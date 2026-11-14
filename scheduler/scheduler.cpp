#include "scheduler/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sutra {

const char* toString(SubmitStatus status) {
    switch (status) {
    case SubmitStatus::Accepted: return "accepted";
    case SubmitStatus::InvalidRequest: return "invalid_request";
    case SubmitStatus::DuplicateRequest: return "duplicate_request";
    case SubmitStatus::QueueFull: return "queue_full";
    case SubmitStatus::OutstandingFull: return "outstanding_full";
    case SubmitStatus::Stopped: return "stopped";
    }
    return "unknown";
}

const char* toString(CompletionCode code) {
    switch (code) {
    case CompletionCode::Completed: return "completed";
    case CompletionCode::InferenceFailed: return "inference_failed";
    case CompletionCode::InvalidResponse: return "invalid_response";
    case CompletionCode::RequestTimeout: return "request_timeout";
    case CompletionCode::WorkerDisconnected: return "worker_disconnected";
    case CompletionCode::WorkerTimeout: return "worker_timeout";
    case CompletionCode::WorkerUnavailable: return "worker_unavailable";
    case CompletionCode::Shutdown: return "shutdown";
    }
    return "unknown";
}

Scheduler::Scheduler(SchedulerConfig config) : config_(config) {
    if (config_.queue_capacity == 0 || config_.outstanding_capacity == 0 ||
        config_.max_batch_size == 0 || config_.max_prompt_bytes == 0 ||
        config_.max_batch_wait.count() < 0 || config_.aging_threshold.count() < 0 ||
        config_.request_timeout.count() <= 0 || config_.worker_timeout.count() <= 0) {
        throw std::invalid_argument("Scheduler capacities/timeouts must be positive; batch wait and aging may be zero");
    }
}

double Scheduler::elapsedMs(TimePoint from, TimePoint to) {
    // Normal operation uses steady_clock. Clamp to zero defensively for an
    // accidentally out-of-order explicit test timestamp, rather than emit negatives.
    return std::max(0.0, std::chrono::duration<double, std::milli>(to - from).count());
}

bool Scheduler::validRequest(const InferenceRequest& request) const {
    return request.request_id != 0 && !request.prompt.empty() &&
           request.prompt.size() <= config_.max_prompt_bytes &&
           request.max_new_tokens >= 1 && request.max_new_tokens <= 512 &&
           request.priority >= 0 && request.priority <= 2 && request.arrival_time_ms >= 0;
}

SubmitStatus Scheduler::submit(InferenceRequest request, TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expireLocked(now);
    if (stopped_) return SubmitStatus::Stopped;
    if (!validRequest(request)) return SubmitStatus::InvalidRequest;
    if (retained_ids_.count(request.request_id)) return SubmitStatus::DuplicateRequest;
    if (queue_.size() >= config_.queue_capacity) return SubmitStatus::QueueFull;
    if (retained_ids_.size() >= config_.outstanding_capacity) return SubmitStatus::OutstandingFull;

    retained_ids_.insert(request.request_id);
    queue_.push_back({std::move(request), now, now + config_.request_timeout});
    changed_.notify_all();
    return SubmitStatus::Accepted;
}

bool Scheduler::registerWorker(WorkerRegistration worker, TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expireLocked(now);
    if (stopped_ || worker.worker_id < 0 || worker.request_capacity == 0 ||
        worker.max_batch_size == 0 || worker.batch_capacity == 0 ||
        workers_.count(worker.worker_id)) return false;
    workers_.emplace(worker.worker_id, WorkerState{worker, now});
    changed_.notify_all();
    return true;
}

bool Scheduler::setWorkerReady(int worker_id, bool ready, TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expireLocked(now);
    auto worker = workers_.find(worker_id);
    if (stopped_ || worker == workers_.end()) return false;
    worker->second.ready = ready;
    changed_.notify_all();
    return true;
}

bool Scheduler::heartbeat(int worker_id, TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Expire first: a heartbeat arriving after its deadline cannot revive the
    // old session. The transport must perform a new, model-ready registration.
    expireLocked(now);
    auto worker = workers_.find(worker_id);
    if (stopped_ || worker == workers_.end()) return false;
    worker->second.last_seen = std::max(worker->second.last_seen, now);
    changed_.notify_all();
    return true;
}

bool Scheduler::removeWorker(int worker_id, const std::string& reason, TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expireLocked(now);
    if (!workers_.count(worker_id)) return false;
    retireWorkerLocked(worker_id, CompletionCode::WorkerDisconnected, reason, now);
    changed_.notify_all();
    return true;
}

void Scheduler::failureLocked(std::uint64_t request_id, int worker_id,
                              TimePoint enqueued_at, double queue_ms,
                              CompletionCode code, const std::string& reason, TimePoint now) {
    InferenceResponse response;
    response.request_id = request_id;
    response.worker_id = worker_id;
    response.queue_time_ms = queue_ms;
    response.latency_ms = elapsedMs(enqueued_at, now);
    response.error_message = reason;
    completions_.push_back({std::move(response), code});
}

void Scheduler::retireWorkerLocked(int worker_id, CompletionCode code,
                                   const std::string& reason, TimePoint now) {
    // Erase every assignment on this connection, without retrying uncertain
    // execution. Batch IDs remain unique even if this numeric worker ID returns.
    for (auto batch = batches_.begin(); batch != batches_.end();) {
        if (batch->second.worker_id != worker_id) { ++batch; continue; }
        for (const auto& item : batch->second.pending) {
            const auto& assignment = item.second;
            const bool expired = code != CompletionCode::Shutdown && assignment.deadline <= now;
            failureLocked(item.first, worker_id, assignment.enqueued_at,
                          assignment.queue_time_ms,
                          expired ? CompletionCode::RequestTimeout : code,
                          expired ? "Request deadline exceeded during execution" : reason, now);
        }
        batch = batches_.erase(batch);
    }
    workers_.erase(worker_id);
}

void Scheduler::expireLocked(TimePoint now) {
    bool changed = false;
    for (auto item = queue_.begin(); item != queue_.end();) {
        if (item->deadline > now) { ++item; continue; }
        failureLocked(item->request.request_id, -1, item->enqueued_at,
                      elapsedMs(item->enqueued_at, now), CompletionCode::RequestTimeout,
                      "Request deadline exceeded while queued", now);
        item = queue_.erase(item);
        changed = true;
    }

    // An execution deadline is NOT proof the model stopped. Retire the entire
    // session instead of treating a still-running GPU as newly idle. Other
    // assignments on it receive WorkerUnavailable unless their own time expired.
    std::map<int, CompletionCode> retired;
    for (const auto& worker : workers_) {
        if (now >= worker.second.last_seen + config_.worker_timeout)
            retired.emplace(worker.first, CompletionCode::WorkerTimeout);
    }
    for (const auto& batch : batches_) {
        for (const auto& item : batch.second.pending) {
            if (item.second.deadline <= now)
                retired.emplace(batch.second.worker_id, CompletionCode::WorkerUnavailable);
        }
    }
    for (const auto& worker : retired) {
        retireWorkerLocked(worker.first, worker.second,
                           worker.second == CompletionCode::WorkerTimeout
                               ? "Worker heartbeat deadline exceeded"
                               : "Worker session retired after an in-flight request timeout", now);
        changed = true;
    }
    if (changed) changed_.notify_all();
}

Scheduler::WorkerIterator Scheduler::chooseWorkerLocked() {
    if (workers_.empty()) return workers_.end();
    auto start = last_worker_ ? workers_.upper_bound(*last_worker_) : workers_.begin();
    if (start == workers_.end()) start = workers_.begin();
    auto best = workers_.end();
    auto candidate = start;
    do {
        const auto& worker = candidate->second;
        const bool eligible = worker.ready &&
            worker.in_flight_requests < worker.registration.request_capacity &&
            worker.in_flight_batches < worker.registration.batch_capacity;
        if (eligible) {
            if (config_.worker_policy == WorkerPolicy::RoundRobin) return candidate;
            if (best == workers_.end() ||
                worker.in_flight_requests < best->second.in_flight_requests) best = candidate;
            // Equal loads keep the first cyclic candidate, giving round-robin ties.
        }
        if (++candidate == workers_.end()) candidate = workers_.begin();
    } while (candidate != start);
    return best;
}

std::list<Scheduler::PendingRequest>::iterator Scheduler::chooseRequestLocked(TimePoint now) {
    auto best = queue_.begin();
    if (config_.queue_policy == QueuePolicy::Fifo) return best;
    // Aging changes selection metadata, never the original priority field.
    // Scan in admission order: the first aged request wins, otherwise highest
    // priority wins with admission-order ties. A mutable heap key is unnecessary.
    for (auto item = queue_.begin(); item != queue_.end(); ++item) {
        if (now >= item->enqueued_at + config_.aging_threshold) return item;
        if (item->request.priority > best->request.priority) best = item;
    }
    return best;
}

std::optional<DispatchBatch> Scheduler::dispatchLocked(TimePoint now) {
    if (stopped_ || queue_.empty()) return std::nullopt;
    auto chosen = chooseWorkerLocked();
    if (chosen == workers_.end()) return std::nullopt;
    auto& worker = chosen->second;
    const auto limit = std::min({config_.max_batch_size, worker.registration.max_batch_size,
                                worker.registration.request_capacity - worker.in_flight_requests});
    const auto oldest = std::min_element(queue_.begin(), queue_.end(),
        [](const PendingRequest& a, const PendingRequest& b) { return a.enqueued_at < b.enqueued_at; });
    if (queue_.size() < limit && now < oldest->enqueued_at + config_.max_batch_wait)
        return std::nullopt;
    if (next_batch_id_ == 0) throw std::overflow_error("Coordinator batch IDs exhausted");

    DispatchBatch dispatch{next_batch_id_++, chosen->first, {}};
    const auto count = std::min(limit, queue_.size());
    dispatch.requests.reserve(count);
    BatchState state{chosen->first, {}};
    for (std::size_t index = 0; index < count; ++index) {
        auto item = chooseRequestLocked(now);
        state.pending.emplace(item->request.request_id,
                              Assignment{item->enqueued_at, item->deadline,
                                         elapsedMs(item->enqueued_at, now)});
        dispatch.requests.push_back(std::move(item->request));
        queue_.erase(item);
    }
    // Reservation and removal happen under the same lock. Two dispatch callers
    // cannot both observe and reserve the same apparently-free capacity.
    worker.in_flight_requests += count;
    ++worker.in_flight_batches;
    batches_.emplace(dispatch.batch_id, std::move(state));
    last_worker_ = chosen->first;
    changed_.notify_all();
    return dispatch;
}

std::optional<DispatchBatch> Scheduler::tryDispatch(TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expireLocked(now);
    return dispatchLocked(now);
}

std::optional<Scheduler::TimePoint> Scheduler::nextWakeLocked(TimePoint now) {
    std::optional<TimePoint> wake;
    auto include = [&wake](TimePoint value) {
        if (!wake || value < *wake) wake = value;
    };
    // A past collection timer with no free worker must NOT cause a busy loop.
    const bool available = chooseWorkerLocked() != workers_.end();
    for (const auto& item : queue_) {
        include(item.deadline);
        const auto batch_ready = item.enqueued_at + config_.max_batch_wait;
        if (available && batch_ready > now) include(batch_ready);
    }
    for (const auto& worker : workers_) include(worker.second.last_seen + config_.worker_timeout);
    for (const auto& batch : batches_)
        for (const auto& item : batch.second.pending) include(item.second.deadline);
    return wake;
}

std::optional<DispatchBatch> Scheduler::waitForDispatch() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        const auto now = Clock::now();
        expireLocked(now);
        if (stopped_) return std::nullopt;
        if (auto dispatch = dispatchLocked(now)) return dispatch;
        if (auto wake = nextWakeLocked(now)) changed_.wait_until(lock, *wake);
        else changed_.wait(lock);
        // Recheck everything after notification, timeout, or a spurious wakeup.
    }
}

bool Scheduler::complete(std::uint64_t batch_id, int worker_id,
                         InferenceResponse response, TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expireLocked(now);
    auto batch = batches_.find(batch_id);
    if (stopped_ || batch == batches_.end() || batch->second.worker_id != worker_id ||
        response.worker_id != worker_id) return false;
    auto item = batch->second.pending.find(response.request_id);
    if (item == batch->second.pending.end()) return false;
    const auto assignment = item->second;

    auto valid_duration = [](double value) { return std::isfinite(value) && value >= 0.0; };
    const bool valid = response.input_tokens >= 0 && response.generated_tokens >= 0 &&
        valid_duration(response.queue_time_ms) && valid_duration(response.inference_time_ms) &&
        valid_duration(response.time_to_first_token_ms) && valid_duration(response.tokens_per_second);
    if (!valid) {
        failureLocked(response.request_id, worker_id, assignment.enqueued_at,
                      assignment.queue_time_ms, CompletionCode::InvalidResponse,
                      "Worker returned negative or non-finite token counts/timings", now);
    } else {
        // Worker queue time is LOCAL receipt-to-execution waiting, not a copy
        // of the coordinator queue duration. Never subtract cross-host clocks.
        response.queue_time_ms += assignment.queue_time_ms;
        response.latency_ms = elapsedMs(assignment.enqueued_at, now);
        const auto code = response.success ? CompletionCode::Completed : CompletionCode::InferenceFailed;
        completions_.push_back({std::move(response), code});
    }

    auto& worker = workers_.at(worker_id);
    --worker.in_flight_requests;
    batch->second.pending.erase(item);
    if (batch->second.pending.empty()) {
        --worker.in_flight_batches;
        batches_.erase(batch);
    }
    changed_.notify_all();
    return true;
}

void Scheduler::poll(TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    expireLocked(now);
}

std::vector<Completion> Scheduler::takeCompletions() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Completion> result;
    result.reserve(completions_.size());
    while (!completions_.empty()) {
        retained_ids_.erase(completions_.front().response.request_id);
        result.push_back(std::move(completions_.front()));
        completions_.pop_front();
    }
    changed_.notify_all();
    return result;
}

std::optional<Completion> Scheduler::waitForCompletion() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        const auto now = Clock::now();
        expireLocked(now);
        if (!completions_.empty()) {
            Completion result = std::move(completions_.front());
            completions_.pop_front();
            retained_ids_.erase(result.response.request_id);
            changed_.notify_all();
            return result;
        }
        if (stopped_) return std::nullopt;
        // This thread consumes terminal events, not batches. A batch collection
        // timer is irrelevant here and would repeatedly wake it before dispatch.
        std::optional<TimePoint> wake;
        auto include = [&wake](TimePoint value) { if (!wake || value < *wake) wake = value; };
        for (const auto& item : queue_) include(item.deadline);
        for (const auto& worker : workers_) include(worker.second.last_seen + config_.worker_timeout);
        for (const auto& batch : batches_)
            for (const auto& item : batch.second.pending) include(item.second.deadline);
        if (wake) changed_.wait_until(lock, *wake);
        else changed_.wait(lock);
    }
}

void Scheduler::shutdown(TimePoint now) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) return;
    stopped_ = true;
    for (const auto& item : queue_)
        failureLocked(item.request.request_id, -1, item.enqueued_at,
                      elapsedMs(item.enqueued_at, now), CompletionCode::Shutdown,
                      "Coordinator shutting down", now);
    queue_.clear();
    while (!workers_.empty())
        retireWorkerLocked(workers_.begin()->first, CompletionCode::Shutdown,
                           "Coordinator shutting down", now);
    changed_.notify_all();
}

SchedulerSnapshot Scheduler::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t in_flight = 0;
    for (const auto& batch : batches_) in_flight += batch.second.pending.size();
    return {queue_.size(), in_flight, completions_.size(), workers_.size(), stopped_};
}

}  // namespace sutra
