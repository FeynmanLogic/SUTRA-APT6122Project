#include "scheduler/scheduler.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;
using namespace sutra;

// Do not use assert(): Release builds define NDEBUG and would skip the checks.
#define CHECK(expression) do { if (!(expression)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #expression); } while (false)

namespace {
Scheduler::TimePoint at(int milliseconds) {
    return Scheduler::TimePoint{5s} + std::chrono::milliseconds(milliseconds);
}
SchedulerConfig immediate() {
    SchedulerConfig config;
    config.max_batch_wait = 0ms;
    config.worker_timeout = 10s;
    config.request_timeout = 5s;
    return config;
}
InferenceRequest request(std::uint64_t id, int priority = 0) {
    InferenceRequest value;
    value.request_id = id;
    value.prompt = "Keep each prompt independent: " + std::to_string(id);
    value.priority = priority;
    value.arrival_time_ms = 1793217600000LL;
    return value;
}
InferenceResponse response(const DispatchBatch& batch, std::size_t index = 0) {
    InferenceResponse value;
    value.request_id = batch.requests.at(index).request_id;
    value.worker_id = batch.worker_id;
    value.success = true;
    value.generated_text = "synthetic test reply";
    value.generated_tokens = 1;
    value.inference_time_ms = 2;
    value.tokens_per_second = 500;
    return value;
}
DispatchBatch dispatch(Scheduler& scheduler, int milliseconds = 0) {
    auto value = scheduler.tryDispatch(at(milliseconds));
    CHECK(value.has_value());
    return std::move(*value);
}
void accept(Scheduler& scheduler, std::uint64_t id, int priority = 0, int milliseconds = 0) {
    CHECK(scheduler.submit(request(id, priority), at(milliseconds)) == SubmitStatus::Accepted);
}

void configuration_and_input_validation() {
    auto config = immediate();
    config.queue_capacity = 0;
    bool threw = false;
    try { Scheduler invalid(config); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    config = immediate();
    config.max_batch_wait = -1ms;
    threw = false;
    try { Scheduler invalid(config); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    Scheduler scheduler(immediate());
    for (int invalid = 0; invalid < 7; ++invalid) {
        auto value = request(1);
        switch (invalid) {
        case 0: value.request_id = 0; break;
        case 1: value.prompt.clear(); break;
        case 2: value.prompt.assign(16385, 'x'); break;
        case 3: value.max_new_tokens = 0; break;
        case 4: value.max_new_tokens = 513; break;
        case 5: value.priority = 3; break;
        case 6: value.arrival_time_ms = -1; break;
        }
        CHECK(scheduler.submit(value, at(0)) == SubmitStatus::InvalidRequest);
    }
    CHECK(!scheduler.registerWorker({-1, 1, 1, 1}, at(0)));
    CHECK(!scheduler.registerWorker({0, 0, 1, 1}, at(0)));
    CHECK(!scheduler.registerWorker({0, 1, 0, 1}, at(0)));
    CHECK(!scheduler.registerWorker({0, 1, 1, 0}, at(0)));
    CHECK(scheduler.snapshot().queued == 0);
}

void bounded_admission_and_outbox() {
    auto config = immediate();
    config.queue_capacity = 1;
    config.outstanding_capacity = 1;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(0)));
    accept(scheduler, 1);
    CHECK(scheduler.submit(request(1), at(0)) == SubmitStatus::DuplicateRequest);
    CHECK(scheduler.submit(request(2), at(0)) == SubmitStatus::QueueFull);
    auto batch = dispatch(scheduler);
    CHECK(scheduler.submit(request(2), at(0)) == SubmitStatus::OutstandingFull);
    CHECK(scheduler.complete(batch.batch_id, 0, response(batch), at(1)));
    CHECK(scheduler.submit(request(1), at(1)) == SubmitStatus::DuplicateRequest);
    CHECK(scheduler.submit(request(2), at(1)) == SubmitStatus::OutstandingFull);
    CHECK(scheduler.takeCompletions().size() == 1);
    CHECK(scheduler.submit(request(2), at(1)) == SubmitStatus::Accepted);
}

void fifo_preserves_request_fields() {
    Scheduler scheduler(immediate());
    CHECK(scheduler.registerWorker({0, 4, 4, 1}, at(0)));
    auto original = request(41, 0);
    original.max_new_tokens = 17;
    CHECK(scheduler.submit(original, at(0)) == SubmitStatus::Accepted);
    accept(scheduler, 9, 2);
    const auto batch = dispatch(scheduler);
    CHECK(batch.requests.size() == 2);
    CHECK(batch.requests[0].request_id == 41);
    CHECK(batch.requests[1].request_id == 9);
    CHECK(batch.requests[0].prompt == original.prompt);
    CHECK(batch.requests[0].max_new_tokens == 17);
    CHECK(batch.requests[0].priority == 0);
    CHECK(batch.requests[0].arrival_time_ms == original.arrival_time_ms);
}

void priority_and_fifo_ties() {
    auto config = immediate();
    config.queue_policy = QueuePolicy::Priority;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 4, 4, 1}, at(0)));
    accept(scheduler, 1, 0);
    accept(scheduler, 2, 2);
    accept(scheduler, 3, 1);
    accept(scheduler, 4, 2);
    const auto batch = dispatch(scheduler);
    const std::vector<std::uint64_t> expected{2, 4, 3, 1};
    for (std::size_t index = 0; index < expected.size(); ++index)
        CHECK(batch.requests[index].request_id == expected[index]);
}

void aging_prevents_priority_starvation() {
    auto config = immediate();
    config.queue_policy = QueuePolicy::Priority;
    config.aging_threshold = 100ms;
    config.max_batch_size = 1;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(0)));
    accept(scheduler, 1, 0);
    accept(scheduler, 2, 2, 99);
    auto first = dispatch(scheduler, 99);
    CHECK(first.requests[0].request_id == 2);
    CHECK(scheduler.complete(first.batch_id, 0, response(first), at(99)));
    accept(scheduler, 3, 2, 100);
    const auto second = dispatch(scheduler, 100);
    CHECK(second.requests[0].request_id == 1);
    CHECK(second.requests[0].priority == 0);
}

void count_and_oldest_request_timer() {
    auto config = immediate();
    config.max_batch_wait = 10ms;
    config.max_batch_size = 3;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 3, 3, 1}, at(0)));
    accept(scheduler, 1);
    CHECK(!scheduler.tryDispatch(at(9)));
    accept(scheduler, 2, 0, 9);
    auto partial = dispatch(scheduler, 10);
    CHECK(partial.requests.size() == 2);  // new arrivals did not reset the timer
    for (std::size_t index = 0; index < partial.requests.size(); ++index)
        CHECK(scheduler.complete(partial.batch_id, 0, response(partial, index), at(10)));
    accept(scheduler, 3, 0, 11);
    accept(scheduler, 4, 0, 11);
    accept(scheduler, 5, 0, 11);
    CHECK(dispatch(scheduler, 11).requests.size() == 3);  // full dispatches early
}

void worker_request_and_batch_capacity() {
    Scheduler scheduler(immediate());
    CHECK(scheduler.registerWorker({0, 3, 2, 2}, at(0)));
    CHECK(!scheduler.registerWorker({0, 99, 99, 99}, at(0)));
    for (std::uint64_t id = 1; id <= 5; ++id) accept(scheduler, id);
    const auto first = dispatch(scheduler);
    const auto second = dispatch(scheduler);
    CHECK(first.requests.size() == 2);
    CHECK(second.requests.size() == 1);
    CHECK(!scheduler.tryDispatch(at(0)));
    CHECK(scheduler.complete(first.batch_id, 0, response(first, 0), at(1)));
    // One request slot is free, but BOTH batch slots are still occupied.
    CHECK(!scheduler.tryDispatch(at(1)));
    CHECK(scheduler.complete(first.batch_id, 0, response(first, 1), at(1)));
    CHECK(dispatch(scheduler, 1).requests.size() == 2);
    CHECK(scheduler.snapshot().in_flight == 3);
}

void round_robin_skips_busy_and_unready_workers() {
    auto config = immediate();
    config.max_batch_size = 1;
    Scheduler scheduler(config);
    for (int id : {2, 7, 11}) CHECK(scheduler.registerWorker({id, 1, 1, 1}, at(0)));
    CHECK(scheduler.setWorkerReady(7, false, at(0)));
    accept(scheduler, 1);
    accept(scheduler, 2);
    accept(scheduler, 3);
    const auto first = dispatch(scheduler);
    CHECK(first.worker_id == 2);
    CHECK(dispatch(scheduler).worker_id == 11);
    CHECK(!scheduler.tryDispatch(at(0)));
    CHECK(scheduler.setWorkerReady(7, true, at(1)));
    CHECK(dispatch(scheduler, 1).worker_id == 7);
}

void least_loaded_uses_outstanding_requests() {
    auto config = immediate();
    config.worker_policy = WorkerPolicy::LeastLoaded;
    config.max_batch_size = 2;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 8, 2, 8}, at(0)));
    CHECK(scheduler.registerWorker({1, 8, 2, 8}, at(0)));
    accept(scheduler, 1);
    accept(scheduler, 2);
    CHECK(dispatch(scheduler).worker_id == 0);  // worker 0 now has two
    accept(scheduler, 3);
    CHECK(dispatch(scheduler).worker_id == 1);  // worker 1 now has one
    accept(scheduler, 4);
    CHECK(dispatch(scheduler).worker_id == 1);  // differs from pure round-robin
}

void result_correlation_and_duplicate_suppression() {
    Scheduler scheduler(immediate());
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(0)));
    accept(scheduler, 1);
    const auto batch = dispatch(scheduler);
    CHECK(!scheduler.complete(batch.batch_id + 1, 0, response(batch), at(1)));
    CHECK(!scheduler.complete(batch.batch_id, 99, response(batch), at(1)));
    auto wrong = response(batch);
    wrong.request_id = 999;
    CHECK(!scheduler.complete(batch.batch_id, 0, wrong, at(1)));
    wrong = response(batch);
    wrong.worker_id = 99;
    CHECK(!scheduler.complete(batch.batch_id, 0, wrong, at(1)));
    CHECK(scheduler.snapshot().in_flight == 1);
    CHECK(scheduler.complete(batch.batch_id, 0, response(batch), at(2)));
    CHECK(!scheduler.complete(batch.batch_id, 0, response(batch), at(2)));
    CHECK(scheduler.takeCompletions().size() == 1);
    CHECK(scheduler.snapshot().in_flight == 0);
}

void out_of_order_partial_results_and_metrics() {
    auto config = immediate();
    config.max_batch_wait = 10ms;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 4, 4, 1}, at(0)));
    accept(scheduler, 1);
    accept(scheduler, 2, 0, 5);
    const auto batch = dispatch(scheduler, 10);
    auto second = response(batch, 1);
    second.queue_time_ms = 2;
    second.latency_ms = -999;  // overwritten using coordinator-local steady time
    CHECK(scheduler.complete(batch.batch_id, 0, second, at(30)));
    CHECK(scheduler.complete(batch.batch_id, 0, response(batch, 0), at(31)));
    const auto events = scheduler.takeCompletions();
    CHECK(events.size() == 2);
    CHECK(events[0].response.request_id == 2);
    CHECK(events[0].response.queue_time_ms == 7);
    CHECK(events[0].response.latency_ms == 25);
    CHECK(events[1].response.queue_time_ms == 10);
    CHECK(events[1].response.latency_ms == 31);
}

void engine_failure_and_invalid_metrics_release_capacity() {
    Scheduler scheduler(immediate());
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(0)));
    accept(scheduler, 1);
    auto first = dispatch(scheduler);
    auto failure = response(first);
    failure.success = false;
    failure.error_message = "model rejected context overflow";
    CHECK(scheduler.complete(first.batch_id, 0, failure, at(1)));
    CHECK(scheduler.takeCompletions().at(0).code == CompletionCode::InferenceFailed);
    accept(scheduler, 2, 0, 1);
    auto second = dispatch(scheduler, 1);
    auto invalid = response(second);
    invalid.inference_time_ms = std::numeric_limits<double>::quiet_NaN();
    CHECK(scheduler.complete(second.batch_id, 0, invalid, at(2)));
    const auto event = scheduler.takeCompletions().at(0);
    CHECK(event.code == CompletionCode::InvalidResponse);
    CHECK(!event.response.success);
    CHECK(event.response.generated_text.empty());
    CHECK(scheduler.snapshot().in_flight == 0);
}

void queued_request_timeout_without_workers() {
    auto config = immediate();
    config.request_timeout = 20ms;
    Scheduler scheduler(config);
    accept(scheduler, 1);
    scheduler.poll(at(19));
    CHECK(scheduler.takeCompletions().empty());
    scheduler.poll(at(20));
    const auto events = scheduler.takeCompletions();
    CHECK(events.size() == 1);
    CHECK(events[0].code == CompletionCode::RequestTimeout);
    CHECK(events[0].response.worker_id == -1);
    CHECK(events[0].response.queue_time_ms == 20);
    scheduler.poll(at(21));
    CHECK(scheduler.takeCompletions().empty());
}

void heartbeats_and_expired_session_reconnect() {
    auto config = immediate();
    config.worker_timeout = 10ms;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(0)));
    accept(scheduler, 1);
    const auto old_batch = dispatch(scheduler);
    CHECK(scheduler.heartbeat(0, at(9)));
    scheduler.poll(at(18));
    CHECK(scheduler.snapshot().in_flight == 1);
    CHECK(!scheduler.heartbeat(0, at(19)));  // exactly at deadline is too late
    CHECK(scheduler.takeCompletions().at(0).code == CompletionCode::WorkerTimeout);
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(20)));
    accept(scheduler, 2, 0, 20);
    const auto fresh = dispatch(scheduler, 20);
    CHECK(fresh.batch_id != old_batch.batch_id);
    CHECK(!scheduler.complete(old_batch.batch_id, 0, response(old_batch), at(21)));
    CHECK(scheduler.snapshot().in_flight == 1);
    CHECK(scheduler.complete(fresh.batch_id, 0, response(fresh), at(21)));
}

void in_flight_timeout_retires_session_not_gpu_capacity() {
    auto config = immediate();
    config.request_timeout = 20ms;
    config.max_batch_size = 1;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 4, 1, 4}, at(0)));
    accept(scheduler, 1);
    const auto expired = dispatch(scheduler);
    accept(scheduler, 2, 0, 5);
    dispatch(scheduler, 5);
    CHECK(scheduler.registerWorker({1, 1, 1, 1}, at(5)));
    accept(scheduler, 3, 0, 5);
    const auto healthy = dispatch(scheduler, 5);
    CHECK(healthy.worker_id == 1);
    scheduler.poll(at(20));
    const auto events = scheduler.takeCompletions();
    CHECK(events.size() == 2);
    CHECK(events[0].code == CompletionCode::RequestTimeout);
    CHECK(events[1].code == CompletionCode::WorkerUnavailable);
    CHECK(!scheduler.heartbeat(0, at(20)));
    CHECK(!scheduler.complete(expired.batch_id, 0, response(expired), at(20)));
    CHECK(scheduler.snapshot().in_flight == 1);
    CHECK(scheduler.complete(healthy.batch_id, 1, response(healthy), at(21)));
}

void disconnect_fails_only_assigned_work_without_retry() {
    auto config = immediate();
    config.max_batch_size = 1;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(0)));
    accept(scheduler, 1);
    accept(scheduler, 2);
    auto first = dispatch(scheduler);
    CHECK(scheduler.removeWorker(0, "send failed or connection closed", at(1)));
    CHECK(!scheduler.removeWorker(0, "duplicate close", at(1)));
    CHECK(scheduler.takeCompletions().at(0).code == CompletionCode::WorkerDisconnected);
    CHECK(scheduler.snapshot().queued == 1);
    CHECK(scheduler.registerWorker({1, 1, 1, 1}, at(2)));
    CHECK(dispatch(scheduler, 2).requests[0].request_id == 2);
    CHECK(!scheduler.complete(first.batch_id, 0, response(first), at(2)));
}

void shutdown_is_idempotent_and_terminal() {
    auto config = immediate();
    config.max_batch_size = 1;
    config.request_timeout = 1ms;  // explicit shutdown wins even at a request deadline
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 1, 1, 1}, at(0)));
    accept(scheduler, 1);
    accept(scheduler, 2);
    auto batch = dispatch(scheduler);
    scheduler.shutdown(at(1));
    scheduler.shutdown(at(2));
    const auto events = scheduler.takeCompletions();
    CHECK(events.size() == 2);
    CHECK(events[0].code == CompletionCode::Shutdown);
    CHECK(events[1].code == CompletionCode::Shutdown);
    CHECK(scheduler.submit(request(3), at(2)) == SubmitStatus::Stopped);
    CHECK(!scheduler.registerWorker({1, 1, 1, 1}, at(2)));
    CHECK(!scheduler.tryDispatch(at(2)));
    CHECK(!scheduler.complete(batch.batch_id, 0, response(batch), at(2)));
    const auto state = scheduler.snapshot();
    CHECK(state.stopped && state.queued == 0 && state.in_flight == 0);
}

void blocked_consumers_wake_on_shutdown() {
    Scheduler scheduler;
    auto dispatch_waiter = std::async(std::launch::async, [&] { return scheduler.waitForDispatch(); });
    auto completion_waiter = std::async(std::launch::async, [&] { return scheduler.waitForCompletion(); });
    scheduler.shutdown();
    CHECK(dispatch_waiter.wait_for(2s) == std::future_status::ready);
    CHECK(completion_waiter.wait_for(2s) == std::future_status::ready);
    CHECK(!dispatch_waiter.get());
    CHECK(!completion_waiter.get());
}

void blocking_dispatch_observes_collection_timer() {
    auto config = immediate();
    config.max_batch_wait = 15ms;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 4, 4, 1}));
    const auto start = Scheduler::Clock::now();
    CHECK(scheduler.submit(request(1), start) == SubmitStatus::Accepted);
    auto waiter = std::async(std::launch::async, [&] { return scheduler.waitForDispatch(); });
    const bool ready = waiter.wait_for(2s) == std::future_status::ready;
    if (!ready) scheduler.shutdown();
    const auto value = waiter.get();
    scheduler.shutdown();
    CHECK(ready && value.has_value());
    CHECK(Scheduler::Clock::now() - start >= 15ms);
}

void blocking_completion_drives_request_deadlines() {
    auto config = immediate();
    config.request_timeout = 15ms;
    Scheduler scheduler(config);
    CHECK(scheduler.submit(request(1)) == SubmitStatus::Accepted);
    auto waiter = std::async(std::launch::async, [&] { return scheduler.waitForCompletion(); });
    const bool ready = waiter.wait_for(2s) == std::future_status::ready;
    if (!ready) scheduler.shutdown();
    const auto event = waiter.get();
    scheduler.shutdown();
    CHECK(ready && event.has_value());
    CHECK(event->code == CompletionCode::RequestTimeout);
}

void concurrent_producers_have_no_loss_or_duplicate_completion() {
    auto config = immediate();
    config.max_batch_wait = 1ms;
    Scheduler scheduler(config);
    CHECK(scheduler.registerWorker({0, 4, 4, 1}));
    CHECK(scheduler.registerWorker({1, 4, 4, 1}));
    constexpr std::size_t producer_count = 8;
    constexpr std::size_t per_producer = 16;
    constexpr std::size_t total = producer_count * per_producer;
    std::atomic<bool> accepted{true};
    auto consumer = std::async(std::launch::async, [&] {
        std::size_t count = 0;
        while (count < total) {
            auto batch = scheduler.waitForDispatch();
            if (!batch) break;
            for (std::size_t index = batch->requests.size(); index > 0; --index) {
                if (!scheduler.complete(batch->batch_id, batch->worker_id, response(*batch, index - 1))) {
                    scheduler.shutdown();
                    return std::size_t{0};
                }
                ++count;
            }
        }
        return count;
    });
    std::vector<std::thread> producers;
    for (std::size_t producer = 0; producer < producer_count; ++producer) {
        producers.emplace_back([&, producer] {
            for (std::size_t item = 0; item < per_producer; ++item) {
                const auto id = producer * per_producer + item + 1;
                if (scheduler.submit(request(id)) != SubmitStatus::Accepted) {
                    accepted = false;
                    scheduler.shutdown();
                    return;
                }
            }
        });
    }
    for (auto& producer : producers) producer.join();
    const bool ready = consumer.wait_for(5s) == std::future_status::ready;
    scheduler.shutdown();
    const auto count = consumer.get();
    CHECK(ready && accepted && count == total);
    const auto events = scheduler.takeCompletions();
    CHECK(events.size() == total);
    std::set<std::uint64_t> seen;
    for (const auto& event : events) {
        CHECK(event.code == CompletionCode::Completed);
        CHECK(seen.insert(event.response.request_id).second);
    }
    CHECK(seen.size() == total);
    CHECK(scheduler.snapshot().in_flight == 0);
}
}  // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> cases{
        {"configuration and input validation", configuration_and_input_validation},
        {"bounded admission and outbox", bounded_admission_and_outbox},
        {"FIFO preserves all request fields", fifo_preserves_request_fields},
        {"priority and FIFO ties", priority_and_fifo_ties},
        {"aging prevents starvation", aging_prevents_priority_starvation},
        {"batch count and oldest-request timer", count_and_oldest_request_timer},
        {"worker request and batch capacity", worker_request_and_batch_capacity},
        {"round-robin ready/capacity filtering", round_robin_skips_busy_and_unready_workers},
        {"least-loaded outstanding counts", least_loaded_uses_outstanding_requests},
        {"result correlation and duplicate suppression", result_correlation_and_duplicate_suppression},
        {"out-of-order partial results and metrics", out_of_order_partial_results_and_metrics},
        {"engine failure and invalid response", engine_failure_and_invalid_metrics_release_capacity},
        {"queued timeout without workers", queued_request_timeout_without_workers},
        {"heartbeats, expiry and reconnect", heartbeats_and_expired_session_reconnect},
        {"in-flight timeout quarantines worker", in_flight_timeout_retires_session_not_gpu_capacity},
        {"disconnect without automatic retry", disconnect_fails_only_assigned_work_without_retry},
        {"idempotent terminal shutdown", shutdown_is_idempotent_and_terminal},
        {"blocked consumers wake on shutdown", blocked_consumers_wake_on_shutdown},
        {"blocking dispatch collection timer", blocking_dispatch_observes_collection_timer},
        {"blocking completion deadline", blocking_completion_drives_request_deadlines},
        {"eight concurrent producers, 128 unique completions", concurrent_producers_have_no_loss_or_duplicate_completion},
    };
    std::size_t failed = 0;
    for (const auto& test : cases) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << cases.size() - failed << "/" << cases.size() << " scheduler cases passed\n";
    return failed == 0 ? 0 : 1;
}
