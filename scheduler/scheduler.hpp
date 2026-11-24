#pragma once

#include "common/inference_types.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace sutra {

enum class QueuePolicy { Fifo, Priority };
enum class WorkerPolicy { RoundRobin, LeastLoaded };
enum class SubmitStatus {
    Accepted, InvalidRequest, DuplicateRequest, QueueFull, OutstandingFull, Stopped
};
enum class CompletionCode {
    Completed, InferenceFailed, InvalidResponse, RequestTimeout,
    WorkerDisconnected, WorkerTimeout, WorkerUnavailable, Shutdown
};
const char* toString(SubmitStatus status);
const char* toString(CompletionCode code);

struct SchedulerConfig {
    std::size_t queue_capacity{1024};
    // Includes queued, executing AND completed-but-not-collected requests.
    // A slow response consumer therefore cannot grow the outbox indefinitely.
    std::size_t outstanding_capacity{4096};
    std::size_t max_batch_size{4};
    std::size_t max_prompt_bytes{16 * 1024};
    std::chrono::milliseconds max_batch_wait{10};
    std::chrono::milliseconds aging_threshold{1000};
    std::chrono::milliseconds request_timeout{120000};
    std::chrono::milliseconds worker_timeout{3000};
    QueuePolicy queue_policy{QueuePolicy::Fifo};
    WorkerPolicy worker_policy{WorkerPolicy::RoundRobin};
};

struct WorkerRegistration {
    int worker_id{-1};
    std::size_t request_capacity{4};
    std::size_t max_batch_size{4};
    // One batch at a time by default: a single model context has one owner.
    // Raise this only when the worker has an explicitly bounded serial queue
    // or independently safe execution contexts. These are NOT GPU token limits.
    std::size_t batch_capacity{1};
};

struct DispatchBatch {
    std::uint64_t batch_id{0};
    int worker_id{-1};
    RequestBatch requests;
};

struct Completion {
    InferenceResponse response;
    CompletionCode code{CompletionCode::Completed};
};

struct SchedulerSnapshot {
    std::size_t queued{0};
    std::size_t in_flight{0};
    std::size_t buffered_completions{0};
    std::size_t registered_workers{0};
    bool stopped{false};
};

// Owns scheduling state, not threads, sockets, model objects, or callbacks.
// All public operations are synchronized. Network/model I/O happens AFTER a
// returned dispatch leaves the mutex. Use one dispatch thread per coordinator.
class Scheduler {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit Scheduler(SchedulerConfig config = {});
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Production callers omit 'now'. Tests may supply a synthetic monotonic
    // timeline to nonblocking operations; never mix it with blocking methods.
    // IDs must be nonzero and unique for the entire coordinator run.
    SubmitStatus submit(InferenceRequest request, TimePoint now = Clock::now());
    // Register only after model-ready. Duplicate live worker IDs are rejected,
    // so a reconnect cannot silently erase reservations on an old connection.
    bool registerWorker(WorkerRegistration worker, TimePoint now = Clock::now());
    bool setWorkerReady(int worker_id, bool ready, TimePoint now = Clock::now());
    bool heartbeat(int worker_id, TimePoint now = Clock::now());
    bool removeWorker(int worker_id, const std::string& reason,
                      TimePoint now = Clock::now());

    std::optional<DispatchBatch> tryDispatch(TimePoint now = Clock::now());
    // Blocks without spinning; wakes for admission, capacity, timers or shutdown.
    std::optional<DispatchBatch> waitForDispatch();

    // Match connection/session + batch ID + request ID. A true return means
    // the result was consumed, including conversion of invalid metrics to an
    // explicit InvalidResponse failure. Duplicate/late/wrong-worker replies
    // return false and cannot release capacity twice.
    bool complete(std::uint64_t batch_id, int worker_id, InferenceResponse response,
                  TimePoint now = Clock::now());

    // A nonblocking reactor must call poll regularly even when no traffic arrives.
    // Blocking consumers also drive timers, so no internal timer thread is needed.
    void poll(TimePoint now = Clock::now());
    std::vector<Completion> takeCompletions();
    std::optional<Completion> waitForCompletion();

    // Immediate cancellation, not a graceful drain. Collect terminal failures,
    // close transports, and JOIN caller-owned threads before destroying this object.
    void shutdown(TimePoint now = Clock::now());
    SchedulerSnapshot snapshot() const;

private:
    struct PendingRequest {
        InferenceRequest request;
        TimePoint enqueued_at;
        TimePoint deadline;
    };
    struct Assignment {
        TimePoint enqueued_at;
        TimePoint deadline;
        double queue_time_ms;
    };
    struct BatchState {
        int worker_id;
        std::map<std::uint64_t, Assignment> pending;
    };
    struct WorkerState {
        WorkerRegistration registration;
        TimePoint last_seen;
        bool ready{true};
        std::size_t in_flight_requests{0};
        std::size_t in_flight_batches{0};
    };

    using WorkerIterator = std::map<int, WorkerState>::iterator;
    static double elapsedMs(TimePoint from, TimePoint to);
    bool validRequest(const InferenceRequest& request) const;
    void expireLocked(TimePoint now);
    void retireWorkerLocked(int worker_id, CompletionCode code,
                            const std::string& reason, TimePoint now);
    void failureLocked(std::uint64_t request_id, int worker_id,
                       TimePoint enqueued_at, double queue_ms,
                       CompletionCode code, const std::string& reason, TimePoint now);
    WorkerIterator chooseWorkerLocked();
    std::list<PendingRequest>::iterator chooseRequestLocked(TimePoint now);
    std::optional<DispatchBatch> dispatchLocked(TimePoint now);
    std::optional<TimePoint> nextWakeLocked(TimePoint now);

    const SchedulerConfig config_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::list<PendingRequest> queue_;
    std::map<int, WorkerState> workers_;
    std::map<std::uint64_t, BatchState> batches_;
    // Keep an ID until its terminal event is consumed; this set also enforces
    // outstanding_capacity without a second, potentially inconsistent counter.
    std::unordered_set<std::uint64_t> retained_ids_;
    std::deque<Completion> completions_;
    std::optional<int> last_worker_;
    std::uint64_t next_batch_id_{1};
    bool stopped_{false};
};

}  // namespace sutra
