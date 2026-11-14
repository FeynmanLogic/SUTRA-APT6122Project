#include "scheduler/scheduler.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    sutra::Scheduler scheduler;
    scheduler.registerWorker({0, 4, 4, 1});
    scheduler.registerWorker({1, 4, 4, 1});
    constexpr std::uint64_t count = 8;
    for (std::uint64_t id = 1; id <= count; ++id) {
        sutra::InferenceRequest request;
        request.request_id = id;
        request.prompt = "Explain scheduling.";
        if (scheduler.submit(request) != sutra::SubmitStatus::Accepted)
            throw std::runtime_error("Demo request was rejected");
    }

    std::cout << "SCHEDULER-ONLY DEMO: synthetic replies; no TCP, llama.cpp or GPU\n";
    std::size_t completed = 0;
    while (completed < count) {
        auto batch = scheduler.waitForDispatch();
        if (!batch) return 1;
        std::cout << "batch " << batch->batch_id << " -> worker " << batch->worker_id
                  << " (" << batch->requests.size() << " requests)\n";
        // This is feedback to exercise the scheduler contract, not an inference
        // implementation. Dhruv's worker supplies real engine responses later.
        for (const auto& request : batch->requests) {
            sutra::InferenceResponse response;
            response.request_id = request.request_id;
            response.worker_id = batch->worker_id;
            response.success = true;
            response.generated_text = "Synthetic scheduler demonstration reply";
            if (!scheduler.complete(batch->batch_id, batch->worker_id, response)) return 1;
        }
        for (const auto& event : scheduler.takeCompletions()) {
            std::cout << "request " << event.response.request_id << ": "
                      << sutra::toString(event.code) << '\n';
            ++completed;
        }
    }
    scheduler.shutdown();
    std::cout << completed << "/" << count << " requests completed\n";
}
