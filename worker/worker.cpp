/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10
*/

#include "worker.hpp"

#include <exception>
#include <stdexcept>
#include <utility>

namespace sutra {

InferenceWorker::InferenceWorker(
    int worker_id,
    std::unique_ptr<InferenceEngine> engine
)
    : worker_id_(worker_id),
      engine_(std::move(engine)) {
    if (worker_id_ < 0) {
        throw std::invalid_argument(
            "Worker ID cannot be negative."
        );
    }

    if (engine_ == nullptr) {
        throw std::invalid_argument(
            "InferenceWorker requires a valid engine."
        );
    }
}

ResponseBatch InferenceWorker::processBatch(
    const RequestBatch& requests
) {
    if (requests.empty()) {
        return {};
    }

    try {
        ResponseBatch responses =
            engine_->generatefullBatch(requests);

        if (responses.size() != requests.size()) {
            throw std::runtime_error(
                "The number of responses does not match "
                "the number of requests."
            );
        }

        for (std::size_t index = 0;
             index < responses.size();
             ++index) {
            responses[index].worker_id = worker_id_;

            if (
                responses[index].request_id !=
                requests[index].request_id
            ) {
                // Preserve the input ID so the scheduler can terminate this
                // request rather than rejecting it and waiting for a timeout.
                // Discard another request's text and metrics as well.
                responses[index] = InferenceResponse{};
                responses[index].request_id = requests[index].request_id;
                responses[index].worker_id = worker_id_;
                responses[index].success = false;
                responses[index].error_message =
                    "Request and response IDs do not match.";
            }
        }

        return responses;
    } catch (const std::exception& exception) {
        ResponseBatch failed_responses;
        failed_responses.reserve(requests.size());

        for (const auto& request : requests) {
            InferenceResponse response;
            response.request_id = request.request_id;
            response.worker_id = worker_id_;
            response.success = false;
            response.error_message = exception.what();

            failed_responses.push_back(
                std::move(response)
            );
        }

        return failed_responses;
    }
}

int InferenceWorker::getWorkerId() const {
    return worker_id_;
}

}  // namespace sutra
