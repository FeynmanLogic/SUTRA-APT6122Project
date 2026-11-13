/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10
*/


#pragma once

#include "../common/inference_types.hpp"
#include "../inference/inference_engine.hpp"

#include <memory>

namespace sutra {

class InferenceWorker {
public:
    InferenceWorker(
        int worker_id,
        std::unique_ptr<InferenceEngine> engine
    );

    ResponseBatch processBatch(
        const RequestBatch& requests
    );

    int getWorkerId() const;

private:
    int worker_id_;
    std::unique_ptr<InferenceEngine> engine_;
};

}  // namespace sutra