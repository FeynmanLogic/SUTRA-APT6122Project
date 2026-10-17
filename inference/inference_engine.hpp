/*
Written by Dhruv Kulkarni, 6122APT, last modified on Oct 17
*/

#pragma once

#include "../common/inference_types.hpp"

#include <string>

namespace sutra {

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    //this following function will load the model
    virtual bool loadModel(const std::string& model_path) = 0;
//this will generate one request
    virtual InferenceResponse generatesingle(
        const InferenceRequest& request
    ) = 0;
//this will generate a batch of requests, initially it will loop through requests, until Jayani/Ashish write the scheduler and router.
    virtual ResponseBatch generatefullBatch(
        const RequestBatch& requests
    ) {
        ResponseBatch responses;
        responses.reserve(requests.size());

        for (const auto& request : requests) {
            responses.push_back(generatesingle(request));
        }

        return responses;
    }
};

}  // namespace sutra has been defined for readability.