#pragma once

#include "inference_engine.hpp"

namespace sutra {

// Deterministic integration fixture; its timings are not inference measurements.
class MockInferenceEngine final : public InferenceEngine {
public:
    bool loadModel(const std::string&) override { return true; }
    InferenceResponse generatesingle(const InferenceRequest& request) override {
        InferenceResponse response;
        response.request_id = request.request_id;
        response.success = true;
        response.generated_text = "Synthetic response to: " + request.prompt;
        return response;
    }
};

}  // namespace sutra
