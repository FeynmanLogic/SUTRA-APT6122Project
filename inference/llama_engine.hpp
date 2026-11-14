/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10
*/
#pragma once

#include "inference_engine.hpp"

#include "llama.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace sutra {

class LlamaEngine final : public InferenceEngine {
public:
    explicit LlamaEngine(
        int gpu_layers = 99,
        std::uint32_t context_size = 4096
    );

    ~LlamaEngine() override;

    LlamaEngine(const LlamaEngine&) = delete;
    LlamaEngine& operator=(const LlamaEngine&) = delete;

    bool loadModel(const std::string& model_path) override;

    InferenceResponse generatesingle(
        const InferenceRequest& request
    ) override;

private:
    static void setBatchTokens(
        llama_batch_ext* batch,
        const llama_token* tokens,
        std::int32_t token_count,
        llama_pos starting_position
    );
    std::string formatPrompt(
    const std::string& prompt
) const;

    std::string tokenToText(llama_token token) const;

    void releaseResources();

    int gpu_layers_;
    std::uint32_t context_size_;

    llama_model* model_{nullptr};
    llama_context* context_{nullptr};
    const llama_vocab* vocabulary_{nullptr};

    bool backend_initialized_{false};
    std::mutex inference_mutex_;
};

}  // namespace sutra