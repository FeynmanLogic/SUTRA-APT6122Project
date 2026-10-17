
/*
Written by Dhruv Kulkarni, 6122APT, last modified on Oct 17
This file will define how workload is generated.
*/

#pragma once

#include "../common/inference_types.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sutra {

class WorkloadGenerator {
public:
    explicit WorkloadGenerator(std::uint64_t seed);

    RequestBatch fromPrompt(
        const std::string& prompt,
        int request_count,
        int max_new_tokens,
        int priority = 0
    );

    RequestBatch fromPromptFile(
        const std::string& file_path,
        int request_count,
        int max_new_tokens,
        int priority = 0
    );

    RequestBatch generateUniform(
        int request_count,
        int approximate_prompt_tokens,
        int max_new_tokens,
        int priority = 0
    );

    RequestBatch generateMixed(
        int request_count,
        int max_new_tokens
    );

private:
    InferenceRequest createRequest(
        const std::string& prompt,
        int max_new_tokens,
        int priority
    );

    std::string createDefaultPrompt(int approximate_tokens);

    std::mt19937_64 random_engine_;
    std::uint64_t next_request_id_{1};
};

}  // namespace sutra