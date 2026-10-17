/*
Written by Dhruv Kulkarni, 6122APT, last modified on Oct 17
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sutra {

struct InferenceRequest {
    std::uint64_t request_id{0};
    std::string prompt;
    int max_new_tokens{64};
    int priority{0};

    // Unix epoch milliseconds, assigned by the client/router.
    std::int64_t arrival_time_ms{0};
};

struct InferenceResponse {
    std::uint64_t request_id{0};
    int worker_id{-1};

    std::string generated_text;
    int input_tokens{0};
    int generated_tokens{0};
//these metrics will be really useful in benchmarking and actively checking contracts
    double queue_time_ms{0.0};
    double inference_time_ms{0.0};
    double time_to_first_token_ms{0.0};
    double latency_ms{0.0};
    double tokens_per_second{0.0};

    bool success{false};
    std::string error_message;
};

using RequestBatch = std::vector<InferenceRequest>;
using ResponseBatch = std::vector<InferenceResponse>;

}  // namespace sutra