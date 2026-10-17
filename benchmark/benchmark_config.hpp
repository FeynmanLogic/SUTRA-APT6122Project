/*
Written by Dhruv Kulkarni, 6122APT, last modified on Oct 17
This file will define how benchmark requests are to be handled.
*/


#pragma once

#include <cstdint>
#include <string>

namespace sutra {

enum class Backend {
    CPU,
    CUDA
};
//two options each for the backend and scheduling policy.
enum class SchedulingPolicy {
    FIFO,
    PRIORITY
};

struct BenchmarkConfig {
    std::string run_id{"default"};
    std::string model_path;

    Backend backend{Backend::CUDA};
    SchedulingPolicy scheduling_policy{SchedulingPolicy::FIFO};

    int total_requests{30};
    //have defined sensible defaults
    int warmup_requests{5};
    int concurrency{1};
    int batch_size{1};
    int worker_count{1};
    std::string prompt;
std::string prompt_file;

bool mixed_workload{false}; 
    int approximate_prompt_tokens{128};
    int max_new_tokens{64};

    std::uint64_t random_seed{42};

    std::string output_csv{"results/requests.csv"};
};

inline std::string toString(Backend backend) {
    return backend == Backend::CUDA ? "cuda" : "cpu";
}
//ternary operator, use CUDA if defined else use CPU.

inline std::string toString(SchedulingPolicy policy) {
    return policy == SchedulingPolicy::PRIORITY
        ? "priority"
        : "fifo";
}
//ternary operator, use priority scheduling if defined else use FIFO.

}  // namespace sutra