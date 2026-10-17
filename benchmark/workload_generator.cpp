//this file will actually elaborate on the functions for generation.

#include "workload_generator.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace sutra {

WorkloadGenerator::WorkloadGenerator(std::uint64_t seed)
    : random_engine_(seed) {}

InferenceRequest WorkloadGenerator::createRequest(
    const std::string& prompt,
    int max_new_tokens,
    int priority
) {
    InferenceRequest request;

    request.request_id = next_request_id_++;
    request.prompt = prompt;
    request.max_new_tokens = max_new_tokens;
    request.priority = priority;

    // BenchmarkRunner sets this immediately before submission.
    request.arrival_time_ms = 0;

    return request;
}

std::string WorkloadGenerator::createDefaultPrompt(
    int approximate_tokens
) {
    static const std::vector<std::string> sentences = {
        "Parallel computing divides a large task into smaller operations.",
        "Modern GPUs execute many arithmetic operations simultaneously.",
        "Efficient scheduling improves throughput and request latency.",
        "Language models generate text by repeatedly predicting tokens.",
        "Distributed workers process independent requests concurrently.",
        "Dynamic batching can improve hardware utilization."
    };

    const int target_characters =
        std::max(1, approximate_tokens) * 4;

    std::uniform_int_distribution<std::size_t> distribution(
        0,
        sentences.size() - 1
    );

    std::string prompt;

    while (static_cast<int>(prompt.size()) < target_characters) {
        prompt += sentences[distribution(random_engine_)];
        prompt += ' ';
    }

    prompt += "Summarize the preceding information clearly.";

    return prompt;
}

RequestBatch WorkloadGenerator::fromPrompt(
    const std::string& prompt,
    int request_count,
    int max_new_tokens,
    int priority
) {
    if (prompt.empty()) {
        throw std::invalid_argument("The input prompt cannot be empty.");
    }

    RequestBatch requests;
    requests.reserve(request_count);

    for (int index = 0; index < request_count; ++index) {
        requests.push_back(
            createRequest(prompt, max_new_tokens, priority)
        );
    }

    return requests;
}

RequestBatch WorkloadGenerator::fromPromptFile(
    const std::string& file_path,
    int request_count,
    int max_new_tokens,
    int priority
) {
    std::ifstream input(file_path);

    if (!input) {
        throw std::runtime_error(
            "Unable to open prompt file: " + file_path
        );
    }

    std::vector<std::string> prompts;
    std::string line;

    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!line.empty()) {
            prompts.push_back(line);
        }
    }

    if (prompts.empty()) {
        throw std::runtime_error(
            "Prompt file contains no non-empty prompts."
        );
    }

    RequestBatch requests;
    requests.reserve(request_count);

    for (int index = 0; index < request_count; ++index) {
        const std::string& selected_prompt =
            prompts[index % prompts.size()];

        requests.push_back(
            createRequest(
                selected_prompt,
                max_new_tokens,
                priority
            )
        );
    }

    return requests;
}

RequestBatch WorkloadGenerator::generateUniform(
    int request_count,
    int approximate_prompt_tokens,
    int max_new_tokens,
    int priority
) {
    RequestBatch requests;
    requests.reserve(request_count);

    for (int index = 0; index < request_count; ++index) {
        requests.push_back(
            createRequest(
                createDefaultPrompt(approximate_prompt_tokens),
                max_new_tokens,
                priority
            )
        );
    }

    return requests;
}

RequestBatch WorkloadGenerator::generateMixed(
    int request_count,
    int max_new_tokens
) {
    static constexpr int prompt_lengths[] = {32, 128, 512};
    static constexpr int priorities[] = {0, 0, 0, 1, 1, 2};

    std::uniform_int_distribution<int> length_distribution(0, 2);
    std::uniform_int_distribution<int> priority_distribution(0, 5);

    RequestBatch requests;
    requests.reserve(request_count);

    for (int index = 0; index < request_count; ++index) {
        const int prompt_length =
            prompt_lengths[length_distribution(random_engine_)];

        const int priority =
            priorities[priority_distribution(random_engine_)];

        requests.push_back(
            createRequest(
                createDefaultPrompt(prompt_length),
                max_new_tokens,
                priority
            )
        );
    }

    return requests;
}

}  // namespace sutra