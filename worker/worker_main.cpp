#include "../inference/llama_engine.hpp"
#include "worker.hpp"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr
            << "Usage: sutra_worker <model.gguf> [prompt]\n";
        return 1;
    }

    const std::string model_path = argv[1];

    const std::string prompt =
        argc >= 3
        ? argv[2]
        : "Explain CUDA in one sentence.";

    auto engine =
        std::make_unique<sutra::LlamaEngine>(
            99,
            4096
        );

    if (!engine->loadModel(model_path)) {
        std::cerr << "Failed to load model.\n";
        return 1;
    }

    sutra::InferenceWorker worker(
        0,
        std::move(engine)
    );

    sutra::RequestBatch requests;

    sutra::InferenceRequest request;
    request.request_id = 1;
    request.prompt = prompt;
    request.max_new_tokens = 64;
    request.priority = 0;
    request.arrival_time_ms = 0;
 sutra::InferenceRequest request2;
    request2.request_id = 2;
    request2.prompt = "Who am I";
    request2.max_new_tokens = 64;
    request2.priority = 1;
    request2.arrival_time_ms = 1;
 sutra::InferenceRequest request3;
    request3.request_id = 3;
    request3.prompt = "Who are you";
    request3.max_new_tokens = 64;
    request3.priority = 2;
    request3.arrival_time_ms = 2;
    requests.push_back(request);
    requests.push_back(request2);
    requests.push_back(request3);
//insert three requests to check whether naive batch processing works or not

    const sutra::ResponseBatch responses =
        worker.processBatch(requests);

    for (const auto& response : responses) {
        if (!response.success) {
            std::cerr
                << "Request "
                << response.request_id
                << " failed: "
                << response.error_message
                << '\n';

            continue;
        }

        std::cout
            << "\nRequest ID: "
            << response.request_id
            << "\nWorker ID: "
            << response.worker_id
            << "\nGenerated text:\n"
            << response.generated_text
            << "\n\nInput tokens: "
            << response.input_tokens
            << "\nOutput tokens: "
            << response.generated_tokens
            << "\nTTFT: "
            << response.time_to_first_token_ms
            << " ms\nInference time: "
            << response.inference_time_ms
            << " ms\nThroughput: "
            << response.tokens_per_second
            << " tokens/s\n";
    }

    return 0;
}