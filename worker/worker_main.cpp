
/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10
*/
#include "../common/inference_types.hpp"
#include "../inference/llama_engine.hpp"

#include <iostream>
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

    sutra::LlamaEngine engine(99, 4096);

    if (!engine.loadModel(model_path)) {
        std::cerr << "Failed to load model.\n";
        return 1;
    }

    sutra::InferenceRequest request;
    request.request_id = 1;
    request.prompt = prompt;
    request.max_new_tokens = 64;

    const sutra::InferenceResponse response =
        engine.generatesingle(request);

    if (!response.success) {
        std::cerr
            << "Inference failed: "
            << response.error_message
            << '\n';
        return 1;
    }

    std::cout
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

    return 0;
}