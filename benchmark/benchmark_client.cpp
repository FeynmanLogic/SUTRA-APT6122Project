//Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10


#include "benchmark_config.hpp"
#include "benchmark_runner.hpp"
#include "workload_generator.hpp"

#include "../inference/llama_engine.hpp"
#include "../worker/worker.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << program
        << " --model <model.gguf> [options]\n\n"
        << "Options:\n"
        << "  --backend cpu|cuda\n"
        << "  --requests N\n"
        << "  --warmup N\n"
        << "  --concurrency N\n"
        << "  --batch-size N\n"
        << "  --prompt \"text\"\n"
        << "  --prompt-file file.txt\n"
        << "  --mixed\n"
        << "  --output file.csv\n"
        << "  --run-id name\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        sutra::BenchmarkConfig config;

        for (int index = 1;
             index < argc;
             ++index) {
            const std::string argument =
                argv[index];

            auto nextValue = [&]() -> std::string {
                if (index + 1 >= argc) {
                    throw std::runtime_error(
                        "Missing value for " +
                        argument
                    );
                }

                return argv[++index];
            };

            if (argument == "--model") {
                config.model_path = nextValue();
            } else if (argument == "--backend") {
                const std::string backend =
                    nextValue();

                if (backend == "cpu") {
                    config.backend =
                        sutra::Backend::CPU;
                } else if (backend == "cuda") {
                    config.backend =
                        sutra::Backend::CUDA;
                } else {
                    throw std::runtime_error(
                        "Backend must be cpu or cuda."
                    );
                }
            } else if (argument == "--requests") {
                config.total_requests =
                    std::stoi(nextValue());
            } else if (argument == "--warmup") {
                config.warmup_requests =
                    std::stoi(nextValue());
            } else if (
                argument == "--concurrency"
            ) {
                config.concurrency =
                    std::stoi(nextValue());
            } else if (
                argument == "--batch-size"
            ) {
                config.batch_size =
                    std::stoi(nextValue());
            } else if (argument == "--prompt") {
                config.prompt = nextValue();
            } else if (
                argument == "--prompt-file"
            ) {
                config.prompt_file =
                    nextValue();
            } else if (argument == "--mixed") {
                config.mixed_workload = true;
            } else if (argument == "--output") {
                config.output_csv = nextValue();
            } else if (argument == "--run-id") {
                config.run_id = nextValue();
            } else if (
                argument == "--help" ||
                argument == "-h"
            ) {
                printUsage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error(
                    "Unknown argument: " +
                    argument
                );
            }
        }

        if (config.model_path.empty()) {
            printUsage(argv[0]);
            return 1;
        }

        sutra::WorkloadGenerator generator(
            config.random_seed
        );

        sutra::RequestBatch requests;

        if (!config.prompt.empty()) {
            requests = generator.fromPrompt(
                config.prompt,
                config.total_requests,
                config.max_new_tokens
            );
        } else if (!config.prompt_file.empty()) {
            requests = generator.fromPromptFile(
                config.prompt_file,
                config.total_requests,
                config.max_new_tokens
            );
        } else if (config.mixed_workload) {
            requests = generator.generateMixed(
                config.total_requests,
                config.max_new_tokens
            );
        } else {
            requests = generator.generateUniform(
                config.total_requests,
                config.approximate_prompt_tokens,
                config.max_new_tokens
            );
        }

        const int gpu_layers =
            config.backend ==
                    sutra::Backend::CUDA
                ? 99
                : 0;

        auto engine =
            std::make_unique<
                sutra::LlamaEngine
            >(gpu_layers, 4096);

        if (
            !engine->loadModel(
                config.model_path
            )
        ) {
            throw std::runtime_error(
                "Failed to load model."
            );
        }

        sutra::InferenceWorker worker(
            0,
            std::move(engine)
        );

        sutra::BenchmarkRunner runner(worker);

        const sutra::BenchmarkSummary summary =
            runner.run(requests, config);

        std::cout
            << "\nBenchmark complete\n"
            << "Completed requests: "
            << summary.completed_requests
            << "\nFailed requests: "
            << summary.failed_requests
            << "\nWall time: "
            << summary.wall_time_seconds
            << " seconds\nRequests/second: "
            << summary.requests_per_second
            << "\nTokens/second: "
            << summary.tokens_per_second
            << "\nMean latency: "
            << summary.mean_latency_ms
            << " ms\nP50 latency: "
            << summary.p50_latency_ms
            << " ms\nP95 latency: "
            << summary.p95_latency_ms
            << " ms\nP99 latency: "
            << summary.p99_latency_ms
            << " ms\nP50 TTFT: "
            << summary.p50_ttft_ms
            << " ms\nP95 TTFT: "
            << summary.p95_ttft_ms
            << " ms\nCSV: "
            << config.output_csv
            << '\n';
    } catch (const std::exception& exception) {
        std::cerr
            << "Benchmark failed: "
            << exception.what()
            << '\n';

        return 1;
    }

    return 0;
}