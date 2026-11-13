//Written by Dhruv Kulkarni, 6122APT, last modified on Nov 12


#include "benchmark_config.hpp"
#include "benchmark_runner.hpp"
#include "workload_generator.hpp"

#include "../inference/llama_engine.hpp"
#include "../worker/worker.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --model <model.gguf> [options]\n\n"

        << "Required:\n"
        << "  --model file.gguf\n\n"

        << "Inference options:\n"
        << "  --backend cpu|cuda\n"
        << "  --requests N\n"
        << "  --warmup N\n"
        << "  --concurrency N\n"
        << "  --batch-size N\n\n"

        << "Workload options:\n"
        << "  --prompt \"text\"\n"
        << "  --prompt-file file.txt\n"
        << "  --mixed\n\n"

        << "Benchmark output:\n"
        << "  --output file.csv\n"
        << "  --run-id name\n\n"

        << "GPU monitoring:\n"
        << "  --gpu-monitor\n"
        << "  --gpu-output file.csv\n"
        << "  --gpu-interval-ms N\n\n"

        << "Other:\n"
        << "  --help, -h\n";
}

void validateConfig(
    const sutra::BenchmarkConfig& config
) {
    if (config.model_path.empty()) {
        throw std::invalid_argument(
            "A model path is required."
        );
    }

    if (config.total_requests <= 0) {
        throw std::invalid_argument(
            "Request count must be positive."
        );
    }

    if (config.warmup_requests < 0) {
        throw std::invalid_argument(
            "Warm-up count cannot be negative."
        );
    }

    if (config.concurrency <= 0) {
        throw std::invalid_argument(
            "Concurrency must be positive."
        );
    }

    if (config.batch_size <= 0) {
        throw std::invalid_argument(
            "Batch size must be positive."
        );
    }

    if (config.gpu_sample_interval_ms <= 0) {
        throw std::invalid_argument(
            "GPU sampling interval must be positive."
        );
    }

    if (
        config.enable_gpu_monitor &&
        config.backend != sutra::Backend::CUDA
    ) {
        throw std::invalid_argument(
            "GPU monitoring requires the CUDA backend."
        );
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        sutra::BenchmarkConfig config;

        for (
            int index = 1;
            index < argc;
            ++index
        ) {
            const std::string argument =
                argv[index];

            auto nextValue =
                [&]() -> std::string {
                    if (index + 1 >= argc) {
                        throw std::runtime_error(
                            "Missing value for " +
                            argument
                        );
                    }

                    return argv[++index];
                };

            if (argument == "--model") {
                config.model_path =
                    nextValue();
            } else if (
                argument == "--backend"
            ) {
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
            } else if (
                argument == "--requests"
            ) {
                config.total_requests =
                    std::stoi(nextValue());
            } else if (
                argument == "--warmup"
            ) {
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
            } else if (
                argument == "--prompt"
            ) {
                config.prompt =
                    nextValue();
            } else if (
                argument == "--prompt-file"
            ) {
                config.prompt_file =
                    nextValue();
            } else if (
                argument == "--mixed"
            ) {
                config.mixed_workload = true;
            } else if (
                argument == "--output"
            ) {
                config.output_csv =
                    nextValue();
            } else if (
                argument == "--run-id"
            ) {
                config.run_id =
                    nextValue();
            } else if (
                argument == "--gpu-monitor"
            ) {
                config.enable_gpu_monitor = true;
            } else if (
                argument == "--gpu-output"
            ) {
                config.gpu_metrics_csv =
                    nextValue();
            } else if (
                argument == "--gpu-interval-ms"
            ) {
                config.gpu_sample_interval_ms =
                    std::stoi(nextValue());
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

        validateConfig(config);

        sutra::WorkloadGenerator generator(
            config.random_seed
        );

        sutra::RequestBatch requests;

        /*
         * Workload priority:
         *
         * 1. Direct prompt
         * 2. Prompt file
         * 3. Mixed synthetic workload
         * 4. Default uniform workload
         */
        if (!config.prompt.empty()) {
            requests =
                generator.fromPrompt(
                    config.prompt,
                    config.total_requests,
                    config.max_new_tokens
                );
        } else if (
            !config.prompt_file.empty()
        ) {
            requests =
                generator.fromPromptFile(
                    config.prompt_file,
                    config.total_requests,
                    config.max_new_tokens
                );
        } else if (
            config.mixed_workload
        ) {
            requests =
                generator.generateMixed(
                    config.total_requests,
                    config.max_new_tokens
                );
        } else {
            requests =
                generator.generateUniform(
                    config.total_requests,
                    config.approximate_prompt_tokens,
                    config.max_new_tokens
                );
        }

        /*
         * Zero GPU layers gives CPU-only execution.
         * A large value requests complete GPU offloading.
         */
        const int gpu_layers =
            config.backend ==
                    sutra::Backend::CUDA
                ? 99
                : 0;

        auto engine =
            std::make_unique<
                sutra::LlamaEngine
            >(
                gpu_layers,
                4096
            );

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

        sutra::BenchmarkRunner runner(
            worker
        );

        const sutra::BenchmarkSummary summary =
            runner.run(
                requests,
                config
            );

        std::cout
            << "\nBenchmark complete\n"
            << "Completed requests: "
            << summary.completed_requests
            << "\nFailed requests: "
            << summary.failed_requests
            << "\nWall time: "
            << summary.wall_time_seconds
            << " seconds"
            << "\nRequests/second: "
            << summary.requests_per_second
            << "\nTokens/second: "
            << summary.tokens_per_second
            << "\nMean latency: "
            << summary.mean_latency_ms
            << " ms"
            << "\nP50 latency: "
            << summary.p50_latency_ms
            << " ms"
            << "\nP95 latency: "
            << summary.p95_latency_ms
            << " ms"
            << "\nP99 latency: "
            << summary.p99_latency_ms
            << " ms"
            << "\nP50 TTFT: "
            << summary.p50_ttft_ms
            << " ms"
            << "\nP95 TTFT: "
            << summary.p95_ttft_ms
            << " ms"
            << "\nRequest CSV: "
            << config.output_csv;

        if (config.enable_gpu_monitor) {
            std::cout
                << "\nGPU CSV: "
                << config.gpu_metrics_csv;
        }

        std::cout << '\n';
    } catch (
        const std::exception& exception
    ) {
        std::cerr
            << "Benchmark failed: "
            << exception.what()
            << '\n';

        return 1;
    }

    return 0;
}