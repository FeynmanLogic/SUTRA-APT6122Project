# Sutra

**Sutra** is a multithreaded and distributed C++ framework for efficient LLM inference. It uses CUDA and multiple GPU workers to improve inference performance while supporting concurrent requests, batching, scheduling, load balancing, monitoring, and fault handling.

## Team Responsibilities

### Dhruv Kulkarni: LLM Inference and Benchmarking

- Integrate llama.cpp with the C++ application.
- Run a small pretrained LLM on CPU and GPU.
- Receive batches of requests from the scheduler.
- Execute the model and return generated tokens.
- Measure GPU utilization, latency, and throughput.
- Build a client that generates concurrent requests.
- Test execution using multiple worker processes.

### Aashish Rahate: C++ Scheduler

- Implement FIFO and priority scheduling.
- Build configurable batching policies.
- Enforce maximum batch size and waiting time.
- Implement round-robin and least-loaded worker selection.
- Prevent starvation of low-priority requests.
- Compare scheduling policies experimentally.

### Jayani Malik: C++ Router

- Build a TCP server using C++ sockets.
- Handle multiple clients using C++ threads.
- Maintain thread-safe queues and unique request IDs.
- Forward scheduled requests to available workers.
- Return responses to the correct clients.
- Detect disconnected workers and report failures.

## Planned Architecture

```text
Benchmark Client
      |
      v
C++ Router
      |
      v
Scheduler and Batch Builder
      |
      v
Inference Worker Processes
      |
      v
llama.cpp with CPU or CUDA
      |
      v
Generated Responses and Metrics
```

## Repository Structure

```text
sutra/
├── benchmark/
│   ├── benchmark_client.cpp
│   └── workload_generator.cpp
├── common/
│   └── inference_types.hpp
├── inference/
│   ├── inference_engine.hpp
│   ├── llama_engine.cpp
│   └── llama_engine.hpp
├── monitoring/
│   ├── gpu_monitor.cpp
│   ├── metrics.cpp
│   └── metrics.hpp
├── tests/
│   └── test_worker.cpp
├── worker/
│   ├── worker.cpp
│   ├── worker.hpp
│   └── worker_main.cpp
├── CMakeLists.txt
├── .gitignore
└── README.md
```

## Module Overview

### `common`

Contains the shared request and response structures used by the router, scheduler, benchmark client, and inference workers.

### `inference`

Contains the common inference interface and the llama.cpp implementation. This module loads the model, tokenizes prompts, generates tokens, and returns inference results.

### `worker`

Contains the inference-worker implementation. Each worker receives a batch, executes its requests, preserves request IDs, records performance, and returns the responses.

### `benchmark`

Contains the workload generator and concurrent benchmark client. It will test different concurrency levels, batch sizes, prompt lengths, and worker counts.

### `monitoring`

Records inference latency, time to first token, token throughput, request throughput, failures, GPU utilization, and GPU memory usage.

### `tests`

Contains unit tests and integration tests for inference and worker execution.

## Proposed Integration Contract

```cpp
struct InferenceRequest {
    std::uint64_t request_id;
    std::string prompt;
    int max_new_tokens;
    int priority;
    std::int64_t arrival_time_ms;
};

struct InferenceResponse {
    std::uint64_t request_id;
    std::string generated_text;
    int generated_tokens;
    double latency_ms;
    double time_to_first_token_ms;
    double tokens_per_second;
    bool success;
    std::string error_message;
};
```

The scheduler will send a `std::vector<InferenceRequest>` to an available worker. The worker will return a corresponding `std::vector<InferenceResponse>`. The team should confirm these fields before integration.

## Development Plan

1. Confirm the common request and response interface.
2. Build llama.cpp and run a small GGUF model on CPU.
3. Enable CUDA and verify GPU inference.
4. Implement the inference-engine abstraction.
5. Implement the llama.cpp engine.
6. Implement worker batch processing.
7. Add latency, throughput, and GPU monitoring.
8. Build the concurrent benchmark client.
9. Test multiple worker processes.
10. Integrate the worker with the scheduler and router.

## Batching Note

Passing several requests to a worker is request aggregation, but it is not necessarily true GPU batching. True batched inference requires multiple sequences to be evaluated together. Continuous batching additionally allows new requests to enter while existing requests are still generating.

The first milestone is therefore a correct single-request inference path. Request aggregation, true batching, and continuous batching can be added incrementally.

## Git Workflow

Each team member should work on a separate branch:

- `dhruv-inference`
- `aashish-scheduler`
- `jayani-router`

Create a branch using:

```bash
git switch -c branch-name
```

Open a pull request before merging major changes into `main`.

## Schedule

Dhruv plans to complete the inference, monitoring, and benchmarking components either during the weekend of **October 17 to 18, 2026**, or, if additional CUDA and integration testing is required, during the weekend of **October 24 to 25, 2026**.

## Current Status

The initial repository structure has been created. Source code, dependency setup, build instructions, and benchmark results will be added as implementation progresses.
