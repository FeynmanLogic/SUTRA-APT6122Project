# Sutra

**Sutra** is a multithreaded and distributed C++ framework for efficient LLM inference. It uses CUDA and multiple GPU workers to improve inference performance while supporting concurrent requests, batching, scheduling, load balancing, monitoring, and fault handling.

Planning and execution: [delivery plan and checklist](PROJECT_DELIVERY_PLAN.md), [step-by-step implementation guide](IMPLEMENTATION_GUIDE.md), and [PACE-ICE precheck/demo notebook](notebooks/Sutra_PACE_ICE_Demo.ipynb). The notebook can probe the environment and run an optional llama.cpp baseline now; the Sutra runtime/demo stages remain pending implementation.

## Team Integration Guide

Sutra is divided into three independently developed subsystems:

| Subsystem | Owner | Primary responsibility |
|---|---|---|
| Router | Jayani | Accept clients, assign request IDs, serialize messages, and return responses |
| Scheduler | Aashish | Queue requests, apply scheduling policies, construct batches, and select workers |
| Inference and benchmarking | Dhruv | Load the model, execute batches, collect metrics, and benchmark performance |

### Shared request contract

All components must use the types declared in:

```text
common/inference_types.hpp
```

The logical request and response structures are:

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
    int worker_id;

    std::string generated_text;
    int input_tokens;
    int generated_tokens;

    double queue_time_ms;
    double inference_time_ms;
    double time_to_first_token_ms;
    double latency_ms;
    double tokens_per_second;

    bool success;
    std::string error_message;
};
```

`arrival_time_ms` means Unix-epoch time in milliseconds and should be assigned when the router accepts a request.

The TCP router must serialize these fields into a defined wire format. It must not transmit the raw in-memory bytes of the C++ structures because objects such as `std::string` contain process-local memory addresses.

### Request flow

```text
Client
  → Router creates InferenceRequest
  → Scheduler queues and batches requests
  → Scheduler selects an available worker
  → Worker calls LlamaEngine
  → LlamaEngine generates tokens through llama.cpp
  → Worker returns InferenceResponse
  → Router matches request_id and replies to the client
```

The value of `request_id` must remain unchanged throughout this complete path.

### Inference interface

The worker accesses model execution through the `InferenceEngine` interface:

```cpp
/*
Written by Dhruv Kulkarni, 6122APT, last modified on Oct 17
*/

#pragma once

#include "../common/inference_types.hpp"

#include <string>

namespace sutra {

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    //this following function will load the model
    virtual bool loadModel(const std::string& model_path) = 0;
//this will generate one request
    virtual InferenceResponse generatesingle(
        const InferenceRequest& request
    ) = 0;
//this will generate a batch of requests, initially it will loop through requests, until Jayani/Ashish write the scheduler and router.
    virtual ResponseBatch generatefullBatch(
        const RequestBatch& requests
    ) {
        ResponseBatch responses;
        responses.reserve(requests.size());

        for (const auto& request : requests) {
            responses.push_back(generatesingle(request));
        }

        return responses;
    }
};

}  // namespace sutra has been defined for readability.
```

Two implementations are planned:

- `MockInferenceEngine` returns deterministic fake responses for scheduler, router, and worker tests.
- `LlamaEngine` loads a GGUF model once and performs real inference through llama.cpp.

The router and scheduler must not call llama.cpp directly.

### Router integration instructions

The router is responsible for:

1. Assigning a unique `request_id`.
2. Recording `arrival_time_ms`.
3. Reading the prompt, token limit, and priority.
4. Serializing the request for transfer.
5. Sending the request to the scheduler.
6. Matching the returned response by `request_id`.
7. Returning generated text or an error to the correct client.

The router should not calculate model-execution time. It may calculate client-observed end-to-end latency.

### Scheduler integration instructions

The scheduler receives individual `InferenceRequest` objects and produces a `RequestBatch`:

```cpp
using RequestBatch = std::vector<InferenceRequest>;
```

The scheduler is responsible for:

1. FIFO or priority queueing.
2. Enforcing maximum batch size.
3. Enforcing maximum batch waiting time.
4. Preventing indefinite low-priority starvation.
5. Selecting a worker using round-robin or least-loaded selection.
6. Sending the complete batch to the selected worker.

The scheduler must not modify request IDs or prompts.

Initially, a batch may be processed sequentially by the worker. This is request aggregation, not true batched GPU inference. True multi-sequence and continuous batching are later optimizations.

### Worker integration instructions

Each worker owns:

- One worker ID
- One `InferenceEngine`
- One loaded model
- One llama.cpp context or managed context set
- One assigned GPU device

The worker exposes the logical operation:

```cpp
ResponseBatch processBatch(const RequestBatch& requests);
```

For every input request, the worker must return exactly one response with the same request ID. An inference failure should produce `success = false` and an error message instead of terminating the worker process.

A single llama.cpp context must not be called concurrently by unrelated threads unless synchronization or independent contexts are provided.

### Metrics ownership

The worker records:

- Input-token count
- Output-token count
- Queue time
- Inference time
- Time to first token
- Request latency
- Generated tokens per second
- Worker ID
- Success or failure

The benchmark runner calculates:

- Completed and failed request counts
- Requests per second
- Aggregate output tokens per second
- Mean latency
- Median latency
- P95 and P99 latency
- Median and P95 time to first token

Aggregate token throughput must be calculated as total generated tokens divided by total benchmark wall time. It must not be calculated by averaging per-request token rates.

### Repository workflow

Clone the project and its llama.cpp dependency with:

```bash
git clone --recurse-submodules \
  https://github.com/FeynmanLogic/SUTRA-APT6122Project.git
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Team members should avoid editing files owned by another subsystem unless the change is coordinated. Changes to `common/inference_types.hpp` affect every subsystem and must be communicated to the complete team.

Recommended branches are:

```text
dhruv-inference
aashish-scheduler
jayani-router
```

The model files, generated binaries, and raw benchmark output are not stored in Git. They are excluded through `.gitignore`.

### Current status

The following setup has been verified on PACE ICE:

- NVIDIA Tesla V100 16 GB
- CUDA 12.6
- llama.cpp with CUDA support
- Qwen2.5 0.5B Instruct using Q4_K_M quantization
- CPU and GPU llama.cpp microbenchmarks

The standalone scheduler is implemented and tested locally. It supports bounded admission, FIFO/priority with aging, count/timer batches, round-robin/least-loaded worker selection, capacity accounting, deadlines, heartbeat expiry and shutdown. Router/worker wiring, real inference and end-to-end benchmarking remain pending. Jayani's initial TCP echo router is on `feature/tcp-router` and has not been merged into this branch. See [the scheduler handoff](PROJECT_DELIVERY_PLAN.md#10-scheduler-delivery-and-teammate-handoff--28-october-2026) for the API and owner checklists.

Build and exercise the scheduler without CUDA, a model, or the llama.cpp submodule:

```bash
cmake -S . -B build-scheduler -DCMAKE_BUILD_TYPE=Release -DSUTRA_ENABLE_LLAMA=OFF
cmake --build build-scheduler --parallel 4
ctest --test-dir build-scheduler --output-on-failure --no-tests=error
./build-scheduler/sutra_scheduler_demo
```

The demo uses synthetic replies and does not measure inference or networking performance. `SUTRA_ENABLE_LLAMA=ON` explicitly fails until the worker/backend integration is implemented.
