# Sutra

Sutra is a multithreaded and distributed C++ framework for efficient LLM inference. It integrates [llama.cpp](https://github.com/ggml-org/llama.cpp) for CUDA-accelerated model execution and provides interfaces for request routing, scheduling, batching, worker execution, benchmarking, and GPU monitoring.

Integration status (14 November 2026): the scheduler, current `main` inference/benchmark code, and TCP-router branch are combined on `integration/scheduler-router-inference`. See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for local builds, PACE GPU validation, the wire protocol, and Git publishing steps. Earlier milestones remain in [PROJECT_DELIVERY_PLAN.md](PROJECT_DELIVERY_PLAN.md) and [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## Quick local checks

The core build requires no model, CUDA, or downloaded dependencies:

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release
cmake --build build-core --parallel 4
ctest --test-dir build-core --output-on-failure --no-tests=error
```

Enable the TCP integration test with `-DSUTRA_BUILD_ROUTER=ON`. This requires nlohmann_json >= 3.11; CMake uses an installed package or downloads v3.11.3. See the integration guide for offline configuration and Windows compiler setup.

## Team

| Subsystem | Owner | Primary responsibility |
|---|---|---|
| Router | Jayani Malik | Accept clients, assign request IDs, serialize messages, and return responses |
| Scheduler | Aashish Rahate | Queue requests, apply scheduling policies, construct batches, and select workers |
| Inference and benchmarking | Dhruv Kulkarni | Load models, execute requests, collect metrics, monitor GPUs, and benchmark performance |

## Architecture

```text
Client
  |
  v
TCP Router
  |
  v
Request Queue and Scheduler
  |
  v
Inference Worker
  |
  v
LlamaEngine and llama.cpp
  |
  v
NVIDIA GPU
```

The response follows the reverse path. The original `request_id` must remain unchanged throughout the complete request lifecycle.

## Repository Structure

```text
SUTRA-APT6122Project/
├── benchmark/       Benchmark configuration, workload generation, and runner
├── common/          Shared request and response structures
├── inference/       Inference interface and llama.cpp implementation
├── monitoring/      GPU utilization, memory, temperature, and power monitoring
├── worker/          Inference worker abstraction
├── external/
│   └── llama.cpp/   llama.cpp Git submodule
├── tests/           Project tests
└── CMakeLists.txt
```

## Shared Request Contract

All subsystems use the request and response types declared in:

```text
common/inference_types.hpp
```

The logical structures are:

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

`arrival_time_ms` is Unix epoch time in milliseconds and should be assigned when the router accepts a request.

The router must serialize these fields into an explicitly defined wire format. It must not transmit the raw bytes of these structures because objects such as `std::string` contain process-local memory addresses.

## Inference Interface

Workers access model execution through the `InferenceEngine` interface:

```cpp
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    virtual bool loadModel(
        const std::string& model_path
    ) = 0;

    virtual InferenceResponse generatesingle(
        const InferenceRequest& request
    ) = 0;

    virtual ResponseBatch generatefullBatch(
        const RequestBatch& requests
    ) {
        ResponseBatch responses;
        responses.reserve(requests.size());

        for (const auto& request : requests) {
            responses.push_back(
                generatesingle(request)
            );
        }

        return responses;
    }
};
```

`LlamaEngine` implements this interface using llama.cpp. It loads the model once, applies the model's chat template, tokenizes the prompt, generates output tokens, and records performance metrics.

The current default batch implementation processes aggregated requests sequentially. This is request batching at the worker interface, not simultaneous multi-sequence GPU inference.

## Integration Responsibilities

### Router

The router is responsible for:

1. Assigning a unique `request_id`.
2. Recording `arrival_time_ms`.
3. Reading the prompt, output-token limit, and priority.
4. Serializing requests for network transfer.
5. Sending requests to the scheduler.
6. Matching returned responses using `request_id`.
7. Returning generated text or an error to the correct client.

The router may calculate client-observed end-to-end latency, but it should not calculate internal model-execution time.

### Scheduler

The scheduler receives individual requests and creates batches:

```cpp
using RequestBatch =
    std::vector<InferenceRequest>;
```

The scheduler is responsible for:

1. FIFO or priority-based queueing.
2. Enforcing a maximum batch size.
3. Enforcing a maximum batch waiting time.
4. Preventing indefinite low-priority starvation.
5. Selecting an available worker.
6. Sending the complete batch to that worker.

The scheduler must not change request IDs or prompt contents.

### Worker

Each worker owns:

- One worker ID
- One inference engine
- One loaded model
- One llama.cpp context
- One assigned GPU device

The worker exposes:

```cpp
ResponseBatch processBatch(
    const RequestBatch& requests
);
```

For each input request, the worker must return exactly one response with the same request ID.

Inference failures should return `success = false` and an explanatory `error_message`. A failed request should not terminate the worker process.

A single llama.cpp context is protected from concurrent access. True parallel model execution requires independent worker processes or separately managed contexts.

## Metrics

### Per-request metrics

The inference and worker layers record:

- Input tokens
- Generated tokens
- Queue time
- Inference time
- Time to first token
- Total request latency
- Generated tokens per second
- Worker ID
- Success or failure

### Aggregate benchmark metrics

The benchmark runner calculates:

- Completed requests
- Failed requests
- Requests per second
- Aggregate generated tokens per second
- Mean latency
- Median latency
- P95 latency
- P99 latency
- Median time to first token
- P95 time to first token

Aggregate token throughput is calculated as:

```text
total generated tokens / benchmark wall time
```

It is not calculated by averaging individual request token rates.

## GPU Monitoring

The benchmark runner can launch a background GPU monitor during the measured portion of an experiment.

The monitor records:

- GPU utilization
- GPU memory usage
- Total GPU memory
- GPU temperature
- GPU power consumption
- Unix timestamp
- Elapsed benchmark time

GPU statistics are collected through `nvidia-smi` and written to a separate CSV file.

Example output:

```csv
unix_time_ms,elapsed_ms,gpu_index,gpu_name,gpu_utilization_percent,memory_used_mib,memory_total_mib,temperature_c,power_draw_w
1794496693002,0.000,0,Tesla V100-PCIE-16GB,81,1086,16384,47,86.91
```

## Cloning the Repository

Clone the repository together with its llama.cpp dependency:

```bash
git clone --recurse-submodules \
  https://github.com/FeynmanLogic/SUTRA-APT6122Project.git

cd SUTRA-APT6122Project
```

If the repository was cloned without its submodules:

```bash
git submodule update --init --recursive
```

## PACE ICE Environment

The implementation has been tested with:

- NVIDIA Tesla V100-PCIE-16GB
- CUDA 12.6
- GCC 12.3
- CMake 3.31
- Qwen2.5-0.5B-Instruct Q4_K_M

Load the required modules:

```bash
module load gcc/12.3.0
module load cuda/12.6.1
```

Confirm that the GPU and CUDA compiler are available:

```bash
nvidia-smi
nvcc --version
```

The CUDA module must be loaded both when building and when running the executables. Otherwise, shared libraries such as `libcudart.so.12` and `libcublas.so.12` may not be found.

## Downloading the Model

Model files are not stored in Git.

Download the tested GGUF model with:

```bash
mkdir -p models

curl -L --fail --progress-bar \
  -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

## Building Sutra

Configure the project for the V100 GPU, which has CUDA compute capability 7.0:

```bash
cmake -S . -B build-sutra \
  -DSUTRA_ENABLE_LLAMA=ON \
  -DSUTRA_BUILD_ROUTER=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release
```

Build the project:

```bash
cmake --build build-sutra -j
```

## Running Inference

Run the worker or inference executable using the downloaded model:

```bash
./build-sutra/sutra_worker \
  models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

This executable is a standalone worker smoke test. For TCP requests through the scheduler, run `sutra_router --backend llama --model <model.gguf> --gpu-layers 99`; see the integration guide.

## Running a Benchmark

The benchmark program accepts custom workloads and also provides a generated default workload.

Example CUDA benchmark:

```bash
mkdir -p results

./build-sutra/sutra_benchmark \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --backend cuda \
  --requests 20 \
  --warmup 3 \
  --concurrency 1 \
  --batch-size 1 \
  --prompt "Explain distributed computing briefly." \
  --output results/request_metrics.csv \
  --run-id cuda-test
```

### GPU-monitored benchmark

```bash
./build-sutra/sutra_benchmark \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --backend cuda \
  --requests 20 \
  --warmup 3 \
  --concurrency 1 \
  --batch-size 1 \
  --prompt "Explain distributed computing briefly." \
  --output results/monitor_test_requests.csv \
  --run-id monitor-test \
  --gpu-monitor \
  --gpu-output results/monitor_test_gpu.csv \
  --gpu-interval-ms 200
```

### Workload options

The benchmark supports:

```text
--prompt "text"          Use one prompt for every request
--prompt-file file.txt   Read prompts from a file
--mixed                  Generate a mixed workload
```

If none of these options is supplied, the workload generator creates a default uniform workload.

### CPU comparison

Run the same experiment on the CPU by changing:

```bash
--backend cuda
```

to:

```bash
--backend cpu
```

## Previously Reported Results

The following results were measured on one Tesla V100 using the Qwen2.5-0.5B-Instruct Q4_K_M model.

### llama.cpp microbenchmark

| Operation | CPU throughput | GPU throughput | GPU speedup |
|---|---:|---:|---:|
| Prompt processing, 128 tokens | 1244.9 tokens/s | 8816.5 tokens/s | 7.08x |
| Token generation, 64 tokens | 88.0 tokens/s | 422.5 tokens/s | 4.80x |

### Sutra end-to-end benchmark

For 20 requests generating 64 tokens each:

| Metric | Result |
|---|---:|
| Completed requests | 20 |
| Failed requests | 0 |
| Requests per second | 5.57 |
| Generated tokens per second | 356.37 |
| Mean latency | 179.58 ms |
| P50 latency | 179.47 ms |
| P95 latency | 179.86 ms |
| P99 latency | 180.64 ms |
| P50 time to first token | 7.81 ms |
| P95 time to first token | 7.94 ms |

During the run, GPU utilization remained approximately 80 to 82%, model memory usage remained at approximately 1086 MiB, and GPU temperature remained between 47°C and 48°C.

These results were recorded on the earlier inference branch and describe one inference worker using one llama.cpp context. The benchmark calls the worker directly; it excludes TCP routing and scheduler queueing. Re-run on PACE before attributing these numbers to the integrated service.

## Concurrency Interpretation

The current `LlamaEngine` protects its llama.cpp context with a mutex. Multiple benchmark client threads may submit requests concurrently, but model execution inside one engine remains serialized.

Therefore:

- `--concurrency 1` measures the single-worker baseline.
- Higher concurrency values measure queueing and contention around one worker.
- Higher concurrency does not represent parallel GPU execution.
- Parallel execution requires multiple workers or independent llama.cpp contexts.
- Sequential `generatefullBatch()` is not true multi-sequence llama.cpp batching.

This distinction must be preserved when reporting performance results.

## Current Status

Completed:

- llama.cpp integration as a Git submodule
- CUDA model loading
- GGUF model inference
- Model chat-template application
- Single-request generation
- Sequential worker batch processing
- Per-request performance metrics
- Configurable workload generation
- Concurrent benchmark submission
- Request-level CSV output
- Aggregate benchmark statistics
- CPU and GPU microbenchmarks
- Background GPU monitoring
- GPU metrics CSV output

Integrated on this branch:

- Scheduler admission, queue policies, batch formation, worker accounting, and deadlines
- TCP -> scheduler -> `InferenceWorker` -> mock/llama engine -> TCP response path
- Explicit `--backend mock|llama` selection; llama mode requires a successful model load
- Windows and Linux socket support for `sutra_router`
- Joined service/client threads with SIGINT/SIGTERM shutdown on Linux
- Core, optional inference, worker smoke, benchmark, and router CMake targets
- Scheduler, worker contract, and automated TCP integration tests

Remaining validation and extensions:

- PACE CUDA build and real-model TCP smoke run
- Linux signal/shutdown checks (Windows tests terminate the child process)
- Separate remote worker processes, worker registration protocol, and multiple GPUs
- True multi-sequence and continuous GPU batching
- Service-level benchmark experiments including TCP and scheduler delays

The current router embeds **one worker in its own process**, binds to loopback, and uses the scheduler's default FIFO policy with a maximum batch size of four. The scheduler library supports priority/aging and least-loaded selection, but router CLI policy selection and multiple workers are not exposed yet. A coordinator heartbeat thread keeps the embedded worker registered; it does not prove that a hung model call is healthy.

## Repository Policy

The following files are intentionally excluded from Git:

- Model files
- Build directories
- Generated executables
- Raw benchmark output
- Temporary result files

Changes to `common/inference_types.hpp` affect every subsystem and must be communicated to the complete team.