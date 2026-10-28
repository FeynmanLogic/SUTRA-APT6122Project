# Sutra: delivery plan, deliverables, and transaction specification

Prepared 24 October 2026; scheduler implementation and teammate handoff updated 28 October 2026. Initial inventory was based on checkout `e3a4a53`; the current scheduler work builds on `e806735`. **Existing** means present in this checkout unless a separate branch is named; **proposed** means still to agree and implement. No PACE job or GPU performance result was verified during this work. The local course schedule lists a tentative December 8, 2026 deadline; confirm current announcements and the rubric. See section 10 for the implemented scheduler API, verification and Jayani/Dhruv integration tasks; see `IMPLEMENTATION_GUIDE.md` for ICE setup.

## 1. What we are building

Sutra is a C++ service that accepts several text-generation requests, decides their execution order, groups compatible requests, and distributes them across GPU worker processes. Each worker runs an existing pretrained model through llama.cpp. Clients receive generated text, success/error status, and performance measurements.

Our engineering contribution is concurrent networking, safe queues, scheduling, batching, worker distribution, failure handling, and experimental comparison. CUDA model kernels come from llama.cpp. Training, fine-tuning, a web UI, and splitting one model across GPUs are outside the proposed delivery scope. Check the course rubric early for any requirement to write custom CUDA kernels.

```text
Concurrent benchmark clients
          | TCP: framed UTF-8 JSON
          v
Router + Scheduler (one C++ coordinator process)
  validate -> bounded queue -> FIFO/priority -> batch -> worker selection
          | TCP: batch messages                    ^ responses/status
          +-------------------+--------------------+
          v                   v
Worker 0 process        Worker 1 process
llama.cpp + model       llama.cpp + model
assigned GPU 0          assigned GPU 1
```

Each worker holds a full model replica and processes different requests. Start on one compute node; use the same protocol across nodes after connectivity is verified. Multiple processes on one GPU can test correctness, but do not demonstrate multi-GPU scaling.

## 2. Actual starting point

| Component | Evidence in this checkout | Work remaining |
|---|---|---|
| Shared contract | `common/inference_types.hpp` defines requests, responses, vector batch aliases | Freeze semantics, validation, wire format, timing ownership |
| Engine interface | `inference/inference_engine.hpp` defines `loadModel`, `generatesingle`, `generatefullBatch` | Mock engine and real model execution |
| Workload generation | `benchmark/workload_generator.cpp` implements prompt/file/uniform/mixed workloads; config header exists | Validate negative counts/limits; tokenize to measure actual lengths; build client and runner |
| Scheduler | `scheduler/scheduler.hpp/.cpp`: bounded admission, FIFO/priority+aging, count/timer batching, readiness, round-robin/least-loaded, reservations, deadlines, heartbeat expiry, disconnects and shutdown | Wire into Jayani's coordinator and Dhruv's workers; verify actual TCP/GPU behavior |
| Build/tests | CMake builds `sutra_scheduler`, `sutra_scheduler_demo`, and `sutra_scheduler_tests` without external dependencies; 21 scheduler cases pass locally | Add coordinator/client/worker targets and optional llama linkage; run on ICE and Linux ThreadSanitizer |
| Other runtime | Worker, metrics, runner and original worker-test files remain empty; LlamaEngine files contain comments only. Jayani's `origin/feature/tcp-router` at `098383a` adds a sequential loopback TCP echo server; it is reviewed but not merged here | Complete networking/framing, model execution, metrics and end-to-end tests; preserve Jayani's branch |
| Dependency | `.gitmodules` points to llama.cpp; recorded revision `9d1e03616efb7520514d65552c7376296ef9208f` | Initialize submodule; verify API/toolchain compatibility before changing revision |
| PACE baseline | README reports V100 16 GB, CUDA 12.6, Qwen2.5-0.5B-Instruct Q4_K_M, CPU/GPU microbenchmarks | Reproduce and preserve commands, environment, and raw results; these are reported, not locally verified |

Important integration fixes: the header comment permits client/router arrival timestamps, while the README assigns them to the router. The generator also assigns local IDs. For network serving, **the router owns authoritative IDs and arrival timestamps**; benchmark-local IDs must not collide or override them. The existing `generatefullBatch` loops over requests: that is sequential aggregation, not simultaneous GPU inference.

## 3. Delivery sequence and ownership

Owners follow the proposal: Jayani = router/networking; Aashish = scheduler; Dhruv = inference/monitoring/benchmarking. Shared contract/build changes need team coordination.

Suggested budget: 10-14 working days for the team, with overlapping component work after M0. Reserve the last two days for integration and evidence. Rebase these estimates against the real deadline before committing to dates.

| Milestone | Budget / dependency | Deliverables | Acceptance gate |
|---|---|---|---|
| M0: freeze interfaces and reproduce ICE baseline | Days 1-2; first | Shared protocol/spec, CMake/CTest setup, pinned dependency/model manifest, ICE environment capture and CPU/CUDA smoke scripts; all owners, Dhruv leads ICE | Mock build runs without GPU; one allocated NVIDIA GPU produces real text; model loads once; exact revisions recorded |
| M1: complete mock transaction | Days 2-4; M0 contract | Jayani: concurrent TCP coordinator/client and worker transport. Aashish: bounded FIFO queue, batch size 1, worker registry. Dhruv: deterministic mock engine/worker | 100 requests from 8 concurrent clients: correct IDs/client mapping, no loss/duplicate terminal replies; malformed/partial TCP frames handled |
| M2: real single-worker serving | Days 4-6; M1 + ICE baseline | Dhruv: LlamaEngine, tokenization/chat template, context reset, limits, CPU/CUDA choice, measurements. Integrate network path | Entire client-to-GPU path works; isolation between prompts; token limits and errors verified; warmup excluded from measurements |
| M3: scheduling, distribution, faults | Days 5-8; M1, then M2 integration | Aashish: FIFO/priority with aging, batch timers, round-robin/least-loaded. Jayani: readiness, heartbeats, deadlines, disconnect cleanup. Dhruv: two worker replicas | At least two processes execute distinct requests; use two assigned GPUs for scaling evidence. Priority/aging, saturation, and killed-worker cases pass |
| M4: true dynamic GPU batching | Days 8-10; M2/M3 stable | Dhruv + Aashish: multi-sequence llama batches, per-request sequence/KV state, batch token budget, capacity reporting | Multiple independent sequences share decode calls; outputs remain isolated; compare batch sizes 1/2/4. Keep sequential mode as an explicitly labeled baseline |
| M5: experiments and submission | Days 10-14; M3/M4 | Reproducible scripts/configs, per-request CSV, summaries/plots, report, architecture/data-flow explanation, demo and saved fallback output; all owners | A fresh ICE allocation reproduces the demo and benchmark procedure; each performance claim traces to a run manifest and raw data |

Critical path: **contract + ICE access -> mock transaction -> real GPU transaction -> multi-worker integration -> real batching -> measurements -> report/demo**. Router and scheduler can progress with the mock while inference work waits for GPUs. Do not defer the first real inference integration until the final week.

Scope gates: deliver all listed proposal features, including dynamic batching. Continuous batching (admitting new sequences between decode steps) follows working static multi-sequence batches if time permits; confirm whether the proposal's "dynamic/continuous" wording requires it. Cross-node deployment is a planned extension after same-node worker processes; confirm whether the rubric requires multiple physical nodes. If either is mandatory, promote it into M4 and reserve allocation time now. Any scope reduction must be explicit in the report and agreed against course expectations.

## 4. PACE-ICE dependencies and operating plan

Georgia Tech's current public resource page identifies ICE as RHEL 9/x86_64 with NVIDIA **and** AMD GPUs. This project targets NVIDIA/CUDA. The detailed resource link redirects to GT authentication, so partition names, account/QoS requirements, module versions, GPU limits, storage paths, and walltimes remain **environment checks**, not assumed facts. [GT resource page](https://sox.cc.gatech.edu/facilities/instructional-labs/academic-hpc-resources), [authenticated ICE guide](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095).

Local course instructions add a concrete starting workflow: Open OnDemand GPU Jupyter/VS Code, `module load cuda`, a system C++ compiler, Release builds, and `pace-quota`. The CUDA starter notebook specifically excludes the `ece4122-6122` conda build environment; follow that CUDA-specific guidance rather than the generic course conda setup. The new implementation guide records the exact local source files and setup cells. Verify the active module/compiler/GPU in the allocated job.

| Dependency | Concrete action and evidence | If unavailable |
|---|---|---|
| Course access and connection | Confirm each teammate can connect with the institution's required authentication/VPN and submit a course job | Continue mock development; resolve course provisioning before GPU integration |
| Slurm allocation | Inspect `sinfo`, `scontrol show partition`, and current course instructions; confirm GPU type/count and walltime before writing launch scripts | Schedule short one-GPU smoke jobs first; reserve longer/two-GPU experiments early |
| Compiler/CMake/CUDA | Capture `module list`, compiler/CMake versions, `nvcc --version`, driver and `nvidia-smi` inside allocation; test C++17 and the pinned llama.cpp requirements | Use a compatible available module combination; coordinate and record a dependency revision change if necessary |
| GPU assignment | One model-owning worker per allocated GPU; preserve Slurm's visibility and binding; record each worker's actual GPU UUID | One-GPU functionality remains testable; postpone scaling claims |
| Model and storage | Stage one selected GGUF plus checksum/revision in approved persistent/project storage; use documented scratch/local storage for temporary data if available; archive results before scratch expiry | Pre-stage via an approved transfer path; do not depend on outbound network access from compute nodes |
| Process networking | Run client and coordinator within the allocation; verify internal ports, worker registration and cross-node reachability | Demonstrate multiple processes on one allocated node; record any cross-node limitation |
| Job lifecycle | Launch workers, wait for model-ready signals, run warmup/measurement, flush results, terminate children on exit; capture Slurm job ID, exit status, logs | Save a short successful run and demo recording before the final session |

Use login nodes for connection, submission, and light file management; place builds, model execution, and load generation in allocated compute resources according to current PACE rules. No public endpoint is needed for the demo.

Proposed initial allocation requests, subject to current course limits: one NVIDIA GPU, 8 CPU cores, 16 GiB host RAM and 30 minutes for smoke tests; then two matching NVIDIA GPUs, 16 CPU cores, 32 GiB host RAM and 60 minutes for integration. These are starting budgets, not measured requirements or cluster guarantees. Profile and adjust. V100/CUDA 12.6 is the README's reproduction target, not a guaranteed allocation.

M0 script deliverables: `scripts/ice_smoke.sbatch`, `scripts/ice_benchmark.sbatch`, `scripts/run_demo.sh`, and an environment/model manifest. Confirm account/partition directives before submission. Initialize using `git submodule update --init --recursive`; build distinct CPU and CUDA configurations. llama.cpp's CUDA build uses CMake's `GGML_CUDA` option; check the checked-out revision before fixing compiler/architecture flags. [Upstream build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

## 5. Proposed runtime specification

These defaults are implementation targets, not existing functionality or optimal measured settings.

| Item | Specification |
|---|---|
| Language/platform | C++17, Linux/POSIX sockets, CMake, `Threads::Threads`; pin a JSON library version; Python only for offline plots if useful |
| Coordinator concurrency | Bounded connection/I/O handling plus one scheduler thread and worker-response handling; `std::thread`, `std::mutex`, `std::condition_variable`; avoid unbounded thread creation and holding queue locks during network/model calls |
| Model | Qwen2.5-0.5B-Instruct, GGUF Q4_K_M; 0.49B parameters; 24 layers; hidden width 896; 14 query heads, 2 KV heads; head width 64; vocabulary 151,936; model supports 32,768 positions. Pin actual GGUF metadata/revision/checksum. Q4_K_M uses mixed quantized tensor types, not a uniform four-bit representation of all runtime data |
| Context | Start with 2,048 tokens per sequence including template/special tokens and reserved generation; reject overflow without silent truncation. For four active sequences, configure and verify sufficient total llama context/KV capacity; do not assume `n_ctx=2048` provides 2048 per sequence |
| Generation | Default 64 new tokens, accepted range 1-512; greedy sampling for controlled comparisons; terminate on model end-of-generation token or limit; independent sampler and sequence state |
| Scheduler | FIFO or priority 0/1/2, with 2 highest; FIFO tie-break. Proposed aging: oldest eligible request waiting at least 1 second goes first at each dispatch, ahead of priority order. Recompute eligibility at dispatch |
| Batch policy | Maximum 4 requests, target collection wait 10 ms from oldest queued request, plus tokenizer-based total token/KV budget. Dispatch early when full; never wait merely to fill a batch past the timer. Saturation can still extend queue delay beyond 10 ms |
| Backpressure | Initial limits: 1,024 queued requests, 128 active client connections, 16 KiB prompt UTF-8 bytes, 1 MiB frame payload; reject overload explicitly. Token/context checks follow template/tokenization and are stricter where necessary |
| Worker choice | Round-robin among ready workers, or least outstanding request count including in-flight work; reserve capacity atomically. Extend to token-weighted load only after the basic comparison |
| Worker safety | One execution owner per llama context; do not call the same context concurrently. Clear/release per-request KV state; isolate sequences during real batching |
| Failure behavior | Proposed heartbeat every 1 s, unavailable after 3 missed heartbeats; 120 s request deadline for the first small-model demo, configurable. Model-not-ready, bad input, overload, timeout, and disconnect produce explicit errors |

Batch admission has two layers: the scheduler applies request count/wait limits; the worker formats/tokenizes candidates and enforces actual token/KV capacity before model execution. Split a candidate group into smaller execution batches when necessary and record effective batch sizes. The scheduler must not call llama.cpp directly or assume the character-based workload estimate is an exact token count. Distinguish maximum request count from llama.cpp's token batch/microbatch settings; log both.

Model dimensions above come from the [publisher configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json) and [GGUF model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF). Runtime context and precision are our choices and must be logged separately from pretrained-model specifications.

No automatic retry of dispatched requests initially: a disconnect does not prove the GPU did no work. Enforce one terminal outcome per accepted request while the coordinator remains running, discard late responses after timeout, and continue routing to healthy workers. Coordinator crash recovery and durable exactly-once execution are outside scope.

## 6. Exact request, response, and wire contract

Preserve the existing C++ fields; put transport/control metadata in an envelope. Use UTF-8 JSON preceded by a **4-byte unsigned network-byte-order payload length**. TCP is a byte stream: implement read/write loops for partial transfers, reject oversized/truncated frames, and handle multiple frames arriving together. Never send raw C++ struct memory.

Envelope fields: `version` (integer, initially 1), `type` (string: `infer`, `batch`, `result`, `ready`, `heartbeat`, or `error`), and `payload` (object). Batch payload contains `batch_id` (decimal string), `worker_id` (integer), and `requests` (array). Worker readiness reports backend/model identity and request/token capacity. Define these message schemas in M0.

| Existing request field | C++ type / proposed wire encoding | Meaning and mutation |
|---|---|---|
| `request_id` | `std::uint64_t` / decimal JSON string | Router-generated unique ID within coordinator run; unchanged thereafter. Log `run_id` separately for uniqueness across runs. String encoding preserves all 64 bits in JSON clients |
| `prompt` | `std::string` / UTF-8 JSON string | Original text; unchanged through scheduling/networking. Worker creates a separate templated model-input string |
| `max_new_tokens` | `int` / JSON integer | Generation cap; validate 1-512; unchanged |
| `priority` | `int` / JSON integer | Base priority 0-2; unchanged. Aging is internal scheduling metadata, not a rewrite of this field |
| `arrival_time_ms` | `std::int64_t` / decimal JSON string | Unix-epoch milliseconds set at router acceptance; diagnostic timestamp, unchanged. Never use unsynchronized cross-host clock subtraction for durations |

`int` is expected to be 32 bits on the target; enforce this or use fixed-width internal/wire validation. `std::string` has variable-length bytes and process-local storage; its `sizeof` does not measure text length. JSON escaping can change wire length without changing decoded text. Client input requires only prompt, generation cap, and priority; clients do not choose authoritative IDs/timestamps.

| Existing response field(s) | C++ type / wire encoding | Owner and definition |
|---|---|---|
| `request_id` | `std::uint64_t` / decimal string | Same ID, used by coordinator to find client |
| `worker_id` | `int` / integer | Executing worker; -1 for rejection before assignment |
| `generated_text` | `std::string` / UTF-8 string | Detokenized generated output only; exclude original prompt/template and control tokens |
| `input_tokens`, `generated_tokens` | `int` / integers | Engine counts actual templated input tokens and emitted output tokens; exclude end-of-generation control token from output count; record this convention |
| `queue_time_ms` | `double` / JSON number | Scheduler monotonic enqueue-to-dispatch duration plus any local worker waiting before execution; carry durations in internal metadata |
| `inference_time_ms` | `double` / number | Worker monotonic execution start through completion, including formatting/tokenization, prefill, decode and detokenization; synchronize completion where needed |
| `time_to_first_token_ms` | `double` / number | Worker execution start to first emitted output token. Worker-internal TTFT: final-response-only TCP delivery does not expose it live to the client |
| `latency_ms` | `double` / number | Coordinator finalizes acceptance-to-response-ready duration on its own monotonic clock; benchmark separately measures client send-to-full-reply time |
| `tokens_per_second` | `double` / number | `1000 * generated_tokens / inference_time_ms`, zero if duration is zero; includes prefill, so label it per-request overall generation rate |
| `success` | `bool` / JSON boolean | True on completed generation, including token-limit termination |
| `error_message` | `std::string` / string | Empty on success; readable error on failure. Log a stable error code in the envelope/CSV |

Existing response numeric fields cannot represent missing metrics. On failures use zero placeholders, `success=false`, an error code, and exclude unavailable measurements from successful-request percentiles. Add a `metrics_valid` boolean in the response envelope for completions without a first output token; use it to exclude undefined TTFT. Router/scheduler/worker keep local `std::chrono::steady_clock::time_point` values; serialize elapsed durations only. Epoch timestamp is for correlation. Worker timing fields proposed in the README need coordinator/scheduler participation for distributed queue and end-to-end latency accuracy.

## 7. One complete transaction: data at every stage

Client asks for `"Say hello."` (10 ASCII/UTF-8 bytes), at most 16 new tokens, priority 2. The following IDs, counts, answer and timings are **illustrative**, not measured output or an asserted tokenization.

```json
{"version":1,"type":"infer","payload":{"prompt":"Say hello.","max_new_tokens":16,"priority":2}}
```

| Stage | Representation, shape, and transformation |
|---|---|
| 1. Client -> socket | JSON UTF-8 byte buffer, e.g. `std::vector<std::uint8_t>` of length L; prefix is `uint32_t(L)` in network order. This is text and metadata, not model tensors |
| 2. Router | Parse/validate -> `InferenceRequest{42, "Say hello.", 16, 2, T}`. ID 42 is `uint64_t`; T is `int64_t` epoch ms. Save `42 -> client connection`, coordinator-run ID, deadline and monotonic acceptance time |
| 3. Scheduler | Store request plus enqueue time/sequence number in a mutex-protected queue. Choose FIFO or effective priority. With a second compatible request, create `RequestBatch = std::vector<InferenceRequest>` of size 2. Prompts and IDs stay separate; strings are never concatenated into one request |
| 4. Dispatch | Select ready worker 0; add batch envelope and elapsed scheduler queue time; serialize two requests into JSON bytes. Worker parses back to typed objects. Model weights are already loaded; they never travel with a request |
| 5. Model formatting | Worker applies GGUF's instruction/chat template: one user message plus assistant-generation prefix, with a fixed documented system-message policy. New `std::string` contains role/control markers plus the original prompt; original request text is preserved |
| 6. Tokenization | Templated UTF-8 -> `std::vector<llama_token>`, where token IDs are signed 32-bit integers. Let actual length be P. Token count differs from byte/word count; count special/template tokens. Enforce `P + 16 <= 2048`; do not invent exact IDs or P before using the actual tokenizer |
| 7. GPU prefill | llama.cpp embeds token IDs into width-896 vectors and executes 24 transformer layers. Logical single-sequence hidden shape is `[P,896]`; fused/internal layouts vary. CPU token arrays enter backend-managed execution; weights are resident on the assigned GPU when fully offloaded. Q4_K_M weight storage stays quantized; activation/compute types are backend-dependent FP16/FP32 paths and must be inspected/logged |
| 8. KV state | Attention keys/values are per request/sequence, not shared between independent users. Proposed FP16 KV: logical per-layer K and V each `[used_tokens,2,64]`. Each emitted step grows used state; reserve capacity for the configured context. Derived ideal KV payload is `24 layers * 2(K,V) * 2 heads * 64 * 2 bytes = 12 KiB/token`, about 24 MiB per 2048-token sequence, excluding padding/metadata/workspaces |
| 9. Next-token scores | Latest-position logits exposed to sampling are float32 scores, logical shape `[151936]` per active sequence, roughly 594 KiB. Greedy sampling chooses one `int32_t` token ID. Scores are not words, and a softmax/probability array is unnecessary for argmax |
| 10. Decode loop | Feed the chosen token back with its position and sequence ID; update that sequence's KV state; repeat until end-of-generation or 16 emitted tokens. In sequential mode only one request advances. In real batching, active sequences advance in shared decode calls with separate sequence IDs, positions, sampling and outputs |
| 11. Text assembly | Token IDs -> token pieces -> generated UTF-8 `std::string`. A piece can contain only part of a Unicode character, so assemble valid text before final serialization. Suppose the final text is `"Hello, world!"` (13 bytes). This does not imply 13 tokens; assume G=4 only for the timing illustration |
| 12. Worker -> coordinator | Build `InferenceResponse`, preserve ID 42, attach worker 0, actual P/G and measured durations. Serialize response bytes; no logits, KV tensors or weights are sent back |
| 13. Coordinator -> client | Merge queue metrics, finalize coordinator latency, find client using 42, send one framed JSON result, and remove pending state. Client parses generated text/status/metrics; completion may occur out of submission order |
| 14. Cleanup and evidence | Release request-specific KV/sampler state and worker capacity, retain loaded model, append CSV row plus run/config metadata. Repeated requests must not inherit unrelated conversation state |

Example timing arithmetic: assume scheduler/worker waiting totals 5 ms, worker inference 40 ms, first token at worker +10 ms, and coordinator latency 48 ms. Then `queue_time_ms=5`, `inference_time_ms=40`, `time_to_first_token_ms=10`, `latency_ms=48`, and for illustrative G=4, `tokens_per_second=100`. Remaining 3 ms covers transport/coordinator overhead. Client latency includes its additional transport. TTFT is already inside inference time: do not add it again.

llama token/sequence types and logit interfaces are documented in the [upstream C API](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h). The local pinned revision's header must be inspected during M0; upstream master is explanatory evidence, not a promise of API compatibility. Tensor shapes/KV sizing above are logical derivations from the model dimensions, not measured allocations.

## 8. Experiments and final evidence

Use the same pinned model, prompt set, output limits, sampler, CPU allocation and GPU class when comparing one factor. Existing uniform prompts estimate tokens as roughly four characters each; actual tokenizer counts must be recorded. Use repeated identical prompts for controlled comparisons and mixed 32/128/512-token targets for scheduling experiments.

| Question | Runs / evidence |
|---|---|
| Does CUDA help? | CPU vs CUDA, one worker, concurrency 1, batch 1; record actual offload and hardware |
| Does concurrency help? | Fixed backend/policy, client concurrency 1/4/8/16; observe throughput, p95 latency and saturation |
| Do batching and scheduling help? | Sequential aggregation vs true multi-sequence mode, max batch 1/2/4, wait 0/5/10 ms; FIFO vs priority+aging with identical mixed arrivals; include per-priority wait/latency |
| Does distribution help? | One vs two workers on one vs two assigned, matching GPUs; round-robin vs least-loaded; report throughput speedup and per-GPU efficiency |
| Are failures contained? | Bad input, queue full, stalled/disconnected worker and late response; healthy workers continue, each accepted request reaches terminal status |

Avoid a full Cartesian sweep. Establish a baseline, vary one factor, then combine the best settings. First use config defaults (30 measured requests, 5 warmups) for smoke only; for reported comparisons target at least 200 measured requests and 3 repetitions, increasing sample size for credible tails. Treat p99 from small samples as exploratory; publish counts and per-run variation. Add a controlled arrival-rate test to expose queue growth; closed-loop concurrency alone can hide offered load.

Report completed/failed/timed-out/rejected counts; successful requests/s; aggregate emitted tokens/s = total successful output tokens divided by measured benchmark wall time; median/p95/p99 client latency; worker TTFT; queue latency by priority; GPU utilization/peak memory via sampled `nvidia-smi` or NVML. Do not average per-request token rates to claim aggregate throughput. Exclude loading/warmup from steady-state results and report startup cost separately. Report regressions honestly rather than promise speedup.

Every run bundle: coordinator/submodule commits, model source/revision/SHA-256, CLI/config, seed and prompt-set hash, actual token counts, hardware/GPU UUIDs, driver/CUDA/compiler versions, Slurm job ID, timestamps, raw CSV/logs, summary, plot-generation command. Store large artifacts outside Git with stable references; commit small configs/summaries and report figures. Existing `.gitignore` excludes `models/` and `results/`.

## 9. Submission checklist and deliverable inventory

### Build and interfaces

- [ ] M0: Confirm actual deadline, rubric, cross-node/continuous-batching/custom-CUDA requirements.
- [ ] M0: Reproduce one CPU and one CUDA inference run on ICE; record exact environment/model/dependency.
- [ ] M0: Agree `common/` contracts, framed message schemas, timing definitions, limits and error codes.
- [ ] M0: Supply working CMake targets for coordinator, worker, benchmark client and focused tests.
- [ ] M1: Supply deterministic mock inference so router/scheduler integration requires no GPU.

### Functionality and correctness

- [ ] M1: Concurrent clients, unique router IDs, partial/multiple TCP frames, bounded queues and correct response mapping.
- [ ] M2: Persistent model loading, actual token counts, valid templating, generation/context limits, CPU/CUDA selection and state isolation.
- [ ] M3: FIFO, priority+aging, maximum batch count/wait, token budget, round-robin and least-loaded selection.
- [ ] M3: Multiple worker processes; two-GPU execution evidence; cross-node test if required.
- [ ] M3: Readiness, heartbeat/deadline/disconnect handling, overload, late-response suppression and clean shutdown.
- [ ] M4: Real multi-sequence GPU batching with independent KV/sampling/output state; label sequential baseline accurately.
- [ ] M4: Continuous slot replacement between decode steps if required by the agreed course scope.
- [ ] M1-M4: Focused tests for wire framing, queue fairness/timers, capacity accounting, engine failures and end-to-end mapping; CPU thread-sanitizer run where supported.

### Reproducibility and submission

- [ ] M5: Deliver ICE smoke/benchmark/demo scripts with verified allocation directives and bounded runtime.
- [ ] M5: Deliver benchmark configs, seeded workloads, per-request CSV, run manifests and analysis/plots.
- [ ] M5: Deliver report mapping every proposal feature to code, correctness evidence, performance evidence and limitations.
- [ ] M5: Deliver a short demo: concurrent traffic -> scheduling/batching -> two workers -> metrics -> killed-worker recovery.
- [ ] M5: Preserve a successful demo recording/results bundle in case the final live allocation is delayed.
- [ ] M5: Update README with build/run commands, architecture, ownership and actual implemented status; tag the tested submission commit.

The submission is complete when another teammate can use the documented ICE allocation/build/run procedure to reproduce correct concurrent serving, scheduling and failure behavior, and regenerate every reported chart from preserved measurements.

## 10. Scheduler delivery and teammate handoff — 28 October 2026

### Aashish: implemented component and remaining integration

The scheduler component is implemented in `scheduler/scheduler.hpp` and `scheduler/scheduler.cpp`. `examples/scheduler_demo.cpp` exercises two synthetic workers without networking or model inference. No code on Jayani's or Dhruv's branches was replaced. No shared request/response fields, engine interfaces, llama.cpp revision or benchmark-owned configuration types were changed. The initial CMake file was empty; the new build only covers the scheduler. Extend it with additional targets during integration instead of replacing it.

Implemented: bounded queued/outstanding admission; input and active-ID validation; FIFO; priority 2 > 1 > 0 with FIFO ties; oldest-admitted aged-request promotion; count/timer candidate batches; worker readiness; bounded request and batch reservations; round-robin; least outstanding requests with round-robin ties; per-response capacity release; correlation and duplicate/late-response suppression; queued/in-flight request deadlines; heartbeat expiry; disconnect failures; immediate idempotent shutdown; condition-variable dispatch/completion waits; coordinator queue/latency measurements; invalid numeric worker-metric rejection. The outstanding limit includes uncollected terminal events, bounding the internal completion outbox.

These are scheduler-component results, not a completed M1/M3 network service or a true GPU batching result. Aashish's integration work remains: connect the coordinator, verify FIFO/priority and worker policies over TCP, run killed-worker and real-GPU tests, and measure/tune settings. Actual tokenizer/KV admission, independent GPU sequences and continuous slot replacement remain joint M4 work with Dhruv; request aggregation is not GPU batching.

### Concrete API to use

Include `scheduler/scheduler.hpp` and link `sutra_scheduler`. All public calls synchronize internal state; no socket/model callbacks execute under its mutex. One coordinator owns one scheduler and one dispatch loop. Returned values own their data; do transport work after the call returns.

| Operation | Integration contract |
|---|---|
| `Scheduler(SchedulerConfig)` | Defaults: queue 1024, total retained requests 4096, batch 4, wait 10 ms, aging 1 s, request deadline 120 s, heartbeat deadline 3 s, prompt limit 16 KiB; default policies FIFO/round-robin. Timers must be positive except batch wait/aging may be zero |
| `submit(request)` | Router supplies a nonzero coordinator-run-unique ID, nonempty prompt, 1–512 output tokens, priority 0–2 and nonnegative epoch timestamp. Scheduler copies/moves the request without changing fields. Return status is `accepted`, `invalid_request`, `duplicate_request`, `queue_full`, `outstanding_full`, or `stopped`. Rejections have NO completion event: router replies immediately |
| `registerWorker({id, request_capacity, max_batch_size, batch_capacity})` | Call only on the identified current connection after model-ready. ID >= 0; all capacities positive. Duplicate live IDs rejected. Defaults `{id,4,4,1}`: up to four assigned requests in one batch. Advertise `{id,1,1,1}` for initial single-request integration. These are request counts, not tokens or GPU memory |
| `setWorkerReady(id, false/true)` | Stop/resume new dispatches without discarding existing assignments. Does not refresh heartbeat or change configured capacity. To change capacity, drain, remove, then re-register a fresh session |
| `heartbeat(id)` | Refresh current registered session. Returns false for unknown/expired sessions; never silently resurrects an expired worker. Send heartbeats while inference runs; output responses do not refresh heartbeat automatically |
| `waitForDispatch()` / `tryDispatch()` | Returns `DispatchBatch{batch_id, worker_id, requests}` with capacity already reserved. Candidate count is limited by scheduler maximum, worker maximum and remaining request slots. Batch wait starts at the oldest queued request; full candidates dispatch early. A busy system can wait longer than the collection timer |
| `complete(batch_id, worker_id, response)` | Worker response must retain request ID and set matching worker ID. Match all three identifiers. Accept individual results in any order. `false` means unknown/stale/duplicate/wrong-worker result, with no capacity release. `true` consumes one result, even if malformed numeric metrics are converted into `invalid_response` |
| `removeWorker(id, reason)` | Use for disconnect, send failure, or uncertain delivery. Fails all still-assigned requests once and removes session; queued/unrelated worker requests remain. Never automatically retry uncertain execution |
| `waitForCompletion()` / `takeCompletions()` | Collect each accepted request's single terminal `Completion{response, code}` and route by request ID. Blocking consumer drives deadline timers; nonblocking consumer must call `poll()` regularly even with no traffic. Multiple consumers divide events; they do not each receive copies |
| `shutdown()` | Immediate cancellation of queued/in-flight requests, reject new work, wake waiters, preserve terminal events for collection. Stop transport activity and join caller-owned threads before destroying scheduler. `waitForCompletion()` drains buffered events before returning empty |

The scheduler does not retain a lifetime history of completed request IDs: IDs remain reserved until their terminal event is collected, then memory is freed. **The router must never reuse IDs within a coordinator run.** Batch IDs increase for the scheduler's lifetime, including worker reconnects. Explicit `TimePoint` parameters exist for deterministic tests; production should omit them and use `steady_clock`. If passing the router acceptance time to `submit`, it must be from the same coordinator's steady clock. Admission/FIFO ties follow mutex-protected insertion order.

Safety on in-flight timeout: the scheduler retires the whole worker session. Expired requests receive `request_timeout`; other still-assigned requests on that session receive `worker_unavailable` (or `worker_timeout` if heartbeats also expired). This avoids freeing capacity on a GPU that may still be executing. Jayani must close/quarantine that connection; Dhruv must confirm cancellation/drain or restart before new model-ready registration. Timed-out queued requests have worker ID -1. A stale connection's heartbeat/disconnect MUST be filtered by Jayani's connection-generation map before calling scheduler methods keyed by worker ID; otherwise it could affect a newly registered session with that numeric ID. Late old results are additionally rejected by batch ID.

### Jayani: expected changes and checks

- [ ] Keep `router/tcp_server.cpp` as the starting point for transport; its `098383a` echo prototype is still on `feature/tcp-router`, not merged into this checkout. Extend/refactor on your branch without adding scheduler policy or model code to socket loops.
- [ ] Implement section 6 framing: four-byte network-order length, UTF-8 JSON envelopes, bounded payloads, complete read/write loops (including interrupted calls), malformed/truncated frame errors and concurrent clients. Current single `recv()`/sequential handling is insufficient. Keep loopback for same-node tests; configure bind/connect addresses before cross-node work.
- [ ] Generate authoritative IDs/timestamps, maintain pending-ID-to-client and current-worker-connection-generation maps, call `submit`, handle every rejection immediately, and collect terminal events even when no worker response arrives. Deliver one terminal reply where the client connection is still available.
- [ ] Run a dispatch loop: call `waitForDispatch`, serialize batch ID as a decimal string, worker ID and requests, then send OUTSIDE scheduler calls. A send failure means uncertain execution: `removeWorker` on that current session, with no automatic retry. Add serialization/dispatch-failure containment so an exception cannot leave a reserved batch stranded until its deadline.
- [ ] Read worker ready/heartbeat/result messages; call `registerWorker`, `heartbeat`, `complete` on the matching current session. Add `batch_id` to the result envelope without changing `InferenceResponse`. Ignore stale connections. On scheduler retirement, close/quarantine the old connection; never allow an old close/heartbeat to act on a replacement session.
- [ ] Return `toString(Completion.code)` as stable error/status metadata; finalize any extra router-acceptance/client-send timing separately. The scheduler's `latency_ms` is submit-to-result-ready, not socket-send duration.
- [ ] Coordinate CMake additions for coordinator and transport tests. Verify fragmented/coalesced frames, 8 concurrent clients/100 requests, client disconnects, queue/outbox backpressure, worker disconnects/timeouts, stale reconnect traffic and exactly one terminal outcome per accepted request.

### Dhruv: expected changes and checks

- [ ] Implement deterministic mock and real `InferenceEngine`/worker paths using the existing shared types. Load a model once; send ready only after successful loading. Start with `{id,1,1,1}`; advertise larger request/batch limits only after corresponding processing works.
- [ ] Parse `DispatchBatch.requests` into `RequestBatch`, preserve request IDs, and echo the envelope batch ID on EACH result (or unpack a result array into individual scheduler completions). Set `response.worker_id` to the assigned worker ID. Exactly one response for every request, including model/validation failures.
- [ ] Keep one execution owner per llama context. Default one in-flight batch; any larger `batch_capacity` needs an explicit bounded serial queue or safe independent contexts. Run heartbeat I/O independently of a long inference call; proposed cadence 1 s, scheduler timeout 3 s. Timer settings are starting values to tune, not measured guarantees.
- [ ] Tokenize formatted prompts, enforce actual input+output context and total token/KV budgets, split candidate groups if needed, and record effective GPU batch sizes. Scheduler count/timer aggregation does not replace tokenizer admission or implement simultaneous GPU sequences.
- [ ] Report finite, nonnegative token counts/timings/rates. Set worker `queue_time_ms` ONLY to local receipt-to-execution waiting; scheduler adds its own enqueue-to-dispatch time once. Keep engine `inference_time_ms` and TTFT local; scheduler replaces `latency_ms` with coordinator elapsed time. Failed/undefined metrics need the agreed envelope validity metadata; no cross-host epoch subtraction.
- [ ] Handle retired-session cancellation/drain/restart before reconnect. Do not send a new ready just because the scheduler deadline elapsed while old model work is still running.
- [ ] Extend CMake rather than replace scheduler targets. `SUTRA_ENABLE_LLAMA=OFF` is working; `ON` deliberately errors until real backend linkage exists. Map benchmark `SchedulingPolicy::FIFO/PRIORITY` explicitly to `QueuePolicy::Fifo/Priority` (existing benchmark types are unchanged), and expose worker-policy/batch settings in benchmark metadata.
- [ ] Add worker/model isolation and failure tests, then one/two-GPU tests and benchmark CSV/manifests. Verify real offload, true sequence isolation and throughput separately from the synthetic scheduler demonstration.

### Integration order, branch preservation and evidence

1. Review this API and the shared wire envelope together. Start with batch size 1 and one mock worker, then four candidates and two workers.
2. Integrate feature branches via reviewed merges/PRs; no force pushes or copying whole files over teammate work. The `.gitignore` edits conflict with Jayani's append: retain both sets, deduplicate `/router/tcp_server`, and keep the personal learning-guide exclusion. Existing `build*/` covers `/build/`.
3. Extend CMake and run component tests on every integration. Keep CPU scheduler builds independent of CUDA and dependency downloads.
4. Complete full mock TCP transaction, faults/reconnect tests, real GPU path, then GPU batching/scaling. Do not mark combined milestone checkboxes complete solely because scheduler unit tests pass.

Verified locally on Windows with MinGW GCC 16.1.0, CMake 3.29.2, Ninja, Release: build succeeds; CTest scheduler suite passes all 21 named cases, including 8 producer threads and 128 unique successful completions; synthetic demo completes 8/8 requests across workers 0 and 1. `git diff --check` passes. No Linux/ICE/CUDA/TCP integration or ThreadSanitizer run is claimed.

```bash
# Linux / ICE allocated CPU session; no CUDA, model or submodule needed.
cmake -S . -B build-scheduler -DCMAKE_BUILD_TYPE=Release -DSUTRA_ENABLE_LLAMA=OFF
cmake --build build-scheduler --parallel 4
ctest --test-dir build-scheduler --output-on-failure --no-tests=error
./build-scheduler/sutra_scheduler_demo
```

For a future supported Linux GCC/Clang race check, use a separate build with `-DCMAKE_CXX_FLAGS="-fsanitize=thread -g"` and `-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"`, then run the same CTest suite. This check remains pending; the local Windows build does not establish race freedom on ICE.
