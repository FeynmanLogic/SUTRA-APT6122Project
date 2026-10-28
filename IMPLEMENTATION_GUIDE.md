# Sutra: implementation and PACE-ICE execution guide

Updated 28 October 2026. Companion to `PROJECT_DELIVERY_PLAN.md` and `notebooks/Sutra_PACE_ICE_Demo.ipynb`. The standalone scheduler is now implemented; section 10 of the delivery plan records its exact API and Jayani/Dhruv follow-up work. Full-service notebook execution remains gated until transport/worker/launcher integration.

## 1. Use C++ source files, scripts, and a notebook together

Develop/debug the service in **VS Code on PACE-ICE**. Keep implementation in `.hpp/.cpp`, build with CMake, and run focused tests with CTest. Use **Jupyter as the demo control panel**: prechecks, build commands, short runs, logs, and result tables. Use **Slurm batch jobs for longer experiments**. The same scripts should work from a terminal or notebook; no project source should depend on notebook cell execution order.

The supplied notebook has working configuration/repository checks, allocation/toolchain checks, independent CPU/CUDA probes, optional llama.cpp baseline builds and smoke runs, and a results viewer. Sutra runtime/demo cells are explicitly gated until the C++ executables and `scripts/run_demo.sh` are implemented. No real ICE execution has been performed here.

## 2. Course-specific setup findings

Reviewed local course sources (references are relative to this repository):

- `../../CourseMaterial/code/VSCode_on_PACE-ICE.md`: GT VPN/SSO, Open OnDemand, compute-node VS Code, uploading/cloning, quota and debugging. Course support goes through the instructor/TA.
- `../../CourseMaterial/code/CUDA_Starter_Projects_21-26.ipynb`, setup cell 3: request a GPU, `module load cuda`, use the system compiler; **do not use the `ece4122-6122` conda build environment for CUDA**. Use Release for performance measurements.
- `../../CourseMaterial/notebooks/Lecture_14_CUDA_Intro.ipynb`, cell 9: launch Jupyter or VS Code with a GPU allocation; inspect the actual device rather than assume V100.
- `../../CourseMaterial/notebooks/Disk_Space_Check_and_Cleanup.ipynb`: inspect `pace-quota` for both byte and file quotas. `df` alone does not show your personal quota.
- `../../CourseMaterial/SCHEDULE.md`: **tentative final-project deadline December 8, 2026**. Check current course announcements for changes.

The README's generic conda instructions cover the wider course. For this CUDA project, follow the more specific CUDA setup. A standard Python kernel can drive subprocesses; its C++ build subprocess must use the system/compiler-module toolchain. `scripts/ice_env.sh` loads CUDA for each GPU command and selects configurable system compilers. It rejects the course conda environment rather than silently changing it. A module loaded in one `%%bash` cell does not persist into another cell.

## 3. Get the first runnable version started

First target: **one mock request through the full C++ path**, followed by 100 requests from 8 concurrent clients. Begin with batch size 1, FIFO and one worker; defer real inference while constructing the transport/scheduler path.

| Order | Owner | Implement | Verify before proceeding |
|---|---|---|---|
| 1 | All | Freeze fields, timing ownership, framed JSON schemas and CLI names from the delivery plan | Every component serializes/parses the same sample; 64-bit IDs round-trip |
| 2 | All; Dhruv leads build | `CMakeLists.txt`: C++17, threads, core library, mock engine, worker/client/coordinator targets, CTest, optional llama linkage | A mock-only build needs neither CUDA nor llama.cpp; no automatic dependency download during configure |
| 3 | Dhruv | `MockInferenceEngine`: deterministic text, bounded optional artificial delay, controlled failure input; worker delegates to engine | Every input gets the same ID back and one response; a failure does not stop the worker |
| 4 | Aashish | `scheduler/`: bounded queue, mutex/CV, FIFO pop, worker registry/capacity, dispatch outside lock | Empty queue waits without spinning; full queue rejects; busy workers are not over-assigned; shutdown wakes waiters |
| 5 | Jayani | `router/` and shared transport: length prefix, complete read/write loops, client registry, router IDs, result correlation, worker registration | Fragmented/coalesced frames, malformed JSON, disconnects and concurrent clients behave correctly |
| 6 | All | Wire coordinator -> worker process -> result -> client | 100/100 requests terminate; correct client/ID mapping; failure count reported |
| 7 | Dhruv | LlamaEngine: persistent model, chat formatting/tokenization, context limit, greedy generation, state cleanup, CPU/CUDA switch, timings | One real request works over the exact same TCP path; logs prove actual GPU offload |
| 8 | Aashish + Jayani | Priority+aging, timer/count batch formation, round-robin/least-loaded, deadlines/heartbeats, capacity release on every exit path | Deterministic queue/policy tests; killed worker is removed; late replies cannot complete twice |
| 9 | Dhruv + Aashish | True multi-sequence GPU batches; worker checks token/KV budget and splits oversized candidate groups | Separate sequence IDs/KV state; effective batch count logged; compare with sequential aggregation |
| 10 | All | Two GPUs, then multiple nodes if required; continuous slot replacement if required by rubric | Separate GPU UUIDs and worker IDs visible; cross-node connectivity proven before performance claims |
| 11 | Dhruv + all | Benchmark runner, CSV/manifest, plots, scripts and report | Same config/seed/workload across comparisons; warmup omitted; figures regenerate from raw results |

**Aashish's scheduler component is implemented:** `scheduler/scheduler.hpp/.cpp` provides `submit`, `registerWorker`, `setWorkerReady`, `heartbeat`, `waitForDispatch`/`tryDispatch`, `complete`, completion collection, deadline polling and `shutdown`. Policy tests pass explicit steady-clock timestamps; production blocking paths use condition variables. Next work is teammate integration and Linux/ICE validation, using the API and owner checklists in delivery-plan section 10. Socket and llama.cpp calls stay outside queue policy code.

Suggested file boundaries:

```text
common/       existing types + protocol/serialization and transport helpers
router/       TCP clients/workers, pending-ID map, coordinator main
scheduler/    bounded queues, policy, batching, worker selection
inference/    existing engine interface + mock and LlamaEngine
worker/       worker main, readiness, heartbeat, serial execution owner
benchmark/    existing workload generator + client/runner
monitoring/   metrics and GPU sampling
tests/        protocol, scheduler, worker and small end-to-end checks
scripts/      environment wrapper, future demo launcher and Slurm jobs
notebooks/    demo control panel only
```

## 4. First PACE session: exact sequence

1. Connect using the course's GT VPN/SSO instructions. Open `https://ondemand-ice.pace.gatech.edu`.
2. Use Files or a login shell to upload/clone this repository. If cloning fresh: `git clone --recurse-submodules https://github.com/FeynmanLogic/SUTRA-APT6122Project.git`. Upload the new notebook/scripts too if they are not yet pushed. Do not commit model weights/build output.
3. Check `pace-quota` before loading models. Choose an approved writable model/result location with space; keep persistent evidence out of expiring scratch.
4. Launch a **GPU Jupyter session** through Interactive Apps. Initial request: 1 NVIDIA GPU, 8 CPU cores, 16 GiB memory, 1 hour, subject to the current form/course limits. Launch a CPU VS Code session for mock coding if no GPU is available.
5. Open `notebooks/Sutra_PACE_ICE_Demo.ipynb` with a Python 3 kernel. This must be the remote ICE kernel, not local Windows. Avoid selecting the course conda build environment for this CUDA workflow.
6. Edit the configuration cell: repository path if needed, staged GGUF path, CUDA module name/version, and CPU budget. The notebook locates the repository automatically when opened inside it.
7. Run the first configuration, repository and diagnostic cells. Confirm Linux, a Slurm job ID, a running compute-node session, compiler/CMake, quota and the expected GPUs. A Slurm ID is a guard, not proof by itself; inspect hostname/device output.
8. Run the C++17/thread and CUDA-runtime probes. **Both must pass** before a model build. `nvidia-smi` alone proves driver visibility, not successful CUDA compilation/execution.
9. Initialize the pinned submodule if absent using the gated cell. An upload without Git metadata needs a proper clone for dependency/revision tracking.
10. Stage the team's known GGUF (README: Qwen2.5-0.5B-Instruct Q4_K_M); set its absolute Linux path. Record its source/revision and checksum. The notebook hashes it and does not automatically download a large model.
11. Set `RUN_BASELINE=True` and run the build/baseline cells: CPU build, CUDA build, short CPU/GPU generation and a one-repeat microbenchmark. The first build can take several minutes. Save output logs; inspect GPU-offload messages and actual generated text. These are **llama.cpp baseline results**, not Sutra results.
12. When M1 is implemented, build the mock C++ targets and run tests. When M2 is implemented, run the same client through the real GPU worker. Only enable the notebook's Sutra demo cell after the launcher exists and its CLI matches the contract below.

An SSH alternative to OnDemand (outside the notebook): course material uses `salloc -N1 --gres=gpu:1 --mem=8G -t 1:00:00`. Confirm current partition/account requirements; then use an appropriate `srun --pty bash -l` job step so commands run on the allocated compute node. Do not call `salloc` inside a notebook or assume it moves an existing kernel. Launch a fresh remote Jupyter session for the simplest route.

## 5. Execution contract to implement once, then reuse

The CPU CMake/CTest build now runs scheduler tests, but coordinator/worker/client targets, CUDA backend linkage and `scripts/run_demo.sh` remain proposed. The working scheduler-only executable is `./build-scheduler/sutra_scheduler_demo` after building into `build-scheduler`; it uses synthetic replies. The following full-service workflow is still an implementation target; the first three commands already configure/build/test the scheduler into `build-mock`.

```bash
# Mock configure/build/test: first implement this CMake option and these targets.
cmake -S . -B build-mock -DCMAKE_BUILD_TYPE=Release -DSUTRA_ENABLE_LLAMA=OFF
cmake --build build-mock --parallel 4
ctest --test-dir build-mock --output-on-failure --no-tests=error

# Real backend: same runtime, optional llama.cpp linked through CMake.
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSUTRA_ENABLE_LLAMA=ON -DGGML_CUDA=ON
cmake --build build-cuda --parallel 4

# Implement this launcher for both terminal and notebook execution.
bash scripts/run_demo.sh --backend mock --workers 1 --requests 100 --concurrency 8
bash scripts/run_demo.sh --backend cuda --model /absolute/path/model.gguf --workers 1 --requests 30 --concurrency 4
```

`run_demo.sh` must: validate arguments/allocation/model -> start coordinator and worker(s) -> wait for explicit ready signals with a timeout -> run warmup -> run the client -> save results -> gracefully stop all child processes on success/failure/interrupt. Use PID tracking and `trap`; never rely on a notebook `!server` command that blocks forever or leaves orphan workers. Reserve ports for the run and fail clearly on conflicts. The first version supports one node; bind each GPU worker only to its allocated device, preserving Slurm's assignment. Add two-GPU and cross-node launch modes after the one-worker path passes.

Standard output bundle per run, under a unique `results/<run_id>/` directory:

- `manifest.json`: revisions, model checksum, host/GPU assignment, job ID, toolchain, backend, workload/seed, policy, batching, warmup count and `measurement_wall_seconds` for the measured interval.
- `requests.csv`: one terminal row per request with `request_id`, `worker_id`, `success`, `error_code`, `input_tokens`, `generated_tokens`, `queue_time_ms`, `inference_time_ms`, `time_to_first_token_ms`, `latency_ms`, and **`client_latency_ms`**; optional validity fields for undefined timings. CSV `success` uses lowercase `true`/`false`.
- `summary.json`: counts, throughput and latency quantiles; coordinator/worker/client logs.

Treat the notebook as a viewer for these files. Compute aggregate token throughput using the manifest's measured wall interval, not a mean of per-request rates. Scripts also support Slurm `.sbatch` jobs so long measurements survive browser/kernel disconnects. Do not run a long experiment and a live demo simultaneously on the same allocated GPUs.

## 6. Demo sequence and prechecks

Before presenting: allocation has enough remaining time; exact submission commit built in Release; model staged/checksummed; correct GPU count; ports available; no stale workers; warmup completed; result directory writable; baseline/policy configurations fixed; saved fallback outputs available.

Suggested 6-8 minute demonstration:

1. Show allocation/GPU identity and model checksum (30 seconds).
2. Submit one real text prompt and show its ID, selected worker and response (1 minute).
3. Submit concurrent requests and show FIFO vs priority/aging (1-2 minutes).
4. Show real GPU batches and one-vs-two-worker results from the same workload (2 minutes).
5. Deliberately stop one worker using the launcher's tracked PID; show failures reported and healthy worker progress (1 minute).
6. Open saved CSV/summary, explain latency vs throughput, and show reproducibility commands (1 minute).

Build and warm up **before** the presentation. A notebook makes visible cells and results convenient, but it is not the durable process supervisor or benchmark timing authority. Keep a pre-executed notebook/result bundle for an allocation delay; label historical results with their run ID/hardware rather than presenting them as live.

## 7. Calendar targets from the tentative course schedule

| Target | Finish |
|---|---|
| Oct. 24-30 | Freeze interfaces; ICE probes/model baseline; mock build and scheduler/transport skeleton |
| Oct. 31-Nov. 6 | Complete concurrent mock path, then real one-worker transaction |
| Nov. 7-13 | Priority/fairness, batching timers, multiple workers and failures |
| Nov. 14-20 | True multi-sequence batching, two-GPU evidence; cross-node/continuous work if required |
| Nov. 21-24 | Main experiments and first report/plots; capture successful demo before Thanksgiving |
| Nov. 28-Dec. 1 | Repeat key measurements, final integration and feature freeze |
| Dec. 2-7 | Reproduction check, report/demo polish, submission packaging |
| Dec. 8 | Tentative submission deadline; confirm current course announcement/time |

## 8. Troubleshooting

| Symptom | Next check |
|---|---|
| Windows/local hostname or no Slurm job | Notebook kernel is local or outside an allocated compute session; relaunch on ICE |
| `nvidia-smi` works, `nvcc` missing | CUDA module name/version and module initialization; driver and toolkit are separate |
| Unsupported compiler / conda path in build | Fresh session outside course conda environment; inspect `CC`, `CXX`, `CUDAHOSTCXX`; choose a supported system/module compiler; configure a new build directory |
| CUDA probe fails | Read compilation/runtime output before debugging llama.cpp; inspect allocation, device binding, driver/toolkit compatibility |
| Model missing or no space | Staged Linux path, file size/SHA-256 and `pace-quota`; local Windows paths cannot address remote files |
| Model runs on CPU during CUDA test | Build uses `GGML_CUDA=ON`, GPU layers requested, assigned GPU visible, and logs show offload |
| Cell hangs launching the service | Use the bounded launcher, ready timeout and child cleanup; do not run a foreground endless server in a cell |
| Empty CMake / no tests / no launcher | Runtime is still scaffolding; complete the corresponding implementation stage, rather than treating absence as a passing test |

Upstream flag/API references (the checked-out submodule revision remains authoritative): [llama.cpp build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md), [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md). Baseline probes establish environment readiness, not end-to-end Sutra correctness.
