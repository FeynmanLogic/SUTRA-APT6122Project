# Integration handoff — 14 November 2026

## What was combined

Work is on `integration/scheduler-router-inference`. Existing local branches were preserved; local `main` was not reset or rewritten.

| Source | Integrated tip | Contribution |
|---|---|---|
| `feature/aashish-scheduler` | `02bdbde` | Scheduler, tests, planning docs, ICE notebook |
| `origin/main` | `2bbef4f` | llama.cpp backend, sequential worker, benchmarks, GPU monitoring |
| `origin/feature/tcp-router` | `192803f` | Concurrent framed-JSON TCP router and client checks |
| `origin/feature/aashish-planning-ice` | `e806735` | Already an ancestor of the scheduler branch |

Merge conflicts in CMake, README, and `.gitignore` were resolved. CMake now preserves the dependency-free scheduler build, adds the worker core and tests, and exposes optional router and llama.cpp targets. The router uses `InferenceWorker` in both mock and llama modes. Worker failures with mismatched IDs are mapped back to the original request ID so clients receive a terminal error.

The runtime is one router/coordinator process with one embedded worker and one model context. TCP clients can run concurrently; worker batches execute sequentially. Distributed worker processes, multiple GPUs, true GPU batching, and a service-level benchmark remain future work. The existing `sutra_benchmark` calls the worker directly.

Verified locally on Windows/MinGW in Release:

- Core build without router/llama dependencies and its scheduler/worker tests.
- Router build without llama: all three CTest suites pass.
- Inference-enabled CPU build against pinned llama.cpp `9d1e03616efb7520514d65552c7376296ef9208f`: worker, benchmark, and router compile; all three CTest suites pass using mock inference.
- Scheduler demo completes 8/8 requests; notebook code cells parse.

No model or CUDA toolkit was available locally. Real token generation, GPU offloading, Linux socket/signal behavior, and PACE performance remain manual validation items.

## TCP request contract

Each connection carries one request and one reply. A frame is a four-byte unsigned big-endian payload length followed by that many UTF-8 JSON bytes. Maximum frame size is 1 MiB; zero-length/oversized/incomplete frames close the connection. Malformed JSON and invalid request values receive a framed `type: "error"` reply.

```json
{"version":1,"type":"infer","payload":{"prompt":"Explain CUDA briefly.","max_new_tokens":64,"priority":0}}
```

`prompt` must contain 1–16384 UTF-8 bytes; `max_new_tokens` is an integer 1–512 (default 64); `priority` is an integer 0–2 (default 0). The server assigns the request ID and arrival time. The current default FIFO policy does not prioritize requests; priority scheduling exists in the scheduler library.

Replies contain `version`, `type: "infer_result"`, `request_id`, `success`, `status`, `worker_id`, `generated_text`, `error_message`, `input_tokens`, `generated_tokens`, `queue_time_ms`, `inference_time_ms`, `time_to_first_token_ms`, `latency_ms`, and `tokens_per_second`. Millisecond values are durations. There is no streaming protocol. The earlier `router/tcp_server.cpp` echo experiment is preserved for history and is not a build target.

## Local Windows checks

Run these PowerShell commands from the repository directory. This machine has MinGW and Ninja; adding MinGW to PATH is necessary for the compiler and runtime DLLs.

```powershell
$env:PATH = 'C:\msys64\mingw64\bin;' + $env:PATH
cmake -S . -B build-integration -G Ninja `
  -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe `
  -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe `
  -DCMAKE_BUILD_TYPE=Release `
  -DSUTRA_ENABLE_LLAMA=OFF -DSUTRA_BUILD_ROUTER=ON
cmake --build build-integration --parallel 4
ctest --test-dir build-integration --output-on-failure --no-tests=error
```

The TCP test starts its own server on an available port and stops that child process. It checks 30 concurrent submissions, unique IDs, correct prompt routing, metrics fields, invalid inputs (including integer overflow), fragmented frames, and a disconnected client. Worker tests cover successful batches, empty batches, engine exceptions, missing responses, and incorrect IDs. Scheduler tests retain their existing deterministic coverage.

For a manual two-terminal demo, start the server:

```powershell
.\build-integration\sutra_router.exe --backend mock
```

In another terminal in the same repository:

```powershell
python router/test_router.py
python router/test_invalid.py
```

Mock responses are synthetic and cannot establish inference performance. Press Ctrl+C to stop the manual server. Automated Windows tests terminate their server process; Linux signal cleanup needs the PACE check below.

`SUTRA_BUILD_ROUTER` defaults to OFF, keeping the core build free of downloaded dependencies. When ON, CMake first looks for nlohmann_json >= 3.11, then fetches v3.11.3 if absent. On an offline machine, install a package first or pass an existing JSON source checkout via `-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/absolute/path/to/json`. Its source root must contain `CMakeLists.txt`.

## Manual PACE GPU validation

1. Publish the integration branch using the Git instructions below, then check it out on PACE. Use a GPU compute allocation through your usual course/PACE workflow. Load the compiler and CUDA modules in that terminal; the versions below are those recorded by the inference branch. Confirm availability on your allocation.

   ```bash
   git fetch origin
   git switch integration/scheduler-router-inference
   git pull --ff-only origin integration/scheduler-router-inference
   git submodule update --init --recursive
   module load gcc/12.3.0
   module load cuda/12.6.1
   nvidia-smi
   nvcc --version
   ```

   Keep Slurm's GPU assignment. Use the system/module compiler rather than the course conda environment. Check `pace-quota` before downloading dependencies or models.

2. Stage the team's GGUF model at `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`, or substitute its actual path below. Model files are ignored by Git. The README includes the download command.

3. Build Release with CUDA and router support. Architecture 70 is for the recorded V100; substitute the architecture of your actual GPU.

   ```bash
   cmake -S . -B build-sutra \
     -DCMAKE_BUILD_TYPE=Release \
     -DSUTRA_ENABLE_LLAMA=ON -DSUTRA_BUILD_ROUTER=ON \
     -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
   cmake --build build-sutra --parallel 4
   ctest --test-dir build-sutra --output-on-failure --no-tests=error
   ./build-sutra/sutra_worker models/qwen2.5-0.5b-instruct-q4_k_m.gguf
   ```

   If an inaccessible compiler cache causes `ccache: Permission denied`, reconfigure with `-DGGML_CCACHE=OFF`. CTest exercises the mock TCP path even in this CUDA build. The worker command separately exercises real inference.

4. Start the real TCP service on the compute node:

   ```bash
   ./build-sutra/sutra_router --backend llama \
     --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
     --gpu-layers 99
   ```

   Confirm llama.cpp's startup logs show GPU offloading; a successful build alone does not prove GPU execution. The service announces readiness only after model loading and socket binding. In a second terminal **on the same compute node**, with the same modules loaded, run:

   ```bash
   python3 router/test_router.py
   python3 router/test_invalid.py
   ```

   Successful requests must contain real generated text and unique request IDs. The existing client uses 15-second response timeouts, so a slow allocation may require increasing `sock.settimeout(15)` in `router/test_router.py`. The scheduler has a 120-second request deadline. Press Ctrl+C in the server terminal and verify the process exits; background and client threads are joined. A model call already in progress must finish before the execution thread can exit.

5. Re-run the direct-worker baseline and retain its CSV and GPU logs:

   ```bash
   mkdir -p results
   ./build-sutra/sutra_benchmark \
     --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
     --backend cuda --requests 20 --warmup 3 \
     --concurrency 1 --batch-size 1 \
     --prompt "Explain distributed computing briefly." \
     --output results/integrated_worker_requests.csv --run-id integration-check \
     --gpu-monitor --gpu-output results/integrated_worker_gpu.csv
   ```

   This measures the inference worker, excluding TCP and scheduler time. Do not present it as a distributed-service benchmark. Router `latency_ms` measures scheduler admission through batch completion; network/client time is excluded. Router TTFT is the engine-reported value, not a streamed client measurement. Batches return together after sequential worker processing.

## Git publishing and review

The integration is committed locally; no remote branch was pushed and no PR was merged during this work. From the repository directory on this computer:

```bash
git status
git log --oneline --graph --decorate -12
git push -u origin integration/scheduler-router-inference
```

Open a pull request with base `main` and compare `integration/scheduler-router-inference`:

[Open the GitHub compare page](https://github.com/FeynmanLogic/SUTRA-APT6122Project/compare/main...integration/scheduler-router-inference?expand=1)

Suggested title: **Integrate scheduler, TCP router, and inference backend**. Include the local CTest results and later PACE validation results in the PR. Coordinate review with Jayani and Dhruv because the router entry point and worker contract handling changed. No shared request/response fields were changed. Merging this PR includes the commits from the feature branches, so do not separately merge them again.

If `main` advances before review, integrate it on this branch and repeat the build/tests:

```bash
git fetch origin
git switch integration/scheduler-router-inference
git merge origin/main
# Resolve any conflicts, then build and run CTest before pushing.
git push
```

After the PR is merged, preserve the existing local planning commit by updating local main with a normal fast-forward when possible:

```bash
git switch main
git fetch origin
git merge --ff-only origin/main
```

The planning commit is included in this integration history. If GitHub squash-merges the PR, `--ff-only` may refuse because the original local history differs; inspect that history before choosing a normal merge. Do not force-push or reset main to resolve that situation.
