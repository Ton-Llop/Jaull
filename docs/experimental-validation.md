# Cross-machine experimental validation

This procedure collects real evidence; it does not make a GPU result comparable by
itself. Use the existing experiment, benchmark, case and bundle contracts. Do not
calibrate the estimator from one run.

1. Freeze the source commit (`git rev-parse HEAD`) and record a clean/dirty worktree,
   Jaull version, OS image, GPU identity, total and available VRAM, driver version,
   runtime family/build and selected backend. Run `uv run jaull scan` and
   `uv run jaull doctor` on each machine. Record the actual runtime build separately;
   backend detection is not proof that a binary supports it.
2. Pin repository, revision, artifact filename, quantization and SHA-256. Recalculate
   the hash on each machine. For a repeatable reference use
   `bartowski/Qwen2.5-7B-Instruct-GGUF`,
   `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, SHA-256
   `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`,
   llama.cpp, context 4096 and one user. The existing RTX 2060 result is documented
   in [B001-R10](qwen2.5-tests/b001-r10-experiment-record.md); it does not stand in
   for a Windows RTX 4060 or rented Linux GPU result.
3. In `uv run jaull ui`, select the same task, workload mode and execution path.
   Record context, concurrency, optional SLOs and all execution flags. The fixed
   Validate prompt is an execution sample, not a concurrent workload test. If raw
   logs are needed, select **Save raw runtime logs** before Validate. Keep the
   displayed experiment ID and saved record path. The `.runtime-log` sidecar is
   opt-in and may contain prompt/model output; review it before sharing.
4. Inspect the persisted `ExperimentRecord`: timestamp, hardware, artifact, SHA-256,
   backend trace, runtime flags/build where available, prediction input, workload
   profile, observation and `PredictionComparison`. Historical records without a
   profile remain valid. An SLO value is requested intent, never observed TPS/TTFT.
5. Run Benchmark on the *same selected plan* and retain benchmark IDs, methodology,
   repetitions and raw logs. A benchmark's recorded context is provenance; do not
   claim its `llama-bench` invocation applied `--ctx-size` when it did not. Do not
   compare validation latency and benchmark throughput as the same measurement.
6. From a campaign evidence directory, create a case using the IDs and optional
   relative log path:

   ```sh
   uv run jaull experiments case create --experiment EXP_ID --benchmark BENCH_ID --evidence logs/run.runtime-log:runtime_log --json
   uv run jaull experiments case validate CASE_ID --json
   uv run jaull experiments case export CASE_ID ./bundle --evidence-root . --json
   uv run jaull experiments case bundle validate ./bundle --json
   ```

   Omit `--benchmark` or `--evidence` if no such record/file exists. Preserve the
   original records and log files; case validation is read-only.
7. Compare only metrics whose method gates pass. `runtime_reported_allocation`
   (llama.cpp device buffers) and NVML process allocation are distinct sources;
   keep both on Linux when available. WDDM may leave process-attributed NVML
   unavailable. Device reserve and safety margin are budget, not process
   allocation; compare physical bytes to measured allocation. Partial-offload
   host RSS may follow `mmap` rather than the predicted host share. When the
   comparison reports `methodologically_unavailable`, retain that result rather
   than replacing it with a percentage or treating it as zero.

Repeat the reference case separately on RTX 2060, Windows RTX 4060 and a rented
Linux NVIDIA GPU. Driver, runtime build and available memory must be recorded for
each run. No result from one GPU is evidence for another.
