# Improvement loop

Lōkahi gets faster through one repeated loop. It starts at a kernel on one Mac
and ends in the Common Compute fleet. Every iteration has the same shape, and
each level only promotes a change that survives the level above it. This page
says how an iteration runs, which tools measure it, and what counts as a win.
Claims still follow [`PERFORMANCE_CONTRACT.md`](PERFORMANCE_CONTRACT.md).

## One iteration

1. **Measure** the fixed matrix on the current head:
   - Qwen3 0.6B, Llama 3.2 1B and Gemma 3 1B;
   - 512 prompt tokens and 128 generated, greedy, batch 1;
   - a fresh MLX-LM control on the same run.
2. **Attribute** decode time to sections (embed, qkv, attention, mlp, lm_head,
   and so on). Each section gets the bandwidth it achieved against the chip's
   published peak.
3. **Rank** the candidates by time share × distance from the bandwidth floor
   ÷ effort. The first number comes for free: `recoverable_seconds` is the
   time a section spends above `bytes / peak`.
4. **Change one thing**, aimed at the top of the ranking.
5. **Verify correctness.** Oracle parity on generated fixtures, then exact
   greedy tokens against the previous head and against MLX-LM.
6. **A/B on hardware.** Use warm, serialized runs with the same workload and
   OS build.
7. **Promote or reject.**
   - A change is kept only if it wins beyond noise at its own level and still
     wins one level up: kernel → phase program → end-to-end model →
     Common Compute canary.
   - Rejections are recorded too, so the same idea is not tried twice.

A ≥10% end-to-end win with exact tokens is the bar for an MVP claim against
MLX-LM. Smaller verified wins still move the recorded best.

## Levels and speeds

| Level | Evidence ([ladder](PERFORMANCE_CONTRACT.md#evidence-ladder)) | Where it runs | Speed |
|---|---|---|---|
| Kernel | L1 correctness, then L2 kernel hardware | Cloud and Linux CI for L1; self-hosted Mac for L2 | Inner: minutes |
| Phase program | L3: a whole prefill or decode step | Self-hosted Mac (`lokahi profile`, `lokahi bench`) | Middle: one push |
| Standalone model | L4: the scoreboard matrix vs MLX-LM | Self-hosted Mac (scoreboard) | Middle: one push |
| Common Compute canary | L5, then [gates](COMMON_COMPUTE_RUNTIME.md#acceptance-ladder) E–G | Fleet `lokahi_bench` workload on every chip family | Outer: days |

The middle loop is closed. Each push to a `claude/**` branch goes through
these steps:

1. The self-hosted Mac builds the head and runs the parity suites.
2. It runs the scoreboard.
3. The scoreboard is posted as one sticky comment on the branch's pull
   request, and a regression fails the check.
4. The comment and the check wake the agent session that owns the PR, which
   picks the next change from the ranking.

## Tools

| Tool | What it produces | Evidence kind |
|---|---|---|
| `lokahi profile` ([native/engine](../native/engine/README.md#profiling)) | Per-section time, calls and minimum traffic for prefill and decode | Attribution only |
| [`benchmarks/attribution.py`](../benchmarks/attribution.py) | Sections ranked by recoverable time, with achieved GB/s against the chip's peak ([bandwidth table](../benchmarks/targets/apple_silicon_memory_bandwidth_v1.json)) | Attribution only |
| [`benchmarks/scoreboard.py`](../benchmarks/scoreboard.py) | The matrix vs MLX-LM and vs the recorded best; promotion records; the PR comment | Hardware |
| [`apple-silicon` workflow](../.github/workflows/apple-silicon.yml) | Runs all of it on the self-hosted Mac and posts the comment | Hardware |

**Profiling changes the timing.** In profiling mode the Metal backend ends,
commits and waits for a command buffer after every section. It reads that
buffer's GPU start and end times. Each section's time therefore includes one
command buffer's launch cost, and wall time includes every synchronization.
So:

- use shares and recoverable time from `profile` to decide what to work on;
- use `bench` and the scoreboard to decide whether a change won.

Section traffic is the minimum the math requires: each weight is read once,
float32 activations are read and written once, and attention reads K and V
once per KV head for each visible key. Real traffic is at least this, so a
section's achieved GB/s is an upper bound.

## Scoreboard rules

- **Comparable or not compared.** A run is compared with the recorded best
  only for the same chip, memory size, model revision, workload and backend.
  Otherwise the verdict is "no baseline", with the reason.
- **Noise threshold.** A change counts only beyond
  `max(3%, 2·√(cv_candidate² + cv_best²))`. Here `cv` is a run's coefficient
  of variation across its repetitions. A noisy run widens its own threshold
  rather than producing a false win.
- **Gating metrics.** Decode tok/s (higher is better) and time to first token
  (lower). Prompt tok/s and peak memory are reported but do not gate.
- **Statuses.**

  | Status | Meaning |
  |---|---|
  | improved | Some gating metric is better beyond noise and none is worse |
  | regressed | A gating metric is worse beyond noise, or tokens changed between repetitions |
  | unchanged | Neither of the above |
  | no baseline | No comparable best is recorded |
  | not staged | The pinned snapshot is not on the runner |
  | failed | The engine did not produce a report |

  The run exits non-zero on "regressed" or "failed".
- **Bests are committed, not inferred.** Recorded bests live in
  [`benchmarks/results/scoreboard/<hardware>/<model>.json`](../benchmarks/results/scoreboard/README.md).
  For deterministic runs that improved or had no baseline, the scoreboard
  writes a compact record under `promote/` in the evidence artifact and in the
  comment. Committing that record is the promotion. Review sees the change and
  `git log` keeps the history.

## Ranking in practice

Decode at batch 1 reads every weight once per token, so the floor for a step
is about `weight_bytes / peak`. The table below sets the recorded MLX-LM M1
baselines (hardware evidence, August 2026, 512/128 greedy) against that
floor. The floor is arithmetic from Apple's published 68.25 GB/s, not a
measurement.

| Model | Weights | Floor | Floor tok/s | MLX-LM tok/s | MLX-LM share of floor |
|---|---:|---:|---:|---:|---:|
| Qwen3 0.6B 4-bit | 335 MB | 4.9 ms | 203 | 81.4 | 40% |
| Llama 3.2 1B 4-bit | 695 MB | 10.2 ms | 98 | 69.9 | 71% |
| Gemma 3 1B QAT 4-bit | 733 MB | 10.7 ms | 93 | 58.2 | 62% |

The smallest model is furthest from its floor, which points at fixed
per-token costs (dispatches, host round trips) rather than bandwidth. The
scoreboard shows how far each section is from its own floor.

Expected order of work once the Metal path is verified on hardware
([roadmap G1–G2](ROADMAP.md#gpu-metal-primary-engine)):

1. **Gaps between dispatches.** If the decode wall time is well above the
   sum of the fast-path section times, the GPU is idle between dispatches.
   The fixes are fewer dispatches, persistent encoders and indirect command
   buffers.
2. **Matrix–vector kernels below about 70% of peak** (qkv, mlp, lm_head).
   The fixes are rows per SIMD group, vectorized unpacking and better weight
   layout.
3. **Attention at long context.** The fix is half-precision KV, which halves
   attention traffic.

## The loop improves itself

Some findings are about the loop rather than the engine. These are worked on
like any other ranked item:

- **Unexplained time.** Section times that sum well below the measured wall
  time mean a missing section.
- **Too much noise.** A noise threshold above about 5% means runs are too
  noisy to resolve the wins being chased. The fixes are more repetitions, an
  idle machine, or a pinned power state.
- **A fault that got through.** A regression that the scoreboard passed but
  the fleet caught means the matrix is missing a shape. Add it.
- **No proof yet.** Any claim the scoreboard cannot yet check (ANE
  participation, SSD paging, batching) gets a correctness oracle and a
  hardware record before it gets an optimization.

## Ledger

Promoted and rejected changes, newest first. Every row names the commit, the
level reached and the evidence.

| Date | Change | Level reached | Result | Kept |
|---|---|---|---|---|
| — | First entries follow the first self-hosted scoreboard run | — | — | — |
