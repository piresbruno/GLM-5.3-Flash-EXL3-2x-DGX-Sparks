# R1 freeze forensics — 2026-08-31 in-image torch-profiling freeze (dgx1)

Boot: adopted config (auto pool, no pin, MNBT 3584 / LPT 1792 / async off /
retention 0, mixed=1024) **plus `--profiler-config {"profiler":"torch",
"torch_profiler_dir":"/root/.cache/vllm/profiling"}` via EXTRA_ARGS** — a
prefill-decomposition experiment (RESEARCH-PERF-NEXT addendum, "layerwise
NVTX/profiler" lever).

## Verdict

**Same NVRM OOM freeze class as C4 — triggered by the torch profiler itself.**
First occurrence without a pin and at the stock 0.85 util: 65
`NV_ERR_NO_MEMORY` entries in the frozen boot's kernel journal (b −1), and
journald logging `Under memory pressure, flushing caches` every ~20–40 s from
01:08:09 until the journal ends 01:09:21 (Aug 31). Power cycle required; dgx2
stayed up, its worker container exited cleanly (0).

## Timeline

| Time (Aug 31) | Event |
|---|---|
| 00:33 | profiler boot up (second attempt — first failed validation: `torch_profiler_dir` requires `profiler: "torch"`) |
| 00:43–00:44 | cache-flusher `drop_caches` fires twice; last NVRM warning `refcntRequestReference_IMPL` 00:44:34 |
| ~00:47–01:07 | warmup, `/start_profile`, one ~78k-token prefill (max_tokens=1), `/stop_profile` — request never returned a trace |
| 01:08:09–01:09:21 | journald memory-pressure flushing loop (trace post-processing on host RAM) |
| 01:09:21 | journal ends — host froze; power cycle required |

## Causal chain

On GB10 UMA, host RAM and GPU RAM are the same physical pool. The torch
profiler (CUPTI buffers + on-stop trace serialization/post-processing of ~56
chunked-prefill steps × thousands of CUDA events) explodes host memory while
the engine holds ~12 GiB of GPU reservations. NVRM cannot get pages → the
documented assertion storm → UVM/driver stall → hard freeze. Identical
signature to the C4 pin freezes (`freeze-20260830.md`): 65 NV_ERR_NO_MEMORY,
assertion at `mem_desc.c:1359`, journald thrash, power cycle.

**The profiler — not a pin, not util >0.85 — was the trigger.** Auto-pool
boots had never produced this signature before.

## Rule (standing, added to AB-PLAN discipline)

**Do not run in-image torch profiling of long prefills on this UMA kit.**
Acceptable alternatives, in order of safety:
1. Standalone kernel microbench (`exllamav3_ext` direct, short shapes,
   outside the serving process, bounded memory).
2. Profile only tiny prompts (≤4k tokens) if a serving-path trace is
   unavoidable — never a 100k-class prefill.
3. Out-of-image nsys on a non-serving instance.

## Recovery (per the ritual — followed)

Power cycle → `./start.sh stop` both ranks (worker had exited 0) → relaunch
adopted config (no profiler) → rendezvous clean after machine reboot (no C5
GID drift), pool **1,014,285 tokens = 1.69×**, smoke 200. No residual damage.

## Addendum — guarded reproduction abandoned (2026-08-31 01:0x)

Two guarded boots (profiler config, live MemFree/NV_ERR polling, instant
`docker rm -f` kill-switch on both ranks) failed to reach the pool-sizing
line safely and were abandoned:

1. Window too short (300 s — weight load alone is ~5.5 min at 2.5 s/shard).
2. Threshold miscalibrated — MemFree 24 GiB during weight load is NORMAL
   (~24 GiB weights + page cache in UMA); killed on MemFree < 25. Corrected
   thresholds: storm = NV_ERR > 3, valley = MemFree < 10 GiB. Both boots:
   0 NV_ERR, clean release, no damage.

**Decision: the in-image profiler path is abandoned entirely.** Engine init
on a profiler boot is normal (first freeze's blowup correlated with the
profiled workload itself — CUPTI + `torch_profiler_with_stack=True` during
the trace), so no guarded boot can observe the failure safely. The prefill
decomposition moves to a standalone kernel microbench (exllamav3_ext direct,
short shapes, outside the serving process). Note for the next session:
`--profiler-config` defaults to `torch_profiler_with_stack=True` — any
future in-image profiling attempt (tiny prompts only) should set it False.
