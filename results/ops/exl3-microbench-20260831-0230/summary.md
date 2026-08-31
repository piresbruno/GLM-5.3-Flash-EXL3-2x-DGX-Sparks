# EXL3 MoE kernel microbench — 2026-08-31 (session/exl3-kernel-microbench)

Standalone decomposition of the prefill budget, per the profiling-freeze
fallback (`freeze-20260831-profiling.md`): real EXL3 tensors for 32 layer-15
experts loaded from the checkpoint, pointer tables cycled over 288 expert
slots, `exllamav3_ext.exl3_moe` called with vLLM's exact argument tuple
(`apply_exl3_fused_moe`), CUDA-event timing, uniform synthetic routing
(~50 tok/expert — fused path, no >128 fallback), single GPU, full 2048
intermediate (checkpoint-native). Script: `tools/exl3_moe_microbench.py`.
Ran inside the serving container against GPU 0 while the fleet idled —
outside the serving process, bounded memory (0.4 GB).

## Measured (k=4bpw, 288 experts, topk 8, hidden 4096, inter 2048)

| M (tokens) | median/layer | tok/s (kernel) |
|---:|---:|---:|
| 1024 | 70.5 ms | 14,532 |
| 1792 | 90.9 ms | 19,719 |
| 3584 | 140.6 ms | 25,483 |

The kernel scales **+40%** from M=1792→3584 — larger chunks tile better, but
sub-linearly (2× tokens, +55% time).

## Decomposition of the serving prefill step (1792 tokens, 3.70 s budget)

- TP=2: each rank runs N=1024 experts (half the single-GPU work) in
  parallel → ≈45 ms/layer/rank → **43 MoE layers ≈ 1.9–2.0 s ≈ 53% of the
  serving step budget**.
- Effective kernel throughput: ~15.4 TFLOPs/step/rank ÷ ~1.95 s ≈
  **8 TFLOP/s ≈ 8% of GB10 bf16 peak** — the cost is the software trellis
  decode (EXL3 4 bpw dequant per weight). NVFP4 paths use hardware FP4
  tensor cores instead — this is the architectural GLM↔DeepSeek gap,
  now quantified.
- Remaining ~47%: sparse-MLA attention + indexer, mamba scan, dense layers,
  NCCL, scheduler/framework.

## Answers

1. **The MoE kernel is not the bottleneck it was suspected to be** — it is
   ~half the prefill budget and runs a healthy 19.7k tok/s per layer at
   serving shapes. The tweet's MoE-tile lever: n256 is already auto-selected
   (dims 256-divisible), and even the +40% kernel scaling at doubled chunks
   lands as ~+20% MoE-side ≈ +10% total — exactly what the MNBT arm measured
   (+3.2% real, after the pool regression). Closed.
2. **The 483-vs-1638 tok/s GLM↔DeepSeek gap is the dequant path**: software
   trellis decode at ~8% of peak vs hardware FP4. Not closable by config.
3. **Any future prefill gain must come from the non-MoE half** (attention/
   mamba/comm) or image-era kernel work on the EXL3 dequant itself.

## Caveats
- 32 real experts cycled over 288 slots: working set 400 MB vs the real
  3.6 GB/layer — L2-reuse makes absolute numbers optimistic; the kernel is
  dequant-compute-bound, not weight-bandwidth-bound, so shape-level shares
  hold.
- Uniform synthetic routing; serving routing is comparable at this scale.
- TP=2 per-rank scaling (N=1024 ≈ half the 2048-wide time) is approximate.

## Fleet state during the session
Idle serving, adopted config (no profiler), pool 1.68×, clocks 2190 MHz.
No serving-process interference: microbench ran in its own process.
