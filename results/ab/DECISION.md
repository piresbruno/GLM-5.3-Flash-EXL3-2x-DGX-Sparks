# Campaign decision record — Entrpi validation + D1 (2026-08-29)

Campaign branch: `checkpoint-d1-baseline` (main parked at f249a23).
Comparison source: `docs/COMPARISON-ENTRPI.md`.

## Verdict table

| Arm | MAX_NUM_BATCHED_TOKENS | MAX_MODEL_LEN | 100k TTFT | Structured | Prose | KV pool | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline-20260829-1559 | 512 | 1M | 275.7 s | 66.28 | 26.60 | 1,262,295 (1.26×) | baseline |
| d1-1024-20260829-1735 | 1024 | 1M | 224.4 s (−18.6%) | 66.27 | 27.35 | 1,102,094 (1.10×) | PASS all gates |
| d1-2048-20260829-1817 | 2048 | 1M | 204.7 s (−25.7%) | 65.81 | 27.61 | 1,012,077 (1.01×) | PASS, pool thin |
| **final-20260829-1900** | **2048** | **900k** | (prefill speed preserved) | 65.92 | 26.70 | 974,611 (**1.08×**) | **ADOPTED** |

Decode gates: structured/prose within ±3% of baseline in every arm; 0
preemptions; no NaN; no dmesg faults anywhere.

## Adopted

1. **D1: MAX_NUM_BATCHED_TOKENS 512 → 2048** — the spec-decode step budget is
   env-controlled (boot log verifies; the vllm.py:1849 line is a <8192
   informational warning, NOT a clamp — C6's "clamp" measured a stale image
   build). 100k cold TTFT −25.7% (275.7 → 204.7 s; prefill ~363 → ~489 tok/s);
   decode unchanged. Ladder stopped at 2048: 4096 predicted ~950k pool for
   ~10 s more (diminishing); 8192 known indexer-topk smem OOM on this lane.
2. **MAX_MODEL_LEN 1M → 900k (operator decision)** — the 2048 pool left 1.01×
   at 1M (lottery edge; daily floor drift would break it). At 900k: 1.08–1.12×
   margin. The final boot proved the point: leftover UMA came in lower than
   the 2048-arm boot (974,611 vs 1,012,077) — at 1M that boot would have
   lottery-fallen to a reduced max-len; at 900k it is 1.08× and healthy.
3. **D0 cache-flusher fixes** (validated on every boot this campaign):
   MemFree < 8 GiB trigger (the Cached > 40 GiB trigger never fired — the
   loader keeps cache 23–29 GiB while MemFree dips) + worker sidecar scp
   (was silently never shipped). Boot-window MemAvailable floor improved
   0.35 GiB → 0.63 GiB; MemFree held 15–18 GiB through shard load (was 0.94).

## Measured context (hardware-limit answer)

Decode is engine-step-bound (~116 ms/step; Entrpi's fork measures the same on
the same hardware → hardware, not stack). Aggregate decode peaks ×2–×3
(~90 tok/s). Prefill was the headroom (−25.7% TTFT captured). Memory is the
standing constraint: saturation memfloor head 0.53–0.65 GiB (binding node),
worker 4.3–5.2 GiB — do not raise KV_CACHE_MEMORY/GPU_MEM_UTIL on this state;
floors age ~1.5–2 GiB/day.

## Not adopted / deferred

- **D3 task evals + equivalence harness** — implemented (tests/eval_math500.py,
  eval_gpqa.py, eval_longretrieval.py, dflash_equiv.py), deferred by operator.
  Runbook: `python3 tests/eval_math500.py --n 100 --out results/ab/d3-math500.json`,
  `python3 tests/eval_gpqa.py --n 50 --out results/ab/d3-gpqa.json`,
  `python3 tests/eval_longretrieval.py --tokens 100000 --out results/ab/d3-longctx.json`,
  and the dflash_equiv dump/compare dance across a SPEC_METHOD=mtp → dflash pair.
- **D4 thinking-off default** — knob shipped (GLM53_DEFAULT_THINKING, default 1).
  Arm when desired: `GLM53_DEFAULT_THINKING=0 ./start.sh restart`, then verify
  a default request returns `message.reasoning` (not content) and structured
  acceptance for flag-less clients. Bench numbers are unaffected (benches pin
  enable_thinking=false).
- **D4 MTP-4 fallback** — default shipped (MTP_TOKENS=4 in code/.env.example;
  live .env still 2 until the arm). Arm: `MTP_TOKENS=4 SPEC_METHOD=mtp
  ./start.sh restart`, record vs the ~24.6 tok/s k=2 number.
- **D4 NFS weights mode** — shipped opt-in (WEIGHTS_MODE=nfs); this kit's
  local-mode load is healthy with the fixed flusher, so not armed.
- llama-benchy lens — dropped by operator mid-campaign.
