# Arm: MNBT=7168 + LPT=3584 (tweet investigation, issue #43 follow-up) — 2026-08-30

Branch `arm/mnbt7168-lpt3584`. Motivation: a community report (tweet) claimed
vLLM's long-prefill fairness threshold silently caps chunks (making MNBT
dead weight) and that MNBT 7168 + LPT 3584 + "MoE tile 256" bought +11%
cold prefill (77s→70s on 64K).

## Mechanism: VERIFIED in our image

`vllm/v1/core/sched/scheduler.py:554` caps every chunk at
`long_prefill_token_threshold` before the MNBT budget min — with LPT=1792,
solo cold-prefill chunks are 1792 tokens and **MNBT=3584 never binds**. This
explains our own R1 result: the bundle's MNBT bump (2048→3584) moved 100k
TTFT by only 0.9s — both configs ran identical 1792-token chunks.

## Results (vs adopted reference `mixed-chunk-1024-20260830-1804`)

| Metric | Gate | ref (3584/1792) | arm (7168/3584) | Verdict |
|---|---|---:|---:|---|
| **100k cold prefill TTFT** | ≥10% better | 207.3 s | **200.7 s (−3.2%)** | **FAIL** |
| **KV pool** | ≥ 1.6× window | ~980k (1.63×) | **805,714 (1.34×)** | **REGRESSION** |
| Structured ×1 | ±3% | 65.59 | 65.54 | PASS (−0.1%) |
| Prose ×1 | ±3% | 27.69 | 25.85 | drift caveat (evening band) |
| Conc ×1 / ×2 / ×4 agg | ±3% | 38.0 / 45.2 / 47.5 | 35.6 / 38.4 / 43.7 | mixed, state-sensitive |
| Preemptions | 0 | 0 | 0 | PASS |
| dmesg indexer smem | clean | — | clean (0 faults) | PASS |
| GPU clocks | same | 2190 MHz | 2190 MHz | valid A/B |

memfloor artifact: sampler failed this run ("Floor missing on one or both
nodes") — moot for a rejected arm; the boot-log pool number above is the
memory evidence.

## Verdict: REJECTED (recorded) — revert to MNBT=3584 / LPT=1792

Doubling the chunk size halved the step count (56→28) and bought only 3.2% —
**on this kit the 100k cold prefill is kernel-bound (EXL3 dequant-MoE +
fp8_ds_mla sparse MLA + hybrid mamba/DFlash2 state), not
step-overhead-bound**, and the price was a 1.63×→1.34× concurrency-margin
regression (≈175k KV tokens diverted to activations).

## Cross-checks that close the tweet's thesis

- Sibling DeepSeek recipe on this same cluster: `LONG_PREFILL_TOKEN_
  THRESHOLD=1024` (smaller chunks!) and ~1638 tok/s cold prefill (128K→80 s)
  — chunk size cannot explain a 3.4× per-chunk efficiency gap. The GLM↔
  DeepSeek prefill gap is the kernel path, image-era work.
- The tweet's "+11%" bundled THREE changes; our isolated scheduler test
  attributes only ~3% to the scheduler knobs — their remaining gain likely
  came from the MoE-tile patch (`exl3_moe_inst_0_256.cu` exists in our image
  but selection is internal; no env knob — overlay-patch territory).
- Their absolute baseline (831 tok/s on 64K) is 1.7× ours (483 on 100K);
  config unknown — DFlash2 residency + padded slot-share + vision tower are
  candidate explains. Worth asking the author for their config before
  chasing the gap further.
