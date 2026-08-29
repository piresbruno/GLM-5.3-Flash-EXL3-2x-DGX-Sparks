# D1-1024 (MAX_NUM_BATCHED_TOKENS 512 → 1024)

All gates PASS — budget verified in the boot log (no scheduler clamp on the
current image; C6's "1024 clamp" measured a stale image build).

| Metric | Baseline (512) | D1-1024 | Gate | Verdict |
|---|---:|---:|---|---|
| 100k cold TTFT | 275.7 s | **224.4 s (−18.6%)** | ≥10% better | PASS |
| Structured tok/s | 66.28 | 66.27 | ±3% | PASS |
| Prose tok/s | 26.60 | 27.35 | ±3% | PASS |
| KV pool @1M | 1,262,295 (1.26×) | 1,102,094 (1.10×) | fits 1M | PASS |
| Boot-window floor (head) | 0.35 GiB | 0.63 GiB | improved | PASS |
| Preemptions / NaN / dmesg | 0 | 0 / 0 / clean | clean | PASS |

TTFT / ITL (per-run, temp 0): structured TTFT med 0.464 s / ITL ~15.1 ms;
prose TTFT med 0.472 s / ITL ~36.5 ms; 100k decode 35.5 tok/s.

Saturation memfloor (×1/×2/×4 @15k): TTFT 32.9 / 52.2 / 90.7 s; decode 33.1 /
30.8 / 27.3 tok/s. Floors under saturation: head 0.65 GiB (binding),
worker 5.17 GiB — first measured saturation floor on this kit.

Pool cost note: 1024 costs −160k KV tokens vs 512 (the memory profiler sizes
activations from MNBT). Still fits 1M with the boot-lottery unused.

Flusher re-validation: MemFree held 15–18 GiB through the whole shard load
(baseline boot: 0.94 GiB); worker sidecar running after the scp fix.

llama-benchy lens dropped by operator decision.
