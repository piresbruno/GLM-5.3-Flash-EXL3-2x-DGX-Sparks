# Phase 3.5 baseline re-serve (`skip`) — 2026-08-30

Reference serve for the mixed-chunk-1024 arm (issue #43). Same protocol,
same clocks (SM 2190 MHz underclock, identical at arm serve). Full verdict
table + decision: `results/ab/mixed-chunk-1024-20260830-1804/summary.md`
and `results/ab/DECISION-R1.md`.

| Metric | Value |
|---|---|
| Structured ×1 | 63.50 tok/s |
| Prose ×1 | 26.62 tok/s |
| Conc ×2 / ×4 agg | 26.7 / 34.5 tok/s |
| Conc ×2 / ×4 TTFT | 7046 / 12233 ms |
| 100k cold prefill TTFT | 207.7 s |
| Cache burst / solo | 98.6% / 96.9% |
| HOL first token | 5.99 s |
| Preemptions | 0 |
