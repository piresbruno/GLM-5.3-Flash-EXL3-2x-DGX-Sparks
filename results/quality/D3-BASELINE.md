# D3 quality baseline — 2026-08-30 (first measurement)

Fleet: adopted config (mixed=1024, R1 bundle, 600k window), temp 0,
thinking **off** (the evals' design), GPU clocks 2190 MHz (underclock).

| Eval | Result | Gate | Note |
|---|---|---|---|
| math500 (n=100) | **66/100 = 66.0%** | 86 (script default) | first D3 measurement = the quality floor; the 86% default gate was calibrated for a different (likely thinking-ON) regime — the recorded floor is what future arms diff against |
| gpqa_diamond (n=50) | **BLOCKED** | 70 | `Idavidrein/gpqa` is gated on HF and the local token has not accepted the dataset terms (404 with auth) — **operator action: accept the gate at https://huggingface.co/datasets/Idavidrein/gpqa, then rerun `python3 tests/eval_gpqa.py`** |

## Eval harness fixes shipped with this baseline
- `fetch_rows` now authenticates datasets-server with the local HF token
  (MATH-500/GPQA went gated upstream — 39aff6f).
- Retries with backoff + **pagination** (datasets-server caps `length` at
  100; `fetch_rows(500)` could never succeed — 4e0ed4f, a8ce7a8).

## Interpretation guardrail
66% at thinking-off is the **floor**, not a regression signal — there is no
prior measurement to regress from. When re-running after any arm, compare
like-for-like (same thinking setting). A thinking-ON run is the natural
companion measurement if the served default (thinking on) is the production
regime; not run here to keep the D3 floor cheap and deterministic.
