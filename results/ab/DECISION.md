# Campaign decision record — Entrpi validation + D1 (2026-08-29)

Campaign branch: `checkpoint-d1-baseline` (main parked at f249a23).
Comparison source: `docs/COMPARISON-ENTRPI.md`.
R1 campaign runbook: `docs/CAMPAIGN-R1.md` (Reederey87 bundle adoption + DSD
concurrency arm; ported components credited in `NOTICE`).

## R1 bundle — implementation status (2026-08-30)

The Reederey87 production kit (same image digest `9bb1557a…`, byte-identical
overlays, same bench protocol) measures 70.4 tok/s structured / 29.5 prose /
~893 tok/s prefill vs our recorded 65.9 / 26.7 / ~489 — configuration-level
wins, ported and re-gated.

**Implemented + offline-verified (this pass):**
- Phase 0 (partial): driver 580.173.02 confirmed on BOTH nodes (580.x branch,
  no downgrade); CX7 ports ACTIVE/200Gb with all error counters 0 on the
  pinned rail; second rail (`rocepP2p1s0f1` class) also active on both nodes
  (dual-rail A/B seed). 30-min `ib_write_bw` soak still required before
  Phase 2. Digest pin + `SKIP_PULL=1` live in `.env`.
- C1 serving config: MNBT 3584 (page-exact), `LONG_PREFILL_TOKEN_THRESHOLD`
  1792 emitted as `--long-prefill-token-threshold` on both ranks (outside
  EXTRA_ARGS), `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` forwarded to both
  ranks (non-empty only — an empty string crashes the fork's env parser),
  `--no-async-scheduling` via the `ASYNC_SCHEDULING` knob (0=off bundle /
  1=on DSD / auto=baseline), `FLASHINFER_WORKSPACE_BASE` inside the mounted
  vLLM cache, KV pin via the exact `--kv-cache-memory-bytes` flag (quoted),
  digest-pinned `IMAGE` + `SKIP_PULL=1` + BUILD-refuse guard, driver-branch
  preflight gate (590.x refuses to boot).
- C2 ops kit: `local/` (cache-burst/cache-probe/acceptance/serving-probe/
  toolcall-probe/xid-check/metrics-alert/check-updates/prod-start + systemd
  user units + installer), watchdog upgrade (crash vs wedge vs deliberate-stop
  + 900 s backoff + optional liveness probe), `NOTICE` attribution.
- C3 wiring tests: `tests/test_r1_bundle.py` 11/11 (inner-script argv executed
  with a stubbed vllm; .env digest-pin assertions; launch/stop/watchdog
  wiring), `tests/test_dsd_wiring.py` 7/7 (now also enforces
  DSD-requires-async), override-capture tests passing.

**Pending (on-GPU, runbook `docs/CAMPAIGN-R1.md`):** Phase 2 bundle A/B
(re-benched baseline arm vs R1 arm, interleaved, gates in the runbook),
Phase 3 DSD concurrency arm (`DSD_TABLE=1:1:7,2:999:5 ASYNC_SCHEDULING=1`,
pin off — recorded deviation), Phase 4 ops activation + `recipe-r1-<date>`
tag. Recorded deviations from upstream: loopback bind NOT adopted (operator
keeps LAN; exposure documented in README); "MNBT < 3584 → ~0% hits" not
adopted blind (our 93% solo hits at MNBT=1024 contradict it — retention env
is the multi-session lever, re-gated by our own cache probes).

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
