# Campaign R1 verdicts — 2026-08-30

Branch: `checkpoint-d1-baseline`. Image: `…@sha256:9bb1557a…` (digest-pinned,
identical both ranks). Comparison source: `docs/CAMPAIGN-R1.md` /
`results/ab/r1-phase0/`. Prior record: D1 verdict table above.

## Verdict table (fresh re-bench, same protocol, interleaved boots)

| Arm | MNBT | LPT | async | retention | 100k TTFT | Structured ×1 | Prose ×1 | Cache burst r2-3 | Solo replay | HOL first tok | Verdict |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline-r1ref-20260830-0715 | 2048 | — | auto (ON) | — | 200.3 s | 70.21 | 27.15 | PASS | 98.6/98.7% | **461.3 s FAIL** | reference |
| **r1-auto-20260830-0633** | **3584** | **1792** | **0 (off)** | **0** | **199.4 s** | **71.97** | **27.64** | **98.6%** | **98.7%** | **8.39 s** | **ADOPTED** |
| dsd-r1-20260830-0725 | 3584 | 1792 | 1 (ON) + DSD | 0 | — | 66.91 | 28.87 | — | — | — | REJECTED (dormant) |

KV pools: baseline 951,655 (1.59×) / R1 979,591 (1.63×) / DSD 824,571 (1.37×)
of the 600k window — auto pool everywhere (pin REJECTED, see below).

## Adopted

1. **R1 bundle = new serving baseline** (r1-auto-20260830-0633):
   - MNBT 3584 (page-exact) + long-prefill threshold 1792: the prefill gate
     passes (199.4 s < 204.7 s record; effective ~501 tok/s vs ~489), and the
     threshold knob eliminates head-of-line blocking — first token for a
     1.2k request behind a 240k cold prefill: **8.39 s vs 461.3 s** on the
     baseline config (55×). This is the bundle's decisive win; their ~893
     tok/s prefill claim did NOT transfer (~501 here), but the HOL behavior
     does.
   - Structured ×1 **71.97** tok/s (accept **1.0**) — +2.5% vs the fresh
     baseline, +9.2% vs the stale D1 record; the stale-acceptance question is
     resolved: the xgrammar backports put structured acceptance at 1.0 and
     account for most of the delta vs the old 65.9 record.
   - Prose ×1 **27.64** (+1.8% vs fresh baseline).
   - Cache: burst rounds 2–3 **98.6%** (gate 90%), solo replay **98.7%** at
     ~200k (gate 93%) — with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` the
     shared-prefix junctions and the latest boundary survive; the
     "MNBT < 3584 → ~0% hits" claim remains unproven (retention env + aligned
     blocks are what matter), recorded as such.
   - Quality: serving 6/6, acceptance 7/7 (tools 5/5, thinking on/off, vision,
     needles, cache-aware replay).
   - Memory: floors recorded head 0.45–1.15 GiB / worker 4.84–5.04 GiB — the
     D1 standing state persists; nothing raised.
2. **Async OFF in the bundle** (`ASYNC_SCHEDULING=0`): their A/B (async-off ≥
   async-on at ×1) holds here — R1 async-off beat the async-on baseline at
   ×1 structured (71.97 vs 70.21).
3. **KV pin REJECTED** (third freeze, NVRM `NV_ERR_NO_MEMORY` signature;
   `results/ab/r1-phase0/freeze-20260830.md`); auto pool satisfies the pool
   gate at 1.61–1.63×. Hard `ALLOW_KV_PIN` guard shipped.
4. **Ops kit + hardening live**: flusher-until-stable with logged flushes,
   post-init cache drain on both ranks (would have fed NVRM pages ahead of
   the MM warmup burst in the freeze window), watchdog crash/wedge/
   deliberate-stop distinction, Xid/metrics/check-updates monitors installed
   (timers enabled on operator approval).

## Rejected / dormant

- **DSD (Phase 3) — REJECTED, ships dormant**: `DSD_TABLE=1:1:7,2:999:5` +
  async ON. Receipt ACTIVE (per-position acceptance freezes at pos ≥5 under
  c≥2; DSD capture ladder `1 2 4 8 12 18 24`; zero cudagraph downgrades), and
  aggregate ×2/×4 beat R1 by +71%/+35% — **but ×1 structured regressed −7.0%**
  (66.91 vs 71.97), beyond the unconditional −3% rejection threshold. The
  regression is the async-ON cost (matches the Reederey87 async A/B). DSD
  stays default-off. Follow-up arm (not run): isolate `ASYNC_SCHEDULING=1`
  WITHOUT DSD at ×1 to price async alone; if async-only ×1 holds ±3%, revisit
  DSD with a table whose solo row compensates.

## Measurement notes

- `usage.prompt_tokens_details.cached_tokens` is NOT populated by this fork;
  every hit ratio here is a `vllm:prefix_cache_hits_total` delta.
- c2/c4 aggregates are state-sensitive (warm cache from prior probes
  inflates/subtracts TTFT components): DSD-vs-R1 aggregates were taken
  fresh-boot-to-fresh-boot with identical protocols.
- Tags: `baseline-r1-20260830` (this verdict), `recipe-r1-20260830`.

## Phase 3.5 — mixed-chunk-1024 (issue #43) — 2026-08-30

Arm: `GLM53_MIXED_PREFILL_CHUNK=1024` vs `skip`, everything else = R1 bundle
(AB-PLAN Phase 3.5 runbook). Single serve per arm, same-day back-to-back,
identical protocol; **GPU clocks identical both serves (SM 2190 MHz operator
underclock — absolute values sit ~10% under the morning records; the fresh
baseline re-serve is the comparison target per rule 6)**.

| Arm | Struct ×1 | Prose ×1 | Conc ×2 agg | Conc ×4 agg | TTFT ×2 | TTFT ×4 | 100k prefill | HOL | Cache b/solo | Preempt | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| mixed-chunk-base-20260830-1741 (`skip`) | 63.50 | 26.62 | 26.7 | 34.5 | 7046 ms | 12233 ms | 207.7 s | 5.99 s | 98.6 / 96.9% | 0 | reference |
| **mixed-chunk-1024-20260830-1804** | **65.59** | **27.69** | **45.2** | **47.5** | **428 ms** | **528 ms** | 207.3 s | 6.23 s | 98.6 / 96.9% | 0 | **ADOPTED** |

**ADOPTED.** Peer cold prefills now mix (≤1024 tokens/step) into decode
steps: TTFT at ×2/×4 collapses −94%/−96% and aggregate rises +69%/+38%, with
×1 decode *improving* (+3.3%/+4.0%), acceptance unchanged, zero preemptions,
APC hits identical, solo prefill untouched. Cost, recorded: per-stream decode
tok/s during contention drops (×2 27.3→24.8, ×4 16.5→12.6 — the issue #6
TPOT-jitter concern, mild at c≥2) while every round finishes 27–37% sooner.
Answers issue #43's "more than 1 session clogs everything": queueing below
4 running is now prefill-budget-bound only for solo prefills; interactive
multi-client traffic admits and drains ~2× faster. Default flipped to `1024`
(in the adoption commit); `skip` remains available as an inline override.

## Phase 3.5 follow-ups — async-solo + NCCL QPS4 — 2026-08-30 (both REJECTED)

Single-serve arms on the adopted mixed-chunk-1024 config, clocks identical
(2190 MHz underclock) all serves. Evening prose drift caveat recorded
(−9% across post-18:00 serves; see async-solo summary).

| Arm | Struct ×1 | Prose ×1 | Conc ×2 agg | Conc ×4 agg | 100k prefill | HOL | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| async-solo-20260830-1858 (`ASYNC=1`, no DSD) | 64.14 | 25.01 | 36.2 | 44.4 | 214.3 s | 8.0 s | **REJECTED** — async costs ×1, no ×2/×4 win post-mixing; **closes the D5 follow-up: DSD dormant permanently** |
| nccl-qps4-20260830-1932 (`NCCL_IB_QPS_PER_CONNECTION=4`) | 64.58 | 25.18 | 37.3 | 50.3 | 207.8 s | 6.06 s | **REJECTED (recorded)** — target prefill metric unchanged on 2-node CX7; wiring kept for post-reseat re-run |

Reference: mixed-chunk-1024-20260830-1804 (65.59 / 27.69 / 45.2 / 47.5 /
207.3 s / 6.23 s). Zero preemptions and PASS cache gates on both arms.

## Tweet-investigation arm — MNBT 7168 + LPT 3584 — 2026-08-30 (REJECTED, recorded)

Mechanism verified in-image: `long_prefill_token_threshold` caps every chunk
before the MNBT min (scheduler.py:554) — MNBT=3584 was dead weight for cold
prefills while LPT=1792 (explains the R1 bundle's 0.9s MNBT delta). The arm
isolated the scheduler knobs: 100k TTFT −3.2% (gate ≥10% FAIL) and the KV
pool regressed 1.63×→1.34× (805,714 tokens). Verdict: prefill on this kit is
kernel-bound, not step-overhead-bound; revert to 3584/1792 (done). The
tweet's +11% likely came mostly from its bundled MoE-tile patch
(`exl3_moe_inst_0_256.cu` exists in-image, selection internal — overlay
territory). Sibling DeepSeek recipe does ~1638 tok/s with 1024-chunks on the
same cluster — the GLM↔DeepSeek gap is the kernel path. Full analysis:
`results/ab/mnbt7168-lpt3584-20260830-2234/summary.md`.
