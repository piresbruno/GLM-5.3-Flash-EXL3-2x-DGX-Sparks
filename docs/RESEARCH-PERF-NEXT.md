# Perf research: what's left to win (2026-08-29)

> **R1 update (2026-08-30):** the campaign ran. The Reederey87 bundle was
> adopted as the new serving config (MNBT 3584 page-exact,
> long-prefill threshold 1792, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`,
> async OFF via `ASYNC_SCHEDULING=0`, FlashInfer JIT workspace persisted,
> digest-pinned image, KV-pin procedure via `--kv-cache-memory-bytes`).
> Verdicts below that are superseded: P0-1 (DSD) is armed as the follow-on
> concurrency arm on top of the bundle — requires `ASYNC_SCHEDULING=1` and
> pin OFF (async double-counts the SW-family reservation under a pin),
> enforced by `dsd_validate` since the R1 implementation pass. P0-2 is
> resolved OPPOSITE to the earlier guess: their A/B measured async-off at or
> above async-on at ×1 (structured 69.8 vs 67.4), so the bundle ships async
> OFF and the DSD arm must win back the async cost at ×2–×4 or stays
> default-off. See `docs/CAMPAIGN-R1.md` for the pending on-GPU gates.

Scope: further decode/prefill/throughput improvements for this recipe, researched
against the current upstream (vLLM main, FlashInfer, SGLang, GB10-specific findings)
**and** verified directly inside the live `:exl3` image (`vllm 0.1.dev20051+g487ecf187`).
Companion to `results/ab/DECISION.md` (D0–D4 adopted set).

Headline: **decode is engine-step-bound (~115 ms/step, confirmed identical on the
Entrpi fork → not a stack defect), prefill was already captured (D1, −25.7% TTFT),
and memory is the standing constraint.** The one big unadopted lever left is
**Dynamic Speculative Decoding** — and it turns out the pinned image already
ships the full machinery for it.

---

## P0-1 · Dynamic Speculative Decoding (DSD) — highest-value arm

**What it is.** vLLM "Dynamic SD" (`num_speculative_tokens_per_batch_size` in
`--speculative-config`): a `[start_bs, end_bs, K]` table that changes the draft
length **per concurrency**, e.g. K=7 at batch 1, K=5 at batch 2, K=4 at batch 4.
Merged upstream as PR #32374 (v0.26.0, 2026-07-27); upstream docs list **DFlash
as a tested method**.

**Why this recipe wants it.** C7 measured exactly the trade DSD captures:
k=5 gives **+7.1% (×2) / +5.4% (×4) aggregate** on mixed prompts but **−19%
structured** at ×1. That failed as a *static* re-pin — but a per-batch-size table
keeps k=7 at ×1 (structured 62.9 preserved) and drops k at ×2/×4. The README
concurrency row (×2: 103.3, ×4: 146.5 aggregate) is the direct upside.

**Verified in the live image (not assumed):**

- `SpeculativeConfig.num_speculative_tokens_per_batch_size` exists; DSD lookup is
  live in the scheduler (`dynamic_sd_lookup` → `num_spec_tokens_to_schedule` in
  `SchedulerOutput`).
- The boot runs **V2 Model Runner** (log: `Using V2 Model Runner`) with
  `cudagraph_mode=FULL_AND_PIECEWISE` — the code path that would downgrade DSD to
  PIECEWISE only fires on MRv1 (`_maybe_override_dynamic_sd_cudagraph_mode`
  returns early when `use_v2_model_runner`). No graph-mode penalty.
- `dflash` ∈ `EagleModelTypes` → not blocked by the eagle-family gate; DP=1 so the
  DP-disable doesn't fire.

**The gap that requires an overlay patch — RESOLVED, none needed:** deeper
tracing (2026-08-29 implementation pass) found the active path is **V2 runner
(`gpu/model_runner.py`, `self.speculator` = DFlash2Speculator) + AsyncScheduler**
(auto-enabled: MP executor supports it, dflash ∈ EagleModelTypes, no disable
warning in the boot log). In that combination DSD works natively: the async
scheduler sizes per-step draft placeholders from `num_spec_tokens_to_schedule`
(`_spec_token_placeholders = [-1] * K`), `update_draft_token_ids_in_output`
trims the drafter's static-7 output to K, and the next verify batch is `1+K`.
The static-k `DFlashProposer` only matters on the **V1/MRv1** runner — which
this boot does not use. Consequence: the DSD arm is launcher-wiring only
(`DSD_TABLE` env → speculative-config + DSD-derived capture ladder), plus a
behavioral receipt (`tests/verify_dsd.py`: per-position acceptance must freeze
at pos ≥ K during a c≥2 burst). The live log confirms the seam:
`SchedulerOutput(..., num_spec_tokens_to_schedule=7)` is plumbed every step.
Work items:

1. ~~Thread `num_spec_tokens_to_schedule` through the proposer~~ — not needed:
   the AsyncScheduler placeholder mechanism already consumes it (see above).
   The drafter keeps proposing static k=7; the scheduler trims to K. The
   drafter-side trim (skip the 2 wasted walk steps at K=5) is a possible
   micro-optimization if profiling ever shows the drafter chain matters.
2. Extend the target CUDA-graph capture ladder. With table `[[1,1,7],[2,999,5]]`
   and MAX_NUM_SEQS=4 the verify shapes are 8/12/18/24 → ladder
   `1 2 4 8 12 18 24` (16/32 unreachable under DSD — net graph count drops from
   7 to 7 with a lower max shape). Implemented as `dsd_capture_sizes` in
   `start.sh`; the DFlash2 drafter's own 4-graph ladder is keyed to static k=7
   and unchanged.
3. Arm per AB-PLAN rules: structured/prose ±3%, acceptance ±2 pt, aggregate ×2/×4
   primary metric, memfloor + dmesg gates; DSD receipt via `tests/verify_dsd.py`.

Expected: +5–7% aggregate at ×2–×4, zero ×1 regression. This is the single
remaining measurable, lossless win.

## P0-2 · Async scheduling — likely already on

`MultiprocExecutor.supports_async_scheduling() == True` and dflash passes the
eagle-family gate, so `async_scheduling=None` auto-enables on this config. The
boot log contains **no** "async scheduling ... will be disabled" warning, which
is what the disable path logs — so it is active. If a future image ever shows
that warning, `--async-scheduling` is the free arm that overlaps CPU scheduling
with GPU work in exactly this step-bound regime (zero-bubble async speculative
decoding, vLLM #29957 lineage).

## P1-1 · NCCL micro-tuning for prefill

From independently measured CX7/RoCE baselines (pulsar-gb10-vllm-stack TUNING.md):

- **`NCCL_IB_QPS_PER_CONNECTION=4`**: +9% at ≥256 MB payloads, no small-message
  penalty. D1 moved prefill chunks to 2048-token steps — large collective payloads
  — so this targets the remaining TTFT directly. One-env arm, cheap gate.
- `NCCL_CROSS_NIC=1`: used by both sfxnz's NVFP4 recipe and the DeepSeek-DSpark
  recipe on this hardware class; this recipe pins `NCCL_CROSS_NIC=0`. Likely
  neutral on single-cable kits (single NIC path), but it is the one NCCL delta
  never measured on this kit — low-cost C5-adjacent arm.

## P1-2 · FlashInfer version audit (correctness-adjacent)

**Confirmed: the image runs FlashInfer 0.6.17** (`flashinfer.__version__` =
0.6.17; NCCL 2.30.7 matches sfxnz's pin). sfxnz's v8 layer pins
`flashinfer 0.6.18.dev20260819` + `nvidia-nccl-cu13==2.30.7` +
`cutlass-dsl 4.6.2` and documents **"0.6.17 FA2 MLA NaNs on 64–256 row batches
on sm_121"**. This recipe's C1 indexer hardening kills one NaN *source*, but the
base image carries exactly the version sfxnz flags; the same class of bug may
exist elsewhere in the MLA path. Action: a version bump is a candidate *fenced*
arm (overlay patches target exact source lines — a bump invalidates them, so
treat as an image-era change, not a config knob).

## P2 · Low-confidence knobs (flagged, not recommended blind)

- **`selector_top_k`** (DFlash2 config: `selector_rank 256, selector_top_k 16`,
  `block_size 8`): narrower beam = less selector walk compute, but 16 is the
  trained config; deviating is unvalidated acceptance risk. Only worth an arm if
  the drafter chain shows up in a profile as a meaningful slice of the 115 ms.
- **Draft quantization (MXFP8/W8A8 DFlash2)**: Entrpi measured boot crash on
  their fork, parked there too. Drafter is ~2.3 GiB bf16 with TP=1; the chain is
  kernel-bound small-shape attention, not weight-read-bound — quant gain is
  likely small. Stay parked.
- **FULL cudagraphs for the drafter chain** (vLLM #45258): author's own
  measurement retracted the win to ~1.3% e2e (chain is kernel-bound, not
  eager-overhead-bound). Not worth the engineering here.

## Explicitly evaluated and rejected (do not revisit)

- **SGLang DFlash2 lane** (randomllama 2×Spark): 77.4 agg @ c8, 28.6/23.6 c1 —
  this recipe's 146.5 agg @ c4 / 62.9 c1 leads on every row. No change.
- **NVFP4 packed-KV lane** (incl. the sfxnz recipe): measured 61.9 structured /
  27.3 prose c1, 107.7 agg @ c2 — **at parity with this EXL3 lane** (62.9 /
  26.9 / 103.3), and it caps at 327k context vs this recipe's 900k. No change.
- **5/6bpw EXL3**: K6 is 254 GB — does not fit 2×120 GiB UMA with KV; no public
  5bpw checkpoint exists. No change.
- **FlashInfer sparse-MLA autotune / fused-MoE autotune / PDL**: already gated
  off in the Dockerfile (rank-0 kills / KDA races on SM121 — pre-verified).
- **Official vLLM GLM-5.3 recipe geometry** (GB200 TP8 PD-disaggregation,
  NixlConnector): needs 8 GPUs; not portable to 2×Spark UMA.

## sfxnz repo — what transfers, what doesn't

Repo: `sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark` (user-suggested reference).
Performance: at parity (table above), so nothing perf-positive to import directly.
Three things worth keeping:

1. **KV-pin UMA evidence** — their ladder: 4.14 GiB safe; 5.0 GiB slowed decode
   ~20% at *every* concurrency (UMA pressure); 5.14 GiB crashed under concurrent
   load. Independent confirmation of D2's "explicit budgets are non-linear +
   floors age" methodology; also a reason to *keep* the D1 pool (1.08×) instead
   of pinning more.
2. **`--default-chat-template-kwargs {"enable_thinking": false}`** — semantic
   equivalent of the already-shipped `GLM53_DEFAULT_THINKING=0` (D4). Nothing new.
3. **`expandable_segments:True`, PDL gate, NCCL 2.30.7 pin** — all already
   present in this recipe's Dockerfile/start.sh. Convergent engineering.

## Ops finding: the inter-Spark CX7 link is flaking (fleet died 19:46)

The current head container's log shows the full failure chain, and it is **this
kit's own fabric**, not foreign traffic (the GIDs `::ffff:10.100.24.2` →
`::ffff:10.100.24.1` are the two nodes' native CX7 stacking addresses; NCCL
sockets ride the 10.0.0.x aliases, but the RoCE GIDs carry the real interface
IPs):

| Time | Event |
|---|---|
| 19:00:09 | `rocep1s0f1:1 port error(10)` → 19:00:46 `port active(9)` — link flap under load |
| 19:07–19:10 | `shm_broadcast` "no block found in 60 s" ×2 — TP collectives stalling |
| 19:26–19:29 | repeated `IBV_WC_RETRY_EXC_ERR` (IBV_WC_RECV_RDMA_WITH_IMM / IBV_WC_SEND) between the two nodes |
| 19:15–19:40 | engine `dump_input` error dumps (spec-decode verify steps) |
| **19:46:37** | `RuntimeError: NCCL error: remote process exited or there was a network error` in `pynccl all_reduce` → **fleet death** |

RETRY_EXC_ERR on a directly-attached QSFP DAC is the classic signature of a
marginal physical link (cable/connector/thermal) or a transceiver dropping
under sustained RDMA load. The watchdog recovers the fleet, but the root cause
is hardware-side. Before any further A/B campaign:

1. Reseat / swap the QSFP DAC, check port state and transceiver diagnostics
   (`ibstat`, `ethtool -m`), and run an `ib_write_bw`/`ib_write_lat` soak
   between the nodes to reproduce.
2. Re-check the C5 GID-index story after any re-seat (boot-log GID drift risk).
3. Rule the link clean before benchmarking DSD — otherwise every arm measures
   link jitter, violating AB-PLAN rule 3.

## Suggested next campaign (priority order)

| # | Arm | Gate | Expected |
|---|---|---|---|
| 0 | **CX7 link soak / DAC reseat (ops blocker)** | `ib_write_bw` soak clean ≥30 min | fleet died on this 2026-08-29 19:46 |
| 1 | DSD overlay (dynamic k, capture ladder +10/12/20) | structured ±3% at ×1; aggregate ×2/×4 primary; memfloor | +5–7% aggregate at ×2–×4 |
| 2 | Verify async_scheduling in boot config | record only | resolved: likely already on |
| 3 | `NCCL_IB_QPS_PER_CONNECTION=4` | TTFT gate (C6-style, ≥10% not required; record) | small prefill win |
| 4 | FlashInfer 0.6.17→0.6.18 audit | fingerprint confirmed 0.6.17; fenced image-era arm only | correctness hygiene |
| 5 | File upstream DFlash DSD issue (static-K gap) | link in report | upstream hygiene |

D3 task evals (already implemented, deferred by operator) remain the recommended
pre-campaign quality baseline: run `tests/eval_math500.py` / `eval_gpqa.py` once
on the current final state so the DSD overlay patch has a quality floor to
diff against.

---

## Addendum (2026-08-30): community prefill report — investigated, isolated, closed

A community report (2× Spark, GLM-5.3-Flash EXL3) claimed: (1) the long-prefill
fairness threshold silently caps chunks making MNBT dead weight, (2) MNBT 7168 +
LPT 3584 + "MoE tile 256" bought +11% cold prefill, (3) GLM ~900 tok/s vs
DeepSeek ~1900 leaves a 2× prefill win available.

Findings on this kit (full record: `results/ab/mnbt7168-lpt3584-20260830-2234/`):

1. **Mechanism verified** — `scheduler.py:554` caps chunks at LPT before the
   MNBT min; MNBT=3584 was dead weight while LPT=1792 (explains R1's 0.9 s
   MNBT delta).
2. **Scheduler arm REJECTED** — isolated MNBT 7168/LPT 3584: 100k TTFT −3.2%
   (gate ≥10%) and the KV pool regressed 1.63×→1.34×. Prefill here is
   **kernel-bound**, not step-overhead-bound.
3. **MoE tile: already active** — `exl3_moe.cu` selects the n256 variant
   automatically when hidden%256==0 && intermediate%256==0; GLM-5.3-Flash is
   4096/2048 → **n256 is already dispatched**. The report's tile bump was a
   no-op fix for their own stale selection (or an older exllamav3); nothing to
   import.
4. **The GLM↔DeepSeek gap is architectural** — the sibling DeepSeek recipe on
   this same cluster runs LPT=1024 (smaller chunks) at ~1638 tok/s cold
   prefill (128K→80 s). The gap is the EXL3 4 bpw dequant-MoE + fp8_ds_mla +
   hybrid mamba/DFlash2 path vs NVFP4-KV + DSpark. Config cannot close it.

**Remaining prefill levers, in order of cost:**
- `EXLLAMAV3_TUNE_CACHE` — the coop-autotune cache is env-addressable; a
  seeded cache could benchmark alternative tile/warp configurations without
  patching. Bounded experiment, needs a cache-format read first.
- Layerwise NVTX trace (`enable_layerwise_nvtx_tracing` in-image) + nsys to
  decompose a prefill step (MoE vs attention vs mamba vs NCCL) before touching
  kernels. Recommended before any overlay work.
- Open question for the community report author: their absolute baseline is
  831 tok/s vs our 483 on the same model class — DFlash2 on/off, bpw, and
  util would explain most of it and may reveal a config import.

**UPDATE (2026-08-31, post-freeze):** the "layerwise NVTX/profiler"
decomposition lever is **withdrawn for long prefills** — an in-image torch
profiler capture of a ~78k prefill froze the head node with the documented
NVRM OOM signature (65 × `NV_ERR_NO_MEMORY`; CUPTI + trace post-processing on
host RAM starves NVRM under UMA). Full forensics:
`results/ab/r1-phase0/freeze-20260831-profiling.md`. Standing rule: no
in-image torch profiling of long prefills on this kit. The prefill
decomposition falls back to standalone kernel microbenches (exllamav3_ext
direct, short shapes, outside the serving process). The
`EXLLAMAV3_TUNE_CACHE` experiment remains viable (cache seeding, no CUPTI).
