<h1 align="center">GLM-5.3 Flash EXL3 for 2x DGX Sparks</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

OpenAI-compatible vLLM serve of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) as
**[Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)**
— a byte-identical public mirror of
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
snapshot `5ab363a8…` (uniform-K4 EXL3/TR3 routed-experts, 4 bpw, ~164 GiB, 120 shards)
so this recipe stays fetchable if the upstream Hub id moves. On a **2× NVIDIA GB10**
kit: tensor-parallel size 2 over CX7, native `sm_121a` cubins, API on `:8888`.
Served model id: **`GLM-5.3-Flash-EXL3`**. EXL3/TR3 quant by
[brandonmusic](https://huggingface.co/brandonmusic).

This is **EXL3 weights + fp8 KV** on GB10. Do not pass `--moe-backend marlin`.
The Hub card on brandonmusic (TP2/EP2/DCP2 + calibrated NVFP4 MLA KV) is the SM120 B12X
image (`verdictai/glm53-flash-exl3-k4:…-v84-dflash2`), not this overlay. Target KV
stays packed **`fp8_ds_mla`**. Speculator is **DFlash2 k=7**
([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2));
draft attention is **FLASH_ATTN** (do not pin `TRITON_ATTN` — that mask is causal
inside the draft block on this image and collapses later-position accept).

## Decode (this kit, 2026-08-28)

Official numbers: sparkDash Decode bench, DFlash2 k=7, **Structured** (count 1→200) and **Code** (`clamp_00`…`clamp_49`) — same high-accept regime. Temp **0**, thinking **off**, 400 tokens, CUDA graphs, fused EXL3 MoE. Prompt types, not grammar / schema. Stream tok/s is per request; aggregate is all streams.

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| **×1** | **719 ms** | **62.9** | **62.9** |
| **×2** | 6.62 s | 51.7 | 103.3 |
| **×4** | 6.30 s | 37.1 | 146.5 |

Serve recipe is `--max-model-len 1000000` with KV pool **1,754,237** tokens (1.75× a full 1M request) at util 0.87. These runs are warm / empty KV — they do not need a filled 1M cache.

Lab `tests/bench_decode.py` on the same protocol (median of 5 × 400, C1): Structured **61.7** tok/s (0.918 accept / 6.43 per step); Prose (hash-map) **26.9** (0.332 / 2.33). Long context / mixed (~60–100k KV) 24–27. MTP k=2 baseline ~24.6.

Structured per-pos (lab median): **0.98 / 0.98 / 0.94 / 0.94 / 0.91 / 0.83 / 0.83**.
Prose per-pos: **0.75 / 0.58 / 0.41 / 0.28 / 0.16 / 0.09 / 0.06**.
Pinning `attention_backend=TRITON_ATTN` dropped structured to ~29 tok/s / 0.31 accept
(pos0 healthy, later positions collapsed).

Re-measure:

```bash
# structured (count 1→200)
python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence --out /tmp/glm53-structured.json
# prose (hash-map explanation)
python3 tests/bench_decode.py --phase prose --runs 5 --max-tokens 400 --skip-coherence --out /tmp/glm53-prose.json
```

## Quality (KLD)

Independent teacher-logit panel from
[malaiwah on the 4bpw discussion](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1#6a9144846b0bdba943bfe86f):
KLD(teacher ‖ model), five cold runs, 25 sealed windows (51,175 positions). This
scores the **weights**, not this GB10 overlay. We serve the **4bpw** row.

| Model | Mean KLD (nats) | Size |
|---|---:|---:|
| TR3 K6 (6bpw) | 0.013723 | 254 GB |
| Official FP8 (cross-stack) | 0.020615 | 328 GB |
| **This checkpoint — EXL3 4bpw** | **0.024555** | **176 GB** |
| Official FP8 (brandonmusic stack, v44) | 0.024629 | 328 GB |
| NVFP4 (brandonmusic stack, v44) | 0.060535 | ~180 GB |

On the same stack, 4bpw matches official FP8 (~1.00× KLD) at **54%** of the bytes.
K6 (`malaiwah/GLM-5.3-Flash-TR3-6bpw`) is a different checkpoint. Padded DFlash
slot-share is an allocator change only — target KV stays packed `fp8_ds_mla`,
same path as the compact-64 fp8 serve (not NVFP4 KV).

## What runs

| Layer | Runtime |
|---|---|
| API | vLLM OpenAI (`/v1/chat/completions`) on the head, port **8888** |
| Weights | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (mirror of `brandonmusic/…` snapshot `5ab363a8…`) |
| Model id | `GLM-5.3-Flash-EXL3` (`--served-model-name`) |
| Image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` FROM `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (arm64, CUDA 13.0) |
| Executor | `mp`, `--nnodes 2`, `--tensor-parallel-size 2` |
| Head | this machine, `HEAD_IP=10.0.0.1`, container `glm53-exl3-head` |
| Worker | `WORKER_USER@WORKER_IP` (this kit: `zurih@10.0.0.2`), `--headless`, `glm53-exl3-worker` |
| Fabric | CX7 QSFP: `enp1s0f1np1`/`rocep1s0f1` ↔ `enp1s0f0np0`/`rocep1s0f0`. Image NCCL (`USE_HOST_NCCL=0`) |
| Attention | `FLASHINFER_MLA_SPARSE_SM120` (NoPE MLA padded into GLM_NSA 576-wide) |
| KV | `--kv-cache-dtype fp8` → packed **`fp8_ds_mla`** (target). Draft DFlash2 KV is `auto`/bf16. Live pool **1,754,237** tokens / **1.75×** at 1M / 690 GPU blocks / 18.67 GiB. `--enable-prefix-caching` (block-aligned hits; see Prefix caching) |
| Experts | packed trellis + suh + svh + mcg, codebook MCG, **one fused `exllamav3_ext.exl3_moe` launch per layer** |
| Dense / shared / attn / embed / lm_head | native (unquantized) |
| Spec | **DFlash2 k=7** (`incoai/GLM-5.3-Flash-DFlash2`); draft KV `auto`/bf16, draft TP=1, FLASH_ATTN. Rollback `SPEC_METHOD=mtp` |
| Context | **1M** (`MAX_MODEL_LEN=1000000`). Live pool **1,754,237** tokens (1.75×) / 690 GPU blocks / 18.67 GiB. Padded slot-share is why 1M allocates; the old 900k cap was 1.95× on this same pool. Do not drop to 256k to “free” slots (hybrid mamba + DFlash window block-id demand is mostly length-independent) |
| Tools / reasoning | `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45` |
| Graphs | on (`ENFORCE_EAGER=0`) — MTP capture `1 2 3 4 6 8 12`; DFlash2 capture `1 2 4 8 16 24 32` |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) — image + video, `--limit-mm-per-prompt {image:4,video:1}`, `--skip-mm-profiling` |

Kernels: `TORCH_CUDA_ARCH_LIST=12.1a`. ExLlamaV3 pin `c5d9c657` (0.0.43) exposes
`exl3_moe` / `exl3_moe_max_concurrency`; aarch64 CPU allreduce stubs in
`overlay/patch_exl3_ext_aarch64.py`.

## Why the overlay exists

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads this checkpoint and dies on
the first forward: `pe_dim must be 64 for fp8_ds_mla`. GLM-5.3-Flash is **NoPE MLA**
(`qk_rope_head_dim=0`, `kv_lora_rank=512`). On SM12x the only sparse-MLA backend is
`FLASHINFER_MLA_SPARSE_SM120`, whose packed record is 512 NoPE + 16 B scales + 128 B
RoPE (656 B). The overlay zero-pads the 512-d latent into that GLM_NSA geometry
(RoPE pad is zeros; the QK dot is unchanged) and registers a real EXL3 method so
routed experts stay packed instead of expanding to BF16.

Registering the name `"exl3"` is not enough. Experts must stay **trellis + suh +
svh + mcg** and run Trellis/MCG. Shared experts, attention, embeddings, and
`lm_head` stay native. TP=2 shards gate/up **column-wise** and down **row-wise**;
the MoE runner all-reduces once per layer.

DFlash2 on this fork also needs three GLM-specific hooks the stock image lacks:
EAGLE3 aux capture at mHC (`hc_post` then `hc_contract` → 4096-wide, taps log as
`(6, 15, 25, 34, 43)`), drafter SWA **padded slot-share** onto the MLA tensors
(`block_size=64`, `page_size_padded` equal to the MLA page so drafter layer i
co-owns MLA tensor i; the layout validator allows that one padded case), and
checkpoint `is_causal: false` so draft attention is bidirectional inside the
block. Draft KV is forced `auto` because dense DFlash2 cannot use the target's
`fp8_ds_mla` layout and SM121 has no FA3/FA4 for plain FP8.

The pinned vLLM `487ecf187` also predates two merged XGrammar speculative-decode
fixes. `overlay/patch_xgrammar_termination.py` source-exactly backports
[vLLM PR #52805](https://github.com/vllm-project/vllm/pull/52805)
([commit `12f64b39`](https://github.com/vllm-project/vllm/commit/12f64b39d29282437e35be9aa5db432fb2a1a6e6))
and [vLLM PR #53046](https://github.com/vllm-project/vllm/pull/53046)
([commit `c6e19b3`](https://github.com/vllm-project/vllm/commit/c6e19b3be24338759a443e03c8325d76da9ee202)).
The first stops `accept_tokens()` and `validate_tokens()` at the first
terminating token, ignores later advances after termination, and clears the
cached flag on reset. The second validates drafts produced before a mid-window
reasoning-end marker before advancing the newly active grammar, avoiding a
spurious `Failed to advance FSM` error for invalid drafts. The fail-closed,
idempotent script preflights both source files before writing and is already
mounted and run on both ranks. These address issue #19's matcher-error paths;
they do not reinterpret a client request that combines GLM XML tool output
with a JSON response schema, nor do they remove cold-prefill queue time.

`overlay/patch_glm_video_placeholders.py` routes Glm5Next video timestamps through
the glm46v path and aligns placeholder blocks to encoder `grid_t`. The overlay
also disables GB10 `persistent_topk` so long-history decode uses
`top_k_per_row_decode`.

## KV cache (this kit, 2026-08-29)

`--kv-cache-dtype fp8` is required. The SM12x sparse-MLA kernel only accepts packed
`fp8_ds_mla`. **bf16 KV has no sparse kernel** on this arch. Metrics report
`cache_dtype=fp8`; that is the **target** path. The logged **1,754,237** tokens
are hybrid BlockPool accounting, not 1.75M tokens of uniform fp8 tensors.

| Piece | Dtype / layout | Notes |
|---|---|---|
| Target MLA (12 layers) | packed **`fp8_ds_mla`**, 656 B/token/layer | `FLASHINFER_MLA_SPARSE_SM120` |
| Indexer / kpool tail | follows the GLM-5-Next hybrid groups | kernel block 64 |
| Mamba (33 layers, 3 groups) | `mamba_cache_dtype=auto` | window / state, mostly length-independent |
| DFlash2 draft (5 SWA layers) | **`auto`/bf16**, 2048 B/token this boot | no MLA FP8 backend on SM121 |

With DFlash2 + vision + util **0.87**, the pool is leftover UMA after weights and
CUDA graphs. Live boot (confirm `padded slot-share` and `Maximum concurrency`
in the log):

| | |
|---|---|
| GPU KV cache size | **1,754,237** tokens |
| Max concurrency at 1M | **1.75×** (same 1.75M pool; was 1.95× at 900k) |
| GPU blocks | **690** (`block_size=64`, `mamba_block_size=16`) |
| Available KV memory | **18.67 GiB** |
| `kv_cache_max_concurrency` | 1.949… |
| Boot line | `padded slot-share block=64 mla_page=2351104 (was block=16); draft_bytes/token=2048` |

DFlash2 cannot exact-fit the 656 B MLA page, so the five SWA layers **padded
slot-share** the MLA tensors: manager `block_size=64` (indexer kernel size; not
the 3584-token mamba-aligned MLA manager) and `page_size_padded` equal to the
MLA page. Drafter layer *i* co-owns MLA tensor *i* at window-bounded BlockPool
IDs, like mamba. Per-block pool bytes unchanged. This is an **allocator**
change — target attention is still the same `fp8_ds_mla` kernel and scales as
the compact-64 fp8 serve. It is not NVFP4 KV.

Inheriting the MLA manager block (1152, later 3584) made each of 5 draft layers
tens of MiB per pool block and pinned logged concurrency near 1×
`max-model-len`. Compact-64 without slot-share still burned unique IDs per
draft layer.

Live occupancy, temp **0**, thinking **off**, unique pads, `max_tokens=8`:

| Load | HTTP | Peak KV | Wall / TTFT | Notes |
|---|---|---:|---:|---|
| ~36k ×1 (compact-64, no slot-share) | 200 | **44.6%** | — | five standalone DFlash ID sets |
| ~36k ×1 (padded slot-share) | 200 | **~16%** | — | one shared ID set |
| ~36k ×3 concurrent | **3× 200** | **21%** (two in flight) | 54 / 96 / 137 s | `GLM53_MIXED_PREFILL_CHUNK=skip` still serializes prefills (`Running: 1`, others wait on capacity, then deferred) |
| ~256k ×3 concurrent | **3× 200** | **29.5%** (two in flight) | 305 / 608 / 916 s | live on the 900k boot; 256,013 prompt tokens each, gen `OK`; third waited (skip) |
| ~300k ×1 streamed | **200** | **26.0%** | **356 s** TTFT (~840 tok/s) | 299,213 prompt tokens, gen `OK` |

Live **3×256k** held (the original failure). Prefills still serialize under skip; two 256k contexts were in KV at once at 29.5%. One 256k sat ~25%. Hybrid occupancy is a large length-independent floor (mamba + DFlash window) plus MLA pages that scale: 36k → 16%, 256k → ~25%, 300k → 26%.
**Shipped default is `600000`** (`.env.example`, D1 operator decision
2026-08-29; the first `./start.sh` run copies it to `.env`). The `start.sh`
fallback is **1M** only when `MAX_MODEL_LEN` is unset or empty. Do **not**
drop the window to 256k to “free” slots — logged tokens ≈ concurrency × that
cap, and the hybrid floor then shrinks the pool. The cap is a **ceiling, not
a reservation** (issue #43): lowering it changes admission only, never the
pool size.

### Queueing vs `MAX_NUM_SEQS`: read the gauges right (issue #43)

`--max-num-seqs 4` is four *in-flight* generations — it reserves nothing.
Admission is KV-capacity-bound per request: the R1 auto pool is **~950–980k**
tokens ≈ **1.6×** the 600k window, and an independent second-fleet
measurement (issue #43) read **~31%** on `vllm:kv_cache_usage_perc` for one
258k prompt — the *effective* pool sits well below nominal (hybrid mamba +
SWA + DFlash2 draft state take the rest). One long request can therefore
block the next admission while seq slots are free: below-4 queueing with
large requests is expected behavior, not a bug.

`num_requests_waiting_by_reason{reason="capacity"}` is **not** a KV
diagnostic in this build — the fork's `loggers.py:1117` sets it from plain
`num_waiting_reqs`, so it labels *every* waiting request “capacity”.
Diagnose with `vllm:kv_cache_usage_perc` plus prefill tok/s counter deltas
instead.

**Profile guidance.** Interactive fleets (many concurrent sessions ≤200k,
e.g. 4 × 90k) are not served better by shrinking the window — such traffic
never touches a 600k cap, and follow-up turns already ride the 93% prefix
cache above. The cap binds only when single requests grow past it: at 200k,
~4 full-length requests fit concurrently but >200k prompts are rejected; at
600k/1M they are admitted and queue on capacity instead. Choose per
deployment — do not flip the fleet default mid-campaign (arm geometry).

**Cold-prefill serialization.** Under `GLM53_MIXED_PREFILL_CHUNK=skip` a
second *cold* prefill waits for the in-flight one (`Deferred` above). Decode
head-of-line blocking behind a long prefill is already fixed by the R1
bundle's long-prefill threshold (first token behind a 240k cold prefill:
**8.39 s vs 461.3 s** pre-R1). Remaining levers, each a gated arm:
`GLM53_MIXED_PREFILL_CHUNK=N>0` (mixes prefill chunks into decode steps —
queued cold prefills start sooner, decode TPOT jitters; A/B pending),
`ASYNC_SCHEDULING=1` solo (never isolated from DSD), `MAX_NUM_SEQS=8`
(per-request mamba/draft floor grows per admission), and bounded default
output (upstream decode-hygiene PR). `MAX_NUM_BATCHED_TOKENS` is already
tuned: do not raise it to 8192 (GB10 indexer oversubscribe) or to “fix” APC.

Keep **`SKIP_MM_PROFILING=1`** — a max-size image+video dummy profile OOMs this UMA.
`LIMIT_MM={"image":4,"video":1}`.

**NVFP4 KV is not available here.** FlashInfer’s SM12x NVFP4 kernels are dense MHA,
not sparse MLA. Do not confuse that with NVFP4 **weights** (`--moe-backend marlin`).

## Prefix caching (this kit, 2026-08-29)

`--enable-prefix-caching` is on. The OpenAI API is **stateless**: the client
resends the full history each turn; vLLM hashes that prefix. Concurrent chats
do **not** mix activations. `--max-num-seqs 4` is four **in-flight** generations,
not four parked sessions. MLA `KpoolTailManager` disables **fine-grained**
hits — only **block-aligned** tokens count (3584-token hybrid align).
`KpoolTail` already opts out of the hybrid min (1-block circular scratch).

`dflash` is `use_eagle()`. GLM never sets `is_eagle_group` (that annotator is
DeepseekV4-only), so stock HybridKVCacheCoordinator flagged **every** group.
MLA dropped its last 3584-token page, and the DFlash2 SlidingWindow group
re-aligned the min by another scheduler page. Overlay
`patch_hybrid_prefix_hit.py` flags only the drafter SWA group as EAGLE and
does **not** let that group shrink the MLA+mamba hit. Mamba stays in the min
(skipping a mamba miss is a correctness hole). Do not raise
`--max-num-batched-tokens` to “fix” APC.

**Live retest** (thinking off, temp 0, real user + assistant + follow-up), 1M serve:

| Turn | Hits | Compute | Prompt tok | TTFT |
|---|---:|---:|---:|---:|
| ~7.7k cold | 0 | 7696 | 7696 | 9.7 s |
| ~7.7k follow-up | **7168** | 549 | 7717 | **1.17 s** |
| ~12k cold | 0 | 11994 | 11994 | 13.4 s |
| ~12k follow-up | **10752** | 1263 | 12015 | **1.94 s** |
| ~16k cold | 0 | 15994 | 15994 | 17.7 s |
| ~16k follow-up | **14336** | 1679 | 16015 | **2.18 s** |
| 4× ~7.5k concurrent follow-ups | **7168 each** (28672 total) | rest | 7515 each | **1.86–2.50 s** |

A ~7.7k follow-up reuses **93%** of the prompt (7168 / 7717), not 46%.
Hits work **below** UserHIJ’s 14,336-token floor (that floor is 896-chunk ×
2048-align LCM on a different geometry; this kit’s 3584 is 4×896). Isolation
held (`STILL_READY_S` / `STILL_C0`…`C3`). Idle chats are not reserved; after
the pool drains, a later turn of an old window prefills again. Concurrent
colds still serialize under `GLM53_MIXED_PREFILL_CHUNK=skip` (`Deferred`).

This 1M boot: **1,670,157** tokens / **1.67×** / 638 GPU blocks (padded
slot-share still applied). The 900k process measured 1,754,237 / 690 blocks
on the same recipe; the delta is leftover UMA, not a slot-share collapse.

## Campaign R1 — production bundle (2026-08-30)

The Reederey87 production kit (a downstream fork of this recipe on the **same
image digest**, same overlays, same bench protocol) measures **70.4 tok/s
structured / 29.5 prose / ~893 tok/s prefill** against our recorded
65.9 / 26.7 / ~489. Those wins are configuration-level. R1 adopts the full
bundle and re-gates every claim with our own benches (`docs/CAMPAIGN-R1.md`);
ported components are credited in `NOTICE`.

**Serving-config bundle** (all default in `start.sh` + `.env`):

| Knob | R1 value | Why |
|---|---|---|
| `MAX_NUM_BATCHED_TOKENS` | `3584` | page-exact prefill chunks (14 × 256-token pages) |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1792` | requests above it take the long-prefill scheduling path; half the step budget, emitted as `--long-prefill-token-threshold` directly (never via `EXTRA_ARGS`) |
| `ASYNC_SCHEDULING` | `0` | async scheduling **off** in the bundle (their A/B: async-off ≥ async-on at ×1). The DSD arm sets `ASYNC_SCHEDULING=1` — `start.sh` refuses `DSD_TABLE` without it |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `0` | sparse KDA retention (fork env): retain only prompt boundaries + shared-prefix junctions; forwarded to **both ranks**, only when non-empty (an empty string crashes boot) |
| `FLASHINFER_WORKSPACE_BASE` | `/root/.cache/vllm/flashinfer` | FlashInfer JIT workspace inside the mounted vLLM cache — kernels survive container recreate; watchdog heals pay no re-JIT |
| `IMAGE` | `…@sha256:9bb1557a…` | digest pin + `SKIP_PULL=1`: a restart can never silently upgrade. `BUILD=1` with a digest ref is refused |
| `KV_CACHE_MEMORY` | *(unset — pin REJECTED)* | three pinned boots froze this kit with the NVRM `NV_ERR_NO_MEMORY` kernel signature (17.7 GiB ×2 C4, 14.64 GiB R1 — `results/ab/r1-phase0/freeze-20260830.md`). Auto pool = 963,265 tokens = **1.61×** the 600k window. A pin requires `ALLOW_KV_PIN=1` + a measured memfloor artifact (D2) |

**R1 measured (2026-08-30, fresh re-bench vs the pre-R1 config, `results/ab/DECISION-R1.md`):**
structured ×1 **71.97** tok/s (accept **1.0** — the xgrammar backports resolved the
stale 0.9588), prose ×1 **27.64**, 100k cold TTFT **199.4 s**, cache burst rounds 2–3
**98.6%** / solo replay **98.7%** at ~200k, HOL first-token behind a 240k cold prefill
**8.39 s vs 461.3 s** on the pre-R1 config (the long-prefill threshold eliminates
head-of-line blocking — the bundle's decisive win). DSD concurrency arm: receipt
ACTIVE, aggregates ×2/×4 +71%/+35%, but ×1 structured −7.0% (async-ON cost) →
**REJECTED, ships dormant** per the unconditional ×1 gate. KV pin: **REJECTED** after
the third freeze (NVRM `NV_ERR_NO_MEMORY` at API bring-up; auto pool 963–980k tokens
= 1.61–1.63× adopted; `ALLOW_KV_PIN` hard guard).

**DSD concurrency arm (D5, on top of R1):** `DSD_TABLE=1:1:7,2:999:5
ASYNC_SCHEDULING=1 ./start.sh restart` — async ON, **pin OFF** (auto pool;
recorded deviation: under a KV pin, async double-counts the SW-family
reservation and can fail the admission check). Receipt: `python3
tests/verify_dsd.py`. DSD survives as default-on only if it beats the R1
bundle at ×2/×4 by ≥3% without regressing ×1 (the async cost may offset —
their A/B measured async-off 69.8 vs async-on 67.4 structured).

**Security note (recorded operator decision):** the API binds `0.0.0.0` —
vLLM's OpenAI server has **no authentication** (`VLLM_API_KEY` unset), so
everything on the LAN can query the fleet and the metrics endpoint. The
upstream kit's loopback-bind hardening was deliberately **not** adopted; if
that ever changes, set the bind in the inner scripts and re-run the serving
probes. Interim mitigations: firewall the head port (`PORT`), or front it
with an authenticating proxy.

**Ops kit** (`local/`, ported — see `NOTICE`):

| Tool | Purpose |
|---|---|
| `local/serving-probe.sh` | 6-probe serving liveness battery (health/models/chat/stream/tools/metrics) |
| `local/acceptance.sh` | 7-probe quality battery (tools / thinking / vision / needle ×3 incl. cache-aware replay) |
| `local/toolcall-probe.py` | tool-call acceptance battery (5 cases × N reps) |
| `local/cache-burst.py` + `local/cache-probe.sh` | multi-session cache gates: 4×60k ×3 rounds hit ≥90%, solo 110k replay ≥93% |
| `tests/bench_prefix_cache.py` | the same protocols wired into an AB-PLAN arm artifact (fingerprint + gates) |
| `local/xid-check.sh` | NVRM Xid monitor (fatal classes 13/31/43/45/48/62/79) — timer-friendly |
| `local/metrics-alert.sh` | spec-decode acceptance alerting from `/metrics` (consecutive-strike model) |
| `local/check-updates.sh` | registry digest vs `.env` pin + 590.x driver warning (Phase-0 gate, automated) |
| `local/prod-start.sh` | memory-gated restart (MemFree ≥ 8 GiB both nodes) + config-shape hash that wipes stale JIT caches on geometry change |
| `local/install-ops-units.sh` | installs the systemd **user** units (timers default OFF; `--enable` is the operator-approved Phase-4 step) |

The watchdog now distinguishes **crash** (container gone → immediate recover),
**wedge** (running but `/health` failing, optional `WATCHDOG_LIVENESS=1`
decode probe) and **deliberate stop** (`./start.sh stop` writes a sentinel
the watchdog honors; any launch clears it), with a 900 s recovery backoff.

## Quick start (2× Spark)

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git
cd GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cp .env.example .env          # edit HEAD_IP / WORKER_IP / WORKER_USER if needed
./download.sh                 # optional: EXL3 + DFlash2 into the head HF cache only
./start.sh                    # pull public GHCR :exl3, download if missing, rsync, launch TP=2
```

First run of `./start.sh` copies `.env.example` → `.env` if missing. Prefix env
wins over `.env` (`SPEC_METHOD=dflash SKIP_DOWNLOAD=1 ./start.sh restart`).

`./start.sh` downloads weights automatically when the HF cache is incomplete
(120 shards of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, falling back to
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` if the mirror is incomplete, plus DFlash2 when
`SPEC_METHOD=dflash`). `./download.sh` is the same Hub fetch **on this machine
only** — no docker, no SSH, no worker rsync. Use it to stage ~164 GiB before
the worker is ready. `REFRESH_WEIGHTS=1 ./download.sh` re-fetches.
Already present: both scripts skip. `./start.sh` still rsyncs the cache to the
worker unless `SKIP_SYNC=1`.

DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`, ~2.3 GiB BF16, CC BY-NC-ND 4.0 research/eval)
is the default. Rollback:

```bash
SPEC_METHOD=mtp ./start.sh restart      # MTP k=2
```

`./start.sh` will:

1. Preflight docker/ssh/disk on both nodes
2. `docker pull` `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` (public; no login) on the head, then the same pull on the worker if GHCR is reachable. If the worker cannot pull, `docker save --platform linux/arm64 | ssh docker load`. `SKIP_PULL=1` keeps a local copy. `SKIP_SHIP=1` never copies.
3. Download the TR3 EXL3 repo into `$HF_HOME` / `~/.cache/huggingface` (~164 GiB, 120 shards) if missing. Same job as `./download.sh`, which stops here (head only).
4. `rsync` that cache to `${WORKER_HOME}/.cache/huggingface`
5. Start rank 1 `--headless` on the worker, rank 0 + API on the head
6. Poll `/health` (weight load + warmup is slow; `READY_TIMEOUT` default 3600s), then a **nonfatal** DFlash2/sampler shape sweep so the first client is not the first JIT on TP=2. `GLM53_BOOT_SHAPE_WARMUP=0` skips it.

The worker does not need GHCR access — start.sh pulls there when it can, otherwise it ships a single-platform tar over SSH.

```bash
./download.sh                              # head HF cache only (no worker); same as ./start.sh download
SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh     # weights already local on both nodes
SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart  # keep local image, no GHCR
BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart  # rebuild overlay from this repo + ship
./start.sh status
./start.sh logs                # head
./start.sh logs worker
./start.sh stop                # or ./stop.sh
```

Do not pull `glm53-flash-sm121:v8` — that is the older NVFP4/Ray kernel.

API: `http://127.0.0.1:8888/v1` (LAN: `http://10.0.0.1:8888/v1`).

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash-EXL3",
    "messages": [{"role": "user", "content": "hello!"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Thinking defaults on (D4: `GLM53_DEFAULT_THINKING` — 0 flips the served
default to off, Entrpi-validated +7% structured acceptance; reasoning then
arrives in `message.reasoning`). Disable it per request with the **top-level**
JSON field `"chat_template_kwargs": {"enable_thinking": false}`. This
closes the empty thinking block in the generation prompt and omits the
reasoning-effort hint.
Do not send a literal nested `extra_body` object over raw HTTP; `extra_body` is
an OpenAI Python SDK option that merges its contents into the top-level request.
The Hub `generation_config.json` stamps `temperature=1.0` / `top_p=0.95` unless
the request overrides. The launcher sets
`--chat-template /opt/glm53/chat_template.jinja` (checkpoint jinja is language-only).

Needs: Docker (no sudo) on both nodes, passwordless SSH head → worker,
`hf` / `huggingface-cli` + `curl` + `rsync` on the head, ~180 GiB free per
node for the first download. The GHCR image is public; login is only needed
if you hit anonymous pull rate limits (`GHCR_TOKEN` + `GHCR_USER`).
Mixed OS accounts: set `WORKER_USER` (this kit uses `zurih` on spark2).

NCCL cannot use the `10.0.0.x` loopback aliases — leave the CX7 pins unless
your cabling differs. `ncclCommInitRank` hangs without them.

## Running on a different 2×Spark kit

Independently reproduced on a second GB10 pair (2026-08-28) — decode within the
same bands (structured 38–62, prose 27.1) after three kit-specific adjustments
that are now documented/enforced:

- **NIC names differ per kit.** Set all four of `HEAD_CX7_IF/IB`,
  `WORKER_CX7_IF/IB` in `.env` (some pairs use the same names on both nodes,
  e.g. `enP2p1s0f1np1`/`roceP2p1s0f1`). Exporting generic
  `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` does **not** override the per-node values.
- **`NCCL_IB_GID_INDEX` (default 3) may be an all-zero entry on one node** —
  the launch then dies ~60 s in with `ibv_modify_qp` errno 61 on the worker
  rank. `preflight` now validates the index on both devices and prints both
  GID tables when it refuses; pick the index carrying the `::ffff:<ip>`
  RoCEv2 entry on both nodes.
- **`GPU_MEM_UTIL=0.87` needs ≥105.9 GiB free *after* vLLM's own ~9 GiB
  init.** Nodes running resident services (dashboards, TTS, desktop) can miss
  it by well under 1 GiB and fail the startup memory check; `GPU_MEM_UTIL=0.86`
  with `MAX_MODEL_LEN=800000` (the previously published pair) fits with margin.
  If :8888 is taken on your head node, `PORT` moves the API cleanly.

## .env

| Knob | Default | What |
|---|---|---|
| `HEAD_IP` | `10.0.0.1` | this node, NCCL/vLLM master |
| `WORKER_IP` | `10.0.0.2` | other Spark |
| `WORKER_USER` | *(unset = `$USER`)* | SSH user on the worker |
| `WORKER_HOME` | `$HOME` if same user, else `/home/$WORKER_USER` | worker HF cache |
| `MODEL` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | Hub repo into the HF cache (mirror) |
| `MODEL_FALLBACK` | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` | Used if the mirror 404s or has fewer than 120 shards |
| `SERVED_MODEL_NAME` | `GLM-5.3-Flash-EXL3` | OpenAI `model` id (`/v1/models`) |
| `IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | public GHCR tag; pulled on every start. `SKIP_PULL=1` skips. `BUILD=1` rebuilds the overlay. R1: pin the exact digest (`…@sha256:…`) + `SKIP_PULL=1` so a restart can never silently upgrade; `BUILD=1` with a digest ref is refused |
| `GHCR_TOKEN` / `GHCR_USER` | *(unset)* | optional login if anonymous GHCR pull is rate-limited |
| `PORT` | `8888` | OpenAI API on the head |
| `TP` / `NNODES` | `2` / `2` | do not change for this recipe |
| `QUANTIZATION` | `exl3` | overlay method; never `marlin` |
| `MTP_TOKENS` | `4` | MTP speculative tokens (`SPEC_METHOD=mtp`; k=4 measured faster than k=2, k=5 regresses — D4) |
| `SPEC_METHOD` | `dflash` | `dflash` / `mtp` / `none`. Rollback: `SPEC_METHOD=mtp ./start.sh restart` |
| `DFLASH_MODEL` | `incoai/GLM-5.3-Flash-DFlash2` | DFlash2 draft Hub repo (~2.3 GiB BF16) |
| `DFLASH_TOKENS` | `7` | DFlash2 speculative tokens (trained block 8) |
| `DSD_TABLE` | *(unset)* | D5: Dynamic Speculative Decoding (vLLM PR #32374, present in this image). `start_bs:end_bs:k,...` draft-length schedule; e.g. `1:1:7,2:999:5` = k7 solo, k5 from 2 concurrent. Empty = static k. Requires `ASYNC_SCHEDULING=1` (enforced) + KV pin OFF (recorded R1 deviation). Verify shapes `seq*(1+K)` are captured by a DSD-derived ladder (`12 18 24` replace `16 32`). Receipt: `python3 tests/verify_dsd.py` |
| `DFLASH_DRAFT_TP` | `1` | keep the 2.3 GiB drafter on rank 0 (no CX7 per draft step). Empty = inherit TP |
| DFlash2 draft KV | `auto` (bf16) | target stays `fp8`/`fp8_ds_mla`; dense draft has no MLA FP8 backend on SM121 |
| DFlash2 attention | *(unset)* | SM121 picks FLASH_ATTN for non-causal SWA. Do not pin `TRITON_ATTN` |
| `ENFORCE_EAGER` | `0` | CUDA graphs; MTP capture `1 2 3 4 6 8 12`, DFlash2 `1 2 4 8 16 24 32` |
| `EXL3_FUSED_MOE` | `1` | `exl3_moe` per layer; `0` = LinearEXL3 loop |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`; not `nvfp4`, not bf16 |
| `GPU_MEM_UTIL` | `0.85` | GB10 UMA budget — hard ceiling (crash review: >0.85 froze both nodes; start.sh refuses). The live pool at 600k window gives ~1.6× concurrency margin |
| `MAX_MODEL_LEN` | `600000` | Operator decision (2026-08-29, DSD campaign): 600k window — same UMA pool gives ~1.6× concurrency margin (was 1.08× at 900k). Do not drop to 256k to “free” KV — logged tokens ≈ concurrency × this cap; hybrid block-id overhead then shrinks the pool |
| `MAX_NUM_SEQS` | `4` | decode batch; MTP adds k+1 tokens/seq. **Ceiling, not a reservation** — admission is KV-capacity-bound; a full 600k request ≈ 62% of the ~950–980k auto pool, so below-4 queueing with large requests is expected (issue #43) |
| `MAX_NUM_BATCHED_TOKENS` | `3584` | R1 bundle (page-exact, 14 × 256-token pages; was D1's 2048, ladder 512→1024→2048). Spec-decode step budget; 8192 oversubscribes GB10 indexer topk. Raising further shrinks the pool — re-run the D1 gate + memfloor |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1792` | R1 bundle: requests above this take the long-prefill scheduling path; emitted as `--long-prefill-token-threshold` directly (never via `EXTRA_ARGS`). Empty = scheduler default (cap off) |
| `ASYNC_SCHEDULING` | `0` | R1 bundle: `0` = `--no-async-scheduling` (bundle baseline), `1` = `--async-scheduling` (DSD arm), `auto` = pass neither (vLLM decides). `start.sh` refuses `DSD_TABLE` without `1` |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `0` | R1 bundle: sparse KDA retention (fork env) — `0` retains only prompt boundaries + shared-prefix junctions; unset/empty = dense. Forwarded to both ranks only when non-empty (empty string crashes boot) |
| `FLASHINFER_WORKSPACE_BASE` | `/root/.cache/vllm/flashinfer` | R1 bundle: FlashInfer JIT workspace inside the mounted vLLM cache — survives container recreate |
| `KV_CACHE_MEMORY` | *(unset)* | **pin REJECTED** (3 freezes, NVRM OOM signature — see the R1 section). Hard guard: dies unless `ALLOW_KV_PIN=1`. Emits `--kv-cache-memory-bytes "N"` (quoted) when allowed |
| `GLM53_MIXED_PREFILL_CHUNK` | `skip` | do not mix a peer prefill into a decode step (issue #6). `N>0` = cap tokens; `0` = off. Solo prefill stays 1024 |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1` | ignore client `stop` strings until `</think>` (thinking-on default) |
| `GLM53_BOOT_SHAPE_WARMUP` | `1` | after `/health`, burn DFlash2 BLOCK / sampler / kpool shapes (nonfatal) |
| `TRITON_HOST_CACHE` / `TILELANG_HOST_CACHE` | `$CACHE_ROOT/triton` / `tilelang` | persist JIT caches across container recreate |
| `GLM53_DEFAULT_THINKING` | `1` | served thinking default; `0` = off (D4: reasoning → `message.reasoning`; clients can override per request) |
| `WEIGHTS_MODE` | `local` | `nfs` = head reads the worker's HF cache over NFS (`NFS_PORT` 12049, opt-in; avoids the measured head load wedge — D4) |
| `LANGUAGE_MODEL_ONLY` | `0` | load vision tower (image + video) |
| `SKIP_MM_PROFILING` | `1` | skip max-size MM dummy at init (OOM otherwise) |
| `LIMIT_MM` | `{"image":4,"video":1}` | `--limit-mm-per-prompt` |
| `HEAD_CX7_IF` / `WORKER_CX7_IF` | `enp1s0f1np1` / `enp1s0f0np0` | NCCL sockets |
| `HEAD_CX7_IB` / `WORKER_CX7_IB` | `rocep1s0f1` / `rocep1s0f0` | NCCL HCAs |
| `USE_HOST_NCCL` | `0` | image nvidia-nccl; host preload duplicates DeepEP |

## Image / overlay

```bash
docker build -t glm53-flash-sm121:local .
# or: BUILD=1 ./start.sh
```

`./start.sh` **pulls** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
(public) on every start unless `SKIP_PULL=1`. `BUILD=1` rebuilds the overlay from
this Dockerfile instead. After CUDA compile, Python overlay edits
(`overlay/exl3.py`, tests) are a cheap layer so they do not rebuild
`exllamav3_ext`.

| Path | Role |
|---|---|
| `Dockerfile` | NoPE sparse-MLA patches + EXL3 install (`sm_121a`) + self-check |
| `overlay/exl3.py` | `Exl3Config` / packed load / TP shard / fused `exl3_moe` apply |
| `overlay/patch_exl3_ext_aarch64.py` | stub AVX CPU allreduce so the ext builds on GB10 |
| `overlay/patch_model_overrides.py` | `"exl3"` in ModelConfig overrides |
| `tests/test_exl3_overlay.py` | registry, TP shard, `sm_121a` cubin, fused vs loop GEMM, `EXL3_FUSED_MOE=0` |
| `tests/bench_decode.py` | streaming decode + coherence; `--structured` is the count-1→200 median |
| `start.sh` / `stop.sh` / `download.sh` | 2-node launch; Hub fetch on the head only |
| `files/chat_template.jinja` | GLM-5.3 MM template (`<|image|>` / `<|video|>`); checkpoint jinja is language-only |
| `overlay/qwen3_dflash2.py` | DFlash2 draft (grouped conv + candidate selector) |
| `overlay/dflash2_speculator.py` | DFlash2 selector walk (V2 speculator) |
| `overlay/patch_dflash2.py` | registry + `decoder_layer_cls` + speculator dispatch + draft KV `auto` on MLA/FP8 |
| `overlay/patch_glm_eagle3.py` | Glm5Next EAGLE3 aux-hidden layers (mHC `hc_post` + contract) |
| `overlay/patch_glm5_drafter_group.py` | GLM KV fast path + DFlash2 padded slot-share (`block=64`, `page_size_padded=mla_page`); runtime-mounted by `start.sh` (`DRAFTER_PATCH_HOST`) |
| `overlay/patch_glm_video_placeholders.py` | align video timestamp blocks to encoder `grid_t` |
| `overlay/patch_suppress_stops_in_reasoning.py` | fail-closed detokenizer guard: client `stop` dormant until `</think>` |
| `overlay/patch_scheduler_decode_floor.py` | skip (or cap) peer prefill while another seq is decoding |
| `overlay/patch_xgrammar_termination.py` | source-exact vLLM #52805/#53046 backports; stop at termination and validate post-reasoning speculative drafts before FSM advance |
| `tests/test_xgrammar_termination.py` | exact two-file patch, idempotence, cross-file fail-closed drift, termination/rollback/reset and post-reasoning draft behavior, launcher wiring |
| `scripts/boot-shape-warmup.sh` | post-`/health` DFlash2 k=7 BLOCK ladder + sampler/kpool arms |

Image-build runs `EXL3_SELFCHECK_GPU=0`. `./start.sh` runs the GPU self-check
(`docker run --gpus all`) before shipping unless `SKIP_OVERLAY_VERIFY=1`.

## Cross-stack validation (Entrpi) & memory-floor methodology

[`docs/COMPARISON-ENTRPI.md`](docs/COMPARISON-ENTRPI.md) is the honest
side-by-side against [Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)
— the same checkpoint + drafter on the same hardware through a custom vLLM
fork. Verdict: this recipe leads on robustness tooling, KV pool size and 1M
context; Entrpi leads on long-prompt prefill chunking (D1), memory-floor
methodology (D2), task evals + spec-equivalence (D3) and a few knobs (D4) —
adopted here as gated arms, see `results/ab/DECISION.md`.

**Memory floors (D2):** GB10 fails as a swap wedge, not a graceful OOM. Any
`KV_CACHE_MEMORY` pin or `GPU_MEM_UTIL` raise must be backed by a measured
floor: `tools/memfloor.sh <label> -- <workload>` samples both nodes at 1 Hz
(`tools/memlog.sh`) and writes `results/ab/memfloor-<label>-<stamp>/`.
Rules: explicit budgets bypass the profiler reserve (raises are non-linear in
floor cost — this kit's 17.7 GiB pin crashed twice); floors age ~1.5-2 GiB/day;
keep the binding floor ≥ 5 GiB; `CACHE_FLUSHER=1` is validated-required on
this kit's load window (MemFree 3.2 GiB idle floor).

## Do not

- Destroy HF weights, requantize, or `docker rm` HF caches. `REFRESH_WEIGHTS=1 ./download.sh` only if you intend to re-fetch
- `--moe-backend marlin`, NVFP4 weights, or `glm53-flash-sm121:v8` as this serve
- qemu / amd64 / `cstechdev/vllm:glm53-flash-nope-sm120-*` / verdictai SM120 B12X
- `--kv-cache-dtype nvfp4` or bf16 (no sparse-MLA kernel)
- `"attention_backend": "TRITON_ATTN"` in speculative-config (causal-in-block on this image)
- Change TP, CX7 pins, or `USE_HOST_NCCL` unless you are re-plumbing NCCL
- Force-push

## License

This repository (serve scripts, overlay, docs) is **MIT**. The EXL3/TR3
checkpoint stays [ShapleyMCG License 1.0](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/main/LICENSE)
(unmodified upstream LICENSE; also on
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)).
DFlash2 stays [CC BY-NC-ND 4.0](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

## Credits

- **EXL3/TR3 weights:** [brandonmusic](https://huggingface.co/brandonmusic) —
  [GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  (uniform-K4 routed-experts, ShapleyMCG License 1.0). Public mirror for this
  recipe: [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
- **EXL3 format / kernels:** [turboderp](https://github.com/turboderp-org/exllamav3) (ExLlamaV3)
- **Base model:** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- **DFlash2 drafter:** [IncoAI](https://huggingface.co/incoai) —
  [GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0, research/eval)
- **KLD panel:** [malaiwah](https://huggingface.co/malaiwah) —
  [discussion #1](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1#6a9144846b0bdba943bfe86f)
