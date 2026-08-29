# Cross-stack validation: this recipe vs Entrpi/glm-5.3-flash-exl3-2x-spark

Date: 2026-08-29 · Campaign: D1–D4 (see `results/ab/DECISION.md` for the adopted set)

[Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)
(local clone: `~/developer/recipes/glm-5.3-flash-exl3-2x-spark`) serves the same
model, the same brandonmusic EXL3/TR3 4bpw checkpoint, and the same incoai
DFlash2 k=7 drafter on the same 2× GB10 hardware — through a **completely
different engine**: a custom vLLM fork branch
(Entrpi/vllm-glm-5.3-flash-spark, local-inference-lab lineage, DeepGEMM kpool,
b12x trellis, breakable CUDA graphs) baked into its own image. This recipe
serves the stock vLLM day-0 image plus runtime overlay patches
(FlashInfer sparse-MLA SM120, packed `fp8_ds_mla`, turboderp `exl3_moe`).

Both projects were developed independently. Each README already carries a
cross-stack section; this document is the honest side-by-side from this
recipe's point of view, and the source of the D1–D4 adoption decisions.

## Side-by-side (at validation time)

| | This recipe (MiaAI fork + hardening) | Entrpi recipe |
|---|---|---|
| Engine | stock vLLM image + 12 runtime overlay patches | custom vLLM fork branch, image rebuilt from source |
| Attention / KV lane | `FLASHINFER_MLA_SPARSE_SM120`, NoPE latent zero-padded into GLM_NSA 576 | fork NoPE-MLA lane, DeepGEMM kpool, block 2304/4608 |
| KV format | packed `fp8_ds_mla` (mandatory; bf16 has no sparse kernel on SM12x) | bf16 default until 2026-08-29; `fp8_e4m3` gated option |
| Context / pool (live) | 1M, 1,221,311 tok pool @0.85/CG_ESTIMATE=0 (1.22×) | 524k default, 1,435,070 tok @12.4 GB pin (2.74×) |
| Prefill chunk | 512 (crash-review value); engine honors env | 8192 (~1,400 tok/s on 133k prefill) |
| Fused EXL3 MoE | turboderp `exllamav3_ext.exl3_moe` (one launch/layer) | tpurtell b12x CuTeDSL trellis (`standard_fused_moe`) |
| Draft placement | TP=1 on rank 0 (no CX7 per draft step) | TP=2, sharded with the target |
| Draft KV pooling | padded slot-share (block=64, `page_size_padded`=MLA page) | fixed ring + positional remap kernel |
| Spec fallback | MTP k=2 (~24.6 tok/s measured) | MTP k=4 (28.6 tok/s) |
| Quality evidence | weights-level KLD panel (cited) | task evals (math_500 87%, gpqa_diamond 72%, estonia 9/10) + equivalence harness |
| Robustness tooling | fleet watchdog, cache flusher, memory ritual, GID preflight, suppress-stops, xgrammar backports, boot-lottery fallback | hotfix bind-mount hooks, NFS weights mode, memlog floor sampler |
| Memory methodology | C4 pin rejected (17.7 GiB → 2 node crashes); CG_ESTIMATE=0; 0.85 hard ceiling | pinned 12.4 GB budget + 1 Hz floor sampler; "floors age 1.5–2 GiB/day" |

## Same protocol, both stacks (best-on-each-stack, not single-variable)

Entrpi vendored this repo's `bench_decode.py` unmodified as
`scripts/bench_decode_miaai.py`. Their side is from their published README /
COMPARISON.md; our side is our own lab runs.

| Protocol phase | This recipe | Entrpi |
|---|---:|---:|
| Structured (count 1→200), tok/s | 61.7 (lab median) | 72.4 (post template-fix era) |
| — accept/step (of 7) | 6.43 | 6.86 |
| Prose (hash-map), tok/s | 26.9 | 27.4 |
| — accept/step | 2.33 | 2.17 |
| TTFT (short prompt) | ~0.72 s | 0.43–0.47 s |
| Per engine step | ~115 ms | ~115 ms |

Reading: per-step cost is at par — two different fused-MoE engines landing on
the same number. The structured gap is mostly acceptance and the TTFT gap is
mostly prefill chunking (see D1).

## Convergent engineering (both solved the same three problems)

No upstream solution exists for DFlash2-on-GLM-5.3 in either lineage; both
projects independently arrived at the same three fixes (strong evidence both
implement the drafter correctly):

1. **Drafter KV on the glm5 hybrid** — plain sliding-window drafter specs eject
   the model from the KV fast path. Entrpi: dedicated draft groups + fixed
   ring of draft pages. This recipe: exact-fit block rescale that slot-shares
   drafter layers into the MLA tensors. Both end at ~zero pool cost.
2. **mHC aux capture** — materialize deferred `hc_post` then `hc_contract` at
   tap layers (6, 15, 25, 34, 43), identical contract on both stacks.
3. **Non-causal draft attention** — a causal mask inside the draft block
   silently collapses later-position acceptance; both stacks independently
   discovered and fixed it.

## Adopted from Entrpi (campaign D1–D4)

| Bundle | What | Status |
|---|---|---|
| D1 | Prefill step budget: raise the spec-decode per-step budget above 512/1024 (Entrpi serves 8192 on the same hardware class; ~1,400 vs ~840 tok/s prefill) | verified env-only on the current image (no scheduler clamp found in source); arms 2048/4096 with KV-pool + memfloor + dmesg gates |
| D2 | Memory-floor methodology: 1 Hz MemAvailable sampler (`tools/memlog.sh`), non-linear cost of explicit KV budgets, floors age 1.5–2 GiB/day | ported; baseline floor artifact + per-arm floor recording |
| D3 | Task-level quality gates (math_500 n=100, gpqa_diamond n=50, ≥100k-token retrieval) + spec-equivalence harness (`tests/dflash_equiv.py`) | ported/implemented |
| D4 | NFS weights mode (worker-hosted weights, NFS-paced load), MTP fallback k=4, thinking-off serving default | implemented (NFS opt-in per gate; kit's local-mode load is healthy) |

## Explicitly NOT ported (with reasons)

- **Ring draft KV** — this recipe's padded slot-share already achieves the
  same ~zero pool cost through a different mechanism; acceptance is
  equivalent.
- **MXFP8 DFlash2 drafter** — Entrpi measured a boot crash at draft load on
  their fork (draft-side quant hydration is exl3-only there) and parked it.
  Note: their README headline mentions "MXFP8 DFlash2 drafter" while their
  FINDINGS §8 documents it as boot-crash/parked and the shipping config uses
  the bf16 incoai repo — a README discrepancy; no MXFP8 work is planned here.
- **Draft TP=2** — this recipe's draft TP=1 (2.3 GiB drafter on rank 0) is
  the better choice for a small drafter; Entrpi's own COMPARISON.md calls
  draft-TP=1 "inert on this fork".
- **KV block 2304→4608 auto-bump** — a fork-lane geometry trick (fp32 KDA
  state page parity); this recipe's FlashInfer lane uses block=64 /
  `page_size_padded`=MLA page. Not portable.
- **1M declaration** — Entrpi hits a kernel wall at 524k+ (`persistent_topk`
  grid oversubscription, TopK=512); this recipe serves 1M live. No change.
- **KV pool size** — this recipe's pool (1,221,311 @ the crash-review config)
  is already the largest published at fp8 on this hardware class; Entrpi's
  +46% pool claim vs the older 982,612-token MiaAI config predates this
  recipe's CG_ESTIMATE=0 + padded slot-share work.
- **Strict token-equality equivalence** — Entrpi's own FINDINGS §4 shows it
  is misleading (tie-flips ≤0.75 nats, run-to-run nondeterminism); their
  shipped `tools/dflash_equiv.py` is a greedy-equality harness, so we port it
  with that caveat documented and use it as a smoke gate, not a proof.
- **MTP-5** — measured regression on both stacks (pos-5 prose acceptance
  ~0.23). Confirmed, do not use.

## Discrepancies / notes

- **C6's "1024 clamp" vs the current image.** The C6 arm (2026-08-29 01:29)
  saw the boot config pinned at 1024 with env=2048 and concluded a scheduler
  clamp exists. The current image (pulled 14:22, mutable `:exl3` tag) has no
  such clamp in `vllm/config/vllm.py` or the scheduler: `max_num_scheduled_tokens`
  defaults to `max_num_batched_tokens`, and the vllm.py:1849 warning is a
  `<8192` informational threshold, not a clamp (live boot with env=512 shows
  512). C6 likely measured a stale image build. D1 re-verifies empirically.
- **Memory fragility is the binding constraint on this kit** (crash review
  2026-08-29: 0.87 froze both nodes twice; a 17.7 GiB pin crashed twice; the
  load window is a dice roll with MemFree at 3.2 GiB idle). Entrpi's floor
  methodology maps 1:1 onto that history; the cache-flusher sidecar
  (CACHE_FLUSHER=1) is the unvalidated fix and is validated on the campaign's
  first boot.
