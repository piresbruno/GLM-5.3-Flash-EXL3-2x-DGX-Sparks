# GLM-5.3-Flash EXL3 — Benchmark → Implement → A/B → Decide

**Date:** 2026-08-28
**Recipe:** `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (this repo)
**Stack:** 2× DGX Spark (GB10 / SM121), TP=2 over CX7, vLLM `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
**Served model:** `GLM-5.3-Flash-EXL3` on `:8081` (live `.env` `PORT=8081`; earlier drafts/docs said 8888 — 8081 is the live truth)
**Weights:** `brandonmusic/GLM-5.3-Flash-EXL3-4bpw` (EXL3 4bpw, snapshot `a38a2eeb`, in the local HF cache) + `incoai/GLM-5.3-Flash-DFlash2` (k=7)
**Base for every arm:** `1df71c1` — padded slot-share DFlash2 KV, hybrid-APC fix (issue #7), and the 1M default (`MAX_MODEL_LEN=1000000`, live pool ~1,754,237 tokens ≈ 1.75× a full 1M request at util 0.87; occupancy receipt: 3×256k concurrent sessions, peak KV 29.5%). Cut the baseline on this geometry — do not baseline before it.

Goal: freeze a repeatable **baseline**, apply the four robustness improvements
(ported from the NVFP4 + DFlash2 bring-up and the DeepSeek DSpark recipe), A/B each
against baseline, then ship only the changes that measurably help with **zero
regression**. No final decision until the Phase 3/4 gates pass.

---

## 0. Cross-cutting rules (every phase)

1. **One change per commit** — indexer hardening / memory ritual / watchdog / KV pin /
   GID auto-resolve are independent commits. Revert = revert the commit.
2. **Rebuild + recreate between arms** — overlay/Dockerfile changes use
   `BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart`. Always `./start.sh stop`
   (both ranks) before relaunching either — never let a new rank rendezvous with a
   dying one (NVFP4 lesson).
3. **Identical workload per arm** — same `tests/bench_decode.py` flags, same prompt
   phases, same run counts; only the arm variable differs. Record the image digest
   (`docker image inspect -f '{{.Id}}'`) and `git rev-parse HEAD` per arm.
4. **Warm before measuring** — first request after restart is cold autotune; fire ≥2
   warmup gens before timing.
5. **Results dir** `results/ab/<arm>-<YYYYMMDD-HHMM>/` with `arm.json` + `summary.md`.
6. **Baseline is the only comparison target.** Never compare arms to memory.
7. **Full env fingerprint per arm.** `arm.json` records: `git rev-parse HEAD`, docker image id on the **head AND the worker** (`worker_ssh "docker image inspect -f '{{.Id}}' $IMAGE"` — since af8c111 the worker can silently run a stale image; mismatched rank images void the arm), plus the volatile knobs: `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `MAX_NUM_BATCHED_TOKENS`, `DFLASH_TOKENS`, `GLM53_MIXED_PREFILL_CHUNK`, `KV_CACHE_DTYPE`, `SPEC_METHOD`, and the benchmark temperature.
8. **Cold boot, two definitions.** *Fleet cold boot* = `./start.sh stop` → relaunch (drops page cache; exercised by C2/C3). *Machine reboot* = full node reboot (resets NVRM/GID driver state; required to exercise C5 GID drift). Gates name which one they mean.
9. **Temperature discipline.** Every arm benchmarks at temp 0 AND at the production temperature (tool-eval runs temp 1.0). Temp 0 is +13–21% free throughput — exact top-1 rejection sampling vs a strictly harder probabilistic ratio test at T>0, with greedy drafting pinning draft probability to 1 (NVFP4 OPEN-PROBLEMS #7). Never compare arms across different temperatures.

---

## Phase 0 — Fabric reference (already collected, do not change)

Network config currently in place for DeepSeek TP=2 on this kit (from
`~/developer/recipes/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`, README + docs/SETUP.md):

| Key | Head (rank 0) | Worker (rank 1) |
|---|---|---|
| IP | `10.0.0.1` | `10.0.0.2` |
| `MASTER_ADDR` / `VLLM_HOST_IP` | `10.0.0.1` | — |
| `WORKER_HOST` / `WORKER_VLLM_HOST_IP` | — | `10.0.0.2` |
| `MASTER_PORT` | `25000` | — |
| `NCCL_SOCKET_IFNAME` | `enp1s0f1np1` | `enp1s0f0np0` |
| `TP_` / `GLOO_SOCKET_IFNAME` | `enp1s0f1np1` | `enp1s0f0np0` |
| `NCCL_IB_HCA` | `rocep1s0f1` | `rocep1s0f0` |
| `NCCL_IB_GID` | `NCCL_IB_GID_AUTO=1` (sysfs resolve; pin `=3` optional) | same |
| `NCCL_CROSS_NIC` | `1` | — |
| `NCCL_NET` / `NCCL_IB_DISABLE` | `IB` / `0` | — |
| `NCCL_CUMEM_ENABLE` / `NCCL_NVLS_ENABLE` | `0` / `0` | — |
| `NCCL_IGNORE_CPU_AFFINITY` / `NCCL_DEBUG` | `1` / `WARN` | — |

Deltas vs this EXL3 recipe: DeepSeek uses `MASTER_PORT=25000`, `NCCL_CROSS_NIC=1`,
and **auto-resolves the RoCE GID per node** instead of hardcoding `NCCL_IB_GID_INDEX=3`.
EXL3 currently: `MASTER_PORT=29521`, `NCCL_CROSS_NIC=0`, pinned `GID=3`.
**Do not touch these in Phase 1 (baseline must be stock).** Evaluate the GID
auto-resolve as optional change C5 in Phase 3.

---

## Phase 1 — Baseline freeze (test tooling only, no serving changes)

0. **C0 tooling commit (before the tag):** port `probes/bench_c1c6.py` from
   `~/developer/recipes/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` as
   `tests/bench_concurrency.py` (unique salt per request to defeat prefix caching,
   ≥2 waves/level, end-to-end wall clock, acceptance per level). This is the only
   commit allowed in Phase 1 — without it the ×1/×2/×4 sweep would be ad-hoc per arm,
   violating rule 3.
1. Clean tree: `git status --porcelain` must be empty; record `git rev-parse HEAD`.
2. Launch stock:
   ```bash
   ./start.sh stop && ./start.sh restart
   # record image digest + the vLLM startup log's "Available KV cache memory" /
   # "GPU KV cache size" lines (the suggested --kv-cache-memory number)
   # also record: the "padded slot-share block=64 mla_page=..." boot line and
   # the "Maximum concurrency" line — proof the 1M window allocated on this kit
   ```
   - **APC regression guard (issue #7):** fire a follow-up request that shares a prefix
     with the first (same system prompt, longer user turn) and confirm nonzero
     `vllm:gpu_prefix_cache_hits_total` / a much shorter TTFT than a cold prompt. The
     hybrid-APC patch fail-closes on anchor drift; a zero-hit follow-up means the
     coordinator anchor moved and the fleet must not be benchmarked until understood.
3. Wait for `/health` (already polled by `start.sh`).
4. Run the repo's own protocol (3 repeats, median):
   ```bash
   python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence --out results/ab/baseline-structured.json
   python3 tests/bench_decode.py --phase prose       --runs 5 --max-tokens 400 --skip-coherence --out results/ab/baseline-prose.json
   ```
5. Concurrency sweep (×1/×2/×4) — `tests/bench_concurrency.py` (the C0 port; keep its
   unique per-request salt — it defeats prefix caching so APC gains cannot mask A/B
   deltas); record aggregate + per-stream tok/s and accepted÷drafted per level.
6. Extra baseline-only metrics: boot-to-ready wall time, first-request TTFT after restart,
   KV pool tokens, `/metrics` `vllm:spec_decode_num_drafted_tokens_total` +
   `_accepted_tokens_total` (acceptance), per-position acceptance, `vllm:num_preemptions_total`.
   - **Spec-decode sanity (vllm#53030 trap):** per-position acceptance must be a realistic
     decay (e.g. 0.9/0.5/0.3…). Flat **1.00** = piecewise-graph `BatchDescriptor` collision —
     speculation is silently naive while metrics look perfect. Fail the baseline if seen.
     (EXL3 runs CUDA graphs ON; NVFP4 ran eager, so this trap is untested on this stack.)
   - **Bimodality check:** decode windows of ~37 tok/s interleaved with ~0 = host swap
     pressure, not a uniform slowdown (NVFP4 OPEN-PROBLEMS #6). Check before blaming prefill.
   - **Fenced extra (one DEBUG boot):** `VLLM_LOGGING_LEVEL=DEBUG` capture of
     `gpu_worker.py:552-560` `peak_activation_memory` / `non_kv_cache_memory` on BOTH ranks
     (rank-1 KV asymmetry, NVFP4 OPEN-PROBLEMS #1, worth ~+54% pool). Record only — see
     Appendix C.
7. Tag `git tag baseline-2026-08-28`; store `results/ab/baseline-<stamp>/arm.json`.

**Exit gate:** baseline JSON + summary.md committed. Every later arm diffs against this.

---

## Phase 2 — Implement (one commit each)

### C1 — Indexer NaN-hardening (port of NVFP4 `docker/patch_v7.py`) · HIGH
- Add to the Dockerfile's patch `RUN python3 <<'PY'` block:
  - `sparse_attn_indexer_kpool.py`: `pool_topk = torch.empty(...)` → `torch.full(..., -1, dtype=torch.int32, ...)` in **both** prefill and decode branches.
  - `ops/kpool_compress.py`: `tl.where(pid >= 0, hist_val, -1)` → `tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)`.
- Add self-check asserts (mirroring the existing verify block).
- **First diff the target snippets against the actual image source** — this is a different
  vLLM build than the NVFP4 recipe the line numbers come from; keep self-checks grep-based,
  not line-based.
- Build + verify: `BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart` (runs GPU self-check).

### C2 — GB10 memory ritual + cache-flusher sidecar · MED-HIGH
- `start.sh` `launch_cluster()`: before `docker run`, on **both** nodes run
  `sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory`
  (sudo -n, WARN-only on failure, `SKIP_MEM_RITUAL=1` opt-out).
- New `cache_flusher.sh` (port of NVFP4): flush when `Cached > 40 GiB` for ≤25 min.
  Optionally launched as a detached sidecar during weight load when `CACHE_FLUSHER=1`.
- **UVM livelock guard (NVFP4 OPEN-PROBLEMS #4):** on both nodes set `vm.swappiness=0`
  (persist in `/etc/sysctl.d/` — the workaround does not survive a reboot) and cycle swap
  (`swapoff -a && swapon -a`) before launch. Never disable swap entirely: without swap the
  worker is killed outright during MoE repack, which has no valve for that spike.

### C3 — `fleet_watchdog.sh` auto-recovery · MED
- New script, adapted to 2 nodes / `:8081` / containers `glm53-exl3-head` + `glm53-exl3-worker`:
  - Runs on the head node, **outside** the container; sources `.env` (needs `WORKER_SSH`,
    `MASTER_PORT`, container names, `PORT` — do not hardcode).
  - Probe **`/health`** (not `/v1/models` — returns 200 even with a dead engine).
  - `FAIL_THRESHOLD` consecutive misses → **snapshot `docker logs` of both ranks first**
    (NVRM failures surface minutes after the real event; teardown destroys the evidence)
    → memory ritual → relaunch worker (rank 1) then head (rank 0) via `./start.sh` → wait ready.
  - `flock` single-instance lock; log to `results/` or `logs/`.

### C4 — `KV_CACHE_MEMORY` knob + suggested-value capture · MED
- `.env.example`: add `KV_CACHE_MEMORY` (empty = auto/gmu path as today).
- `start.sh` inner scripts: pass `--kv-cache-memory ${KV_CACHE_MEMORY}` only when set.
- `wait_for_health()`/`on_ready()`: grep the startup log for vLLM's suggested
  `--kv-cache-memory` and log it prominently (plus WARN if `GPU_MEM_UTIL` path yields a
  pool far above it). GB10 context: the suggested number is **not advisory** — the UMA
  driver fails fast on MemFree, not MemAvailable (NVFP4 KV-ladder). At the live
  `MAX_MODEL_LEN=1000000` the ~1.75M-token pool is ~1.75× a full request (receipt:
  3×256k concurrent sessions at peak KV 29.5%), so this knob stays

### C5 — NCCL GID auto-resolve (optional, from DeepSeek) · LOW-MED
- Port `NCCL_IB_GID_AUTO=1` sysfs GID resolution (or a minimal per-node `show_gids` probe)
  into `start.sh`; default OFF (`NCCL_IB_GID_AUTO=0` keeps pinned `GID=3`).
- Only evaluated because the DeepSeek docs flag pinned GID drift after reboot as a
  `ncclCommInitRank` wedge risk.

### C6 — Prefill chunk cap raise `MAX_NUM_BATCHED_TOKENS` 1024 → 2048 · MED-HIGH (pp/TTFT)
- Only 8192 is known-bad (GB10 indexer topk smem OOM at long context); 2048/4096 untested.
- Every 1,024-token chunk pays a full engine step (2×TP collectives per layer, scheduler
  step, the indexer's large fixed per-step cost) — this is the direct pp/TTFT lever:
  measured TTFT ≈ prompt_tokens ÷ pp tok/s, i.e. prefill-bound end to end.
- `.env`/`.env.example` comment + README row change; no code.

### C7 — DFlash2 draft length `DFLASH_TOKENS` 7 → 4 (or 5) · MED (tg/responsiveness)
- At the prose/agent acceptance regime (~0.33–0.41 measured on this kit), k=7 wastes later
  draft positions (README: later-position accept collapses inside the draft block). Fewer
  drafted tokens per verify step can raise effective tok/s and cut per-step latency.
- Hard constraint: `num_speculative_tokens ≤ block_size − 1 = 7` (k=8 drafts an untrained
  position). Gate must include the structured/high-acceptance regime, where k=7 wins.

### C8 — `LANGUAGE_MODEL_ONLY=1` (text-only serving) · LOW-MED
- Vision requests get no spec-decode speedup anyway (drafter is text-only), and the vision
  tower is pure allocation pressure on the UMA. If the benchmark/serving workload is
  text-only, `=1` frees several GiB → KV headroom + boot stability.
- Only evaluated if the workload really is text-only; otherwise skip.

**Phase gate:** each change committed separately; `BUILD=1` boot clean + self-checks pass
for C1; `bash -n` + dry-run for C2–C8 (C6–C8 are env-only: verify via `docker exec ... env`).

---

## Phase 3 — A/B per change

For each C1…C8 (skip C5 unless time permits): arms `base` vs `<cX>`, A/B/B/A interleaved
order preferred (mandatory for C2/C6/C7 — their noise sources are exactly what's being
measured; carry raw per-run values in `arm.json`), else single serve per arm, warmed per
rule 4. Metrics per arm:

| Metric | Source | Pass/fail vs baseline |
|---|---|---|
| Structured decode tok/s (median) | bench_decode `--structured` | ±3% (no regression) |
| Prose decode tok/s | bench_decode `--prose` | ±3% |
| Aggregate ×1/×2/×4 | concurrency sweep | ±3% |
| Acceptance (drafted/accepted) | `/metrics` or bench output | ±2 pt |
| TTFT / boot-to-ready | timing | record only |
| Prefix-cache hit rate on follow-ups | `/metrics` `gpu_prefix_cache_*` | ≥ baseline (issue #7 guard) |
| Preemptions | `/metrics` | **0** |
| NaN / garbage logits | smoke + coherence check | none |

Decision rules:
- **C1 (indexer)**: adopt if no regression and self-check passes (defense-in-depth; it
  removes a latent NaN source, so the bar is "no harm").
- **C2 (ritual/flusher)**: adopt if boot-to-ready and KV-pool tokens are unchanged-or-better
  and no rank dies across ≥3 cold boots.
- **C3 (watchdog)**: adopt if kill-the-head drill recovers within `READY_TIMEOUT` (intentional
  failure injection: `docker rm -f glm53-exl3-head` → watchdog must bring the fleet back).
- **C4 (KV pin)**: adopt only if pinning vLLM's suggested value stays within the ±3%
  regression gate — same bar as every arm (the earlier ≥90% allowance contradicted the
  zero-regression rule) — and eliminates any observed allocation-lottery variance;
  otherwise keep the knob documented but default-off.
- **C5 (GID auto)**: adopt only if `ncclCommInitRank` is demonstrably as-stable-or-better
  across ≥3 **machine reboots** (worker reboot exercises GID drift); else keep pinned `=3`.
- **C6 (chunk cap)**: adopt if pp and TTFT improve ≥10% with structured/prose decode within
  ±3% and no smem OOM (watch dmesg for indexer `topk` shared-memory faults) across the
  sweep and a ≥28K-token long-context run.
- **C7 (draft length)**: pick the k that wins at the production temperature; structured
  (high-accept) decode must stay within ±3% at temp 0. If k=4 wins at temp 1.0 but loses at
  temp 0, expose as a knob rather than re-pinning.
- **C8 (language-only)**: adopt if KV pool tokens increase, boot-to-ready is unchanged-or-
  better, and no decode metric regresses; skip entirely if vision is needed.

---

## Phase 4 — Final combined state + regression

1. Assemble the winning set into one commit (`BUILD=1` if C1 included).
2. Full regression: Phase 1 protocol ×2 separate serves, plus one cold-boot stability run.
3. Produce `results/ab/final-<stamp>/` with a baseline-vs-final table (Δ% per metric).
   Add a `tool-eval-bench` Responsiveness score run (median turn latency) as the final
   acceptance metric — it weights exactly the TTFT+decode latency this plan targets.
4. Update README `.env` table + "What runs" for any new knobs (C2/C3/C4/C5) and note the
   indexer hardening in the overlay docs.
5. Tag `recipe-2026-08-XX`.
6. **Final decision record:** a short `results/ab/DECISION.md` listing adopted/skipped
   changes and the measured justification for each.

---

## Appendix A — Reference commands

```bash
# per-arm relaunch (overlay change = rebuild; otherwise skip BUILD)
./start.sh stop && BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart

# endpoint + engine checks
curl -sf http://127.0.0.1:8081/health                       # 503 = EngineDeadError
curl -s  http://127.0.0.1:8081/metrics | grep -E 'spec_decode_num_(drafted|accepted)|num_preemptions'
# per-position acceptance (vllm#53030 sanity: must decay, never flat 1.00)
curl -s  http://127.0.0.1:8081/metrics | grep spec_decode_num_accepted_tokens_per_pos

# bench (the repo's own protocol)
python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence --out results/ab/<arm>-structured.json
python3 tests/bench_decode.py --phase prose       --runs 5 --max-tokens 400 --skip-coherence --out results/ab/<arm>-prose.json

# env / image fingerprint
docker image inspect -f '{{.Id}}' ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3
ssh -T "$WORKER_SSH" "docker image inspect -f '{{.Id}}' ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3"   # must equal the head's id
docker exec glm53-exl3-head env | grep -E 'NCCL|KV_CACHE|GPU_MEM|SPEC|EXL3' | sort
```

## Appendix B — Per-arm record (`arm.json`)

```json
{
  "arm": "c1-indexer",
  "date": "2026-08-28T00:00:00Z",
  "git_sha": "<sha>",
  "image_id": "<docker image id>",
  "worker_image_id": "<worker docker image id — must equal image_id>",
  "temperature": {"bench_temp0": null, "production_temp": null},
  "env_fingerprint": {
    "MAX_MODEL_LEN": 1000000, "GPU_MEM_UTIL": "0.85", "MAX_NUM_BATCHED_TOKENS": 1024,
    "DFLASH_TOKENS": 7, "GLM53_MIXED_PREFILL_CHUNK": "skip", "KV_CACHE_DTYPE": "fp8",
    "SPEC_METHOD": "dflash"
  },
  "env_diff_vs_baseline": {},
  "metrics": {
    "structured_decode_tok_s": 0, "prose_decode_tok_s": 0,
    "agg_x1": 0, "agg_x2": 0, "agg_x4": 0,
    "acceptance": 0, "acceptance_per_pos": [],
    "ttft_s": 0, "boot_to_ready_s": 0, "kv_pool_tokens": 0,
    "preemptions": 0, "nan_or_garbage": 0
  },
  "notes": ""
}
```

## Appendix C — Fenced investigation arms (do NOT mix into the A/B set)

1. **Rank-1 KV asymmetry** (NVFP4 OPEN-PROBLEMS #1): the worker rank profiles 3–5 GiB less
   KV headroom than the head; the pool is `min()` across ranks. One DEBUG boot in Phase 1
   (step 6, fenced extra). Closing it ≈ +54% pool. Outcome: a named differing term
   (`peak_activation_memory` vs `non_kv_cache_memory`), filed upstream if it is vLLM's.
2. **InstantTensor direct-I/O loader** (OPEN-PROBLEMS #3): 610 s → 40–100 s loads would
   make every A/B arm ~4× cheaper (more reps per arm). In all four NVFP4 TP2 boots a rank
   died silently ~60–90 s after load (exit code None, no dmesg trace). Rules if attempted:
   never the baseline serving config; dedicated arm only; re-pin `nvidia-nccl-cu13` to
   2.30.7 in the same Docker layer (the pip install silently downgrades it); any silent
   rank death = immediate abort.
