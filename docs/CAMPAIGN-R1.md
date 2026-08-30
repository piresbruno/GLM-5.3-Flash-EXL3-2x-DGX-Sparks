# Campaign R1 — runbook: bundle A/B + DSD concurrency arm (2026-08-30)

Companion to `AB-PLAN.md` (protocol rules apply to every arm) and
`docs/RESEARCH-PERF-NEXT.md`. Implementation state is recorded in
`results/ab/DECISION.md`. This is the execution runbook for the on-GPU phases;
the implementation + offline verification is complete (see "Status").

## Status

| Phase | State |
|---|---|
| 0 — ops pre-flight | DONE. Driver 580.173.02 both nodes; CX7 soak clean (109.25 Gb/s, 25 min, zero errors — `results/ab/r1-phase0/soak.md`); digest pinned. Freeze forensics: `results/ab/r1-phase0/freeze-20260830.md`. |
| 1 — bundle implementation | DONE (C1/C2/C3 committed; tests 14/14 offline). |
| 2 — bundle A/B | **DONE — ADOPTED** (`results/ab/DECISION-R1.md`, arms `baseline-r1ref-20260830-0715` + `r1-auto-20260830-0633`). |
| 3 — DSD arm | **DONE — REJECTED, dormant** (arm `dsd-r1-20260830-0725`: ×1 −7.0% unconditional reject; aggregates +71%/+35% ride on async ON). |
| 4 — ops activation + close | Timers installed, NOT enabled (operator-approved step: `local/install-ops-units.sh --enable`). Tags `baseline-r1-20260830`, `recipe-r1-20260830`. |

## Phase 0 — finish the ops pre-flight (blocking)

1. **Driver branch** — done 2026-08-29: both nodes 580.173.02. Re-check after
   any reboot/maintenance. `./start.sh` and `local/check-updates.sh` now
   refuse/warn on 590.x automatically.
2. **CX7 link soak** (fleet died 2026-08-29 19:46 on this link):
   ```bash
   # server on head, client on worker; >= 30 min, then swap directions
   ib_write_bw -d rocep1s0f1 -R -D 1800          # head
   ib_write_bw -d rocep1s0f0 10.100.24.2 -D 1800 # worker
   ```
   Gate: zero `RETRY_EXC` events, no port-state flaps in `dmesg`, and
   `grep -c 'port error' ` on both nodes' logs unchanged. A second active rail
   exists (`rocepP2p1s0f1` head ↔ `rocepP2p1s0f0` worker): optional dual-rail
   A/B via `NCCL_IB_MERGE_NICS=1` — record only, do not adopt blind.
3. **Digest pin** — already applied: `.env` `IMAGE=…@sha256:9bb1557a…` +
   `SKIP_PULL=1`. Verify both nodes still hold it:
   `docker image inspect $IMAGE -f '{{json .RepoDigests}}'` (head) and the
   same over `WORKER_SSH`. `local/check-updates.sh` automates this.

## Phase 2 — bundle A/B (new baseline)

Baseline arm = the pre-R1 committed config **re-benched fresh** (resolves the
stale-acceptance question: our 65.9 / 6.43-per-step numbers may predate the
xgrammar backports — their W1 gained 0.9804 → 1.0000 acceptance, 67.4 → 70.4).
Arm R1 = the bundle. Same-protocol benches, interleaved, fingerprint per arm
(`tests/collect_fingerprint.sh`), image digest + git sha per arm (AB-PLAN
rule 7).

**Baseline arm boot** (explicitly disable the bundle knobs):

```bash
./start.sh stop
MAX_NUM_BATCHED_TOKENS=2048 \
LONG_PREFILL_TOKEN_THRESHOLD= \
ASYNC_SCHEDULING=auto \
VLLM_PREFIX_CACHE_RETENTION_INTERVAL= \
FLASHINFER_WORKSPACE_BASE= \
./start.sh restart
```

**R1 arm boot** (bundle defaults, AUTO POOL — the pin is REJECTED):

```bash
./start.sh stop
CACHE_FLUSHER=1 ./start.sh restart   # bundle defaults from start.sh/.env
```

**KV pin: REJECTED (2026-08-30).** Three pinned boots froze this kit with the
NVRM `NV_ERR_NO_MEMORY` kernel signature — 17.7 GiB ×2 (C4, 08-29, both nodes)
and 14.64 GiB (R1, 08-30, dgx1 at API bring-up; full forensics in
`results/ab/r1-phase0/freeze-20260830.md`). The pin reserves KV upfront from
MemFree and consumes the auto path's slack, which the post-init window (API
bring-up + MM warmup) needs. **Auto pool is the R1 config**: 963,265 tokens =
**1.61×** the 600k window — satisfies the pool gate with margin. start.sh now
refuses any pin unless `ALLOW_KV_PIN=1` (break-glass + measured memfloor
artifact, D2 methodology).

Freeze-window hardening now in every boot: the cache-flusher sidecar runs
**until the fleet is serving-stable** (health OK ×5 past 900 s, cap 45 min,
logged to `logs/cache_flusher.log`), and a **post-init cache drain** runs on
both nodes after health and before the MM/shape warmup burst.

Boot once per arm, then run, interleaved, per AB-PLAN:

```bash
# decode benches (structured + prose), concurrency ladder, warm before timing
python3 tests/bench_decode.py --out results/ab/<arm>/decode.json
python3 tests/bench_concurrency.py --out results/ab/<arm>/conc.json
# 100k cold prefill TTFT (their evidence: ~893 tok/s prefill → 100k ≈ 112 s)
# cache gates
local/cache-probe.sh --out results/ab/<arm>/cache.json
# or, as an arm artifact with fingerprint:
python3 tests/bench_prefix_cache.py --label <arm> --out results/ab/<arm>
# quality batteries
local/serving-probe.sh
local/acceptance.sh
python3 tests/eval_math500.py --n 100 --out results/ab/<arm>/math500.json   # optional pre-campaign floor
# memory floor per arm (D2 methodology)
tools/memfloor.sh <arm> -- python3 tests/bench_concurrency.py
# HOL probe: 1.2k-token request behind a 240k cold prefill — first token ≤ 30 s
# (their evidence: 7.9 s)
```

### Gates (arm R1 vs the re-benched baseline record)

| Metric | Gate |
|---|---|
| Structured ×1 | ≥ +3% expected; hard floor ±3% no-regress |
| Prose ×1 | ≥ +3% expected; floor ±3% |
| 100k cold prefill | TTFT < 204.7 s (the D1 record) — their ~893 tok/s evidence |
| Cache burst (4×60k ×3 rounds) | rounds 2–3 mean hit ≥ 90% |
| Cache solo 110k replay | hit ≥ 93% held |
| HOL (1.2k behind 240k cold) | first token ≤ 30 s |
| Pool / memory | pool ≥ 1.0× of the 600k window (auto: 963,265 = **1.61×** ✓); memfloor artifact recorded per arm (D1 standing state: head 0.53 / worker 4.3–5.2 GiB) |
| Stability | 0 preemptions, no NaN, no dmesg faults; DSD off |

**Exit:** bundle adopted as baseline → `results/ab/DECISION.md` R1 entry,
README/`.env` table updated (done in the implementation pass), tag
`baseline-r1-<date>`.

## Phase 3 — DSD concurrency arm (on the R1 baseline)

```bash
./start.sh stop
DSD_TABLE=1:1:7,2:999:5 ASYNC_SCHEDULING=1 ./start.sh restart   # pin OFF (unset KV_CACHE_MEMORY)
python3 tests/verify_dsd.py                                     # receipt must print ACTIVE
```

- MNBT/threshold/retention stay bundle. Pin off = recorded deviation
  (KV pin × async double-counts the SW-family reservation — vLLM #47728
  class; admission check can fail).
- `on_ready` warns if the boot downgraded cudagraph mode — do not benchmark
  such a boot.
- Gates: structured ×1 within ±3% of R1 (if ×1 regresses > 3%, DSD is
  rejected regardless of ×2/×4 — the async cost is the price of admission);
  aggregate ×2 and ×4 ≥ +3% vs R1; memfloor artifact; verify_dsd receipt.
- Outcome: default-on only if it wins; otherwise ships dormant (current
  default) and the async-cost data point goes into DECISION.md.

## Phase 4 — ops activation + close

1. `local/install-ops-units.sh --enable` (operator-approved; xid-check,
   metrics-alert, check-updates user timers).
2. README R1 section, security note and ops-kit table — already in place from
   the implementation pass; update the bench tables once Phases 2–3 land.
3. `docs/RESEARCH-PERF-NEXT.md` verdicts updated (done for the adopted
   config; append measured numbers post-bench).
4. `results/ab/DECISION.md` R1 + D5 entries with the arm artifacts, then tag
   `recipe-r1-<date>`.

## Recorded deviations from the upstream kit

1. **Loopback bind NOT adopted** — operator keeps the LAN bind
   (`0.0.0.0`); the unauthenticated-LAN exposure is documented in README.
2. **"MNBT < 3584 → ~0% cache hits" not adopted blind** — our measured 93%
   solo hits at MNBT=1024 (D1-era) contradicts it; the retention env, not
   MNBT alone, is treated as the multi-session lever. MNBT 3584 is adopted
   for the prefill benefit and re-gated by `local/cache-probe.sh` (D1-style).
3. **Async off via a first-class knob** (`ASYNC_SCHEDULING`), not a literal
   `--no-async-scheduling` in `EXTRA_ARGS` — the knob makes the DSD coupling
   enforceable (`dsd_validate` refuses DSD without async) and arms cannot
   double-add scheduler flags.
4. **Pin flag spelled `--kv-cache-memory-bytes`** — the exact flag the pinned
   image registers (the kit historically emitted the abbreviated
   `--kv-cache-memory` form).
5. **KV pin REJECTED entirely** (2026-08-30 freeze): the upstream kit's 14.36 GiB
   pin reference does NOT transfer — our kit froze at 14.64 GiB (and twice at
   17.7 GiB in C4) with the NVRM `NV_ERR_NO_MEMORY` kernel signature, always in
   the post-init window. Auto pool (1.61× margin) is the R1 geometry; the DSD
   arm's "pin off" deviation is now moot (nothing is pinned).
6. **DSD arm runs unpinned** — moot after the pin rejection (nothing is
   pinned); the async admission-check interaction is avoided by construction.
