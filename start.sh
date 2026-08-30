#!/usr/bin/env bash
# ============================================================================
# start.sh — Spark runtime for GLM-5.3-Flash EXL3 (SM121 / GB10)
# ============================================================================
#
# We serve Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw (mirror of
# brandonmusic/GLM-5.3-Flash-tr3-4bpw @ 5ab363a8) on this 2× DGX Spark (GB10 /
# SM121) kit: vLLM TP=2 over CX7, OpenAI API on :8888, NoPE-MLA overlay image.
# DFlash2-7 is the default speculator. Target KV stays packed fp8_ds_mla;
# the SM120 B12X recipe (EP2/DCP2 + nvfp4_ds_mla) is a different image/arch.
#
#   head   : this machine (HEAD_IP, default 10.0.0.1) — vLLM rank 0 + API
#   worker : WORKER_USER@WORKER_IP (default: $USER@10.0.0.2) — vLLM rank 1, --headless
#   layout : --tensor-parallel-size 2, --nnodes 2, mp executor (not Ray)
#
# EXL3, not NVFP4. Do not pass --moe-backend marlin.
#
# What we do:
#   1. preflight  — docker/ssh/disk on both nodes
#   2. image      — docker pull IMAGE from GHCR (public :exl3 tag). If the
#                   worker is missing that digest, try docker pull there,
#                   then fall back to docker save --platform | ssh docker
#                   load (issue #8). SKIP_PULL=1 keeps a local copy.
#                   BUILD=1 rebuilds from this repo instead. Local-only
#                   tags (no slash) skip pull. SKIP_SHIP=1 never copies.
#   3. download   — EXL3/TR3 (+ DFlash2) into the local HF cache if missing
#   4. sync       — rsync that cache to the worker (each rank loads local disk)
#   5. launch     — worker --headless, then head + `vllm serve` (both
#                   --network host --ipc=host)
#   6. wait       — poll /health up to READY_TIMEOUT, then a nonfatal
#                   DFlash2/sampler shape warmup (GLM53_BOOT_SHAPE_WARMUP)
#
# Usage:
#   ./start.sh                    start (download/sync/launch) — default
#   ./start.sh download           EXL3 (+ DFlash2) into the head HF cache only
#                                 (no worker). Same as ./download.sh
#   ./start.sh stop               stop both nodes
#   ./start.sh restart            stop + start
#   ./start.sh status             containers + API health
#   ./start.sh logs               follow head logs
#   ./start.sh logs worker        follow worker container logs
#
# Node IPs live in .env (copied from .env.example on first run).
# Handy overrides: SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_PULL=1 SKIP_SHIP=1 PULL=1 BUILD=1 TAIL=1 HF_TOKEN=...
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || {
        echo "ERROR: missing .env.example" >&2
        exit 1
    }
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    printf '\033[1;36m[glm53-exl3]\033[0m wrote .env from .env.example — edit HEAD_IP / WORKER_IP if needed\n'
fi
# Caller exports (MTP_TOKENS=2 ./start.sh restart) must win over .env.
_cli_mtp="${MTP_TOKENS-}"
_cli_spec="${SPEC_METHOD-}"
_cli_eager="${ENFORCE_EAGER-}"
_cli_fused="${EXL3_FUSED_MOE-}"
_cli_image="${IMAGE-}"
_cli_util="${GPU_MEM_UTIL-}"
_cli_lm="${LANGUAGE_MODEL_ONLY-}"
_cli_max_num_seqs="${MAX_NUM_SEQS-}"
# A/B arm knobs (C6/C7/C8 + C4/C2) must also survive the .env source so
# inline overrides like DFLASH_TOKENS=5 ./start.sh restart actually win.
_cli_dflash_tokens="${DFLASH_TOKENS-}"
_cli_dflash_draft_tp="${DFLASH_DRAFT_TP-}"
_cli_max_batched="${MAX_NUM_BATCHED_TOKENS-}"
_cli_kv_dtype="${KV_CACHE_DTYPE-}"
_cli_kv_mem="${KV_CACHE_MEMORY-}"
_cli_cg="${CG_ESTIMATE-}"
_cli_max_len="${MAX_MODEL_LEN-}"
_cli_mixed="${GLM53_MIXED_PREFILL_CHUNK-}"
_cli_flusher="${CACHE_FLUSHER-}"
_cli_dsd="${DSD_TABLE-}"
# R1 bundle knobs survive the .env source too, so inline arm overrides like
# ASYNC_SCHEDULING=1 DSD_TABLE=... ./start.sh restart actually win. These four
# use set-vs-unset semantics ("${var+set}"): an explicitly EMPTY inline value
# (e.g. LONG_PREFILL_TOKEN_THRESHOLD= on the Phase-2 baseline arm) OVERRIDES the
# start.sh bundle default — that is the documented way to disable a bundle knob
# for an arm.
_cli_lpt="${LONG_PREFILL_TOKEN_THRESHOLD-}"
_cli_lpt_set="${LONG_PREFILL_TOKEN_THRESHOLD+set}"
_cli_async="${ASYNC_SCHEDULING-}"
_cli_async_set="${ASYNC_SCHEDULING+set}"
_cli_retention="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL-}"
_cli_retention_set="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL+set}"
_cli_fiws="${FLASHINFER_WORKSPACE_BASE-}"
_cli_fiws_set="${FLASHINFER_WORKSPACE_BASE+set}"
_cli_qps="${NCCL_IB_QPS_PER_CONNECTION-}"
_cli_qps_set="${NCCL_IB_QPS_PER_CONNECTION+set}"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
[ -n "${_cli_mtp}" ] && MTP_TOKENS="$_cli_mtp"
[ -n "${_cli_spec}" ] && SPEC_METHOD="$_cli_spec"
[ -n "${_cli_eager}" ] && ENFORCE_EAGER="$_cli_eager"
[ -n "${_cli_fused}" ] && EXL3_FUSED_MOE="$_cli_fused"
[ -n "${_cli_image}" ] && IMAGE="$_cli_image"
[ -n "${_cli_util}" ] && GPU_MEM_UTIL="$_cli_util"
[ -n "${_cli_lm}" ] && LANGUAGE_MODEL_ONLY="$_cli_lm"
[ -n "${_cli_max_num_seqs}" ] && MAX_NUM_SEQS="$_cli_max_num_seqs"
[ -n "${_cli_dflash_tokens}" ] && DFLASH_TOKENS="$_cli_dflash_tokens"
[ -n "${_cli_dflash_draft_tp}" ] && DFLASH_DRAFT_TP="$_cli_dflash_draft_tp"
[ -n "${_cli_max_batched}" ] && MAX_NUM_BATCHED_TOKENS="$_cli_max_batched"
[ -n "${_cli_kv_dtype}" ] && KV_CACHE_DTYPE="$_cli_kv_dtype"
[ -n "${_cli_kv_mem}" ] && KV_CACHE_MEMORY="$_cli_kv_mem"
[ -n "${_cli_cg}" ] && CG_ESTIMATE="$_cli_cg"
[ -n "${_cli_max_len}" ] && MAX_MODEL_LEN="$_cli_max_len"
[ -n "${_cli_mixed}" ] && GLM53_MIXED_PREFILL_CHUNK="$_cli_mixed"
[ -n "${_cli_flusher}" ] && CACHE_FLUSHER="$_cli_flusher"
[ -n "${_cli_dsd}" ] && DSD_TABLE="${_cli_dsd}"
[ "${_cli_lpt_set:-}" = set ] && LONG_PREFILL_TOKEN_THRESHOLD="$_cli_lpt"
[ "${_cli_async_set:-}" = set ] && ASYNC_SCHEDULING="$_cli_async"
[ "${_cli_retention_set:-}" = set ] && VLLM_PREFIX_CACHE_RETENTION_INTERVAL="$_cli_retention"
[ "${_cli_fiws_set:-}" = set ] && FLASHINFER_WORKSPACE_BASE="$_cli_fiws"
[ "${_cli_qps_set:-}" = set ] && NCCL_IB_QPS_PER_CONNECTION="$_cli_qps"

# Helpers are defined BEFORE the configuration section: the config-side guards
# (GPU_MEM_UTIL hard limit, ASYNC_SCHEDULING validation, dsd_validate) refuse
# with die() during parsing, before the old post-config helper block ran.
log()  { printf '\033[1;36m[glm53-exl3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-exl3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-exl3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# ----------------------------- configuration -------------------------------
MODEL="${MODEL:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}"
# If the durable mirror is empty/moved, download.sh falls back to this id.
MODEL_FALLBACK="${MODEL_FALLBACK:-brandonmusic/GLM-5.3-Flash-tr3-4bpw}"
MODEL_CACHE_NAME="${MODEL_CACHE_NAME:-models--${MODEL//\//--}}"
MODEL_FALLBACK_CACHE_NAME="${MODEL_FALLBACK_CACHE_NAME:-models--${MODEL_FALLBACK//\//--}}"
# Hub commit on the Mia-AiLab mirror (the 5ab363a8-byte-identical upload).
MODEL_REVISION="${MODEL_REVISION:-25a44fdbf16862a46b7cc9921142c6c81350af2f}"
IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
GHCR_USER="${GHCR_USER:-MiaAI-Lab}"

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
# Same OS user on both Sparks unless .env sets WORKER_USER (mixed-account kits).
WORKER_USER="${WORKER_USER:-$USER}"
if [ "$WORKER_USER" = "$USER" ]; then
    WORKER_HOME="${WORKER_HOME:-$HOME}"
else
    WORKER_HOME="${WORKER_HOME:-/home/${WORKER_USER}}"
fi
WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"

HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# vLLM subtracts a CUDA-graph memory ESTIMATE from the KV pool. On this kit the
# estimate is 2.43 GiB while the captured graphs actually consume -0.19 GiB, so
# ~2.6 GiB of KV is reserved and never used. 0 keeps CUDA graphs ON and drops only
# the deduction. 1 = upstream default.
CG_ESTIMATE="${CG_ESTIMATE:-1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-$WORKER_HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
# glm53-flash already ships nvidia-nccl. LD_PRELOAD of the host 2.30.7 SO
# makes DeepEP assert duplicate NCCL (/nccl/... vs nvidia/nccl/lib/...).
# Set USE_HOST_NCCL=1 only if image NCCL cannot talk CX7.
USE_HOST_NCCL="${USE_HOST_NCCL:-0}"

TP="${TP:-2}"
NNODES="${NNODES:-2}"
PORT="${PORT:-8888}"
MASTER_PORT="${MASTER_PORT:-29521}"
# C2: GB10 memory ritual (drop_caches + compact + swappiness=0 + swap cycle)
# before docker run on both nodes; WARN-only on failure. Opt out: SKIP_MEM_RITUAL=1
SKIP_MEM_RITUAL="${SKIP_MEM_RITUAL:-0}"
# C2 sidecar: flush page cache >40 GiB during weight load (25 min max).
# Validated 2026-08-29 (D2 baseline boot): required on this kit — the load
# window is a dice roll otherwise (MemFree 3.2 GiB idle floor, load pushes it
# through zero; crash review 2026-08-29).
CACHE_FLUSHER="${CACHE_FLUSHER:-0}"
# D4: WEIGHTS_MODE=local (default; weights on both boxes, rsync) | nfs
# (head reads the worker's cache over NFS — NFS-paced load avoids the measured
# head wedge when a local NVMe read outruns UMA page-cache reclaim; the head
# keeps its local copy, the point is load pacing). NFS_PORT = host port of
# the worker's containerized export.
WEIGHTS_MODE="${WEIGHTS_MODE:-local}"
NFS_PORT="${NFS_PORT:-12049}"
NFS_VOL_NAME="${NFS_VOL_NAME:-glm53-exl3-weights}"
# D4: served thinking default. 1 = thinking on (pre-D4 behavior), 0 = off
# (Entrpi-validated +7% structured acceptance; reasoning goes to
# message.reasoning). Clients can still override per request.
GLM53_DEFAULT_THINKING="${GLM53_DEFAULT_THINKING:-1}"

MTP_TOKENS="${MTP_TOKENS:-4}"  # D4: MTP fallback k=4 (Entrpi-measured 28.6 vs ~24.6 tok/s at k=2; k=5 regresses)
# dflash (default, incoai/GLM-5.3-Flash-DFlash2, k=7) | mtp | none
SPEC_METHOD="${SPEC_METHOD:-dflash}"
DFLASH_MODEL="${DFLASH_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH_CACHE_NAME="${DFLASH_CACHE_NAME:-models--${DFLASH_MODEL//\//--}}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
# 1 = keep the ~2.3 GiB drafter on rank 0 (no CX7 on every draft step).
# Empty = inherit target TP. Do not pin attention_backend: SM121 already
# prefers FLASH_ATTN for non-causal dense SWA. TRITON_ATTN was an SM120
# mask-fix copy this image does not have.
DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}"
# HARD LIMIT (2026-08-29 crash review): on this kit GPU_MEM_UTIL > 0.85 has
# frozen both nodes twice (UVM livelock: shm_broadcast starvation -> power
# cycle). 0.85 auto is the profiled-safe envelope (~13.7 GiB KV @1M, 1.37x).
# Refuse louder-than-0.85 rather than reproduce the freeze.
if awk "BEGIN{exit !(${GPU_MEM_UTIL:-0.85} > 0.85)}" 2>/dev/null; then
    die "GPU_MEM_UTIL=${GPU_MEM_UTIL} > 0.85 is not allowed on this kit (crash review 2026-08-29: >0.85 froze both nodes). Set 0.85."
fi
# Pin guard (HARD, 2026-08-30): pins have frozen this kit three times — 17.7 GiB
# x2 (C4, 08-29) and 14.64 GiB (R1, 08-30, dgx1 at API bring-up) — all with the
# NVRM NV_ERR_NO_MEMORY kernel signature (results/ab/r1-phase0/freeze-20260830.md).
# The pin reserves KV upfront from MemFree and consumes the auto path's slack,
# which the post-init window (API bring-up + MM warmup) needs. Auto pool is the
# R1 config: 963,265 tokens = 1.61x the 600k window. A pin requires the
# explicit ALLOW_KV_PIN=1 break-glass plus a measured memfloor artifact (D2).
if [ -n "${KV_CACHE_MEMORY:-}" ]; then
    if [ "${ALLOW_KV_PIN:-0}" != "1" ]; then
        die "KV_CACHE_MEMORY is set but pins are REJECTED on this kit (3 freezes: 17.7 GiB x2 C4, 14.64 GiB R1 — NVRM NV_ERR_NO_MEMORY at API bring-up; auto pool 963k tokens = 1.61x covers the 600k window). Unset KV_CACHE_MEMORY, or export ALLOW_KV_PIN=1 to override with a measured memfloor artifact (D2)."
    fi
    warn "ALLOW_KV_PIN=1: pinning KV_CACHE_MEMORY=${KV_CACHE_MEMORY} — run tools/memfloor.sh during the next saturation and watch MemFree; three pinned boots have frozen this kit"
    if [ "${CG_ESTIMATE:-1}" != "0" ]; then
        warn "KV_CACHE_MEMORY is set while CG_ESTIMATE=1: the CUDA-graph deduction (~1.4 GiB) is still reserved from the pool — prefer CG_ESTIMATE=0 over pinning"
    fi
fi
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# R1 bundle (2026-08-30, derived from the Reederey87 kit): 3584 = 14 × 256-token
# pages — page-exact prefill chunks. 8192 is still the known indexer-topk smem
# OOM on this lane. D1 measured 512->2048 (−25.7% 100k TTFT); R1 3584 re-gates
# the prefill step budget with the long-prefill path (LPT below).
# 8192 chunk × long history oversubscribes GB10 persistent_topk smem (300k crash).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3584}"

# ---- R1 bundle knobs (scheduler / cache geometry) --------------------------
# These are start.sh's bundle defaults so a restart never silently loses the
# bundle; arms override explicitly (see docs/CAMPAIGN-R1.md):
#   LONG_PREFILL_TOKEN_THRESHOLD=        (empty = scheduler default 0 = cap off)
#   ASYNC_SCHEDULING=auto                (vLLM auto — the Phase-2 baseline arm)
#   VLLM_PREFIX_CACHE_RETENTION_INTERVAL= (empty = unset = dense checkpoints)
# ---
# Requests with more prompt tokens than this take the long-prefill scheduling
# path (chunked prefill / overlap-with-decode). Bundle value 1792 = 7 pages,
# exactly half the 3584 step budget. Scheduler flag, emitted directly by the
# inner scripts (kept OUT of EXTRA_ARGS so an arm cannot double-add it).
# Bundle-knob defaults use ${var-default} (NOT :-): an explicitly EMPTY value
# (arm override, set-vs-unset semantics above) must stay empty and disable the
# knob, not be re-defaulted. MAX_NUM_BATCHED_TOKENS keeps :- (no empty
# override semantics; an empty MNBT would break the flag emission).
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD-1792}"
# 0 = --no-async-scheduling (bundle baseline; async off). 1 = --async-scheduling
# (required by the DSD arm — dsd_validate enforces it). auto = pass neither
# flag, vLLM decides (the Phase-2 baseline-arm setting, resolves the
# "async is not free" A/B: async off measured at/above async on at c=1).
ASYNC_SCHEDULING="${ASYNC_SCHEDULING-0}"
case "$ASYNC_SCHEDULING" in
    0|1|auto|"") ;;   # empty = no flag emitted = vLLM auto (same as auto)
    *) die "ASYNC_SCHEDULING=$ASYNC_SCHEDULING must be 0, 1 or auto (0=off bundle, 1=on DSD, auto=vLLM decides)" ;;
esac
# Sparse KDA retention for prefix caching (fork env, vLLM reads it in-process):
#   unset/empty = dense checkpoints (default); 0 = retain only the latest
#   completed prompt boundary + shared-prefix junctions; N>0 = checkpoints at
#   every N-th boundary. This fork extends it to the SWA/MambaSpec (KDA) groups.
# Forwarded to BOTH ranks conditionally on non-empty — the fork's env parser
# runs int() on the value, so an EMPTY STRING crashes boot (must mean "unset").
VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL-0}"
# FlashInfer JIT workspace root, INSIDE the already-mounted vLLM cache dir
# (-v $CACHE_ROOT:/root/.cache/vllm on the head, $WORKER_VLLM_CACHE on the
# worker), so JIT kernels survive container recreate and watchdog heals pay no
# re-JIT. Path is in-container and identical on both ranks.
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE-/root/.cache/vllm/flashinfer}"

# ---- D5: dynamic speculative decoding (DSD) --------------------------------
# vLLM Dynamic SD (PR #32374, present in this image): per-batch-size draft
# length. Empty = OFF (static k=DFLASH_TOKENS; byte-identical to pre-D5).
# Format "start_bs:end_bs:k,...", e.g. DSD_TABLE=1:1:7,2:999:5 (k=7 solo,
# k=5 from 2 concurrent). Active path on this kit: AsyncScheduler sizes the
# per-step draft placeholders from the table (num_spec_tokens_to_schedule)
# and update_draft_token_ids_in_output trims the drafter's static-k output
# to it -> the drafter keeps proposing k=7 (no drafter patch) while the
# verify batch runs 1+K. Verify-shape graphs: seq*(1+K) per table row.
DSD_TABLE="${DSD_TABLE:-}"

dsd_k_for() {
    # K for a batch size (table MUST be dsd_validate'd first).
    local bs="$1" e rest start end k
    for e in $(printf '%s' "$DSD_TABLE" | tr ',' ' '); do
        start=${e%%:*}; rest=${e#*:}; end=${rest%%:*}; k=${rest##*:}
        [ "$bs" -ge "$start" ] && [ "$bs" -le "$end" ] && { echo "$k"; return; }
    done
    echo "$DFLASH_TOKENS"
}

dsd_validate() {
    # Dies on any malformation; empty table is a valid no-op (DSD off).
    [ -n "$DSD_TABLE" ] || return 0
    [ "$SPEC_METHOD" = "dflash" ] || die "DSD_TABLE requires SPEC_METHOD=dflash (got $SPEC_METHOD)"
    # R1: the DSD active path sizes per-step draft placeholders inside the
    # AsyncScheduler — DSD without async scheduling silently degrades to the
    # static ladder. The bundle defaults async OFF, so the arm must opt back in.
    [ "${ASYNC_SCHEDULING:-0}" = "1" ] || die "DSD_TABLE requires ASYNC_SCHEDULING=1 (the DSD active path is the AsyncScheduler; the R1 bundle defaults async off)"
    local e rest start end k prev_end=0 count=0
    for e in $(printf '%s' "$DSD_TABLE" | tr ',' ' '); do
        count=$((count + 1))
        case "$e" in
            *[!0-9:]*) die "DSD_TABLE entry '$e' malformed (want start:end:k)" ;;
        esac
        case "$(printf '%s' "$e" | tr -cd ':' | wc -c)" in
            2) ;;
            *) die "DSD_TABLE entry '$e' malformed (want start:end:k)" ;;
        esac
        start=${e%%:*}; rest=${e#*:}; end=${rest%%:*}; k=${rest##*:}
        [ "$start" -ge 1 ] || die "DSD_TABLE batch sizes are 1-based ('$e')"
        [ "$start" -le "$end" ] || die "DSD_TABLE inverted range ('$e')"
        [ "$start" -eq "$((prev_end + 1))" ] || die "DSD_TABLE ranges must be contiguous from 1 ('$e' after end=$prev_end)"
        [ "$k" -ge 1 ] || die "DSD_TABLE k must be >= 1 ('$e')"
        [ "$k" -le "$DFLASH_TOKENS" ] || die "DSD_TABLE k=$k exceeds trained block DFLASH_TOKENS=$DFLASH_TOKENS ('$e')"
        [ "$k" -le 7 ] || die "DSD_TABLE k=$k exceeds the trained DFlash2 block (8 slots = k 7)"
        prev_end=$end
    done
    [ "$prev_end" -ge "$MAX_NUM_SEQS" ] || die "DSD_TABLE must cover 1..MAX_NUM_SEQS=$MAX_NUM_SEQS (last range ends at $prev_end)"
    [ "$count" -le 8 ] || die "DSD_TABLE: too many ranges ($count); keep the ladder small"
    return 0
}

dsd_capture_sizes() {
    # Target verify-shape graph ladder: union of the piecewise base sizes and
    # seq*(1+K) per concurrency 1..MAX_NUM_SEQS. Replaces the static
    # 1 2 4 8 16 24 32 ladder (16/32 unreachable under DSD; 12/18/24 new).
    local out="1 2 4" s k size
    s=1
    while [ "$s" -le "$MAX_NUM_SEQS" ]; do
        k=$(dsd_k_for "$s")
        size=$(( s * (k + 1) ))
        out="$out $size"
        s=$((s + 1))
    done
    printf '%s\n' $out | sort -n -u | tr '\n' ' ' | sed 's/ $//'
}
# ---- end D5 -----------------------------------------------------------------
dsd_validate
CHAT_TEMPLATE_HOST="${CHAT_TEMPLATE_HOST:-$SCRIPT_DIR/files/chat_template.jinja}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/opt/glm53/chat_template.jinja}"
VIDEO_PATCH_HOST="${VIDEO_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm_video_placeholders.py}"
STOP_PATCH_HOST="${STOP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_suppress_stops_in_reasoning.py}"
SCHED_PATCH_HOST="${SCHED_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_scheduler_decode_floor.py}"
DRAFTER_PATCH_HOST="${DRAFTER_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_glm5_drafter_group.py}"
APC_PATCH_HOST="${APC_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_hybrid_prefix_hit.py}"
XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_xgrammar_termination.py}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
QUANTIZATION="${QUANTIZATION:-exl3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
# JSON default cannot sit in ${LIMIT_MM:-{...}} — } ends the expansion.
if [ -z "${LIMIT_MM:-}" ]; then
    LIMIT_MM='{"image":4,"video":1}'
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
# Graph-safe fused apply (device-side expert grouping). MTP k=2 decode is
# 1..4 seqs × 3 tokens (must include 3). DFlash2 k=7 is 1..4 seqs × 8 tokens
# (must include 8, 16, 24, 32).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
if [ "${ENFORCE_EAGER}" != "1" ]; then
    case " ${EXTRA_ARGS:-} " in
        *" --cudagraph-capture-sizes "*|*" cudagraph-capture-sizes "*) ;;
        *)
            if [ "$SPEC_METHOD" = "dflash" ]; then
                if [ -n "$DSD_TABLE" ]; then
                    EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes $(dsd_capture_sizes)"
                else
                    EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes 1 2 4 8 16 24 32"
                fi
            else
                EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes 1 2 3 4 6 8 12"
            fi
            ;;
    esac
fi
# 1 = fused exl3_moe (decode). 0 restores the unique-expert LinearEXL3 loop.
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"

READY_TIMEOUT="${READY_TIMEOUT:-3600}"
# 1 = suppress client stop strings until </think> (DSpark #42 class).
GLM53_SUPPRESS_STOPS_IN_REASONING="${GLM53_SUPPRESS_STOPS_IN_REASONING:-1}"
# Mixed-step prefill policy when a peer is already decoding (issue #6).
# skip = do not mix; N>0 = cap tokens; 0 = off.
GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-skip}"
# EngineCore stock timeout is 300s; mid-serve Triton/TileLang JIT on TP=2 can
# exceed that without being a true hang. NCCL watchdog is still 600s.
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
# 1 = after /health, burn DFlash2 BLOCK / sampler / kpool shapes. Nonfatal.
GLM53_BOOT_SHAPE_WARMUP="${GLM53_BOOT_SHAPE_WARMUP:-1}"
GLM53_WARMUP_REQ_TIMEOUT="${GLM53_WARMUP_REQ_TIMEOUT:-240}"

CONTAINER_HEAD="${CONTAINER_HEAD:-glm53-exl3-head}"
CONTAINER_WORKER="${CONTAINER_WORKER:-glm53-exl3-worker}"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"
FALLBACK_MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_FALLBACK_CACHE_NAME"
DFLASH_PATH="$HF_CACHE_DIR/hub/$DFLASH_CACHE_NAME"
WORKER_CACHE_DIR="$WORKER_HOME/.cache/huggingface"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-$WORKER_HOME/.cache/vllm-glm53-flash}"
# Overlay FS ~/.triton and ~/.tilelang die on container recreate (TP=2 JIT
# stall → 600s NCCL watchdog). Persist next to the vLLM cache.
TRITON_HOST_CACHE="${TRITON_HOST_CACHE:-$CACHE_ROOT/triton}"
TILELANG_HOST_CACHE="${TILELANG_HOST_CACHE:-$CACHE_ROOT/tilelang}"
WORKER_TRITON_CACHE="${WORKER_TRITON_CACHE:-$WORKER_VLLM_CACHE/triton}"
WORKER_TILELANG_CACHE="${WORKER_TILELANG_CACHE:-$WORKER_VLLM_CACHE/tilelang}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton/cache}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/root/.tilelang/cache}"

LOGDIR="$SCRIPT_DIR/logs"
CLUSTER_LOCK="$LOGDIR/cluster.lock"
CLUSTER_LOCK_PID="$LOGDIR/cluster.lock.pid"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-exl3-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-exl3-worker.inner.sh"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-120}"

# ------------------------------- helpers -----------------------------------

memfree_gib() {  # MemFree of a node: "local" | "remote" — prints N.N (GiB)
    if [ "${1:-local}" = "remote" ]; then
        worker_ssh "grep '^MemFree:' /proc/meminfo" 2>/dev/null | awk '{printf "%.1f", $2/1048576}'
    else
        awk '/^MemFree:/{printf "%.1f", $2/1048576}' /proc/meminfo
    fi
}

post_init_cache_drop() {
    # R1 freeze hardening (2026-08-30): the pinned boot froze with NVRM
    # NV_ERR_NO_MEMORY at API bring-up/MM warmup while the load's 164 GiB of
    # page cache still held RAM; drop_caches during the thrash could not save
    # it because the deficit was structural. Drain the page cache BEFORE the
    # warmup burst, on both nodes, and log the receipt
    # (results/ab/r1-phase0/freeze-20260830.md).
    log "post-init cache drain (freeze-window hardening, before MM/shape warmup) ..."
    log "  head  : MemFree=$(memfree_gib local) GiB before drop"
    worker_ssh "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
    log "  worker: MemFree=$(memfree_gib remote) GiB after drop"
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    log "  head  : MemFree=$(memfree_gib local) GiB after drop"
}

# GLM53 numeric config guard (begin)
_glm53_canonical_positive_int() {
    local name="$1" value="$2" maximum="$3" canonical
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a positive base-10 integer (got: $value)" >&2
        return 2
    fi
    canonical="$value"
    while [ "${canonical#0}" != "$canonical" ]; do canonical="${canonical#0}"; done
    [ -n "$canonical" ] || canonical=0
    if [ "$canonical" = 0 ] \
       || [ "${#canonical}" -gt "${#maximum}" ] \
       || [ "$canonical" -gt "$maximum" ]; then
        echo "$name must be between 1 and $maximum (got: $value)" >&2
        return 2
    fi
    printf -v "$name" '%s' "$canonical"
    # $name is one of three fixed names below.
    # shellcheck disable=SC2163
    export "$name"
}

validate_numeric_config() {
    if ! [[ "$GPU_MEM_UTIL" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        echo "GPU_MEM_UTIL must be greater than 0 and at most 1 (got: $GPU_MEM_UTIL)" >&2
        return 2
    fi
    _glm53_canonical_positive_int MAX_MODEL_LEN "$MAX_MODEL_LEN" 1000000 || return
    _glm53_canonical_positive_int MAX_NUM_SEQS "$MAX_NUM_SEQS" 4096 || return
    _glm53_canonical_positive_int MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS" 8388608 || return
}
# GLM53 numeric config guard (end)

# Serialize start/restart so a second launcher cannot docker-rm the first's
# containers mid wait_for_health (false "head container exited" + empty logs).
with_cluster_lock() {
    mkdir -p "$LOGDIR"
    exec 9>"$CLUSTER_LOCK"
    if ! flock -n 9; then
        local holder
        holder="$(tr -d '[:space:]' <"$CLUSTER_LOCK_PID" 2>/dev/null || true)"
        die "another start.sh is already running${holder:+ (pid $holder)}"
    fi
    echo $$ >"$CLUSTER_LOCK_PID"
}

steal_cluster_lock_for_stop() {
    mkdir -p "$LOGDIR"
    exec 9>"$CLUSTER_LOCK"
    if flock -n 9; then
        echo $$ >"$CLUSTER_LOCK_PID"
        return 0
    fi
    local holder
    holder="$(tr -d '[:space:]' <"$CLUSTER_LOCK_PID" 2>/dev/null || true)"
    if [ -n "$holder" ] && [ "$holder" != "$$" ] && kill -0 "$holder" 2>/dev/null; then
        warn "terminating in-flight start.sh pid $holder before stop"
        kill "$holder" 2>/dev/null || true
    fi
    if ! flock -w 30 9; then
        warn "cluster lock still busy — removing containers anyway"
    fi
    echo $$ >"$CLUSTER_LOCK_PID"
}

banner() {
    local label="${1:-start.sh}"
    printf '\n'
    printf '  \033[1;36m┌────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mGLM-5.3 Flash EXL3\033[0m  \033[2m·  %-11s\033[0m        \033[1;36m│\033[0m\n' "$label"
    printf '  \033[1;36m└────────────────────────────────────────────┘\033[0m\n'
    printf '\n'
}

worker_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

count_shards() {
    find "$1/snapshots" -name '*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

ensure_refs_main() {
    local ref="$MODEL_PATH/refs/main" snap
    [ -f "$ref" ] && [ -n "$(<"$ref")" ] && return 0
    snap="$(ls -1t "$MODEL_PATH/snapshots" 2>/dev/null | head -n 1 || true)"
    [ -n "$snap" ] || die "no snapshots under $MODEL_PATH — re-run download"
    mkdir -p "$MODEL_PATH/refs"
    printf '%s' "$snap" >"$ref"
    log "wrote refs/main -> $snap (hf download left it empty)"
}

resolve_model_dir() {
    local ref="$MODEL_PATH/refs/main" hash dir
    ensure_refs_main
    hash="$(<"$ref")"
    dir="$MODEL_PATH/snapshots/$hash"
    [ -f "$dir/config.json" ] || die "config.json missing in $dir — re-run with REFRESH_WEIGHTS=1"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$MODEL_CACHE_NAME" "$hash"
}

ensure_dflash_refs_main() {
    local ref="$DFLASH_PATH/refs/main" snap
    [ -f "$ref" ] && [ -n "$(<"$ref")" ] && return 0
    snap="$(ls -1t "$DFLASH_PATH/snapshots" 2>/dev/null | head -n 1 || true)"
    [ -n "$snap" ] || die "no snapshots under $DFLASH_PATH — re-run download"
    mkdir -p "$DFLASH_PATH/refs"
    printf '%s' "$snap" >"$ref"
    log "wrote DFlash2 refs/main -> $snap"
}

resolve_dflash_dir() {
    local ref="$DFLASH_PATH/refs/main" hash dir
    ensure_dflash_refs_main
    hash="$(<"$ref")"
    dir="$DFLASH_PATH/snapshots/$hash"
    [ -f "$dir/config.json" ] || die "DFlash2 config.json missing in $dir"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$DFLASH_CACHE_NAME" "$hash"
}

check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start.sh restart' or './start.sh stop' first"
        fi
        die "port ${port} is already in use — stop it or rerun with ${envname}=<free-port>"
    fi
}

# GLM53 preflight memory guard (begin)
read_meminfo_kib() {
    local source_file="${1:-/proc/meminfo}"
    awk '
      /^MemTotal:/ { total=$2 }
      /^MemAvailable:/ { available=$2 }
      END {
        if (!total || !available) exit 1
        print total, available
      }
    ' "$source_file"
}

preflight_memory() {
    local label="$1" total_kib="$2" available_kib="$3" util="$4"
    local headroom_kib="${GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB:-2097152}"
    local total_gib available_gib requested_gib headroom_gib

    if ! [[ "$total_kib" =~ ^[0-9]+$ && "$available_kib" =~ ^[0-9]+$ && "$headroom_kib" =~ ^[0-9]+$ ]]; then
        echo "PREFLIGHT FAIL [$label]: invalid memory reading" >&2
        return 2
    fi
    if ! [[ "$util" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$util" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        echo "PREFLIGHT FAIL [$label]: GPU_MEM_UTIL must be greater than 0 and at most 1: $util" >&2
        return 2
    fi

    total_gib=$(awk -v k="$total_kib" 'BEGIN { printf "%.2f", k/1048576 }')
    available_gib=$(awk -v k="$available_kib" 'BEGIN { printf "%.2f", k/1048576 }')
    requested_gib=$(awk -v t="$total_kib" -v u="$util" 'BEGIN { printf "%.2f", (t*u)/1048576 }')
    headroom_gib=$(awk -v k="$headroom_kib" 'BEGIN { printf "%.2f", k/1048576 }')

    if ! awk -v a="$available_kib" -v t="$total_kib" -v u="$util" -v h="$headroom_kib" \
        'BEGIN { exit !(a >= (t*u)+h) }'; then
        echo "PREFLIGHT FAIL [$label]: MemAvailable=${available_gib} GiB of ${total_gib} GiB; GPU_MEM_UTIL=${util} requests ${requested_gib} GiB plus ${headroom_gib} GiB headroom" >&2
        return 1
    fi
    echo "PREFLIGHT OK [$label]: MemAvailable=${available_gib}/${total_gib} GiB; request=${requested_gib} GiB plus ${headroom_gib} GiB headroom"
}
# GLM53 preflight memory guard (end)

trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set it in .env"

    log "checking worker ${WORKER_SSH} ..."
    worker_ssh true 2>/dev/null \
        || die "cannot ssh (key-based) to ${WORKER_SSH} — set up passwordless ssh first"
    worker_ssh "docker info >/dev/null 2>&1" \
        || die "worker cannot talk to its docker daemon (docker group?)"
    worker_ssh "nvidia-smi -L 2>/dev/null | grep -q GB10" \
        || warn "no GB10 GPU visible on worker"

    # R1 Phase-0 gate: stay on the 580.x driver branch. 590.x deadlocks
    # CUDAGraph capture on GB10 (Reederey87 evidence; this kit measured 590.x
    # hangs). If 590.x is installed, stop here and plan a downgrade before any
    # boot — never re-capture graphs on 590.x.
    local drv_head drv_worker side
    drv_head="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
    drv_worker="$(worker_ssh "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null | tr -d '[:space:]' || true)"
    for side in "head:${drv_head:-}" "worker:${drv_worker:-}"; do
        case "${side#*:}" in
            590*) die "driver ${side#*:} on ${side%%:*} is the 590.x branch — 590.x deadlocks CUDAGraph capture on GB10 (R1 Phase 0). Plan a downgrade to 580.x before booting the fleet." ;;
            580*) ;;
            "") warn "driver version unknown on ${side%%:*} (nvidia-smi missing?) — R1 Phase-0 gate unverified" ;;
            *) warn "driver ${side#*:} on ${side%%:*} is not 580.x — R1 Phase-0 gate expects the 580.x branch" ;;
        esac
    done
    log "driver branch: head=${drv_head:-unknown} worker=${drv_worker:-unknown} (R1 Phase-0 gate: 580.x)"

    # NCCL_IB_GID_INDEX must name a populated GID on BOTH nodes' CX7 devices.
    # An empty (all-zero) entry passes every earlier check and then kills the
    # worker rank ~60 s in with ibv_modify_qp errno 61 "No data available" —
    # kits differ: on some GB10 pairs gid 3 is populated on one node and
    # all-zero on the other. Fail here, in seconds, with the fix in hand.
    local gid_head gid_worker gid_path
    gid_path="/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gids/${NCCL_IB_GID_INDEX}"
    gid_head=$(cat "$gid_path" 2>/dev/null | tr -d ':0' || true)
    gid_path="/sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gids/${NCCL_IB_GID_INDEX}"
    gid_worker=$(worker_ssh "cat '$gid_path' 2>/dev/null" | tr -d ':0' || true)
    if [ -z "$gid_head" ] || [ -z "$gid_worker" ]; then
        warn "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} is EMPTY on $( [ -z "$gid_head" ] && echo "head(${HEAD_CX7_IB})" ) $( [ -z "$gid_worker" ] && echo "worker(${WORKER_CX7_IB})" )"
        warn "GID tables (pick an index whose entry is non-zero on BOTH nodes — the ::ffff:<ip> RoCEv2 one):"
        for i in 0 1 2 3; do
            printf '    head   gid%s: %s\n' "$i" "$(cat "/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gids/$i" 2>/dev/null)" >&2
        done
        worker_ssh "for i in 0 1 2 3; do printf '    worker gid%s: %s\n' \"\$i\" \"\$(cat /sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gids/\$i 2>/dev/null)\"; done" >&2 || true
        die "set NCCL_IB_GID_INDEX in .env to a populated index (this kills the worker rank with ibv_modify_qp errno 61 otherwise)"
    fi

    [ "$TP" = "2" ] || warn "TP=${TP} on a 2×1-GPU cluster — expected TP=2"
    [ "$NNODES" = "2" ] || warn "NNODES=${NNODES} — expected 2"

    local others
    others=$(worker_ssh "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  ${CONTAINER_WORKER}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the worker:"
        echo "$others" >&2
        warn "this model needs most of each GB10 — stop GPU containers on the worker first"
    fi

    check_port_free "$PORT" PORT
    check_port_free "$MASTER_PORT" MASTER_PORT

    local head_mem worker_mem head_total head_available worker_total worker_available
    head_mem="$(read_meminfo_kib /proc/meminfo)" \
        || die "cannot read MemTotal/MemAvailable on head"
    worker_mem="$(worker_ssh "cat /proc/meminfo" | read_meminfo_kib /dev/stdin)" \
        || die "cannot read MemTotal/MemAvailable on worker"
    read -r head_total head_available <<< "$head_mem"
    read -r worker_total worker_available <<< "$worker_mem"
    preflight_memory head "$head_total" "$head_available" "$GPU_MEM_UTIL" || return
    preflight_memory worker "$worker_total" "$worker_available" "$GPU_MEM_UTIL" || return

    [ -f "$STOP_PATCH_HOST" ] || die "$STOP_PATCH_HOST missing"
    [ -f "$SCHED_PATCH_HOST" ] || die "$SCHED_PATCH_HOST missing"
    [ -f "$DRAFTER_PATCH_HOST" ] || die "$DRAFTER_PATCH_HOST missing"
    [ -f "$APC_PATCH_HOST" ] || die "$APC_PATCH_HOST missing"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "$XGRAMMAR_PATCH_HOST missing"

    case "$WEIGHTS_MODE" in
        local|nfs) ;;
        *) die "WEIGHTS_MODE=$WEIGHTS_MODE must be local or nfs (D4)" ;;
    esac
    if [ "$WEIGHTS_MODE" = "nfs" ]; then
        warn "WEIGHTS_MODE=nfs: head reads weights over NFS from ${WORKER_IP}:${NFS_PORT} (erichough/nfs-server must be pullable on the worker)"
    fi

    local need_kb=$((180 * 1024 * 1024)) avail
    mkdir -p "$HF_CACHE_DIR"
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on head for a ~164 GiB model"
    avail=$(worker_ssh "df -Pk '$WORKER_HOME' 2>/dev/null" | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on worker for a ~164 GiB model"

    log "preflight OK (head=$(hostname) ${HEAD_IP}, worker=${WORKER_SSH})"
}

# ------------------------------ image --------------------------------------
image_from_registry() {
    case "$IMAGE" in
        */*) return 0 ;;
        *) return 1 ;;
    esac
}

login_ghcr_if_token() {
    [ -n "${GHCR_TOKEN:-}" ] || return 0
    log "docker login ghcr.io as ${GHCR_USER} (GHCR_TOKEN)"
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null
}

login_ghcr_if_token_worker() {
    [ -n "${GHCR_TOKEN:-}" ] || return 0
    log "docker login ghcr.io on worker as ${GHCR_USER} (GHCR_TOKEN)"
    echo "$GHCR_TOKEN" | worker_ssh "docker login ghcr.io -u '$GHCR_USER' --password-stdin" >/dev/null
}

# RepoDigest is stable across overlay2 vs containerd. Those snapshotters
# disagree on .Id (config digest vs index digest), so start.sh used to
# ship even after the worker had already pulled the same GHCR tag (issue #8).
# Local builds have no RepoDigest — use RootFS layer diffs, then .Id.
_IMAGE_KEY_FMT='{{if .RepoDigests}}{{index .RepoDigests 0}}{{else if .RootFS.Layers}}{{join .RootFS.Layers ","}}{{else}}{{.Id}}{{end}}'

parse_image_key() {
    tr -d '\r' | sed -n 's/^GLM53KEY //p' | tail -n 1
}

local_image_key() {
    docker image inspect -f "GLM53KEY ${_IMAGE_KEY_FMT}" "$IMAGE" 2>/dev/null | parse_image_key
}

worker_image_key() {
    worker_ssh "docker image inspect -f 'GLM53KEY ${_IMAGE_KEY_FMT}' '$IMAGE' 2>/dev/null" | parse_image_key
}

images_match() {
    [ -n "${1:-}" ] && [ -n "${2:-}" ] && [ "$1" = "$2" ]
}

image_platform() {
    if [ -n "${IMAGE_PLATFORM:-}" ]; then
        printf '%s' "$IMAGE_PLATFORM"
        return
    fi
    local p
    p="$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$IMAGE" 2>/dev/null || true)"
    printf '%s' "${p:-linux/arm64}"
}

build_image() {
    log "building ${IMAGE} from Dockerfile (log: $LOGDIR/build-sm121.log) ..."
    docker build -t "$IMAGE" "$SCRIPT_DIR" \
        >"$LOGDIR/build-sm121.log" 2>&1 \
        || { tail -n 40 "$LOGDIR/build-sm121.log" >&2; die "docker build of $IMAGE failed"; }
}

pull_image() {
    login_ghcr_if_token
    log "pulling ${IMAGE} ..."
    docker pull "$IMAGE" && return 0
    die "docker pull ${IMAGE} failed.
  :exl3 is a public GHCR package — check network / disk.
  If you still get 401/403: echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
  Or set GHCR_TOKEN + GHCR_USER in .env. Overlay rebuild: BUILD=1 ./start.sh"
}

pull_image_on_worker() {
    login_ghcr_if_token_worker
    log "pulling ${IMAGE} on worker ..."
    worker_ssh "docker pull '$IMAGE'"
}

ship_image_to_worker() {
    local platform
    platform="$(image_platform)"
    log "shipping ${IMAGE} (${platform}) to worker via docker save | ssh docker load ..."
    # A multi-arch OCI index references blobs docker save does not pack
    # (only the native platform is local). docker load then dies with:
    #   open /var/lib/docker/tmp/docker-import-*/blobs/sha256/<id>: no such file
    # (issue #8). --platform emits a complete single-manifest tar.
    if docker save --platform "$platform" "$IMAGE" | worker_ssh docker load; then
        return 0
    fi
    warn "docker save --platform ${platform} failed — retrying without --platform"
    docker save "$IMAGE" | worker_ssh docker load
}

ensure_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 worker_ok=0 head_key="" worker_key=""
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        head_ok=1
        head_key="$(local_image_key)"
    fi
    if worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
        worker_key="$(worker_image_key)"
        if images_match "$head_key" "$worker_key"; then
            worker_ok=1
        else
            worker_ok=0
            log "worker image differs (head=${head_key:-none} worker=${worker_key:-none}) — will refresh worker"
        fi
    fi
    local skip_pull="${SKIP_PULL:-0}"
    [ "${PULL:-0}" = "1" ] && skip_pull=0
    if [ "${BUILD:-0}" = "1" ]; then
        case "$IMAGE" in
            *@sha256:*) die "BUILD=1 with a digest-pinned IMAGE (${IMAGE}) cannot tag a digest ref — set IMAGE to a local tag or drop the digest pin" ;;
        esac
        build_image
        head_key="$(local_image_key)"
        head_ok=1
        worker_ok=0
    elif image_from_registry && [ "$skip_pull" != "1" ]; then
        local before_key="$head_key"
        pull_image
        head_key="$(local_image_key)"
        head_ok=1
        if [ "$head_key" != "$before_key" ]; then
            log "pulled ${IMAGE} (${before_key:-missing} -> ${head_key})"
        else
            log "${IMAGE} already current"
        fi
        if images_match "$worker_key" "$head_key"; then
            worker_ok=1
        else
            worker_ok=0
        fi
    elif [ "$head_ok" = "0" ]; then
        if image_from_registry && [ "$skip_pull" = "1" ]; then
            die "SKIP_PULL=1 but ${IMAGE} is not on the head"
        fi
        build_image
        head_key="$(local_image_key)"
        head_ok=1
        worker_ok=0
    fi
    if [ "${SKIP_SHIP:-0}" = "1" ]; then
        [ "$worker_ok" = "1" ] || warn "SKIP_SHIP=1 — not copying ${IMAGE} to the worker"
    elif [ "$worker_ok" = "0" ]; then
        if image_from_registry && [ "$skip_pull" != "1" ] && [ "${BUILD:-0}" != "1" ]; then
            if pull_image_on_worker; then
                worker_key="$(worker_image_key)"
                if images_match "$head_key" "$worker_key"; then
                    worker_ok=1
                    log "worker pulled ${IMAGE} — matches head"
                else
                    warn "worker pull left a different image (head=${head_key:-none} worker=${worker_key:-none}) — shipping"
                fi
            else
                warn "worker docker pull failed — shipping over SSH (worker does not need GHCR)"
            fi
        fi
        if [ "$worker_ok" = "0" ]; then
            ship_image_to_worker
            worker_key="$(worker_image_key)"
            if images_match "$head_key" "$worker_key"; then
                worker_ok=1
            elif worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
                warn "worker has ${IMAGE} after ship but keys still differ (head=${head_key:-none} worker=${worker_key:-none}) — continuing"
                worker_ok=1
            else
                die "worker still missing ${IMAGE} after ship"
            fi
        fi
    fi
    if [ "${SKIP_OVERLAY_VERIFY:-0}" != "1" ]; then
        log "GPU EXL3 self-check on ${IMAGE} (log: $LOGDIR/overlay-verify.log) ..."
        docker run --rm --gpus all \
            -e EXL3_SELFCHECK_GPU=1 \
            --entrypoint python3 "$IMAGE" /opt/glm53/test_exl3_overlay.py \
            >"$LOGDIR/overlay-verify.log" 2>&1 \
            || { tail -n 80 "$LOGDIR/overlay-verify.log" >&2; die "EXL3 overlay GPU self-check failed"; }
        log "overlay verify OK"
    fi
    log "image ready on both nodes"
}

# ---------------------------- weight download ------------------------------
# Use an already-complete local tree (primary or upstream fallback). If the
# durable Mia-AiLab mirror is still filling / 404s, keep serving from the
# brandonmusic cache folder without a second 164 GiB pull.
adopt_complete_weights() {
    local have
    have="$(count_shards "$MODEL_PATH")"
    if [ "${have:-0}" -ge "$EXPECTED_SHARDS" ]; then
        ensure_refs_main
        log "weights already present: $MODEL_PATH ($have shards)"
        return 0
    fi
    have="$(count_shards "$FALLBACK_MODEL_PATH")"
    if [ "${have:-0}" -ge "$EXPECTED_SHARDS" ]; then
        log "primary cache incomplete — using fallback ${MODEL_FALLBACK} at $FALLBACK_MODEL_PATH ($have shards)"
        MODEL_PATH="$FALLBACK_MODEL_PATH"
        MODEL_CACHE_NAME="$MODEL_FALLBACK_CACHE_NAME"
        ensure_refs_main
        return 0
    fi
    return 1
}

hf_download_repo() {
    local repo="$1"
    local hf_bin="$2"
    shift 2
    local -a args=("$repo")
    if [ -n "${MODEL_REVISION:-}" ] && [ "$repo" = "$MODEL" ]; then
        args+=(--revision "$MODEL_REVISION")
    fi
    args+=("$@")
    HF_HOME="$HF_CACHE_DIR" "$hf_bin" download "${args[@]}"
}

download_weights() {
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping download check"; return; }
    if [ "${REFRESH_WEIGHTS:-0}" != "1" ] && adopt_complete_weights; then
        return
    fi

    local hf
    hf="$(command -v hf || command -v huggingface-cli || true)"
    [ -n "$hf" ] || die "neither 'hf' nor 'huggingface-cli' found — pip install --user -U 'huggingface_hub[cli]'"

    mkdir -p "$HF_CACHE_DIR"
    local -a hf_excl=()
    local pat
    IFS=',' read -ra _excl_pats <<< "${HF_DOWNLOAD_EXCLUDE:-runtime-results/**,src/**,runtime/src/**,scripts/**,docs/**,results/**,.materialization/**,runtime/scripts/**}"
    for pat in "${_excl_pats[@]}"; do
        [ -n "$pat" ] && hf_excl+=(--exclude "$pat")
    done

    log "downloading ${MODEL} (~164 GiB / ${EXPECTED_SHARDS} shards) into ${HF_CACHE_DIR} ..."
    hf_download_repo "$MODEL" "$hf" "${hf_excl[@]}" || warn "download of ${MODEL} failed — will try ${MODEL_FALLBACK}"
    if adopt_complete_weights; then
        return
    fi

    if [ "$MODEL_FALLBACK" != "$MODEL" ]; then
        log "falling back to ${MODEL_FALLBACK} ..."
        hf_download_repo "$MODEL_FALLBACK" "$hf" "${hf_excl[@]}" \
            || die "download of ${MODEL} and ${MODEL_FALLBACK} both failed"
    fi
    adopt_complete_weights \
        || die "download finished with $(count_shards "$MODEL_PATH") / $EXPECTED_SHARDS shards"
}

download_dflash() {
    [ "$SPEC_METHOD" = "dflash" ] || return 0
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping DFlash2 download check"; return; }
    local have
    have="$(find "$DFLASH_PATH/snapshots" -name 'model.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true)"
    if [ "${have:-0}" -ge 1 ] && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "DFlash2 already present: $DFLASH_PATH"
        ensure_dflash_refs_main
        return
    fi
    local hf
    hf="$(command -v hf || command -v huggingface-cli || true)"
    [ -n "$hf" ] || die "neither 'hf' nor 'huggingface-cli' found — pip install --user -U 'huggingface_hub[cli]'"
    mkdir -p "$HF_CACHE_DIR"
    log "downloading ${DFLASH_MODEL} (~2.3 GiB) into ${HF_CACHE_DIR} ..."
    HF_HOME="$HF_CACHE_DIR" "$hf" download "$DFLASH_MODEL"
    ensure_dflash_refs_main
    have="$(find "$DFLASH_PATH/snapshots" -name 'model.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true)"
    [ "${have:-0}" -ge 1 ] || die "DFlash2 download finished without model.safetensors"
    log "DFlash2 download complete"
}

# Head-only Hub fetch. No docker, no SSH, no worker rsync.
download_only() {
    local hf have
    hf="$(command -v hf || command -v huggingface-cli || true)"
    [ -n "$hf" ] || die "neither 'hf' nor 'huggingface-cli' found — pip install --user -U 'huggingface_hub[cli]'"
    mkdir -p "$HF_CACHE_DIR"
    local need_kb=$((180 * 1024 * 1024)) avail
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on this disk for a ~164 GiB model"

    # Explicit download: do not honor SKIP_DOWNLOAD from .env.
    SKIP_DOWNLOAD=0
    download_weights
    download_dflash

    have="$(count_shards "$MODEL_PATH")"
    log "======================================================================"
    log "head HF cache : ${HF_CACHE_DIR}"
    log "  target      : ${MODEL}  (${have} / ${EXPECTED_SHARDS} shards)"
    log "  snapshot    : ${MODEL_PATH}"
    if [ "$SPEC_METHOD" = "dflash" ]; then
        log "  DFlash2     : ${DFLASH_MODEL}"
        log "  draft cache : ${DFLASH_PATH}"
    else
        log "  DFlash2     : skipped (SPEC_METHOD=${SPEC_METHOD})"
    fi
    log "worker was not touched. ./start.sh will rsync on launch unless SKIP_SYNC=1."
    log "======================================================================"
}

# ------------------------------ weight sync --------------------------------
sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not syncing to worker"; return; }
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    log "syncing weights to worker (first run moves ~164 GiB over the p2p link) ..."
    worker_ssh "mkdir -p '$WORKER_CACHE_DIR/hub'"
    rsync -a --partial --info=progress2 \
        "$MODEL_PATH/" "${WORKER_SSH}:${WORKER_CACHE_DIR}/hub/${MODEL_CACHE_NAME}/"
    if [ "$SPEC_METHOD" = "dflash" ]; then
        [ -d "$DFLASH_PATH" ] || die "DFlash2 weights missing at $DFLASH_PATH"
        log "syncing DFlash2 draft to worker ..."
        rsync -a --partial --info=progress2 \
            "$DFLASH_PATH/" "${WORKER_SSH}:${WORKER_CACHE_DIR}/hub/${DFLASH_CACHE_NAME}/"
    fi
    log "worker weights in sync"
}

# ------------------------ inner container scripts --------------------------
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-head] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 0
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${KV_CACHE_MEMORY:-}" ] && ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
# R1 bundle scheduler flags — emitted directly, OUTSIDE EXTRA_ARGS, so an arm
# appending to EXTRA_ARGS can never double-add or shadow them.
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
if [ "${ASYNC_SCHEDULING:-0}" = "0" ]; then
    ARGS+=(--no-async-scheduling)
elif [ "${ASYNC_SCHEDULING:-0}" = "1" ]; then
    ARGS+=(--async-scheduling)
fi
if [ "${SPEC_METHOD:-mtp}" = "dflash" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
spec={"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}
tp=os.environ.get("DFLASH_DRAFT_TP","").strip()
if tp:
    spec["draft_tensor_parallel_size"]=int(tp)
# D5 dynamic speculative decoding: per-batch-size K table (vLLM PR #32374).
# Launcher validated the format (dsd_validate); empty = static k, unchanged.
dsd=os.environ.get("DSD_TABLE","").strip()
if dsd:
    spec["num_speculative_tokens_per_batch_size"]=[[int(x) for x in e.split(":")] for e in dsd.split(",")]
print(json.dumps(spec,separators=(",",":")))')")
elif [ "${SPEC_METHOD:-mtp}" = "none" ]; then
    :
elif [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
# D4: served thinking default (GLM53_DEFAULT_THINKING 1=on 0=off). Clients
# can still override per request via chat_template_kwargs.
if [ "${GLM53_DEFAULT_THINKING:-1}" = "1" ]; then
    ARGS+=(--default-chat-template-kwargs '{"enable_thinking": true}')
else
    ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} skip-mm-profiling=${SKIP_MM_PROFILING:-1} chat-template=${CHAT_TEMPLATE:-}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi
say "bundle: mnbt=${MAX_NUM_BATCHED_TOKENS:-} lpt=${LONG_PREFILL_TOKEN_THRESHOLD:-} async=${ASYNC_SCHEDULING:-0} retention=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-unset} finfer-ws=${FLASHINFER_WORKSPACE_BASE:-} pin=${KV_CACHE_MEMORY:-}"

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
if [ -f /opt/glm53/patch_glm_video_placeholders.py ]; then
    python3 /opt/glm53/patch_glm_video_placeholders.py
fi
if [ -f /opt/glm53/patch_suppress_stops_in_reasoning.py ]; then
    python3 /opt/glm53/patch_suppress_stops_in_reasoning.py
fi
if [ -f /opt/glm53/patch_scheduler_decode_floor.py ]; then
    python3 /opt/glm53/patch_scheduler_decode_floor.py
fi
if [ -f /opt/glm53/patch_glm5_drafter_group.py ]; then
    python3 /opt/glm53/patch_glm5_drafter_group.py
fi
if [ -f /opt/glm53/patch_hybrid_prefix_hit.py ]; then
    python3 /opt/glm53/patch_hybrid_prefix_hit.py
fi
if [ -f /opt/glm53/patch_xgrammar_termination.py ]; then
    python3 /opt/glm53/patch_xgrammar_termination.py
fi
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF

    cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-worker] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 1
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --headless
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${KV_CACHE_MEMORY:-}" ] && ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
# R1 bundle scheduler flags — emitted directly, OUTSIDE EXTRA_ARGS, so an arm
# appending to EXTRA_ARGS can never double-add or shadow them.
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
if [ "${ASYNC_SCHEDULING:-0}" = "0" ]; then
    ARGS+=(--no-async-scheduling)
elif [ "${ASYNC_SCHEDULING:-0}" = "1" ]; then
    ARGS+=(--async-scheduling)
fi
if [ "${SPEC_METHOD:-mtp}" = "dflash" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
spec={"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}
tp=os.environ.get("DFLASH_DRAFT_TP","").strip()
if tp:
    spec["draft_tensor_parallel_size"]=int(tp)
# D5 dynamic speculative decoding: per-batch-size K table (vLLM PR #32374).
# Launcher validated the format (dsd_validate); empty = static k, unchanged.
dsd=os.environ.get("DSD_TABLE","").strip()
if dsd:
    spec["num_speculative_tokens_per_batch_size"]=[[int(x) for x in e.split(":")] for e in dsd.split(",")]
print(json.dumps(spec,separators=(",",":")))')")
elif [ "${SPEC_METHOD:-mtp}" = "none" ]; then
    :
elif [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
# D4: served thinking default (GLM53_DEFAULT_THINKING 1=on 0=off). Clients
# can still override per request via chat_template_kwargs.
if [ "${GLM53_DEFAULT_THINKING:-1}" = "1" ]; then
    ARGS+=(--default-chat-template-kwargs '{"enable_thinking": true}')
else
    ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi
say "bundle: mnbt=${MAX_NUM_BATCHED_TOKENS:-} lpt=${LONG_PREFILL_TOKEN_THRESHOLD:-} async=${ASYNC_SCHEDULING:-0} retention=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-unset} finfer-ws=${FLASHINFER_WORKSPACE_BASE:-} pin=${KV_CACHE_MEMORY:-}"

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
if [ -f /opt/glm53/patch_glm_video_placeholders.py ]; then
    python3 /opt/glm53/patch_glm_video_placeholders.py
fi
if [ -f /opt/glm53/patch_suppress_stops_in_reasoning.py ]; then
    python3 /opt/glm53/patch_suppress_stops_in_reasoning.py
fi
if [ -f /opt/glm53/patch_scheduler_decode_floor.py ]; then
    python3 /opt/glm53/patch_scheduler_decode_floor.py
fi
if [ -f /opt/glm53/patch_glm5_drafter_group.py ]; then
    python3 /opt/glm53/patch_glm5_drafter_group.py
fi
if [ -f /opt/glm53/patch_hybrid_prefix_hit.py ]; then
    python3 /opt/glm53/patch_hybrid_prefix_hit.py
fi
if [ -f /opt/glm53/patch_xgrammar_termination.py ]; then
    python3 /opt/glm53/patch_xgrammar_termination.py
fi
say "joining TP2 at ${HEAD_IP}:${MASTER_PORT} as rank 1"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}

# ------------------------------- launch ------------------------------------
mem_ritual() {  # $1 = label, $2 = "remote" for the worker node
  local label="$1" where="${2:-local}"
  if [ "${SKIP_MEM_RITUAL:-0}" = "1" ]; then
    log "$label: memory ritual skipped (SKIP_MEM_RITUAL=1)"
    return 0
  fi
  log "$label: GB10 memory ritual (drop_caches, compact, swappiness=0, swap cycle) ..."
  local cmd='sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || echo "WARN: drop_caches failed (sudo -n?)"
    echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || echo "WARN: compact_memory failed"
    sysctl -w vm.swappiness=0 >/dev/null 2>&1 || echo "WARN: swappiness=0 failed"
    if ! grep -qs "^vm.swappiness=0" /etc/sysctl.d/90-glm53.conf 2>/dev/null; then
      echo "vm.swappiness=0" | sudo -n tee /etc/sysctl.d/90-glm53.conf >/dev/null 2>&1 || echo "WARN: could not persist /etc/sysctl.d/90-glm53.conf (reboot reverts swappiness!)"
    fi
    if command -v swapon >/dev/null 2>&1 && [ -n "$(swapon --show 2>/dev/null)" ]; then
      sudo -n swapoff -a >/dev/null 2>&1 && sudo -n swapon -a >/dev/null 2>&1 || echo "WARN: swap cycle failed"
    fi
    grep -E "MemFree|MemAvailable" /proc/meminfo | head -2'
  local out
  if [ "$where" = "remote" ]; then
    out=$(worker_ssh "$cmd" 2>&1) || out="$out
WARN: worker_ssh ritual failed"
  else
    out=$(bash -c "$cmd" 2>&1)
  fi
  log "$label ritual: $(echo "$out" | tr '\n' ' ' | tr -s ' ')"
}

launch_cache_flusher() {  # $1 = "remote" or "local"
  [ "${CACHE_FLUSHER:-0}" = "1" ] || return 0
  mkdir -p "$LOGDIR"
  # R1 2026-08-30: the sidecar runs until the fleet is serving-stable and its
  # flushes are LOGGED (the frozen pinned boot's sidecar was invisible and its
  # fixed cap ended inside the danger window).
  if [ "${1:-local}" = "remote" ]; then
    scp -q -o BatchMode=yes "$SCRIPT_DIR/cache_flusher.sh" "${WORKER_SSH}:/tmp/cache_flusher.sh"  # was never shipped — worker sidecar silently failed before
    worker_ssh "GLM53_HEALTH_URL='http://${HEAD_IP}:${PORT}/health' GLM53_FLUSHER_LOG='/tmp/cache_flusher-worker.log' nohup bash /tmp/cache_flusher.sh >/dev/null 2>&1 & echo sidecar up"
  else
    GLM53_HEALTH_URL="http://127.0.0.1:${PORT}/health" \
    GLM53_FLUSHER_LOG="$LOGDIR/cache_flusher.log" \
    nohup bash "$SCRIPT_DIR/cache_flusher.sh" >/dev/null 2>&1 &
    echo "sidecar up (pid $!, log: $LOGDIR/cache_flusher.log)"
  fi
}

ensure_worker_nfs_export() {  # D4: containerized NFS export of the worker HF cache
  if worker_ssh "docker ps --format '{{.Names}}' | grep -q '^glm53-nfs\$'"; then
    log "worker NFS export already up"
    return 0
  fi
  log "starting worker NFS export of ${WORKER_CACHE_DIR} on :${NFS_PORT} ..."
  worker_ssh "docker rm -f glm53-nfs 2>/dev/null; docker run -d --name glm53-nfs --restart unless-stopped --privileged -p ${NFS_PORT}:2049 -v '${WORKER_CACHE_DIR}:/export:ro' -v /lib/modules:/lib/modules:ro -e NFS_EXPORT_0='/export *(ro,no_subtree_check,fsid=0,insecure)' erichough/nfs-server" \
    || die "NFS export container failed on the worker (WEIGHTS_MODE=nfs needs docker pull erichough/nfs-server on the worker)"
  log "worker NFS export ready (${WORKER_IP}:${NFS_PORT})"
}

launch_cluster() {
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || true

    mkdir -p "$CACHE_ROOT" "$TRITON_HOST_CACHE" "$TILELANG_HOST_CACHE"
    worker_ssh "mkdir -p '$WORKER_VLLM_CACHE' '$WORKER_TRITON_CACHE' '$WORKER_TILELANG_CACHE'"
    # A launch is a deliberate start — release the watchdog pause written by
    # the last ./start.sh stop.
    rm -f "$LOGDIR/.watchdog-paused"
    scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${WORKER_SSH}:/tmp/${CONTAINER_WORKER}.sh"
    [ -f "$CHAT_TEMPLATE_HOST" ] || die "missing chat template: $CHAT_TEMPLATE_HOST"
    scp -q -o BatchMode=yes "$CHAT_TEMPLATE_HOST" "${WORKER_SSH}:/tmp/glm53-chat_template.jinja"
    [ -f "$VIDEO_PATCH_HOST" ] || die "missing $VIDEO_PATCH_HOST"
    scp -q -o BatchMode=yes "$VIDEO_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_glm_video_placeholders.py"
    [ -f "$STOP_PATCH_HOST" ] || die "missing $STOP_PATCH_HOST"
    scp -q -o BatchMode=yes "$STOP_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_suppress_stops_in_reasoning.py"
    [ -f "$SCHED_PATCH_HOST" ] || die "missing $SCHED_PATCH_HOST"
    scp -q -o BatchMode=yes "$SCHED_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_scheduler_decode_floor.py"
    [ -f "$DRAFTER_PATCH_HOST" ] || die "missing $DRAFTER_PATCH_HOST"
    scp -q -o BatchMode=yes "$DRAFTER_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_glm5_drafter_group.py"
    [ -f "$APC_PATCH_HOST" ] || die "missing $APC_PATCH_HOST"
    scp -q -o BatchMode=yes "$APC_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_hybrid_prefix_hit.py"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "missing $XGRAMMAR_PATCH_HOST"
    scp -q -o BatchMode=yes "$XGRAMMAR_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_xgrammar_termination.py"

    local -a nccl_common=(
        -e NCCL_IB_DISABLE=0
        -e NCCL_IB_ROCE_VERSION_NUM=2
        -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
        -e NCCL_NET=IB
        -e NCCL_NET_PLUGIN=none
        -e NCCL_NVLS_ENABLE=0
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_MERGE_NICS=0
        -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
        -e NCCL_IGNORE_CPU_AFFINITY=1
        -e "NCCL_DEBUG=$NCCL_DEBUG"
        -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1
        -e HF_HOME=/root/.cache/huggingface
        -e VLLM_CACHE_ROOT=/root/.cache/vllm
        -e "GLM53_SUPPRESS_STOPS_IN_REASONING=$GLM53_SUPPRESS_STOPS_IN_REASONING"
        -e "GLM53_MIXED_PREFILL_CHUNK=$GLM53_MIXED_PREFILL_CHUNK"
        -e "GLM53_DEFAULT_THINKING=$GLM53_DEFAULT_THINKING"
        -e "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
        -e "TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
        -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
        -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        -e "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        -e FLASHINFER_DISABLE_VERSION_CHECK=1
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        -e "VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT"
        # py-cpuinfo JSON-parses empty output on Grace/aarch64; the usage
        # thread then dumps JSONDecodeError. Stats are off on this private kit.
        -e VLLM_NO_USAGE_STATS=1
        -e DO_NOT_TRACK=1
        -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$CG_ESTIMATE"
    )
    local worker_nccl="" e
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        worker_nccl+=" -e $e"
    done

    # R1 bundle envs. Conditional on non-empty: the fork's env parser int()s
    # VLLM_PREFIX_CACHE_RETENTION_INTERVAL, so forwarding an EMPTY string would
    # crash boot — empty means "unset" (dense checkpoints / FlashInfer default).
    local -a bundle_env=()
    [ -n "${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-}" ] && \
        bundle_env+=(-e "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=$VLLM_PREFIX_CACHE_RETENTION_INTERVAL")
    [ -n "${FLASHINFER_WORKSPACE_BASE:-}" ] && \
        bundle_env+=(-e "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE")
    # NCCL_IB_QPS_PER_CONNECTION (P1-1 arm, RESEARCH-PERF-NEXT): forward only
    # when the operator sets it — unset = NCCL default (1 QPC per connection).
    [ -n "${NCCL_IB_QPS_PER_CONNECTION:-}" ] && \
        bundle_env+=(-e "NCCL_IB_QPS_PER_CONNECTION=$NCCL_IB_QPS_PER_CONNECTION")
    local worker_bundle="" be
    for be in "${bundle_env[@]}"; do
        [ "$be" = "-e" ] && continue
        worker_bundle+=" -e $be"
    done

    local -a head_preload=() worker_preload=""
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
        if worker_ssh "test -f '$WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME'"; then
            worker_preload="-v '$WORKER_NCCL_HOST_DIR:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
            log "worker: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "worker: $WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    local serve_env=""
    local v
    for v in SERVED_MODEL_NAME PORT TP NNODES HEAD_IP MASTER_PORT QUANTIZATION \
             MAX_MODEL_LEN GPU_MEM_UTIL KV_CACHE_MEMORY MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
             KV_CACHE_DTYPE MTP_TOKENS SPEC_METHOD DFLASH_TOKENS DFLASH_MODEL_DIR \
             DFLASH_DRAFT_TP DSD_TABLE \
             LONG_PREFILL_TOKEN_THRESHOLD ASYNC_SCHEDULING \
             LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING \
             LIMIT_MM CHAT_TEMPLATE ENFORCE_EAGER EXL3_FUSED_MOE MODEL_DIR EXTRA_ARGS; do
        serve_env+=" -e $v='${!v:-}'"
    done

    log "starting worker on ${WORKER_SSH} (NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
    mem_ritual "worker" "remote"
    launch_cache_flusher remote
    worker_ssh "docker run -d --name '$CONTAINER_WORKER' \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v '$WORKER_CACHE_DIR:/root/.cache/huggingface' \
        -v '$WORKER_VLLM_CACHE:/root/.cache/vllm' \
        -v '$WORKER_TRITON_CACHE:/root/.triton/cache' \
        -v '$WORKER_TILELANG_CACHE:/root/.tilelang/cache' \
        -v '/tmp/${CONTAINER_WORKER}.sh:/start.sh:ro' \
        -v '/tmp/glm53-chat_template.jinja:${CHAT_TEMPLATE}:ro' \
        -v '/tmp/patch_glm_video_placeholders.py:/opt/glm53/patch_glm_video_placeholders.py:ro' \
        -v '/tmp/patch_suppress_stops_in_reasoning.py:/opt/glm53/patch_suppress_stops_in_reasoning.py:ro' \
        -v '/tmp/patch_scheduler_decode_floor.py:/opt/glm53/patch_scheduler_decode_floor.py:ro' \
        -v '/tmp/patch_glm5_drafter_group.py:/opt/glm53/patch_glm5_drafter_group.py:ro' \
        -v '/tmp/patch_hybrid_prefix_hit.py:/opt/glm53/patch_hybrid_prefix_hit.py:ro' \
        -v '/tmp/patch_xgrammar_termination.py:/opt/glm53/patch_xgrammar_termination.py:ro' \
        ${worker_preload} \
        ${worker_nccl} \
        ${worker_bundle} \
        -e NCCL_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e GLOO_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e NCCL_IB_HCA='$WORKER_CX7_IB' \
        -e VLLM_HOST_IP='$WORKER_IP' \
        ${serve_env} \
        --entrypoint bash '$IMAGE' /start.sh" >/dev/null

    log "starting head (vLLM API :${PORT}; NCCL if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
    mem_ritual "head"
    launch_cache_flusher local
    local head_hf_mounts=(-v "$HF_CACHE_DIR:/root/.cache/huggingface")
    if [ "$WEIGHTS_MODE" = "nfs" ]; then
        ensure_worker_nfs_export
        docker volume create --driver local \
            --opt type=nfs --opt "o=addr=${WORKER_IP},ro,vers=4.2,rsize=1048576,port=${NFS_PORT}" \
            --opt device=:/ "$NFS_VOL_NAME" >/dev/null 2>&1 || true
        head_hf_mounts=(-v "$NFS_VOL_NAME:/root/.cache/huggingface:ro")
        log "head weights via NFS volume ${NFS_VOL_NAME} (${WORKER_IP}:${NFS_PORT})"
    fi
    docker run -d --name "$CONTAINER_HEAD" \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        "${head_hf_mounts[@]}" \
        -v "$CACHE_ROOT:/root/.cache/vllm" \
        -v "$TRITON_HOST_CACHE:/root/.triton/cache" \
        -v "$TILELANG_HOST_CACHE:/root/.tilelang/cache" \
        -v "$HEAD_SCRIPT:/start.sh:ro" \
        -v "$CHAT_TEMPLATE_HOST:$CHAT_TEMPLATE:ro" \
        -v "$VIDEO_PATCH_HOST:/opt/glm53/patch_glm_video_placeholders.py:ro" \
        -v "$STOP_PATCH_HOST:/opt/glm53/patch_suppress_stops_in_reasoning.py:ro" \
        -v "$SCHED_PATCH_HOST:/opt/glm53/patch_scheduler_decode_floor.py:ro" \
        -v "$DRAFTER_PATCH_HOST:/opt/glm53/patch_glm5_drafter_group.py:ro" \
        -v "$APC_PATCH_HOST:/opt/glm53/patch_hybrid_prefix_hit.py:ro" \
        -v "$XGRAMMAR_PATCH_HOST:/opt/glm53/patch_xgrammar_termination.py:ro" \
        "${head_preload[@]}" \
        "${nccl_common[@]}" \
        -e NCCL_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e GLOO_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e NCCL_IB_HCA="$HEAD_CX7_IB" \
        -e VLLM_HOST_IP="$HEAD_IP" \
        -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        -e PORT="$PORT" -e TP="$TP" -e NNODES="$NNODES" \
        -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="$MASTER_PORT" \
        -e QUANTIZATION="$QUANTIZATION" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        -e KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e MTP_TOKENS="$MTP_TOKENS" \
        -e SPEC_METHOD="$SPEC_METHOD" \
        -e DFLASH_TOKENS="${DFLASH_TOKENS:-7}" \
        -e DFLASH_MODEL_DIR="${DFLASH_MODEL_DIR:-}" \
        -e DFLASH_DRAFT_TP="${DFLASH_DRAFT_TP:-}" \
        -e DSD_TABLE="${DSD_TABLE:-}" \
        -e LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}" \
        -e ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}" \
        "${bundle_env[@]}" \
        -e LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
        -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e LIMIT_MM="$LIMIT_MM" \
        -e CHAT_TEMPLATE="$CHAT_TEMPLATE" \
        -e ENFORCE_EAGER="$ENFORCE_EAGER" \
        -e EXL3_FUSED_MOE="$EXL3_FUSED_MOE" \
        -e MODEL_DIR="$MODEL_DIR" \
        -e EXTRA_ARGS="${EXTRA_ARGS:-}" \
        --entrypoint bash "$IMAGE" /start.sh >/dev/null

    log "containers up — head=${CONTAINER_HEAD}, worker=${CONTAINER_WORKER}"
}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (weight load + warmup on a 320B MoE is slow; timeout ${READY_TIMEOUT}s) ..."
    log "streaming head logs live — Ctrl-C detaches, the server keeps running"

    local logpid=""
    _stop_logtail() {
        [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true
        wait "$logpid" 2>/dev/null || true
        logpid=""
    }
    trap '_stop_logtail; warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT
    docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 &
    logpid=$!

    local elapsed=0 healthy=0 exited=0 dead_side="" worker_fail=0 head_fail=0
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        # Keep the inspect result out of a grep -q pipeline. With pipefail,
        # grep can close early and make a running container look dead.
        # Same 3-strike window as the worker: one transient docker miss must
        # not abort a multi-minute weight load.
        if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null || true)" = "true" ]; then
            head_fail=0
        else
            head_fail=$((head_fail + 1))
            if [ "$head_fail" -ge 3 ]; then
                if docker inspect "$CONTAINER_HEAD" >/dev/null 2>&1; then
                    log "head container not running during startup (3 consecutive checks)"
                else
                    log "head container missing during startup (removed by concurrent stop/restart?)"
                fi
                exited=1; dead_side="head"; break
            fi
        fi
        # A dead worker rank can never make the head healthy — fail fast with
        # the log dump instead of polling for the full READY_TIMEOUT (issue
        # #22, item 4). Transient ssh/docker hiccups are tolerated; only
        # three consecutive non-running answers (~30 s) count as a dead
        # worker.
        if [ "$(worker_ssh "docker inspect -f '{{.State.Running}}' '$CONTAINER_WORKER' 2>/dev/null" || true)" = "true" ]; then
            worker_fail=0
        else
            worker_fail=$((worker_fail + 1))
            if [ "$worker_fail" -ge 3 ]; then
                log "worker container '$CONTAINER_WORKER' not running on ${WORKER_SSH} (3 consecutive checks)"
                exited=1; dead_side="worker"; break
            fi
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    _stop_logtail
    trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT

    if [ "$healthy" = "1" ]; then
        log "health check passed after ${elapsed}s — server is up"
    elif [ "$exited" = "1" ]; then
        warn "${dead_side:-head} container exited/stopped after ${elapsed}s"
    else
        warn "timed out after ${elapsed}s without becoming healthy"
    fi
    [ "$healthy" = "1" ]
}

post_ready_warmup() {
    if [ "${GLM53_BOOT_SHAPE_WARMUP:-1}" = "0" ]; then
        log "boot shape warmup skipped (GLM53_BOOT_SHAPE_WARMUP=0)"
        return 0
    fi
    [ -f "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" ] \
        || { warn "boot-shape-warmup.sh missing — skipping"; return 0; }
    log "post-ready DFlash2/sampler warmup (nonfatal; timeout ${GLM53_WARMUP_REQ_TIMEOUT}s/req) ..."
    GLM53_WARMUP_MAX_CONCURRENCY="$MAX_NUM_SEQS" \
    GLM53_WARMUP_REQ_TIMEOUT="$GLM53_WARMUP_REQ_TIMEOUT" \
    GLM53_WARMUP_DFLASH_K="${DFLASH_TOKENS:-7}" \
    GLM53_WARMUP_TRITON_CACHE_DIR="$TRITON_HOST_CACHE" \
    GLM53_WARMUP_BEARER="${VLLM_API_KEY:-}" \
        bash "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" \
            "http://127.0.0.1:${PORT}" "$SERVED_MODEL_NAME" \
        || warn "boot shape warmup incomplete — uncovered shapes may JIT mid-serve on TP=2"
}

collect_failure_logs() {
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head.log" 2>&1 || true
    worker_ssh "docker logs '$CONTAINER_WORKER' 2>&1" >"$LOGDIR/worker.log" 2>&1 || true
}

on_ready() {
    log "======================================================================"
    log "GLM-5.3-Flash EXL3 is UP (TP=${TP}, nnodes=${NNODES})"
    log "  endpoints  : http://127.0.0.1:${PORT}/v1   (LAN: ${HEAD_IP}:${PORT})"
    log "  model name : ${SERVED_MODEL_NAME}"
    log "  weights    : ${MODEL}  quant=${QUANTIZATION}  kv=${KV_CACHE_DTYPE}"
    local vision=on
    [ "${LANGUAGE_MODEL_ONLY}" = "1" ] && vision=off
    local spec="MTP k=${MTP_TOKENS}"
    [ "$SPEC_METHOD" = "dflash" ] && spec="DFlash2 k=${DFLASH_TOKENS} (${DFLASH_MODEL})"
    [ "$SPEC_METHOD" = "none" ] && spec=off
    if [ -n "${DSD_TABLE:-}" ]; then
        spec="${spec} + DSD[${DSD_TABLE}]"
    fi
    log "  features   : tools=glm47+auto, reasoning=glm45, spec=${spec}, vision=${vision}"
    log "  prefill    : max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS} (D1/R1 spec-decode step budget — verify the boot log shows this number)"
    log "  bundle     : lpt=${LONG_PREFILL_TOKEN_THRESHOLD:-off} async=${ASYNC_SCHEDULING:-0} retention=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-unset} flashinfer-ws=${FLASHINFER_WORKSPACE_BASE:-} qps=${NCCL_IB_QPS_PER_CONNECTION:-nccl-default}"
    log "  mem        : head MemFree=$(memfree_gib local) GiB / worker MemFree=$(memfree_gib remote) GiB (post-init cache drain applied before warmup)"
    log "  image      : ${IMAGE}"
    log "  thinking   : served default=$( [ "${GLM53_DEFAULT_THINKING:-1}" = "1" ] && echo on || echo off ) (GLM53_DEFAULT_THINKING; clients can override)"
    log "  weights    : mode=${WEIGHTS_MODE} (nfs => head reads ${WORKER_IP}:${NFS_PORT})"
    log "  quick test :"
    log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
    log "      -H 'Content-Type: application/json' \\"
    log "      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"hello!\"}]}'"
    log "  manage     : ./start.sh status | ./start.sh logs | ./start.sh logs worker | ./start.sh stop"
    log "======================================================================"
    # D5 boot verification (fail-noisy, non-fatal): the only bad failure mode
    # is the DSD->PIECEWISE cudagraph downgrade, which means the V2 model
    # runner assumption broke and the arm must not be benchmarked.
    if [ -n "${DSD_TABLE:-}" ]; then
        if docker logs "$CONTAINER_HEAD" 2>&1 | grep -q "Overriding cudagraph_mode from"; then
            warn "DSD forced PIECEWISE cudagraphs (V2 model runner regression?) — do NOT benchmark this boot"
        else
            log "  dsd-check  : no cudagraph_mode downgrade in boot log (V2 path intact)"
        fi
        log "  dsd-receipt: run tests/verify_dsd.py (per-position acceptance must freeze at pos K during a c>=2 burst)"
    fi
    # R1 KV-pin procedure (docs/CAMPAIGN-R1.md): boot UNPINNED once, read the
    # suggested budget from the log, then pin KV_CACHE_MEMORY at (suggested −
    # margin) — never raise. vLLM logs TWO suggestions; only the FIRST
    # ("to fit into requested memory" — the profiled envelope) is defensible.
    # The SECOND ("to fully utilize") is the C4 crash class. The gate is 3
    # cold boots with a byte-identical pool line.
    local pin_line pin_to_fit pin_to_gpu
    pin_line=$(docker logs "$CONTAINER_HEAD" 2>&1 | grep -E 'to fit into requested memory' | tail -1)
    pin_to_fit=$(printf '%s' "$pin_line" | grep -oP -- '--kv-cache-memory=\K[0-9]+' | head -1)
    pin_to_gpu=$(printf '%s' "$pin_line" | grep -oP -- 'or `--kv-cache-memory=\K[0-9]+' | head -1)
    if [ -n "${pin_to_fit:-}" ]; then
        if [ -n "${KV_CACHE_MEMORY:-}" ]; then
            log "  kv-pin     : pinned KV_CACHE_MEMORY=${KV_CACHE_MEMORY} bytes (envelope to-fit: ${pin_to_fit}; full-utilize ${pin_to_gpu:-n/a} = C4 crash class — never)"
        else
            log "  kv-pin     : unpinned — pin KV_CACHE_MEMORY at (to-fit ${pin_to_fit} − margin), NEVER at the full-utilize value ${pin_to_gpu:-n/a}"
        fi
    elif [ -n "${KV_CACHE_MEMORY:-}" ]; then
        log "  kv-pin     : pinned KV_CACHE_MEMORY=${KV_CACHE_MEMORY} bytes (no engine suggestion line found)"
    fi
    if [ "${TAIL:-0}" = "1" ]; then
        log "tailing head logs — Ctrl-C just detaches, the server keeps running"
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER_HEAD" || true
        trap 'warn "interrupted — containers keep running"; exit 130' INT
    fi
}

# ------------------------------- start -------------------------------------
start_unlocked() {
    preflight
    ensure_image
    download_weights
    download_dflash
    sync_weights
    write_inner_scripts

    MODEL_DIR="$(resolve_model_dir)"
    DFLASH_MODEL_DIR=""
    if [ "$SPEC_METHOD" = "dflash" ]; then
        DFLASH_MODEL_DIR="$(resolve_dflash_dir)"
        log "DFlash2 load path (in-container): ${DFLASH_MODEL_DIR}"
    fi
    log "model load path (in-container): ${MODEL_DIR}"
    log "config: image=${IMAGE} tp=${TP} nnodes=${NNODES} quant=${QUANTIZATION} spec=${SPEC_METHOD} mtp=${MTP_TOKENS} dflash_k=${DFLASH_TOKENS} max-len=${MAX_MODEL_LEN} gpu-util=${GPU_MEM_UTIL} kv=${KV_CACHE_DTYPE} lm-only=${LANGUAGE_MODEL_ONLY} port=${PORT}"

    launch_cluster
    if wait_for_health; then
        post_init_cache_drop
        post_ready_warmup
        on_ready
        return 0
    fi
    # Boot-lottery fallback: on a clean memory-shortfall refuse (9.xx < 9.66 GiB
    # needed for 1M), vLLM suggests the actual max length. Retry ONCE with that
    # value instead of failing the whole launch. Never raises util (0.85 cap).
    suggested_len=$(grep -oP "estimated maximum model length is \\K[0-9]+" "$LOGDIR/head.log" | tail -1)
    if [ -n "${suggested_len:-}" ] && [ "${suggested_len:-0}" -lt "${MAX_MODEL_LEN:-1000000}" ]; then
        log "KV shortfall: engine estimates max length ${suggested_len} < ${MAX_MODEL_LEN} — retrying once with MAX_MODEL_LEN=${suggested_len}"
        MAX_MODEL_LEN="$suggested_len"
        write_inner_scripts   # regenerate both rank scripts with the new max len
        launch_cluster
        if wait_for_health; then
            post_init_cache_drop
            post_ready_warmup
            on_ready
            return
        fi
    fi
    collect_failure_logs
    echo "---- last 60 lines of head log ($LOGDIR/head.log) ----"
    tail -n 60 "$LOGDIR/head.log" || true
    echo "---- last 40 lines of worker log ($LOGDIR/worker.log) ----"
    tail -n 40 "$LOGDIR/worker.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}

start() {
    with_cluster_lock
    start_unlocked
}

stop_containers() {
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    log "stopping worker container on ${WORKER_SSH} ..."
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 \
        || log "  (no worker container was running)"
    # R1 watchdog guard: a deliberate stop must not be resurrected by the
    # watchdog's crash/wedge recovery. fleet_watchdog.sh honors this sentinel;
    # any launch clears it.
    mkdir -p "$LOGDIR"
    : > "$LOGDIR/.watchdog-paused"
    log "stopped (watchdog paused via $LOGDIR/.watchdog-paused until the next launch)"
}

# ------------------------------- stop --------------------------------------
stop() {
    steal_cluster_lock_for_stop
    stop_containers
    rm -f "$CLUSTER_LOCK_PID"
}

# ------------------------------ status -------------------------------------
status() {
    log "head (${CONTAINER_HEAD} on $(hostname)):"
    docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "  API: not responding"
    fi
    log "worker (${CONTAINER_WORKER} on ${WORKER_SSH}):"
    worker_ssh "docker ps -a --filter name=${CONTAINER_WORKER} --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
        || log "  (worker unreachable)"
}

# ------------------------------- logs --------------------------------------
logs() {
    case "${1:-head}" in
        worker)
            log "following worker container logs on ${WORKER_SSH} ..."
            trap '' INT
            worker_ssh "docker logs -f --tail 100 '$CONTAINER_WORKER'" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
        head|*)
            log "following head logs (driver + API server) ..."
            trap '' INT
            docker logs -f --tail 100 "$CONTAINER_HEAD" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
    esac
}

# ------------------------------- main --------------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        start|restart) validate_numeric_config ;;
    esac
    case "$cmd" in
        stop)     banner stop.sh ;;
        download) banner download.sh ;;
        *)        banner start.sh ;;
    esac
    case "$cmd" in
        start)    shift || true; start ;;
        download) download_only ;;
        stop)     stop ;;
        restart)
            with_cluster_lock
            stop_containers
            start_unlocked
            ;;
        status)   status ;;
        logs)     shift || true; logs "$@" ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
