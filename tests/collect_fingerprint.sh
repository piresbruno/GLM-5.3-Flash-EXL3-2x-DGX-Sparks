#!/usr/bin/env bash
# collect_fingerprint.sh — per-arm env fingerprint (AB-PLAN.md rule 7 / Appendix B).
# Usage: ./tests/collect_fingerprint.sh > results/ab/<arm>/fingerprint.json
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ./.env 2>/dev/null || true
IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"
WORKER_SSH_TARGET="${WORKER_SSH:-${WORKER_USER:-$USER}@${WORKER_IP:-}}"

img_id() { docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo null; }
img_repo_digest() {
  docker image inspect -f '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "$IMAGE" 2>/dev/null || echo null
}
worker_img_id() {
  if [ -n "${WORKER_IP:-}" ]; then
    ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH_TARGET" \
      "docker image inspect -f '{{.Id}}' '$IMAGE' 2>/dev/null" 2>/dev/null || echo null
  else
    echo null
  fi
}
worker_repo_digest() {
  if [ -n "${WORKER_IP:-}" ]; then
    ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH_TARGET" \
      "docker image inspect -f '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' '$IMAGE' 2>/dev/null" 2>/dev/null || echo null
  else
    echo null
  fi
}

kv_pool() {  # KV pool tokens from the head container log, if reachable
  docker logs glm53-exl3-head 2>&1 | grep -oP 'GPU KV cache size: \K[0-9,]+' | head -1 || true
}

jstr() { python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$1"; }

cat <<EOF
{
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_sha": "$(git rev-parse HEAD)",
  "image": $(jstr "$IMAGE"),
  "image_id": $(jstr "$(img_id)"),
  "repo_digest": $(jstr "$(img_repo_digest)"),
  "worker_image_id": $(jstr "$(worker_img_id)"),
  "worker_repo_digest": $(jstr "$(worker_repo_digest)"),
  "_note": "rank images match if repo_digest == worker_repo_digest (same manifest). .Id may differ across docker-load paths — not a mismatch.",
  "env_fingerprint": {
    "MAX_MODEL_LEN": ${MAX_MODEL_LEN:-1000000},
    "GPU_MEM_UTIL": "${GPU_MEM_UTIL:-0.87}",
    "MAX_NUM_BATCHED_TOKENS": ${MAX_NUM_BATCHED_TOKENS:-1024},
    "DFLASH_TOKENS": ${DFLASH_TOKENS:-7},
    "GLM53_MIXED_PREFILL_CHUNK": "${GLM53_MIXED_PREFILL_CHUNK:-skip}",
    "KV_CACHE_DTYPE": "${KV_CACHE_DTYPE:-fp8}",
    "SPEC_METHOD": "${SPEC_METHOD:-dflash}",
    "LANGUAGE_MODEL_ONLY": ${LANGUAGE_MODEL_ONLY:-0},
    "ENFORCE_EAGER": ${ENFORCE_EAGER:-0}
  },
  "kv_pool_tokens": $(jstr "$(kv_pool | tr -d ',' || true)")
}
EOF
