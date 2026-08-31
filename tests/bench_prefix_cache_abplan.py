#!/usr/bin/env python3
"""bench_prefix_cache_abplan.py — AB-PLAN prefix-cache bench for campaign R1.

Wraps local/cache-burst.py (the Reederey87-derived protocol) with the
per-arm fingerprint required by AB-PLAN rule 7 (git sha + image digest +
volatile env knobs) and writes an arm artifact:

    results/ab/<label>/cache.json     protocol results
    results/ab/<label>/cache.md       human summary

Usage:
    python3 tests/bench_prefix_cache_abplan.py --label r1-bundle --out results/ab/r1-bundle
    python3 tests/bench_prefix_cache_abplan.py --label baseline --sessions 4 \
        --tokens 60000 --rounds 3 --solo-tokens 110000

Gates (evaluated into the summary, exit 1 on failure):
    burst rounds 2+ mean hit >= 0.90   (4 x 60k x 3 rounds)
    solo replay hit >= 0.93            (110k)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BURST = REPO / "local" / "cache-burst.py"


def fingerprint() -> dict:
    def sh(cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, shell=True, capture_output=True, text=True, cwd=REPO
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    env: dict[str, str] = {}
    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    knobs = [
        "MAX_NUM_BATCHED_TOKENS",
        "LONG_PREFILL_TOKEN_THRESHOLD",
        "ASYNC_SCHEDULING",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
        "FLASHINFER_WORKSPACE_BASE",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "GPU_MEM_UTIL",
        "KV_CACHE_MEMORY",
        "IMAGE",
    ]
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sh("git rev-parse HEAD"),
        "image_repo_digest": sh(
            "docker image inspect -f '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "
            f"{env.get('IMAGE', '')} 2>/dev/null"
        ),
        "env": {k: env.get(k, "") for k in knobs},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=60000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--solo-tokens", type=int, default=110000)
    ap.add_argument("--round-hit-min", type=float, default=0.90)
    ap.add_argument("--solo-hit-min", type=float, default=0.93)
    args = ap.parse_args()

    outdir = Path(args.out) if args.out else REPO / "results" / "ab" / args.label
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(BURST), "--protocol", "both", "--sessions", str(args.sessions),
           "--tokens", str(args.tokens), "--rounds", str(args.rounds),
           "--round-hit-min", str(args.round_hit_min),
           "--solo-hit-min", str(args.solo_hit_min)]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    # First the burst protocol at its own token size, then solo at its size.
    burst_cmd = cmd + ["--protocol", "burst", "--out", str(outdir / "cache-burst.json")]
    solo_cmd = [c for c in cmd] + ["--protocol", "solo", "--tokens", str(args.solo_tokens),
                                   "--out", str(outdir / "cache-solo.json")]
    print("+ " + " ".join(burst_cmd), flush=True)
    rb = subprocess.run(burst_cmd, capture_output=True, text=True)
    print(rb.stdout[-2000:], rb.stderr[-500:], sep="\n")
    print("+ " + " ".join(solo_cmd), flush=True)
    rs = subprocess.run(solo_cmd, capture_output=True, text=True)
    print(rs.stdout[-2000:], rs.stderr[-500:], sep="\n")

    burst = json.loads((outdir / "cache-burst.json").read_text())["burst"]
    solo = json.loads((outdir / "cache-solo.json").read_text())["solo"]

    gates = {
        "burst_rounds2plus_hit>=0.90": bool(
            burst["gate_hit_mean_min"] >= args.round_hit_min and burst["gate_pass"]
        ),
        "solo_replay_hit>=0.93": bool(solo["gate_hit_min"] >= args.solo_hit_min and solo["gate_pass"]),
    }
    art = {
        "label": args.label,
        "fingerprint": fingerprint(),
        "burst": burst,
        "solo": solo,
        "gates": gates,
        "pass": all(gates.values()),
    }
    (outdir / "cache.json").write_text(json.dumps(art, indent=2) + "\n")
    lines = [
        f"# cache bench — {args.label}",
        "",
        f"- burst {burst['sessions']}x{burst['target_tokens']} x{len(burst['rounds'])} rounds: "
        f"rounds {burst['gate_rounds_from']}+ hit_mean_min={burst['gate_hit_mean_min']:.3f} "
        f"({'PASS' if gates['burst_rounds2plus_hit>=0.90'] else 'FAIL'} @ >= {args.round_hit_min})",
        f"- solo {solo['target_tokens']} replay: hit={solo['gate_hit_min']:.3f} "
        f"({'PASS' if gates['solo_replay_hit>=0.93'] else 'FAIL'} @ >= {args.solo_hit_min})",
        f"- git: {art['fingerprint']['git_sha']}",
        f"- image: {art['fingerprint']['image_repo_digest']}",
        "",
    ]
    (outdir / "cache.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if art["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
