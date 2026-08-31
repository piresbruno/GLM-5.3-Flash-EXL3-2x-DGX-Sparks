#!/usr/bin/env python3
"""cache-burst.py — multi-session prefix-cache gate (R1 bundle receipt).

Ported from the Reederey87 production kit (see NOTICE), adapted to this
repo's layout (.env PORT, GLM-5.3-Flash-EXL3 served name, :8081 default).

Hit measurement (R1 fix, 2026-08-30): this fork's OpenAI usage does NOT
populate ``prompt_tokens_details.cached_tokens`` (verified: always None), so
hits are measured from the ``vllm:prefix_cache_hits_total`` /metrics counter
delta around each request/round instead — the same channel the D1-era
retests used. Hit ratio = delta_hits / measured prompt tokens.

Two protocols, both prefill-dominated:

burst   : S concurrent sessions share one long common prefix and each ends
          with a unique question. R rounds. Rounds 2+ should hit >= the
          round gate (the R1 gate is 90% at 4 sessions x ~60k tokens x 3
          rounds) — this exercises shared-prefix junctions under the sparse
          KDA retention env (VLLM_PREFIX_CACHE_RETENTION_INTERVAL).
solo    : one session replays a long prompt twice; the replay should hit
          >= the solo gate (R1: 93% at ~110k tokens — our own D1-era
          measurement this campaign re-gates).

Exit 0 only when every requested gate passes (with --gate). Without --gate
the tool just reports and exits 0 (bench harness evaluates).

Usage:
  python3 local/cache-burst.py --protocol burst --sessions 4 --tokens 60000 \
      --rounds 3 --out results/ab/r1-xxx/cache-burst.json [--gate]
  python3 local/cache-burst.py --protocol solo --tokens 110000 --gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Read the live .env for PORT so the probe always matches the serving kit.
_PORT = "8081"
try:  # minimal KEY=VALUE parse; never fatal
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("PORT=") and not line.lstrip().startswith("#"):
            _PORT = line.split("=", 1)[1].strip()
except OSError:
    pass

BASE = os.environ.get("GLM53_BENCH_BASE", f"http://127.0.0.1:{_PORT}")
MODEL = os.environ.get("GLM53_BENCH_MODEL", "GLM-5.3-Flash-EXL3")

HAYSTACK = (
    "The logistics ledger for convoy {i} records the manifest of crates, "
    "seals, and duty stamps transferred at the mid-way depot, including the "
    "inspection notes for each seal and the counter-signed receipt for every "
    "crate that left the loading bay before dusk. "
)


HIT_METRIC = "vllm:prefix_cache_hits_total"


def _read_hits(base: str) -> float:
    """Sum of the prefix-cache hit-token counters across engines."""
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as resp:
        text = resp.read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith(f"{HIT_METRIC}{{") or line.startswith(f"{HIT_METRIC} "):
            try:
                total += float(line.rsplit("}", 1)[1].strip())
            except (ValueError, IndexError):
                continue
    return total


def _chat(base: str, model: str, content: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _hit_ratio(resp: dict) -> tuple[int, int]:
    """(cached, prompt) from the response — cached is None on this fork.

    Kept for the measured prompt size; the HIT count comes from the /metrics
    counter delta (see _read_hits)."""
    usage = resp.get("usage", {})
    detail = (usage.get("prompt_tokens_details") or {}) if isinstance(usage, dict) else {}
    cached = int(detail.get("cached_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    return cached, prompt


def _build_prompt(target_tokens: int, session_idx: int, question: str) -> str:
    # ~1 token per short word + padding factor measured from usage; the tool
    # always REPORTS the true prompt_tokens, so the operator sees the real size.
    words = max(1, int(target_tokens / 1.35))
    blocks = words // 40 + 1
    filler = " ".join(HAYSTACK.format(i=i % 97) for i in range(blocks))
    return filler + "\n" + question


def run_burst(args: argparse.Namespace) -> dict:
    questions = [
        f"[session-{i}] In one word, what color is the crate marked with seal {i}?"
        for i in range(args.sessions)
    ]
    prompts = [_build_prompt(args.tokens, i, questions[i]) for i in range(args.sessions)]
    rounds = []
    for r in range(args.rounds):
        hits_before = _read_hits(args.base_url)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.sessions) as pool:
            futs = [
                pool.submit(_chat, args.base_url, args.model, p, args.max_tokens, args.timeout)
                for p in prompts
            ]
            hits = []
            errs = []
            for f in futs:
                try:
                    _cached, prompt = _hit_ratio(f.result())
                    hits.append(prompt)
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    errs.append(str(e))
        dt = time.time() - t0
        hits_after = _read_hits(args.base_url)
        round_hits = hits_after - hits_before
        prompt_sum = float(sum(hits))
        ratio = round_hits / prompt_sum if prompt_sum > 0 else 0.0
        round_stat = {
            "round": r + 1,
            "ok": len(hits),
            "errors": errs,
            "prompt_tokens": hits,
            "round_hit_tokens": round(round_hits, 1),
            "hit_ratio_mean": round(ratio, 4),
            "hit_ratio_min": round(ratio, 4),
            "wall_s": round(dt, 1),
        }
        rounds.append(round_stat)
        print(
            f"burst round {r + 1}: ok={round_stat['ok']}/{args.sessions} "
            f"hit={ratio:.3f} ({round_hits:.0f}/{prompt_sum:.0f} tok) ({dt:.0f}s)",
            flush=True,
        )
    gate_rounds = rounds[args.rounds_gate - 1 :]
    gate_val = min((r["hit_ratio_mean"] for r in gate_rounds), default=0.0)
    return {
        "protocol": "burst",
        "sessions": args.sessions,
        "target_tokens": args.tokens,
        "rounds": rounds,
        "gate_rounds_from": args.rounds_gate,
        "gate_hit_mean_min": gate_val,
        "gate_pass": bool(gate_val >= args.round_hit_min and all(r["ok"] == args.sessions for r in gate_rounds)),
    }


def run_solo(args: argparse.Namespace) -> dict:
    prompt = _build_prompt(args.tokens, 0, "[solo-replay] Summarize the manifest rules in one sentence.")
    out = []
    for attempt in range(2):
        hits_before = _read_hits(args.base_url)
        t0 = time.time()
        _cached, prompt_tokens = _hit_ratio(
            _chat(args.base_url, args.model, prompt, args.max_tokens, args.timeout)
        )
        hit_tokens = _read_hits(args.base_url) - hits_before
        ratio = hit_tokens / prompt_tokens if prompt_tokens else 0.0
        out.append(
            {
                "attempt": attempt + 1,
                "prompt_tokens": prompt_tokens,
                "hit_tokens": round(hit_tokens, 1),
                "hit_ratio": round(ratio, 4),
                "wall_s": round(time.time() - t0, 1),
            }
        )
        print(f"solo attempt {attempt + 1}: hit={ratio:.3f} ({prompt_tokens} prompt tokens)", flush=True)
    replay = out[-1]["hit_ratio"]
    return {
        "protocol": "solo",
        "target_tokens": args.tokens,
        "attempts": out,
        "gate_hit_min": replay,
        "gate_pass": bool(replay >= args.solo_hit_min),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--protocol", choices=["burst", "solo", "both"], default="burst")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=60000, help="approx prompt size per session")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rounds-gate", type=int, default=2, help="evaluate hit gate from this round on")
    ap.add_argument("--round-hit-min", type=float, default=0.90)
    ap.add_argument("--solo-hit-min", type=float, default=0.93)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--gate", action="store_true", help="exit 1 when a gate fails")
    ap.add_argument("--out", default="", help="write the JSON result here")
    args = ap.parse_args()

    results: dict = {}
    ok = True
    if args.protocol in ("burst", "both"):
        r = run_burst(args)
        results["burst"] = r
        ok = ok and r["gate_pass"]
        print(f"burst gate: {r['gate_hit_mean_min']:.3f} vs {args.round_hit_min:.2f} -> {'PASS' if r['gate_pass'] else 'FAIL'}")
    if args.protocol in ("solo", "both"):
        r = run_solo(args)
        results["solo"] = r
        ok = ok and r["gate_pass"]
        print(f"solo gate: {r['gate_hit_min']:.3f} vs {args.solo_hit_min:.2f} -> {'PASS' if r['gate_pass'] else 'FAIL'}")

    payload = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n")
        print(f"wrote {args.out}")
    else:
        print(payload)
    # Without --gate the tool only reports (bench harness evaluates the JSON).
    return (0 if ok else 1) if args.gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
