#!/usr/bin/env python3
"""verify_dsd.py — runtime receipt that Dynamic Speculative Decoding is live.

The boot config's SpeculativeConfig repr does not print the DSD table, so the
only trustworthy proof is behavioral: under DSD with table [1,1,K1],[2,..,K2],
a c>=2 burst must draft exactly K2 tokens per verify step per sequence —
per-position accepted counts at positions >= K2 freeze while positions < K2
grow. Under static k, all 7 positions keep growing.

Protocol:
  1. snapshot /metrics  (spec_decode_num_accepted_tokens_per_pos vector)
  2. fire GEN_SEQUENCES concurrent short gens (temp 0, thinking off) at
     --concurrency (>= 2), unique salt so prefix caching cannot interfere
  3. re-read /metrics; diff per-position accepted counters
  4. verdict:
     - positions < K grow AND positions >= K frozen  -> DSD ACTIVE
     - all positions grew                            -> DSD INACTIVE
     - nothing grew                                  -> INCONCLUSIVE

Non-fatal by design: exit 0 on ACTIVE, 1 otherwise (launcher WARNs).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8081"
MODEL = "GLM-5.3-Flash-EXL3"
POS_METRIC = "vllm:spec_decode_num_accepted_tokens_per_pos"


def http_json(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def read_pos_counters(base: str) -> dict[int, int]:
    """Parse the per-position accepted counters from the Prometheus endpoint."""
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as resp:
        text = resp.read().decode()
    counters: dict[int, int] = {}
    for line in text.splitlines():
        if not line.startswith(f"{POS_METRIC}{{") and not line.startswith(
            f"{POS_METRIC}_total{{"
        ):
            continue
        if "position=" not in line:
            continue
        pos_part = line.split("position=")[1].split('"')[1]
        value_part = line.rsplit("}", 1)[1].strip()
        try:
            counters[int(pos_part)] = int(float(value_part))
        except ValueError:
            continue
    return counters


def gen_once(base: str, model: str, salt: str, max_tokens: int) -> bool:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"[dsd-receipt {salt}] Count from 1 to 20, digits only.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        http_json(f"{base}/v1/chat/completions", payload)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def fire_concurrent(base: str, model: str, n: int, max_tokens: int) -> int:
    salt = f"{time.time_ns()}"
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(
            pool.map(
                lambda i: gen_once(base, model, f"{salt}-{i}", max_tokens),
                range(n),
            )
        )
    return sum(results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--gens", type=int, default=2, help="waves of --concurrency requests")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--expected-k", type=int, default=5,
                    help="K the DSD table assigns at --concurrency")
    args = ap.parse_args()

    if args.concurrency < 2:
        print("verify_dsd: --concurrency must be >= 2 (DSD at c=1 is the static k)")
        return 1

    before = read_pos_counters(args.base_url)
    ok = 0
    for wave in range(args.gens):
        ok += fire_concurrent(
            args.base_url, args.model, args.concurrency, args.max_tokens
        )
    if ok < args.concurrency * args.gens:
        print(f"verify_dsd: INCONCLUSIVE — only {ok} requests succeeded")
        return 1
    time.sleep(1)  # let the metrics counters settle
    after = read_pos_counters(args.base_url)

    grew = {p: after.get(p, 0) - before.get(p, 0) for p in sorted(after)}
    if not any(d > 0 for d in grew.values()):
        print(f"verify_dsd: INCONCLUSIVE — no accepted-position growth: {grew}")
        return 1

    frozen = [p for p, d in grew.items() if d == 0 and p >= args.expected_k]
    active_below = [p for p, d in grew.items() if d > 0 and p < args.expected_k]
    leaked = [p for p, d in grew.items() if d > 0 and p >= args.expected_k]

    print("verify_dsd: per-position accepted deltas:", grew)
    if leaked:
        print(
            f"verify_dsd: INACTIVE — positions {leaked} grew during a "
            f"c={args.concurrency} burst (expected K={args.expected_k} cap)"
        )
        return 1
    if not active_below or not frozen:
        print(
            f"verify_dsd: INCONCLUSIVE — no growth below K={args.expected_k} "
            f"({active_below}) or no frozen positions (frozen={frozen})"
        )
        return 1
    print(
        f"verify_dsd: ACTIVE — positions {active_below} grew, positions "
        f"{frozen} frozen at a c={args.concurrency} burst (K={args.expected_k})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
