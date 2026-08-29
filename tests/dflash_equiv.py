#!/usr/bin/env python3
"""Speculative-equivalence smoke harness (port of Entrpi's tools/dflash_equiv.py, MIT).

Greedy token-equivalence compare: `dump` against the spec-off (or MTP)
server, relaunch with SPEC_METHOD=dflash, `dump` again, then `compare`.

Methodology caveat (Entrpi FINDINGS §4, adopted here): strict token equality
is NOT a losslessness proof on a healthy stack — batch-shape numerics flip
argmax ties (<=0.75 nats) and the speculative server is run-to-run
non-deterministic while the baseline is deterministic. Use this harness as a
smoke gate (large structural diffs, crashes, finish-reason changes), and
treat isolated single-token diffs as inconclusive rather than failures.

Usage:
  API=http://127.0.0.1:8081/v1/chat/completions MODEL=GLM-5.3-Flash-EXL3 \
    python3 tests/dflash_equiv.py dump baseline      # SPEC_METHOD=mtp or none
  # ... ./start.sh restart with SPEC_METHOD=dflash ...
  API=http://127.0.0.1:8081/v1/chat/completions MODEL=GLM-5.3-Flash-EXL3 \
    python3 tests/dflash_equiv.py dump dflash
  python3 tests/dflash_equiv.py compare baseline dflash
"""

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("API", "http://127.0.0.1:8081/v1/chat/completions")
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")
OUTDIR = os.environ.get("OUTDIR", os.path.expanduser("~/dflash_equiv"))

CASES = [
    ("short_eos", "What is 2+2? Answer with just the number.", 64),
    ("mid_math", "Compute 17 * 23 step by step, then state the answer.", 256),
    ("long_prose", "Write a detailed 400-word explanation of how tides work.", 640),
    ("code", "Write a Python function that merges two sorted lists. Code only.", 384),
    ("list_gen", "List the first 25 prime numbers, comma separated.", 256),
    ("prefix_repeat", "Compute 17 * 23 step by step, then state the answer.", 256),
    # Exercises the draft's non-causal sliding window: generation must run
    # well past 2048 tokens of draft KV.
    (
        "very_long",
        "Write a thorough, chapter-structured 2500-word essay on the history "
        "of numerical linear algebra, from Gauss to GPUs.",
        3072,
    ),
]
CONCURRENT_CASE = ("concurrent", "Count from 1 to 40, one number per line.", 320)


def query(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 12345,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.load(r)
    choice = out["choices"][0]
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "completion_tokens": out["usage"]["completion_tokens"],
    }


def dump(label: str) -> None:
    results = {}
    for name, prompt, max_tokens in CASES:
        results[name] = query(prompt, max_tokens)
        print(f"  {name}: {results[name]['completion_tokens']} tokens")
    name, prompt, max_tokens = CONCURRENT_CASE
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(query, prompt, max_tokens) for _ in range(5)]
        results[name] = [f.result() for f in futs]
    print(f"  {name}: {[r['completion_tokens'] for r in results[name]]} tokens x5")
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{label}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {path}")


def compare(a: str, b: str) -> None:
    with open(os.path.join(OUTDIR, f"{a}.json")) as f:
        da = json.load(f)
    with open(os.path.join(OUTDIR, f"{b}.json")) as f:
        db = json.load(f)
    failures = 0
    benign = 0
    for name in da:
        va, vb = da[name], db[name]
        pairs = zip(va, vb) if isinstance(va, list) else [(va, vb)]
        for i, (xa, xb) in enumerate(pairs):
            tag = f"{name}[{i}]" if isinstance(va, list) else name
            if xa == xb:
                print(f"  MATCH {tag} ({xa['completion_tokens']} tokens)")
                continue
            failures += 1
            print(f"  DIFF  {tag}:")
            for k in ("finish_reason", "completion_tokens"):
                if xa[k] != xb[k]:
                    print(f"        {k}: {xa[k]} vs {xb[k]}")
            ca, cb = xa["content"], xb["content"]
            if ca != cb:
                pos = next(
                    (j for j, (p, q) in enumerate(zip(ca, cb)) if p != q),
                    min(len(ca), len(cb)),
                )
                div = abs(len(ca) - len(cb))
                if pos > 300 and div <= 4:
                    benign += 1
                    print(
                        f"        content diverges at char {pos} (late, <=4-char "
                        "length delta — likely an argmax tie-flip; inconclusive per FINDINGS §4)"
                    )
                else:
                    print(f"        content diverges at char {pos}:")
                    print(f"          {a}: ...{ca[max(0, pos - 40) : pos + 40]!r}")
                    print(f"          {b}: ...{cb[max(0, pos - 40) : pos + 40]!r}")
    print(("EQUIVALENT" if failures == 0 else f"NOT EQUIVALENT: {failures} diffs"))
    if benign:
        print(
            f"  ({benign} late single-token diffs flagged benign per FINDINGS §4 "
            "tie-flip caveat — inspect above)"
        )
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
