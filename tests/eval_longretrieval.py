#!/usr/bin/env python3
"""Long-context retrieval gate (D3) — >=100k-token synthetic ledger, estonia-style.

Self-contained probe: builds a deterministic ledger of ~100k tokens of
"RECORD <id>: key_<hex> = value_<hex>" lines, embeds 5 needles at depths
20/40/60/75/90%, then asks for the values. Gate >= 4/5 exact matches
(Entrpi's estonia 133k band: 9/10; this is 100k with 5 needles -> 4/5).

Prefix caching makes re-runs fast; the first run pays a ~100k prefill.

Usage:
  python3 tests/eval_longretrieval.py --tokens 100000 --out results/ab/d3-longctx.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request

TOKENS_PER_RECORD = 10  # ~35 chars/record / ~3.5 chars/token


def build_ledger(tokens: int, seed: int) -> tuple[str, list[tuple[str, str, int]]]:
    rng = random.Random(seed)
    n_records = tokens // TOKENS_PER_RECORD
    lines, needles = [], []
    for i in range(n_records):
        key = f"key_{rng.getrandbits(32):08x}"
        value = f"val_{rng.getrandbits(32):08x}"
        lines.append(f"RECORD {i}: {key} = {value}")
        if i in [n_records // 5, 2 * n_records // 5, 3 * n_records // 5,
                 (3 * n_records) // 4, 9 * n_records // 10]:
            needles.append((key, value, i))
    ledger = "\n".join(lines)
    return ledger, needles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--gate", type=int, default=4, help="pass threshold, /5")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ledger, needles = build_ledger(a.tokens, a.seed)
    qs = "\n".join(
        f"Q{i + 1}: What is the value for {key}? Answer with the value only."
        for i, (key, _, _) in enumerate(needles)
    )
    prompt = (
        "You are a precise retrieval assistant. Below is a ledger of records. "
        "Answer each question with exactly the value (val_xxxxxxxx), one per line.\n\n"
        f"{ledger}\n\n{qs}"
    )
    body = {
        "model": a.model,
        "temperature": 0,
        "max_tokens": a.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    print(f"longctx: {a.tokens} tokens, {len(needles)} needles, temp 0 (prefill first)")
    req = urllib.request.Request(
        a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        content = json.load(r)["choices"][0]["message"]["content"]

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    passed = 0
    results = []
    for i, (key, value, depth) in enumerate(needles):
        got = None
        for ln in lines:
            if value in ln or (got is None and ln.startswith(("val_",)) and i < len(lines)):
                pass
            if value in ln:
                got = ln
                break
        # fallback: i-th answer line
        if got is None and i < len(lines):
            got = lines[i]
        ok = got is not None and value in got
        passed += ok
        results.append({"key": key, "expected": value, "depth_pct": depth / (a.tokens // TOKENS_PER_RECORD) * 100, "got": got, "ok": ok})
        print(f"  Q{i + 1} @{results[-1]['depth_pct']:.0f}% depth: {'PASS' if ok else 'FAIL'}  got={got!r} expected={value!r}")

    score = passed / len(needles)
    print(f"\nlongctx retrieval: {passed}/{len(needles)} exact  (gate {a.gate}/5)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {"tokens": a.tokens, "passed": passed, "total": len(needles),
                 "gate": a.gate, "results": results, "raw": content}, f, indent=1
            )
        print(f"saved {a.out}")
    return 0 if passed >= a.gate else 1


if __name__ == "__main__":
    sys.exit(main())
