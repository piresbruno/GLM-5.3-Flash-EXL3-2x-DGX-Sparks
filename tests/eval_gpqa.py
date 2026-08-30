#!/usr/bin/env python3
"""gpqa_diamond quality gate (D3) — end-to-end multiple-choice accuracy.

Methodology (Entrpi FINDINGS §9/§10, adopted): greedy, temp 0, thinking off,
robust grading (never first-line). Gate >= 70% at n=50 (Entrpi: 72%).

Usage:
  python3 tests/eval_gpqa.py --n 50 --out results/ab/d3-gpqa.json

Uses the HuggingFace datasets-server API (host needs internet + HF access).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request

API_BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "Idavidrein%2Fgpqa"
CONFIG = "gpqa_diamond"
SPLIT = "train"


def _hf_auth_header() -> dict:
    """MATH-500/GPQA went gated upstream — auth with the local HF token if present."""
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        p = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(p):
            token = open(p).read().strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_rows(length: int) -> list[dict]:
    url = (
        f"{API_BASE}?dataset={DATASET}&config={CONFIG}&split={SPLIT}"
        f"&offset=0&length={length}"
    )
    last: Exception = RuntimeError("unreachable")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_hf_auth_header()), timeout=60) as r:
                data = json.load(r)
            return [row["row"] for row in data["rows"]]
        except urllib.error.HTTPError as e:
            last = e
            print(f"  datasets-server HTTP {e.code} (attempt {attempt + 1}/4): {e.read()[:200]}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise last


def build_mcq(row: dict, rng: random.Random) -> tuple[str, str]:
    """Return (prompt, correct_letter). Options shuffled with seeded RNG."""
    q = row["Question"]
    correct = row["Correct Answer"]
    wrongs = [row[f"Incorrect Answer {i}"] for i in range(1, 4)]
    opts = [(correct, True)] + [(w, False) for w in wrongs]
    rng.shuffle(opts)
    letters = "ABCD"
    lines = [q, ""]
    for letter, (text, _) in zip(letters, opts):
        lines.append(f"{letter}. {text}")
    prompt = (
        "Answer the following multiple-choice question with the single best "
        "letter (A, B, C or D). Output only the letter.\n\n"
        + "\n".join(lines)
    )
    correct_letter = next(l for l, (_, is_c) in zip(letters, opts) if is_c)
    return prompt, correct_letter


def extract_letter(pred: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", pred)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gate", type=int, default=70, help="pass threshold, %")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = fetch_rows(500)
    rng = random.Random(a.seed)
    sample = rng.sample(rows, min(a.n, len(rows)))
    print(f"gpqa_diamond: {len(sample)} items, temp 0, thinking off")

    body_base = {
        "model": a.model,
        "temperature": 0,
        "max_tokens": a.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    passed = 0
    results = []
    for i, row in enumerate(sample):
        prompt, correct_letter = build_mcq(row, random.Random(a.seed * 1000 + i))
        body = {**body_base, "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] REQUEST FAIL: {e}")
            results.append({"ok": False, "correct_letter": correct_letter, "pred": ""})
            continue
        letter = extract_letter(content)
        ok = letter == correct_letter
        passed += ok
        results.append({"ok": ok, "correct_letter": correct_letter, "letter": letter, "pred": content})
        print(f"  [{i}] {'PASS' if ok else 'FAIL'}  letter={letter} correct={correct_letter}")

    score = passed / len(sample) * 100
    print(f"\ngpqa_diamond: {passed}/{len(sample)} = {score:.1f}%  (gate {a.gate}%)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {"score": score, "passed": passed, "total": len(sample),
                 "gate": a.gate, "results": results}, f, indent=1
            )
        print(f"saved {a.out}")
    return 0 if score >= a.gate else 1


if __name__ == "__main__":
    sys.exit(main())
