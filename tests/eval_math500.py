#!/usr/bin/env python3
"""math_500 quality gate (D3) — end-to-end task accuracy of the live serve.

Methodology (from Entrpi FINDINGS §9/§10, adopted): greedy decoding, temp 0,
thinking off, robust answer extraction (never first-line grading), n=100 gate
band 86-88% at fp8 KV on this hardware class.

Usage:
  python3 tests/eval_math500.py --n 100 --out results/ab/d3-math500.json

Exit 0 = gate passed (>= --gate, default 86/100), 1 = gate failed.
Uses the HuggingFace datasets-server API (host needs internet + HF access).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

API_BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "HuggingFaceH4%2FMATH-500"
SPLIT = "test"

import random


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
    import time
    rows: list[dict] = []
    offset = 0
    # datasets-server caps length at 100 — paginate
    while len(rows) < length:
        chunk = min(100, length - len(rows))
        url = (
            f"{API_BASE}?dataset={DATASET}&config=default&split={SPLIT}"
            f"&offset={offset}&length={chunk}"
        )
        last: Exception = RuntimeError("unreachable")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=_hf_auth_header()), timeout=60) as r:
                    data = json.load(r)
                rows.extend(row["row"] for row in data["rows"])
                last = RuntimeError("unreachable")
                break
            except urllib.error.HTTPError as e:
                last = e
                print(f"  datasets-server HTTP {e.code} (attempt {attempt + 1}/4): {e.read()[:200]}", flush=True)
                time.sleep(10 * (attempt + 1))
        if isinstance(last, Exception) and str(last) != "unreachable":
            raise last
        if chunk == 0:
            break
        offset += chunk
    return rows[:length]


def extract_answer(text: str) -> str:
    """Robust extraction: \\boxed{...} first, then trailing '#### ' answer."""
    text = text.strip()
    # \boxed{...} with nested braces handled
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        start = idx + len(r"\boxed{")
        depth, i = 1, start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        return text[start : i - 1].strip()
    if "#### " in text:
        return text.rsplit("#### ", 1)[-1].strip()
    # last line fallback (never first-line grading)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


def normalize(ans: str) -> str:
    ans = ans.replace(",", "").replace("$", "").strip()
    ans = re.sub(r"\s+", " ", ans)
    return ans


def numeric_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def correct(pred: str, ref: str) -> bool:
    p, r = normalize(extract_answer(pred)), normalize(extract_answer(ref))
    if not p or not r:
        return False
    if p == r:
        return True
    return numeric_equal(p, r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gate", type=int, default=86, help="pass threshold, n=100")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = fetch_rows(500)
    rng = random.Random(a.seed)
    sample = rng.sample(rows, min(a.n, len(rows)))
    print(f"math_500: {len(sample)} items, temp 0, thinking off, url={a.url}")

    body_base = {
        "model": a.model,
        "temperature": 0,
        "max_tokens": a.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    passed = 0
    results = []
    for i, row in enumerate(sample):
        prompt = (
            "Solve the following math problem. Provide the final answer "
            "clearly, e.g. boxed.\n\n"
            f"Problem: {row['problem']}\n\nFinal answer:"
        )
        body = {**body_base, "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] REQUEST FAIL: {e}")
            results.append({"problem": row["problem"], "ref": row["answer"], "pred": "", "ok": False})
            continue
        ok = correct(content, row["answer"])
        passed += ok
        results.append(
            {
                "problem": row["problem"],
                "ref": row["answer"],
                "extracted_ref": extract_answer(row["answer"]),
                "pred": content,
                "extracted_pred": extract_answer(content),
                "ok": ok,
            }
        )
        print(f"  [{i}] {'PASS' if ok else 'FAIL'}  pred={extract_answer(content)!r} ref={extract_answer(row['answer'])!r}")

    score = passed / len(sample) * 100
    print(f"\nmath_500: {passed}/{len(sample)} = {score:.1f}%  (gate {a.gate}%)")
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
