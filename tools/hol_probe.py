#!/usr/bin/env python3
"""hol_probe.py — head-of-line blocking probe (campaign R1 gate).

Fires a ~240k-token cold prefill (the blocking request), then ~2 s later a
small ~1.2k-token request behind it, and measures the small request's
FIRST-TOKEN latency. Gate (R1): first token <= 30 s (the Reederey87 kit
measured 7.9 s).

The blocking request uses max_tokens=1 so it leaves the scheduler as soon as
its prefill completes; the small request competes only with the prefill.

Usage:
  python3 tools/hol_probe.py --out results/ab/<arm>/hol.json
Gates evaluated into the JSON: hol_first_token_s <= --gate-s (default 30).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_PORT = "8081"
try:
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("PORT=") and not line.lstrip().startswith("#"):
            _PORT = line.split("=", 1)[1].strip()
except OSError:
    pass

BASE = "http://127.0.0.1:" + _PORT
MODEL = "GLM-5.3-Flash-EXL3"

FILLER = "The logistics ledger lists crates, seals, and duty stamps transferred at the mid-way depot. "


def _stream_once(content: str, max_tokens: int, on_first_token, timeout: float = 3600.0) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first = True
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if first and line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        on_first_token()
                        first = False
                except json.JSONDecodeError:
                    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--background-tokens", type=int, default=240000)
    ap.add_argument("--hol-tokens", type=int, default=1200)
    ap.add_argument("--stagger-s", type=float, default=2.0)
    ap.add_argument("--gate-s", type=float, default=30.0)
    ap.add_argument("--filler", default=FILLER,
                    help="filler sentence; use a UNIQUE one so the cold prefill "
                         "cannot hit blocks cached by earlier probes")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    globals()["BASE"] = args.base_url
    globals()["MODEL"] = args.model

    # ~240k-token blocking prompt (unique salt so prefix caching cannot help a
    # re-run; the small request must be cold too).
    salt = str(time.time_ns())
    filler = args.filler
    bg = filler * int(args.background_tokens / 16) + f"\n[salt {salt}] End of ledger."
    small_fill = filler * int(args.hol_tokens / 16)
    small = (
        small_fill
        + f"\n[salt {salt}] Question: reply with the single word: ready"
    )

    result: dict = {"background_tokens": args.background_tokens, "hol_tokens": args.hol_tokens}
    bg_done = threading.Event()

    def bg_run() -> None:
        try:
            _stream_once(bg, 1, lambda: None)
            result["background_ok"] = True
        except Exception as e:  # noqa: BLE001
            result["background_ok"] = False
            result["background_error"] = str(e)[:200]
        finally:
            bg_done.set()

    bg_t0 = time.time()
    threading.Thread(target=bg_run, daemon=True).start()
    time.sleep(args.stagger_s)

    hol_first: list[float] = []

    def on_first() -> None:
        hol_first.append(time.time())

    hol_t0 = time.time()
    err = None
    try:
        _stream_once(small, 8, on_first)
    except Exception as e:  # noqa: BLE001
        err = str(e)[:200]
    hol_ttft = (hol_first[0] - hol_t0) if hol_first else None
    bg_total = time.time() - bg_t0
    bg_done.wait(timeout=5)

    result.update(
        {
            "hol_first_token_s": round(hol_ttft, 2) if hol_ttft else None,
            "gate_s": args.gate_s,
            "gate_pass": bool(hol_ttft is not None and hol_ttft <= args.gate_s),
            "background_still_running_at_print": not bg_done.is_set(),
            "background_wall_s": round(bg_total, 1),
            "hol_error": err,
        }
    )
    payload = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n")
    print(payload)
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
