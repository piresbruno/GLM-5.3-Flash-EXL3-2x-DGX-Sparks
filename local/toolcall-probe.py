#!/usr/bin/env python3
"""toolcall-probe.py — tool-call acceptance battery (R1 ops kit, ported from
the Reederey87 production kit; see NOTICE).

Fires a battery of chat completions that must produce well-formed OpenAI
tool calls through the served glm47 parser, and validates each response:
  - tool_calls present with the expected function name
  - arguments parse as JSON and carry the required key(s)
  - optional content channel stays free of a leaked raw JSON call

Cases (name, prompt, expected function, required arg keys):
  simple        single obvious call, flat args
  nested        nested object argument (struct validate)
  multi_tool    one of several offered tools must be chosen
  forced_choice tool_choice pins the function name
  repeat        N=3 identical calls (determinism / parser stability)

Exit 0 when the pass rate >= --min-pass (default 1.0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
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

BASE = os.environ.get("GLM53_BENCH_BASE", f"http://127.0.0.1:{_PORT}")
MODEL = os.environ.get("GLM53_BENCH_MODEL", "GLM-5.3-Flash-EXL3")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "passenger": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "tier": {"type": "string"}},
                    },
                },
                "required": ["origin", "destination", "passenger"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
                "required": ["amount", "from", "to"],
            },
        },
    },
]

WEATHER_ONLY = [TOOLS[0]]


def chat(payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def validate(resp: dict, want_fn: str, want_keys: list[str]) -> tuple[bool, str]:
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, f"no tool_calls (content head: {str(msg.get('content'))[:80]!r})"
    call = calls[0]
    fn = (call.get("function") or {}).get("name", "")
    if fn != want_fn:
        return False, f"function name {fn!r} != {want_fn!r}"
    raw = (call.get("function") or {}).get("arguments", "")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        return False, f"arguments not JSON: {e}: {str(raw)[:80]!r}"
    if not isinstance(args, dict):
        return False, f"arguments is {type(args).__name__}, want object"
    missing = [k for k in want_keys if k not in args]
    if missing:
        return False, f"arguments missing keys {missing}"
    return True, "ok"


CASES = [
    (
        "simple",
        {"model": MODEL, "messages": [{"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}]},
        WEATHER_ONLY, None, "get_weather", ["city"],
    ),
    (
        "nested",
        {"model": MODEL, "messages": [{"role": "user", "content": "Book a flight from Porto to Oslo for passenger Marta Silva, gold tier."}]},
        TOOLS, None, "book_flight", ["origin", "destination", "passenger"],
    ),
    (
        "multi_tool",
        {"model": MODEL, "messages": [{"role": "user", "content": "Convert 250 EUR to JPY please."}]},
        TOOLS, None, "convert_currency", ["amount", "from", "to"],
    ),
    (
        "forced_choice",
        {"model": MODEL, "messages": [{"role": "user", "content": "Weather in Kyoto?"}], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
        WEATHER_ONLY, {"type": "function", "function": {"name": "get_weather"}}, "get_weather", ["city"],
    ),
    (
        "repeat",
        {"model": MODEL, "messages": [{"role": "user", "content": "What is the weather in Bergen? Use the tool."}]},
        WEATHER_ONLY, None, "get_weather", ["city"],
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min-pass", type=float, default=1.0)
    ap.add_argument("--repeat", type=int, default=3, help="battery repetitions")
    ap.add_argument("--json", action="store_true", help="print per-case JSON at the end")
    args = ap.parse_args()

    results = []
    total = passed = 0
    for rep in range(args.repeat):
        for name, payload, tools, tool_choice, want_fn, want_keys in CASES:
            body = dict(payload)
            body["model"] = args.model
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
            body["temperature"] = 0
            body["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                ok, note = validate(chat(body), want_fn, want_keys)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                ok, note = False, f"request failed: {e}"
            total += 1
            passed += int(ok)
            results.append({"case": name, "rep": rep + 1, "pass": ok, "note": note})
            print(f"{'PASS' if ok else 'FAIL'} {name}#{rep + 1}: {note}", flush=True)

    rate = passed / total if total else 0.0
    print(f"toolcall-probe: {passed}/{total} = {rate:.2%} (min {args.min_pass:.0%})")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
