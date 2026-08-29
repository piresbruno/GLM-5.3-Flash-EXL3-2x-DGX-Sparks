#!/usr/bin/env python3
"""bench_concurrency.py — C0 A/B harness: decode throughput + TTFT at concurrency levels.

Port of GLM-5.3-Flash-NVFP4-DFlash2 probes/bench_c1c6.py, adapted to this recipe
(:8081, GLM-5.3-Flash-EXL3, DFlash2 spec-decode metrics). AB-PLAN.md Phase 1 step 5.

Usage:
  python3 tests/bench_concurrency.py --out results/ab/<arm>-sweep.json \
      [--url http://127.0.0.1:8081] [--levels 1,2,4] [--rounds 2] \
      [--max-tokens 400] [--temperature 1.0]

Per level: `rounds` waves of c parallel streaming requests (unique per-request salt
defeats prefix caching so APC gains cannot mask arm deltas). Reports aggregate tok/s,
per-stream decode tok/s (first→last token, excludes TTFT), TTFT, failures, and the
DFlash2 accepted÷drafted ratio over the level's window, scraped from /metrics.
Also captures per-position acceptance + preemptions (vllm#53030 sanity).
"""
import argparse
import json
import random
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that parses an nginx access log line into a dict, with a regex, and explain each group.",
    "Implement a rate limiter class in Python using the token bucket algorithm, then show example usage.",
    "A warehouse ships 340 orders/day growing 6% weekly. Model 8 weeks of volume in a Python list comprehension and explain.",
    "Write a SQL query for the top 5 customers by 90-day revenue, then rewrite it as a window function version.",
    "Explain the difference between TCP slow start and congestion avoidance, then pseudocode both.",
    "Implement binary search in Python, then walk through the trace on [2,5,8,12,17,23] searching for 17.",
    "Write a short blog paragraph about why distributed tracing matters, in a casual tone.",
]


def stream_one(url, model, prompt, max_tokens, temperature, out, idx):
    """One streaming chat request: TTFT + decode tok/s from SSE deltas."""
    salt = f"[run {random.randint(1, 10**9)}] "
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": salt + prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    ntok = 0
    t_last = None
    try:
        resp = urllib.request.urlopen(req, timeout=900)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                ntok += 1
                t_last = now
        dt = time.time() - t0
        t_first_abs = t0 + ttft if ttft is not None else None
        out[idx] = {
            "ok": True,
            "seconds": round(dt, 3),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "completion_tokens": ntok,
            # decode-only speed: first->last content token, excludes TTFT
            "decode_toks_per_s": round((ntok - 1) / (t_last - t_first_abs), 1)
            if ttft is not None and ntok > 1 and t_last > t_first_abs
            else None,
        }
    except Exception as e:  # noqa: BLE001
        out[idx] = {"ok": False, "err": str(e)[:120], "seconds": round(time.time() - t0, 3)}


def metrics_snapshot(url):
    """Scrape spec-decode totals, per-position acceptance, preemptions, APC hits."""
    snap = {}
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return snap
    for line in txt.splitlines():
        for key, prefix in (
            ("drafted", "vllm:spec_decode_num_draft_tokens_total"),
            ("accepted", "vllm:spec_decode_num_accepted_tokens_total"),
            ("preemptions", "vllm:num_preemptions_total"),
            ("apc_hits", "vllm:prefix_cache_hits_total"),
            ("apc_queries", "vllm:prefix_cache_queries_total"),
        ):
            if line.startswith(prefix):
                try:
                    snap[key] = snap.get(key, 0.0) + float(line.split()[-1])
                except ValueError:
                    pass
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            try:
                pos = int(line.split('position="')[1].split('"')[0])
                snap.setdefault("accept_per_pos", {})[pos] = float(line.split()[-1])
            except (IndexError, ValueError):
                pass
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--levels", default="1,2,4")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rec = {
        "harness": "bench_concurrency.py",
        "url": a.url,
        "levels": [int(x) for x in a.levels.split(",")],
        "rounds": a.rounds,
        "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "ts": time.time(),
        "levels_data": {},
    }

    # warmup (≥1 gen; plan rule 4 wants the engine warm before timing)
    warm = {}
    stream_one(a.url, a.model, PROMPTS[0], 32, a.temperature, warm, 0)
    rec["warmup"] = warm.get(0)

    print(f"{'c':>2} {'reqs':>4} {'agg_tok/s':>10} {'decode_tok/s':>12} {'TTFT_ms':>8} "
          f"{'mean_s':>7} {'fails':>5}  accept_ratio", flush=True)
    for c in rec["levels"]:
        m0 = metrics_snapshot(a.url)
        agg_toks, agg_time, decodes, ttfts, fails, nreq = 0, 0.0, [], [], 0, 0
        for _ in range(a.rounds):
            out = {}
            threads = [
                threading.Thread(
                    target=stream_one,
                    args=(a.url, a.model, PROMPTS[(i * 7 + c) % len(PROMPTS)],
                          a.max_tokens, a.temperature, out, i),
                )
                for i in range(c)
            ]
            t0 = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.time() - t0
            ok = [v for v in out.values() if v.get("ok")]
            fails += c - len(ok)
            nreq += c
            agg_toks += sum(v["completion_tokens"] for v in ok)
            agg_time += wall
            decodes += [v["decode_toks_per_s"] for v in ok if v.get("decode_toks_per_s")]
            ttfts += [v["ttft_s"] for v in ok if v.get("ttft_s") is not None]
        m1 = metrics_snapshot(a.url)
        d_delta = m1.get("drafted", 0) - m0.get("drafted", 0)
        a_delta = m1.get("accepted", 0) - m0.get("accepted", 0)
        ratio = round(a_delta / d_delta, 3) if d_delta > 0 else None
        agg = agg_toks / agg_time if agg_time else 0.0
        dec = sum(decodes) / len(decodes) if decodes else 0.0
        ttft_ms = (sum(ttfts) / len(ttfts) * 1000) if ttfts else 0.0
        ms = agg_time / a.rounds
        lvl = {
            "requests": nreq,
            "agg_toks_per_s": round(agg, 1),
            "decode_toks_per_s": round(dec, 1),
            "ttft_ms_mean": round(ttft_ms),
            "mean_wall_s": round(ms, 1),
            "fails": fails,
            "accept_ratio": ratio,
        }
        rec["levels_data"][c] = lvl
        print(f"{c:>2} {nreq:>4} {agg:>10.1f} {dec:>12.1f} {ttft_ms:>8.0f} "
              f"{ms:>7.1f} {fails:>5}  {ratio}", flush=True)

    rec["final_metrics"] = metrics_snapshot(a.url)
    pp = rec["final_metrics"].get("accept_per_pos", {})
    if pp:
        total = sum(pp.values())
        rec["accept_per_pos_ratio"] = {
            p: round(v / total, 3) for p, v in sorted(pp.items())
        } if total else {}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
