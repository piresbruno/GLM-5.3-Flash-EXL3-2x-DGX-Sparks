#!/usr/bin/env python3
"""bench_vision.py — vision (image+text) smoke/perf probe for the A/B arms.

AB-PLAN.md: vision requests get NO spec-decode speedup (DFlash2 drafts text-only),
so vision tok/s is a separate metric from text decode. Sends a synthetic image with
a verifiable question, records TTFT + tok/s + correctness. Usage:

  python3 tests/bench_vision.py --out results/ab/<arm>/vision.json \
      [--runs 3] [--max-tokens 120] [--temperature 0]

Generates distinctive images (solid color / shape / digits) with PIL so the
correctness check is deterministic. Requires the vision tower loaded
(LANGUAGE_MODEL_ONLY=0).
"""
import argparse
import base64
import io
import json
import time
import urllib.request

def make_png(kind: str) -> str:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (336, 336), "white")
    d = ImageDraw.Draw(img)
    if kind == "red_circle":
        d.ellipse([68, 68, 268, 268], fill=(220, 30, 30))
        q = "What shape is shown and what color is it? Answer in under 8 words."
        expect = ("circle", "red")
    elif kind == "blue_square":
        d.rectangle([68, 68, 268, 268], fill=(30, 60, 220))
        q = "What shape is shown and what color is it? Answer in under 8 words."
        expect = ("square", "blue")
    else:  # digits
        d.text((110, 120), "17", fill=(0, 0, 0), font_size=120)
        d.text((110, 240), "23", fill=(0, 0, 0), font_size=120)
        q = "Two numbers are written in the image. Multiply them and answer with the number only."
        expect = ("391",)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), q, expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kinds = ["red_circle", "blue_square", "digits"]
    rec = {"harness": "bench_vision.py", "runs": [], "temperature": a.temperature,
           "max_tokens": a.max_tokens, "ts": time.time(), "thinking": False}
    any_nan = False
    for i in range(a.runs):
        kind = kinds[i % len(kinds)]
        b64, q, expect = make_png(kind)
        payload = {
            "model": a.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": q},
            ]}],
            "max_tokens": a.max_tokens,
            "temperature": a.temperature,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            a.url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time(); ttft = None; text = []; ntok = 0; t_last = None; usage_ntok = None
        try:
            resp = urllib.request.urlopen(req, timeout=600)
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
                if chunk.get("usage"):
                    usage_ntok = chunk["usage"].get("completion_tokens", usage_ntok)
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    text.append(delta["content"]); ntok += 1; t_last = now
            dt = time.time() - t0
            full = "".join(text)
            ntok = usage_ntok if usage_ntok else ntok
            low = full.lower()
            ok = all(e in low for e in expect)
            nan = "nan" in low
            any_nan = any_nan or nan
            rec["runs"].append({
                "kind": kind, "ok_content": ok, "expected": list(expect),
                "nan": nan, "ttft_s": round(ttft, 3) if ttft else None,
                "seconds": round(dt, 3), "completion_tokens": ntok,
                "tok_s": round((ntok - 1) / (t_last - (t0 + ttft)), 1)
                if ttft and ntok > 1 and t_last > t0 + ttft else None,
                "text": full[:160],
            })
            print(f"{kind}: {rec['runs'][-1]['tok_s']} tok/s ttft={rec['runs'][-1]['ttft_s']}s "
                  f"ok={ok} nan={nan}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["runs"].append({"kind": kind, "ok_content": False, "err": str(e)[:160]})
            print(f"{kind}: ERROR {str(e)[:120]}", flush=True)

    ok = sum(1 for r in rec["runs"] if r.get("ok_content"))
    rec["summary"] = {"passed": ok, "total": a.runs, "any_nan": any_nan}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"vision: {ok}/{a.runs} passed, any_nan={any_nan} -> {a.out}", flush=True)
    return 0 if ok == a.runs and not any_nan else 2


if __name__ == "__main__":
    raise SystemExit(main())
