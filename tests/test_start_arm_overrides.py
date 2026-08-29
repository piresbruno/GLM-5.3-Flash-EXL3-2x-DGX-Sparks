#!/usr/bin/env python3
"""Regression tests for the extended caller-override capture (start.sh).

Upstream fixed MAX_NUM_SEQS (test_start_overrides.py); this extends the
coverage to the A/B-arm knobs that were silently losing to .env:
  - DFLASH_TOKENS            (C7 arm: `DFLASH_TOKENS=5` ran at k=7 — invalid arm)
  - MAX_NUM_BATCHED_TOKENS   (C6 arm: 2048 was silently dropped to 1024)
  - CG_ESTIMATE              (upstream 0195390: 0 returns graph deduction to KV)
  - MAX_MODEL_LEN            (boot-lottery fallback target)
  - KV_CACHE_MEMORY          (C4 pin knob)
Also asserts the 0.85 GPU_MEM_UTIL hard guard refuses >0.85.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# ----------------------------- configuration -------------------------------"

KNOBS = {
    "DFLASH_TOKENS": ("7", "5"),
    "MAX_NUM_BATCHED_TOKENS": ("1024", "2048"),
    "CG_ESTIMATE": ("1", "0"),
    "MAX_MODEL_LEN": ("1000000", "980000"),
    "KV_CACHE_MEMORY": ("", "19010164736"),
    "DFLASH_DRAFT_TP": ("1", "2"),
    "GLM53_MIXED_PREFILL_CHUNK": ("skip", "256"),
}


def _preamble() -> str:
    source = (ROOT / "start.sh").read_text()
    preamble, sep, _rest = source.partition(MARKER)
    assert sep, "start.sh configuration marker is missing"
    # die/warn/log are defined after the marker; the guard block calls them.
    helpers = (
        'log() { printf "[log] %s\\n" "$*"; }\n'
        'warn() { printf "[warn] %s\\n" "$*" >&2; }\n'
        'die() { printf "[die] %s\\n" "$*" >&2; exit 1; }\n'
    )
    return helpers + preamble


def test_all_arm_knobs_inline_override_wins() -> None:
    pre = _preamble()
    prints = "\n".join(f'printf "{k}=%s " "${{{k}:-unset}}"' for k in KNOBS)
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = tmp / "start.sh"
        script.write_text(pre + "\n" + prints + '\nprintf "\\n"\n')
        script.chmod(0o755)
        env_lines = "\n".join(f"{k}={dflt}" for k, (dflt, _) in KNOBS.items())
        (tmp / ".env").write_text(env_lines + "\n")
        env = dict(os.environ)
        for k, (_dflt, val) in KNOBS.items():
            if val:
                env[k] = val
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        got = r.stdout.strip().split()
        want = [f"{k}={v or 'unset'}" for k, (_d, v) in KNOBS.items()]
        assert got == want, f"override lost:\n got={got}\nwant={want}"


def test_util_085_hard_guard() -> None:
    pre = _preamble()
    source = (ROOT / "start.sh").read_text()
    guard = source.split("# HARD LIMIT (2026-08-29 crash review)")[1].split("MAX_NUM_SEQS=")[0]
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = tmp / "start.sh"
        script.write_text(pre + guard + '\nprintf "SURVIVED\\n"\n')
        script.chmod(0o755)
        (tmp / ".env").write_text("GPU_MEM_UTIL=0.85\n")
        env = dict(os.environ)
        env["GPU_MEM_UTIL"] = "0.87"
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
        assert r.returncode != 0, "0.87 must be refused"
        assert "0.85" in r.stderr + r.stdout, "refusal must mention the 0.85 cap"
        env["GPU_MEM_UTIL"] = "0.85"
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and "SURVIVED" in r.stdout


if __name__ == "__main__":
    test_all_arm_knobs_inline_override_wins()
    test_util_085_hard_guard()
    print("extended caller-override + util guard regression OK")
