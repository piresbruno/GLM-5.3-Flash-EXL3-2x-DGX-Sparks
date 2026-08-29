#!/usr/bin/env python3
"""D5 wiring tests: DSD_TABLE parsing/validation, capture-ladder derivation,
inner-script speculative-config emission, and launcher env passing.

Offline by construction: the DSD block is sliced out of start.sh between its
# ---- D5 / # ---- end D5 markers and evaluated in a bash harness; the inner
scripts are regenerated via write_inner_scripts() into a temp dir and their
python -S -c snippet is executed to assert the emitted speculative-config JSON.

Runtime (on-GPU) receipt lives in tests/verify_dsd.py — per-position
acceptance must freeze at pos K during a c>=2 burst.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
D5_BEGIN = "# ---- D5: dynamic speculative decoding (DSD) ---"
D5_END = "# ---- end D5 ----"

HELPERS = (
    'log() { printf "[log] %s\\n" "$*"; }\n'
    'warn() { printf "[warn] %s\\n" "$*" >&2; }\n'
    'die() { printf "[die] %s\\n" "$*" >&2; exit 1; }\n'
)


def dsd_block() -> str:
    src = START.read_text()
    m = re.search(
        re.escape(D5_BEGIN) + r".*?" + re.escape(D5_END), src, re.S
    )
    assert m, "D5 block markers missing from start.sh"
    return m.group(0)


def run_dsd(env: dict[str, str], snippet: str) -> subprocess.CompletedProcess[str]:
    """Evaluate the D5 block (plus an optional extra snippet) in bash."""
    script = HELPERS + dsd_block() + "\n" + snippet
    with tempfile.TemporaryDirectory() as raw:
        sh = Path(raw) / "dsd.sh"
        sh.write_text(script)
        e = dict(os.environ)
        e.update(env)
        return subprocess.run(
            ["bash", str(sh)], capture_output=True, text=True, env=e
        )


BASE = {
    "SPEC_METHOD": "dflash",
    "DFLASH_TOKENS": "7",
    "MAX_NUM_SEQS": "4",
    "DSD_TABLE": "",
    # R1: DSD requires ASYNC_SCHEDULING=1 (the active path is the
    # AsyncScheduler) — dsd_validate dies otherwise.
    "ASYNC_SCHEDULING": "1",
}


def test_off_by_default_is_noop() -> None:
    r = run_dsd(BASE, 'dsd_validate && printf "sizes=[%s]" "$(dsd_capture_sizes)"')
    assert r.returncode == 0, r.stderr
    # Empty table -> dsd_k_for falls back to DFLASH_TOKENS=7 everywhere, so the
    # derived ladder is byte-identical to the static DFlash2 ladder.
    assert "sizes=[1 2 4 8 16 24 32]" in r.stdout


def test_two_step_table_ladder() -> None:
    env = dict(BASE, DSD_TABLE="1:1:7,2:999:5")
    r = run_dsd(
        env,
        'dsd_validate || exit 1\n'
        'printf "k=%s,%s sizes=[%s]" "$(dsd_k_for 1)" "$(dsd_k_for 4)" "$(dsd_capture_sizes)"',
    )
    assert r.returncode == 0, r.stderr
    assert "k=7,5" in r.stdout
    # seq*(1+K): 1*8=8, 2*6=12, 3*6=18, 4*6=24; 16/32 unreachable under DSD.
    assert "sizes=[1 2 4 8 12 18 24]" in r.stdout


def test_three_range_table_ladder() -> None:
    env = dict(BASE, DSD_TABLE="1:1:7,2:3:5,4:999:4")
    r = run_dsd(
        env,
        'dsd_validate || exit 1\nprintf "sizes=[%s]" "$(dsd_capture_sizes)"',
    )
    assert r.returncode == 0, r.stderr
    # 1*8=8, 2*6=12, 3*6=18, 4*5=20
    assert "sizes=[1 2 4 8 12 18 20]" in r.stdout


def test_validation_rejects() -> None:
    cases = {
        "gap": "1:1:7,3:999:5",
        "k_above_trained": "1:1:8",
        "k_above_dflash_tokens": None,  # DFLASH_TOKENS=5 vs k=7, set below
        "malformed": "1:1:7,x",
        "zero_start": "0:999:5",
        "inverted": "4:2:5",
        "not_covering_max_num_seqs": "1:1:7,2:2:5",
    }
    for name, table in cases.items():
        env = dict(BASE, DSD_TABLE=table or "1:1:7,2:999:5")
        if name == "k_above_dflash_tokens":
            env["DFLASH_TOKENS"] = "5"
        r = run_dsd(env, "dsd_validate")
        assert r.returncode != 0, f"{name} was accepted"
        assert "[die]" in r.stderr


def test_dsd_requires_dflash() -> None:
    env = dict(BASE, SPEC_METHOD="mtp", DSD_TABLE="1:1:7,2:999:5")
    r = run_dsd(env, "dsd_validate")
    assert r.returncode != 0
    assert "SPEC_METHOD=dflash" in r.stderr


def test_launcher_wiring_present() -> None:
    src = START.read_text()
    # CLI override survives the .env source (pattern of the other arm knobs)
    assert '_cli_dsd="${DSD_TABLE-}"' in src
    assert '[ -n "${_cli_dsd}" ] && DSD_TABLE="${_cli_dsd}"' in src
    # worker serve_env list and head docker run both pass the table
    assert re.search(r"DFLASH_DRAFT_TP DSD_TABLE \\", src), "worker serve_env"
    assert '-e DSD_TABLE="${DSD_TABLE:-}"' in src, "head -e DSD_TABLE"
    # DSD-aware ladder replaces the static one only when the table is set
    assert "--cudagraph-capture-sizes $(dsd_capture_sizes)" in src
    # boot verification greps exist
    assert "Overriding cudagraph_mode from" in src
    assert "tests/verify_dsd.py" in src


def _write_inner_scripts(tmp: Path) -> tuple[Path, Path]:
    src = START.read_text()
    m = re.search(
        r"write_inner_scripts\(\) \{\n(.*?)\n\}\n", src, re.S
    )
    assert m, "write_inner_scripts() not found"
    script = tmp / "gen.sh"
    script.write_text(
        HELPERS
        + "SCRIPT_DIR="
        + str(tmp)
        + "\n"
        + "HEAD_SCRIPT=$SCRIPT_DIR/.head.inner.sh\n"
        + "WORKER_SCRIPT=$SCRIPT_DIR/.worker.inner.sh\n"
        + m.group(1)
        + "\n"
    )
    script.chmod(0o755)
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return tmp / ".head.inner.sh", tmp / ".worker.inner.sh"


def _spec_json(inner: Path, dsd: str) -> dict:
    text = inner.read_text()
    assert "num_speculative_tokens_per_batch_size" in text
    m = re.search(r"python3 -S -c '(.*?)'\)", text, re.S)
    assert m, "spec-config python snippet not found in generated inner script"
    env = dict(
        os.environ,
        DFLASH_MODEL_DIR="/models/dflash2",
        DFLASH_TOKENS="7",
        DFLASH_DRAFT_TP="1",
        DSD_TABLE=dsd,
    )
    r = subprocess.run(
        ["python3", "-S", "-c", m.group(1)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_inner_scripts_emit_dsd_table() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        head, worker = _write_inner_scripts(tmp)
        for inner in (head, worker):
            cfg = _spec_json(inner, "1:1:7,2:999:5")
            assert cfg["method"] == "dflash"
            assert cfg["num_speculative_tokens"] == 7
            assert cfg["num_speculative_tokens_per_batch_size"] == [
                [1, 1, 7],
                [2, 999, 5],
            ]
            off = _spec_json(inner, "")
            assert "num_speculative_tokens_per_batch_size" not in off


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
