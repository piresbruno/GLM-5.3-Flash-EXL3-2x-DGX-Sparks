#!/usr/bin/env python3
"""R1 bundle wiring tests (campaign R1 — Reederey87 bundle adoption).

Offline by construction, in the start.sh-slicing pattern:
  - config-preamble slices are evaluated in a bash harness (override-capture,
    MNBT default, ASYNC_SCHEDULING validation);
  - write_inner_scripts() is regenerated into a temp dir and EXECUTED with a
    stubbed `vllm` on PATH to assert the exact argv both ranks receive;
  - the repo .env is parsed for the digest pin and bundle values;
  - launch/stop/watchdog wiring is asserted textually against start.sh.
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
ENV_FILE = ROOT / ".env"
MARKER = "# ----------------------------- configuration -------------------------------"

HELPERS = (
    'log() { printf "[log] %s\\n" "$*"; }\n'
    'warn() { printf "[warn] %s\\n" "$*" >&2; }\n'
    'die() { printf "[die] %s\\n" "$*" >&2; exit 1; }\n'
)

BUNDLE_ENV = {
    "MAX_NUM_BATCHED_TOKENS": "3584",
    "LONG_PREFILL_TOKEN_THRESHOLD": "1792",
    "ASYNC_SCHEDULING": "0",
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "0",
    "FLASHINFER_WORKSPACE_BASE": "/root/.cache/vllm/flashinfer",
}


def _preamble() -> str:
    src = START.read_text()
    pre, sep, _rest = src.partition(MARKER)
    assert sep, "start.sh configuration marker missing"
    return HELPERS + pre


def _config_slice() -> str:
    src = START.read_text()
    _pre, sep, rest = src.partition(MARKER)
    assert sep
    body = rest.split("# ---- D5: dynamic speculative decoding (DSD) ---")[0]
    return HELPERS + _pre + body + '\nprintf "CFG %s %s %s\\n" "${MAX_NUM_BATCHED_TOKENS:-unset}" "${LONG_PREFILL_TOKEN_THRESHOLD:-unset}" "${ASYNC_SCHEDULING:-unset}"\n'


def _write_inner_scripts(tmp: Path) -> tuple[Path, Path]:
    src = START.read_text()
    m = re.search(r"write_inner_scripts\(\) \{\n(.*?)\n\}\n", src, re.S)
    assert m, "write_inner_scripts() not found"
    script = tmp / "gen.sh"
    script.write_text(
        HELPERS
        + "SCRIPT_DIR=" + str(tmp) + "\n"
        + "HEAD_SCRIPT=$SCRIPT_DIR/.head.inner.sh\n"
        + "WORKER_SCRIPT=$SCRIPT_DIR/.worker.inner.sh\n"
        + m.group(1) + "\n"
    )
    script.chmod(0o755)
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return tmp / ".head.inner.sh", tmp / ".worker.inner.sh"


def _run_inner(inner: Path, env: dict[str, str]) -> list[str]:
    """Execute a generated inner script with a stubbed vllm; return argv."""
    bin_dir = inner.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "vllm"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*"\n')
    stub.chmod(0o755)
    model_dir = inner.parent / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text("{}")
    base_env = {
        "SERVED_MODEL_NAME": "GLM-5.3-Flash-EXL3",
        "PORT": "8081",
        "TP": "2",
        "NNODES": "2",
        "HEAD_IP": "10.0.0.1",
        "MASTER_PORT": "29521",
        "MODEL_DIR": str(model_dir),
        "SPEC_METHOD": "dflash",
        "DFLASH_MODEL_DIR": str(model_dir),
        "DFLASH_TOKENS": "7",
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    e = {**base_env, **env}
    r = subprocess.run(["bash", str(inner)], capture_output=True, text=True, env=e)
    assert r.returncode == 0, f"inner script failed: {r.stderr}"
    return r.stdout.strip().split()


def _spec_json(args: list[str]) -> dict:
    i = args.index("--speculative-config")
    return json.loads(args[i + 1])


def test_env_carries_bundle_and_digest_pin() -> None:
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    assert env["IMAGE"] == (
        "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:"
        "9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58"
    ), "digest pin missing/changed in .env"
    assert env.get("SKIP_PULL") == "1", "SKIP_PULL=1 must accompany the digest pin"
    assert env["MAX_NUM_BATCHED_TOKENS"] == "3584"
    assert env["LONG_PREFILL_TOKEN_THRESHOLD"] == "1792"
    assert env["ASYNC_SCHEDULING"] == "0"
    assert env["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"] == "0"
    assert env["FLASHINFER_WORKSPACE_BASE"] == "/root/.cache/vllm/flashinfer"


def test_mnbt_bundle_default_is_3584() -> None:
    src = START.read_text()
    m = re.search(r'MAX_NUM_BATCHED_TOKENS="\$\{MAX_NUM_BATCHED_TOKENS:-(\d+)\}"', src)
    assert m, "MAX_NUM_BATCHED_TOKENS default line missing"
    assert m.group(1) == "3584", f"start.sh MNBT default is {m.group(1)}, want 3584"


def test_async_scheduling_validation() -> None:
    src = START.read_text()
    assert 'case "$ASYNC_SCHEDULING" in' in src
    # config slice dies on a nonsense value, accepts 0/1/auto
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = tmp / "s.sh"
        script.write_text(_config_slice() + '\nprintf "SURVIVED %s\\n" "$ASYNC_SCHEDULING"\n')
        script.chmod(0o755)
        (tmp / ".env").write_text("GPU_MEM_UTIL=0.85\n")
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env={**os.environ, "ASYNC_SCHEDULING": "banana"})
        assert r.returncode != 0 and "banana" in r.stderr, "invalid ASYNC_SCHEDULING accepted"
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env={**os.environ, "ASYNC_SCHEDULING": "auto"})
        assert r.returncode == 0 and "SURVIVED auto" in r.stdout


def test_inner_scripts_carry_bundle_flags() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        head, worker = _write_inner_scripts(tmp)
        env = {**BUNDLE_ENV, "KV_CACHE_MEMORY": "12345678901"}
        for inner in (head, worker):
            args = _run_inner(inner, env)
            s = " ".join(args)
            assert "--max-num-batched-tokens 3584" in s
            assert "--long-prefill-token-threshold 1792" in s
            assert "--no-async-scheduling" in s
            assert "--async-scheduling" not in s
            # pin present + QUOTED form (single arg, no word splitting)
            i = args.index("--kv-cache-memory-bytes")
            assert args[i + 1] == "12345678901"
            # threshold flag must appear BEFORE the EXTRA_ARGS tail markers:
            # it is a scheduler flag emitted directly, never smuggled via
            # EXTRA_ARGS (double-add protection).
            assert args.index("--long-prefill-token-threshold") < len(args) - 2
            cfg = _spec_json(args)
            assert cfg["method"] == "dflash"
            assert "num_speculative_tokens_per_batch_size" not in cfg, "DSD_TABLE absent must stay absent"


def test_inner_scripts_async_and_pin_variants() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        head, worker = _write_inner_scripts(tmp)
        # DSD arm: async ON
        args = _run_inner(head, {**BUNDLE_ENV, "ASYNC_SCHEDULING": "1",
                                 "DSD_TABLE": "1:1:7,2:999:5"})
        s = " ".join(args)
        assert "--async-scheduling" in s and "--no-async-scheduling" not in s
        cfg = _spec_json(args)
        assert cfg["num_speculative_tokens_per_batch_size"] == [[1, 1, 7], [2, 999, 5]]
        # auto: neither flag
        for inner in (head, worker):
            args = _run_inner(inner, {**BUNDLE_ENV, "ASYNC_SCHEDULING": "auto",
                                      "LONG_PREFILL_TOKEN_THRESHOLD": ""})
            s = " ".join(args)
            assert "--async-scheduling" not in s and "--no-async-scheduling" not in s
            assert "--long-prefill-token-threshold" not in s, "empty knob must mean 'off'"
            assert "--kv-cache-memory-bytes" not in s, "empty pin must not be passed"


def test_dsd_validate_requires_async_on() -> None:
    src = START.read_text()
    m = re.search(r"dsd_validate\(\) \{.*?\n\}", src, re.S)
    assert m
    body = m.group(0)
    assert "ASYNC_SCHEDULING" in body and "die" in body, "DSD/async coupling missing"


def test_launch_wiring_env_forwarding() -> None:
    src = START.read_text()
    # serve_env carries the inner-script knobs to BOTH ranks
    assert re.search(r"LONG_PREFILL_TOKEN_THRESHOLD ASYNC_SCHEDULING \\", src)
    # bundle envs are conditional (-e only when non-empty; the fork int()s the
    # retention value, so an empty string would crash boot)
    assert 'bundle_env+=(-e "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=$VLLM_PREFIX_CACHE_RETENTION_INTERVAL")' in src
    assert 'bundle_env+=(-e "FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE")' in src
    assert "${worker_bundle}" in src and '"${bundle_env[@]}"' in src
    # CLI overrides survive the .env source for every bundle knob (set-vs-unset
    # semantics: an explicitly EMPTY inline value overrides the default)
    for cap, capset, var in (("_cli_lpt", "_cli_lpt_set", "LONG_PREFILL_TOKEN_THRESHOLD"),
                             ("_cli_async", "_cli_async_set", "ASYNC_SCHEDULING"),
                             ("_cli_retention", "_cli_retention_set", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"),
                             ("_cli_fiws", "_cli_fiws_set", "FLASHINFER_WORKSPACE_BASE")):
        assert f'{cap}="${{{var}-}}"' in src, f"no CLI capture for {var}"
        assert f'"${{{var}+set}}"' in src, f"no set-vs-unset capture for {var}"
        assert f'[ "${{{capset}:-}}" = set ] && {var}="${cap}"' in src, f"no CLI restore for {var}"


def test_preflight_driver_gate_and_build_guard() -> None:
    src = START.read_text()
    assert "590.x deadlocks CUDAGraph capture on GB10" in src, "driver-branch gate missing"
    assert "driver branch: head=" in src, "driver receipt line missing"
    assert 'case "$IMAGE" in' in src and "cannot tag a digest ref" in src, "BUILD/digest guard missing"


def test_watchdog_pause_wiring() -> None:
    src = START.read_text()
    assert ".watchdog-paused" in src, "stop()/launch sentinel wiring missing"
    # stop writes it, launch clears it
    stop_part = src.split("stop() {")[1].split("\n}", 1)[0]
    launch_part = src.split("launch_cluster() {")[1].split("\n}", 1)[0]
    assert ".watchdog-paused" in stop_part
    assert 'rm -f "$LOGDIR/.watchdog-paused"' in launch_part
    # and the watchdog honors it
    wd = (ROOT / "fleet_watchdog.sh").read_text()
    assert "deliberate stop detected" in wd
    assert "head_running" in wd and "CRASH:" in wd, "crash/wedge distinction missing"
    assert "LIVENESS" in wd, "wedge liveness probe missing"


def test_on_ready_bundle_receipts() -> None:
    src = START.read_text()
    assert "bundle     : lpt=" in src
    # the kv-pin receipt must surface BOTH engine suggestions and mark the
    # full-utilize value as the C4 crash class (never pin it)
    assert "to fit into requested memory" in src, "kv-suggest receipt missing"
    assert "to-fit ${pin_to_fit} − margin" in src, "to-fit recommendation missing"
    assert "C4 crash class" in src, "full-utilize warning missing"


def test_kv_pin_flag_is_the_exact_image_flag() -> None:
    # The pinned image registers --kv-cache-memory-bytes (arg_utils.py); the
    # older abbreviated --kv-cache-memory spelling must not be emitted.
    src = START.read_text()
    assert '--kv-cache-memory-bytes "${KV_CACHE_MEMORY}"' in src
    assert '--kv-cache-memory "${KV_CACHE_MEMORY}"' not in src, "abbreviated flag form still emitted"


def test_pin_guard_hard_refusal() -> None:
    src = START.read_text()
    assert "ALLOW_KV_PIN" in src and "REJECTED on this kit" in src, "hard pin guard missing"
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = tmp / "s.sh"
        script.write_text(_config_slice() + '\nprintf "SURVIVED\\n"\n')
        script.chmod(0o755)
        (tmp / ".env").write_text("GPU_MEM_UTIL=0.85\nKV_CACHE_MEMORY=15724154880\n")
        # no ALLOW_KV_PIN -> refused
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env={**os.environ, "KV_CACHE_MEMORY": "15724154880"})
        assert r.returncode != 0 and "REJECTED" in r.stderr, "pin allowed without ALLOW_KV_PIN"
        # ALLOW_KV_PIN=1 -> survives with a warning
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env={**os.environ, "KV_CACHE_MEMORY": "15724154880", "ALLOW_KV_PIN": "1"})
        assert r.returncode == 0 and "SURVIVED" in r.stdout, "break-glass pin refused"
        assert "ALLOW_KV_PIN=1" in r.stdout + r.stderr
        # auto pool (no pin in .env, no env override) -> survives without any flag
        (tmp / ".env").write_text("GPU_MEM_UTIL=0.85\n")
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env={k: v for k, v in os.environ.items() if k != "KV_CACHE_MEMORY"})
        assert r.returncode == 0 and "SURVIVED" in r.stdout, "auto pool refused"


def test_flusher_stability_exit_and_cap() -> None:
    import http.server
    import threading
    flusher = ROOT / "cache_flusher.sh"
    assert "GLM53_FLUSHER_CAP" in flusher.read_text(), "flusher cap not env-overridable"
    # cap path: health URL unreachable -> exits at the (short) cap with a log line
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        logf = tmp / "flusher.log"
        r = subprocess.run(
            ["bash", str(flusher)], capture_output=True, text=True, timeout=120,
            env={**os.environ, "GLM53_FLUSHER_CAP": "3", "GLM53_FLUSHER_MIN_STABLE": "999999",
                 "GLM53_HEALTH_URL": "http://127.0.0.1:1/health",
                 "GLM53_FLUSHER_LOG": str(logf)},
        )
        assert r.returncode == 0
        assert "cap: " in logf.read_text() and "flusher exiting" in logf.read_text(), "cap exit not logged"
    # stability path: healthy endpoint -> exits after min_stable + 5 probes
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            logf = tmp / "flusher.log"
            r = subprocess.run(
                ["bash", str(flusher)], capture_output=True, text=True, timeout=120,
                env={**os.environ, "GLM53_FLUSHER_CAP": "600", "GLM53_FLUSHER_MIN_STABLE": "2",
                     "GLM53_HEALTH_URL": f"http://127.0.0.1:{srv.server_port}/health",
                     "GLM53_FLUSHER_LOG": str(logf)},
            )
            assert r.returncode == 0
            assert "stable: health OK x5" in logf.read_text(), "stability exit not logged"
    finally:
        srv.shutdown()


def test_bundle_knobs_empty_inline_override_wins() -> None:
    # Baseline-arm requirement: an explicitly EMPTY inline value must override
    # the .env/start.sh default (set-vs-unset semantics on the four bundle
    # knobs) — the boot4 arm contamination showed the old [ -n ] restore
    # silently dropped empty overrides.
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        script = tmp / "s.sh"
        script.write_text(_config_slice()
                          + '\nprintf "GOT %s|%s|%s|%s\\n" '
                            '"${LONG_PREFILL_TOKEN_THRESHOLD-X}" "${ASYNC_SCHEDULING-X}" '
                            '"${VLLM_PREFIX_CACHE_RETENTION_INTERVAL-X}" "${FLASHINFER_WORKSPACE_BASE-X}"\n')
        script.chmod(0o755)
        env_line = ("LONG_PREFILL_TOKEN_THRESHOLD=9999\nASYNC_SCHEDULING=1\n"
                    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=7\nFLASHINFER_WORKSPACE_BASE=/x\n")
        (tmp / ".env").write_text("GPU_MEM_UTIL=0.85\n" + env_line)
        e = {k: v for k, v in os.environ.items()
             if k not in ("LONG_PREFILL_TOKEN_THRESHOLD", "ASYNC_SCHEDULING",
                          "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "FLASHINFER_WORKSPACE_BASE")}
        # inline EMPTY values must win over .env
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env={**e, "LONG_PREFILL_TOKEN_THRESHOLD": "",
                                "ASYNC_SCHEDULING": "",
                                "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "",
                                "FLASHINFER_WORKSPACE_BASE": ""})
        assert r.returncode == 0, r.stderr
        assert "GOT |||" in r.stdout, f"empty inline overrides lost: {r.stdout}"
        # no inline values -> .env wins
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=e)
        assert r.returncode == 0, r.stderr
        assert "GOT 9999|1|7|/x" in r.stdout, f".env values lost: {r.stdout}"


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
