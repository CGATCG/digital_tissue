import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from backend.env_keys import apply_keys_to_environ
except Exception:
    apply_keys_to_environ = None  # type: ignore


_ROOT = Path(__file__).resolve().parents[2]


def _benchmarks_state_dir() -> Path:
    return _ROOT / "var" / "runs" / "benchmarks"


_BENCHMARKS_PID_FILE = _benchmarks_state_dir() / "streamlit.pid"
_BENCHMARKS_META_FILE = _benchmarks_state_dir() / "streamlit_meta.json"


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\x00") if p]
        return " ".join(parts)
    except Exception:
        return ""


def _pid_matches_expected_app(pid: int) -> bool:
    expected = _ROOT / "apps" / "benchmarks" / "app.py"
    cmd = _proc_cmdline(int(pid))
    if not cmd.strip():
        return False
    return str(expected) in cmd


def _read_pid() -> int | None:
    try:
        raw = _BENCHMARKS_PID_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return int(raw)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_monitor_meta() -> dict:
    try:
        if not _BENCHMARKS_META_FILE.exists():
            return {}
        obj = json.loads(_BENCHMARKS_META_FILE.read_text(encoding="utf-8", errors="replace") or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _run_monitor_ctl(args: list[str]) -> int:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.tools.benchmarks_ctl",
            *args,
        ],
        cwd=str(_ROOT),
        env=dict(os.environ),
    )
    return int(proc.returncode)


def _kill_process_group(pid: int, sig: int) -> None:
    os.killpg(int(pid), int(sig))


def _terminate_proc(proc: subprocess.Popen, *, timeout_s: float = 5.0) -> None:
    try:
        if proc.poll() is not None:
            return
        _kill_process_group(proc.pid, signal.SIGTERM)
    except Exception:
        return

    t0 = time.time()
    while time.time() - t0 < float(timeout_s):
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        _kill_process_group(proc.pid, signal.SIGKILL)
    except Exception:
        pass


def main() -> int:
    try:
        if apply_keys_to_environ is not None:
            apply_keys_to_environ()
    except Exception:
        pass

    p = argparse.ArgumentParser(prog="run_ui")
    p.add_argument("--runtime-port", type=int, default=8000)
    p.add_argument("--benchmarks-port", type=int, default=8001)
    p.add_argument("--benchmarks-address", type=str, default="0.0.0.0")
    p.add_argument("--benchmarks-timeout", type=float, default=10.0)
    p.add_argument("--monitor-port", type=int, default=None)
    p.add_argument("--monitor-address", type=str, default="")
    p.add_argument("--monitor-timeout", type=float, default=None)
    args = p.parse_args()

    runtime_port = int(args.runtime_port)
    benchmarks_port = int(args.benchmarks_port)
    benchmarks_address = str(args.benchmarks_address)
    benchmarks_timeout = float(args.benchmarks_timeout)

    if args.monitor_port is not None:
        benchmarks_port = int(args.monitor_port)
    if str(args.monitor_address or "").strip():
        benchmarks_address = str(args.monitor_address)
    if args.monitor_timeout is not None:
        benchmarks_timeout = float(args.monitor_timeout)

    pre_pid = _read_pid()
    monitor_preexisting = (
        pre_pid is not None
        and _pid_alive(int(pre_pid))
        and _pid_matches_expected_app(int(pre_pid))
    )

    if monitor_preexisting:
        meta = _read_monitor_meta()
        try:
            meta_port = int(meta.get("port"))
            if meta_port > 0:
                benchmarks_port = int(meta_port)
        except Exception:
            pass

    os.environ["DT_BENCHMARKS_PORT"] = str(benchmarks_port)
    os.environ["DT_MONITOR_PORT"] = str(benchmarks_port)

    rc = _run_monitor_ctl(["start", "--port", str(benchmarks_port), "--address", benchmarks_address])
    if rc != 0:
        return int(rc)

    env = dict(os.environ)
    env["DT_BENCHMARKS_PORT"] = str(benchmarks_port)
    env["DT_MONITOR_PORT"] = str(benchmarks_port)

    srv = subprocess.Popen(
        [sys.executable, "-m", "backend.runtime_server", str(runtime_port)],
        cwd=str(_ROOT),
        env=env,
        start_new_session=True,
    )

    try:
        return int(srv.wait())
    except KeyboardInterrupt:
        return 0
    finally:
        _terminate_proc(srv)
        if not monitor_preexisting:
            _run_monitor_ctl(["stop", "--timeout", str(float(benchmarks_timeout))])


if __name__ == "__main__":
    raise SystemExit(main())
