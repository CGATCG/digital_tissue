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
_STATE_DIR = _ROOT / "var" / "runs" / "benchmarks"
_PID_FILE = _STATE_DIR / "streamlit.pid"
_LOG_FILE = _STATE_DIR / "streamlit.log"
_META_FILE = _STATE_DIR / "streamlit_meta.json"
_APP_FILE = _ROOT / "apps" / "benchmarks" / "app.py"


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\x00") if p]
        return " ".join(parts)
    except Exception:
        return ""


def _pid_matches_expected_app(pid: int) -> bool:
    cmd = _proc_cmdline(int(pid))
    if not cmd.strip():
        return False
    return str(_APP_FILE) in cmd


def _read_pid() -> int | None:
    try:
        raw = _PID_FILE.read_text(encoding="utf-8").strip()
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


def _ensure_state_dir() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def _write_pid(pid: int) -> None:
    _ensure_state_dir()
    _PID_FILE.write_text(str(int(pid)) + "\n", encoding="utf-8")


def _clear_pid() -> None:
    try:
        _PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _write_meta(*, pid: int, port: int, address: str) -> None:
    _ensure_state_dir()
    obj = {
        "pid": int(pid),
        "port": int(port),
        "address": str(address),
        "updated_at": float(time.time()),
    }
    _META_FILE.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _clear_meta() -> None:
    try:
        _META_FILE.unlink()
    except FileNotFoundError:
        pass


def _kill_group(pid: int, sig: int) -> None:
    os.killpg(pid, sig)


def start(*, port: int, address: str) -> None:
    if not _APP_FILE.exists():
        raise RuntimeError(f"Missing Streamlit app: {_APP_FILE}")

    pid = _read_pid()
    if pid is not None and _pid_alive(pid):
        if _pid_matches_expected_app(int(pid)):
            print(f"benchmarks already running (pid={pid})")
            print(f"URL: http://{address}:{port}")
            return
        try:
            _kill_group(int(pid), signal.SIGTERM)
        except Exception:
            pass
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if not _pid_alive(int(pid)):
                break
            time.sleep(0.1)
        if _pid_alive(int(pid)):
            try:
                _kill_group(int(pid), signal.SIGKILL)
            except Exception:
                pass
        _clear_pid()
        _clear_meta()

    _clear_pid()
    _ensure_state_dir()

    log_f = open(_LOG_FILE, "ab", buffering=0)

    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_APP_FILE),
        "--server.headless=true",
        f"--server.port={int(port)}",
        f"--server.address={str(address)}",
        "--browser.gatherUsageStats=false",
    ]

    proc = subprocess.Popen(
        args,
        cwd=str(_ROOT),
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
        close_fds=True,
        env=dict(os.environ),
    )
    _write_pid(proc.pid)
    _write_meta(pid=proc.pid, port=int(port), address=str(address))

    time.sleep(0.2)
    print(f"benchmarks started (pid={proc.pid})")
    print(f"URL: http://{address}:{port}")
    print(f"Log: {_LOG_FILE}")


def stop(*, timeout_s: float) -> None:
    pid = _read_pid()
    if pid is None:
        print("benchmarks not running (no pid file)")
        return
    if not _pid_alive(pid):
        _clear_pid()
        _clear_meta()
        print("benchmarks not running (stale pid file removed)")
        return

    try:
        _kill_group(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_pid()
        print("benchmarks already stopped")
        return

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not _pid_alive(pid):
            _clear_pid()
            _clear_meta()
            print("benchmarks stopped")
            return
        time.sleep(0.2)

    try:
        _kill_group(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    _clear_pid()
    _clear_meta()
    print("benchmarks stopped (SIGKILL)")


def pause() -> None:
    pid = _read_pid()
    if pid is None or not _pid_alive(pid):
        print("benchmarks not running")
        return
    _kill_group(pid, signal.SIGSTOP)
    print("benchmarks paused")


def resume() -> None:
    pid = _read_pid()
    if pid is None or not _pid_alive(pid):
        print("benchmarks not running")
        return
    _kill_group(pid, signal.SIGCONT)
    print("benchmarks resumed")


def status(*, port: int, address: str) -> None:
    pid = _read_pid()
    if pid is None:
        print("benchmarks: stopped")
        return
    if not _pid_alive(pid):
        print("benchmarks: stopped (stale pid file)")
        return
    print(f"benchmarks: running (pid={pid})")
    print(f"URL: http://{address}:{port}")
    print(f"Log: {_LOG_FILE}")


def main() -> int:
    try:
        if apply_keys_to_environ is not None:
            apply_keys_to_environ()
    except Exception:
        pass

    p = argparse.ArgumentParser(prog="benchmarks_ctl")
    p.add_argument("command", choices=["start", "stop", "restart", "status", "pause", "resume"])
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--address", type=str, default="0.0.0.0")
    p.add_argument("--timeout", type=float, default=10.0)
    args = p.parse_args()

    if args.command == "start":
        start(port=args.port, address=args.address)
        return 0
    if args.command == "stop":
        stop(timeout_s=float(args.timeout))
        return 0
    if args.command == "restart":
        stop(timeout_s=float(args.timeout))
        start(port=args.port, address=args.address)
        return 0
    if args.command == "pause":
        pause()
        return 0
    if args.command == "resume":
        resume()
        return 0
    if args.command == "status":
        status(port=args.port, address=args.address)
        return 0

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
