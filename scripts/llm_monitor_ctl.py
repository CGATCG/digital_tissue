import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_STATE_DIR = _ROOT / "runs" / "llm_monitor"
_PID_FILE = _STATE_DIR / "streamlit.pid"
_LOG_FILE = _STATE_DIR / "streamlit.log"
_APP_FILE = _ROOT / "llm_monitor" / "app.py"


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


def _kill_group(pid: int, sig: int) -> None:
    os.killpg(pid, sig)


def start(*, port: int, address: str) -> None:
    if not _APP_FILE.exists():
        raise RuntimeError(f"Missing Streamlit app: {_APP_FILE}")

    pid = _read_pid()
    if pid is not None and _pid_alive(pid):
        print(f"llm_monitor already running (pid={pid})")
        print(f"URL: http://{address}:{port}")
        return

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

    time.sleep(0.2)
    print(f"llm_monitor started (pid={proc.pid})")
    print(f"URL: http://{address}:{port}")
    print(f"Log: {_LOG_FILE}")


def stop(*, timeout_s: float) -> None:
    pid = _read_pid()
    if pid is None:
        print("llm_monitor not running (no pid file)")
        return
    if not _pid_alive(pid):
        _clear_pid()
        print("llm_monitor not running (stale pid file removed)")
        return

    try:
        _kill_group(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_pid()
        print("llm_monitor already stopped")
        return

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not _pid_alive(pid):
            _clear_pid()
            print("llm_monitor stopped")
            return
        time.sleep(0.2)

    try:
        _kill_group(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    _clear_pid()
    print("llm_monitor stopped (SIGKILL)")


def pause() -> None:
    pid = _read_pid()
    if pid is None or not _pid_alive(pid):
        print("llm_monitor not running")
        return
    _kill_group(pid, signal.SIGSTOP)
    print("llm_monitor paused")


def resume() -> None:
    pid = _read_pid()
    if pid is None or not _pid_alive(pid):
        print("llm_monitor not running")
        return
    _kill_group(pid, signal.SIGCONT)
    print("llm_monitor resumed")


def status(*, port: int, address: str) -> None:
    pid = _read_pid()
    if pid is None:
        print("llm_monitor: stopped")
        return
    if not _pid_alive(pid):
        print("llm_monitor: stopped (stale pid file)")
        return
    print(f"llm_monitor: running (pid={pid})")
    print(f"URL: http://{address}:{port}")
    print(f"Log: {_LOG_FILE}")


def main() -> int:
    p = argparse.ArgumentParser(prog="llm_monitor_ctl")
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
