import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _default_server_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _check_import(python_exe: str, module: str) -> None:
    p = subprocess.run(
        [python_exe, "-c", f"import {module}"],
        cwd=str(REPO_ROOT),
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Preflight failed: {python_exe} cannot import '{module}'.\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
            "Fix by installing dependencies for the interpreter used to run the server, e.g.\n"
            "  .venv/bin/pip install -r requirements.txt\n"
        )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get_json(base_url: str, path: str, timeout_s: float = 10.0):
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _http_post_json(base_url: str, path: str, payload: dict, timeout_s: float = 30.0):
    url = base_url.rstrip("/") + path
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out_raw = resp.read()
        return json.loads(out_raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read() or b""
        except Exception:
            body = b""
        msg = body.decode("utf-8", errors="replace") if body else str(e)
        raise RuntimeError(f"HTTP {e.code} POST {path} failed: {msg}")


def _wait_for_health(base_url: str, timeout_s: float = 10.0) -> None:
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            out = _http_get_json(base_url, "/api/health", timeout_s=2.0)
            if out and out.get("ok") is True:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(f"Server did not become healthy within {timeout_s}s. Last error: {last_err}")


def _tail_text_file(path: Path, max_bytes: int = 16_384) -> str:
    try:
        if not path.exists():
            return ""
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


class _StepFail(Exception):
    pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise _StepFail(msg)


def _run_steps(base_url: str, docs_dir: Path, workspace_dir: Path) -> None:
    test_doc_name = "smoke_test.json"
    test_doc_path = docs_dir / test_doc_name
    test_doc_path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    st = _http_get_json(base_url, "/api/doc/status")
    _assert(st.get("ok") is True, "status ok")
    _assert(bool(st.get("loaded")) is False, "starts unloaded")

    ls = _http_get_json(base_url, "/api/doc/list")
    _assert(ls.get("ok") is True, "list ok")
    names = [f.get("name") for f in (ls.get("files") or []) if isinstance(f, dict)]
    _assert(test_doc_name in names, "test doc present in list")

    opened = _http_post_json(base_url, "/api/doc/open", {"name": test_doc_name})
    _assert(opened.get("ok") is True, "open ok")
    payload_text = str(opened.get("payload_text") or "")
    _assert("\"a\"" in payload_text, "open payload contains expected content")

    got = _http_post_json(base_url, "/api/doc/get", {})
    _assert(got.get("ok") is True, "get ok")
    _assert(str(got.get("payload_text") or "").strip() == payload_text.strip(), "get payload matches open")

    new_payload = {"b": 2}
    autos = _http_post_json(base_url, "/api/doc/autosave", {"payload_text": json.dumps(new_payload), "path": test_doc_name})
    _assert(autos.get("ok") is True, "autosave ok")

    st2 = _http_get_json(base_url, "/api/doc/status")
    _assert(st2.get("ok") is True, "status ok after autosave")
    _assert(bool(st2.get("dirty")) is True, "dirty after autosave")
    _assert(bool(st2.get("has_autosave")) is True, "has_autosave after autosave")

    rec = _http_post_json(base_url, "/api/doc/recover", {})
    _assert(rec.get("ok") is True, "recover ok")
    rec_payload = json.loads(str(rec.get("payload_text") or "{}"))
    _assert(rec_payload == new_payload, "recover returns latest autosaved payload")

    saved = _http_post_json(base_url, "/api/doc/save", {})
    _assert(saved.get("ok") is True, "save ok")

    save_as_name = "smoke_save_as.json"
    saved_as = _http_post_json(base_url, "/api/doc/save", {"name": save_as_name})
    _assert(saved_as.get("ok") is True, "save as ok")

    st_save_as = _http_get_json(base_url, "/api/doc/status")
    _assert(st_save_as.get("ok") is True, "status ok after save as")
    _assert(str(st_save_as.get("path") or "") == save_as_name, "path updates after save as")

    ls2 = _http_get_json(base_url, "/api/doc/list")
    _assert(ls2.get("ok") is True, "list ok after save as")
    names2 = [f.get("name") for f in (ls2.get("files") or []) if isinstance(f, dict)]
    _assert(save_as_name in names2, "save as file present in list")

    save_as_path = docs_dir / save_as_name
    _assert(save_as_path.exists(), "save as file exists on disk")
    on_disk = json.loads(save_as_path.read_text(encoding="utf-8"))
    _assert(on_disk == new_payload, "save as file has expected payload")

    st3 = _http_get_json(base_url, "/api/doc/status")
    _assert(bool(st3.get("dirty")) is False, "not dirty after save")
    _assert(bool(st3.get("loaded")) is True, "loaded after save")

    cleared = _http_post_json(base_url, "/api/doc/clear", {})
    _assert(cleared.get("ok") is True, "clear ok")
    _assert(bool(cleared.get("loaded")) is False, "unloaded after clear")
    _assert(bool(cleared.get("has_autosave")) is False, "no autosave after clear")

    autosave_file = workspace_dir / "doc_autosave.json"
    _assert(not autosave_file.exists(), "autosave file removed after clear")

    try:
        _http_post_json(base_url, "/api/doc/recover", {})
        raise _StepFail("recover should fail after clear")
    except RuntimeError as e:
        _assert("no autosave" in str(e).lower(), "recover fails with no autosave")

    opened2 = _http_post_json(base_url, "/api/doc/open", {"name": test_doc_name})
    _assert(opened2.get("ok") is True, "open works after clear")

    for i in range(25):
        _http_post_json(base_url, "/api/doc/open", {"name": test_doc_name})
        _http_post_json(base_url, "/api/doc/clear", {})
    _http_post_json(base_url, "/api/doc/open", {"name": test_doc_name})

    delete_other_name = "smoke_delete_other.json"
    (docs_dir / delete_other_name).write_text(json.dumps({"z": 9}), encoding="utf-8")
    ls3 = _http_get_json(base_url, "/api/doc/list")
    _assert(ls3.get("ok") is True, "list ok before delete")
    names3 = [f.get("name") for f in (ls3.get("files") or []) if isinstance(f, dict)]
    _assert(delete_other_name in names3, "delete-other doc present in list")

    st_before_del = _http_get_json(base_url, "/api/doc/status")
    _assert(str(st_before_del.get("path") or "") == test_doc_name, "current path before delete")

    del_other = _http_post_json(base_url, "/api/doc/delete", {"name": delete_other_name})
    _assert(del_other.get("ok") is True, "delete other ok")
    _assert(not (docs_dir / delete_other_name).exists(), "delete other removed file on disk")

    st_after_other = _http_get_json(base_url, "/api/doc/status")
    _assert(bool(st_after_other.get("loaded")) is True, "still loaded after deleting other")
    _assert(str(st_after_other.get("path") or "") == test_doc_name, "still on same doc after deleting other")

    del_cur = _http_post_json(base_url, "/api/doc/delete", {"name": test_doc_name})
    _assert(del_cur.get("ok") is True, "delete current ok")
    _assert(not (docs_dir / test_doc_name).exists(), "delete current removed file on disk")

    st_after_cur = _http_get_json(base_url, "/api/doc/status")
    _assert(bool(st_after_cur.get("loaded")) is False, "unloaded after deleting current")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--python", type=str, default="")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    port = int(args.port or 0)
    if port <= 0:
        port = _pick_free_port()

    python_exe = str(args.python or "").strip() or _default_server_python()
    _check_import(python_exe, "numpy")

    with tempfile.TemporaryDirectory(prefix="dt_docs_") as docs_tmp, tempfile.TemporaryDirectory(prefix="dt_ws_") as ws_tmp:
        docs_dir = Path(docs_tmp)
        workspace_dir = Path(ws_tmp)

        env = dict(os.environ)
        env["DT_DOCS_DIR"] = str(docs_dir)
        env["DT_WORKSPACE_DIR"] = str(workspace_dir)
        env["PYTHONUNBUFFERED"] = "1"

        stdout_path = Path(docs_tmp) / "server_stdout.txt"
        stderr_path = Path(docs_tmp) / "server_stderr.txt"

        stdout_f = open(stdout_path, "wb")
        stderr_f = open(stderr_path, "wb")

        proc = subprocess.Popen(
            [python_exe, str(REPO_ROOT / "runtime_server.py"), str(port)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
        )
        base_url = f"http://127.0.0.1:{port}"

        try:
            try:
                _wait_for_health(base_url, timeout_s=float(args.timeout))
            except Exception as e:
                rc = proc.poll()
                if rc is not None:
                    try:
                        stdout_f.flush()
                        stderr_f.flush()
                    except Exception:
                        pass
                    so = _tail_text_file(stdout_path)
                    se = _tail_text_file(stderr_path)
                    raise RuntimeError(
                        "Server failed to start. "
                        f"exit_code={rc} error={e}\n"
                        f"--- stdout (tail) ---\n{so}\n"
                        f"--- stderr (tail) ---\n{se}\n"
                    )
                raise
            t0 = time.time()
            print(f"OK: server healthy at {base_url} ({(time.time() - t0) * 1000.0:.1f}ms)")

            try:
                _run_steps(base_url, docs_dir=docs_dir, workspace_dir=workspace_dir)
            except _StepFail as e:
                print(f"FAIL: {e}")
                return 2

            print("PASS: doc api smoke test")
            return 0
        finally:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                stdout_f.close()
            except Exception:
                pass
            try:
                stderr_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
