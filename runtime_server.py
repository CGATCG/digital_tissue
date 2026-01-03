import concurrent.futures
import faulthandler
import hashlib
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from multiprocessing import shared_memory
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from apply_layer_ops import _decode_float32_b64, _encode_float32_b64, apply_layer_ops_inplace
from output_calc import _ExprEval


_LOG = logging.getLogger("digital_tissue.runtime")
_FAULT_FH: Optional[Any] = None
_EVO_DEBUG = bool(int(os.environ.get("DT_EVO_DEBUG", "0") or "0"))


def _setup_logging() -> None:
    if _LOG.handlers:
        return

    _LOG.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(stream=sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    _LOG.addHandler(sh)

    try:
        log_path = (Path(__file__).resolve().parent / "runtime_server.log").resolve()
        fh = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        _LOG.addHandler(fh)
    except Exception:
        pass

    global _FAULT_FH
    try:
        fault_path = (Path(__file__).resolve().parent / "runtime_server_faulthandler.log").resolve()
        _FAULT_FH = open(fault_path, "a", encoding="utf-8")
    except Exception:
        _FAULT_FH = None


def _install_exception_hooks() -> None:
    def _sys_hook(exc_type, exc, tb):
        try:
            _LOG.error("Uncaught exception", exc_info=(exc_type, exc, tb))
        except Exception:
            pass

    sys.excepthook = _sys_hook

    def _thread_hook(args):
        try:
            _LOG.error(
                "Uncaught thread exception in %s", str(getattr(args, "thread", None)),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        except Exception:
            pass

    try:
        threading.excepthook = _thread_hook  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        if _FAULT_FH is not None:
            faulthandler.enable(file=_FAULT_FH, all_threads=True)
        else:
            faulthandler.enable(all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass


def _compute_distribution_score(values: list[float], method: str = "entropy") -> float:
    """
    Compute a distribution score for a list of per-tick values.
    Higher score = more evenly distributed across ticks.
    
    Methods:
    - entropy: Shannon entropy (normalized)
    - cv: 1 / (1 + coefficient of variation)
    - spread: fraction of non-zero ticks
    """
    if not values or len(values) < 2:
        return 0.0
    
    arr = np.array(values, dtype=np.float64)
    total = float(arr.sum())
    
    if total <= 0:
        return 0.0
    
    if method == "entropy":
        probs = arr / total
        probs = probs[probs > 0]
        if len(probs) <= 1:
            return 0.0
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(values))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    elif method == "cv":
        mean = float(arr.mean())
        if mean <= 0:
            return 0.0
        std = float(arr.std())
        cv = std / mean
        return float(1.0 / (1.0 + cv))
    
    elif method == "spread":
        nonzero = int((arr > 0).sum())
        return float(nonzero) / float(len(values))
    
    return 0.0


def _decoded_layers_and_kinds(payload: Dict[str, Any]) -> tuple[int, int, Dict[str, Any], Dict[str, str]]:
    H = int(payload.get("H") or 0)
    W = int(payload.get("W") or 0)
    if H <= 0 or W <= 0:
        return 0, 0, {}, {}

    kinds: Dict[str, str] = {}
    layer_meta = payload.get("layers")
    if isinstance(layer_meta, list):
        for m in layer_meta:
            if not isinstance(m, dict):
                continue
            nm = m.get("name")
            if isinstance(nm, str) and nm:
                kinds[nm] = str(m.get("kind") or "continuous")

    data = payload.get("data")
    if not isinstance(data, dict):
        return H, W, {}, kinds

    layers: Dict[str, Any] = {}
    for name, entry in data.items():
        if not isinstance(name, str):
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("dtype") != "float32":
            continue
        b64 = entry.get("b64")
        if not isinstance(b64, str) or not b64:
            continue
        try:
            layers[name] = _decode_float32_b64(b64, expected_len=H * W, layer_name=name)
        except Exception:
            continue

    return H, W, layers, kinds


def _compute_layer_scalars_from_layers(layers: Dict[str, Any], kinds: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, arr in layers.items():
        try:
            s = float(arr.sum())
            m = float(arr.mean())
            nz = int((arr != 0).sum())
            if kinds.get(name) == "categorical":
                eq1 = int((arr == 1).sum())
                out[name] = {"sum": s, "mean": m, "nonzero": nz, "eq1": eq1}
            else:
                out[name] = {"sum": s, "mean": m, "nonzero": nz}
        except Exception:
            continue
    return out


def _compute_measurements_from_layers(payload: Dict[str, Any], layers: Dict[str, Any], H: int, W: int) -> Dict[str, Any]:
    cfg = payload.get("measurements_config")
    if not isinstance(cfg, dict):
        return {}
    if int(cfg.get("version") or 0) != 3:
        return {}
    measurements = cfg.get("measurements")
    if not isinstance(measurements, list):
        return {}

    ev = _ExprEval(layers=layers, H=H, W=W)
    out: Dict[str, Any] = {}
    for m in measurements:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        expr = str(m.get("expr") or "").strip()
        if not name or not expr:
            continue
        try:
            v = ev.eval(expr)
            out[name] = v
        except Exception:
            out[name] = None
    return out


def _compute_selected_measurements_from_layers(
    payload: Dict[str, Any],
    layers: Dict[str, Any],
    H: int,
    W: int,
    selected: set[str],
) -> Dict[str, Any]:
    if not selected:
        return {}
    cfg = payload.get("measurements_config")
    if not isinstance(cfg, dict):
        return {}
    if int(cfg.get("version") or 0) != 3:
        return {}
    measurements = cfg.get("measurements")
    if not isinstance(measurements, list):
        return {}

    ev = _ExprEval(layers=layers, H=H, W=W)
    out: Dict[str, Any] = {}
    for m in measurements:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name or name not in selected:
            continue
        expr = str(m.get("expr") or "").strip()
        if not expr:
            continue
        try:
            out[name] = ev.eval(expr)
        except Exception:
            out[name] = None
    return out


_WEB_DIR = Path(__file__).resolve().parent / "web_editor"
_RUNS_DIR = Path(__file__).resolve().parent / "runs" / "evolution"
_DOCS_DIR = Path(os.environ.get("DT_DOCS_DIR") or (Path(__file__).resolve().parent / "documents"))
_WORKSPACE_DIR = Path(os.environ.get("DT_WORKSPACE_DIR") or (Path(__file__).resolve().parent / "workspace"))


def _find_cell_layer_name(payload: Dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        return ""
    for nm in ("cell", "cell_type"):
        ent = data.get(nm)
        if isinstance(ent, dict) and ent.get("dtype") == "float32" and isinstance(ent.get("b64"), str):
            return nm
    return ""


def _deepcopy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload))


def _safe_read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _ensure_dirs() -> None:
    try:
        _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _docs_safe_path(name: str) -> Path:
    nm = str(name or "").strip()
    if not nm:
        raise ValueError("missing name")
    if "\x00" in nm:
        raise ValueError("bad name")
    if nm.startswith("/") or nm.startswith("\\"):
        raise ValueError("bad name")
    if ":" in nm:
        raise ValueError("bad name")
    p = Path(nm)
    if p.is_absolute():
        raise ValueError("bad name")
    if any(part in ("..", "") for part in p.parts):
        raise ValueError("bad name")
    out = (_DOCS_DIR / p).resolve()
    base = _DOCS_DIR.resolve()
    if str(out).startswith(str(base) + os.sep) or out == base:
        return out
    raise ValueError("bad name")


class _DocWorkspace:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.doc_id = ""
        self.path = ""
        self.payload_text = ""
        self.payload_hash = ""
        self.last_saved_hash = ""
        self.last_autosave_ts = 0.0
        self.last_saved_ts = 0.0
        self.has_autosave = False

        try:
            _ensure_dirs()
        except Exception:
            pass
        try:
            self._load_meta_from_disk()
        except Exception:
            pass

    def _load_meta_from_disk(self) -> None:
        mp = self._meta_path()
        dd = _safe_read_json(mp)
        if isinstance(dd, dict):
            self.doc_id = str(dd.get("doc_id") or "")
            self.path = str(dd.get("path") or "")
            self.payload_hash = str(dd.get("payload_hash") or "")
            self.last_saved_hash = str(dd.get("last_saved_hash") or "")
            try:
                self.last_autosave_ts = float(dd.get("last_autosave_ts") or 0.0)
            except Exception:
                self.last_autosave_ts = 0.0
            try:
                self.last_saved_ts = float(dd.get("last_saved_ts") or 0.0)
            except Exception:
                self.last_saved_ts = 0.0
        # Auto-load autosave payload on startup for seamless restore (Google Docs-like)
        ap = self._autosave_path()
        self.has_autosave = ap.exists()
        if self.has_autosave and not self.payload_text:
            try:
                txt = ap.read_text(encoding="utf-8")
                if txt.strip():
                    self.payload_text = txt
                    self.payload_hash = _sha256_text(txt)
                    if not self.doc_id:
                        self.doc_id = uuid.uuid4().hex
            except Exception:
                pass

    def _meta_path(self) -> Path:
        return _WORKSPACE_DIR / "doc_meta.json"

    def _autosave_path(self) -> Path:
        return _WORKSPACE_DIR / "doc_autosave.json"

    def _write_meta(self) -> None:
        meta = {
            "doc_id": self.doc_id,
            "path": self.path,
            "payload_hash": self.payload_hash,
            "last_saved_hash": self.last_saved_hash,
            "last_autosave_ts": self.last_autosave_ts,
            "last_saved_ts": self.last_saved_ts,
            "has_autosave": self.has_autosave,
        }
        _atomic_write_json(self._meta_path(), meta)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            dirty = bool(self.payload_hash and self.payload_hash != self.last_saved_hash)
            has_autosave = bool(self.has_autosave) or bool(self._autosave_path().exists())
            return {
                "ok": True,
                "loaded": bool(self.payload_text),
                "doc_id": self.doc_id,
                "path": self.path,
                "dirty": dirty,
                "has_autosave": has_autosave,
                "last_autosave_ts": float(self.last_autosave_ts),
                "last_saved_ts": float(self.last_saved_ts),
            }

    def clear_active(self) -> None:
        with self._lock:
            self.doc_id = ""
            self.path = ""
            self.payload_text = ""
            self.payload_hash = ""
            self.last_saved_hash = ""
            self.last_autosave_ts = 0.0
            self.last_saved_ts = 0.0
            ap = self._autosave_path()
            try:
                if ap.exists():
                    ap.unlink()
            except Exception:
                pass
            self.has_autosave = False
            self._write_meta()

    def set_payload_from_text(self, payload_text: str, path: str = "") -> Dict[str, Any]:
        txt = str(payload_text or "")
        if not txt.strip():
            raise ValueError("missing payload")
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            raise ValueError("payload must be object")
        h = _sha256_text(txt)
        with self._lock:
            if not self.doc_id:
                self.doc_id = uuid.uuid4().hex
            self.payload_text = txt
            self.payload_hash = h
            if isinstance(path, str) and path.strip():
                self.path = str(path).strip()
            self.last_autosave_ts = time.time()
            self.has_autosave = True
            _atomic_write_text(self._autosave_path(), txt)
            self._write_meta()
            dirty = bool(self.payload_hash and self.payload_hash != self.last_saved_hash)
            return {"ok": True, "doc_id": self.doc_id, "path": self.path, "dirty": dirty}

    def recover_autosave(self) -> Dict[str, Any]:
        ap = self._autosave_path()
        if not ap.exists():
            raise ValueError("no autosave")
        try:
            txt = ap.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(str(e))
        self.set_payload_from_text(txt)
        with self._lock:
            return {"ok": True, "payload_text": self.payload_text, **self.status()}

    def get_payload_text(self) -> str:
        with self._lock:
            return str(self.payload_text or "")

    def open_doc(self, name: str) -> Dict[str, Any]:
        path = _docs_safe_path(name)
        if not path.exists():
            raise ValueError("file not found")
        try:
            txt = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(str(e))
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            raise ValueError("file must be json object")
        h = _sha256_text(txt)
        with self._lock:
            self.doc_id = uuid.uuid4().hex
            self.path = str(Path(name).as_posix())
            self.payload_text = txt
            self.payload_hash = h
            self.last_saved_hash = h
            self.last_saved_ts = time.time()
            self.last_autosave_ts = time.time()
            self.has_autosave = True
            _atomic_write_text(self._autosave_path(), txt)
            self._write_meta()
            return {"ok": True, "payload_text": self.payload_text, **self.status()}

    def save_doc(self, name: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            txt = str(self.payload_text or "")
            if not txt.strip():
                raise ValueError("no active document")
            cur_path = str(self.path or "")
        target_name = (str(name).strip() if isinstance(name, str) else "") or cur_path
        if not target_name:
            raise ValueError("missing name")
        if not target_name.lower().endswith(".json"):
            target_name = target_name + ".json"
        path = _docs_safe_path(target_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        _atomic_write_text(path, txt)
        h = _sha256_text(txt)
        with self._lock:
            self.path = str(Path(target_name).as_posix())
            self.last_saved_hash = h
            self.last_saved_ts = time.time()
            self._write_meta()
            return {"ok": True, "path": self.path, **self.status()}

    def delete_doc(self, name: str) -> Dict[str, Any]:
        target_name = str(name or "").strip()
        if not target_name:
            raise ValueError("missing name")
        if not target_name.lower().endswith(".json"):
            target_name = target_name + ".json"
        path = _docs_safe_path(target_name)
        if not path.exists():
            raise ValueError("file not found")

        cur_path = ""
        with self._lock:
            cur_path = str(self.path or "")

        try:
            path.unlink()
        except Exception as e:
            raise ValueError(str(e))

        if path.exists():
            raise ValueError("delete failed")

        if cur_path and cur_path == str(Path(target_name).as_posix()):
            self.clear_active()
            return {"ok": True, "deleted": str(Path(target_name).as_posix()), **self.status()}
        return {"ok": True, "deleted": str(Path(target_name).as_posix())}

    def list_docs(self) -> Dict[str, Any]:
        base = _DOCS_DIR
        out = []
        try:
            for p in base.rglob("*.json"):
                try:
                    rel = p.relative_to(base).as_posix()
                    st = p.stat()
                    out.append({"name": rel, "size": int(st.st_size), "mtime": float(st.st_mtime)})
                except Exception:
                    continue
        except Exception:
            pass
        out.sort(key=lambda x: str(x.get("name") or ""))
        return {"ok": True, "files": out}


_DOC = _DocWorkspace()


def _evo_runs_ensure_dir() -> None:
    try:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _evo_state_path() -> Path:
    return _RUNS_DIR / "state.json"


def _evo_cfg_path() -> Path:
    return _RUNS_DIR / "cfg.json"


def _evo_base_payload_path() -> Path:
    return _RUNS_DIR / "base_payload.json"


def _evo_candidates_path() -> Path:
    return _RUNS_DIR / "candidates.json"


def _evo_stop_flag_path() -> Path:
    return _RUNS_DIR / "stop.flag"


def _evo_extract_measurement_defs(payload: Dict[str, Any]) -> list[Dict[str, str]]:
    measurements_cfg = payload.get("measurements_config")
    if not isinstance(measurements_cfg, dict) or int(measurements_cfg.get("version") or 0) != 3:
        return []
    meas_list = measurements_cfg.get("measurements")
    if not isinstance(meas_list, list):
        return []
    out: list[Dict[str, str]] = []
    for m in meas_list:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        expr = str(m.get("expr") or "").strip()
        if not name or not expr:
            continue
        out.append({"name": name, "expr": expr})
    return out


def _evo_resolve_target_layers(payload: Dict[str, Any], patterns: list[str]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    pats = [str(p).strip() for p in (patterns or []) if isinstance(p, str) and str(p).strip()]
    if not pats:
        pats = ["gene_*", "rna_*", "protein_*"]
    import fnmatch

    out: list[str] = []
    for name, ent in data.items():
        if not isinstance(name, str) or not name:
            continue
        if not any(fnmatch.fnmatch(name, pat) for pat in pats):
            continue
        if not isinstance(ent, dict):
            continue
        if ent.get("dtype") != "float32" or not isinstance(ent.get("b64"), str):
            continue
        out.append(name)
    out.sort()
    return out


def _evo_start_runner(payload: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    _evo_runs_ensure_dir()

    pid = _evo_runner_pid()
    if _pid_alive(pid):
        if not _pid_looks_like_evo_runner(pid):
            try:
                _atomic_write_json(_RUNS_DIR / "runner.pid", {"pid": 0, "stale_pid": int(pid), "cleared_at": float(time.time())})
            except Exception:
                pass
        else:
            try:
                if _evo_stop_flag_path().exists():
                    t0 = time.time()
                    while time.time() - t0 < 2.0:
                        if not _pid_alive(pid):
                            break
                        time.sleep(0.05)
            except Exception:
                pass
            if _pid_alive(pid):
                raise ValueError("evolution already running")

    try:
        if _evo_stop_flag_path().exists():
            _evo_stop_flag_path().unlink()
    except Exception:
        pass

    job_id = uuid.uuid4().hex
    cfg2 = dict(cfg)
    cfg2["job_id"] = job_id

    meas_defs = _evo_extract_measurement_defs(payload)
    if meas_defs:
        cfg2["measurement_defs"] = meas_defs

    tgt = cfg2.get("target_layers")
    if not isinstance(tgt, list):
        tgt = ["gene_*", "rna_*", "protein_*"]
        cfg2["target_layers"] = tgt
    resolved = _evo_resolve_target_layers(payload, [str(p) for p in tgt if isinstance(p, str)])
    cfg2["target_layers_resolved"] = resolved

    _atomic_write_json(_evo_cfg_path(), cfg2)
    _atomic_write_json(_evo_base_payload_path(), payload)
    try:
        _atomic_write_json(_evo_candidates_path(), {})
    except Exception:
        pass

    state0 = {
        "ok": True,
        "job_id": job_id,
        "running": True,
        "error": "",
        "cfg": cfg2,
        "progress": {},
        "history": {"best": [], "mean": [], "median": [], "p10": [], "p90": []},
        "baseline": {},
        "series": {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []},
        "perf": {},
        "top": [],
    }
    try:
        _atomic_write_json(_evo_state_path(), state0)
    except Exception:
        pass

    runner_path = (Path(__file__).resolve().parent / "evolution_runner.py").resolve()
    if not runner_path.exists():
        raise ValueError("missing evolution_runner.py")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        subprocess.Popen(
            [sys.executable, str(runner_path), "--dir", str(_RUNS_DIR)],
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        raise ValueError(str(e))

    return str(job_id)


def _evo_stop_runner() -> None:
    _evo_runs_ensure_dir()
    try:
        _evo_stop_flag_path().touch(exist_ok=True)
    except Exception:
        pass

    pid = _evo_runner_pid()
    if not _pid_alive(pid):
        return

    if not _pid_looks_like_evo_runner(pid):
        try:
            _atomic_write_json(_RUNS_DIR / "runner.pid", {"pid": 0, "stale_pid": int(pid), "cleared_at": float(time.time())})
        except Exception:
            pass
        return

    sent = False
    try:
        os.killpg(int(pid), signal.SIGTERM)
        sent = True
    except Exception:
        pass

    if not sent:
        try:
            os.kill(int(pid), signal.SIGTERM)
            sent = True
        except Exception:
            pass

    if not sent:
        return

    t0 = time.time()
    while time.time() - t0 < 1.5:
        if not _pid_alive(pid):
            try:
                os.waitpid(int(pid), os.WNOHANG)
            except Exception:
                pass
            return
        time.sleep(0.05)

    try:
        os.killpg(int(pid), signal.SIGKILL)
        sent = True
    except Exception:
        pass

    try:
        os.kill(int(pid), signal.SIGKILL)
        sent = True
    except Exception:
        pass

    if not sent:
        return

    t1 = time.time()
    while time.time() - t1 < 2.0:
        if not _pid_alive(pid):
            break
        time.sleep(0.05)

    try:
        os.waitpid(int(pid), os.WNOHANG)
    except Exception:
        pass

    if _pid_is_zombie(pid):
        try:
            os.waitpid(int(pid), os.WNOHANG)
        except Exception:
            pass

    if not _pid_alive(pid):
        try:
            cur = _safe_read_json(_RUNS_DIR / "runner.pid")
            if isinstance(cur, dict) and int(cur.get("pid") or 0) == int(pid):
                _atomic_write_json(_RUNS_DIR / "runner.pid", {"pid": 0, "started_at": float(cur.get("started_at") or 0.0), "cleared_at": float(time.time())})
        except Exception:
            pass


def _evo_status_from_disk() -> Dict[str, Any]:
    _evo_runs_ensure_dir()
    st = _safe_read_json(_evo_state_path())
    if isinstance(st, dict):
        pid = 0
        try:
            pid = int(st.get("runner_pid") or 0)
        except Exception:
            pid = 0
        if pid <= 0:
            pid = _evo_runner_pid()
        if pid and not _pid_alive(pid):
            st = dict(st)
            st["running"] = False
            st["runner_pid"] = int(pid)
        return st

    pid = _evo_runner_pid()
    if _pid_alive(pid):
        return {
            "ok": True,
            "job_id": "",
            "running": True,
            "error": "",
            "cfg": {},
            "progress": {},
            "history": {"best": [], "mean": [], "median": [], "p10": [], "p90": []},
            "baseline": {},
            "series": {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []},
            "perf": {},
            "top": [],
            "runner_pid": int(pid),
        }

    return {
        "ok": True,
        "job_id": "",
        "running": False,
        "error": "",
        "cfg": {},
        "progress": {},
        "history": {"best": [], "mean": [], "median": [], "p10": [], "p90": []},
        "baseline": {},
        "series": {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []},
        "perf": {},
        "top": [],
    }


def _evo_runner_pid() -> int:
    dd = _safe_read_json(_RUNS_DIR / "runner.pid")
    if isinstance(dd, dict):
        try:
            return int(dd.get("pid") or 0)
        except Exception:
            return 0
    return 0


def _pid_is_zombie(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="ignore")
        # /proc/<pid>/stat format: pid (comm) state ...
        parts = stat.split()
        if len(parts) >= 3 and parts[2] == "Z":
            return True
    except Exception:
        return False
    return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _pid_is_zombie(pid):
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _pid_looks_like_evo_runner(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        s = cmd.decode("utf-8", errors="ignore")
        return "evolution_runner.py" in s
    except Exception:
        return False


def _evo_build_candidate_payload(
    base: Dict[str, Any],
    cfg: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    genome = candidate.get("genome")
    if not isinstance(genome, dict):
        raise ValueError("candidate genome missing")

    huge = float((cfg or {}).get("huge") or 1e9)
    H = int(base.get("H") or 0)
    W = int(base.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("base payload invalid H/W")

    kinds: Dict[str, str] = {}
    layer_meta = base.get("layers")
    if isinstance(layer_meta, list):
        for m in layer_meta:
            if not isinstance(m, dict):
                continue
            nm = m.get("name")
            if isinstance(nm, str) and nm:
                kinds[nm] = str(m.get("kind") or "continuous")

    cell_layer = _find_cell_layer_name(base)
    cell_mask = None
    if cell_layer:
        dd0 = base.get("data")
        if isinstance(dd0, dict):
            ent0 = dd0.get(cell_layer)
            if isinstance(ent0, dict) and ent0.get("dtype") == "float32" and isinstance(ent0.get("b64"), str):
                try:
                    cell_arr = _decode_float32_b64(
                        str(ent0.get("b64") or ""), expected_len=H * W, layer_name=cell_layer
                    )
                    cell_mask = np.asarray(cell_arr, dtype=np.float32).reshape(H * W) > 0.5
                except Exception:
                    cell_mask = None

    payload = _deepcopy_payload(base)
    payload.pop("event_counters", None)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("candidate payload missing data")

    for nm, gb in genome.items():
        if not isinstance(nm, str) or not nm:
            continue
        if not isinstance(gb, dict):
            continue
        ent = data.get(nm)
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str) or not b64:
            continue
        arr = _decode_float32_b64(b64, expected_len=H * W, layer_name=nm)
        if "delta_b64" in gb and isinstance(gb.get("delta_b64"), str):
            db64 = str(gb.get("delta_b64") or "")
            delta = _decode_float32_b64(db64, expected_len=H * W, layer_name=f"{nm}:delta")
            arr2 = np.asarray(arr + delta, dtype=np.float32)
        else:
            s = float(gb.get("scale", 1.0))
            b = float(gb.get("bias", 0.0))
            arr2 = np.asarray(arr * s + b, dtype=np.float32)
            if isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr2.size:
                arr2[~cell_mask] = np.asarray(arr, dtype=np.float32).reshape(-1)[~cell_mask]
        arr2 = np.nan_to_num(arr2, nan=0.0, posinf=huge, neginf=0.0)
        arr2 = np.clip(arr2, 0.0, huge)
        if kinds.get(nm) == "counts":
            arr2 = np.clip(np.rint(arr2), 0.0, huge)
        ent["b64"] = _encode_float32_b64(arr2)

    return payload


_EVO_WORKER_BASE: Optional[Dict[str, Any]] = None
_EVO_WORKER_BASE_DATA: Optional[Dict[str, np.ndarray]] = None
_EVO_WORKER_PAYLOAD: Optional[Dict[str, Any]] = None
_EVO_WORKER_DATA: Optional[Dict[str, Any]] = None
_EVO_WORKER_KINDS: Optional[Dict[str, str]] = None
_EVO_WORKER_CELL_LAYER: str = ""
_EVO_WORKER_CELL_MASK: Optional[np.ndarray] = None
_EVO_WORKER_HUGE: float = 1e9

_EVO_CEM_MUTABLE_NAMES: Optional[list[str]] = None
_EVO_CEM_MU: Optional[np.ndarray] = None
_EVO_CEM_SIG: Optional[np.ndarray] = None
_EVO_CEM_MASK_F: Optional[np.ndarray] = None
_EVO_CEM_MU_SHM: Optional[shared_memory.SharedMemory] = None
_EVO_CEM_SIG_SHM: Optional[shared_memory.SharedMemory] = None


def _evo_worker_prepare_process() -> None:
    try:
        nice = int(os.environ.get("DT_WORKER_NICE", "10") or "10")
        if nice != 0:
            os.nice(int(nice))
    except Exception:
        pass

    try:
        port = int(os.environ.get("DT_RUNTIME_PORT", "8000") or "8000")
    except Exception:
        port = 8000

    try:
        for fd_name in os.listdir("/proc/self/fd"):
            try:
                fd = int(fd_name)
            except Exception:
                continue
            if fd < 3:
                continue
            try:
                s = socket.socket(fileno=fd)
            except Exception:
                continue
            try:
                if s.family not in (socket.AF_INET, socket.AF_INET6):
                    continue
                try:
                    acc = int(s.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) or 0)
                except Exception:
                    acc = 0
                if not acc:
                    continue
                try:
                    addr = s.getsockname()
                except Exception:
                    continue
                if isinstance(addr, tuple) and len(addr) >= 2 and int(addr[1]) == int(port):
                    try:
                        s.close()
                    except Exception:
                        pass
            finally:
                try:
                    if s.fileno() != -1:
                        s.detach()
                except Exception:
                    pass
    except Exception:
        pass


def _evo_worker_init(
    base_payload_fast: Dict[str, Any],
    kinds: Dict[str, str],
    cell_layer: str,
    huge: float,
) -> None:
    global _EVO_WORKER_BASE, _EVO_WORKER_BASE_DATA, _EVO_WORKER_PAYLOAD, _EVO_WORKER_DATA
    global _EVO_WORKER_KINDS, _EVO_WORKER_CELL_LAYER, _EVO_WORKER_CELL_MASK, _EVO_WORKER_HUGE

    _evo_worker_prepare_process()

    _EVO_WORKER_BASE = base_payload_fast
    _EVO_WORKER_KINDS = dict(kinds)
    _EVO_WORKER_CELL_LAYER = str(cell_layer)
    _EVO_WORKER_HUGE = float(huge)

    dd = base_payload_fast.get("data")
    if not isinstance(dd, dict):
        raise ValueError("worker init: base payload missing data")
    base_data: Dict[str, np.ndarray] = {}
    for nm, ent in dd.items():
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        arr = ent.get("arr")
        if not isinstance(arr, np.ndarray):
            continue
        base_data[str(nm)] = np.asarray(arr, dtype=np.float32).reshape(-1)
    _EVO_WORKER_BASE_DATA = base_data

    if cell_layer and cell_layer in base_data:
        cm = np.asarray(base_data.get(cell_layer), dtype=np.float32).reshape(-1) > 0.5
        _EVO_WORKER_CELL_MASK = cm
    else:
        _EVO_WORKER_CELL_MASK = None

    p = dict(base_payload_fast)
    out_data: Dict[str, Any] = {}
    for nm, arr in base_data.items():
        out_data[nm] = {"dtype": "float32", "arr": arr.copy()}
    p["data"] = out_data
    p["_skip_b64_writeback"] = True
    _EVO_WORKER_PAYLOAD = p
    _EVO_WORKER_DATA = out_data


def _evo_fitness_from_metrics(metrics: Dict[str, Any], fitness_w: Dict[str, Any]) -> float:
    meas_weights = fitness_w.get("measurements") if isinstance(fitness_w, dict) else None
    if isinstance(meas_weights, dict):
        mm = metrics.get("measurements") if isinstance(metrics, dict) else None
        if isinstance(mm, dict):
            fit = 0.0
            for meas_name, w in meas_weights.items():
                if meas_name in mm and isinstance(w, (int, float)):
                    fit += float(w) * float(mm.get(meas_name) or 0.0)
            return float(fit)
        return 0.0
    return 0.0


def _evo_worker_eval_affine(
    gen: int,
    vi: int,
    genome: Dict[str, Dict[str, float]],
    seed: int,
    ticks: int,
    replicates: int,
    fitness_w: Dict[str, float],
) -> Dict[str, Any]:
    if _EVO_WORKER_PAYLOAD is None or _EVO_WORKER_DATA is None or _EVO_WORKER_BASE_DATA is None:
        raise RuntimeError("worker not initialized")

    kinds = _EVO_WORKER_KINDS or {}
    huge = float(_EVO_WORKER_HUGE)
    cell_layer = str(_EVO_WORKER_CELL_LAYER)
    cell_mask = _EVO_WORKER_CELL_MASK

    p = _EVO_WORKER_PAYLOAD
    dd = _EVO_WORKER_DATA

    rep_metrics = []
    rep_per_tick_events = []
    rep_measurements = []
    
    for ri in range(int(replicates)):
        seed0 = int(seed) + (int(gen) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)

        p.pop("event_counters", None)
        for nm, src in _EVO_WORKER_BASE_DATA.items():
            ent = dd.get(nm)
            if not isinstance(ent, dict):
                continue
            dst = ent.get("arr")
            if not isinstance(dst, np.ndarray):
                continue
            np.copyto(dst, src)

        for nm, gb in genome.items():
            if not isinstance(nm, str) or not nm:
                continue
            ent = dd.get(nm)
            if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                continue
            arr = ent.get("arr")
            if not isinstance(arr, np.ndarray):
                continue

            s = float(gb.get("scale", 1.0))
            b = float(gb.get("bias", 0.0))
            arr *= np.float32(s)
            arr += np.float32(b)
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=huge, neginf=0.0)
            np.clip(arr, 0.0, huge, out=arr)
            if kinds.get(nm) == "counts":
                np.rint(arr, out=arr)
                np.clip(arr, 0.0, huge, out=arr)

            if isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr.size:
                src = _EVO_WORKER_BASE_DATA.get(nm) if isinstance(_EVO_WORKER_BASE_DATA, dict) else None
                if isinstance(src, np.ndarray) and src.shape[0] == arr.size:
                    arr[~cell_mask] = src[~cell_mask]

        H = int(p.get("H") or 0)
        W = int(p.get("W") or 0)

        meas_aggs = fitness_w.get("measurement_aggs", {})
        if not isinstance(meas_aggs, dict):
            meas_aggs = {}
        agg_modes = {
            str(k): str(v)
            for k, v in meas_aggs.items()
            if isinstance(k, str) and str(v) in ("mean", "median")
        }
        agg_names = set(agg_modes.keys())
        per_tick_meas: Dict[str, list[float]] = {k: [] for k in agg_names}

        per_tick_divs = []
        per_tick_starv = []
        per_tick_dmg = []
        
        for t in range(int(ticks)):
            apply_layer_ops_inplace(p, seed_offset=seed0 + int(t))
            
            events = p.get("event_counters") if isinstance(p, dict) else None
            last_events = events.get("last") if isinstance(events, dict) else None
            if isinstance(last_events, dict):
                per_tick_divs.append(int(last_events.get("divisions") or 0))
                per_tick_starv.append(int(last_events.get("starvation_deaths") or 0))
                per_tick_dmg.append(int(last_events.get("damage_deaths") or 0))
            else:
                per_tick_divs.append(0)
                per_tick_starv.append(0)
                per_tick_dmg.append(0)

            if agg_names:
                layers_dict_tick: Dict[str, np.ndarray] = {}
                for nm2, ent2 in dd.items():
                    if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                        arr2 = ent2.get("arr")
                        if isinstance(arr2, np.ndarray):
                            layers_dict_tick[nm2] = arr2

                sel = _compute_selected_measurements_from_layers(p, layers_dict_tick, H, W, agg_names)
                for nm in agg_names:
                    try:
                        per_tick_meas[nm].append(float(sel.get(nm) or 0.0))
                    except Exception:
                        per_tick_meas[nm].append(0.0)

        cell_ent2 = dd.get(cell_layer)
        if not isinstance(cell_ent2, dict) or cell_ent2.get("dtype") != "float32":
            raise ValueError("payload missing cell layer")
        cell_arr = cell_ent2.get("arr")
        if not isinstance(cell_arr, np.ndarray):
            raise ValueError("payload missing cell layer array")

        alive = int((np.asarray(cell_arr, dtype=np.float32).reshape(-1) > 0.5).sum())
        events = p.get("event_counters") if isinstance(p, dict) else None
        totals = events.get("totals") if isinstance(events, dict) else None
        if not isinstance(totals, dict):
            totals = {}
        
        layers_dict_end: Dict[str, np.ndarray] = {}
        for nm2, ent2 in dd.items():
            if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                arr2 = ent2.get("arr")
                if isinstance(arr2, np.ndarray):
                    layers_dict_end[nm2] = arr2

        measurements = _compute_measurements_from_layers(p, layers_dict_end, H, W)
        if not isinstance(measurements, dict):
            measurements = {}
        if agg_names:
            for nm, mode in agg_modes.items():
                vals = per_tick_meas.get(nm) or []
                if not vals:
                    continue
                if mode == "mean":
                    measurements[nm] = float(np.mean(np.asarray(vals, dtype=np.float64)))
                elif mode == "median":
                    measurements[nm] = float(np.median(np.asarray(vals, dtype=np.float64)))
        
        rep_metrics.append(
            {
                "alive": alive,
                "divisions": int(totals.get("divisions") or 0),
                "starvation_deaths": int(totals.get("starvation_deaths") or 0),
                "damage_deaths": int(totals.get("damage_deaths") or 0),
            }
        )
        rep_per_tick_events.append(
            {
                "divisions": per_tick_divs,
                "starvation_deaths": per_tick_starv,
                "damage_deaths": per_tick_dmg,
            }
        )
        rep_measurements.append(measurements)

    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
    
    avg_per_tick_divs = []
    avg_per_tick_starv = []
    avg_per_tick_dmg = []
    if rep_per_tick_events:
        n_ticks = len(rep_per_tick_events[0]["divisions"])
        for t in range(n_ticks):
            avg_per_tick_divs.append(float(np.mean([r["divisions"][t] for r in rep_per_tick_events])))
            avg_per_tick_starv.append(float(np.mean([r["starvation_deaths"][t] for r in rep_per_tick_events])))
            avg_per_tick_dmg.append(float(np.mean([r["damage_deaths"][t] for r in rep_per_tick_events])))
    
    merged_measurements = {}
    if rep_measurements:
        all_keys = set()
        for m in rep_measurements:
            if isinstance(m, dict):
                all_keys.update(m.keys())
        for k in all_keys:
            vals = [float(m.get(k) or 0) for m in rep_measurements if isinstance(m, dict) and k in m]
            if vals:
                merged_measurements[k] = float(np.mean(vals))
    
    metrics = {
        "alive": int(round(alive_m)),
        "divisions": div_m,
        "starvation_deaths": starv_m,
        "damage_deaths": dmg_m,
        "per_tick_divisions": avg_per_tick_divs,
        "per_tick_starvation_deaths": avg_per_tick_starv,
        "per_tick_damage_deaths": avg_per_tick_dmg,
        "measurements": merged_measurements,
    }
    
    fit = _evo_fitness_from_metrics(metrics, fitness_w)

    return {
        "vi": int(vi),
        "genome": genome,
        "metrics": metrics,
        "fitness": float(fit),
        "evals_done": int(replicates),
    }


def _evo_worker_init_cem_delta(
    base_payload_fast: Dict[str, Any],
    kinds: Dict[str, str],
    cell_layer: str,
    huge: float,
    mu_shm_name: str,
    sig_shm_name: str,
    k_layers: int,
    n_cells: int,
    mutable_names: list[str],
) -> None:
    global _EVO_CEM_MUTABLE_NAMES, _EVO_CEM_MU, _EVO_CEM_SIG, _EVO_CEM_MASK_F
    global _EVO_CEM_MU_SHM, _EVO_CEM_SIG_SHM

    _evo_worker_init(base_payload_fast, kinds, cell_layer, huge)
    _EVO_CEM_MUTABLE_NAMES = list(mutable_names)

    if _EVO_WORKER_BASE_DATA is not None and cell_layer in _EVO_WORKER_BASE_DATA:
        cm = np.asarray(_EVO_WORKER_BASE_DATA.get(cell_layer), dtype=np.float32).reshape(-1) > 0.5
        if int(n_cells) > 0:
            cm = cm[: int(n_cells)]
        _EVO_CEM_MASK_F = cm.astype(np.float32)
    else:
        _EVO_CEM_MASK_F = None

    _EVO_CEM_MU_SHM = shared_memory.SharedMemory(name=str(mu_shm_name))
    _EVO_CEM_SIG_SHM = shared_memory.SharedMemory(name=str(sig_shm_name))
    _EVO_CEM_MU = np.ndarray(
        (int(k_layers), int(n_cells)), dtype=np.float32, buffer=_EVO_CEM_MU_SHM.buf
    )
    _EVO_CEM_SIG = np.ndarray(
        (int(k_layers), int(n_cells)), dtype=np.float32, buffer=_EVO_CEM_SIG_SHM.buf
    )


def _evo_worker_eval_cem_delta(
    gen: int,
    vi: int,
    seed: int,
    ticks: int,
    replicates: int,
    fitness_w: Dict[str, float],
    use_cell_mask: bool,
) -> Dict[str, Any]:
    if _EVO_WORKER_PAYLOAD is None or _EVO_WORKER_DATA is None or _EVO_WORKER_BASE_DATA is None:
        raise RuntimeError("worker not initialized")
    if _EVO_CEM_MUTABLE_NAMES is None or _EVO_CEM_MU is None or _EVO_CEM_SIG is None:
        raise RuntimeError("cem worker not initialized")

    kinds = _EVO_WORKER_KINDS or {}
    huge = float(_EVO_WORKER_HUGE)
    cell_layer = str(_EVO_WORKER_CELL_LAYER)

    p = _EVO_WORKER_PAYLOAD
    dd = _EVO_WORKER_DATA

    mu = _EVO_CEM_MU
    sig = _EVO_CEM_SIG
    mask_f = _EVO_CEM_MASK_F if bool(use_cell_mask) else None
    n_cells = int(mu.shape[1])

    rr = np.random.default_rng(int(seed) + 1234567 + (int(gen) * 1000003) + (int(vi) * 1009))
    deltas: list[np.ndarray] = []
    for i, _nm in enumerate(_EVO_CEM_MUTABLE_NAMES):
        eps = rr.normal(0.0, 1.0, size=(n_cells,)).astype(np.float32)
        delta = np.asarray(mu[i] + sig[i] * eps, dtype=np.float32)
        if mask_f is not None:
            delta = np.asarray(delta * mask_f, dtype=np.float32)
        deltas.append(delta)

    rep_metrics = []
    rep_per_tick_events = []
    rep_measurements = []
    
    for ri in range(int(replicates)):
        seed0 = int(seed) + (int(gen) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)

        p.pop("event_counters", None)
        for nm, src in _EVO_WORKER_BASE_DATA.items():
            ent = dd.get(nm)
            if not isinstance(ent, dict):
                continue
            dst = ent.get("arr")
            if not isinstance(dst, np.ndarray):
                continue
            np.copyto(dst, src)

        for i, nm in enumerate(_EVO_CEM_MUTABLE_NAMES):
            ent = dd.get(nm)
            if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                continue
            arr = ent.get("arr")
            if not isinstance(arr, np.ndarray):
                continue

            arr += deltas[int(i)]
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=huge, neginf=0.0)
            np.clip(arr, 0.0, huge, out=arr)

        H = int(p.get("H") or 0)
        W = int(p.get("W") or 0)

        meas_aggs = fitness_w.get("measurement_aggs", {})
        if not isinstance(meas_aggs, dict):
            meas_aggs = {}
        agg_modes = {
            str(k): str(v)
            for k, v in meas_aggs.items()
            if isinstance(k, str) and str(v) in ("mean", "median")
        }
        agg_names = set(agg_modes.keys())
        per_tick_meas: Dict[str, list[float]] = {k: [] for k in agg_names}

        per_tick_divs = []
        per_tick_starv = []
        per_tick_dmg = []
        
        for t in range(int(ticks)):
            apply_layer_ops_inplace(p, seed_offset=seed0 + int(t))
            
            events = p.get("event_counters") if isinstance(p, dict) else None
            last_events = events.get("last") if isinstance(events, dict) else None
            if isinstance(last_events, dict):
                per_tick_divs.append(int(last_events.get("divisions") or 0))
                per_tick_starv.append(int(last_events.get("starvation_deaths") or 0))
                per_tick_dmg.append(int(last_events.get("damage_deaths") or 0))
            else:
                per_tick_divs.append(0)
                per_tick_starv.append(0)
                per_tick_dmg.append(0)

            if agg_names:
                layers_dict_tick: Dict[str, np.ndarray] = {}
                for nm2, ent2 in dd.items():
                    if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                        arr2 = ent2.get("arr")
                        if isinstance(arr2, np.ndarray):
                            layers_dict_tick[nm2] = arr2

                sel = _compute_selected_measurements_from_layers(p, layers_dict_tick, H, W, agg_names)
                for nm in agg_names:
                    try:
                        per_tick_meas[nm].append(float(sel.get(nm) or 0.0))
                    except Exception:
                        per_tick_meas[nm].append(0.0)

        cell_ent2 = dd.get(cell_layer)
        if not isinstance(cell_ent2, dict) or cell_ent2.get("dtype") != "float32":
            raise ValueError("payload missing cell layer")
        cell_arr = cell_ent2.get("arr")
        if not isinstance(cell_arr, np.ndarray):
            raise ValueError("payload missing cell layer array")

        alive = int((np.asarray(cell_arr, dtype=np.float32).reshape(-1) > 0.5).sum())
        events = p.get("event_counters") if isinstance(p, dict) else None
        totals = events.get("totals") if isinstance(events, dict) else None
        if not isinstance(totals, dict):
            totals = {}
        
        layers_dict_end: Dict[str, np.ndarray] = {}
        for nm2, ent2 in dd.items():
            if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                arr2 = ent2.get("arr")
                if isinstance(arr2, np.ndarray):
                    layers_dict_end[nm2] = arr2

        measurements = _compute_measurements_from_layers(p, layers_dict_end, H, W)
        if not isinstance(measurements, dict):
            measurements = {}
        if agg_names:
            for nm, mode in agg_modes.items():
                vals = per_tick_meas.get(nm) or []
                if not vals:
                    continue
                if mode == "mean":
                    measurements[nm] = float(np.mean(np.asarray(vals, dtype=np.float64)))
                elif mode == "median":
                    measurements[nm] = float(np.median(np.asarray(vals, dtype=np.float64)))
        
        rep_metrics.append(
            {
                "alive": alive,
                "divisions": int(totals.get("divisions") or 0),
                "starvation_deaths": int(totals.get("starvation_deaths") or 0),
                "damage_deaths": int(totals.get("damage_deaths") or 0),
            }
        )
        rep_per_tick_events.append(
            {
                "divisions": per_tick_divs,
                "starvation_deaths": per_tick_starv,
                "damage_deaths": per_tick_dmg,
            }
        )
        rep_measurements.append(measurements)

    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
    
    avg_per_tick_divs = []
    avg_per_tick_starv = []
    avg_per_tick_dmg = []
    if rep_per_tick_events:
        n_ticks = len(rep_per_tick_events[0]["divisions"])
        for t in range(n_ticks):
            avg_per_tick_divs.append(float(np.mean([r["divisions"][t] for r in rep_per_tick_events])))
            avg_per_tick_starv.append(float(np.mean([r["starvation_deaths"][t] for r in rep_per_tick_events])))
            avg_per_tick_dmg.append(float(np.mean([r["damage_deaths"][t] for r in rep_per_tick_events])))
    
    merged_measurements = {}
    if rep_measurements:
        all_keys = set()
        for m in rep_measurements:
            if isinstance(m, dict):
                all_keys.update(m.keys())
        for k in all_keys:
            vals = [float(m.get(k) or 0) for m in rep_measurements if isinstance(m, dict) and k in m]
            if vals:
                merged_measurements[k] = float(np.mean(vals))
    
    metrics = {
        "alive": int(round(alive_m)),
        "divisions": div_m,
        "starvation_deaths": starv_m,
        "damage_deaths": dmg_m,
        "per_tick_divisions": avg_per_tick_divs,
        "per_tick_starvation_deaths": avg_per_tick_starv,
        "per_tick_damage_deaths": avg_per_tick_dmg,
        "measurements": merged_measurements,
    }
    
    fit = _evo_fitness_from_metrics(metrics, fitness_w)

    return {"vi": int(vi), "metrics": metrics, "fitness": float(fit), "evals_done": int(replicates)}


def _evo_worker_eval_cem_delta_batch(
    gen: int,
    vis: list[int],
    seed: int,
    ticks: int,
    replicates: int,
    fitness_w: Dict[str, float],
    use_cell_mask: bool,
) -> Dict[str, Any]:
    out: list[Dict[str, Any]] = []
    evals_done_total = 0
    for vi in vis:
        res = _evo_worker_eval_cem_delta(
            int(gen),
            int(vi),
            int(seed),
            int(ticks),
            int(replicates),
            dict(fitness_w),
            bool(use_cell_mask),
        )
        if isinstance(res, dict):
            out.append(res)
            evals_done_total += int(res.get("evals_done") or 0)
    return {"results": out, "evals_done": int(evals_done_total)}


class _EvolutionJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.job_id: str = ""
        self.running: bool = False
        self.error: str = ""

        self.auto: Dict[str, Any] = {}

        self._base_payload: Optional[Dict[str, Any]] = None

        self.baseline: Dict[str, Any] = {}
        self.series: Dict[str, Any] = {
            "offset": 0,
            "fitness": [],
            "best": [],
            "mean": [],
            "median": [],
        }
        self._series_sum: float = 0.0
        self._series_n: int = 0
        self._series_best: float = float("-inf")

        self.cfg: Dict[str, Any] = {}
        self.progress: Dict[str, Any] = {
            "generation": 0,
            "variant": 0,
            "total_generations": 0,
            "total_variants": 0,
            "evaluations_done": 0,
            "evaluations_total": 0,
            "started_at": 0.0,
            "updated_at": 0.0,
        }

        self.history: Dict[str, list] = {
            "best": [],
            "mean": [],
            "median": [],
            "p10": [],
            "p90": [],
        }

        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.top_ids: list[str] = []

        self.perf: Dict[str, Any] = {
            "evals": 0,
            "apply_s": 0.0,
            "apply_copy_s": 0.0,
            "apply_decode_s": 0.0,
            "apply_math_s": 0.0,
            "apply_encode_s": 0.0,
            "sample_s": 0.0,
            "tick_by_type_s": {},
            "ticks_s": 0.0,
            "decode_cell_s": 0.0,
            "total_s": 0.0,
        }

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            top = []
            for cid in self.top_ids[:10]:
                c = self.candidates.get(cid) or {}
                top.append(
                    {
                        "id": cid,
                        "fitness": c.get("fitness"),
                        "metrics": c.get("metrics"),
                        "gen": c.get("gen"),
                    }
                )
            baseline = dict(self.baseline) if isinstance(self.baseline, dict) else {}
            series = self.series if isinstance(self.series, dict) else {}
            series_out = {
                "offset": int(series.get("offset") or 0),
                "fitness": list(series.get("fitness") or []),
                "best": list(series.get("best") or []),
                "mean": list(series.get("mean") or []),
                "median": list(series.get("median") or []),
            }
            history = self.history if isinstance(self.history, dict) else {}
            history_out = {
                "best": list(history.get("best") or []),
                "mean": list(history.get("mean") or []),
                "median": list(history.get("median") or []),
                "p10": list(history.get("p10") or []),
                "p90": list(history.get("p90") or []),
            }
            perf_out = dict(self.perf) if isinstance(self.perf, dict) else {}
            return {
                "ok": True,
                "job_id": self.job_id,
                "running": self.running,
                "error": self.error,
                "cfg": self.cfg,
                "auto": dict(self.auto) if isinstance(self.auto, dict) else {},
                "progress": dict(self.progress),
                "history": history_out,
                "baseline": baseline,
                "series": series_out,
                "perf": perf_out,
                "top": top,
            }

    def candidate(self, candidate_id: str) -> Dict[str, Any]:
        with self._lock:
            c = self.candidates.get(candidate_id)
            if not c:
                raise ValueError("unknown candidate")

            base = self._base_payload
            if not isinstance(base, dict):
                raise ValueError("evolution base payload missing")

            genome = c.get("genome")
            if not isinstance(genome, dict):
                raise ValueError("candidate genome missing")

            huge = float((self.cfg or {}).get("huge") or 1e9)
            H = int(base.get("H") or 0)
            W = int(base.get("W") or 0)
            if H <= 0 or W <= 0:
                raise ValueError("base payload invalid H/W")

            kinds: Dict[str, str] = {}
            layer_meta = base.get("layers")
            if isinstance(layer_meta, list):
                for m in layer_meta:
                    if not isinstance(m, dict):
                        continue
                    nm = m.get("name")
                    if isinstance(nm, str) and nm:
                        kinds[nm] = str(m.get("kind") or "continuous")

            cell_layer = _find_cell_layer_name(base)
            cell_mask = None
            if cell_layer:
                dd0 = base.get("data")
                if isinstance(dd0, dict):
                    ent0 = dd0.get(cell_layer)
                    if isinstance(ent0, dict) and ent0.get("dtype") == "float32" and isinstance(ent0.get("b64"), str):
                        try:
                            cell_arr = _decode_float32_b64(str(ent0.get("b64") or ""), expected_len=H * W, layer_name=cell_layer)
                            cell_mask = np.asarray(cell_arr, dtype=np.float32).reshape(H * W) > 0.5
                        except Exception:
                            cell_mask = None

            payload = _deepcopy_payload(base)
            payload.pop("event_counters", None)
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("candidate payload missing data")

            for nm, gb in genome.items():
                if not isinstance(nm, str) or not nm:
                    continue
                if not isinstance(gb, dict):
                    continue
                ent = data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if not isinstance(b64, str) or not b64:
                    continue
                arr = _decode_float32_b64(b64, expected_len=H * W, layer_name=nm)
                if "delta_b64" in gb and isinstance(gb.get("delta_b64"), str):
                    db64 = str(gb.get("delta_b64") or "")
                    delta = _decode_float32_b64(db64, expected_len=H * W, layer_name=f"{nm}:delta")
                    arr2 = np.asarray(arr + delta, dtype=np.float32)
                else:
                    s = float(gb.get("scale", 1.0))
                    b = float(gb.get("bias", 0.0))
                    arr2 = np.asarray(arr * s + b, dtype=np.float32)
                    if isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr2.size:
                        arr2[~cell_mask] = np.asarray(arr, dtype=np.float32).reshape(-1)[~cell_mask]
                arr2 = np.nan_to_num(arr2, nan=0.0, posinf=huge, neginf=0.0)
                arr2 = np.clip(arr2, 0.0, huge)
                if kinds.get(nm) == "counts":
                    arr2 = np.clip(np.rint(arr2), 0.0, huge)
                ent["b64"] = _encode_float32_b64(arr2)
            return {
                "ok": True,
                "id": candidate_id,
                "fitness": c.get("fitness"),
                "metrics": c.get("metrics"),
                "genome": c.get("genome"),
                "payload": payload,
            }

    def start(self, base_payload: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        if self.running:
            raise ValueError("evolution already running")

        self._stop.clear()
        self.job_id = str(uuid.uuid4())
        self.running = True
        self.error = ""
        self.cfg = cfg
        self._base_payload = _deepcopy_payload(base_payload)
        self.auto = {}
        self.candidates = {}
        self.top_ids = []
        self.history = {"best": [], "mean": [], "median": [], "p10": [], "p90": []}
        self.baseline = {}
        self.series = {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []}
        self._series_sum = 0.0
        self._series_n = 0
        self._series_best = float("-inf")
        self.perf = {
            "evals": 0,
            "apply_s": 0.0,
            "apply_copy_s": 0.0,
            "apply_decode_s": 0.0,
            "apply_math_s": 0.0,
            "apply_encode_s": 0.0,
            "sample_s": 0.0,
            "tick_by_type_s": {},
            "ticks_s": 0.0,
            "decode_cell_s": 0.0,
            "total_s": 0.0,
        }
        self.progress = {
            "generation": 0,
            "variant": 0,
            "total_generations": int(cfg.get("generations") or 0),
            "total_variants": int(cfg.get("variants") or 0),
            "evaluations_done": 0,
            "evaluations_total": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
        }

        self._thread = threading.Thread(
            target=self._run,
            args=(base_payload, cfg),
            daemon=True,
        )
        self._thread.start()

    def _run(self, base_payload: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        try:
            self._run_impl(base_payload, cfg)
        except Exception as e:
            with self._lock:
                self.error = str(e)
        finally:
            with self._lock:
                self.running = False
                self.progress["updated_at"] = time.time()

    def _run_impl(self, base_payload: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        algo = str(cfg.get("algo") or "affine")
        variants = int(cfg.get("variants") or 0)
        generations = int(cfg.get("generations") or 0)
        ticks = int(cfg.get("ticks") or 0)
        elites = int(cfg.get("elites") or 0)
        replicates = int(cfg.get("replicates") or 1)
        workers = int(cfg.get("workers") or 0)
        cem_sigma_init = float(cfg.get("cem_sigma_init") or 0.5)
        cem_alpha = float(cfg.get("cem_alpha") or 0.7)
        cem_sigma_floor = float(cfg.get("cem_sigma_floor") or 0.05)
        cem_mask = str(cfg.get("cem_mask") or "cell")
        cem_batch = int(cfg.get("cem_batch") or 0)
        seed = int(cfg.get("seed") or 0)
        mut_rate = float(cfg.get("mutation_rate") or 0.15)
        sigma_scale = float(cfg.get("sigma_scale") or 0.25)
        sigma_bias = float(cfg.get("sigma_bias") or 0.25)
        huge = float(cfg.get("huge") or 1e9)
        fitness_w = cfg.get("fitness_weights")
        if not isinstance(fitness_w, dict):
            fitness_w = {}
        meas_weights = fitness_w.get("measurements")
        if not isinstance(meas_weights, dict):
            meas_weights = {}

        active_w = [
            float(w)
            for w in meas_weights.values()
            if isinstance(w, (int, float)) and float(w) != 0.0
        ]
        if not active_w:
            raise ValueError("no active fitness objectives: set at least one non-zero measurement weight")

        if variants <= 0 or generations <= 0 or ticks <= 0:
            raise ValueError("variants/generations/ticks must be > 0")
        elites = max(1, min(variants, elites if elites > 0 else min(10, variants)))
        replicates = max(1, min(50, replicates))
        if workers <= 0:
            workers = max(1, int(min(4, os.cpu_count() or 1)))
        workers = max(1, min(32, workers))

        worker_mode = str(cfg.get("worker_mode") or "process").strip().lower()
        if worker_mode not in ("thread", "process"):
            worker_mode = "process"

        H, W, base_layers, kinds = _decoded_layers_and_kinds(base_payload)
        if H <= 0 or W <= 0:
            raise ValueError("payload invalid H/W")

        data = base_payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("payload missing data")

        cell_layer = _find_cell_layer_name(base_payload)
        if not cell_layer:
            raise ValueError("payload missing cell layer (expected 'cell' or 'cell_type')")

        cell0 = base_layers.get(cell_layer)
        if not isinstance(cell0, np.ndarray):
            raise ValueError("invalid cell layer")
        cell_mask = cell0.reshape(H * W) > 0.5

        # Get target layers from config (list of layer names or glob patterns)
        # If not specified, default to gene_*, rna_*, protein_*
        target_layers_cfg = cfg.get("target_layers")
        if isinstance(target_layers_cfg, list) and len(target_layers_cfg) > 0:
            target_patterns = [str(p).strip() for p in target_layers_cfg if isinstance(p, str) and str(p).strip()]
        else:
            target_patterns = ["gene_*", "rna_*", "protein_*"]

        def _matches_any_pattern(name: str, patterns: list) -> bool:
            import fnmatch
            for pat in patterns:
                if fnmatch.fnmatch(name, pat):
                    return True
            return False

        mutable_names: list[str] = []
        layer_stats: Dict[str, Dict[str, float]] = {}
        for name, ent in data.items():
            if not isinstance(name, str):
                continue
            if not _matches_any_pattern(name, target_patterns):
                continue
            if not isinstance(ent, dict) or ent.get("dtype") != "float32" or not isinstance(ent.get("b64"), str):
                continue
            mutable_names.append(name)
            try:
                arr = base_layers.get(name)
                if not isinstance(arr, np.ndarray):
                    raise ValueError("missing layer array")
                v = arr[cell_mask]
                if v.size <= 1:
                    mu = float(arr.mean())
                    sd = float(arr.std())
                else:
                    mu = float(v.mean())
                    sd = float(v.std())
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1.0
                layer_stats[name] = {"mean": mu, "std": sd}
            except Exception:
                layer_stats[name] = {"mean": 0.0, "std": 1.0}

        if not mutable_names:
            raise ValueError("no gene_/rna_/protein_ layers found to mutate")
        
        if _EVO_DEBUG:
            try:
                _LOG.info("DEBUG CEM: Found %s mutable layers", str(len(mutable_names)))
                for nm in mutable_names[:5]:
                    st = layer_stats.get(nm, {})
                    _LOG.info(
                        "DEBUG CEM: Layer '%s' - mean=%0.4f, std=%0.4f",
                        str(nm),
                        float(st.get("mean", 0.0) or 0.0),
                        float(st.get("std", 1.0) or 1.0),
                    )
            except Exception:
                pass

        eval_total = generations * variants * replicates
        with self._lock:
            self.progress["evaluations_total"] = int(eval_total)
            self.progress["updated_at"] = time.time()

        rng = np.random.default_rng(seed)

        base_payload_fast: Dict[str, Any] = {k: v for k, v in base_payload.items() if k not in ("data", "event_counters")}
        fast_data: Dict[str, Any] = {}
        for nm, ent in data.items():
            if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                continue
            arr0 = base_layers.get(nm)
            if not isinstance(arr0, np.ndarray):
                continue
            fast_data[nm] = {"dtype": "float32", "arr": np.asarray(arr0, dtype=np.float32).reshape(H * W)}
        if not fast_data:
            raise ValueError("no float32 layers found")
        base_payload_fast["data"] = fast_data
        base_payload_fast["_skip_b64_writeback"] = True

        tls = threading.local()

        def _get_thread_workspace() -> tuple[Dict[str, Any], Dict[str, Any]]:
            ws = getattr(tls, "ws", None)
            if isinstance(ws, tuple) and len(ws) == 2:
                p0, d0 = ws
                if isinstance(p0, dict) and isinstance(d0, dict):
                    return p0, d0

            p = dict(base_payload_fast)
            dd = p.get("data")
            assert isinstance(dd, dict)
            out_data: Dict[str, Any] = {}
            for nm, ent in dd.items():
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                arr = ent.get("arr")
                if not isinstance(arr, np.ndarray):
                    continue
                out_data[nm] = {"dtype": "float32", "arr": arr.copy()}
            p["data"] = out_data
            tls.ws = (p, out_data)
            return p, out_data

        def _copy_payload_fast() -> Dict[str, Any]:
            p, out_data = _get_thread_workspace()
            p.pop("event_counters", None)
            for nm, ent in out_data.items():
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                dst = ent.get("arr")
                src_ent = fast_data.get(nm)
                if not isinstance(dst, np.ndarray) or not isinstance(src_ent, dict):
                    continue
                src = src_ent.get("arr")
                if not isinstance(src, np.ndarray):
                    continue
                np.copyto(dst, src)
            return p

        def _perf_add(
            apply_s: float,
            ticks_s: float,
            decode_cell_s: float,
            total_s: float,
            apply_copy_s: float = 0.0,
            apply_decode_s: float = 0.0,
            apply_math_s: float = 0.0,
            apply_encode_s: float = 0.0,
            sample_s: float = 0.0,
            tick_by_type_s: Optional[Dict[str, float]] = None,
        ) -> None:
            with self._lock:
                p = self.perf if isinstance(self.perf, dict) else {}
                p["evals"] = int(p.get("evals") or 0) + 1
                p["apply_s"] = float(p.get("apply_s") or 0.0) + float(apply_s)
                p["apply_copy_s"] = float(p.get("apply_copy_s") or 0.0) + float(apply_copy_s)
                p["apply_decode_s"] = float(p.get("apply_decode_s") or 0.0) + float(apply_decode_s)
                p["apply_math_s"] = float(p.get("apply_math_s") or 0.0) + float(apply_math_s)
                p["apply_encode_s"] = float(p.get("apply_encode_s") or 0.0) + float(apply_encode_s)
                p["sample_s"] = float(p.get("sample_s") or 0.0) + float(sample_s)

                if isinstance(tick_by_type_s, dict):
                    cur = p.get("tick_by_type_s")
                    if not isinstance(cur, dict):
                        cur = {}
                    for k, v in tick_by_type_s.items():
                        if not isinstance(k, str) or not k:
                            continue
                        try:
                            dv = float(v)
                        except Exception:
                            continue
                        cur[k] = float(cur.get(k) or 0.0) + dv
                    p["tick_by_type_s"] = cur

                p["ticks_s"] = float(p.get("ticks_s") or 0.0) + float(ticks_s)
                p["decode_cell_s"] = float(p.get("decode_cell_s") or 0.0) + float(decode_cell_s)
                p["total_s"] = float(p.get("total_s") or 0.0) + float(total_s)
                self.perf = p

        def _mutate_genome(parent: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
            g = {k: {"scale": float(v.get("scale", 1.0)), "bias": float(v.get("bias", 0.0))} for k, v in parent.items()}
            for nm in mutable_names:
                if rng.random() > mut_rate:
                    continue
                cur = g.get(nm) or {"scale": 1.0, "bias": 0.0}
                cur_scale = float(cur.get("scale", 1.0))
                cur_bias = float(cur.get("bias", 0.0))
                cur_scale *= float(np.exp(rng.normal(0.0, sigma_scale)))
                cur_bias += float(rng.normal(0.0, sigma_bias) * layer_stats.get(nm, {}).get("std", 1.0))
                g[nm] = {"scale": cur_scale, "bias": cur_bias}
            return g

        def _apply_genome_to_payload(
            payload: Dict[str, Any], genome: Dict[str, Any]
        ) -> tuple[Dict[str, Any], float, float, float, float]:
            t_copy0 = time.perf_counter()
            out = _deepcopy_payload(payload)
            t_copy = time.perf_counter() - t_copy0
            out.pop("event_counters", None)
            out_data = out.get("data")
            if not isinstance(out_data, dict):
                raise ValueError("payload missing data")

            t_decode = 0.0
            t_math = 0.0
            t_encode = 0.0
            for nm, gb in genome.items():
                ent = out_data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if not isinstance(b64, str) or not b64:
                    continue

                td0 = time.perf_counter()
                arr = _decode_float32_b64(b64, expected_len=H * W, layer_name=nm)
                t_decode += time.perf_counter() - td0
                arr2: np.ndarray

                tm0 = time.perf_counter()
                if isinstance(gb, dict) and "delta" in gb and isinstance(gb.get("delta"), np.ndarray):
                    delta = gb.get("delta")
                    arr2 = arr + np.asarray(delta, dtype=np.float32)
                elif isinstance(gb, dict) and "delta_b64" in gb and isinstance(gb.get("delta_b64"), str):
                    delta = _decode_float32_b64(str(gb.get("delta_b64") or ""), expected_len=H * W, layer_name=f"{nm}:delta")
                    arr2 = arr + delta
                else:
                    if not isinstance(gb, dict):
                        continue
                    s = float(gb.get("scale", 1.0))
                    b = float(gb.get("bias", 0.0))
                    arr2 = arr * s + b
                arr2 = np.asarray(arr2, dtype=np.float32)
                arr2 = np.nan_to_num(arr2, nan=0.0, posinf=huge, neginf=0.0)
                arr2 = np.clip(arr2, 0.0, huge)
                if kinds.get(nm) == "counts":
                    arr2 = np.clip(np.rint(arr2), 0.0, huge)
                t_math += time.perf_counter() - tm0

                te0 = time.perf_counter()
                ent["b64"] = _encode_float32_b64(arr2)
                t_encode += time.perf_counter() - te0
            return out, float(t_copy), float(t_decode), float(t_math), float(t_encode)

        def _apply_genome_to_payload_fast(genome: Dict[str, Any]) -> tuple[Dict[str, Any], float, float, float, float]:
            t_copy0 = time.perf_counter()
            out = _copy_payload_fast()
            t_copy = time.perf_counter() - t_copy0

            out_data = out.get("data")
            if not isinstance(out_data, dict):
                raise ValueError("payload missing data")

            t_math0 = time.perf_counter()
            for nm, gb in genome.items():
                if not isinstance(nm, str) or not nm:
                    continue
                ent = out_data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                arr = ent.get("arr")
                if not isinstance(arr, np.ndarray):
                    continue

                if isinstance(gb, dict) and "delta" in gb and isinstance(gb.get("delta"), np.ndarray):
                    delta = np.asarray(gb.get("delta"), dtype=np.float32).reshape(H * W)
                    arr += delta
                else:
                    if not isinstance(gb, dict):
                        continue
                    s = float(gb.get("scale", 1.0))
                    b = float(gb.get("bias", 0.0))
                    arr *= np.float32(s)
                    arr += np.float32(b)

                np.nan_to_num(arr, copy=False, nan=0.0, posinf=huge, neginf=0.0)
                np.clip(arr, 0.0, huge, out=arr)
                if kinds.get(nm) == "counts":
                    np.rint(arr, out=arr)
                    np.clip(arr, 0.0, huge, out=arr)

                if isinstance(gb, dict) and "delta" not in gb and isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr.size:
                    src_ent = fast_data.get(nm)
                    if isinstance(src_ent, dict):
                        src = src_ent.get("arr")
                        if isinstance(src, np.ndarray) and src.shape[0] == arr.size:
                            arr[~cell_mask] = src[~cell_mask]

                ent["arr"] = np.asarray(arr, dtype=np.float32).reshape(H * W)

            t_math = time.perf_counter() - t_math0
            return out, float(t_copy), 0.0, float(t_math), 0.0

        profile_ticks = bool(cfg.get("profile_ticks") or cfg.get("profile_layer_ops"))

        meas_aggs = fitness_w.get("measurement_aggs", {})
        if not isinstance(meas_aggs, dict):
            meas_aggs = {}
        agg_modes = {
            str(k): str(v)
            for k, v in meas_aggs.items()
            if isinstance(k, str) and str(v) in ("mean", "median")
        }
        agg_names = set(agg_modes.keys())

        def _eval_genome(genome: Dict[str, Any], seed0: int, sample_s: float = 0.0) -> Dict[str, Any]:
            t_total0 = time.perf_counter()
            t_apply0 = time.perf_counter()
            p, t_copy, t_apply_decode, t_math, t_encode = _apply_genome_to_payload_fast(genome)
            t_apply = time.perf_counter() - t_apply0

            lop_cfg = p.get("layer_ops_config") if isinstance(p, dict) else None
            if isinstance(lop_cfg, dict):
                if "opt_env_cache" not in lop_cfg and "optimize_env_cache" not in lop_cfg:
                    lop_cfg["opt_env_cache"] = True
                if "opt_expr_cache" not in lop_cfg and "optimize_expr_cache" not in lop_cfg:
                    lop_cfg["opt_expr_cache"] = True

                if not profile_ticks:
                    for k in ("profile_expr", "profile_step_names", "profile_deep"):
                        if k in lop_cfg:
                            lop_cfg[k] = False

            if profile_ticks:
                p["_profile_layer_ops"] = True

            dd0 = p.get("data")
            if not isinstance(dd0, dict):
                raise ValueError("payload missing data")
            per_tick_meas: Dict[str, list[float]] = {k: [] for k in agg_names}

            t_ticks = 0.0
            for t in range(ticks):
                if self._stop.is_set():
                    raise RuntimeError("stopped")
                tt0 = time.perf_counter()
                apply_layer_ops_inplace(p, seed_offset=seed0 + t)
                t_ticks += time.perf_counter() - tt0

                if agg_names:
                    layers_dict_tick: Dict[str, np.ndarray] = {}
                    for nm2, ent2 in dd0.items():
                        if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                            arr2 = ent2.get("arr")
                            if isinstance(arr2, np.ndarray):
                                layers_dict_tick[str(nm2)] = arr2

                    sel = _compute_selected_measurements_from_layers(p, layers_dict_tick, int(H), int(W), agg_names)
                    for nm in agg_names:
                        try:
                            per_tick_meas[nm].append(float(sel.get(nm) or 0.0))
                        except Exception:
                            per_tick_meas[nm].append(0.0)
            dd = p.get("data")
            if not isinstance(dd, dict):
                raise ValueError("payload missing data")
            cell_ent2 = dd.get(cell_layer)
            if not isinstance(cell_ent2, dict) or cell_ent2.get("dtype") != "float32":
                raise ValueError("payload missing cell layer")

            t_dec0 = time.perf_counter()
            cell_arr0 = cell_ent2.get("arr")
            if isinstance(cell_arr0, np.ndarray):
                cell_arr = np.asarray(cell_arr0, dtype=np.float32).reshape(H * W)
                t_cell_decode = time.perf_counter() - t_dec0
            else:
                cell_arr = _decode_float32_b64(str(cell_ent2.get("b64") or ""), expected_len=H * W, layer_name=cell_layer)
                t_cell_decode = time.perf_counter() - t_dec0
            alive = int((cell_arr > 0.5).sum())
            events = p.get("event_counters") if isinstance(p, dict) else None
            totals = events.get("totals") if isinstance(events, dict) else None
            if not isinstance(totals, dict):
                totals = {}
            divisions = int(totals.get("divisions") or 0)
            starv = int(totals.get("starvation_deaths") or 0)
            dmg = int(totals.get("damage_deaths") or 0)

            layers_dict: Dict[str, np.ndarray] = {}
            for nm, ent in dd.items():
                if isinstance(ent, dict) and ent.get("dtype") == "float32":
                    arr = ent.get("arr")
                    if isinstance(arr, np.ndarray):
                        layers_dict[str(nm)] = arr
            measurements = _compute_measurements_from_layers(p, layers_dict, int(H), int(W))
            if not isinstance(measurements, dict):
                measurements = {}

            if agg_names:
                for nm, mode in agg_modes.items():
                    vals = per_tick_meas.get(nm) or []
                    if not vals:
                        continue
                    if mode == "mean":
                        measurements[nm] = float(np.mean(np.asarray(vals, dtype=np.float64)))
                    elif mode == "median":
                        measurements[nm] = float(np.median(np.asarray(vals, dtype=np.float64)))

            tick_by_type_s: Dict[str, float] = {}
            if profile_ticks:
                ev = p.get("event_counters") if isinstance(p, dict) else None
                lop = ev.get("layer_ops_perf") if isinstance(ev, dict) else None
                bt = lop.get("by_type_s") if isinstance(lop, dict) else None
                if isinstance(bt, dict):
                    for k, v in bt.items():
                        if not isinstance(k, str) or not k:
                            continue
                        try:
                            tick_by_type_s[k] = float(v)
                        except Exception:
                            continue

            t_total = time.perf_counter() - t_total0
            _perf_add(
                apply_s=t_apply,
                ticks_s=t_ticks,
                decode_cell_s=t_cell_decode,
                total_s=t_total,
                apply_copy_s=t_copy,
                apply_decode_s=t_apply_decode,
                apply_math_s=t_math,
                apply_encode_s=t_encode,
                sample_s=sample_s,
                tick_by_type_s=tick_by_type_s if profile_ticks else None,
            )
            return {
                "alive": alive,
                "divisions": divisions,
                "starvation_deaths": starv,
                "damage_deaths": dmg,
                "measurements": measurements,
            }

        def _fitness(metrics: Dict[str, Any]) -> float:
            return _evo_fitness_from_metrics(metrics, fitness_w)

        def _merge_measurements(rep_metrics: list[Dict[str, Any]]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            if not rep_metrics:
                return out
            all_keys: set[str] = set()
            for m in rep_metrics:
                mm = m.get("measurements") if isinstance(m, dict) else None
                if isinstance(mm, dict):
                    for k in mm.keys():
                        if isinstance(k, str) and k:
                            all_keys.add(k)
            for k in all_keys:
                vals = []
                for m in rep_metrics:
                    mm = m.get("measurements") if isinstance(m, dict) else None
                    if not isinstance(mm, dict) or k not in mm:
                        continue
                    try:
                        vals.append(float(mm.get(k) or 0.0))
                    except Exception:
                        continue
                if vals:
                    out[k] = float(np.mean(vals))
            return out

        base_rep_metrics = []
        for ri in range(replicates):
            if self._stop.is_set():
                raise RuntimeError("stopped")
            seed0 = seed + (0 * 1000003) + (0 * 1009) + (ri * 97)
            base_rep_metrics.append(_eval_genome({}, seed0=seed0, sample_s=0.0))
        alive_m0 = float(np.mean([mm["alive"] for mm in base_rep_metrics]))
        div_m0 = float(np.mean([mm["divisions"] for mm in base_rep_metrics]))
        starv_m0 = float(np.mean([mm["starvation_deaths"] for mm in base_rep_metrics]))
        dmg_m0 = float(np.mean([mm["damage_deaths"] for mm in base_rep_metrics]))
        base_metrics = {
            "alive": int(round(alive_m0)),
            "divisions": div_m0,
            "starvation_deaths": starv_m0,
            "damage_deaths": dmg_m0,
            "measurements": _merge_measurements(base_rep_metrics),
        }
        # Calculate baseline fitness using exactly the same approach as candidates
        base_fit = float(_fitness(base_metrics))
        
        if _EVO_DEBUG:
            try:
                _LOG.info("DEBUG: Baseline fitness calculation: %s", str(base_fit))
                _LOG.info("DEBUG: Baseline metrics: %s", str(base_metrics))
                _LOG.info("DEBUG: Using measurement weights: %s", str(meas_weights))
            except Exception:
                pass
        
        with self._lock:
            self.baseline = {"fitness": base_fit, "metrics": base_metrics}
            self.progress["updated_at"] = time.time()

        max_points = int(cfg.get("plot_max_points") or 5000)

        if algo == "auto_switch":
            auto_first = str(cfg.get("auto_first") or "cem_delta").strip().lower()
            if auto_first not in ("cem_delta", "affine"):
                auto_first = "cem_delta"
            auto_patience = max(1, int(cfg.get("auto_patience") or 5))
            auto_min_delta = float(cfg.get("auto_min_delta") or 0.0)
            auto_max_switches = int(cfg.get("auto_max_switches") or 20)
            auto_max_switches = max(0, auto_max_switches)

            use_cell_mask = cem_mask != "all"
            n_cells = int(H * W)

            def _clone_affine_genome(g: Any) -> Dict[str, Dict[str, float]]:
                out2: Dict[str, Dict[str, float]] = {}
                if not isinstance(g, dict):
                    return out2
                for nm, ent in g.items():
                    if not isinstance(nm, str) or not nm:
                        continue
                    if not isinstance(ent, dict):
                        continue
                    out2[nm] = {"scale": float(ent.get("scale", 1.0)), "bias": float(ent.get("bias", 0.0))}
                return out2

            def _delta_seed_from_affine(g: Any) -> Dict[str, np.ndarray]:
                gg = _clone_affine_genome(g)
                out2: Dict[str, np.ndarray] = {}
                for nm in mutable_names:
                    base_arr0 = base_layers.get(nm)
                    if not isinstance(base_arr0, np.ndarray):
                        continue
                    ent = gg.get(nm) or {"scale": 1.0, "bias": 0.0}
                    s = float(ent.get("scale", 1.0))
                    b = float(ent.get("bias", 0.0))
                    base_arr = np.asarray(base_arr0, dtype=np.float32).reshape(n_cells)
                    d = np.asarray(base_arr * np.float32(s - 1.0) + np.float32(b), dtype=np.float32)
                    if use_cell_mask:
                        d[~cell_mask] = 0.0
                    out2[nm] = d
                return out2

            def _affine_seed_from_delta(delta_seed: Any) -> Dict[str, Dict[str, float]]:
                out2: Dict[str, Dict[str, float]] = {}
                if not isinstance(delta_seed, dict):
                    return out2
                for nm in mutable_names:
                    d0 = delta_seed.get(nm)
                    base_arr0 = base_layers.get(nm)
                    if not isinstance(d0, np.ndarray) or not isinstance(base_arr0, np.ndarray):
                        continue
                    x = np.asarray(base_arr0, dtype=np.float64).reshape(n_cells)
                    y = x + np.asarray(d0, dtype=np.float64).reshape(n_cells)
                    if use_cell_mask:
                        xx = x[cell_mask]
                        yy = y[cell_mask]
                    else:
                        xx = x
                        yy = y
                    if xx.size <= 1:
                        s = 1.0
                        b = float(np.mean(yy - xx)) if xx.size else 0.0
                    else:
                        xm = float(xx.mean())
                        ym = float(yy.mean())
                        denom = float(np.sum((xx - xm) ** 2))
                        if denom <= 1e-12:
                            s = 1.0
                        else:
                            s = float(np.sum((xx - xm) * (yy - ym)) / denom)
                        if not np.isfinite(s):
                            s = 1.0
                        if s < 0.0:
                            s = 0.0
                        b = float(ym - s * xm)
                        if not np.isfinite(b):
                            b = 0.0
                    out2[nm] = {"scale": float(s), "bias": float(b)}
                return out2

            def _series_push(vi: int, fit: float, evals_done: int) -> None:
                with self._lock:
                    self.progress["variant"] = int(vi)
                    self.progress["evaluations_done"] = int(
                        self.progress.get("evaluations_done", 0) + max(0, int(evals_done))
                    )
                    self._series_sum += float(fit)
                    self._series_n += 1
                    if float(fit) > self._series_best:
                        self._series_best = float(fit)
                    self.series.setdefault("median", [])
                    self.series["fitness"].append(float(fit))
                    self.series["best"].append(float(self._series_best))
                    self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                    try:
                        self.series["median"].append(
                            float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                        )
                    except Exception:
                        self.series["median"].append(float(fit))
                    while len(self.series["fitness"]) > max_points:
                        self.series["fitness"].pop(0)
                        self.series["best"].pop(0)
                        self.series["mean"].pop(0)
                        self.series["median"].pop(0)
                        self.series["offset"] = int(self.series.get("offset") or 0) + 1
                    self.progress["updated_at"] = time.time()

            cem_alpha2 = float(np.clip(cem_alpha, 0.0, 1.0))
            cem_sigma_init2 = max(0.0, cem_sigma_init)
            cem_sigma_floor2 = max(0.0, cem_sigma_floor)
            cem_topk = max(1, min(variants, elites if elites > 0 else max(1, variants // 10)))

            def _cem_init(seed_delta: Optional[Dict[str, np.ndarray]]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
                mu2: Dict[str, np.ndarray] = {}
                sig2: Dict[str, np.ndarray] = {}
                sig_floor2: Dict[str, float] = {}
                sd = seed_delta if isinstance(seed_delta, dict) else {}
                for nm in mutable_names:
                    layer_mean = float(layer_stats.get(nm, {}).get("mean", 1.0))
                    layer_std = float(layer_stats.get(nm, {}).get("std", 1.0))
                    scale_ref = min(layer_std, abs(layer_mean) * 0.1) if abs(layer_mean) > 1.0 else layer_std
                    if scale_ref <= 0:
                        scale_ref = 1.0
                    sig_floor2[nm] = float(cem_sigma_floor2 * scale_ref)
                    m0 = np.zeros((n_cells,), dtype=np.float32)
                    sd0 = sd.get(nm)
                    if isinstance(sd0, np.ndarray) and int(sd0.size) == int(n_cells):
                        m0 = np.asarray(sd0, dtype=np.float32).reshape(n_cells).copy()
                    if use_cell_mask:
                        m0[~cell_mask] = 0.0
                    mu2[nm] = m0
                    s0 = np.full((n_cells,), float(cem_sigma_init2 * scale_ref), dtype=np.float32)
                    if use_cell_mask:
                        s0[~cell_mask] = 0.0
                    sig2[nm] = s0
                return mu2, sig2, sig_floor2

            def _cem_delta_for_vi(gen_i: int, vi: int, mu0: Dict[str, np.ndarray], sig0: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                rr0 = np.random.default_rng(int(seed) + 1234567 + (int(gen_i) * 1000003) + (int(vi) * 1009))
                out2: Dict[str, np.ndarray] = {}
                for nm in mutable_names:
                    eps = rr0.normal(0.0, 1.0, size=(n_cells,)).astype(np.float32)
                    d = np.asarray(mu0[nm] + sig0[nm] * eps, dtype=np.float32)
                    if use_cell_mask:
                        d = np.where(cell_mask, d, 0.0).astype(np.float32)
                    out2[nm] = d
                return out2

            def _cem_encode_genome(gen_i: int, vi: int, mu0: Dict[str, np.ndarray], sig0: Dict[str, np.ndarray]) -> Dict[str, Any]:
                dd0 = _cem_delta_for_vi(gen_i, int(vi), mu0, sig0)
                g2: Dict[str, Any] = {}
                for nm in mutable_names:
                    g2[nm] = {"delta_b64": _encode_float32_b64(np.asarray(dd0[nm], dtype=np.float32))}
                return g2

            def _run_cem_gen(ex: concurrent.futures.ThreadPoolExecutor, gen_i: int, mu2: Dict[str, np.ndarray], sig2: Dict[str, np.ndarray], sig_floor2: Dict[str, float]) -> tuple[float, Dict[str, np.ndarray]]:
                if self._stop.is_set():
                    return float("-inf"), {}
                with self._lock:
                    self.progress["generation"] = int(gen_i)
                    self.progress["variant"] = 0
                    self.progress["updated_at"] = time.time()
                mu0 = {nm: mu2[nm].copy() for nm in mutable_names}
                sig0 = {nm: sig2[nm].copy() for nm in mutable_names}
                plan = list(range(int(variants)))
                gen_workers = max(1, min(int(workers), len(plan)))

                def _eval_variant(vi: int) -> Optional[Dict[str, Any]]:
                    rr0 = np.random.default_rng(int(seed) + 1234567 + (int(gen_i) * 1000003) + (int(vi) * 1009))
                    t_sample0 = time.perf_counter()
                    genome: Dict[str, Any] = {}
                    for nm in mutable_names:
                        eps = rr0.normal(0.0, 1.0, size=(n_cells,)).astype(np.float32)
                        d = np.asarray(mu0[nm] + sig0[nm] * eps, dtype=np.float32)
                        if use_cell_mask:
                            d = np.where(cell_mask, d, 0.0).astype(np.float32)
                        genome[nm] = {"delta": d}
                    t_sample = time.perf_counter() - t_sample0
                    rep_metrics = []
                    for ri in range(int(replicates)):
                        if self._stop.is_set():
                            return None
                        seed0 = int(seed) + (int(gen_i) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)
                        rep_metrics.append(_eval_genome(genome, seed0=seed0, sample_s=t_sample if ri == 0 else 0.0))
                    if not rep_metrics:
                        return None
                    merged_measurements = _merge_measurements(rep_metrics)
                    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                    metrics = {
                        "alive": int(round(alive_m)),
                        "divisions": div_m,
                        "starvation_deaths": starv_m,
                        "damage_deaths": dmg_m,
                        "measurements": merged_measurements,
                    }
                    fit = float(_fitness(metrics))
                    return {"vi": int(vi), "metrics": metrics, "fitness": fit, "evals_done": int(replicates)}

                pending: set[concurrent.futures.Future] = set()
                it = iter(plan)

                def _submit_one() -> None:
                    if self._stop.is_set():
                        return
                    try:
                        vi0 = next(it)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_eval_variant, int(vi0)))

                for _ in range(gen_workers):
                    _submit_one()

                candidates_this_gen: list[Dict[str, Any]] = []
                while pending:
                    if self._stop.is_set():
                        break
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        if self._stop.is_set():
                            break
                        try:
                            res = fut.result()
                        except Exception:
                            _submit_one()
                            continue
                        if not isinstance(res, dict):
                            _submit_one()
                            continue
                        vi = int(res.get("vi") or 0)
                        metrics = res.get("metrics")
                        fit = float(res.get("fitness") or 0.0)
                        evals_done = int(res.get("evals_done") or 0)
                        if not isinstance(metrics, dict):
                            _submit_one()
                            continue
                        _series_push(vi, fit, evals_done)
                        cid = str(uuid.uuid4())
                        candidates_this_gen.append({"id": cid, "gen": int(gen_i), "fitness": fit, "metrics": metrics, "vi": vi})
                        _submit_one()

                if not candidates_this_gen:
                    return float("-inf"), {}
                candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                best = float(fits.max())
                mean = float(fits.mean())
                median = float(np.quantile(fits, 0.50))
                p10 = float(np.quantile(fits, 0.10))
                p90 = float(np.quantile(fits, 0.90))

                top = candidates_this_gen[:cem_topk]
                top_deltas: list[Dict[str, np.ndarray]] = []
                for c in top:
                    vi = int(c.get("vi") or 0)
                    top_deltas.append(_cem_delta_for_vi(gen_i, vi, mu0, sig0))
                for nm in mutable_names:
                    stack = np.stack([d[nm] for d in top_deltas], axis=0)
                    mu_new = stack.mean(axis=0).astype(np.float32)
                    sig_new = stack.std(axis=0).astype(np.float32)
                    mu2[nm] = ((1.0 - cem_alpha2) * mu2[nm] + cem_alpha2 * mu_new).astype(np.float32)
                    sig2[nm] = ((1.0 - cem_alpha2) * sig2[nm] + cem_alpha2 * sig_new).astype(np.float32)
                    sig2[nm] = np.maximum(sig2[nm], float(sig_floor2.get(nm, 0.0))).astype(np.float32)
                    if use_cell_mask:
                        mu2[nm][~cell_mask] = 0.0
                        sig2[nm][~cell_mask] = 0.0

                with self._lock:
                    for c in candidates_this_gen[: max(int(elites), 10)]:
                        vi = int(c.get("vi") or 0)
                        c2 = dict(c)
                        c2["genome"] = _cem_encode_genome(gen_i, vi, mu0, sig0)
                        c2["genome_type"] = "delta"
                        self.candidates[str(c2["id"])] = c2
                    self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                    self.history["best"].append(best)
                    self.history["mean"].append(mean)
                    self.history.setdefault("median", [])
                    self.history["median"].append(median)
                    self.history["p10"].append(p10)
                    self.history["p90"].append(p90)
                    self.progress["updated_at"] = time.time()

                best_vi = int(candidates_this_gen[0].get("vi") or 0)
                best_delta = _cem_delta_for_vi(gen_i, best_vi, mu0, sig0)
                return float(best), best_delta

            def _run_affine_gen(ex: concurrent.futures.ThreadPoolExecutor, gen_i: int, parents2: list[Dict[str, Dict[str, float]]]) -> tuple[float, Dict[str, Dict[str, float]], list[Dict[str, Dict[str, float]]]]:
                if self._stop.is_set():
                    return float("-inf"), {}, parents2
                with self._lock:
                    self.progress["generation"] = int(gen_i)
                    self.progress["variant"] = 0
                    self.progress["updated_at"] = time.time()

                plan: list[tuple[int, Dict[str, Dict[str, float]]]] = []
                for vi in range(int(variants)):
                    if self._stop.is_set():
                        break
                    parent = parents2[int(rng.integers(0, len(parents2)))]
                    genome = _mutate_genome(parent)
                    plan.append((int(vi), genome))

                gen_workers = max(1, min(int(workers), len(plan)))

                def _eval_variant(vi: int, genome: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
                    rep_metrics = []
                    for ri in range(int(replicates)):
                        if self._stop.is_set():
                            return None
                        seed0 = int(seed) + (int(gen_i) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)
                        rep_metrics.append(_eval_genome(genome, seed0=seed0))
                    if not rep_metrics:
                        return None
                    merged_measurements = _merge_measurements(rep_metrics)
                    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                    metrics = {
                        "alive": int(round(alive_m)),
                        "divisions": div_m,
                        "starvation_deaths": starv_m,
                        "damage_deaths": dmg_m,
                        "measurements": merged_measurements,
                    }
                    fit = float(_fitness(metrics))
                    return {"vi": int(vi), "genome": genome, "metrics": metrics, "fitness": fit, "evals_done": int(replicates)}

                pending: set[concurrent.futures.Future] = set()
                it = iter(plan)

                def _submit_one() -> None:
                    if self._stop.is_set():
                        return
                    try:
                        vi0, genome0 = next(it)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_eval_variant, int(vi0), genome0))

                for _ in range(gen_workers):
                    _submit_one()

                candidates_this_gen: list[Dict[str, Any]] = []
                while pending:
                    if self._stop.is_set():
                        break
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        if self._stop.is_set():
                            break
                        try:
                            res = fut.result()
                        except Exception:
                            _submit_one()
                            continue
                        if not isinstance(res, dict):
                            _submit_one()
                            continue
                        vi = int(res.get("vi") or 0)
                        genome = res.get("genome")
                        metrics = res.get("metrics")
                        fit = float(res.get("fitness") or 0.0)
                        evals_done = int(res.get("evals_done") or 0)
                        if not isinstance(genome, dict) or not isinstance(metrics, dict):
                            _submit_one()
                            continue
                        _series_push(vi, fit, evals_done)
                        cid = str(uuid.uuid4())
                        candidates_this_gen.append({"id": cid, "gen": int(gen_i), "fitness": fit, "metrics": metrics, "genome": genome})
                        _submit_one()

                if not candidates_this_gen:
                    return float("-inf"), {}, parents2
                candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                best = float(fits.max())
                mean = float(fits.mean())
                median = float(np.quantile(fits, 0.50))
                p10 = float(np.quantile(fits, 0.10))
                p90 = float(np.quantile(fits, 0.90))

                with self._lock:
                    for c in candidates_this_gen[: max(int(elites), 10)]:
                        self.candidates[str(c["id"])] = c
                    self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                    self.history["best"].append(best)
                    self.history["mean"].append(mean)
                    self.history.setdefault("median", [])
                    self.history["median"].append(median)
                    self.history["p10"].append(p10)
                    self.history["p90"].append(p90)
                    self.progress["updated_at"] = time.time()

                new_parents = [c["genome"] for c in candidates_this_gen[: int(elites)]]
                best_genome = _clone_affine_genome(candidates_this_gen[0].get("genome"))
                return float(best), best_genome, new_parents if new_parents else parents2

            active_algo = str(auto_first)
            plateau_count = 0
            last_switch_gen = -1
            switches = 0

            best_so_far = float(base_fit)
            best_delta_seed: Dict[str, np.ndarray] = {}
            best_affine_seed: Dict[str, Dict[str, float]] = {}
            parents: list[Dict[str, Dict[str, float]]] = [{}]
            mu, sig, sig_floor = _cem_init(best_delta_seed)

            with self._lock:
                self.auto = {
                    "active_algo": str(active_algo),
                    "plateau_count": int(plateau_count),
                    "last_switch_gen": int(last_switch_gen),
                    "switches": int(switches),
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for gen_i in range(int(generations)):
                    if self._stop.is_set():
                        break
                    if active_algo == "affine":
                        best_fit_gen, best_genome, parents = _run_affine_gen(ex, int(gen_i), parents)
                        if isinstance(best_genome, dict) and best_fit_gen > float("-inf"):
                            improved = bool(best_fit_gen > (best_so_far + float(auto_min_delta)))
                            if improved:
                                best_so_far = float(best_fit_gen)
                                plateau_count = 0
                                best_affine_seed = _clone_affine_genome(best_genome)
                                best_delta_seed = _delta_seed_from_affine(best_genome)
                            else:
                                plateau_count += 1
                    else:
                        best_fit_gen, best_delta = _run_cem_gen(ex, int(gen_i), mu, sig, sig_floor)
                        if isinstance(best_delta, dict) and best_fit_gen > float("-inf"):
                            improved = bool(best_fit_gen > (best_so_far + float(auto_min_delta)))
                            if improved:
                                best_so_far = float(best_fit_gen)
                                plateau_count = 0
                                best_delta_seed = {k: np.asarray(v, dtype=np.float32).reshape(n_cells) for k, v in best_delta.items() if isinstance(k, str) and isinstance(v, np.ndarray)}
                                best_affine_seed = _affine_seed_from_delta(best_delta_seed)
                            else:
                                plateau_count += 1

                    if (
                        auto_max_switches > 0
                        and switches < auto_max_switches
                        and plateau_count >= int(auto_patience)
                        and int(gen_i) < int(generations) - 1
                    ):
                        switches += 1
                        last_switch_gen = int(gen_i)
                        plateau_count = 0
                        active_algo = "affine" if active_algo == "cem_delta" else "cem_delta"
                        if active_algo == "affine":
                            parents = [best_affine_seed] if best_affine_seed else [{}]
                        else:
                            mu, sig, sig_floor = _cem_init(best_delta_seed)

                    with self._lock:
                        self.auto = {
                            "active_algo": str(active_algo),
                            "plateau_count": int(plateau_count),
                            "last_switch_gen": int(last_switch_gen),
                            "switches": int(switches),
                            "patience": int(auto_patience),
                            "min_delta": float(auto_min_delta),
                            "max_switches": int(auto_max_switches),
                            "best_fitness": float(best_so_far),
                            "updated_at": float(time.time()),
                        }

            return

        if algo == "cem_delta":
            topk = max(1, min(variants, elites if elites > 0 else max(1, variants // 10)))
            cem_alpha = float(np.clip(cem_alpha, 0.0, 1.0))
            cem_sigma_init = max(0.0, cem_sigma_init)
            cem_sigma_floor = max(0.0, cem_sigma_floor)
            use_cell_mask = cem_mask != "all"

            mu: Dict[str, np.ndarray] = {}
            sig: Dict[str, np.ndarray] = {}
            sig_floor: Dict[str, float] = {}
            for nm in mutable_names:
                layer_mean = float(layer_stats.get(nm, {}).get("mean", 1.0))
                layer_std = float(layer_stats.get(nm, {}).get("std", 1.0))
                # Use the smaller of std or 10% of mean to avoid huge perturbations
                # This ensures sigma is reasonable relative to actual layer values
                scale_ref = min(layer_std, abs(layer_mean) * 0.1) if abs(layer_mean) > 1.0 else layer_std
                if scale_ref <= 0:
                    scale_ref = 1.0
                sig_floor[nm] = float(cem_sigma_floor * scale_ref)
                mu[nm] = np.zeros((H * W,), dtype=np.float32)
                sig[nm] = np.full((H * W,), float(cem_sigma_init * scale_ref), dtype=np.float32)
                if use_cell_mask:
                    mu[nm][~cell_mask] = 0.0
                    sig[nm][~cell_mask] = 0.0
            
            if _EVO_DEBUG:
                try:
                    _LOG.info("DEBUG CEM: cem_sigma_init=%s, cem_alpha=%s", str(cem_sigma_init), str(cem_alpha))
                    for nm in mutable_names[:3]:
                        layer_mean = layer_stats.get(nm, {}).get("mean", 1.0)
                        layer_std = layer_stats.get(nm, {}).get("std", 1.0)
                        scale_ref = min(layer_std, abs(layer_mean) * 0.1) if abs(layer_mean) > 1.0 else layer_std
                        init_sig = cem_sigma_init * scale_ref
                        _LOG.info(
                            "DEBUG CEM: Layer '%s' initial sigma=%0.4f (mean=%0.1f, std=%0.1f, scale_ref=%0.1f)",
                            str(nm),
                            float(init_sig),
                            float(layer_mean),
                            float(layer_std),
                            float(scale_ref),
                        )
                except Exception:
                    pass

            if worker_mode == "process":
                ctx = mp.get_context("spawn")
                k_layers = int(len(mutable_names))
                n_cells = int(H * W)
                mu_shm = shared_memory.SharedMemory(create=True, size=int(k_layers * n_cells * 4))
                sig_shm = shared_memory.SharedMemory(create=True, size=int(k_layers * n_cells * 4))
                mu_shm_arr = np.ndarray((k_layers, n_cells), dtype=np.float32, buffer=mu_shm.buf)
                sig_shm_arr = np.ndarray((k_layers, n_cells), dtype=np.float32, buffer=sig_shm.buf)
                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=ctx,
                        initializer=_evo_worker_init_cem_delta,
                        initargs=(
                            base_payload_fast,
                            kinds,
                            cell_layer,
                            huge,
                            str(mu_shm.name),
                            str(sig_shm.name),
                            int(k_layers),
                            int(n_cells),
                            list(mutable_names),
                        ),
                    ) as ex:
                        for gen in range(generations):
                            if self._stop.is_set():
                                break

                            with self._lock:
                                self.progress["generation"] = int(gen)
                                self.progress["variant"] = 0
                                self.progress["updated_at"] = time.time()

                            mu0: Dict[str, np.ndarray] = {nm: mu[nm].copy() for nm in mutable_names}
                            sig0: Dict[str, np.ndarray] = {nm: sig[nm].copy() for nm in mutable_names}

                            for i, nm in enumerate(mutable_names):
                                np.copyto(mu_shm_arr[int(i)], np.asarray(mu0[nm], dtype=np.float32).reshape(n_cells))
                                np.copyto(sig_shm_arr[int(i)], np.asarray(sig0[nm], dtype=np.float32).reshape(n_cells))

                            cem_batch0 = int(cem_batch)
                            if cem_batch0 <= 0:
                                target_tasks = max(1, int(workers * 3))
                                cem_batch0 = max(1, (int(variants) + int(target_tasks) - 1) // int(target_tasks))
                            cem_batch0 = max(1, min(64, int(cem_batch0)))

                            todo: deque[tuple[list[int], int]] = deque()
                            buf: list[int] = []
                            for vi in range(int(variants)):
                                buf.append(int(vi))
                                if len(buf) >= cem_batch0:
                                    todo.append((buf, 0))
                                    buf = []
                            if buf:
                                todo.append((buf, 0))

                            gen_workers = max(1, min(workers, len(todo)))

                            candidates_this_gen: list[Dict[str, Any]] = []
                            pending: set[concurrent.futures.Future] = set()
                            future_meta: dict[concurrent.futures.Future, tuple[list[int], int]] = {}

                            def _submit_one() -> None:
                                if self._stop.is_set():
                                    return
                                if not todo:
                                    return
                                vis, retries = todo.popleft()
                                fut = ex.submit(
                                    _evo_worker_eval_cem_delta_batch,
                                    int(gen),
                                    list(vis),
                                    int(seed),
                                    int(ticks),
                                    int(replicates),
                                    dict(fitness_w),
                                    bool(use_cell_mask),
                                )
                                pending.add(fut)
                                future_meta[fut] = (vis, int(retries))

                            for _ in range(gen_workers):
                                _submit_one()

                            while pending:
                                if self._stop.is_set():
                                    break
                                done, pending = concurrent.futures.wait(
                                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for fut in done:
                                    if self._stop.is_set():
                                        break
                                    try:
                                        res = fut.result()
                                    except Exception:
                                        vis0, retries0 = future_meta.pop(fut, ([], 0))
                                        if vis0 and int(retries0) < 2:
                                            todo.append((vis0, int(retries0) + 1))
                                        else:
                                            pass
                                        _submit_one()
                                        continue
                                    if not isinstance(res, dict):
                                        future_meta.pop(fut, None)
                                        _submit_one()
                                        continue

                                    future_meta.pop(fut, None)
                                    batch_results = res.get("results")
                                    if not isinstance(batch_results, list) or not batch_results:
                                        _submit_one()
                                        continue

                                    for r1 in batch_results:
                                        if not isinstance(r1, dict):
                                            continue
                                        vi = int(r1.get("vi") or 0)
                                        metrics = r1.get("metrics")
                                        fit = float(r1.get("fitness") or 0.0)
                                        evals_done = int(r1.get("evals_done") or 0)
                                        if not isinstance(metrics, dict):
                                            continue

                                        with self._lock:
                                            self.progress["variant"] = int(vi)
                                            self.progress["evaluations_done"] = int(
                                                self.progress.get("evaluations_done", 0) + max(0, evals_done)
                                            )
                                            self._series_sum += fit
                                            self._series_n += 1
                                            if fit > self._series_best:
                                                self._series_best = fit
                                            self.series.setdefault("median", [])
                                            self.series["fitness"].append(fit)
                                            self.series["best"].append(float(self._series_best))
                                            self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                                            try:
                                                self.series["median"].append(
                                                    float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                                                )
                                            except Exception:
                                                self.series["median"].append(float(fit))
                                            while len(self.series["fitness"]) > max_points:
                                                self.series["fitness"].pop(0)
                                                self.series["best"].pop(0)
                                                self.series["mean"].pop(0)
                                                self.series["median"].pop(0)
                                                self.series["offset"] = int(self.series.get("offset") or 0) + 1
                                            self.progress["updated_at"] = time.time()

                                        cid = str(uuid.uuid4())
                                        candidates_this_gen.append(
                                            {"id": cid, "gen": gen, "fitness": fit, "metrics": metrics, "vi": vi}
                                        )

                                    _submit_one()

                            if not candidates_this_gen:
                                break

                            candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                            fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                            best = float(fits.max())
                            mean = float(fits.mean())
                            median = float(np.quantile(fits, 0.50))
                            p10 = float(np.quantile(fits, 0.10))
                            p90 = float(np.quantile(fits, 0.90))

                            top = candidates_this_gen[:topk]
                            top_deltas: list[Dict[str, np.ndarray]] = []
                            for c in top:
                                vi = int(c.get("vi") or 0)
                                rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                                dd0: Dict[str, np.ndarray] = {}
                                for nm in mutable_names:
                                    eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                                    delta = mu0[nm] + sig0[nm] * eps
                                    if use_cell_mask:
                                        delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                                    dd0[nm] = delta
                                top_deltas.append(dd0)

                            for nm in mutable_names:
                                stack = np.stack([d[nm] for d in top_deltas], axis=0)
                                mu_new = stack.mean(axis=0).astype(np.float32)
                                sig_new = stack.std(axis=0).astype(np.float32)
                                mu[nm] = ((1.0 - cem_alpha) * mu[nm] + cem_alpha * mu_new).astype(np.float32)
                                sig[nm] = ((1.0 - cem_alpha) * sig[nm] + cem_alpha * sig_new).astype(np.float32)
                                sig[nm] = np.maximum(sig[nm], float(sig_floor.get(nm, 0.0))).astype(np.float32)
                                if use_cell_mask:
                                    mu[nm][~cell_mask] = 0.0
                                    sig[nm][~cell_mask] = 0.0

                            def _encode_delta_genome_for_vi(vi: int) -> Dict[str, Any]:
                                rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                                g: Dict[str, Any] = {}
                                for nm in mutable_names:
                                    eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                                    delta = mu0[nm] + sig0[nm] * eps
                                    if use_cell_mask:
                                        delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                                    g[nm] = {"delta_b64": _encode_float32_b64(np.asarray(delta, dtype=np.float32))}
                                return g

                            with self._lock:
                                for c in candidates_this_gen[: max(elites, 10)]:
                                    vi = int(c.get("vi") or 0)
                                    c2 = dict(c)
                                    c2["genome"] = _encode_delta_genome_for_vi(vi)
                                    c2["genome_type"] = "delta"
                                    self.candidates[str(c2["id"])] = c2
                                self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                                self.history["best"].append(best)
                                self.history["mean"].append(mean)
                                self.history.setdefault("median", [])
                                self.history["median"].append(median)
                                self.history["p10"].append(p10)
                                self.history["p90"].append(p90)
                                self.progress["updated_at"] = time.time()
                finally:
                    try:
                        mu_shm.close()
                        mu_shm.unlink()
                    except Exception:
                        pass
                    try:
                        sig_shm.close()
                        sig_shm.unlink()
                    except Exception:
                        pass

                return

            # Reuse a single executor across generations to reduce overhead and improve scaling.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for gen in range(generations):
                    if self._stop.is_set():
                        break

                    with self._lock:
                        self.progress["generation"] = int(gen)
                        self.progress["variant"] = 0
                        self.progress["updated_at"] = time.time()

                    mu0: Dict[str, np.ndarray] = {nm: mu[nm].copy() for nm in mutable_names}
                    sig0: Dict[str, np.ndarray] = {nm: sig[nm].copy() for nm in mutable_names}

                    plan = list(range(variants))
                    gen_workers = max(1, min(workers, len(plan)))

                    def _eval_variant(vi: int) -> Optional[Dict[str, Any]]:
                        rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                        t_sample0 = time.perf_counter()
                        genome: Dict[str, Any] = {}
                        for nm in mutable_names:
                            eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                            delta = mu0[nm] + sig0[nm] * eps
                            if use_cell_mask:
                                delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                            genome[nm] = {"delta": delta}
                        t_sample = time.perf_counter() - t_sample0

                        rep_metrics = []
                        for ri in range(replicates):
                            if self._stop.is_set():
                                return None
                            seed0 = seed + (gen * 1000003) + (vi * 1009) + (ri * 97)
                            # Sampling the genome is done once per variant; attribute that time once.
                            m = _eval_genome(genome, seed0=seed0, sample_s=t_sample if ri == 0 else 0.0)
                            rep_metrics.append(m)

                        if not rep_metrics:
                            return None

                        merged_measurements = _merge_measurements(rep_metrics)
                        alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                        div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                        starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                        dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                        metrics = {
                            "alive": int(round(alive_m)),
                            "divisions": div_m,
                            "starvation_deaths": starv_m,
                            "damage_deaths": dmg_m,
                            "measurements": merged_measurements,
                        }
                        fit = float(_fitness(metrics))
                        return {"vi": int(vi), "metrics": metrics, "fitness": fit, "evals_done": int(replicates)}

                    candidates_this_gen: list[Dict[str, Any]] = []
                    pending: set[concurrent.futures.Future] = set()
                    it = iter(plan)

                    def _submit_one() -> None:
                        if self._stop.is_set():
                            return
                        try:
                            vi = next(it)
                        except StopIteration:
                            return
                        pending.add(ex.submit(_eval_variant, vi))

                    for _ in range(gen_workers):
                        _submit_one()

                    while pending:
                        if self._stop.is_set():
                            break
                        done, pending = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            if self._stop.is_set():
                                break
                            try:
                                res = fut.result()
                            except Exception:
                                _submit_one()
                                continue
                            if not isinstance(res, dict):
                                _submit_one()
                                continue

                            vi = int(res.get("vi") or 0)
                            metrics = res.get("metrics")
                            fit = float(res.get("fitness") or 0.0)
                            evals_done = int(res.get("evals_done") or 0)
                            if not isinstance(metrics, dict):
                                _submit_one()
                                continue

                            with self._lock:
                                self.progress["variant"] = int(vi)
                                self.progress["evaluations_done"] = int(
                                    self.progress.get("evaluations_done", 0) + max(0, evals_done)
                                )
                                self._series_sum += fit
                                self._series_n += 1
                                if fit > self._series_best:
                                    self._series_best = fit
                                self.series.setdefault("median", [])
                                self.series["fitness"].append(fit)
                                self.series["best"].append(float(self._series_best))
                                self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                                try:
                                    self.series["median"].append(
                                        float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                                    )
                                except Exception:
                                    self.series["median"].append(float(fit))
                                while len(self.series["fitness"]) > max_points:
                                    self.series["fitness"].pop(0)
                                    self.series["best"].pop(0)
                                    self.series["mean"].pop(0)
                                    self.series["median"].pop(0)
                                    self.series["offset"] = int(self.series.get("offset") or 0) + 1
                                self.progress["updated_at"] = time.time()

                            cid = str(uuid.uuid4())
                            candidates_this_gen.append(
                                {"id": cid, "gen": gen, "fitness": fit, "metrics": metrics, "vi": vi}
                            )

                            _submit_one()

                    if not candidates_this_gen:
                        break

                    candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                    fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                    best = float(fits.max())
                    mean = float(fits.mean())
                    median = float(np.quantile(fits, 0.50))
                    p10 = float(np.quantile(fits, 0.10))
                    p90 = float(np.quantile(fits, 0.90))

                    top = candidates_this_gen[:topk]
                    top_deltas: list[Dict[str, np.ndarray]] = []
                    for c in top:
                        vi = int(c.get("vi") or 0)
                        rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                        dd: Dict[str, np.ndarray] = {}
                        for nm in mutable_names:
                            eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                            delta = mu0[nm] + sig0[nm] * eps
                            if use_cell_mask:
                                delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                            dd[nm] = delta
                        top_deltas.append(dd)

                    for nm in mutable_names:
                        stack = np.stack([d[nm] for d in top_deltas], axis=0)
                        mu_new = stack.mean(axis=0).astype(np.float32)
                        sig_new = stack.std(axis=0).astype(np.float32)
                        mu[nm] = ((1.0 - cem_alpha) * mu[nm] + cem_alpha * mu_new).astype(np.float32)
                        sig[nm] = ((1.0 - cem_alpha) * sig[nm] + cem_alpha * sig_new).astype(np.float32)
                        sig[nm] = np.maximum(sig[nm], float(sig_floor.get(nm, 0.0))).astype(np.float32)
                        if use_cell_mask:
                            mu[nm][~cell_mask] = 0.0
                            sig[nm][~cell_mask] = 0.0

                    def _encode_delta_genome_for_vi(vi: int) -> Dict[str, Any]:
                        rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                        g: Dict[str, Any] = {}
                        for nm in mutable_names:
                            eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                            delta = mu0[nm] + sig0[nm] * eps
                            if use_cell_mask:
                                delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                            g[nm] = {"delta_b64": _encode_float32_b64(np.asarray(delta, dtype=np.float32))}
                        return g

                    with self._lock:
                        for c in candidates_this_gen[: max(elites, 10)]:
                            vi = int(c.get("vi") or 0)
                            c2 = dict(c)
                            c2["genome"] = _encode_delta_genome_for_vi(vi)
                            c2["genome_type"] = "delta"
                            self.candidates[str(c2["id"])] = c2
                        self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                        self.history["best"].append(best)
                        self.history["mean"].append(mean)
                        self.history.setdefault("median", [])
                        self.history["median"].append(median)
                        self.history["p10"].append(p10)
                        self.history["p90"].append(p90)
                        self.progress["updated_at"] = time.time()

            return

        parents: list[Dict[str, Dict[str, float]]] = [{}]

        if worker_mode == "process":
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_evo_worker_init,
                initargs=(base_payload_fast, kinds, cell_layer, huge),
            ) as ex:
                for gen in range(generations):
                    if self._stop.is_set():
                        break

                    with self._lock:
                        self.progress["generation"] = int(gen)
                        self.progress["variant"] = 0
                        self.progress["updated_at"] = time.time()

                    candidates_this_gen: list[Dict[str, Any]] = []
                    plan: list[tuple[int, Dict[str, Dict[str, float]]]] = []
                    for vi in range(variants):
                        if self._stop.is_set():
                            break
                        parent = parents[int(rng.integers(0, len(parents)))]
                        genome = _mutate_genome(parent)
                        plan.append((vi, genome))

                    gen_workers = max(1, min(workers, len(plan)))

                    pending: set[concurrent.futures.Future] = set()
                    it = iter(plan)

                    def _submit_one() -> None:
                        if self._stop.is_set():
                            return
                        try:
                            vi, genome = next(it)
                        except StopIteration:
                            return
                        pending.add(
                            ex.submit(
                                _evo_worker_eval_affine,
                                int(gen),
                                int(vi),
                                genome,
                                int(seed),
                                int(ticks),
                                int(replicates),
                                dict(fitness_w),
                            )
                        )

                    for _ in range(gen_workers):
                        _submit_one()

                    while pending:
                        if self._stop.is_set():
                            break
                        done, pending = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            if self._stop.is_set():
                                break
                            try:
                                res = fut.result()
                            except Exception:
                                _submit_one()
                                continue
                            if not isinstance(res, dict):
                                _submit_one()
                                continue

                            vi = int(res.get("vi") or 0)
                            genome = res.get("genome")
                            metrics = res.get("metrics")
                            fit = float(res.get("fitness") or 0.0)
                            evals_done = int(res.get("evals_done") or 0)
                            if not isinstance(genome, dict) or not isinstance(metrics, dict):
                                _submit_one()
                                continue

                            with self._lock:
                                self.progress["variant"] = int(vi)
                                self.progress["evaluations_done"] = int(
                                    self.progress.get("evaluations_done", 0) + max(0, evals_done)
                                )
                                self._series_sum += fit
                                self._series_n += 1
                                if fit > self._series_best:
                                    self._series_best = fit
                                self.series["fitness"].append(fit)
                                self.series["best"].append(float(self._series_best))
                                self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                                while len(self.series["fitness"]) > max_points:
                                    self.series["fitness"].pop(0)
                                    self.series["best"].pop(0)
                                    self.series["mean"].pop(0)
                                    self.series["offset"] = int(self.series.get("offset") or 0) + 1
                                self.progress["updated_at"] = time.time()

                            cid = str(uuid.uuid4())
                            cand = {
                                "id": cid,
                                "gen": gen,
                                "fitness": fit,
                                "metrics": metrics,
                                "genome": genome,
                            }
                            candidates_this_gen.append(cand)

                            _submit_one()

                    if not candidates_this_gen:
                        break

                    candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                    fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                    best = float(fits.max())
                    mean = float(fits.mean())
                    p10 = float(np.quantile(fits, 0.10))
                    p90 = float(np.quantile(fits, 0.90))

                    with self._lock:
                        for c in candidates_this_gen[: max(elites, 10)]:
                            self.candidates[str(c["id"])] = c
                        self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                        self.history["best"].append(best)
                        self.history["mean"].append(mean)
                        self.history["p10"].append(p10)
                        self.history["p90"].append(p90)
                        self.progress["updated_at"] = time.time()

                    parents = [c["genome"] for c in candidates_this_gen[:elites]]

            return

        # Thread mode (default)
        # Reuse a single executor across generations to reduce overhead and improve scaling.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for gen in range(generations):
                if self._stop.is_set():
                    break

                with self._lock:
                    self.progress["generation"] = int(gen)
                    self.progress["variant"] = 0
                    self.progress["updated_at"] = time.time()

                candidates_this_gen: list[Dict[str, Any]] = []
                plan: list[tuple[int, Dict[str, Dict[str, float]]]] = []
                for vi in range(variants):
                    if self._stop.is_set():
                        break
                    parent = parents[int(rng.integers(0, len(parents)))]
                    genome = _mutate_genome(parent)
                    plan.append((vi, genome))

                gen_workers = max(1, min(workers, len(plan)))

                def _eval_variant(vi: int, genome: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
                    rep_metrics = []
                    for ri in range(replicates):
                        if self._stop.is_set():
                            return None
                        seed0 = seed + (gen * 1000003) + (vi * 1009) + (ri * 97)
                        m = _eval_genome(genome, seed0=seed0)
                        rep_metrics.append(m)

                    if not rep_metrics:
                        return None

                    merged_measurements = _merge_measurements(rep_metrics)
                    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                    metrics = {
                        "alive": int(round(alive_m)),
                        "divisions": div_m,
                        "starvation_deaths": starv_m,
                        "damage_deaths": dmg_m,
                        "measurements": merged_measurements,
                    }
                    fit = float(_fitness(metrics))
                    return {
                        "vi": int(vi),
                        "genome": genome,
                        "metrics": metrics,
                        "fitness": fit,
                        "evals_done": int(replicates),
                    }

                pending: set[concurrent.futures.Future] = set()
                it = iter(plan)

                def _submit_one() -> None:
                    if self._stop.is_set():
                        return
                    try:
                        vi, genome = next(it)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_eval_variant, vi, genome))

                for _ in range(gen_workers):
                    _submit_one()

                while pending:
                    if self._stop.is_set():
                        break
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done:
                        if self._stop.is_set():
                            break
                        try:
                            res = fut.result()
                        except Exception:
                            _submit_one()
                            continue
                        if not isinstance(res, dict):
                            _submit_one()
                            continue

                        vi = int(res.get("vi") or 0)
                        genome = res.get("genome")
                        metrics = res.get("metrics")
                        fit = float(res.get("fitness") or 0.0)
                        evals_done = int(res.get("evals_done") or 0)
                        if not isinstance(genome, dict) or not isinstance(metrics, dict):
                            _submit_one()
                            continue

                        with self._lock:
                            self.progress["variant"] = int(vi)
                            self.progress["evaluations_done"] = int(
                                self.progress.get("evaluations_done", 0) + max(0, evals_done)
                            )
                            self._series_sum += fit
                            self._series_n += 1
                            if fit > self._series_best:
                                self._series_best = fit
                            self.series.setdefault("median", [])
                            self.series["fitness"].append(fit)
                            self.series["best"].append(float(self._series_best))
                            self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                            try:
                                self.series["median"].append(
                                    float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                                )
                            except Exception:
                                self.series["median"].append(float(fit))
                            while len(self.series["fitness"]) > max_points:
                                self.series["fitness"].pop(0)
                                self.series["best"].pop(0)
                                self.series["mean"].pop(0)
                                self.series["median"].pop(0)
                                self.series["offset"] = int(self.series.get("offset") or 0) + 1
                            self.progress["updated_at"] = time.time()

                        cid = str(uuid.uuid4())
                        cand = {
                            "id": cid,
                            "gen": gen,
                            "fitness": fit,
                            "metrics": metrics,
                            "genome": genome,
                        }
                        candidates_this_gen.append(cand)

                        _submit_one()

                if not candidates_this_gen:
                    break

                candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                best = float(fits.max())
                mean = float(fits.mean())
                median = float(np.quantile(fits, 0.50))
                p10 = float(np.quantile(fits, 0.10))
                p90 = float(np.quantile(fits, 0.90))

                with self._lock:
                    for c in candidates_this_gen[: max(elites, 10)]:
                        self.candidates[str(c["id"])] = c
                    self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                    self.history["best"].append(best)
                    self.history["mean"].append(mean)
                    self.history.setdefault("median", [])
                    self.history["median"].append(median)
                    self.history["p10"].append(p10)
                    self.history["p90"].append(p90)
                    self.progress["updated_at"] = time.time()

                parents = [c["genome"] for c in candidates_this_gen[:elites]]


_EVO = _EvolutionJob()


class _RuntimeState:
    def __init__(self) -> None:
        self.payload: Optional[Dict[str, Any]] = None
        self.tick: int = 0
        self._lock = threading.Lock()

    def _default_layer_names(self, max_layers: int = 4) -> list:
        if self.payload is None:
            return []
        layers = self.payload.get("layers")
        if not isinstance(layers, list):
            return []
        names = []
        for m in layers:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                names.append(m["name"])
        if not names:
            return []

        prefer = ["cell", "circulation", "damage", "molecule_glucose", "molecule_atp"]
        picked = []
        s = set(names)
        for p in prefer:
            if p in s:
                picked.append(p)
            if len(picked) >= max_layers:
                break
        if len(picked) < max_layers:
            for nm in names:
                if nm in picked:
                    continue
                picked.append(nm)

                if len(picked) >= max_layers:
                    break
        return picked

    def reset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            if payload.get("version") != 1:
                raise ValueError("gridstate version must be 1")
            if "H" not in payload or "W" not in payload:
                raise ValueError("gridstate missing H/W")
            if not isinstance(payload.get("layers"), list):
                raise ValueError("gridstate missing layers[]")
            if not isinstance(payload.get("data"), dict):
                raise ValueError("gridstate missing data{}")

            # Keep in-memory copy; server is stateful.
            self.payload = payload
            self.tick = 0

            layers_meta = []
            for m in payload.get("layers", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                kind = m.get("kind")
                if isinstance(name, str):
                    layers_meta.append({"name": name, "kind": str(kind or "continuous")})

            return {
                "ok": True,
                "tick": self.tick,
                "H": int(payload["H"]),
                "W": int(payload["W"]),
                "layers": layers_meta,
            }

    def frame(self, layer_names: Optional[list]) -> Dict[str, Any]:
        with self._lock:
            if self.payload is None:
                raise ValueError("runtime not initialized: call /api/runtime/reset first")

            data = self.payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("payload missing data")

            if not layer_names:
                layer_names = self._default_layer_names()

            out_data: Dict[str, Any] = {}
            if layer_names:
                for nm in layer_names:
                    if not isinstance(nm, str):
                        continue
                    ent = data.get(nm)
                    if not isinstance(ent, dict):
                        continue
                    dtype = ent.get("dtype")
                    b64 = ent.get("b64")
                    if dtype != "float32" or not isinstance(b64, str):
                        continue
                    out_data[nm] = {"dtype": "float32", "b64": b64}

            H, W, layers, kinds = _decoded_layers_and_kinds(self.payload)
            events = self.payload.get("event_counters") if isinstance(self.payload, dict) else None
            if not isinstance(events, dict):
                events = {}
            return {
                "ok": True,
                "tick": self.tick,
                "data": out_data,
                "scalars": _compute_layer_scalars_from_layers(layers, kinds),
                "measurements": _compute_measurements_from_layers(self.payload, layers, H=H, W=W),
                "events": events,
            }

    def export(self) -> Dict[str, Any]:
        with self._lock:
            if self.payload is None:
                raise ValueError("runtime not initialized: call /api/runtime/reset first")

            base = self.payload
            H = int(base.get("H") or 0)
            W = int(base.get("W") or 0)
            if H <= 0 or W <= 0:
                raise ValueError("payload invalid H/W")

            layers_meta = base.get("layers")
            if not isinstance(layers_meta, list):
                raise ValueError("payload missing layers")

            data = base.get("data")
            if not isinstance(data, dict):
                raise ValueError("payload missing data")

            out: Dict[str, Any] = {
                "version": 1,
                "H": H,
                "W": W,
                "tick": int(self.tick),
                "layers": json.loads(json.dumps(layers_meta)),
                "data": {},
            }
            if "measurements_config" in base:
                out["measurements_config"] = json.loads(json.dumps(base.get("measurements_config")))
            if "layer_ops_config" in base:
                out["layer_ops_config"] = json.loads(json.dumps(base.get("layer_ops_config")))

            out_data: Dict[str, Any] = {}
            for m in layers_meta:
                if not isinstance(m, dict):
                    continue
                nm = m.get("name")
                if not isinstance(nm, str) or not nm:
                    continue
                ent = data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if isinstance(b64, str) and b64:
                    out_data[nm] = {"dtype": "float32", "b64": b64}
                    continue
                arr = ent.get("arr")
                if isinstance(arr, np.ndarray):
                    arr2 = np.asarray(arr, dtype=np.float32).reshape(H * W)
                    out_data[nm] = {"dtype": "float32", "b64": _encode_float32_b64(arr2)}
                    continue
            out["data"] = out_data
            return {"ok": True, "tick": int(self.tick), "payload": out}

    def step(self, layer_names: Optional[list]) -> Dict[str, Any]:
        with self._lock:
            if self.payload is None:
                raise ValueError("runtime not initialized: call /api/runtime/reset first")

            apply_layer_ops_inplace(self.payload, seed_offset=self.tick)
            self.tick += 1

            data = self.payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("payload missing data")

            if not layer_names:
                layer_names = self._default_layer_names()

            out_data: Dict[str, Any] = {}
            if layer_names:
                for nm in layer_names:
                    if not isinstance(nm, str):
                        continue
                    ent = data.get(nm)
                    if not isinstance(ent, dict):
                        continue
                    dtype = ent.get("dtype")
                    b64 = ent.get("b64")
                    if dtype != "float32" or not isinstance(b64, str):
                        continue
                    out_data[nm] = {"dtype": "float32", "b64": b64}

            H, W, layers, kinds = _decoded_layers_and_kinds(self.payload)
            events = self.payload.get("event_counters") if isinstance(self.payload, dict) else None
            if not isinstance(events, dict):
                events = {}
            return {
                "ok": True,
                "tick": self.tick,
                "data": out_data,
                "scalars": _compute_layer_scalars_from_layers(layers, kinds),
                "measurements": _compute_measurements_from_layers(self.payload, layers, H=H, W=W),
                "events": events,
            }


_RT = _RuntimeState()


class RuntimeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(_WEB_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        try:
            _LOG.debug("HTTP %s", (fmt % args) if args else str(fmt))
        except Exception:
            pass

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.settimeout(30)
        except Exception:
            pass

    def end_headers(self) -> None:
        # This is a local dev server; always disable caching so HTML/JS/CSS updates are picked up.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, code: int, obj: Dict[str, Any]) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length") or "0")
        except Exception:
            n = 0
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"bad json: {e}")
        if not isinstance(obj, dict):
            raise ValueError("json body must be an object")
        return obj

    def do_GET(self):  # noqa: N802
        t0 = time.time()
        try:
            if self.path == "/api/health":
                self._send_json(200, {"ok": True, "tick": int(_RT.tick)})
                return
            if self.path == "/api/doc/status":
                self._send_json(200, _DOC.status())
                return
            if self.path == "/api/doc/list":
                self._send_json(200, _DOC.list_docs())
                return
            super().do_GET()
        finally:
            try:
                dt_ms = (time.time() - t0) * 1000.0
                _LOG.info("GET %s %.1fms", str(self.path), float(dt_ms))
            except Exception:
                pass

    def do_POST(self):  # noqa: N802
        t0 = time.time()
        try:
            if self.path == "/api/doc/clear":
                _DOC.clear_active()
                self._send_json(200, _DOC.status())
                return

            if self.path == "/api/doc/get":
                self._send_json(200, {"ok": True, "payload_text": _DOC.get_payload_text(), **_DOC.status()})
                return

            if self.path == "/api/doc/autosave":
                body = self._read_json_body()
                payload_text = body.get("payload_text")
                path = body.get("path")
                if not isinstance(payload_text, str):
                    raise ValueError("payload_text must be string")
                out = _DOC.set_payload_from_text(payload_text, path=str(path or "") if isinstance(path, str) else "")
                self._send_json(200, out)
                return

            if self.path == "/api/doc/open":
                body = self._read_json_body()
                name = body.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("missing name")
                out = _DOC.open_doc(name)
                self._send_json(200, out)
                return

            if self.path == "/api/doc/save":
                body = self._read_json_body()
                name = body.get("name")
                if name is not None and (not isinstance(name, str) or not name.strip()):
                    raise ValueError("bad name")
                out = _DOC.save_doc(name if isinstance(name, str) else None)
                self._send_json(200, out)
                return

            if self.path == "/api/doc/delete":
                body = self._read_json_body()
                name = body.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("missing name")
                out = _DOC.delete_doc(name)
                self._send_json(200, out)
                return

            if self.path == "/api/doc/recover":
                out = _DOC.recover_autosave()
                self._send_json(200, out)
                return

            if self.path == "/api/evolution/start":
                body = self._read_json_body()
                payload = body.get("payload")
                cfg = body.get("config")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                if not isinstance(cfg, dict):
                    raise ValueError("missing config")
                job_id = _evo_start_runner(payload, cfg)
                self._send_json(200, {"ok": True, "job_id": job_id})
                return

            if self.path == "/api/evolution/stop":
                _evo_stop_runner()
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/evolution/status":
                self._send_json(200, _evo_status_from_disk())
                return

            if self.path == "/api/evolution/candidate":
                body = self._read_json_body()
                cid = body.get("id")
                if not isinstance(cid, str) or not cid:
                    raise ValueError("missing id")
                base = _safe_read_json(_evo_base_payload_path())
                cfg = _safe_read_json(_evo_cfg_path())
                cands = _safe_read_json(_evo_candidates_path())
                if not isinstance(base, dict) or not isinstance(cfg, dict) or not isinstance(cands, dict):
                    raise ValueError("candidate store missing")
                c = cands.get(cid)
                if not isinstance(c, dict):
                    raise ValueError("unknown candidate")

                # Auto-switch persists a stable "__winner__" snapshot that may already be
                # reflected in base_payload.json. In that case, return the base payload directly
                # so the latest winner can always be fetched even if candidates were cleared.
                if bool(c.get("payload_inline")) and isinstance(c.get("payload"), dict):
                    payload = c.get("payload")
                elif bool(c.get("payload_is_base")):
                    payload = base
                else:
                    payload = _evo_build_candidate_payload(base, cfg, c)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "id": cid,
                        "fitness": c.get("fitness"),
                        "metrics": c.get("metrics"),
                        "genome": c.get("genome"),
                        "payload": payload,
                    },
                )
                return

            if self.path == "/api/evolution/fitness-config":
                body = self._read_json_body()
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                
                measurements_cfg = payload.get("measurements_config")
                available_measurements = []
                if isinstance(measurements_cfg, dict) and int(measurements_cfg.get("version") or 0) == 3:
                    meas_list = measurements_cfg.get("measurements")
                    if isinstance(meas_list, list):
                        for m in meas_list:
                            if isinstance(m, dict):
                                name = str(m.get("name") or "").strip()
                                expr = str(m.get("expr") or "").strip()
                                if name and expr:
                                    available_measurements.append({
                                        "name": name,
                                        "expr": expr,
                                    })
                
                self._send_json(200, {
                    "ok": True,
                    "measurements": available_measurements,
                    "events": ["divisions", "starvation_deaths", "damage_deaths"],
                    "distribution_methods": [
                        {"value": "entropy", "label": "Entropy (normalized Shannon)"},
                        {"value": "cv", "label": "Coefficient of Variation"},
                        {"value": "spread", "label": "Spread (non-zero ticks)"},
                    ],
                })
                return

            if self.path == "/api/runtime/reset":
                body = self._read_json_body()
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                out = _RT.reset(payload)
                self._send_json(200, out)
                return

            if self.path == "/api/runtime/frame":
                body = self._read_json_body()
                layers = body.get("layers")
                if layers is not None and not isinstance(layers, list):
                    raise ValueError("layers must be a list")
                out = _RT.frame(layers)
                self._send_json(200, out)
                return

            if self.path == "/api/runtime/step":
                body = self._read_json_body()
                layers = body.get("layers")
                if layers is not None and not isinstance(layers, list):
                    raise ValueError("layers must be a list")
                out = _RT.step(layers)
                self._send_json(200, out)
                return

            if self.path == "/api/runtime/export":
                out = _RT.export()
                self._send_json(200, out)
                return

            self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as e:
            err_id = uuid.uuid4().hex[:10]
            try:
                _LOG.exception("POST %s failed (error_id=%s)", str(self.path), err_id)
            except Exception:
                pass
            self._send_json(400, {"ok": False, "error": str(e), "error_id": err_id})
        finally:
            try:
                dt_ms = (time.time() - t0) * 1000.0
                _LOG.info("POST %s %.1fms", str(self.path), float(dt_ms))
            except Exception:
                pass


def main() -> int:
    _setup_logging()
    _install_exception_hooks()
    _ensure_dirs()
    if not _WEB_DIR.exists():
        try:
            _LOG.error("web editor dir not found: %s", str(_WEB_DIR))
        except Exception:
            pass
        return 2

    port = 8000
    try:
        if len(sys.argv) >= 2:
            port = int(sys.argv[1])
    except Exception:
        port = 8000

    # Avoid caching during dev.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("DT_RUNTIME_PORT", str(port))

    srv = ThreadingHTTPServer(("0.0.0.0", port), RuntimeHandler)
    try:
        srv.daemon_threads = True
    except Exception:
        pass
    try:
        _LOG.info("Runtime server: http://0.0.0.0:%s/", str(port))
        _LOG.info("Open Functions → Runtime")
    except Exception:
        pass
    try:
        _LOG.info("Runtime server starting on port %s", str(port))
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        srv.server_close()
        try:
            _LOG.info("Runtime server stopped")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
