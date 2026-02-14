from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union


_PathLike = Union[str, Path]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_keys_file() -> Path:
    p = str(os.environ.get("DT_KEYS_FILE") or "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return (_repo_root() / "keys.txt").resolve()


def _strip_quotes(v: str) -> str:
    s = str(v or "")
    if len(s) >= 2 and ((s[0] == s[-1] == "\"") or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def parse_keys_text(txt: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in str(txt or "").splitlines():
        s = str(raw).strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = str(k).strip()
        if not k:
            continue
        v = _strip_quotes(str(v).strip())
        out[k] = v
    return out


def load_keys_file(path: Optional[_PathLike] = None) -> Dict[str, str]:
    p = Path(path).expanduser().resolve() if path is not None else _default_keys_file()
    if not p.exists() or not p.is_file():
        return {}
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    return parse_keys_text(txt)


def apply_keys_to_environ(*, path: Optional[_PathLike] = None, override: bool = False) -> Dict[str, str]:
    keys = load_keys_file(path)
    if not keys:
        return {}

    applied: Dict[str, str] = {}
    for k, v in keys.items():
        if not override:
            cur = os.environ.get(str(k))
            if isinstance(cur, str) and cur.strip():
                continue
        os.environ[str(k)] = str(v)
        applied[str(k)] = str(v)
    return applied


def with_keys_env(env: Optional[Dict[str, str]] = None, *, path: Optional[_PathLike] = None, override: bool = False) -> Dict[str, str]:
    base = dict(env) if env is not None else dict(os.environ)
    keys = load_keys_file(path)
    if not keys:
        return base

    for k, v in keys.items():
        if not override:
            cur = base.get(str(k))
            if isinstance(cur, str) and cur.strip():
                continue
        base[str(k)] = str(v)
    return base
