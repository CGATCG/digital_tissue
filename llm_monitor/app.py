import json
import concurrent.futures
import hashlib
import importlib.util
import os
import random
import re
import signal
import subprocess
import sys
import time
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    events_path: Path
    report_path: Path
    files_dir: Path
    stdout_path: Path
    stderr_path: Path
    pid_path: Path
    state_path: Path


@dataclass
class SuitePaths:
    suite_id: str
    suite_dir: Path
    pid_path: Path
    stdout_path: Path
    stderr_path: Path
    specs_path: Path
    summary_csv_path: Path
    aggregate_csv_path: Path
    manifest_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@st.cache_resource
def _load_runner_module() -> Any:
    p = (_repo_root() / "trials" / "run_llm_benchmark.py").resolve()
    spec = importlib.util.spec_from_file_location("llm_benchmark_runner", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load runner module spec: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource
def _story_executor() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="story")


def _runs_root() -> Path:
    return _repo_root() / "runs" / "llm_bench"


def _suites_root() -> Path:
    return _runs_root() / "suites"


def _active_run_id_path() -> Path:
    return _runs_root() / "active_run_id.txt"


def _active_suite_id_path() -> Path:
    return _runs_root() / "active_suite_id.txt"


def _write_active_run_id(run_id: str) -> None:
    try:
        rid = str(run_id or "").strip()
        if not rid:
            return
        root = _runs_root()
        root.mkdir(parents=True, exist_ok=True)
        _active_run_id_path().write_text(rid + "\n", encoding="utf-8")
    except Exception:
        pass


def _bootstrap_active_suite_id() -> None:
    try:
        if st.session_state.get("suite_bootstrapped") is True:
            return
        st.session_state["suite_bootstrapped"] = True
        cur = str(st.session_state.get("active_suite_id") or "").strip()
        if cur:
            return

        saved = _read_active_suite_id()
        if saved:
            try:
                if _paths_for_suite(saved).suite_dir.exists():
                    st.session_state["active_suite_id"] = str(saved)
                    return
            except Exception:
                pass

        running = _running_suites()
        if running:
            st.session_state["active_suite_id"] = str(running[0])
            _write_active_suite_id(str(running[0]))
    except Exception:
        pass


def _read_active_run_id() -> str:
    try:
        p = _active_run_id_path()
        if not p.exists() or not p.is_file():
            return ""
        return str(p.read_text(encoding="utf-8", errors="replace") or "").strip()
    except Exception:
        return ""


def _write_active_suite_id(suite_id: str) -> None:
    try:
        sid = str(suite_id or "").strip()
        if not sid:
            return
        root = _runs_root()
        root.mkdir(parents=True, exist_ok=True)
        _active_suite_id_path().write_text(sid + "\n", encoding="utf-8")
    except Exception:
        pass


def _read_active_suite_id() -> str:
    try:
        p = _active_suite_id_path()
        if not p.exists() or not p.is_file():
            return ""
        return str(p.read_text(encoding="utf-8", errors="replace") or "").strip()
    except Exception:
        return ""


def _new_run_id() -> str:
    t = int(time.time())
    r = random.randint(1000, 9999)
    return f"run_{t}_{r}"


def _new_suite_id() -> str:
    t = int(time.time())
    r = random.randint(1000, 9999)
    return f"suite_{t}_{r}"


def _paths_for_run(run_id: str) -> RunPaths:
    root = _runs_root()
    run_dir = root / run_id
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        events_path=run_dir / "events.jsonl",
        report_path=run_dir / "report.json",
        files_dir=run_dir / "files",
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        pid_path=run_dir / "runner.pid",
        state_path=run_dir / "state.json",
    )


def _paths_for_suite(suite_id: str) -> SuitePaths:
    suite_dir = _suites_root() / str(suite_id)
    return SuitePaths(
        suite_id=str(suite_id),
        suite_dir=suite_dir,
        pid_path=suite_dir / "suite.pid",
        stdout_path=suite_dir / "stdout.log",
        stderr_path=suite_dir / "stderr.log",
        specs_path=suite_dir / "specs.txt",
        summary_csv_path=suite_dir / "suite_summary.csv",
        aggregate_csv_path=suite_dir / "suite_aggregate.csv",
        manifest_path=suite_dir / "suite_manifest.json",
    )


def _read_pid(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
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


def _write_pid(path: Path, pid: int) -> None:
    try:
        path.write_text(str(int(pid)) + "\n", encoding="utf-8")
    except Exception:
        pass


def _clear_pid(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _list_runs() -> List[str]:
    root = _runs_root()
    if not root.exists() or not root.is_dir():
        return []
    out: List[str] = []
    for p in root.iterdir():
        try:
            if p.is_dir():
                out.append(p.name)
        except Exception:
            continue
    out.sort(reverse=True)
    return out


def _list_suites() -> List[str]:
    root = _suites_root()
    if not root.exists() or not root.is_dir():
        return []
    out: List[str] = []
    for p in root.iterdir():
        try:
            if p.is_dir():
                out.append(p.name)
        except Exception:
            continue
    out.sort(reverse=True)
    return out


def _running_runs() -> List[str]:
    out: List[str] = []
    for rid in _list_runs():
        p = _paths_for_run(rid)
        pid = _read_pid(p.pid_path)
        if pid is None:
            continue
        if _pid_alive(int(pid)):
            out.append(str(rid))
    return out


def _running_suites() -> List[str]:
    out: List[str] = []
    for sid in _list_suites():
        p = _paths_for_suite(sid)
        pid = _read_pid(p.pid_path)
        if pid is None:
            continue
        if _pid_alive(int(pid)):
            out.append(str(sid))
    return out


def _preflight_inspect_row(ent: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    llm_path = str(ent.get("llm_path") or "")
    method = str(ent.get("method") or "")
    st0 = ent.get("http_status")
    sec0 = ent.get("seconds")
    label = str(ent.get("label") or "").strip()

    try:
        st_i = int(st0)
    except Exception:
        st_i = None
    try:
        sec_f = float(sec0)
    except Exception:
        sec_f = None

    tr = ent.get("tool_result") if isinstance(ent.get("tool_result"), dict) else {}
    rj = tr.get("response_json") if isinstance(tr.get("response_json"), dict) else None
    rt = tr.get("response_text") if isinstance(tr.get("response_text"), str) else ""
    api_rj = ent.get("api_response_json") if isinstance(ent.get("api_response_json"), dict) else None
    shape_rj = api_rj if isinstance(api_rj, dict) else rj

    if st_i is None:
        issues.append({"severity": "error", "kind": "missing_http_status", "summary": f"Missing http_status for {method} {llm_path}"})
    elif st_i >= 400:
        sev = "error" if st_i >= 500 else "warn"
        kind = "rate_limited" if st_i == 429 else "http_error"
        issues.append({"severity": sev, "kind": kind, "summary": f"HTTP {st_i} for {method} {llm_path}"})

    if sec_f is not None:
        slow = 60.0
        if llm_path in ("/api/health",):
            slow = 5.0
        if llm_path in ("/api/game/state", "/api/tests/disease/models", "/api/tests/disease/proteins", "/api/bulk_omics/sets", "/api/spatial_tx/gene_sets", "/api/omics/inventory"):
            slow = 10.0
        if llm_path in ("/api/tests/disease/bulk_omics", "/api/tests/disease/spatial_tx", "/api/tests/disease/characterization", "/api/tests/disease/protein_screen", "/api/tests/disease/claim_cure"):
            slow = 120.0
        if llm_path in ("/api/omics/analyze",):
            slow = 240.0
        if sec_f >= float(slow):
            issues.append({"severity": "warn", "kind": "slow", "summary": f"Slow response: {sec_f:.1f}s for {method} {llm_path}"})

    if isinstance(rj, dict):
        if rj.get("ok") is False:
            issues.append({"severity": "warn", "kind": "ok_false", "summary": f"ok=false in TOOL_RESULT for {method} {llm_path}"})
        if isinstance(rj.get("error"), str) and str(rj.get("error") or "").strip():
            issues.append({"severity": "warn", "kind": "error_field", "summary": f"error field present in TOOL_RESULT for {method} {llm_path}"})
    else:
        if st_i is not None and st_i < 400:
            issues.append({"severity": "warn", "kind": "missing_response_json", "summary": f"Missing TOOL_RESULT.response_json for {method} {llm_path}"})

    if isinstance(api_rj, dict):
        if api_rj.get("ok") is False:
            issues.append({"severity": "warn", "kind": "api_ok_false", "summary": f"ok=false in raw API response for {method} {llm_path}"})
        if isinstance(api_rj.get("error"), str) and str(api_rj.get("error") or "").strip():
            issues.append({"severity": "warn", "kind": "api_error_field", "summary": f"error field present in raw API response for {method} {llm_path}"})

    if isinstance(rt, str) and rt.strip():
        issues.extend(_issues_from_text(text=rt, source="preflight_response_text", max_items=20))

    if llm_path == "/api/health":
        if not (isinstance(shape_rj, dict) and shape_rj.get("ok") is True):
            issues.append({"severity": "warn", "kind": "health_unexpected", "summary": "Expected /api/health to return ok=true"})

    if llm_path == "/api/game/state":
        game = shape_rj.get("game") if isinstance(shape_rj, dict) else None
        if not isinstance(game, dict):
            issues.append({"severity": "warn", "kind": "missing_game", "summary": "Expected /api/game/state TOOL_RESULT to include game{}"})

    if llm_path == "/api/tests/disease/models":
        models = shape_rj.get("models") if isinstance(shape_rj, dict) else None
        if not (isinstance(models, list) and any(str(x or "").strip() for x in models)):
            issues.append({"severity": "warn", "kind": "empty_models", "summary": "Expected non-empty models list"})

    if llm_path == "/api/tests/disease/proteins":
        prots = shape_rj.get("proteins") if isinstance(shape_rj, dict) else None
        if not (isinstance(prots, list) and any(str(x or "").strip() for x in prots)):
            issues.append({"severity": "warn", "kind": "empty_proteins", "summary": "Expected non-empty proteins list"})

    if llm_path == "/api/bulk_omics/sets":
        sets0 = shape_rj.get("sets") if isinstance(shape_rj, dict) else None
        if not (isinstance(sets0, list) and any(str(x or "").strip() for x in sets0)):
            issues.append({"severity": "warn", "kind": "empty_bulk_sets", "summary": "Expected non-empty bulk omics sets"})

    if llm_path == "/api/spatial_tx/gene_sets":
        gs = shape_rj.get("gene_sets") if isinstance(shape_rj, dict) else None
        if not (isinstance(gs, list) and any(str(x or "").strip() for x in gs)):
            issues.append({"severity": "warn", "kind": "empty_gene_sets", "summary": "Expected non-empty gene_sets"})

    if llm_path == "/api/tests/disease/estimate_cost":
        charge = shape_rj.get("charge") if isinstance(shape_rj, dict) else None
        if not isinstance(charge, dict):
            issues.append({"severity": "warn", "kind": "missing_charge", "summary": "Expected estimate_cost to return charge{}"})

    if llm_path in ("/api/tests/disease/characterization", "/api/tests/disease/bulk_omics", "/api/tests/disease/spatial_tx", "/api/tests/disease/protein_screen"):
        files0 = shape_rj.get("files") if isinstance(shape_rj, dict) else None
        arts0 = shape_rj.get("artifacts") if isinstance(shape_rj, dict) else None
        if not (isinstance(files0, list) or isinstance(arts0, list)):
            issues.append({"severity": "warn", "kind": "missing_files", "summary": "Expected experiment response to include files/artifacts"})

    if llm_path == "/api/omics/inventory":
        files0 = shape_rj.get("files") if isinstance(shape_rj, dict) else None
        if not (isinstance(files0, list) and files0):
            issues.append({"severity": "warn", "kind": "empty_inventory", "summary": "Expected omics inventory to include non-empty files"})

    if llm_path == "/api/omics/analyze":
        out_txt = shape_rj.get("output_text") if isinstance(shape_rj, dict) else None
        if not (isinstance(out_txt, str) and out_txt.strip()):
            issues.append({"severity": "warn", "kind": "empty_output_text", "summary": "Expected /api/omics/analyze to return non-empty output_text"})

    if llm_path == "/preflight/llm_ping":
        out_txt = shape_rj.get("output_text") if isinstance(shape_rj, dict) else None
        if not (isinstance(out_txt, str) and "pong" in out_txt.lower()):
            extra = f" ({label})" if label else ""
            issues.append({"severity": "warn", "kind": "llm_ping_failed", "summary": f"LLM ping did not return 'pong'{extra}"})

    return issues


def _preflight_inspect(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ent in enumerate(rows or []):
        if not isinstance(ent, dict):
            continue
        label = ent.get("label")
        issues = _preflight_inspect_row(ent)
        if not issues:
            out.append(
                {
                    "i": int(i + 1),
                    "severity": "ok",
                    "kind": "ok",
                    "method": ent.get("method"),
                    "llm_path": ent.get("llm_path"),
                    "label": label,
                    "http_status": ent.get("http_status"),
                    "seconds": ent.get("seconds"),
                    "summary": "OK",
                    "details": None,
                }
            )
            continue
        for iss in issues:
            out.append(
                {
                    "i": int(i + 1),
                    "severity": str(iss.get("severity") or "warn"),
                    "kind": str(iss.get("kind") or "issue"),
                    "method": ent.get("method"),
                    "llm_path": ent.get("llm_path"),
                    "label": label,
                    "http_status": ent.get("http_status"),
                    "seconds": ent.get("seconds"),
                    "summary": str(iss.get("summary") or ""),
                    "details": iss,
                }
            )
    return out


def _latest_game_money(events: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[float]]:
    # Prefer /api/game/state responses.
    cents: Optional[int] = None
    usd: Optional[float] = None
    for ev in reversed(events):
        if not isinstance(ev, dict):
            continue
        if ev.get("type") != "api":
            continue
        if ev.get("path") != "/api/game/state":
            continue
        rj = ev.get("response_json")
        if not isinstance(rj, dict):
            continue
        game = rj.get("game")
        if not isinstance(game, dict):
            continue
        try:
            cents = int(game.get("money_spent_cents"))
        except Exception:
            cents = None
        try:
            usd = float(game.get("money_spent_usd"))
        except Exception:
            usd = None
        break
    return cents, usd


def _latest_end_metrics(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for ev in reversed(events):
        if isinstance(ev, dict) and ev.get("type") == "end":
            return ev
    return None


def _csv_to_rows(text: str, *, max_rows: int = 25, max_cols: int = 24) -> List[Dict[str, Any]]:
    s = str(text or "").strip("\n")
    if not s:
        return []
    buf = io.StringIO(s)
    try:
        reader = csv.reader(buf)
        rows = list(reader)
    except Exception:
        return []

    if not rows:
        return []
    hdr = rows[0]
    data = rows[1:]
    hdr = [str(h) for h in rows[0][:max_cols]]
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i >= int(max_rows):
            break
        out.append({str(k): v for k, v in zip(hdr, row) if str(k)})
    return out


def _count_csv_rows(path: Path) -> int:
    try:
        if not path.exists() or not path.is_file():
            return 0
        n = 0
        with open(str(path), "r", encoding="utf-8", errors="replace") as f:
            rr = csv.DictReader(f)
            for _ in rr:
                n += 1
        return int(n)
    except Exception:
        return 0


def _issues_from_text(*, text: str, source: str, max_items: int = 120) -> List[Dict[str, Any]]:
    s = str(text or "")
    if not s.strip():
        return []
    out: List[Dict[str, Any]] = []
    pats: List[Tuple[str, str, str]] = [
        (r"\btraceback\b", "error", "traceback"),
        (r"\b(exception|assertionerror|runtimeerror|valueerror|typeerror|keyerror)\b", "error", "exception"),
        (r"\b(llm_call_failed|http\s+\d{3})\b", "error", "llm_or_http_error"),
        (r"\b(429|rate\s*limit|quota|retry[-\s]?after|retrydelay)\b", "warn", "rate_limit_or_retry"),
        (r"\b(max[_\s]?tokens|max[_\s]?output[_\s]?tokens|stop_reason\s*[:=]\s*max_tokens)\b", "warn", "possible_truncation"),
        (r"\b(timeout|timed\s*out)\b", "warn", "timeout"),
    ]
    for ln in s.splitlines():
        if len(out) >= int(max_items):
            break
        line = str(ln or "")
        if not line.strip():
            continue
        hit: Optional[Tuple[str, str]] = None
        for pat, sev, kind in pats:
            try:
                if re.search(pat, line, flags=re.IGNORECASE):
                    hit = (sev, kind)
                    break
            except Exception:
                continue
        if hit is None:
            continue
        sev, kind = hit
        out.append(
            {
                "severity": str(sev),
                "kind": str(kind),
                "source": str(source),
                "seq": None,
                "ts": None,
                "summary": line[:3200],
                "details": line,
            }
        )
    return out


def _detect_issues(events: List[Dict[str, Any]], paths: RunPaths) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        t = str(ev.get("type") or "")
        seq = ev.get("seq")
        ts = ev.get("ts")

        if t == "llm_error":
            msg = str(ev.get("error") or "")
            out.append(
                {
                    "severity": "error",
                    "kind": "llm_error",
                    "source": "events",
                    "seq": seq,
                    "ts": ts,
                    "summary": msg[:3200] if msg else "llm_error",
                    "details": ev,
                }
            )
            continue

        if t == "player_id_mismatch":
            details = ev.get("details") if isinstance(ev.get("details"), dict) else {}
            found = details.get("found")
            exp = details.get("expected")
            out.append(
                {
                    "severity": "warn",
                    "kind": "player_id_mismatch",
                    "source": "events",
                    "seq": seq,
                    "ts": ts,
                    "summary": f"player_id mismatch (found={found} expected={exp})",
                    "details": ev,
                }
            )
            continue

        if t == "llm":
            txt = str(ev.get("text") or "")
            obj = _parse_first_json(txt)
            if obj is None:
                out.append(
                    {
                        "severity": "warn",
                        "kind": "llm_malformed_output",
                        "source": "events",
                        "seq": seq,
                        "ts": ts,
                        "summary": "LLM output did not parse as JSON object (possible truncation / formatting issue)",
                        "details": txt[:12000],
                    }
                )
            else:
                act = str(obj.get("action") or "").strip().lower()
                if act and act not in ("call_api", "final"):
                    out.append(
                        {
                            "severity": "warn",
                            "kind": "llm_unexpected_action",
                            "source": "events",
                            "seq": seq,
                            "ts": ts,
                            "summary": f"Unexpected action='{act}'",
                            "details": obj,
                        }
                    )
            continue

        if t == "final_rejected":
            out.append(
                {
                    "severity": "warn",
                    "kind": "final_rejected",
                    "source": "events",
                    "seq": seq,
                    "ts": ts,
                    "summary": "Final action was rejected (format/validation issue)",
                    "details": ev,
                }
            )
            continue

        if t == "api":
            status = ev.get("http_status")
            path = str(ev.get("path") or "")
            method = str(ev.get("method") or "")
            seconds = ev.get("seconds")
            sev = None
            kind = None
            summary = ""
            try:
                st_i = int(status)
            except Exception:
                st_i = None

            if st_i is not None and st_i >= 400:
                sev = "error" if st_i >= 500 else "warn"
                kind = "api_http_error"
                summary = f"API {method} {path} -> HTTP {st_i}"
                if st_i == 429:
                    sev = "warn"
                    kind = "api_rate_limited"
            rj = ev.get("response_json")
            if sev is None and isinstance(rj, dict):
                if rj.get("ok") is False:
                    sev = "warn"
                    kind = "api_ok_false"
                    summary = f"API {method} {path} returned ok=false"
                if isinstance(rj.get("error"), str) and str(rj.get("error") or "").strip():
                    sev = "warn" if sev is None else sev
                    kind = "api_error_field" if kind is None else kind
                    if not summary:
                        summary = f"API {method} {path} returned error field"

            if sev is not None:
                out.append(
                    {
                        "severity": str(sev),
                        "kind": str(kind or "api_issue"),
                        "source": "events",
                        "seq": seq,
                        "ts": ts,
                        "summary": summary[:3200] if summary else f"API {method} {path}",
                        "details": ev,
                    }
                )
            try:
                if seconds is not None and float(seconds) >= 120.0:
                    out.append(
                        {
                            "severity": "warn",
                            "kind": "api_slow",
                            "source": "events",
                            "seq": seq,
                            "ts": ts,
                            "summary": f"Slow API call: {method} {path} ({float(seconds):.1f}s)",
                            "details": ev,
                        }
                    )
            except Exception:
                pass
            continue

        if t == "tool_result":
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            status = payload.get("http_status")
            path = str(ev.get("path") or payload.get("path") or "")
            sev = None
            kind = None
            try:
                st_i = int(status)
            except Exception:
                st_i = None
            if st_i is not None and st_i >= 400:
                sev = "error" if st_i >= 500 else "warn"
                kind = "tool_http_error"
                if st_i == 429:
                    sev = "warn"
                    kind = "tool_rate_limited"
            rj = payload.get("response_json")
            if sev is None and isinstance(rj, dict):
                if rj.get("ok") is False:
                    sev = "warn"
                    kind = "tool_ok_false"
                if isinstance(rj.get("error"), str) and str(rj.get("error") or "").strip():
                    sev = "warn" if sev is None else sev
                    kind = "tool_error_field" if kind is None else kind
            rt = payload.get("response_text")
            if sev is None and isinstance(rt, str) and rt.strip():
                if re.search(r"max[_\s]?tokens|stop_reason\s*[:=]\s*max_tokens", rt, flags=re.IGNORECASE):
                    sev = "warn"
                    kind = "possible_truncation"
            if sev is not None:
                out.append(
                    {
                        "severity": str(sev),
                        "kind": str(kind or "tool_issue"),
                        "source": "events",
                        "seq": seq,
                        "ts": ts,
                        "summary": f"Tool result issue for {path}"[:3200],
                        "details": ev,
                    }
                )
            continue

    out.extend(_issues_from_text(text=_tail_text(paths.stderr_path), source="runner_stderr"))
    out.extend(_issues_from_text(text=_tail_text(paths.stdout_path), source="runner_stdout"))

    def _sort_key(x: Dict[str, Any]) -> Tuple[int, float]:
        seq0 = x.get("seq")
        try:
            s0 = int(seq0)
        except Exception:
            s0 = -1
        ts0 = x.get("ts")
        try:
            t0 = float(ts0)
        except Exception:
            t0 = 0.0
        return (s0, t0)

    out.sort(key=_sort_key)
    return out


def _tail_text(path: Path, *, max_bytes: int = 80_000) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        sz = path.stat().st_size
        if sz <= 0:
            return ""
        with path.open("rb") as f:
            if sz > max_bytes:
                f.seek(sz - max_bytes)
            raw = f.read()
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def _tail_jsonl_lines(path: Path, *, max_lines: int, max_scan_bytes: int = 24_000_000, chunk_size: int = 128_000) -> List[str]:
    try:
        if not path.exists() or not path.is_file():
            return []
        want = max(1, int(max_lines))
        sz = path.stat().st_size
        if sz <= 0:
            return []
        buf = b""
        pos = int(sz)
        scanned = 0
        with path.open("rb") as f:
            while pos > 0 and buf.count(b"\n") <= (want + 2) and scanned < int(max_scan_bytes):
                read_sz = min(int(chunk_size), pos)
                pos -= read_sz
                f.seek(pos)
                chunk = f.read(read_sz)
                buf = chunk + buf
                scanned += int(read_sz)

        if pos > 0:
            i = buf.find(b"\n")
            if i >= 0:
                buf = buf[i + 1 :]
        lines_b = [ln for ln in buf.splitlines() if ln.strip()]
        if len(lines_b) > want:
            lines_b = lines_b[-want:]
        return [ln.decode("utf-8", errors="replace") for ln in lines_b]
    except Exception:
        return []


def _parse_first_json(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "")
    i0 = s.find("{")
    if i0 < 0:
        return None
    i1 = s.rfind("}")
    if i1 < 0 or i1 <= i0:
        return None
    try:
        obj = json.loads(s[i0 : i1 + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _llm_reason_fields(ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    lrs = ev.get("last_result_summary")
    nsr = ev.get("next_step_rationale")
    if isinstance(lrs, str) or isinstance(nsr, str):
        return (lrs if isinstance(lrs, str) else None, nsr if isinstance(nsr, str) else None)
    obj = _parse_first_json(str(ev.get("text") or ""))
    if not isinstance(obj, dict):
        return (None, None)
    lrs2 = obj.get("last_result_summary")
    nsr2 = obj.get("next_step_rationale")
    return (lrs2 if isinstance(lrs2, str) else None, nsr2 if isinstance(nsr2, str) else None)


def _read_events(path: Path, *, max_events: Optional[int] = 600, read_all: bool = False) -> List[Dict[str, Any]]:
    try:
        if not path.exists() or not path.is_file():
            return []
        events: List[Dict[str, Any]] = []
        if read_all or max_events is None:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    if not str(ln).strip():
                        continue
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict):
                            events.append(obj)
                    except Exception:
                        continue
        else:
            lines = _tail_jsonl_lines(path, max_lines=int(max_events))
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        events.append(obj)
                except Exception:
                    continue
        events.sort(key=lambda e: int(e.get("seq") or 0))
        if max_events is not None and len(events) > int(max_events):
            events = events[-int(max_events) :]
        return events
    except Exception:
        return []


def _api_response_summary(rj: Any) -> Any:
    if not isinstance(rj, dict):
        if isinstance(rj, list):
            return {"type": "list", "n": len(rj)}
        if isinstance(rj, str):
            return {"type": "str", "chars": len(rj)}
        return {"type": str(type(rj))}
    keys = list(rj.keys())
    keep: Dict[str, Any] = {}
    for k in (
        "ok",
        "error",
        "experiment",
        "win",
        "delta_median_ticks",
        "extra_days",
        "lifespan_recovery_pct",
        "score",
        "score_lifedays_per_usd",
        "money_spent_usd",
        "money_spent_cents",
    ):
        if k in rj:
            keep[k] = rj.get(k)
    files0 = rj.get("files") if isinstance(rj.get("files"), list) else None
    if files0 is not None:
        keep["files_n"] = int(len(files0))
    arts0 = rj.get("artifacts") if isinstance(rj.get("artifacts"), list) else None
    if arts0 is not None:
        keep["artifacts_n"] = int(len(arts0))
    keep["keys_n"] = int(len(keys))
    keep["keys_head"] = [str(k) for k in keys[:40]]
    return keep


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _llm_payloads_dir(paths: RunPaths) -> Path:
    return paths.run_dir / "llm_payloads"


def _latest_llm_payload_path(paths: RunPaths) -> Optional[Path]:
    d = _llm_payloads_dir(paths)
    if not d.exists() or not d.is_dir():
        return None
    best: Optional[Tuple[int, Path]] = None
    for p in d.glob("llm_payload_step_*.json"):
        try:
            m = re.search(r"llm_payload_step_(\d+)\.json$", p.name)
            if not m:
                continue
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
        except Exception:
            continue
    return best[1] if best is not None else None


def _payload_text_blocks(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts: List[str] = []
        for b in v:
            if not isinstance(b, dict):
                continue
            if str(b.get("type") or "").strip().lower() == "text":
                parts.append(str(b.get("text") or ""))
        return "".join(parts)
    return str(v)


def _latest_llm_prompt_text(paths: RunPaths, *, max_chars: int = 120_000) -> Tuple[str, str]:
    p = _latest_llm_payload_path(paths)
    if p is None:
        return "", ""
    raw = ""
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", ""

    h = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    obj: Any = None
    try:
        obj = json.loads(raw)
    except Exception:
        obj = None

    header = f"LLM_PAYLOAD_FILE: {p.name}"
    if isinstance(obj, dict):
        prov = str(obj.get("provider") or "").strip().lower()
        if prov == "openai" and isinstance(obj.get("request"), dict):
            req = obj.get("request")
            msgs = req.get("messages") if isinstance(req.get("messages"), list) else []
            lines: List[str] = [header, f"provider=openai model={req.get('model')}", "", "MESSAGES:"]
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "")
                content = _payload_text_blocks(m.get("content"))
                lines.append(f"[{role}]\n{content}\n")
            txt = "\n".join(lines).strip()
        elif ("messages" in obj) and ("model" in obj):
            msgs = obj.get("messages") if isinstance(obj.get("messages"), list) else []
            sys_blocks = obj.get("system")
            sys_txt = _payload_text_blocks(sys_blocks)
            lines2: List[str] = [header, f"provider=anthropic model={obj.get('model')}"]
            if sys_txt.strip():
                lines2.extend(["", "SYSTEM:", sys_txt, ""])
            lines2.append("MESSAGES:")
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "")
                content = _payload_text_blocks(m.get("content"))
                lines2.append(f"[{role}]\n{content}\n")
            txt = "\n".join(lines2).strip()
        else:
            txt = header + "\n\n" + raw
    else:
        txt = header + "\n\n" + raw

    if len(txt) > int(max_chars):
        txt = txt[-int(max_chars) :]
    return txt, h


def _read_lab_notebook(paths: RunPaths) -> str:
    st_state = _load_json(paths.state_path)
    if not isinstance(st_state, dict):
        return ""
    return str(st_state.get("notebook") or "").strip()


def _extract_files(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") != "api":
            continue
        items = ev.get("files")
        if not isinstance(items, list):
            items = ev.get("artifacts")
        if not isinstance(items, list):
            continue
        for a in items:
            if not isinstance(a, dict):
                continue
            ent = dict(a)
            ent["event_seq"] = ev.get("seq")
            ent["path"] = ev.get("path")
            ent["method"] = ev.get("method")
            out.append(ent)
    return out


def _best_claim_cure(events: List[Dict[str, Any]], *, challenge: str = "cancer") -> Tuple[Optional[float], Optional[bool]]:
    best_abs: Optional[float] = None
    best_val: Optional[float] = None
    best_win: Optional[bool] = None
    ch = str(challenge or "").strip().lower()
    if ch == "cancer":
        want = "/api/tests/cancer/claim_cure"
    elif ch == "hereditary_disease":
        want = "/api/tests/hereditary_disease/claim_cure"
    else:
        want = "/api/tests/aging/claim_cure"
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") != "api":
            continue
        if ev.get("path") != want:
            continue
        rj = ev.get("response_json")
        if not isinstance(rj, dict):
            continue
        try:
            d = float(rj.get("delta_median_ticks"))
        except Exception:
            continue
        w = bool(rj.get("win") is True)
        a = abs(d)
        if best_abs is None or a < best_abs:
            best_abs = a
            best_val = d
            best_win = w
    return best_val, best_win


def _best_claim_cure_recovery(events: List[Dict[str, Any]], *, challenge: str = "cancer") -> Optional[float]:
    best_pct: Optional[float] = None
    ch = str(challenge or "").strip().lower()
    if ch == "cancer":
        want = "/api/tests/cancer/claim_cure"
    elif ch == "hereditary_disease":
        want = "/api/tests/hereditary_disease/claim_cure"
    else:
        want = "/api/tests/aging/claim_cure"
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") != "api":
            continue
        if ev.get("path") != want:
            continue
        rj = ev.get("response_json")
        if not isinstance(rj, dict):
            continue
        try:
            pct = rj.get("lifespan_recovery_pct")
            pct_f = float(pct) if pct is not None else None
        except Exception:
            pct_f = None
        if pct_f is None:
            continue
        if best_pct is None or float(pct_f) > float(best_pct):
            best_pct = float(pct_f)
    return best_pct


def _best_claim_cure_score(events: List[Dict[str, Any]], *, challenge: str = "cancer") -> Tuple[Optional[float], Optional[float], Optional[int]]:
    best_score: Optional[float] = None
    best_extra: Optional[float] = None
    best_seq: Optional[int] = None
    ch = str(challenge or "").strip().lower()
    if ch == "cancer":
        want = "/api/tests/cancer/claim_cure"
    elif ch == "hereditary_disease":
        want = "/api/tests/hereditary_disease/claim_cure"
    else:
        want = "/api/tests/aging/claim_cure"
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") != "api":
            continue
        if ev.get("path") != want:
            continue
        rj = ev.get("response_json")
        if not isinstance(rj, dict):
            continue
        try:
            s = rj.get("score")
            score = float(s) if s is not None else None
        except Exception:
            score = None
        if score is None:
            try:
                s2 = rj.get("score_lifedays_per_usd")
                score2 = float(s2) if s2 is not None else None
            except Exception:
                score2 = None
            if score2 is not None:
                try:
                    score = float(score2) * 10000.0
                except Exception:
                    score = None
        if score is None:
            continue
        if best_score is None or float(score) > float(best_score):
            best_score = float(score)
            try:
                best_extra = float(rj.get("extra_days"))
            except Exception:
                best_extra = None
            try:
                best_seq = int(ev.get("seq") or 0)
            except Exception:
                best_seq = None
    return best_score, best_extra, best_seq


def _api_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") != "api":
            continue
        rj = ev.get("response_json") if isinstance(ev.get("response_json"), dict) else {}
        rows.append(
            {
                "seq": ev.get("seq"),
                "method": ev.get("method"),
                "path": ev.get("path"),
                "status": ev.get("http_status"),
                "seconds": ev.get("seconds"),
                "experiment": rj.get("experiment") if isinstance(rj, dict) else None,
                "win": rj.get("win") if isinstance(rj, dict) else None,
                "delta_median_ticks": rj.get("delta_median_ticks") if isinstance(rj, dict) else None,
                "extra_days": rj.get("extra_days") if isinstance(rj, dict) else None,
                "score_lifedays_per_usd": rj.get("score_lifedays_per_usd") if isinstance(rj, dict) else None,
                "lifespan_recovery_pct": rj.get("lifespan_recovery_pct") if isinstance(rj, dict) else None,
                "score": rj.get("score") if isinstance(rj, dict) else None,
            }
        )
    return rows


def _truncate(s: Any, n: int) -> str:
    t = str(s or "")
    t = " ".join(t.split())
    if len(t) <= n:
        return t
    return t[: max(0, n - 1)] + "…"


def _mask_disease_term(text: str) -> str:
    t = str(text or "")
    if not t:
        return ""
    t = t.replace("/api/tests/cancer/", "/api/tests/disease/")
    t = t.replace("/api/tests/cancer", "/api/tests/disease")
    t = t.replace("/api/tests/hereditary_disease/", "/api/tests/disease/")
    t = t.replace("/api/tests/hereditary_disease", "/api/tests/disease")
    t = re.sub(r"cancerous", "diseased", t, flags=re.IGNORECASE)
    t = re.sub(r"cancer", "disease", t, flags=re.IGNORECASE)
    return t


def _json_compact(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def _preflight_call(
    *,
    base_url: str,
    llm_path: str,
    method: str,
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 120.0,
    omics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runner = _load_runner_module()

    server_path = runner._llm_path_to_server(str(llm_path or ""))
    q = dict(query) if isinstance(query, dict) else None
    if isinstance(q, dict) and "model" in q:
        q = dict(q)
        q["model"] = runner._llm_model_key_to_server(q.get("model"))

    b = runner._llm_body_to_server(server_path, body) if isinstance(body, dict) else body
    res = runner.call_local_api(
        base_url=str(base_url),
        method=str(method),
        path=str(server_path),
        query=q,
        body=b,
        timeout_s=float(timeout_s),
    )

    llm_json = runner._llm_response_json(server_path, res.response_json)
    if isinstance(llm_json, dict):
        if server_path in (
            "/api/tests/cancer/bulk_omics",
            "/api/tests/cancer/spatial_tx",
            "/api/tests/hereditary_disease/bulk_omics",
            "/api/tests/hereditary_disease/spatial_tx",
        ):
            llm_json.pop("matrix_noisy_csv", None)
            llm_json.pop("metadata_csv", None)

    compact = runner._llm_tool_result_compact(
        server_path,
        llm_json if isinstance(llm_json, dict) else None,
        omics_state=omics_state if isinstance(omics_state, dict) else {},
    )

    tool_payload: Dict[str, Any] = {
        "http_status": res.http_status,
        "seconds": res.seconds,
        "response_json": compact,
    }
    if res.response_json is None:
        tool_payload["response_text"] = runner._llm_sanitize_text(str(res.response_text or "")[:400])

    return {
        "method": str(method).upper(),
        "llm_path": str(llm_path),
        "server_path": str(server_path),
        "query": q,
        "body": b,
        "http_status": int(res.http_status),
        "seconds": float(res.seconds),
        "tool_result": tool_payload,
        "api_response_json": res.response_json,
    }


def _preflight_llm_ping(*, provider: str, model: str, timeout_s: float) -> Dict[str, Any]:
    runner = _load_runner_module()
    prov0 = str(provider or "").strip().lower() or "openai"
    if prov0 == "grok":
        prov0 = "xai"
    label = f"{prov0}:{str(model or '').strip()}"
    temp = 0.0
    if prov0 in ("claude", "anthropic") and str(model or "").strip() == "claude-opus-4-5-20251101":
        temp = 1.0
    max_tokens_i = 32
    if prov0 in ("claude", "anthropic") and str(model or "").strip() == "claude-opus-4-5-20251101":
        max_tokens_i = 1025
    if prov0 in ("gemini",):
        max_tokens_i = max(int(max_tokens_i), 256)
    t0 = time.time()
    ok = False
    out_text = ""
    err_text = ""
    try:
        out_text = runner.llm_generate(
            provider=str(prov0),
            model=str(model),
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            temperature=float(temp),
            max_tokens=int(max_tokens_i),
            timeout_s=float(timeout_s),
        )
        ok = bool(isinstance(out_text, str) and ("pong" in out_text.lower()))
    except Exception as e:
        err_text = str(e) or repr(e)
        ok = False

    sec = float(time.time() - t0)
    http_status = 200 if ok else 500
    api_rj: Dict[str, Any] = {"ok": bool(ok), "provider": str(prov0), "model": str(model), "output_text": str(out_text or "")}
    if not ok and err_text:
        api_rj["error"] = str(err_text)

    tool_payload: Dict[str, Any] = {"http_status": int(http_status), "seconds": float(sec), "response_json": dict(api_rj)}
    if not ok and err_text:
        tool_payload["response_text"] = runner._llm_sanitize_text(str(err_text)[:400])

    return {
        "method": "LLM",
        "llm_path": "/preflight/llm_ping",
        "server_path": "/preflight/llm_ping",
        "label": str(label),
        "query": None,
        "body": {"provider": str(prov0), "model": str(model)},
        "http_status": int(http_status),
        "seconds": float(sec),
        "tool_result": tool_payload,
        "api_response_json": dict(api_rj),
    }


def _llm_provider_model_options() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    openai_models = [
        "gpt-5.2",
        "gpt-5.2-medium",
        "gpt-5.2-high",
        "gpt-5.2-extra-high",
    ]
    for m in openai_models:
        out.append(("openai", str(m)))

    xai_models = [
        "grok-4",
        "grok-4-1-fast",
        "grok-4-1-fast-reasoning",
    ]
    for m in xai_models:
        out.append(("xai", str(m)))

    gemini_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3-pro-preview",
    ]
    for m in gemini_models:
        out.append(("gemini", str(m)))

    claude_models = [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
    ]
    for m in claude_models:
        out.append(("claude", str(m)))

    return out


def _run_preflight_checks(
    *,
    challenge: str,
    base_url: str,
    preflight_player_id: str,
    ticks: int,
    replicates: int,
    api_timeout_s: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    omics_state: Dict[str, Any] = {}

    runner = _load_runner_module()
    try:
        runner._BENCH_CHALLENGE = str(getattr(runner, "_normalize_challenge")(challenge))
    except Exception:
        runner._BENCH_CHALLENGE = "cancer"

    pid = str(preflight_player_id or "").strip() or "preflight"
    ticks_i = max(1, int(ticks))
    reps_i = max(1, int(replicates))
    test_ticks_i = int(min(int(ticks_i), 60))

    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/game/reset",
            method="POST",
            body={"player_id": pid},
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )

    for prov, model in _llm_provider_model_options():
        out.append(_preflight_llm_ping(provider=str(prov), model=str(model), timeout_s=min(120.0, float(api_timeout_s))))

    out.append(_preflight_call(base_url=base_url, llm_path="/api/health", method="GET", timeout_s=api_timeout_s, omics_state=omics_state))
    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/game/state",
            method="GET",
            query={"player_id": pid},
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )

    models_resp = _preflight_call(base_url=base_url, llm_path="/api/tests/disease/models", method="GET", timeout_s=api_timeout_s, omics_state=omics_state)
    out.append(models_resp)
    model = "cell_culture_cancer"
    screen_model = "cell_culture_cancer"
    try:
        mrj = models_resp.get("api_response_json") if isinstance(models_resp, dict) else None
        if isinstance(mrj, dict) and isinstance(mrj.get("models"), list) and mrj.get("models"):
            mkeys: List[str] = []
            iv_keys: List[str] = []
            for m0 in (mrj.get("models") or []):
                if isinstance(m0, dict):
                    k0 = str(m0.get("key") or "").strip()
                    dom0 = str(m0.get("domain") or "").strip().lower()
                    if k0:
                        mkeys.append(k0)
                        if dom0 == "in_vitro":
                            iv_keys.append(k0)
                else:
                    k0 = str(m0 or "").strip()
                    if k0:
                        mkeys.append(k0)
            if mkeys:
                model = str(mkeys[0])
            if iv_keys:
                ch0 = str(challenge or "").strip().lower()
                preferred = ""
                if ch0 == "cancer":
                    preferred = "cell_culture_cancer"
                elif ch0 in ("hereditary_disease", "heredetary_disease"):
                    preferred = "cell_culture_disease"
                elif ch0 == "aging":
                    preferred = "cell_culture"
                if preferred and preferred in iv_keys:
                    screen_model = str(preferred)
                else:
                    screen_model = str(iv_keys[-1])
    except Exception:
        pass

    prot_resp = _preflight_call(
        base_url=base_url,
        llm_path="/api/tests/disease/proteins",
        method="GET",
        query={"model": str(model)},
        timeout_s=api_timeout_s,
        omics_state=omics_state,
    )
    out.append(prot_resp)
    prot_list = []
    try:
        prj = prot_resp.get("api_response_json") if isinstance(prot_resp, dict) else None
        if isinstance(prj, dict) and isinstance(prj.get("proteins"), list):
            prot_list = [str(x) for x in prj.get("proteins") if str(x).strip()]
    except Exception:
        prot_list = []
    example_iv = []
    if prot_list:
        example_iv = [{"layer": prot_list[0], "direction": "down", "dose": 1}]

    bulk_sets = _preflight_call(base_url=base_url, llm_path="/api/bulk_omics/sets", method="GET", timeout_s=api_timeout_s, omics_state=omics_state)
    out.append(bulk_sets)
    omics_set = "rna/Bulk RNAseq"
    try:
        brj = bulk_sets.get("api_response_json") if isinstance(bulk_sets, dict) else None
        if isinstance(brj, dict) and isinstance(brj.get("sets"), list) and brj.get("sets"):
            omics_set = str(brj.get("sets")[0])
    except Exception:
        pass

    gene_sets = _preflight_call(base_url=base_url, llm_path="/api/spatial_tx/gene_sets", method="GET", timeout_s=api_timeout_s, omics_state=omics_state)
    out.append(gene_sets)
    gene_set = "spatial transcriptomics"
    try:
        grj = gene_sets.get("api_response_json") if isinstance(gene_sets, dict) else None
        if isinstance(grj, dict) and isinstance(grj.get("gene_sets"), list) and grj.get("gene_sets"):
            gene_set = str(grj.get("gene_sets")[0])
    except Exception:
        pass

    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/tests/disease/estimate_cost",
            method="POST",
            body={
                "player_id": pid,
                "model": str(model),
                "ticks": int(test_ticks_i),
                "replicates": int(reps_i),
                "experiment": "characterization",
                "interventions": list(example_iv),
            },
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )

    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/tests/disease/characterization",
            method="POST",
            body={
                "player_id": pid,
                "model": str(model),
                "ticks": int(test_ticks_i),
                "replicates": int(reps_i),
                "interventions": list(example_iv),
            },
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )
    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/tests/disease/bulk_omics",
            method="POST",
            body={
                "player_id": pid,
                "model": str(model),
                "ticks": int(test_ticks_i),
                "replicates": int(reps_i),
                "omics_set": str(omics_set),
                "interventions": list(example_iv),
            },
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )
    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/tests/disease/spatial_tx",
            method="POST",
            body={
                "player_id": pid,
                "model": str(model),
                "ticks": int(test_ticks_i),
                "replicates": int(reps_i),
                "gene_set": str(gene_set),
                "interventions": list(example_iv),
            },
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )

    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/tests/disease/protein_screen",
            method="POST",
            body={
                "player_id": pid,
                "model": str(screen_model or model),
                "ticks": int(ticks_i),
                "replicates": 1,
                "interventions": list(example_iv),
            },
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )

    out.append(
        _preflight_call(
            base_url=base_url,
            llm_path="/api/tests/disease/claim_cure",
            method="POST",
            body={
                "player_id": pid,
                "replicates": 1,
                "interventions": list(example_iv),
            },
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
    )

    inv = _preflight_call(
        base_url=base_url,
        llm_path="/api/omics/inventory",
        method="GET",
        query={"player_id": pid},
        timeout_s=api_timeout_s,
        omics_state=omics_state,
    )
    out.append(inv)

    inv_raw = inv.get("api_response_json") if isinstance(inv, dict) else None
    inv_files = inv_raw.get("files") if isinstance(inv_raw, dict) else None
    file_ids: List[str] = []
    if isinstance(inv_files, list):
        counts = [f for f in inv_files if isinstance(f, dict) and str(f.get("role") or "").startswith("counts")]
        metas = [f for f in inv_files if isinstance(f, dict) and str(f.get("role") or "") in ("run_metadata", "cell_metadata")]
        if counts:
            file_ids.append(str(counts[0].get("file_id") or ""))
        if metas:
            file_ids.append(str(metas[0].get("file_id") or ""))
        for f in inv_files:
            if len(file_ids) >= 3:
                break
            if not isinstance(f, dict):
                continue
            fid = str(f.get("file_id") or "").strip()
            if fid and fid not in file_ids:
                file_ids.append(fid)
    file_ids = [x for x in file_ids if x]

    for prov, model in _llm_provider_model_options():
        ent = _preflight_call(
            base_url=base_url,
            llm_path="/api/omics/analyze",
            method="POST",
            body={
                "player_id": pid,
                "file_ids": list(file_ids),
                "provider": str(prov),
                "model": str(model),
                "instructions": "Reply with a single word: ok",
            },
            timeout_s=api_timeout_s,
            omics_state=omics_state,
        )
        ent["label"] = f"{str(prov)}:{str(model)}"
        out.append(ent)

    return out


def _render_preflight_ui(*, challenge: str, base_url: str, api_timeout_s: float, player_id: str) -> None:
    st.subheader("Preflight: test API endpoints (LLM-facing TOOL_RESULT)")
    st.caption("Runs quick calls using a separate player_id and shows exactly what the LLM would receive as TOOL_RESULT. Does not start the benchmark run.")

    default_pid = (str(player_id or "").strip() or "preflight") + "_preflight"
    with st.expander("Preflight settings", expanded=True):
        pid = st.text_input("Preflight player_id", value=default_pid, key="preflight_player_id")
        ticks = int(st.number_input("ticks", min_value=1, max_value=50, value=5, step=1, key="preflight_ticks"))
        reps = int(st.number_input("replicates", min_value=1, max_value=5, value=1, step=1, key="preflight_reps"))

    run_clicked = st.button("Run preflight checks", use_container_width=True, key="preflight_run_btn")
    if run_clicked:
        fp = _json_compact({"base_url": base_url, "pid": pid, "ticks": ticks, "reps": reps})
        with st.spinner("Running preflight calls..."):
            try:
                rows = _run_preflight_checks(
                    challenge=str(challenge),
                    base_url=base_url,
                    preflight_player_id=pid,
                    ticks=ticks,
                    replicates=reps,
                    api_timeout_s=api_timeout_s,
                )
            except Exception as e:
                rows = [{"error": str(e)}]
        st.session_state["preflight_last"] = {"fp": fp, "rows": rows, "ts": float(time.time())}

    last = st.session_state.get("preflight_last")
    if not isinstance(last, dict) or not isinstance(last.get("rows"), list):
        st.info("Run preflight checks to see example TOOL_RESULT payloads.")
        return

    rows = last.get("rows")
    ts = last.get("ts")
    if isinstance(ts, (int, float)):
        st.caption(f"Last preflight run: {max(0.0, time.time() - float(ts)):.1f}s ago")

    insp = _preflight_inspect(rows)
    bundle = {
        "bundle_format": "preflight_bundle_v1",
        "meta": {
            "fp": last.get("fp"),
            "ts": ts,
            "base_url": str(base_url),
            "challenge": str(challenge),
        },
        "rows": rows,
        "inspection": insp,
    }
    st.download_button(
        "Download preflight bundle JSON",
        data=json.dumps(bundle, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="preflight_bundle.json",
        mime="application/json",
        use_container_width=False,
    )
    errs = [x for x in insp if isinstance(x, dict) and str(x.get("severity") or "") == "error"]
    warns = [x for x in insp if isinstance(x, dict) and str(x.get("severity") or "") == "warn"]
    oks = [x for x in insp if isinstance(x, dict) and str(x.get("severity") or "") == "ok"]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Errors", len(errs))
    col_b.metric("Warnings", len(warns))
    col_c.metric("OK", len(oks))

    if errs:
        st.error("Preflight inspection found errors.")
    elif warns:
        st.warning("Preflight inspection found warnings.")
    else:
        st.success("Preflight inspection: all checks OK.")

    with st.expander("Inspection summary", expanded=True):
        st.dataframe(
            [
                {k: v for k, v in r.items() if k in ("i", "severity", "kind", "method", "llm_path", "label", "http_status", "seconds", "summary")}
                for r in insp
                if isinstance(r, dict)
            ],
            use_container_width=True,
            hide_index=True,
        )

    for i, ent in enumerate(rows):
        if not isinstance(ent, dict):
            continue
        llm_path = ent.get("llm_path")
        method = ent.get("method")
        st0 = ent.get("http_status")
        sec = ent.get("seconds")
        lbl = str(ent.get("label") or "").strip()
        lbl_txt = f" [{lbl}]" if lbl else ""
        title = f"{i+1}. {method} {llm_path}{lbl_txt} (status={st0} sec={sec})"
        with st.expander(title, expanded=False):
            issues_here = _preflight_inspect_row(ent)
            if issues_here:
                st.markdown("Inspection")
                for iss in issues_here[:12]:
                    sev = str(iss.get("severity") or "warn")
                    msg = str(iss.get("summary") or "")
                    if sev == "error":
                        st.error(msg)
                    else:
                        st.warning(msg)
            st.write({"llm_path": ent.get("llm_path"), "server_path": ent.get("server_path")})
            if ent.get("query") is not None:
                st.markdown("Query")
                st.json(ent.get("query"))
            if ent.get("body") is not None:
                st.markdown("Body")
                st.json(ent.get("body"))
            tr = ent.get("tool_result")
            if isinstance(tr, dict):
                st.markdown("LLM-facing TOOL_RESULT")
                st.json(tr)
            else:
                st.json(ent)


def _full_context_text(events: List[Dict[str, Any]]) -> str:
    evs = [e for e in events if isinstance(e, dict)]
    lines: List[str] = []
    for ev in evs:
        t = str(ev.get("type") or "")
        seq = ev.get("seq")
        if t == "start":
            lines.append(f"EVENT #{seq} type=start")
            lines.append(f"base_url={ev.get('base_url')} provider={ev.get('provider')} model={ev.get('model')} player_id={ev.get('player_id')}")
            p = ev.get("prompt")
            if isinstance(p, str) and p.strip():
                lines.append("PROMPT:")
                lines.append(p)
        elif t == "resume":
            lines.append(f"EVENT #{seq} type=resume player_id={ev.get('player_id')} next_step={ev.get('next_step')}")
        elif t == "llm":
            lines.append(f"EVENT #{seq} type=llm step={ev.get('step')}")
            lines.append("LLM_OUTPUT:")
            lines.append(str(ev.get("text") or ""))
        elif t == "api":
            lines.append(
                f"EVENT #{seq} type=api method={ev.get('method')} path={ev.get('path')} status={ev.get('http_status')} seconds={ev.get('seconds')}"
            )
            if ev.get("query") is not None:
                lines.append("QUERY_JSON:")
                lines.append(_json_compact(ev.get("query")))
            if ev.get("body") is not None:
                lines.append("BODY_JSON:")
                lines.append(_json_compact(ev.get("body")))
            rj = ev.get("response_json")
            if rj is not None:
                lines.append("RESPONSE_JSON:")
                lines.append(_json_compact(rj))
            items = ev.get("files")
            if not isinstance(items, list):
                items = ev.get("artifacts")
            if isinstance(items, list) and items:
                lines.append("FILES:")
                for a in items:
                    if isinstance(a, dict) and isinstance(a.get("path"), str) and a.get("path"):
                        lines.append(str(a.get("path")))
        elif t == "tool_result":
            lines.append(f"EVENT #{seq} type=tool_result tool={ev.get('tool')} path={ev.get('path')}")
            payload = ev.get("payload")
            if payload is not None:
                lines.append("PAYLOAD_JSON:")
                lines.append(_json_compact(payload))
        elif t == "llm_error":
            lines.append(f"EVENT #{seq} type=llm_error")
            lines.append(str(ev.get("error") or ""))
        elif t in ("final", "final_rejected", "end"):
            lines.append(f"EVENT #{seq} type={t}")
            if t == "end":
                lines.append(_json_compact(ev))
            else:
                lines.append(str(ev.get("text") or ev.get("payload") or ""))
        else:
            lines.append(f"EVENT #{seq} type={t}")
            lines.append(_json_compact(ev))

        lines.append("")

    return "\n".join(lines).strip()


def _split_chunks(text: str, *, max_chars: int = 14_000) -> List[str]:
    t = str(text or "")
    if not t:
        return [""]
    if len(t) <= int(max_chars):
        return [t]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for ln in t.splitlines(True):
        if cur and (cur_len + len(ln)) > int(max_chars):
            chunks.append("".join(cur))
            cur = []
            cur_len = 0
        cur.append(ln)
        cur_len += len(ln)
    if cur:
        chunks.append("".join(cur))
    return chunks


def _openai_chunk_notes(*, api_key: str, chunk: str, run_id: str, proc_alive: bool, chunk_i: int, chunk_n: int) -> str:
    if OpenAI is None:
        raise RuntimeError("openai python package not available")
    client = OpenAI(api_key=str(api_key))
    sys_msg = (
        "You are monitoring an LLM benchmark run. Summarize ONLY the provided log excerpt into factual notes. "
        "Do not speculate. Keep it short and concrete."
    )
    user_msg = (
        f"Run: {run_id}\n"
        f"Status: {'RUNNING' if proc_alive else 'STOPPED/DONE'}\n"
        f"Excerpt: {int(chunk_i) + 1}/{int(chunk_n)}\n"
        "\n"
        "Write 6-12 short bullet-like lines (plain text, one per line) capturing: key actions, key results, errors, and next intentions if stated. "
        "Do not add anything not present in the excerpt.\n"
        "\n"
        "LOG EXCERPT:\n"
        f"{_mask_disease_term(str(chunk or ''))}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=420,
            reasoning_effort="medium",
        )
    except TypeError:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=420,
        )
    out = None
    try:
        out = resp.choices[0].message.content
    except Exception:
        out = None
    return _mask_disease_term(str(out or "").strip())


def _openai_story_full(*, api_key: str, full_text: str, run_id: str, proc_alive: bool) -> str:
    chunks = _split_chunks(str(full_text or ""), max_chars=14_000)
    if len(chunks) <= 1:
        return _openai_story(api_key=api_key, digest=str(full_text or ""), run_id=run_id, proc_alive=proc_alive)
    notes: List[str] = []
    for i, ch in enumerate(chunks):
        notes.append(_openai_chunk_notes(api_key=api_key, chunk=ch, run_id=run_id, proc_alive=proc_alive, chunk_i=i, chunk_n=len(chunks)))
    merged = "\n\n".join([f"EXCERPT_NOTES {i+1}/{len(notes)}\n{n}" for i, n in enumerate(notes)]).strip()
    return _openai_story(api_key=api_key, digest=merged, run_id=run_id, proc_alive=proc_alive)


def _event_digest(events: List[Dict[str, Any]], *, max_events: int = 220, max_chars: int = 16_000) -> str:
    evs = [e for e in events if isinstance(e, dict)]
    evs = evs[-max_events:]
    lines: List[str] = []
    for ev in evs:
        t = str(ev.get("type") or "")
        seq = ev.get("seq")
        if t == "llm":
            act_obj = _parse_first_json(str(ev.get("text") or ""))
            act = act_obj.get("action") if isinstance(act_obj, dict) else None
            method = act_obj.get("method") if isinstance(act_obj, dict) else None
            path = act_obj.get("path") if isinstance(act_obj, dict) else None
            lrs, nsr = _llm_reason_fields(ev)
            lines.append(
                f"#{seq} LLM step={ev.get('step')} action={act} {method} {path} | lrs={_truncate(lrs, 160)} | nsr={_truncate(nsr, 180)}"
            )
        elif t == "api":
            rj = ev.get("response_json") if isinstance(ev.get("response_json"), dict) else {}
            extra = ""
            if isinstance(rj, dict):
                if "experiment" in rj:
                    extra += f" exp={rj.get('experiment')}"
                if "win" in rj:
                    extra += f" win={rj.get('win')}"
                if "delta_median_ticks" in rj:
                    extra += f" delta={rj.get('delta_median_ticks')}"
                if "error" in rj:
                    extra += f" error={_truncate(rj.get('error'), 120)}"
            lines.append(
                f"#{seq} API {ev.get('method')} {ev.get('path')} status={ev.get('http_status')} sec={ev.get('seconds')}{extra}"
            )
        elif t == "tool_result":
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            rj = payload.get("response_json") if isinstance(payload.get("response_json"), dict) else {}
            err = rj.get("error") if isinstance(rj, dict) else None
            err_s = f" error={_truncate(err, 140)}" if err else ""
            lines.append(
                f"#{seq} TOOL_RESULT {ev.get('tool')} {ev.get('path')} status={payload.get('http_status')} sec={payload.get('seconds')}{err_s}"
            )
        elif t == "llm_error":
            lines.append(f"#{seq} LLM_ERROR {_truncate(ev.get('error'), 240)}")
        elif t in ("final", "final_rejected", "end", "resume", "start"):
            lines.append(f"#{seq} {t} {_truncate(ev.get('path') or ev.get('payload') or '', 120)}")

    txt = "\n".join(lines).strip()
    if len(txt) <= max_chars:
        return txt
    return txt[-max_chars:]


def _local_story(events: List[Dict[str, Any]]) -> str:
    steps: List[str] = []
    for ev in reversed(events):
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        if t == "llm":
            act_obj = _parse_first_json(str(ev.get("text") or ""))
            act = act_obj.get("action") if isinstance(act_obj, dict) else None
            path = act_obj.get("path") if isinstance(act_obj, dict) else None
            lrs, nsr = _llm_reason_fields(ev)
            steps.append(
                f"step {ev.get('step')}: {act} {path} | { _truncate(lrs, 140) } -> { _truncate(nsr, 160) }"
            )
        if len(steps) >= 6:
            break
    if not steps:
        return "No steps yet."
    steps.reverse()
    return "\n".join(steps)


def _openai_story(*, api_key: str, digest: str, run_id: str, proc_alive: bool) -> str:
    if OpenAI is None:
        raise RuntimeError("openai python package not available")
    client = OpenAI(api_key=str(api_key))

    sys_msg = (
        "You are monitoring an LLM benchmark where an agent tries to solve a biological puzzle. "
        "Your job is to help a human evaluator understand what happened and what the agent is doing now. "
        "Do not speculate. Only use information present in the log digest. "
        "Be ultra-concise and clear."
    )
    user_msg = (
        f"Run: {run_id}\n"
        f"Status: {'RUNNING' if proc_alive else 'STOPPED/DONE'}\n"
        "\n"
        "Provide a high-level ultra concise summary of what is happening in just a few sentences. "
        "Frame it as a story. Mention: what it tried, what it learned, and what it will do next. "
        "If there were errors or obvious misinterpretations, mention them briefly. "
        "Constraints: 2-4 sentences total, <= 80 words, no bullets, no speculation.\n"
        "\n"
        "LOG DIGEST:\n"
        f"{_mask_disease_term(digest)}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=260,
            reasoning_effort="medium",
        )
    except TypeError:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=260,
        )
    out = None
    try:
        out = resp.choices[0].message.content
    except Exception:
        out = None
    return _mask_disease_term(str(out or "").strip())


def _ensure_session_state() -> None:
    st.session_state.setdefault("active_run_id", "")
    st.session_state.setdefault("proc", None)
    st.session_state.setdefault("active_suite_id", "")
    st.session_state.setdefault("suite_proc", None)
    st.session_state.setdefault("story_cache", {})
    st.session_state.setdefault("story_job", None)
    st.session_state.setdefault("story_job_meta", {})
    st.session_state.setdefault("bootstrapped", False)
    st.session_state.setdefault("suite_bootstrapped", False)


def _bootstrap_active_run_id() -> None:
    try:
        if st.session_state.get("bootstrapped") is True:
            return
        st.session_state["bootstrapped"] = True
        cur = str(st.session_state.get("active_run_id") or "").strip()
        if cur:
            return

        saved = _read_active_run_id()
        if saved:
            try:
                if _paths_for_run(saved).run_dir.exists():
                    st.session_state["active_run_id"] = str(saved)
                    return
            except Exception:
                pass

        running = _running_runs()
        if running:
            st.session_state["active_run_id"] = str(running[0])
            _write_active_run_id(str(running[0]))
    except Exception:
        pass


def _start_run(paths: RunPaths, *, cmd: List[str], cwd: Path, env: Dict[str, str]) -> subprocess.Popen:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.files_dir.mkdir(parents=True, exist_ok=True)

    out_f = open(paths.stdout_path, "ab", buffering=0)
    err_f = open(paths.stderr_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
        )
    finally:
        # Close parent handles; child keeps its own file descriptors.
        try:
            out_f.close()
        except Exception:
            pass
        try:
            err_f.close()
        except Exception:
            pass
    return proc


def _start_suite(paths: SuitePaths, *, cmd: List[str], cwd: Path, env: Dict[str, str]) -> subprocess.Popen:
    paths.suite_dir.mkdir(parents=True, exist_ok=True)

    out_f = open(paths.stdout_path, "ab", buffering=0)
    err_f = open(paths.stderr_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
        )
    finally:
        try:
            out_f.close()
        except Exception:
            pass
        try:
            err_f.close()
        except Exception:
            pass
    return proc


def _stop_run(paths: RunPaths) -> None:
    pid = _read_pid(paths.pid_path)
    if pid is None:
        return
    if not _pid_alive(pid):
        _clear_pid(paths.pid_path)
        return

    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < 3.0:
        if not _pid_alive(pid):
            _clear_pid(paths.pid_path)
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    _clear_pid(paths.pid_path)


def _stop_suite(paths: SuitePaths) -> None:
    pid = _read_pid(paths.pid_path)
    if pid is None:
        return
    if not _pid_alive(pid):
        _clear_pid(paths.pid_path)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < 3.0:
        if not _pid_alive(pid):
            _clear_pid(paths.pid_path)
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    _clear_pid(paths.pid_path)


def _stop_proc(proc: Any) -> None:
    try:
        if proc is None:
            return
        if getattr(proc, "poll")() is not None:
            return
        proc.terminate()
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if getattr(proc, "poll")() is not None:
                return
            time.sleep(0.1)
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        pass


st.set_page_config(page_title="LLM Benchmark Monitor", layout="wide")

_ensure_session_state()
_bootstrap_active_run_id()
_bootstrap_active_suite_id()

st.title("LLM Benchmark Monitor")

running_runs_global = _running_runs()
running_suites_global = _running_suites()
ui_locked_global = bool(running_runs_global or running_suites_global)

if ui_locked_global:
    try:
        if running_runs_global:
            rr0 = str(running_runs_global[0])
            if rr0 and str(st.session_state.get("active_run_id") or "").strip() != rr0:
                st.session_state["active_run_id"] = rr0
                _write_active_run_id(rr0)
        if running_suites_global:
            ss0 = str(running_suites_global[0])
            if ss0 and str(st.session_state.get("active_suite_id") or "").strip() != ss0:
                st.session_state["active_suite_id"] = ss0
                _write_active_suite_id(ss0)
    except Exception:
        pass

with st.sidebar:
    tab_settings, tab_run, tab_suite = st.tabs(["Settings", "Run", "Suite"])

    if ui_locked_global:
        rr0 = str(running_runs_global[0]) if running_runs_global else ""
        ss0 = str(running_suites_global[0]) if running_suites_global else ""
        st.warning({"locked": True, "running_run": rr0 or None, "running_suite": ss0 or None})

    with tab_settings:
        base_url = st.text_input("Runtime base URL", value="http://127.0.0.1:8000", disabled=ui_locked_global)
        st.caption(
            ("OPENAI_API_KEY=ok" if str(os.environ.get("OPENAI_API_KEY") or "").strip() else "OPENAI_API_KEY=missing")
            + " | "
            + ("ANTHROPIC_API_KEY=ok" if str(os.environ.get("ANTHROPIC_API_KEY") or "").strip() else "ANTHROPIC_API_KEY=missing")
            + " | "
            + ("XAI_API_KEY=ok" if str(os.environ.get("XAI_API_KEY") or "").strip() else "XAI_API_KEY=missing")
            + " | "
            + ("GEMINI_API_KEY=ok" if str(os.environ.get("GEMINI_API_KEY") or "").strip() else "GEMINI_API_KEY=missing")
        )
        challenge = st.selectbox(
            "Challenge",
            options=["cancer", "hereditary_disease", "aging"],
            index=0,
            key="challenge",
            disabled=ui_locked_global,
        )

        st.subheader("Limits")
        max_steps = int(st.number_input("Max steps", min_value=1, max_value=400, value=40, step=1, disabled=ui_locked_global))
        temperature = float(st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1, disabled=ui_locked_global))
        max_tokens = int(st.number_input("Max tokens", min_value=32, max_value=8000, value=8000, step=50, disabled=ui_locked_global))

        st.subheader("Timeouts")
        api_timeout = float(st.number_input("API timeout (s)", min_value=5.0, max_value=5000.0, value=5000.0, step=30.0, disabled=ui_locked_global))
        llm_timeout = float(st.number_input("LLM timeout (s)", min_value=5.0, max_value=5000.0, value=5000.0, step=5.0, disabled=ui_locked_global))

        st.subheader("Game")
        player_id = st.text_input("Player ID (optional)", value="", disabled=ui_locked_global)
        player_id_for_reset = player_id.strip() or "(default)"
        reset_first = False
        with st.expander("Danger zone", expanded=False):
            arm_reset = bool(
                st.checkbox(
                    f"I understand: this will wipe the current game state for player_id='{player_id_for_reset}'",
                    value=False,
                    key="arm_reset",
                    disabled=ui_locked_global,
                )
            )
            if arm_reset:
                reset_first = bool(
                    st.checkbox(
                        "Reset game state before starting a new run",
                        value=False,
                        key="reset_first",
                        disabled=ui_locked_global,
                    )
                )

        st.subheader("Refresh")
        auto_refresh = bool(st.checkbox("Auto refresh", value=True))
        refresh_s = float(st.number_input("Refresh interval (s)", min_value=0.2, max_value=10.0, value=1.0, step=0.2))
        manual_refresh_clicked = st.button("Refresh now", use_container_width=True)

    with tab_run:
        st.header("Run")
        if str(st.session_state.get("provider") or "") == "grok":
            st.session_state["provider"] = "xai"
        provider = st.selectbox(
            "Provider",
            options=["openai", "claude", "xai", "gemini", "human"],
            index=0,
            key="provider",
            disabled=ui_locked_global,
        )

        if provider == "anthropic":
            provider = "claude"
        if provider == "grok":
            provider = "xai"

        model = ""
        human_exec_provider = ""
        human_exec_model = ""
        human_poll = 0.5

        if provider == "human":
            st.subheader("Human mode")
            if str(st.session_state.get("human_exec_provider") or "") == "grok":
                st.session_state["human_exec_provider"] = "xai"
            human_exec_provider = st.selectbox(
                "Executor provider",
                options=["claude", "openai", "xai", "gemini"],
                index=0,
                key="human_exec_provider",
                help="The human provides directives; this LLM translates directives into Action JSON and executes tool calls.",
                disabled=ui_locked_global,
            )
            if human_exec_provider == "openai":
                openai_models = [
                    "gpt-5.2",
                    "gpt-5.2-medium",
                    "gpt-5.2-high",
                    "gpt-5.2-extra-high",
                ]
                default_model = str(st.session_state.get("human_exec_model") or "").strip()
                if default_model not in openai_models:
                    default_model = openai_models[0]
                chosen = st.selectbox(
                    "Executor model",
                    options=openai_models,
                    index=int(openai_models.index(default_model)),
                    key="human_openai_model_choice",
                    disabled=ui_locked_global,
                )
                human_exec_model = str(chosen)
                st.session_state["human_exec_model"] = str(human_exec_model)
            elif human_exec_provider == "gemini":
                gemini_models = [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-3-pro-preview",
                ]
                default_model = str(st.session_state.get("human_exec_model") or "").strip()
                if default_model not in gemini_models:
                    default_model = gemini_models[0]
                chosen = st.selectbox(
                    "Executor model",
                    options=gemini_models,
                    index=int(gemini_models.index(default_model)),
                    key="human_gemini_model_choice",
                    disabled=ui_locked_global,
                )
                human_exec_model = str(chosen)
                st.session_state["human_exec_model"] = str(human_exec_model)
            elif human_exec_provider in ("xai", "grok"):
                xai_models = [
                    "grok-4",
                    "grok-4-1-fast",
                    "grok-4-1-fast-reasoning",
                ]
                default_model = str(st.session_state.get("human_exec_model") or "").strip()
                if default_model not in xai_models:
                    default_model = xai_models[0]
                chosen = st.selectbox(
                    "Executor model",
                    options=xai_models,
                    index=int(xai_models.index(default_model)),
                    key="human_xai_model_choice",
                    disabled=ui_locked_global,
                )
                human_exec_model = str(chosen)
                st.session_state["human_exec_model"] = str(human_exec_model)
            else:
                claude_models = [
                    "claude-sonnet-4-5-20250929",
                    "claude-opus-4-5-20251101",
                ]
                default_model = str(st.session_state.get("human_exec_model") or "").strip()
                if default_model not in claude_models:
                    default_model = claude_models[0]
                chosen = st.selectbox(
                    "Executor model",
                    options=claude_models,
                    index=int(claude_models.index(default_model)),
                    key="human_claude_model_choice",
                    disabled=ui_locked_global,
                )
                human_exec_model = str(chosen)
                st.session_state["human_exec_model"] = str(human_exec_model)
            human_poll = float(st.number_input("Human poll interval (s)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, disabled=ui_locked_global))
            model = str(human_exec_model)
        elif provider == "openai":
            st.subheader("OpenAI")
            openai_models = [
                "gpt-5.2",
                "gpt-5.2-medium",
                "gpt-5.2-high",
                "gpt-5.2-extra-high",
            ]
            default_model = str(st.session_state.get("model") or "").strip()
            if default_model not in openai_models:
                default_model = openai_models[0]
            chosen = st.selectbox("Model", options=openai_models, index=int(openai_models.index(default_model)), key="openai_model_choice", disabled=ui_locked_global)
            model = str(chosen)
            st.session_state["model"] = str(model)
        elif provider == "xai":
            st.subheader("xAI (Grok)")
            xai_models = [
                "grok-4",
                "grok-4-1-fast",
                "grok-4-1-fast-reasoning",
            ]
            default_model = str(st.session_state.get("model") or "").strip()
            if default_model not in xai_models:
                default_model = xai_models[0]
            chosen = st.selectbox("Model", options=xai_models, index=int(xai_models.index(default_model)), key="xai_model_choice", disabled=ui_locked_global)
            model = str(chosen)
            st.session_state["model"] = str(model)
        elif provider == "gemini":
            st.subheader("Google Gemini")
            gemini_models = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-3-pro-preview",
            ]
            default_model = str(st.session_state.get("model") or "").strip()
            if default_model not in gemini_models:
                default_model = gemini_models[0]
            chosen = st.selectbox(
                "Model",
                options=gemini_models,
                index=int(gemini_models.index(default_model)),
                key="gemini_model_choice",
                disabled=ui_locked_global,
            )
            model = str(chosen)
            st.session_state["model"] = str(model)
        elif provider in ("anthropic", "claude"):
            st.subheader("Anthropic")
            claude_models = [
                "claude-sonnet-4-5-20250929",
                "claude-opus-4-5-20251101",
            ]
            default_model = str(st.session_state.get("model") or "").strip()
            if default_model not in claude_models:
                default_model = claude_models[0]
            chosen = st.selectbox("Model", options=claude_models, index=int(claude_models.index(default_model)), key="claude_model_choice", disabled=ui_locked_global)
            model = str(chosen)
            st.session_state["model"] = str(model)
        else:
            model = st.text_input("Model", value="", key="model", disabled=ui_locked_global)

        st.subheader("Saved runs")
        runs = _list_runs()
        running_set = set(_running_runs())
        active_run_id = str(st.session_state.get("active_run_id") or "").strip()
        any_running = bool(running_set)
        selected_running = bool(active_run_id and active_run_id in running_set)
        options = [""] + runs
        idx = 0
        try:
            if active_run_id and active_run_id in options:
                idx = int(options.index(active_run_id))
        except Exception:
            idx = 0
        sel_run = st.selectbox(
            "Select run",
            options=options,
            index=int(idx),
            format_func=lambda x: (str(x) + (" (running)" if str(x) in running_set else "")) if str(x) else "",
            disabled=ui_locked_global,
        )
        if sel_run:
            st.session_state["active_run_id"] = str(sel_run)
            _write_active_run_id(str(sel_run))

        resumable = False
        try:
            if active_run_id:
                resumable = bool(_paths_for_run(active_run_id).state_path.exists())
        except Exception:
            resumable = False

        col_a, col_b, col_c = st.columns(3)
        start_clicked = col_a.button("Start new", use_container_width=True, disabled=bool(ui_locked_global))
        stop_clicked = col_b.button("Stop", use_container_width=True, disabled=bool((not any_running) or bool(running_suites_global)))
        resume_clicked = col_c.button("Resume", use_container_width=True, disabled=bool(ui_locked_global or selected_running or (not resumable)))

    with tab_suite:
        st.header("Suite")
        suites = _list_suites()
        suite_running_set = set(_running_suites())
        active_suite_id = str(st.session_state.get("active_suite_id") or "").strip()
        any_suite_running = bool(suite_running_set)
        selected_suite_running = bool(active_suite_id and active_suite_id in suite_running_set)

        suite_options = [""] + suites
        suite_idx = 0
        try:
            if active_suite_id and active_suite_id in suite_options:
                suite_idx = int(suite_options.index(active_suite_id))
        except Exception:
            suite_idx = 0

        sel_suite = st.selectbox(
            "Select suite",
            options=suite_options,
            index=int(suite_idx),
            format_func=lambda x: (str(x) + (" (running)" if str(x) in suite_running_set else "")) if str(x) else "",
            disabled=ui_locked_global,
        )
        if sel_suite:
            st.session_state["active_suite_id"] = str(sel_suite)
            _write_active_suite_id(str(sel_suite))
            active_suite_id = str(sel_suite)
            selected_suite_running = bool(active_suite_id and active_suite_id in suite_running_set)

        any_running_now = bool(set(_running_runs()))
        running_runs_now = _running_runs()
        running_run_id = str(running_runs_now[0]) if running_runs_now else ""
        if running_run_id:
            st.caption(f"Running run: {running_run_id}")
        sp_sel = _paths_for_suite(active_suite_id) if active_suite_id else None
        suite_resumable = False
        if sp_sel is not None:
            suite_resumable = bool(
                sp_sel.specs_path.exists()
                and sp_sel.specs_path.is_file()
                and sp_sel.summary_csv_path.exists()
                and sp_sel.summary_csv_path.is_file()
            )

        st.subheader("Start new suite")
        pairs = _llm_provider_model_options()
        avail_specs = [str(p) + ":" + str(m) for p, m in pairs]
        prev_sel = st.session_state.get("suite_models")
        if not isinstance(prev_sel, list):
            prev_sel = []
        sel_models = st.multiselect(
            "Models",
            options=avail_specs,
            default=[x for x in prev_sel if x in avail_specs],
            disabled=ui_locked_global,
        )
        st.session_state["suite_models"] = list(sel_models)
        with st.expander("Advanced", expanded=False):
            suite_extra_specs = st.text_area(
                "Extra spec lines",
                value=str(st.session_state.get("suite_extra_specs") or "").strip(),
                height=100,
                disabled=ui_locked_global,
            )
            st.session_state["suite_extra_specs"] = str(suite_extra_specs or "")

        suite_reps = int(st.number_input("Replicates", min_value=1, max_value=50, value=1, step=1, disabled=ui_locked_global))
        suite_cooldown_s = float(st.number_input("Cooldown between runs (s)", min_value=0.0, max_value=60.0, value=0.0, step=0.5, disabled=ui_locked_global))
        suite_stop_on_error = bool(st.checkbox("Stop on error", value=False, disabled=ui_locked_global))

        start_suite_clicked = st.button(
            "Start new suite",
            use_container_width=True,
            disabled=bool(ui_locked_global or any_running_now or any_suite_running),
            help="Suites run sequentially and refuse to start if any benchmark run is already running.",
        )

        st.subheader("Resume selected suite")
        resume_suite_clicked = st.button(
            "Resume selected suite",
            use_container_width=True,
            disabled=bool(ui_locked_global or any_running_now or any_suite_running or selected_suite_running or (not suite_resumable)),
        )

        stop_suite_clicked = st.button(
            "Stop selected suite",
            use_container_width=True,
            disabled=bool(not any_suite_running),
        )


active_run_id = str(st.session_state.get("active_run_id") or "").strip()
proc = st.session_state.get("proc")

active_suite_id = str(st.session_state.get("active_suite_id") or "").strip()
suite_proc = st.session_state.get("suite_proc")

if manual_refresh_clicked:
    st.rerun()

paths = _paths_for_run(active_run_id) if active_run_id else None

if stop_clicked:
    if paths is not None:
        _stop_run(paths)
    _stop_proc(proc)
    st.session_state["proc"] = None

if stop_suite_clicked:
    if active_suite_id:
        sp = _paths_for_suite(active_suite_id)
        _stop_suite(sp)
    running = _running_runs()
    if running:
        try:
            rp = _paths_for_run(str(running[0]))
            _stop_run(rp)
        except Exception:
            pass
    _stop_proc(suite_proc)
    st.session_state["suite_proc"] = None

if resume_clicked:
    if not active_run_id:
        st.sidebar.error("Select a run to resume.")
    elif paths is None:
        st.sidebar.error("No run selected.")
    elif not paths.state_path.exists():
        st.sidebar.error("No state.json found for this run (nothing to resume).")
    else:
        st_state = _load_json(paths.state_path)
        if not isinstance(st_state, dict):
            st.sidebar.error("Failed to read state.json for this run.")
            st.stop()
        st_base_url = str(st_state.get("base_url") or "").strip() or str(base_url)
        st_provider = str(st_state.get("provider") or "").strip() or str(provider)
        if st_provider == "anthropic":
            st_provider = "claude"
        st_model = str(st_state.get("model") or "").strip() or str(model)
        st_challenge = str(st_state.get("challenge") or "").strip() or str(challenge)

        exec_provider = ""
        exec_model = ""
        exec_poll = 0.5
        if st_provider == "human":
            exec_provider = str(st_state.get("executor_provider") or "").strip() or str(human_exec_provider)
            if exec_provider == "anthropic":
                exec_provider = "claude"
            if not exec_provider:
                exec_provider = "claude"
            exec_model = str(st_state.get("executor_model") or "").strip() or str(human_exec_model or st_model)
            try:
                exec_poll = float(st_state.get("human_poll") or human_poll)
            except Exception:
                exec_poll = float(human_poll)

            if exec_provider == "openai":
                key_present = bool(str(os.environ.get("OPENAI_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("OpenAI API key is required for Human executor (set OPENAI_API_KEY).")
                    st.stop()
            if exec_provider in ("anthropic", "claude"):
                key_present = bool(str(os.environ.get("ANTHROPIC_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("Anthropic API key is required for Human executor (set ANTHROPIC_API_KEY).")
                    st.stop()
        else:
            if st_provider == "openai":
                key_present = bool(str(os.environ.get("OPENAI_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("OpenAI API key is required to resume this run (set OPENAI_API_KEY in your environment).")
                    st.stop()
            if st_provider in ("anthropic", "claude"):
                key_present = bool(str(os.environ.get("ANTHROPIC_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("Anthropic API key is required to resume this run (set ANTHROPIC_API_KEY in your environment).")
                    st.stop()

        _stop_run(paths)
        _stop_proc(proc)
        env = dict(os.environ)
        cmd = [
            sys.executable,
            "trials/run_llm_benchmark.py",
            "--run-id",
            str(active_run_id),
            "--base-url",
            str(st_base_url),
            "--provider",
            str(st_provider),
            "--model",
            str(st_model),
        ]
        if st_provider == "human":
            cmd.extend(["--executor-provider", str(exec_provider), "--executor-model", str(exec_model), "--human-poll", str(float(exec_poll))])
        cmd.extend([
            "--challenge",
            str(st_challenge),
            "--max-steps",
            str(int(max_steps)),
            "--temperature",
            str(float(temperature)),
            "--max-tokens",
            str(int(max_tokens)),
            "--api-timeout",
            str(float(api_timeout)),
            "--llm-timeout",
            str(float(llm_timeout)),
            "--events-out",
            str(paths.events_path),
            "--out",
            str(paths.report_path),
            "--files-dir",
            str(paths.files_dir),
            "--state-out",
            str(paths.state_path),
            "--resume-state",
            str(paths.state_path),
        ])
        st.session_state["proc"] = _start_run(paths, cmd=cmd, cwd=_repo_root(), env=env)
        proc = st.session_state.get("proc")
        try:
            _write_pid(paths.pid_path, int(getattr(proc, "pid")))
        except Exception:
            pass
        _write_active_run_id(str(active_run_id))

if start_clicked:
    if not model.strip():
        st.sidebar.error("Model is required.")
    else:
        if provider == "human":
            if human_exec_provider == "openai":
                key_present = bool(str(os.environ.get("OPENAI_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("OpenAI API key is required for Human executor (set OPENAI_API_KEY).")
                    st.stop()
            if human_exec_provider in ("anthropic", "claude"):
                key_present = bool(str(os.environ.get("ANTHROPIC_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("Anthropic API key is required for Human executor (set ANTHROPIC_API_KEY).")
                    st.stop()
        else:
            if provider == "openai":
                key_present = bool(str(os.environ.get("OPENAI_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("OpenAI API key is required (set OPENAI_API_KEY in your environment).")
                    st.stop()
            if provider in ("anthropic", "claude"):
                key_present = bool(str(os.environ.get("ANTHROPIC_API_KEY") or "").strip())
                if not key_present:
                    st.sidebar.error("Anthropic API key is required (set ANTHROPIC_API_KEY in your environment).")
                    st.stop()

        if paths is not None:
            _stop_run(paths)
        _stop_proc(proc)
        run_id = _new_run_id()
        st.session_state["active_run_id"] = run_id
        paths = _paths_for_run(run_id)

        cmd = [
            sys.executable,
            "trials/run_llm_benchmark.py",
            "--run-id",
            str(run_id),
            "--base-url",
            str(base_url),
            "--provider",
            str(provider),
            "--model",
            str(model),
        ]
        if provider == "human":
            cmd.extend([
                "--executor-provider",
                str(human_exec_provider),
                "--executor-model",
                str(human_exec_model or model),
                "--human-poll",
                str(float(human_poll)),
            ])
        cmd.extend([
            "--challenge",
            str(challenge),
            "--max-steps",
            str(int(max_steps)),
            "--temperature",
            str(float(temperature)),
            "--max-tokens",
            str(int(max_tokens)),
            "--api-timeout",
            str(float(api_timeout)),
            "--llm-timeout",
            str(float(llm_timeout)),
            "--events-out",
            str(paths.events_path),
            "--out",
            str(paths.report_path),
            "--files-dir",
            str(paths.files_dir),
            "--state-out",
            str(paths.state_path),
        ])
        if reset_first:
            cmd.append("--reset-first")
        if player_id.strip():
            cmd.extend(["--player-id", player_id.strip()])

        env = dict(os.environ)
        st.session_state["proc"] = _start_run(paths, cmd=cmd, cwd=_repo_root(), env=env)
        proc = st.session_state.get("proc")
        try:
            _write_pid(paths.pid_path, int(getattr(proc, "pid")))
        except Exception:
            pass
        _write_active_run_id(str(run_id))

if start_suite_clicked:
    specs_in: List[str] = []
    for s in list(st.session_state.get("suite_models") or []):
        if not isinstance(s, str):
            continue
        s2 = str(s).strip()
        if not s2:
            continue
        specs_in.append(s2)
    for ln in str(st.session_state.get("suite_extra_specs") or "").splitlines():
        s3 = str(ln or "").strip()
        if not s3:
            continue
        if s3.startswith("#"):
            continue
        specs_in.append(s3)
    if not specs_in:
        st.sidebar.error("Select at least one model (or provide an advanced spec line).")
    else:
        suite_id = _new_suite_id()
        st.session_state["active_suite_id"] = str(suite_id)
        _write_active_suite_id(str(suite_id))
        sp = _paths_for_suite(str(suite_id))
        try:
            sp.suite_dir.mkdir(parents=True, exist_ok=True)
            sp.specs_path.write_text("\n".join(specs_in) + "\n", encoding="utf-8")
        except Exception as e:
            st.sidebar.error(f"Failed to write suite specs: {e}")
            st.stop()

        cmd = [
            sys.executable,
            "trials/run_llm_suite.py",
            "--suite-id",
            str(suite_id),
            "--base-url",
            str(base_url),
            "--challenge",
            str(challenge),
            "--max-steps",
            str(int(max_steps)),
            "--temperature",
            str(float(temperature)),
            "--max-tokens",
            str(int(max_tokens)),
            "--api-timeout",
            str(float(api_timeout)),
            "--llm-timeout",
            str(float(llm_timeout)),
            "--replicates",
            str(int(suite_reps)),
            "--cooldown-s",
            str(float(suite_cooldown_s)),
            "--spec-file",
            str(sp.specs_path),
        ]
        if reset_first:
            cmd.append("--reset-first")
        if suite_stop_on_error:
            cmd.append("--stop-on-error")

        env = dict(os.environ)
        st.session_state["suite_proc"] = _start_suite(sp, cmd=cmd, cwd=_repo_root(), env=env)
        suite_proc = st.session_state.get("suite_proc")

if resume_suite_clicked:
    active_suite_id = str(st.session_state.get("active_suite_id") or "").strip()
    if not active_suite_id:
        st.sidebar.error("Select a suite to resume.")
    else:
        sp = _paths_for_suite(active_suite_id)
        if not (sp.specs_path.exists() and sp.specs_path.is_file()):
            st.sidebar.error("No specs file found for this suite (cannot resume).")
        else:
            cmd = [
                sys.executable,
                "trials/run_llm_suite.py",
                "--suite-id",
                str(active_suite_id),
                "--base-url",
                str(base_url),
                "--challenge",
                str(challenge),
                "--max-steps",
                str(int(max_steps)),
                "--temperature",
                str(float(temperature)),
                "--max-tokens",
                str(int(max_tokens)),
                "--api-timeout",
                str(float(api_timeout)),
                "--llm-timeout",
                str(float(llm_timeout)),
                "--replicates",
                str(int(suite_reps)),
                "--cooldown-s",
                str(float(suite_cooldown_s)),
                "--spec-file",
                str(sp.specs_path),
                "--resume",
            ]
            if reset_first:
                cmd.append("--reset-first")
            if suite_stop_on_error:
                cmd.append("--stop-on-error")
            env = dict(os.environ)
            st.session_state["suite_proc"] = _start_suite(sp, cmd=cmd, cwd=_repo_root(), env=env)
            suite_proc = st.session_state.get("suite_proc")


if (not active_run_id) and (not active_suite_id):
    st.info("Start a new run or select a saved run. You can also run preflight checks below.")
    _render_preflight_ui(challenge=str(challenge), base_url=base_url, api_timeout_s=float(api_timeout), player_id=str(player_id))
    if auto_refresh:
        time.sleep(refresh_s)
        st.rerun()
    raise SystemExit(0)

if (not active_run_id) and active_suite_id:
    sp = _paths_for_suite(active_suite_id)
    suite_pid = _read_pid(sp.pid_path)
    suite_alive = bool(suite_pid is not None and _pid_alive(int(suite_pid)))
    cols0 = st.columns(4)
    cols0[0].metric("Suite", str(active_suite_id))
    cols0[1].metric("Status", "Running" if suite_alive else "Idle/Done")
    cols0[2].metric("Suite dir", str(sp.suite_dir))
    running = _running_runs()
    cols0[3].metric("Running run", str(running[0]) if running else "—")

    if running:
        if st.button("Attach to running run", use_container_width=True, disabled=ui_locked_global):
            st.session_state["active_run_id"] = str(running[0])
            _write_active_run_id(str(running[0]))
            st.rerun()

    st.subheader("Suite outputs")
    if sp.aggregate_csv_path.exists():
        try:
            rows0 = _csv_to_rows(sp.aggregate_csv_path.read_text(encoding="utf-8", errors="replace"), max_rows=50)
            if rows0:
                st.dataframe(rows0, use_container_width=True)
        except Exception:
            pass
    if sp.summary_csv_path.exists():
        try:
            rows1 = _csv_to_rows(sp.summary_csv_path.read_text(encoding="utf-8", errors="replace"), max_rows=50)
            if rows1:
                st.dataframe(rows1, use_container_width=True)
        except Exception:
            pass

    st.subheader("Suite logs")
    st.text_area("suite stdout", value=_tail_text(sp.stdout_path), height=200)
    st.text_area("suite stderr", value=_tail_text(sp.stderr_path), height=200)

    if auto_refresh:
        time.sleep(refresh_s)
        st.rerun()
    raise SystemExit(0)

paths = _paths_for_run(active_run_id)

pid = _read_pid(paths.pid_path)
proc_alive = bool(pid is not None and _pid_alive(int(pid)))
if (not proc_alive) and proc is not None:
    try:
        if getattr(proc, "poll")() is not None:
            st.session_state["proc"] = None
            proc = None
    except Exception:
        pass


events = _read_events(paths.events_path)
report = _load_json(paths.report_path)

end_metrics = _latest_end_metrics(events)
money_cents, money_usd = _latest_game_money(events)

best_delta, best_win = _best_claim_cure(events, challenge=str(challenge))
best_recovery_pct = _best_claim_cure_recovery(events, challenge=str(challenge))
best_score, best_extra_days, best_score_seq = _best_claim_cure_score(events, challenge=str(challenge))

header_cols = st.columns(6)
header_cols[0].metric("Run", active_run_id)
header_cols[1].metric("Status", "Running" if proc_alive else "Idle/Done")
header_cols[2].metric("Best score", "—" if best_score is None else f"{best_score:.2f}")
header_cols[3].metric("Best extra days", "—" if best_extra_days is None else f"{best_extra_days:.2f}")
header_cols[4].metric("Best Lifespan Recovery", "—" if best_recovery_pct is None else f"{best_recovery_pct:.2f}%")
header_cols[5].metric("Best delta ticks", "—" if best_delta is None else f"{best_delta:.3f}")

if money_usd is not None:
    st.caption(f"Money spent: ${money_usd:.2f} ({money_cents} cents)")
elif end_metrics and end_metrics.get("money_spent_usd") is not None:
    st.caption(f"Money spent: ${float(end_metrics.get('money_spent_usd')):.2f}")

if str(challenge or "").strip().lower() != "aging":
    if best_win is True:
        st.success("Best observed claim_cure: WIN")
    elif best_win is False and best_delta is not None:
        st.warning("Best observed claim_cure: not yet win")

tabs = st.tabs(["Human", "Story", "Live", "Errors", "Preflight", "CSV files", "Prompt", "Report", "Logs", "Suite"])

if events:
    last = events[-1] if isinstance(events[-1], dict) else None
    if isinstance(last, dict):
        last_seq = last.get("seq")
        last_type = last.get("type")
        last_ts = last.get("ts")
        if isinstance(last_ts, (int, float)):
            age_s = max(0.0, float(time.time()) - float(last_ts))
            st.caption(f"Last event: #{last_seq} {last_type} ({age_s:.1f}s ago)")

with tabs[0]:
    st.subheader("Human-in-the-loop")
    human_dir = paths.run_dir / "human_mode"
    pending = _load_json(human_dir / "pending.json")
    if not isinstance(pending, dict) or pending.get("waiting") is not True:
        st.info("Runner is not currently waiting for human input.")
    else:
        step = int(pending.get("step") or 0)
        expecting = str(pending.get("expecting") or f"input_step_{step:06d}.json")
        st.write({"step": step, "expecting": expecting, "player_id": pending.get("player_id")})
        if isinstance(pending.get("error"), str) and str(pending.get("error") or "").strip():
            st.warning(str(pending.get("error")))
        mode = st.radio("Submit type", options=["Directive text", "Full action_json"], index=0, horizontal=True)
        with st.form(key=f"human_submit_{active_run_id}_{step}"):
            txt = st.text_area("Your directive", value="", height=140)
            action_json_raw = ""
            if mode == "Full action_json":
                action_json_raw = st.text_area("Action JSON", value="{}", height=220)
            submitted = st.form_submit_button("Submit to runner", use_container_width=True)
        col_a, col_b = st.columns(2)
        stop_clicked2 = col_a.button("Stop run (write stop.json)", use_container_width=True)
        if stop_clicked2:
            try:
                _write_json_atomic(human_dir / "stop.json", {"ok": True, "stop": True, "ts": float(time.time())})
                st.success("stop.json written")
            except Exception as e:
                st.error(str(e))

        if submitted:
            payload: Dict[str, Any] = {"text": str(txt or ""), "ts": float(time.time())}
            if mode == "Full action_json":
                try:
                    obj = json.loads(str(action_json_raw or "").strip() or "{}")
                    if not isinstance(obj, dict):
                        raise ValueError("action_json must be a JSON object")
                    payload["action_json"] = obj
                except Exception as e:
                    st.error(f"Invalid Action JSON: {str(e)}")
                    st.stop()
            try:
                _write_json_atomic(human_dir / expecting, payload)
                st.success(f"Wrote {expecting}")
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Concise run story")

    max_events_for_story = 220
    min_refresh_s = 45.0
    show_digest = False
    allow_auto = False
    use_lab_notebook = True
    prefer_latest_prompt = True
    with st.expander("Story settings", expanded=False):
        max_events_for_story = int(
            st.number_input("Events window", min_value=50, max_value=800, value=220, step=10)
        )
        min_refresh_s = float(
            st.number_input("Min story refresh (s)", min_value=5.0, max_value=600.0, value=45.0, step=5.0)
        )
        allow_auto = bool(st.checkbox("Auto-update story (costs tokens)", value=False))
        use_lab_notebook = bool(st.checkbox("Use LAB_NOTEBOOK (recommended)", value=True))
        prefer_latest_prompt = bool(st.checkbox("Prefer latest outbound LLM prompt (llm_payloads)", value=True))
        show_digest = bool(st.checkbox("Show digest (debug)", value=False))

    notebook_txt = _read_lab_notebook(paths) if use_lab_notebook else ""
    nb_hash = hashlib.sha256(notebook_txt.encode("utf-8", errors="replace")).hexdigest() if notebook_txt else ""
    latest_prompt_txt, latest_prompt_hash = ("", "")
    if prefer_latest_prompt:
        latest_prompt_txt, latest_prompt_hash = _latest_llm_prompt_text(paths)
    last_ev = events[-1] if isinstance(events, list) and events and isinstance(events[-1], dict) else {}
    fp = _json_compact(
        {
            "run": active_run_id,
            "last_seq": last_ev.get("seq"),
            "last_type": last_ev.get("type"),
            "last_ts": last_ev.get("ts"),
            "use_lab_notebook": bool(use_lab_notebook),
            "nb_hash": nb_hash,
            "prefer_latest_prompt": bool(prefer_latest_prompt),
            "latest_prompt_hash": str(latest_prompt_hash or ""),
        }
    )
    digest_hash = hashlib.sha256(fp.encode("utf-8", errors="replace")).hexdigest()

    cache = st.session_state.get("story_cache")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state["story_cache"] = cache
    cache_ent = cache.get(active_run_id) if isinstance(cache.get(active_run_id), dict) else {}
    last_hash = cache_ent.get("digest_hash")
    last_story = cache_ent.get("story")
    last_ts = cache_ent.get("ts")

    stale = bool(last_hash != digest_hash)
    if isinstance(last_ts, (int, float)):
        st.caption(f"Story cached: {max(0.0, time.time() - float(last_ts)):.1f}s ago | stale={stale}")
    else:
        st.caption(f"Story cached: none | stale={stale}")

    job = st.session_state.get("story_job")
    job_meta = st.session_state.get("story_job_meta")
    job_running = isinstance(job, concurrent.futures.Future) and (not job.done())

    if isinstance(job, concurrent.futures.Future) and job.done():
        try:
            story_out = job.result(timeout=0.0)
        except Exception as e:
            story_out = None
            st.warning(f"Story generation failed. ({str(e)})")
        try:
            if isinstance(job_meta, dict) and str(job_meta.get("run_id") or "") == str(active_run_id):
                dh = str(job_meta.get("digest_hash") or "")
                if dh:
                    cache[active_run_id] = {"digest_hash": dh, "story": str(story_out or "").strip(), "ts": float(time.time())}
                    st.session_state["story_cache"] = cache
        except Exception:
            pass
        st.session_state["story_job"] = None
        st.session_state["story_job_meta"] = {}
        job = None
        job_meta = {}
        job_running = False

    refresh_now = st.button("Update story now", use_container_width=True, disabled=job_running)
    should_update = bool(refresh_now)
    if allow_auto and stale:
        if not isinstance(last_ts, (int, float)):
            should_update = True
        elif (time.time() - float(last_ts)) >= float(min_refresh_s):
            should_update = True

    story = str(last_story or "").strip()
    if should_update:
        key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
        if key:
            if not job_running:
                try:
                    if prefer_latest_prompt and latest_prompt_txt.strip():
                        full_txt = "LATEST_LLM_PROMPT:\n" + latest_prompt_txt
                    elif use_lab_notebook and notebook_txt.strip():
                        full_txt = "LAB_NOTEBOOK:\n" + notebook_txt
                    else:
                        digest = _event_digest(events, max_events=int(max_events_for_story))
                        full_txt = "EVENT_DIGEST:\n" + digest
                    fut = _story_executor().submit(
                        _openai_story_full,
                        api_key=key,
                        full_text=full_txt,
                        run_id=str(active_run_id),
                        proc_alive=bool(proc_alive),
                    )
                    st.session_state["story_job"] = fut
                    st.session_state["story_job_meta"] = {
                        "run_id": str(active_run_id),
                        "digest_hash": str(digest_hash),
                        "ts": float(time.time()),
                    }
                except Exception as e:
                    story = _local_story(events)
                    st.warning(f"Story generation failed; showing fallback. ({str(e)})")
        else:
            story = _local_story(events)
            st.info("No OPENAI_API_KEY set; showing fallback summary.")

        if not (isinstance(st.session_state.get("story_job"), concurrent.futures.Future)):
            cache[active_run_id] = {"digest_hash": digest_hash, "story": story, "ts": float(time.time())}
            st.session_state["story_cache"] = cache

    job2 = st.session_state.get("story_job")
    job_meta2 = st.session_state.get("story_job_meta")
    if isinstance(job2, concurrent.futures.Future) and (not job2.done()):
        started = job_meta2.get("ts") if isinstance(job_meta2, dict) else None
        if isinstance(started, (int, float)):
            st.caption(f"Story generation in progress... ({max(0.0, time.time() - float(started)):.1f}s)")
        else:
            st.caption("Story generation in progress...")

    if story.strip():
        st.markdown(story)
    else:
        st.info("No story yet. Click 'Update story now'.")

    if show_digest:
        if prefer_latest_prompt and latest_prompt_txt.strip():
            st.text_area("LATEST_LLM_PROMPT (debug)", value=latest_prompt_txt, height=240)
        elif use_lab_notebook:
            st.text_area("LAB_NOTEBOOK (debug)", value=notebook_txt, height=240)
        else:
            dig = _event_digest(events, max_events=int(max_events_for_story))
            st.text_area("Digest", value=dig, height=240)

with tabs[2]:
    st.subheader("Live events")
    preset = st.selectbox(
        "View",
        options=["Key events", "LLM only", "API only", "Errors only", "Custom"],
        index=0,
    )
    auto_expand_latest_llm = bool(st.checkbox("Auto-expand latest LLM step", value=True))
    if preset == "LLM only":
        show_types = ["llm", "llm_error"]
    elif preset == "API only":
        show_types = ["api", "tool_result"]
    elif preset == "Errors only":
        show_types = ["llm_error", "player_id_mismatch", "final_rejected"]
    elif preset == "Key events":
        show_types = ["llm", "api", "tool_result", "final", "end", "llm_error", "player_id_mismatch", "final_rejected"]
    else:
        show_types = st.multiselect(
            "Event types",
            options=["start", "resume", "llm", "api", "tool_result", "final", "final_rejected", "end", "llm_error", "player_id_mismatch"],
            default=["llm", "api", "tool_result", "final", "end", "llm_error", "final_rejected", "player_id_mismatch"],
        )
    max_show = int(st.number_input("Max events shown", min_value=50, max_value=2000, value=300, step=50))

    with st.expander("Performance", expanded=False):
        lazy_api = bool(st.checkbox("Lazy-load API/tool JSON (recommended)", value=True))
        show_full_api_default = bool(st.checkbox("Show full API response_json by default", value=False, disabled=lazy_api))
        show_full_tool_default = bool(st.checkbox("Show full TOOL_RESULT by default", value=False, disabled=lazy_api))

    shown = [ev for ev in events if str(ev.get("type")) in set(show_types)]
    shown = shown[-max_show:]

    expanded_llm = False

    for ev in reversed(shown):
        t = str(ev.get("type") or "")
        seq = ev.get("seq")
        title = f"#{seq} {t}"
        if t == "api":
            title = f"#{seq} API {ev.get('method')} {ev.get('path')} ({ev.get('http_status')})"
        if t == "llm":
            title = f"#{seq} LLM step={ev.get('step')}"
        expand = False
        if auto_expand_latest_llm and (not expanded_llm) and t == "llm":
            expand = True
            expanded_llm = True
        with st.expander(title, expanded=expand):
            if t == "llm":
                lrs, nsr = _llm_reason_fields(ev)
                act_obj = _parse_first_json(str(ev.get("text") or ""))
                if isinstance(act_obj, dict):
                    st.write(
                        {
                            "action": act_obj.get("action"),
                            "method": act_obj.get("method"),
                            "path": act_obj.get("path"),
                        }
                    )
                if isinstance(lrs, str) and lrs.strip():
                    st.markdown("Last result summary")
                    st.write(lrs)
                if isinstance(nsr, str) and nsr.strip():
                    st.markdown("Next step rationale")
                    st.write(nsr)
                st.code(str(ev.get("text") or ""), language="")
            elif t in ("api", "tool_result"):
                if t == "api":
                    rj = ev.get("response_json") if isinstance(ev.get("response_json"), dict) else None
                    st.write({"seconds": ev.get("seconds"), "status": ev.get("http_status")})
                    if rj is not None:
                        if lazy_api:
                            st.markdown("Response summary")
                            st.json(_api_response_summary(rj))
                            if st.button("Load full response_json", key=f"api_full_{seq}"):
                                st.json(rj)
                        else:
                            if show_full_api_default:
                                st.json(rj)
                            else:
                                st.markdown("Response summary")
                                st.json(_api_response_summary(rj))
                    items = ev.get("files")
                    if not isinstance(items, list):
                        items = ev.get("artifacts")
                    if isinstance(items, list) and items:
                        st.markdown("Files")
                        for a in items:
                            if not isinstance(a, dict):
                                continue
                            p = a.get("path")
                            if isinstance(p, str) and p:
                                st.code(p)
                            prev = str(a.get("preview") or "")
                            if prev:
                                st.code(prev, language="csv")
                else:
                    payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else None
                    if payload is None:
                        st.json(ev)
                    else:
                        st.write(
                            {
                                "tool": ev.get("tool"),
                                "path": ev.get("path"),
                                "seconds": payload.get("seconds"),
                                "status": payload.get("http_status"),
                            }
                        )
                        rj = payload.get("response_json") if isinstance(payload.get("response_json"), dict) else None
                        if rj is not None:
                            st.markdown("LLM-facing TOOL_RESULT (compact)")
                            if lazy_api:
                                st.json(_api_response_summary(rj))
                                if st.button("Load full TOOL_RESULT", key=f"tool_full_{seq}"):
                                    st.json(rj)
                            else:
                                if show_full_tool_default:
                                    st.json(rj)
                                else:
                                    st.json(_api_response_summary(rj))
                        rj2 = (
                            payload.get("api_response_json_summary")
                            if isinstance(payload.get("api_response_json_summary"), dict)
                            else None
                        )
                        if rj2 is not None:
                            st.markdown("Raw API response summary (debug)")
                            st.json(rj2)
                        rt = payload.get("response_text")
                        if isinstance(rt, str) and rt.strip():
                            st.code(rt, language="")
            else:
                st.json(ev)

with tabs[3]:
    st.subheader("Issues / Errors")
    st.caption("Heuristic detector: flags anything that looks off even if the run continues (errors, retries, truncation signals, malformed LLM outputs, etc.).")
    issues = _detect_issues(events, paths)

    if not issues:
        st.info("No issues detected in the current events/log tail.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        sev_vals = sorted({str(x.get("severity") or "") for x in issues if str(x.get("severity") or "").strip()})
        kind_vals = sorted({str(x.get("kind") or "") for x in issues if str(x.get("kind") or "").strip()})
        src_vals = sorted({str(x.get("source") or "") for x in issues if str(x.get("source") or "").strip()})

        sev_pick = c1.multiselect("Severity", options=sev_vals, default=sev_vals)
        kind_pick = c2.multiselect("Kind", options=kind_vals, default=kind_vals)
        src_pick = c3.multiselect("Source", options=src_vals, default=src_vals)
        query = c4.text_input("Search", value="", help="Substring match against summary/details.")

        def _keep(x: Dict[str, Any]) -> bool:
            sev0 = str(x.get("severity") or "")
            kind0 = str(x.get("kind") or "")
            src0 = str(x.get("source") or "")
            if sev_pick and sev0 not in set(sev_pick):
                return False
            if kind_pick and kind0 not in set(kind_pick):
                return False
            if src_pick and src0 not in set(src_pick):
                return False
            q = str(query or "").strip().lower()
            if q:
                hay = (str(x.get("summary") or "") + "\n" + str(x.get("details") or "")).lower()
                if q not in hay:
                    return False
            return True

        issues_f = [x for x in issues if isinstance(x, dict) and _keep(x)]
        st.caption(f"Issues shown: {len(issues_f)} / {len(issues)}")

        st.download_button(
            "Download issues.json",
            data=json.dumps(issues_f, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{active_run_id}_issues.json",
            mime="application/json",
            use_container_width=True,
        )

        for i, it in enumerate(reversed(issues_f[-200:])):
            sev0 = str(it.get("severity") or "")
            kind0 = str(it.get("kind") or "")
            src0 = str(it.get("source") or "")
            seq0 = it.get("seq")
            hdr = f"{sev0.upper()} | {kind0} | {src0}"
            if seq0 is not None:
                hdr += f" | seq={seq0}"
            with st.expander(hdr, expanded=False):
                st.write(str(it.get("summary") or ""))
                st.json(it.get("details"))

with tabs[4]:
    _render_preflight_ui(challenge=str(challenge), base_url=str(base_url), api_timeout_s=float(api_timeout), player_id=str(player_id))

with tabs[5]:
    st.subheader("CSV files")
    arts = _extract_files(events)
    if not arts:
        st.info("No CSV files detected yet.")
    else:
        for a in reversed(arts[-200:]):
            p = a.get("path")
            if not isinstance(p, str) or not p:
                continue
            ev_seq = a.get("event_seq")
            path0 = a.get("path")
            label = f"#{ev_seq} {path0}"
            with st.expander(label, expanded=False):
                st.write({"from_api": a.get("path"), "bytes": a.get("bytes"), "file": p})
                prev = str(a.get("preview") or "")
                if prev:
                    st.code(prev, language="csv")
                    rows = _csv_to_rows(prev)
                    if rows:
                        st.dataframe(rows, use_container_width=True)
                if isinstance(p, str) and p:
                    fp = Path(p)
                    if fp.exists() and fp.is_file():
                        data = fp.read_bytes()
                        st.download_button(
                            "Download CSV",
                            data=data,
                            file_name=fp.name,
                            mime="text/csv",
                            use_container_width=False,
                        )

with tabs[6]:
    st.subheader("Prompt (what the LLM is reading)")
    prompt_txt = ""
    for ev in events:
        if isinstance(ev, dict) and ev.get("type") == "start":
            prompt_txt = str(ev.get("prompt") or "")
            break
    if not prompt_txt and report and isinstance(report.get("prompt"), str):
        prompt_txt = str(report.get("prompt") or "")
    st.text_area("Prompt", value=prompt_txt, height=500)

with tabs[7]:
    st.subheader("Final report")
    if report is None:
        st.info("No report.json yet (run may still be in progress).")
    else:
        st.json(report)
        st.download_button(
            "Download report.json",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name=f"{active_run_id}_report.json",
            mime="application/json",
        )

with tabs[8]:
    st.subheader("Runner logs")
    st.caption(str(paths.run_dir))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("stdout")
        st.code(_tail_text(paths.stdout_path), language="")
    with col2:
        st.markdown("stderr")
        st.code(_tail_text(paths.stderr_path), language="")


with tabs[9]:
    st.subheader("Suite")
    sid = str(st.session_state.get("active_suite_id") or "").strip()
    if not sid:
        st.info("No suite selected.")
    else:
        sp = _paths_for_suite(sid)
        suite_pid = _read_pid(sp.pid_path)
        suite_alive = bool(suite_pid is not None and _pid_alive(int(suite_pid)))
        spec_count = 0
        try:
            if sp.specs_path.exists() and sp.specs_path.is_file():
                for ln in str(sp.specs_path.read_text(encoding="utf-8", errors="replace") or "").splitlines():
                    s0 = str(ln or "").strip()
                    if not s0 or s0.startswith("#"):
                        continue
                    spec_count += 1
        except Exception:
            spec_count = 0
        done_count = _count_csv_rows(sp.summary_csv_path)
        reps0: Optional[int] = None
        manifest0 = _load_json(sp.manifest_path)
        if isinstance(manifest0, dict):
            try:
                reps0 = int(manifest0.get("replicates"))
            except Exception:
                reps0 = None
        total_count: Optional[int] = None
        if reps0 is not None and spec_count > 0:
            total_count = int(spec_count) * int(reps0)
        if total_count is not None and total_count > 0:
            try:
                st.progress(min(1.0, max(0.0, float(done_count) / float(total_count))))
            except Exception:
                pass
        c0, c1, c2 = st.columns(3)
        c0.metric("Suite", str(sid))
        c1.metric("Status", "Running" if suite_alive else "Idle/Done")
        c2.metric("Suite dir", str(sp.suite_dir))

        c3, c4, c5 = st.columns(3)
        c3.metric("Models", str(int(spec_count)))
        c4.metric("Completed", str(int(done_count)))
        c5.metric("Total", "—" if total_count is None else str(int(total_count)))

        running = _running_runs()
        running_run_id = str(running[0]) if running else ""
        if running_run_id:
            st.subheader("Current run")
            colr0, colr1 = st.columns(2)
            colr0.metric("Running run", str(running_run_id))
            if colr1.button("Attach to running run", use_container_width=True, key=f"attach_run_from_suite_main_{sid}", disabled=ui_locked_global):
                st.session_state["active_run_id"] = str(running_run_id)
                _write_active_run_id(str(running_run_id))
                st.rerun()

            rp = _paths_for_run(str(running_run_id))
            cx, cy = st.columns(2)
            with cx:
                st.markdown("stdout")
                st.code(_tail_text(rp.stdout_path), language="")
            with cy:
                st.markdown("stderr")
                st.code(_tail_text(rp.stderr_path), language="")

        st.subheader("Manifest")
        if isinstance(manifest0, dict):
            st.json(manifest0)
        else:
            st.info("No suite_manifest.json yet.")

        st.subheader("Aggregate")
        if sp.aggregate_csv_path.exists() and sp.aggregate_csv_path.is_file():
            try:
                txt = sp.aggregate_csv_path.read_text(encoding="utf-8", errors="replace")
                rows0 = _csv_to_rows(txt, max_rows=200)
                if rows0:
                    st.dataframe(rows0, use_container_width=True)
                st.download_button(
                    "Download suite_aggregate.csv",
                    data=txt.encode("utf-8", errors="replace"),
                    file_name=f"{sid}_suite_aggregate.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.warning(str(e))
        else:
            st.info("No suite_aggregate.csv yet.")

        st.subheader("Per-run summary")
        if sp.summary_csv_path.exists() and sp.summary_csv_path.is_file():
            try:
                txt2 = sp.summary_csv_path.read_text(encoding="utf-8", errors="replace")
                rows1 = _csv_to_rows(txt2, max_rows=200)
                if rows1:
                    st.dataframe(rows1, use_container_width=True)
                st.download_button(
                    "Download suite_summary.csv",
                    data=txt2.encode("utf-8", errors="replace"),
                    file_name=f"{sid}_suite_summary.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.warning(str(e))
        else:
            st.info("No suite_summary.csv yet.")

        st.subheader("Suite logs")
        colx, coly = st.columns(2)
        with colx:
            st.markdown("stdout")
            st.code(_tail_text(sp.stdout_path), language="")
        with coly:
            st.markdown("stderr")
            st.code(_tail_text(sp.stderr_path), language="")


if auto_refresh:
    time.sleep(refresh_s)
    st.rerun()
