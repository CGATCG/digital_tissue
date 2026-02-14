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
import tempfile
import time
import csv
import io
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from backend.env_keys import apply_keys_to_environ
except Exception:
    apply_keys_to_environ = None  # type: ignore


@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    events_path: Path
    report_path: Path
    issues_path: Path
    story_path: Path
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


try:
    if apply_keys_to_environ is not None:
        apply_keys_to_environ()
except Exception:
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prompts_dir() -> Path:
    return _repo_root() / "assets" / "prompts"


def _list_prompt_files() -> List[str]:
    try:
        d = _prompts_dir()
        if not (d.exists() and d.is_dir()):
            return []
        out: List[str] = []
        for p in sorted(d.glob("*.txt")):
            try:
                if p.is_file():
                    out.append(p.name)
            except Exception:
                continue
        return out
    except Exception:
        return []


def _runner_module_mtime() -> float:
    try:
        p = (_repo_root() / "trials" / "run_llm_benchmark.py").resolve()
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


@st.cache_resource
def _load_runner_module_cached(_runner_mtime: float) -> Any:
    p = (_repo_root() / "trials" / "run_llm_benchmark.py").resolve()
    spec = importlib.util.spec_from_file_location("llm_benchmark_runner", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load runner module spec: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_runner_module() -> Any:
    return _load_runner_module_cached(_runner_module_mtime())


@st.cache_resource
def _story_executor() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="story")


def _runs_root() -> Path:
    return _repo_root() / "var" / "runs" / "llm_bench"


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


def _set_active_run_id(run_id: str, *, rerun: bool = False) -> None:
    try:
        rid = str(run_id or "").strip()
        st.session_state["active_run_id"] = rid
        st.session_state["_active_run_override"] = True
        if rid:
            _write_active_run_id(rid)
        else:
            try:
                p = _active_run_id_path()
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass
        if rerun:
            st.rerun()
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
        issues_path=run_dir / "issues.json",
        story_path=run_dir / "story.md",
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


def _suite_run_entries(manifest: Any) -> List[Dict[str, Any]]:
    if not isinstance(manifest, dict):
        return []
    rr = manifest.get("runs")
    if not isinstance(rr, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in rr:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("run_id") or "").strip()
        if not rid:
            continue
        out.append(dict(r))
    return out


def _run_latest_llm_reason_fields(paths: RunPaths, *, events: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
    evs: List[Dict[str, Any]] = []
    if isinstance(events, list):
        evs = list(events)
    else:
        try:
            evs = _read_events(paths.events_path, max_events=2200)
        except Exception:
            evs = []

    last_llm: Optional[Dict[str, Any]] = None
    for ev in reversed(evs or []):
        if not isinstance(ev, dict):
            continue
        if str(ev.get("type") or "") == "llm":
            last_llm = ev
            break
    if not isinstance(last_llm, dict):
        return "", ""
    lrs0, nsr0 = _llm_reason_fields(last_llm)
    return str(lrs0 or "").strip(), str(nsr0 or "").strip()


def _llm_step_log_markdown(events: List[Dict[str, Any]], *, max_llm_steps: int = 160) -> str:
    llm_events = [ev for ev in (events or []) if isinstance(ev, dict) and str(ev.get("type") or "") == "llm"]
    if int(max_llm_steps) > 0 and len(llm_events) > int(max_llm_steps):
        llm_events = llm_events[-int(max_llm_steps) :]
    out: List[str] = []
    for ev in llm_events:
        step = ev.get("step")
        obj = _parse_first_json(str(ev.get("text") or ""))
        act = obj.get("action") if isinstance(obj, dict) else None
        method = obj.get("method") if isinstance(obj, dict) else None
        path = obj.get("path") if isinstance(obj, dict) else None
        action_line = " ".join([str(x or "").strip() for x in [act, method, path] if str(x or "").strip()]).strip()
        lrs0, nsr0 = _llm_reason_fields(ev)

        hdr = f"#### Step {step}" if step is not None else "#### Step"
        if action_line:
            hdr = hdr + f": {action_line}"
        out.append(hdr)
        out.append("")
        out.append("**last_result_summary:**")
        out.append("")
        out.append(str(lrs0 or "").strip() or "(missing)")
        out.append("")
        out.append("**next_step_rationale:**")
        out.append("")
        out.append(str(nsr0 or "").strip() or "(missing)")
        out.append("")
    return "\n".join(out).strip()


def _suite_runs_bundle_markdown(
    suite_id: str,
    *,
    include_issues: bool = True,
    include_saved_story: bool = True,
    include_latest_reason_fields: bool = True,
) -> str:
    sid = str(suite_id or "").strip()
    if not sid:
        return ""
    sp = _paths_for_suite(sid)
    manifest0 = _load_json(sp.manifest_path)
    runs = _suite_run_entries(manifest0)

    lines: List[str] = []
    lines.append(f"# Suite bundle: {sid}")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append(f"Suite dir: {sp.suite_dir}")
    lines.append(f"Runs: {len(runs)}")
    lines.append("")

    for i, ent in enumerate(runs):
        rid = str(ent.get("run_id") or "").strip()
        if not rid:
            continue
        provider = str(ent.get("provider") or "").strip()
        model = str(ent.get("model") or "").strip()
        replicate = str(ent.get("replicate") or "").strip()
        exit_code = ent.get("exit_code")

        hdr = f"## Run {i + 1}/{len(runs)}: {rid}"
        meta = " | ".join(
            [
                x
                for x in [
                    (f"provider={provider}" if provider else ""),
                    (f"model={model}" if model else ""),
                    (f"replicate={replicate}" if replicate else ""),
                    (f"exit_code={exit_code}" if exit_code is not None else ""),
                ]
                if x
            ]
        ).strip()

        lines.append(hdr)
        if meta:
            lines.append(meta)
        lines.append("")

        rp = _paths_for_run(rid)

        need_events = False
        if include_latest_reason_fields:
            need_events = True
        if include_issues:
            try:
                if not (rp.issues_path.exists() and rp.issues_path.is_file()):
                    need_events = True
            except Exception:
                need_events = True
        if include_saved_story:
            try:
                if not (rp.story_path.exists() and rp.story_path.is_file()):
                    need_events = True
            except Exception:
                need_events = True

        events: List[Dict[str, Any]] = []
        if need_events:
            try:
                events = _read_events(rp.events_path, max_events=2200)
            except Exception:
                events = []

        if include_latest_reason_fields:
            lrs, nsr = _run_latest_llm_reason_fields(rp, events=events)
            lines.append("### Latest last_result_summary")
            lines.append("")
            lines.append(lrs if lrs else "(missing)")
            lines.append("")
            lines.append("### Latest next_step_rationale")
            lines.append("")
            lines.append(nsr if nsr else "(missing)")
            lines.append("")

        if include_issues:
            issues_obj = _load_json_any(rp.issues_path)
            if issues_obj is None and events:
                try:
                    issues_obj = _detect_issues(events, rp)
                except Exception:
                    issues_obj = None
            lines.append("### issues.json")
            lines.append("")
            if issues_obj is None:
                lines.append("(missing)")
                lines.append("")
            else:
                try:
                    issues_txt = json.dumps(issues_obj, indent=2, ensure_ascii=False)
                except Exception:
                    issues_txt = str(issues_obj)
                lines.append("```json")
                lines.append(issues_txt)
                lines.append("```")
                lines.append("")

        if include_saved_story:
            story_txt = _load_text(rp.story_path)
            lines.append("### story.md")
            lines.append("")
            if story_txt.strip():
                lines.append(story_txt.strip())
            else:
                lines.append("(missing)")
                if events:
                    step_log = _llm_step_log_markdown(events)
                    if step_log.strip():
                        lines.append("")
                        lines.append("### Fallback step log")
                        lines.append("")
                        lines.append(step_log)
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


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


def _write_json_atomic(path: Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd: Optional[int] = None
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=str(p.name) + ".", suffix=".tmp", dir=str(p.parent))
        with os.fdopen(int(fd), "w", encoding="utf-8") as f:
            f.write(txt)
        fd = None
        os.replace(str(tmp_path), str(p))
    finally:
        try:
            if fd is not None:
                try:
                    os.close(int(fd))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if tmp_path and os.path.exists(str(tmp_path)):
                os.unlink(str(tmp_path))
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
            if p.is_dir() and str(p.name).startswith("run_"):
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
        if llm_path in (
            "/api/game/state",
            "/api/tests/disease/models",
            "/api/tests/disease/proteins",
            "/api/bulk_omics/sets",
            "/api/spatial_tx/gene_sets",
            "/api/spatial_omics/type",
            "/api/omics/inventory",
        ):
            slow = 10.0
        if llm_path in (
            "/api/tests/disease/bulk_omics",
            "/api/tests/disease/spatial_tx",
            "/api/tests/disease/spatial_omics",
            "/api/tests/disease/characterization",
            "/api/tests/disease/protein_screen",
            "/api/tests/disease/claim_cure",
        ):
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

    if llm_path in ("/api/spatial_tx/gene_sets", "/api/spatial_omics/type"):
        key = "gene_sets" if llm_path == "/api/spatial_tx/gene_sets" else "types"
        gs = shape_rj.get(key) if isinstance(shape_rj, dict) else None
        if not (isinstance(gs, list) and any(str(x or "").strip() for x in gs)):
            issues.append({"severity": "warn", "kind": "empty_gene_sets", "summary": f"Expected non-empty {key}"})

    if llm_path == "/api/tests/disease/estimate_cost":
        charge = shape_rj.get("charge") if isinstance(shape_rj, dict) else None
        if not isinstance(charge, dict):
            issues.append({"severity": "warn", "kind": "missing_charge", "summary": "Expected estimate_cost to return charge{}"})

    if llm_path in (
        "/api/tests/disease/characterization",
        "/api/tests/disease/bulk_omics",
        "/api/tests/disease/spatial_tx",
        "/api/tests/disease/spatial_omics",
        "/api/tests/disease/protein_screen",
    ):
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


def _extract_omics_csv_refs_from_report(report: Any, *, base_url: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return []
    if not isinstance(report, dict):
        return []
    tr = report.get("transcript")
    if not isinstance(tr, list):
        return []
    for ent in tr:
        if not isinstance(ent, dict):
            continue
        if str(ent.get("type") or "") != "api":
            continue
        res = ent.get("result")
        rj = res.get("response_json") if isinstance(res, dict) else None
        if not isinstance(rj, dict):
            continue
        rid = str(rj.get("run_id") or "").strip()
        files = rj.get("files")
        if not rid or (not isinstance(files, list)) or (not files):
            continue
        for f in files:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name") or "").strip()
            if not name.lower().endswith(".csv"):
                continue
            try:
                nbytes = int(f.get("bytes") or 0)
            except Exception:
                nbytes = 0
            q_rid = urllib.parse.quote(rid, safe="")
            q_name = urllib.parse.quote(name, safe="")
            url = f"{base}/api/omics/file?run_id={q_rid}&name={q_name}"
            out.append(
                {
                    "source": "omics",
                    "event_seq": ent.get("seq"),
                    "api_path": ent.get("path"),
                    "omics_run_id": rid,
                    "name": name,
                    "bytes": nbytes,
                    "url": url,
                }
            )
    return out


def _scan_local_csv_files(files_dir: Path, *, limit: int = 3000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        if not (files_dir.exists() and files_dir.is_dir()):
            return []
        for p in files_dir.rglob("*.csv"):
            if len(out) >= int(limit):
                break
            try:
                if not p.is_file():
                    continue
            except Exception:
                continue
            try:
                sz = int(p.stat().st_size)
            except Exception:
                sz = 0
            out.append({"path": str(p), "bytes": int(sz), "event_seq": None})
    except Exception:
        return []
    out.sort(key=lambda d: str(d.get("path") or ""))
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
            rj = ev.get("response_json")
            ignore_err = bool(isinstance(rj, dict) and str(rj.get("error_kind") or "").strip() == "all_replicates_died")
            try:
                st_i = int(status)
            except Exception:
                st_i = None

            if (not ignore_err) and st_i is not None and st_i >= 400:
                sev = "error" if st_i >= 500 else "warn"
                kind = "api_http_error"
                summary = f"API {method} {path} -> HTTP {st_i}"
                if st_i == 429:
                    sev = "warn"
                    kind = "api_rate_limited"
            if (not ignore_err) and sev is None and isinstance(rj, dict):
                if rj.get("ok") is False:
                    sev = "warn"
                    kind = "api_ok_false"
                    summary = f"API {method} {path} returned ok=false"
                if isinstance(rj.get("error"), str) and str(rj.get("error") or "").strip():
                    sev = "warn" if sev is None else sev
                    kind = "api_error_field" if kind is None else kind
                    if not summary:
                        summary = f"API {method} {path} returned error field"

            if (not ignore_err) and sev is not None:
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
            rj = payload.get("response_json")
            ignore_err = bool(isinstance(rj, dict) and str(rj.get("error_kind") or "").strip() == "all_replicates_died")
            try:
                st_i = int(status)
            except Exception:
                st_i = None
            if (not ignore_err) and st_i is not None and st_i >= 400:
                sev = "error" if st_i >= 500 else "warn"
                kind = "tool_http_error"
                if st_i == 429:
                    sev = "warn"
                    kind = "tool_rate_limited"
            if (not ignore_err) and sev is None and isinstance(rj, dict):
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
            if (not ignore_err) and sev is not None:
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

    out.extend(_issues_from_text(text=_tail_text_cached(paths.stderr_path), source="runner_stderr"))
    out.extend(_issues_from_text(text=_tail_text_cached(paths.stdout_path), source="runner_stdout"))

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


def _tail_text_cached(path: Path, *, max_bytes: int = 80_000) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        st0 = path.stat()
        key = f"{str(path)}::{int(max_bytes)}"

        cache0 = st.session_state.get("tail_cache")
        if not isinstance(cache0, dict):
            cache0 = {}
            st.session_state["tail_cache"] = cache0

        ent = cache0.get(key)
        if isinstance(ent, dict):
            try:
                if float(ent.get("mtime") or 0.0) == float(st0.st_mtime) and int(ent.get("size") or 0) == int(st0.st_size):
                    txt0 = ent.get("text")
                    return str(txt0 or "")
            except Exception:
                pass

        txt = _tail_text(path, max_bytes=int(max_bytes))
        cache0[key] = {
            "mtime": float(st0.st_mtime),
            "size": int(st0.st_size),
            "text": str(txt or ""),
        }
        st.session_state["tail_cache"] = cache0
        return txt
    except Exception:
        return _tail_text(path, max_bytes=int(max_bytes))


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


def _read_events_incremental(path: Path, *, run_key: str, max_events: int = 600) -> List[Dict[str, Any]]:
    try:
        if not path.exists() or not path.is_file():
            return []

        st0 = path.stat()
        inode = (int(getattr(st0, "st_dev", 0) or 0), int(getattr(st0, "st_ino", 0) or 0))
        size = int(getattr(st0, "st_size", 0) or 0)

        cache0 = st.session_state.get("events_cache")
        if not isinstance(cache0, dict):
            cache0 = {}
            st.session_state["events_cache"] = cache0

        ent0 = cache0.get(str(run_key))
        ent: Dict[str, Any] = ent0 if isinstance(ent0, dict) else {}

        prev_inode = ent.get("inode")
        prev_pos = ent.get("pos")
        prev_buf = ent.get("buf")
        prev_max = ent.get("max_events")
        events0 = ent.get("events")
        events: List[Dict[str, Any]] = events0 if isinstance(events0, list) else []
        buf = str(prev_buf or "") if isinstance(prev_buf, str) else ""
        pos = int(prev_pos) if isinstance(prev_pos, int) else None

        need_reset = False
        if prev_inode != inode:
            need_reset = True
        if pos is None:
            need_reset = True
        if pos is not None and size < int(pos):
            need_reset = True
        if prev_max is None or int(prev_max) != int(max_events):
            need_reset = True

        if need_reset:
            events = []
            buf = ""
            lines = _tail_jsonl_lines(path, max_lines=int(max_events))
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        events.append(obj)
                except Exception:
                    continue
            events.sort(key=lambda e: int(e.get("seq") or 0))
            if len(events) > int(max_events):
                events = events[-int(max_events) :]
            pos = int(size)
        else:
            raw = b""
            with path.open("rb") as f:
                try:
                    f.seek(int(pos or 0))
                except Exception:
                    f.seek(0)
                raw = f.read()
            pos = int(size)
            txt = raw.decode("utf-8", errors="replace") if raw else ""
            if buf:
                txt = buf + txt

            if txt and (not txt.endswith("\n")):
                i = txt.rfind("\n")
                if i >= 0:
                    buf = txt[i + 1 :]
                    txt = txt[: i + 1]
                else:
                    buf = txt
                    txt = ""
            else:
                buf = ""

            if txt.strip():
                for ln in txt.splitlines():
                    if not str(ln).strip():
                        continue
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict):
                            events.append(obj)
                    except Exception:
                        continue

            events.sort(key=lambda e: int(e.get("seq") or 0))
            if len(events) > int(max_events):
                events = events[-int(max_events) :]

        cache0[str(run_key)] = {
            "inode": inode,
            "pos": int(pos or 0),
            "buf": str(buf or ""),
            "events": events,
            "max_events": int(max_events),
        }
        st.session_state["events_cache"] = cache0
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
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _load_json_any(path: Path) -> Any:
    try:
        if not path.exists() or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _load_text(path: Path) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        return str(path.read_text(encoding="utf-8", errors="replace") or "")
    except Exception:
        return ""


def _http_get_bytes(url: str, *, timeout_s: float = 30.0) -> bytes:
    u = str(url or "").strip()
    if not u:
        return b""
    req = urllib.request.Request(u, method="GET")
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        return bytes(resp.read())


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


def _latest_llm_prompt_text(paths: RunPaths, *, max_chars: int = 0) -> Tuple[str, str]:
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
            if not msgs:
                msgs = req.get("input") if isinstance(req.get("input"), list) else []
            lines: List[str] = [header, f"provider=openai model={req.get('model')}", "", "MESSAGES:"]
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "")
                content = _payload_text_blocks(m.get("content"))
                lines.append(f"[{role}]\n{content}\n")
            txt = "\n".join(lines).strip()
            if len(msgs) <= 0:
                txt = header + "\n\n" + raw
            elif str(txt or "").strip().endswith("MESSAGES:"):
                txt = header + "\n\n" + raw
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

    if isinstance(max_chars, int) and int(max_chars) > 0 and len(txt) > int(max_chars):
        txt = txt[: int(max_chars)]
    return txt, h


def _extract_context_summary_from_messages(msgs: Any) -> str:
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if not isinstance(m, dict):
            continue
        c = str(m.get("content") or "")
        c2 = c.lstrip()
        if not c2.startswith("CONTEXT_SUMMARY_ENTRY:"):
            continue
        s = c2[len("CONTEXT_SUMMARY_ENTRY:") :]
        return str(s).strip()
    return ""


def _read_context_summary_cached(paths: RunPaths, *, run_key: str) -> Tuple[str, str]:
    try:
        p = paths.state_path
        if not p.exists() or not p.is_file():
            return "", ""
        st0 = p.stat()
        cache0 = st.session_state.get("context_summary_cache")
        if not isinstance(cache0, dict):
            cache0 = {}
            st.session_state["context_summary_cache"] = cache0

        ent = cache0.get(str(run_key))
        if isinstance(ent, dict) and float(ent.get("mtime") or 0.0) == float(st0.st_mtime):
            return str(ent.get("text") or ""), str(ent.get("hash") or "")

        st_state = _load_json(p)
        txt = ""
        if isinstance(st_state, dict):
            txt = _extract_context_summary_from_messages(st_state.get("messages"))
        h = hashlib.sha256(txt.encode("utf-8", errors="replace")).hexdigest() if txt else ""
        cache0[str(run_key)] = {"mtime": float(st0.st_mtime), "text": str(txt or ""), "hash": str(h or "")}
        st.session_state["context_summary_cache"] = cache0
        return str(txt or ""), str(h or "")
    except Exception:
        return "", ""


def _read_lab_notebook(paths: RunPaths) -> str:
    st_state = _load_json(paths.state_path)
    if not isinstance(st_state, dict):
        return ""
    return str(st_state.get("notebook") or "").strip()


def _read_lab_notebook_cached(paths: RunPaths, *, run_key: str) -> Tuple[str, str]:
    try:
        p = paths.state_path
        if not p.exists() or not p.is_file():
            return "", ""
        st0 = p.stat()
        cache0 = st.session_state.get("notebook_cache")
        if not isinstance(cache0, dict):
            cache0 = {}
            st.session_state["notebook_cache"] = cache0
        ent = cache0.get(str(run_key))
        if isinstance(ent, dict) and float(ent.get("mtime") or 0.0) == float(st0.st_mtime):
            return str(ent.get("text") or ""), str(ent.get("hash") or "")

        txt = _read_lab_notebook(paths)
        h = hashlib.sha256(txt.encode("utf-8", errors="replace")).hexdigest() if txt else ""
        cache0[str(run_key)] = {"mtime": float(st0.st_mtime), "text": str(txt or ""), "hash": str(h or "")}
        st.session_state["notebook_cache"] = cache0
        return str(txt or ""), str(h or "")
    except Exception:
        return "", ""


_LATEST_LLM_PROMPT_CACHE_VERSION = 3


def _latest_llm_prompt_text_cached(paths: RunPaths, *, run_key: str, max_chars: int = 0) -> Tuple[str, str]:
    try:
        p = _latest_llm_payload_path(paths)
        if p is None or (not p.exists()) or (not p.is_file()):
            return "", ""
        st0 = p.stat()
        cache0 = st.session_state.get("latest_prompt_cache")
        if not isinstance(cache0, dict):
            cache0 = {}
            st.session_state["latest_prompt_cache"] = cache0
        ent = cache0.get(str(run_key))
        if isinstance(ent, dict):
            if (
                int(ent.get("v") or 0) == int(_LATEST_LLM_PROMPT_CACHE_VERSION)
                and str(ent.get("path") or "") == str(p)
                and float(ent.get("mtime") or 0.0) == float(st0.st_mtime)
            ):
                return str(ent.get("text") or ""), str(ent.get("hash") or "")

        txt, h = _latest_llm_prompt_text(paths, max_chars=int(max_chars))
        cache0[str(run_key)] = {
            "v": int(_LATEST_LLM_PROMPT_CACHE_VERSION),
            "path": str(p),
            "mtime": float(st0.st_mtime),
            "text": str(txt or ""),
            "hash": str(h or ""),
        }
        st.session_state["latest_prompt_cache"] = cache0
        return str(txt or ""), str(h or "")
    except Exception:
        return "", ""


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


def _extract_omics_csv_refs(events: List[Dict[str, Any]], *, base_url: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("type") or "") != "api":
            continue
        rj = ev.get("response_json")
        if not isinstance(rj, dict):
            continue
        rid = str(rj.get("run_id") or "").strip()
        files = rj.get("files")
        if not rid or (not isinstance(files, list)) or (not files):
            continue
        for f in files:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name") or "").strip()
            if not name.lower().endswith(".csv"):
                continue
            try:
                nbytes = int(f.get("bytes") or 0)
            except Exception:
                nbytes = 0
            q_rid = urllib.parse.quote(rid, safe="")
            q_name = urllib.parse.quote(name, safe="")
            url = f"{base}/api/omics/file?run_id={q_rid}&name={q_name}"
            out.append(
                {
                    "source": "omics",
                    "event_seq": ev.get("seq"),
                    "api_path": ev.get("path"),
                    "omics_run_id": rid,
                    "name": name,
                    "bytes": nbytes,
                    "url": url,
                }
            )
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
    t = re.sub(r"\bcancerous\b", "diseased", t, flags=re.IGNORECASE)
    t = re.sub(r"\bcancer\b", "disease", t, flags=re.IGNORECASE)
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
            "/api/tests/cancer/spatial_omics",
            "/api/tests/hereditary_disease/bulk_omics",
            "/api/tests/hereditary_disease/spatial_tx",
            "/api/tests/hereditary_disease/spatial_omics",
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
    max_tokens_i = 256
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
        "gpt-5.2-none",
        "gpt-5.2-low",
        "gpt-5.2-medium",
        "gpt-5.2-high",
        "gpt-5.2-extra-high",
    ]
    for m in openai_models:
        out.append(("openai", str(m)))

    xai_models = [
        "grok-4",
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
    ]
    for m in xai_models:
        out.append(("xai", str(m)))

    gemini_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]
    for m in gemini_models:
        out.append(("gemini", str(m)))

    claude_models = [
        "claude-haiku-4-5-20251001",
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
    model = "cancer_cell_culture"
    screen_model = "cancer_cell_culture"
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
                    preferred = "cancer_cell_culture"
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

    gene_sets = _preflight_call(base_url=base_url, llm_path="/api/spatial_omics/type", method="GET", timeout_s=api_timeout_s, omics_state=omics_state)
    out.append(gene_sets)
    gene_set = "spatial_rna"
    try:
        grj = gene_sets.get("api_response_json") if isinstance(gene_sets, dict) else None
        if isinstance(grj, dict) and isinstance(grj.get("types"), list) and grj.get("types"):
            gene_set = str(grj.get("types")[0])
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
            llm_path="/api/tests/disease/spatial_omics",
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
                "max_tokens": 4096,
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
        key="download_preflight_bundle_json",
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
        "You are monitoring an LLM benchmark where an agent takes actions in a biology strategy puzzle. "
        "Summarize ONLY the provided log excerpt into factual notes about actions, results, and errors. "
        "Assume the excerpt is valid and do not comment on truncation or missing context. Do not speculate."
    )
    user_msg = (
        f"Run: {run_id}\n"
        f"Status: {'RUNNING' if proc_alive else 'STOPPED/DONE'}\n"
        f"Excerpt: {int(chunk_i) + 1}/{int(chunk_n)}\n"
        "\n"
        "The log below represents a series of actions that an agent took to solve a biological puzzle.\n"
        "Summarize the events at a high level, focusing on actions, outcomes, and decisions.\n"
        "Write 18-30 short bullet-like lines (plain text, one per line) capturing: key actions, key results, errors, and next intentions if stated. "
        "Do not add anything not present in the excerpt.\n"
        "\n"
        "LOG EXCERPT:\n"
        f"{_mask_disease_term(str(chunk or ''))}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=1200,
            reasoning_effort="medium",
        )
    except TypeError:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=1200,
        )
    out = None
    try:
        out = resp.choices[0].message.content
    except Exception:
        out = None
    return _mask_disease_term(str(out or "").strip())


def _openai_story_concise_from_llm_reasons(*, api_key: str, reason_log: str, run_id: str, proc_alive: bool) -> str:
    if OpenAI is None:
        raise RuntimeError("openai python package not available")
    client = OpenAI(api_key=str(api_key))

    sys_msg = (
        "You are monitoring an LLM benchmark where an agent tries to solve a biological puzzle. "
        "The input is the agent's own step-by-step notes: last_result_summary and next_step_rationale. "
        "Write a concise, high-level story for a human evaluator. "
        "Only use information present in the input. Do not speculate." 
    )
    user_msg = (
        f"Run: {run_id}\n"
        f"Status: {'RUNNING' if proc_alive else 'STOPPED/DONE'}\n"
        "\n"
        "Turn the step summaries below into a concise high-level story. "
        "Focus on: what the agent tried, what it learned, and what it intends to do next. "
        "Constraints: 3-6 sentences total, <= 180 words, no bullets, no speculation.\n"
        "\n"
        "STEP SUMMARIES:\n"
        f"{_mask_disease_term(str(reason_log or ''))}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=520,
            reasoning_effort="medium",
        )
    except TypeError:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=520,
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


def _event_digest(events: List[Dict[str, Any]], *, max_events: int = 220, max_chars: int = 0) -> str:
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
    if not isinstance(max_chars, int) or int(max_chars) <= 0:
        return txt
    if len(txt) <= int(max_chars):
        return txt
    return txt[-int(max_chars) :]


def _llm_reason_digest(
    events: List[Dict[str, Any]],
    report: Any,
    *,
    max_steps: int = 80,
    max_chars: int = 0,
) -> str:
    src: List[Dict[str, Any]] = []
    tr = report.get("transcript") if isinstance(report, dict) else None
    if isinstance(tr, list):
        src = [e for e in tr if isinstance(e, dict)]
    else:
        src = [e for e in (events or []) if isinstance(e, dict)]

    llm_rows: List[str] = []
    for ev in src:
        if str(ev.get("type") or "") != "llm":
            continue
        step = ev.get("step")
        lrs, nsr = _llm_reason_fields(ev)
        if not (isinstance(lrs, str) or isinstance(nsr, str)):
            continue
        lrs_s = str(lrs or "").strip()
        nsr_s = str(nsr or "").strip()
        if not (lrs_s or nsr_s):
            continue
        llm_rows.append(f"step {step}: last_result_summary={_truncate(lrs_s, 500)} | next_step_rationale={_truncate(nsr_s, 700)}")

    if not llm_rows:
        return ""
    llm_rows = llm_rows[-max(1, int(max_steps)) :]
    txt = "\n".join(llm_rows).strip()
    if not isinstance(max_chars, int) or int(max_chars) <= 0:
        return txt
    if len(txt) <= int(max_chars):
        return txt
    return txt[-int(max_chars) :]


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
        "The input is a log of the agent's messages/actions and tool/API results. "
        "Your job is to help a human evaluator understand what happened and what the agent is doing now. "
        "Only use information present in the log. Do not speculate. Do not comment on truncation or missing context. "
        "Be ultra-concise and clear."
    )
    user_msg = (
        f"Run: {run_id}\n"
        f"Status: {'RUNNING' if proc_alive else 'STOPPED/DONE'}\n"
        "\n"
        "The log below represents a series of actions an agent took to solve a biological puzzle.\n"
        "Summarize the events at a high level, as completely as possible. Mention: what it tried, what it learned, and what it will do next. "
        "If there were errors or obvious misinterpretations, mention them briefly. "
        "Constraints: 10-18 sentences total, <= 500 words, no bullets, no speculation.\n"
        "\n"
        "LOG DIGEST:\n"
        f"{_mask_disease_term(digest)}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=1200,
            reasoning_effort="medium",
        )
    except TypeError:
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=1200,
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
    st.session_state.setdefault("context_summary_cache", {})
    st.session_state.setdefault("_active_run_override", False)
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
                    _set_active_run_id(str(saved), rerun=False)
                    return
            except Exception:
                pass

        running = _running_runs()
        if running:
            _set_active_run_id(str(running[0]), rerun=False)
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


st.set_page_config(page_title="LLM Benchmarks", layout="wide")

_ensure_session_state()
_bootstrap_active_run_id()
_bootstrap_active_suite_id()

col_title, col_refresh = st.columns([0.78, 0.22])
with col_title:
    st.title("LLM Benchmarks")
with col_refresh:
    manual_refresh_clicked = st.button("Refresh now", use_container_width=True, key="refresh_now_top")

proc0 = st.session_state.get("proc")
proc0_alive = False
try:
    if proc0 is not None and getattr(proc0, "poll")() is None:
        proc0_alive = True
except Exception:
    proc0_alive = False

suite_proc0 = st.session_state.get("suite_proc")
suite_proc0_alive = False
try:
    if suite_proc0 is not None and getattr(suite_proc0, "poll")() is None:
        suite_proc0_alive = True
except Exception:
    suite_proc0_alive = False

running_runs_global = _running_runs()
running_suites_global = _running_suites()
ui_locked_global = bool(running_runs_global or running_suites_global or proc0_alive or suite_proc0_alive)

try:
    prev_suite_id0 = str(st.session_state.get("_prev_active_suite_id") or "").strip()
    cur_suite_id0 = str(st.session_state.get("active_suite_id") or "").strip()
    if cur_suite_id0 != prev_suite_id0:
        st.session_state["_prev_active_suite_id"] = str(cur_suite_id0)
except Exception:
    pass

try:
    # Keep the sidebar run selector (run_choice) and the global run state (active_run_id)
    # synchronized. Programmatic run switches set _active_run_override=True.
    override0 = bool(st.session_state.get("_active_run_override") is True)
    active0 = str(st.session_state.get("active_run_id") or "").strip()
    choice0 = str(st.session_state.get("run_choice") or "").strip()

    if override0:
        st.session_state["_active_run_override"] = False
        st.session_state["run_choice"] = str(active0)
    else:
        if choice0 and choice0 != active0:
            try:
                if _paths_for_run(str(choice0)).run_dir.exists():
                    _set_active_run_id(str(choice0), rerun=False)
            except Exception:
                pass
        elif active0 and (not choice0) and ("run_choice" not in st.session_state):
            st.session_state["run_choice"] = str(active0)
except Exception:
    pass

if ui_locked_global:
    try:
        cur_run0 = str(st.session_state.get("active_run_id") or "").strip()
        if running_runs_global and (not cur_run0):
            rr0 = str(running_runs_global[0])
            if rr0:
                _set_active_run_id(str(rr0), rerun=False)
        cur_suite0 = str(st.session_state.get("active_suite_id") or "").strip()
        if running_suites_global and (not cur_suite0):
            ss0 = str(running_suites_global[0])
            if ss0:
                st.session_state["active_suite_id"] = ss0
                _write_active_suite_id(ss0)
    except Exception:
        pass

active_run_id0 = str(st.session_state.get("active_run_id") or "").strip()
active_suite_id0 = str(st.session_state.get("active_suite_id") or "").strip()
suite_running0 = bool(running_suites_global or suite_proc0_alive)
if active_suite_id0 and suite_running0 and (not active_run_id0):
    cand = ""
    try:
        if running_runs_global:
            cand = str(running_runs_global[0])
        else:
            cand = str(_read_active_run_id() or "").strip()
    except Exception:
        cand = ""

    if cand:
        try:
            if not _paths_for_run(str(cand)).run_dir.exists():
                cand = ""
        except Exception:
            cand = ""

    if str(cand or "").strip() != active_run_id0:
        _set_active_run_id(str(cand or "").strip(), rerun=False)

if ui_locked_global:
    try:
        if not str(st.session_state.get("run_choice") or "").strip():
            st.session_state["run_choice"] = str(st.session_state.get("active_run_id") or "").strip()
        if not str(st.session_state.get("suite_choice") or "").strip():
            st.session_state["suite_choice"] = str(st.session_state.get("active_suite_id") or "").strip()
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
        max_tokens = 8000
        api_timeout = 5000.0
        llm_timeout = 5000.0
        player_id = ""
        reset_first = False

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
                    "gpt-5.2-none",
                    "gpt-5.2-low",
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
                    "gemini-3-flash-preview",
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
                    "grok-4-1-fast-reasoning",
                    "grok-4-1-fast-non-reasoning",
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
                    "claude-haiku-4-5-20251001",
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
                "gpt-5.2-none",
                "gpt-5.2-low",
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
                "grok-4-1-fast-reasoning",
                "grok-4-1-fast-non-reasoning",
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
                "gemini-3-flash-preview",
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
                "claude-haiku-4-5-20251001",
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

        st.subheader("Prompt")
        prompt_files = _list_prompt_files()
        if not prompt_files:
            prompt_files = ["default.txt"]
        prev_prompt = str(st.session_state.get("prompt_file") or "").strip()
        ch0 = str(challenge or "").strip().lower()
        preferred_prompt = "aging.txt" if ch0 == "aging" else "default.txt"
        last_ch0 = str(st.session_state.get("_last_challenge_prompt") or "").strip().lower()
        if last_ch0 and last_ch0 != ch0:
            if preferred_prompt in prompt_files:
                prev_prompt = preferred_prompt
        if prev_prompt not in prompt_files:
            if preferred_prompt in prompt_files:
                prev_prompt = preferred_prompt
            else:
                prev_prompt = prompt_files[0]
        prompt_file = st.selectbox(
            "Initial prompt",
            options=prompt_files,
            index=int(prompt_files.index(prev_prompt)),
            key="prompt_file_choice",
            disabled=ui_locked_global,
        )
        st.session_state["prompt_file"] = str(prompt_file)
        st.session_state["_last_challenge_prompt"] = str(ch0)

        st.subheader("Saved runs")
        runs = _list_runs()
        running_set = set(_running_runs())
        active_run_id = str(st.session_state.get("active_run_id") or "").strip()
        any_running = bool(running_set)
        options = [""] + runs
        if active_run_id and active_run_id not in options:
            _set_active_run_id("", rerun=False)
            active_run_id = ""
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
            key="run_choice",
            disabled=ui_locked_global,
        )
        selected_run_id = str(sel_run or "").strip()
        if str(selected_run_id) != str(st.session_state.get("active_run_id") or "").strip():
            _set_active_run_id(str(selected_run_id), rerun=False)

        selected_running = bool(selected_run_id and selected_run_id in running_set)

        resumable = False
        try:
            if selected_run_id:
                resumable = bool(_paths_for_run(selected_run_id).state_path.exists())
        except Exception:
            resumable = False

        selected_has_run = bool(selected_run_id)
        col_a, col_b, col_c = st.columns(3)
        start_clicked = col_a.button("Start new", use_container_width=True, disabled=bool(ui_locked_global or selected_has_run))
        stop_clicked = col_b.button(
            "Stop",
            use_container_width=True,
            disabled=bool((not any_running) or bool(running_suites_global) or bool(suite_proc0_alive)),
        )
        resume_clicked = col_c.button(
            "Resume",
            use_container_width=True,
            disabled=bool(ui_locked_global or (not selected_has_run) or selected_running or (not resumable)),
        )

    with tab_suite:
        st.header("Suite")
        suites = _list_suites()
        suite_running_set = set(_running_suites())
        suite_proc = st.session_state.get("suite_proc")
        suite_proc_alive = False
        try:
            if suite_proc is not None and getattr(suite_proc, "poll")() is None:
                suite_proc_alive = True
        except Exception:
            suite_proc_alive = False
        active_suite_id = str(st.session_state.get("active_suite_id") or "").strip()
        any_suite_running = bool(suite_running_set) or bool(suite_proc_alive)
        selected_suite_running = bool(active_suite_id and (active_suite_id in suite_running_set or suite_proc_alive))

        suite_options = [""] + suites
        suite_idx = 0
        try:
            if active_suite_id and active_suite_id in suite_options:
                suite_idx = int(suite_options.index(active_suite_id))
        except Exception:
            suite_idx = 0

        if active_suite_id and active_suite_id not in suite_options:
            st.session_state["active_suite_id"] = ""
            active_suite_id = ""

        sel_suite = st.selectbox(
            "Select suite",
            options=suite_options,
            index=int(suite_idx),
            format_func=lambda x: (str(x) + (" (running)" if str(x) in suite_running_set else "")) if str(x) else "",
            key="suite_choice",
            disabled=ui_locked_global,
        )
        selected_suite_id = str(sel_suite or "").strip()
        st.session_state["active_suite_id"] = str(selected_suite_id)

        sel_suite_running_now = bool(selected_suite_id and (selected_suite_id in suite_running_set or suite_proc_alive))
        if selected_suite_id and sel_suite_running_now:
            pass
        if selected_suite_id and (not sel_suite_running_now):
            try:
                cur_run = str(st.session_state.get("active_run_id") or "").strip()
                cur_choice = str(st.session_state.get("run_choice") or "").strip() if ("run_choice" in st.session_state) else None
                if (not cur_run) and not (cur_choice is not None and (not cur_choice)):
                    sp0 = _paths_for_suite(str(selected_suite_id))
                    man0 = _load_json(sp0.manifest_path)
                    run_ids0: List[str] = []
                    if isinstance(man0, dict):
                        rr0 = man0.get("runs")
                        if isinstance(rr0, list):
                            for r in rr0:
                                if not isinstance(r, dict):
                                    continue
                                rid = str(r.get("run_id") or "").strip()
                                if rid:
                                    run_ids0.append(rid)
                    run_ids0 = [r for r in run_ids0 if r]
                    run_ids0 = list(dict.fromkeys(run_ids0))
                    if run_ids0:
                        last_rid = str(run_ids0[-1])
                        if str(st.session_state.get("_suite_run_bootstrap") or "") != str(selected_suite_id):
                            _set_active_run_id(str(last_rid), rerun=False)
                            st.session_state["_suite_run_bootstrap"] = str(selected_suite_id)
                            st.rerun()
            except Exception:
                pass

        active_suite_id = str(st.session_state.get("active_suite_id") or "").strip()
        if active_suite_id:
            _write_active_suite_id(str(active_suite_id))
        else:
            try:
                p = _active_suite_id_path()
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass

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

        prompt_files2 = _list_prompt_files()
        if not prompt_files2:
            prompt_files2 = ["default.txt"]
        prev_suite_prompt = str(st.session_state.get("suite_prompt_file") or "").strip()
        ch1 = str(challenge or "").strip().lower()
        preferred_suite_prompt = "aging.txt" if ch1 == "aging" else "default.txt"
        last_ch1 = str(st.session_state.get("_last_challenge_suite_prompt") or "").strip().lower()
        if last_ch1 and last_ch1 != ch1:
            if preferred_suite_prompt in prompt_files2:
                prev_suite_prompt = preferred_suite_prompt
        if prev_suite_prompt not in prompt_files2:
            if preferred_suite_prompt in prompt_files2:
                prev_suite_prompt = preferred_suite_prompt
            else:
                prev_suite_prompt = prompt_files2[0]
        suite_prompt_file = st.selectbox(
            "Suite initial prompt",
            options=prompt_files2,
            index=int(prompt_files2.index(prev_suite_prompt)),
            key="suite_prompt_file_choice",
            disabled=ui_locked_global,
        )
        st.session_state["suite_prompt_file"] = str(suite_prompt_file)
        st.session_state["_last_challenge_suite_prompt"] = str(ch1)

        pairs = _llm_provider_model_options()
        avail_specs = [str(p) + ":" + str(m) for p, m in pairs]
        prev_sel = st.session_state.get("suite_models")
        if not isinstance(prev_sel, list):
            prev_sel = []
        prev_set = set(str(x) for x in prev_sel)
        sel_models: List[str] = []
        st.markdown("Models")
        with st.container(height=260):
            for spec0 in avail_specs:
                k0 = hashlib.md5(str(spec0).encode("utf-8", errors="replace")).hexdigest()[:10]
                checked = bool(
                    st.checkbox(
                        str(spec0),
                        value=bool(str(spec0) in prev_set),
                        key=f"suite_model_cb_{k0}",
                        disabled=ui_locked_global,
                    )
                )
                if checked:
                    sel_models.append(str(spec0))
        st.session_state["suite_models"] = list(sel_models)

        with st.expander("Advanced", expanded=False):
            suite_max_parallel = int(
                st.number_input(
                    "Max parallel runs",
                    min_value=1,
                    max_value=64,
                    value=int(st.session_state.get("suite_max_parallel") or 1),
                    step=1,
                    disabled=ui_locked_global,
                    help="How many benchmark runs to execute concurrently inside a suite.",
                )
            )
            st.session_state["suite_max_parallel"] = int(suite_max_parallel)
            suite_max_per_provider = int(
                st.number_input(
                    "Max parallel per provider",
                    min_value=1,
                    max_value=8,
                    value=int(st.session_state.get("suite_max_per_provider") or 1),
                    step=1,
                    disabled=ui_locked_global,
                    help="Hard cap on concurrent runs for the same provider.",
                )
            )
            st.session_state["suite_max_per_provider"] = int(suite_max_per_provider)

        suite_reps = int(st.number_input("Replicates", min_value=1, max_value=50, value=1, step=1, disabled=ui_locked_global))
        suite_stop_on_error = bool(st.checkbox("Stop on error", value=False, disabled=ui_locked_global))

        start_suite_clicked = st.button(
            "Start new suite",
            use_container_width=True,
            disabled=bool(ui_locked_global or any_running_now or any_suite_running),
            help="Suites refuse to start if any benchmark run is already running.",
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
    sid0 = str(active_suite_id or "").strip()
    if not sid0:
        try:
            rs = _running_suites()
            sid0 = str(rs[0]) if rs else ""
        except Exception:
            sid0 = ""

    if sid0:
        sp = _paths_for_suite(sid0)
        _stop_suite(sp)
    running = _running_runs()
    for rid0 in list(running or []):
        try:
            rp = _paths_for_run(str(rid0))
            _stop_run(rp)
        except Exception:
            continue
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
        st_prompt_file = str(st_state.get("prompt_file") or "").strip()

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
        if st_prompt_file:
            cmd.extend(["--prompt-file", str(st_prompt_file)])
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
        _set_active_run_id(str(run_id), rerun=False)
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
        prompt_file = str(st.session_state.get("prompt_file") or "").strip()
        if prompt_file:
            cmd.extend(["--prompt-file", str(prompt_file)])
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
    if not specs_in:
        st.sidebar.error("Select at least one model.")
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
            str(float(5.0)),
            "--max-parallel",
            str(int(st.session_state.get("suite_max_parallel") or 1)),
            "--max-per-provider",
            str(int(st.session_state.get("suite_max_per_provider") or 1)),
            "--spec-file",
            str(sp.specs_path),
        ]
        suite_prompt_file = str(st.session_state.get("suite_prompt_file") or "").strip()
        if suite_prompt_file:
            cmd.extend(["--prompt-file", str(suite_prompt_file)])
        if reset_first:
            cmd.append("--reset-first")
        if suite_stop_on_error:
            cmd.append("--stop-on-error")

        env = dict(os.environ)
        st.session_state["suite_proc"] = _start_suite(sp, cmd=cmd, cwd=_repo_root(), env=env)
        suite_proc = st.session_state.get("suite_proc")
        _set_active_run_id("", rerun=True)

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
                str(float(5.0)),
                "--max-parallel",
                str(int(st.session_state.get("suite_max_parallel") or 1)),
                "--max-per-provider",
                str(int(st.session_state.get("suite_max_per_provider") or 1)),
                "--spec-file",
                str(sp.specs_path),
                "--resume",
            ]
            suite_prompt_file = str(st.session_state.get("suite_prompt_file") or "").strip()
            if suite_prompt_file:
                cmd.extend(["--prompt-file", str(suite_prompt_file)])
            if reset_first:
                cmd.append("--reset-first")
            if suite_stop_on_error:
                cmd.append("--stop-on-error")
            env = dict(os.environ)
            st.session_state["suite_proc"] = _start_suite(sp, cmd=cmd, cwd=_repo_root(), env=env)
            suite_proc = st.session_state.get("suite_proc")
            _set_active_run_id("", rerun=True)


active_run_id = str(st.session_state.get("active_run_id") or "").strip()
active_suite_id = str(st.session_state.get("active_suite_id") or "").strip()


if (not active_run_id) and (not active_suite_id):
    st.info("Start a new run or select a saved run. You can also run preflight checks below.")
    _render_preflight_ui(challenge=str(challenge), base_url=base_url, api_timeout_s=float(api_timeout), player_id=str(player_id))
    raise SystemExit(0)

if (not active_run_id) and active_suite_id:
    sp = _paths_for_suite(active_suite_id)
    suite_pid = _read_pid(sp.pid_path)
    suite_alive = bool(suite_pid is not None and _pid_alive(int(suite_pid)))

    try:
        if (not suite_alive) and str(st.session_state.get("_suite_done_auto_attached") or "") != str(active_suite_id):
            manifest0 = _load_json(sp.manifest_path)
            run_ids0: List[str] = []
            if isinstance(manifest0, dict):
                rr0 = manifest0.get("runs")
                if isinstance(rr0, list):
                    for r in rr0:
                        if not isinstance(r, dict):
                            continue
                        rid = str(r.get("run_id") or "").strip()
                        if rid:
                            run_ids0.append(rid)
            run_ids0 = [r for r in run_ids0 if r]
            run_ids0 = list(dict.fromkeys(run_ids0))
            if run_ids0:
                last_rid = str(run_ids0[-1])
                _set_active_run_id(str(last_rid), rerun=True)
    except Exception:
        pass

    running = _running_runs()
    cols0 = st.columns(4)
    cols0[0].metric("Suite", str(active_suite_id))
    cols0[1].metric("Status", "Running" if suite_alive else "Idle/Done")
    cols0[2].metric("Running runs", str(int(len(running))))
    cols0[3].metric("Suite dir", str(sp.suite_dir))
    if running:
        tail = "" if len(running) <= 8 else f" (+{len(running) - 8} more)"
        st.caption("Running: " + ", ".join([str(x) for x in running[:8]]) + tail)

    try:
        manifest_live0 = _load_json(sp.manifest_path)
        run_ents0 = _suite_run_entries(manifest_live0)
        rid_label: Dict[str, str] = {}
        all_rids0: List[str] = []
        running_rids0: List[str] = []
        for ent in run_ents0:
            if not isinstance(ent, dict):
                continue
            rid = str(ent.get("run_id") or "").strip()
            if not rid:
                continue
            all_rids0.append(rid)
            provider = str(ent.get("provider") or "").strip()
            model = str(ent.get("model") or "").strip()
            replicate = str(ent.get("replicate") or "").strip()
            status0 = str(ent.get("status") or "").strip()
            pid0 = ent.get("pid")
            try:
                pid_i = int(pid0) if pid0 is not None else None
            except Exception:
                pid_i = None
            pid_alive = bool(pid_i is not None and _pid_alive(int(pid_i)))
            if status0 == "running" and pid_alive:
                running_rids0.append(rid)
            bits = [rid]
            meta_bits = []
            if provider:
                meta_bits.append(provider)
            if model:
                meta_bits.append(model)
            if replicate:
                meta_bits.append(f"rep={replicate}")
            if meta_bits:
                bits.append("[" + "/".join(meta_bits) + "]")
            if status0:
                bits.append(status0)
            rid_label[rid] = " ".join([b for b in bits if b]).strip()

        all_rids0 = [r for r in all_rids0 if r]
        all_rids0 = list(dict.fromkeys(all_rids0))
        running_rids0 = [r for r in running_rids0 if r]
        running_rids0 = list(dict.fromkeys(running_rids0))
        opts0 = [""] + running_rids0 + [r for r in all_rids0 if r not in set(running_rids0)]
        if len(opts0) > 1:
            picked_rid0 = st.selectbox(
                "Inspect a run",
                options=opts0,
                index=0,
                format_func=lambda x: rid_label.get(str(x), str(x)),
                key=f"suite_inspect_run_{active_suite_id}",
            )
            picked_rid0 = str(picked_rid0 or "").strip()
            if picked_rid0:
                _set_active_run_id(str(picked_rid0), rerun=True)
    except Exception:
        pass

    try:
        manifest_live = _load_json(sp.manifest_path)
        run_ents = _suite_run_entries(manifest_live)
        rows_live: List[Dict[str, Any]] = []
        for ent in run_ents:
            if not isinstance(ent, dict):
                continue
            rid = str(ent.get("run_id") or "").strip()
            if not rid:
                continue
            pid0 = ent.get("pid")
            try:
                pid_i = int(pid0) if pid0 is not None else None
            except Exception:
                pid_i = None
            pid_alive = bool(pid_i is not None and _pid_alive(int(pid_i)))
            status0 = str(ent.get("status") or "").strip()
            if status0 == "running" and (not pid_alive):
                status0 = "running (stale)"
            rows_live.append(
                {
                    "run_id": rid,
                    "provider": str(ent.get("provider") or ""),
                    "model": str(ent.get("model") or ""),
                    "replicate": str(ent.get("replicate") or ""),
                    "status": status0,
                    "pid": pid_i,
                    "pid_alive": bool(pid_alive),
                    "exit_code": ent.get("exit_code"),
                }
            )
        if rows_live:
            st.subheader("Suite runs")
            st.dataframe(rows_live, use_container_width=True)
    except Exception:
        pass

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

    st.subheader("Suite bundle (issues + story)")
    try:
        bundle_cache = st.session_state.get("suite_bundle_cache")
        if not isinstance(bundle_cache, dict):
            bundle_cache = {}
            st.session_state["suite_bundle_cache"] = bundle_cache

        bundle_include_issues = True
        bundle_include_story = True
        bundle_include_latest = True
        cba, cbb, cbc = st.columns(3)
        bundle_include_issues = bool(cba.checkbox("Include issues.json", value=True, key=f"suite_bundle_issues_only_{active_suite_id}"))
        bundle_include_story = bool(cbb.checkbox("Include story.md", value=True, key=f"suite_bundle_story_only_{active_suite_id}"))
        bundle_include_latest = bool(cbc.checkbox("Include latest summary/rationale", value=True, key=f"suite_bundle_latest_only_{active_suite_id}"))

        try:
            st0 = sp.manifest_path.stat() if sp.manifest_path.exists() else None
            manifest_sig = (float(getattr(st0, "st_mtime", 0.0) or 0.0), int(getattr(st0, "st_size", 0) or 0))
        except Exception:
            manifest_sig = (0.0, 0)

        cache_key = (
            str(active_suite_id),
            bool(bundle_include_issues),
            bool(bundle_include_story),
            bool(bundle_include_latest),
            float(manifest_sig[0]),
            int(manifest_sig[1]),
        )

        if st.button("Build suite bundle", use_container_width=True, key=f"suite_bundle_build_only_{active_suite_id}"):
            md = _suite_runs_bundle_markdown(
                str(active_suite_id),
                include_issues=bool(bundle_include_issues),
                include_saved_story=bool(bundle_include_story),
                include_latest_reason_fields=bool(bundle_include_latest),
            )
            bundle_cache[str(cache_key)] = md
            st.session_state["suite_bundle_cache"] = bundle_cache

        if (not suite_alive) and (not isinstance(bundle_cache.get(str(cache_key)), str)):
            md = _suite_runs_bundle_markdown(
                str(active_suite_id),
                include_issues=bool(bundle_include_issues),
                include_saved_story=bool(bundle_include_story),
                include_latest_reason_fields=bool(bundle_include_latest),
            )
            bundle_cache[str(cache_key)] = md
            st.session_state["suite_bundle_cache"] = bundle_cache

        md0 = bundle_cache.get(str(cache_key))
        if isinstance(md0, str) and md0.strip():
            st.download_button(
                "Download suite_bundle.md",
                data=md0.encode("utf-8", errors="replace"),
                file_name=f"{active_suite_id}_bundle.md",
                mime="text/markdown",
                key=f"download_suite_bundle_only_{active_suite_id}_{float(manifest_sig[0])}_{int(manifest_sig[1])}_{int(bundle_include_issues)}_{int(bundle_include_story)}_{int(bundle_include_latest)}",
                use_container_width=True,
            )
    except Exception as e:
        st.warning(str(e))

    manifest0 = _load_json(sp.manifest_path)
    run_ids: List[str] = []
    if isinstance(manifest0, dict):
        rr0 = manifest0.get("runs")
        if isinstance(rr0, list):
            for r in rr0:
                if not isinstance(r, dict):
                    continue
                rid = str(r.get("run_id") or "").strip()
                if rid:
                    run_ids.append(rid)
    run_ids = [r for r in run_ids if r]
    run_ids = list(dict.fromkeys(run_ids))

    if run_ids:
        st.subheader("Open a run from this suite")
        pick = st.selectbox("Run", options=[""] + run_ids, index=0)
        if str(pick or "").strip():
            if st.button("Open selected run", use_container_width=True):
                _set_active_run_id(str(pick).strip(), rerun=True)
    else:
        st.info("No run_ids found in suite_manifest.json yet.")
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


try:
    events_window
except Exception:
    events_window = 600
try:
    incremental_events
except Exception:
    incremental_events = True

if incremental_events:
    events = _read_events_incremental(paths.events_path, run_key=str(active_run_id), max_events=int(events_window))
else:
    events = _read_events(paths.events_path, max_events=int(events_window))
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
    waiting = bool(isinstance(pending, dict) and pending.get("waiting") is True)
    step = int(pending.get("step") or 0) if isinstance(pending, dict) else 0
    expecting = str(pending.get("expecting") or f"input_step_{step:06d}.json") if isinstance(pending, dict) else ""

    last_llm: Optional[Dict[str, Any]] = None
    for ev in reversed(events or []):
        if not isinstance(ev, dict):
            continue
        if str(ev.get("type") or "") == "llm":
            last_llm = ev
            break

    last_action = ""
    last_lrs = ""
    last_nsr = ""
    last_step = None
    if isinstance(last_llm, dict):
        act_obj = _parse_first_json(str(last_llm.get("text") or ""))
        act = act_obj.get("action") if isinstance(act_obj, dict) else None
        method = act_obj.get("method") if isinstance(act_obj, dict) else None
        path = act_obj.get("path") if isinstance(act_obj, dict) else None
        last_step = last_llm.get("step")
        last_action = " ".join([str(x or "").strip() for x in [act, method, path] if str(x or "").strip()]).strip()
        lrs0, nsr0 = _llm_reason_fields(last_llm)
        last_lrs = str(lrs0 or "").strip()
        last_nsr = str(nsr0 or "").strip()

    last_result_line = ""
    for ev in reversed(events or []):
        if not isinstance(ev, dict):
            continue
        t = str(ev.get("type") or "")
        if t == "tool_result":
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            st0 = payload.get("http_status")
            pth = str(ev.get("path") or payload.get("path") or "")
            err = ""
            rj = payload.get("response_json") if isinstance(payload.get("response_json"), dict) else {}
            if isinstance(rj, dict) and isinstance(rj.get("error"), str) and str(rj.get("error") or "").strip():
                err = str(rj.get("error") or "")
            last_result_line = f"TOOL_RESULT {pth} status={st0} {(_truncate(err, 120) if err else '')}".strip()
            break
        if t == "api":
            st0 = ev.get("http_status")
            pth = str(ev.get("path") or "")
            err = ""
            rj = ev.get("response_json") if isinstance(ev.get("response_json"), dict) else {}
            if isinstance(rj, dict) and isinstance(rj.get("error"), str) and str(rj.get("error") or "").strip():
                err = str(rj.get("error") or "")
            last_result_line = f"API {ev.get('method')} {pth} status={st0} {(_truncate(err, 120) if err else '')}".strip()
            break
        if t == "llm_error":
            last_result_line = f"LLM_ERROR {_truncate(ev.get('error'), 200)}".strip()
            break

    turn = "YOUR TURN" if waiting else ("RUNNING" if proc_alive else "IDLE/DONE")
    cols = st.columns(3)
    cols[0].metric("Turn", turn)
    cols[1].metric("Current step", str(step if waiting else (last_step if last_step is not None else "—")))
    cols[2].metric("Awaiting", expecting if waiting else "—")

    if waiting:
        st.success("The runner is paused and waiting for your instructions.")
    else:
        if proc_alive:
            st.info("The runner is still executing. You can submit instructions when it pauses for human input.")
        else:
            st.info("The runner is not currently waiting for human input.")

    if last_action or last_lrs or last_nsr or last_result_line:
        with st.expander("What’s happening now", expanded=True):
            if last_action:
                st.markdown(f"**Last agent action:** {last_action}")
            if last_lrs:
                st.markdown(f"**Last result summary:** {_truncate(last_lrs, 1200)}")
            if last_nsr:
                st.markdown(f"**Next step rationale:** {_truncate(last_nsr, 1200)}")
            if last_result_line:
                st.markdown(f"**Latest result:** {last_result_line}")

    if isinstance(pending, dict) and (pending.get("waiting") is True):
        with st.expander("Raw human-mode state", expanded=False):
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
    st.subheader("Story")

    sid_story = str(st.session_state.get("active_suite_id") or "").strip()
    if sid_story:
        sp_story = _paths_for_suite(sid_story)
        suite_pid_story = _read_pid(sp_story.pid_path)
        suite_alive_story = bool(suite_pid_story is not None and _pid_alive(int(suite_pid_story)))
        with st.expander("Suite bundle download", expanded=bool(not suite_alive_story)):
            try:
                bundle_cache = st.session_state.get("suite_bundle_cache")
                if not isinstance(bundle_cache, dict):
                    bundle_cache = {}
                    st.session_state["suite_bundle_cache"] = bundle_cache

                cb1, cb2, cb3 = st.columns(3)
                inc_issues = bool(cb1.checkbox("Include issues.json", value=True, key=f"suite_bundle_issues_story_{sid_story}"))
                inc_story = bool(cb2.checkbox("Include story.md", value=True, key=f"suite_bundle_story_story_{sid_story}"))
                inc_latest = bool(cb3.checkbox("Include latest summary/rationale", value=True, key=f"suite_bundle_latest_story_{sid_story}"))

                try:
                    st0 = sp_story.manifest_path.stat() if sp_story.manifest_path.exists() else None
                    manifest_sig = (float(getattr(st0, "st_mtime", 0.0) or 0.0), int(getattr(st0, "st_size", 0) or 0))
                except Exception:
                    manifest_sig = (0.0, 0)

                cache_key = (
                    str(sid_story),
                    bool(inc_issues),
                    bool(inc_story),
                    bool(inc_latest),
                    float(manifest_sig[0]),
                    int(manifest_sig[1]),
                )

                if st.button("Build suite bundle", use_container_width=True, key=f"suite_bundle_build_story_{sid_story}"):
                    md = _suite_runs_bundle_markdown(
                        str(sid_story),
                        include_issues=bool(inc_issues),
                        include_saved_story=bool(inc_story),
                        include_latest_reason_fields=bool(inc_latest),
                    )
                    bundle_cache[str(cache_key)] = md
                    st.session_state["suite_bundle_cache"] = bundle_cache

                if (not suite_alive_story) and (not isinstance(bundle_cache.get(str(cache_key)), str)):
                    md = _suite_runs_bundle_markdown(
                        str(sid_story),
                        include_issues=bool(inc_issues),
                        include_saved_story=bool(inc_story),
                        include_latest_reason_fields=bool(inc_latest),
                    )
                    bundle_cache[str(cache_key)] = md
                    st.session_state["suite_bundle_cache"] = bundle_cache

                md0 = bundle_cache.get(str(cache_key))
                if isinstance(md0, str) and md0.strip():
                    st.download_button(
                        "Download suite_bundle.md",
                        data=md0.encode("utf-8", errors="replace"),
                        file_name=f"{sid_story}_bundle.md",
                        mime="text/markdown",
                        key=f"download_suite_bundle_story_{sid_story}_{float(manifest_sig[0])}_{int(manifest_sig[1])}_{int(inc_issues)}_{int(inc_story)}_{int(inc_latest)}",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(str(e))

    saved_story = ""
    try:
        saved_story = _load_text(paths.story_path)
    except Exception:
        saved_story = ""

    ctx_summary, _ctx_hash = _read_context_summary_cached(paths, run_key=str(active_run_id))
    if ctx_summary.strip():
        with st.expander("Rolling summary (CONTEXT_SUMMARY_ENTRY)", expanded=True):
            st.write(ctx_summary)

    max_events_for_story = 220
    min_refresh_s = 45.0
    show_digest = False
    allow_auto = False
    story_source = "LLM step summaries (last_result_summary + next_step_rationale)"
    max_llm_steps_for_story = 80
    with st.expander("Story settings", expanded=False):
        max_events_for_story = int(
            st.number_input("Events window", min_value=50, max_value=800, value=220, step=10)
        )
        max_llm_steps_for_story = int(
            st.number_input("LLM steps window", min_value=10, max_value=400, value=80, step=10)
        )
        min_refresh_s = float(
            st.number_input("Min story refresh (s)", min_value=5.0, max_value=600.0, value=45.0, step=5.0)
        )
        allow_auto = bool(st.checkbox("Auto-update story (costs tokens)", value=False))
        story_source = str(
            st.selectbox(
                "Story input",
                options=[
                    "LLM step summaries (last_result_summary + next_step_rationale)",
                    "Latest outbound LLM prompt (llm_payloads)",
                    "LAB_NOTEBOOK",
                    "EVENT_DIGEST",
                ],
                index=0,
            )
        )
        show_digest = bool(st.checkbox("Show digest (debug)", value=False))

    story_options = ["Step cards", "Auto story summary"]
    if saved_story.strip():
        story_options = ["Saved story"] + story_options
    story_default_idx = 0
    if (not proc_alive) and saved_story.strip():
        story_default_idx = int(story_options.index("Saved story"))
    story_mode = st.radio(
        "View",
        options=story_options,
        index=int(story_default_idx),
        horizontal=True,
    )

    if story_mode == "Saved story":
        if saved_story.strip():
            st.caption("Loaded from story.md")
            st.markdown(saved_story)
        else:
            st.info("No saved story.md found for this run.")
    elif story_mode == "Step cards":
        llm_events = [ev for ev in (events or []) if isinstance(ev, dict) and str(ev.get("type") or "") == "llm"]
        if int(max_llm_steps_for_story) > 0:
            llm_events = llm_events[-int(max_llm_steps_for_story) :]

        expand_latest = bool(st.checkbox("Expand latest step", value=True))
        if not llm_events:
            st.info("No LLM steps yet.")
        else:
            last_ev2 = llm_events[-1]
            for ev in llm_events:
                step_i = ev.get("step")
                act_obj = _parse_first_json(str(ev.get("text") or ""))
                act = act_obj.get("action") if isinstance(act_obj, dict) else None
                method = act_obj.get("method") if isinstance(act_obj, dict) else None
                path = act_obj.get("path") if isinstance(act_obj, dict) else None
                hdr = " ".join([str(x or "").strip() for x in [f"Step {step_i}", act, method, path] if str(x or "").strip()])
                lrs, nsr = _llm_reason_fields(ev)
                expanded = bool(expand_latest and (ev is last_ev2))
                with st.expander(hdr or f"Step {step_i}", expanded=expanded):
                    if isinstance(lrs, str) and lrs.strip():
                        st.markdown("**Last result summary**")
                        st.write(lrs)
                    if isinstance(nsr, str) and nsr.strip():
                        st.markdown("**Next step rationale**")
                        st.write(nsr)
    elif story_mode != "Step cards":
        st.markdown("Auto story summary")

        notebook_txt = ""
        nb_hash = ""
        latest_prompt_txt, latest_prompt_hash = ("", "")
        if story_source == "LAB_NOTEBOOK":
            notebook_txt, nb_hash = _read_lab_notebook_cached(paths, run_key=str(active_run_id))
        elif story_source == "Latest outbound LLM prompt (llm_payloads)":
            latest_prompt_txt, latest_prompt_hash = _latest_llm_prompt_text_cached(paths, run_key=str(active_run_id))
        last_ev = events[-1] if isinstance(events, list) and events and isinstance(events[-1], dict) else {}

        report_mtime_story = 0.0
        try:
            if paths is not None and paths.report_path.exists():
                report_mtime_story = float(paths.report_path.stat().st_mtime)
        except Exception:
            report_mtime_story = 0.0

        fp = _json_compact(
            {
                "run": active_run_id,
                "proc_alive": bool(proc_alive),
                "latest_prompt_parser_v": int(_LATEST_LLM_PROMPT_CACHE_VERSION),
                "last_seq": last_ev.get("seq"),
                "last_type": last_ev.get("type"),
                "last_ts": last_ev.get("ts"),
                "story_source": str(story_source),
                "max_llm_steps_for_story": int(max_llm_steps_for_story),
                "report_mtime": float(report_mtime_story),
                "nb_hash": nb_hash,
                "latest_prompt_hash": str(latest_prompt_hash or ""),
                "max_events_for_story": int(max_events_for_story),
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
                        if story_source == "LLM step summaries (last_result_summary + next_step_rationale)":
                            reason = _llm_reason_digest(
                                events,
                                report,
                                max_steps=int(max_llm_steps_for_story),
                            )
                            full_txt = "LLM_STEP_SUMMARIES:\n" + str(reason or "")
                            fut = _story_executor().submit(
                                _openai_story_concise_from_llm_reasons,
                                api_key=key,
                                reason_log=full_txt,
                                run_id=str(active_run_id),
                                proc_alive=bool(proc_alive),
                            )
                        else:
                            if story_source == "Latest outbound LLM prompt (llm_payloads)" and latest_prompt_txt.strip():
                                full_txt = "LATEST_LLM_PROMPT:\n" + latest_prompt_txt
                            elif story_source == "LAB_NOTEBOOK" and notebook_txt.strip():
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
            if story_source == "LLM step summaries (last_result_summary + next_step_rationale)":
                dig = _llm_reason_digest(events, report, max_steps=int(max_llm_steps_for_story))
                st.text_area("LLM step summaries (debug)", value=dig, height=240)
            elif story_source == "Latest outbound LLM prompt (llm_payloads)" and latest_prompt_txt.strip():
                st.text_area("LATEST_LLM_PROMPT (debug)", value=latest_prompt_txt, height=240)
            elif story_source == "LAB_NOTEBOOK":
                st.text_area("LAB_NOTEBOOK (debug)", value=notebook_txt, height=240)
            else:
                dig = _event_digest(events, max_events=int(max_events_for_story))
                st.text_area("EVENT_DIGEST (debug)", value=dig, height=240)

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

    fast_live_view = bool(st.checkbox("Fast live view (recommended)", value=True))

    with st.expander("Performance", expanded=False):
        lazy_api = bool(st.checkbox("Lazy-load API/tool JSON (recommended)", value=True))
        show_full_api_default = bool(st.checkbox("Show full API response_json by default", value=False, disabled=lazy_api))
        show_full_tool_default = bool(st.checkbox("Show full TOOL_RESULT by default", value=False, disabled=lazy_api))

    shown = [ev for ev in events if str(ev.get("type")) in set(show_types)]
    shown = shown[-max_show:]

    def _event_title(ev: Dict[str, Any]) -> str:
        t = str(ev.get("type") or "")
        seq = ev.get("seq")
        title = f"#{seq} {t}"
        if t == "api":
            title = f"#{seq} API {ev.get('method')} {ev.get('path')} ({ev.get('http_status')})"
        if t == "llm":
            title = f"#{seq} LLM step={ev.get('step')}"
        return title

    def _render_event_body(ev: Dict[str, Any], *, lazy_api: bool, show_full_api_default: bool, show_full_tool_default: bool) -> None:
        t = str(ev.get("type") or "")
        seq = ev.get("seq")
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
            return

        if t in ("api", "tool_result"):
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
                return

            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else None
            if payload is None:
                st.json(ev)
                return
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
            rj2 = payload.get("api_response_json_summary") if isinstance(payload.get("api_response_json_summary"), dict) else None
            if rj2 is not None:
                st.markdown("Raw API response summary (debug)")
                st.json(rj2)
            rt = payload.get("response_text")
            if isinstance(rt, str) and rt.strip():
                st.code(rt, language="")
            return

        st.json(ev)

    if fast_live_view:
        if not shown:
            st.info("No events in the current window.")
        else:
            titles = [_event_title(ev) for ev in shown]
            pick = st.selectbox("Event", options=list(range(len(shown))), format_func=lambda i: titles[int(i)], index=len(shown) - 1)
            try:
                ev0 = shown[int(pick)]
            except Exception:
                ev0 = shown[-1]
            st.caption(_event_title(ev0))
            _render_event_body(ev0, lazy_api=lazy_api, show_full_api_default=show_full_api_default, show_full_tool_default=show_full_tool_default)
    else:
        expanded_llm = False

        for ev in reversed(shown):
            t = str(ev.get("type") or "")
            title = _event_title(ev)
            expand = False
            if auto_expand_latest_llm and (not expanded_llm) and t == "llm":
                expand = True
                expanded_llm = True
            with st.expander(title, expanded=expand):
                _render_event_body(ev, lazy_api=lazy_api, show_full_api_default=show_full_api_default, show_full_tool_default=show_full_tool_default)

with tabs[3]:
    st.subheader("Issues / Errors")
    st.caption("Heuristic detector: flags anything that looks off even if the run continues (errors, retries, truncation signals, malformed LLM outputs, etc.).")

    issues_from_disk: Optional[List[Dict[str, Any]]] = None
    if paths is not None:
        try:
            obj0 = _load_json_any(paths.issues_path)
            if isinstance(obj0, list) and all(isinstance(x, dict) for x in obj0):
                issues_from_disk = list(obj0)
        except Exception:
            issues_from_disk = None

    issues_cache = st.session_state.get("issues_cache")
    if not isinstance(issues_cache, dict):
        issues_cache = {}
        st.session_state["issues_cache"] = issues_cache

    last_seq0: Optional[int] = None
    try:
        if isinstance(events, list) and events and isinstance(events[-1], dict):
            last_seq0 = int(events[-1].get("seq") or 0)
    except Exception:
        last_seq0 = None

    def _mtime(p: Path) -> float:
        try:
            if p.exists() and p.is_file():
                return float(p.stat().st_mtime)
        except Exception:
            return 0.0
        return 0.0

    cache_key = (
        str(active_run_id),
        int(last_seq0 or 0),
        float(_mtime(paths.stderr_path)),
        float(_mtime(paths.stdout_path)),
    )
    cached_ent = issues_cache.get(str(active_run_id))
    if isinstance(issues_from_disk, list):
        issues = issues_from_disk
        st.caption("Loaded from issues.json")
    elif isinstance(cached_ent, dict) and cached_ent.get("key") == cache_key and isinstance(cached_ent.get("issues"), list):
        issues = cached_ent.get("issues")
    else:
        issues = _detect_issues(events, paths)
        issues_cache[str(active_run_id)] = {"key": cache_key, "issues": issues}
        st.session_state["issues_cache"] = issues_cache

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
            key=f"download_issues_{active_run_id}",
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

    fast_csv_view = bool(st.checkbox("Fast CSV view (recommended)", value=True))

    file_bytes_cache = st.session_state.get("file_bytes_cache")
    if not isinstance(file_bytes_cache, dict):
        file_bytes_cache = {}
        st.session_state["file_bytes_cache"] = file_bytes_cache

    files_cache = st.session_state.get("files_cache")
    if not isinstance(files_cache, dict):
        files_cache = {}
        st.session_state["files_cache"] = files_cache

    last_seq1: Optional[int] = None
    try:
        if isinstance(events, list) and events and isinstance(events[-1], dict):
            last_seq1 = int(events[-1].get("seq") or 0)
    except Exception:
        last_seq1 = None

    report_mtime = 0.0
    files_mtime = 0.0
    try:
        if paths is not None and paths.report_path.exists():
            report_mtime = float(paths.report_path.stat().st_mtime)
    except Exception:
        report_mtime = 0.0
    try:
        if paths is not None and paths.files_dir.exists():
            files_mtime = float(paths.files_dir.stat().st_mtime)
    except Exception:
        files_mtime = 0.0

    cached_files = files_cache.get(str(active_run_id))
    if (
        isinstance(cached_files, dict)
        and cached_files.get("last_seq") == int(last_seq1 or 0)
        and float(cached_files.get("report_mtime") or 0.0) == float(report_mtime)
        and float(cached_files.get("files_mtime") or 0.0) == float(files_mtime)
        and isinstance(cached_files.get("arts"), list)
        and isinstance(cached_files.get("omics"), list)
    ):
        arts = cached_files.get("arts")
        omics = cached_files.get("omics")
    else:
        arts = _extract_files(events)
        if paths is not None:
            arts = list(arts or []) + _scan_local_csv_files(paths.files_dir)
        omics = _extract_omics_csv_refs(events, base_url=str(base_url))
        omics = list(omics or []) + _extract_omics_csv_refs_from_report(report, base_url=str(base_url))
        files_cache[str(active_run_id)] = {
            "last_seq": int(last_seq1 or 0),
            "report_mtime": float(report_mtime),
            "files_mtime": float(files_mtime),
            "arts": arts,
            "omics": omics,
        }
        st.session_state["files_cache"] = files_cache

    local_csv = [a for a in (arts or []) if isinstance(a, dict) and isinstance(a.get("path"), str) and str(a.get("path") or "").strip()]
    local_csv = local_csv[-500:]
    omics_csv = [a for a in (omics or []) if isinstance(a, dict) and isinstance(a.get("url"), str) and str(a.get("url") or "").strip()]
    omics_csv = omics_csv[-800:]

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for a in local_csv:
        p = str(a.get("path") or "").strip()
        k = f"local::{p}"
        if k in seen:
            continue
        seen.add(k)
        ent = dict(a)
        ent["source"] = "local"
        ent["id"] = k
        items.append(ent)
    for a in omics_csv:
        rid = str(a.get("omics_run_id") or "").strip()
        name = str(a.get("name") or "").strip()
        k = f"omics::{rid}::{name}"
        if k in seen:
            continue
        seen.add(k)
        ent = dict(a)
        ent["id"] = k
        items.append(ent)

    if not items:
        st.info("No CSV files detected yet.")
    else:
        items = items[-1000:]

        def _lab(i: int) -> str:
            try:
                a0 = items[int(i)]
            except Exception:
                a0 = items[-1]
            src = str(a0.get("source") or "")
            ev_seq = a0.get("event_seq")
            if src == "omics":
                rid = str(a0.get("omics_run_id") or "")
                name = str(a0.get("name") or "")
                return f"#{ev_seq} OMICS {rid} {name}"
            p = str(a0.get("path") or "")
            return f"#{ev_seq} LOCAL {p}"

        pick = st.selectbox("File", options=list(range(len(items))), index=len(items) - 1, format_func=_lab) if fast_csv_view else None

        def _render_one(a: Dict[str, Any]) -> None:
            src = str(a.get("source") or "")
            if src == "omics":
                rid = str(a.get("omics_run_id") or "")
                name = str(a.get("name") or "")
                url = str(a.get("url") or "")
                st.write({"source": "omics", "from_api": a.get("api_path"), "omics_run_id": rid, "name": name, "bytes": a.get("bytes"), "url": url})

                k = f"omics::{rid}::{name}::{int(a.get('bytes') or 0)}"
                cached = file_bytes_cache.get(k)
                have = isinstance(cached, (bytes, bytearray))
                if (not have) and st.button("Load for preview/download", key=f"load_omics_csv_{k}"):
                    try:
                        cached = _http_get_bytes(url)
                        file_bytes_cache[k] = cached
                        st.session_state["file_bytes_cache"] = file_bytes_cache
                        have = True
                    except Exception as e:
                        st.warning(str(e))

                if have:
                    txt = bytes(cached).decode("utf-8", errors="replace")
                    prev = "\n".join(txt.splitlines()[:80])
                    if prev.strip():
                        st.code(prev, language="csv")
                        rows = _csv_to_rows(prev)
                        if rows:
                            st.dataframe(rows, use_container_width=True)
                    try:
                        fn = f"{rid}_{Path(name).name}"
                    except Exception:
                        fn = Path(name).name if name else "omics.csv"
                    st.download_button(
                        "Download CSV",
                        data=bytes(cached),
                        file_name=fn,
                        mime="text/csv",
                        key=f"download_omics_csv_{k}",
                        use_container_width=False,
                    )
                return

            p = str(a.get("path") or "")
            st.write({"source": "local", "from_api": a.get("path"), "bytes": a.get("bytes"), "file": p})
            prev = str(a.get("preview") or "")
            if prev:
                st.code(prev, language="csv")
                rows = _csv_to_rows(prev)
                if rows:
                    st.dataframe(rows, use_container_width=True)

            fp = Path(p)
            if not (fp.exists() and fp.is_file()):
                return
            try:
                st0 = fp.stat()
                k = f"local::{str(fp)}::{int(st0.st_mtime)}::{int(st0.st_size)}"
            except Exception:
                k = f"local::{str(fp)}"
            cached = file_bytes_cache.get(k)
            have = isinstance(cached, (bytes, bytearray))
            if (not have) and st.button("Load for preview/download", key=f"load_local_csv_{k}"):
                try:
                    cached = fp.read_bytes()
                    file_bytes_cache[k] = cached
                    st.session_state["file_bytes_cache"] = file_bytes_cache
                    have = True
                except Exception as e:
                    st.warning(str(e))
            if have:
                if not prev:
                    try:
                        txt = bytes(cached).decode("utf-8", errors="replace")
                        prev2 = "\n".join(txt.splitlines()[:80])
                        if prev2.strip():
                            st.code(prev2, language="csv")
                            rows = _csv_to_rows(prev2)
                            if rows:
                                st.dataframe(rows, use_container_width=True)
                    except Exception:
                        pass
                st.download_button(
                    "Download CSV",
                    data=bytes(cached),
                    file_name=fp.name,
                    mime="text/csv",
                    key=f"download_local_csv_{k}",
                    use_container_width=False,
                )

        if fast_csv_view:
            try:
                a = items[int(pick)]
            except Exception:
                a = items[-1]
            _render_one(a)
        else:
            for a in reversed(items[-200:]):
                ev_seq = a.get("event_seq")
                if str(a.get("source") or "") == "omics":
                    label = f"#{ev_seq} OMICS {a.get('omics_run_id')} {a.get('name')}"
                else:
                    label = f"#{ev_seq} LOCAL {a.get('path')}"
                with st.expander(label, expanded=False):
                    _render_one(a)

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
            key=f"download_report_{active_run_id}",
        )

with tabs[8]:
    st.subheader("Runner logs")
    st.caption(str(paths.run_dir))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("stdout")
        st.code(_tail_text_cached(paths.stdout_path), language="")
    with col2:
        st.markdown("stderr")
        st.code(_tail_text_cached(paths.stderr_path), language="")


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
                _set_active_run_id(str(running_run_id), rerun=True)

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

        st.subheader("Suite bundle (issues + story)")
        try:
            bundle_cache = st.session_state.get("suite_bundle_cache")
            if not isinstance(bundle_cache, dict):
                bundle_cache = {}
                st.session_state["suite_bundle_cache"] = bundle_cache

            cb1, cb2, cb3 = st.columns(3)
            inc_issues = bool(cb1.checkbox("Include issues.json", value=True, key=f"suite_bundle_issues_tab_{sid}"))
            inc_story = bool(cb2.checkbox("Include story.md", value=True, key=f"suite_bundle_story_tab_{sid}"))
            inc_latest = bool(cb3.checkbox("Include latest summary/rationale", value=True, key=f"suite_bundle_latest_tab_{sid}"))

            try:
                st0 = sp.manifest_path.stat() if sp.manifest_path.exists() else None
                manifest_sig = (float(getattr(st0, "st_mtime", 0.0) or 0.0), int(getattr(st0, "st_size", 0) or 0))
            except Exception:
                manifest_sig = (0.0, 0)

            cache_key = (
                str(sid),
                bool(inc_issues),
                bool(inc_story),
                bool(inc_latest),
                float(manifest_sig[0]),
                int(manifest_sig[1]),
            )

            if st.button("Build suite bundle", use_container_width=True, key=f"suite_bundle_build_tab_{sid}"):
                md = _suite_runs_bundle_markdown(
                    str(sid),
                    include_issues=bool(inc_issues),
                    include_saved_story=bool(inc_story),
                    include_latest_reason_fields=bool(inc_latest),
                )
                bundle_cache[str(cache_key)] = md
                st.session_state["suite_bundle_cache"] = bundle_cache

            if (not suite_alive) and (not isinstance(bundle_cache.get(str(cache_key)), str)):
                md = _suite_runs_bundle_markdown(
                    str(sid),
                    include_issues=bool(inc_issues),
                    include_saved_story=bool(inc_story),
                    include_latest_reason_fields=bool(inc_latest),
                )
                bundle_cache[str(cache_key)] = md
                st.session_state["suite_bundle_cache"] = bundle_cache

            md0 = bundle_cache.get(str(cache_key))
            if isinstance(md0, str) and md0.strip():
                st.download_button(
                    "Download suite_bundle.md",
                    data=md0.encode("utf-8", errors="replace"),
                    file_name=f"{sid}_bundle.md",
                    mime="text/markdown",
                    key=f"download_suite_bundle_tab_{sid}_{float(manifest_sig[0])}_{int(manifest_sig[1])}_{int(inc_issues)}_{int(inc_story)}_{int(inc_latest)}",
                    use_container_width=True,
                )
        except Exception as e:
            st.warning(str(e))

        run_ids2: List[str] = []
        if isinstance(manifest0, dict):
            rr2 = manifest0.get("runs")
            if isinstance(rr2, list):
                for r in rr2:
                    if not isinstance(r, dict):
                        continue
                    rid = str(r.get("run_id") or "").strip()
                    if rid:
                        run_ids2.append(rid)
        run_ids2 = [r for r in run_ids2 if r]
        run_ids2 = list(dict.fromkeys(run_ids2))
        if run_ids2:
            st.subheader("Open a run")
            pick2 = st.selectbox("Run from this suite", options=[""] + run_ids2, index=0, key=f"suite_run_pick_{sid}")
            if str(pick2 or "").strip():
                if st.button("Open", use_container_width=True, key=f"suite_open_run_{sid}"):
                    _set_active_run_id(str(pick2).strip(), rerun=True)

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
                    key=f"download_suite_aggregate_{sid}",
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
                    key=f"download_suite_summary_{sid}",
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
