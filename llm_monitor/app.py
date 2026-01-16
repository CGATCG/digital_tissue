import json
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


def _runs_root() -> Path:
    return _repo_root() / "runs" / "llm_bench"


def _new_run_id() -> str:
    t = int(time.time())
    r = random.randint(1000, 9999)
    return f"run_{t}_{r}"


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
    hdr = [str(h) for h in hdr[:max_cols]]
    out: List[Dict[str, Any]] = []
    for r in data[:max_rows]:
        ent: Dict[str, Any] = {}
        for i, h in enumerate(hdr):
            ent[h] = r[i] if i < len(r) else ""
        out.append(ent)
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
        if read_all:
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
            raw = _tail_text(path, max_bytes=400_000)
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            if max_events is None:
                for ln in lines:
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict):
                            events.append(obj)
                    except Exception:
                        continue
            else:
                for ln in lines[-int(max_events) :]:
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


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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
            s = rj.get("score_lifedays_per_usd")
            score = float(s) if s is not None else None
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


def _run_preflight_checks(
    *,
    challenge: str,
    base_url: str,
    preflight_player_id: str,
    ticks: int,
    replicates: int,
    api_timeout_s: float,
    reset_player: bool,
    run_claim_cure: bool,
    run_protein_screen: bool,
    run_omics_analyze: bool,
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

    if reset_player:
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
    out.append(_preflight_call(base_url=base_url, llm_path="/api/tests/disease/models", method="GET", timeout_s=api_timeout_s, omics_state=omics_state))

    prot_resp = _preflight_call(
        base_url=base_url,
        llm_path="/api/tests/disease/proteins",
        method="GET",
        query={"model": "cell_culture_disease"},
        timeout_s=api_timeout_s,
        omics_state=omics_state,
    )
    out.append(prot_resp)
    prot_list = []
    try:
        rj = prot_resp.get("tool_result", {}).get("response_json")
        if isinstance(rj, dict) and isinstance(rj.get("proteins"), list):
            prot_list = [str(x) for x in rj.get("proteins") if str(x).strip()]
    except Exception:
        prot_list = []
    example_iv = []
    if prot_list:
        example_iv = [{"layer": prot_list[0], "direction": "down", "dose": 1}]

    bulk_sets = _preflight_call(base_url=base_url, llm_path="/api/bulk_omics/sets", method="GET", timeout_s=api_timeout_s, omics_state=omics_state)
    out.append(bulk_sets)
    omics_set = "rna/Bulk RNAseq"
    try:
        rj = bulk_sets.get("tool_result", {}).get("response_json")
        if isinstance(rj, dict) and isinstance(rj.get("sets"), list) and rj.get("sets"):
            omics_set = str(rj.get("sets")[0])
    except Exception:
        pass

    gene_sets = _preflight_call(base_url=base_url, llm_path="/api/spatial_tx/gene_sets", method="GET", timeout_s=api_timeout_s, omics_state=omics_state)
    out.append(gene_sets)
    gene_set = "spatial transcriptomics"
    try:
        rj = gene_sets.get("tool_result", {}).get("response_json")
        if isinstance(rj, dict) and isinstance(rj.get("gene_sets"), list) and rj.get("gene_sets"):
            gene_set = str(rj.get("gene_sets")[0])
    except Exception:
        pass

    for exp in ("bulk_omics", "spatial_tx", "characterization"):
        body: Dict[str, Any] = {
            "player_id": pid,
            "experiment": exp,
            "model": "cell_culture_disease",
            "ticks": int(ticks_i),
            "replicates": int(reps_i),
            "interventions": list(example_iv),
        }
        if exp == "bulk_omics":
            body["omics_set"] = str(omics_set)
        if exp == "spatial_tx":
            body["gene_set"] = str(gene_set)
        out.append(
            _preflight_call(
                base_url=base_url,
                llm_path="/api/tests/disease/estimate_cost",
                method="POST",
                body=body,
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
                "model": "cell_culture_disease",
                "ticks": int(ticks_i),
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
                "model": "cell_culture_disease",
                "ticks": int(ticks_i),
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
                "model": "cell_culture_disease",
                "ticks": int(ticks_i),
                "replicates": int(reps_i),
                "gene_set": str(gene_set),
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

    if run_protein_screen:
        out.append(
            _preflight_call(
                base_url=base_url,
                llm_path="/api/tests/disease/estimate_cost",
                method="POST",
                body={
                    "player_id": pid,
                    "experiment": "protein_screen",
                    "model": "cell_culture_disease",
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
                llm_path="/api/tests/disease/protein_screen",
                method="POST",
                body={
                    "player_id": pid,
                    "model": "cell_culture_disease",
                    "ticks": int(ticks_i),
                    "replicates": 1,
                    "interventions": list(example_iv),
                },
                timeout_s=api_timeout_s,
                omics_state=omics_state,
            )
        )

    if run_claim_cure:
        out.append(
            _preflight_call(
                base_url=base_url,
                llm_path="/api/tests/disease/estimate_cost",
                method="POST",
                body={
                    "player_id": pid,
                    "experiment": "claim_cure",
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

    if run_omics_analyze:
        try:
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
            file_ids = [x for x in file_ids if x]
            if file_ids:
                out.append(
                    _preflight_call(
                        base_url=base_url,
                        llm_path="/api/omics/analyze",
                        method="POST",
                        body={
                            "player_id": pid,
                            "file_ids": file_ids,
                            "instructions": "Briefly describe these file(s) (rows/cols, key columns) and suggest one useful comparison.",
                        },
                        timeout_s=api_timeout_s,
                        omics_state=omics_state,
                    )
                )
        except Exception:
            pass

    return out


def _render_preflight_ui(*, challenge: str, base_url: str, api_timeout_s: float, player_id: str) -> None:
    st.subheader("Preflight: test API endpoints (LLM-facing TOOL_RESULT)")
    st.caption("Runs quick calls using a separate player_id and shows exactly what the LLM would receive as TOOL_RESULT. Does not start the benchmark run.")

    default_pid = (str(player_id or "").strip() or "preflight") + "_preflight"
    with st.expander("Preflight settings", expanded=True):
        pid = st.text_input("Preflight player_id", value=default_pid, key="preflight_player_id")
        ticks = int(st.number_input("ticks", min_value=1, max_value=50, value=5, step=1, key="preflight_ticks"))
        reps = int(st.number_input("replicates", min_value=1, max_value=5, value=1, step=1, key="preflight_reps"))
        reset_pid = bool(st.checkbox("Reset preflight player state first", value=True, key="preflight_reset_pid"))
        run_claim = bool(st.checkbox("Run claim_cure (can take time + spend money)", value=False, key="preflight_run_claim"))
        run_screen = bool(st.checkbox("Run protein_screen (slow/expensive)", value=False, key="preflight_run_screen"))
        run_analyze = bool(st.checkbox("Run /api/omics/analyze (may trigger OpenAI calls on server)", value=False, key="preflight_run_analyze"))

    run_clicked = st.button("Run preflight checks", use_container_width=True, key="preflight_run_btn")
    if run_clicked:
        fp = _json_compact({"base_url": base_url, "pid": pid, "ticks": ticks, "reps": reps, "reset": reset_pid, "claim": run_claim, "screen": run_screen, "analyze": run_analyze})
        with st.spinner("Running preflight calls..."):
            try:
                rows = _run_preflight_checks(
                    challenge=str(challenge),
                    base_url=base_url,
                    preflight_player_id=pid,
                    ticks=ticks,
                    replicates=reps,
                    api_timeout_s=api_timeout_s,
                    reset_player=reset_pid,
                    run_claim_cure=run_claim,
                    run_protein_screen=run_screen,
                    run_omics_analyze=run_analyze,
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

    st.download_button(
        "Download preflight JSON",
        data=json.dumps(rows, indent=2).encode("utf-8"),
        file_name="preflight_results.json",
        mime="application/json",
        use_container_width=False,
    )

    for i, ent in enumerate(rows):
        if not isinstance(ent, dict):
            continue
        llm_path = ent.get("llm_path")
        method = ent.get("method")
        st0 = ent.get("http_status")
        sec = ent.get("seconds")
        title = f"{i+1}. {method} {llm_path} (status={st0} sec={sec})"
        with st.expander(title, expanded=False):
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
    st.session_state.setdefault("openai_api_key", "")
    st.session_state.setdefault("anthropic_api_key", "")
    st.session_state.setdefault("story_cache", {})


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

st.title("LLM Benchmark Monitor")

openai_key_current = ""
anthropic_key_current = ""

with st.sidebar:
    st.header("Run")
    base_url = st.text_input("Runtime base URL", value="http://127.0.0.1:8000")
    challenge = st.selectbox(
        "Challenge",
        options=["cancer", "hereditary_disease", "aging"],
        index=0,
        key="challenge",
    )
    provider = st.selectbox(
        "Provider",
        options=["openai", "claude"],
        index=0,
        key="provider",
    )

    if provider == "anthropic":
        provider = "claude"

    model = ""
    if provider == "openai":
        st.subheader("OpenAI")
        openai_key_input = st.text_input(
            "OpenAI API key",
            value=str(st.session_state.get("openai_api_key") or ""),
            type="password",
            help="Kept only in Streamlit session memory and passed to the runner as OPENAI_API_KEY.",
            key="openai_api_key_input",
        )
        openai_key_current = str(openai_key_input or "")
        remember_openai_key = bool(st.checkbox("Remember key for this session", value=True))
        if remember_openai_key:
            st.session_state["openai_api_key"] = str(openai_key_current or "")
        else:
            st.session_state["openai_api_key"] = ""

        openai_models = [
            "gpt-5.2",
        ]
        chosen = st.selectbox("Model", options=openai_models, index=0, key="openai_model_choice")
        model = str(chosen)
    elif provider in ("anthropic", "claude"):
        st.subheader("Anthropic")
        anthropic_key_input = st.text_input(
            "Anthropic API key",
            value=str(st.session_state.get("anthropic_api_key") or ""),
            type="password",
            help="Kept only in Streamlit session memory and passed to the runner as ANTHROPIC_API_KEY.",
            key="anthropic_api_key_input",
        )
        anthropic_key_current = str(anthropic_key_input or "")
        remember_anthropic_key = bool(st.checkbox("Remember key for this session", value=True, key="remember_anthropic_key"))
        if remember_anthropic_key:
            st.session_state["anthropic_api_key"] = str(anthropic_key_current or "")
        else:
            st.session_state["anthropic_api_key"] = ""

        model = st.text_input("Model", value=str(st.session_state.get("model") or ""), key="model")
    else:
        model = st.text_input("Model", value="", key="model")

    st.subheader("Limits")
    max_steps = int(st.number_input("Max steps", min_value=1, max_value=400, value=40, step=1))
    temperature = float(st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1))
    max_tokens = int(st.number_input("Max tokens", min_value=32, max_value=8000, value=700, step=50))

    st.subheader("Timeouts")
    api_timeout = float(st.number_input("API timeout (s)", min_value=5.0, max_value=3600.0, value=900.0, step=30.0))
    llm_timeout = float(st.number_input("LLM timeout (s)", min_value=5.0, max_value=900.0, value=120.0, step=5.0))

    st.subheader("Game")
    player_id = st.text_input("Player ID (optional)", value="")
    player_id_for_reset = player_id.strip() or "(default)"
    reset_first = False
    with st.expander("Danger zone", expanded=False):
        arm_reset = bool(
            st.checkbox(
                f"I understand: this will wipe the current game state for player_id='{player_id_for_reset}'",
                value=False,
                key="arm_reset",
            )
        )
        if arm_reset:
            reset_first = bool(
                st.checkbox(
                    "Reset game state before starting a new run",
                    value=False,
                    key="reset_first",
                )
            )

    st.subheader("Refresh")
    auto_refresh = bool(st.checkbox("Auto refresh", value=True))
    refresh_s = float(st.number_input("Refresh interval (s)", min_value=0.2, max_value=10.0, value=1.0, step=0.2))
    manual_refresh_clicked = st.button("Refresh now", use_container_width=True)

    st.subheader("Saved runs")
    runs = _list_runs()
    sel_run = st.selectbox("Select run", options=[""] + runs, index=0)
    if sel_run:
        st.session_state["active_run_id"] = sel_run

    col_a, col_b, col_c = st.columns(3)
    start_clicked = col_a.button("Start new", use_container_width=True)
    stop_clicked = col_b.button("Stop", use_container_width=True)
    resume_clicked = col_c.button("Resume", use_container_width=True)


active_run_id = str(st.session_state.get("active_run_id") or "").strip()
proc = st.session_state.get("proc")

if manual_refresh_clicked:
    st.rerun()

paths = _paths_for_run(active_run_id) if active_run_id else None

if stop_clicked:
    if paths is not None:
        _stop_run(paths)
    _stop_proc(proc)
    st.session_state["proc"] = None

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
        if st_provider == "openai":
            key_present = bool(str(openai_key_current or "").strip() or str(st.session_state.get("openai_api_key") or "").strip() or str(os.environ.get("OPENAI_API_KEY") or "").strip())
            if not key_present:
                st.sidebar.error("OpenAI API key is required to resume this run (set OPENAI_API_KEY or paste it above).")
                st.stop()
        if st_provider in ("anthropic", "claude"):
            key_present = bool(
                str(anthropic_key_current or "").strip()
                or str(st.session_state.get("anthropic_api_key") or "").strip()
                or str(os.environ.get("ANTHROPIC_API_KEY") or "").strip()
            )
            if not key_present:
                st.sidebar.error("Anthropic API key is required to resume this run (set ANTHROPIC_API_KEY or paste it above).")
                st.stop()

        _stop_run(paths)
        _stop_proc(proc)
        env = dict(os.environ)
        if st_provider == "openai":
            k = str(openai_key_current or "").strip() or str(st.session_state.get("openai_api_key") or "").strip()
            if k:
                env["OPENAI_API_KEY"] = k
        if st_provider in ("anthropic", "claude"):
            k = str(anthropic_key_current or "").strip() or str(st.session_state.get("anthropic_api_key") or "").strip()
            if k:
                env["ANTHROPIC_API_KEY"] = k
        cmd = [
            sys.executable,
            "trials/run_llm_benchmark.py",
            "--base-url",
            str(st_base_url),
            "--provider",
            str(st_provider),
            "--model",
            str(st_model),
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
        ]
        st.session_state["proc"] = _start_run(paths, cmd=cmd, cwd=_repo_root(), env=env)
        proc = st.session_state.get("proc")
        try:
            _write_pid(paths.pid_path, int(getattr(proc, "pid")))
        except Exception:
            pass

if start_clicked:
    if not model.strip():
        st.sidebar.error("Model is required.")
    else:
        if provider == "openai":
            key_present = bool(str(openai_key_current or "").strip() or str(os.environ.get("OPENAI_API_KEY") or "").strip())
            if not key_present:
                st.sidebar.error("OpenAI API key is required (set OPENAI_API_KEY or paste it above).")
                st.stop()
        if provider in ("anthropic", "claude"):
            key_present = bool(str(anthropic_key_current or "").strip() or str(os.environ.get("ANTHROPIC_API_KEY") or "").strip())
            if not key_present:
                st.sidebar.error("Anthropic API key is required (set ANTHROPIC_API_KEY or paste it above).")
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
            "--base-url",
            str(base_url),
            "--provider",
            str(provider),
            "--model",
            str(model),
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
        ]
        if reset_first:
            cmd.append("--reset-first")
        if player_id.strip():
            cmd.extend(["--player-id", player_id.strip()])

        env = dict(os.environ)
        if provider == "openai":
            k = str(openai_key_current or "").strip()
            if k:
                env["OPENAI_API_KEY"] = k
        if provider in ("anthropic", "claude"):
            k = str(anthropic_key_current or "").strip()
            if k:
                env["ANTHROPIC_API_KEY"] = k
        st.session_state["proc"] = _start_run(paths, cmd=cmd, cwd=_repo_root(), env=env)
        proc = st.session_state.get("proc")
        try:
            _write_pid(paths.pid_path, int(getattr(proc, "pid")))
        except Exception:
            pass


if not active_run_id:
    st.info("Start a new run or select a saved run. You can also run preflight checks below.")
    _render_preflight_ui(challenge=str(challenge), base_url=base_url, api_timeout_s=float(api_timeout), player_id=str(player_id))
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
best_score, best_extra_days, best_score_seq = _best_claim_cure_score(events, challenge=str(challenge))

header_cols = st.columns(6)
header_cols[0].metric("Run", active_run_id)
header_cols[1].metric("Status", "Running" if proc_alive else "Idle/Done")
header_cols[2].metric("Best score", "—" if best_score is None else f"{best_score:.6f}")
header_cols[3].metric("Best extra days", "—" if best_extra_days is None else f"{best_extra_days:.2f}")
header_cols[4].metric("Best |delta|", "—" if best_delta is None else f"{abs(best_delta):.3f}")
header_cols[5].metric("Best delta", "—" if best_delta is None else f"{best_delta:.3f}")

if money_usd is not None:
    st.caption(f"Money spent: ${money_usd:.2f} ({money_cents} cents)")
elif end_metrics and end_metrics.get("money_spent_usd") is not None:
    st.caption(f"Money spent: ${float(end_metrics.get('money_spent_usd')):.2f}")

if str(challenge or "").strip().lower() != "aging":
    if best_win is True:
        st.success("Best observed claim_cure: WIN")
    elif best_win is False and best_delta is not None:
        st.warning("Best observed claim_cure: not yet win")

tabs = st.tabs(["Story", "Live", "Preflight", "CSV files", "Prompt", "Report", "Logs"])

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
    st.subheader("Concise run story")

    max_events_for_story = 220
    min_refresh_s = 45.0
    show_digest = False
    allow_auto = False
    use_full_history = True
    with st.expander("Story settings", expanded=False):
        max_events_for_story = int(
            st.number_input("Events window", min_value=50, max_value=800, value=220, step=10)
        )
        min_refresh_s = float(
            st.number_input("Min story refresh (s)", min_value=5.0, max_value=600.0, value=45.0, step=5.0)
        )
        allow_auto = bool(st.checkbox("Auto-update story (costs tokens)", value=False))
        use_full_history = bool(st.checkbox("Use full run context", value=True))
        show_digest = bool(st.checkbox("Show digest (debug)", value=False))

    last_ev = events[-1] if isinstance(events, list) and events and isinstance(events[-1], dict) else {}
    fp = _json_compact({"run": active_run_id, "last_seq": last_ev.get("seq"), "last_type": last_ev.get("type"), "last_ts": last_ev.get("ts"), "full": bool(use_full_history)})
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

    refresh_now = st.button("Update story now", use_container_width=True)
    should_update = bool(refresh_now)
    if allow_auto and stale:
        if not isinstance(last_ts, (int, float)):
            should_update = True
        elif (time.time() - float(last_ts)) >= float(min_refresh_s):
            should_update = True

    story = str(last_story or "").strip()
    if should_update:
        key = str(openai_key_current or "").strip() or str(st.session_state.get("openai_api_key") or "").strip() or str(os.environ.get("OPENAI_API_KEY") or "").strip()
        if key:
            try:
                if use_full_history:
                    evs_all = _read_events(paths.events_path, max_events=None, read_all=True)
                    full_txt = _full_context_text(evs_all)
                    story = _openai_story_full(api_key=key, full_text=full_txt, run_id=active_run_id, proc_alive=proc_alive)
                else:
                    digest = _event_digest(events, max_events=int(max_events_for_story))
                    story = _openai_story(api_key=key, digest=digest, run_id=active_run_id, proc_alive=proc_alive)
            except Exception as e:
                story = _local_story(events)
                st.warning(f"LLM summary failed; showing fallback. ({str(e)})")
        else:
            story = _local_story(events)
            st.info("No OpenAI key set; showing fallback summary.")

        cache[active_run_id] = {"digest_hash": digest_hash, "story": story, "ts": float(time.time())}
        st.session_state["story_cache"] = cache

    if story.strip():
        st.markdown(story)
    else:
        st.info("No story yet. Click 'Update story now'.")

    if show_digest:
        dig = _event_digest(events, max_events=int(max_events_for_story))
        st.text_area("Digest", value=dig, height=240)

with tabs[1]:
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
        show_types = ["llm_error"]
    elif preset == "Key events":
        show_types = ["llm", "api", "tool_result", "final", "end", "llm_error"]
    else:
        show_types = st.multiselect(
            "Event types",
            options=["start", "resume", "llm", "api", "tool_result", "final", "final_rejected", "end", "llm_error"],
            default=["llm", "api", "tool_result", "final", "end", "llm_error"],
        )
    max_show = int(st.number_input("Max events shown", min_value=50, max_value=2000, value=300, step=50))

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
                        st.json(rj)
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
                            st.json(rj)
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

with tabs[2]:
    _render_preflight_ui(challenge=str(challenge), base_url=str(base_url), api_timeout_s=float(api_timeout), player_id=str(player_id))

with tabs[3]:
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

with tabs[4]:
    st.subheader("Prompt (what the LLM is reading)")
    prompt_txt = ""
    for ev in events:
        if isinstance(ev, dict) and ev.get("type") == "start":
            prompt_txt = str(ev.get("prompt") or "")
            break
    if not prompt_txt and report and isinstance(report.get("prompt"), str):
        prompt_txt = str(report.get("prompt") or "")
    st.text_area("Prompt", value=prompt_txt, height=500)

with tabs[5]:
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

with tabs[6]:
    st.subheader("Runner logs")
    st.caption(str(paths.run_dir))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("stdout")
        st.code(_tail_text(paths.stdout_path), language="")
    with col2:
        st.markdown("stderr")
        st.code(_tail_text(paths.stderr_path), language="")


if auto_refresh:
    time.sleep(refresh_s)
    st.rerun()
