import argparse
import csv
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from backend.env_keys import apply_keys_to_environ
except Exception:
    apply_keys_to_environ = None  # type: ignore


@dataclass
class RunSpec:
    provider: str
    model: str


@dataclass
class RunJob:
    suite_id: str
    spec: RunSpec
    replicate: str
    run_id: str
    run_dir: Path
    files_dir: Path
    events_path: Path
    report_path: Path
    state_path: Path
    pid_path: Path
    stdout_path: Path
    stderr_path: Path
    proc: subprocess.Popen
    out_f: Any
    err_f: Any
    started_ts: float


def _read_pid(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return None
        return int(raw)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _running_runs() -> List[str]:
    out: List[str] = []
    root = _runs_root()
    if not root.exists() or not root.is_dir():
        return out
    for p in root.iterdir():
        try:
            if not p.is_dir():
                continue
            if not str(p.name).startswith("run_"):
                continue
            pid = _read_pid(p / "runner.pid")
            if pid is None:
                continue
            if _pid_alive(int(pid)):
                out.append(str(p.name))
        except Exception:
            continue
    out.sort(reverse=True)
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runs_root() -> Path:
    return _repo_root() / "var" / "runs" / "llm_bench"


def _suites_root() -> Path:
    return _runs_root() / "suites"


def _active_run_id_path() -> Path:
    return _runs_root() / "active_run_id.txt"


def _active_suite_id_path() -> Path:
    return _runs_root() / "active_suite_id.txt"


def _new_suite_id() -> str:
    t = int(time.time())
    r = random.randint(1000, 9999)
    return f"suite_{t}_{r}"


def _new_run_id() -> str:
    t = int(time.time())
    r = random.randint(1000, 9999)
    return f"run_{t}_{r}"


def _write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(txt or "").strip() + "\n", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mean_std(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    xs: List[float] = []
    for v in vals or []:
        try:
            if v is None:
                continue
            xs.append(float(v))
        except Exception:
            continue
    if not xs:
        return None, None
    mu = sum(xs) / float(len(xs))
    if len(xs) < 2:
        return float(mu), None
    var = sum((x - mu) ** 2 for x in xs) / float(len(xs) - 1)
    return float(mu), float(math.sqrt(max(0.0, var)))


def _is_true(v: Any) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _git_cmd(args2: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(args2, cwd=str(_repo_root()), stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _parse_run_spec_line(line: str) -> Optional[RunSpec]:
    s = str(line or "").strip()
    if not s:
        return None
    if s.startswith("#"):
        return None
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            return RunSpec(provider=str(parts[0]), model=str(parts[1]))
        return None
    if ":" in s:
        parts = [p.strip() for p in s.split(":") if p.strip()]
        if len(parts) >= 2:
            return RunSpec(provider=str(parts[0]), model=str(parts[1]))
        return None
    return None


def _load_specs(specs: List[str], spec_file: str) -> List[RunSpec]:
    out: List[RunSpec] = []
    for s in specs or []:
        ent = _parse_run_spec_line(str(s))
        if ent is not None:
            out.append(ent)

    p = str(spec_file or "").strip()
    if p:
        raw = Path(p).read_text(encoding="utf-8", errors="replace")
        for line in raw.splitlines():
            ent = _parse_run_spec_line(line)
            if ent is not None:
                out.append(ent)

    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _steps_from_events(events_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        if not events_path.exists() or not events_path.is_file():
            return None, None
        max_step: Optional[int] = None
        with open(str(events_path), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = str(line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if str(obj.get("type") or "") != "llm":
                    continue
                st = obj.get("step")
                try:
                    si = int(st)
                except Exception:
                    continue
                if max_step is None or si > int(max_step):
                    max_step = int(si)
        if max_step is None:
            return None, None
        return int(max_step), int(max_step) + 1
    except Exception:
        return None, None


def _run_one(
    *,
    suite_id: str,
    suite_dir: Path,
    spec: RunSpec,
    base_url: str,
    challenge: str,
    max_steps: int,
    temperature: float,
    max_tokens: int,
    api_timeout: float,
    llm_timeout: float,
    reset_first: bool,
    prompt_file: Optional[str],
) -> Tuple[str, int, Optional[Dict[str, Any]]]:
    job = _start_one(
        suite_id=suite_id,
        suite_dir=suite_dir,
        spec=spec,
        base_url=base_url,
        challenge=challenge,
        max_steps=max_steps,
        temperature=temperature,
        max_tokens=max_tokens,
        api_timeout=api_timeout,
        llm_timeout=llm_timeout,
        reset_first=reset_first,
        prompt_file=prompt_file,
        run_id_override=None,
    )
    try:
        rc = int(job.proc.wait())
    except KeyboardInterrupt:
        _terminate_job(job)
        raise
    finally:
        _finalize_job(job)
    rep = _read_report(job.report_path)
    return str(job.run_id), int(rc or 0), rep


def _read_report(report_path: Path) -> Optional[Dict[str, Any]]:
    rep = None
    try:
        if report_path.exists() and report_path.is_file():
            rep_obj = _read_json(report_path)
            if isinstance(rep_obj, dict):
                rep = rep_obj
    except Exception:
        rep = None
    return rep


def _start_one(
    *,
    suite_id: str,
    suite_dir: Path,
    spec: RunSpec,
    base_url: str,
    challenge: str,
    max_steps: int,
    temperature: float,
    max_tokens: int,
    api_timeout: float,
    llm_timeout: float,
    reset_first: bool,
    prompt_file: Optional[str],
    run_id_override: Optional[str],
) -> RunJob:
    run_id = str(run_id_override or "").strip() or _new_run_id()
    run_dir = _runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    events_path = run_dir / "events.jsonl"
    report_path = run_dir / "report.json"
    state_path = run_dir / "state.json"
    pid_path = run_dir / "runner.pid"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    cmd: List[str] = [
        sys.executable,
        "trials/run_llm_benchmark.py",
        "--run-id",
        str(run_id),
        "--suite-id",
        str(suite_id),
        "--base-url",
        str(base_url),
        "--provider",
        str(spec.provider),
        "--model",
        str(spec.model),
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
        str(events_path),
        "--out",
        str(report_path),
        "--files-dir",
        str(files_dir),
        "--state-out",
        str(state_path),
    ]
    pf = str(prompt_file or "").strip()
    if pf:
        cmd.extend(["--prompt-file", str(pf)])
    if reset_first:
        cmd.append("--reset-first")

    out_f = open(stdout_path, "ab", buffering=0)
    err_f = open(stderr_path, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        env=dict(os.environ),
        stdout=out_f,
        stderr=err_f,
        start_new_session=True,
    )
    _write_text(pid_path, str(int(proc.pid)))
    _write_text(_active_run_id_path(), run_id)
    return RunJob(
        suite_id=str(suite_id),
        spec=spec,
        replicate="",
        run_id=str(run_id),
        run_dir=run_dir,
        files_dir=files_dir,
        events_path=events_path,
        report_path=report_path,
        state_path=state_path,
        pid_path=pid_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        proc=proc,
        out_f=out_f,
        err_f=err_f,
        started_ts=float(time.time()),
    )


def _finalize_job(job: RunJob) -> None:
    try:
        job.out_f.close()
    except Exception:
        pass
    try:
        job.err_f.close()
    except Exception:
        pass
    try:
        if job.proc.poll() is not None:
            job.pid_path.unlink()
    except Exception:
        pass


def _terminate_job(job: RunJob) -> None:
    try:
        if job.proc.poll() is not None:
            return
    except Exception:
        return
    try:
        os.killpg(int(job.proc.pid), signal.SIGTERM)
    except Exception:
        try:
            job.proc.terminate()
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < 3.0:
        try:
            if job.proc.poll() is not None:
                break
        except Exception:
            break
        time.sleep(0.1)
    try:
        if job.proc.poll() is None:
            try:
                os.killpg(int(job.proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    job.proc.kill()
                except Exception:
                    pass
    except Exception:
        pass


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "suite_id",
        "run_id",
        "run_dir",
        "challenge",
        "provider",
        "model",
        "replicate",
        "player_id",
        "exit_code",
        "ok",
        "max_step",
        "steps_completed",
        "seconds_total",
        "llm_calls",
        "tool_calls",
        "api_calls",
        "experiment_calls",
        "win",
        "final_delta_median_ticks",
        "best_extra_days",
        "best_lifespan_recovery_pct",
        "best_score",
        "best_score_lifedays_per_usd",
        "best_score_seq",
        "money_spent_usd",
        "money_spent_cents",
    ]
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _write_aggregate_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        ch = str(r.get("challenge") or "").strip()
        prov = str(r.get("provider") or "").strip()
        model = str(r.get("model") or "").strip()
        if not prov or not model:
            continue
        k = (ch, prov, model)
        groups.setdefault(k, []).append(r)

    cols = [
        "suite_id",
        "challenge",
        "provider",
        "model",
        "n_runs",
        "n_ok",
        "win_rate",
        "best_score_mean",
        "best_score_std",
        "steps_completed_mean",
        "steps_completed_std",
        "seconds_total_mean",
        "seconds_total_std",
        "money_spent_usd_mean",
        "money_spent_usd_std",
    ]
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (ch, prov, model), items in sorted(groups.items()):
            suite_id = str(items[0].get("suite_id") or "")
            oks = [it for it in items if _is_true(it.get("ok"))]
            wins_n = 0
            for it in oks:
                if _is_true(it.get("win")):
                    wins_n += 1
            win_rate = float(wins_n) / float(len(oks)) if oks else None

            score_mu, score_sd = _mean_std([it.get("best_score") for it in oks])
            steps_mu, steps_sd = _mean_std([it.get("steps_completed") for it in oks])
            sec_mu, sec_sd = _mean_std([it.get("seconds_total") for it in oks])
            usd_mu, usd_sd = _mean_std([it.get("money_spent_usd") for it in oks])

            w.writerow(
                {
                    "suite_id": suite_id,
                    "challenge": ch,
                    "provider": prov,
                    "model": model,
                    "n_runs": int(len(items)),
                    "n_ok": int(len(oks)),
                    "win_rate": win_rate,
                    "best_score_mean": score_mu,
                    "best_score_std": score_sd,
                    "steps_completed_mean": steps_mu,
                    "steps_completed_std": steps_sd,
                    "seconds_total_mean": sec_mu,
                    "seconds_total_std": sec_sd,
                    "money_spent_usd_mean": usd_mu,
                    "money_spent_usd_std": usd_sd,
                }
            )


def main() -> int:
    try:
        if apply_keys_to_environ is not None:
            apply_keys_to_environ()
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Run multiple LLM benchmark configs sequentially and write a suite summary CSV.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--challenge", default="cancer", choices=["cancer", "hereditary_disease", "aging"])
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--api-timeout", type=float, default=5000.0)
    ap.add_argument("--llm-timeout", type=float, default=5000.0)
    ap.add_argument("--prompt-file", default="", help="Prompt file under assets/prompts/ to use for all runs in this suite.")
    ap.add_argument("--reset-first", action="store_true")
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--cooldown-s", type=float, default=0.0)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--suite-id", default="")
    ap.add_argument("--max-parallel", type=int, default=1, help="Max concurrent benchmark runs (default: 1).")
    ap.add_argument(
        "--max-per-provider",
        type=int,
        default=1,
        help="Max concurrent runs per provider (default: 1).",
    )
    ap.add_argument(
        "--spec",
        action="append",
        default=[],
        help="One run spec like 'openai:gpt-5.2-extra-high' or 'claude,claude-opus-4-5-20251101'. Can be repeated.",
    )
    ap.add_argument("--spec-file", default="", help="Path to a text file with one spec per line (provider:model or provider,model).")

    args = ap.parse_args()

    max_parallel_arg = max(1, int(args.max_parallel))
    max_per_provider_arg = max(1, int(args.max_per_provider))

    specs = _load_specs(list(args.spec or []), str(args.spec_file or ""))
    if not specs:
        sys.stderr.write("No specs provided. Use --spec and/or --spec-file.\n")
        return 2

    running = _running_runs()
    if running:
        sys.stderr.write("A benchmark run is already running; refusing to start a suite in parallel.\n")
        for rid in running[:5]:
            sys.stderr.write(f"- {rid}\n")
        return 3

    suite_id = str(args.suite_id or "").strip() or _new_suite_id()
    suite_dir = _suites_root() / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_pid_path = suite_dir / "suite.pid"
    manifest_path = suite_dir / "suite_manifest.json"

    _write_text(_active_suite_id_path(), suite_id)
    _write_text(suite_pid_path, str(os.getpid()))

    rows: List[Dict[str, Any]] = []
    if bool(args.resume):
        try:
            p0 = suite_dir / "suite_summary.csv"
            if p0.exists() and p0.is_file():
                with open(str(p0), "r", encoding="utf-8", errors="replace") as f:
                    rr = csv.DictReader(f)
                    for r in rr:
                        if isinstance(r, dict):
                            rows.append(dict(r))
        except Exception:
            rows = []

    done_keys: set = set()
    try:
        for r in rows:
            if not isinstance(r, dict):
                continue
            key = (str(r.get("provider") or ""), str(r.get("model") or ""), str(r.get("replicate") or ""))
            if key[0] and key[1] and key[2]:
                done_keys.add(key)
    except Exception:
        done_keys = set()

    git_commit = _git_cmd(["git", "rev-parse", "HEAD"])
    git_dirty = None
    try:
        st = _git_cmd(["git", "status", "--porcelain"])
        if st is not None:
            git_dirty = bool(str(st).strip())
    except Exception:
        git_dirty = None
    manifest: Dict[str, Any] = {}
    if bool(args.resume):
        try:
            man0 = _read_json(manifest_path)
            if isinstance(man0, dict):
                manifest = dict(man0)
        except Exception:
            manifest = {}

    if not isinstance(manifest, dict) or not manifest:
        manifest = {
            "suite_id": suite_id,
            "base_url": str(args.base_url),
            "challenge": str(args.challenge),
            "prompt_file": str(args.prompt_file or "").strip(),
            "max_steps": int(args.max_steps),
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
            "api_timeout": float(args.api_timeout),
            "llm_timeout": float(args.llm_timeout),
            "reset_first": bool(args.reset_first),
            "replicates": int(args.replicates),
            "cooldown_s": float(args.cooldown_s),
            "stop_on_error": bool(args.stop_on_error),
            "resume": bool(args.resume),
            "max_parallel": int(max_parallel_arg),
            "max_per_provider": int(max_per_provider_arg),
            "meta": {
                "created_ts": float(time.time()),
                "argv": list(sys.argv),
                "cwd": os.getcwd(),
                "pid": int(os.getpid()),
                "python_executable": str(sys.executable),
                "python_version": str(sys.version),
                "platform": str(platform.platform()),
                "git": {"commit": git_commit, "dirty": git_dirty},
            },
            "runs": [],
        }
    else:
        manifest["suite_id"] = str(manifest.get("suite_id") or suite_id)
        manifest["max_parallel"] = int(max_parallel_arg)
        manifest["max_per_provider"] = int(max_per_provider_arg)
        manifest["prompt_file"] = str(args.prompt_file or "").strip()
        if not isinstance(manifest.get("runs"), list):
            manifest["runs"] = []

    def _sig_term(_sig: int, _frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _sig_term)
    signal.signal(signal.SIGINT, _sig_term)

    try:
        reps = int(args.replicates) if int(args.replicates) > 0 else 1
        total = int(len(specs)) * int(reps)
        max_parallel = int(max_parallel_arg)
        max_per_provider = int(max_per_provider_arg)
        k0 = 0

        pending: List[Tuple[RunSpec, str]] = []
        for spec in specs:
            for rep_i in range(int(reps)):
                rep_s = str(int(rep_i) + 1)
                skip_key = (str(spec.provider), str(spec.model), rep_s)
                if bool(args.resume) and skip_key in done_keys:
                    continue
                pending.append((spec, rep_s))

        provider_inflight: Dict[str, int] = {}
        running_jobs: List[RunJob] = []
        stopping = False

        def _emit_row(*, spec: RunSpec, rep_s: str, run_id: str, rc: int, rep: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            metrics = rep.get("metrics") if isinstance(rep, dict) and isinstance(rep.get("metrics"), dict) else {}
            max_step, steps_completed = _steps_from_events(events_path=(_runs_root() / str(run_id) / "events.jsonl"))
            return {
                "suite_id": suite_id,
                "run_id": str(run_id),
                "run_dir": str((_runs_root() / str(run_id)).resolve()),
                "challenge": (rep.get("challenge") if isinstance(rep, dict) else None) or str(args.challenge),
                "provider": (rep.get("provider") if isinstance(rep, dict) else None) or str(spec.provider),
                "model": (rep.get("model") if isinstance(rep, dict) else None) or str(spec.model),
                "replicate": rep_s,
                "player_id": (rep.get("player_id") if isinstance(rep, dict) else None) if isinstance(rep, dict) else None,
                "exit_code": int(rc),
                "ok": bool(isinstance(rep, dict) and rep.get("ok") is True and int(rc) == 0),
                "max_step": max_step,
                "steps_completed": steps_completed,
                "seconds_total": metrics.get("seconds_total") if isinstance(metrics, dict) else None,
                "llm_calls": metrics.get("llm_calls") if isinstance(metrics, dict) else None,
                "tool_calls": metrics.get("tool_calls") if isinstance(metrics, dict) else None,
                "api_calls": metrics.get("api_calls") if isinstance(metrics, dict) else None,
                "experiment_calls": metrics.get("experiment_calls") if isinstance(metrics, dict) else None,
                "win": metrics.get("win") if isinstance(metrics, dict) else None,
                "final_delta_median_ticks": metrics.get("final_delta_median_ticks") if isinstance(metrics, dict) else None,
                "best_extra_days": metrics.get("best_extra_days") if isinstance(metrics, dict) else None,
                "best_lifespan_recovery_pct": metrics.get("best_lifespan_recovery_pct") if isinstance(metrics, dict) else None,
                "best_score": metrics.get("best_score") if isinstance(metrics, dict) else None,
                "best_score_lifedays_per_usd": metrics.get("best_score_lifedays_per_usd") if isinstance(metrics, dict) else None,
                "best_score_seq": metrics.get("best_score_seq") if isinstance(metrics, dict) else None,
                "money_spent_usd": metrics.get("money_spent_usd") if isinstance(metrics, dict) else None,
                "money_spent_cents": metrics.get("money_spent_cents") if isinstance(metrics, dict) else None,
            }

        while pending or running_jobs:
            started_any = False
            if (not stopping) and (len(running_jobs) < int(max_parallel)) and pending:
                for i in range(len(pending)):
                    spec, rep_s = pending[i]
                    prov = str(spec.provider)
                    if int(provider_inflight.get(prov, 0)) >= int(max_per_provider):
                        continue
                    pending.pop(i)
                    k0 += 1
                    sys.stdout.write(
                        f"[{k0}/{total}] starting {spec.provider}:{spec.model} (replicate {rep_s}/{reps})\n"
                    )
                    sys.stdout.flush()
                    job = _start_one(
                        suite_id=suite_id,
                        suite_dir=suite_dir,
                        spec=spec,
                        base_url=str(args.base_url),
                        challenge=str(args.challenge),
                        max_steps=int(args.max_steps),
                        temperature=float(args.temperature),
                        max_tokens=int(args.max_tokens),
                        api_timeout=float(args.api_timeout),
                        llm_timeout=float(args.llm_timeout),
                        reset_first=bool(args.reset_first),
                        prompt_file=str(args.prompt_file or "").strip() or None,
                        run_id_override=None,
                    )
                    job.replicate = str(rep_s)
                    running_jobs.append(job)
                    provider_inflight[prov] = int(provider_inflight.get(prov, 0)) + 1
                    try:
                        runs0 = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
                        if not isinstance(runs0, list):
                            runs0 = []
                        runs0.append(
                            {
                                "run_id": str(job.run_id),
                                "provider": str(job.spec.provider),
                                "model": str(job.spec.model),
                                "replicate": str(job.replicate),
                                "status": "running",
                                "pid": int(job.proc.pid),
                                "started_ts": float(job.started_ts),
                            }
                        )
                        manifest["runs"] = runs0
                        _write_json(manifest_path, manifest)
                    except Exception:
                        pass
                    started_any = True
                    break

            done: List[RunJob] = []
            for job in list(running_jobs):
                rc0 = job.proc.poll()
                if rc0 is None:
                    continue
                done.append(job)

            for job in done:
                running_jobs.remove(job)
                prov = str(job.spec.provider)
                provider_inflight[prov] = max(0, int(provider_inflight.get(prov, 0)) - 1)
                rc = int(job.proc.returncode or 0)
                _finalize_job(job)
                rep = _read_report(job.report_path)

                row = _emit_row(spec=job.spec, rep_s=str(job.replicate), run_id=str(job.run_id), rc=int(rc), rep=rep)
                rows.append(row)
                try:
                    rr0 = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
                    if not isinstance(rr0, list):
                        rr0 = []
                    found = False
                    for ent in rr0:
                        if not isinstance(ent, dict):
                            continue
                        if str(ent.get("run_id") or "") != str(job.run_id):
                            continue
                        ent["exit_code"] = int(rc)
                        ent["status"] = "done"
                        ent["ended_ts"] = float(time.time())
                        found = True
                        break
                    if not found:
                        rr0.append(
                            {
                                "run_id": str(job.run_id),
                                "provider": str(job.spec.provider),
                                "model": str(job.spec.model),
                                "replicate": str(job.replicate),
                                "exit_code": int(rc),
                                "status": "done",
                                "ended_ts": float(time.time()),
                            }
                        )
                    manifest["runs"] = rr0
                except Exception:
                    pass

                _write_summary_csv(suite_dir / "suite_summary.csv", rows)
                _write_aggregate_csv(suite_dir / "suite_aggregate.csv", rows)
                _write_json(manifest_path, manifest)

                sys.stdout.write(f"[{k0}/{total}] done run_id={job.run_id} exit_code={rc}\n")
                sys.stdout.flush()

                if (not bool(row.get("ok"))) and bool(args.stop_on_error):
                    sys.stderr.write("Stopping suite due to error (stop-on-error enabled).\n")
                    stopping = True
                    for jb in list(running_jobs):
                        _terminate_job(jb)
                        _finalize_job(jb)
                    return 1

                try:
                    cd = float(args.cooldown_s)
                except Exception:
                    cd = 0.0
                if cd > 0:
                    time.sleep(float(cd))

            if (not started_any) and (not done):
                time.sleep(0.2)

    except KeyboardInterrupt:
        for jb in list(running_jobs):
            _terminate_job(jb)
            _finalize_job(jb)
        sys.stderr.write("Suite interrupted.\n")
        return 130
    finally:
        try:
            suite_pid_path.unlink()
        except Exception:
            pass
        try:
            p = _active_suite_id_path()
            if p.exists():
                p.unlink()
        except Exception:
            pass

    sys.stdout.write(f"Suite complete: {suite_id}\n")
    sys.stdout.write(f"Summary CSV: {str(suite_dir / 'suite_summary.csv')}\n")
    sys.stdout.write(f"Aggregate CSV: {str(suite_dir / 'suite_aggregate.csv')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
