import argparse
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", dest="run_dir", default="runs/evolution")
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "cfg.json"
    base_path = run_dir / "base_payload.json"
    state_path = run_dir / "state.json"
    cands_path = run_dir / "candidates.json"
    stop_path = run_dir / "stop.flag"
    pid_path = run_dir / "runner.pid"

    pid = os.getpid()
    _atomic_write_json(pid_path, {"pid": int(pid), "started_at": float(time.time())})

    if not cfg_path.exists() or not base_path.exists():
        _atomic_write_json(
            state_path,
            {
                "ok": False,
                "job_id": "",
                "running": False,
                "error": "missing cfg.json or base_payload.json",
                "cfg": {},
                "progress": {},
                "history": {},
                "baseline": {},
                "series": {},
                "perf": {},
                "top": [],
                "runner_pid": int(pid),
            },
        )
        return 2

    cfg = _read_json(cfg_path)
    base_payload = _read_json(base_path)
    if not isinstance(cfg, dict):
        cfg = {}
    if not isinstance(base_payload, dict):
        base_payload = {}

    try:
        import runtime_server as rs

        rs._setup_logging()
        rs._install_exception_hooks()

        job = rs._EvolutionJob()
    except Exception as e:
        _atomic_write_json(
            state_path,
            {
                "ok": False,
                "job_id": str(cfg.get("job_id") or ""),
                "running": False,
                "error": str(e),
                "cfg": dict(cfg) if isinstance(cfg, dict) else {},
                "progress": {},
                "history": {},
                "baseline": {},
                "series": {},
                "perf": {},
                "top": [],
                "runner_pid": int(pid),
            },
        )
        return 3

    stop_requested = {"v": False}

    def _request_stop() -> None:
        stop_requested["v"] = True
        try:
            stop_path.touch(exist_ok=True)
        except Exception:
            pass
        try:
            job.stop()
        except Exception:
            pass

    def _sig_handler(_signum, _frame):
        _request_stop()

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass

    try:
        if not isinstance(cfg.get("job_id"), str) or not cfg.get("job_id"):
            cfg["job_id"] = uuid.uuid4().hex
    except Exception:
        cfg["job_id"] = uuid.uuid4().hex

    cfg_algo = str(cfg.get("algo") or "affine").strip().lower()

    if cfg_algo != "auto_switch":
        try:
            job.start(base_payload, cfg)
        except Exception as e:
            _atomic_write_json(
                state_path,
                {
                    "ok": False,
                    "job_id": str(cfg.get("job_id") or ""),
                    "running": False,
                    "error": str(e),
                    "cfg": dict(cfg) if isinstance(cfg, dict) else {},
                    "progress": {},
                    "history": {},
                    "baseline": {},
                    "series": {},
                    "perf": {},
                    "top": [],
                    "runner_pid": int(pid),
                },
            )
            return 4
    else:
        pass

    last_written = 0.0
    candidates_store: Dict[str, Any] = {}

    def _persist_status(st: Dict[str, Any]) -> None:
        nonlocal last_written
        st["runner_pid"] = int(pid)
        st["stop_requested"] = bool(stop_requested.get("v"))
        now = time.time()
        if now - last_written < 0.5:
            return
        try:
            top = st.get("top")
            if isinstance(top, list):
                top_ids: list[str] = []
                for ent in top:
                    if isinstance(ent, dict) and isinstance(ent.get("id"), str):
                        top_ids.append(str(ent.get("id")))
                try:
                    with job._lock:  # type: ignore[attr-defined]
                        for cid in top_ids:
                            c = job.candidates.get(cid) if isinstance(job.candidates, dict) else None
                            if isinstance(c, dict) and isinstance(c.get("genome"), dict):
                                candidates_store[cid] = c
                except Exception:
                    pass

            _atomic_write_json(state_path, st)
            if candidates_store:
                _atomic_write_json(cands_path, candidates_store)
            last_written = now
        except Exception:
            pass

    def _read_status_safe() -> Dict[str, Any]:
        try:
            st0 = job.status()
            if isinstance(st0, dict):
                return st0
        except Exception as e:
            return {
                "ok": False,
                "job_id": str(cfg.get("job_id") or ""),
                "running": False,
                "error": str(e),
                "cfg": dict(cfg) if isinstance(cfg, dict) else {},
                "progress": {},
                "history": {},
                "baseline": {},
                "series": {},
                "perf": {},
                "top": [],
            }
        return {
            "ok": False,
            "job_id": str(cfg.get("job_id") or ""),
            "running": False,
            "error": "status unavailable",
            "cfg": dict(cfg) if isinstance(cfg, dict) else {},
            "progress": {},
            "history": {},
            "baseline": {},
            "series": {},
            "perf": {},
            "top": [],
        }

    if cfg_algo != "auto_switch":
        while True:
            try:
                if stop_path.exists():
                    _request_stop()
            except Exception:
                pass

            st = _read_status_safe()
            _persist_status(st)

            if not bool(st.get("running")):
                break
            time.sleep(0.15)
    else:
        auto_first = str(cfg.get("auto_first") or "cem_delta").strip().lower()
        if auto_first not in ("cem_delta", "affine"):
            auto_first = "cem_delta"
        patience_gens = max(1, int(cfg.get("auto_patience") or 5))
        min_improve_pct = float(cfg.get("auto_min_improve_pct") if cfg.get("auto_min_improve_pct") is not None else (cfg.get("auto_min_delta") or 0.0))
        if not (min_improve_pct >= 0.0):
            min_improve_pct = 0.0
        # Accept either a fraction (0.05) or a percent (5 => 5%).
        if min_improve_pct > 1.0:
            min_improve_pct = float(min_improve_pct) / 100.0
        max_switches = max(0, int(cfg.get("auto_max_switches") or 20))

        total_gens = max(1, int(cfg.get("generations") or 1))
        global_gen_base = 0
        global_last_switch_gen = -1
        switches = 0
        active_algo = str(auto_first)
        best_fitness = float("-inf")

        origin_baseline = None
        global_history = {"best": [], "mean": [], "median": [], "p10": [], "p90": []}
        auto_segments = [{"start_gen": 0, "algo": str(active_algo)}]
        last_top_snapshot: list[Dict[str, Any]] = []
        best_snapshot: Dict[str, Any] | None = None
        best_snapshot_fit = float("-inf")

        window_start_gen = 0
        window_start_best = None
        window_improve_pct = 0.0
        gens_in_window = 0

        cur_base_payload = base_payload

        try:
            _atomic_write_json(base_path, cur_base_payload)
            _atomic_write_json(cands_path, {})
            candidates_store = {}
        except Exception:
            pass

        while global_gen_base < total_gens and not bool(stop_requested.get("v")):
            remaining = int(total_gens - global_gen_base)
            if remaining <= 0:
                break

            cfg_seg = dict(cfg)
            cfg_seg["algo"] = str(active_algo)
            cfg_seg["generations"] = int(remaining)
            try:
                job = rs._EvolutionJob()
            except Exception:
                job = rs._EvolutionJob()

            try:
                job.start(cur_base_payload, cfg_seg)
            except Exception as e:
                st_err = {
                    "ok": False,
                    "job_id": str(cfg.get("job_id") or ""),
                    "running": False,
                    "error": str(e),
                    "cfg": dict(cfg) if isinstance(cfg, dict) else {},
                    "progress": {},
                    "history": {},
                    "baseline": {},
                    "series": {},
                    "perf": {},
                    "top": [],
                    "auto": {
                        "active_algo": str(active_algo),
                        "switches": int(switches),
                        "last_switch_gen": int(global_last_switch_gen),
                        "patience_gens": int(patience_gens),
                        "min_improve_pct": float(min_improve_pct),
                        "best_fitness": float(best_fitness) if best_fitness > float("-inf") else None,
                    },
                }
                _persist_status(st_err)
                break

            seg_plateau_triggered = False
            seg_stop_due_to_max_switch = False
            seg_history_view = {"best": [], "mean": [], "median": [], "p10": [], "p90": []}

            while True:
                try:
                    if stop_path.exists():
                        _request_stop()
                except Exception:
                    pass

                st = _read_status_safe()

                top_now = st.get("top")
                if isinstance(top_now, list) and top_now:
                    try:
                        last_top_snapshot = [dict(x) for x in top_now if isinstance(x, dict)][:10]
                    except Exception:
                        pass

                try:
                    if isinstance(top_now, list) and top_now and isinstance(top_now[0], dict):
                        t0 = dict(top_now[0])
                        t0_fit = float(t0.get("fitness") or float("-inf"))
                        t0_id = t0.get("id")
                        if isinstance(t0_id, str) and t0_id and t0_fit > best_snapshot_fit:
                            cand0 = None
                            if isinstance(getattr(job, "candidates", None), dict):
                                cand0 = job.candidates.get(t0_id)
                            if isinstance(cand0, dict):
                                payload0 = rs._evo_build_candidate_payload(cur_base_payload, cfg_seg, cand0)
                                best_snapshot_fit = float(t0_fit)
                                best_snapshot = {
                                    "id": "__best__",
                                    "fitness": cand0.get("fitness"),
                                    "metrics": cand0.get("metrics"),
                                    "gen": cand0.get("gen"),
                                    "genome": cand0.get("genome"),
                                    "payload_inline": True,
                                    "payload": payload0,
                                }
                                candidates_store["__best__"] = dict(best_snapshot)
                except Exception:
                    pass

                if origin_baseline is None:
                    bb = st.get("baseline")
                    if isinstance(bb, dict) and bb:
                        origin_baseline = dict(bb)

                prog = st.get("progress")
                if isinstance(prog, dict):
                    try:
                        g_local = int(prog.get("generation") or 0)
                    except Exception:
                        g_local = 0
                    prog = dict(prog)
                    prog["generation"] = int(global_gen_base + g_local)
                    prog["total_generations"] = int(total_gens)
                    st["progress"] = prog

                hist = st.get("history")
                best_hist = hist.get("best") if isinstance(hist, dict) else None
                mean_hist = hist.get("mean") if isinstance(hist, dict) else None
                median_hist = hist.get("median") if isinstance(hist, dict) else None
                p10_hist = hist.get("p10") if isinstance(hist, dict) else None
                p90_hist = hist.get("p90") if isinstance(hist, dict) else None

                if isinstance(best_hist, list):
                    seg_history_view["best"] = list(best_hist)
                if isinstance(mean_hist, list):
                    seg_history_view["mean"] = list(mean_hist)
                if isinstance(median_hist, list):
                    seg_history_view["median"] = list(median_hist)
                if isinstance(p10_hist, list):
                    seg_history_view["p10"] = list(p10_hist)
                if isinstance(p90_hist, list):
                    seg_history_view["p90"] = list(p90_hist)

                seg_len = 0
                try:
                    seg_len = int(len(seg_history_view.get("best") or []))
                except Exception:
                    seg_len = 0
                if seg_len > 0:
                    if not isinstance(seg_history_view.get("mean"), list) or len(seg_history_view["mean"]) != seg_len:
                        mm = list(seg_history_view.get("mean") or []) if isinstance(seg_history_view.get("mean"), list) else []
                        if len(mm) < seg_len:
                            mm.extend([float("nan")] * int(seg_len - len(mm)))
                        if len(mm) > seg_len:
                            mm = mm[:seg_len]
                        seg_history_view["mean"] = mm

                    med = list(seg_history_view.get("median") or []) if isinstance(seg_history_view.get("median"), list) else []
                    if len(med) < seg_len:
                        fill = list(seg_history_view.get("mean") or []) if isinstance(seg_history_view.get("mean"), list) else []
                        for i in range(len(med), seg_len):
                            med.append(float(fill[i]) if i < len(fill) else float("nan"))
                    if len(med) > seg_len:
                        med = med[:seg_len]
                    seg_history_view["median"] = med

                    for kk in ("p10", "p90"):
                        vv = list(seg_history_view.get(kk) or []) if isinstance(seg_history_view.get(kk), list) else []
                        if len(vv) < seg_len:
                            vv.extend([float("nan")] * int(seg_len - len(vv)))
                        if len(vv) > seg_len:
                            vv = vv[:seg_len]
                        seg_history_view[kk] = vv
                if isinstance(best_hist, list) and best_hist:
                    try:
                        cur_best = float(max([float(x) for x in best_hist if x is not None]))
                    except Exception:
                        cur_best = float("-inf")
                    if cur_best > best_fitness:
                        best_fitness = float(cur_best)

                    if window_start_best is None:
                        window_start_best = float(best_fitness)
                        window_start_gen = int(global_gen_base)

                    if int(global_gen_base) + len(best_hist) >= int(window_start_gen + patience_gens):
                        denom = abs(float(window_start_best))
                        if denom < 1e-9:
                            denom = 1e-9
                        improve_pct = float((best_fitness - float(window_start_best)) / denom)
                        window_improve_pct = float(improve_pct)
                        if improve_pct < float(min_improve_pct):
                            seg_plateau_triggered = True
                        else:
                            # Start a new window from the current best.
                            window_start_best = float(best_fitness)
                            window_start_gen = int(global_gen_base) + int(len(best_hist))

                        if seg_plateau_triggered:
                            pass

                gens_in_window = max(0, int(global_gen_base + len(best_hist) - window_start_gen)) if isinstance(best_hist, list) else 0
                st["auto"] = {
                    "active_algo": str(active_algo),
                    "switches": int(switches),
                    "last_switch_gen": int(global_last_switch_gen),
                    # Switching criteria (requested): X% per Y generations
                    "patience_gens": int(patience_gens),
                    "min_improve_pct": float(min_improve_pct),
                    "window_improve_pct": float(window_improve_pct),
                    "gens_in_window": int(gens_in_window),
                    "window_start_gen": int(window_start_gen),
                    "best_fitness": float(best_fitness) if best_fitness > float("-inf") else None,
                    "plateau": bool(seg_plateau_triggered),
                    # Back-compat with earlier UI/status naming
                    "patience": int(patience_gens),
                    "min_delta": float(min_improve_pct),
                    "plateau_count": int(1 if seg_plateau_triggered else 0),
                    "segments": list(auto_segments),
                    "updated_at": float(time.time()),
                }

                merged_history = {
                    "best": list(global_history["best"]) + list(seg_history_view["best"]),
                    "mean": list(global_history["mean"]) + list(seg_history_view["mean"]),
                    "median": list(global_history["median"]) + list(seg_history_view["median"]),
                    "p10": list(global_history["p10"]) + list(seg_history_view["p10"]),
                    "p90": list(global_history["p90"]) + list(seg_history_view["p90"]),
                }
                st["history"] = merged_history
                if isinstance(origin_baseline, dict):
                    st["baseline"] = dict(origin_baseline)
                st["series"] = {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []}

                job_running = bool(st.get("running"))
                merged_n = 0
                try:
                    merged_n = int(len(merged_history.get("best") or []))
                except Exception:
                    merged_n = 0

                # IMPORTANT: During auto-switch, a segment finishing sets running=false briefly.
                # If a plateau-triggered switch is pending, keep running=true in persisted status
                # so the UI doesn't go idle and stop polling between segments.
                if (
                    (not job_running)
                    and bool(seg_plateau_triggered)
                    and (not bool(stop_requested.get("v")))
                    and (merged_n < int(total_gens))
                    and (max_switches <= 0 or int(switches) < int(max_switches))
                ):
                    st["running"] = True

                st_cfg = dict(st.get("cfg") or {}) if isinstance(st.get("cfg"), dict) else {}
                st_cfg["algo"] = "auto_switch"
                st["cfg"] = st_cfg

                if (not isinstance(st.get("top"), list) or not st.get("top")) and last_top_snapshot:
                    st["top"] = list(last_top_snapshot)
                if isinstance(best_snapshot, dict):
                    best_row = {
                        "id": "__best__",
                        "fitness": best_snapshot.get("fitness"),
                        "metrics": best_snapshot.get("metrics"),
                        "gen": best_snapshot.get("gen"),
                    }
                    cur_top = st.get("top")
                    rest: list[Dict[str, Any]] = []
                    if isinstance(cur_top, list):
                        for ent in cur_top:
                            if isinstance(ent, dict) and ent.get("id") != "__best__":
                                rest.append(dict(ent))
                    st["top"] = [best_row] + rest[:9]

                _persist_status(st)

                if not job_running:
                    break

                if seg_plateau_triggered or bool(stop_requested.get("v")):
                    try:
                        job.stop()
                    except Exception:
                        pass

                time.sleep(0.15)

            try:
                st_end = _read_status_safe()
                hist_end = st_end.get("history")
                best_hist_end = hist_end.get("best") if isinstance(hist_end, dict) else None
                mean_hist_end = hist_end.get("mean") if isinstance(hist_end, dict) else None
                median_hist_end = hist_end.get("median") if isinstance(hist_end, dict) else None
                p10_hist_end = hist_end.get("p10") if isinstance(hist_end, dict) else None
                p90_hist_end = hist_end.get("p90") if isinstance(hist_end, dict) else None

                best_list = list(best_hist_end) if isinstance(best_hist_end, list) else []
                seg_completed = int(len(best_list))

                mean_list = list(mean_hist_end) if isinstance(mean_hist_end, list) else []
                if len(mean_list) < seg_completed:
                    mean_list.extend([float("nan")] * int(seg_completed - len(mean_list)))
                if len(mean_list) > seg_completed:
                    mean_list = mean_list[:seg_completed]

                median_list = list(median_hist_end) if isinstance(median_hist_end, list) else []
                if len(median_list) < seg_completed:
                    for i in range(len(median_list), seg_completed):
                        median_list.append(float(mean_list[i]) if i < len(mean_list) else float("nan"))
                if len(median_list) > seg_completed:
                    median_list = median_list[:seg_completed]

                p10_list = list(p10_hist_end) if isinstance(p10_hist_end, list) else []
                if len(p10_list) < seg_completed:
                    p10_list.extend([float("nan")] * int(seg_completed - len(p10_list)))
                if len(p10_list) > seg_completed:
                    p10_list = p10_list[:seg_completed]

                p90_list = list(p90_hist_end) if isinstance(p90_hist_end, list) else []
                if len(p90_list) < seg_completed:
                    p90_list.extend([float("nan")] * int(seg_completed - len(p90_list)))
                if len(p90_list) > seg_completed:
                    p90_list = p90_list[:seg_completed]

                global_history["best"].extend(best_list)
                global_history["mean"].extend(mean_list)
                global_history["median"].extend(median_list)
                global_history["p10"].extend(p10_list)
                global_history["p90"].extend(p90_list)

                global_gen_base = int(len(global_history["best"]))
            except Exception:
                pass

            if bool(stop_requested.get("v")):
                break

            if not seg_plateau_triggered:
                # Completed remaining generations without plateau.
                break

            if max_switches > 0 and switches >= max_switches:
                seg_stop_due_to_max_switch = True

            if seg_stop_due_to_max_switch:
                break

            st_final = _read_status_safe()
            top = st_final.get("top")
            best_id = None
            if isinstance(top, list) and top and isinstance(top[0], dict):
                bid = top[0].get("id")
                if isinstance(bid, str) and bid:
                    best_id = bid
            if best_id and isinstance(getattr(job, "candidates", None), dict):
                cand = job.candidates.get(best_id)
                if isinstance(cand, dict):
                    try:
                        cur_base_payload = rs._evo_build_candidate_payload(cur_base_payload, cfg_seg, cand)
                        try:
                            _atomic_write_json(base_path, cur_base_payload)
                        except Exception:
                            pass
                        # IMPORTANT: switching segments historically cleared candidates.json.
                        # If the user stops during this boundary, the UI loses the top table and
                        # can't fetch the latest winner. Persist a stable snapshot record that can
                        # always be loaded.
                        winner_id = "__winner__"
                        winner_ent: Dict[str, Any] = {
                            "id": winner_id,
                            "fitness": cand.get("fitness"),
                            "metrics": cand.get("metrics"),
                            "gen": cand.get("gen"),
                            "genome": cand.get("genome"),
                            "payload_is_base": True,
                        }
                        candidates_store = {winner_id: winner_ent}
                        if isinstance(best_snapshot, dict):
                            candidates_store["__best__"] = dict(best_snapshot)
                        last_top_snapshot = [
                            {
                                "id": winner_id,
                                "fitness": cand.get("fitness"),
                                "metrics": cand.get("metrics"),
                                "gen": cand.get("gen"),
                            }
                        ]
                        try:
                            _atomic_write_json(cands_path, candidates_store)
                        except Exception:
                            pass
                    except Exception:
                        pass

            switches += 1
            global_last_switch_gen = int(global_gen_base)
            active_algo = "affine" if active_algo == "cem_delta" else "cem_delta"
            auto_segments.append({"start_gen": int(global_gen_base), "algo": str(active_algo)})

        st_final2 = _read_status_safe()
        st_final2["auto"] = {
            "active_algo": str(active_algo),
            "switches": int(switches),
            "last_switch_gen": int(global_last_switch_gen),
            "patience_gens": int(patience_gens),
            "min_improve_pct": float(min_improve_pct),
            "best_fitness": float(best_fitness) if best_fitness > float("-inf") else None,
            "segments": list(auto_segments),
            "updated_at": float(time.time()),
        }
        st_final2["history"] = {
            "best": list(global_history["best"]),
            "mean": list(global_history["mean"]),
            "median": list(global_history["median"]),
            "p10": list(global_history["p10"]),
            "p90": list(global_history["p90"]),
        }
        if isinstance(origin_baseline, dict):
            st_final2["baseline"] = dict(origin_baseline)
        st_final2["series"] = {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []}
        st_cfg2 = dict(st_final2.get("cfg") or {}) if isinstance(st_final2.get("cfg"), dict) else {}
        st_cfg2["algo"] = "auto_switch"
        st_final2["cfg"] = st_cfg2
        _persist_status(st_final2)

    try:
        # Ensure the final top candidates are persisted even if the run finished
        # between periodic write intervals.
        try:
            top = st.get("top")
            if isinstance(top, list):
                top_ids = []
                for ent in top:
                    if isinstance(ent, dict) and isinstance(ent.get("id"), str):
                        top_ids.append(str(ent.get("id")))
                if top_ids:
                    try:
                        with job._lock:  # type: ignore[attr-defined]
                            for cid in top_ids:
                                c = job.candidates.get(cid) if isinstance(job.candidates, dict) else None
                                if isinstance(c, dict) and isinstance(c.get("genome"), dict):
                                    candidates_store[cid] = c
                    except Exception:
                        pass
        except Exception:
            pass

        _atomic_write_json(state_path, st)
        if candidates_store:
            _atomic_write_json(cands_path, candidates_store)
    except Exception:
        pass

    try:
        cur = _read_json(pid_path)
        if isinstance(cur, dict) and int(cur.get("pid") or 0) == int(pid):
            _atomic_write_json(pid_path, {"pid": 0, "started_at": float(cur.get("started_at") or 0.0), "finished_at": float(time.time())})
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
