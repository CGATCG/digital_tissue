import concurrent.futures
import json
import os
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from apply_layer_ops import _decode_float32_b64, _encode_float32_b64, apply_layer_ops_inplace
from output_calc import _ExprEval


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


_WEB_DIR = Path(__file__).resolve().parent / "web_editor"


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


class _EvolutionJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.job_id: str = ""
        self.running: bool = False
        self.error: str = ""

        self._base_payload: Optional[Dict[str, Any]] = None

        self.baseline: Dict[str, Any] = {}
        self.series: Dict[str, Any] = {
            "offset": 0,
            "fitness": [],
            "best": [],
            "mean": [],
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
            "p10": [],
            "p90": [],
        }

        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.top_ids: list[str] = []

        self.perf: Dict[str, Any] = {
            "evals": 0,
            "apply_s": 0.0,
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
            }
            history = self.history if isinstance(self.history, dict) else {}
            history_out = {
                "best": list(history.get("best") or []),
                "mean": list(history.get("mean") or []),
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
        self.candidates = {}
        self.top_ids = []
        self.history = {"best": [], "mean": [], "p10": [], "p90": []}
        self.baseline = {}
        self.series = {"offset": 0, "fitness": [], "best": [], "mean": []}
        self._series_sum = 0.0
        self._series_n = 0
        self._series_best = float("-inf")
        self.perf = {
            "evals": 0,
            "apply_s": 0.0,
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
        seed = int(cfg.get("seed") or 0)
        mut_rate = float(cfg.get("mutation_rate") or 0.15)
        sigma_scale = float(cfg.get("sigma_scale") or 0.25)
        sigma_bias = float(cfg.get("sigma_bias") or 0.25)
        huge = float(cfg.get("huge") or 1e9)
        fitness_w = cfg.get("fitness_weights")
        if not isinstance(fitness_w, dict):
            fitness_w = {}

        if variants <= 0 or generations <= 0 or ticks <= 0:
            raise ValueError("variants/generations/ticks must be > 0")
        elites = max(1, min(variants, elites if elites > 0 else min(10, variants)))
        replicates = max(1, min(50, replicates))
        if workers <= 0:
            workers = max(1, int(min(4, os.cpu_count() or 1)))
        workers = max(1, min(32, workers))

        layer_meta = base_payload.get("layers")
        if not isinstance(layer_meta, list):
            raise ValueError("payload missing layers")
        kinds: Dict[str, str] = {}
        for m in layer_meta:
            if not isinstance(m, dict):
                continue
            nm = m.get("name")
            if isinstance(nm, str) and nm:
                kinds[nm] = str(m.get("kind") or "continuous")

        data = base_payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("payload missing data")
        H = int(base_payload.get("H") or 0)
        W = int(base_payload.get("W") or 0)
        if H <= 0 or W <= 0:
            raise ValueError("payload invalid H/W")

        cell_layer = _find_cell_layer_name(base_payload)
        if not cell_layer:
            raise ValueError("payload missing cell layer (expected 'cell' or 'cell_type')")

        cell_ent = data.get(cell_layer)
        if not isinstance(cell_ent, dict) or cell_ent.get("dtype") != "float32":
            raise ValueError("invalid cell layer")
        cell0 = _decode_float32_b64(str(cell_ent.get("b64") or ""), expected_len=H * W, layer_name=cell_layer)
        cell_mask = cell0.reshape(H * W) > 0.5

        mutable_names: list[str] = []
        layer_stats: Dict[str, Dict[str, float]] = {}
        for name, ent in data.items():
            if not isinstance(name, str):
                continue
            if not (name.startswith("gene_") or name.startswith("rna_") or name.startswith("protein_")):
                continue
            if not isinstance(ent, dict) or ent.get("dtype") != "float32" or not isinstance(ent.get("b64"), str):
                continue
            mutable_names.append(name)
            try:
                arr = _decode_float32_b64(str(ent.get("b64") or ""), expected_len=H * W, layer_name=name)
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

        eval_total = generations * variants * replicates
        with self._lock:
            self.progress["evaluations_total"] = int(eval_total)
            self.progress["updated_at"] = time.time()

        rng = np.random.default_rng(seed)

        def _perf_add(apply_s: float, ticks_s: float, decode_cell_s: float, total_s: float) -> None:
            with self._lock:
                p = self.perf if isinstance(self.perf, dict) else {}
                p["evals"] = int(p.get("evals") or 0) + 1
                p["apply_s"] = float(p.get("apply_s") or 0.0) + float(apply_s)
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

        def _apply_genome_to_payload(payload: Dict[str, Any], genome: Dict[str, Any]) -> Dict[str, Any]:
            out = _deepcopy_payload(payload)
            out.pop("event_counters", None)
            out_data = out.get("data")
            if not isinstance(out_data, dict):
                raise ValueError("payload missing data")
            for nm, gb in genome.items():
                ent = out_data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if not isinstance(b64, str) or not b64:
                    continue
                arr = _decode_float32_b64(b64, expected_len=H * W, layer_name=nm)
                arr2: np.ndarray
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
                ent["b64"] = _encode_float32_b64(arr2)
            return out

        def _eval_genome(genome: Dict[str, Any], seed0: int) -> Dict[str, Any]:
            t_total0 = time.perf_counter()
            t_apply0 = time.perf_counter()
            p = _apply_genome_to_payload(base_payload, genome)
            t_apply = time.perf_counter() - t_apply0

            t_ticks = 0.0
            for t in range(ticks):
                if self._stop.is_set():
                    raise RuntimeError("stopped")
                tt0 = time.perf_counter()
                apply_layer_ops_inplace(p, seed_offset=seed0 + t)
                t_ticks += time.perf_counter() - tt0
            dd = p.get("data")
            if not isinstance(dd, dict):
                raise ValueError("payload missing data")
            cell_ent2 = dd.get(cell_layer)
            if not isinstance(cell_ent2, dict) or cell_ent2.get("dtype") != "float32":
                raise ValueError("payload missing cell layer")

            t_dec0 = time.perf_counter()
            cell_arr = _decode_float32_b64(str(cell_ent2.get("b64") or ""), expected_len=H * W, layer_name=cell_layer)
            t_decode = time.perf_counter() - t_dec0
            alive = int((cell_arr > 0.5).sum())
            events = p.get("event_counters") if isinstance(p, dict) else None
            totals = events.get("totals") if isinstance(events, dict) else None
            if not isinstance(totals, dict):
                totals = {}
            divisions = int(totals.get("divisions") or 0)
            starv = int(totals.get("starvation_deaths") or 0)
            dmg = int(totals.get("damage_deaths") or 0)

            t_total = time.perf_counter() - t_total0
            _perf_add(apply_s=t_apply, ticks_s=t_ticks, decode_cell_s=t_decode, total_s=t_total)
            return {
                "alive": alive,
                "divisions": divisions,
                "starvation_deaths": starv,
                "damage_deaths": dmg,
            }

        def _fitness(metrics: Dict[str, Any]) -> float:
            w_alive = float(fitness_w.get("alive", 1.0))
            w_div = float(fitness_w.get("divisions", 2.0))
            w_starv = float(fitness_w.get("starvation_deaths", -1.0))
            w_dmg = float(fitness_w.get("damage_deaths", -1.0))
            return (
                w_alive * float(metrics.get("alive") or 0)
                + w_div * float(metrics.get("divisions") or 0)
                + w_starv * float(metrics.get("starvation_deaths") or 0)
                + w_dmg * float(metrics.get("damage_deaths") or 0)
            )

        base_rep_metrics = []
        for ri in range(replicates):
            if self._stop.is_set():
                raise RuntimeError("stopped")
            seed0 = seed + (0 * 1000003) + (0 * 1009) + (ri * 97)
            base_rep_metrics.append(_eval_genome({}, seed0=seed0))
        alive_m0 = float(np.mean([mm["alive"] for mm in base_rep_metrics]))
        div_m0 = float(np.mean([mm["divisions"] for mm in base_rep_metrics]))
        starv_m0 = float(np.mean([mm["starvation_deaths"] for mm in base_rep_metrics]))
        dmg_m0 = float(np.mean([mm["damage_deaths"] for mm in base_rep_metrics]))
        base_metrics = {
            "alive": int(round(alive_m0)),
            "divisions": div_m0,
            "starvation_deaths": starv_m0,
            "damage_deaths": dmg_m0,
        }
        base_fit = float(_fitness(base_metrics))
        with self._lock:
            self.baseline = {"fitness": base_fit, "metrics": base_metrics}
            self.progress["updated_at"] = time.time()

        max_points = int(cfg.get("plot_max_points") or 5000)

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
                std = float(layer_stats.get(nm, {}).get("std", 1.0))
                sig_floor[nm] = float(cem_sigma_floor * std)
                mu[nm] = np.zeros((H * W,), dtype=np.float32)
                sig[nm] = np.full((H * W,), float(cem_sigma_init * std), dtype=np.float32)
                if use_cell_mask:
                    mu[nm][~cell_mask] = 0.0
                    sig[nm][~cell_mask] = 0.0

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
                    genome: Dict[str, Any] = {}
                    for nm in mutable_names:
                        eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                        delta = mu0[nm] + sig0[nm] * eps
                        if use_cell_mask:
                            delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                        genome[nm] = {"delta": delta}
                    rep_metrics = []
                    for ri in range(replicates):
                        if self._stop.is_set():
                            return None
                        seed0 = seed + (gen * 1000003) + (vi * 1009) + (ri * 97)
                        m = _eval_genome(genome, seed0=seed0)
                        rep_metrics.append(m)
                        with self._lock:
                            self.progress["evaluations_done"] = int(self.progress.get("evaluations_done", 0) + 1)
                            self.progress["updated_at"] = time.time()

                    if not rep_metrics:
                        return None

                    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                    metrics = {
                        "alive": int(round(alive_m)),
                        "divisions": div_m,
                        "starvation_deaths": starv_m,
                        "damage_deaths": dmg_m,
                    }
                    fit = float(_fitness(metrics))
                    return {"vi": int(vi), "metrics": metrics, "fitness": fit}

                ex: Optional[concurrent.futures.ThreadPoolExecutor] = None
                futs: Dict[concurrent.futures.Future, int] = {}
                candidates_this_gen: list[Dict[str, Any]] = []
                try:
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=gen_workers)
                    for vi in plan:
                        if self._stop.is_set():
                            break
                        futs[ex.submit(_eval_variant, vi)] = vi

                    for fut in concurrent.futures.as_completed(list(futs.keys())):
                        if self._stop.is_set():
                            break
                        try:
                            res = fut.result()
                        except Exception:
                            continue
                        if not isinstance(res, dict):
                            continue

                        vi = int(res.get("vi") or 0)
                        metrics = res.get("metrics")
                        fit = float(res.get("fitness") or 0.0)
                        if not isinstance(metrics, dict):
                            continue

                        with self._lock:
                            self.progress["variant"] = int(vi)
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
                        candidates_this_gen.append({"id": cid, "gen": gen, "fitness": fit, "metrics": metrics, "vi": vi})
                finally:
                    if ex is not None:
                        try:
                            ex.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            try:
                                ex.shutdown(wait=False)
                            except Exception:
                                pass

                if not candidates_this_gen:
                    break

                candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                best = float(fits.max())
                mean = float(fits.mean())
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
                    self.history["p10"].append(p10)
                    self.history["p90"].append(p90)
                    self.progress["updated_at"] = time.time()

            return

        parents: list[Dict[str, Dict[str, float]]] = [{}]

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
                    with self._lock:
                        self.progress["evaluations_done"] = int(self.progress.get("evaluations_done", 0) + 1)
                        self.progress["updated_at"] = time.time()

                if not rep_metrics:
                    return None

                alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                metrics = {
                    "alive": int(round(alive_m)),
                    "divisions": div_m,
                    "starvation_deaths": starv_m,
                    "damage_deaths": dmg_m,
                }
                fit = float(_fitness(metrics))
                return {
                    "vi": int(vi),
                    "genome": genome,
                    "metrics": metrics,
                    "fitness": fit,
                }

            ex: Optional[concurrent.futures.ThreadPoolExecutor] = None
            futs: Dict[concurrent.futures.Future, int] = {}
            try:
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=gen_workers)
                for vi, genome in plan:
                    if self._stop.is_set():
                        break
                    futs[ex.submit(_eval_variant, vi, genome)] = vi

                for fut in concurrent.futures.as_completed(list(futs.keys())):
                    if self._stop.is_set():
                        break
                    try:
                        res = fut.result()
                    except Exception:
                        continue
                    if not isinstance(res, dict):
                        continue

                    vi = int(res.get("vi") or 0)
                    genome = res.get("genome")
                    metrics = res.get("metrics")
                    fit = float(res.get("fitness") or 0.0)
                    if not isinstance(genome, dict) or not isinstance(metrics, dict):
                        continue

                    with self._lock:
                        self.progress["variant"] = int(vi)
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
            finally:
                if ex is not None:
                    try:
                        ex.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        try:
                            ex.shutdown(wait=False)
                        except Exception:
                            pass

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


_EVO = _EvolutionJob()


class _RuntimeState:
    def __init__(self) -> None:
        self.payload: Optional[Dict[str, Any]] = None
        self.tick: int = 0

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

    def step(self, layer_names: Optional[list]) -> Dict[str, Any]:
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

    def do_POST(self):  # noqa: N802
        try:
            if self.path == "/api/evolution/start":
                body = self._read_json_body()
                payload = body.get("payload")
                cfg = body.get("config")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                if not isinstance(cfg, dict):
                    raise ValueError("missing config")
                _EVO.start(payload, cfg)
                self._send_json(200, {"ok": True, "job_id": _EVO.job_id})
                return

            if self.path == "/api/evolution/stop":
                _EVO.stop()
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/evolution/status":
                self._send_json(200, _EVO.status())
                return

            if self.path == "/api/evolution/candidate":
                body = self._read_json_body()
                cid = body.get("id")
                if not isinstance(cid, str) or not cid:
                    raise ValueError("missing id")
                self._send_json(200, _EVO.candidate(cid))
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

            self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as e:
            self._send_json(400, {"ok": False, "error": str(e)})


def main() -> int:
    if not _WEB_DIR.exists():
        print(f"web editor dir not found: {_WEB_DIR}", file=sys.stderr)
        return 2

    port = 8000
    try:
        if len(sys.argv) >= 2:
            port = int(sys.argv[1])
    except Exception:
        port = 8000

    # Avoid caching during dev.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    srv = ThreadingHTTPServer(("127.0.0.1", port), RuntimeHandler)
    print(f"Runtime server: http://127.0.0.1:{port}/")
    print("Open Functions → Runtime")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
