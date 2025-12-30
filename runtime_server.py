import concurrent.futures
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from multiprocessing import shared_memory
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from apply_layer_ops import _decode_float32_b64, _encode_float32_b64, apply_layer_ops_inplace
from output_calc import _ExprEval


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


def _evo_worker_init(
    base_payload_fast: Dict[str, Any],
    kinds: Dict[str, str],
    cell_layer: str,
    huge: float,
) -> None:
    global _EVO_WORKER_BASE, _EVO_WORKER_BASE_DATA, _EVO_WORKER_PAYLOAD, _EVO_WORKER_DATA
    global _EVO_WORKER_KINDS, _EVO_WORKER_CELL_LAYER, _EVO_WORKER_CELL_MASK, _EVO_WORKER_HUGE

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
        layers_dict: Dict[str, np.ndarray] = {}
        for nm, ent in dd.items():
            if isinstance(ent, dict) and ent.get("dtype") == "float32":
                arr = ent.get("arr")
                if isinstance(arr, np.ndarray):
                    layers_dict[nm] = arr

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
                sel = _compute_selected_measurements_from_layers(p, layers_dict, H, W, agg_names)
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
        
        measurements = _compute_measurements_from_layers(p, layers_dict, H, W)
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
    
    # Fitness calculation: measurement-based only
    fit = 0.0
    meas_weights = fitness_w.get("measurements", {})
    if isinstance(meas_weights, dict) and merged_measurements:
        for meas_name, w in meas_weights.items():
            if isinstance(w, (int, float)) and meas_name in merged_measurements:
                fit += float(w) * float(merged_measurements[meas_name])

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
        layers_dict: Dict[str, np.ndarray] = {}
        for nm, ent in dd.items():
            if isinstance(ent, dict) and ent.get("dtype") == "float32":
                arr = ent.get("arr")
                if isinstance(arr, np.ndarray):
                    layers_dict[nm] = arr

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
                sel = _compute_selected_measurements_from_layers(p, layers_dict, H, W, agg_names)
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
        
        measurements = _compute_measurements_from_layers(p, layers_dict, H, W)
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
    
    # Fitness calculation: measurement-based only
    fit = 0.0
    meas_weights = fitness_w.get("measurements", {})
    if isinstance(meas_weights, dict) and merged_measurements:
        for meas_name, w in meas_weights.items():
            if isinstance(w, (int, float)) and meas_name in merged_measurements:
                fit += float(w) * float(merged_measurements[meas_name])

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
        
        # Debug: print layer stats for CEM initialization
        print(f"DEBUG CEM: Found {len(mutable_names)} mutable layers")
        for nm in mutable_names[:5]:  # Show first 5
            st = layer_stats.get(nm, {})
            print(f"DEBUG CEM: Layer '{nm}' - mean={st.get('mean', 0):.4f}, std={st.get('std', 1):.4f}")

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
            layers_dict0: Dict[str, np.ndarray] = {}
            for nm, ent in dd0.items():
                if isinstance(ent, dict) and ent.get("dtype") == "float32":
                    arr = ent.get("arr")
                    if isinstance(arr, np.ndarray):
                        layers_dict0[str(nm)] = arr
            per_tick_meas: Dict[str, list[float]] = {k: [] for k in agg_names}

            t_ticks = 0.0
            for t in range(ticks):
                if self._stop.is_set():
                    raise RuntimeError("stopped")
                tt0 = time.perf_counter()
                apply_layer_ops_inplace(p, seed_offset=seed0 + t)
                t_ticks += time.perf_counter() - tt0

                if agg_names:
                    sel = _compute_selected_measurements_from_layers(p, layers_dict0, int(H), int(W), agg_names)
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
            if meas_weights:
                mm = metrics.get("measurements")
                if isinstance(mm, dict):
                    fit = 0.0
                    for meas_name, w in meas_weights.items():
                        if meas_name in mm and isinstance(w, (int, float)):
                            fit += float(w) * float(mm.get(meas_name) or 0.0)
                    return float(fit)
                return 0.0
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
        
        # Log baseline details for debugging
        print(f"DEBUG: Baseline fitness calculation: {base_fit}")
        print(f"DEBUG: Baseline metrics: {base_metrics}")
        print(f"DEBUG: Using measurement weights: {meas_weights}")
        
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
            
            # Debug: print initial CEM sigma values
            print(f"DEBUG CEM: cem_sigma_init={cem_sigma_init}, cem_alpha={cem_alpha}")
            for nm in mutable_names[:3]:  # Show first 3
                layer_mean = layer_stats.get(nm, {}).get("mean", 1.0)
                layer_std = layer_stats.get(nm, {}).get("std", 1.0)
                scale_ref = min(layer_std, abs(layer_mean) * 0.1) if abs(layer_mean) > 1.0 else layer_std
                init_sig = cem_sigma_init * scale_ref
                print(f"DEBUG CEM: Layer '{nm}' initial sigma={init_sig:.4f} (mean={layer_mean:.1f}, std={layer_std:.1f}, scale_ref={scale_ref:.1f})")

            if worker_mode == "process":
                k_layers = int(len(mutable_names))
                n_cells = int(H * W)
                mu_shm = shared_memory.SharedMemory(create=True, size=int(k_layers * n_cells * 4))
                sig_shm = shared_memory.SharedMemory(create=True, size=int(k_layers * n_cells * 4))
                mu_shm_arr = np.ndarray((k_layers, n_cells), dtype=np.float32, buffer=mu_shm.buf)
                sig_shm_arr = np.ndarray((k_layers, n_cells), dtype=np.float32, buffer=sig_shm.buf)
                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=workers,
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

        if worker_mode == "process":
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
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

    srv = ThreadingHTTPServer(("0.0.0.0", port), RuntimeHandler)
    print(f"Runtime server: http://0.0.0.0:{port}/")
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
