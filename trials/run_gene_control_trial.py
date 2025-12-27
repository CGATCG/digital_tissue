import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apply_layer_ops import apply_layer_ops_inplace
from output_calc import _ExprEval


@dataclass(frozen=True)
class CandidateResult:
    score_t1: Optional[float]
    multipliers: Dict[str, float]


def _decode_f32(b64: str, expected_len: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size != expected_len:
        raise ValueError(f"decoded len={arr.size} expected={expected_len}")
    return arr


def _encode_f32(arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=np.float32)
    return base64.b64encode(a.tobytes()).decode("ascii")


def _clone_payload_shallow(base_payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base_payload)
    data = base_payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("payload missing data")
    out["data"] = {k: dict(v) for k, v in data.items() if isinstance(v, dict)}
    return out


def _gene_layer_names(payload: Dict[str, Any]) -> List[str]:
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return []
    out: List[str] = []
    for m in layers:
        if not isinstance(m, dict):
            continue
        nm = m.get("name")
        if isinstance(nm, str) and nm.startswith("gene_"):
            out.append(nm)
    out.sort()
    return out


def _get_measurement_expr(payload: Dict[str, Any], name: str) -> Optional[str]:
    cfg = payload.get("measurements_config")
    if not isinstance(cfg, dict):
        return None
    if int(cfg.get("version") or 0) != 3:
        return None
    ms = cfg.get("measurements")
    if not isinstance(ms, list):
        return None
    for m in ms:
        if not isinstance(m, dict):
            continue
        if str(m.get("name") or "").strip() != name:
            continue
        expr = m.get("expr")
        if isinstance(expr, str) and expr.strip():
            return expr.strip()
    return None


def _eval_measurement(payload: Dict[str, Any], expr: str) -> Optional[float]:
    H = int(payload["H"])
    W = int(payload["W"])
    expected_len = H * W
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("payload missing data")

    layers: Dict[str, np.ndarray] = {}
    for nm, ent in data.items():
        if not isinstance(ent, dict):
            continue
        if ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str):
            continue
        layers[str(nm)] = _decode_f32(b64, expected_len)

    ev = _ExprEval(layers=layers, H=H, W=W)
    return ev.eval(expr)


class _DefaultRngPatch:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._orig = None

    def __enter__(self):
        self._orig = np.random.default_rng

        def _wrapped_default_rng(*args, **kwargs):
            if args or kwargs:
                return self._orig(*args, **kwargs)
            return self._orig(self.seed)

        np.random.default_rng = _wrapped_default_rng
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            np.random.default_rng = self._orig
        return False


def _apply_gene_multipliers_inplace(
    payload: Dict[str, Any],
    gene_names: List[str],
    multipliers: Dict[str, float],
    cell_mask: np.ndarray,
    clip_lo: float,
    clip_hi: float,
) -> None:
    H = int(payload["H"])
    W = int(payload["W"])
    expected_len = H * W
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("payload missing data")

    for g in gene_names:
        ent = data.get(g)
        if not isinstance(ent, dict):
            continue
        if ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str):
            continue
        a = _decode_f32(b64, expected_len).reshape(H, W)
        mul = float(multipliers.get(g, 1.0))
        a2 = a.copy()
        a2[cell_mask] = np.clip(np.rint(a2[cell_mask] * mul), clip_lo, clip_hi)
        ent["b64"] = _encode_f32(a2.reshape(expected_len))


def _sample_candidate(
    rng: np.random.Generator,
    gene_names: List[str],
    active_k: int,
    multiplier_choices: List[float],
) -> Dict[str, float]:
    n = len(gene_names)
    k = max(1, min(int(active_k), n))
    idx = rng.choice(n, size=k, replace=False)
    out: Dict[str, float] = {}
    for i in idx:
        out[gene_names[int(i)]] = float(rng.choice(multiplier_choices))
    return out


def _evaluate_candidate(
    base_payload: Dict[str, Any],
    gene_names: List[str],
    multipliers: Dict[str, float],
    cell_mask: np.ndarray,
    seed: int,
    measurement_expr: str,
    clip_lo: float,
    clip_hi: float,
) -> CandidateResult:
    payload = _clone_payload_shallow(base_payload)
    _apply_gene_multipliers_inplace(payload, gene_names, multipliers, cell_mask, clip_lo=clip_lo, clip_hi=clip_hi)
    with _DefaultRngPatch(seed):
        apply_layer_ops_inplace(payload, seed_offset=0)
    score = _eval_measurement(payload, measurement_expr)
    return CandidateResult(score_t1=score, multipliers=multipliers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridstate", type=str, default=str(Path("array") / "gridstate.json"))
    ap.add_argument("--measurement", type=str, default="morphological_health")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--active-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-lo", type=float, default=0.0)
    ap.add_argument("--clip-hi", type=float, default=None)
    ap.add_argument("--out", type=str, default=str(Path("trials") / "results.json"))
    args = ap.parse_args()

    grid_path = Path(args.gridstate)
    base_payload = json.loads(grid_path.read_text())

    gene_names = _gene_layer_names(base_payload)
    if not gene_names:
        raise ValueError("No gene_* layers found")

    expr = _get_measurement_expr(base_payload, args.measurement)
    if expr is None:
        raise ValueError(f"Measurement {args.measurement!r} not found or not v3 expr")

    H = int(base_payload["H"])
    W = int(base_payload["W"])
    expected_len = H * W
    base_cell = _decode_f32(base_payload["data"]["cell"]["b64"], expected_len).reshape(H, W)
    cell_mask = base_cell == 1

    max_gene_in_cells = 0.0
    for g in gene_names:
        ent = base_payload.get("data", {}).get(g)
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str):
            continue
        a = _decode_f32(b64, expected_len).reshape(H, W)
        if int(cell_mask.sum()) == 0:
            continue
        max_gene_in_cells = max(max_gene_in_cells, float(a[cell_mask].max(initial=0.0)))

    clip_hi = float(args.clip_hi) if args.clip_hi is not None else max(10.0, 2.0 * max_gene_in_cells)
    clip_lo = float(args.clip_lo)

    if args.clip_hi is not None and float(args.clip_hi) < max_gene_in_cells:
        print(
            f"WARNING: --clip-hi={float(args.clip_hi)} is below current max gene value in cells ({max_gene_in_cells}); "
            "this will clamp many/most edits and can collapse the search.",
            file=sys.stderr,
        )

    t0 = time.time()
    score_t = _eval_measurement(base_payload, expr)

    baseline_payload = _clone_payload_shallow(base_payload)
    with _DefaultRngPatch(int(args.seed)):
        apply_layer_ops_inplace(baseline_payload, seed_offset=0)
    baseline_t1 = _eval_measurement(baseline_payload, expr)

    rng = np.random.default_rng(int(args.seed) + 123)
    multiplier_choices = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    best: Optional[CandidateResult] = None
    cand_rows: List[dict] = []

    noop = _evaluate_candidate(
        base_payload,
        gene_names,
        {},
        cell_mask,
        seed=int(args.seed),
        measurement_expr=expr,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )
    cand_rows.append({"i": -1, "score_t1": noop.score_t1, "multipliers": noop.multipliers})
    best = noop

    for i in range(int(args.candidates)):
        multipliers = _sample_candidate(rng, gene_names, active_k=int(args.active_k), multiplier_choices=multiplier_choices)
        res = _evaluate_candidate(
            base_payload,
            gene_names,
            multipliers,
            cell_mask,
            seed=int(args.seed),
            measurement_expr=expr,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )
        cand_rows.append({"i": i, "score_t1": res.score_t1, "multipliers": res.multipliers})
        if (res.score_t1 is not None) and (best.score_t1 is None or res.score_t1 > best.score_t1):
            best = res

    dt_ms = (time.time() - t0) * 1000.0

    out_obj = {
        "gridstate": str(grid_path),
        "H": H,
        "W": W,
        "measurement": args.measurement,
        "expr": expr,
        "genes": gene_names,
        "params": {
            "candidates": int(args.candidates),
            "active_k": int(args.active_k),
            "seed": int(args.seed),
            "clip_lo": clip_lo,
            "clip_hi": clip_hi,
            "multiplier_choices": multiplier_choices,
        },
        "baseline": {
            "score_t": score_t,
            "score_t1": baseline_t1,
            "delta": None if (score_t is None or baseline_t1 is None) else (baseline_t1 - score_t),
        },
        "best_candidate": {
            "score_t1": None if best is None else best.score_t1,
            "multipliers": {} if best is None else best.multipliers,
            "improvement_over_baseline_t1": None
            if best is None or best.score_t1 is None or baseline_t1 is None
            else (best.score_t1 - baseline_t1),
        },
        "candidates": cand_rows,
        "runtime_ms": dt_ms,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))

    print(json.dumps({"baseline_t": score_t, "baseline_t1": baseline_t1, "best_t1": out_obj["best_candidate"]["score_t1"], "runtime_ms": dt_ms}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
