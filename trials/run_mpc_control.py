import argparse
import base64
import json
import math
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
class RolloutResult:
    objective: Optional[float]
    health: List[Optional[float]]
    log_multipliers: List[float]


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


def _layers_by_prefix(payload: Dict[str, Any], prefixes: Tuple[str, ...]) -> List[str]:
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return []
    out: List[str] = []
    for m in layers:
        if not isinstance(m, dict):
            continue
        nm = m.get("name")
        if not isinstance(nm, str):
            continue
        if any(nm.startswith(p) for p in prefixes):
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


def _compute_default_clip_hi(
    base_payload: Dict[str, Any],
    layer_names: List[str],
    apply_cell_mask: bool,
) -> float:
    H = int(base_payload["H"])
    W = int(base_payload["W"])
    expected_len = H * W
    data = base_payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("payload missing data")

    cell_mask = None
    if apply_cell_mask:
        base_cell = _decode_f32(data["cell"]["b64"], expected_len).reshape(H, W)
        cell_mask = base_cell == 1

    mx = 0.0
    for nm in layer_names:
        ent = data.get(nm)
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str):
            continue
        a = _decode_f32(b64, expected_len).reshape(H, W)
        if cell_mask is None:
            mx = max(mx, float(a.max(initial=0.0)))
        else:
            if int(cell_mask.sum()) == 0:
                continue
            mx = max(mx, float(a[cell_mask].max(initial=0.0)))
    return max(10.0, 2.0 * mx)


def _apply_log_multipliers_inplace(
    payload: Dict[str, Any],
    layer_names: List[str],
    log_multipliers: np.ndarray,
    action_mode: str,
    apply_cell_mask: bool,
    clip_lo: float,
    clip_hi: float,
    round_counts: bool,
) -> None:
    if action_mode not in {"global", "two_region"}:
        raise ValueError(f"invalid action_mode={action_mode!r}")
    expected = len(layer_names) if action_mode == "global" else 2 * len(layer_names)
    if expected != int(log_multipliers.size):
        raise ValueError("log_multipliers size mismatch")

    H = int(payload["H"])
    W = int(payload["W"])
    expected_len = H * W
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("payload missing data")

    cell_mask = None
    if apply_cell_mask:
        cell = _decode_f32(data["cell"]["b64"], expected_len).reshape(H, W)
        cell_mask = cell == 1

    region_masks: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if action_mode == "two_region":
        morph = _decode_f32(data["morphology"]["b64"], expected_len).reshape(H, W)
        inside = morph == 1
        outside = ~inside
        if cell_mask is not None:
            inside = inside & cell_mask
            outside = outside & cell_mask
        region_masks = (inside, outside)

    for i, nm in enumerate(layer_names):
        ent = data.get(nm)
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str):
            continue
        a = _decode_f32(b64, expected_len).reshape(H, W)
        a2 = a.copy()

        if action_mode == "global":
            mul = float(math.exp(float(log_multipliers[i])))
            if cell_mask is None:
                v = a2 * mul
                if round_counts:
                    v = np.rint(v)
                a2 = np.clip(v, clip_lo, clip_hi)
            else:
                v = a2[cell_mask] * mul
                if round_counts:
                    v = np.rint(v)
                a2[cell_mask] = np.clip(v, clip_lo, clip_hi)
        else:
            assert region_masks is not None
            inside, outside = region_masks
            mul_in = float(math.exp(float(log_multipliers[2 * i])))
            mul_out = float(math.exp(float(log_multipliers[2 * i + 1])))

            if bool(inside.any()):
                v = a2[inside] * mul_in
                if round_counts:
                    v = np.rint(v)
                a2[inside] = np.clip(v, clip_lo, clip_hi)
            if bool(outside.any()):
                v = a2[outside] * mul_out
                if round_counts:
                    v = np.rint(v)
                a2[outside] = np.clip(v, clip_lo, clip_hi)

        ent["b64"] = _encode_f32(a2.reshape(expected_len))


def _rollout(
    base_payload: Dict[str, Any],
    layer_names: List[str],
    log_multipliers: np.ndarray,
    horizon: int,
    seed: int,
    measurement_expr: str,
    gamma: float,
    effort_lambda: float,
    apply_each_step: bool,
    action_mode: str,
    apply_cell_mask: bool,
    clip_lo: float,
    clip_hi: float,
    round_counts: bool,
) -> RolloutResult:
    payload = _clone_payload_shallow(base_payload)

    health: List[Optional[float]] = []
    obj = 0.0
    any_valid = False

    for t in range(int(horizon)):
        if apply_each_step or t == 0:
            _apply_log_multipliers_inplace(
                payload,
                layer_names,
                log_multipliers,
                action_mode=action_mode,
                apply_cell_mask=apply_cell_mask,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
                round_counts=round_counts,
            )

        with _DefaultRngPatch(int(seed) + 10007 * t):
            apply_layer_ops_inplace(payload, seed_offset=0)

        h = _eval_measurement(payload, measurement_expr)
        health.append(h)
        if h is not None:
            obj += (float(gamma) ** t) * float(h)
            any_valid = True

    if not any_valid:
        return RolloutResult(objective=None, health=health, log_multipliers=[float(x) for x in log_multipliers.tolist()])

    if float(effort_lambda) > 0.0:
        obj -= float(effort_lambda) * float(np.mean(np.square(log_multipliers)))

    return RolloutResult(objective=float(obj), health=health, log_multipliers=[float(x) for x in log_multipliers.tolist()])


def _optimize_random(
    base_payload: Dict[str, Any],
    layer_names: List[str],
    horizon: int,
    seed: int,
    measurement_expr: str,
    gamma: float,
    effort_lambda: float,
    samples: int,
    sigma: float,
    max_abs_log: float,
    apply_each_step: bool,
    action_mode: str,
    apply_cell_mask: bool,
    clip_lo: float,
    clip_hi: float,
    round_counts: bool,
) -> Tuple[RolloutResult, List[dict]]:
    rng = np.random.default_rng(int(seed) + 123)

    dim = len(layer_names) if action_mode == "global" else 2 * len(layer_names)
    best = _rollout(
        base_payload,
        layer_names,
        np.zeros((dim,), dtype=np.float32),
        horizon=horizon,
        seed=seed,
        measurement_expr=measurement_expr,
        gamma=gamma,
        effort_lambda=effort_lambda,
        apply_each_step=apply_each_step,
        action_mode=action_mode,
        apply_cell_mask=apply_cell_mask,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        round_counts=round_counts,
    )

    rows: List[dict] = [{"i": -1, "objective": best.objective, "health": best.health}]

    for i in range(int(samples)):
        z = rng.normal(loc=0.0, scale=float(sigma), size=(dim,)).astype(np.float32)
        z = np.clip(z, -float(max_abs_log), float(max_abs_log))
        r = _rollout(
            base_payload,
            layer_names,
            z,
            horizon=horizon,
            seed=seed,
            measurement_expr=measurement_expr,
            gamma=gamma,
            effort_lambda=effort_lambda,
            apply_each_step=apply_each_step,
            action_mode=action_mode,
            apply_cell_mask=apply_cell_mask,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            round_counts=round_counts,
        )
        rows.append({"i": i, "objective": r.objective, "health": r.health})
        if r.objective is not None and (best.objective is None or r.objective > best.objective):
            best = r

    return best, rows


def _optimize_cem(
    base_payload: Dict[str, Any],
    layer_names: List[str],
    horizon: int,
    seed: int,
    measurement_expr: str,
    gamma: float,
    effort_lambda: float,
    pop_size: int,
    iters: int,
    elite_frac: float,
    init_sigma: float,
    min_sigma: float,
    max_abs_log: float,
    apply_each_step: bool,
    action_mode: str,
    apply_cell_mask: bool,
    clip_lo: float,
    clip_hi: float,
    round_counts: bool,
) -> Tuple[RolloutResult, List[dict]]:
    rng = np.random.default_rng(int(seed) + 456)

    dim = len(layer_names) if action_mode == "global" else 2 * len(layer_names)
    mu = np.zeros((dim,), dtype=np.float32)
    sig = np.full((dim,), float(init_sigma), dtype=np.float32)

    best = _rollout(
        base_payload,
        layer_names,
        np.zeros((dim,), dtype=np.float32),
        horizon=horizon,
        seed=seed,
        measurement_expr=measurement_expr,
        gamma=gamma,
        effort_lambda=effort_lambda,
        apply_each_step=apply_each_step,
        action_mode=action_mode,
        apply_cell_mask=apply_cell_mask,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        round_counts=round_counts,
    )

    rows: List[dict] = [{"iter": -1, "i": -1, "objective": best.objective, "health": best.health}]

    elite_n = max(1, int(round(float(elite_frac) * float(pop_size))))
    elite_n = min(elite_n, int(pop_size))

    for it in range(int(iters)):
        zs = rng.normal(loc=mu, scale=sig, size=(int(pop_size), dim)).astype(np.float32)
        zs = np.clip(zs, -float(max_abs_log), float(max_abs_log))

        scored: List[Tuple[float, int, RolloutResult]] = []
        for i in range(int(pop_size)):
            r = _rollout(
                base_payload,
                layer_names,
                zs[i],
                horizon=horizon,
                seed=seed,
                measurement_expr=measurement_expr,
                gamma=gamma,
                effort_lambda=effort_lambda,
                apply_each_step=apply_each_step,
                action_mode=action_mode,
                apply_cell_mask=apply_cell_mask,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
                round_counts=round_counts,
            )
            rows.append({"iter": it, "i": i, "objective": r.objective, "health": r.health})
            if r.objective is not None:
                scored.append((float(r.objective), i, r))

        if not scored:
            break

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = scored[:elite_n]

        if elites[0][2].objective is not None and (best.objective is None or elites[0][2].objective > best.objective):
            best = elites[0][2]

        elite_z = np.stack([zs[i] for _, i, _ in elites], axis=0)
        mu = np.mean(elite_z, axis=0).astype(np.float32)
        sig = np.std(elite_z, axis=0).astype(np.float32)
        sig = np.maximum(sig, float(min_sigma)).astype(np.float32)

    return best, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridstate", type=str, default=str(Path("array") / "gridstate.json"))
    ap.add_argument("--measurement", type=str, default="morphological_health")
    ap.add_argument("--control-mode", type=str, default="gene", choices=["gene", "rna", "protein", "molecule", "all"])
    ap.add_argument("--optimizer", type=str, default="random", choices=["random", "cem"])
    ap.add_argument("--action-mode", type=str, default="global", choices=["global", "two_region"])
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--effort-lambda", type=float, default=0.0)
    ap.add_argument("--apply-each-step", action="store_true")
    ap.add_argument("--apply-all-cells", action="store_true")
    ap.add_argument("--clip-lo", type=float, default=0.0)
    ap.add_argument("--clip-hi", type=float, default=None)
    ap.add_argument("--round", action="store_true")

    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.35)

    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--init-sigma", type=float, default=0.6)
    ap.add_argument("--min-sigma", type=float, default=0.05)

    ap.add_argument("--max-abs-log", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(Path("trials") / "mpc_results.json"))
    args = ap.parse_args()

    base_payload = json.loads(Path(args.gridstate).read_text())

    prefixes: Tuple[str, ...]
    if args.control_mode == "gene":
        prefixes = ("gene_",)
    elif args.control_mode == "rna":
        prefixes = ("rna_",)
    elif args.control_mode == "protein":
        prefixes = ("protein_",)
    elif args.control_mode == "molecule":
        prefixes = ("molecule_",)
    else:
        prefixes = ("gene_", "rna_", "protein_", "molecule_")

    layer_names = _layers_by_prefix(base_payload, prefixes)
    if not layer_names:
        raise ValueError(f"No layers found for control_mode={args.control_mode!r}")

    expr = _get_measurement_expr(base_payload, args.measurement)
    if expr is None:
        raise ValueError(f"Measurement {args.measurement!r} not found or not v3 expr")

    apply_cell_mask = not bool(args.apply_all_cells)
    if args.control_mode == "molecule" and not bool(args.apply_all_cells):
        apply_cell_mask = False

    clip_hi = float(args.clip_hi) if args.clip_hi is not None else _compute_default_clip_hi(base_payload, layer_names, apply_cell_mask=apply_cell_mask)
    clip_lo = float(args.clip_lo)

    if args.clip_hi is not None:
        default_hi = _compute_default_clip_hi(base_payload, layer_names, apply_cell_mask=apply_cell_mask)
        if float(args.clip_hi) < 0.5 * float(default_hi):
            print(
                f"WARNING: --clip-hi={float(args.clip_hi)} looks small versus inferred scale ({default_hi}); may clamp search.",
                file=sys.stderr,
            )

    round_counts = bool(args.round)
    if args.control_mode == "molecule" and args.round is False:
        round_counts = False

    t0 = time.time()

    dim = len(layer_names) if args.action_mode == "global" else 2 * len(layer_names)

    baseline = _rollout(
        base_payload,
        layer_names,
        np.zeros((dim,), dtype=np.float32),
        horizon=int(args.horizon),
        seed=int(args.seed),
        measurement_expr=expr,
        gamma=float(args.gamma),
        effort_lambda=float(args.effort_lambda),
        apply_each_step=bool(args.apply_each_step),
        action_mode=str(args.action_mode),
        apply_cell_mask=apply_cell_mask,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        round_counts=round_counts,
    )

    if args.optimizer == "random":
        best, rows = _optimize_random(
            base_payload,
            layer_names,
            horizon=int(args.horizon),
            seed=int(args.seed),
            measurement_expr=expr,
            gamma=float(args.gamma),
            effort_lambda=float(args.effort_lambda),
            samples=int(args.samples),
            sigma=float(args.sigma),
            max_abs_log=float(args.max_abs_log),
            apply_each_step=bool(args.apply_each_step),
            action_mode=str(args.action_mode),
            apply_cell_mask=apply_cell_mask,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            round_counts=round_counts,
        )
    else:
        best, rows = _optimize_cem(
            base_payload,
            layer_names,
            horizon=int(args.horizon),
            seed=int(args.seed),
            measurement_expr=expr,
            gamma=float(args.gamma),
            effort_lambda=float(args.effort_lambda),
            pop_size=int(args.pop),
            iters=int(args.iters),
            elite_frac=float(args.elite_frac),
            init_sigma=float(args.init_sigma),
            min_sigma=float(args.min_sigma),
            max_abs_log=float(args.max_abs_log),
            apply_each_step=bool(args.apply_each_step),
            action_mode=str(args.action_mode),
            apply_cell_mask=apply_cell_mask,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
            round_counts=round_counts,
        )

    dt_ms = (time.time() - t0) * 1000.0

    out_obj = {
        "gridstate": str(args.gridstate),
        "measurement": str(args.measurement),
        "expr": expr,
        "control_mode": str(args.control_mode),
        "optimizer": str(args.optimizer),
        "layer_names": layer_names,
        "params": {
            "horizon": int(args.horizon),
            "gamma": float(args.gamma),
            "effort_lambda": float(args.effort_lambda),
            "apply_each_step": bool(args.apply_each_step),
            "action_mode": str(args.action_mode),
            "apply_cell_mask": apply_cell_mask,
            "clip_lo": clip_lo,
            "clip_hi": clip_hi,
            "round_counts": round_counts,
            "seed": int(args.seed),
            "random": {"samples": int(args.samples), "sigma": float(args.sigma)},
            "cem": {
                "pop": int(args.pop),
                "iters": int(args.iters),
                "elite_frac": float(args.elite_frac),
                "init_sigma": float(args.init_sigma),
                "min_sigma": float(args.min_sigma),
            },
            "max_abs_log": float(args.max_abs_log),
        },
        "baseline": {"objective": baseline.objective, "health": baseline.health},
        "best": {
            "objective": best.objective,
            "health": best.health,
            "log_multipliers": best.log_multipliers,
            "multipliers": [float(math.exp(x)) for x in best.log_multipliers],
            "by_layer": (
                {layer_names[i]: float(math.exp(best.log_multipliers[i])) for i in range(len(layer_names))}
                if str(args.action_mode) == "global"
                else {
                    layer_names[i]: {
                        "inside": float(math.exp(best.log_multipliers[2 * i])),
                        "outside": float(math.exp(best.log_multipliers[2 * i + 1])),
                    }
                    for i in range(len(layer_names))
                }
            ),
        },
        "rows": rows,
        "runtime_ms": dt_ms,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))

    print(
        json.dumps(
            {
                "baseline_obj": baseline.objective,
                "best_obj": best.objective,
                "baseline_h1": baseline.health[0] if baseline.health else None,
                "best_h1": best.health[0] if best.health else None,
                "baseline_hH": baseline.health[-1] if baseline.health else None,
                "best_hH": best.health[-1] if best.health else None,
                "runtime_ms": dt_ms,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
