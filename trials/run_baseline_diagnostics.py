import argparse
import base64
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apply_layer_ops import apply_layer_ops_inplace  # noqa: E402


class _DefaultRngPatch:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._orig = None

    def __enter__(self):
        self._orig = np.random.default_rng

        def _patched_default_rng(*args, **kwargs):
            if args or kwargs:
                return self._orig(*args, **kwargs)
            return self._orig(self.seed)

        np.random.default_rng = _patched_default_rng
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            np.random.default_rng = self._orig
        return False


def _decode_f32(b64: str, expected_len: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.float32)
    if int(arr.size) != int(expected_len):
        raise ValueError(f"decoded length mismatch: got {arr.size}, expected {expected_len}")
    return arr


def _get_layer2d(payload: Dict[str, Any], name: str) -> np.ndarray:
    H = int(payload["H"])
    W = int(payload["W"])
    data = payload["data"]
    ent = data.get(name)
    if not isinstance(ent, dict) or ent.get("dtype") != "float32":
        raise KeyError(name)
    b64 = ent.get("b64")
    if not isinstance(b64, str):
        raise KeyError(name)
    return _decode_f32(b64, H * W).reshape(H, W)


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.asarray(x, dtype=np.float64).mean())


def _safe_min(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.asarray(x, dtype=np.float64).min())


def _safe_max(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.asarray(x, dtype=np.float64).max())


def _metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    cell = np.rint(_get_layer2d(payload, "cell")).astype(np.int64)
    morph = np.rint(_get_layer2d(payload, "morphology")).astype(np.int64)
    is_cell = cell == 1

    H = int(payload["H"])
    W = int(payload["W"])
    n_total = int(H * W)
    eq = cell == morph
    n_match = int(eq.sum())
    n_mismatch = int(n_total - n_match)
    n_cell_outside_morph = int(((cell == 1) & (morph != 1)).sum())
    n_morph_without_cell = int(((morph == 1) & (cell != 1)).sum())

    mean_match = float((cell == morph).mean())
    health = float(math.pow(mean_match, 20))

    atp = _get_layer2d(payload, "molecule_atp")
    glu = _get_layer2d(payload, "molecule_glucose")
    dmg = _get_layer2d(payload, "damage")
    atp_maker = _get_layer2d(payload, "protein_atp_maker")
    fixer = _get_layer2d(payload, "protein_fixer")

    atp_generated = np.where(is_cell, np.clip(glu, 0, atp_maker), 0.0).astype(np.float32)

    def _frac_lt(x: np.ndarray, thr: float) -> float:
        if x.size == 0:
            return float("nan")
        return float((np.asarray(x) < float(thr)).mean())

    atp_cell = atp[is_cell]
    glu_cell = glu[is_cell]
    dmg_cell = dmg[is_cell]
    gen_cell = atp_generated[is_cell]
    fixer_cell = fixer[is_cell]

    repair_req = fixer_cell * 0.01
    repair_act = np.minimum(np.minimum(repair_req, dmg_cell), atp_cell / 100.0)

    return {
        "n_cell": int(is_cell.sum()),
        "n_total": n_total,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "n_cell_outside_morph": n_cell_outside_morph,
        "n_morph_without_cell": n_morph_without_cell,
        "mean_match": mean_match,
        "morphological_health": health,
        "atp_mean": _safe_mean(atp_cell),
        "atp_min": _safe_min(atp_cell),
        "atp_max": _safe_max(atp_cell),
        "frac_atp_lt1": _frac_lt(atp_cell, 1.0),
        "glucose_mean": _safe_mean(glu_cell),
        "glucose_min": _safe_min(glu_cell),
        "damage_mean": _safe_mean(dmg_cell),
        "damage_max": _safe_max(dmg_cell),
        "protein_fixer_mean": _safe_mean(fixer_cell),
        "protein_fixer_max": _safe_max(fixer_cell),
        "repair_act_mean": _safe_mean(repair_act),
        "repair_act_max": _safe_max(repair_act),
        "atp_generated_mean": _safe_mean(gen_cell),
        "atp_generated_max": _safe_max(gen_cell),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridstate", type=str, default=str(Path("array") / "gridstate.json"))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(Path("trials") / "baseline_diagnostics.json"))
    args = ap.parse_args()

    payload = json.loads(Path(args.gridstate).read_text())

    rows = []
    rows.append({"t": 0, **_metrics(payload)})
    for t in range(int(args.steps)):
        with _DefaultRngPatch(int(args.seed) + 10007 * t):
            apply_layer_ops_inplace(payload, export_let_layers=False, seed_offset=0)
        rows.append({"t": t + 1, **_metrics(payload)})

    out_obj: Dict[str, Any] = {
        "gridstate": str(args.gridstate),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(json.dumps(out_obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
