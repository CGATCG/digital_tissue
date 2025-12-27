from __future__ import annotations

import argparse
import ast
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np


@dataclass(frozen=True)
class MeasurementSpec:
    output_name: str
    layer_name: str


MEASUREMENTS: List[MeasurementSpec] = [
    MeasurementSpec(output_name="inflammation", layer_name="cytokine"),
    MeasurementSpec(output_name="glucose", layer_name="glucose"),
    MeasurementSpec(output_name="protein", layer_name="amino_acids"),
    MeasurementSpec(output_name="toxins", layer_name="toxins"),
    MeasurementSpec(output_name="bacterial_infection", layer_name="bacterial_antigen"),
]


def _default_config_dict() -> dict:
    return {
        "version": 3,
        "measurements": [
            {"name": "inflammation", "expr": "mean(cytokine, where=(circulation==1))"},
            {"name": "glucose_per_circ_cell", "expr": "sum(glucose, where=(circulation==1)) / count(where=(circulation==1))"},
            {"name": "protein_per_circ_cell", "expr": "sum(amino_acids, where=(circulation==1)) / count(where=(circulation==1))"},
            {"name": "toxins_per_circ_cell", "expr": "sum(toxins, where=(circulation==1)) / count(where=(circulation==1))"},
            {"name": "bacterial_infection", "expr": "mean(bacterial_antigen, where=(circulation==1))"},
        ],
    }


def _load_config_json(path: Path) -> Tuple[str, Optional[dict], List[dict]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Invalid config: root must be an object")
    version = payload.get("version")
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported config version: {version!r} (expected 1, 2, or 3)")

    # v3: expression-based measurements, no global mask
    if version == 3:
        specs_raw = payload.get("measurements")
        if not isinstance(specs_raw, list):
            raise ValueError("Invalid config v3: 'measurements' must be a list")
        out: List[dict] = []
        for s in specs_raw:
            if not isinstance(s, dict):
                continue
            name = s.get("name")
            expr = s.get("expr")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(expr, str) or not expr.strip():
                continue
            out.append({"name": name.strip(), "expr": expr})
        if not out:
            raise ValueError("Invalid config v3: no valid measurements")
        return "expr", None, out

    # v1/v2: op-based calculations with a global mask
    mask = payload.get("mask")
    if not isinstance(mask, dict):
        raise ValueError("Invalid config: 'mask' must be an object")
    layer = mask.get("layer")
    op = mask.get("op")
    value = mask.get("value")
    invert = mask.get("invert", False)
    if not isinstance(layer, str) or not layer.strip():
        raise ValueError("Invalid config: mask.layer must be a non-empty string")
    if op not in {"==", "!=", ">", ">=", "<", "<="}:
        raise ValueError("Invalid config: mask.op must be one of == != > >= < <=")
    try:
        value_f = float(value)
    except Exception:
        raise ValueError("Invalid config: mask.value must be a number")
    if not isinstance(invert, bool):
        raise ValueError("Invalid config: mask.invert must be a boolean")

    calcs: List[dict] = []
    if version == 1:
        specs_raw = payload.get("measurements")
        if not isinstance(specs_raw, list):
            raise ValueError("Invalid config v1: 'measurements' must be a list")
        for s in specs_raw:
            if not isinstance(s, dict):
                continue
            outn = s.get("output")
            lay = s.get("layer")
            if not isinstance(outn, str) or not outn.strip():
                continue
            if not isinstance(lay, str) or not lay.strip():
                continue
            calcs.append({"output": outn, "op": "masked_mean", "layer": lay})
    else:
        specs_raw = payload.get("calculations")
        if not isinstance(specs_raw, list):
            raise ValueError("Invalid config v2: 'calculations' must be a list")
        for s in specs_raw:
            if not isinstance(s, dict):
                continue
            outn = s.get("output")
            op_name = s.get("op")
            if not isinstance(outn, str) or not outn.strip():
                continue
            if not isinstance(op_name, str) or not op_name.strip():
                continue

            layer_a = s.get("layer")
            layer_b = s.get("layer_b")
            calc = {"output": outn, "op": op_name}
            if isinstance(layer_a, str) and layer_a.strip():
                calc["layer"] = layer_a
            if isinstance(layer_b, str) and layer_b.strip():
                calc["layer_b"] = layer_b
            calcs.append(calc)

    if not calcs:
        raise ValueError("Invalid config: no valid calculations")

    return "calc", {"layer": layer, "op": op, "value": value_f, "invert": invert}, calcs


def _expr_required_layer_names(expr: str) -> List[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
    # Do not filter here; caller can decide which are layers.
    return names


class _ExprEval:
    def __init__(self, layers: Dict[str, np.ndarray], H: int, W: int):
        self.layers = layers
        self.H = H
        self.W = W

    def _as_array(self, v) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        raise ValueError("Expected array")

    def _as_float(self, v) -> float:
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        raise ValueError("Expected number")

    def _ensure_bool_mask(self, m) -> np.ndarray:
        if not isinstance(m, np.ndarray):
            raise ValueError("where must evaluate to an array")
        if m.dtype != np.bool_:
            m = m.astype(bool)
        return m

    def _masked_reduce(self, arr: np.ndarray, where: Optional[np.ndarray], fn: str, q: Optional[float] = None) -> Optional[float]:
        if where is None:
            m = np.ones((self.H, self.W), dtype=bool)
        else:
            m = self._ensure_bool_mask(where)
        n = int(m.sum())
        if n == 0:
            return None
        a = arr[m]
        if fn == "mean":
            return float(a.mean())
        if fn == "sum":
            return float(a.sum())
        if fn == "min":
            return float(a.min())
        if fn == "max":
            return float(a.max())
        if fn == "std":
            return float(a.std())
        if fn == "var":
            return float(a.var())
        if fn == "median":
            return float(np.median(a))
        if fn == "quantile":
            if q is None:
                raise ValueError("quantile requires q")
            return float(np.quantile(a, q))
        raise ValueError(f"Unknown reduction: {fn}")

    def eval(self, expr: str) -> Optional[float]:
        tree = ast.parse(expr, mode="eval")
        v = self._eval_node(tree.body)
        if isinstance(v, np.ndarray):
            raise ValueError("Expression evaluated to an array; expected scalar")
        if v is None:
            return None
        return float(v)

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ("True", "False"):
                return bool(nm == "True")
            if nm in self.layers:
                return self.layers[nm].reshape(self.H, self.W)
            raise ValueError(f"Unknown identifier: {nm}")

        if isinstance(node, ast.UnaryOp):
            v = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -self._as_float(v)
            if isinstance(node.op, ast.UAdd):
                return +self._as_float(v)
            if isinstance(node.op, ast.Invert):
                a = self._as_array(v)
                return np.logical_not(a.astype(bool))
            if isinstance(node.op, ast.Not):
                if isinstance(v, np.ndarray):
                    return np.logical_not(v.astype(bool))
                return not bool(v)
            raise ValueError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            # mask composition
            if isinstance(node.op, ast.BitAnd):
                return self._as_array(left).astype(bool) & self._as_array(right).astype(bool)
            if isinstance(node.op, ast.BitOr):
                return self._as_array(left).astype(bool) | self._as_array(right).astype(bool)

            # scalar arithmetic only
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                raise ValueError("Array arithmetic is not supported at the top level; use reductions like mean(x)")

            a = self._as_float(left)
            b = self._as_float(right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return a**b
            raise ValueError("Unsupported binary operator")

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Chained comparisons are not supported")
            left = self._eval_node(node.left)
            right = self._eval_node(node.comparators[0])
            op = node.ops[0]

            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                if left.shape != right.shape:
                    raise ValueError("Array comparison requires same shape")
                if isinstance(op, ast.Eq):
                    return left == right
                if isinstance(op, ast.NotEq):
                    return left != right
                if isinstance(op, ast.Gt):
                    return left > right
                if isinstance(op, ast.GtE):
                    return left >= right
                if isinstance(op, ast.Lt):
                    return left < right
                if isinstance(op, ast.LtE):
                    return left <= right

            if isinstance(left, np.ndarray) and not isinstance(right, np.ndarray):
                r = np.float32(self._as_float(right))
                if isinstance(op, ast.Eq):
                    return left == r
                if isinstance(op, ast.NotEq):
                    return left != r
                if isinstance(op, ast.Gt):
                    return left > r
                if isinstance(op, ast.GtE):
                    return left >= r
                if isinstance(op, ast.Lt):
                    return left < r
                if isinstance(op, ast.LtE):
                    return left <= r

            if not isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                l = np.float32(self._as_float(left))
                if isinstance(op, ast.Eq):
                    return l == right
                if isinstance(op, ast.NotEq):
                    return l != right
                if isinstance(op, ast.Gt):
                    return l > right
                if isinstance(op, ast.GtE):
                    return l >= right
                if isinstance(op, ast.Lt):
                    return l < right
                if isinstance(op, ast.LtE):
                    return l <= right

            if not isinstance(left, np.ndarray) and not isinstance(right, np.ndarray):
                a = self._as_float(left)
                b = self._as_float(right)
                if isinstance(op, ast.Eq):
                    return a == b
                if isinstance(op, ast.NotEq):
                    return a != b
                if isinstance(op, ast.Gt):
                    return a > b
                if isinstance(op, ast.GtE):
                    return a >= b
                if isinstance(op, ast.Lt):
                    return a < b
                if isinstance(op, ast.LtE):
                    return a <= b

            raise ValueError("Invalid comparison")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            fn = node.func.id
            allowed = {"mean", "sum", "min", "max", "std", "var", "median", "quantile", "count"}
            if fn not in allowed:
                raise ValueError(f"Unsupported function: {fn}")

            kwargs = {kw.arg: self._eval_node(kw.value) for kw in node.keywords if kw.arg}
            where = kwargs.get("where")
            if where is not None:
                where = self._ensure_bool_mask(where)

            if fn == "count":
                if node.args:
                    raise ValueError("count() does not take positional args")
                if where is None:
                    return float(self.H * self.W)
                return float(int(where.sum()))

            if not node.args:
                raise ValueError(f"{fn}() requires a layer argument")
            arr = self._as_array(self._eval_node(node.args[0]))

            q = None
            if fn == "quantile":
                if len(node.args) >= 2:
                    q = self._as_float(self._eval_node(node.args[1]))
                elif "q" in kwargs:
                    q = self._as_float(kwargs["q"])
                else:
                    raise ValueError("quantile() requires q")
            return self._masked_reduce(arr, where=where, fn=fn, q=q)

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _calc_required_layers(calc: dict) -> List[str]:
    op = str(calc.get("op") or "")
    a = calc.get("layer")
    b = calc.get("layer_b")
    needs: List[str] = []
    if op in ("masked_mean", "masked_sum", "masked_sum_over_count"):
        if isinstance(a, str) and a:
            needs.append(a)
    if op == "masked_sum_ratio":
        if isinstance(a, str) and a:
            needs.append(a)
        if isinstance(b, str) and b:
            needs.append(b)
    return needs


def _masked_sum(arr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    n = int(mask.sum())
    if n == 0:
        return None
    return float(arr[mask].sum())


def _eval_calc(calc: dict, layers: Dict[str, np.ndarray], mask: np.ndarray, H: int, W: int) -> Optional[float]:
    op = str(calc.get("op") or "")
    if op == "mask_count":
        return float(int(mask.sum()))

    layer = calc.get("layer")
    if op in ("masked_mean", "masked_sum", "masked_sum_over_count"):
        if not isinstance(layer, str) or layer not in layers:
            return None
        arr = layers[layer].reshape(H, W)
        if op == "masked_mean":
            return _masked_mean(arr, mask)
        s = _masked_sum(arr, mask)
        if s is None:
            return None
        if op == "masked_sum":
            return s
        denom = int(mask.sum())
        if denom == 0:
            return None
        return float(s / float(denom))

    if op == "masked_sum_ratio":
        layer_b = calc.get("layer_b")
        if not isinstance(layer, str) or layer not in layers:
            return None
        if not isinstance(layer_b, str) or layer_b not in layers:
            return None
        a = layers[layer].reshape(H, W)
        b = layers[layer_b].reshape(H, W)
        sa = _masked_sum(a, mask)
        sb = _masked_sum(b, mask)
        if sa is None or sb is None or sb == 0:
            return None
        return float(sa / sb)

    raise ValueError(f"Unsupported calculation op: {op!r}")


def _apply_mask_op(arr: np.ndarray, op: str, value: float) -> np.ndarray:
    if op == "==":
        return arr == np.float32(value)
    if op == "!=":
        return arr != np.float32(value)
    if op == ">":
        return arr > np.float32(value)
    if op == ">=":
        return arr >= np.float32(value)
    if op == "<":
        return arr < np.float32(value)
    if op == "<=":
        return arr <= np.float32(value)
    raise ValueError(f"Unsupported mask op: {op!r}")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _decode_float32_b64(b64: str, expected_len: int, layer_name: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise ValueError(f"Failed to base64-decode layer '{layer_name}': {e}")

    if len(raw) % 4 != 0:
        raise ValueError(
            f"Layer '{layer_name}' has invalid byte length {len(raw)} (not divisible by 4 for float32)"
        )

    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size != expected_len:
        raise ValueError(
            f"Layer '{layer_name}' has length {arr.size}, expected {expected_len} (H*W)"
        )
    return arr


def _load_gridstate_json(path: Path) -> Tuple[int, int, Dict[str, np.ndarray], Optional[dict]]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise ValueError(f"Failed to parse JSON '{path}': {e}")

    if not isinstance(payload, dict):
        raise ValueError("Invalid gridstate.json: root must be an object")

    version = payload.get("version")
    if version != 1:
        raise ValueError(f"Unsupported gridstate version: {version!r} (expected 1)")

    try:
        H = int(payload["H"])
        W = int(payload["W"])
    except Exception:
        raise ValueError("Invalid gridstate.json: missing/invalid H/W")

    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid H/W: {H}x{W}")

    expected_len = H * W

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Invalid gridstate.json: 'data' must be an object")

    embedded_cfg = payload.get("measurements_config")
    if embedded_cfg is not None and not isinstance(embedded_cfg, dict):
        embedded_cfg = None

    layers: Dict[str, np.ndarray] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        dtype = entry.get("dtype")
        b64 = entry.get("b64")
        if dtype != "float32" or not isinstance(b64, str):
            continue
        layers[str(name)] = _decode_float32_b64(b64, expected_len=expected_len, layer_name=str(name))

    return H, W, layers, embedded_cfg


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    n = int(mask.sum())
    if n == 0:
        return None
    return float(arr[mask].mean())


def compute_measurements(gridstate_path: Path, config_path: Optional[Path]) -> int:
    H, W, layers, embedded_cfg = _load_gridstate_json(gridstate_path)

    if config_path is None:
        if embedded_cfg is not None:
            cfg_mode, mask_cfg, calcs = _load_config_json_payload(embedded_cfg)
        else:
            mask_cfg = {"layer": "circulation", "op": "==", "value": 1.0, "invert": False}
            calcs = [{"output": m.output_name, "op": "masked_mean", "layer": m.layer_name} for m in MEASUREMENTS]
            cfg_mode = "calc"
    else:
        cfg_mode, mask_cfg, calcs = _load_config_json(config_path)

    missing: List[str] = []
    if cfg_mode == "calc":
        assert mask_cfg is not None
        if mask_cfg["layer"] not in layers:
            missing.append(mask_cfg["layer"])
        for calc in calcs:
            for req in _calc_required_layers(calc):
                if req not in layers:
                    missing.append(req)
    else:
        # best-effort pre-scan for unknown layers
        allowed = {"mean", "sum", "min", "max", "std", "var", "median", "quantile", "count", "where", "True", "False"}
        for m in calcs:
            expr = str(m.get("expr") or "")
            for nm in _expr_required_layer_names(expr):
                if nm in allowed:
                    continue
                if nm not in layers:
                    missing.append(nm)

    if missing:
        _eprint("Missing required layer(s):")
        for nm in sorted(set(missing)):
            _eprint(f"- {nm}")

    print(f"grid: {H}x{W}")
    results: Dict[str, Optional[float]] = {}

    if cfg_mode == "calc":
        assert mask_cfg is not None
        if mask_cfg["layer"] not in layers:
            _eprint(f"Cannot compute measurements without mask layer '{mask_cfg['layer']}'.")
            return 2
        mask_src = layers[mask_cfg["layer"]].reshape(H, W)
        mask = _apply_mask_op(mask_src, mask_cfg["op"], float(mask_cfg["value"]))
        if bool(mask_cfg["invert"]):
            mask = ~mask
        n_mask = int(mask.sum())
        inv_txt = " (inverted)" if bool(mask_cfg["invert"]) else ""
        print(f"mask: {mask_cfg['layer']}{mask_cfg['op']}{mask_cfg['value']}{inv_txt} cells: {n_mask}")
        if n_mask == 0:
            _eprint("No cells in mask; outputs are undefined.")

        for calc in calcs:
            out_name = str(calc.get("output") or "").strip()
            if not out_name:
                continue
            try:
                results[out_name] = _eval_calc(calc, layers, mask, H=H, W=W)
            except ValueError:
                results[out_name] = None

        print("\ncalculations:")
        for calc in calcs:
            out_name = str(calc.get("output") or "").strip()
            if not out_name:
                continue
            v = results.get(out_name)
            if v is None:
                print(f"- {out_name}: MISSING")
            else:
                print(f"- {out_name}: {v:.6g}")
        return 0

    # v3 expression mode
    ev = _ExprEval(layers=layers, H=H, W=W)
    errors: Dict[str, str] = {}
    for m in calcs:
        nm = str(m.get("name") or "").strip()
        expr = str(m.get("expr") or "")
        if not nm:
            continue
        try:
            results[nm] = ev.eval(expr)
        except Exception as e:
            results[nm] = None
            errors[nm] = str(e)

    if errors:
        _eprint("Expression errors:")
        for k in sorted(errors.keys()):
            _eprint(f"- {k}: {errors[k]}")

    print("\nmeasurements:")
    for m in calcs:
        nm = str(m.get("name") or "").strip()
        if not nm:
            continue
        v = results.get(nm)
        if v is None:
            print(f"- {nm}: MISSING")
        else:
            print(f"- {nm}: {v:.6g}")

    return 0


def _load_config_json_payload(payload: Dict[str, Any]) -> Tuple[str, Optional[dict], List[dict]]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid config: root must be an object")
    version = payload.get("version")
    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported config version: {version!r} (expected 1, 2, or 3)")

    # v3: expression-based measurements, no global mask
    if version == 3:
        specs_raw = payload.get("measurements")
        if not isinstance(specs_raw, list):
            raise ValueError("Invalid config v3: 'measurements' must be a list")
        out: List[dict] = []
        for s in specs_raw:
            if not isinstance(s, dict):
                continue
            name = s.get("name")
            expr = s.get("expr")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(expr, str) or not expr.strip():
                continue
            out.append({"name": name.strip(), "expr": expr})
        if not out:
            raise ValueError("Invalid config v3: no valid measurements")
        return "expr", None, out

    # v1/v2: op-based calculations with a global mask
    mask = payload.get("mask")
    if not isinstance(mask, dict):
        raise ValueError("Invalid config: 'mask' must be an object")
    layer = mask.get("layer")
    op = mask.get("op")
    value = mask.get("value")
    invert = mask.get("invert", False)
    if not isinstance(layer, str) or not layer.strip():
        raise ValueError("Invalid config: mask.layer must be a non-empty string")
    if op not in {"==", "!=", ">", ">=", "<", "<="}:
        raise ValueError("Invalid config: mask.op must be one of == != > >= < <=")
    try:
        value_f = float(value)
    except Exception:
        raise ValueError("Invalid config: mask.value must be a number")
    if not isinstance(invert, bool):
        raise ValueError("Invalid config: mask.invert must be a boolean")

    calcs: List[dict] = []
    if version == 1:
        specs_raw = payload.get("measurements")
        if not isinstance(specs_raw, list):
            raise ValueError("Invalid config v1: 'measurements' must be a list")
        for s in specs_raw:
            if not isinstance(s, dict):
                continue
            outn = s.get("output")
            lay = s.get("layer")
            if not isinstance(outn, str) or not outn.strip():
                continue
            if not isinstance(lay, str) or not lay.strip():
                continue
            calcs.append({"output": outn, "op": "masked_mean", "layer": lay})
    else:
        specs_raw = payload.get("calculations")
        if not isinstance(specs_raw, list):
            raise ValueError("Invalid config v2: 'calculations' must be a list")
        for s in specs_raw:
            if not isinstance(s, dict):
                continue
            outn = s.get("output")
            op_name = s.get("op")
            if not isinstance(outn, str) or not outn.strip():
                continue
            if not isinstance(op_name, str) or not op_name.strip():
                continue

            layer_a = s.get("layer")
            layer_b = s.get("layer_b")
            calc = {"output": outn, "op": op_name}
            if isinstance(layer_a, str) and layer_a.strip():
                calc["layer"] = layer_a
            if isinstance(layer_b, str) and layer_b.strip():
                calc["layer_b"] = layer_b
            calcs.append(calc)

    if not calcs:
        raise ValueError("Invalid config: no valid calculations")

    return "calc", {"layer": layer, "op": op, "value": value_f, "invert": invert}, calcs


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute circulation-masked summary measurements from gridstate.json")
    p.add_argument(
        "--path",
        type=Path,
        default=Path("array/gridstate.json"),
        help="Path to gridstate.json (default: array/gridstate.json)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to measurements config JSON (default: none; uses built-in defaults)",
    )
    p.add_argument(
        "--write-default-config",
        type=Path,
        default=None,
        help="Write a default measurements config JSON to this path and exit",
    )
    args = p.parse_args(argv)

    try:
        if args.write_default_config is not None:
            args.write_default_config.write_text(json.dumps(_default_config_dict(), indent=2) + "\n")
            return 0
        return compute_measurements(args.path, args.config)
    except Exception as e:
        _eprint(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
