import argparse
import base64
import fnmatch
import json
import time
import functools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import copy
import re

import numpy as np

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


def _encode_float32_b64(arr: np.ndarray) -> str:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _load_gridstate_payload(path: Path) -> Dict[str, Any]:
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

    return payload


def _extract_layers(payload: Dict[str, Any]) -> Tuple[int, int, Dict[str, np.ndarray], Dict[str, str]]:
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

    layer_meta = payload.get("layers")
    if not isinstance(layer_meta, list):
        raise ValueError("Invalid gridstate.json: 'layers' must be a list")

    kinds: Dict[str, str] = {}
    for meta in layer_meta:
        if not isinstance(meta, dict):
            continue
        name = meta.get("name")
        kind = meta.get("kind")
        if isinstance(name, str) and isinstance(kind, str):
            kinds[name] = kind

    layers: Dict[str, np.ndarray] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        dtype = entry.get("dtype")
        b64 = entry.get("b64")
        if dtype != "float32" or not isinstance(b64, str):
            continue
        layers[str(name)] = _decode_float32_b64(b64, expected_len=expected_len, layer_name=str(name))

    if not layers:
        raise ValueError("No float32 layers found in gridstate.json")

    return H, W, layers, kinds


_ALLOWED_FUNCS = {
    "where": np.where,
    "clip": np.clip,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "minimum": np.minimum,
    "maximum": np.maximum,
}


def _validate_expr_ast(expr: str, env: Dict[str, Any]) -> None:
    import ast

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expr syntax: {e}")

    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Subscript,
        ast.Slice,
        ast.Tuple,
        ast.List,
        # operator / comparator nodes (these appear in ast.walk)
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        ast.And,
        ast.Or,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            raise ValueError(f"Disallowed syntax in expr: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")
            fn = node.func.id
            if fn not in env or not callable(env.get(fn)):
                raise ValueError(f"Function '{fn}' is not allowed")

        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"Unknown identifier '{node.id}'")

        # explicitly forbid attribute access (np.where etc.)
        if hasattr(ast, "Attribute") and isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed")


def _eval_expr(expr: str, env: Dict[str, Any]) -> np.ndarray:
    import ast

    _validate_expr_ast(expr, env)
    code = compile(ast.parse(expr, mode="eval"), "<layer_op>", "eval")
    out = eval(code, {"__builtins__": {}}, env)  # noqa: S307
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


@functools.lru_cache(maxsize=512)
def _compile_expr_cached(expr: str) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    import ast

    tree = ast.parse(expr, mode="eval")

    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Subscript,
        ast.Slice,
        ast.Tuple,
        ast.List,
        # operator / comparator nodes (these appear in ast.walk)
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        ast.And,
        ast.Or,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    names: set[str] = set()
    call_fns: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            raise ValueError(f"Disallowed syntax in expr: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")
            call_fns.add(node.func.id)

        if isinstance(node, ast.Name):
            names.add(node.id)

        if hasattr(ast, "Attribute") and isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed")

    code = compile(tree, "<layer_op>", "eval")
    return code, tuple(sorted(names)), tuple(sorted(call_fns))


def _eval_expr_fast(expr: str, env: Dict[str, Any]) -> np.ndarray:
    code, names, call_fns = _compile_expr_cached(expr)
    for nm in names:
        if nm not in env:
            raise ValueError(f"Unknown identifier '{nm}'")
    for fn in call_fns:
        if fn not in env or not callable(env.get(fn)):
            raise ValueError(f"Function '{fn}' is not allowed")
    out = eval(code, {"__builtins__": {}}, env)  # noqa: S307
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


def _subst_template(s: str, m: re.Match) -> str:
    def repl(mm: re.Match) -> str:
        try:
            idx = int(mm.group(1))
        except Exception:
            return mm.group(0)
        try:
            g = m.group(idx)
        except Exception:
            return ""
        return "" if g is None else str(g)

    return re.sub(r"\$\{(\d+)\}", repl, str(s))


def _escape_regex_literal(s: str) -> str:
    return re.sub(r"([\\^$.|?*+()\[\]{}])", r"\\\1", str(s))


def _glob_to_regex_source_with_groups(glob: str) -> str:
    s = str(glob or "")
    out = "^"
    i = 0
    while i < len(s):
        ch = s[i]

        if ch == "\\":
            if i + 1 < len(s):
                out += _escape_regex_literal(s[i + 1])
                i += 2
                continue
            out += "\\\\"
            i += 1
            continue

        if ch == "*":
            out += "(.*?)"
            i += 1
            continue
        if ch == "?":
            out += "(.)"
            i += 1
            continue

        if ch == "[":
            j = i + 1
            while j < len(s) and s[j] != "]":
                j += 1
            if j < len(s):
                cls = s[i + 1 : j]
                if cls.startswith("!"):
                    cls = "^" + cls[1:]
                out += "[" + cls + "]"
                i = j + 1
                continue
            out += "\\["
            i += 1
            continue

        out += _escape_regex_literal(ch)
        i += 1

    out += "$"
    return out


def _is_glob_like_pattern(pat: str) -> bool:
    p = str(pat or "").strip()
    if not p:
        return False
    if p.startswith("glob:"):
        return True
    if p.startswith("re:") or p.startswith("regex:"):
        return False
    if not re.search(r"[\*\?\[]", p):
        return False
    if re.search(r"[\(\)\|\+\{\}]", p):
        return False
    if ".*" in p:
        return False
    return True


def _expand_foreach_steps(
    steps: List[Dict[str, Any]],
    layer_names: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("enabled", True) is False:
            continue

        step_type = str(step.get("type") or "op").strip().lower()
        if step_type != "foreach":
            out.append(step)
            continue

        raw_pat = str(step.get("match") or "").strip()
        if not raw_pat:
            raise ValueError(f"Step #{i}: foreach missing 'match' pattern")
        try:
            if raw_pat.startswith("glob:"):
                glob_pat = raw_pat[len("glob:") :].strip()
                rx = re.compile(_glob_to_regex_source_with_groups(glob_pat))
            elif raw_pat.startswith("re:") or raw_pat.startswith("regex:"):
                pat = raw_pat[3:].strip() if raw_pat.startswith("re:") else raw_pat[len("regex:") :].strip()
                rx = re.compile(pat)
            elif _is_glob_like_pattern(raw_pat):
                rx = re.compile(_glob_to_regex_source_with_groups(raw_pat))
            else:
                rx = re.compile(raw_pat)
        except Exception as e:
            raise ValueError(f"Step #{i}: invalid foreach match pattern {raw_pat!r}: {e}") from e

        template_steps = step.get("steps")
        if not isinstance(template_steps, list) or not template_steps:
            raise ValueError(f"Step #{i}: foreach missing non-empty 'steps' array")

        matched_any = False
        for layer_name in layer_names:
            m = rx.match(layer_name)
            if not m:
                continue
            matched_any = True

            for j, tstep in enumerate(template_steps):
                if not isinstance(tstep, dict):
                    continue
                expanded = copy.deepcopy(tstep)
                expanded_type = str(expanded.get("type") or "op").strip().lower()
                if expanded_type not in ("op", "let"):
                    raise ValueError(
                        f"Step #{i}: foreach substep #{j} has invalid type {expanded_type!r} (expected 'op' or 'let')"
                    )

                for k in ("name", "group", "expr", "target", "var"):
                    if k in expanded and isinstance(expanded[k], str):
                        expanded[k] = _subst_template(expanded[k], m)

                # Inherit enabled/name/group defaults from the foreach wrapper if omitted
                if "enabled" not in expanded:
                    expanded["enabled"] = True
                if "group" not in expanded and step.get("group"):
                    expanded["group"] = step.get("group")
                if "name" not in expanded and step.get("name"):
                    expanded["name"] = step.get("name")

                out.append(expanded)

        if not matched_any and step.get("require_match", False):
            raise ValueError(f"Step #{i}: foreach matched no layers for pattern {raw_pat!r}")

    return out


def apply_layer_ops_inplace(
    payload: Dict[str, Any], export_let_layers: bool = False, seed_offset: int = 0
) -> int:
    H, W, layers, kinds = _extract_layers(payload)
    cfg = payload.get("layer_ops_config")
    if cfg is None:
        return 0
    if not isinstance(cfg, dict):
        raise ValueError("Invalid layer_ops_config: expected object")
    v = int(cfg.get("version", -1))
    if v not in (1, 2):
        raise ValueError("Invalid layer_ops_config: expected version=1 or version=2")

    if v == 1:
        ops = cfg.get("ops")
        if not isinstance(ops, list):
            return 0
        steps = [{"type": "op", **op} for op in ops if isinstance(op, dict)]
    else:
        steps = cfg.get("steps")
        if not isinstance(steps, list):
            return 0

    # Expand foreach steps into concrete op/let steps.
    steps = _expand_foreach_steps(steps, layer_names=sorted(layers.keys()))

    def _refs_layer_name(obj: Any, layer_name: str) -> bool:
        if isinstance(obj, dict):
            for v in obj.values():
                if _refs_layer_name(v, layer_name):
                    return True
            return False
        if isinstance(obj, list):
            for v in obj:
                if _refs_layer_name(v, layer_name):
                    return True
            return False
        if isinstance(obj, str):
            if obj == layer_name:
                return True
            return re.search(rf"\\b{re.escape(layer_name)}\\b", obj) is not None
        return False

    if "molecule_atp" not in layers and _refs_layer_name(steps, "molecule_atp"):
        raise ValueError(
            "Layer Ops config references missing required layer 'molecule_atp' (ATP is required)"
        )

    event_counters = payload.get("event_counters")
    if not isinstance(event_counters, dict):
        event_counters = {}
        payload["event_counters"] = event_counters
    totals = event_counters.get("totals")
    if not isinstance(totals, dict):
        totals = {}
        event_counters["totals"] = totals

    profile_expr_requested = bool(payload.get("_profile_expr") or cfg.get("profile_expr") or cfg.get("profile_deep"))
    profile_step_names_requested = bool(
        payload.get("_profile_step_names") or cfg.get("profile_step_names") or cfg.get("profile_deep")
    )

    profile_layer_ops = bool(
        payload.get("_profile_layer_ops")
        or cfg.get("profile")
        or cfg.get("profile_layer_ops")
        or profile_expr_requested
    )
    if profile_layer_ops:
        perf = event_counters.get("layer_ops_perf")
        if not isinstance(perf, dict):
            perf = {}
            event_counters["layer_ops_perf"] = perf
        by_type_s = perf.get("by_type_s")
        if not isinstance(by_type_s, dict):
            by_type_s = {}
            perf["by_type_s"] = by_type_s
        perf["calls"] = int(perf.get("calls") or 0) + 1
        perf.setdefault("total_s", 0.0)

        if profile_step_names_requested:
            step_perf = perf.get("step_perf")
            if not isinstance(step_perf, dict):
                step_perf = {}
                perf["step_perf"] = step_perf

    profile_expr = bool(profile_layer_ops and profile_expr_requested)
    if profile_expr:
        perf0 = event_counters.get("layer_ops_perf")
        if isinstance(perf0, dict):
            expr_perf = perf0.get("expr_perf")
            if not isinstance(expr_perf, dict):
                expr_perf = {}
                perf0["expr_perf"] = expr_perf
            expr_perf.setdefault("calls", 0)
            for k in ("env_s", "validate_s", "compile_s", "eval_s", "asarray_s", "writeback_s", "total_s"):
                expr_perf.setdefault(k, 0.0)

    def _prof_add(step_type: str, dt_s: float) -> None:
        if not profile_layer_ops:
            return
        perf = event_counters.get("layer_ops_perf")
        if not isinstance(perf, dict):
            return
        perf["total_s"] = float(perf.get("total_s") or 0.0) + float(dt_s)
        by_type_s = perf.get("by_type_s")
        if not isinstance(by_type_s, dict):
            by_type_s = {}
            perf["by_type_s"] = by_type_s
        by_type_s[step_type] = float(by_type_s.get(step_type) or 0.0) + float(dt_s)

    def _expr_prof_add(key: str, dt_s: float) -> None:
        if not profile_expr:
            return
        perf = event_counters.get("layer_ops_perf")
        if not isinstance(perf, dict):
            return
        ep = perf.get("expr_perf")
        if not isinstance(ep, dict):
            return
        ep[key] = float(ep.get(key) or 0.0) + float(dt_s)

    def _step_prof_add(step_key: str, field: str, dv: float) -> None:
        if not (profile_layer_ops and profile_step_names_requested):
            return
        perf = event_counters.get("layer_ops_perf")
        if not isinstance(perf, dict):
            return
        sp = perf.get("step_perf")
        if not isinstance(sp, dict):
            return
        ent = sp.get(step_key)
        if not isinstance(ent, dict):
            ent = {"calls": 0}
            sp[step_key] = ent
        if field == "calls":
            ent["calls"] = int(ent.get("calls") or 0) + int(dv)
            return
        ent[field] = float(ent.get(field) or 0.0) + float(dv)

    def _inc_total(key: str, dv: int) -> None:
        try:
            dv_i = int(dv)
        except Exception:
            dv_i = 0
        if dv_i == 0:
            return
        cur = totals.get(key, 0)
        try:
            cur_i = int(cur)
        except Exception:
            cur_i = 0
        totals[key] = int(cur_i + dv_i)

    step_events: Dict[str, int] = {
        "starvation_deaths": 0,
        "damage_deaths": 0,
        "divisions": 0,
    }

    def _is_identifier(name: str) -> bool:
        import re

        return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name or "") is not None

    vars2d: Dict[str, np.ndarray] = {}

    seed_offset_i = int(seed_offset)
    base_seed_raw = cfg.get("seed", cfg.get("rng_seed"))
    if base_seed_raw is None or str(base_seed_raw).strip() == "":
        rng = np.random.default_rng()
    else:
        try:
            rng = np.random.default_rng(int(base_seed_raw) + seed_offset_i)
        except Exception:
            rng = np.random.default_rng()

    _GROUP_FUNC_NAMES = {
        "sum_layers",
        "mean_layers",
        "min_layers",
        "max_layers",
        "sum_layer",
        "rand_beta",
        "rand_logitnorm",
    }

    def _normalize_glob_pattern(pat: str) -> str:
        pat = str(pat or "").strip()
        if not pat:
            raise ValueError("Pattern must be a non-empty string")
        # Convenience: treat bare prefixes like "gene_" as "gene_*".
        if not any(ch in pat for ch in "*?[]"):
            pat = pat + "*"
        return pat

    def _match_layer_names_glob(pat: str) -> List[str]:
        pat = _normalize_glob_pattern(pat)
        matched = [nm for nm in layers.keys() if fnmatch.fnmatchcase(nm, pat)]
        matched.sort()
        return matched

    def _reduce_layers(pat: str, op: str) -> np.ndarray:
        names = _match_layer_names_glob(pat)
        if not names:
            raise ValueError(f"No layers match pattern {str(pat)!r}")
        stack = np.stack([layers[nm].reshape(H, W) for nm in names], axis=0)
        if op == "sum":
            return stack.sum(axis=0)
        if op == "mean":
            return stack.mean(axis=0)
        if op == "min":
            return stack.min(axis=0)
        if op == "max":
            return stack.max(axis=0)
        raise ValueError(f"Unknown reduction op: {op!r}")

    # Env uses 2D arrays for user convenience
    def make_env() -> Dict[str, Any]:
        env: Dict[str, Any] = {**_ALLOWED_FUNCS, "True": True, "False": False}
        for nm, arr in layers.items():
            env[nm] = arr.reshape(H, W)
        for nm, arr2d in vars2d.items():
            env[nm] = arr2d

        # Layer-group helpers (glob patterns)
        env["sum_layers"] = lambda pat: _reduce_layers(pat, "sum")
        env["mean_layers"] = lambda pat: _reduce_layers(pat, "mean")
        env["min_layers"] = lambda pat: _reduce_layers(pat, "min")
        env["max_layers"] = lambda pat: _reduce_layers(pat, "max")

        # Scalar reductions
        env["sum_layer"] = lambda x: float(np.asarray(x).sum())

        def _rand_beta(alpha: Any = 1.0, beta: Any = 1.0) -> np.ndarray:
            a = np.asarray(alpha)
            b = np.asarray(beta)
            if np.any(a <= 0) or np.any(b <= 0):
                raise ValueError("rand_beta(alpha, beta): alpha and beta must be > 0")
            return rng.beta(a, b, size=(H, W)).astype(np.float32)

        def _rand_logitnorm(mu: Any = 0.0, sigma: Any = 1.0) -> np.ndarray:
            s = np.asarray(sigma)
            if np.any(s < 0):
                raise ValueError("rand_logitnorm(mu, sigma): sigma must be non-negative")
            z = rng.normal(loc=mu, scale=sigma, size=(H, W)).astype(np.float32)
            out = (np.float32(1.0) / (np.float32(1.0) + np.exp(-z))).astype(np.float32)
            return np.clip(out, 0.0, 1.0)

        env["rand_beta"] = _rand_beta
        env["rand_logitnorm"] = _rand_logitnorm
        return env

    opt_env_cache = bool(cfg.get("opt_env_cache") or cfg.get("optimize_env_cache"))
    opt_expr_cache = bool(cfg.get("opt_expr_cache") or cfg.get("optimize_expr_cache"))
    opt_diffusion_buffers = bool(cfg.get("opt_diffusion_buffers") or cfg.get("optimize_diffusion_buffers"))
    env_cache: Optional[Dict[str, Any]] = None

    def _env_get() -> Dict[str, Any]:
        nonlocal env_cache
        if not opt_env_cache:
            return make_env()
        if env_cache is None:
            env_cache = make_env()
        return env_cache

    def _env_update_layer(name: str) -> None:
        if not opt_env_cache or env_cache is None:
            return
        arr = layers.get(name)
        if arr is None:
            return
        env_cache[name] = arr.reshape(H, W)

    def _env_update_var(name: str) -> None:
        if not opt_env_cache or env_cache is None:
            return
        arr2d = vars2d.get(name)
        if arr2d is None:
            return
        env_cache[name] = arr2d

    _DIR_ALIASES = {
        "n": "north",
        "north": "north",
        "s": "south",
        "south": "south",
        "e": "east",
        "east": "east",
        "w": "west",
        "west": "west",
    }
    _DIR_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}

    def _normalize_dir(d: Any) -> str:
        s = str(d or "").strip().lower()
        s = _DIR_ALIASES.get(s, s)
        if s not in _DIR_OPP:
            raise ValueError(f"Invalid dir {d!r} (expected one of north/south/east/west)")
        return s

    def _counts2d_or_zeros(name: str) -> np.ndarray:
        arr = layers.get(name)
        if arr is None:
            return np.zeros((H, W), dtype=np.int64)
        return np.clip(np.rint(arr.reshape(H, W)), 0, None).astype(np.int64)

    def _compute_edge_caps(
        is_cell: np.ndarray,
        exporter_src: np.ndarray,
        importer_dst: np.ndarray,
        src_slice_y: slice,
        src_slice_x: slice,
        dst_slice_y: slice,
        dst_slice_x: slice,
    ) -> np.ndarray:
        cell_s = is_cell[src_slice_y, src_slice_x]
        cell_d = is_cell[dst_slice_y, dst_slice_x]
        e = exporter_src[src_slice_y, src_slice_x]
        i = importer_dst[dst_slice_y, dst_slice_x]
        slots = np.where(
            cell_s & cell_d,
            np.minimum(e, i),
            np.where(
                cell_s & (~cell_d),
                e,
                np.where((~cell_s) & cell_d, i, 0),
            ),
        )
        out = np.zeros((H, W), dtype=np.int64)
        out[src_slice_y, src_slice_x] = np.asarray(slots, dtype=np.int64)
        return out

    applied = 0
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("enabled", True) is False:
            continue

        step_type = str(step.get("type") or "op").strip().lower()
        t_step0 = time.perf_counter() if profile_layer_ops else 0.0

        step_name = str(step.get("name") or "").strip()
        step_key = f"{step_type}:{step_name}" if step_name else f"{step_type}:#{i}"
        if step_type == "diffusion":
            t0_step = time.perf_counter() if (profile_layer_ops and profile_step_names_requested) else 0.0
            molecules_spec = step.get("molecules", "molecule_*")
            if isinstance(molecules_spec, str):
                pat = molecules_spec.strip()
                if pat.startswith("glob:"):
                    pat = pat[len("glob:") :].strip()
                molecule_layers = _match_layer_names_glob(pat)
            elif isinstance(molecules_spec, list):
                molecule_layers = [str(x) for x in molecules_spec if isinstance(x, str) and str(x).strip()]
            else:
                raise ValueError(f"Step #{i}: diffusion 'molecules' must be a string glob or list of names")

            if not molecule_layers:
                raise ValueError(f"Step #{i}: diffusion matched no molecule layers")

            _step_prof_add(step_key, "calls", 1)
            _step_prof_add(step_key, "molecules", float(len(molecule_layers)))

            cell_layer = str(step.get("cell_layer", "cell") or "").strip()
            if not cell_layer:
                raise ValueError(f"Step #{i}: diffusion missing 'cell_layer'")
            if cell_layer not in layers:
                raise ValueError(f"Step #{i}: diffusion unknown cell_layer '{cell_layer}'")
            cell_mode = str(step.get("cell_mode", "eq") or "eq").strip().lower()
            if cell_mode != "eq":
                raise ValueError(f"Step #{i}: diffusion unsupported cell_mode {cell_mode!r} (expected 'eq')")
            cell_value = int(step.get("cell_value", 1))
            cell2d = np.rint(layers[cell_layer].reshape(H, W)).astype(np.int64)
            is_cell = cell2d == cell_value
            non_cell = ~is_cell

            rate = step.get("rate", None)
            rate_layer = step.get("rate_layer", None)
            if rate is None and rate_layer is None:
                raise ValueError(f"Step #{i}: diffusion missing 'rate' or 'rate_layer'")

            if rate is None:
                rate_scalar = 1.0
            else:
                try:
                    rate_scalar = float(rate)
                except Exception as e:
                    raise ValueError(f"Step #{i}: diffusion invalid rate {rate!r}: {e}") from e

            if rate_layer is None or str(rate_layer).strip() == "":
                rate2d = np.full((H, W), rate_scalar, dtype=np.float32)
            else:
                rl = str(rate_layer).strip()
                if rl not in layers:
                    raise ValueError(f"Step #{i}: diffusion unknown rate_layer '{rl}'")
                rate2d = np.asarray(layers[rl].reshape(H, W), dtype=np.float32) * float(rate_scalar)

            rate2d = np.clip(rate2d, 0.0, 1.0)
            rate2d = np.where(non_cell, rate2d, 0.0).astype(np.float32)

            seed = step.get("seed")
            rng_diff = rng
            if seed is not None:
                try:
                    rng_diff = np.random.default_rng(int(seed) + seed_offset_i + 1000003 * i)
                except Exception as e:
                    raise ValueError(f"Step #{i}: diffusion invalid seed {seed!r}: {e}") from e

            openN = np.zeros((H, W), dtype=bool)
            openS = np.zeros((H, W), dtype=bool)
            openE = np.zeros((H, W), dtype=bool)
            openW = np.zeros((H, W), dtype=bool)
            if H > 1:
                openN[1:, :] = non_cell[1:, :] & non_cell[0 : H - 1, :]
                openS[0 : H - 1, :] = non_cell[0 : H - 1, :] & non_cell[1:, :]
            if W > 1:
                openE[:, 0 : W - 1] = non_cell[:, 0 : W - 1] & non_cell[:, 1:]
                openW[:, 1:] = non_cell[:, 1:] & non_cell[:, 0 : W - 1]

            eps = np.float32(1e-9)
            gN = np.zeros((H, W), dtype=np.float32)
            gS = np.zeros((H, W), dtype=np.float32)
            gE = np.zeros((H, W), dtype=np.float32)
            gW = np.zeros((H, W), dtype=np.float32)
            if H > 1:
                r0 = rate2d[0 : H - 1, :]
                r1 = rate2d[1:, :]
                gV = (np.float32(2.0) * r0 * r1) / (r0 + r1 + eps)
                gS[0 : H - 1, :] = np.where(openS[0 : H - 1, :], gV, np.float32(0.0))
                gN[1:, :] = np.where(openN[1:, :], gV, np.float32(0.0))
            if W > 1:
                r0 = rate2d[:, 0 : W - 1]
                r1 = rate2d[:, 1:]
                gH = (np.float32(2.0) * r0 * r1) / (r0 + r1 + eps)
                gE[:, 0 : W - 1] = np.where(openE[:, 0 : W - 1], gH, np.float32(0.0))
                gW[:, 1:] = np.where(openW[:, 1:], gH, np.float32(0.0))

            dt = step.get("dt", None)
            if dt is None:
                dt0 = np.float32(0.25)
            else:
                try:
                    dt0 = np.float32(float(dt))
                except Exception as e:
                    raise ValueError(f"Step #{i}: diffusion invalid dt {dt!r}: {e}") from e
                if dt0 < 0:
                    raise ValueError(f"Step #{i}: diffusion dt must be non-negative (got {float(dt0)!r})")
            if dt0 > np.float32(0.25):
                dt0 = np.float32(0.25)

            pN = (dt0 * gN).astype(np.float32)
            pS = (dt0 * gS).astype(np.float32)
            pE = (dt0 * gE).astype(np.float32)
            pW = (dt0 * gW).astype(np.float32)
            pMove = np.clip(pN + pS + pE + pW, 0.0, 1.0).astype(np.float32)

            qN_buf = None
            qS_buf = None
            qE_buf = None
            pRem_buf = None
            inflow_buf = None
            if opt_diffusion_buffers:
                qN_buf = np.zeros((H, W), dtype=np.float32)
                qS_buf = np.zeros((H, W), dtype=np.float32)
                qE_buf = np.zeros((H, W), dtype=np.float32)
                pRem_buf = np.zeros((H, W), dtype=np.float32)
                inflow_buf = np.zeros((H, W), dtype=np.int64)

            t_setup_done = time.perf_counter() if (profile_layer_ops and profile_step_names_requested) else 0.0
            if profile_layer_ops and profile_step_names_requested:
                _step_prof_add(step_key, "setup_s", t_setup_done - t0_step)

            for mol_name in molecule_layers:
                t_mol0 = time.perf_counter() if (profile_layer_ops and profile_step_names_requested) else 0.0
                if mol_name not in layers:
                    raise ValueError(f"Step #{i}: diffusion unknown molecule layer '{mol_name}'")
                if kinds.get(mol_name) != "counts":
                    raise ValueError(
                        f"Step #{i}: diffusion molecule layer '{mol_name}' must be kind='counts'"
                    )

                m_old = np.clip(np.rint(layers[mol_name].reshape(H, W)), 0, None).astype(np.int64)

                n_out = rng_diff.binomial(m_old, pMove).astype(np.int64)

                qN = qN_buf if qN_buf is not None else np.zeros((H, W), dtype=np.float32)
                if qN_buf is not None:
                    qN.fill(0)
                np.divide(pN, pMove, out=qN, where=pMove > 0)
                np.clip(qN, 0.0, 1.0, out=qN)
                outN = rng_diff.binomial(n_out, qN).astype(np.int64)

                rem = n_out - outN
                pRem = pRem_buf if pRem_buf is not None else np.maximum(np.float32(0.0), pMove - pN)
                if pRem_buf is not None:
                    np.subtract(pMove, pN, out=pRem)
                    np.maximum(pRem, np.float32(0.0), out=pRem)
                qS = qS_buf if qS_buf is not None else np.zeros((H, W), dtype=np.float32)
                if qS_buf is not None:
                    qS.fill(0)
                np.divide(pS, pRem, out=qS, where=pRem > 0)
                np.clip(qS, 0.0, 1.0, out=qS)
                outS = rng_diff.binomial(rem, qS).astype(np.int64)

                rem = rem - outS
                if pRem_buf is not None:
                    np.subtract(pRem, pS, out=pRem)
                    np.maximum(pRem, np.float32(0.0), out=pRem)
                else:
                    pRem = np.maximum(np.float32(0.0), pRem - pS)
                qE = qE_buf if qE_buf is not None else np.zeros((H, W), dtype=np.float32)
                if qE_buf is not None:
                    qE.fill(0)
                np.divide(pE, pRem, out=qE, where=pRem > 0)
                np.clip(qE, 0.0, 1.0, out=qE)
                outE = rng_diff.binomial(rem, qE).astype(np.int64)
                outW = (rem - outE).astype(np.int64)

                inflow = inflow_buf if inflow_buf is not None else np.zeros((H, W), dtype=np.int64)
                if inflow_buf is not None:
                    inflow.fill(0)
                inflow[1:, :] += outS[0 : H - 1, :]
                inflow[0 : H - 1, :] += outN[1:, :]
                inflow[:, 1:] += outE[:, 0 : W - 1]
                inflow[:, 0 : W - 1] += outW[:, 1:]

                m_new = m_old - (outN + outS + outE + outW) + inflow
                layers[mol_name] = np.asarray(m_new, dtype=np.float32).reshape(H * W)
                _env_update_layer(mol_name)

                if profile_layer_ops and profile_step_names_requested:
                    _step_prof_add(step_key, "molecule_s", time.perf_counter() - t_mol0)

            applied += 1
            if profile_layer_ops and profile_step_names_requested:
                _step_prof_add(step_key, "total_s", time.perf_counter() - t0_step)
            if profile_layer_ops:
                _prof_add(step_type, time.perf_counter() - t_step0)
            continue

        if step_type == "divide_cells":
            t0_step = time.perf_counter() if (profile_layer_ops and profile_step_names_requested) else 0.0
            _step_prof_add(step_key, "calls", 1)
            cell_layer = str(step.get("cell_layer", "cell") or "").strip()
            if not cell_layer:
                raise ValueError(f"Step #{i}: divide_cells missing 'cell_layer'")
            if cell_layer not in layers:
                raise ValueError(f"Step #{i}: divide_cells unknown cell_layer '{cell_layer}'")

            cell_value = int(step.get("cell_value", 1))
            empty_value = int(step.get("empty_value", 0))
            cell2d = np.rint(layers[cell_layer].reshape(H, W)).astype(np.int64)
            is_cell = cell2d == cell_value
            is_empty = cell2d == empty_value

            trigger_layer = str(step.get("trigger_layer", "protein_divider") or "").strip()
            if not trigger_layer:
                raise ValueError(f"Step #{i}: divide_cells missing 'trigger_layer'")
            if trigger_layer not in layers and trigger_layer not in vars2d:
                raise ValueError(
                    f"Step #{i}: divide_cells unknown trigger_layer '{trigger_layer}' (not a layer or let var)"
                )

            try:
                threshold = float(step.get("threshold", step.get("trigger_value", 50)))
            except Exception as e:
                raise ValueError(f"Step #{i}: divide_cells invalid threshold: {e}") from e

            frac_raw = step.get("split_fraction", 0.5)
            try:
                split_fraction = float(frac_raw)
            except Exception as e:
                raise ValueError(f"Step #{i}: divide_cells invalid split_fraction {frac_raw!r}: {e}") from e
            if split_fraction <= 0.0 or split_fraction >= 1.0:
                raise ValueError(
                    f"Step #{i}: divide_cells split_fraction must be in (0, 1) (got {split_fraction!r})"
                )

            max_radius_raw = step.get("max_radius", None)
            if max_radius_raw is None or str(max_radius_raw).strip() == "":
                max_radius = int(max(H, W))
            else:
                try:
                    max_radius = int(max_radius_raw)
                except Exception as e:
                    raise ValueError(f"Step #{i}: divide_cells invalid max_radius {max_radius_raw!r}: {e}") from e
                if max_radius < 1:
                    max_radius = 1

            prefixes_raw = step.get("layer_prefixes", None)
            if prefixes_raw is None:
                layer_prefixes = ["molecule", "protein", "rna", "damage", "gene"]
            elif isinstance(prefixes_raw, list) and prefixes_raw:
                layer_prefixes = [str(x) for x in prefixes_raw if isinstance(x, (str, int, float)) and str(x).strip()]
                if not layer_prefixes:
                    raise ValueError(f"Step #{i}: divide_cells layer_prefixes must be a non-empty list")
            else:
                raise ValueError(f"Step #{i}: divide_cells layer_prefixes must be a non-empty list")

            seed = step.get("seed")
            rng_div = rng
            if seed is not None:
                try:
                    rng_div = np.random.default_rng(int(seed) + seed_offset_i + 1000003 * i)
                except Exception as e:
                    raise ValueError(f"Step #{i}: divide_cells invalid seed {seed!r}: {e}") from e

            if trigger_layer in layers:
                trig2d = np.asarray(layers[trigger_layer].reshape(H, W), dtype=np.float32)
            else:
                trig2d = np.asarray(vars2d[trigger_layer], dtype=np.float32)
            div_mask = is_cell & (trig2d > np.float32(threshold))
            if not np.any(div_mask) or not np.any(is_empty):
                applied += 1
                if profile_layer_ops and profile_step_names_requested:
                    _step_prof_add(step_key, "total_s", time.perf_counter() - t0_step)
                if profile_layer_ops:
                    _prof_add(step_type, time.perf_counter() - t_step0)
                continue

            available = np.array(is_empty, dtype=bool)

            target_mask_layer = step.get("target_mask_layer", None)
            if target_mask_layer is not None and str(target_mask_layer).strip() != "":
                ml = str(target_mask_layer).strip()
                mask_value_raw = step.get("target_mask_value", 1)
                try:
                    mask_value = int(mask_value_raw)
                except Exception as e:
                    raise ValueError(f"Step #{i}: divide_cells invalid target_mask_value {mask_value_raw!r}: {e}") from e

                if ml in layers:
                    m2d = np.rint(layers[ml].reshape(H, W)).astype(np.int64)
                elif ml in vars2d:
                    m2d = np.rint(vars2d[ml]).astype(np.int64)
                else:
                    raise ValueError(f"Step #{i}: divide_cells unknown target_mask_layer '{ml}'")

                available &= m2d == mask_value

            sources = np.argwhere(div_mask)

            def _find_nearest_available(src_y: int, src_x: int):
                for r in range(1, max_radius + 1):
                    y0 = max(0, src_y - r)
                    y1 = min(H - 1, src_y + r)
                    x0 = max(0, src_x - r)
                    x1 = min(W - 1, src_x + r)
                    sub = available[y0 : y1 + 1, x0 : x1 + 1]
                    if not np.any(sub):
                        continue
                    ys, xs = np.nonzero(sub)
                    ys = ys + y0
                    xs = xs + x0
                    dy = ys - int(src_y)
                    dx = xs - int(src_x)
                    d2 = (dy * dy + dx * dx).astype(np.int64)
                    md2 = int(d2.min())
                    ties = np.where(d2 == md2)[0]
                    pick = int(ties[0]) if ties.size == 1 else int(rng_div.choice(ties))
                    return int(ys[pick]), int(xs[pick]), md2
                return None

            unassigned = [int(x) for x in range(int(sources.shape[0]))]
            assignments = []
            while unassigned:
                proposals = {}
                for si in unassigned:
                    sy = int(sources[si, 0])
                    sx = int(sources[si, 1])
                    res = _find_nearest_available(sy, sx)
                    if res is None:
                        continue
                    ty, tx, d2 = res
                    proposals.setdefault((int(ty), int(tx)), []).append((int(si), int(d2)))

                if not proposals:
                    break

                round_winners = []
                winners_src = set()
                for (ty, tx), cand in proposals.items():
                    dmin = min(d for _, d in cand)
                    tied = [si for si, d in cand if d == dmin]
                    win_si = int(tied[0]) if len(tied) == 1 else int(rng_div.choice(np.asarray(tied, dtype=np.int64)))
                    if win_si in winners_src:
                        continue
                    winners_src.add(win_si)
                    round_winners.append((win_si, int(ty), int(tx)))

                if not round_winners:
                    break

                for win_si, ty, tx in round_winners:
                    available[ty, tx] = False
                assignments.extend(round_winners)
                unassigned = [si for si in unassigned if si not in winners_src]

            if not assignments:
                applied += 1
                if profile_layer_ops and profile_step_names_requested:
                    _step_prof_add(step_key, "total_s", time.perf_counter() - t0_step)
                if profile_layer_ops:
                    _prof_add(step_type, time.perf_counter() - t_step0)
                continue

            split_layers = []
            for nm in layers.keys():
                if nm == cell_layer:
                    continue
                if any(str(nm).startswith(pfx) for pfx in layer_prefixes):
                    if kinds.get(nm) == "categorical":
                        continue
                    split_layers.append(nm)

            div_success = 0
            for win_si, ty, tx in assignments:
                sy = int(sources[win_si, 0])
                sx = int(sources[win_si, 1])
                if not is_empty[ty, tx]:
                    continue

                cell2d[ty, tx] = cell_value
                is_empty[ty, tx] = False
                div_success += 1

                for nm in split_layers:
                    arr2d = layers[nm].reshape(H, W)
                    kind = kinds.get(nm)
                    if str(nm).startswith("damage"):
                        v = float(arr2d[sy, sx])
                        arr2d[ty, tx] = np.float32(v)
                        layers[nm] = np.asarray(arr2d, dtype=np.float32).reshape(H * W)
                        _env_update_layer(nm)
                        continue
                    if kind == "counts":
                        v = int(np.clip(np.rint(arr2d[sy, sx]), 0, None))
                        moved = int(v * split_fraction)
                        if moved < 0:
                            moved = 0
                        if moved > v:
                            moved = v
                        arr2d[sy, sx] = np.float32(v - moved)
                        arr2d[ty, tx] = np.float32(np.clip(np.rint(arr2d[ty, tx]), 0, None) + moved)
                    else:
                        v = float(arr2d[sy, sx])
                        moved = v * split_fraction
                        arr2d[sy, sx] = np.float32(v - moved)
                        arr2d[ty, tx] = np.float32(float(arr2d[ty, tx]) + moved)
                    layers[nm] = np.asarray(arr2d, dtype=np.float32).reshape(H * W)
                    _env_update_layer(nm)

            layers[cell_layer] = np.asarray(cell2d, dtype=np.float32).reshape(H * W)
            _env_update_layer(cell_layer)
            if div_success:
                step_events["divisions"] = int(step_events.get("divisions", 0) + int(div_success))
            applied += 1
            if profile_layer_ops and profile_step_names_requested:
                _step_prof_add(step_key, "total_s", time.perf_counter() - t0_step)
            if profile_layer_ops:
                _prof_add(step_type, time.perf_counter() - t_step0)
            continue

        if step_type == "transport":
            t0_step = time.perf_counter() if (profile_layer_ops and profile_step_names_requested) else 0.0
            molecules_spec = step.get("molecules", "molecule_*")
            if isinstance(molecules_spec, str):
                pat = molecules_spec.strip()
                if pat.startswith("glob:"):
                    pat = pat[len("glob:") :].strip()
                molecule_layers = _match_layer_names_glob(pat)
            elif isinstance(molecules_spec, list):
                molecule_layers = [str(x) for x in molecules_spec if isinstance(x, str) and str(x).strip()]
            else:
                raise ValueError(f"Step #{i}: transport 'molecules' must be a string glob or list of names")

            if not molecule_layers:
                raise ValueError(f"Step #{i}: transport matched no molecule layers")

            _step_prof_add(step_key, "calls", 1)
            _step_prof_add(step_key, "molecules", float(len(molecule_layers)))

            dirs_raw = step.get("dirs")
            if dirs_raw is None:
                dirs = ["north", "south", "east", "west"]
            elif isinstance(dirs_raw, list):
                dirs = [_normalize_dir(d) for d in dirs_raw]
            else:
                raise ValueError(f"Step #{i}: transport 'dirs' must be a list")
            dirs = list(dict.fromkeys(dirs))

            protein_prefix = str(step.get("protein_prefix", "protein_") or "")
            molecule_prefix = str(step.get("molecule_prefix", "molecule_") or "")

            cell_layer = str(step.get("cell_layer", "cell") or "").strip()
            if not cell_layer:
                raise ValueError(f"Step #{i}: transport missing 'cell_layer'")
            if cell_layer not in layers:
                raise ValueError(f"Step #{i}: transport unknown cell_layer '{cell_layer}'")
            cell_mode = str(step.get("cell_mode", "eq") or "eq").strip().lower()
            if cell_mode != "eq":
                raise ValueError(f"Step #{i}: transport unsupported cell_mode {cell_mode!r} (expected 'eq')")
            cell_value = int(step.get("cell_value", 1))
            cell2d = np.rint(layers[cell_layer].reshape(H, W)).astype(np.int64)
            is_cell = cell2d == cell_value

            per_pair_rate = float(step.get("per_pair_rate", 1.0))
            if per_pair_rate < 0.0 or per_pair_rate > 1.0:
                raise ValueError(
                    f"Step #{i}: transport per_pair_rate must be in [0, 1] (got {per_pair_rate!r})"
                )

            seed = step.get("seed")
            rng_transport = rng
            if seed is not None:
                try:
                    rng_transport = np.random.default_rng(int(seed) + seed_offset_i + 1000003 * i)
                except Exception as e:
                    raise ValueError(f"Step #{i}: transport invalid seed {seed!r}: {e}") from e

            t_setup_done = time.perf_counter() if (profile_layer_ops and profile_step_names_requested) else 0.0
            if profile_layer_ops and profile_step_names_requested:
                _step_prof_add(step_key, "setup_s", t_setup_done - t0_step)

            for mol_name in molecule_layers:
                t_mol0 = time.perf_counter() if (profile_layer_ops and profile_step_names_requested) else 0.0
                if mol_name not in layers:
                    raise ValueError(f"Step #{i}: transport unknown molecule layer '{mol_name}'")
                if kinds.get(mol_name) != "counts":
                    raise ValueError(
                        f"Step #{i}: transport molecule layer '{mol_name}' must be kind='counts'"
                    )

                suffix = mol_name
                if molecule_prefix and mol_name.startswith(molecule_prefix):
                    suffix = mol_name[len(molecule_prefix) :]

                capN = np.zeros((H, W), dtype=np.int64)
                capS = np.zeros((H, W), dtype=np.int64)
                capE = np.zeros((H, W), dtype=np.int64)
                capW = np.zeros((H, W), dtype=np.int64)

                if "north" in dirs:
                    e = _counts2d_or_zeros(f"{protein_prefix}north_exporter_{suffix}")
                    i2d = _counts2d_or_zeros(f"{protein_prefix}{_DIR_OPP['north']}_importer_{suffix}")
                    capN = _compute_edge_caps(
                        is_cell,
                        exporter_src=e,
                        importer_dst=i2d,
                        src_slice_y=slice(1, H),
                        src_slice_x=slice(0, W),
                        dst_slice_y=slice(0, H - 1),
                        dst_slice_x=slice(0, W),
                    )
                if "south" in dirs:
                    e = _counts2d_or_zeros(f"{protein_prefix}south_exporter_{suffix}")
                    i2d = _counts2d_or_zeros(f"{protein_prefix}{_DIR_OPP['south']}_importer_{suffix}")
                    capS = _compute_edge_caps(
                        is_cell,
                        exporter_src=e,
                        importer_dst=i2d,
                        src_slice_y=slice(0, H - 1),
                        src_slice_x=slice(0, W),
                        dst_slice_y=slice(1, H),
                        dst_slice_x=slice(0, W),
                    )
                if "east" in dirs:
                    e = _counts2d_or_zeros(f"{protein_prefix}east_exporter_{suffix}")
                    i2d = _counts2d_or_zeros(f"{protein_prefix}{_DIR_OPP['east']}_importer_{suffix}")
                    capE = _compute_edge_caps(
                        is_cell,
                        exporter_src=e,
                        importer_dst=i2d,
                        src_slice_y=slice(0, H),
                        src_slice_x=slice(0, W - 1),
                        dst_slice_y=slice(0, H),
                        dst_slice_x=slice(1, W),
                    )
                if "west" in dirs:
                    e = _counts2d_or_zeros(f"{protein_prefix}west_exporter_{suffix}")
                    i2d = _counts2d_or_zeros(f"{protein_prefix}{_DIR_OPP['west']}_importer_{suffix}")
                    capW = _compute_edge_caps(
                        is_cell,
                        exporter_src=e,
                        importer_dst=i2d,
                        src_slice_y=slice(0, H),
                        src_slice_x=slice(1, W),
                        dst_slice_y=slice(0, H),
                        dst_slice_x=slice(0, W - 1),
                    )

                if per_pair_rate < 1.0:
                    capN = rng_transport.binomial(capN, per_pair_rate).astype(np.int64)
                    capS = rng_transport.binomial(capS, per_pair_rate).astype(np.int64)
                    capE = rng_transport.binomial(capE, per_pair_rate).astype(np.int64)
                    capW = rng_transport.binomial(capW, per_pair_rate).astype(np.int64)

                m_old = np.clip(np.rint(layers[mol_name].reshape(H, W)), 0, None).astype(np.int64)

                C = capN + capS + capE + capW
                n_move = np.minimum(m_old, C)

                outN = rng_transport.hypergeometric(capN, C - capN, n_move).astype(np.int64)
                rem_move = n_move - outN
                rem_cap = C - capN

                outS = rng_transport.hypergeometric(capS, rem_cap - capS, rem_move).astype(np.int64)
                rem_move = rem_move - outS
                rem_cap = rem_cap - capS

                outE = rng_transport.hypergeometric(capE, rem_cap - capE, rem_move).astype(np.int64)
                outW = (rem_move - outE).astype(np.int64)

                inflow = np.zeros((H, W), dtype=np.int64)
                inflow[1:, :] += outS[0 : H - 1, :]
                inflow[0 : H - 1, :] += outN[1:, :]
                inflow[:, 1:] += outE[:, 0 : W - 1]
                inflow[:, 0 : W - 1] += outW[:, 1:]

                m_new = m_old - (outN + outS + outE + outW) + inflow
                layers[mol_name] = np.asarray(m_new, dtype=np.float32).reshape(H * W)
                _env_update_layer(mol_name)

                if profile_layer_ops and profile_step_names_requested:
                    _step_prof_add(step_key, "molecule_s", time.perf_counter() - t_mol0)

            applied += 1
            if profile_layer_ops and profile_step_names_requested:
                _step_prof_add(step_key, "total_s", time.perf_counter() - t0_step)
            continue

        if step_type not in ("op", "let"):
            raise ValueError(
                f"Step #{i}: invalid type {step_type!r} (expected 'op', 'let', 'transport', 'diffusion', or 'divide_cells')"
            )

        expr = str(step.get("expr") or "").strip()
        if not expr:
            if profile_layer_ops:
                _prof_add(step_type, time.perf_counter() - t_step0)
            continue

        t_expr0 = time.perf_counter() if profile_expr else 0.0
        if profile_expr:
            perf = event_counters.get("layer_ops_perf")
            ep = perf.get("expr_perf") if isinstance(perf, dict) else None
            if isinstance(ep, dict):
                ep["calls"] = int(ep.get("calls") or 0) + 1

        t0 = time.perf_counter() if profile_expr else 0.0
        env = _env_get() if opt_env_cache else make_env()
        if profile_expr:
            _expr_prof_add("env_s", time.perf_counter() - t0)

        try:
            if profile_expr:
                import ast

                if opt_expr_cache:
                    tc0 = time.perf_counter()
                    code, names, call_fns = _compile_expr_cached(expr)
                    _expr_prof_add("compile_s", time.perf_counter() - tc0)

                    tchk0 = time.perf_counter()
                    for nm in names:
                        if nm not in env:
                            raise ValueError(f"Unknown identifier '{nm}'")
                    for fn in call_fns:
                        if fn not in env or not callable(env.get(fn)):
                            raise ValueError(f"Function '{fn}' is not allowed")
                    _expr_prof_add("validate_s", time.perf_counter() - tchk0)

                    te0 = time.perf_counter()
                    out2d = eval(code, {"__builtins__": {}}, env)  # noqa: S307
                    _expr_prof_add("eval_s", time.perf_counter() - te0)

                    if not isinstance(out2d, np.ndarray):
                        ta0 = time.perf_counter()
                        out2d = np.asarray(out2d)
                        _expr_prof_add("asarray_s", time.perf_counter() - ta0)
                else:
                    tv0 = time.perf_counter()
                    _validate_expr_ast(expr, env)
                    _expr_prof_add("validate_s", time.perf_counter() - tv0)

                    tc0 = time.perf_counter()
                    code = compile(ast.parse(expr, mode="eval"), "<layer_op>", "eval")
                    _expr_prof_add("compile_s", time.perf_counter() - tc0)

                    te0 = time.perf_counter()
                    out2d = eval(code, {"__builtins__": {}}, env)  # noqa: S307
                    _expr_prof_add("eval_s", time.perf_counter() - te0)

                    if not isinstance(out2d, np.ndarray):
                        ta0 = time.perf_counter()
                        out2d = np.asarray(out2d)
                        _expr_prof_add("asarray_s", time.perf_counter() - ta0)
            else:
                out2d = _eval_expr_fast(expr, env) if opt_expr_cache else _eval_expr(expr, env)
        except Exception as e:
            raise ValueError(
                f"Step #{i} (type='{step_type}') failed for expr: {expr!r}; error: {e}"
            ) from e

        if profile_expr:
            _expr_prof_add("total_s", time.perf_counter() - t_expr0)
        if out2d.shape != (H, W):
            # Allow scalar outputs; treat them as constants by broadcasting over the grid.
            if np.asarray(out2d).size == 1:
                v = float(np.asarray(out2d).reshape(-1)[0])
                out2d = np.full((H, W), v, dtype=np.float32)
            else:
                raise ValueError(
                    f"Step #{i}: expression result has shape {tuple(out2d.shape)}, expected {(H, W)}"
                )

        t_wb0 = time.perf_counter() if profile_expr else 0.0

        if step_type == "let":
            var = str(step.get("var") or "").strip()
            if not var:
                raise ValueError(f"Step #{i}: missing 'var' for let")
            if not _is_identifier(var):
                raise ValueError(f"Step #{i}: invalid var name {var!r}")
            if var in _ALLOWED_FUNCS or var in _GROUP_FUNC_NAMES or var in ("True", "False"):
                raise ValueError(f"Step #{i}: var name not allowed: {var!r}")
            if var in layers:
                raise ValueError(f"Step #{i}: var conflicts with existing layer: {var!r}")
            if var in vars2d:
                raise ValueError(f"Step #{i}: var already defined: {var!r}")
            vars2d[var] = np.asarray(out2d)
            _env_update_var(var)
            if profile_expr:
                _expr_prof_add("writeback_s", time.perf_counter() - t_wb0)
            if profile_layer_ops:
                _prof_add(step_type, time.perf_counter() - t_step0)
            continue

        target = str(step.get("target") or "").strip()
        if not target:
            if profile_layer_ops:
                _prof_add(step_type, time.perf_counter() - t_step0)
            continue

        if target not in layers:
            raise ValueError(f"Step #{i}: unknown target layer '{target}'")

        kind = kinds.get(target)
        if kind == "categorical":
            allowed_raw = step.get("allowed_values", None)
            if allowed_raw is None:
                cur = np.rint(layers[target].reshape(H, W)).astype(np.int64)
                allowed = np.unique(cur)
            else:
                if not isinstance(allowed_raw, list) or not allowed_raw:
                    raise ValueError(f"Step #{i}: categorical allowed_values must be a non-empty list")
                try:
                    allowed = np.asarray([int(x) for x in allowed_raw], dtype=np.int64)
                except Exception as e:
                    raise ValueError(f"Step #{i}: categorical allowed_values invalid: {e}") from e
                allowed = np.unique(allowed)

            if allowed.size <= 0:
                raise ValueError(f"Step #{i}: categorical target '{target}' has no allowed values")

            out_i = np.rint(out2d).astype(np.int64)

            mn = int(allowed.min())
            mx = int(allowed.max())
            if allowed.size == (mx - mn + 1) and np.all(allowed == np.arange(mn, mx + 1, dtype=np.int64)):
                out_i = np.clip(out_i, mn, mx)
            else:
                best = np.full(out_i.shape, int(allowed[0]), dtype=np.int64)
                bestd = np.abs(out_i - int(allowed[0]))
                for v in allowed[1:]:
                    vv = int(v)
                    d = np.abs(out_i - vv)
                    m = d < bestd
                    if np.any(m):
                        best[m] = vv
                        bestd[m] = d[m]
                out_i = best

            old_i = None
            if target == "cell" and out_i.shape == (H, W):
                old_i = np.rint(layers[target].reshape(H, W)).astype(np.int64)
            layers[target] = np.asarray(out_i, dtype=np.float32).reshape(H * W)
            _env_update_layer(target)
            if profile_expr:
                _expr_prof_add("writeback_s", time.perf_counter() - t_wb0)
            if old_i is not None:
                step_name = str(step.get("name") or "").strip()
                deaths = int(((old_i == 1) & (out_i == 0)).sum())
                if deaths:
                    if step_name == "atp_starvation_death":
                        step_events["starvation_deaths"] = int(step_events.get("starvation_deaths", 0) + deaths)
                    elif step_name == "cell_death":
                        step_events["damage_deaths"] = int(step_events.get("damage_deaths", 0) + deaths)
            applied += 1
            if profile_layer_ops:
                _prof_add(step_type, time.perf_counter() - t_step0)
            continue

        if kind == "counts":
            # counts are non-negative integers; store as float32 but enforce integer semantics
            out2d = np.clip(np.rint(out2d), 0, None)

        layers[target] = np.asarray(out2d, dtype=np.float32).reshape(H * W)
        _env_update_layer(target)
        applied += 1
        if profile_expr:
            _expr_prof_add("writeback_s", time.perf_counter() - t_wb0)
        if profile_layer_ops:
            _prof_add(step_type, time.perf_counter() - t_step0)

    # Write updated b64 buffers back into payload
    data = payload.get("data")
    assert isinstance(data, dict)

    if export_let_layers and vars2d:
        layer_meta = payload.get("layers")
        assert isinstance(layer_meta, list)
        existing = set()
        for m in layer_meta:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                existing.add(m["name"])

        for var, arr2d in vars2d.items():
            debug_name = f"__let__{var}"
            if debug_name not in existing:
                layer_meta.append({"name": debug_name, "kind": "continuous"})
                existing.add(debug_name)
            kinds[debug_name] = "continuous"
            layers[debug_name] = np.asarray(arr2d, dtype=np.float32).reshape(H * W)
            data[debug_name] = {"dtype": "float32", "b64": ""}
    for name, arr in layers.items():
        entry = data.get(name)
        if not isinstance(entry, dict):
            continue
        if entry.get("dtype") != "float32":
            continue
        entry["b64"] = _encode_float32_b64(arr)

    event_counters["last"] = {
        "starvation_deaths": int(step_events.get("starvation_deaths", 0)),
        "damage_deaths": int(step_events.get("damage_deaths", 0)),
        "divisions": int(step_events.get("divisions", 0)),
    }
    _inc_total("starvation_deaths", step_events.get("starvation_deaths", 0))
    _inc_total("damage_deaths", step_events.get("damage_deaths", 0))
    _inc_total("divisions", step_events.get("divisions", 0))

    return applied


def main() -> int:
    p = argparse.ArgumentParser(description="Apply layer_ops_config operations to a gridstate.json bundle")
    p.add_argument("--in", dest="in_path", required=True, help="Input gridstate.json")
    p.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output gridstate.json (default: <in> with .ops.json suffix)",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file (ignored if --out is provided)",
    )
    p.add_argument(
        "--export-let-layers",
        action="store_true",
        help="Export let variables as debug layers named '__let__<var>' for inspection",
    )
    p.add_argument(
        "--ticks",
        dest="ticks",
        type=int,
        default=1,
        help="Number of ticks to run (default: 1)",
    )
    p.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Optional deterministic seed for rand_* funcs when layer_ops_config omits per-step seeds",
    )
    p.add_argument(
        "--opt-env-cache",
        dest="opt_env_cache",
        action="store_true",
        help="Enable env caching optimization for op/let steps",
    )
    p.add_argument(
        "--opt-expr-cache",
        dest="opt_expr_cache",
        action="store_true",
        help="Enable compiled expression caching optimization for op/let steps",
    )
    p.add_argument(
        "--opt-diffusion-buffers",
        dest="opt_diffusion_buffers",
        action="store_true",
        help="Enable diffusion buffer reuse optimization",
    )
    p.add_argument(
        "--ab-test",
        dest="ab_test",
        action="store_true",
        help="Run X ticks with and without optimizations and assert identical outputs",
    )
    args = p.parse_args()

    in_path = Path(args.in_path)
    payload = _load_gridstate_payload(in_path)

    ticks = int(args.ticks or 1)
    if ticks < 1:
        ticks = 1

    def _set_cfg_flag(p0: Dict[str, Any], key: str, val: Any) -> None:
        cfg0 = p0.get("layer_ops_config")
        if not isinstance(cfg0, dict):
            return
        cfg0[key] = val

    def _run_ticks(p0: Dict[str, Any]) -> None:
        applied0 = 0
        for t in range(ticks):
            applied0 += int(
                apply_layer_ops_inplace(p0, export_let_layers=args.export_let_layers, seed_offset=t)
            )
        p0.setdefault("_cli_stats", {})
        if isinstance(p0.get("_cli_stats"), dict):
            p0["_cli_stats"]["applied_ops"] = int(applied0)

    if args.seed is not None:
        _set_cfg_flag(payload, "seed", int(args.seed))

    if args.ab_test:
        base_txt = json.dumps(payload)
        a_payload = json.loads(base_txt)
        b_payload = json.loads(base_txt)

        if args.seed is None:
            _set_cfg_flag(a_payload, "seed", 0)
            _set_cfg_flag(b_payload, "seed", 0)

        opt_env = bool(args.opt_env_cache)
        opt_expr = bool(args.opt_expr_cache)
        opt_diff = bool(args.opt_diffusion_buffers)
        if not opt_env and not opt_expr and not opt_diff:
            # preserve previous behavior: default to env-cache on
            opt_env = True

        _set_cfg_flag(a_payload, "opt_env_cache", False)
        _set_cfg_flag(a_payload, "opt_expr_cache", False)
        _set_cfg_flag(a_payload, "opt_diffusion_buffers", False)
        _set_cfg_flag(b_payload, "opt_env_cache", opt_env)
        _set_cfg_flag(b_payload, "opt_expr_cache", opt_expr)
        _set_cfg_flag(b_payload, "opt_diffusion_buffers", opt_diff)

        t0 = time.perf_counter()
        _run_ticks(a_payload)
        t_a = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_ticks(b_payload)
        t_b = time.perf_counter() - t0

        a_stats = a_payload.get("_cli_stats") if isinstance(a_payload, dict) else None
        b_stats = b_payload.get("_cli_stats") if isinstance(b_payload, dict) else None
        a_applied = int(a_stats.get("applied_ops") or 0) if isinstance(a_stats, dict) else 0
        b_applied = int(b_stats.get("applied_ops") or 0) if isinstance(b_stats, dict) else 0

        a_data = a_payload.get("data") if isinstance(a_payload, dict) else None
        b_data = b_payload.get("data") if isinstance(b_payload, dict) else None
        if not isinstance(a_data, dict) or not isinstance(b_data, dict):
            raise ValueError("missing data")

        mismatches = []
        names = sorted(set(a_data.keys()) | set(b_data.keys()))
        for nm in names:
            ea = a_data.get(nm)
            eb = b_data.get(nm)
            if not isinstance(ea, dict) or not isinstance(eb, dict):
                continue
            if ea.get("dtype") != "float32" or eb.get("dtype") != "float32":
                continue
            ba = ea.get("b64")
            bb = eb.get("b64")
            if isinstance(ba, str) and isinstance(bb, str) and ba == bb:
                continue
            mismatches.append(nm)

        print(f"ticks={ticks}")
        print(f"baseline_s={t_a:.6f}")
        print(f"optimized_s={t_b:.6f}")
        print(f"baseline_applied_ops={a_applied}")
        print(f"optimized_applied_ops={b_applied}")
        if t_b > 0:
            print(f"speedup_x={t_a / t_b:.3f}")

        if mismatches:
            print(f"mismatched_layers={len(mismatches)}")
            for nm in mismatches[:10]:
                print(f"- {nm}")
            return 2
        print("outputs_match=true")
        return 0

    if args.opt_env_cache:
        _set_cfg_flag(payload, "opt_env_cache", True)
    if args.opt_expr_cache:
        _set_cfg_flag(payload, "opt_expr_cache", True)
    if args.opt_diffusion_buffers:
        _set_cfg_flag(payload, "opt_diffusion_buffers", True)

    _run_ticks(payload)
    stats = payload.get("_cli_stats") if isinstance(payload, dict) else None
    applied = int(stats.get("applied_ops") or 0) if isinstance(stats, dict) else 0

    if args.out_path:
        out_path = Path(args.out_path)
    elif args.inplace:
        out_path = in_path
    else:
        out_path = in_path.with_suffix("")
        out_path = out_path.with_name(out_path.name + ".ops.json")

    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"applied_ops: {applied}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
