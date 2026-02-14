import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class RunSpec:
    name: str
    args: List[str]


def _run_one(cmd: List[str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = p.stdout.strip()
    err = p.stderr.strip()
    if p.returncode != 0:
        raise RuntimeError(f"command failed (code={p.returncode})\nCMD: {' '.join(cmd)}\nSTDERR:\n{err}\nSTDOUT:\n{out}")

    if not out:
        raise RuntimeError(f"no stdout from command\nCMD: {' '.join(cmd)}\nSTDERR:\n{err}")

    # run_mpc_control prints JSON (possibly pretty-printed). If any non-JSON text
    # sneaks in, strip it by finding the first '{'.
    start = out.find("{")
    if start == -1:
        raise RuntimeError(f"stdout did not contain JSON\nCMD: {' '.join(cmd)}\nSTDERR:\n{err}\nSTDOUT:\n{out}")
    summary = json.loads(out[start:])

    summary["stderr"] = err
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridstate", type=str, default=str(Path("array") / "gridstate.json"))
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(Path("trials") / "compare_results.json"))

    ap.add_argument("--cem-pop", type=int, default=48)
    ap.add_argument("--cem-iters", type=int, default=8)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--init-sigma", type=float, default=0.7)
    ap.add_argument("--min-sigma", type=float, default=0.08)
    ap.add_argument("--max-abs-log", type=float, default=1.0)

    ap.add_argument("--random-samples", type=int, default=128)
    ap.add_argument("--random-sigma", type=float, default=0.35)

    args = ap.parse_args()

    mpc = str(Path(__file__).resolve().parent / "run_mpc_control.py")

    base = [
        sys.executable,
        mpc,
        "--gridstate",
        str(args.gridstate),
        "--horizon",
        str(int(args.horizon)),
        "--gamma",
        str(float(args.gamma)),
        "--seed",
        str(int(args.seed)),
    ]

    specs: List[RunSpec] = [
        RunSpec(
            name="gene_random_global",
            args=base
            + [
                "--control-mode",
                "gene",
                "--optimizer",
                "random",
                "--action-mode",
                "global",
                "--round",
                "--samples",
                str(int(args.random_samples)),
                "--sigma",
                str(float(args.random_sigma)),
                "--max-abs-log",
                str(float(args.max_abs_log)),
                "--out",
                str(Path("trials") / "cmp_gene_random_global.json"),
            ],
        ),
        RunSpec(
            name="gene_cem_global",
            args=base
            + [
                "--control-mode",
                "gene",
                "--optimizer",
                "cem",
                "--action-mode",
                "global",
                "--round",
                "--pop",
                str(int(args.cem_pop)),
                "--iters",
                str(int(args.cem_iters)),
                "--elite-frac",
                str(float(args.elite_frac)),
                "--init-sigma",
                str(float(args.init_sigma)),
                "--min-sigma",
                str(float(args.min_sigma)),
                "--max-abs-log",
                str(float(args.max_abs_log)),
                "--out",
                str(Path("trials") / "cmp_gene_cem_global.json"),
            ],
        ),
        RunSpec(
            name="gene_cem_two_region",
            args=base
            + [
                "--control-mode",
                "gene",
                "--optimizer",
                "cem",
                "--action-mode",
                "two_region",
                "--round",
                "--pop",
                str(int(args.cem_pop)),
                "--iters",
                str(int(args.cem_iters)),
                "--elite-frac",
                str(float(args.elite_frac)),
                "--init-sigma",
                str(float(args.init_sigma)),
                "--min-sigma",
                str(float(args.min_sigma)),
                "--max-abs-log",
                str(float(args.max_abs_log)),
                "--out",
                str(Path("trials") / "cmp_gene_cem_two_region.json"),
            ],
        ),
        RunSpec(
            name="protein_cem_teacher",
            args=base
            + [
                "--control-mode",
                "protein",
                "--optimizer",
                "cem",
                "--action-mode",
                "global",
                "--round",
                "--apply-each-step",
                "--pop",
                str(int(args.cem_pop)),
                "--iters",
                str(int(args.cem_iters)),
                "--elite-frac",
                str(float(args.elite_frac)),
                "--init-sigma",
                str(float(args.init_sigma)),
                "--min-sigma",
                str(float(args.min_sigma)),
                "--max-abs-log",
                str(float(args.max_abs_log)),
                "--out",
                str(Path("trials") / "cmp_protein_cem_teacher.json"),
            ],
        ),
        RunSpec(
            name="molecule_cem_teacher",
            args=base
            + [
                "--control-mode",
                "molecule",
                "--optimizer",
                "cem",
                "--action-mode",
                "global",
                "--apply-each-step",
                "--pop",
                str(int(args.cem_pop)),
                "--iters",
                str(int(args.cem_iters)),
                "--elite-frac",
                str(float(args.elite_frac)),
                "--init-sigma",
                str(float(args.init_sigma)),
                "--min-sigma",
                str(float(args.min_sigma)),
                "--max-abs-log",
                str(float(args.max_abs_log)),
                "--out",
                str(Path("trials") / "cmp_molecule_cem_teacher.json"),
            ],
        ),
        RunSpec(
            name="all_cem_teacher",
            args=base
            + [
                "--control-mode",
                "all",
                "--optimizer",
                "cem",
                "--action-mode",
                "two_region",
                "--round",
                "--apply-each-step",
                "--pop",
                str(int(args.cem_pop)),
                "--iters",
                str(int(args.cem_iters)),
                "--elite-frac",
                str(float(args.elite_frac)),
                "--init-sigma",
                str(float(args.init_sigma)),
                "--min-sigma",
                str(float(args.min_sigma)),
                "--max-abs-log",
                str(float(args.max_abs_log)),
                "--out",
                str(Path("trials") / "cmp_all_cem_teacher.json"),
            ],
        ),
    ]

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    for spec in specs:
        r = _run_one(spec.args)
        results.append(
            {
                "name": spec.name,
                "baseline_obj": r.get("baseline_obj"),
                "best_obj": r.get("best_obj"),
                "baseline_h1": r.get("baseline_h1"),
                "best_h1": r.get("best_h1"),
                "baseline_hH": r.get("baseline_hH"),
                "best_hH": r.get("best_hH"),
                "runtime_ms": r.get("runtime_ms"),
            }
        )

    dt_ms = (time.time() - t0) * 1000.0
    out_obj = {
        "gridstate": str(args.gridstate),
        "horizon": int(args.horizon),
        "gamma": float(args.gamma),
        "seed": int(args.seed),
        "runs": results,
        "total_runtime_ms": dt_ms,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(json.dumps(out_obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
