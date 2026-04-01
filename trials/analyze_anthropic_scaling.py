import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    run_dir: str
    challenge: str
    provider: str
    model: str
    model_family: str
    win: Optional[bool]

    claim_cure_calls: int
    claim_cure_intervention_calls: int
    claim_cure_empty_calls: int
    claim_cure_avg_n_interventions: float
    claim_cure_max_n_interventions: int
    claim_cure_unique_layers: int

    analyze_calls: int
    analyze_cell_mentions: int
    analyze_instr_chars_total: int

    experiments_total: int
    experiments_cell: int
    experiments_organism: int

    screen_calls: int
    cell_screen_calls: int
    organism_screen_calls: int

    protein_screen_calls: int
    cell_protein_screen_calls: int
    organism_protein_screen_calls: int

    bulk_omics_calls: int
    cell_bulk_omics_calls: int
    organism_bulk_omics_calls: int

    characterization_calls: int
    cell_characterization_calls: int
    organism_characterization_calls: int

    cell_frac: float
    cell_last3_frac: float
    cell_first: int
    organism_absent: int

    seq_first_experiment: int
    seq_first_cell_experiment: int
    seq_first_organism_experiment: int
    seq_first_claim_cure: int
    seq_first_claim_cure_with_interventions: int
    seq_first_analyze: int


def _read_json_lines(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except Exception:
        return


def _resolve_run_dir(run_id: str, repo_root: Path) -> Optional[Path]:
    run_id = str(run_id or "").strip()
    if not run_id:
        return None

    candidates = [
        repo_root / "var" / "runs" / "llm_bench" / run_id,
        repo_root / "runs" / "llm_bench" / run_id,
        repo_root / "var" / "runs" / "llm_bench" / "runs" / run_id,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def _parse_bool(x: Any) -> Optional[bool]:
    if x is True:
        return True
    if x is False:
        return False
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def _is_cell_model(model: str) -> bool:
    s = str(model or "").lower()
    if "cell_culture" in s:
        return True
    if "in_vitro" in s or "in-vitro" in s:
        return True
    return False


def _is_organism_model(model: str) -> bool:
    s = str(model or "").lower()
    if not s:
        return False
    if _is_cell_model(s):
        return False
    if "organism" in s or "in_vivo" in s or "in-vivo" in s:
        return True
    if s in ("healthy", "disease"):
        return True
    if s.startswith("healthy_") or s.startswith("cancer_") or s.endswith("_organism"):
        if "cell" not in s:
            return True
    return False


def _experiment_key_from_path(path: str) -> str:
    p = str(path or "")
    if not p:
        return ""
    parts = [x for x in p.split("/") if x]
    if not parts:
        return p
    return parts[-1]


def _model_family(model: str) -> str:
    s = str(model or "").lower()
    if "haiku" in s:
        return "haiku"
    if "sonnet" in s:
        return "sonnet"
    if "opus" in s:
        return "opus"
    return "other"


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / float(len(xs)))


def analyze_run(run_id: str, repo_root: Path) -> Optional[RunMetrics]:
    run_dir = _resolve_run_dir(run_id, repo_root)
    if run_dir is None:
        return None
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return None

    provider = ""
    model = ""
    challenge = ""

    claim_cure_calls = 0
    claim_cure_intervention_calls = 0
    claim_cure_empty_calls = 0
    claim_cure_n_interventions: List[int] = []
    claim_cure_layers = set()
    win: Optional[bool] = None

    analyze_calls = 0
    analyze_cell_mentions = 0
    analyze_instr_chars_total = 0

    exp: List[Tuple[int, str, str]] = []  # (seq, key, model)

    screen_calls = 0
    cell_screen_calls = 0
    organism_screen_calls = 0

    protein_screen_calls = 0
    cell_protein_screen_calls = 0
    organism_protein_screen_calls = 0

    bulk_omics_calls = 0
    cell_bulk_omics_calls = 0
    organism_bulk_omics_calls = 0

    characterization_calls = 0
    cell_characterization_calls = 0
    organism_characterization_calls = 0

    seq_first_experiment: Optional[int] = None
    seq_first_cell_experiment: Optional[int] = None
    seq_first_organism_experiment: Optional[int] = None
    seq_first_claim_cure: Optional[int] = None
    seq_first_claim_cure_with_interventions: Optional[int] = None
    seq_first_analyze: Optional[int] = None

    for ev in _read_json_lines(events_path):
        ev_type = str(ev.get("type") or "")
        if ev_type == "start":
            challenge = str(ev.get("challenge") or challenge)
            provider = str(ev.get("provider") or provider)
            model = str(ev.get("model") or model)
            continue

        if ev_type != "api":
            continue

        seq = ev.get("seq")
        try:
            seq_i = int(seq)
        except Exception:
            seq_i = -1

        method = str(ev.get("method") or "").upper()
        path = str(ev.get("path") or "")
        body = ev.get("body") if isinstance(ev.get("body"), dict) else {}
        query = ev.get("query") if isinstance(ev.get("query"), dict) else {}
        resp = ev.get("response_json") if isinstance(ev.get("response_json"), dict) else {}

        if path.endswith("/claim_cure"):
            claim_cure_calls += 1
            if seq_first_claim_cure is None:
                seq_first_claim_cure = seq_i

            interventions = body.get("interventions")
            if isinstance(interventions, list):
                n_int = len(interventions)
                claim_cure_n_interventions.append(n_int)
                if n_int > 0:
                    claim_cure_intervention_calls += 1
                    if seq_first_claim_cure_with_interventions is None:
                        seq_first_claim_cure_with_interventions = seq_i
                    for it in interventions:
                        if isinstance(it, dict):
                            layer = it.get("layer")
                            if isinstance(layer, str) and layer:
                                claim_cure_layers.add(layer)
                else:
                    claim_cure_empty_calls += 1
            else:
                claim_cure_n_interventions.append(0)
                claim_cure_empty_calls += 1

            w = _parse_bool(resp.get("win"))
            if w is not None:
                win = w
            continue

        if path == "/api/omics/analyze" and method == "POST":
            analyze_calls += 1
            if seq_first_analyze is None:
                seq_first_analyze = seq_i
            instr = body.get("instructions")
            if isinstance(instr, str):
                analyze_instr_chars_total += len(instr)
                if "cell" in instr.lower():
                    analyze_cell_mentions += 1
            continue

        if not path.startswith("/api/tests/"):
            continue
        if method != "POST":
            continue

        if path.endswith("/estimate_cost"):
            continue
        if path.endswith("/models") or path.endswith("/proteins"):
            continue

        exp_key = _experiment_key_from_path(path)
        m = body.get("model")
        if m is None:
            m = query.get("model")
        model_s = str(m or "")

        exp.append((seq_i, exp_key, model_s))

        if seq_first_experiment is None:
            seq_first_experiment = seq_i

        is_cell = _is_cell_model(model_s)
        is_org = _is_organism_model(model_s)
        if is_cell and seq_first_cell_experiment is None:
            seq_first_cell_experiment = seq_i
        if is_org and seq_first_organism_experiment is None:
            seq_first_organism_experiment = seq_i

        is_screen = bool(exp_key) and exp_key.endswith("_screen")
        is_protein_screen = exp_key == "protein_screen"

        if is_screen:
            screen_calls += 1
            if is_cell:
                cell_screen_calls += 1
            if is_org:
                organism_screen_calls += 1

        if is_protein_screen:
            protein_screen_calls += 1
            if is_cell:
                cell_protein_screen_calls += 1
            if is_org:
                organism_protein_screen_calls += 1

        if exp_key == "bulk_omics":
            bulk_omics_calls += 1
            if is_cell:
                cell_bulk_omics_calls += 1
            if is_org:
                organism_bulk_omics_calls += 1

        if exp_key == "characterization":
            characterization_calls += 1
            if is_cell:
                cell_characterization_calls += 1
            if is_org:
                organism_characterization_calls += 1

    experiments_total = len(exp)
    experiments_cell = sum(1 for _, _, m in exp if _is_cell_model(m))
    experiments_organism = sum(1 for _, _, m in exp if _is_organism_model(m))

    cell_frac = float(experiments_cell) / float(experiments_total) if experiments_total > 0 else 0.0

    last3 = exp[-3:] if len(exp) >= 3 else list(exp)
    cell_last3_frac = 0.0
    if last3:
        cell_last3_frac = float(sum(1 for _, _, m in last3 if _is_cell_model(m))) / float(len(last3))

    cell_first = 0
    if exp:
        cell_first = 1 if _is_cell_model(exp[0][2]) else 0

    organism_absent = 1 if (experiments_organism == 0 and experiments_cell > 0) else 0

    avg_n_int = _mean([float(x) for x in claim_cure_n_interventions]) if claim_cure_n_interventions else 0.0
    max_n_int = int(max(claim_cure_n_interventions)) if claim_cure_n_interventions else 0

    def _seq_or_neg1(x: Optional[int]) -> int:
        return int(x) if x is not None else -1

    return RunMetrics(
        run_id=str(run_id),
        run_dir=str(run_dir),
        challenge=str(challenge),
        provider=str(provider),
        model=str(model),
        model_family=_model_family(model),
        win=win,
        claim_cure_calls=int(claim_cure_calls),
        claim_cure_intervention_calls=int(claim_cure_intervention_calls),
        claim_cure_empty_calls=int(claim_cure_empty_calls),
        claim_cure_avg_n_interventions=float(avg_n_int),
        claim_cure_max_n_interventions=int(max_n_int),
        claim_cure_unique_layers=int(len(claim_cure_layers)),
        analyze_calls=int(analyze_calls),
        analyze_cell_mentions=int(analyze_cell_mentions),
        analyze_instr_chars_total=int(analyze_instr_chars_total),
        experiments_total=int(experiments_total),
        experiments_cell=int(experiments_cell),
        experiments_organism=int(experiments_organism),
        screen_calls=int(screen_calls),
        cell_screen_calls=int(cell_screen_calls),
        organism_screen_calls=int(organism_screen_calls),
        protein_screen_calls=int(protein_screen_calls),
        cell_protein_screen_calls=int(cell_protein_screen_calls),
        organism_protein_screen_calls=int(organism_protein_screen_calls),
        bulk_omics_calls=int(bulk_omics_calls),
        cell_bulk_omics_calls=int(cell_bulk_omics_calls),
        organism_bulk_omics_calls=int(organism_bulk_omics_calls),
        characterization_calls=int(characterization_calls),
        cell_characterization_calls=int(cell_characterization_calls),
        organism_characterization_calls=int(organism_characterization_calls),
        cell_frac=float(cell_frac),
        cell_last3_frac=float(cell_last3_frac),
        cell_first=int(cell_first),
        organism_absent=int(organism_absent),
        seq_first_experiment=_seq_or_neg1(seq_first_experiment),
        seq_first_cell_experiment=_seq_or_neg1(seq_first_cell_experiment),
        seq_first_organism_experiment=_seq_or_neg1(seq_first_organism_experiment),
        seq_first_claim_cure=_seq_or_neg1(seq_first_claim_cure),
        seq_first_claim_cure_with_interventions=_seq_or_neg1(seq_first_claim_cure_with_interventions),
        seq_first_analyze=_seq_or_neg1(seq_first_analyze),
    )


def _read_run_ids(args: argparse.Namespace) -> List[str]:
    run_ids: List[str] = []
    for rid in getattr(args, "run_id", []) or []:
        rid = str(rid or "").strip()
        if rid:
            run_ids.append(rid)

    fpath = getattr(args, "run_ids_file", None)
    if fpath:
        p = Path(str(fpath))
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="replace")
            for token in re.split(r"\s+", txt.strip()):
                if token.startswith("run_"):
                    run_ids.append(token.strip())

    if not run_ids:
        stdin = sys.stdin.read()
        for token in re.split(r"\s+", str(stdin or "").strip()):
            if token.startswith("run_"):
                run_ids.append(token.strip())

    out: List[str] = []
    seen = set()
    for rid in run_ids:
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def _family_summary(rows: List[RunMetrics]) -> List[Dict[str, Any]]:
    families = sorted({r.model_family for r in rows})

    metrics: List[Tuple[str, str]] = [
        ("win_rate", "continuous"),
        ("claim_cure_calls", "continuous"),
        ("claim_cure_empty_calls", "continuous"),
        ("claim_cure_intervention_calls", "continuous"),
        ("claim_cure_avg_n_interventions", "continuous"),
        ("claim_cure_unique_layers", "continuous"),
        ("analyze_calls", "continuous"),
        ("analyze_cell_mentions", "continuous"),
        ("analyze_instr_chars_total", "continuous"),
        ("experiments_total", "continuous"),
        ("experiments_cell", "continuous"),
        ("experiments_organism", "continuous"),
        ("cell_frac", "continuous"),
        ("cell_last3_frac", "continuous"),
        ("screen_calls", "continuous"),
        ("cell_screen_calls", "continuous"),
        ("protein_screen_calls", "continuous"),
        ("cell_protein_screen_calls", "continuous"),
        ("bulk_omics_calls", "continuous"),
        ("organism_bulk_omics_calls", "continuous"),
        ("characterization_calls", "continuous"),
        ("organism_characterization_calls", "continuous"),
        ("organism_absent", "continuous"),
        ("seq_first_organism_experiment", "continuous"),
        ("seq_first_claim_cure_with_interventions", "continuous"),
        ("seq_first_analyze", "continuous"),
    ]

    out: List[Dict[str, Any]] = []
    for fam in families:
        rs = [r for r in rows if r.model_family == fam]
        usable = [r for r in rs if r.win is not None]
        win_rate = float(sum(1 for r in usable if r.win is True) / float(len(usable))) if usable else float("nan")

        def values(metric: str) -> List[float]:
            if metric == "win_rate":
                return [win_rate]
            return [float(getattr(r, metric)) for r in rs]

        row: Dict[str, Any] = {
            "model_family": fam,
            "n_runs": len(rs),
            "n_with_win": len(usable),
            "n_wins": sum(1 for r in usable if r.win is True),
            "n_losses": sum(1 for r in usable if r.win is False),
            "win_rate": win_rate,
        }
        for metric, _kind in metrics:
            if metric == "win_rate":
                continue
            row[metric + "_mean"] = _mean(values(metric))
        out.append(row)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", action="append", default=[])
    ap.add_argument("--run-ids-file", default="")
    ap.add_argument(
        "--out",
        default="",
        help="CSV output path. Default: var/runs/llm_bench/analyses/anthropic_scaling_report.csv under repo root.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    run_ids = _read_run_ids(args)
    if not run_ids:
        print("No run_ids provided", file=sys.stderr)
        return 2

    out_path = str(args.out or "").strip()
    if not out_path:
        out_path = str(repo_root / "var" / "runs" / "llm_bench" / "analyses" / "anthropic_scaling_report.csv")
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    results: List[RunMetrics] = []
    missing: List[str] = []
    for rid in run_ids:
        rr = analyze_run(rid, repo_root)
        if rr is None:
            missing.append(rid)
            continue
        results.append(rr)

    cols = [f.name for f in RunMetrics.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    with open(str(outp), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            d = r.__dict__.copy()
            d["win"] = "" if r.win is None else str(bool(r.win))
            w.writerow({k: d.get(k, "") for k in cols})

    outp_family = outp.with_name(outp.stem + "__family_summary.csv")
    family_rows = _family_summary(results)
    if family_rows:
        family_cols = list(family_rows[0].keys())
        with open(str(outp_family), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=family_cols)
            w.writeheader()
            for r in family_rows:
                w.writerow({k: r.get(k, "") for k in family_cols})

    print(f"Wrote CSV: {outp}")
    if family_rows:
        print(f"Wrote CSV: {outp_family}")
    print(f"Runs requested: {len(run_ids)}")
    print(f"Runs found: {len(results)} (missing={len(missing)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
