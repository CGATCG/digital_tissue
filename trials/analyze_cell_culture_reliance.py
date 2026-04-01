import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: str
    challenge: str
    provider: str
    model: str
    win: Optional[bool]
    claim_cure_calls: int
    analyze_calls: int
    analyze_cell_mentions: int
    experiments_total: int
    experiments_cell: int
    experiments_organism: int
    cell_frac: float
    cell_last3_frac: float
    cell_first: int
    organism_absent: int
    reliance_score: float
    reliance_label: str
    experiment_timeline: str
    screen_calls: int
    cell_screen_calls: int
    organism_screen_calls: int
    protein_screen_calls: int
    cell_protein_screen_calls: int
    organism_protein_screen_calls: int
    intervention_test_calls: int
    cell_intervention_test_calls: int
    organism_intervention_test_calls: int
    claim_cure_intervention_calls: int


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


def _short(s: Any, n: int = 240) -> str:
    if s is None:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip()
    if len(t) <= n:
        return t
    return t[: max(0, n - 3)] + "..."


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


def _experiment_key_from_path(path: str) -> str:
    p = str(path or "")
    if not p:
        return ""
    parts = [x for x in p.split("/") if x]
    if not parts:
        return p
    return parts[-1]


def _compute_reliance_score(
    *,
    cell_frac: float,
    cell_last3_frac: float,
    cell_first: int,
    organism_absent: int,
    analyze_calls: int,
    analyze_cell_mentions: int,
) -> float:
    a_frac = 0.0
    if analyze_calls > 0:
        a_frac = min(1.0, float(analyze_cell_mentions) / float(max(1, analyze_calls)))
    score = 2.0 * float(cell_frac) + 1.0 * float(cell_last3_frac) + 0.5 * float(cell_first) + 1.5 * float(organism_absent) + 0.5 * float(a_frac)
    return float(score)


def _label_from_score(score: float) -> str:
    if score >= 3.0:
        return "high"
    if score >= 1.5:
        return "medium"
    return "low"


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= 0 or deny <= 0:
        return float("nan")
    return float(num / (denx * deny))


def _perm_test_diff_means(scores: List[float], y_loss: List[int], iters: int = 20000, seed: int = 0) -> float:
    rnd = random.Random(int(seed))
    idx_loss = [i for i, y in enumerate(y_loss) if y == 1]
    idx_win = [i for i, y in enumerate(y_loss) if y == 0]
    if not idx_loss or not idx_win:
        return float("nan")

    obs = (sum(scores[i] for i in idx_loss) / len(idx_loss)) - (sum(scores[i] for i in idx_win) / len(idx_win))

    n = len(scores)
    y = list(y_loss)
    count = 0
    for _ in range(int(iters)):
        rnd.shuffle(y)
        idx_loss_p = [i for i, yy in enumerate(y) if yy == 1]
        idx_win_p = [i for i, yy in enumerate(y) if yy == 0]
        if not idx_loss_p or not idx_win_p:
            continue
        diff = (sum(scores[i] for i in idx_loss_p) / len(idx_loss_p)) - (sum(scores[i] for i in idx_win_p) / len(idx_win_p))
        if abs(diff) >= abs(obs):
            count += 1
    return float((count + 1) / float(iters + 1))


def analyze_run(run_id: str, repo_root: Path) -> Optional[RunResult]:
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
    win: Optional[bool] = None

    analyze_calls = 0
    analyze_cell_mentions = 0

    exp: List[Tuple[str, str]] = []

    screen_calls = 0
    cell_screen_calls = 0
    organism_screen_calls = 0
    protein_screen_calls = 0
    cell_protein_screen_calls = 0
    organism_protein_screen_calls = 0

    intervention_test_calls = 0
    cell_intervention_test_calls = 0
    organism_intervention_test_calls = 0

    claim_cure_intervention_calls = 0

    for ev in _read_json_lines(events_path):
        ev_type = str(ev.get("type") or "")
        if ev_type == "start":
            challenge = str(ev.get("challenge") or challenge)
            provider = str(ev.get("provider") or provider)
            model = str(ev.get("model") or model)
            continue

        if ev_type != "api":
            continue

        method = str(ev.get("method") or "").upper()
        path = str(ev.get("path") or "")
        body = ev.get("body") if isinstance(ev.get("body"), dict) else {}
        query = ev.get("query") if isinstance(ev.get("query"), dict) else {}
        resp = ev.get("response_json") if isinstance(ev.get("response_json"), dict) else {}

        if path.endswith("/claim_cure"):
            claim_cure_calls += 1
            interventions = body.get("interventions")
            if isinstance(interventions, list) and len(interventions) > 0:
                claim_cure_intervention_calls += 1
            w = _parse_bool(resp.get("win"))
            if w is not None:
                win = w
            continue

        if path == "/api/omics/analyze" and method == "POST":
            analyze_calls += 1
            instr = body.get("instructions")
            if isinstance(instr, str) and "cell" in instr.lower():
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
        exp.append((exp_key, model_s))

        is_cell = _is_cell_model(model_s)
        is_org = _is_organism_model(model_s)
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

        interventions = body.get("interventions")
        has_interventions = isinstance(interventions, list) and len(interventions) > 0
        if has_interventions:
            intervention_test_calls += 1
            if is_cell:
                cell_intervention_test_calls += 1
            if is_org:
                organism_intervention_test_calls += 1

    experiments_total = len(exp)
    experiments_cell = sum(1 for _, m in exp if _is_cell_model(m))
    experiments_organism = sum(1 for _, m in exp if _is_organism_model(m))

    cell_frac = float(experiments_cell) / float(experiments_total) if experiments_total > 0 else 0.0

    last3 = exp[-3:] if len(exp) >= 3 else list(exp)
    cell_last3_frac = 0.0
    if last3:
        cell_last3_frac = float(sum(1 for _, m in last3 if _is_cell_model(m))) / float(len(last3))

    cell_first = 0
    if exp:
        cell_first = 1 if _is_cell_model(exp[0][1]) else 0

    organism_absent = 1 if (experiments_organism == 0 and experiments_cell > 0) else 0

    score = _compute_reliance_score(
        cell_frac=cell_frac,
        cell_last3_frac=cell_last3_frac,
        cell_first=cell_first,
        organism_absent=organism_absent,
        analyze_calls=analyze_calls,
        analyze_cell_mentions=analyze_cell_mentions,
    )
    label = _label_from_score(score)

    timeline_parts = []
    for k, m in exp[:30]:
        mm = str(m or "")
        if _is_cell_model(mm):
            mm2 = "cell"
        elif _is_organism_model(mm):
            mm2 = "organism"
        elif mm.strip():
            mm2 = _short(mm, 24)
        else:
            mm2 = "unknown"
        timeline_parts.append(f"{k}[{mm2}]")
    if len(exp) > 30:
        timeline_parts.append("...")
        for k, m in exp[-10:]:
            mm = str(m or "")
            if _is_cell_model(mm):
                mm2 = "cell"
            elif _is_organism_model(mm):
                mm2 = "organism"
            elif mm.strip():
                mm2 = _short(mm, 24)
            else:
                mm2 = "unknown"
            timeline_parts.append(f"{k}[{mm2}]")

    return RunResult(
        run_id=str(run_id),
        run_dir=str(run_dir),
        challenge=str(challenge),
        provider=str(provider),
        model=str(model),
        win=win,
        claim_cure_calls=int(claim_cure_calls),
        analyze_calls=int(analyze_calls),
        analyze_cell_mentions=int(analyze_cell_mentions),
        experiments_total=int(experiments_total),
        experiments_cell=int(experiments_cell),
        experiments_organism=int(experiments_organism),
        cell_frac=float(cell_frac),
        cell_last3_frac=float(cell_last3_frac),
        cell_first=int(cell_first),
        organism_absent=int(organism_absent),
        reliance_score=float(score),
        reliance_label=str(label),
        experiment_timeline=";".join(timeline_parts),
        screen_calls=int(screen_calls),
        cell_screen_calls=int(cell_screen_calls),
        organism_screen_calls=int(organism_screen_calls),
        protein_screen_calls=int(protein_screen_calls),
        cell_protein_screen_calls=int(cell_protein_screen_calls),
        organism_protein_screen_calls=int(organism_protein_screen_calls),
        intervention_test_calls=int(intervention_test_calls),
        cell_intervention_test_calls=int(cell_intervention_test_calls),
        organism_intervention_test_calls=int(organism_intervention_test_calls),
        claim_cure_intervention_calls=int(claim_cure_intervention_calls),
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

    # stable unique
    out: List[str] = []
    seen = set()
    for rid in run_ids:
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", action="append", default=[])
    ap.add_argument("--run-ids-file", default="")
    ap.add_argument(
        "--out",
        default="",
        help="CSV output path. Default: var/runs/llm_bench/analyses/cell_culture_reliance_report.csv under repo root.",
    )
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    run_ids = _read_run_ids(args)
    if not run_ids:
        print("No run_ids provided", file=sys.stderr)
        return 2

    out_path = str(args.out or "").strip()
    if not out_path:
        out_path = str(repo_root / "var" / "runs" / "llm_bench" / "analyses" / "cell_culture_reliance_report.csv")
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    missing: List[str] = []
    for rid in run_ids:
        rr = analyze_run(rid, repo_root)
        if rr is None:
            missing.append(rid)
            continue
        results.append(rr)

    cols = [
        "run_id",
        "run_dir",
        "challenge",
        "provider",
        "model",
        "win",
        "claim_cure_calls",
        "claim_cure_intervention_calls",
        "analyze_calls",
        "analyze_cell_mentions",
        "experiments_total",
        "experiments_cell",
        "experiments_organism",
        "screen_calls",
        "cell_screen_calls",
        "organism_screen_calls",
        "protein_screen_calls",
        "cell_protein_screen_calls",
        "organism_protein_screen_calls",
        "intervention_test_calls",
        "cell_intervention_test_calls",
        "organism_intervention_test_calls",
        "cell_frac",
        "cell_last3_frac",
        "cell_first",
        "organism_absent",
        "reliance_score",
        "reliance_label",
        "experiment_timeline",
    ]

    with open(str(outp), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            d = r.__dict__.copy()
            d["win"] = "" if r.win is None else str(bool(r.win))
            w.writerow({k: d.get(k, "") for k in cols})

    outp_metrics_long = outp.with_name(outp.stem + "__metrics_long.csv")
    outp_metrics_summary = outp.with_name(outp.stem + "__metrics_summary.csv")

    usable = [r for r in results if r.win is not None]
    losses = [r for r in usable if r.win is False]
    wins = [r for r in usable if r.win is True]

    metric_defs: List[Tuple[str, str]] = [
        ("used_cell_screen_any", "binary"),
        ("cell_screen_calls", "continuous"),
        ("used_cell_protein_screen_any", "binary"),
        ("cell_protein_screen_calls", "continuous"),
        ("used_any_screen", "binary"),
        ("screen_calls", "continuous"),
        ("used_cell_intervention_test_any", "binary"),
        ("cell_intervention_test_calls", "continuous"),
        ("used_any_intervention_test", "binary"),
        ("intervention_test_calls", "continuous"),
        ("used_claim_cure_any", "binary"),
        ("claim_cure_calls", "continuous"),
        ("used_claim_cure_with_interventions", "binary"),
        ("claim_cure_intervention_calls", "continuous"),
        ("used_cell_experiment_any", "binary"),
        ("experiments_cell", "continuous"),
        ("used_organism_experiment_any", "binary"),
        ("experiments_organism", "continuous"),
        ("experiments_total", "continuous"),
        ("cell_frac", "continuous"),
        ("cell_last3_frac", "continuous"),
        ("organism_absent", "binary"),
        ("analyze_calls", "continuous"),
        ("analyze_cell_mentions", "continuous"),
        ("reliance_score", "continuous"),
    ]

    def _metric_value(r: RunResult, metric: str) -> float:
        if metric == "used_cell_screen_any":
            return float(1 if r.cell_screen_calls > 0 else 0)
        if metric == "cell_screen_calls":
            return float(r.cell_screen_calls)
        if metric == "used_cell_protein_screen_any":
            return float(1 if r.cell_protein_screen_calls > 0 else 0)
        if metric == "cell_protein_screen_calls":
            return float(r.cell_protein_screen_calls)
        if metric == "used_any_screen":
            return float(1 if r.screen_calls > 0 else 0)
        if metric == "screen_calls":
            return float(r.screen_calls)
        if metric == "used_cell_intervention_test_any":
            return float(1 if r.cell_intervention_test_calls > 0 else 0)
        if metric == "cell_intervention_test_calls":
            return float(r.cell_intervention_test_calls)
        if metric == "used_any_intervention_test":
            return float(1 if r.intervention_test_calls > 0 else 0)
        if metric == "intervention_test_calls":
            return float(r.intervention_test_calls)
        if metric == "used_claim_cure_any":
            return float(1 if r.claim_cure_calls > 0 else 0)
        if metric == "claim_cure_calls":
            return float(r.claim_cure_calls)
        if metric == "used_claim_cure_with_interventions":
            return float(1 if r.claim_cure_intervention_calls > 0 else 0)
        if metric == "claim_cure_intervention_calls":
            return float(r.claim_cure_intervention_calls)
        if metric == "used_cell_experiment_any":
            return float(1 if r.experiments_cell > 0 else 0)
        if metric == "experiments_cell":
            return float(r.experiments_cell)
        if metric == "used_organism_experiment_any":
            return float(1 if r.experiments_organism > 0 else 0)
        if metric == "experiments_organism":
            return float(r.experiments_organism)
        if metric == "experiments_total":
            return float(r.experiments_total)
        if metric == "cell_frac":
            return float(r.cell_frac)
        if metric == "cell_last3_frac":
            return float(r.cell_last3_frac)
        if metric == "organism_absent":
            return float(r.organism_absent)
        if metric == "analyze_calls":
            return float(r.analyze_calls)
        if metric == "analyze_cell_mentions":
            return float(r.analyze_cell_mentions)
        if metric == "reliance_score":
            return float(r.reliance_score)
        raise KeyError(metric)

    cols_long = [
        "run_id",
        "challenge",
        "provider",
        "model",
        "win",
        "loss",
        "reliance_label",
        "metric",
        "kind",
        "value",
    ]
    with open(str(outp_metrics_long), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols_long)
        w.writeheader()
        for r in results:
            win_s = "" if r.win is None else str(bool(r.win))
            loss_s = "" if r.win is None else str(bool(not r.win))
            for metric, kind in metric_defs:
                w.writerow(
                    {
                        "run_id": r.run_id,
                        "challenge": r.challenge,
                        "provider": r.provider,
                        "model": r.model,
                        "win": win_s,
                        "loss": loss_s,
                        "reliance_label": r.reliance_label,
                        "metric": metric,
                        "kind": kind,
                        "value": _metric_value(r, metric),
                    }
                )

    cols_summary = [
        "metric",
        "kind",
        "win_n",
        "loss_n",
        "win_true",
        "loss_true",
        "win_true_frac",
        "loss_true_frac",
        "diff_true_frac_loss_minus_win",
        "win_mean",
        "loss_mean",
        "diff_mean_loss_minus_win",
    ]

    def _mean(xs: List[float]) -> float:
        if not xs:
            return float("nan")
        return float(sum(xs) / float(len(xs)))

    with open(str(outp_metrics_summary), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols_summary)
        w.writeheader()
        for metric, kind in metric_defs:
            win_vals = [float(_metric_value(r, metric)) for r in wins]
            loss_vals = [float(_metric_value(r, metric)) for r in losses]
            row: Dict[str, Any] = {
                "metric": metric,
                "kind": kind,
                "win_n": len(win_vals),
                "loss_n": len(loss_vals),
                "win_true": "",
                "loss_true": "",
                "win_true_frac": "",
                "loss_true_frac": "",
                "diff_true_frac_loss_minus_win": "",
                "win_mean": "",
                "loss_mean": "",
                "diff_mean_loss_minus_win": "",
            }
            if kind == "binary":
                win_true = int(sum(1 for v in win_vals if v != 0.0))
                loss_true = int(sum(1 for v in loss_vals if v != 0.0))
                win_frac = float(win_true) / float(len(win_vals)) if win_vals else float("nan")
                loss_frac = float(loss_true) / float(len(loss_vals)) if loss_vals else float("nan")
                row["win_true"] = win_true
                row["loss_true"] = loss_true
                row["win_true_frac"] = win_frac
                row["loss_true_frac"] = loss_frac
                if not (math.isnan(win_frac) or math.isnan(loss_frac)):
                    row["diff_true_frac_loss_minus_win"] = float(loss_frac - win_frac)
            else:
                win_mean = _mean(win_vals)
                loss_mean = _mean(loss_vals)
                row["win_mean"] = win_mean
                row["loss_mean"] = loss_mean
                if not (math.isnan(win_mean) or math.isnan(loss_mean)):
                    row["diff_mean_loss_minus_win"] = float(loss_mean - win_mean)
            w.writerow(row)

    scores = [float(r.reliance_score) for r in usable]
    y_loss = [1 if r.win is False else 0 for r in usable]

    diff_means = float("nan")
    if wins and losses:
        diff_means = (sum(r.reliance_score for r in losses) / len(losses)) - (sum(r.reliance_score for r in wins) / len(wins))

    corr = _pearson(scores, [float(y) for y in y_loss])
    p = _perm_test_diff_means(scores, y_loss, iters=int(args.permutations), seed=int(args.seed))

    by_label: Dict[str, List[RunResult]] = {"low": [], "medium": [], "high": []}
    for r in usable:
        by_label.setdefault(r.reliance_label, []).append(r)

    def _loss_rate(rs: List[RunResult]) -> float:
        if not rs:
            return float("nan")
        return float(sum(1 for r in rs if r.win is False) / float(len(rs)))

    print(f"Wrote CSV: {outp}")
    print(f"Wrote CSV: {outp_metrics_long}")
    print(f"Wrote CSV: {outp_metrics_summary}")
    print(f"Runs requested: {len(run_ids)}")
    print(f"Runs found: {len(results)} (missing={len(missing)})")
    print(f"Runs with win/loss (claim_cure): {len(usable)}")
    print(f"Wins: {len(wins)} Losses: {len(losses)}")
    print(f"Mean reliance_score(loss) - mean reliance_score(win): {diff_means:.4f}")
    print(f"Pearson corr(reliance_score, loss): {corr:.4f}")
    print(f"Permutation p-value (diff in means): {p:.6f}")
    print("Loss rate by reliance_label:")
    for k in ("low", "medium", "high"):
        rs = by_label.get(k, [])
        if not rs:
            continue
        print(f"  {k}: n={len(rs)} loss_rate={_loss_rate(rs):.3f} mean_score={(sum(r.reliance_score for r in rs)/len(rs)):.3f}")

    top_losses = sorted([r for r in losses], key=lambda r: r.reliance_score, reverse=True)[:10]
    if top_losses:
        print("Top high-reliance losses:")
        for r in top_losses:
            print(f"  {r.run_id} score={r.reliance_score:.3f} cell_frac={r.cell_frac:.2f} exp_cell/org={r.experiments_cell}/{r.experiments_organism} timeline={_short(r.experiment_timeline, 200)}")

    if missing:
        print("Missing run directories:")
        for rid in missing[:50]:
            print(f"  {rid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
