import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunScan:
    benchmark_validity: str
    errors: str
    analyze_calls: int
    analyze_failures: int
    api_error_calls: int
    claim_cure_calls: int
    tests_calls: int
    files_returned_calls: int


def _short(s: Any, n: int = 240) -> str:
    if s is None:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip()
    if len(t) <= n:
        return t
    return t[: max(0, n - 3)] + "..."


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


def _extract_error_text(resp: Any) -> str:
    if isinstance(resp, dict):
        for k in ("error", "message", "detail"):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                vv = v.get("message") or v.get("detail") or v.get("error")
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
        return ""
    if isinstance(resp, str):
        return resp.strip()
    return ""


def _is_true(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def _scan_run(events_path: Path) -> RunScan:
    if not events_path.exists() or not events_path.is_file():
        return RunScan(
            benchmark_validity="FLAG: missing_events_jsonl",
            errors="missing events.jsonl",
            analyze_calls=0,
            analyze_failures=0,
            api_error_calls=0,
            claim_cure_calls=0,
            tests_calls=0,
            files_returned_calls=0,
        )

    events = list(_read_json_lines(events_path))

    analyze_calls = 0
    analyze_failures = 0
    api_error_calls = 0
    claim_cure_calls = 0
    tests_calls = 0
    files_returned_calls = 0

    reasons: List[str] = []

    api_events: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for ev in events:
        ev_type = str(ev.get("type") or "").strip()

        if ev_type in ("error", "llm_error", "runner_error"):
            reasons.append(f"event_type={ev_type}")
            continue

        if ev_type != "api":
            continue

        path = str(ev.get("path") or "")
        http_status = ev.get("http_status")
        resp = ev.get("response_json")
        resp_ok = resp.get("ok") if isinstance(resp, dict) else None

        method = str(ev.get("method") or "").strip().upper()
        seq = ev.get("seq")

        if path.startswith("/api/tests/"):
            tests_calls += 1

        files_field = ev.get("files")
        if isinstance(files_field, list) and len(files_field) > 0:
            files_returned_calls += 1

        if path.endswith("/claim_cure"):
            claim_cure_calls += 1

        err_txt = _extract_error_text(resp)
        api_ev = {
            "seq": seq,
            "method": method,
            "path": path,
            "http_status": http_status,
            "resp_ok": resp_ok,
            "err_txt": err_txt,
            "analyze_diag_failed": False,
        }

        if isinstance(http_status, int) and http_status >= 400:
            api_error_calls += 1
            if err_txt:
                reasons.append(f"api_http_{http_status}:{path}:{_short(err_txt, 200)}")
            else:
                reasons.append(f"api_http_{http_status}:{path}")

            failures.append(
                {
                    "seq": seq,
                    "method": method,
                    "path": path,
                    "http_status": http_status,
                    "kind": "http_error",
                    "message": err_txt,
                }
            )

        if path == "/api/omics/analyze":
            analyze_calls += 1

            # Hard failures
            if http_status != 200 or resp_ok is False:
                analyze_failures += 1
                if err_txt:
                    reasons.append(f"analyze_failed:{_short(err_txt, 200)}")
                else:
                    reasons.append("analyze_failed")
                failures.append(
                    {
                        "seq": seq,
                        "method": method,
                        "path": path,
                        "http_status": http_status,
                        "kind": "analyze_http_or_ok_failed",
                        "message": err_txt,
                    }
                )
                api_events.append(api_ev)
                continue

            # Soft failures: code execution produced an error but API returned ok=true
            if isinstance(resp, dict):
                out_text = resp.get("output_text")
                if isinstance(out_text, str) and "ANALYZE_FAILED" in out_text:
                    analyze_failures += 1
                    reasons.append("analyze_diagnostics_failed")
                    api_ev["analyze_diag_failed"] = True
                    failures.append(
                        {
                            "seq": seq,
                            "method": method,
                            "path": path,
                            "http_status": http_status,
                            "kind": "analyze_diagnostics_failed",
                            "message": "ANALYZE_FAILED",
                        }
                    )
                    api_events.append(api_ev)
                    continue
                diag = resp.get("analysis_diagnostics")
                if isinstance(diag, dict):
                    failed = diag.get("failed")
                    if failed is True:
                        analyze_failures += 1
                        reasons.append("analyze_diagnostics_failed")
                        api_ev["analyze_diag_failed"] = True
                        failures.append(
                            {
                                "seq": seq,
                                "method": method,
                                "path": path,
                                "http_status": http_status,
                                "kind": "analyze_diagnostics_failed",
                                "message": _extract_error_text(diag) or "analysis_diagnostics.failed",
                            }
                        )
                        api_events.append(api_ev)
                        continue

        api_events.append(api_ev)

    # Quota/credits detection as a reason (common technical unfairness)
    quota_markers = ("insufficient_quota", "out of credits", "credits", "billing", "payment required")
    for r in list(reasons):
        if any(m in r.lower() for m in quota_markers):
            if "quota_or_credits" not in reasons:
                reasons.append("quota_or_credits")
            break

    # Decide validity string
    if analyze_failures > 0:
        validity = f"FLAG: analyze_failed (failures={analyze_failures}, calls={analyze_calls})"
    elif analyze_calls == 0 and files_returned_calls > 0:
        validity = f"FLAG: no_analyze_despite_files (files_calls={files_returned_calls})"
    elif analyze_calls == 0 and claim_cure_calls >= 5:
        validity = f"FLAG: claim_cure_guessing_no_analyze (claim_cure_calls={claim_cure_calls})"
    elif analyze_calls <= 1 and claim_cure_calls >= 10:
        validity = f"FLAG: claim_cure_spam_low_analyze (claim_cure_calls={claim_cure_calls}, analyze_calls={analyze_calls})"
    elif api_error_calls > 0:
        validity = f"FLAG: api_errors (n={api_error_calls})"
    elif reasons:
        validity = "FLAG: " + "; ".join(sorted(set(reasons))[:3])
    else:
        validity = "OK"

    errors = ""
    if validity != "OK":
        # Build an error narrative based on failure points and what happened afterward.
        parts: List[str] = []
        failures_sorted = [f for f in failures if isinstance(f, dict)]
        failures_sorted.sort(key=lambda x: (x.get("seq") is None, x.get("seq")))

        if failures_sorted:
            # Summarize each failing endpoint.
            by_path: Dict[str, List[Dict[str, Any]]] = {}
            for f in failures_sorted:
                by_path.setdefault(str(f.get("path") or ""), []).append(f)

            for pth, fs in by_path.items():
                first = fs[0]
                first_seq = first.get("seq")
                first_status = first.get("http_status")
                first_kind = str(first.get("kind") or "")
                first_msg = _short(first.get("message") or "", 160)

                # What happened immediately after the failure (next API call overall)?
                next_api: Optional[Dict[str, Any]] = None
                for e in api_events:
                    if first_seq is None:
                        continue
                    if e.get("seq") is None:
                        continue
                    if int(e.get("seq")) > int(first_seq):
                        next_api = e
                        break
                next_note = ""
                if isinstance(next_api, dict):
                    nxt_path = str(next_api.get("path") or "")
                    nxt_status = next_api.get("http_status")
                    if isinstance(nxt_status, int):
                        next_note = f"next={nxt_path} (HTTP {int(nxt_status)})"
                    else:
                        next_note = f"next={nxt_path}"

                # Determine retry + recovery behavior.
                subsequent = [
                    e
                    for e in api_events
                    if str(e.get("path") or "") == pth and (first_seq is None or (e.get("seq") is not None and e.get("seq") > first_seq))
                ]
                retried = len(subsequent)
                recovered = any(
                    (isinstance(e.get("http_status"), int) and int(e.get("http_status")) < 400)
                    and (e.get("resp_ok") is not False)
                    and (e.get("analyze_diag_failed") is not True)
                    for e in subsequent
                )

                retry_note = "no_retry"
                if retried > 0:
                    retry_note = f"retried={retried}, recovered={'yes' if recovered else 'no'}"

                head = f"{pth}"
                if isinstance(first_status, int):
                    head += f" HTTP {int(first_status)}"
                if first_kind and first_kind != "http_error":
                    head += f" ({first_kind})"
                if first_msg:
                    head += f": {first_msg}"
                head += f" [{retry_note}]"
                if next_note:
                    head += f" [{next_note}]"
                parts.append(head)

            # After the last failure, what did the run do?
            last_seq = failures_sorted[-1].get("seq")
            after = [e for e in api_events if last_seq is None or (e.get("seq") is not None and e.get("seq") > last_seq)]
            after_analyze = sum(1 for e in after if e.get("path") == "/api/omics/analyze")
            after_claim = sum(1 for e in after if str(e.get("path") or "").endswith("/claim_cure"))
            after_tests = sum(1 for e in after if str(e.get("path") or "").startswith("/api/tests/"))
            if after:
                parts.append(
                    f"after_last_error: analyze_calls={after_analyze}, claim_cure_calls={after_claim}, tests_calls={after_tests}"
                )
        else:
            # Flagged due to ok_false / guessing heuristics without explicit HTTP failures.
            parts.append(
                f"run_summary: analyze_calls={analyze_calls}, analyze_failures={analyze_failures}, api_error_calls={api_error_calls}, "
                f"files_returned_calls={files_returned_calls}, claim_cure_calls={claim_cure_calls}, tests_calls={tests_calls}"
            )
            if analyze_calls == 0 and files_returned_calls > 0:
                parts.append("pattern: produced_files_but_never_called_analyze")
            if analyze_calls == 0 and claim_cure_calls >= 5:
                parts.append("pattern: claim_cure_probing_without_analyze")
            if analyze_calls <= 1 and claim_cure_calls >= 10:
                parts.append("pattern: claim_cure_spam_with_low_analyze")

        errors = " | ".join([_short(p, 420) for p in parts])

    return RunScan(
        benchmark_validity=validity,
        errors=errors,
        analyze_calls=analyze_calls,
        analyze_failures=analyze_failures,
        api_error_calls=api_error_calls,
        claim_cure_calls=claim_cure_calls,
        tests_calls=tests_calls,
        files_returned_calls=files_returned_calls,
    )


def _discover_suite_ids(suites_root: Path) -> List[str]:
    out: List[str] = []
    for p in suites_root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("suite_"):
            continue
        if (p / "suite_summary.csv").exists():
            out.append(p.name)
    out.sort()
    return out


def _resolve_run_dir(*, run_id: str, run_dir_from_csv: str, runs_root: Path) -> Path:
    p = Path(str(run_dir_from_csv or "").strip())
    if str(p) and p.exists() and p.is_dir():
        return p
    # Fallback to canonical runs_root/run_id
    if str(run_id or "").strip():
        p2 = runs_root / str(run_id)
        if p2.exists() and p2.is_dir():
            return p2
    return p if str(p) else (runs_root / str(run_id))


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge suite_summary.csv files and add a benchmark validity flag per run.")
    ap.add_argument(
        "--suites-root",
        default="/home/cga/Desktop/digital_tissue/var/runs/llm_bench/suites",
        help="Directory containing suite_* folders.",
    )
    ap.add_argument(
        "--runs-root",
        default="/home/cga/Desktop/digital_tissue/var/runs/llm_bench",
        help="Directory containing run_* folders.",
    )
    ap.add_argument("--suite-id", action="append", default=[], help="Suite id(s) to include (repeatable).")
    ap.add_argument(
        "--out",
        default="/home/cga/Desktop/digital_tissue/var/runs/llm_bench/suites/combined_suite_summary__validity.csv",
        help="Output CSV path.",
    )
    args = ap.parse_args()

    suites_root = Path(str(args.suites_root))
    runs_root = Path(str(args.runs_root))
    out_path = Path(str(args.out))

    suite_ids: List[str] = [str(s).strip() for s in (args.suite_id or []) if str(s).strip()]
    if not suite_ids:
        suite_ids = _discover_suite_ids(suites_root)

    all_rows: List[Dict[str, Any]] = []

    for sid in suite_ids:
        summ = suites_root / sid / "suite_summary.csv"
        if not summ.exists():
            continue
        with open(str(summ), "r", encoding="utf-8", errors="replace") as f:
            rr = csv.DictReader(f)
            for r in rr:
                if not isinstance(r, dict):
                    continue

                run_id = str(r.get("run_id") or "").strip()
                run_dir = _resolve_run_dir(run_id=run_id, run_dir_from_csv=str(r.get("run_dir") or ""), runs_root=runs_root)
                events_path = run_dir / "events.jsonl"

                scan = _scan_run(events_path)

                row = dict(r)
                row["_source_suite_summary"] = str(summ)
                row["benchmark_validity"] = scan.benchmark_validity
                row["errors"] = scan.errors
                row["benchmark_validity__analyze_calls"] = scan.analyze_calls
                row["benchmark_validity__analyze_failures"] = scan.analyze_failures
                row["benchmark_validity__api_error_calls"] = scan.api_error_calls
                row["benchmark_validity__claim_cure_calls"] = scan.claim_cure_calls
                row["benchmark_validity__tests_calls"] = scan.tests_calls
                row["benchmark_validity__files_returned_calls"] = scan.files_returned_calls

                if not _is_true(row.get("ok")):
                    # Keep this explicit even if events look fine
                    if str(row.get("benchmark_validity") or "") == "OK":
                        row["benchmark_validity"] = "FLAG: ok_false"
                    else:
                        row["benchmark_validity"] = str(row["benchmark_validity"]) + "; ok_false"
                    if str(row.get("errors") or "").strip():
                        row["errors"] = str(row["errors"]) + " | suite_summary_ok=false"
                    else:
                        row["errors"] = "suite_summary_ok=false"

                all_rows.append(row)

    # Determine column order
    base_cols: List[str] = []
    if all_rows:
        # Prefer the canonical suite_summary order if present
        preferred = [
            "suite_id",
            "run_id",
            "run_dir",
            "challenge",
            "provider",
            "model",
            "replicate",
            "player_id",
            "exit_code",
            "ok",
            "benchmark_validity",
            "errors",
            "benchmark_validity__analyze_calls",
            "benchmark_validity__analyze_failures",
            "benchmark_validity__api_error_calls",
            "benchmark_validity__claim_cure_calls",
            "benchmark_validity__tests_calls",
            "benchmark_validity__files_returned_calls",
            "max_step",
            "steps_completed",
            "seconds_total",
            "llm_calls",
            "tool_calls",
            "api_calls",
            "experiment_calls",
            "win",
            "final_delta_median_ticks",
            "best_extra_days",
            "best_lifespan_recovery_pct",
            "best_score",
            "best_score_lifedays_per_usd",
            "best_score_seq",
            "money_spent_usd",
            "money_spent_cents",
        ]
        seen = set()
        for c in preferred:
            if any(c in r for r in all_rows):
                base_cols.append(c)
                seen.add(c)

        extra_cols = sorted({k for r in all_rows for k in r.keys() if k not in seen})
        cols = base_cols + extra_cols
    else:
        cols = ["suite_id", "run_id", "benchmark_validity", "_source_suite_summary"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in cols})

    print(f"Wrote {len(all_rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
