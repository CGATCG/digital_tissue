import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _default_server_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _check_import(python_exe: str, module: str) -> None:
    p = subprocess.run(
        [python_exe, "-c", f"import {module}"],
        cwd=str(REPO_ROOT),
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Preflight failed: {python_exe} cannot import '{module}'.\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get_json(base_url: str, path: str, timeout_s: float = 10.0):
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _http_post_json(base_url: str, path: str, payload: dict, timeout_s: float = 60.0):
    url = base_url.rstrip("/") + path
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out_raw = resp.read()
        return json.loads(out_raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read() or b""
        except Exception:
            body = b""
        msg = body.decode("utf-8", errors="replace") if body else str(e)
        raise RuntimeError(f"HTTP {e.code} POST {path} failed: {msg}")


def _wait_for_health(base_url: str, timeout_s: float = 10.0) -> None:
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            out = _http_get_json(base_url, "/api/health", timeout_s=2.0)
            if out and out.get("ok") is True:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(f"Server did not become healthy within {timeout_s}s. Last error: {last_err}")


class _StepFail(Exception):
    pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise _StepFail(msg)


def _load_payload(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"payload not object: {path}")
    return obj


def _run_steps(base_url: str) -> None:
    healthy = _load_payload(REPO_ROOT / "documents" / "stable_healthy.json")
    sick = _load_payload(REPO_ROOT / "documents" / "stable.json")
    aging = _load_payload(REPO_ROOT / "documents" / "healthy_aging.json")

    interventions = [
        {"layer": "molecule_glucose", "direction": "down", "dose": 1},
    ]

    player_id = "smoke_player"

    gene_set = ""
    try:
        gs = _http_get_json(base_url, "/api/spatial_tx/gene_sets", timeout_s=10.0)
        cand = gs.get("gene_sets") if isinstance(gs, dict) else None
        if isinstance(cand, list):
            for nm in cand:
                if isinstance(nm, str) and nm.strip():
                    gene_set = nm.strip()
                    break
    except Exception:
        gene_set = ""
    _assert(bool(gene_set), "spatial gene_set available")

    omics_set = ""
    try:
        bs = _http_get_json(base_url, "/api/bulk_omics/sets", timeout_s=10.0)
        cand2 = bs.get("sets") if isinstance(bs, dict) else None
        if isinstance(cand2, list):
            for nm in cand2:
                if isinstance(nm, str) and nm.strip().lower().startswith("rna/"):
                    omics_set = nm.strip()
                    break
            if not omics_set:
                for nm in cand2:
                    if isinstance(nm, str) and nm.strip():
                        omics_set = nm.strip()
                        break
    except Exception:
        omics_set = ""
    _assert(bool(omics_set), "bulk omics_set available")

    req = {
        "player_id": player_id,
        "ticks": 2,
        "replicates": 1,
        "seed": 1,
        "gene_set": gene_set,
        "healthy": healthy,
        "sick": sick,
        "interventions": interventions,
    }

    out = _http_post_json(base_url, "/api/experiments/spatial_tx", req, timeout_s=120.0)
    _assert(out.get("ok") is True, "ok")
    _assert(out.get("experiment") == "spatial_tx_v1", "experiment type")

    game = out.get("game")
    _assert(isinstance(game, dict), "game present")
    _assert(game.get("player_id") == player_id, "player_id returned")
    _assert(isinstance(game.get("money_spent_cents"), int), "money_spent_cents int")

    runs = out.get("runs")
    _assert(isinstance(runs, list) and len(runs) == 2, "2 runs (healthy+sick)")

    conds = sorted([str(r.get("condition") or "") for r in runs if isinstance(r, dict)])
    _assert(conds == ["healthy", "sick"], "run conditions")

    genes = out.get("genes")
    _assert(isinstance(genes, list) and len(genes) >= 1, "genes list")

    _assert(out.get("gene_set") == gene_set, "gene_set")

    matrix_csv = out.get("matrix_csv")
    truth_csv = out.get("matrix_truth_csv")
    noisy_csv = out.get("matrix_noisy_csv")
    meta_csv = out.get("metadata_csv")
    _assert(isinstance(matrix_csv, str) and len(matrix_csv) > 0, "matrix_csv present")
    _assert(isinstance(truth_csv, str) and len(truth_csv) > 0, "matrix_truth_csv present")
    _assert(isinstance(noisy_csv, str) and len(noisy_csv) > 0, "matrix_noisy_csv present")
    _assert(isinstance(meta_csv, str) and len(meta_csv) > 0, "metadata_csv present")

    matrix_lines = [ln for ln in matrix_csv.splitlines() if ln.strip()]
    truth_lines = [ln for ln in truth_csv.splitlines() if ln.strip()]
    noisy_lines = [ln for ln in noisy_csv.splitlines() if ln.strip()]
    meta_lines = [ln for ln in meta_csv.splitlines() if ln.strip()]
    _assert(len(matrix_lines) >= 2, "matrix has header + rows")
    _assert(len(truth_lines) >= 2, "truth matrix has header + rows")
    _assert(len(noisy_lines) >= 2, "noisy matrix has header + rows")
    _assert(len(meta_lines) >= 2, "metadata has header + rows")

    _assert(matrix_lines[0] == noisy_lines[0], "matrix_csv header matches matrix_noisy_csv header")
    _assert(len(truth_lines) == len(noisy_lines), "truth/noisy row counts match")
    _assert(len(meta_lines) == len(truth_lines), "metadata row count matches matrix row count")

    # Basic headers
    _assert(matrix_lines[0].startswith("cell_id,"), "matrix header starts with cell_id")
    _assert("x" in meta_lines[0].split(","), "metadata has x")
    _assert("y" in meta_lines[0].split(","), "metadata has y")

    # Spot-check a few gene columns from the noisy matrix
    header2 = noisy_lines[0].split(",")
    row2 = noisy_lines[1].split(",")
    _assert(header2 and header2[0] == "cell_id", "matrix header cell_id (noisy)")
    _assert(len(row2) == len(header2), "matrix row matches header (noisy)")
    for i in range(1, min(len(row2), 6)):
        try:
            v = int(float(row2[i]))
        except Exception:
            raise _StepFail("noisy matrix contains non-numeric value")
        _assert(v >= 0, "noisy matrix has non-negative counts")

    req2 = {
        "player_id": player_id,
        "ticks": 2,
        "replicates": 1,
        "seed": 1,
        "omics_set": omics_set,
        "healthy": healthy,
        "sick": sick,
        "interventions": interventions,
    }

    out2 = _http_post_json(base_url, "/api/experiments/bulk_omics", req2, timeout_s=120.0)
    _assert(out2.get("ok") is True, "bulk ok")
    _assert(out2.get("experiment") == "bulk_omics_v1", "bulk experiment type")

    game2 = out2.get("game")
    _assert(isinstance(game2, dict), "bulk game present")
    _assert(game2.get("player_id") == player_id, "bulk player_id returned")
    _assert(isinstance(game2.get("money_spent_cents"), int), "bulk money_spent_cents int")
    _assert(int(game2.get("money_spent_cents") or 0) > int(game.get("money_spent_cents") or 0), "money spent increases")

    _assert(out2.get("omics_set") == omics_set, "omics_set")
    genes2 = out2.get("genes")
    _assert(isinstance(genes2, list) and len(genes2) >= 1, "bulk genes list")

    # Reset money and ensure it goes back to zero
    out_reset = _http_post_json(base_url, "/api/game/reset", {"player_id": player_id}, timeout_s=30.0)
    _assert(out_reset.get("ok") is True, "reset ok")
    game_reset = out_reset.get("game")
    _assert(isinstance(game_reset, dict), "reset game present")
    _assert(int(game_reset.get("money_spent_cents") or 0) == 0, "money reset to 0")

    truth_csv2 = out2.get("matrix_truth_csv")
    noisy_csv2 = out2.get("matrix_noisy_csv")
    meta_csv2 = out2.get("metadata_csv")
    _assert(isinstance(truth_csv2, str) and len(truth_csv2) > 0, "bulk truth matrix present")
    _assert(isinstance(noisy_csv2, str) and len(noisy_csv2) > 0, "bulk noisy matrix present")
    _assert(isinstance(meta_csv2, str) and len(meta_csv2) > 0, "bulk metadata present")

    truth_lines2 = [ln for ln in truth_csv2.splitlines() if ln.strip()]
    noisy_lines2 = [ln for ln in noisy_csv2.splitlines() if ln.strip()]
    meta_lines2 = [ln for ln in meta_csv2.splitlines() if ln.strip()]
    _assert(len(truth_lines2) >= 2, "bulk truth has header + rows")
    _assert(len(noisy_lines2) >= 2, "bulk noisy has header + rows")
    _assert(len(meta_lines2) >= 2, "bulk metadata has header + rows")
    _assert(len(truth_lines2) == len(noisy_lines2), "bulk truth/noisy row counts match")
    _assert(len(meta_lines2) == len(truth_lines2), "bulk metadata row count matches")

    rowb = noisy_lines2[1].split(",")
    _assert(len(rowb) >= 2, "bulk row has at least one feature")
    try:
        v = int(float(rowb[1]))
    except Exception:
        raise _StepFail("bulk noisy matrix contains non-numeric value")
    _assert(v >= 0, "bulk noisy matrix has non-negative counts")

    req3 = {
        "player_id": player_id,
        "ticks": 5,
        "replicates": 2,
        "seed": 1,
        "healthy": healthy,
        "sick": sick,
        "interventions": interventions,
    }

    out3 = _http_post_json(base_url, "/api/experiments/in_vivo_trial", req3, timeout_s=120.0)
    _assert(out3.get("ok") is True, "in vivo ok")
    _assert(out3.get("experiment") == "in_vivo_trial_v1", "in vivo experiment type")
    ticks_out = int(out3.get("ticks") or 0)
    _assert(ticks_out >= 5, "in vivo ticks >= requested")
    _assert(int(out3.get("requested_ticks") or 0) == 5, "in vivo requested_ticks")
    _assert(int(out3.get("replicates") or 0) == 2, "in vivo replicates")

    game3 = out3.get("game")
    _assert(isinstance(game3, dict), "in vivo game present")
    _assert(game3.get("player_id") == player_id, "in vivo player_id returned")
    _assert(isinstance(game3.get("money_spent_cents"), int), "in vivo money_spent_cents int")

    meas = out3.get("measurements")
    _assert(isinstance(meas, list) and len(meas) >= 1, "in vivo measurements list")
    series = out3.get("series")
    _assert(isinstance(series, dict), "in vivo series present")
    sh = series.get("healthy")
    ss = series.get("sick")
    _assert(isinstance(sh, dict) and isinstance(ss, dict), "in vivo series healthy/sick dict")
    k0 = str(meas[0])
    _assert(k0 in sh and k0 in ss, "series contains first measurement")
    _assert(isinstance(sh.get(k0), list) and len(sh.get(k0)) == ticks_out, "healthy series length")
    _assert(isinstance(ss.get(k0), list) and len(ss.get(k0)) == ticks_out, "sick series length")

    cure = out3.get("cure")
    _assert(isinstance(cure, dict), "in vivo cure present")
    _assert(isinstance(cure.get("score_pct"), (int, float)), "in vivo cure score_pct")
    _assert(isinstance(cure.get("win"), bool), "in vivo cure win bool")

    req4 = {
        "payload": aging,
        "ticks": 50,
        "replicates": 20,
        "seed": 1,
    }
    out4 = _http_post_json(base_url, "/api/lifespan/run", req4, timeout_s=120.0)
    _assert(out4.get("ok") is True, "lifespan ok")
    _assert(out4.get("experiment") == "lifespan_v1", "lifespan experiment type")
    _assert(int(out4.get("ticks") or 0) == 50, "lifespan ticks")
    _assert(int(out4.get("replicates") or 0) == 20, "lifespan replicates")
    dn = out4.get("death_measurements")
    _assert(isinstance(dn, list) and len(dn) >= 1, "lifespan death_measurements list")
    reps = out4.get("replicates_out")
    _assert(isinstance(reps, list) and len(reps) == 20, "lifespan replicate outputs")
    curve = out4.get("curve")
    _assert(isinstance(curve, dict), "lifespan curve present")
    times = curve.get("times")
    surv = curve.get("survival")
    _assert(isinstance(times, list) and len(times) == 51, "lifespan curve times")
    _assert(isinstance(surv, list) and len(surv) == 51, "lifespan curve survival")
    _assert(isinstance(curve.get("deaths"), int), "lifespan curve deaths int")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--python", type=str, default="")
    args = ap.parse_args()

    port = int(args.port or 0)
    if port <= 0:
        port = _pick_free_port()

    python_exe = str(args.python or "").strip() or _default_server_python()
    _check_import(python_exe, "numpy")

    with tempfile.TemporaryDirectory(prefix="dt_docs_") as docs_tmp, tempfile.TemporaryDirectory(prefix="dt_ws_") as ws_tmp:
        docs_dir = Path(docs_tmp)
        workspace_dir = Path(ws_tmp)

        env = dict(os.environ)
        env["DT_DOCS_DIR"] = str(docs_dir)
        env["DT_WORKSPACE_DIR"] = str(workspace_dir)
        env["PYTHONUNBUFFERED"] = "1"

        stdout_path = docs_dir / "server_stdout.txt"
        stderr_path = docs_dir / "server_stderr.txt"

        stdout_f = open(stdout_path, "wb")
        stderr_f = open(stderr_path, "wb")

        proc = subprocess.Popen(
            [python_exe, str(REPO_ROOT / "runtime_server.py"), str(port)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
        )
        base_url = f"http://127.0.0.1:{port}"

        try:
            _wait_for_health(base_url, timeout_s=float(args.timeout))
            _run_steps(base_url)
            print("PASS: experiments api smoke test")
            return 0
        except _StepFail as e:
            print(f"FAIL: {e}")
            return 2
        finally:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                stdout_f.close()
            except Exception:
                pass
            try:
                stderr_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
