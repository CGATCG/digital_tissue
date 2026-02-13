import argparse
import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _guess_mime_type(path: Optional[Path], *, name: str = "") -> str:
    suf = ""
    if path is not None:
        suf = str(path.suffix or "").lower()
    if not suf and name:
        suf = str(Path(name).suffix or "").lower()

    if suf == ".csv":
        return "text/csv"
    if suf == ".tsv":
        return "text/tab-separated-values"
    if suf == ".txt":
        return "text/plain"
    if suf == ".json":
        return "application/json"
    return "application/octet-stream"


def _mask_disease_term(text: str) -> str:
    t = str(text or "")
    if not t:
        return ""
    t = t.replace("/api/tests/cancer/", "/api/tests/disease/")
    t = t.replace("/api/tests/cancer", "/api/tests/disease")
    t = t.replace("/api/tests/hereditary_disease/", "/api/tests/disease/")
    t = t.replace("/api/tests/hereditary_disease", "/api/tests/disease")
    t = t.replace("/api/tests/aging/", "/api/tests/disease/")
    t = t.replace("/api/tests/aging", "/api/tests/disease")
    t = re.sub(r"cancerous", "diseased", t, flags=re.IGNORECASE)
    t = re.sub(r"cancer", "disease", t, flags=re.IGNORECASE)
    return t


def _server_instructions_prefix() -> str:
    return (
        "You will be communicating with an LLM. Do not generate files, plots, or charts. "
        "Describe your findings in as much detail as possible using pure text. "
        "If helpful, you may include a small table in plain text."
    )


def _mask_text_file_bytes(raw_bytes: bytes) -> bytes:
    try:
        txt0 = raw_bytes.decode("utf-8", errors="replace")
        txt1 = _mask_disease_term(txt0)
        return txt1.encode("utf-8")
    except Exception:
        return raw_bytes


def _truncate_bytes(raw: bytes, *, max_bytes: int) -> bytes:
    try:
        n = int(max_bytes)
    except Exception:
        n = 0
    if n <= 0:
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return b""
    b = bytes(raw)
    if len(b) <= n:
        return b
    return b[:n]


def _filter_resolved(
    resolved: List[Dict[str, Any]],
    *,
    include_name_regex: Optional[str],
    exclude_name_regex: Optional[str],
    max_files: Optional[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    inc = None
    exc = None
    try:
        if isinstance(include_name_regex, str) and include_name_regex.strip():
            inc = re.compile(str(include_name_regex))
    except Exception:
        inc = None
    try:
        if isinstance(exclude_name_regex, str) and exclude_name_regex.strip():
            exc = re.compile(str(exclude_name_regex))
    except Exception:
        exc = None

    lim = None
    try:
        if max_files is not None:
            lim = max(0, int(max_files))
    except Exception:
        lim = None

    for r in resolved:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or "")
        if inc is not None and not bool(inc.search(nm)):
            continue
        if exc is not None and bool(exc.search(nm)):
            continue
        out.append(r)
        if lim is not None and lim > 0 and len(out) >= lim:
            break
    return out


def _retry_delay_seconds_from_error_text(txt: str) -> Optional[float]:
    s = str(txt or "")
    if not s.strip():
        return None
    try:
        m = re.search(r"retryDelay\"\s*:\s*\"\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*\"", s, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"Please\s+retry\s+in\s*([0-9]+(?:\.[0-9]+)?)\s*s", s, flags=re.IGNORECASE)
        if not m:
            return None
        v = float(m.group(1))
        if not (v >= 0.0):
            return None
        return float(min(300.0, v))
    except Exception:
        return None


def _omics_file_id(run_id: str, name: str) -> str:
    s = f"{str(run_id)}:{str(name)}".encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _default_omics_runs_dir() -> Path:
    try:
        return (Path(__file__).resolve().parent.parent / "workspace" / "omics_runs").resolve()
    except Exception:
        return Path("workspace/omics_runs").resolve()


def _resolve_file_ids_from_omics_runs(
    *,
    omics_runs_dir: Path,
    player_id: str,
    file_ids: List[str],
) -> List[Dict[str, Any]]:
    want = [str(x or "").strip() for x in (file_ids or [])]
    want = [x for x in want if x]
    if not want:
        return []

    base = omics_runs_dir.resolve()
    if not base.exists() or not base.is_dir():
        raise ValueError(f"omics_runs_dir not found: {str(base)}")

    found: Dict[str, Dict[str, Any]] = {}

    for ent in base.iterdir():
        if not ent.is_dir():
            continue
        mp = (ent / "manifest.json").resolve()
        if not mp.exists() or not mp.is_file():
            continue
        try:
            mf = json.loads(mp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(mf, dict):
            continue
        if str(mf.get("player_id") or "") != str(player_id):
            continue

        run_id = str(mf.get("run_id") or ent.name or "").strip()
        if not run_id:
            continue
        files = mf.get("files")
        if not isinstance(files, list):
            continue

        for f in files:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name") or "").strip()
            if not name:
                continue
            fid = _omics_file_id(run_id, name)
            if fid not in want or fid in found:
                continue

            p = (base / run_id / name).resolve()
            try:
                if base not in p.parents:
                    continue
            except Exception:
                continue
            if not p.exists() or not p.is_file():
                continue

            try:
                raw = p.read_bytes()
            except Exception:
                raw = b""

            found[fid] = {
                "file_id": str(fid),
                "run_id": str(run_id),
                "name": str(name),
                "path": str(p),
                "bytes": int(len(raw or b"")),
                "raw_bytes": raw,
            }

        if len(found) >= len(want):
            break

    missing = [fid for fid in want if fid not in found]
    if missing:
        raise ValueError(
            "Could not resolve some file_ids for this player_id from omics_runs_dir. "
            f"missing={missing} omics_runs_dir={str(base)}"
        )

    out: List[Dict[str, Any]] = []
    for fid in want:
        ent = found.get(str(fid))
        if ent is not None:
            out.append(ent)
    return out


def _load_analyze_request_from_events_jsonl(
    *,
    events_jsonl_path: Path,
    seq: Optional[int],
) -> Dict[str, Any]:
    p = events_jsonl_path.expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise ValueError(f"events_jsonl not found: {str(p)}")

    first_503: Optional[Dict[str, Any]] = None
    last_any: Optional[Dict[str, Any]] = None
    exact_seq: Optional[Dict[str, Any]] = None

    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = str(ln or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type") or "") != "api":
            continue
        if str(obj.get("path") or "") != "/api/omics/analyze":
            continue

        last_any = obj

        if seq is not None:
            try:
                if int(obj.get("seq")) == int(seq):
                    exact_seq = obj
                    break
            except Exception:
                pass

        try:
            st = int(obj.get("http_status") or 0)
        except Exception:
            st = 0
        if st == 503 and first_503 is None:
            first_503 = obj

    use = exact_seq if exact_seq is not None else (first_503 if first_503 is not None else last_any)
    if not isinstance(use, dict):
        raise ValueError("no /api/omics/analyze api events found")

    body = use.get("body")
    if not isinstance(body, dict):
        raise ValueError("events.jsonl analyze event missing body")

    return {
        "seq": use.get("seq"),
        "http_status": use.get("http_status"),
        "player_id": body.get("player_id"),
        "file_ids": body.get("file_ids"),
        "instructions": body.get("instructions"),
        "provider": body.get("provider"),
        "model": body.get("model"),
    }


def _post_json(*, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: float) -> Tuple[int, str, Dict[str, Any]]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(str(url), data=raw, method="POST", headers=dict(headers))

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            b = resp.read()
            txt = b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else str(b)
    except urllib.error.HTTPError as e:
        status = int(getattr(e, "code", 0) or 0)
        try:
            b = e.read()
            txt = b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else str(b)
        except Exception:
            txt = str(e)
    except Exception as e:
        return 0, str(e), {}

    obj: Dict[str, Any] = {}
    try:
        parsed = json.loads(txt) if isinstance(txt, str) and txt.strip() else None
        obj = parsed if isinstance(parsed, dict) else {}
    except Exception:
        obj = {}

    return status, txt, obj


def _count_tokens(
    *,
    base_url: str,
    model: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Tuple[int, str, Dict[str, Any]]:
    url = str(base_url).rstrip("/") + "/models/" + str(model) + ":countTokens"
    headers = {
        "x-goog-api-key": str(api_key),
        "content-type": "application/json",
    }
    req_payload: Dict[str, Any] = {}
    contents = payload.get("contents")
    if isinstance(contents, list):
        req_payload["contents"] = contents
    return _post_json(url=url, headers=headers, payload=req_payload, timeout_s=float(timeout_s))


def _payload_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    inline_parts = 0
    inline_data_total_len_chars = 0

    try:
        contents = payload.get("contents")
        if isinstance(contents, list):
            for c in contents:
                if not isinstance(c, dict):
                    continue
                parts = c.get("parts")
                if not isinstance(parts, list):
                    continue
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    inline = p.get("inlineData")
                    if not isinstance(inline, dict):
                        continue
                    inline_parts += 1
                    data = inline.get("data")
                    if isinstance(data, str):
                        inline_data_total_len_chars += int(len(data))
    except Exception:
        pass

    raw_json_len = 0
    try:
        raw_json_len = int(len(json.dumps(payload, ensure_ascii=False)))
    except Exception:
        raw_json_len = 0

    return {
        "inline_parts": int(inline_parts),
        "inline_data_total_len_chars": int(inline_data_total_len_chars),
        "payload_json_len_chars": int(raw_json_len),
    }


def _make_inline_part_from_bytes(*, data: bytes, mime_type: str) -> Dict[str, Any]:
    b64 = base64.b64encode(data or b"").decode("ascii")
    return {"inlineData": {"mimeType": str(mime_type), "data": str(b64)}}


def _make_synthetic_csv_bytes(*, target_bytes: int, cols: int = 12) -> bytes:
    n = max(0, int(target_bytes))
    if n <= 0:
        return b"col0\n"

    header = ",".join([f"col{i}" for i in range(int(max(1, cols)))]) + "\n"
    row = ",".join(["0" for _ in range(int(max(1, cols)))]) + "\n"

    out = bytearray(header.encode("utf-8"))
    row_b = row.encode("utf-8")

    while len(out) + len(row_b) <= n:
        out.extend(row_b)

    if len(out) < n:
        out.extend(b"\n" * int(n - len(out)))

    return bytes(out)


def _build_payload(
    *,
    instructions: str,
    file_parts: List[Dict[str, Any]],
    max_output_tokens: int,
    thinking_level: Optional[str],
) -> Dict[str, Any]:
    parts: list[Dict[str, Any]] = [{"text": str(instructions)}]
    parts.extend(list(file_parts))

    payload: Dict[str, Any] = {
        "tools": [{"code_execution": {}}],
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": int(max(1, int(max_output_tokens))),
        },
    }
    if thinking_level:
        payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": str(thinking_level)}
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default="", help="Gemini API key. Defaults to env GEMINI_API_KEY.")
    ap.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta", help="Gemini API base URL.")
    ap.add_argument("--model", default="gemini-3-pro-preview", help="Gemini model name.")
    ap.add_argument("--max-output-tokens", type=int, default=8192, help="generationConfig.maxOutputTokens")
    ap.add_argument("--thinking", default="", help="Set thinkingLevel (e.g. high). Empty => match server default.")
    ap.add_argument("--instructions", default="Summarize each CSV and report basic stats.", help="Prompt text.")
    ap.add_argument("--instructions-file", default="", help="Optional path to a text file containing the prompt.")
    ap.add_argument("--no-server-prefix", action="store_true", help="Do not prepend the same instruction prefix used by /api/omics/analyze.")

    ap.add_argument("--events-jsonl", default="", help="Load a /api/omics/analyze request from an events.jsonl file.")
    ap.add_argument("--events-seq", type=int, default=-1, help="If set, choose the /api/omics/analyze api event with this seq.")

    ap.add_argument("--file", action="append", default=[], help="Path to a local file to include as inlineData. Repeatable.")
    ap.add_argument("--player-id", default="", help="player_id used to resolve file_ids from workspace/omics_runs")
    ap.add_argument("--file-id", action="append", default=[], help="Omics file_id to resolve from workspace/omics_runs. Repeatable.")
    ap.add_argument("--omics-runs-dir", default="", help="Override workspace/omics_runs directory used for file_id resolution")
    ap.add_argument("--include-name-regex", default="", help="Only include resolved omics files whose name matches this regex")
    ap.add_argument("--exclude-name-regex", default="", help="Exclude resolved omics files whose name matches this regex")
    ap.add_argument("--max-files", type=int, default=0, help="Max number of resolved omics files to include (0 => no limit)")
    ap.add_argument("--max-file-bytes", type=int, default=0, help="Truncate each included file to at most this many bytes (0 => no truncation)")
    ap.add_argument("--max-total-bytes", type=int, default=0, help="Stop adding files once this raw-byte total is reached (0 => no cap)")
    ap.add_argument("--gen-csv-bytes", type=int, default=0, help="Generate a synthetic CSV of this size (bytes) and include it as inlineData.")
    ap.add_argument("--gen-csv-count", type=int, default=1, help="Number of synthetic CSVs to include.")

    ap.add_argument("--repeat", type=int, default=1, help="Number of requests to send.")
    ap.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between requests.")
    ap.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout per request.")
    ap.add_argument("--dry-run", action="store_true", help="Build payload and print stats, but do not call the API.")
    ap.add_argument("--auto-retry-429", action="store_true", help="If Gemini returns 429 with retryDelay, sleep and retry (within --timeout budget)")
    ap.add_argument("--count-tokens", action="store_true", help="Call Gemini :countTokens with this payload and print the result")

    args = ap.parse_args()

    if str(args.events_jsonl or "").strip():
        seq0: Optional[int] = None
        try:
            if int(args.events_seq) >= 0:
                seq0 = int(args.events_seq)
        except Exception:
            seq0 = None

        ev = _load_analyze_request_from_events_jsonl(events_jsonl_path=Path(str(args.events_jsonl)), seq=seq0)
        print("EVENTS_SELECTED:", json.dumps(ev, ensure_ascii=False))

        if not str(args.player_id or "").strip():
            args.player_id = str(ev.get("player_id") or "")
        if not (args.file_id or []):
            fids0 = ev.get("file_ids")
            if isinstance(fids0, list):
                args.file_id = [str(x or "").strip() for x in fids0 if str(x or "").strip()]

        if not str(args.instructions_file or "").strip():
            if str(args.instructions or "") == "Summarize each CSV and report basic stats.":
                args.instructions = str(ev.get("instructions") or "")

        if str(args.model or "") == "gemini-3-pro-preview":
            m = str(ev.get("model") or "").strip()
            if m:
                args.model = m

    api_key = str(args.api_key or "").strip() or str(os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key and not (bool(args.dry_run) and not bool(args.count_tokens)):
        raise SystemExit("Missing API key. Provide --api-key or set GEMINI_API_KEY.")

    instructions = str(args.instructions or "")
    if str(args.instructions_file or "").strip():
        instructions = Path(str(args.instructions_file)).read_text(encoding="utf-8", errors="replace")

    instructions_eff = str(instructions)
    if not bool(args.no_server_prefix):
        instructions_eff = _server_instructions_prefix() + "\n\n" + str(instructions_eff)
    instructions_eff = _mask_disease_term(instructions_eff)

    file_parts: List[Dict[str, Any]] = []
    uploaded_bytes_total = 0
    max_total_bytes = 0
    try:
        max_total_bytes = int(args.max_total_bytes or 0)
    except Exception:
        max_total_bytes = 0
    max_total_bytes = max(0, int(max_total_bytes))

    max_file_bytes = 0
    try:
        max_file_bytes = int(args.max_file_bytes or 0)
    except Exception:
        max_file_bytes = 0
    max_file_bytes = max(0, int(max_file_bytes))

    for fp in args.file or []:
        p = Path(str(fp)).expanduser().resolve()
        raw = p.read_bytes()
        mime_type = _guess_mime_type(p)
        if str(p.suffix or "").lower() in (".csv", ".tsv", ".txt", ".json"):
            raw = _mask_text_file_bytes(raw)
        if max_file_bytes > 0:
            raw = _truncate_bytes(raw, max_bytes=int(max_file_bytes))
        raw_len = 0
        try:
            raw_len = int(len(raw or b""))
        except Exception:
            raw_len = 0
        if max_total_bytes > 0 and (int(uploaded_bytes_total) + int(raw_len)) > int(max_total_bytes):
            break
        uploaded_bytes_total = int(uploaded_bytes_total) + int(raw_len)
        file_parts.append(_make_inline_part_from_bytes(data=raw, mime_type=mime_type))

    file_ids = [str(x or "").strip() for x in (args.file_id or [])]
    file_ids = [x for x in file_ids if x]
    if file_ids:
        player_id = str(args.player_id or "").strip()
        if not player_id:
            raise SystemExit("--player-id is required when using --file-id")

        omics_runs_dir = str(args.omics_runs_dir or "").strip()
        omics_dir = Path(omics_runs_dir).expanduser().resolve() if omics_runs_dir else _default_omics_runs_dir()

        resolved0 = _resolve_file_ids_from_omics_runs(omics_runs_dir=omics_dir, player_id=player_id, file_ids=file_ids)
        max_files = None
        try:
            if int(args.max_files or 0) > 0:
                max_files = int(args.max_files)
        except Exception:
            max_files = None
        resolved = _filter_resolved(
            resolved0,
            include_name_regex=str(args.include_name_regex or "").strip() or None,
            exclude_name_regex=str(args.exclude_name_regex or "").strip() or None,
            max_files=max_files,
        )
        print("RESOLVED_FILE_IDS:", json.dumps([{k: v for k, v in r.items() if k != "raw_bytes"} for r in resolved], ensure_ascii=False))

        for r in resolved:
            name = str(r.get("name") or "")
            raw = r.get("raw_bytes")
            if not isinstance(raw, (bytes, bytearray)):
                raw = b""

            mime_type = _guess_mime_type(None, name=name)
            if str(Path(name).suffix or "").lower() in (".csv", ".tsv", ".txt", ".json"):
                raw = _mask_text_file_bytes(bytes(raw))
            if max_file_bytes > 0:
                raw = _truncate_bytes(bytes(raw), max_bytes=int(max_file_bytes))
            raw_len = 0
            try:
                raw_len = int(len(raw or b""))
            except Exception:
                raw_len = 0
            if max_total_bytes > 0 and (int(uploaded_bytes_total) + int(raw_len)) > int(max_total_bytes):
                break
            uploaded_bytes_total = int(uploaded_bytes_total) + int(raw_len)
            file_parts.append(_make_inline_part_from_bytes(data=bytes(raw), mime_type=mime_type))

    gen_n = max(0, int(args.gen_csv_count))
    if int(args.gen_csv_bytes) > 0 and gen_n > 0:
        for _ in range(gen_n):
            raw = _make_synthetic_csv_bytes(target_bytes=int(args.gen_csv_bytes))
            try:
                uploaded_bytes_total += int(len(raw or b""))
            except Exception:
                pass
            file_parts.append(_make_inline_part_from_bytes(data=raw, mime_type="text/csv"))

    thinking = str(args.thinking or "").strip() or None
    if thinking is None:
        if str(args.model or "").strip().startswith("gemini-3-"):
            thinking = "high"

    payload = _build_payload(
        instructions=instructions_eff,
        file_parts=file_parts,
        max_output_tokens=int(args.max_output_tokens),
        thinking_level=thinking,
    )

    url = str(args.base_url).rstrip("/") + "/models/" + str(args.model) + ":generateContent"
    headers = {
        "x-goog-api-key": str(api_key),
        "content-type": "application/json",
    }

    st = _payload_stats(payload)
    print("URL:", url)
    print("MODEL:", str(args.model))
    print("PAYLOAD_STATS:", json.dumps(st, ensure_ascii=False))
    print("UPLOADED_BYTES_TOTAL:", int(uploaded_bytes_total))
    if int(uploaded_bytes_total) > 18 * 1024 * 1024:
        raise SystemExit("files too large for gemini inlineData (server would reject this request)")

    if bool(args.count_tokens):
        if not api_key:
            raise SystemExit("Missing API key for --count-tokens")
        st0, txt0, obj0 = _count_tokens(
            base_url=str(args.base_url),
            model=str(args.model),
            api_key=str(api_key),
            payload=payload,
            timeout_s=float(args.timeout),
        )
        print("COUNT_TOKENS_STATUS:", int(st0))
        if int(st0) != 200:
            print("COUNT_TOKENS_ERROR_BODY (first 4000 chars):")
            print(str(txt0 or "")[:4000])
        else:
            print("COUNT_TOKENS_RESPONSE:", json.dumps(obj0, ensure_ascii=False))

    if args.dry_run:
        return 0

    repeat_n = max(1, int(args.repeat))

    for i in range(repeat_n):
        t0 = time.time()
        status, txt, obj = _post_json(url=url, headers=headers, payload=payload, timeout_s=float(args.timeout))
        dt = time.time() - t0

        print("-")
        print("REQ", i + 1, "/", repeat_n, "status=", status, "seconds=", round(dt, 3))

        if status != 200:
            print("ERROR_BODY (first 4000 chars):")
            print(str(txt or "")[:4000])
            if int(status) == 429 and bool(args.auto_retry_429):
                wait_s = _retry_delay_seconds_from_error_text(str(txt or ""))
                if wait_s is not None and wait_s > 0.0:
                    try:
                        time.sleep(float(wait_s))
                    except Exception:
                        pass
                    t1 = time.time()
                    status2, txt2, obj2 = _post_json(url=url, headers=headers, payload=payload, timeout_s=float(args.timeout))
                    dt2 = time.time() - t1
                    print("-")
                    print("RETRY status=", status2, "seconds=", round(dt2, 3))
                    if status2 != 200:
                        print("ERROR_BODY (first 4000 chars):")
                        print(str(txt2 or "")[:4000])
                    else:
                        usage2 = obj2.get("usageMetadata") if isinstance(obj2, dict) else None
                        if isinstance(usage2, dict):
                            print("USAGE:", json.dumps(usage2, ensure_ascii=False))
        else:
            usage = obj.get("usageMetadata") if isinstance(obj, dict) else None
            if isinstance(usage, dict):
                print("USAGE:", json.dumps(usage, ensure_ascii=False))
            candidates = obj.get("candidates") if isinstance(obj, dict) else None
            if isinstance(candidates, list) and candidates:
                c0 = candidates[0] if isinstance(candidates[0], dict) else {}
                finish = c0.get("finishReason")
                if finish is not None:
                    print("FINISH_REASON:", str(finish))

        if i < repeat_n - 1 and float(args.delay) > 0:
            time.sleep(float(args.delay))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
