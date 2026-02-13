import concurrent.futures
import base64
import contextlib
import copy
import csv
import faulthandler
import gzip
import hashlib
import io
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from multiprocessing import shared_memory
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse
import urllib.error
import urllib.request

import numpy as np

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"


class RateLimitError(Exception):
    def __init__(self, message: str, *, retry_after_s: Optional[float] = None, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.provider = provider
        self.model = model


class TemporaryUnavailableError(Exception):
    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


def _scheduler_enabled() -> bool:
    v = str(os.environ.get("DT_RESOURCE_SCHEDULER") or "").strip().lower()
    if not v:
        return False
    return v in ("1", "true", "yes", "on")


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = [p for p in line.split() if p]
                if len(parts) >= 2:
                    kb = float(parts[1])
                    return float(kb) / (1024.0 * 1024.0)
    except Exception:
        return float("inf")
    return float("inf")


class _ResourceScheduler:
    def __init__(self) -> None:
        cpu0 = os.cpu_count() or 1
        try:
            cpu0 = int(os.environ.get("DT_RESOURCE_SCHEDULER_CPU"))
        except Exception:
            cpu0 = int(cpu0)
        self.cpu_total = max(1, int(cpu0))
        try:
            self.mem_reserve_gb = float(os.environ.get("DT_RESOURCE_SCHEDULER_MEM_RESERVE_GB"))
        except Exception:
            self.mem_reserve_gb = 8.0
        self._cv = threading.Condition()
        self._cpu_used = 0
        self._mem_used_gb = 0.0

    def _can_acquire(self, *, cpu: int, mem_gb: float) -> bool:
        if int(cpu) <= 0:
            cpu = 0
        if float(mem_gb) <= 0:
            mem_gb = 0.0
        if int(self._cpu_used) + int(cpu) > int(self.cpu_total):
            return False
        avail_gb = float(_mem_available_gb())
        if not (avail_gb == float("inf")):
            if float(avail_gb) - float(self.mem_reserve_gb) - float(self._mem_used_gb) < float(mem_gb):
                return False
        return True

    @contextlib.contextmanager
    def acquire(self, *, cpu: int, mem_gb: float) -> Any:
        if not _scheduler_enabled():
            yield
            return
        cpu_req = max(0, int(cpu))
        mem_req = max(0.0, float(mem_gb))
        if cpu_req > int(self.cpu_total):
            cpu_req = int(self.cpu_total)
        with self._cv:
            while not self._can_acquire(cpu=cpu_req, mem_gb=mem_req):
                self._cv.wait(timeout=0.2)
            self._cpu_used += int(cpu_req)
            self._mem_used_gb += float(mem_req)
        try:
            yield
        finally:
            with self._cv:
                self._cpu_used = max(0, int(self._cpu_used) - int(cpu_req))
                self._mem_used_gb = max(0.0, float(self._mem_used_gb) - float(mem_req))
                self._cv.notify_all()


_RESOURCE_SCHED = _ResourceScheduler()


_GEMINI_DEBUG_LOCK = threading.Lock()


def _gemini_debug_dir() -> Path:
    try:
        _ensure_dirs()
    except Exception:
        pass

    base = _WORKSPACE_DIR / "llm_debug" / "gemini"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base


def _gemini_debug_include_inline_data() -> bool:
    return str(os.environ.get("DT_GEMINI_DEBUG_INCLUDE_INLINE_DATA") or "").strip().lower() in ("1", "true", "yes")


def _gemini_debug_hash_inline_data() -> bool:
    return str(os.environ.get("DT_GEMINI_DEBUG_HASH_INLINE_DATA") or "").strip().lower() in ("1", "true", "yes")


def _gemini_debug_redact_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (headers or {}).items():
        kk = str(k or "")
        vv = v
        if kk.strip().lower() in ("x-goog-api-key", "authorization"):
            vv = "***"
        out[kk] = vv
    return out


def _gemini_debug_inline_summary(inline: Dict[str, Any]) -> Dict[str, Any]:
    mime = str(inline.get("mimeType") or "") if isinstance(inline, dict) else ""
    data = inline.get("data") if isinstance(inline, dict) else None
    b64_len = int(len(str(data))) if isinstance(data, str) else 0
    approx_bytes = int((3 * b64_len) // 4) if b64_len > 0 else 0
    out: Dict[str, Any] = {
        "mimeType": mime,
        "data_len_chars": int(b64_len),
        "approx_decoded_bytes": int(approx_bytes),
    }
    if _gemini_debug_hash_inline_data() and isinstance(data, str) and data:
        try:
            out["data_sha256"] = str(_sha256_text(str(data)))
        except Exception:
            pass
    return out


def _gemini_debug_redact_payload(obj: Any) -> Any:
    include_inline = _gemini_debug_include_inline_data()
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            kk = str(k or "")
            if kk in ("api_key", "key", "x-goog-api-key", "authorization"):
                out[kk] = "***"
                continue
            if kk == "inlineData" and isinstance(v, dict) and (not include_inline):
                out[kk] = _gemini_debug_inline_summary(v)
                continue
            out[kk] = _gemini_debug_redact_payload(v)
        return out
    if isinstance(obj, list):
        return [_gemini_debug_redact_payload(x) for x in obj]
    return obj


def _gemini_debug_payload_stats(payload: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "inline_parts": 0,
        "inline_data_total_len_chars": 0,
        "inline_data_total_approx_bytes": 0,
    }

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "inlineData" in x and isinstance(x.get("inlineData"), dict):
                inline = x.get("inlineData")
                data = inline.get("data") if isinstance(inline, dict) else None
                if isinstance(data, str):
                    b64_len = int(len(data))
                    stats["inline_parts"] = int(stats.get("inline_parts") or 0) + 1
                    stats["inline_data_total_len_chars"] = int(stats.get("inline_data_total_len_chars") or 0) + b64_len
                    stats["inline_data_total_approx_bytes"] = int(stats.get("inline_data_total_approx_bytes") or 0) + int((3 * b64_len) // 4)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    try:
        walk(payload)
    except Exception:
        pass
    return stats


def _gemini_debug_write_snapshot(
    *,
    event: str,
    provider: str,
    model: str,
    base_url: str,
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
    error: Optional[str],
    attempt: Optional[int],
    attempts: Optional[int],
    sleep_budget_s: Optional[float],
    cooldown_remaining_s: Optional[float],
    path: str,
    run_id: Optional[str] = None,
    player_id: Optional[str] = None,
) -> None:
    try:
        d = _gemini_debug_dir()
        ts = int(time.time())
        uid = uuid.uuid4().hex[:10]
        p = d / f"gemini_{str(event or 'event')}_{ts}_{uid}.json"

        payload_stats = _gemini_debug_payload_stats(payload)
        payload_red = _gemini_debug_redact_payload(payload)

        info: Dict[str, Any] = {
            "ts": int(ts),
            "event": str(event or ""),
            "provider": str(provider or ""),
            "model": str(model or ""),
            "base_url": str(base_url or ""),
            "url": str(url or ""),
            "path": str(path or ""),
            "run_id": str(run_id or "") if run_id is not None else None,
            "player_id": str(player_id or "") if player_id is not None else None,
            "attempt": int(attempt) if attempt is not None else None,
            "attempts": int(attempts) if attempts is not None else None,
            "sleep_budget_s": float(sleep_budget_s) if sleep_budget_s is not None else None,
            "cooldown_remaining_s": float(cooldown_remaining_s) if cooldown_remaining_s is not None else None,
            "error": str(error or "") if error else None,
            "headers": _gemini_debug_redact_headers(headers),
            "payload_stats": payload_stats,
            "payload": payload_red,
        }

        raw = json.dumps(info, ensure_ascii=False).encode("utf-8")
        with _GEMINI_DEBUG_LOCK:
            _atomic_write_bytes(p, raw)
    except Exception:
        return


_ANTHROPIC_INPUT_TPM_LIMIT = 450_000
_ANTHROPIC_TPM_WINDOW_S = 60.0
_ANTHROPIC_TPM_SAFETY_FRAC = 0.60
_ANTHROPIC_TPM_USAGE: deque[tuple[float, int]] = deque()
_ANTHROPIC_TPM_LOCK = threading.Lock()


_GEMINI_COOLDOWN_UNTIL: Dict[str, float] = {}
_GEMINI_COOLDOWN_LOCK = threading.Lock()


def _http_status_from_error(msg: str) -> Optional[int]:
    s = str(msg or "")
    if not s.strip():
        return None
    try:
        m = re.search(r"\bHTTP\s+([0-9]{3})\b", s, flags=re.IGNORECASE)
        if not m:
            return None
        v = int(m.group(1))
        if v < 100 or v > 999:
            return None
        return v
    except Exception:
        return None


def _is_rate_limited_error(msg: str) -> bool:
    st = _http_status_from_error(msg)
    if st == 429:
        return True
    s = str(msg or "").upper()
    return bool(s and ("RESOURCE_EXHAUSTED" in s or "RATE LIMIT" in s or "QUOTA" in s))


def _gemini_cooldown_key(*, base_url: str, model: str) -> str:
    return str(base_url or "").strip() + "|" + str(model or "").strip()


def _gemini_cooldown_remaining_s(*, base_url: str, model: str) -> float:
    key = _gemini_cooldown_key(base_url=str(base_url), model=str(model))
    now = float(time.time())
    with _GEMINI_COOLDOWN_LOCK:
        until = float(_GEMINI_COOLDOWN_UNTIL.get(key, 0.0) or 0.0)
    rem = float(until - now)
    return float(rem) if rem > 0.0 else 0.0


def _gemini_set_cooldown(*, base_url: str, model: str, retry_after_s: Optional[float]) -> float:
    wait_s = 60.0
    try:
        if retry_after_s is not None:
            wait_s = float(retry_after_s)
    except Exception:
        wait_s = 60.0
    wait_s = float(min(300.0, max(1.0, wait_s)))

    key = _gemini_cooldown_key(base_url=str(base_url), model=str(model))
    now = float(time.time())
    until = float(now + wait_s)
    with _GEMINI_COOLDOWN_LOCK:
        prev = float(_GEMINI_COOLDOWN_UNTIL.get(key, 0.0) or 0.0)
        _GEMINI_COOLDOWN_UNTIL[key] = float(max(prev, until))
    return float(wait_s)


def _gemini_retry_attempts() -> int:
    attempts = 6
    try:
        attempts = int(os.environ.get("DT_GEMINI_RETRY_ATTEMPTS", str(attempts)) or attempts)
    except Exception:
        attempts = 6
    return max(1, min(20, int(attempts)))


def _approx_token_count_text(text: str) -> int:
    s = str(text or "")
    if not s:
        return 0
    return max(1, int((len(s) + 3) // 4))


def _anthropic_tpm_used(now: float) -> int:
    with _ANTHROPIC_TPM_LOCK:
        while _ANTHROPIC_TPM_USAGE and (now - float(_ANTHROPIC_TPM_USAGE[0][0])) >= float(_ANTHROPIC_TPM_WINDOW_S):
            _ANTHROPIC_TPM_USAGE.popleft()
        return int(sum(int(n) for (_, n) in _ANTHROPIC_TPM_USAGE))


def _anthropic_tpm_throttle(*, need_tokens: int) -> None:
    need = int(need_tokens or 0)
    if need <= 0:
        return

    limit = int(int(_ANTHROPIC_INPUT_TPM_LIMIT) * float(_ANTHROPIC_TPM_SAFETY_FRAC))
    if limit <= 0:
        return

    while True:
        now = float(time.time())
        used = _anthropic_tpm_used(now)
        if used + need <= limit:
            with _ANTHROPIC_TPM_LOCK:
                _ANTHROPIC_TPM_USAGE.append((now, need))
            return
        with _ANTHROPIC_TPM_LOCK:
            oldest_t = float(_ANTHROPIC_TPM_USAGE[0][0]) if _ANTHROPIC_TPM_USAGE else now
        sleep_s = max(0.0, float(_ANTHROPIC_TPM_WINDOW_S) - (now - oldest_t) + 1.0)
        sleep_s = min(75.0, sleep_s)
        try:
            logging.getLogger(__name__).warning(
                "Anthropic TPM throttle: used=%s need~%s limit=%s. Sleeping %.1fs...",
                used,
                need,
                limit,
                float(sleep_s),
            )
        except Exception:
            pass
        time.sleep(float(sleep_s))


def _anthropic_headers(*, api_key: str, betas: Optional[list[str]] = None, content_type: Optional[str] = "application/json") -> Dict[str, str]:
    h = {
        "x-api-key": str(api_key),
        "anthropic-version": "2023-06-01",
    }
    if content_type:
        h["content-type"] = str(content_type)
    if betas:
        h["anthropic-beta"] = ",".join([str(x) for x in betas if str(x).strip()])
    return h


def _http_post_json(*, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(str(url), data=raw, method="POST", headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            b = resp.read()
            txt = b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else str(b)
    except urllib.error.HTTPError as e:
        try:
            b = e.read()
            txt = b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else str(b)
        except Exception:
            txt = str(e)
        raise ValueError(f"HTTP {int(getattr(e, 'code', 0) or 0)}: {txt[:2000]}") from e
    except Exception as e:
        raise ValueError(str(e)) from e
    obj = {}
    try:
        parsed = json.loads(txt) if isinstance(txt, str) and txt.strip() else None
        obj = parsed if isinstance(parsed, dict) else {}
    except Exception:
        obj = {}
    return obj


def _should_retry_remote_http_error(msg: str) -> bool:
    s = str(msg or "").upper()
    if not s:
        return False
    if "HTTP 503" in s or "\"CODE\"" in s and "503" in s:
        return True
    if "HTTP 529" in s or "OVERLOADED" in s or "OVERLOADED_ERROR" in s:
        return True
    if "UNAVAILABLE" in s or "SERVICE UNAVAILABLE" in s:
        return True
    if "REQUEST TIMED OUT" in s or "TIMED OUT" in s or "TIMEOUT" in s:
        return True
    if "HTTP 429" in s or "RATE LIMIT" in s or "RESOURCE_EXHAUSTED" in s:
        return True
    return False


def _retry_delay_seconds_from_error(msg: str) -> Optional[float]:
    s = str(msg or "")
    if not s.strip():
        return None
    try:
        m = re.search(r"retryDelay\"\s*:\s*\"\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*\"", s, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"please\s+retry\s+in\s*([0-9]+(?:\.[0-9]+)?)\s*s", s, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"retry[-\s]?after\s*([0-9]+(?:\.[0-9]+)?)\s*s", s, flags=re.IGNORECASE)
        if not m:
            return None
        v = float(m.group(1))
        if not (v >= 0.0):
            return None
        return min(300.0, v)
    except Exception:
        return None


def _openai_base_model_and_effort(model: str) -> tuple[str, Optional[str]]:
    m = str(model or ":").strip()
    if not m:
        return "", None
    if m == "gpt-5.2":
        return "gpt-5.2", "medium"
    if m == "gpt-5.2-none":
        return "gpt-5.2", "none"
    if m == "gpt-5.2-low":
        return "gpt-5.2", "low"
    if m == "gpt-5.2-medium":
        return "gpt-5.2", "medium"
    if m == "gpt-5.2-high":
        return "gpt-5.2", "high"
    if m in ("gpt-5.2-extra-high", "gpt-5.2-xhigh"):
        return "gpt-5.2", "xhigh"
    return m, None


def _xai_canonical_model(model: str) -> str:
    m = str(model or "").strip()
    if not m:
        return ""
    if m == "grok-4-1-fast":
        return "grok-4-1-fast-reasoning"
    return str(m)


_DISCUSS_ADVISOR_SYSTEM_PROMPT = (
    "You are a deeply experienced scientific advisor helping a trainee solve problems with in their research. "
    "Your job is to give short, concrete, actionable advice.\n\n"
    "These are the only experiments the trainee can run:\n"
    " - In vivo Characterization studies (biomarkers, survival curves, timecourses; in vivo or cell culture)\n"
    " - Bulk omics snapshots (transcript measurements, protein measurements, metabolite measurements)\n"
    " - Spatial omics (spatial transcript or spatial protein measurements)\n"
    " - Cell-culture perturbation screens (systematic protein up or down perturbations)\n"
    " - Cell-culture perturbation experiments where one can increase or decrease a protein and see how it affects cells\n"
    " - In vivo experiments can be done with perturbations and also do omics after a perturbation\n"
    " - Any type of analysis on python"
    " - For any experiment, the trainee can alter the age of the in vivo model or days in culture or number of replicates or add a perturbation"     
    "Guidelines:\n"
    "Reply with high level advice on how to solve the current problem the trainee is facing\n"
    "Output requirements:\n"
    "- Return at most 3 bullet points.\n"
    "- Each bullet must be an imperative action (e.g., 'Run ...', 'Compare ...', 'Increase replicates to ...').\n"
    "- No long explanations, no markdown fences, no code."
)


def _discuss_postprocess_advice(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    lines = []
    for ln in t.splitlines():
        s = str(ln or "")
        if s.strip().startswith("```"):
            continue
        lines.append(s)
    t2 = "\n".join(lines).strip()
    if len(t2) > 2200:
        t2 = t2[:2200].rstrip()
    return t2


def _discuss_llm_generate(*, provider: str, model: str, system_prompt: str, user_prompt: str, timeout_s: float, max_tokens: int) -> str:
    prov = str(provider or "").strip().lower() or "openai"
    if prov == "claude":
        prov = "anthropic"
    if prov == "grok":
        prov = "xai"

    if prov in ("openai", "openai_compat"):
        if OpenAI is None:
            raise ValueError("openai sdk not installed")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("missing OPENAI_API_KEY")

        client = OpenAI(api_key=str(api_key))
        base_model, effort = _openai_base_model_and_effort(str(model))
        req: Dict[str, Any] = {
            "model": str(base_model or model),
            "input": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "text": {"format": {"type": "text"}, "verbosity": "low"},
            "timeout": float(timeout_s),
        }
        if effort is not None:
            req["reasoning"] = {"effort": str(effort)}

        try:
            resp = client.responses.create(**req, max_output_tokens=int(max_tokens))
        except Exception:
            resp = client.responses.create(**req)

        try:
            return str(getattr(resp, "output_text", "") or "")
        except Exception:
            return ""

    if prov in ("xai",):
        if OpenAI is None:
            raise ValueError("openai sdk not installed")
        api_key = os.environ.get("XAI_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("missing XAI_API_KEY")
        base_url = str(os.environ.get("XAI_BASE_URL") or "").strip() or "https://api.x.ai/v1"

        client = OpenAI(api_key=str(api_key), base_url=str(base_url))
        model_call = _xai_canonical_model(str(model))
        req: Dict[str, Any] = {
            "model": str(model_call),
            "input": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "text": {"format": {"type": "text"}, "verbosity": "low"},
        }
        try:
            resp = client.responses.create(**req, max_output_tokens=int(max_tokens))
        except Exception:
            resp = client.responses.create(**req)
        try:
            return str(getattr(resp, "output_text", "") or "")
        except Exception:
            return ""

    if prov in ("gemini",):
        api_key_g = os.environ.get("GEMINI_API_KEY")
        if not isinstance(api_key_g, str) or not api_key_g.strip():
            raise ValueError("missing GEMINI_API_KEY")
        base_url_g = str(os.environ.get("GEMINI_BASE_URL") or "").strip() or "https://generativelanguage.googleapis.com/v1beta"
        url = str(base_url_g).rstrip("/") + "/models/" + str(model) + ":generateContent"
        headers = {
            "x-goog-api-key": str(api_key_g),
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": str(system_prompt)}]},
            "contents": [{"role": "user", "parts": [{"text": str(user_prompt)}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": int(max_tokens)},
        }

        resp_json: Dict[str, Any] = {}
        last_err: Optional[str] = None
        rate_limited = False
        sleep_budget_s = 300.0
        try:
            sleep_budget_s = float(os.environ.get("DT_GEMINI_RATE_LIMIT_SLEEP_BUDGET_S", str(sleep_budget_s)) or sleep_budget_s)
        except Exception:
            sleep_budget_s = 300.0
        sleep_budget_s = max(0.0, min(1800.0, float(sleep_budget_s)))

        attempts = int(max(3, _gemini_retry_attempts()))
        for attempt in range(int(attempts)):
            rem_s = _gemini_cooldown_remaining_s(base_url=str(base_url_g), model=str(model))
            if rem_s > 0.0:
                if sleep_budget_s <= 0.0:
                    rate_limited = True
                    try:
                        _gemini_debug_write_snapshot(
                            event="cooldown_budget_exhausted",
                            provider="gemini",
                            model=str(model),
                            base_url=str(base_url_g),
                            url=str(url),
                            headers=dict(headers),
                            payload=dict(payload),
                            error=None,
                            attempt=int(attempt),
                            attempts=int(attempts),
                            sleep_budget_s=float(sleep_budget_s),
                            cooldown_remaining_s=float(rem_s),
                            path="/api/discuss",
                        )
                    except Exception:
                        pass
                    break
                try:
                    sleep_s = min(float(rem_s), float(sleep_budget_s))
                    time.sleep(float(max(0.0, sleep_s)))
                    sleep_budget_s -= float(sleep_s)
                except Exception:
                    pass
                continue

            try:
                t_s = float(timeout_s)
                if attempt >= 1:
                    t_s = max(t_s, float(timeout_s) * (1.5 + 0.25 * float(attempt)))
                resp_json = _http_post_json(url=str(url), headers=headers, payload=payload, timeout_s=float(t_s))
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                if _is_rate_limited_error(last_err):
                    rate_limited = True
                    hint_s = _retry_delay_seconds_from_error(last_err)
                    wait_s = _gemini_set_cooldown(base_url=str(base_url_g), model=str(model), retry_after_s=hint_s)
                    try:
                        _gemini_debug_write_snapshot(
                            event="rate_limited",
                            provider="gemini",
                            model=str(model),
                            base_url=str(base_url_g),
                            url=str(url),
                            headers=dict(headers),
                            payload=dict(payload),
                            error=str(last_err),
                            attempt=int(attempt),
                            attempts=int(attempts),
                            sleep_budget_s=float(sleep_budget_s),
                            cooldown_remaining_s=float(wait_s),
                            path="/api/discuss",
                        )
                    except Exception:
                        pass
                    if attempt >= (int(attempts) - 1) or sleep_budget_s <= 0.0:
                        break
                    try:
                        sleep_s = min(float(wait_s), float(sleep_budget_s))
                        time.sleep(float(max(0.0, sleep_s)))
                        sleep_budget_s -= float(sleep_s)
                    except Exception:
                        pass
                    continue
                if _should_retry_remote_http_error(last_err) and attempt < (int(attempts) - 1):
                    try:
                        sleep_s = float(1.0 + (2.0 * float(attempt)))
                        hint_s = _retry_delay_seconds_from_error(last_err)
                        if hint_s is not None:
                            sleep_s = max(float(sleep_s), float(hint_s))
                        sleep_s = min(60.0, max(0.0, float(sleep_s)))
                        time.sleep(float(sleep_s))
                    except Exception:
                        pass
                    continue
                raise

        if rate_limited and last_err is None:
            try:
                _gemini_debug_write_snapshot(
                    event="temporary_unavailable",
                    provider="gemini",
                    model=str(model),
                    base_url=str(base_url_g),
                    url=str(url),
                    headers=dict(headers),
                    payload=dict(payload),
                    error=None,
                    attempt=None,
                    attempts=int(attempts),
                    sleep_budget_s=float(sleep_budget_s),
                    cooldown_remaining_s=_gemini_cooldown_remaining_s(base_url=str(base_url_g), model=str(model)),
                    path="/api/discuss",
                )
            except Exception:
                pass
            raise TemporaryUnavailableError("Gemini temporarily unavailable", provider="gemini", model=str(model))
        if last_err:
            if rate_limited:
                try:
                    _gemini_debug_write_snapshot(
                        event="temporary_unavailable",
                        provider="gemini",
                        model=str(model),
                        base_url=str(base_url_g),
                        url=str(url),
                        headers=dict(headers),
                        payload=dict(payload),
                        error=str(last_err),
                        attempt=None,
                        attempts=int(attempts),
                        sleep_budget_s=float(sleep_budget_s),
                        cooldown_remaining_s=_gemini_cooldown_remaining_s(base_url=str(base_url_g), model=str(model)),
                        path="/api/discuss",
                    )
                except Exception:
                    pass
                raise TemporaryUnavailableError("Gemini temporarily unavailable", provider="gemini", model=str(model))
            raise ValueError(str(last_err))

        try:
            candidates = resp_json.get("candidates") if isinstance(resp_json, dict) else None
            if isinstance(candidates, list) and candidates:
                c0 = candidates[0] if isinstance(candidates[0], dict) else {}
                content0 = c0.get("content") if isinstance(c0, dict) else None
                parts0 = content0.get("parts") if isinstance(content0, dict) else None
                if isinstance(parts0, list) and parts0:
                    out_chunks: list[str] = []
                    for p0 in parts0:
                        if isinstance(p0, dict) and isinstance(p0.get("text"), str):
                            out_chunks.append(str(p0.get("text") or ""))
                    return "".join(out_chunks).strip()
        except Exception:
            return ""
        return ""

    if prov in ("anthropic",):
        api_key_a = os.environ.get("ANTHROPIC_API_KEY")
        if not isinstance(api_key_a, str) or not api_key_a.strip():
            raise ValueError("missing ANTHROPIC_API_KEY")

        try:
            need_tokens = int(_approx_token_count_text(str(system_prompt)))
            need_tokens += int(_approx_token_count_text(str(user_prompt)))
            need_tokens += 500
            _anthropic_tpm_throttle(need_tokens=int(need_tokens))
        except Exception:
            pass

        payload = {
            "model": str(model),
            "max_tokens": int(max(256, min(8192, int(max_tokens)))),
            "temperature": 0.0,
            "system": str(system_prompt),
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": str(user_prompt)}]},
            ],
        }

        last_err: Optional[str] = None
        resp_json: Dict[str, Any] = {}
        for attempt in range(3):
            try:
                t_s = float(timeout_s)
                if attempt == 1:
                    t_s = max(t_s, float(timeout_s) * 2.0)
                if attempt == 2:
                    t_s = max(t_s, float(timeout_s) * 3.0)
                resp_json = _http_post_json(
                    url=f"{_ANTHROPIC_BASE_URL}/messages",
                    headers=_anthropic_headers(api_key=str(api_key_a), betas=[], content_type="application/json"),
                    payload=payload,
                    timeout_s=float(t_s),
                )
                last_err = None
                break
            except Exception as e:
                last_err = str(e)
                if attempt < 2 and ("HTTP 429" in last_err or "rate_limit_error" in last_err or "input tokens per minute" in last_err or "would exceed the rate limit" in last_err):
                    time.sleep(65.0)
                    continue
                if attempt < 2 and ("timed out" in last_err.lower() or "timeout" in last_err.lower()):
                    time.sleep(2.0)
                    continue
                if attempt < 2:
                    time.sleep(float(0.6 * (2**attempt)))
                    continue
                raise
        if last_err:
            raise ValueError(str(last_err))

        return _anthropic_message_text(resp_json if isinstance(resp_json, dict) else {})

    raise ValueError(f"unsupported provider: {prov}")


def _anthropic_upload_file(*, api_key: str, filename: str, file_bytes: bytes, timeout_s: float) -> Dict[str, Any]:
    betas = ["files-api-2025-04-14"]
    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            return _http_post_multipart_file(
                url=f"{_ANTHROPIC_BASE_URL}/files",
                headers=_anthropic_headers(api_key=api_key, betas=betas, content_type=None),
                field_name="file",
                filename=str(filename),
                file_bytes=(file_bytes or b""),
                mime_type="application/octet-stream",
                timeout_s=float(timeout_s),
            )
        except Exception as e:
            last_err = str(e)
            if attempt < 2 and ("HTTP 429" in last_err or "rate_limit_error" in last_err):
                time.sleep(65.0)
                continue
            if attempt < 2 and ("timed out" in last_err.lower() or "timeout" in last_err.lower()):
                time.sleep(2.0)
                continue
            if attempt < 2:
                time.sleep(float(0.6 * (2**attempt)))
                continue
            raise
    raise ValueError(last_err or "Anthropic upload failed")


def _anthropic_messages_code_execution(
    *,
    api_key: str,
    model: str,
    instructions: str,
    file_ids: list[str],
    timeout_s: float,
    max_tokens: int,
    messages: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    betas = ["code-execution-2025-08-25", "files-api-2025-04-14"]

    content_blocks: list[Dict[str, Any]] = [{"type": "text", "text": str(instructions)}]
    for fid in file_ids:
        if str(fid or "").strip():
            content_blocks.append({"type": "container_upload", "file_id": str(fid)})

    msgs: list[Dict[str, Any]] = []
    if isinstance(messages, list) and messages:
        msgs = [m for m in messages if isinstance(m, dict)]
    if not msgs:
        msgs = [
            {
                "role": "user",
                "content": content_blocks,
            }
        ]

    max_tokens_i = 4096
    try:
        max_tokens_i = int(max_tokens)
    except Exception:
        max_tokens_i = 4096
    max_tokens_i = max(256, min(16384, int(max_tokens_i)))

    payload: Dict[str, Any] = {
        "model": str(model),
        "max_tokens": int(max_tokens_i),
        "messages": msgs,
        "tools": [
            {
                "type": "code_execution_20250825",
                "name": "code_execution",
            }
        ],
    }
    if str(model or "").strip() == "claude-opus-4-5-20251101":
        budget = 63_999
        try:
            budget = int(os.environ.get("DT_ANTHROPIC_THINKING_BUDGET", str(budget)) or budget)
        except Exception:
            budget = 63_999
        budget = max(1, min(200_000, int(budget)))
        max_out = max(1, int(payload.get("max_tokens") or max_tokens_i))
        if max_out <= 1024:
            max_out = 1025
            payload["max_tokens"] = int(max_out)
        budget = max(1024, min(int(budget), int(max_out) - 1))
        payload["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
        payload["temperature"] = 1.0

    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            t_s = float(timeout_s)
            if attempt == 1:
                t_s = max(t_s, float(timeout_s) * 2.0)
            if attempt == 2:
                t_s = max(t_s, float(timeout_s) * 3.0)
            return _http_post_json(
                url=f"{_ANTHROPIC_BASE_URL}/messages",
                headers=_anthropic_headers(api_key=api_key, betas=betas, content_type="application/json"),
                payload=payload,
                timeout_s=float(t_s),
            )
        except Exception as e:
            last_err = str(e)
            if "HTTP 529" in last_err or "overloaded_error" in last_err.lower() or "overloaded" in last_err.lower():
                if attempt < 2:
                    time.sleep(float(0.8 * (2**attempt)))
                    continue
                raise TemporaryUnavailableError("Anthropic temporarily unavailable", provider="anthropic", model=str(model))
            if attempt < 2 and ("HTTP 429" in last_err or "rate_limit_error" in last_err or "input tokens per minute" in last_err or "would exceed the rate limit" in last_err):
                time.sleep(65.0)
                continue
            if attempt < 2 and ("timed out" in last_err.lower() or "timeout" in last_err.lower()):
                time.sleep(2.0)
                continue
            if attempt < 2:
                time.sleep(float(0.6 * (2**attempt)))
                continue
            raise
    raise ValueError(last_err or "Anthropic request failed")


def _anthropic_message_text(resp_json: Dict[str, Any]) -> str:
    parts: list[str] = []
    blocks = resp_json.get("content")
    if isinstance(blocks, list):
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
    return "".join(parts)


def _http_post_multipart_file(
    *,
    url: str,
    headers: Dict[str, str],
    field_name: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
    timeout_s: float,
) -> Dict[str, Any]:
    boundary = "----digital_tissue_" + uuid.uuid4().hex
    pre = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8")
    post = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = pre + (file_bytes or b"") + post

    h2 = dict(headers)
    h2["content-type"] = f"multipart/form-data; boundary={boundary}"

    req = urllib.request.Request(str(url), data=body, method="POST", headers=h2)
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            b = resp.read()
            txt = b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else str(b)
    except urllib.error.HTTPError as e:
        try:
            b = e.read()
            txt = b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else str(b)
        except Exception:
            txt = str(e)
        raise ValueError(f"HTTP {int(getattr(e, 'code', 0) or 0)}: {txt[:2000]}") from e
    obj = {}
    try:
        parsed = json.loads(txt) if isinstance(txt, str) and txt.strip() else None
        obj = parsed if isinstance(parsed, dict) else {}
    except Exception:
        obj = {}
    return obj

from apply_layer_ops import _decode_float32_b64, _encode_float32_b64, apply_layer_ops_inplace
from output_calc import _ExprEval


def _pathway_compute_topology(step: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError("step must be a dict")

    pathway_name = str(step.get("pathway_name", step.get("name", "")) or "").strip()
    if not pathway_name:
        raise ValueError("missing pathway_name")

    inputs_raw = step.get("inputs", [])
    if isinstance(inputs_raw, str):
        inputs = [s.strip() for s in inputs_raw.split(",") if s.strip()]
    elif isinstance(inputs_raw, list):
        inputs = [str(x).strip() for x in inputs_raw if x]
    else:
        raise ValueError("inputs must be a list or comma-separated string")

    outputs_raw = step.get("outputs", [])
    if isinstance(outputs_raw, str):
        outputs = [s.strip() for s in outputs_raw.split(",") if s.strip()]
    elif isinstance(outputs_raw, list):
        outputs = [str(x).strip() for x in outputs_raw if x]
    else:
        raise ValueError("outputs must be a list or comma-separated string")

    num_enzymes = int(step.get("num_enzymes", 3))
    if num_enzymes < 1:
        num_enzymes = 1

    topo_seed = sum(ord(c) * (idx + 1) for idx, c in enumerate(pathway_name))
    topo_rng = np.random.default_rng(topo_seed)

    enzyme_connections: list[list[Dict[str, Any]]] = []
    enzyme_norm_weights: list[float] = []

    for e in range(num_enzymes):
        sources: list[tuple[str, int]] = []

        for inp_idx in range(len(inputs)):
            if float(topo_rng.random()) < (0.4 + 0.3 * (e == 0)):
                sources.append(("input", int(inp_idx)))

        for prev_e in range(e):
            prob = 0.6 if prev_e == e - 1 else 0.25
            if float(topo_rng.random()) < float(prob):
                sources.append(("enzyme", int(prev_e)))

        if not sources:
            if e == 0:
                sources.append(("input", 0))
            else:
                sources.append(("enzyme", int(e - 1)))

        n_conn = len(sources)
        norm_w = (1.0 / float(np.sqrt(float(n_conn)))) if n_conn > 1 else 1.0
        enzyme_norm_weights.append(float(norm_w))

        enzyme_connections.append(
            [
                {
                    "source_type": st,
                    "source_idx": int(si),
                }
                for (st, si) in sources
            ]
        )

    output_connections: list[int] = []
    for e in range(num_enzymes):
        prob = 0.3 + 0.5 * (float(e) / float(max(1, num_enzymes - 1)))
        if float(topo_rng.random()) < float(prob) or e == num_enzymes - 1:
            output_connections.append(int(e))

    if not output_connections:
        output_connections.append(int(num_enzymes - 1))

    out_n = len(output_connections)
    output_norm_weight = (1.0 / float(np.sqrt(float(out_n)))) if out_n > 1 else 1.0

    return {
        "ok": True,
        "pathway_name": pathway_name,
        "inputs": inputs,
        "outputs": outputs,
        "num_enzymes": int(num_enzymes),
        "enzyme_connections": enzyme_connections,
        "output_connections": output_connections,
        "enzyme_norm_weights": enzyme_norm_weights,
        "output_norm_weight": float(output_norm_weight),
    }


_LOG = logging.getLogger("digital_tissue.runtime")
_FAULT_FH: Optional[Any] = None
_EVO_DEBUG = bool(int(os.environ.get("DT_EVO_DEBUG", "0") or "0"))


def _setup_logging() -> None:
    if _LOG.handlers:
        return

    _LOG.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(stream=sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    _LOG.addHandler(sh)

    try:
        log_path = (Path(__file__).resolve().parent / "runtime_server.log").resolve()
        fh = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        _LOG.addHandler(fh)
    except Exception:
        pass

    try:
        err_path = (Path(__file__).resolve().parent / "stderr.log").resolve()
        eh = logging.handlers.RotatingFileHandler(
            str(err_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        eh.setLevel(logging.WARNING)
        eh.setFormatter(fmt)
        _LOG.addHandler(eh)
    except Exception:
        pass

    global _FAULT_FH
    try:
        fault_path = (Path(__file__).resolve().parent / "runtime_server_faulthandler.log").resolve()
        _FAULT_FH = open(fault_path, "a", encoding="utf-8")
    except Exception:
        _FAULT_FH = None


def _install_exception_hooks() -> None:
    def _sys_hook(exc_type, exc, tb):
        try:
            _LOG.error("Uncaught exception", exc_info=(exc_type, exc, tb))
        except Exception:
            pass

    sys.excepthook = _sys_hook

    def _thread_hook(args):
        try:
            _LOG.error(
                "Uncaught thread exception in %s", str(getattr(args, "thread", None)),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        except Exception:
            pass

    try:
        threading.excepthook = _thread_hook  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        if _FAULT_FH is not None:
            faulthandler.enable(file=_FAULT_FH, all_threads=True)
        else:
            faulthandler.enable(all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass


def _maybe_log_truncation_alarm(
    text: str,
    *,
    provider: str,
    model: str,
    path: str,
    run_id: Optional[str] = None,
    player_id: Optional[str] = None,
) -> None:
    t = str(text or "")
    if not t:
        return
    tl = t.lower()
    markers = [
        "truncation",
        "truncated",
        "truncate",
        "cut off",
        "cut-off",
        "token limit",
        "max tokens",
        "context length",
    ]
    hit = None
    hit_idx = -1
    for m in markers:
        idx = tl.find(str(m))
        if idx >= 0:
            hit = str(m)
            hit_idx = int(idx)
            break
    if hit is None:
        return

    snippet = ""
    try:
        lo = max(0, int(hit_idx) - 120)
        hi = min(len(t), int(hit_idx) + 240)
        snippet = t[lo:hi]
    except Exception:
        snippet = ""

    try:
        _LOG.error(
            "TRUNCATION_ALARM path=%s provider=%s model=%s run_id=%s player_id=%s marker=%s snippet=%s",
            str(path),
            str(provider),
            str(model),
            str(run_id or ""),
            str(player_id or ""),
            str(hit),
            str(snippet)[:800],
        )
    except Exception:
        pass


def _omics_analyze_diagnostics(output_text: str, response_obj: Any) -> tuple[str, Dict[str, Any]]:
    snippets: list[Dict[str, Any]] = []

    def _add_snippet(kind: str, s: str) -> None:
        if len(snippets) >= 24:
            return
        t = str(s or "")
        if not t.strip():
            return
        if len(t) > 2000:
            t = t[:2000]
        snippets.append({"kind": str(kind), "text": str(t)})

    def _walk(x: Any, depth: int) -> None:
        if len(snippets) >= 24:
            return
        if depth > 7:
            return
        if x is None:
            return
        if isinstance(x, str):
            _add_snippet("text", x)
            return
        if isinstance(x, dict):
            for k, v in list(x.items()):
                ks = str(k or "")
                ksl = ks.lower()
                if any(w in ksl for w in ("b64", "inline", "data", "bytes", "arr", "tensor")):
                    if isinstance(v, str) and len(v) > 2000:
                        continue
                _walk(v, depth + 1)
                if len(snippets) >= 24:
                    break
            return
        if isinstance(x, list):
            for v in list(x)[:200]:
                _walk(v, depth + 1)
                if len(snippets) >= 24:
                    break
            return

    _walk(response_obj, 0)

    out_s = str(output_text or "")
    diagnostics = {
        "ok": True,
        "output_text_chars": int(len(out_s)),
        "response_text_snippets": list(snippets),
        "response_text_snippet_count": int(len(snippets)),
        "note": "heuristic_error_detection_disabled",
    }
    return out_s, diagnostics


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    t0 = str(text or "").strip()
    if not t0:
        return None
    lines: list[str] = []
    for ln in t0.splitlines():
        if str(ln or "").strip().startswith("```"):
            continue
        lines.append(str(ln or ""))
    t = "\n".join(lines).strip()
    if not t:
        return None
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(int(start), int(len(t))):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                cand = t[start : i + 1]
                try:
                    obj = json.loads(cand)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _omics_analyze_judge(
    *,
    player_instructions: str,
    analyze_manifest_entries: list[Dict[str, Any]],
    output_text: str,
    analysis_diagnostics: Dict[str, Any],
    provider: str,
    model: str,
    attempt: int,
    max_attempts: int,
) -> Dict[str, str]:
    enabled = str(os.environ.get("DT_OMICS_ANALYZE_JUDGE", "1") or "1").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        return {"decision": "done", "reason": "judge_disabled"}
    if not str(os.environ.get("XAI_API_KEY") or "").strip():
        return {"decision": "done", "reason": "judge_unavailable_missing_xai_api_key"}

    try:
        max_chars = int(os.environ.get("DT_OMICS_ANALYZE_JUDGE_MAX_OUTPUT_TEXT_CHARS", "14000") or 14000)
    except Exception:
        max_chars = 14000
    max_chars = max(2000, min(80000, int(max_chars)))
    out0 = str(output_text or "")
    if len(out0) > int(max_chars):
        tail = out0[-2000:] if len(out0) > 4000 else ""
        out0 = out0[: int(max_chars) - len(tail)].rstrip() + ("\n\n[...TRUNCATED...]\n\n" + tail if tail else "\n\n[...TRUNCATED...]\n")

    judge_system = (
        "You are a strict evaluator for an automated data-analysis tool output. "
        "Your job is to decide whether the analysis output satisfactorily answers the user's instructions. "
        "Return ONLY a JSON object with keys: decision, reason, retry_instructions. "
        "decision must be exactly either 'retry' or 'done'. "
        "Use 'retry' if there are missing required computed outputs, severe tool/runtime errors, or the response is mostly process narration instead of computed results. "
        "Use 'done' if the output contains the requested computed results in text form. "
        "If decision='retry', set retry_instructions to a SHORT string (<= 1200 chars) that can be appended to the original analysis prompt to help the next attempt succeed. "
        "retry_instructions MUST preserve the user's intent, reference attachments by FILE_XX labels, and should focus on concrete guidance (e.g., 'do not write files', 'compute in-memory', 'print a compact table'). "
        "If decision='done', set retry_instructions to an empty string."
    )

    user_payload = {
        "player_instructions": str(player_instructions or ""),
        "attachment_manifest_entries": list(analyze_manifest_entries or []),
        "analysis_output_text": str(out0),
        "analysis_diagnostics": dict(analysis_diagnostics or {}),
        "provider": str(provider or ""),
        "model": str(model or ""),
        "attempt": int(attempt),
        "max_attempts": int(max_attempts),
    }

    raw = ""
    try:
        raw = _discuss_llm_generate(
            provider="xai",
            model="grok-4-1-fast-reasoning",
            system_prompt=str(judge_system),
            user_prompt=json.dumps(user_payload, ensure_ascii=False),
            timeout_s=float(45.0),
            max_tokens=int(240),
        )
    except Exception as e:
        return {"decision": "done", "reason": "judge_error: " + str(e)[:200]}

    obj = _extract_first_json_object(raw)
    if not isinstance(obj, dict):
        return {"decision": "done", "reason": "judge_unparseable"}
    dec = str(obj.get("decision") or "").strip().lower()
    if dec not in ("retry", "done"):
        dec = "done"
    reason = str(obj.get("reason") or "").strip()
    if not reason:
        reason = "(no_reason)"
    if len(reason) > 800:
        reason = reason[:800]

    retry_instr = str(obj.get("retry_instructions") or "").strip()
    if len(retry_instr) > 2000:
        retry_instr = retry_instr[:2000]
    if dec != "retry":
        retry_instr = ""
    return {"decision": str(dec), "reason": str(reason), "retry_instructions": str(retry_instr)}


def _merge_omics_analyze_retry_guidance(prev: str, new: str) -> str:
    p = str(prev or "").strip()
    n = str(new or "").strip()
    if not n:
        return p
    merged = n if not p else (p + "\n\n" + n)
    if len(merged) > 3500:
        merged = merged[-3500:]
    return str(merged).strip()


def _apply_omics_analyze_retry_guidance(base_instructions: str, guidance: str) -> str:
    b = str(base_instructions or "")
    g = str(guidance or "").strip()
    if not g:
        return b
    return (
        str(b)
        + "\n\n"
        + "RETRY GUIDANCE (from an automated judge; follow strictly; preserve user intent):\n"
        + str(g)
    )


def _compute_distribution_score(values: list[float], method: str = "entropy") -> float:
    """
    Compute a distribution score for a list of per-tick values.
    Higher score = more evenly distributed across ticks.
    
    Methods:
    - entropy: Shannon entropy (normalized)
    - cv: 1 / (1 + coefficient of variation)
    - spread: fraction of non-zero ticks
    """
    if not values or len(values) < 2:
        return 0.0
    
    arr = np.array(values, dtype=np.float64)
    total = float(arr.sum())
    
    if total <= 0:
        return 0.0
    
    if method == "entropy":
        probs = arr / total
        probs = probs[probs > 0]
        if len(probs) <= 1:
            return 0.0
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(values))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    elif method == "cv":
        mean = float(arr.mean())
        if mean <= 0:
            return 0.0
        std = float(arr.std())
        cv = std / mean
        return float(1.0 / (1.0 + cv))
    
    elif method == "spread":
        nonzero = int((arr > 0).sum())
        return float(nonzero) / float(len(values))
    
    return 0.0


def _decoded_layers_and_kinds(payload: Dict[str, Any]) -> tuple[int, int, Dict[str, Any], Dict[str, str]]:
    H = int(payload.get("H") or 0)
    W = int(payload.get("W") or 0)
    if H <= 0 or W <= 0:
        return 0, 0, {}, {}

    kinds: Dict[str, str] = {}
    layer_meta = payload.get("layers")
    if isinstance(layer_meta, list):
        for m in layer_meta:
            if not isinstance(m, dict):
                continue
            nm = m.get("name")
            if isinstance(nm, str) and nm:
                kinds[nm] = str(m.get("kind") or "continuous")

    data = payload.get("data")
    if not isinstance(data, dict):
        return H, W, {}, kinds

    layers: Dict[str, Any] = {}
    for name, entry in data.items():
        if not isinstance(name, str):
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("dtype") != "float32":
            continue
        b64 = entry.get("b64")
        if not isinstance(b64, str) or not b64:
            continue
        try:
            layers[name] = _decode_float32_b64(b64, expected_len=H * W, layer_name=name)
        except Exception:
            continue

    return H, W, layers, kinds


def _compute_layer_scalars_from_layers(layers: Dict[str, Any], kinds: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, arr in layers.items():
        try:
            s = float(arr.sum())
            m = float(arr.mean())
            nz = int((arr != 0).sum())
            if kinds.get(name) == "categorical":
                eq1 = int((arr == 1).sum())
                out[name] = {"sum": s, "mean": m, "nonzero": nz, "eq1": eq1}
            else:
                out[name] = {"sum": s, "mean": m, "nonzero": nz}
        except Exception:
            continue
    return out


def _run_in_vivo_measurement_series_until_death(
    base: Dict[str, Any], *, ticks: int, seed0: int, death_names: list[str]
) -> tuple[Dict[str, list[float]], int, str]:
    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")

    expected_len = int(H * W)
    data = p.get("data")
    if isinstance(data, dict):
        layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
        for nm0, arr0 in layers0.items():
            ent0 = data.get(nm0)
            if not isinstance(ent0, dict) or ent0.get("dtype") != "float32":
                continue
            ent0["arr"] = np.asarray(arr0, dtype=np.float32).reshape(expected_len)
            ent0["b64"] = ""

    names = _measurement_names_from_payload(p)
    if not names:
        return {}, int(max(0, int(ticks))), ""
    selected = set(names)

    out: Dict[str, list[float]] = {nm: [] for nm in names}
    ticks_i = max(0, int(ticks))

    dn = [str(x) for x in death_names if str(x).strip()]
    died = False
    death_tick = int(ticks_i)
    death_measurement = ""
    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))
        layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
        sel = _compute_selected_measurements_from_layers(p, layers, H, W, selected)
        for nm in names:
            try:
                v = float(sel.get(nm) or 0.0)
            except Exception:
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            out[nm].append(float(v))

        hit = ""
        if dn:
            for nm in dn:
                try:
                    v = float(sel.get(nm) or 0.0)
                except Exception:
                    v = 0.0
                if not np.isfinite(v):
                    v = 0.0
                if v >= 0.5:
                    hit = nm
                    break
        else:
            cell0 = layers.get("cell") if isinstance(layers, dict) else None
            if cell0 is not None:
                try:
                    carr = np.asarray(cell0, dtype=np.float32).reshape(-1)
                except Exception:
                    carr = None
                if carr is not None and carr.size > 0:
                    try:
                        alive = int(np.sum(carr > 0.5))
                    except Exception:
                        alive = -1
                    if int(alive) == 0:
                        hit = "cell_extinction"
        if hit:
            died = True
            death_tick = int(t)
            death_measurement = str(hit)
            break

    if not died:
        death_tick = int(ticks_i)
        death_measurement = ""

    return out, int(death_tick), str(death_measurement)


def _death_tick_from_series(series: Dict[str, list[float]], *, ticks: int, death_names: list[str]) -> tuple[int, str]:
    ticks_i = max(0, int(ticks))
    dn = [str(x) for x in death_names if str(x).strip()]
    if not dn:
        return int(ticks_i), ""
    best_tick = int(ticks_i)
    best_nm = ""
    for nm in dn:
        arr = series.get(nm)
        if not isinstance(arr, list) or not arr:
            continue
        m = min(int(ticks_i), len(arr))
        for i in range(m):
            try:
                v = float(arr[i])
            except Exception:
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            if v >= 0.5:
                if int(i) < int(best_tick):
                    best_tick = int(i)
                    best_nm = str(nm)
                break
    return int(best_tick), str(best_nm)


def _mean_measurement_series_ragged(
    series_list: list[Dict[str, list[float]]], ticks: int, names: list[str]
) -> tuple[Dict[str, list[Optional[float]]], Dict[str, list[int]]]:
    ticks_i = max(0, int(ticks))
    out: Dict[str, list[Optional[float]]] = {nm: [None] * ticks_i for nm in names}
    out_n: Dict[str, list[int]] = {nm: [0] * ticks_i for nm in names}
    if not series_list or ticks_i <= 0:
        return out, out_n

    for nm in names:
        acc = np.zeros((ticks_i,), dtype=np.float64)
        cnt = np.zeros((ticks_i,), dtype=np.int64)
        for s in series_list:
            vv = s.get(nm) if isinstance(s, dict) else None
            if not isinstance(vv, list) or not vv:
                continue
            m = min(int(ticks_i), len(vv))
            for i in range(m):
                try:
                    v = float(vv[i])
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                acc[i] += float(v)
                cnt[i] += 1
        for i in range(int(ticks_i)):
            c = int(cnt[i])
            out_n[nm][i] = int(c)
            if c <= 0:
                out[nm][i] = None
            else:
                out[nm][i] = float(acc[i] / float(c))

    return out, out_n


def _pad_measurement_series_to_ticks(
    series: Dict[str, list[float]], *, ticks: int, names: list[str]
) -> Dict[str, list[Optional[float]]]:
    ticks_i = max(0, int(ticks))
    out: Dict[str, list[Optional[float]]] = {nm: [None] * ticks_i for nm in names}
    if ticks_i <= 0:
        return out
    if not isinstance(series, dict):
        return out
    for nm in names:
        vv = series.get(nm)
        if not isinstance(vv, list) or not vv:
            continue
        m = min(int(ticks_i), len(vv))
        for i in range(m):
            try:
                v = float(vv[i])
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            out[nm][i] = float(v)
    return out


def _alive_n_from_death_ticks(death_ticks: list[int], *, ticks: int) -> list[int]:
    ticks_i = max(0, int(ticks))
    out = [0] * ticks_i
    dts: list[int] = []
    for dt in death_ticks:
        try:
            dts.append(int(dt))
        except Exception:
            dts.append(int(ticks_i))
    for t in range(ticks_i):
        n = 0
        for dt in dts:
            if int(dt) >= int(t):
                n += 1
        out[t] = int(n)
    return out


def _preflight_death_before_ticks(
    payload: Dict[str, Any], *, ticks: int, seed0: int
) -> Optional[Dict[str, Any]]:
    ticks_i = max(0, int(ticks))
    if ticks_i <= 0:
        return None
    death_names = _death_measurement_names_from_payload(payload)
    if not death_names:
        return None
    r = _run_lifespan_death_tick(payload, ticks=int(ticks_i), seed0=int(seed0), death_names=death_names)
    died = bool(r.get("died"))
    try:
        dt = int(r.get("death_tick"))
    except Exception:
        dt = int(ticks_i)
    dm = str(r.get("death_measurement") or "")
    if died and int(dt) < int(ticks_i):
        return {
            "died": True,
            "death_tick": int(dt),
            "death_measurement": str(dm),
            "death_names": list(death_names),
        }
    return None


_STX_GENESETS_DIR = (Path(__file__).resolve().parent / "spatial_transcriptomics").resolve()
_BULK_OMICS_DIR = (Path(__file__).resolve().parent / "bulk_omics").resolve()


_BULK_OMICS_MASK_DICT_LOCK = threading.Lock()
_BULK_OMICS_MASK_DICT_CACHE: Optional[Dict[str, Any]] = None


def _list_stx_gene_sets() -> list[str]:
    d = _STX_GENESETS_DIR
    if not d.exists() or not d.is_dir():
        return []
    out: list[str] = []
    for p in d.iterdir():
        try:
            if not p.is_file():
                continue
        except Exception:
            continue
        nm = str(p.name)
        if not nm or nm.startswith("."):
            continue
        out.append(nm)
    out.sort(key=lambda s: s.lower())
    return out


def _load_stx_gene_set(name: str) -> tuple[str, list[str]]:
    nm = _spatial_omics_type_to_gene_set_filename(name)
    all_sets = _list_stx_gene_sets()
    if not nm:
        if "default.txt" in all_sets:
            nm = "default.txt"
        elif all_sets:
            nm = all_sets[0]
        else:
            return "", []

    p = (_STX_GENESETS_DIR / nm).resolve()
    if _STX_GENESETS_DIR not in p.parents:
        raise ValueError("invalid gene_set")
    if not p.exists() or not p.is_file():
        raise ValueError("unknown gene_set")

    txt = p.read_text(encoding="utf-8")
    genes: list[str] = []
    seen = set()
    for line in txt.splitlines():
        s = str(line).strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s in seen:
            continue
        genes.append(s)
        seen.add(s)
    return nm, genes


def _normalize_spatial_omics_type(name: Any) -> str:
    s = str(name or "").strip().lower()
    if not s:
        return ""
    if s in ("spatial transcriptomics", "spatial_rna", "spatial rna", "rna", "rna panel", "spatial rna panel"):
        return "spatial_rna"
    if s in ("spatial proteomics", "spatial_protein", "spatial protein", "protein", "protein panel", "spatial protein panel"):
        return "spatial_protein"
    return str(name or "").strip()


def _list_spatial_omics_types() -> list[str]:
    return ["spatial_protein", "spatial_rna"]


def _spatial_omics_type_to_gene_set_filename(name: Any) -> str:
    nm = _normalize_spatial_omics_type(name)
    if nm == "spatial_rna":
        return "spatial transcriptomics"
    if nm == "spatial_protein":
        return "spatial proteomics"
    return str(name or "").strip()


def _list_bulk_omics_sets() -> list[str]:
    d = _BULK_OMICS_DIR
    if not d.exists() or not d.is_dir():
        return []
    out: list[str] = []
    for p in d.rglob("*"):
        try:
            if not p.is_file():
                continue
        except Exception:
            continue
        rel = None
        try:
            rel = p.relative_to(d)
        except Exception:
            rel = None
        if rel is None:
            continue
        if any(str(part).startswith(".") for part in rel.parts):
            continue
        nm = rel.as_posix()
        if not nm:
            continue
        out.append(nm)
    out.sort(key=lambda s: s.lower())
    return out


def _load_bulk_omics_set(name: str) -> tuple[str, list[str]]:
    nm = str(name or "").strip()
    all_sets = _list_bulk_omics_sets()
    if not nm:
        prefer = "rna/default.txt"
        if prefer in all_sets:
            nm = prefer
        elif all_sets:
            nm = all_sets[0]
        else:
            return "", []

    p = (_BULK_OMICS_DIR / nm).resolve()
    if _BULK_OMICS_DIR not in p.parents:
        raise ValueError("invalid omics_set")
    if not p.exists() or not p.is_file():
        raise ValueError("unknown omics_set")

    txt = p.read_text(encoding="utf-8")
    layers: list[str] = []
    seen = set()
    for line in txt.splitlines():
        s = str(line).strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s in seen:
            continue
        layers.append(s)
        seen.add(s)
    return nm, layers


def _compute_measurements_from_layers(payload: Dict[str, Any], layers: Dict[str, Any], H: int, W: int) -> Dict[str, Any]:
    cfg = payload.get("measurements_config")
    if not isinstance(cfg, dict):
        return {}
    if int(cfg.get("version") or 0) != 3:
        return {}
    measurements = cfg.get("measurements")
    if not isinstance(measurements, list):
        return {}

    events = payload.get("event_counters") if isinstance(payload, dict) else None
    if not isinstance(events, dict):
        events = {}
    ev = _ExprEval(layers=layers, H=H, W=W, events=events)
    out: Dict[str, Any] = {}
    for m in measurements:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        expr = str(m.get("expr") or "").strip()
        if not name or not expr:
            continue
        try:
            v = ev.eval(expr)
            out[name] = v
        except Exception:
            out[name] = None
    return out


def _compute_selected_measurements_from_layers(
    payload: Dict[str, Any],
    layers: Dict[str, Any],
    H: int,
    W: int,
    selected: set[str],
) -> Dict[str, Any]:
    if not selected:
        return {}
    cfg = payload.get("measurements_config")
    if not isinstance(cfg, dict):
        return {}
    if int(cfg.get("version") or 0) != 3:
        return {}
    measurements = cfg.get("measurements")
    if not isinstance(measurements, list):
        return {}

    events = payload.get("event_counters") if isinstance(payload, dict) else None
    if not isinstance(events, dict):
        events = {}
    ev = _ExprEval(layers=layers, H=H, W=W, events=events)
    out: Dict[str, Any] = {}
    for m in measurements:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name or name not in selected:
            continue
        expr = str(m.get("expr") or "").strip()
        if not expr:
            continue
        try:
            out[name] = ev.eval(expr)
        except Exception:
            out[name] = None
    return out


def _profile_measurements_eval(payload: Dict[str, Any], layers: Dict[str, Any], H: int, W: int) -> Dict[str, Any]:
    cfg = payload.get("measurements_config")
    if not isinstance(cfg, dict) or int(cfg.get("version") or 0) != 3:
        return {"total_s": 0.0, "by_name_s": {}, "values": {}}
    measurements = cfg.get("measurements")
    if not isinstance(measurements, list):
        return {"total_s": 0.0, "by_name_s": {}, "values": {}}

    events = payload.get("event_counters") if isinstance(payload, dict) else None
    if not isinstance(events, dict):
        events = {}
    ev = _ExprEval(layers=layers, H=H, W=W, events=events)
    total_s = 0.0
    by_name_s: Dict[str, float] = {}
    values: Dict[str, Any] = {}
    for m in measurements:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        expr = str(m.get("expr") or "").strip()
        if not name or not expr:
            continue
        t0 = time.perf_counter()
        try:
            values[name] = ev.eval(expr)
        except Exception:
            values[name] = None
        dt = time.perf_counter() - t0
        total_s += float(dt)
        by_name_s[name] = float(by_name_s.get(name) or 0.0) + float(dt)

    return {"total_s": float(total_s), "by_name_s": by_name_s, "values": values}


def _layers_dict_from_payload_data(payload: Dict[str, Any], expected_len: int) -> Dict[str, np.ndarray]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return {}
    out: Dict[str, np.ndarray] = {}
    for nm, ent in data.items():
        if not isinstance(nm, str) or not nm:
            continue
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        arr = ent.get("arr")
        if isinstance(arr, np.ndarray):
            out[nm] = np.asarray(arr, dtype=np.float32).reshape(expected_len)
            continue
        b64 = ent.get("b64")
        if isinstance(b64, str) and b64:
            try:
                out[nm] = _decode_float32_b64(b64, expected_len=expected_len, layer_name=nm)
            except Exception:
                continue
    return out


def _merge_num_fields(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if not isinstance(k, str) or not k:
            continue
        if isinstance(v, (int, float, np.floating)):
            dst[k] = float(dst.get(k) or 0.0) + float(v)


def _merge_step_perf(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for step_key, ent in src.items():
        if not isinstance(step_key, str) or not step_key:
            continue
        if not isinstance(ent, dict):
            continue
        cur = dst.get(step_key)
        if not isinstance(cur, dict):
            cur = {}
            dst[step_key] = cur
        for fk, fv in ent.items():
            if fk == "calls":
                try:
                    cur["calls"] = int(cur.get("calls") or 0) + int(fv)
                except Exception:
                    cur["calls"] = int(cur.get("calls") or 0)
                continue
            if isinstance(fv, (int, float, np.floating)):
                cur[fk] = float(cur.get(fk) or 0.0) + float(fv)


def _profile_run_payload(
    payload: Dict[str, Any],
    ticks: int,
    warmup: int,
    repeats: int,
    do_estimate: bool,
    do_breakdown: bool,
) -> Dict[str, Any]:
    ticks_i = max(1, int(ticks))
    warmup_i = max(0, int(warmup))
    reps_i = max(1, int(repeats))

    base = _deepcopy_payload(payload)

    def _ensure_opts(p: Dict[str, Any]) -> None:
        lop_cfg = p.get("layer_ops_config") if isinstance(p, dict) else None
        if isinstance(lop_cfg, dict):
            if "opt_env_cache" not in lop_cfg and "optimize_env_cache" not in lop_cfg:
                lop_cfg["opt_env_cache"] = True
            if "opt_expr_cache" not in lop_cfg and "optimize_expr_cache" not in lop_cfg:
                lop_cfg["opt_expr_cache"] = True

    def _warmup_run(p: Dict[str, Any], seed0: int) -> None:
        p.pop("event_counters", None)
        p.pop("_profile_layer_ops", None)
        p.pop("_profile_step_names", None)
        p.pop("_profile_expr", None)
        p["_skip_b64_writeback"] = True
        _ensure_opts(p)
        for t in range(warmup_i):
            apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))

    out: Dict[str, Any] = {
        "ok": True,
        "ticks": int(ticks_i),
        "warmup": int(warmup_i),
        "repeats": int(reps_i),
    }

    if do_estimate:
        tick_total_s = 0.0
        meas_total_s = 0.0
        for ri in range(reps_i):
            seed0 = 1337 + (ri * 1000003)
            p = _deepcopy_payload(base)
            _warmup_run(p, seed0)

            t0 = time.perf_counter()
            for t in range(ticks_i):
                apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(warmup_i) + int(t))
            tick_total_s += time.perf_counter() - t0

            H = int(p.get("H") or 0)
            W = int(p.get("W") or 0)
            if H > 0 and W > 0:
                layers_dict = _layers_dict_from_payload_data(p, expected_len=H * W)
                mp = _profile_measurements_eval(p, layers_dict, H, W)
                meas_total_s += float(mp.get("total_s") or 0.0)

        denom = float(max(1, reps_i) * max(1, ticks_i))
        out["estimate"] = {
            "ticks_s": float(tick_total_s),
            "ms_per_tick": float(tick_total_s) * 1000.0 / denom,
            "measurements_s": float(meas_total_s),
        }

    if do_breakdown:
        agg_lop_total_s = 0.0
        agg_by_type_s: Dict[str, float] = {}
        agg_expr_perf: Dict[str, Any] = {}
        agg_step_perf: Dict[str, Any] = {}
        agg_tick_s = 0.0

        agg_meas_total_s = 0.0
        agg_meas_by_name_s: Dict[str, float] = {}

        for ri in range(reps_i):
            seed0 = 4242 + (ri * 1000003)
            p = _deepcopy_payload(base)
            _warmup_run(p, seed0)

            p["event_counters"] = {}
            p["_profile_layer_ops"] = True
            p["_profile_step_names"] = True
            p["_profile_expr"] = True
            p["_skip_b64_writeback"] = True
            _ensure_opts(p)

            tt0 = time.perf_counter()
            for t in range(ticks_i):
                apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(warmup_i) + int(t))
            agg_tick_s += time.perf_counter() - tt0

            ev = p.get("event_counters") if isinstance(p, dict) else None
            lop = ev.get("layer_ops_perf") if isinstance(ev, dict) else None
            if isinstance(lop, dict):
                agg_lop_total_s += float(lop.get("total_s") or 0.0)
                bt = lop.get("by_type_s")
                if isinstance(bt, dict):
                    _merge_num_fields(agg_by_type_s, bt)
                ep = lop.get("expr_perf")
                if isinstance(ep, dict):
                    if not agg_expr_perf:
                        agg_expr_perf = {"calls": 0}
                    try:
                        agg_expr_perf["calls"] = int(agg_expr_perf.get("calls") or 0) + int(ep.get("calls") or 0)
                    except Exception:
                        pass
                    for k in ("env_s", "validate_s", "compile_s", "eval_s", "asarray_s", "writeback_s", "total_s"):
                        agg_expr_perf[k] = float(agg_expr_perf.get(k) or 0.0) + float(ep.get(k) or 0.0)
                sp = lop.get("step_perf")
                if isinstance(sp, dict):
                    _merge_step_perf(agg_step_perf, sp)

            H = int(p.get("H") or 0)
            W = int(p.get("W") or 0)
            if H > 0 and W > 0:
                layers_dict = _layers_dict_from_payload_data(p, expected_len=H * W)
                mp = _profile_measurements_eval(p, layers_dict, H, W)
                agg_meas_total_s += float(mp.get("total_s") or 0.0)
                bnm = mp.get("by_name_s")
                if isinstance(bnm, dict):
                    _merge_num_fields(agg_meas_by_name_s, bnm)

        step_rows = []
        for step_key, ent in agg_step_perf.items():
            if not isinstance(ent, dict):
                continue
            total_s = float(ent.get("total_s") or 0.0)
            calls = int(ent.get("calls") or 0)
            step_rows.append(
                {
                    "step": step_key,
                    "calls": int(calls),
                    "total_ms": float(total_s) * 1000.0,
                    "ms_per_call": (float(total_s) * 1000.0 / float(calls)) if calls > 0 else 0.0,
                }
            )
        step_rows.sort(key=lambda r: float(r.get("total_ms") or 0.0), reverse=True)

        meas_rows = [
            {"name": k, "total_ms": float(v) * 1000.0}
            for k, v in agg_meas_by_name_s.items()
            if isinstance(k, str) and k
        ]
        meas_rows.sort(key=lambda r: float(r.get("total_ms") or 0.0), reverse=True)

        denom = float(max(1, reps_i) * max(1, ticks_i))
        out["breakdown"] = {
            "ticks_s": float(agg_tick_s),
            "ms_per_tick": float(agg_tick_s) * 1000.0 / denom,
            "layer_ops": {
                "total_ms": float(agg_lop_total_s) * 1000.0,
                "by_type_ms": {k: float(v) * 1000.0 for k, v in agg_by_type_s.items() if isinstance(k, str) and k},
                "expr_perf_ms": {
                    k: (float(agg_expr_perf.get(k) or 0.0) * 1000.0)
                    for k in (
                        "env_s",
                        "validate_s",
                        "compile_s",
                        "eval_s",
                        "asarray_s",
                        "writeback_s",
                        "total_s",
                    )
                },
                "expr_calls": int(agg_expr_perf.get("calls") or 0),
                "steps": step_rows,
            },
            "measurements": {
                "total_ms": float(agg_meas_total_s) * 1000.0,
                "by_name": meas_rows,
            },
        }

    out["note"] = "Timing breakdowns include profiling overhead; use estimate.ms_per_tick for a closer speed estimate."
    return out


_WEB_DIR = Path(__file__).resolve().parent / "web_editor"
_RUNS_DIR = Path(__file__).resolve().parent / "runs" / "evolution"
_DOCS_DIR = Path(os.environ.get("DT_DOCS_DIR") or (Path(__file__).resolve().parent / "documents"))
_WORKSPACE_DIR = Path(os.environ.get("DT_WORKSPACE_DIR") or (Path(__file__).resolve().parent / "workspace"))
_OMICS_RUNS_DIR = _WORKSPACE_DIR / "omics_runs"
_TESTS_DIR = Path(__file__).resolve().parent / "tests"
_TESTS_PRICING_PATH = _TESTS_DIR / "pricing.json"

_GAME_STATE_PATH = _WORKSPACE_DIR / "game_state.json"
_GAME_LOCK = threading.RLock()
_COST_MODEL_CENTS = {
    "bulk_rnaseq": 20000,
    "bulk_proteomics": 80000,
    "bulk_metabolomics": 50000,
    "spatial_transcriptomics": 250000,
    "in_vivo_trial": 500000,
}

_TESTS_CANCER_MODELS = {
    "healthy_organism": "healthy_organism.json",
    "cancer_organism": "cancer_organism.json",
    "healthy_cell_culture": "healthy_cell_culture.json",
    "cancer_cell_culture": "cancer_cell_culture.json",
}

_TESTS_CANCER_MODEL_ALIASES = {
    "healthy": "healthy_organism",
    "cancer": "cancer_organism",
    "disease": "cancer_organism",
    "cell_culture_healthy": "healthy_cell_culture",
    "cell_culture_cancer": "cancer_cell_culture",
    "cell_culture_disease": "cancer_cell_culture",
}

_TESTS_HEREDITARY_DISEASE_MODELS = {
    "healthy": "healthy.json",
    "disease": "heredetary_disease.json",
    "cell_culture_healthy": "cell_culture.json",
    "cell_culture_disease": "cell_culture_heredetary_disease.json",
}

_TESTS_AGING_MODELS = {
    "healthy": "healthy_organism.json",
    "cell_culture": "healthy_cell_culture.json",
}

def _tests_cancer_model_list() -> list[Dict[str, Any]]:
    return [
        {"key": "healthy_organism", "label": "Healthy (organism)", "domain": "in_vivo"},
        {"key": "cancer_organism", "label": "Disease (organism)", "domain": "in_vivo"},
        {"key": "healthy_cell_culture", "label": "Healthy (cell culture)", "domain": "in_vitro"},
        {"key": "cancer_cell_culture", "label": "Disease (cell culture)", "domain": "in_vitro"},
    ]

def _tests_cancer_model_path(model_key: Any) -> Path:
    key = str(model_key or "").strip()
    key = str(_TESTS_CANCER_MODEL_ALIASES.get(key) or key)
    fn = _TESTS_CANCER_MODELS.get(key)
    if not fn:
        raise ValueError("unknown model")
    base = (_TESTS_DIR / "cancer").resolve()
    p = (base / str(fn)).resolve()
    if base not in p.parents:
        raise ValueError("invalid model")
    if not p.exists() or not p.is_file():
        raise ValueError("model missing")
    return p

def _tests_load_cancer_model_payload(model_key: Any) -> Dict[str, Any]:
    p = _tests_cancer_model_path(model_key)
    obj = _safe_read_json(p)
    if not isinstance(obj, dict):
        raise ValueError("model JSON invalid")
    return obj

def _tests_hereditary_disease_model_list() -> list[Dict[str, Any]]:
    return [
        {"key": "healthy", "label": "Healthy (organism)", "domain": "in_vivo"},
        {"key": "disease", "label": "Disease (organism)", "domain": "in_vivo"},
        {"key": "cell_culture_healthy", "label": "Healthy (cell culture)", "domain": "in_vitro"},
        {"key": "cell_culture_disease", "label": "Disease (cell culture)", "domain": "in_vitro"},
    ]


def _tests_aging_model_list() -> list[Dict[str, Any]]:
    return [
        {"key": "healthy", "label": "Healthy (organism)", "domain": "in_vivo"},
        {"key": "cell_culture", "label": "Healthy (cell culture)", "domain": "in_vitro"},
    ]

def _tests_hereditary_disease_model_path(model_key: Any) -> Path:
    key = str(model_key or "").strip()
    fn = _TESTS_HEREDITARY_DISEASE_MODELS.get(key)
    if not fn:
        raise ValueError("unknown model")
    base = (_TESTS_DIR / "hereditary_disease").resolve()
    p = (base / str(fn)).resolve()
    if base not in p.parents:
        raise ValueError("invalid model")
    if not p.exists() or not p.is_file():
        raise ValueError("model missing")
    return p

def _tests_load_hereditary_disease_model_payload(model_key: Any) -> Dict[str, Any]:
    p = _tests_hereditary_disease_model_path(model_key)
    obj = _safe_read_json(p)
    if not isinstance(obj, dict):
        raise ValueError("model JSON invalid")
    return obj


def _tests_aging_model_path(model_key: Any) -> Path:
    key = str(model_key or "").strip()
    fn = _TESTS_AGING_MODELS.get(key)
    if not fn:
        raise ValueError("unknown model")
    base = (_TESTS_DIR / "aging").resolve()
    p = (base / str(fn)).resolve()
    if base not in p.parents:
        raise ValueError("invalid model")
    if not p.exists() or not p.is_file():
        raise ValueError("model missing")
    return p


def _tests_load_aging_model_payload(model_key: Any) -> Dict[str, Any]:
    p = _tests_aging_model_path(model_key)
    obj = _safe_read_json(p)
    if not isinstance(obj, dict):
        raise ValueError("model JSON invalid")
    return obj

def _tests_normalize_challenge(challenge: Any) -> str:
    ch = str(challenge or "").strip().lower()
    if ch in ("cancer", "hereditary_disease", "aging"):
        return ch
    raise ValueError("unknown challenge")

def _tests_model_list_for_challenge(challenge: Any) -> list[Dict[str, Any]]:
    ch = _tests_normalize_challenge(challenge)
    if ch == "cancer":
        return _tests_cancer_model_list()
    if ch == "hereditary_disease":
        return _tests_hereditary_disease_model_list()
    return _tests_aging_model_list()

def _tests_load_model_payload_for_challenge(challenge: Any, model_key: Any) -> Dict[str, Any]:
    ch = _tests_normalize_challenge(challenge)
    if ch == "cancer":
        return _tests_load_cancer_model_payload(model_key)
    if ch == "hereditary_disease":
        return _tests_load_hereditary_disease_model_payload(model_key)
    return _tests_load_aging_model_payload(model_key)

def _tests_claim_cure_disease_model_key_for_challenge(challenge: Any) -> str:
    ch = _tests_normalize_challenge(challenge)
    if ch == "cancer":
        return "cancer_organism"
    if ch == "hereditary_disease":
        return "disease"
    return "healthy"

def _tests_is_in_vitro_model(model_key: Any) -> bool:
    key = str(model_key or "").strip().lower()
    return "cell_culture" in key

def _tests_pricing_for_challenge(challenge: str) -> Dict[str, Any]:
    dd = _safe_read_json(_TESTS_PRICING_PATH)
    if not isinstance(dd, dict):
        return {}
    tests = dd.get("tests")
    if not isinstance(tests, dict):
        return {}
    ch = tests.get(str(challenge or "").strip())
    if not isinstance(ch, dict):
        return {}
    return ch


def _tests_compute_unit_cost_cents(
    *,
    challenge: str,
    kind: str,
    model_key: Any,
    ticks: int,
    interventions_n: int,
) -> int:
    cfg = _tests_pricing_for_challenge(challenge)
    kinds = cfg.get("kinds") if isinstance(cfg, dict) else None
    kcfg = kinds.get(str(kind)) if isinstance(kinds, dict) else None
    if not isinstance(kcfg, dict):
        return 0

    ticks_i = max(0, int(ticks))
    iv_i = max(0, int(interventions_n))

    # New schema (costs.csv-like): kind has per-context pricing instead of a single
    # config scaled by multipliers.
    ctx_key = "in_vitro" if _tests_is_in_vitro_model(model_key) else "in_vivo"
    ctxs = kcfg.get("contexts") if isinstance(kcfg, dict) else None
    kcfg_ctx = ctxs.get(ctx_key) if isinstance(ctxs, dict) else None
    if isinstance(kcfg_ctx, dict):
        try:
            base = int(kcfg_ctx.get("unit_cents") or 0)
        except Exception:
            base = 0
        try:
            per_tick_base = int(kcfg_ctx.get("unit_per_tick_cents") or 0)
        except Exception:
            per_tick_base = 0
        try:
            per_tick_per_iv = int(kcfg_ctx.get("unit_per_tick_per_intervention_cents") or 0)
        except Exception:
            per_tick_per_iv = 0
        try:
            per_iv = int(kcfg_ctx.get("unit_per_intervention_cents") or 0)
        except Exception:
            per_iv = 0

        unit = (
            int(base)
            + int(ticks_i) * (int(per_tick_base) + int(per_tick_per_iv) * int(iv_i))
            + int(per_iv) * int(iv_i)
        )
        return int(max(0, unit))

    # Backward-compatible schema: base + per_tick*ticks + per_iv*ivs, scaled by
    # in_vitro/in_vivo multiplier.
    try:
        base = int(kcfg.get("unit_cents") or 0)
    except Exception:
        base = 0
    try:
        per_tick = int(kcfg.get("unit_per_tick_cents") or 0)
    except Exception:
        per_tick = 0
    try:
        per_iv = int(kcfg.get("unit_per_intervention_cents") or 0)
    except Exception:
        per_iv = 0

    unit = int(base) + int(per_tick) * int(ticks_i) + int(per_iv) * int(iv_i)

    mults = cfg.get("multipliers") if isinstance(cfg, dict) else None
    mult = 1.0
    if isinstance(mults, dict):
        try:
            mult = float(mults.get(ctx_key) or 1.0)
        except Exception:
            mult = 1.0
    if not np.isfinite(mult) or mult <= 0.0:
        mult = 1.0

    return int(max(0, int(round(float(unit) * float(mult)))))


def _tests_compute_fixed_cost_cents(*, challenge: str, kind: str, model_key: Any, interventions_n: int) -> int:
    cfg = _tests_pricing_for_challenge(challenge)
    kinds = cfg.get("kinds") if isinstance(cfg, dict) else None
    kcfg = kinds.get(str(kind)) if isinstance(kinds, dict) else None
    if not isinstance(kcfg, dict):
        return 0
    iv_i = max(0, int(interventions_n))
    if int(iv_i) <= 0:
        return 0

    ctx_key = "in_vitro" if _tests_is_in_vitro_model(model_key) else "in_vivo"
    ctxs = kcfg.get("contexts") if isinstance(kcfg, dict) else None
    kcfg_ctx = ctxs.get(ctx_key) if isinstance(ctxs, dict) else None
    if isinstance(kcfg_ctx, dict):
        try:
            return int(max(0, int(kcfg_ctx.get("intervention_setup_cents") or 0)))
        except Exception:
            return 0
    return 0


def _tests_make_charge(
    *,
    kind: str,
    samples: int,
    unit_cost_cents: int,
    fixed_cost_cents: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n = max(0, int(samples))
    unit = max(0, int(unit_cost_cents))
    fixed = max(0, int(fixed_cost_cents))
    total = int(unit) * int(n) + int(fixed)
    out = {
        "id": uuid.uuid4().hex[:12],
        "ts": float(time.time()),
        "currency": "USD",
        "kind": str(kind or ""),
        "samples": int(n),
        "unit_cost_cents": int(unit),
        "fixed_cost_cents": int(fixed),
        "total_cost_cents": int(total),
        "cost_cents": int(total),
        "unit_cost_usd": float(unit) / 100.0,
        "fixed_cost_usd": float(fixed) / 100.0,
        "total_cost_usd": float(total) / 100.0,
        "cost_usd": float(total) / 100.0,
    }
    if isinstance(meta, dict) and meta:
        out["meta"] = meta
    return out


def _tests_validate_protein_interventions(interventions_in: Any) -> list[Dict[str, Any]]:
    if interventions_in is None:
        return []
    if not isinstance(interventions_in, list):
        raise ValueError("interventions must be a list")
    out: list[Dict[str, Any]] = []
    for iv in interventions_in:
        if not isinstance(iv, dict):
            continue
        layer = str(iv.get("layer") or "").strip()
        if not layer:
            continue
        if not layer.startswith("protein_"):
            raise ValueError("targeted interventions must be masked protein_<int> ids")
        tail = layer[len("protein_") :]
        if not tail.isdigit():
            raise ValueError("targeted interventions must be masked protein_<int> ids")
        out.append(iv)
    return out


_TESTS_PROTEIN_MASK_LOCK = threading.Lock()
_TESTS_PROTEIN_MASK_CACHE: Dict[str, tuple[Dict[str, str], Dict[str, str]]] = {}


def _tests_get_protein_mask_maps(model_key: Any, *, challenge: str = "cancer") -> tuple[Dict[str, str], Dict[str, str]]:
    ch = str(challenge or "").strip().lower() or "cancer"
    mk = str(model_key or "").strip().lower()
    if not mk:
        if ch == "cancer":
            mk = "cancer"
        elif ch == "hereditary_disease":
            mk = "disease"
        else:
            mk = "healthy"
    cache_key = f"{ch}:{mk}"
    with _TESTS_PROTEIN_MASK_LOCK:
        cached = _TESTS_PROTEIN_MASK_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if ch == "cancer":
        payload = _tests_load_cancer_model_payload(mk)
    elif ch == "hereditary_disease":
        payload = _tests_load_hereditary_disease_model_payload(mk)
    elif ch == "aging":
        payload = _tests_load_aging_model_payload(mk)
    else:
        raise ValueError("unknown challenge")

    real_layers: list[str] = []
    try:
        _, prot_feats = _load_bulk_omics_set("protein/Bulk Proteomics")
    except Exception:
        prot_feats = []
    try:
        data0 = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data0, dict) and prot_feats:
            for f in list(prot_feats):
                ent = data0.get(str(f))
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if not isinstance(b64, str) or not b64:
                    continue
                real_layers.append(str(f))
    except Exception:
        real_layers = []

    if not real_layers:
        real_layers = _protein_layer_names_from_payload(payload)

    try:
        core_to_idx = _bulk_omics_get_core_index_map()
    except Exception:
        core_to_idx = {}

    def _layer_to_mask(layer: str) -> Optional[str]:
        core = _bulk_omics_core_key(layer)
        idx = core_to_idx.get(str(core))
        if idx is None:
            return None
        return f"protein_{int(idx)}"

    if core_to_idx and real_layers and prot_feats:
        real_layers2: list[str] = []
        for rl in real_layers:
            if _layer_to_mask(str(rl)) is not None:
                real_layers2.append(str(rl))
        real_layers2.sort(key=lambda s: int(core_to_idx.get(str(_bulk_omics_core_key(s))) or 10**9))
        real_layers = real_layers2

    real_to_mask: Dict[str, str] = {}
    mask_to_real: Dict[str, str] = {}

    next_unknown_idx: Optional[int] = None
    if core_to_idx:
        try:
            next_unknown_idx = int(max(core_to_idx.values() or [0])) + 1
        except Exception:
            next_unknown_idx = int(len(core_to_idx) + 1)

    for i, real in enumerate(real_layers):
        masked: Optional[str] = None
        if core_to_idx:
            masked = _layer_to_mask(str(real))
        if not masked:
            if core_to_idx and next_unknown_idx is not None:
                core = _bulk_omics_core_key(str(real))
                idx = core_to_idx.get(str(core))
                if idx is None:
                    idx = int(next_unknown_idx)
                    next_unknown_idx = int(next_unknown_idx) + 1
                    if core:
                        core_to_idx[str(core)] = int(idx)
                masked = f"protein_{int(idx)}"
            else:
                masked = f"protein_{int(i + 1)}"
        real_to_mask[str(real)] = str(masked)
        mask_to_real[str(masked)] = str(real)

    with _TESTS_PROTEIN_MASK_LOCK:
        _TESTS_PROTEIN_MASK_CACHE[cache_key] = (real_to_mask, mask_to_real)
        return _TESTS_PROTEIN_MASK_CACHE[cache_key]


def _tests_translate_interventions_masked_to_real(
    interventions: list[Dict[str, Any]], *, model_key: Any, challenge: str = "cancer"
) -> list[Dict[str, Any]]:
    if not interventions:
        return []
    _, mask_to_real = _tests_get_protein_mask_maps(model_key, challenge=str(challenge))
    out: list[Dict[str, Any]] = []
    for iv in interventions:
        if not isinstance(iv, dict):
            continue
        layer = str(iv.get("layer") or "").strip()
        if not layer:
            continue
        real = mask_to_real.get(layer)
        if not real:
            raise ValueError(f"unknown masked protein id: {layer}")
        iv2 = dict(iv)
        iv2["layer"] = str(real)
        out.append(iv2)
    return out


def _tests_mask_gene_list(genes: list[str]) -> tuple[list[str], Dict[str, str]]:
    out: list[str] = []
    real_to_mask: Dict[str, str] = {}
    for i, g in enumerate(list(genes)):
        masked = f"gene_{int(i + 1)}"
        out.append(masked)
        real_to_mask[str(g)] = str(masked)
    return out, real_to_mask


def _tests_mask_feature_list(features: list[str], prefix: str) -> tuple[list[str], Dict[str, str]]:
    out: list[str] = []
    real_to_mask: Dict[str, str] = {}
    pfx = str(prefix or "").strip().lower() or "feature"
    for i, f in enumerate(list(features)):
        masked = f"{str(pfx)}_{int(i + 1)}"
        out.append(masked)
        real_to_mask[str(f)] = str(masked)
    return out, real_to_mask


_BULK_OMICS_CORE_INDEX_MAP: Optional[Dict[str, int]] = None


def _bulk_omics_core_key(name: Any) -> str:
    s = str(name or "")
    if s.startswith("rna_"):
        return s[len("rna_") :]
    if s.startswith("protein_"):
        return s[len("protein_") :]
    return s


def _bulk_omics_mask_dictionary_path() -> Path:
    try:
        p = (_BULK_OMICS_DIR / "omics_mask_dictionary.csv").resolve()
        if _BULK_OMICS_DIR not in p.parents and p != _BULK_OMICS_DIR:
            return (_BULK_OMICS_DIR / "omics_mask_dictionary.csv").resolve()
        return p
    except Exception:
        return (_BULK_OMICS_DIR / "omics_mask_dictionary.csv").resolve()


def _bulk_omics_load_mask_dictionary() -> Dict[str, Any]:
    global _BULK_OMICS_MASK_DICT_CACHE
    with _BULK_OMICS_MASK_DICT_LOCK:
        if isinstance(_BULK_OMICS_MASK_DICT_CACHE, dict):
            return _BULK_OMICS_MASK_DICT_CACHE

        core_to_idx: Dict[str, int] = {}

        p = _bulk_omics_mask_dictionary_path()
        try:
            if p.exists() and p.is_file():
                with p.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in list(reader):
                        if not isinstance(row, dict):
                            continue
                        orig_rna = str(row.get("original_name_rna") or "").strip()
                        orig_prot = str(row.get("original_name_protein") or row.get("orginal_name_protein") or "").strip()
                        new_rna = str(row.get("new_name_rna") or "").strip()
                        new_prot = str(row.get("new_name_protein") or "").strip()

                        idx: Optional[int] = None
                        if new_rna.startswith("rna_") and new_rna[len("rna_") :].isdigit():
                            idx = int(new_rna[len("rna_") :])
                        if idx is None and new_prot.startswith("protein_") and new_prot[len("protein_") :].isdigit():
                            idx = int(new_prot[len("protein_") :])

                        core = ""
                        if orig_rna:
                            core = _bulk_omics_core_key(orig_rna)
                        elif orig_prot:
                            core = _bulk_omics_core_key(orig_prot)

                        if not core or idx is None or int(idx) <= 0:
                            continue

                        prev = core_to_idx.get(str(core))
                        if prev is not None and int(prev) != int(idx):
                            continue
                        core_to_idx[str(core)] = int(idx)
        except Exception:
            core_to_idx = {}

        _BULK_OMICS_MASK_DICT_CACHE = {
            "core_to_idx": core_to_idx,
        }
        return _BULK_OMICS_MASK_DICT_CACHE


def _bulk_omics_get_core_index_map() -> Dict[str, int]:
    global _BULK_OMICS_CORE_INDEX_MAP
    if isinstance(_BULK_OMICS_CORE_INDEX_MAP, dict) and _BULK_OMICS_CORE_INDEX_MAP:
        return _BULK_OMICS_CORE_INDEX_MAP

    try:
        md = _bulk_omics_load_mask_dictionary()
        c2i = md.get("core_to_idx") if isinstance(md, dict) else None
        if isinstance(c2i, dict) and c2i:
            _BULK_OMICS_CORE_INDEX_MAP = dict((str(k), int(v)) for k, v in c2i.items())
            return _BULK_OMICS_CORE_INDEX_MAP
    except Exception:
        pass

    m: Dict[str, int] = {}
    next_i = 1

    sets = _list_bulk_omics_sets()
    prot_sets = sorted([s for s in sets if str(s).startswith("protein/")])
    rna_sets = sorted([s for s in sets if str(s).startswith("rna/")])

    for s in prot_sets:
        try:
            _, feats = _load_bulk_omics_set(str(s))
        except Exception:
            feats = []
        for f in feats:
            core = _bulk_omics_core_key(f)
            if not core or core in m:
                continue
            m[str(core)] = int(next_i)
            next_i += 1

    for s in rna_sets:
        try:
            _, feats = _load_bulk_omics_set(str(s))
        except Exception:
            feats = []
        for f in feats:
            core = _bulk_omics_core_key(f)
            if not core or core in m:
                continue
            m[str(core)] = int(next_i)
            next_i += 1

    _BULK_OMICS_CORE_INDEX_MAP = m
    return m


def _bulk_omics_mask_feature_headers(features: list[str], kind: str) -> list[str]:
    k = str(kind or "")
    if k == "bulk_metabolomics":
        return [
            (str(f)[len("molecule_") :] if str(f).startswith("molecule_") else str(f))
            for f in list(features)
        ]

    if k not in ("bulk_rnaseq", "bulk_proteomics"):
        return list(features)

    prefix = "rna" if k == "bulk_rnaseq" else "protein"
    core_to_idx = _bulk_omics_get_core_index_map()

    out: list[str] = []
    for f in list(features):
        core = _bulk_omics_core_key(f)
        idx = core_to_idx.get(str(core))
        if idx is None:
            try:
                nxt = int(max(core_to_idx.values() or [0])) + 1
            except Exception:
                nxt = int(len(core_to_idx) + 1)
            idx = int(nxt)
            core_to_idx[str(core)] = int(idx)
        out.append(f"{str(prefix)}_{int(idx)}")
    return out


def _stx_kind_from_gene_set_and_genes(gene_set_name: Any, genes: list[str]) -> str:
    nm = str(gene_set_name or "").strip().lower()
    if "proteom" in nm or "protein" in nm:
        return "bulk_proteomics"
    if "transcript" in nm or nm.startswith("rna"):
        return "bulk_rnaseq"
    for g in list(genes):
        s = str(g or "")
        if s.startswith("protein_"):
            return "bulk_proteomics"
        if s.startswith("rna_"):
            return "bulk_rnaseq"
    return "bulk_rnaseq"


def _tests_lifespan_stats(dts: list[int], *, ticks: int) -> Dict[str, Any]:
    ticks_i = max(0, int(ticks))
    arr = np.asarray([int(x) for x in dts], dtype=np.float64) if dts else np.asarray([], dtype=np.float64)
    if arr.size:
        med = float(np.median(arr))
        mean = float(np.mean(arr))
        try:
            p25 = float(np.quantile(arr, 0.25))
        except Exception:
            p25 = float(med)
        try:
            p75 = float(np.quantile(arr, 0.75))
        except Exception:
            p75 = float(med)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
    else:
        med = float(ticks_i)
        mean = float(ticks_i)
        p25 = float(ticks_i)
        p75 = float(ticks_i)
        mn = float(ticks_i)
        mx = float(ticks_i)
    deaths = int(sum(1 for dt in dts if int(dt) < int(ticks_i)))
    return {
        "n": int(len(dts)),
        "ticks": int(ticks_i),
        "median_lifespan_tick": float(med),
        "mean_lifespan_tick": float(mean),
        "p25_lifespan_tick": float(p25),
        "p75_lifespan_tick": float(p75),
        "min_lifespan_tick": float(mn),
        "max_lifespan_tick": float(mx),
        "deaths": int(deaths),
        "survivors": int(len(dts) - deaths),
    }


def _sanitize_player_id(player_id: Any) -> str:
    s = str(player_id or "").strip()
    if not s:
        return "default"
    if len(s) > 64:
        s = s[:64]
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    out = "".join(ch for ch in s if ch in allowed)
    return out or "default"


def _bulk_omics_kind_from_set_name(omics_set: str) -> str:
    nm = str(omics_set or "").strip().replace("\\", "/")
    if nm.startswith("rna/"):
        return "bulk_rnaseq"
    if nm.startswith("protein/"):
        return "bulk_proteomics"
    if nm.startswith("metabolite/") or nm.startswith("metabolomics/"):
        return "bulk_metabolomics"
    parts = [p for p in nm.split("/") if p]
    if parts:
        p0 = parts[0].lower()
        if p0 == "rna":
            return "bulk_rnaseq"
        if p0 == "protein":
            return "bulk_proteomics"
        if p0 in ("metabolite", "metabolomics"):
            return "bulk_metabolomics"
    return "bulk_rnaseq"


def _game_load_state() -> Dict[str, Any]:
    try:
        _ensure_dirs()
    except Exception:
        pass
    st = _safe_read_json(_GAME_STATE_PATH)
    if not isinstance(st, dict):
        st = {"version": 1, "players": {}}
    if not isinstance(st.get("players"), dict):
        st["players"] = {}
    if int(st.get("version") or 0) != 1:
        st = {"version": 1, "players": dict(st.get("players") or {})}
    return st


def _game_player_entry(state: Dict[str, Any], player_id: str) -> Dict[str, Any]:
    players = state.get("players")
    if not isinstance(players, dict):
        players = {}
        state["players"] = players
    ent = players.get(player_id)
    if not isinstance(ent, dict):
        ent = {
            "money_spent_cents": 0,
            "ledger": [],
            "created_at": float(time.time()),
            "updated_at": float(time.time()),
        }
        players[player_id] = ent
    if not isinstance(ent.get("ledger"), list):
        ent["ledger"] = []
    try:
        ent["money_spent_cents"] = int(ent.get("money_spent_cents") or 0)
    except Exception:
        ent["money_spent_cents"] = 0
    return ent


def _game_get_player_int(player_id_in: Any, key: str, *, default: int = 0) -> int:
    player_id = _sanitize_player_id(player_id_in)
    with _GAME_LOCK:
        st = _game_load_state()
        ent = _game_player_entry(st, player_id)
        v = ent.get(str(key))
        try:
            return int(v)
        except Exception:
            return int(default)


def _game_set_player_int(player_id_in: Any, key: str, value: int) -> None:
    player_id = _sanitize_player_id(player_id_in)
    with _GAME_LOCK:
        st = _game_load_state()
        ent = _game_player_entry(st, player_id)
        try:
            ent[str(key)] = int(value)
        except Exception:
            ent[str(key)] = int(0)
        ent["updated_at"] = float(time.time())
        _atomic_write_json(_GAME_STATE_PATH, st)


def _game_public_player(player_id: str, ent: Dict[str, Any], *, last_charge: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cents = 0
    try:
        cents = int(ent.get("money_spent_cents") or 0)
    except Exception:
        cents = 0
    out = {
        "player_id": str(player_id),
        "currency": "USD",
        "money_spent_cents": int(cents),
        "money_spent_usd": float(cents) / 100.0,
        "ledger": ent.get("ledger") if isinstance(ent.get("ledger"), list) else [],
        "created_at": ent.get("created_at"),
        "updated_at": ent.get("updated_at"),
    }
    if isinstance(last_charge, dict):
        out["last_charge"] = last_charge
    return out


def _game_compute_charge(kind: str, samples: int, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    k = str(kind or "").strip() or "bulk_rnaseq"
    unit = int(_COST_MODEL_CENTS.get(k) or 0)
    n = max(0, int(samples))
    total = int(unit) * int(n)
    out = {
        "id": uuid.uuid4().hex[:12],
        "ts": float(time.time()),
        "currency": "USD",
        "kind": k,
        "samples": int(n),
        "unit_cost_cents": int(unit),
        "cost_cents": int(total),
        "unit_cost_usd": float(unit) / 100.0,
        "cost_usd": float(total) / 100.0,
    }
    if isinstance(meta, dict) and meta:
        out["meta"] = meta
    return out


def _game_apply_charge(player_id_in: Any, charge: Dict[str, Any]) -> Dict[str, Any]:
    player_id = _sanitize_player_id(player_id_in)
    if not isinstance(charge, dict):
        charge = {}
    cents = 0
    try:
        cents = int(charge.get("cost_cents") or 0)
    except Exception:
        cents = 0
    cents = max(0, int(cents))

    with _GAME_LOCK:
        st = _game_load_state()
        ent = _game_player_entry(st, player_id)
        try:
            ent["money_spent_cents"] = int(ent.get("money_spent_cents") or 0) + int(cents)
        except Exception:
            ent["money_spent_cents"] = int(cents)

        led = ent.get("ledger")
        if not isinstance(led, list):
            led = []
            ent["ledger"] = led
        led.append(charge)
        if len(led) > 250:
            ent["ledger"] = led[-250:]

        ent["updated_at"] = float(time.time())
        _atomic_write_json(_GAME_STATE_PATH, st)
        return _game_public_player(player_id, ent, last_charge=charge)


def _game_get_player_state(player_id_in: Any) -> Dict[str, Any]:
    player_id = _sanitize_player_id(player_id_in)
    with _GAME_LOCK:
        st = _game_load_state()
        ent = _game_player_entry(st, player_id)
        _atomic_write_json(_GAME_STATE_PATH, st)
        return _game_public_player(player_id, ent)


def _game_reset_player(player_id_in: Any) -> Dict[str, Any]:
    player_id = _sanitize_player_id(player_id_in)
    with _GAME_LOCK:
        st = _game_load_state()
        ent = _game_player_entry(st, player_id)
        ent["money_spent_cents"] = 0
        ent["ledger"] = []
        try:
            ent.pop("aging_claim_cure_min_reps", None)
        except Exception:
            pass
        ent["updated_at"] = float(time.time())
        _atomic_write_json(_GAME_STATE_PATH, st)
        return _game_public_player(player_id, ent)


def _find_cell_layer_name(payload: Dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        return ""
    for nm in ("cell", "cell_type"):
        ent = data.get(nm)
        if isinstance(ent, dict) and ent.get("dtype") == "float32" and isinstance(ent.get("b64"), str):
            return nm
    return ""


def _deepcopy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload))


def _ensure_layer_ops_opts(p: Dict[str, Any]) -> None:
    lop_cfg = p.get("layer_ops_config") if isinstance(p, dict) else None
    if isinstance(lop_cfg, dict):
        if "opt_env_cache" not in lop_cfg and "optimize_env_cache" not in lop_cfg:
            lop_cfg["opt_env_cache"] = True
        if "opt_expr_cache" not in lop_cfg and "optimize_expr_cache" not in lop_cfg:
            lop_cfg["opt_expr_cache"] = True


def _apply_interventions_to_payload_inplace(payload: Dict[str, Any], interventions_in: Any) -> int:
    if not isinstance(interventions_in, list) or not interventions_in:
        return 0

    H = int(payload.get("H") or 0)
    W = int(payload.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")
    expected_len = int(H * W)

    layer_meta = payload.get("layers")
    kinds: Dict[str, str] = {}
    if isinstance(layer_meta, list):
        for m in layer_meta:
            if not isinstance(m, dict):
                continue
            nm = m.get("name")
            if isinstance(nm, str) and nm:
                kinds[nm] = str(m.get("kind") or "continuous")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("payload missing data")

    huge = 1e9
    applied = 0

    for iv in interventions_in:
        if not isinstance(iv, dict):
            continue
        layer = str(iv.get("layer") or "").strip()
        if not layer:
            continue

        direction = str(iv.get("direction") or "").strip().lower()
        if direction in ("+", "inc", "increase", "up", "pos", "positive"):
            direction = "up"
        elif direction in ("-", "dec", "decrease", "down", "neg", "negative"):
            direction = "down"
        else:
            raise ValueError(f"invalid intervention direction for layer '{layer}': {direction!r}")

        try:
            dose = float(iv.get("dose") or 0.0)
        except Exception:
            dose = 0.0
        if not np.isfinite(dose) or dose < 0.0:
            dose = 0.0

        delta = 0.1 * float(dose)
        factor = (1.0 + delta) if direction == "up" else (1.0 - delta)
        if factor < 0.0:
            factor = 0.0
        if abs(factor - 1.0) < 1e-12:
            continue

        kind = kinds.get(layer) or "continuous"
        if kind == "categorical":
            raise ValueError(f"cannot apply interventions to categorical layer '{layer}'")

        ent = data.get(layer)
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            raise ValueError(f"unknown float32 layer: {layer}")
        b64 = ent.get("b64")
        if not isinstance(b64, str) or not b64:
            raise ValueError(f"layer '{layer}' is missing b64 data")

        arr = _decode_float32_b64(b64, expected_len=expected_len, layer_name=layer)
        arr2 = np.asarray(arr, dtype=np.float32) * np.float32(factor)
        arr2 = np.nan_to_num(arr2, nan=0.0, posinf=huge, neginf=0.0)
        arr2 = np.clip(arr2, 0.0, huge)
        if kind == "counts":
            arr2 = np.clip(np.rint(arr2), 0.0, huge)
        ent["b64"] = _encode_float32_b64(np.asarray(arr2, dtype=np.float32).reshape(expected_len))
        ent.pop("arr", None)
        applied += 1

    return int(applied)


def _run_payload_ticks(base: Dict[str, Any], ticks: int, seed0: int) -> Dict[str, Any]:
    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    ticks_i = max(0, int(ticks))
    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))
    return p


def _measurement_names_from_payload(payload: Dict[str, Any]) -> list[str]:
    cfg = payload.get("measurements_config")
    if not isinstance(cfg, dict) or int(cfg.get("version") or 0) != 3:
        return []
    measurements = cfg.get("measurements")
    if not isinstance(measurements, list):
        return []
    out: list[str] = []
    for m in measurements:
        if not isinstance(m, dict):
            continue
        nm = str(m.get("name") or "").strip()
        if not nm:
            continue
        out.append(nm)
    # preserve configured order while de-duping
    seen: set[str] = set()
    out2: list[str] = []
    for nm in out:
        if nm in seen:
            continue
        out2.append(nm)
        seen.add(nm)
    return out2


def _death_measurement_names_from_payload(payload: Dict[str, Any]) -> list[str]:
    names = _measurement_names_from_payload(payload)
    out: list[str] = []
    for nm in names:
        s = str(nm or "")
        s2 = s.lower()
        if not s2:
            continue
        # In vivo death channels are named like "*_death" (e.g. glucose_death).
        # In vitro models may include measurements like "deaths_per_tick" which are not organism death triggers.
        if s2.endswith("_death") or s2.startswith("death_"):
            out.append(s)
    return out


def _cell_culture_metrics_from_payload(
    base: Dict[str, Any],
    *,
    ticks: int,
    seed0: int,
) -> Dict[str, Any]:
    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")
    expected_len = int(H * W)

    data = p.get("data")
    if isinstance(data, dict):
        layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
        for nm0, arr0 in layers0.items():
            ent0 = data.get(nm0)
            if not isinstance(ent0, dict) or ent0.get("dtype") != "float32":
                continue
            ent0["arr"] = np.asarray(arr0, dtype=np.float32).reshape(expected_len)
            ent0["b64"] = ""

    layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
    cell0 = layers0.get("cell") if isinstance(layers0, dict) else None
    n_cells_start = 0
    if cell0 is not None:
        try:
            n_cells_start = int((np.asarray(cell0, dtype=np.float32).reshape(-1) > 0.5).sum())
        except Exception:
            n_cells_start = 0

    ticks_i = max(0, int(ticks))
    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))

    ev = p.get("event_counters") if isinstance(p, dict) else None
    op_totals = ev.get("op_totals") if isinstance(ev, dict) else None
    births_total = int(op_totals.get("divides_cells") or 0) if isinstance(op_totals, dict) else 0
    deaths_total = int(op_totals.get("cell_death") or 0) if isinstance(op_totals, dict) else 0

    layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
    cell_arr = layers.get("cell") if isinstance(layers, dict) else None
    n_cells_end = 0
    if cell_arr is not None:
        try:
            n_cells_end = int((np.asarray(cell_arr, dtype=np.float32).reshape(-1) > 0.5).sum())
        except Exception:
            n_cells_end = 0

    confluency_end = float(n_cells_end) / float(expected_len) if expected_len > 0 else 0.0
    return {
        "n_cells_start": int(n_cells_start),
        "n_cells_end": int(n_cells_end),
        "confluency_end": float(confluency_end),
        "births_total": int(births_total),
        "deaths_total": int(deaths_total),
        "net_births_total": int(int(births_total) - int(deaths_total)),
    }


def _run_cell_culture_measurement_series_and_metrics(
    base: Dict[str, Any],
    *,
    ticks: int,
    seed0: int,
    names: list[str],
) -> tuple[Dict[str, list[float]], Dict[str, Any]]:
    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")
    expected_len = int(H * W)

    data = p.get("data")
    if isinstance(data, dict):
        layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
        for nm0, arr0 in layers0.items():
            ent0 = data.get(nm0)
            if not isinstance(ent0, dict) or ent0.get("dtype") != "float32":
                continue
            ent0["arr"] = np.asarray(arr0, dtype=np.float32).reshape(expected_len)
            ent0["b64"] = ""

    # Initial metrics (before running)
    cell0 = _layers_dict_from_payload_data(p, expected_len=expected_len).get("cell")
    n_cells_start = 0
    if cell0 is not None:
        try:
            n_cells_start = int((np.asarray(cell0, dtype=np.float32).reshape(-1) > 0.5).sum())
        except Exception:
            n_cells_start = 0

    ticks_i = max(0, int(ticks))
    selected = set(str(x) for x in names if str(x).strip())
    series: Dict[str, list[float]] = {nm: [] for nm in names}

    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))
        layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
        sel = _compute_selected_measurements_from_layers(p, layers, H, W, selected)
        for nm in names:
            try:
                v = float(sel.get(nm) or 0.0)
            except Exception:
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            series[nm].append(float(v))

    ev = p.get("event_counters") if isinstance(p, dict) else None
    op_totals = ev.get("op_totals") if isinstance(ev, dict) else None
    births_total = int(op_totals.get("divides_cells") or 0) if isinstance(op_totals, dict) else 0
    deaths_total = int(op_totals.get("cell_death") or 0) if isinstance(op_totals, dict) else 0

    layers_end = _layers_dict_from_payload_data(p, expected_len=expected_len)
    cell_end = layers_end.get("cell") if isinstance(layers_end, dict) else None
    n_cells_end = 0
    if cell_end is not None:
        try:
            n_cells_end = int((np.asarray(cell_end, dtype=np.float32).reshape(-1) > 0.5).sum())
        except Exception:
            n_cells_end = 0

    confluency_end = float(n_cells_end) / float(expected_len) if expected_len > 0 else 0.0
    metrics = {
        "n_cells_start": int(n_cells_start),
        "n_cells_end": int(n_cells_end),
        "confluency_end": float(confluency_end),
        "births_total": int(births_total),
        "deaths_total": int(deaths_total),
        "net_births_total": int(int(births_total) - int(deaths_total)),
    }
    return series, metrics


def _cell_culture_measurements_end_from_payload(
    base: Dict[str, Any],
    *,
    ticks: int,
    seed0: int,
    selected_names: list[str],
) -> Dict[str, Any]:
    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")
    expected_len = int(H * W)

    data = p.get("data")
    if isinstance(data, dict):
        layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
        for nm0, arr0 in layers0.items():
            ent0 = data.get(nm0)
            if not isinstance(ent0, dict) or ent0.get("dtype") != "float32":
                continue
            ent0["arr"] = np.asarray(arr0, dtype=np.float32).reshape(expected_len)
            ent0["b64"] = ""

    ticks_i = max(0, int(ticks))
    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))

    layers_end = _layers_dict_from_payload_data(p, expected_len=expected_len)
    selected = set(str(x) for x in (selected_names or []) if isinstance(x, str) and x)
    out_raw = _compute_selected_measurements_from_layers(p, layers_end, H, W, selected)
    out: Dict[str, Any] = {}
    for k, v in out_raw.items():
        if v is None:
            out[str(k)] = None
            continue
        if isinstance(v, (int, float, np.floating)):
            out[str(k)] = float(v)
            continue
        try:
            out[str(k)] = float(v)
        except Exception:
            out[str(k)] = None
    return out


def _cell_culture_measurements_end_summary_from_payload(
    base: Dict[str, Any],
    *,
    ticks: int,
    seed: int,
    replicates: int,
    selected_names: list[str],
    condition_index: int,
) -> Dict[str, Any]:
    names = [str(x) for x in (selected_names or []) if isinstance(x, str) and x]
    vals_by_name: Dict[str, list[float]] = {nm: [] for nm in names}

    for ri in range(max(0, int(replicates))):
        seed0 = int(seed) + (int(condition_index) * 1000003) + (int(ri) * 97)
        m_end = _cell_culture_measurements_end_from_payload(
            base,
            ticks=int(ticks),
            seed0=int(seed0),
            selected_names=names,
        )
        for nm in names:
            v = m_end.get(nm)
            if v is None:
                continue
            try:
                vf = float(v)
            except Exception:
                continue
            if not np.isfinite(vf):
                continue
            vals_by_name[nm].append(float(vf))

    out_meas: Dict[str, Any] = {}
    for nm in names:
        out_meas[nm] = _float_list_summary([float(x) for x in vals_by_name.get(nm) or []])

    return {
        "replicates": int(replicates),
        "ticks": int(ticks),
        "measurements_end": out_meas,
    }


def _cell_culture_measurements_end_sample_from_payload(
    base: Dict[str, Any],
    *,
    ticks: int,
    seed: int,
    replicates: int,
    selected_names: list[str],
    condition_index: int,
) -> Dict[str, Any]:
    names = [str(x) for x in (selected_names or []) if isinstance(x, str) and x]
    out_sample: list[Dict[str, Any]] = []
    for ri in range(max(0, int(replicates))):
        seed0 = int(seed) + (int(condition_index) * 1000003) + (int(ri) * 97)
        m_end = _cell_culture_measurements_end_from_payload(
            base,
            ticks=int(ticks),
            seed0=int(seed0),
            selected_names=names,
        )
        out_sample.append(
            {
                "replicate": int(ri),
                "seed": int(seed0),
                "measurements_end": dict(m_end),
            }
        )
    return {
        "replicates": int(replicates),
        "ticks": int(ticks),
        "measurements_end_sample": out_sample,
    }


def _float_list_summary(xs: list[float]) -> Dict[str, Any]:
    arr = np.asarray([float(x) for x in xs], dtype=np.float64) if xs else np.asarray([], dtype=np.float64)
    if arr.size <= 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _run_lifespan_death_tick(
    base: Dict[str, Any],
    *,
    ticks: int,
    seed0: int,
    death_names: list[str],
) -> Dict[str, Any]:
    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")

    expected_len = int(H * W)
    data = p.get("data")
    if isinstance(data, dict):
        layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
        for nm0, arr0 in layers0.items():
            ent0 = data.get(nm0)
            if not isinstance(ent0, dict) or ent0.get("dtype") != "float32":
                continue
            ent0["arr"] = np.asarray(arr0, dtype=np.float32).reshape(expected_len)
            ent0["b64"] = ""

    ticks_i = max(0, int(ticks))
    if ticks_i <= 0:
        return {"died": False, "death_tick": 0, "death_measurement": ""}

    selected = set(death_names)
    died = False
    death_tick = ticks_i
    death_measurement = ""
    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))
        layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
        sel = _compute_selected_measurements_from_layers(p, layers, H, W, selected)
        hit = ""
        if death_names:
            for nm in death_names:
                try:
                    v = float(sel.get(nm) or 0.0)
                except Exception:
                    v = 0.0
                if not np.isfinite(v):
                    v = 0.0
                if v >= 0.5:
                    hit = nm
                    break
        else:
            cell0 = layers.get("cell") if isinstance(layers, dict) else None
            if cell0 is not None:
                try:
                    carr = np.asarray(cell0, dtype=np.float32).reshape(-1)
                except Exception:
                    carr = None
                if carr is not None and carr.size > 0:
                    try:
                        alive = int(np.sum(carr > 0.5))
                    except Exception:
                        alive = -1
                    if int(alive) == 0:
                        hit = "cell_extinction"
        if hit:
            died = True
            death_tick = int(t)
            death_measurement = str(hit)
            break

    return {
        "died": bool(died),
        "death_tick": int(death_tick),
        "death_measurement": str(death_measurement),
    }


def _run_lifespan_rep(
    base: Dict[str, Any],
    *,
    ticks: int,
    seed0: int,
    death_names: list[str],
    series_names: list[str],
) -> tuple[Dict[str, Any], Optional[Dict[str, list[Optional[float]]]]]:
    if not series_names:
        return _run_lifespan_death_tick(base, ticks=int(ticks), seed0=int(seed0), death_names=death_names), None

    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")

    expected_len = int(H * W)
    data = p.get("data")
    if isinstance(data, dict):
        layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
        for nm0, arr0 in layers0.items():
            ent0 = data.get(nm0)
            if not isinstance(ent0, dict) or ent0.get("dtype") != "float32":
                continue
            ent0["arr"] = np.asarray(arr0, dtype=np.float32).reshape(expected_len)
            ent0["b64"] = ""

    ticks_i = max(0, int(ticks))
    if ticks_i <= 0:
        return {"died": False, "death_tick": 0, "death_measurement": ""}, {nm: [] for nm in series_names}

    selected = set(series_names)
    for nm in death_names:
        selected.add(nm)

    series: Dict[str, list[Optional[float]]] = {nm: [] for nm in series_names}

    died = False
    death_tick = ticks_i
    death_measurement = ""
    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))
        layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
        sel = _compute_selected_measurements_from_layers(p, layers, H, W, selected)

        for nm in series_names:
            vv = sel.get(nm)
            try:
                v = float(vv) if vv is not None else None
            except Exception:
                v = None
            if v is None or not np.isfinite(v):
                series[nm].append(None)
            else:
                series[nm].append(float(v))

        if not died:
            hit = ""
            for nm in death_names:
                try:
                    v = float(sel.get(nm) or 0.0)
                except Exception:
                    v = 0.0
                if not np.isfinite(v):
                    v = 0.0
                if v >= 0.5:
                    hit = nm
                    break
            if hit:
                died = True
                death_tick = int(t)
                death_measurement = str(hit)
                break

    if died:
        for nm in series_names:
            cur = series.get(nm)
            if not isinstance(cur, list):
                continue
            while len(cur) < ticks_i:
                cur.append(None)

    return {
        "died": bool(died),
        "death_tick": int(death_tick),
        "death_measurement": str(death_measurement),
    }, series


_LIFE_WORKER_BASE: Optional[Dict[str, Any]] = None
_LIFE_WORKER_DEATH_NAMES: Optional[list[str]] = None
_LIFE_WORKER_SERIES_NAMES: Optional[list[str]] = None


def _lifespan_worker_init(base: Dict[str, Any], death_names: list[str], series_names: list[str]) -> None:
    global _LIFE_WORKER_BASE, _LIFE_WORKER_DEATH_NAMES, _LIFE_WORKER_SERIES_NAMES
    _LIFE_WORKER_BASE = base
    _LIFE_WORKER_DEATH_NAMES = list(death_names)
    _LIFE_WORKER_SERIES_NAMES = list(series_names)


def _lifespan_worker_eval(ri: int, ticks: int, seed: int) -> tuple[int, Dict[str, Any], Optional[Dict[str, list[Optional[float]]]]]:
    if _LIFE_WORKER_BASE is None or _LIFE_WORKER_DEATH_NAMES is None or _LIFE_WORKER_SERIES_NAMES is None:
        raise ValueError("lifespan worker not initialized")
    seed0 = int(seed) + (int(ri) * 97)
    r, series = _run_lifespan_rep(
        _LIFE_WORKER_BASE,
        ticks=int(ticks),
        seed0=int(seed0),
        death_names=_LIFE_WORKER_DEATH_NAMES,
        series_names=_LIFE_WORKER_SERIES_NAMES,
    )
    r["seed0"] = int(seed0)
    return int(ri), r, series


_BULK_OMICS_WORKER_BASE: Optional[Dict[str, Any]] = None
_BULK_OMICS_WORKER_FEATURES: Optional[list[str]] = None
_BULK_OMICS_WORKER_TICKS: Optional[int] = None
_BULK_OMICS_WORKER_SEED: Optional[int] = None


def _bulk_omics_worker_init(base: Dict[str, Any], features: list[str], ticks: int, seed: int) -> None:
    global _BULK_OMICS_WORKER_BASE
    global _BULK_OMICS_WORKER_FEATURES
    global _BULK_OMICS_WORKER_TICKS
    global _BULK_OMICS_WORKER_SEED
    _BULK_OMICS_WORKER_BASE = base
    _BULK_OMICS_WORKER_FEATURES = [str(x) for x in (features or []) if isinstance(x, str) and x]
    _BULK_OMICS_WORKER_TICKS = int(ticks)
    _BULK_OMICS_WORKER_SEED = int(seed)


def _bulk_omics_worker_eval(ri: int) -> tuple[int, int, Optional[Dict[str, Any]], Optional[list[float]], int, int]:
    if (
        _BULK_OMICS_WORKER_BASE is None
        or _BULK_OMICS_WORKER_FEATURES is None
        or _BULK_OMICS_WORKER_TICKS is None
        or _BULK_OMICS_WORKER_SEED is None
    ):
        raise ValueError("bulk omics worker not initialized")

    ticks_i = int(_BULK_OMICS_WORKER_TICKS)
    seed_i = int(_BULK_OMICS_WORKER_SEED)
    seed0 = int(seed_i) + (int(ri) * 97)

    pf = _preflight_death_before_ticks(_BULK_OMICS_WORKER_BASE, ticks=int(ticks_i), seed0=int(seed0))
    if isinstance(pf, dict):
        return int(ri), int(seed0), dict(pf), None, 0, 0

    p = _run_payload_ticks(_BULK_OMICS_WORKER_BASE, ticks=int(ticks_i), seed0=int(seed0))
    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")
    expected_len = int(H * W)
    layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
    if not layers:
        raise ValueError("payload has no float32 layers")

    vv: list[float] = []
    for ln in _BULK_OMICS_WORKER_FEATURES:
        arr = layers.get(ln)
        if arr is None:
            vv.append(0.0)
            continue
        try:
            s = float(np.asarray(arr, dtype=np.float64).reshape(-1).sum())
        except Exception:
            s = 0.0
        if not np.isfinite(s) or s < 0.0:
            s = 0.0
        vv.append(float(s))

    return int(ri), int(seed0), None, vv, int(H), int(W)


_STX_TESTS_WORKER_BASE: Optional[Dict[str, Any]] = None
_STX_TESTS_WORKER_GENES: Optional[list[str]] = None
_STX_TESTS_WORKER_TICKS: Optional[int] = None
_STX_TESTS_WORKER_SEED: Optional[int] = None


def _stx_tests_worker_init(base: Dict[str, Any], genes: list[str], ticks: int, seed: int) -> None:
    global _STX_TESTS_WORKER_BASE
    global _STX_TESTS_WORKER_GENES
    global _STX_TESTS_WORKER_TICKS
    global _STX_TESTS_WORKER_SEED
    _STX_TESTS_WORKER_BASE = base
    _STX_TESTS_WORKER_GENES = [str(x) for x in (genes or []) if isinstance(x, str) and x]
    _STX_TESTS_WORKER_TICKS = int(ticks)
    _STX_TESTS_WORKER_SEED = int(seed)


def _stx_tests_worker_eval(ri: int) -> tuple[int, int, Optional[Dict[str, Any]], int, int, list[Dict[str, Any]]]:
    if (
        _STX_TESTS_WORKER_BASE is None
        or _STX_TESTS_WORKER_GENES is None
        or _STX_TESTS_WORKER_TICKS is None
        or _STX_TESTS_WORKER_SEED is None
    ):
        raise ValueError("spatial_tx worker not initialized")

    ticks_i = int(_STX_TESTS_WORKER_TICKS)
    seed_i = int(_STX_TESTS_WORKER_SEED)
    seed0 = int(seed_i) + (int(ri) * 97)

    pf = _preflight_death_before_ticks(_STX_TESTS_WORKER_BASE, ticks=int(ticks_i), seed0=int(seed0))
    if isinstance(pf, dict):
        return int(ri), int(seed0), dict(pf), 0, 0, []

    p = _run_payload_ticks(_STX_TESTS_WORKER_BASE, ticks=int(ticks_i), seed0=int(seed0))
    tx = _spatial_tx_rows(
        p,
        _STX_TESTS_WORKER_GENES,
        cell_layer="",
        min_cell_value=0.0,
        stride=1,
        max_spots=None,
        seed=int(seed0),
    )
    H = int(tx.get("H") or 0)
    W = int(tx.get("W") or 0)
    rows = tx.get("rows")
    if not isinstance(rows, list):
        rows = []
    rows_out: list[Dict[str, Any]] = [r for r in rows if isinstance(r, dict)]
    return int(ri), int(seed0), None, int(H), int(W), rows_out


_CHAR_WORKER_BASE: Optional[Dict[str, Any]] = None
_CHAR_WORKER_TICKS: Optional[int] = None
_CHAR_WORKER_SEED: Optional[int] = None
_CHAR_WORKER_NAMES: Optional[list[str]] = None
_CHAR_WORKER_DEATH_NAMES: Optional[list[str]] = None
_CHAR_WORKER_MODE: Optional[str] = None


def _char_worker_init(
    base: Dict[str, Any],
    ticks: int,
    seed: int,
    names: list[str],
    death_names: list[str],
    mode: str,
) -> None:
    global _CHAR_WORKER_BASE
    global _CHAR_WORKER_TICKS
    global _CHAR_WORKER_SEED
    global _CHAR_WORKER_NAMES
    global _CHAR_WORKER_DEATH_NAMES
    global _CHAR_WORKER_MODE
    _CHAR_WORKER_BASE = base
    _CHAR_WORKER_TICKS = int(ticks)
    _CHAR_WORKER_SEED = int(seed)
    _CHAR_WORKER_NAMES = [str(x) for x in (names or []) if isinstance(x, str) and x]
    _CHAR_WORKER_DEATH_NAMES = [str(x) for x in (death_names or []) if isinstance(x, str) and x]
    _CHAR_WORKER_MODE = str(mode or "").strip().lower()


def _char_worker_eval(ri: int) -> tuple[int, int, Dict[str, list[float]], int, str]:
    if (
        _CHAR_WORKER_BASE is None
        or _CHAR_WORKER_TICKS is None
        or _CHAR_WORKER_SEED is None
        or _CHAR_WORKER_NAMES is None
        or _CHAR_WORKER_DEATH_NAMES is None
        or _CHAR_WORKER_MODE is None
    ):
        raise ValueError("characterization worker not initialized")

    ticks_i = int(_CHAR_WORKER_TICKS)
    seed_i = int(_CHAR_WORKER_SEED)
    seed0 = int(seed_i) + (int(ri) * 97)

    if str(_CHAR_WORKER_MODE) == "invivo":
        s0, dt0, dm0 = _run_in_vivo_measurement_series_until_death(
            _CHAR_WORKER_BASE,
            ticks=int(ticks_i),
            seed0=int(seed0),
            death_names=_CHAR_WORKER_DEATH_NAMES,
        )
        return int(ri), int(seed0), s0, int(dt0), str(dm0)

    s0, _m0 = _run_cell_culture_measurement_series_and_metrics(
        _CHAR_WORKER_BASE,
        ticks=int(ticks_i),
        seed0=int(seed0),
        names=_CHAR_WORKER_NAMES,
    )
    return int(ri), int(seed0), s0, int(ticks_i), ""


_AGING_CLAIM_WORKER_HEALTHY_BASE: Optional[Dict[str, Any]] = None
_AGING_CLAIM_WORKER_HEALTHY_TREATED: Optional[Dict[str, Any]] = None
_AGING_CLAIM_WORKER_DEATH_NAMES: Optional[list[str]] = None
_AGING_CLAIM_WORKER_TICKS: Optional[int] = None
_AGING_CLAIM_WORKER_SEED: Optional[int] = None


def _aging_claim_worker_init(
    healthy_base: Dict[str, Any],
    healthy_treated: Dict[str, Any],
    death_names: list[str],
    ticks: int,
    seed: int,
) -> None:
    global _AGING_CLAIM_WORKER_HEALTHY_BASE
    global _AGING_CLAIM_WORKER_HEALTHY_TREATED
    global _AGING_CLAIM_WORKER_DEATH_NAMES
    global _AGING_CLAIM_WORKER_TICKS
    global _AGING_CLAIM_WORKER_SEED
    _AGING_CLAIM_WORKER_HEALTHY_BASE = healthy_base
    _AGING_CLAIM_WORKER_HEALTHY_TREATED = healthy_treated
    _AGING_CLAIM_WORKER_DEATH_NAMES = list(death_names)
    _AGING_CLAIM_WORKER_TICKS = int(ticks)
    _AGING_CLAIM_WORKER_SEED = int(seed)


def _aging_claim_worker_eval(ri: int) -> tuple[int, int, str, int, str]:
    if (
        _AGING_CLAIM_WORKER_HEALTHY_BASE is None
        or _AGING_CLAIM_WORKER_HEALTHY_TREATED is None
        or _AGING_CLAIM_WORKER_DEATH_NAMES is None
        or _AGING_CLAIM_WORKER_TICKS is None
        or _AGING_CLAIM_WORKER_SEED is None
    ):
        raise ValueError("aging claim worker not initialized")

    ticks_i = int(_AGING_CLAIM_WORKER_TICKS)
    seed_i = int(_AGING_CLAIM_WORKER_SEED)
    seed0_b = int(seed_i) + (0 * 1000003) + (int(ri) * 97)
    seed0_t = int(seed_i) + (1 * 1000003) + (int(ri) * 97)
    rb = _run_lifespan_death_tick(
        _AGING_CLAIM_WORKER_HEALTHY_BASE,
        ticks=int(ticks_i),
        seed0=int(seed0_b),
        death_names=_AGING_CLAIM_WORKER_DEATH_NAMES,
    )
    rt = _run_lifespan_death_tick(
        _AGING_CLAIM_WORKER_HEALTHY_TREATED,
        ticks=int(ticks_i),
        seed0=int(seed0_t),
        death_names=_AGING_CLAIM_WORKER_DEATH_NAMES,
    )
    dt_b = int(rb.get("death_tick")) if isinstance(rb, dict) else int(ticks_i)
    dt_t = int(rt.get("death_tick")) if isinstance(rt, dict) else int(ticks_i)
    dm_b = str(rb.get("death_measurement") or "") if isinstance(rb, dict) else ""
    dm_t = str(rt.get("death_measurement") or "") if isinstance(rt, dict) else ""
    return int(ri), int(dt_b), str(dm_b), int(dt_t), str(dm_t)


_DISEASE_CLAIM_WORKER_HEALTHY: Optional[Dict[str, Any]] = None
_DISEASE_CLAIM_WORKER_SICK_BASE: Optional[Dict[str, Any]] = None
_DISEASE_CLAIM_WORKER_SICK_TREATED: Optional[Dict[str, Any]] = None
_DISEASE_CLAIM_WORKER_DEATH_NAMES: Optional[list[str]] = None
_DISEASE_CLAIM_WORKER_TICKS: Optional[int] = None
_DISEASE_CLAIM_WORKER_SEED: Optional[int] = None
_DISEASE_CLAIM_WORKER_RUN_TREATED: Optional[bool] = None


def _disease_claim_worker_init(
    healthy: Dict[str, Any],
    sick_base: Dict[str, Any],
    sick_treated: Dict[str, Any],
    death_names: list[str],
    ticks: int,
    seed: int,
    run_treated: bool,
) -> None:
    global _DISEASE_CLAIM_WORKER_HEALTHY
    global _DISEASE_CLAIM_WORKER_SICK_BASE
    global _DISEASE_CLAIM_WORKER_SICK_TREATED
    global _DISEASE_CLAIM_WORKER_DEATH_NAMES
    global _DISEASE_CLAIM_WORKER_TICKS
    global _DISEASE_CLAIM_WORKER_SEED
    global _DISEASE_CLAIM_WORKER_RUN_TREATED
    _DISEASE_CLAIM_WORKER_HEALTHY = healthy
    _DISEASE_CLAIM_WORKER_SICK_BASE = sick_base
    _DISEASE_CLAIM_WORKER_SICK_TREATED = sick_treated
    _DISEASE_CLAIM_WORKER_DEATH_NAMES = list(death_names)
    _DISEASE_CLAIM_WORKER_TICKS = int(ticks)
    _DISEASE_CLAIM_WORKER_SEED = int(seed)
    _DISEASE_CLAIM_WORKER_RUN_TREATED = bool(run_treated)


def _disease_claim_worker_eval(ri: int) -> tuple[int, int, str, int, str, int, str]:
    if (
        _DISEASE_CLAIM_WORKER_HEALTHY is None
        or _DISEASE_CLAIM_WORKER_SICK_BASE is None
        or _DISEASE_CLAIM_WORKER_SICK_TREATED is None
        or _DISEASE_CLAIM_WORKER_DEATH_NAMES is None
        or _DISEASE_CLAIM_WORKER_TICKS is None
        or _DISEASE_CLAIM_WORKER_SEED is None
        or _DISEASE_CLAIM_WORKER_RUN_TREATED is None
    ):
        raise ValueError("disease claim worker not initialized")

    ticks_i = int(_DISEASE_CLAIM_WORKER_TICKS)
    seed_i = int(_DISEASE_CLAIM_WORKER_SEED)
    seed0_h = int(seed_i) + (0 * 1000003) + (int(ri) * 97)
    seed0_s = int(seed_i) + (1 * 1000003) + (int(ri) * 97)

    rh = _run_lifespan_death_tick(
        _DISEASE_CLAIM_WORKER_HEALTHY,
        ticks=int(ticks_i),
        seed0=int(seed0_h),
        death_names=_DISEASE_CLAIM_WORKER_DEATH_NAMES,
    )
    rs0 = _run_lifespan_death_tick(
        _DISEASE_CLAIM_WORKER_SICK_BASE,
        ticks=int(ticks_i),
        seed0=int(seed0_s),
        death_names=_DISEASE_CLAIM_WORKER_DEATH_NAMES,
    )
    if bool(_DISEASE_CLAIM_WORKER_RUN_TREATED):
        rs = _run_lifespan_death_tick(
            _DISEASE_CLAIM_WORKER_SICK_TREATED,
            ticks=int(ticks_i),
            seed0=int(seed0_s),
            death_names=_DISEASE_CLAIM_WORKER_DEATH_NAMES,
        )
    else:
        rs = rs0

    dt_h = int(rh.get("death_tick")) if isinstance(rh, dict) else int(ticks_i)
    dt_s0 = int(rs0.get("death_tick")) if isinstance(rs0, dict) else int(ticks_i)
    dt_s = int(rs.get("death_tick")) if isinstance(rs, dict) else int(ticks_i)
    dm_h = str(rh.get("death_measurement") or "") if isinstance(rh, dict) else ""
    dm_s0 = str(rs0.get("death_measurement") or "") if isinstance(rs0, dict) else ""
    dm_s = str(rs.get("death_measurement") or "") if isinstance(rs, dict) else ""
    return int(ri), int(dt_h), str(dm_h), int(dt_s0), str(dm_s0), int(dt_s), str(dm_s)


_INVIVO_WORKER_HEALTHY_BASE: Optional[Dict[str, Any]] = None
_INVIVO_WORKER_SICK_BASE: Optional[Dict[str, Any]] = None


def _invivo_worker_init(healthy: Dict[str, Any], sick: Dict[str, Any]) -> None:
    global _INVIVO_WORKER_HEALTHY_BASE, _INVIVO_WORKER_SICK_BASE
    _INVIVO_WORKER_HEALTHY_BASE = healthy
    _INVIVO_WORKER_SICK_BASE = sick


_INVIVO_WORKER_DEATH_NAMES: Optional[list[str]] = None


def _invivo_worker_init_v2(healthy: Dict[str, Any], sick: Dict[str, Any], death_names: list[str]) -> None:
    global _INVIVO_WORKER_HEALTHY_BASE, _INVIVO_WORKER_SICK_BASE, _INVIVO_WORKER_DEATH_NAMES
    _INVIVO_WORKER_HEALTHY_BASE = healthy
    _INVIVO_WORKER_SICK_BASE = sick
    _INVIVO_WORKER_DEATH_NAMES = list(death_names)


def _invivo_worker_eval(
    ri: int, ticks: int, seed: int
) -> tuple[int, Dict[str, list[float]], Dict[str, list[float]], int, str, int, str]:
    if _INVIVO_WORKER_HEALTHY_BASE is None or _INVIVO_WORKER_SICK_BASE is None:
        raise ValueError("in vivo worker not initialized")
    dn = _INVIVO_WORKER_DEATH_NAMES
    if dn is None:
        dn = []
    seed0_h = int(seed) + (0 * 1000003) + (int(ri) * 97)
    seed0_s = int(seed) + (1 * 1000003) + (int(ri) * 97)
    sh, dt_h, dm_h = _run_in_vivo_measurement_series_until_death(
        _INVIVO_WORKER_HEALTHY_BASE, ticks=int(ticks), seed0=int(seed0_h), death_names=dn
    )
    ss, dt_s, dm_s = _run_in_vivo_measurement_series_until_death(
        _INVIVO_WORKER_SICK_BASE, ticks=int(ticks), seed0=int(seed0_s), death_names=dn
    )
    return int(ri), sh, ss, int(dt_h), str(dm_h), int(dt_s), str(dm_s)


def _invivo_worker_death_eval(cond: int, ri: int, ticks: int, seed: int) -> tuple[int, int, int, str]:
    if _INVIVO_WORKER_HEALTHY_BASE is None or _INVIVO_WORKER_SICK_BASE is None:
        raise ValueError("in vivo worker not initialized")
    dn = _INVIVO_WORKER_DEATH_NAMES
    if dn is None:
        dn = []
    ci = 0 if int(cond) == 0 else 1
    base = _INVIVO_WORKER_HEALTHY_BASE if ci == 0 else _INVIVO_WORKER_SICK_BASE
    seed0 = int(seed) + (int(ci) * 1000003) + (int(ri) * 97)
    r = _run_lifespan_death_tick(base, ticks=int(ticks), seed0=int(seed0), death_names=dn)
    try:
        dt = int(r.get("death_tick"))
    except Exception:
        dt = int(ticks)
    dm = str(r.get("death_measurement") or "")
    return int(ci), int(ri), int(dt), str(dm)


def _protein_layer_names_from_payload(payload: Dict[str, Any]) -> list[str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    for k, ent in data.items():
        if not isinstance(k, str) or not k.startswith("protein_"):
            continue
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str) or not b64:
            continue
        out.append(k)
    out.sort(key=lambda s: s.lower())
    return out


_INVIVO_SCREEN_WORKER_BASE: Optional[Dict[str, Any]] = None
_INVIVO_SCREEN_WORKER_DEATH_NAMES: Optional[list[str]] = None
_INVIVO_SCREEN_WORKER_TICKS: Optional[int] = None
_INVIVO_SCREEN_WORKER_SEED: Optional[int] = None
_INVIVO_SCREEN_WORKER_REPS: Optional[int] = None
_INVIVO_SCREEN_WORKER_DIRECTION: Optional[str] = None
_INVIVO_SCREEN_WORKER_DOSE: Optional[float] = None

_VITRO_SCREEN_WORKER_BASE: Optional[Dict[str, Any]] = None
_VITRO_SCREEN_WORKER_TICKS: Optional[int] = None
_VITRO_SCREEN_WORKER_SEED: Optional[int] = None
_VITRO_SCREEN_WORKER_REPS: Optional[int] = None
_VITRO_SCREEN_WORKER_DIRECTION: Optional[str] = None
_VITRO_SCREEN_WORKER_DOSE: Optional[float] = None
_VITRO_SCREEN_WORKER_MEAS_NAMES: Optional[list[str]] = None


def _invivo_screen_worker_init(
    base: Dict[str, Any],
    death_names: list[str],
    ticks: int,
    seed: int,
    replicates: int,
    direction: str,
    dose: float,
) -> None:
    global _INVIVO_SCREEN_WORKER_BASE
    global _INVIVO_SCREEN_WORKER_DEATH_NAMES
    global _INVIVO_SCREEN_WORKER_TICKS
    global _INVIVO_SCREEN_WORKER_SEED
    global _INVIVO_SCREEN_WORKER_REPS
    global _INVIVO_SCREEN_WORKER_DIRECTION
    global _INVIVO_SCREEN_WORKER_DOSE

    _INVIVO_SCREEN_WORKER_BASE = base
    _INVIVO_SCREEN_WORKER_DEATH_NAMES = list(death_names)
    _INVIVO_SCREEN_WORKER_TICKS = int(ticks)
    _INVIVO_SCREEN_WORKER_SEED = int(seed)
    _INVIVO_SCREEN_WORKER_REPS = int(replicates)
    _INVIVO_SCREEN_WORKER_DIRECTION = str(direction)
    _INVIVO_SCREEN_WORKER_DOSE = float(dose)


def _vitro_screen_worker_init(
    base: Dict[str, Any],
    ticks: int,
    seed: int,
    replicates: int,
    direction: str,
    dose: float,
    meas_names: list[str],
) -> None:
    global _VITRO_SCREEN_WORKER_BASE
    global _VITRO_SCREEN_WORKER_TICKS
    global _VITRO_SCREEN_WORKER_SEED
    global _VITRO_SCREEN_WORKER_REPS
    global _VITRO_SCREEN_WORKER_DIRECTION
    global _VITRO_SCREEN_WORKER_DOSE
    global _VITRO_SCREEN_WORKER_MEAS_NAMES

    _VITRO_SCREEN_WORKER_BASE = base
    _VITRO_SCREEN_WORKER_TICKS = int(ticks)
    _VITRO_SCREEN_WORKER_SEED = int(seed)
    _VITRO_SCREEN_WORKER_REPS = int(replicates)
    _VITRO_SCREEN_WORKER_DIRECTION = str(direction)
    _VITRO_SCREEN_WORKER_DOSE = float(dose)
    _VITRO_SCREEN_WORKER_MEAS_NAMES = [str(x) for x in (meas_names or []) if isinstance(x, str) and x]


def _vitro_screen_worker_eval(layer_index: int, layer_name: str) -> Dict[str, Any]:
    if (
        _VITRO_SCREEN_WORKER_BASE is None
        or _VITRO_SCREEN_WORKER_TICKS is None
        or _VITRO_SCREEN_WORKER_SEED is None
        or _VITRO_SCREEN_WORKER_REPS is None
        or _VITRO_SCREEN_WORKER_DIRECTION is None
        or _VITRO_SCREEN_WORKER_DOSE is None
        or _VITRO_SCREEN_WORKER_MEAS_NAMES is None
    ):
        raise ValueError("in vitro screen worker not initialized")

    base = _VITRO_SCREEN_WORKER_BASE
    ticks_i = int(_VITRO_SCREEN_WORKER_TICKS)
    seed_i = int(_VITRO_SCREEN_WORKER_SEED)
    reps_i = int(_VITRO_SCREEN_WORKER_REPS)
    direction = str(_VITRO_SCREEN_WORKER_DIRECTION)
    dose = float(_VITRO_SCREEN_WORKER_DOSE)
    names = [str(x) for x in (_VITRO_SCREEN_WORKER_MEAS_NAMES or []) if isinstance(x, str) and x]

    p = _deepcopy_payload(base)
    layer = str(layer_name or "").strip()
    if layer:
        tiv0 = p.get("_tick_interventions")
        tiv = list(tiv0) if isinstance(tiv0, list) else []
        tiv.append({"layer": layer, "direction": direction, "dose": dose})
        p["_tick_interventions"] = tiv

    out0 = _cell_culture_measurements_end_sample_from_payload(
        p,
        ticks=int(ticks_i),
        seed=int(seed_i),
        replicates=int(reps_i),
        selected_names=names,
        condition_index=int(layer_index + 2),
    )
    return {
        "layer": str(layer),
        "layer_index": int(layer_index),
        "measurements_end_sample": out0.get("measurements_end_sample"),
    }


def _invivo_screen_worker_eval(layer_index: int, layer_name: str) -> Dict[str, Any]:
    if (
        _INVIVO_SCREEN_WORKER_BASE is None
        or _INVIVO_SCREEN_WORKER_DEATH_NAMES is None
        or _INVIVO_SCREEN_WORKER_TICKS is None
        or _INVIVO_SCREEN_WORKER_SEED is None
        or _INVIVO_SCREEN_WORKER_REPS is None
        or _INVIVO_SCREEN_WORKER_DIRECTION is None
        or _INVIVO_SCREEN_WORKER_DOSE is None
    ):
        raise ValueError("in vivo screen worker not initialized")

    base = _INVIVO_SCREEN_WORKER_BASE
    death_names = _INVIVO_SCREEN_WORKER_DEATH_NAMES
    ticks_i = int(_INVIVO_SCREEN_WORKER_TICKS)
    seed_i = int(_INVIVO_SCREEN_WORKER_SEED)
    reps_i = int(_INVIVO_SCREEN_WORKER_REPS)
    direction = str(_INVIVO_SCREEN_WORKER_DIRECTION)
    dose = float(_INVIVO_SCREEN_WORKER_DOSE)

    p = _deepcopy_payload(base)
    layer = str(layer_name or "").strip()
    if layer:
        tiv0 = p.get("_tick_interventions")
        tiv = list(tiv0) if isinstance(tiv0, list) else []
        tiv.append({"layer": layer, "direction": direction, "dose": dose})
        p["_tick_interventions"] = tiv

    death_ticks: list[int] = []
    for ri in range(int(reps_i)):
        seed0 = int(seed_i) + (int(layer_index + 1) * 1000003) + (int(ri) * 97)
        r = _run_lifespan_death_tick(p, ticks=int(ticks_i), seed0=int(seed0), death_names=death_names)
        try:
            dt = int(r.get("death_tick"))
        except Exception:
            dt = int(ticks_i)
        death_ticks.append(int(dt))

    arr = np.asarray(death_ticks, dtype=np.float64)
    median_tick = float(np.median(arr)) if arr.size else float(ticks_i)
    mean_tick = float(np.mean(arr)) if arr.size else float(ticks_i)
    try:
        p25_tick = float(np.quantile(arr, 0.25)) if arr.size else float(ticks_i)
    except Exception:
        p25_tick = float(ticks_i)
    try:
        p75_tick = float(np.quantile(arr, 0.75)) if arr.size else float(ticks_i)
    except Exception:
        p75_tick = float(ticks_i)
    min_tick = float(np.min(arr)) if arr.size else float(ticks_i)
    max_tick = float(np.max(arr)) if arr.size else float(ticks_i)
    deaths = int(sum(1 for dt in death_ticks if int(dt) < int(ticks_i)))
    return {
        "layer": str(layer),
        "layer_index": int(layer_index),
        "n": int(len(death_ticks)),
        "ticks": int(ticks_i),
        "median_lifespan_tick": float(median_tick),
        "mean_lifespan_tick": float(mean_tick),
        "p25_lifespan_tick": float(p25_tick),
        "p75_lifespan_tick": float(p75_tick),
        "min_lifespan_tick": float(min_tick),
        "max_lifespan_tick": float(max_tick),
        "deaths": int(deaths),
        "survivors": int(len(death_ticks) - deaths),
    }


def _lifespan_survival_curve(death_ticks: list[int], *, ticks: int) -> Dict[str, Any]:
    ticks_i = max(0, int(ticks))
    dts: list[int] = []
    for dt in death_ticks:
        try:
            dts.append(int(dt))
        except Exception:
            dts.append(int(ticks_i))

    n = int(len(dts))
    surv = 1.0
    times: list[int] = [0]
    survival: list[float] = [1.0]
    for t in range(ticks_i):
        at_risk = 0
        deaths = 0
        for dt in dts:
            if int(dt) >= int(t):
                at_risk += 1
            if int(dt) == int(t):
                deaths += 1
        if at_risk > 0 and deaths > 0:
            surv *= float(1.0 - (float(deaths) / float(at_risk)))
        times.append(int(t + 1))
        survival.append(float(surv))

    median_tick = None
    for i in range(len(times)):
        if float(survival[i]) <= 0.5:
            median_tick = int(times[i])
            break

    deaths_total = int(sum(1 for dt in dts if int(dt) < int(ticks_i)))
    return {
        "n": int(n),
        "deaths": int(deaths_total),
        "survivors": int(n - deaths_total),
        "times": times,
        "survival": survival,
        "median_tick": median_tick,
    }


def _run_in_vivo_measurement_series(base: Dict[str, Any], ticks: int, seed0: int) -> Dict[str, list[float]]:
    p = _deepcopy_payload(base)
    p.pop("event_counters", None)
    p.pop("_profile_layer_ops", None)
    p.pop("_profile_step_names", None)
    p.pop("_profile_expr", None)
    p["_skip_b64_writeback"] = True
    _ensure_layer_ops_opts(p)

    H = int(p.get("H") or 0)
    W = int(p.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")

    expected_len = int(H * W)
    data = p.get("data")
    if isinstance(data, dict):
        layers0 = _layers_dict_from_payload_data(p, expected_len=expected_len)
        for nm0, arr0 in layers0.items():
            ent0 = data.get(nm0)
            if not isinstance(ent0, dict) or ent0.get("dtype") != "float32":
                continue
            ent0["arr"] = np.asarray(arr0, dtype=np.float32).reshape(expected_len)
            ent0["b64"] = ""

    names = _measurement_names_from_payload(p)
    if not names:
        return {}
    selected = set(names)

    out: Dict[str, list[float]] = {nm: [] for nm in names}
    ticks_i = max(0, int(ticks))

    for t in range(ticks_i):
        apply_layer_ops_inplace(p, seed_offset=int(seed0) + int(t))
        layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
        sel = _compute_selected_measurements_from_layers(p, layers, H, W, selected)
        for nm in names:
            try:
                v = float(sel.get(nm) or 0.0)
            except Exception:
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            out[nm].append(float(v))

    return out


def _measurement_names_union(healthy: Dict[str, Any], sick: Dict[str, Any]) -> list[str]:
    a = _measurement_names_from_payload(healthy)
    b = _measurement_names_from_payload(sick)
    if not a:
        return b
    if not b:
        return a
    seen = set(a)
    out = list(a)
    for nm in b:
        if nm in seen:
            continue
        out.append(nm)
        seen.add(nm)
    return out


def _mean_measurement_series(series_list: list[Dict[str, list[float]]], ticks: int, names: list[str]) -> Dict[str, list[float]]:
    ticks_i = max(0, int(ticks))
    out: Dict[str, list[float]] = {nm: [0.0] * ticks_i for nm in names}
    if not series_list or ticks_i <= 0:
        return out

    denom = float(len(series_list))
    for nm in names:
        acc = np.zeros((ticks_i,), dtype=np.float64)
        for s in series_list:
            vv = s.get(nm) if isinstance(s, dict) else None
            if not isinstance(vv, list) or len(vv) != ticks_i:
                continue
            try:
                arr = np.asarray([float(x) for x in vv], dtype=np.float64)
            except Exception:
                continue
            if arr.shape != (ticks_i,):
                continue
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            acc += arr
        out[nm] = (acc / denom).astype(np.float64).tolist()
    return out


def _invivo_cure_score(
    healthy_series: Dict[str, list[float]],
    sick_series: Dict[str, list[float]],
    *,
    names: list[str],
    ticks: int,
    win_threshold: float = 0.95,
) -> Dict[str, Any]:
    ticks_i = max(0, int(ticks))
    if ticks_i <= 0:
        return {
            "score": 0.0,
            "score_pct": 0.0,
            "distance": 0.0,
            "win": False,
            "win_threshold": float(win_threshold),
        }

    err_acc = 0.0
    base_acc = 0.0
    n = 0
    for nm in names:
        hv = healthy_series.get(nm)
        sv = sick_series.get(nm)
        if not isinstance(hv, list) or not isinstance(sv, list) or len(hv) != ticks_i or len(sv) != ticks_i:
            continue
        for i in range(ticks_i):
            try:
                a = float(hv[i])
            except Exception:
                a = float("nan")
            try:
                b = float(sv[i])
            except Exception:
                b = float("nan")
            if not np.isfinite(a) or not np.isfinite(b):
                continue
            err_acc += abs(a - b)
            base_acc += abs(a)
            n += 1

    if n <= 0:
        return {
            "score": 0.0,
            "score_pct": 0.0,
            "distance": 0.0,
            "win": False,
            "win_threshold": float(win_threshold),
        }

    mean_err = float(err_acc / float(n))
    mean_base = float(base_acc / float(n))
    scale = max(mean_base, 1e-6)
    distance = float(mean_err / scale)
    score = float(1.0 / (1.0 + max(0.0, distance)))
    score_pct = float(100.0 * score)
    win = bool(score >= float(win_threshold))
    return {
        "score": score,
        "score_pct": score_pct,
        "distance": distance,
        "win": win,
        "win_threshold": float(win_threshold),
    }


def _spatial_tx_rows(
    payload: Dict[str, Any],
    layer_names: list[str],
    *,
    cell_layer: str = "cell",
    min_cell_value: float = 0.5,
    stride: int = 1,
    max_spots: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    H = int(payload.get("H") or 0)
    W = int(payload.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("payload invalid H/W")

    expected_len = int(H * W)
    layers = _layers_dict_from_payload_data(payload, expected_len=expected_len)
    if not layers:
        raise ValueError("payload has no float32 layers")

    if not layer_names:
        layer_names = sorted(list(layers.keys()))[:8]
    else:
        layer_names = [str(x).strip() for x in layer_names if isinstance(x, str) and str(x).strip()]

    stride_i = max(1, int(stride))
    xs = np.arange(0, W, stride_i, dtype=np.int64)
    ys = np.arange(0, H, stride_i, dtype=np.int64)
    if xs.size == 0 or ys.size == 0:
        return {"H": H, "W": W, "layers": layer_names, "rows": []}

    gx, gy = np.meshgrid(xs, ys)
    x_flat = gx.reshape(-1)
    y_flat = gy.reshape(-1)
    idx = (y_flat * int(W) + x_flat).astype(np.int64)

    if isinstance(cell_layer, str) and cell_layer and cell_layer in layers:
        cell_arr = np.asarray(layers[cell_layer], dtype=np.float32).reshape(-1)
        keep = cell_arr[idx] > float(min_cell_value)
        if np.any(keep):
            x_flat = x_flat[keep]
            y_flat = y_flat[keep]
            idx = idx[keep]

    n = int(idx.size)
    if max_spots is not None:
        max_i = max(1, int(max_spots))
        if n > max_i:
            rng = np.random.default_rng(int(seed))
            pick = rng.choice(n, size=max_i, replace=False)
            x_flat = x_flat[pick]
            y_flat = y_flat[pick]
            idx = idx[pick]

    rows: list[Dict[str, Any]] = []
    for j in range(int(idx.size)):
        ii = int(idx[j])
        row: Dict[str, Any] = {"x": int(x_flat[j]), "y": int(y_flat[j])}
        for ln in layer_names:
            arr = layers.get(ln)
            if arr is None:
                row[ln] = None
                continue
            try:
                row[ln] = float(np.asarray(arr, dtype=np.float32).reshape(-1)[ii])
            except Exception:
                row[ln] = None
        rows.append(row)

    return {"H": H, "W": W, "layers": layer_names, "rows": rows}


def _default_stx_gene_list(payload: Dict[str, Any], max_genes: int = 8) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    float_layers = [nm for nm, ent in data.items() if isinstance(nm, str) and isinstance(ent, dict) and ent.get("dtype") == "float32"]
    if not float_layers:
        return []

    prefer = []
    layer_meta = payload.get("layers")
    if isinstance(layer_meta, list):
        for m in layer_meta:
            if not isinstance(m, dict):
                continue
            nm = m.get("name")
            if isinstance(nm, str) and nm in data:
                prefer.append(nm)
    if not prefer:
        prefer = sorted(float_layers)

    out: list[str] = []
    seen = set()
    for nm in prefer:
        if nm in seen:
            continue
        if nm not in float_layers:
            continue
        out.append(nm)
        seen.add(nm)
        if len(out) >= int(max_genes):
            break
    return out


def _stx_values_to_counts(
    values: list[Any],
    *,
    rng: Any,
    target_depth: float = 1000.0,
    depth_sigma: float = 0.0,
    mu_noise_sigma: float = 0.0,
    dropout_p: float = 0.0,
    dropout_mu_scale: float = 0.0,
    model: str = "poisson",
    nb_theta: float = 10.0,
) -> list[int]:
    g = int(len(values))
    if g <= 0:
        return []
    v = np.zeros(g, dtype=np.float64)
    for i, x in enumerate(values):
        try:
            f = float(x)
        except Exception:
            f = 0.0
        if not np.isfinite(f) or f < 0.0:
            f = 0.0
        v[int(i)] = float(f)

    s = float(v.sum())
    if not np.isfinite(s) or s <= 0.0:
        return [0 for _ in range(g)]

    frac = v / float(s)

    try:
        depth = float(target_depth)
    except Exception:
        depth = 0.0
    if not np.isfinite(depth) or depth < 0.0:
        depth = 0.0

    try:
        ds = float(depth_sigma)
    except Exception:
        ds = 0.0
    if np.isfinite(ds) and ds > 0.0:
        depth = float(depth) * float(rng.lognormal(mean=0.0, sigma=float(ds)))

    mu = frac * float(depth)

    try:
        ms = float(mu_noise_sigma)
    except Exception:
        ms = 0.0
    if np.isfinite(ms) and ms > 0.0:
        mu = mu * rng.lognormal(mean=0.0, sigma=float(ms), size=mu.shape)

    m = str(model or "").strip().lower()
    if m in ("nb", "negbin", "negative_binomial"):
        try:
            theta = float(nb_theta)
        except Exception:
            theta = 10.0
        if not np.isfinite(theta) or theta <= 0.0:
            theta = 1.0
        lam = rng.gamma(shape=float(theta), scale=(mu / float(theta)))
        counts = rng.poisson(lam)
    else:
        counts = rng.poisson(mu)

    try:
        dp = float(dropout_p)
    except Exception:
        dp = 0.0
    if np.isfinite(dp) and dp > 0.0:
        if dp > 1.0:
            dp = 1.0
        try:
            dms = float(dropout_mu_scale)
        except Exception:
            dms = 0.0
        if np.isfinite(dms) and dms > 0.0:
            p = float(dp) * np.exp(-mu / float(dms))
        else:
            p = float(dp)
        mask = rng.random(size=counts.shape) < p
        if np.any(mask):
            cc = np.asarray(counts, dtype=np.int64)
            cc[mask] = 0
            counts = cc

    return [int(x) for x in np.asarray(counts, dtype=np.int64).tolist()]


def _stx_synthetic_v3_noisy_counts(
    T: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    H: int,
    W: int,
    rng: Any,
    z_target: Optional[np.ndarray] = None,
) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 2:
        raise ValueError("T must be 2D")
    n, g = int(T.shape[0]), int(T.shape[1])
    if n <= 0 or g <= 0:
        return np.zeros((0, max(g, 0)), dtype=np.int64)
    T = np.where(np.isfinite(T) & (T >= 0.0), T, 0.0)

    spot_total = np.asarray(np.sum(T, axis=1), dtype=np.float64).reshape(n)
    tissue = spot_total > 0.0
    target_median_umi = 2000.0
    if np.any(tissue):
        med = float(np.median(spot_total[tissue]))
        if np.isfinite(med) and med > 0.0:
            scale = float(target_median_umi) / float(med)
        else:
            scale = 1.0
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    if float(scale) > 1.0:
        scale = 1.0
    T = T * float(scale)

    x = np.asarray(np.round(x), dtype=np.int64).reshape(n)
    y = np.asarray(np.round(y), dtype=np.int64).reshape(n)

    sigma_cell = 0.35
    theta = 50.0
    eps = 0.08
    ambient_total_umi = 0.05
    ambient_sigma_cell = 0.25

    s_i = np.exp(rng.normal(loc=-0.5 * sigma_cell * sigma_cell, scale=sigma_cell, size=n))
    mu = T * s_i.reshape(n, 1)

    coords = np.stack([y, x], axis=1)
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    loc_counts = np.bincount(inv, minlength=int(uniq.shape[0])).astype(np.float64)
    loc_counts = np.maximum(loc_counts, 1.0)

    loc_mu = np.zeros((int(uniq.shape[0]), g), dtype=np.float64)
    for j in range(g):
        loc_sum = np.bincount(inv, weights=mu[:, j], minlength=int(uniq.shape[0])).astype(np.float64)
        loc_mu[:, j] = loc_sum / loc_counts

    mu_mixed = np.asarray(mu, dtype=np.float64).copy()
    uy_all = uniq[:, 0].astype(np.int64)
    ux_all = uniq[:, 1].astype(np.int64)
    inb = (uy_all >= 0) & (uy_all < int(H)) & (ux_all >= 0) & (ux_all < int(W))
    loc_to_inb = np.full((int(uniq.shape[0]),), -1, dtype=np.int64)
    inb_idx = np.nonzero(inb)[0]
    loc_to_inb[inb_idx] = np.arange(int(inb_idx.size), dtype=np.int64)
    uy = uy_all[inb]
    ux = ux_all[inb]
    loc_mu_inb = loc_mu[inb]

    loc_tissue = (
        np.bincount(inv, weights=np.asarray(tissue, dtype=np.float64), minlength=int(uniq.shape[0])).astype(np.float64)
        > 0.0
    )
    loc_tissue_inb = np.asarray(loc_tissue[inb], dtype=bool)

    key_to_inb: Dict[tuple[int, int], int] = {}
    for k in range(int(uy.size)):
        key_to_inb[(int(uy[k]), int(ux[k]))] = int(k)

    inv_inb = loc_to_inb[np.asarray(inv, dtype=np.int64)]
    cell_has_loc = inv_inb >= 0

    for j in range(g):
        loc_mix = np.full((int(uy.size),), np.nan, dtype=np.float64)
        for k in range(int(uy.size)):
            if not bool(loc_tissue_inb[int(k)]):
                continue
            yy = int(uy[k])
            xx = int(ux[k])
            neigh: list[float] = []
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                kk = key_to_inb.get((yy + int(dy), xx + int(dx)))
                if kk is None:
                    continue
                if not bool(loc_tissue_inb[int(kk)]):
                    continue
                neigh.append(float(loc_mu_inb[int(kk), int(j)]))
            if not neigh:
                continue
            nb = float(np.mean(np.asarray(neigh, dtype=np.float64)))
            loc_mix[int(k)] = (1.0 - float(eps)) * float(loc_mu_inb[int(k), int(j)]) + float(eps) * nb

        gm = np.full((n,), np.nan, dtype=np.float64)
        if np.any(cell_has_loc):
            gm[cell_has_loc] = loc_mix[inv_inb[cell_has_loc]]
        ok = np.isfinite(gm)
        if np.any(ok):
            mu_mixed[:, j] = np.where(ok, gm, mu[:, j])

    mu = mu_mixed

    meanT = np.mean(T[tissue] if np.any(tissue) else T, axis=0)
    meanT = np.where(np.isfinite(meanT) & (meanT >= 0.0), meanT, 0.0)
    meanT_sum = float(np.sum(meanT))
    if not np.isfinite(meanT_sum) or meanT_sum <= 0.0:
        feat_p = np.ones((g,), dtype=np.float64) / float(g)
    else:
        feat_p = meanT / float(meanT_sum)
    sA = np.exp(rng.normal(loc=-0.5 * ambient_sigma_cell * ambient_sigma_cell, scale=ambient_sigma_cell, size=n))
    tissue_f = np.asarray(tissue, dtype=np.float64).reshape(n)
    mu_total = mu + (sA.reshape(n, 1) * tissue_f.reshape(n, 1) * float(ambient_total_umi)) * feat_p.reshape(1, g)
    mu_total = np.where(np.isfinite(mu_total) & (mu_total >= 0.0), mu_total, 0.0)

    th = float(theta)
    if not np.isfinite(th) or th <= 0.0:
        th = 1.0
    lam = rng.gamma(shape=th, scale=(mu_total / th))
    Y = rng.poisson(lam).astype(np.int64)

    z0 = np.mean(Y == 0, axis=0)
    if z_target is None:
        zt = np.asarray(z0, dtype=np.float64)
    else:
        zt = np.asarray(z_target, dtype=np.float64).reshape(-1)
        if int(zt.size) != int(g):
            raise ValueError("z_target must have length == #features")
        zt = np.where(np.isfinite(zt), zt, z0)
    zt = np.clip(zt, 0.0, 0.98)
    denom = 1.0 - z0
    denom = np.where(denom <= 0.0, 1.0, denom)
    p_extra = np.clip((zt - z0) / denom, 0.0, 1.0)

    for j in range(g):
        pj = float(p_extra[j])
        if pj <= 0.0:
            continue
        r = rng.random(size=n)
        drop = (Y[:, j] > 0) & (r < pj)
        if np.any(drop):
            Y[drop, j] = 0

    return np.asarray(Y, dtype=np.int64)


def _bulk_synthetic_v1_noisy_counts(
    T: np.ndarray,
    *,
    rng: Any,
    z_target: Optional[np.ndarray] = None,
) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 2:
        raise ValueError("T must be 2D")
    n, g = int(T.shape[0]), int(T.shape[1])
    if n <= 0 or g <= 0:
        return np.zeros((0, max(g, 0)), dtype=np.int64)
    T = np.where(np.isfinite(T) & (T >= 0.0), T, 0.0)

    sigma_sample = 0.35
    theta = 50.0
    ambient_frac = 0.001
    ambient_sigma_sample = 0.25

    s_i = np.exp(rng.normal(loc=-0.5 * sigma_sample * sigma_sample, scale=sigma_sample, size=n))
    mu = T * s_i.reshape(n, 1)

    meanT = np.mean(T, axis=0)
    lam_feat = float(ambient_frac) * meanT
    sA = np.exp(rng.normal(loc=-0.5 * ambient_sigma_sample * ambient_sigma_sample, scale=ambient_sigma_sample, size=n))
    mu_total = mu + sA.reshape(n, 1) * lam_feat.reshape(1, g)
    mu_total = np.where(np.isfinite(mu_total) & (mu_total >= 0.0), mu_total, 0.0)

    th = float(theta)
    if not np.isfinite(th) or th <= 0.0:
        th = 1.0
    lam = rng.gamma(shape=th, scale=(mu_total / th))
    Y = rng.poisson(lam).astype(np.int64)

    z0 = np.mean(Y == 0, axis=0)
    if z_target is None:
        zt = np.asarray(z0, dtype=np.float64)
    else:
        zt = np.asarray(z_target, dtype=np.float64).reshape(-1)
        if int(zt.size) != int(g):
            raise ValueError("z_target must have length == #features")
        zt = np.where(np.isfinite(zt), zt, z0)
    zt = np.clip(zt, 0.0, 0.98)
    denom = 1.0 - z0
    denom = np.where(denom <= 0.0, 1.0, denom)
    p_extra = np.clip((zt - z0) / denom, 0.0, 1.0)

    for j in range(g):
        pj = float(p_extra[j])
        if pj <= 0.0:
            continue
        r = rng.random(size=n)
        drop = (Y[:, j] > 0) & (r < pj)
        if np.any(drop):
            Y[drop, j] = 0

    return np.asarray(Y, dtype=np.int64)


def _csv_escape(v: Any) -> str:
    if v is None:
        s = ""
    else:
        s = str(v)
    if any(ch in s for ch in (",", "\n", "\r", "\"")):
        return '"' + s.replace('"', '""') + '"'
    return s


def _csv_from_rows(header: list[str], rows: list[list[Any]]) -> str:
    out_lines = [",".join(_csv_escape(h) for h in header)]
    for r in rows:
        out_lines.append(",".join(_csv_escape(x) for x in r))
    return "\n".join(out_lines) + "\n"


def _stx_pool_matrix_and_metadata_csv(
    matrix_csv_bytes: bytes,
    meta_csv_bytes: bytes,
    *,
    k: int,
) -> tuple[bytes, bytes]:
    kk = max(1, int(k))
    if kk <= 1:
        return bytes(matrix_csv_bytes or b""), bytes(meta_csv_bytes or b"")

    mat_txt = ""
    try:
        mat_txt = (matrix_csv_bytes or b"").decode("utf-8", errors="replace")
    except Exception:
        mat_txt = ""
    meta_txt = ""
    try:
        meta_txt = (meta_csv_bytes or b"").decode("utf-8", errors="replace")
    except Exception:
        meta_txt = ""

    mat_lines = [ln for ln in str(mat_txt).splitlines() if str(ln).strip()]
    meta_lines = [ln for ln in str(meta_txt).splitlines() if str(ln).strip()]
    if len(mat_lines) < 2 or len(meta_lines) < 2:
        return bytes(matrix_csv_bytes or b""), bytes(meta_csv_bytes or b"")

    mat_header = [str(x or "") for x in mat_lines[0].split(",")]
    meta_header = [str(x or "") for x in meta_lines[0].split(",")]
    if len(mat_header) < 2 or not meta_header:
        return bytes(matrix_csv_bytes or b""), bytes(meta_csv_bytes or b"")

    try:
        meta_cell_id_i = meta_header.index("cell_id")
    except Exception:
        meta_cell_id_i = -1
    try:
        meta_x_i = meta_header.index("x")
    except Exception:
        meta_x_i = -1
    try:
        meta_y_i = meta_header.index("y")
    except Exception:
        meta_y_i = -1
    try:
        meta_grid_i = meta_header.index("grid_index")
    except Exception:
        meta_grid_i = -1
    if meta_cell_id_i < 0 or meta_x_i < 0 or meta_y_i < 0:
        return bytes(matrix_csv_bytes or b""), bytes(meta_csv_bytes or b"")

    cell_to_xy: Dict[str, tuple[int, int]] = {}
    cell_to_meta: Dict[str, list[str]] = {}
    block_rep_meta: Dict[tuple[int, int], list[str]] = {}
    max_x = -1
    max_y = -1
    w_votes: Dict[int, int] = {}
    for ln in meta_lines[1:]:
        cols = [str(x or "") for x in str(ln).split(",")]
        if len(cols) < len(meta_header):
            continue
        cid = str(cols[int(meta_cell_id_i)] or "")
        if not cid:
            continue
        try:
            xi = int(float(cols[int(meta_x_i)]))
            yi = int(float(cols[int(meta_y_i)]))
        except Exception:
            continue
        cell_to_xy[str(cid)] = (int(xi), int(yi))
        cell_to_meta[str(cid)] = list(cols)
        key = (int(yi) // int(kk), int(xi) // int(kk))
        if key not in block_rep_meta:
            block_rep_meta[key] = list(cols)
        if int(meta_grid_i) >= 0:
            try:
                gi = int(float(cols[int(meta_grid_i)]))
            except Exception:
                gi = -1
            if int(gi) >= 0 and int(yi) > 0:
                num = int(gi) - int(xi)
                if num >= 0 and (num % int(yi)) == 0:
                    w0 = int(num // int(yi))
                    if w0 > 0:
                        w_votes[int(w0)] = int(w_votes.get(int(w0), 0)) + 1
        if int(xi) > int(max_x):
            max_x = int(xi)
        if int(yi) > int(max_y):
            max_y = int(yi)

    if not cell_to_xy:
        return bytes(matrix_csv_bytes or b""), bytes(meta_csv_bytes or b"")

    w_est = 0
    if w_votes:
        try:
            w_est = int(sorted(w_votes.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0])
        except Exception:
            w_est = 0
    if int(w_est) <= 0:
        w_est = int(max_x + 1) if int(max_x) >= 0 else 0

    g = int(len(mat_header) - 1)
    block_sum: Dict[tuple[int, int], np.ndarray] = {}
    block_root: Dict[tuple[int, int], str] = {}
    block_x_acc: Dict[tuple[int, int], int] = {}
    block_y_acc: Dict[tuple[int, int], int] = {}
    block_n: Dict[tuple[int, int], int] = {}

    for ln in mat_lines[1:]:
        cols = [str(x or "") for x in str(ln).split(",")]
        if len(cols) < int(g + 1):
            continue
        cid = str(cols[0] or "")
        if not cid:
            continue
        xy = cell_to_xy.get(cid)
        if xy is None:
            continue
        xi, yi = xy
        bx = int(xi) // int(kk)
        by = int(yi) // int(kk)
        key = (int(by), int(bx))
        acc = block_sum.get(key)
        if acc is None:
            acc = np.zeros(int(g), dtype=np.int64)
            block_sum[key] = acc
        for j in range(int(g)):
            try:
                vv = int(float(cols[int(j + 1)]))
            except Exception:
                vv = 0
            acc[int(j)] += int(vv)
        if key not in block_root:
            base = str(cid)
            if "_" in base:
                try:
                    base = base.rsplit("_", 1)[0]
                except Exception:
                    base = str(cid)
            block_root[key] = str(base)
        block_x_acc[key] = int(block_x_acc.get(key, 0)) + int(xi)
        block_y_acc[key] = int(block_y_acc.get(key, 0)) + int(yi)
        block_n[key] = int(block_n.get(key, 0)) + 1

    if not block_sum:
        return bytes(matrix_csv_bytes or b""), bytes(meta_csv_bytes or b"")

    keys = sorted(block_sum.keys())
    out_mat_rows: list[list[Any]] = []
    out_meta_rows: list[list[Any]] = []
    for key in keys:
        by, bx = key
        n0 = int(block_n.get(key, 0))
        if n0 <= 0:
            continue
        x_new = int(round(float(block_x_acc.get(key, 0)) / float(n0)))
        y_new = int(round(float(block_y_acc.get(key, 0)) / float(n0)))
        root = str(block_root.get(key, "spot") or "spot")
        new_cid = f"{root}_p{int(kk)}_{int(by)}_{int(bx)}"

        acc = block_sum.get(key)
        if acc is None:
            continue
        out_mat_rows.append([str(new_cid), *[int(x) for x in np.asarray(acc, dtype=np.int64).tolist()]])

        rep_meta = block_rep_meta.get((int(by), int(bx)))
        if rep_meta is None:
            continue
        cols = list(rep_meta)
        cols[int(meta_cell_id_i)] = str(new_cid)
        cols[int(meta_x_i)] = str(int(x_new))
        cols[int(meta_y_i)] = str(int(y_new))
        if int(meta_grid_i) >= 0 and int(w_est) > 0:
            cols[int(meta_grid_i)] = str(int(int(y_new) * int(w_est) + int(x_new)))
        out_meta_rows.append(cols)

    out_mat_txt = _csv_from_rows(list(mat_header), out_mat_rows)
    out_meta_txt = _csv_from_rows(list(meta_header), [[str(x) for x in r] for r in out_meta_rows])
    return out_mat_txt.encode("utf-8"), out_meta_txt.encode("utf-8")


def _safe_read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _ensure_dirs() -> None:
    try:
        _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _OMICS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _docs_safe_path(name: str) -> Path:
    nm = str(name or "").strip()
    if not nm:
        raise ValueError("missing name")
    if "\x00" in nm:
        raise ValueError("bad name")
    if nm.startswith("/") or nm.startswith("\\"):
        raise ValueError("bad name")
    if ":" in nm:
        raise ValueError("bad name")
    p = Path(nm)
    if p.is_absolute():
        raise ValueError("bad name")
    if any(part in ("..", "") for part in p.parts):
        raise ValueError("bad name")
    out = (_DOCS_DIR / p).resolve()
    base = _DOCS_DIR.resolve()
    if str(out).startswith(str(base) + os.sep) or out == base:
        return out
    raise ValueError("bad name")


def _omics_safe_run_id(run_id: Any) -> str:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("missing run_id")
    if "\x00" in rid:
        raise ValueError("bad run_id")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in rid):
        raise ValueError("bad run_id")
    if len(rid) > 80:
        raise ValueError("bad run_id")
    return rid


def _omics_safe_relpath(name: Any) -> Path:
    nm = str(name or "").strip()
    if not nm:
        raise ValueError("missing name")
    if "\x00" in nm:
        raise ValueError("bad name")
    if nm.startswith("/") or nm.startswith("\\"):
        raise ValueError("bad name")
    if ":" in nm:
        raise ValueError("bad name")
    p = Path(nm)
    if p.is_absolute():
        raise ValueError("bad name")
    if any(part in ("..", "") for part in p.parts):
        raise ValueError("bad name")
    return p


def _omics_safe_label(v: Any, default: str = "item") -> str:
    s = str(v or "").strip()
    if not s:
        s = str(default or "item")
    out: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    ss = "".join(out).strip("_")
    if not ss:
        ss = str(default or "item")
    return ss[:80]


class _OmicsWorkspace:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        try:
            _ensure_dirs()
        except Exception:
            pass

    def _run_dir(self, run_id: str) -> Path:
        rid = _omics_safe_run_id(run_id)
        out = (_OMICS_RUNS_DIR / rid).resolve()
        base = _OMICS_RUNS_DIR.resolve()
        if str(out).startswith(str(base) + os.sep):
            return out
        raise ValueError("bad run_id")

    def _run_manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "manifest.json"

    def create_run(self, manifest: Dict[str, Any], files_text: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            created_at = time.time()

            rid_in: Optional[str] = None
            if isinstance(manifest, dict):
                try:
                    rid_in = str(manifest.get("run_id") or "").strip() or None
                except Exception:
                    rid_in = None

            rid = ""
            if rid_in:
                try:
                    rid_cand = _omics_safe_run_id(rid_in)
                    run_dir_cand = (_OMICS_RUNS_DIR / rid_cand).resolve()
                    base = _OMICS_RUNS_DIR.resolve()
                    if str(run_dir_cand).startswith(str(base) + os.sep) and not run_dir_cand.exists():
                        rid = rid_cand
                except Exception:
                    rid = ""
            if not rid:
                rid = uuid.uuid4().hex

        run_dir = self._run_dir(rid)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        mf: Dict[str, Any] = dict(manifest) if isinstance(manifest, dict) else {}
        mf["run_id"] = str(rid)
        mf["created_at"] = float(mf.get("created_at") or created_at)

        written: list[Dict[str, Any]] = []
        if isinstance(files_text, dict):
            for name, content in files_text.items():
                rel = _omics_safe_relpath(name)
                p = (run_dir / rel).resolve()
                if run_dir not in p.parents:
                    raise ValueError("bad name")
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                if isinstance(content, (bytes, bytearray)):
                    _atomic_write_bytes(p, bytes(content))
                else:
                    _atomic_write_text(p, str(content or ""))
                try:
                    sz = int(p.stat().st_size)
                except Exception:
                    sz = 0
                written.append({"name": str(rel.as_posix()), "bytes": int(sz)})

        mf["files"] = list(written)
        _atomic_write_json(self._run_manifest_path(rid), mf)
        return {"run_id": str(rid), "created_at": float(mf.get("created_at") or created_at), "files": list(written)}

    def list_runs(self, limit: int = 200) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        try:
            base = _OMICS_RUNS_DIR.resolve()
            if not base.exists() or not base.is_dir():
                return []
            for ent in base.iterdir():
                if not ent.is_dir():
                    continue
                mp = (ent / "manifest.json").resolve()
                if not mp.exists() or not mp.is_file():
                    continue
                dd = _safe_read_json(mp)
                if isinstance(dd, dict):
                    out.append(dd)
        except Exception:
            return []

        def _key(d: Dict[str, Any]) -> float:
            try:
                return float(d.get("created_at") or 0.0)
            except Exception:
                return 0.0

        out.sort(key=_key, reverse=True)
        return out[: max(1, int(limit))]

    def _file_id(self, run_id: str, name: str) -> str:
        rid = _omics_safe_run_id(run_id)
        rel = _omics_safe_relpath(name)
        s = f"{rid}:{rel.as_posix()}".encode("utf-8")
        return hashlib.sha1(s).hexdigest()[:12]

    def _friendly_kind(self, kind: Any) -> str:
        k = str(kind or "").strip().lower()
        if k == "bulk_rnaseq":
            return "Bulk RNA-seq"
        if k == "bulk_proteomics":
            return "Bulk proteomics"
        if k == "bulk_metabolomics":
            return "Bulk metabolomics"
        if k == "spatial_transcriptomics":
            return "Spatial transcriptomics"
        if k == "characterization":
            return "Characterization"
        if k == "protein_screen":
            return "Drug screen"
        if k == "claim_cure":
            return "In vivo lifespan study (win claim)"
        if k:
            return str(kind)
        return "Omics"

    def _file_role(self, name: str) -> str:
        n = str(name or "")
        if n.endswith("_cell_metadata.csv"):
            return "cell_metadata"
        if n.endswith("_matrix.csv"):
            return "counts_matrix"
        if n.endswith("_measurements.csv"):
            return "measurements"
        bn = Path(n).name
        if bn.startswith("summarized_results_") and bn.endswith(".csv"):
            return "summarized_results"
        if n.endswith("_metadata.csv") or bn == "metadata.csv" or (bn.startswith("metadata_") and bn.endswith(".csv")):
            return "run_metadata"
        if n.startswith("samples_truth/") or ("/samples_truth/" in n) or ("samples_truth/" in n):
            return "counts_truth"
        if n.startswith("samples_noisy/") or ("/samples_noisy/" in n) or ("samples_noisy/" in n):
            return "counts_noisy"
        if n.startswith("samples/"):
            return "counts_noisy"
        if n.endswith("_timecourse_n.csv"):
            return "timecourse_n"
        if n.endswith("_timecourse.csv"):
            return "timecourse"
        if n.endswith("_death.csv"):
            return "death"
        if n.endswith("_survival.csv"):
            return "survival"
        if n.endswith("_culture_metrics.csv"):
            return "culture_metrics"
        return "file"

    def _extract_condition_replicate(self, name: str) -> tuple[str, Optional[int]]:
        n = str(name or "")
        bn = Path(n).name
        if bn == "metadata.csv" or (bn.startswith("metadata_") and bn.endswith(".csv")):
            return "", None
        if bn.startswith("summarized_results_") and bn.endswith(".csv"):
            return "", None
        base = Path(n).stem
        m = re.search(r"_r(\d+)", base)
        rep = None
        if m:
            try:
                rep = int(m.group(1))
            except Exception:
                rep = None
        cond = ""
        if base:
            cond = base
            if "_r" in cond:
                cond = cond.split("_r", 1)[0]
            if "_c" in cond:
                cond = cond.split("_c", 1)[0]
            if cond.startswith("replicates/"):
                cond = cond.split("/", 1)[-1]
            for suf in ("_timecourse_n", "_timecourse", "_death", "_survival", "_culture_metrics", "_measurements"):
                if cond.endswith(suf):
                    cond = cond[: -len(suf)]
                    break
        return str(cond or ""), rep

    def inventory(self, player_id_in: Any, *, limit_runs: int = 2000, view: str = "compact") -> Dict[str, Any]:
        pid = _sanitize_player_id(player_id_in)
        view_s = str(view or "").strip().lower()
        if view_s not in ("compact", "full"):
            view_s = "compact"
        runs = self.list_runs(limit=int(limit_runs))
        files_out: list[Dict[str, Any]] = []
        datasets_out: list[Dict[str, Any]] = []

        for mf in runs:
            if not isinstance(mf, dict):
                continue
            if str(mf.get("player_id") or "") != str(pid):
                continue
            rid = str(mf.get("run_id") or "")
            if not rid:
                continue
            created_at = mf.get("created_at")
            exp = str(mf.get("experiment") or "")
            kind = str(mf.get("kind") or "")
            friendly_kind = self._friendly_kind(kind)

            model_s = str(mf.get("model") or "")
            ticks_i = 0
            age_days_i = 0
            reps_i = 0
            try:
                ticks_i = int(mf.get("ticks") or 0)
            except Exception:
                ticks_i = 0
            try:
                age_days_i = int(mf.get("age_days") or ticks_i)
            except Exception:
                age_days_i = int(ticks_i)
            try:
                reps_i = int(mf.get("replicates") or 0)
            except Exception:
                reps_i = 0
            omics_set_s = str(mf.get("omics_set") or "")
            gene_set_s = str(mf.get("gene_set") or "")

            ds_parts: list[str] = []
            if exp:
                ds_parts.append(exp)
            if friendly_kind:
                ds_parts.append(friendly_kind)
            if model_s:
                ds_parts.append(f"model={model_s}")
            if omics_set_s:
                ds_parts.append(f"omics_set={omics_set_s}")
            if gene_set_s:
                ds_parts.append(f"gene_set={gene_set_s}")
            if age_days_i:
                ds_parts.append(f"age_days={int(age_days_i)}")
            if reps_i:
                ds_parts.append(f"replicates={int(reps_i)}")
            dataset_display_name = " | ".join([p for p in ds_parts if str(p).strip()])
            if not dataset_display_name:
                dataset_display_name = f"{friendly_kind} | run_id={rid}"

            prefix_parts: list[str] = []
            if exp:
                prefix_parts.append(str(exp))
            if kind:
                prefix_parts.append(str(kind))
            if model_s:
                prefix_parts.append(str(model_s))
            if omics_set_s:
                prefix_parts.append(str(omics_set_s))
            if gene_set_s:
                prefix_parts.append(str(gene_set_s))
            if age_days_i:
                prefix_parts.append(f"age_{int(age_days_i)}")
            dataset_prefix = _omics_safe_label("_".join([x for x in prefix_parts if str(x).strip()]), default="omics")

            dataset_obj: Dict[str, Any] = {
                "run_id": str(rid),
                "created_at": created_at,
                "experiment": str(exp),
                "kind": str(kind),
                "friendly_kind": str(friendly_kind),
                "model": str(model_s),
                "ticks": int(ticks_i),
                "age_days": int(age_days_i),
                "replicates": int(reps_i),
                "omics_set": str(omics_set_s),
                "gene_set": str(gene_set_s),
                "display_name": str(dataset_display_name),
                "dataset_prefix": str(dataset_prefix),
                "file_ids": [],
                "data_file_ids": [],
                "metadata_file_ids": [],
                "metadata_map": {},
            }

            f0 = mf.get("files")
            if not isinstance(f0, list):
                f0 = []

            by_name: Dict[str, str] = {}
            by_fid: Dict[str, Dict[str, Any]] = {}
            run_meta_fid = ""
            for ent in f0:
                if not isinstance(ent, dict):
                    continue
                name = str(ent.get("name") or "")
                if not name:
                    continue
                try:
                    fid = self._file_id(rid, name)
                except Exception:
                    continue
                role = self._file_role(name)
                cond, rep = self._extract_condition_replicate(name)

                llm_fn = ""
                if role in ("counts_noisy", "counts_truth", "counts_matrix"):
                    suf = ""
                    if role == "counts_noisy":
                        suf = "counts_noisy"
                    elif role == "counts_truth":
                        suf = "counts_truth"
                    else:
                        suf = "counts_matrix"
                    if rep is not None:
                        llm_fn = f"{dataset_prefix}_replicate_{int(rep)}_{suf}.csv"
                    elif cond:
                        llm_fn = f"{dataset_prefix}_{_omics_safe_label(cond, default='condition')}_{suf}.csv"
                    else:
                        llm_fn = f"{dataset_prefix}_{suf}.csv"
                elif role == "measurements":
                    if rep is not None:
                        llm_fn = f"{dataset_prefix}_replicate_{int(rep)}_measurements.csv"
                    elif cond:
                        llm_fn = f"{dataset_prefix}_{_omics_safe_label(cond, default='condition')}_measurements.csv"
                    else:
                        llm_fn = f"{dataset_prefix}_measurements.csv"
                elif role == "summarized_results":
                    llm_fn = f"{dataset_prefix}_summarized_results.csv"
                elif role == "timecourse":
                    if rep is not None:
                        llm_fn = f"{dataset_prefix}_replicate_{int(rep)}_timecourse.csv"
                    else:
                        llm_fn = f"{dataset_prefix}_timecourse.csv"
                elif role == "timecourse_n":
                    if rep is not None:
                        llm_fn = f"{dataset_prefix}_replicate_{int(rep)}_timecourse_n.csv"
                    else:
                        llm_fn = f"{dataset_prefix}_timecourse_n.csv"
                elif role == "death":
                    llm_fn = f"{dataset_prefix}_death.csv"
                elif role == "survival":
                    llm_fn = f"{dataset_prefix}_survival.csv"
                elif role == "culture_metrics":
                    llm_fn = f"{dataset_prefix}_culture_metrics.csv"
                elif role == "run_metadata":
                    llm_fn = f"{dataset_prefix}_metadata.csv"
                elif role == "cell_metadata":
                    if rep is not None:
                        llm_fn = f"{dataset_prefix}_replicate_{int(rep)}_cell_metadata.csv"
                    else:
                        llm_fn = f"{dataset_prefix}_cell_metadata.csv"
                else:
                    llm_fn = f"{dataset_prefix}_{_omics_safe_label(Path(name).name, default='file')}"

                display = friendly_kind
                if role == "counts_matrix":
                    display += " counts matrix"
                elif role == "counts_noisy":
                    display += " counts (noisy)"
                elif role == "counts_truth":
                    display += " counts (truth)"
                elif role == "measurements":
                    display += " measurements"
                elif role == "summarized_results":
                    display += " summarized results"
                elif role == "timecourse":
                    display += " timecourse"
                elif role == "timecourse_n":
                    display += " timecourse_n"
                elif role == "death":
                    display += " death table"
                elif role == "survival":
                    display += " survival"
                elif role == "culture_metrics":
                    display += " culture metrics"
                elif role == "cell_metadata":
                    display += " cell metadata"
                elif role == "run_metadata":
                    display += " run metadata"
                else:
                    display += " file"
                if cond:
                    display += f" — {cond}"
                if rep is not None:
                    display += f" (replicate {int(rep)})"
                if exp:
                    display += f" [{exp}]"

                files_out.append(
                    {
                        "file_id": str(fid),
                        "llm_filename": str(llm_fn),
                        "display_name": str(display),
                        "role": str(role),
                        "kind": str(kind),
                        "experiment": str(exp),
                        "condition": str(cond),
                        "replicate": int(rep) if rep is not None else None,
                        "run_id": str(rid),
                        "name": str(name),
                        "bytes": int(ent.get("bytes") or 0),
                        "created_at": created_at,
                        "download_url": f"/api/omics/file?player_id={pid}&file_id={fid}",
                    }
                )

                f_ent = dict(files_out[-1])
                by_name[str(name)] = str(fid)
                by_fid[str(fid)] = f_ent
                dataset_obj["file_ids"].append(str(fid))
                if str(role) in (
                    "counts_noisy",
                    "counts_truth",
                    "counts_matrix",
                    "measurements",
                    "summarized_results",
                    "timecourse",
                    "timecourse_n",
                    "death",
                    "survival",
                    "culture_metrics",
                ):
                    dataset_obj["data_file_ids"].append(str(fid))
                if str(role) in ("run_metadata", "cell_metadata"):
                    dataset_obj["metadata_file_ids"].append(str(fid))
                if str(role) == "run_metadata":
                    run_meta_fid = str(fid)

            metadata_map: Dict[str, Any] = {}
            data_ids0 = dataset_obj.get("data_file_ids")
            if not isinstance(data_ids0, list):
                data_ids0 = []
            for dfid in data_ids0:
                if not isinstance(dfid, str) or not dfid:
                    continue
                f_ent = by_fid.get(str(dfid))
                if not isinstance(f_ent, dict):
                    continue
                nm = str(f_ent.get("name") or "")
                metas: list[str] = []
                if run_meta_fid:
                    metas.append(str(run_meta_fid))
                if nm.endswith("_matrix.csv"):
                    meta_name = nm[:-len("_matrix.csv")] + "_cell_metadata.csv"
                    m2 = by_name.get(meta_name)
                    if isinstance(m2, str) and m2:
                        metas.append(str(m2))
                metadata_map[str(dfid)] = {
                    "metadata_file_ids": metas,
                }
            dataset_obj["metadata_map"] = metadata_map

            datasets_out.append(dataset_obj)

        files_out.sort(key=lambda d: (str(d.get("kind") or ""), str(d.get("condition") or ""), str(d.get("role") or ""), int(d.get("replicate") or -1), str(d.get("file_id") or "")))

        groups: Dict[str, Dict[str, Any]] = {}
        for f in files_out:
            role = str(f.get("role") or "")
            kind = str(f.get("kind") or "")
            cond = str(f.get("condition") or "")
            if role not in ("counts_noisy", "counts_truth", "counts_matrix"):
                continue
            gid = f"{kind}:{cond}:{role}".strip(":")
            if gid not in groups:
                gname = self._friendly_kind(kind)
                if role == "counts_noisy":
                    gname += " counts (noisy)"
                elif role == "counts_truth":
                    gname += " counts (truth)"
                else:
                    gname += " counts matrix"
                if cond:
                    gname += f" — {cond}"
                groups[gid] = {
                    "group_id": gid,
                    "display_name": gname,
                    "kind": kind,
                    "role": role,
                    "condition": cond,
                    "file_ids": [],
                }
            groups[gid]["file_ids"].append(str(f.get("file_id") or ""))

        groups_out = list(groups.values())
        groups_out.sort(key=lambda d: str(d.get("group_id") or ""))

        def _ds_key(d: Dict[str, Any]) -> float:
            try:
                return float(d.get("created_at") or 0.0)
            except Exception:
                return 0.0

        datasets_out.sort(key=_ds_key, reverse=True)

        files_by_id: Dict[str, Dict[str, Any]] = {}
        for f in files_out:
            if not isinstance(f, dict):
                continue
            fid = str(f.get("file_id") or "")
            if fid and fid not in files_by_id:
                files_by_id[fid] = f

        llm_lines: list[str] = []
        analysis_data_hint = "data files"
        analysis_step2_note = ""
        if datasets_out:
            recent = datasets_out[0]
            rk = str(recent.get("kind") or "").strip().lower()

            is_spatial = bool(str(recent.get("gene_set") or "").strip())
            if not is_spatial:
                meta_ids = recent.get("metadata_file_ids")
                if not isinstance(meta_ids, list):
                    meta_ids = []
                for mfid in meta_ids:
                    if not isinstance(mfid, str) or not mfid:
                        continue
                    f_ent = files_by_id.get(str(mfid))
                    if not isinstance(f_ent, dict):
                        continue
                    if str(f_ent.get("role") or "") == "cell_metadata":
                        is_spatial = True
                        break

            if rk == "characterization":
                llm_lines.append("Your most recent characterization run produced the following files:")
                analysis_data_hint = "timecourse tables"
            elif rk == "claim_cure":
                llm_lines.append("Your most recent in vivo lifespan study (win claim) produced the following files:")
                analysis_data_hint = "summarized results tables"
            elif rk == "protein_screen":
                llm_lines.append("Your most recent drug screen run produced the following files:")
                analysis_data_hint = "measurements tables"
            elif is_spatial:
                if rk == "bulk_proteomics":
                    llm_lines.append("Your most recent spatial proteomics run produced the following files:")
                else:
                    llm_lines.append("Your most recent spatial transcriptomics run produced the following files:")
                analysis_data_hint = "spatial count matrices"
                analysis_step2_note = " (For spatial: include BOTH run metadata and per-replicate cell_metadata when present.)"
            elif rk.startswith("bulk_"):
                llm_lines.append("Your most recent omics generation produced the following files:")
                analysis_data_hint = "counts matrices"
            else:
                llm_lines.append("Your most recent run produced the following files:")
            ds_name = str(recent.get("display_name") or "").strip()
            if ds_name:
                llm_lines.append(f"Dataset: {ds_name}")

            recent_data_ids = recent.get("data_file_ids")
            if not isinstance(recent_data_ids, list):
                recent_data_ids = []
            for dfid in recent_data_ids:
                if not isinstance(dfid, str) or not dfid:
                    continue
                f_ent = files_by_id.get(str(dfid))
                if not isinstance(f_ent, dict):
                    continue
                fid = str(f_ent.get("file_id") or "")
                if not fid:
                    continue
                if view_s == "compact":
                    role = str(f_ent.get("role") or "")
                    cond = str(f_ent.get("condition") or "")
                    rep = f_ent.get("replicate")
                    rep_s = ""
                    try:
                        if rep is not None:
                            rep_s = str(int(rep))
                    except Exception:
                        rep_s = ""
                    extra: list[str] = []
                    if role:
                        extra.append(f"role={role}")
                    if cond:
                        extra.append(f"condition={cond}")
                    if rep_s:
                        extra.append(f"replicate={rep_s}")
                    if extra:
                        llm_lines.append(f"- file_id={fid} (" + ", ".join(extra) + ")")
                    else:
                        llm_lines.append(f"- file_id={fid}")
                else:
                    fn = str(f_ent.get("llm_filename") or f_ent.get("name") or "")
                    if fn:
                        llm_lines.append(f"- {fn} (file_id={fid})")

            meta_map = recent.get("metadata_map")
            if isinstance(meta_map, dict) and recent_data_ids:
                llm_lines.append("")
                llm_lines.append("Metadata mapping (what to load alongside each data file):")
                for dfid in recent_data_ids:
                    if not isinstance(dfid, str) or not dfid:
                        continue
                    f_ent = files_by_id.get(str(dfid))
                    if not isinstance(f_ent, dict):
                        continue
                    dfn = str(f_ent.get("llm_filename") or f_ent.get("name") or "")
                    mm = meta_map.get(str(dfid))
                    if not isinstance(mm, dict):
                        continue
                    mfiles = mm.get("metadata_file_ids")
                    if not isinstance(mfiles, list) or not mfiles:
                        continue
                    mrefs: list[str] = []
                    for mfid in mfiles:
                        if not isinstance(mfid, str) or not mfid:
                            continue
                        mf = files_by_id.get(str(mfid))
                        if not isinstance(mf, dict):
                            continue
                        if view_s == "compact":
                            mrefs.append(f"file_id={str(mfid)}")
                        else:
                            mfn = str(mf.get("llm_filename") or mf.get("name") or "")
                            if mfn:
                                mrefs.append(f"{mfn} (file_id={str(mfid)})")
                    if mrefs:
                        if view_s == "compact":
                            llm_lines.append(f"- file_id={str(dfid)} -> " + ", ".join(mrefs))
                        else:
                            if not dfn:
                                continue
                            llm_lines.append(f"- {dfn} -> " + ", ".join(mrefs))

            if len(datasets_out) > 1:
                llm_lines.append("")
                llm_lines.append("In addition to the recently produced files you also have the following datasets available:")
                llm_lines.append("(To inspect an older dataset's exact file_ids, call GET /api/omics/run?run_id=<run_id>.)")
                max_other = 30
                for ds in datasets_out[1 : 1 + int(max_other)]:
                    if not isinstance(ds, dict):
                        continue
                    llm_lines.append("")
                    ds_name2 = str(ds.get("display_name") or "").strip()
                    rid2 = str(ds.get("run_id") or "").strip()
                    n_data = 0
                    n_meta = 0
                    try:
                        n_data = int(len(ds.get("data_file_ids") or [])) if isinstance(ds.get("data_file_ids"), list) else 0
                    except Exception:
                        n_data = 0
                    try:
                        n_meta = int(len(ds.get("metadata_file_ids") or [])) if isinstance(ds.get("metadata_file_ids"), list) else 0
                    except Exception:
                        n_meta = 0
                    if ds_name2:
                        if rid2:
                            llm_lines.append(f"Dataset: {ds_name2} (run_id={rid2}, data_file_ids={n_data}, metadata_file_ids={n_meta})")
                        else:
                            llm_lines.append(f"Dataset: {ds_name2} (data_file_ids={n_data}, metadata_file_ids={n_meta})")
                    elif rid2:
                        llm_lines.append(f"Dataset: run_id={rid2} (data_file_ids={n_data}, metadata_file_ids={n_meta})")
                if len(datasets_out) > (1 + int(max_other)):
                    llm_lines.append("")
                    llm_lines.append(f"(Additional datasets omitted from this message: {int(len(datasets_out) - (1 + int(max_other)))}. See JSON field 'datasets' for the full index.)")
        else:
            llm_lines.append("No files are available yet.")

        llm_lines.append("")
        llm_lines.append("To perform an analysis:")
        llm_lines.append(f"1) Choose the file_id(s) for the data you want to analyze (typically {analysis_data_hint}).")
        llm_lines.append(f"2) Include the matching metadata file_id(s) shown in the mapping above.{analysis_step2_note}")
        llm_lines.append("3) Think of the instructions for the analysis you want to run. If you are not sure what the data/metadata looks like, ask to inspect the files first and then ask again to analyze.")
        llm_lines.append("   For in vivo timecourse data: compute death day as the first day any *_death (or other death measurement) becomes 1. If no death event occurs within the recorded time window, treat that replicate as survived-through-study (censored), not as death at the final day.")
        llm_lines.append("4) If you need more samples to run the analysis run more samples before asking for the analysis. For example if you asked for an experiment with a disease-like cell culture model, you might want to run the corresponding healthy cell culture model to get a control to compare to.")
        llm_lines.append("5) Call POST /api/omics/analyze with JSON like:")
        llm_lines.append('{"file_ids":["<data_file_id>","<metadata_file_id>","..."],"instructions":"..."}')

        llm_message = "\n".join([str(x) for x in llm_lines if isinstance(x, str)])

        if view_s == "full":
            return {
                "player_id": str(pid),
                "files": files_out,
                "groups": groups_out,
                "datasets": datasets_out,
                "llm_message": llm_message,
            }

        files_compact: list[Dict[str, Any]] = []
        for f in files_out:
            if not isinstance(f, dict):
                continue
            files_compact.append(
                {
                    "file_id": f.get("file_id"),
                    "display_name": f.get("display_name"),
                    "role": f.get("role"),
                    "kind": f.get("kind"),
                    "experiment": f.get("experiment"),
                    "condition": f.get("condition"),
                    "replicate": f.get("replicate"),
                    "run_id": f.get("run_id"),
                    "bytes": f.get("bytes"),
                    "created_at": f.get("created_at"),
                }
            )

        datasets_compact: list[Dict[str, Any]] = []
        for i, ds in enumerate(datasets_out):
            if not isinstance(ds, dict):
                continue
            ent: Dict[str, Any] = {
                "run_id": ds.get("run_id"),
                "created_at": ds.get("created_at"),
                "experiment": ds.get("experiment"),
                "kind": ds.get("kind"),
                "friendly_kind": ds.get("friendly_kind"),
                "model": ds.get("model"),
                "ticks": ds.get("ticks"),
                "age_days": ds.get("age_days"),
                "replicates": ds.get("replicates"),
                "omics_set": ds.get("omics_set"),
                "gene_set": ds.get("gene_set"),
                "display_name": ds.get("display_name"),
            }

            if i == 0:
                ent["data_file_ids"] = ds.get("data_file_ids")
                ent["metadata_file_ids"] = ds.get("metadata_file_ids")
                ent["metadata_map"] = ds.get("metadata_map")
            else:
                try:
                    ent["data_file_ids_n"] = int(len(ds.get("data_file_ids") or [])) if isinstance(ds.get("data_file_ids"), list) else 0
                except Exception:
                    ent["data_file_ids_n"] = 0
                try:
                    ent["metadata_file_ids_n"] = int(len(ds.get("metadata_file_ids") or [])) if isinstance(ds.get("metadata_file_ids"), list) else 0
                except Exception:
                    ent["metadata_file_ids_n"] = 0

            datasets_compact.append(ent)

        return {
            "player_id": str(pid),
            "files": files_compact,
            "groups": groups_out,
            "datasets": datasets_compact,
            "llm_message": llm_message,
        }

    def resolve_player_file_ids(self, player_id_in: Any, file_ids: list[str]) -> list[Dict[str, Any]]:
        inv = self.inventory(player_id_in, view="full")
        files = inv.get("files")
        if not isinstance(files, list):
            return []
        wanted = set(str(x or "") for x in (file_ids or []) if str(x or "").strip())
        out: list[Dict[str, Any]] = []
        for f in files:
            if not isinstance(f, dict):
                continue
            fid = str(f.get("file_id") or "")
            if fid and fid in wanted:
                out.append(f)
        return out

    def resolve_file_ids(self, file_ids: list[str], *, limit_runs: int = 2000) -> tuple[list[Dict[str, Any]], list[str]]:
        wanted = [str(x or "").strip() for x in (file_ids or [])]
        wanted = [x for x in wanted if x]
        if not wanted:
            return [], []

        wanted_set = set(wanted)
        by_fid: Dict[str, Dict[str, Any]] = {}
        ambiguous: set[str] = set()

        runs = self.list_runs(limit=int(limit_runs))
        for mf in runs:
            if not isinstance(mf, dict):
                continue
            rid = str(mf.get("run_id") or "").strip()
            if not rid:
                continue

            files = mf.get("files")
            if not isinstance(files, list) or not files:
                continue

            kind = str(mf.get("kind") or "")
            friendly_kind = self._friendly_kind(kind)
            exp = str(mf.get("experiment") or "")

            for ent in files:
                if not isinstance(ent, dict):
                    continue
                name = str(ent.get("name") or "")
                if not name:
                    continue
                try:
                    fid = self._file_id(str(rid), str(name))
                except Exception:
                    fid = ""
                if not fid or fid not in wanted_set:
                    continue

                prev = by_fid.get(str(fid))
                if isinstance(prev, dict):
                    if str(prev.get("run_id") or "") != str(rid) or str(prev.get("name") or "") != str(name):
                        ambiguous.add(str(fid))
                    continue

                role = self._file_role(str(name))
                cond, rep = self._extract_condition_replicate(str(name))
                display = str(friendly_kind)
                if role == "counts_matrix":
                    display += " counts matrix"
                elif role == "counts_noisy":
                    display += " counts (noisy)"
                elif role == "counts_truth":
                    display += " counts (truth)"
                elif role == "measurements":
                    display += " measurements"
                elif role == "summarized_results":
                    display += " summarized results"
                elif role == "timecourse":
                    display += " timecourse"
                elif role == "timecourse_n":
                    display += " timecourse_n"
                elif role == "death":
                    display += " death table"
                elif role == "survival":
                    display += " survival"
                elif role == "culture_metrics":
                    display += " culture metrics"
                elif role == "cell_metadata":
                    display += " cell metadata"
                elif role == "run_metadata":
                    display += " run metadata"
                else:
                    display += " file"
                if cond:
                    display += f" — {cond}"
                if rep is not None:
                    try:
                        display += f" (replicate {int(rep)})"
                    except Exception:
                        pass
                if exp:
                    display += f" [{exp}]"

                by_fid[str(fid)] = {
                    "file_id": str(fid),
                    "display_name": str(display),
                    "run_id": str(rid),
                    "name": str(name),
                    "bytes": int(ent.get("bytes") or 0),
                }

            if len(by_fid) >= len(wanted_set):
                break

        out: list[Dict[str, Any]] = []
        for fid in wanted:
            if fid in ambiguous:
                continue
            ent = by_fid.get(str(fid))
            if isinstance(ent, dict):
                out.append(ent)
        return out, sorted(list(ambiguous))

    def get_run(self, run_id: str) -> Dict[str, Any]:
        rid = _omics_safe_run_id(run_id)
        mp = self._run_manifest_path(rid)
        dd = _safe_read_json(mp)
        if not isinstance(dd, dict):
            raise ValueError("run not found")
        return dd

    def list_files(self, run_id: str) -> list[Dict[str, Any]]:
        rid = _omics_safe_run_id(run_id)
        run_dir = self._run_dir(rid)
        files: list[Dict[str, Any]] = []
        try:
            for p in run_dir.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(run_dir)
                if rel.as_posix() == "manifest.json":
                    continue
                try:
                    sz = int(p.stat().st_size)
                except Exception:
                    sz = 0
                files.append({"name": str(rel.as_posix()), "bytes": int(sz)})
        except Exception:
            return []
        files.sort(key=lambda d: str(d.get("name") or ""))
        return files

    def read_file_bytes(self, run_id: str, name: str) -> bytes:
        rid = _omics_safe_run_id(run_id)
        rel = _omics_safe_relpath(name)
        run_dir = self._run_dir(rid)
        p = (run_dir / rel).resolve()
        if run_dir not in p.parents:
            raise ValueError("bad name")
        if not p.exists() or not p.is_file():
            raise ValueError("file not found")
        try:
            return p.read_bytes()
        except Exception as e:
            raise ValueError(str(e))

    def file_path(self, run_id: str, name: str) -> Path:
        rid = _omics_safe_run_id(run_id)
        rel = _omics_safe_relpath(name)
        run_dir = self._run_dir(rid)
        p = (run_dir / rel).resolve()
        if run_dir not in p.parents:
            raise ValueError("bad name")
        if not p.exists() or not p.is_file():
            raise ValueError("file not found")
        return p


class _DocWorkspace:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.doc_id = ""
        self.path = ""
        self.payload_text = ""
        self.payload_hash = ""
        self.last_saved_hash = ""
        self.last_autosave_ts = 0.0
        self.last_saved_ts = 0.0
        self.has_autosave = False

        try:
            _ensure_dirs()
        except Exception:
            pass
        try:
            self._load_meta_from_disk()
        except Exception:
            pass

    def _load_meta_from_disk(self) -> None:
        mp = self._meta_path()
        dd = _safe_read_json(mp)
        if isinstance(dd, dict):
            self.doc_id = str(dd.get("doc_id") or "")
            self.path = str(dd.get("path") or "")
            self.payload_hash = str(dd.get("payload_hash") or "")
            self.last_saved_hash = str(dd.get("last_saved_hash") or "")
            try:
                self.last_autosave_ts = float(dd.get("last_autosave_ts") or 0.0)
            except Exception:
                self.last_autosave_ts = 0.0
            try:
                self.last_saved_ts = float(dd.get("last_saved_ts") or 0.0)
            except Exception:
                self.last_saved_ts = 0.0
        # Auto-load autosave payload on startup for seamless restore (Google Docs-like)
        ap = self._autosave_path()
        self.has_autosave = ap.exists()
        if self.has_autosave and not self.payload_text:
            try:
                txt = ap.read_text(encoding="utf-8")
                if txt.strip():
                    self.payload_text = txt
                    self.payload_hash = _sha256_text(txt)
                    if not self.doc_id:
                        self.doc_id = uuid.uuid4().hex
            except Exception:
                pass

    def _meta_path(self) -> Path:
        return _WORKSPACE_DIR / "doc_meta.json"

    def _autosave_path(self) -> Path:
        return _WORKSPACE_DIR / "doc_autosave.json"

    def _write_meta(self) -> None:
        meta = {
            "doc_id": self.doc_id,
            "path": self.path,
            "payload_hash": self.payload_hash,
            "last_saved_hash": self.last_saved_hash,
            "last_autosave_ts": self.last_autosave_ts,
            "last_saved_ts": self.last_saved_ts,
            "has_autosave": self.has_autosave,
        }
        _atomic_write_json(self._meta_path(), meta)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            dirty = bool(self.payload_hash and self.payload_hash != self.last_saved_hash)
            has_autosave = bool(self.has_autosave) or bool(self._autosave_path().exists())
            return {
                "ok": True,
                "loaded": bool(self.payload_text),
                "doc_id": self.doc_id,
                "path": self.path,
                "dirty": dirty,
                "has_autosave": has_autosave,
                "last_autosave_ts": float(self.last_autosave_ts),
                "last_saved_ts": float(self.last_saved_ts),
            }

    def clear_active(self) -> None:
        with self._lock:
            self.doc_id = ""
            self.path = ""
            self.payload_text = ""
            self.payload_hash = ""
            self.last_saved_hash = ""
            self.last_autosave_ts = 0.0
            self.last_saved_ts = 0.0
            ap = self._autosave_path()
            try:
                if ap.exists():
                    ap.unlink()
            except Exception:
                pass
            self.has_autosave = False
            self._write_meta()

    def set_payload_from_text(self, payload_text: str, path: str = "") -> Dict[str, Any]:
        txt = str(payload_text or "")
        if not txt.strip():
            raise ValueError("missing payload")
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            raise ValueError("payload must be object")
        h = _sha256_text(txt)
        with self._lock:
            if not self.doc_id:
                self.doc_id = uuid.uuid4().hex
            self.payload_text = txt
            self.payload_hash = h
            if isinstance(path, str) and path.strip():
                self.path = str(path).strip()
            self.last_autosave_ts = time.time()
            self.has_autosave = True
            _atomic_write_text(self._autosave_path(), txt)
            self._write_meta()
            dirty = bool(self.payload_hash and self.payload_hash != self.last_saved_hash)
            return {"ok": True, "doc_id": self.doc_id, "path": self.path, "dirty": dirty}

    def recover_autosave(self) -> Dict[str, Any]:
        ap = self._autosave_path()
        if not ap.exists():
            raise ValueError("no autosave")
        try:
            txt = ap.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(str(e))
        self.set_payload_from_text(txt)
        with self._lock:
            return {"ok": True, "payload_text": self.payload_text, **self.status()}

    def get_payload_text(self) -> str:
        with self._lock:
            return str(self.payload_text or "")

    def open_doc(self, name: str) -> Dict[str, Any]:
        path = _docs_safe_path(name)
        if not path.exists():
            raise ValueError("file not found")
        try:
            txt = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(str(e))
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            raise ValueError("file must be json object")
        h = _sha256_text(txt)
        with self._lock:
            self.doc_id = uuid.uuid4().hex
            self.path = str(Path(name).as_posix())
            self.payload_text = txt
            self.payload_hash = h
            self.last_saved_hash = h
            self.last_saved_ts = time.time()
            self.last_autosave_ts = time.time()
            self.has_autosave = True
            _atomic_write_text(self._autosave_path(), txt)
            self._write_meta()
            return {"ok": True, "payload_text": self.payload_text, **self.status()}

    def save_doc(self, name: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            txt = str(self.payload_text or "")
            if not txt.strip():
                raise ValueError("no active document")
            cur_path = str(self.path or "")
        target_name = (str(name).strip() if isinstance(name, str) else "") or cur_path
        if not target_name:
            raise ValueError("missing name")
        if not target_name.lower().endswith(".json"):
            target_name = target_name + ".json"
        path = _docs_safe_path(target_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        _atomic_write_text(path, txt)
        h = _sha256_text(txt)
        with self._lock:
            self.path = str(Path(target_name).as_posix())
            self.last_saved_hash = h
            self.last_saved_ts = time.time()
            self._write_meta()
            return {"ok": True, "path": self.path, **self.status()}

    def delete_doc(self, name: str) -> Dict[str, Any]:
        target_name = str(name or "").strip()
        if not target_name:
            raise ValueError("missing name")
        if not target_name.lower().endswith(".json"):
            target_name = target_name + ".json"
        path = _docs_safe_path(target_name)
        if not path.exists():
            raise ValueError("file not found")

        cur_path = ""
        with self._lock:
            cur_path = str(self.path or "")

        try:
            path.unlink()
        except Exception as e:
            raise ValueError(str(e))

        if path.exists():
            raise ValueError("delete failed")

        if cur_path and cur_path == str(Path(target_name).as_posix()):
            self.clear_active()
            return {"ok": True, "deleted": str(Path(target_name).as_posix()), **self.status()}
        return {"ok": True, "deleted": str(Path(target_name).as_posix())}

    def list_docs(self) -> Dict[str, Any]:
        base = _DOCS_DIR
        out = []
        try:
            for p in base.rglob("*.json"):
                try:
                    rel = p.relative_to(base).as_posix()
                    st = p.stat()
                    out.append({"name": rel, "size": int(st.st_size), "mtime": float(st.st_mtime)})
                except Exception:
                    continue
        except Exception:
            pass
        out.sort(key=lambda x: str(x.get("name") or ""))
        return {"ok": True, "files": out}


_DOC = _DocWorkspace()
_OMICS = _OmicsWorkspace()


def _evo_runs_ensure_dir() -> None:
    try:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _evo_state_path() -> Path:
    return _RUNS_DIR / "state.json"


def _evo_cfg_path() -> Path:
    return _RUNS_DIR / "cfg.json"


def _evo_base_payload_path() -> Path:
    return _RUNS_DIR / "base_payload.json"


def _evo_candidates_path() -> Path:
    return _RUNS_DIR / "candidates.json"


def _evo_stop_flag_path() -> Path:
    return _RUNS_DIR / "stop.flag"


def _evo_extract_measurement_defs(payload: Dict[str, Any]) -> list[Dict[str, str]]:
    measurements_cfg = payload.get("measurements_config")
    if not isinstance(measurements_cfg, dict) or int(measurements_cfg.get("version") or 0) != 3:
        return []
    meas_list = measurements_cfg.get("measurements")
    if not isinstance(meas_list, list):
        return []
    out: list[Dict[str, str]] = []
    for m in meas_list:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        expr = str(m.get("expr") or "").strip()
        if not name or not expr:
            continue
        out.append({"name": name, "expr": expr})
    return out


def _evo_resolve_target_layers(payload: Dict[str, Any], patterns: list[str]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    pats = [str(p).strip() for p in (patterns or []) if isinstance(p, str) and str(p).strip()]
    if not pats:
        pats = ["gene_*", "rna_*", "protein_*"]
    import fnmatch

    out: list[str] = []
    for name, ent in data.items():
        if not isinstance(name, str) or not name:
            continue
        if not any(fnmatch.fnmatch(name, pat) for pat in pats):
            continue
        if not isinstance(ent, dict):
            continue
        if ent.get("dtype") != "float32" or not isinstance(ent.get("b64"), str):
            continue
        out.append(name)
    out.sort()
    return out


def _evo_start_runner(payload: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    _evo_runs_ensure_dir()

    pid = _evo_runner_pid()
    if _pid_alive(pid):
        if not _pid_looks_like_evo_runner(pid):
            try:
                _atomic_write_json(_RUNS_DIR / "runner.pid", {"pid": 0, "stale_pid": int(pid), "cleared_at": float(time.time())})
            except Exception:
                pass
        else:
            try:
                if _evo_stop_flag_path().exists():
                    t0 = time.time()
                    while time.time() - t0 < 2.0:
                        if not _pid_alive(pid):
                            break
                        time.sleep(0.05)
            except Exception:
                pass
            if _pid_alive(pid):
                raise ValueError("evolution already running")

    try:
        if _evo_stop_flag_path().exists():
            _evo_stop_flag_path().unlink()
    except Exception:
        pass

    job_id = uuid.uuid4().hex
    cfg2 = dict(cfg)
    cfg2["job_id"] = job_id

    meas_defs = _evo_extract_measurement_defs(payload)
    if meas_defs:
        cfg2["measurement_defs"] = meas_defs

    tgt = cfg2.get("target_layers")
    if not isinstance(tgt, list):
        tgt = ["gene_*", "rna_*", "protein_*"]
        cfg2["target_layers"] = tgt
    resolved = _evo_resolve_target_layers(payload, [str(p) for p in tgt if isinstance(p, str)])
    cfg2["target_layers_resolved"] = resolved

    _atomic_write_json(_evo_cfg_path(), cfg2)
    _atomic_write_json(_evo_base_payload_path(), payload)
    try:
        _atomic_write_json(_evo_candidates_path(), {})
    except Exception:
        pass

    state0 = {
        "ok": True,
        "job_id": job_id,
        "running": True,
        "error": "",
        "cfg": cfg2,
        "progress": {},
        "history": {"best": [], "mean": [], "median": [], "p10": [], "p90": []},
        "baseline": {},
        "series": {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []},
        "perf": {},
        "top": [],
    }
    try:
        _atomic_write_json(_evo_state_path(), state0)
    except Exception:
        pass

    runner_path = (Path(__file__).resolve().parent / "evolution_runner.py").resolve()
    if not runner_path.exists():
        raise ValueError("missing evolution_runner.py")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        subprocess.Popen(
            [sys.executable, str(runner_path), "--dir", str(_RUNS_DIR)],
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        raise ValueError(str(e))

    return str(job_id)


def _evo_stop_runner() -> None:
    _evo_runs_ensure_dir()
    try:
        _evo_stop_flag_path().touch(exist_ok=True)
    except Exception:
        pass

    pid = _evo_runner_pid()
    if not _pid_alive(pid):
        return

    if not _pid_looks_like_evo_runner(pid):
        try:
            _atomic_write_json(_RUNS_DIR / "runner.pid", {"pid": 0, "stale_pid": int(pid), "cleared_at": float(time.time())})
        except Exception:
            pass
        return

    sent = False
    try:
        os.killpg(int(pid), signal.SIGTERM)
        sent = True
    except Exception:
        pass

    if not sent:
        try:
            os.kill(int(pid), signal.SIGTERM)
            sent = True
        except Exception:
            pass

    if not sent:
        return

    t0 = time.time()
    while time.time() - t0 < 1.5:
        if not _pid_alive(pid):
            try:
                os.waitpid(int(pid), os.WNOHANG)
            except Exception:
                pass
            return
        time.sleep(0.05)

    try:
        os.killpg(int(pid), signal.SIGKILL)
        sent = True
    except Exception:
        pass

    try:
        os.kill(int(pid), signal.SIGKILL)
        sent = True
    except Exception:
        pass

    if not sent:
        return

    t1 = time.time()
    while time.time() - t1 < 2.0:
        if not _pid_alive(pid):
            break
        time.sleep(0.05)

    try:
        os.waitpid(int(pid), os.WNOHANG)
    except Exception:
        pass

    if _pid_is_zombie(pid):
        try:
            os.waitpid(int(pid), os.WNOHANG)
        except Exception:
            pass

    if not _pid_alive(pid):
        try:
            cur = _safe_read_json(_RUNS_DIR / "runner.pid")
            if isinstance(cur, dict) and int(cur.get("pid") or 0) == int(pid):
                _atomic_write_json(_RUNS_DIR / "runner.pid", {"pid": 0, "started_at": float(cur.get("started_at") or 0.0), "cleared_at": float(time.time())})
        except Exception:
            pass


def _evo_status_from_disk() -> Dict[str, Any]:
    _evo_runs_ensure_dir()
    st = _safe_read_json(_evo_state_path())
    if isinstance(st, dict):
        pid = 0
        try:
            pid = int(st.get("runner_pid") or 0)
        except Exception:
            pid = 0
        if pid <= 0:
            pid = _evo_runner_pid()
        if pid and not _pid_alive(pid):
            st = dict(st)
            st["running"] = False
            st["runner_pid"] = int(pid)
        return st

    pid = _evo_runner_pid()
    if _pid_alive(pid):
        return {
            "ok": True,
            "job_id": "",
            "running": True,
            "error": "",
            "cfg": {},
            "progress": {},
            "history": {"best": [], "mean": [], "median": [], "p10": [], "p90": []},
            "baseline": {},
            "series": {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []},
            "perf": {},
            "top": [],
            "runner_pid": int(pid),
        }

    return {
        "ok": True,
        "job_id": "",
        "running": False,
        "error": "",
        "cfg": {},
        "progress": {},
        "history": {"best": [], "mean": [], "median": [], "p10": [], "p90": []},
        "baseline": {},
        "series": {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []},
        "perf": {},
        "top": [],
    }


def _evo_runner_pid() -> int:
    dd = _safe_read_json(_RUNS_DIR / "runner.pid")
    if isinstance(dd, dict):
        try:
            return int(dd.get("pid") or 0)
        except Exception:
            return 0
    return 0


def _pid_is_zombie(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="ignore")
        # /proc/<pid>/stat format: pid (comm) state ...
        parts = stat.split()
        if len(parts) >= 3 and parts[2] == "Z":
            return True
    except Exception:
        return False
    return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _pid_is_zombie(pid):
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _pid_looks_like_evo_runner(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        s = cmd.decode("utf-8", errors="ignore")
        return "evolution_runner.py" in s
    except Exception:
        return False


def _evo_build_candidate_payload(
    base: Dict[str, Any],
    cfg: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    genome = candidate.get("genome")
    if not isinstance(genome, dict):
        raise ValueError("candidate genome missing")

    huge = float((cfg or {}).get("huge") or 1e9)
    H = int(base.get("H") or 0)
    W = int(base.get("W") or 0)
    if H <= 0 or W <= 0:
        raise ValueError("base payload invalid H/W")

    kinds: Dict[str, str] = {}
    layer_meta = base.get("layers")
    if isinstance(layer_meta, list):
        for m in layer_meta:
            if not isinstance(m, dict):
                continue
            nm = m.get("name")
            if isinstance(nm, str) and nm:
                kinds[nm] = str(m.get("kind") or "continuous")

    cell_layer = _find_cell_layer_name(base)
    cell_mask = None
    if cell_layer:
        dd0 = base.get("data")
        if isinstance(dd0, dict):
            ent0 = dd0.get(cell_layer)
            if isinstance(ent0, dict) and ent0.get("dtype") == "float32" and isinstance(ent0.get("b64"), str):
                try:
                    cell_arr = _decode_float32_b64(
                        str(ent0.get("b64") or ""), expected_len=H * W, layer_name=cell_layer
                    )
                    cell_mask = np.asarray(cell_arr, dtype=np.float32).reshape(H * W) > 0.5
                except Exception:
                    cell_mask = None

    payload = _deepcopy_payload(base)
    payload.pop("event_counters", None)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("candidate payload missing data")

    for nm, gb in genome.items():
        if not isinstance(nm, str) or not nm:
            continue
        if not isinstance(gb, dict):
            continue
        ent = data.get(nm)
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        b64 = ent.get("b64")
        if not isinstance(b64, str) or not b64:
            continue
        arr = _decode_float32_b64(b64, expected_len=H * W, layer_name=nm)
        if "delta_b64" in gb and isinstance(gb.get("delta_b64"), str):
            db64 = str(gb.get("delta_b64") or "")
            delta = _decode_float32_b64(db64, expected_len=H * W, layer_name=f"{nm}:delta")
            arr2 = np.asarray(arr + delta, dtype=np.float32)
        else:
            s = float(gb.get("scale", 1.0))
            b = float(gb.get("bias", 0.0))
            arr2 = np.asarray(arr * s + b, dtype=np.float32)
            if isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr2.size:
                arr2[~cell_mask] = np.asarray(arr, dtype=np.float32).reshape(-1)[~cell_mask]
        arr2 = np.nan_to_num(arr2, nan=0.0, posinf=huge, neginf=0.0)
        arr2 = np.clip(arr2, 0.0, huge)
        if kinds.get(nm) == "counts":
            arr2 = np.clip(np.rint(arr2), 0.0, huge)
        ent["b64"] = _encode_float32_b64(arr2)

    return payload


_EVO_WORKER_BASE: Optional[Dict[str, Any]] = None
_EVO_WORKER_BASE_DATA: Optional[Dict[str, np.ndarray]] = None
_EVO_WORKER_PAYLOAD: Optional[Dict[str, Any]] = None
_EVO_WORKER_DATA: Optional[Dict[str, Any]] = None
_EVO_WORKER_KINDS: Optional[Dict[str, str]] = None
_EVO_WORKER_CELL_LAYER: str = ""
_EVO_WORKER_CELL_MASK: Optional[np.ndarray] = None
_EVO_WORKER_HUGE: float = 1e9

_EVO_CEM_MUTABLE_NAMES: Optional[list[str]] = None
_EVO_CEM_MU: Optional[np.ndarray] = None
_EVO_CEM_SIG: Optional[np.ndarray] = None
_EVO_CEM_MASK_F: Optional[np.ndarray] = None
_EVO_CEM_MU_SHM: Optional[shared_memory.SharedMemory] = None
_EVO_CEM_SIG_SHM: Optional[shared_memory.SharedMemory] = None


def _evo_worker_prepare_process() -> None:
    try:
        nice = int(os.environ.get("DT_WORKER_NICE", "10") or "10")
        if nice != 0:
            os.nice(int(nice))
    except Exception:
        pass

    try:
        port = int(os.environ.get("DT_RUNTIME_PORT", "8000") or "8000")
    except Exception:
        port = 8000

    try:
        for fd_name in os.listdir("/proc/self/fd"):
            try:
                fd = int(fd_name)
            except Exception:
                continue
            if fd < 3:
                continue
            try:
                s = socket.socket(fileno=fd)
            except Exception:
                continue
            try:
                if s.family not in (socket.AF_INET, socket.AF_INET6):
                    continue
                try:
                    acc = int(s.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) or 0)
                except Exception:
                    acc = 0
                if not acc:
                    continue
                try:
                    addr = s.getsockname()
                except Exception:
                    continue
                if isinstance(addr, tuple) and len(addr) >= 2 and int(addr[1]) == int(port):
                    try:
                        s.close()
                    except Exception:
                        pass
            finally:
                try:
                    if s.fileno() != -1:
                        s.detach()
                except Exception:
                    pass
    except Exception:
        pass


def _evo_worker_init(
    base_payload_fast: Dict[str, Any],
    kinds: Dict[str, str],
    cell_layer: str,
    huge: float,
) -> None:
    global _EVO_WORKER_BASE, _EVO_WORKER_BASE_DATA, _EVO_WORKER_PAYLOAD, _EVO_WORKER_DATA
    global _EVO_WORKER_KINDS, _EVO_WORKER_CELL_LAYER, _EVO_WORKER_CELL_MASK, _EVO_WORKER_HUGE

    _evo_worker_prepare_process()

    _EVO_WORKER_BASE = base_payload_fast
    _EVO_WORKER_KINDS = dict(kinds)
    _EVO_WORKER_CELL_LAYER = str(cell_layer)
    _EVO_WORKER_HUGE = float(huge)

    dd = base_payload_fast.get("data")
    if not isinstance(dd, dict):
        raise ValueError("worker init: base payload missing data")
    base_data: Dict[str, np.ndarray] = {}
    for nm, ent in dd.items():
        if not isinstance(ent, dict) or ent.get("dtype") != "float32":
            continue
        arr = ent.get("arr")
        if not isinstance(arr, np.ndarray):
            continue
        base_data[str(nm)] = np.asarray(arr, dtype=np.float32).reshape(-1)
    _EVO_WORKER_BASE_DATA = base_data

    if cell_layer and cell_layer in base_data:
        cm = np.asarray(base_data.get(cell_layer), dtype=np.float32).reshape(-1) > 0.5
        _EVO_WORKER_CELL_MASK = cm
    else:
        _EVO_WORKER_CELL_MASK = None

    p = dict(base_payload_fast)
    out_data: Dict[str, Any] = {}
    for nm, arr in base_data.items():
        out_data[nm] = {"dtype": "float32", "arr": arr.copy()}
    p["data"] = out_data
    p["_skip_b64_writeback"] = True
    _EVO_WORKER_PAYLOAD = p
    _EVO_WORKER_DATA = out_data


def _evo_fitness_from_metrics(metrics: Dict[str, Any], fitness_w: Dict[str, Any]) -> float:
    meas_weights = fitness_w.get("measurements") if isinstance(fitness_w, dict) else None
    if isinstance(meas_weights, dict):
        mm = metrics.get("measurements") if isinstance(metrics, dict) else None
        if isinstance(mm, dict):
            fit = 0.0
            for meas_name, w in meas_weights.items():
                if meas_name in mm and isinstance(w, (int, float)):
                    fit += float(w) * float(mm.get(meas_name) or 0.0)
            return float(fit)
        return 0.0
    return 0.0


def _evo_worker_eval_affine(
    gen: int,
    vi: int,
    genome: Dict[str, Dict[str, float]],
    seed: int,
    ticks: int,
    replicates: int,
    fitness_w: Dict[str, float],
) -> Dict[str, Any]:
    if _EVO_WORKER_PAYLOAD is None or _EVO_WORKER_DATA is None or _EVO_WORKER_BASE_DATA is None:
        raise RuntimeError("worker not initialized")

    kinds = _EVO_WORKER_KINDS or {}
    huge = float(_EVO_WORKER_HUGE)
    cell_layer = str(_EVO_WORKER_CELL_LAYER)
    cell_mask = _EVO_WORKER_CELL_MASK

    p = _EVO_WORKER_PAYLOAD
    dd = _EVO_WORKER_DATA

    rep_metrics = []
    rep_per_tick_events = []
    rep_measurements = []
    
    for ri in range(int(replicates)):
        seed0 = int(seed) + (int(gen) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)

        p.pop("event_counters", None)
        for nm, src in _EVO_WORKER_BASE_DATA.items():
            ent = dd.get(nm)
            if not isinstance(ent, dict):
                continue
            dst = ent.get("arr")
            if not isinstance(dst, np.ndarray):
                continue
            np.copyto(dst, src)

        for nm, gb in genome.items():
            if not isinstance(nm, str) or not nm:
                continue
            ent = dd.get(nm)
            if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                continue
            arr = ent.get("arr")
            if not isinstance(arr, np.ndarray):
                continue

            s = float(gb.get("scale", 1.0))
            b = float(gb.get("bias", 0.0))
            arr *= np.float32(s)
            arr += np.float32(b)
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=huge, neginf=0.0)
            np.clip(arr, 0.0, huge, out=arr)
            if kinds.get(nm) == "counts":
                np.rint(arr, out=arr)
                np.clip(arr, 0.0, huge, out=arr)

            if isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr.size:
                src = _EVO_WORKER_BASE_DATA.get(nm) if isinstance(_EVO_WORKER_BASE_DATA, dict) else None
                if isinstance(src, np.ndarray) and src.shape[0] == arr.size:
                    arr[~cell_mask] = src[~cell_mask]

        H = int(p.get("H") or 0)
        W = int(p.get("W") or 0)

        meas_aggs = fitness_w.get("measurement_aggs", {})
        if not isinstance(meas_aggs, dict):
            meas_aggs = {}
        agg_modes = {
            str(k): str(v)
            for k, v in meas_aggs.items()
            if isinstance(k, str) and str(v) in ("mean", "median", "std", "var")
        }
        agg_names = set(agg_modes.keys())
        per_tick_meas: Dict[str, list[float]] = {k: [] for k in agg_names}

        per_tick_divs = []
        per_tick_starv = []
        per_tick_dmg = []
        
        for t in range(int(ticks)):
            apply_layer_ops_inplace(p, seed_offset=seed0 + int(t))
            
            events = p.get("event_counters") if isinstance(p, dict) else None
            last_events = events.get("last") if isinstance(events, dict) else None
            if isinstance(last_events, dict):
                per_tick_divs.append(int(last_events.get("divisions") or 0))
                per_tick_starv.append(int(last_events.get("starvation_deaths") or 0))
                per_tick_dmg.append(int(last_events.get("damage_deaths") or 0))
            else:
                per_tick_divs.append(0)
                per_tick_starv.append(0)
                per_tick_dmg.append(0)

            if agg_names:
                layers_dict_tick: Dict[str, np.ndarray] = {}
                for nm2, ent2 in dd.items():
                    if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                        arr2 = ent2.get("arr")
                        if isinstance(arr2, np.ndarray):
                            layers_dict_tick[nm2] = arr2

                sel = _compute_selected_measurements_from_layers(p, layers_dict_tick, H, W, agg_names)
                for nm in agg_names:
                    try:
                        vv = float(sel.get(nm) or 0.0)
                        if not np.isfinite(vv):
                            vv = 0.0
                        per_tick_meas[nm].append(vv)
                    except Exception:
                        per_tick_meas[nm].append(0.0)

        cell_ent2 = dd.get(cell_layer)
        if not isinstance(cell_ent2, dict) or cell_ent2.get("dtype") != "float32":
            raise ValueError("payload missing cell layer")
        cell_arr = cell_ent2.get("arr")
        if not isinstance(cell_arr, np.ndarray):
            raise ValueError("payload missing cell layer array")

        alive = int((np.asarray(cell_arr, dtype=np.float32).reshape(-1) > 0.5).sum())
        events = p.get("event_counters") if isinstance(p, dict) else None
        totals = events.get("totals") if isinstance(events, dict) else None
        if not isinstance(totals, dict):
            totals = {}
        
        layers_dict_end: Dict[str, np.ndarray] = {}
        for nm2, ent2 in dd.items():
            if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                arr2 = ent2.get("arr")
                if isinstance(arr2, np.ndarray):
                    layers_dict_end[nm2] = arr2

        measurements = _compute_measurements_from_layers(p, layers_dict_end, H, W)
        if not isinstance(measurements, dict):
            measurements = {}
        if agg_names:
            for nm, mode in agg_modes.items():
                vals = per_tick_meas.get(nm) or []
                if not vals:
                    continue
                if mode == "mean":
                    measurements[nm] = float(np.mean(np.asarray(vals, dtype=np.float64)))
                elif mode == "median":
                    measurements[nm] = float(np.median(np.asarray(vals, dtype=np.float64)))
                elif mode == "std":
                    measurements[nm] = float(np.std(np.asarray(vals, dtype=np.float64)))
                elif mode == "var":
                    measurements[nm] = float(np.var(np.asarray(vals, dtype=np.float64)))
        
        rep_metrics.append(
            {
                "alive": alive,
                "divisions": int(totals.get("divisions") or 0),
                "starvation_deaths": int(totals.get("starvation_deaths") or 0),
                "damage_deaths": int(totals.get("damage_deaths") or 0),
            }
        )
        rep_per_tick_events.append(
            {
                "divisions": per_tick_divs,
                "starvation_deaths": per_tick_starv,
                "damage_deaths": per_tick_dmg,
            }
        )
        rep_measurements.append(measurements)

    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
    
    avg_per_tick_divs = []
    avg_per_tick_starv = []
    avg_per_tick_dmg = []
    if rep_per_tick_events:
        n_ticks = len(rep_per_tick_events[0]["divisions"])
        for t in range(n_ticks):
            avg_per_tick_divs.append(float(np.mean([r["divisions"][t] for r in rep_per_tick_events])))
            avg_per_tick_starv.append(float(np.mean([r["starvation_deaths"][t] for r in rep_per_tick_events])))
            avg_per_tick_dmg.append(float(np.mean([r["damage_deaths"][t] for r in rep_per_tick_events])))
    
    merged_measurements = {}
    if rep_measurements:
        all_keys = set()
        for m in rep_measurements:
            if isinstance(m, dict):
                all_keys.update(m.keys())
        for k in all_keys:
            vals = [float(m.get(k) or 0) for m in rep_measurements if isinstance(m, dict) and k in m]
            if vals:
                merged_measurements[k] = float(np.mean(vals))
    
    metrics = {
        "alive": int(round(alive_m)),
        "divisions": div_m,
        "starvation_deaths": starv_m,
        "damage_deaths": dmg_m,
        "per_tick_divisions": avg_per_tick_divs,
        "per_tick_starvation_deaths": avg_per_tick_starv,
        "per_tick_damage_deaths": avg_per_tick_dmg,
        "measurements": merged_measurements,
    }
    
    fit = _evo_fitness_from_metrics(metrics, fitness_w)

    return {
        "vi": int(vi),
        "genome": genome,
        "metrics": metrics,
        "fitness": float(fit),
        "evals_done": int(replicates),
    }


def _evo_worker_init_cem_delta(
    base_payload_fast: Dict[str, Any],
    kinds: Dict[str, str],
    cell_layer: str,
    huge: float,
    mu_shm_name: str,
    sig_shm_name: str,
    k_layers: int,
    n_cells: int,
    mutable_names: list[str],
) -> None:
    global _EVO_CEM_MUTABLE_NAMES, _EVO_CEM_MU, _EVO_CEM_SIG, _EVO_CEM_MASK_F
    global _EVO_CEM_MU_SHM, _EVO_CEM_SIG_SHM

    _evo_worker_init(base_payload_fast, kinds, cell_layer, huge)
    _EVO_CEM_MUTABLE_NAMES = list(mutable_names)

    if _EVO_WORKER_BASE_DATA is not None and cell_layer in _EVO_WORKER_BASE_DATA:
        cm = np.asarray(_EVO_WORKER_BASE_DATA.get(cell_layer), dtype=np.float32).reshape(-1) > 0.5
        if int(n_cells) > 0:
            cm = cm[: int(n_cells)]
        _EVO_CEM_MASK_F = cm.astype(np.float32)
    else:
        _EVO_CEM_MASK_F = None

    _EVO_CEM_MU_SHM = shared_memory.SharedMemory(name=str(mu_shm_name))
    _EVO_CEM_SIG_SHM = shared_memory.SharedMemory(name=str(sig_shm_name))
    _EVO_CEM_MU = np.ndarray(
        (int(k_layers), int(n_cells)), dtype=np.float32, buffer=_EVO_CEM_MU_SHM.buf
    )
    _EVO_CEM_SIG = np.ndarray(
        (int(k_layers), int(n_cells)), dtype=np.float32, buffer=_EVO_CEM_SIG_SHM.buf
    )


def _evo_worker_eval_cem_delta(
    gen: int,
    vi: int,
    seed: int,
    ticks: int,
    replicates: int,
    fitness_w: Dict[str, float],
    use_cell_mask: bool,
) -> Dict[str, Any]:
    if _EVO_WORKER_PAYLOAD is None or _EVO_WORKER_DATA is None or _EVO_WORKER_BASE_DATA is None:
        raise RuntimeError("worker not initialized")
    if _EVO_CEM_MUTABLE_NAMES is None or _EVO_CEM_MU is None or _EVO_CEM_SIG is None:
        raise RuntimeError("cem worker not initialized")

    kinds = _EVO_WORKER_KINDS or {}
    huge = float(_EVO_WORKER_HUGE)
    cell_layer = str(_EVO_WORKER_CELL_LAYER)

    p = _EVO_WORKER_PAYLOAD
    dd = _EVO_WORKER_DATA

    mu = _EVO_CEM_MU
    sig = _EVO_CEM_SIG
    mask_f = _EVO_CEM_MASK_F if bool(use_cell_mask) else None
    n_cells = int(mu.shape[1])

    rr = np.random.default_rng(int(seed) + 1234567 + (int(gen) * 1000003) + (int(vi) * 1009))
    deltas: list[np.ndarray] = []
    for i, _nm in enumerate(_EVO_CEM_MUTABLE_NAMES):
        eps = rr.normal(0.0, 1.0, size=(n_cells,)).astype(np.float32)
        delta = np.asarray(mu[i] + sig[i] * eps, dtype=np.float32)
        if mask_f is not None:
            delta = np.asarray(delta * mask_f, dtype=np.float32)
        deltas.append(delta)

    rep_metrics = []
    rep_per_tick_events = []
    rep_measurements = []
    
    for ri in range(int(replicates)):
        seed0 = int(seed) + (int(gen) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)

        p.pop("event_counters", None)
        for nm, src in _EVO_WORKER_BASE_DATA.items():
            ent = dd.get(nm)
            if not isinstance(ent, dict):
                continue
            dst = ent.get("arr")
            if not isinstance(dst, np.ndarray):
                continue
            np.copyto(dst, src)

        for i, nm in enumerate(_EVO_CEM_MUTABLE_NAMES):
            ent = dd.get(nm)
            if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                continue
            arr = ent.get("arr")
            if not isinstance(arr, np.ndarray):
                continue

            arr += deltas[int(i)]
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=huge, neginf=0.0)
            np.clip(arr, 0.0, huge, out=arr)

        H = int(p.get("H") or 0)
        W = int(p.get("W") or 0)

        meas_aggs = fitness_w.get("measurement_aggs", {})
        if not isinstance(meas_aggs, dict):
            meas_aggs = {}
        agg_modes = {
            str(k): str(v)
            for k, v in meas_aggs.items()
            if isinstance(k, str) and str(v) in ("mean", "median", "std", "var")
        }
        agg_names = set(agg_modes.keys())
        per_tick_meas: Dict[str, list[float]] = {k: [] for k in agg_names}

        per_tick_divs = []
        per_tick_starv = []
        per_tick_dmg = []
        
        for t in range(int(ticks)):
            apply_layer_ops_inplace(p, seed_offset=seed0 + int(t))
            
            events = p.get("event_counters") if isinstance(p, dict) else None
            last_events = events.get("last") if isinstance(events, dict) else None
            if isinstance(last_events, dict):
                per_tick_divs.append(int(last_events.get("divisions") or 0))
                per_tick_starv.append(int(last_events.get("starvation_deaths") or 0))
                per_tick_dmg.append(int(last_events.get("damage_deaths") or 0))
            else:
                per_tick_divs.append(0)
                per_tick_starv.append(0)
                per_tick_dmg.append(0)

            if agg_names:
                layers_dict_tick: Dict[str, np.ndarray] = {}
                for nm2, ent2 in dd.items():
                    if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                        arr2 = ent2.get("arr")
                        if isinstance(arr2, np.ndarray):
                            layers_dict_tick[nm2] = arr2

                sel = _compute_selected_measurements_from_layers(p, layers_dict_tick, int(H), int(W), agg_names)
                for nm in agg_names:
                    try:
                        vv = float(sel.get(nm) or 0.0)
                        if not np.isfinite(vv):
                            vv = 0.0
                        per_tick_meas[nm].append(vv)
                    except Exception:
                        per_tick_meas[nm].append(0.0)

        cell_ent2 = dd.get(cell_layer)
        if not isinstance(cell_ent2, dict) or cell_ent2.get("dtype") != "float32":
            raise ValueError("payload missing cell layer")
        cell_arr = cell_ent2.get("arr")
        if not isinstance(cell_arr, np.ndarray):
            raise ValueError("payload missing cell layer array")

        alive = int((np.asarray(cell_arr, dtype=np.float32).reshape(-1) > 0.5).sum())
        events = p.get("event_counters") if isinstance(p, dict) else None
        totals = events.get("totals") if isinstance(events, dict) else None
        if not isinstance(totals, dict):
            totals = {}
        
        layers_dict_end: Dict[str, np.ndarray] = {}
        for nm2, ent2 in dd.items():
            if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                arr2 = ent2.get("arr")
                if isinstance(arr2, np.ndarray):
                    layers_dict_end[nm2] = arr2

        measurements = _compute_measurements_from_layers(p, layers_dict_end, H, W)
        if not isinstance(measurements, dict):
            measurements = {}
        if agg_names:
            for nm, mode in agg_modes.items():
                vals = per_tick_meas.get(nm) or []
                if not vals:
                    continue
                if mode == "mean":
                    measurements[nm] = float(np.mean(np.asarray(vals, dtype=np.float64)))
                elif mode == "median":
                    measurements[nm] = float(np.median(np.asarray(vals, dtype=np.float64)))
                elif mode == "std":
                    measurements[nm] = float(np.std(np.asarray(vals, dtype=np.float64)))
                elif mode == "var":
                    measurements[nm] = float(np.var(np.asarray(vals, dtype=np.float64)))
        
        rep_metrics.append(
            {
                "alive": alive,
                "divisions": int(totals.get("divisions") or 0),
                "starvation_deaths": int(totals.get("starvation_deaths") or 0),
                "damage_deaths": int(totals.get("damage_deaths") or 0),
            }
        )
        rep_per_tick_events.append(
            {
                "divisions": per_tick_divs,
                "starvation_deaths": per_tick_starv,
                "damage_deaths": per_tick_dmg,
            }
        )
        rep_measurements.append(measurements)

    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
    
    avg_per_tick_divs = []
    avg_per_tick_starv = []
    avg_per_tick_dmg = []
    if rep_per_tick_events:
        n_ticks = len(rep_per_tick_events[0]["divisions"])
        for t in range(n_ticks):
            avg_per_tick_divs.append(float(np.mean([r["divisions"][t] for r in rep_per_tick_events])))
            avg_per_tick_starv.append(float(np.mean([r["starvation_deaths"][t] for r in rep_per_tick_events])))
            avg_per_tick_dmg.append(float(np.mean([r["damage_deaths"][t] for r in rep_per_tick_events])))
    
    merged_measurements = {}
    if rep_measurements:
        all_keys = set()
        for m in rep_measurements:
            if isinstance(m, dict):
                all_keys.update(m.keys())
        for k in all_keys:
            vals = [float(m.get(k) or 0) for m in rep_measurements if isinstance(m, dict) and k in m]
            if vals:
                merged_measurements[k] = float(np.mean(vals))
    
    metrics = {
        "alive": int(round(alive_m)),
        "divisions": div_m,
        "starvation_deaths": starv_m,
        "damage_deaths": dmg_m,
        "per_tick_divisions": avg_per_tick_divs,
        "per_tick_starvation_deaths": avg_per_tick_starv,
        "per_tick_damage_deaths": avg_per_tick_dmg,
        "measurements": merged_measurements,
    }
    
    fit = _evo_fitness_from_metrics(metrics, fitness_w)

    return {"vi": int(vi), "metrics": metrics, "fitness": float(fit), "evals_done": int(replicates)}


def _evo_worker_eval_cem_delta_batch(
    gen: int,
    vis: list[int],
    seed: int,
    ticks: int,
    replicates: int,
    fitness_w: Dict[str, float],
    use_cell_mask: bool,
) -> Dict[str, Any]:
    out: list[Dict[str, Any]] = []
    evals_done_total = 0
    for vi in vis:
        res = _evo_worker_eval_cem_delta(
            int(gen),
            int(vi),
            int(seed),
            int(ticks),
            int(replicates),
            dict(fitness_w),
            bool(use_cell_mask),
        )
        if isinstance(res, dict):
            out.append(res)
            evals_done_total += int(res.get("evals_done") or 0)
    return {"results": out, "evals_done": int(evals_done_total)}


class _EvolutionJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.job_id: str = ""
        self.running: bool = False
        self.error: str = ""

        self.auto: Dict[str, Any] = {}

        self._base_payload: Optional[Dict[str, Any]] = None

        self.baseline: Dict[str, Any] = {}
        self.series: Dict[str, Any] = {
            "offset": 0,
            "fitness": [],
            "best": [],
            "mean": [],
            "median": [],
        }
        self._series_sum: float = 0.0
        self._series_n: int = 0
        self._series_best: float = float("-inf")

        self.cfg: Dict[str, Any] = {}
        self.progress: Dict[str, Any] = {
            "generation": 0,
            "variant": 0,
            "total_generations": 0,
            "total_variants": 0,
            "evaluations_done": 0,
            "evaluations_total": 0,
            "started_at": 0.0,
            "updated_at": 0.0,
        }

        self.history: Dict[str, list] = {
            "best": [],
            "mean": [],
            "median": [],
            "p10": [],
            "p90": [],
        }

        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.top_ids: list[str] = []

        self.perf: Dict[str, Any] = {
            "evals": 0,
            "apply_s": 0.0,
            "apply_copy_s": 0.0,
            "apply_decode_s": 0.0,
            "apply_math_s": 0.0,
            "apply_encode_s": 0.0,
            "sample_s": 0.0,
            "tick_by_type_s": {},
            "ticks_s": 0.0,
            "decode_cell_s": 0.0,
            "total_s": 0.0,
        }

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            top = []
            for cid in self.top_ids[:10]:
                c = self.candidates.get(cid) or {}
                top.append(
                    {
                        "id": cid,
                        "fitness": c.get("fitness"),
                        "metrics": c.get("metrics"),
                        "gen": c.get("gen"),
                    }
                )
            baseline = dict(self.baseline) if isinstance(self.baseline, dict) else {}
            series = self.series if isinstance(self.series, dict) else {}
            series_out = {
                "offset": int(series.get("offset") or 0),
                "fitness": list(series.get("fitness") or []),
                "best": list(series.get("best") or []),
                "mean": list(series.get("mean") or []),
                "median": list(series.get("median") or []),
            }
            history = self.history if isinstance(self.history, dict) else {}
            history_out = {
                "best": list(history.get("best") or []),
                "mean": list(history.get("mean") or []),
                "median": list(history.get("median") or []),
                "p10": list(history.get("p10") or []),
                "p90": list(history.get("p90") or []),
            }
            perf_out = dict(self.perf) if isinstance(self.perf, dict) else {}
            return {
                "ok": True,
                "job_id": self.job_id,
                "running": self.running,
                "error": self.error,
                "cfg": self.cfg,
                "auto": dict(self.auto) if isinstance(self.auto, dict) else {},
                "progress": dict(self.progress),
                "history": history_out,
                "baseline": baseline,
                "series": series_out,
                "perf": perf_out,
                "top": top,
            }

    def candidate(self, candidate_id: str) -> Dict[str, Any]:
        with self._lock:
            c = self.candidates.get(candidate_id)
            if not c:
                raise ValueError("unknown candidate")

            base = self._base_payload
            if not isinstance(base, dict):
                raise ValueError("evolution base payload missing")

            genome = c.get("genome")
            if not isinstance(genome, dict):
                raise ValueError("candidate genome missing")

            huge = float((self.cfg or {}).get("huge") or 1e9)
            H = int(base.get("H") or 0)
            W = int(base.get("W") or 0)
            if H <= 0 or W <= 0:
                raise ValueError("base payload invalid H/W")

            kinds: Dict[str, str] = {}
            layer_meta = base.get("layers")
            if isinstance(layer_meta, list):
                for m in layer_meta:
                    if not isinstance(m, dict):
                        continue
                    nm = m.get("name")
                    if isinstance(nm, str) and nm:
                        kinds[nm] = str(m.get("kind") or "continuous")

            cell_layer = _find_cell_layer_name(base)
            cell_mask = None
            if cell_layer:
                dd0 = base.get("data")
                if isinstance(dd0, dict):
                    ent0 = dd0.get(cell_layer)
                    if isinstance(ent0, dict) and ent0.get("dtype") == "float32" and isinstance(ent0.get("b64"), str):
                        try:
                            cell_arr = _decode_float32_b64(str(ent0.get("b64") or ""), expected_len=H * W, layer_name=cell_layer)
                            cell_mask = np.asarray(cell_arr, dtype=np.float32).reshape(H * W) > 0.5
                        except Exception:
                            cell_mask = None

            payload = _deepcopy_payload(base)
            payload.pop("event_counters", None)
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("candidate payload missing data")

            for nm, gb in genome.items():
                if not isinstance(nm, str) or not nm:
                    continue
                if not isinstance(gb, dict):
                    continue
                ent = data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if not isinstance(b64, str) or not b64:
                    continue
                arr = _decode_float32_b64(b64, expected_len=H * W, layer_name=nm)
                if "delta_b64" in gb and isinstance(gb.get("delta_b64"), str):
                    db64 = str(gb.get("delta_b64") or "")
                    delta = _decode_float32_b64(db64, expected_len=H * W, layer_name=f"{nm}:delta")
                    arr2 = np.asarray(arr + delta, dtype=np.float32)
                else:
                    s = float(gb.get("scale", 1.0))
                    b = float(gb.get("bias", 0.0))
                    arr2 = np.asarray(arr * s + b, dtype=np.float32)
                    if isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr2.size:
                        arr2[~cell_mask] = np.asarray(arr, dtype=np.float32).reshape(-1)[~cell_mask]
                arr2 = np.nan_to_num(arr2, nan=0.0, posinf=huge, neginf=0.0)
                arr2 = np.clip(arr2, 0.0, huge)
                if kinds.get(nm) == "counts":
                    arr2 = np.clip(np.rint(arr2), 0.0, huge)
                ent["b64"] = _encode_float32_b64(arr2)
            return {
                "ok": True,
                "id": candidate_id,
                "fitness": c.get("fitness"),
                "metrics": c.get("metrics"),
                "genome": c.get("genome"),
                "payload": payload,
            }

    def start(self, base_payload: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        if self.running:
            raise ValueError("evolution already running")

        self._stop.clear()
        self.job_id = str(uuid.uuid4())
        self.running = True
        self.error = ""
        self.cfg = cfg
        self._base_payload = _deepcopy_payload(base_payload)
        self.auto = {}
        self.candidates = {}
        self.top_ids = []
        self.history = {"best": [], "mean": [], "median": [], "p10": [], "p90": []}
        self.baseline = {}
        self.series = {"offset": 0, "fitness": [], "best": [], "mean": [], "median": []}
        self._series_sum = 0.0
        self._series_n = 0
        self._series_best = float("-inf")
        self.perf = {
            "evals": 0,
            "apply_s": 0.0,
            "apply_copy_s": 0.0,
            "apply_decode_s": 0.0,
            "apply_math_s": 0.0,
            "apply_encode_s": 0.0,
            "sample_s": 0.0,
            "tick_by_type_s": {},
            "ticks_s": 0.0,
            "decode_cell_s": 0.0,
            "total_s": 0.0,
        }
        self.progress = {
            "generation": 0,
            "variant": 0,
            "total_generations": int(cfg.get("generations") or 0),
            "total_variants": int(cfg.get("variants") or 0),
            "evaluations_done": 0,
            "evaluations_total": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
        }

        self._thread = threading.Thread(
            target=self._run,
            args=(base_payload, cfg),
            daemon=True,
        )
        self._thread.start()

    def _run(self, base_payload: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        try:
            self._run_impl(base_payload, cfg)
        except Exception as e:
            with self._lock:
                self.error = str(e)
        finally:
            with self._lock:
                self.running = False
                self.progress["updated_at"] = time.time()

    def _run_impl(self, base_payload: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        algo = str(cfg.get("algo") or "affine")
        variants = int(cfg.get("variants") or 0)
        generations = int(cfg.get("generations") or 0)
        ticks = int(cfg.get("ticks") or 0)
        elites = int(cfg.get("elites") or 0)
        replicates = int(cfg.get("replicates") or 1)
        workers = int(cfg.get("workers") or 0)
        cem_sigma_init = float(cfg.get("cem_sigma_init") or 0.5)
        cem_alpha = float(cfg.get("cem_alpha") or 0.7)
        cem_sigma_floor = float(cfg.get("cem_sigma_floor") or 0.05)
        cem_mask = str(cfg.get("cem_mask") or "cell")
        cem_batch = int(cfg.get("cem_batch") or 0)
        seed = int(cfg.get("seed") or 0)
        mut_rate = float(cfg.get("mutation_rate") or 0.15)
        sigma_scale = float(cfg.get("sigma_scale") or 0.25)
        sigma_bias = float(cfg.get("sigma_bias") or 0.25)
        huge = float(cfg.get("huge") or 1e9)
        fitness_w = cfg.get("fitness_weights")
        if not isinstance(fitness_w, dict):
            fitness_w = {}
        meas_weights = fitness_w.get("measurements")
        if not isinstance(meas_weights, dict):
            meas_weights = {}

        active_w = [
            float(w)
            for w in meas_weights.values()
            if isinstance(w, (int, float)) and float(w) != 0.0
        ]
        if not active_w:
            raise ValueError("no active fitness objectives: set at least one non-zero measurement weight")

        if variants <= 0 or generations <= 0 or ticks <= 0:
            raise ValueError("variants/generations/ticks must be > 0")
        elites = max(1, min(variants, elites if elites > 0 else min(10, variants)))
        replicates = max(1, min(50, replicates))
        if workers <= 0:
            workers = max(1, int(min(4, os.cpu_count() or 1)))
        workers = max(1, min(32, workers))

        worker_mode = str(cfg.get("worker_mode") or "process").strip().lower()
        if worker_mode not in ("thread", "process"):
            worker_mode = "process"

        H, W, base_layers, kinds = _decoded_layers_and_kinds(base_payload)
        if H <= 0 or W <= 0:
            raise ValueError("payload invalid H/W")

        data = base_payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("payload missing data")

        cell_layer = _find_cell_layer_name(base_payload)
        if not cell_layer:
            raise ValueError("payload missing cell layer (expected 'cell' or 'cell_type')")

        cell0 = base_layers.get(cell_layer)
        if not isinstance(cell0, np.ndarray):
            raise ValueError("invalid cell layer")
        cell_mask = cell0.reshape(H * W) > 0.5

        # Get target layers from config (list of layer names or glob patterns)
        # If not specified, default to gene_*, rna_*, protein_*
        target_layers_cfg = cfg.get("target_layers")
        if isinstance(target_layers_cfg, list) and len(target_layers_cfg) > 0:
            target_patterns = [str(p).strip() for p in target_layers_cfg if isinstance(p, str) and str(p).strip()]
        else:
            target_patterns = ["gene_*", "rna_*", "protein_*"]

        def _matches_any_pattern(name: str, patterns: list) -> bool:
            import fnmatch
            for pat in patterns:
                if fnmatch.fnmatch(name, pat):
                    return True
            return False

        mutable_names: list[str] = []
        layer_stats: Dict[str, Dict[str, float]] = {}
        for name, ent in data.items():
            if not isinstance(name, str):
                continue
            if not _matches_any_pattern(name, target_patterns):
                continue
            if not isinstance(ent, dict) or ent.get("dtype") != "float32" or not isinstance(ent.get("b64"), str):
                continue
            mutable_names.append(name)
            try:
                arr = base_layers.get(name)
                if not isinstance(arr, np.ndarray):
                    raise ValueError("missing layer array")
                v = arr[cell_mask]
                if v.size <= 1:
                    mu = float(arr.mean())
                    sd = float(arr.std())
                else:
                    mu = float(v.mean())
                    sd = float(v.std())
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1.0
                layer_stats[name] = {"mean": mu, "std": sd}
            except Exception:
                layer_stats[name] = {"mean": 0.0, "std": 1.0}

        if not mutable_names:
            raise ValueError("no gene_/rna_/protein_ layers found to mutate")
        
        if _EVO_DEBUG:
            try:
                _LOG.info("DEBUG CEM: Found %s mutable layers", str(len(mutable_names)))
                for nm in mutable_names[:5]:
                    st = layer_stats.get(nm, {})
                    _LOG.info(
                        "DEBUG CEM: Layer '%s' - mean=%0.4f, std=%0.4f",
                        str(nm),
                        float(st.get("mean", 0.0) or 0.0),
                        float(st.get("std", 1.0) or 1.0),
                    )
            except Exception:
                pass

        eval_total = generations * variants * replicates
        with self._lock:
            self.progress["evaluations_total"] = int(eval_total)
            self.progress["updated_at"] = time.time()

        rng = np.random.default_rng(seed)

        base_payload_fast: Dict[str, Any] = {k: v for k, v in base_payload.items() if k not in ("data", "event_counters")}
        fast_data: Dict[str, Any] = {}
        for nm, ent in data.items():
            if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                continue
            arr0 = base_layers.get(nm)
            if not isinstance(arr0, np.ndarray):
                continue
            fast_data[nm] = {"dtype": "float32", "arr": np.asarray(arr0, dtype=np.float32).reshape(H * W)}
        if not fast_data:
            raise ValueError("no float32 layers found")
        base_payload_fast["data"] = fast_data
        base_payload_fast["_skip_b64_writeback"] = True

        tls = threading.local()

        def _get_thread_workspace() -> tuple[Dict[str, Any], Dict[str, Any]]:
            ws = getattr(tls, "ws", None)
            if isinstance(ws, tuple) and len(ws) == 2:
                p0, d0 = ws
                if isinstance(p0, dict) and isinstance(d0, dict):
                    return p0, d0

            p = dict(base_payload_fast)
            dd = p.get("data")
            assert isinstance(dd, dict)
            out_data: Dict[str, Any] = {}
            for nm, ent in dd.items():
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                arr = ent.get("arr")
                if not isinstance(arr, np.ndarray):
                    continue
                out_data[nm] = {"dtype": "float32", "arr": arr.copy()}
            p["data"] = out_data
            tls.ws = (p, out_data)
            return p, out_data

        def _copy_payload_fast() -> Dict[str, Any]:
            p, out_data = _get_thread_workspace()
            p.pop("event_counters", None)
            for nm, ent in out_data.items():
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                dst = ent.get("arr")
                src_ent = fast_data.get(nm)
                if not isinstance(dst, np.ndarray) or not isinstance(src_ent, dict):
                    continue
                src = src_ent.get("arr")
                if not isinstance(src, np.ndarray):
                    continue
                np.copyto(dst, src)
            return p

        def _perf_add(
            apply_s: float,
            ticks_s: float,
            decode_cell_s: float,
            total_s: float,
            apply_copy_s: float = 0.0,
            apply_decode_s: float = 0.0,
            apply_math_s: float = 0.0,
            apply_encode_s: float = 0.0,
            sample_s: float = 0.0,
            tick_by_type_s: Optional[Dict[str, float]] = None,
        ) -> None:
            with self._lock:
                p = self.perf if isinstance(self.perf, dict) else {}
                p["evals"] = int(p.get("evals") or 0) + 1
                p["apply_s"] = float(p.get("apply_s") or 0.0) + float(apply_s)
                p["apply_copy_s"] = float(p.get("apply_copy_s") or 0.0) + float(apply_copy_s)
                p["apply_decode_s"] = float(p.get("apply_decode_s") or 0.0) + float(apply_decode_s)
                p["apply_math_s"] = float(p.get("apply_math_s") or 0.0) + float(apply_math_s)
                p["apply_encode_s"] = float(p.get("apply_encode_s") or 0.0) + float(apply_encode_s)
                p["sample_s"] = float(p.get("sample_s") or 0.0) + float(sample_s)

                if isinstance(tick_by_type_s, dict):
                    cur = p.get("tick_by_type_s")
                    if not isinstance(cur, dict):
                        cur = {}
                    for k, v in tick_by_type_s.items():
                        if not isinstance(k, str) or not k:
                            continue
                        try:
                            dv = float(v)
                        except Exception:
                            continue
                        cur[k] = float(cur.get(k) or 0.0) + dv
                    p["tick_by_type_s"] = cur

                p["ticks_s"] = float(p.get("ticks_s") or 0.0) + float(ticks_s)
                p["decode_cell_s"] = float(p.get("decode_cell_s") or 0.0) + float(decode_cell_s)
                p["total_s"] = float(p.get("total_s") or 0.0) + float(total_s)
                self.perf = p

        def _mutate_genome(parent: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
            g = {k: {"scale": float(v.get("scale", 1.0)), "bias": float(v.get("bias", 0.0))} for k, v in parent.items()}
            for nm in mutable_names:
                if rng.random() > mut_rate:
                    continue
                cur = g.get(nm) or {"scale": 1.0, "bias": 0.0}
                cur_scale = float(cur.get("scale", 1.0))
                cur_bias = float(cur.get("bias", 0.0))
                cur_scale *= float(np.exp(rng.normal(0.0, sigma_scale)))
                cur_bias += float(rng.normal(0.0, sigma_bias) * layer_stats.get(nm, {}).get("std", 1.0))
                g[nm] = {"scale": cur_scale, "bias": cur_bias}
            return g

        def _apply_genome_to_payload(
            payload: Dict[str, Any], genome: Dict[str, Any]
        ) -> tuple[Dict[str, Any], float, float, float, float]:
            t_copy0 = time.perf_counter()
            out = _deepcopy_payload(payload)
            t_copy = time.perf_counter() - t_copy0
            out.pop("event_counters", None)
            out_data = out.get("data")
            if not isinstance(out_data, dict):
                raise ValueError("payload missing data")

            t_decode = 0.0
            t_math = 0.0
            t_encode = 0.0
            for nm, gb in genome.items():
                ent = out_data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if not isinstance(b64, str) or not b64:
                    continue

                td0 = time.perf_counter()
                arr = _decode_float32_b64(b64, expected_len=H * W, layer_name=nm)
                t_decode += time.perf_counter() - td0
                arr2: np.ndarray

                tm0 = time.perf_counter()
                if isinstance(gb, dict) and "delta" in gb and isinstance(gb.get("delta"), np.ndarray):
                    delta = gb.get("delta")
                    arr2 = arr + np.asarray(delta, dtype=np.float32)
                elif isinstance(gb, dict) and "delta_b64" in gb and isinstance(gb.get("delta_b64"), str):
                    delta = _decode_float32_b64(str(gb.get("delta_b64") or ""), expected_len=H * W, layer_name=f"{nm}:delta")
                    arr2 = arr + delta
                else:
                    if not isinstance(gb, dict):
                        continue
                    s = float(gb.get("scale", 1.0))
                    b = float(gb.get("bias", 0.0))
                    arr2 = arr * s + b
                arr2 = np.asarray(arr2, dtype=np.float32)
                arr2 = np.nan_to_num(arr2, nan=0.0, posinf=huge, neginf=0.0)
                arr2 = np.clip(arr2, 0.0, huge)
                if kinds.get(nm) == "counts":
                    arr2 = np.clip(np.rint(arr2), 0.0, huge)
                t_math += time.perf_counter() - tm0

                te0 = time.perf_counter()
                ent["b64"] = _encode_float32_b64(arr2)
                t_encode += time.perf_counter() - te0
            return out, float(t_copy), float(t_decode), float(t_math), float(t_encode)

        def _apply_genome_to_payload_fast(genome: Dict[str, Any]) -> tuple[Dict[str, Any], float, float, float, float]:
            t_copy0 = time.perf_counter()
            out = _copy_payload_fast()
            t_copy = time.perf_counter() - t_copy0

            out_data = out.get("data")
            if not isinstance(out_data, dict):
                raise ValueError("payload missing data")

            t_math0 = time.perf_counter()
            for nm, gb in genome.items():
                if not isinstance(nm, str) or not nm:
                    continue
                ent = out_data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                arr = ent.get("arr")
                if not isinstance(arr, np.ndarray):
                    continue

                if isinstance(gb, dict) and "delta" in gb and isinstance(gb.get("delta"), np.ndarray):
                    delta = np.asarray(gb.get("delta"), dtype=np.float32).reshape(H * W)
                    arr += delta
                else:
                    if not isinstance(gb, dict):
                        continue
                    s = float(gb.get("scale", 1.0))
                    b = float(gb.get("bias", 0.0))
                    arr *= np.float32(s)
                    arr += np.float32(b)

                np.nan_to_num(arr, copy=False, nan=0.0, posinf=huge, neginf=0.0)
                np.clip(arr, 0.0, huge, out=arr)
                if kinds.get(nm) == "counts":
                    np.rint(arr, out=arr)
                    np.clip(arr, 0.0, huge, out=arr)

                if isinstance(gb, dict) and "delta" not in gb and isinstance(cell_mask, np.ndarray) and cell_mask.shape[0] == arr.size:
                    src_ent = fast_data.get(nm)
                    if isinstance(src_ent, dict):
                        src = src_ent.get("arr")
                        if isinstance(src, np.ndarray) and src.shape[0] == arr.size:
                            arr[~cell_mask] = src[~cell_mask]

                ent["arr"] = np.asarray(arr, dtype=np.float32).reshape(H * W)

            t_math = time.perf_counter() - t_math0
            return out, float(t_copy), 0.0, float(t_math), 0.0

        profile_ticks = bool(cfg.get("profile_ticks") or cfg.get("profile_layer_ops"))

        meas_aggs = fitness_w.get("measurement_aggs", {})
        if not isinstance(meas_aggs, dict):
            meas_aggs = {}
        agg_modes = {
            str(k): str(v)
            for k, v in meas_aggs.items()
            if isinstance(k, str) and str(v) in ("mean", "median", "std", "var")
        }
        agg_names = set(agg_modes.keys())

        def _eval_genome(genome: Dict[str, Any], seed0: int, sample_s: float = 0.0) -> Dict[str, Any]:
            t_total0 = time.perf_counter()
            t_apply0 = time.perf_counter()
            p, t_copy, t_apply_decode, t_math, t_encode = _apply_genome_to_payload_fast(genome)
            t_apply = time.perf_counter() - t_apply0

            lop_cfg = p.get("layer_ops_config") if isinstance(p, dict) else None
            if isinstance(lop_cfg, dict):
                if "opt_env_cache" not in lop_cfg and "optimize_env_cache" not in lop_cfg:
                    lop_cfg["opt_env_cache"] = True
                if "opt_expr_cache" not in lop_cfg and "optimize_expr_cache" not in lop_cfg:
                    lop_cfg["opt_expr_cache"] = True

                if not profile_ticks:
                    for k in ("profile_expr", "profile_step_names", "profile_deep"):
                        if k in lop_cfg:
                            lop_cfg[k] = False

            if profile_ticks:
                p["_profile_layer_ops"] = True

            dd0 = p.get("data")
            if not isinstance(dd0, dict):
                raise ValueError("payload missing data")
            per_tick_meas: Dict[str, list[float]] = {k: [] for k in agg_names}

            t_ticks = 0.0
            for t in range(ticks):
                if self._stop.is_set():
                    raise RuntimeError("stopped")
                tt0 = time.perf_counter()
                apply_layer_ops_inplace(p, seed_offset=seed0 + t)
                t_ticks += time.perf_counter() - tt0

                if agg_names:
                    layers_dict_tick: Dict[str, np.ndarray] = {}
                    for nm2, ent2 in dd0.items():
                        if isinstance(ent2, dict) and ent2.get("dtype") == "float32":
                            arr2 = ent2.get("arr")
                            if isinstance(arr2, np.ndarray):
                                layers_dict_tick[str(nm2)] = arr2

                    sel = _compute_selected_measurements_from_layers(p, layers_dict_tick, int(H), int(W), agg_names)
                    for nm in agg_names:
                        try:
                            vv = float(sel.get(nm) or 0.0)
                            if not np.isfinite(vv):
                                vv = 0.0
                            per_tick_meas[nm].append(vv)
                        except Exception:
                            per_tick_meas[nm].append(0.0)
            dd = p.get("data")
            if not isinstance(dd, dict):
                raise ValueError("payload missing data")
            cell_ent2 = dd.get(cell_layer)
            if not isinstance(cell_ent2, dict) or cell_ent2.get("dtype") != "float32":
                raise ValueError("payload missing cell layer")

            t_dec0 = time.perf_counter()
            cell_arr0 = cell_ent2.get("arr")
            if isinstance(cell_arr0, np.ndarray):
                cell_arr = np.asarray(cell_arr0, dtype=np.float32).reshape(H * W)
                t_cell_decode = time.perf_counter() - t_dec0
            else:
                cell_arr = _decode_float32_b64(str(cell_ent2.get("b64") or ""), expected_len=H * W, layer_name=cell_layer)
                t_cell_decode = time.perf_counter() - t_dec0
            alive = int((cell_arr > 0.5).sum())
            events = p.get("event_counters") if isinstance(p, dict) else None
            totals = events.get("totals") if isinstance(events, dict) else None
            if not isinstance(totals, dict):
                totals = {}
            divisions = int(totals.get("divisions") or 0)
            starv = int(totals.get("starvation_deaths") or 0)
            dmg = int(totals.get("damage_deaths") or 0)

            layers_dict: Dict[str, np.ndarray] = {}
            for nm, ent in dd.items():
                if isinstance(ent, dict) and ent.get("dtype") == "float32":
                    arr = ent.get("arr")
                    if isinstance(arr, np.ndarray):
                        layers_dict[str(nm)] = arr
            measurements = _compute_measurements_from_layers(p, layers_dict, int(H), int(W))
            if not isinstance(measurements, dict):
                measurements = {}

            if agg_names:
                for nm, mode in agg_modes.items():
                    vals = per_tick_meas.get(nm) or []
                    if not vals:
                        continue
                    if mode == "mean":
                        measurements[nm] = float(np.mean(np.asarray(vals, dtype=np.float64)))
                    elif mode == "median":
                        measurements[nm] = float(np.median(np.asarray(vals, dtype=np.float64)))
                    elif mode == "std":
                        measurements[nm] = float(np.std(np.asarray(vals, dtype=np.float64)))
                    elif mode == "var":
                        measurements[nm] = float(np.var(np.asarray(vals, dtype=np.float64)))

            tick_by_type_s: Dict[str, float] = {}
            if profile_ticks:
                ev = p.get("event_counters") if isinstance(p, dict) else None
                lop = ev.get("layer_ops_perf") if isinstance(ev, dict) else None
                bt = lop.get("by_type_s") if isinstance(lop, dict) else None
                if isinstance(bt, dict):
                    for k, v in bt.items():
                        if not isinstance(k, str) or not k:
                            continue
                        try:
                            tick_by_type_s[k] = float(v)
                        except Exception:
                            continue

            t_total = time.perf_counter() - t_total0
            _perf_add(
                apply_s=t_apply,
                ticks_s=t_ticks,
                decode_cell_s=t_cell_decode,
                total_s=t_total,
                apply_copy_s=t_copy,
                apply_decode_s=t_apply_decode,
                apply_math_s=t_math,
                apply_encode_s=t_encode,
                sample_s=sample_s,
                tick_by_type_s=tick_by_type_s if profile_ticks else None,
            )
            return {
                "alive": alive,
                "divisions": divisions,
                "starvation_deaths": starv,
                "damage_deaths": dmg,
                "measurements": measurements,
            }

        def _fitness(metrics: Dict[str, Any]) -> float:
            return _evo_fitness_from_metrics(metrics, fitness_w)

        def _merge_measurements(rep_metrics: list[Dict[str, Any]]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            if not rep_metrics:
                return out
            all_keys: set[str] = set()
            for m in rep_metrics:
                mm = m.get("measurements") if isinstance(m, dict) else None
                if isinstance(mm, dict):
                    for k in mm.keys():
                        if isinstance(k, str) and k:
                            all_keys.add(k)
            for k in all_keys:
                vals = []
                for m in rep_metrics:
                    mm = m.get("measurements") if isinstance(m, dict) else None
                    if not isinstance(mm, dict) or k not in mm:
                        continue
                    try:
                        vals.append(float(mm.get(k) or 0.0))
                    except Exception:
                        continue
                if vals:
                    out[k] = float(np.mean(vals))
            return out

        base_rep_metrics = []
        for ri in range(replicates):
            if self._stop.is_set():
                raise RuntimeError("stopped")
            seed0 = seed + (0 * 1000003) + (0 * 1009) + (ri * 97)
            base_rep_metrics.append(_eval_genome({}, seed0=seed0, sample_s=0.0))
        alive_m0 = float(np.mean([mm["alive"] for mm in base_rep_metrics]))
        div_m0 = float(np.mean([mm["divisions"] for mm in base_rep_metrics]))
        starv_m0 = float(np.mean([mm["starvation_deaths"] for mm in base_rep_metrics]))
        dmg_m0 = float(np.mean([mm["damage_deaths"] for mm in base_rep_metrics]))
        base_metrics = {
            "alive": int(round(alive_m0)),
            "divisions": div_m0,
            "starvation_deaths": starv_m0,
            "damage_deaths": dmg_m0,
            "measurements": _merge_measurements(base_rep_metrics),
        }
        # Calculate baseline fitness using exactly the same approach as candidates
        base_fit = float(_fitness(base_metrics))
        
        if _EVO_DEBUG:
            try:
                _LOG.info("DEBUG: Baseline fitness calculation: %s", str(base_fit))
                _LOG.info("DEBUG: Baseline metrics: %s", str(base_metrics))
                _LOG.info("DEBUG: Using measurement weights: %s", str(meas_weights))
            except Exception:
                pass
        
        with self._lock:
            self.baseline = {"fitness": base_fit, "metrics": base_metrics}
            self.progress["updated_at"] = time.time()

        max_points = int(cfg.get("plot_max_points") or 5000)

        if algo == "auto_switch":
            auto_first = str(cfg.get("auto_first") or "cem_delta").strip().lower()
            if auto_first not in ("cem_delta", "affine"):
                auto_first = "cem_delta"
            auto_patience = max(1, int(cfg.get("auto_patience") or 5))
            auto_min_delta = float(cfg.get("auto_min_delta") or 0.0)
            auto_max_switches = int(cfg.get("auto_max_switches") or 20)
            auto_max_switches = max(0, auto_max_switches)

            use_cell_mask = cem_mask != "all"
            n_cells = int(H * W)

            def _clone_affine_genome(g: Any) -> Dict[str, Dict[str, float]]:
                out2: Dict[str, Dict[str, float]] = {}
                if not isinstance(g, dict):
                    return out2
                for nm, ent in g.items():
                    if not isinstance(nm, str) or not nm:
                        continue
                    if not isinstance(ent, dict):
                        continue
                    out2[nm] = {"scale": float(ent.get("scale", 1.0)), "bias": float(ent.get("bias", 0.0))}
                return out2

            def _delta_seed_from_affine(g: Any) -> Dict[str, np.ndarray]:
                gg = _clone_affine_genome(g)
                out2: Dict[str, np.ndarray] = {}
                for nm in mutable_names:
                    base_arr0 = base_layers.get(nm)
                    if not isinstance(base_arr0, np.ndarray):
                        continue
                    ent = gg.get(nm) or {"scale": 1.0, "bias": 0.0}
                    s = float(ent.get("scale", 1.0))
                    b = float(ent.get("bias", 0.0))
                    base_arr = np.asarray(base_arr0, dtype=np.float32).reshape(n_cells)
                    d = np.asarray(base_arr * np.float32(s - 1.0) + np.float32(b), dtype=np.float32)
                    if use_cell_mask:
                        d[~cell_mask] = 0.0
                    out2[nm] = d
                return out2

            def _affine_seed_from_delta(delta_seed: Any) -> Dict[str, Dict[str, float]]:
                out2: Dict[str, Dict[str, float]] = {}
                if not isinstance(delta_seed, dict):
                    return out2
                for nm in mutable_names:
                    d0 = delta_seed.get(nm)
                    base_arr0 = base_layers.get(nm)
                    if not isinstance(d0, np.ndarray) or not isinstance(base_arr0, np.ndarray):
                        continue
                    x = np.asarray(base_arr0, dtype=np.float64).reshape(n_cells)
                    y = x + np.asarray(d0, dtype=np.float64).reshape(n_cells)
                    if use_cell_mask:
                        xx = x[cell_mask]
                        yy = y[cell_mask]
                    else:
                        xx = x
                        yy = y
                    if xx.size <= 1:
                        s = 1.0
                        b = float(np.mean(yy - xx)) if xx.size else 0.0
                    else:
                        xm = float(xx.mean())
                        ym = float(yy.mean())
                        denom = float(np.sum((xx - xm) ** 2))
                        if denom <= 1e-12:
                            s = 1.0
                        else:
                            s = float(np.sum((xx - xm) * (yy - ym)) / denom)
                        if not np.isfinite(s):
                            s = 1.0
                        if s < 0.0:
                            s = 0.0
                        b = float(ym - s * xm)
                        if not np.isfinite(b):
                            b = 0.0
                    out2[nm] = {"scale": float(s), "bias": float(b)}
                return out2

            def _series_push(vi: int, fit: float, evals_done: int) -> None:
                with self._lock:
                    self.progress["variant"] = int(vi)
                    self.progress["evaluations_done"] = int(
                        self.progress.get("evaluations_done", 0) + max(0, int(evals_done))
                    )
                    self._series_sum += float(fit)
                    self._series_n += 1
                    if float(fit) > self._series_best:
                        self._series_best = float(fit)
                    self.series.setdefault("median", [])
                    self.series["fitness"].append(float(fit))
                    self.series["best"].append(float(self._series_best))
                    self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                    try:
                        self.series["median"].append(
                            float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                        )
                    except Exception:
                        self.series["median"].append(float(fit))
                    while len(self.series["fitness"]) > max_points:
                        self.series["fitness"].pop(0)
                        self.series["best"].pop(0)
                        self.series["mean"].pop(0)
                        self.series["median"].pop(0)
                        self.series["offset"] = int(self.series.get("offset") or 0) + 1
                    self.progress["updated_at"] = time.time()

            cem_alpha2 = float(np.clip(cem_alpha, 0.0, 1.0))
            cem_sigma_init2 = max(0.0, cem_sigma_init)
            cem_sigma_floor2 = max(0.0, cem_sigma_floor)
            cem_topk = max(1, min(variants, elites if elites > 0 else max(1, variants // 10)))

            def _cem_init(seed_delta: Optional[Dict[str, np.ndarray]]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
                mu2: Dict[str, np.ndarray] = {}
                sig2: Dict[str, np.ndarray] = {}
                sig_floor2: Dict[str, float] = {}
                sd = seed_delta if isinstance(seed_delta, dict) else {}
                for nm in mutable_names:
                    layer_mean = float(layer_stats.get(nm, {}).get("mean", 1.0))
                    layer_std = float(layer_stats.get(nm, {}).get("std", 1.0))
                    scale_ref = min(layer_std, abs(layer_mean) * 0.1) if abs(layer_mean) > 1.0 else layer_std
                    if scale_ref <= 0:
                        scale_ref = 1.0
                    sig_floor2[nm] = float(cem_sigma_floor2 * scale_ref)
                    m0 = np.zeros((n_cells,), dtype=np.float32)
                    sd0 = sd.get(nm)
                    if isinstance(sd0, np.ndarray) and int(sd0.size) == int(n_cells):
                        m0 = np.asarray(sd0, dtype=np.float32).reshape(n_cells).copy()
                    if use_cell_mask:
                        m0[~cell_mask] = 0.0
                    mu2[nm] = m0
                    s0 = np.full((n_cells,), float(cem_sigma_init2 * scale_ref), dtype=np.float32)
                    if use_cell_mask:
                        s0[~cell_mask] = 0.0
                    sig2[nm] = s0
                return mu2, sig2, sig_floor2

            def _cem_delta_for_vi(gen_i: int, vi: int, mu0: Dict[str, np.ndarray], sig0: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                rr0 = np.random.default_rng(int(seed) + 1234567 + (int(gen_i) * 1000003) + (int(vi) * 1009))
                out2: Dict[str, np.ndarray] = {}
                for nm in mutable_names:
                    eps = rr0.normal(0.0, 1.0, size=(n_cells,)).astype(np.float32)
                    d = np.asarray(mu0[nm] + sig0[nm] * eps, dtype=np.float32)
                    if use_cell_mask:
                        d = np.where(cell_mask, d, 0.0).astype(np.float32)
                    out2[nm] = d
                return out2

            def _cem_encode_genome(gen_i: int, vi: int, mu0: Dict[str, np.ndarray], sig0: Dict[str, np.ndarray]) -> Dict[str, Any]:
                dd0 = _cem_delta_for_vi(gen_i, int(vi), mu0, sig0)
                g2: Dict[str, Any] = {}
                for nm in mutable_names:
                    g2[nm] = {"delta_b64": _encode_float32_b64(np.asarray(dd0[nm], dtype=np.float32))}
                return g2

            def _run_cem_gen(ex: concurrent.futures.ThreadPoolExecutor, gen_i: int, mu2: Dict[str, np.ndarray], sig2: Dict[str, np.ndarray], sig_floor2: Dict[str, float]) -> tuple[float, Dict[str, np.ndarray]]:
                if self._stop.is_set():
                    return float("-inf"), {}
                with self._lock:
                    self.progress["generation"] = int(gen_i)
                    self.progress["variant"] = 0
                    self.progress["updated_at"] = time.time()
                mu0 = {nm: mu2[nm].copy() for nm in mutable_names}
                sig0 = {nm: sig2[nm].copy() for nm in mutable_names}
                plan = list(range(int(variants)))
                gen_workers = max(1, min(int(workers), len(plan)))

                def _eval_variant(vi: int) -> Optional[Dict[str, Any]]:
                    rr0 = np.random.default_rng(int(seed) + 1234567 + (int(gen_i) * 1000003) + (int(vi) * 1009))
                    t_sample0 = time.perf_counter()
                    genome: Dict[str, Any] = {}
                    for nm in mutable_names:
                        eps = rr0.normal(0.0, 1.0, size=(n_cells,)).astype(np.float32)
                        d = np.asarray(mu0[nm] + sig0[nm] * eps, dtype=np.float32)
                        if use_cell_mask:
                            d = np.where(cell_mask, d, 0.0).astype(np.float32)
                        genome[nm] = {"delta": d}
                    t_sample = time.perf_counter() - t_sample0
                    rep_metrics = []
                    for ri in range(int(replicates)):
                        if self._stop.is_set():
                            return None
                        seed0 = int(seed) + (int(gen_i) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)
                        rep_metrics.append(_eval_genome(genome, seed0=seed0, sample_s=t_sample if ri == 0 else 0.0))
                    if not rep_metrics:
                        return None
                    merged_measurements = _merge_measurements(rep_metrics)
                    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                    metrics = {
                        "alive": int(round(alive_m)),
                        "divisions": div_m,
                        "starvation_deaths": starv_m,
                        "damage_deaths": dmg_m,
                        "measurements": merged_measurements,
                    }
                    fit = float(_fitness(metrics))
                    return {"vi": int(vi), "metrics": metrics, "fitness": fit, "evals_done": int(replicates)}

                pending: set[concurrent.futures.Future] = set()
                it = iter(plan)

                def _submit_one() -> None:
                    if self._stop.is_set():
                        return
                    try:
                        vi0 = next(it)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_eval_variant, int(vi0)))

                for _ in range(gen_workers):
                    _submit_one()

                candidates_this_gen: list[Dict[str, Any]] = []
                while pending:
                    if self._stop.is_set():
                        break
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        if self._stop.is_set():
                            break
                        try:
                            res = fut.result()
                        except Exception:
                            _submit_one()
                            continue
                        if not isinstance(res, dict):
                            _submit_one()
                            continue
                        vi = int(res.get("vi") or 0)
                        metrics = res.get("metrics")
                        fit = float(res.get("fitness") or 0.0)
                        evals_done = int(res.get("evals_done") or 0)
                        if not isinstance(metrics, dict):
                            _submit_one()
                            continue
                        _series_push(vi, fit, evals_done)
                        cid = str(uuid.uuid4())
                        candidates_this_gen.append({"id": cid, "gen": int(gen_i), "fitness": fit, "metrics": metrics, "vi": vi})
                        _submit_one()

                if not candidates_this_gen:
                    return float("-inf"), {}
                candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                best = float(fits.max())
                mean = float(fits.mean())
                median = float(np.quantile(fits, 0.50))
                p10 = float(np.quantile(fits, 0.10))
                p90 = float(np.quantile(fits, 0.90))

                top = candidates_this_gen[:cem_topk]
                top_deltas: list[Dict[str, np.ndarray]] = []
                for c in top:
                    vi = int(c.get("vi") or 0)
                    top_deltas.append(_cem_delta_for_vi(gen_i, vi, mu0, sig0))
                for nm in mutable_names:
                    stack = np.stack([d[nm] for d in top_deltas], axis=0)
                    mu_new = stack.mean(axis=0).astype(np.float32)
                    sig_new = stack.std(axis=0).astype(np.float32)
                    mu2[nm] = ((1.0 - cem_alpha2) * mu2[nm] + cem_alpha2 * mu_new).astype(np.float32)
                    sig2[nm] = ((1.0 - cem_alpha2) * sig2[nm] + cem_alpha2 * sig_new).astype(np.float32)
                    sig2[nm] = np.maximum(sig2[nm], float(sig_floor2.get(nm, 0.0))).astype(np.float32)
                    if use_cell_mask:
                        mu2[nm][~cell_mask] = 0.0
                        sig2[nm][~cell_mask] = 0.0

                with self._lock:
                    for c in candidates_this_gen[: max(int(elites), 10)]:
                        vi = int(c.get("vi") or 0)
                        c2 = dict(c)
                        c2["genome"] = _cem_encode_genome(gen_i, vi, mu0, sig0)
                        c2["genome_type"] = "delta"
                        self.candidates[str(c2["id"])] = c2
                    self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                    self.history["best"].append(best)
                    self.history["mean"].append(mean)
                    self.history.setdefault("median", [])
                    self.history["median"].append(median)
                    self.history["p10"].append(p10)
                    self.history["p90"].append(p90)
                    self.progress["updated_at"] = time.time()

                best_vi = int(candidates_this_gen[0].get("vi") or 0)
                best_delta = _cem_delta_for_vi(gen_i, best_vi, mu0, sig0)
                return float(best), best_delta

            def _run_affine_gen(ex: concurrent.futures.ThreadPoolExecutor, gen_i: int, parents2: list[Dict[str, Dict[str, float]]]) -> tuple[float, Dict[str, Dict[str, float]], list[Dict[str, Dict[str, float]]]]:
                if self._stop.is_set():
                    return float("-inf"), {}, parents2
                with self._lock:
                    self.progress["generation"] = int(gen_i)
                    self.progress["variant"] = 0
                    self.progress["updated_at"] = time.time()

                plan: list[tuple[int, Dict[str, Dict[str, float]]]] = []
                for vi in range(int(variants)):
                    if self._stop.is_set():
                        break
                    parent = parents2[int(rng.integers(0, len(parents2)))]
                    genome = _mutate_genome(parent)
                    plan.append((int(vi), genome))

                gen_workers = max(1, min(int(workers), len(plan)))

                def _eval_variant(vi: int, genome: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
                    rep_metrics = []
                    for ri in range(int(replicates)):
                        if self._stop.is_set():
                            return None
                        seed0 = int(seed) + (int(gen_i) * 1000003) + (int(vi) * 1009) + (int(ri) * 97)
                        rep_metrics.append(_eval_genome(genome, seed0=seed0))
                    if not rep_metrics:
                        return None
                    merged_measurements = _merge_measurements(rep_metrics)
                    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                    metrics = {
                        "alive": int(round(alive_m)),
                        "divisions": div_m,
                        "starvation_deaths": starv_m,
                        "damage_deaths": dmg_m,
                        "measurements": merged_measurements,
                    }
                    fit = float(_fitness(metrics))
                    return {"vi": int(vi), "genome": genome, "metrics": metrics, "fitness": fit, "evals_done": int(replicates)}

                pending: set[concurrent.futures.Future] = set()
                it = iter(plan)

                def _submit_one() -> None:
                    if self._stop.is_set():
                        return
                    try:
                        vi0, genome0 = next(it)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_eval_variant, int(vi0), genome0))

                for _ in range(gen_workers):
                    _submit_one()

                candidates_this_gen: list[Dict[str, Any]] = []
                while pending:
                    if self._stop.is_set():
                        break
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        if self._stop.is_set():
                            break
                        try:
                            res = fut.result()
                        except Exception:
                            _submit_one()
                            continue
                        if not isinstance(res, dict):
                            _submit_one()
                            continue
                        vi = int(res.get("vi") or 0)
                        genome = res.get("genome")
                        metrics = res.get("metrics")
                        fit = float(res.get("fitness") or 0.0)
                        evals_done = int(res.get("evals_done") or 0)
                        if not isinstance(genome, dict) or not isinstance(metrics, dict):
                            _submit_one()
                            continue
                        _series_push(vi, fit, evals_done)
                        cid = str(uuid.uuid4())
                        candidates_this_gen.append({"id": cid, "gen": int(gen_i), "fitness": fit, "metrics": metrics, "genome": genome})
                        _submit_one()

                if not candidates_this_gen:
                    return float("-inf"), {}, parents2
                candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                best = float(fits.max())
                mean = float(fits.mean())
                median = float(np.quantile(fits, 0.50))
                p10 = float(np.quantile(fits, 0.10))
                p90 = float(np.quantile(fits, 0.90))

                with self._lock:
                    for c in candidates_this_gen[: max(int(elites), 10)]:
                        self.candidates[str(c["id"])] = c
                    self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                    self.history["best"].append(best)
                    self.history["mean"].append(mean)
                    self.history.setdefault("median", [])
                    self.history["median"].append(median)
                    self.history["p10"].append(p10)
                    self.history["p90"].append(p90)
                    self.progress["updated_at"] = time.time()

                new_parents = [c["genome"] for c in candidates_this_gen[: int(elites)]]
                best_genome = _clone_affine_genome(candidates_this_gen[0].get("genome"))
                return float(best), best_genome, new_parents if new_parents else parents2

            active_algo = str(auto_first)
            plateau_count = 0
            last_switch_gen = -1
            switches = 0

            best_so_far = float(base_fit)
            best_delta_seed: Dict[str, np.ndarray] = {}
            best_affine_seed: Dict[str, Dict[str, float]] = {}
            parents: list[Dict[str, Dict[str, float]]] = [{}]
            mu, sig, sig_floor = _cem_init(best_delta_seed)

            with self._lock:
                self.auto = {
                    "active_algo": str(active_algo),
                    "plateau_count": int(plateau_count),
                    "last_switch_gen": int(last_switch_gen),
                    "switches": int(switches),
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for gen_i in range(int(generations)):
                    if self._stop.is_set():
                        break
                    if active_algo == "affine":
                        best_fit_gen, best_genome, parents = _run_affine_gen(ex, int(gen_i), parents)
                        if isinstance(best_genome, dict) and best_fit_gen > float("-inf"):
                            improved = bool(best_fit_gen > (best_so_far + float(auto_min_delta)))
                            if improved:
                                best_so_far = float(best_fit_gen)
                                plateau_count = 0
                                best_affine_seed = _clone_affine_genome(best_genome)
                                best_delta_seed = _delta_seed_from_affine(best_genome)
                            else:
                                plateau_count += 1
                    else:
                        best_fit_gen, best_delta = _run_cem_gen(ex, int(gen_i), mu, sig, sig_floor)
                        if isinstance(best_delta, dict) and best_fit_gen > float("-inf"):
                            improved = bool(best_fit_gen > (best_so_far + float(auto_min_delta)))
                            if improved:
                                best_so_far = float(best_fit_gen)
                                plateau_count = 0
                                best_delta_seed = {k: np.asarray(v, dtype=np.float32).reshape(n_cells) for k, v in best_delta.items() if isinstance(k, str) and isinstance(v, np.ndarray)}
                                best_affine_seed = _affine_seed_from_delta(best_delta_seed)
                            else:
                                plateau_count += 1

                    if (
                        auto_max_switches > 0
                        and switches < auto_max_switches
                        and plateau_count >= int(auto_patience)
                        and int(gen_i) < int(generations) - 1
                    ):
                        switches += 1
                        last_switch_gen = int(gen_i)
                        plateau_count = 0
                        active_algo = "affine" if active_algo == "cem_delta" else "cem_delta"
                        if active_algo == "affine":
                            parents = [best_affine_seed] if best_affine_seed else [{}]
                        else:
                            mu, sig, sig_floor = _cem_init(best_delta_seed)

                    with self._lock:
                        self.auto = {
                            "active_algo": str(active_algo),
                            "plateau_count": int(plateau_count),
                            "last_switch_gen": int(last_switch_gen),
                            "switches": int(switches),
                            "patience": int(auto_patience),
                            "min_delta": float(auto_min_delta),
                            "max_switches": int(auto_max_switches),
                            "best_fitness": float(best_so_far),
                            "updated_at": float(time.time()),
                        }

            return

        if algo == "cem_delta":
            topk = max(1, min(variants, elites if elites > 0 else max(1, variants // 10)))
            cem_alpha = float(np.clip(cem_alpha, 0.0, 1.0))
            cem_sigma_init = max(0.0, cem_sigma_init)
            cem_sigma_floor = max(0.0, cem_sigma_floor)
            use_cell_mask = cem_mask != "all"

            mu: Dict[str, np.ndarray] = {}
            sig: Dict[str, np.ndarray] = {}
            sig_floor: Dict[str, float] = {}
            for nm in mutable_names:
                layer_mean = float(layer_stats.get(nm, {}).get("mean", 1.0))
                layer_std = float(layer_stats.get(nm, {}).get("std", 1.0))
                # Use the smaller of std or 10% of mean to avoid huge perturbations
                # This ensures sigma is reasonable relative to actual layer values
                scale_ref = min(layer_std, abs(layer_mean) * 0.1) if abs(layer_mean) > 1.0 else layer_std
                if scale_ref <= 0:
                    scale_ref = 1.0
                sig_floor[nm] = float(cem_sigma_floor * scale_ref)
                mu[nm] = np.zeros((H * W,), dtype=np.float32)
                sig[nm] = np.full((H * W,), float(cem_sigma_init * scale_ref), dtype=np.float32)
                if use_cell_mask:
                    mu[nm][~cell_mask] = 0.0
                    sig[nm][~cell_mask] = 0.0
            
            if _EVO_DEBUG:
                try:
                    _LOG.info("DEBUG CEM: cem_sigma_init=%s, cem_alpha=%s", str(cem_sigma_init), str(cem_alpha))
                    for nm in mutable_names[:3]:
                        layer_mean = layer_stats.get(nm, {}).get("mean", 1.0)
                        layer_std = layer_stats.get(nm, {}).get("std", 1.0)
                        scale_ref = min(layer_std, abs(layer_mean) * 0.1) if abs(layer_mean) > 1.0 else layer_std
                        init_sig = cem_sigma_init * scale_ref
                        _LOG.info(
                            "DEBUG CEM: Layer '%s' initial sigma=%0.4f (mean=%0.1f, std=%0.1f, scale_ref=%0.1f)",
                            str(nm),
                            float(init_sig),
                            float(layer_mean),
                            float(layer_std),
                            float(scale_ref),
                        )
                except Exception:
                    pass

            if worker_mode == "process":
                ctx = mp.get_context("spawn")
                k_layers = int(len(mutable_names))
                n_cells = int(H * W)
                mu_shm = shared_memory.SharedMemory(create=True, size=int(k_layers * n_cells * 4))
                sig_shm = shared_memory.SharedMemory(create=True, size=int(k_layers * n_cells * 4))
                mu_shm_arr = np.ndarray((k_layers, n_cells), dtype=np.float32, buffer=mu_shm.buf)
                sig_shm_arr = np.ndarray((k_layers, n_cells), dtype=np.float32, buffer=sig_shm.buf)
                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=ctx,
                        initializer=_evo_worker_init_cem_delta,
                        initargs=(
                            base_payload_fast,
                            kinds,
                            cell_layer,
                            huge,
                            str(mu_shm.name),
                            str(sig_shm.name),
                            int(k_layers),
                            int(n_cells),
                            list(mutable_names),
                        ),
                    ) as ex:
                        for gen in range(generations):
                            if self._stop.is_set():
                                break

                            with self._lock:
                                self.progress["generation"] = int(gen)
                                self.progress["variant"] = 0
                                self.progress["updated_at"] = time.time()

                            mu0: Dict[str, np.ndarray] = {nm: mu[nm].copy() for nm in mutable_names}
                            sig0: Dict[str, np.ndarray] = {nm: sig[nm].copy() for nm in mutable_names}

                            for i, nm in enumerate(mutable_names):
                                np.copyto(mu_shm_arr[int(i)], np.asarray(mu0[nm], dtype=np.float32).reshape(n_cells))
                                np.copyto(sig_shm_arr[int(i)], np.asarray(sig0[nm], dtype=np.float32).reshape(n_cells))

                            cem_batch0 = int(cem_batch)
                            if cem_batch0 <= 0:
                                target_tasks = max(1, int(workers * 3))
                                cem_batch0 = max(1, (int(variants) + int(target_tasks) - 1) // int(target_tasks))
                            cem_batch0 = max(1, min(64, int(cem_batch0)))

                            todo: deque[tuple[list[int], int]] = deque()
                            buf: list[int] = []
                            for vi in range(int(variants)):
                                buf.append(int(vi))
                                if len(buf) >= cem_batch0:
                                    todo.append((buf, 0))
                                    buf = []
                            if buf:
                                todo.append((buf, 0))

                            gen_workers = max(1, min(workers, len(todo)))

                            candidates_this_gen: list[Dict[str, Any]] = []
                            pending: set[concurrent.futures.Future] = set()
                            future_meta: dict[concurrent.futures.Future, tuple[list[int], int]] = {}

                            def _submit_one() -> None:
                                if self._stop.is_set():
                                    return
                                if not todo:
                                    return
                                vis, retries = todo.popleft()
                                fut = ex.submit(
                                    _evo_worker_eval_cem_delta_batch,
                                    int(gen),
                                    list(vis),
                                    int(seed),
                                    int(ticks),
                                    int(replicates),
                                    dict(fitness_w),
                                    bool(use_cell_mask),
                                )
                                pending.add(fut)
                                future_meta[fut] = (vis, int(retries))

                            for _ in range(gen_workers):
                                _submit_one()

                            while pending:
                                if self._stop.is_set():
                                    break
                                done, pending = concurrent.futures.wait(
                                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for fut in done:
                                    if self._stop.is_set():
                                        break
                                    try:
                                        res = fut.result()
                                    except Exception:
                                        vis0, retries0 = future_meta.pop(fut, ([], 0))
                                        if vis0 and int(retries0) < 2:
                                            todo.append((vis0, int(retries0) + 1))
                                        else:
                                            pass
                                        _submit_one()
                                        continue
                                    if not isinstance(res, dict):
                                        future_meta.pop(fut, None)
                                        _submit_one()
                                        continue

                                    future_meta.pop(fut, None)
                                    batch_results = res.get("results")
                                    if not isinstance(batch_results, list) or not batch_results:
                                        _submit_one()
                                        continue

                                    for r1 in batch_results:
                                        if not isinstance(r1, dict):
                                            continue
                                        vi = int(r1.get("vi") or 0)
                                        metrics = r1.get("metrics")
                                        fit = float(r1.get("fitness") or 0.0)
                                        evals_done = int(r1.get("evals_done") or 0)
                                        if not isinstance(metrics, dict):
                                            continue

                                        with self._lock:
                                            self.progress["variant"] = int(vi)
                                            self.progress["evaluations_done"] = int(
                                                self.progress.get("evaluations_done", 0) + max(0, evals_done)
                                            )
                                            self._series_sum += fit
                                            self._series_n += 1
                                            if fit > self._series_best:
                                                self._series_best = fit
                                            self.series.setdefault("median", [])
                                            self.series["fitness"].append(fit)
                                            self.series["best"].append(float(self._series_best))
                                            self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                                            try:
                                                self.series["median"].append(
                                                    float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                                                )
                                            except Exception:
                                                self.series["median"].append(float(fit))
                                            while len(self.series["fitness"]) > max_points:
                                                self.series["fitness"].pop(0)
                                                self.series["best"].pop(0)
                                                self.series["mean"].pop(0)
                                                self.series["median"].pop(0)
                                                self.series["offset"] = int(self.series.get("offset") or 0) + 1
                                            self.progress["updated_at"] = time.time()

                                        cid = str(uuid.uuid4())
                                        candidates_this_gen.append(
                                            {"id": cid, "gen": gen, "fitness": fit, "metrics": metrics, "vi": vi}
                                        )

                                    _submit_one()

                            if not candidates_this_gen:
                                break

                            candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                            fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                            best = float(fits.max())
                            mean = float(fits.mean())
                            median = float(np.quantile(fits, 0.50))
                            p10 = float(np.quantile(fits, 0.10))
                            p90 = float(np.quantile(fits, 0.90))

                            top = candidates_this_gen[:topk]
                            top_deltas: list[Dict[str, np.ndarray]] = []
                            for c in top:
                                vi = int(c.get("vi") or 0)
                                rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                                dd0: Dict[str, np.ndarray] = {}
                                for nm in mutable_names:
                                    eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                                    delta = mu0[nm] + sig0[nm] * eps
                                    if use_cell_mask:
                                        delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                                    dd0[nm] = delta
                                top_deltas.append(dd0)

                            for nm in mutable_names:
                                stack = np.stack([d[nm] for d in top_deltas], axis=0)
                                mu_new = stack.mean(axis=0).astype(np.float32)
                                sig_new = stack.std(axis=0).astype(np.float32)
                                mu[nm] = ((1.0 - cem_alpha) * mu[nm] + cem_alpha * mu_new).astype(np.float32)
                                sig[nm] = ((1.0 - cem_alpha) * sig[nm] + cem_alpha * sig_new).astype(np.float32)
                                sig[nm] = np.maximum(sig[nm], float(sig_floor.get(nm, 0.0))).astype(np.float32)
                                if use_cell_mask:
                                    mu[nm][~cell_mask] = 0.0
                                    sig[nm][~cell_mask] = 0.0

                            def _encode_delta_genome_for_vi(vi: int) -> Dict[str, Any]:
                                rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                                g: Dict[str, Any] = {}
                                for nm in mutable_names:
                                    eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                                    delta = mu0[nm] + sig0[nm] * eps
                                    if use_cell_mask:
                                        delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                                    g[nm] = {"delta_b64": _encode_float32_b64(np.asarray(delta, dtype=np.float32))}
                                return g

                            with self._lock:
                                for c in candidates_this_gen[: max(elites, 10)]:
                                    vi = int(c.get("vi") or 0)
                                    c2 = dict(c)
                                    c2["genome"] = _encode_delta_genome_for_vi(vi)
                                    c2["genome_type"] = "delta"
                                    self.candidates[str(c2["id"])] = c2
                                self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                                self.history["best"].append(best)
                                self.history["mean"].append(mean)
                                self.history.setdefault("median", [])
                                self.history["median"].append(median)
                                self.history["p10"].append(p10)
                                self.history["p90"].append(p90)
                                self.progress["updated_at"] = time.time()
                finally:
                    try:
                        mu_shm.close()
                        mu_shm.unlink()
                    except Exception:
                        pass
                    try:
                        sig_shm.close()
                        sig_shm.unlink()
                    except Exception:
                        pass

                return

            # Reuse a single executor across generations to reduce overhead and improve scaling.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for gen in range(generations):
                    if self._stop.is_set():
                        break

                    with self._lock:
                        self.progress["generation"] = int(gen)
                        self.progress["variant"] = 0
                        self.progress["updated_at"] = time.time()

                    mu0: Dict[str, np.ndarray] = {nm: mu[nm].copy() for nm in mutable_names}
                    sig0: Dict[str, np.ndarray] = {nm: sig[nm].copy() for nm in mutable_names}

                    plan = list(range(variants))
                    gen_workers = max(1, min(workers, len(plan)))

                    def _eval_variant(vi: int) -> Optional[Dict[str, Any]]:
                        rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                        t_sample0 = time.perf_counter()
                        genome: Dict[str, Any] = {}
                        for nm in mutable_names:
                            eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                            delta = mu0[nm] + sig0[nm] * eps
                            if use_cell_mask:
                                delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                            genome[nm] = {"delta": delta}
                        t_sample = time.perf_counter() - t_sample0

                        rep_metrics = []
                        for ri in range(replicates):
                            if self._stop.is_set():
                                return None
                            seed0 = seed + (gen * 1000003) + (vi * 1009) + (ri * 97)
                            # Sampling the genome is done once per variant; attribute that time once.
                            m = _eval_genome(genome, seed0=seed0, sample_s=t_sample if ri == 0 else 0.0)
                            rep_metrics.append(m)

                        if not rep_metrics:
                            return None

                        merged_measurements = _merge_measurements(rep_metrics)
                        alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                        div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                        starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                        dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                        metrics = {
                            "alive": int(round(alive_m)),
                            "divisions": div_m,
                            "starvation_deaths": starv_m,
                            "damage_deaths": dmg_m,
                            "measurements": merged_measurements,
                        }
                        fit = float(_fitness(metrics))
                        return {"vi": int(vi), "metrics": metrics, "fitness": fit, "evals_done": int(replicates)}

                    candidates_this_gen: list[Dict[str, Any]] = []
                    pending: set[concurrent.futures.Future] = set()
                    it = iter(plan)

                    def _submit_one() -> None:
                        if self._stop.is_set():
                            return
                        try:
                            vi = next(it)
                        except StopIteration:
                            return
                        pending.add(ex.submit(_eval_variant, vi))

                    for _ in range(gen_workers):
                        _submit_one()

                    while pending:
                        if self._stop.is_set():
                            break
                        done, pending = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            if self._stop.is_set():
                                break
                            try:
                                res = fut.result()
                            except Exception:
                                _submit_one()
                                continue
                            if not isinstance(res, dict):
                                _submit_one()
                                continue

                            vi = int(res.get("vi") or 0)
                            metrics = res.get("metrics")
                            fit = float(res.get("fitness") or 0.0)
                            evals_done = int(res.get("evals_done") or 0)
                            if not isinstance(metrics, dict):
                                _submit_one()
                                continue

                            with self._lock:
                                self.progress["variant"] = int(vi)
                                self.progress["evaluations_done"] = int(
                                    self.progress.get("evaluations_done", 0) + max(0, evals_done)
                                )
                                self._series_sum += fit
                                self._series_n += 1
                                if fit > self._series_best:
                                    self._series_best = fit
                                self.series.setdefault("median", [])
                                self.series["fitness"].append(fit)
                                self.series["best"].append(float(self._series_best))
                                self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                                try:
                                    self.series["median"].append(
                                        float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                                    )
                                except Exception:
                                    self.series["median"].append(float(fit))
                                while len(self.series["fitness"]) > max_points:
                                    self.series["fitness"].pop(0)
                                    self.series["best"].pop(0)
                                    self.series["mean"].pop(0)
                                    self.series["median"].pop(0)
                                    self.series["offset"] = int(self.series.get("offset") or 0) + 1
                                self.progress["updated_at"] = time.time()

                            cid = str(uuid.uuid4())
                            candidates_this_gen.append(
                                {"id": cid, "gen": gen, "fitness": fit, "metrics": metrics, "vi": vi}
                            )

                            _submit_one()

                    if not candidates_this_gen:
                        break

                    candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                    fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                    best = float(fits.max())
                    mean = float(fits.mean())
                    median = float(np.quantile(fits, 0.50))
                    p10 = float(np.quantile(fits, 0.10))
                    p90 = float(np.quantile(fits, 0.90))

                    top = candidates_this_gen[:topk]
                    top_deltas: list[Dict[str, np.ndarray]] = []
                    for c in top:
                        vi = int(c.get("vi") or 0)
                        rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                        dd: Dict[str, np.ndarray] = {}
                        for nm in mutable_names:
                            eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                            delta = mu0[nm] + sig0[nm] * eps
                            if use_cell_mask:
                                delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                            dd[nm] = delta
                        top_deltas.append(dd)

                    for nm in mutable_names:
                        stack = np.stack([d[nm] for d in top_deltas], axis=0)
                        mu_new = stack.mean(axis=0).astype(np.float32)
                        sig_new = stack.std(axis=0).astype(np.float32)
                        mu[nm] = ((1.0 - cem_alpha) * mu[nm] + cem_alpha * mu_new).astype(np.float32)
                        sig[nm] = ((1.0 - cem_alpha) * sig[nm] + cem_alpha * sig_new).astype(np.float32)
                        sig[nm] = np.maximum(sig[nm], float(sig_floor.get(nm, 0.0))).astype(np.float32)
                        if use_cell_mask:
                            mu[nm][~cell_mask] = 0.0
                            sig[nm][~cell_mask] = 0.0

                    def _encode_delta_genome_for_vi(vi: int) -> Dict[str, Any]:
                        rr = np.random.default_rng(seed + 1234567 + (gen * 1000003) + (vi * 1009))
                        g: Dict[str, Any] = {}
                        for nm in mutable_names:
                            eps = rr.normal(0.0, 1.0, size=(H * W,)).astype(np.float32)
                            delta = mu0[nm] + sig0[nm] * eps
                            if use_cell_mask:
                                delta = np.where(cell_mask, delta, 0.0).astype(np.float32)
                            g[nm] = {"delta_b64": _encode_float32_b64(np.asarray(delta, dtype=np.float32))}
                        return g

                    with self._lock:
                        for c in candidates_this_gen[: max(elites, 10)]:
                            vi = int(c.get("vi") or 0)
                            c2 = dict(c)
                            c2["genome"] = _encode_delta_genome_for_vi(vi)
                            c2["genome_type"] = "delta"
                            self.candidates[str(c2["id"])] = c2
                        self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                        self.history["best"].append(best)
                        self.history["mean"].append(mean)
                        self.history.setdefault("median", [])
                        self.history["median"].append(median)
                        self.history["p10"].append(p10)
                        self.history["p90"].append(p90)
                        self.progress["updated_at"] = time.time()

            return

        parents: list[Dict[str, Dict[str, float]]] = [{}]

        if worker_mode == "process":
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_evo_worker_init,
                initargs=(base_payload_fast, kinds, cell_layer, huge),
            ) as ex:
                for gen in range(generations):
                    if self._stop.is_set():
                        break

                    with self._lock:
                        self.progress["generation"] = int(gen)
                        self.progress["variant"] = 0
                        self.progress["updated_at"] = time.time()

                    candidates_this_gen: list[Dict[str, Any]] = []
                    plan: list[tuple[int, Dict[str, Dict[str, float]]]] = []
                    for vi in range(variants):
                        if self._stop.is_set():
                            break
                        parent = parents[int(rng.integers(0, len(parents)))]
                        genome = _mutate_genome(parent)
                        plan.append((vi, genome))

                    gen_workers = max(1, min(workers, len(plan)))

                    pending: set[concurrent.futures.Future] = set()
                    it = iter(plan)

                    def _submit_one() -> None:
                        if self._stop.is_set():
                            return
                        try:
                            vi, genome = next(it)
                        except StopIteration:
                            return
                        pending.add(
                            ex.submit(
                                _evo_worker_eval_affine,
                                int(gen),
                                int(vi),
                                genome,
                                int(seed),
                                int(ticks),
                                int(replicates),
                                dict(fitness_w),
                            )
                        )

                    for _ in range(gen_workers):
                        _submit_one()

                    while pending:
                        if self._stop.is_set():
                            break
                        done, pending = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            if self._stop.is_set():
                                break
                            try:
                                res = fut.result()
                            except Exception:
                                _submit_one()
                                continue
                            if not isinstance(res, dict):
                                _submit_one()
                                continue

                            vi = int(res.get("vi") or 0)
                            genome = res.get("genome")
                            metrics = res.get("metrics")
                            fit = float(res.get("fitness") or 0.0)
                            evals_done = int(res.get("evals_done") or 0)
                            if not isinstance(genome, dict) or not isinstance(metrics, dict):
                                _submit_one()
                                continue

                            with self._lock:
                                self.progress["variant"] = int(vi)
                                self.progress["evaluations_done"] = int(
                                    self.progress.get("evaluations_done", 0) + max(0, evals_done)
                                )
                                self._series_sum += fit
                                self._series_n += 1
                                if fit > self._series_best:
                                    self._series_best = fit
                                self.series["fitness"].append(fit)
                                self.series["best"].append(float(self._series_best))
                                self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                                while len(self.series["fitness"]) > max_points:
                                    self.series["fitness"].pop(0)
                                    self.series["best"].pop(0)
                                    self.series["mean"].pop(0)
                                    self.series["offset"] = int(self.series.get("offset") or 0) + 1
                                self.progress["updated_at"] = time.time()

                            cid = str(uuid.uuid4())
                            cand = {
                                "id": cid,
                                "gen": gen,
                                "fitness": fit,
                                "metrics": metrics,
                                "genome": genome,
                            }
                            candidates_this_gen.append(cand)

                            _submit_one()

                    if not candidates_this_gen:
                        break

                    candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                    fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                    best = float(fits.max())
                    mean = float(fits.mean())
                    p10 = float(np.quantile(fits, 0.10))
                    p90 = float(np.quantile(fits, 0.90))

                    with self._lock:
                        for c in candidates_this_gen[: max(elites, 10)]:
                            self.candidates[str(c["id"])] = c
                        self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                        self.history["best"].append(best)
                        self.history["mean"].append(mean)
                        self.history["p10"].append(p10)
                        self.history["p90"].append(p90)
                        self.progress["updated_at"] = time.time()

                    parents = [c["genome"] for c in candidates_this_gen[:elites]]

            return

        # Thread mode (default)
        # Reuse a single executor across generations to reduce overhead and improve scaling.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for gen in range(generations):
                if self._stop.is_set():
                    break

                with self._lock:
                    self.progress["generation"] = int(gen)
                    self.progress["variant"] = 0
                    self.progress["updated_at"] = time.time()

                candidates_this_gen: list[Dict[str, Any]] = []
                plan: list[tuple[int, Dict[str, Dict[str, float]]]] = []
                for vi in range(variants):
                    if self._stop.is_set():
                        break
                    parent = parents[int(rng.integers(0, len(parents)))]
                    genome = _mutate_genome(parent)
                    plan.append((vi, genome))

                gen_workers = max(1, min(workers, len(plan)))

                def _eval_variant(vi: int, genome: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
                    rep_metrics = []
                    for ri in range(replicates):
                        if self._stop.is_set():
                            return None
                        seed0 = seed + (gen * 1000003) + (vi * 1009) + (ri * 97)
                        m = _eval_genome(genome, seed0=seed0)
                        rep_metrics.append(m)

                    if not rep_metrics:
                        return None

                    merged_measurements = _merge_measurements(rep_metrics)
                    alive_m = float(np.mean([mm["alive"] for mm in rep_metrics]))
                    div_m = float(np.mean([mm["divisions"] for mm in rep_metrics]))
                    starv_m = float(np.mean([mm["starvation_deaths"] for mm in rep_metrics]))
                    dmg_m = float(np.mean([mm["damage_deaths"] for mm in rep_metrics]))
                    metrics = {
                        "alive": int(round(alive_m)),
                        "divisions": div_m,
                        "starvation_deaths": starv_m,
                        "damage_deaths": dmg_m,
                        "measurements": merged_measurements,
                    }
                    fit = float(_fitness(metrics))
                    return {
                        "vi": int(vi),
                        "genome": genome,
                        "metrics": metrics,
                        "fitness": fit,
                        "evals_done": int(replicates),
                    }

                pending: set[concurrent.futures.Future] = set()
                it = iter(plan)

                def _submit_one() -> None:
                    if self._stop.is_set():
                        return
                    try:
                        vi, genome = next(it)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_eval_variant, vi, genome))

                for _ in range(gen_workers):
                    _submit_one()

                while pending:
                    if self._stop.is_set():
                        break
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done:
                        if self._stop.is_set():
                            break
                        try:
                            res = fut.result()
                        except Exception:
                            _submit_one()
                            continue
                        if not isinstance(res, dict):
                            _submit_one()
                            continue

                        vi = int(res.get("vi") or 0)
                        genome = res.get("genome")
                        metrics = res.get("metrics")
                        fit = float(res.get("fitness") or 0.0)
                        evals_done = int(res.get("evals_done") or 0)
                        if not isinstance(genome, dict) or not isinstance(metrics, dict):
                            _submit_one()
                            continue

                        with self._lock:
                            self.progress["variant"] = int(vi)
                            self.progress["evaluations_done"] = int(
                                self.progress.get("evaluations_done", 0) + max(0, evals_done)
                            )
                            self._series_sum += fit
                            self._series_n += 1
                            if fit > self._series_best:
                                self._series_best = fit
                            self.series.setdefault("median", [])
                            self.series["fitness"].append(fit)
                            self.series["best"].append(float(self._series_best))
                            self.series["mean"].append(float(self._series_sum / max(1, self._series_n)))
                            try:
                                self.series["median"].append(
                                    float(np.median(np.asarray(self.series["fitness"], dtype=np.float64)))
                                )
                            except Exception:
                                self.series["median"].append(float(fit))
                            while len(self.series["fitness"]) > max_points:
                                self.series["fitness"].pop(0)
                                self.series["best"].pop(0)
                                self.series["mean"].pop(0)
                                self.series["median"].pop(0)
                                self.series["offset"] = int(self.series.get("offset") or 0) + 1
                            self.progress["updated_at"] = time.time()

                        cid = str(uuid.uuid4())
                        cand = {
                            "id": cid,
                            "gen": gen,
                            "fitness": fit,
                            "metrics": metrics,
                            "genome": genome,
                        }
                        candidates_this_gen.append(cand)

                        _submit_one()

                if not candidates_this_gen:
                    break

                candidates_this_gen.sort(key=lambda c: float(c.get("fitness") or 0.0), reverse=True)
                fits = np.array([float(c.get("fitness") or 0.0) for c in candidates_this_gen], dtype=np.float64)
                best = float(fits.max())
                mean = float(fits.mean())
                median = float(np.quantile(fits, 0.50))
                p10 = float(np.quantile(fits, 0.10))
                p90 = float(np.quantile(fits, 0.90))

                with self._lock:
                    for c in candidates_this_gen[: max(elites, 10)]:
                        self.candidates[str(c["id"])] = c
                    self.top_ids = [str(c["id"]) for c in candidates_this_gen[:10]]
                    self.history["best"].append(best)
                    self.history["mean"].append(mean)
                    self.history.setdefault("median", [])
                    self.history["median"].append(median)
                    self.history["p10"].append(p10)
                    self.history["p90"].append(p90)
                    self.progress["updated_at"] = time.time()

                parents = [c["genome"] for c in candidates_this_gen[:elites]]


_EVO = _EvolutionJob()


class _RuntimeState:
    def __init__(self) -> None:
        self.payload: Optional[Dict[str, Any]] = None
        self.tick: int = 0
        self._lock = threading.Lock()

    def _default_layer_names(self, max_layers: int = 4) -> list:
        if self.payload is None:
            return []
        layers = self.payload.get("layers")
        if not isinstance(layers, list):
            return []
        names = []
        for m in layers:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                names.append(m["name"])
        if not names:
            return []

        prefer = ["cell", "circulation", "damage", "molecule_glucose", "molecule_atp"]
        picked = []
        s = set(names)
        for p in prefer:
            if p in s:
                picked.append(p)
            if len(picked) >= max_layers:
                break
        if len(picked) < max_layers:
            for nm in names:
                if nm in picked:
                    continue
                picked.append(nm)

                if len(picked) >= max_layers:
                    break
        return picked

    def reset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            if payload.get("version") != 1:
                raise ValueError("gridstate version must be 1")
            if "H" not in payload or "W" not in payload:
                raise ValueError("gridstate missing H/W")
            if not isinstance(payload.get("layers"), list):
                raise ValueError("gridstate missing layers[]")
            if not isinstance(payload.get("data"), dict):
                raise ValueError("gridstate missing data{}")

            # Keep in-memory copy; server is stateful.
            self.payload = payload
            self.tick = 0

            layers_meta = []
            for m in payload.get("layers", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                kind = m.get("kind")
                if isinstance(name, str):
                    layers_meta.append({"name": name, "kind": str(kind or "continuous")})

            return {
                "ok": True,
                "tick": self.tick,
                "H": int(payload["H"]),
                "W": int(payload["W"]),
                "layers": layers_meta,
            }

    def frame(self, layer_names: Optional[list]) -> Dict[str, Any]:
        with self._lock:
            if self.payload is None:
                raise ValueError("runtime not initialized: call /api/runtime/reset first")

            data = self.payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("payload missing data")

            if not layer_names:
                layer_names = self._default_layer_names()

            out_data: Dict[str, Any] = {}
            if layer_names:
                for nm in layer_names:
                    if not isinstance(nm, str):
                        continue
                    ent = data.get(nm)
                    if not isinstance(ent, dict):
                        continue
                    dtype = ent.get("dtype")
                    b64 = ent.get("b64")
                    if dtype != "float32" or not isinstance(b64, str):
                        continue
                    out_data[nm] = {"dtype": "float32", "b64": b64}

            H, W, layers, kinds = _decoded_layers_and_kinds(self.payload)
            events = self.payload.get("event_counters") if isinstance(self.payload, dict) else None
            if not isinstance(events, dict):
                events = {}
            return {
                "ok": True,
                "tick": self.tick,
                "data": out_data,
                "scalars": _compute_layer_scalars_from_layers(layers, kinds),
                "measurements": _compute_measurements_from_layers(self.payload, layers, H=H, W=W),
                "events": events,
            }

    def export(self) -> Dict[str, Any]:
        with self._lock:
            if self.payload is None:
                raise ValueError("runtime not initialized: call /api/runtime/reset first")

            base = self.payload
            H = int(base.get("H") or 0)
            W = int(base.get("W") or 0)
            if H <= 0 or W <= 0:
                raise ValueError("payload invalid H/W")

            layers_meta = base.get("layers")
            if not isinstance(layers_meta, list):
                raise ValueError("payload missing layers")

            data = base.get("data")
            if not isinstance(data, dict):
                raise ValueError("payload missing data")

            out: Dict[str, Any] = {
                "version": 1,
                "H": H,
                "W": W,
                "tick": int(self.tick),
                "layers": json.loads(json.dumps(layers_meta)),
                "data": {},
            }
            if "measurements_config" in base:
                out["measurements_config"] = json.loads(json.dumps(base.get("measurements_config")))
            if "layer_ops_config" in base:
                out["layer_ops_config"] = json.loads(json.dumps(base.get("layer_ops_config")))

            out_data: Dict[str, Any] = {}
            for m in layers_meta:
                if not isinstance(m, dict):
                    continue
                nm = m.get("name")
                if not isinstance(nm, str) or not nm:
                    continue
                ent = data.get(nm)
                if not isinstance(ent, dict) or ent.get("dtype") != "float32":
                    continue
                b64 = ent.get("b64")
                if isinstance(b64, str) and b64:
                    out_data[nm] = {"dtype": "float32", "b64": b64}
                    continue
                arr = ent.get("arr")
                if isinstance(arr, np.ndarray):
                    arr2 = np.asarray(arr, dtype=np.float32).reshape(H * W)
                    out_data[nm] = {"dtype": "float32", "b64": _encode_float32_b64(arr2)}
                    continue
            out["data"] = out_data
            return {"ok": True, "tick": int(self.tick), "payload": out}

    def step(self, layer_names: Optional[list]) -> Dict[str, Any]:
        with self._lock:
            if self.payload is None:
                raise ValueError("runtime not initialized: call /api/runtime/reset first")

            apply_layer_ops_inplace(self.payload, seed_offset=self.tick)
            self.tick += 1

            data = self.payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("payload missing data")

            if not layer_names:
                layer_names = self._default_layer_names()

            out_data: Dict[str, Any] = {}
            if layer_names:
                for nm in layer_names:
                    if not isinstance(nm, str):
                        continue
                    ent = data.get(nm)
                    if not isinstance(ent, dict):
                        continue
                    dtype = ent.get("dtype")
                    b64 = ent.get("b64")
                    if dtype != "float32" or not isinstance(b64, str):
                        continue
                    out_data[nm] = {"dtype": "float32", "b64": b64}

            H, W, layers, kinds = _decoded_layers_and_kinds(self.payload)
            events = self.payload.get("event_counters") if isinstance(self.payload, dict) else None
            if not isinstance(events, dict):
                events = {}
            return {
                "ok": True,
                "tick": self.tick,
                "data": out_data,
                "scalars": _compute_layer_scalars_from_layers(layers, kinds),
                "measurements": _compute_measurements_from_layers(self.payload, layers, H=H, W=W),
                "events": events,
            }


_RT = _RuntimeState()


class RuntimeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(_WEB_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        try:
            _LOG.debug("HTTP %s", (fmt % args) if args else str(fmt))
        except Exception:
            pass

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.settimeout(30)
        except Exception:
            pass

    def end_headers(self) -> None:
        # This is a local dev server; always disable caching so HTML/JS/CSS updates are picked up.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, code: int, obj: Dict[str, Any], *, extra_headers: Optional[Dict[str, str]] = None) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if isinstance(extra_headers, dict):
            for k, v in extra_headers.items():
                try:
                    if str(k or "").strip() and str(v or "").strip():
                        self.send_header(str(k), str(v))
                except Exception:
                    pass
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, code: int, data: bytes, *, content_type: str = "application/octet-stream", filename: str = "") -> None:
        raw = bytes(data or b"")
        self.send_response(int(code))
        self.send_header("Content-Type", str(content_type or "application/octet-stream"))
        if isinstance(filename, str) and filename.strip():
            safe_fn = str(Path(filename).name)
            self.send_header("Content-Disposition", f"attachment; filename=\"{safe_fn}\"")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except Exception:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"bad json: {e}")
        if not isinstance(obj, dict):
            raise ValueError("json body must be an object")
        return obj

    def do_HEAD(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/testing_api_points":
            self.path = "/testing_api_points.html"
        return super().do_HEAD()

    def do_GET(self):  # noqa: N802
        t0 = time.time()
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/testing_api_points":
                self.path = "/testing_api_points.html"
                super().do_GET()
                return
            if path == "/api/health":
                self._send_json(200, {"ok": True, "tick": int(_RT.tick)})
                return
            if path in ("/api/tests/cancer/models", "/api/tests/hereditary_disease/models", "/api/tests/aging/models"):
                if path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                self._send_json(200, {"ok": True, "challenge": challenge, "models": _tests_model_list_for_challenge(challenge)})
                return
            if path in ("/api/tests/cancer/proteins", "/api/tests/hereditary_disease/proteins", "/api/tests/aging/proteins"):
                if path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                mk = ""
                try:
                    qv = qs.get("model")
                    if isinstance(qv, list) and qv:
                        mk = str(qv[0] or "")
                except Exception:
                    mk = ""
                real_to_mask, _ = _tests_get_protein_mask_maps(mk, challenge=challenge)
                vals = [str(v) for v in list(real_to_mask.values()) if isinstance(v, str) and v]
                vals2 = sorted(set(vals), key=lambda s: int(s[len("protein_") :]) if s.startswith("protein_") and s[len("protein_") :].isdigit() else 10**9)
                self._send_json(200, {"ok": True, "challenge": challenge, "model": str(mk), "proteins": vals2})
                return
            if path in ("/api/tests/cancer/protein_layers", "/api/tests/hereditary_disease/protein_layers", "/api/tests/aging/protein_layers"):
                if path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                mk = ""
                try:
                    qv = qs.get("model")
                    if isinstance(qv, list) and qv:
                        mk = str(qv[0] or "")
                except Exception:
                    mk = ""
                real_to_mask, _ = _tests_get_protein_mask_maps(mk, challenge=challenge)
                vals = [str(v) for v in list(real_to_mask.values()) if isinstance(v, str) and v]
                vals2 = sorted(set(vals), key=lambda s: int(s[len("protein_") :]) if s.startswith("protein_") and s[len("protein_") :].isdigit() else 10**9)
                self._send_json(200, {"ok": True, "challenge": challenge, "model": str(mk), "protein_layers": vals2})
                return
            if path == "/api/spatial_tx/gene_sets":
                self._send_json(200, {"ok": True, "gene_sets": _list_stx_gene_sets()})
                return
            if path == "/api/spatial_omics/type":
                self._send_json(200, {"ok": True, "types": _list_spatial_omics_types()})
                return
            if path == "/api/bulk_omics/sets":
                self._send_json(200, {"ok": True, "sets": _list_bulk_omics_sets()})
                return
            if path == "/api/game/state":
                pid = ""
                try:
                    qv = qs.get("player_id")
                    if isinstance(qv, list) and qv:
                        pid = str(qv[0] or "")
                except Exception:
                    pid = ""
                self._send_json(200, {"ok": True, "game": _game_get_player_state(pid)})
                return
            if path == "/api/omics/runs":
                limit_i = 200
                try:
                    qv = qs.get("limit")
                    if isinstance(qv, list) and qv:
                        limit_i = int(qv[0])
                except Exception:
                    limit_i = 200
                runs = _OMICS.list_runs(limit=int(limit_i))
                out_runs: list[Dict[str, Any]] = []
                for r in runs:
                    if not isinstance(r, dict):
                        continue
                    out_runs.append(
                        {
                            "run_id": str(r.get("run_id") or ""),
                            "created_at": float(r.get("created_at") or 0.0),
                            "experiment": str(r.get("experiment") or ""),
                            "kind": str(r.get("kind") or ""),
                            "model": str(r.get("model") or ""),
                            "ticks": int(r.get("ticks") or 0),
                            "replicates": int(r.get("replicates") or 0),
                            "omics_set": str(r.get("omics_set") or ""),
                            "gene_set": str(r.get("gene_set") or ""),
                        }
                    )

                self._send_json(200, {"ok": True, "runs": out_runs})
                return
            if path == "/api/omics/inventory":
                pid = ""
                view = "compact"
                try:
                    qv = qs.get("player_id")
                    if isinstance(qv, list) and qv:
                        pid = str(qv[0] or "")
                except Exception:
                    pid = ""
                try:
                    qv = qs.get("view")
                    if isinstance(qv, list) and qv:
                        view = str(qv[0] or "compact")
                except Exception:
                    view = "compact"
                if not pid:
                    raise ValueError("missing player_id")
                inv = _OMICS.inventory(pid, view=str(view or "compact"))
                self._send_json(200, {"ok": True, **inv})
                return
            if path == "/api/omics/run":
                rid = ""
                try:
                    qv = qs.get("run_id")
                    if isinstance(qv, list) and qv:
                        rid = str(qv[0] or "")
                except Exception:
                    rid = ""
                if not rid:
                    raise ValueError("missing run_id")
                manifest = _OMICS.get_run(rid)
                files = _OMICS.list_files(rid)
                out_files: list[Dict[str, Any]] = []
                if isinstance(files, list):
                    for ent in files:
                        if not isinstance(ent, dict):
                            continue
                        name = str(ent.get("name") or "")
                        if not name:
                            continue
                        try:
                            fid = _OMICS._file_id(str(rid), str(name))
                        except Exception:
                            fid = ""
                        out_files.append(
                            {
                                "name": str(name),
                                "bytes": int(ent.get("bytes") or 0),
                                "file_id": str(fid),
                                "download_url": f"/api/omics/file?run_id={str(rid)}&name={str(name)}",
                            }
                        )
                self._send_json(200, {"ok": True, "run": manifest, "files": out_files})
                return
            if path == "/api/omics/file":
                rid = ""
                name = ""
                pid = ""
                fid = ""
                try:
                    qv = qs.get("run_id")
                    if isinstance(qv, list) and qv:
                        rid = str(qv[0] or "")
                except Exception:
                    rid = ""
                try:
                    qv = qs.get("player_id")
                    if isinstance(qv, list) and qv:
                        pid = str(qv[0] or "")
                except Exception:
                    pid = ""
                try:
                    qv = qs.get("file_id")
                    if isinstance(qv, list) and qv:
                        fid = str(qv[0] or "")
                except Exception:
                    fid = ""
                try:
                    qv = qs.get("name")
                    if isinstance(qv, list) and qv:
                        name = str(qv[0] or "")
                except Exception:
                    name = ""

                if fid:
                    matches: list[Dict[str, Any]] = []
                    if pid:
                        matches = _OMICS.resolve_player_file_ids(pid, [fid])
                    else:
                        matches, ambiguous = _OMICS.resolve_file_ids([fid])
                        if ambiguous:
                            raise ValueError("file_id is ambiguous")
                    if not matches:
                        raise ValueError("file not found")
                    m0 = matches[0]
                    rid = str(m0.get("run_id") or "")
                    name = str(m0.get("name") or "")
                if not rid:
                    raise ValueError("missing run_id")
                if not name:
                    raise ValueError("missing name")

                p = _OMICS.file_path(rid, name)
                data = p.read_bytes()
                ct = "application/octet-stream"
                suf = p.suffix.lower()
                if suf == ".csv":
                    ct = "text/csv"
                elif suf == ".json":
                    ct = "application/json"
                self._send_bytes(200, data, content_type=ct, filename=p.name)
                return
            if path == "/api/doc/status":
                self._send_json(200, _DOC.status())
                return
            if path == "/api/doc/list":
                self._send_json(200, _DOC.list_docs())
                return
            super().do_GET()
        except Exception as e:
            err_id = uuid.uuid4().hex[:12]
            try:
                _LOG.exception("GET handler error id=%s path=%s", str(err_id), str(self.path))
            except Exception:
                pass
            if str(path).startswith("/api/"):
                code = 500
                if isinstance(e, ValueError):
                    code = 400
                try:
                    self._send_json(int(code), {"ok": False, "error": str(e), "error_id": str(err_id)})
                    return
                except Exception:
                    return
            raise
        finally:
            try:
                dt_ms = (time.time() - t0) * 1000.0
                _LOG.info("GET %s %.1fms", str(self.path), float(dt_ms))
            except Exception:
                pass

    def do_POST(self):  # noqa: N802
        t0 = time.time()
        try:
            if self.path == "/api/doc/clear":
                _DOC.clear_active()
                self._send_json(200, _DOC.status())
                return

            if self.path == "/api/discuss":
                body = self._read_json_body()
                player_id = body.get("player_id")
                pid = _sanitize_player_id(player_id)
                if not pid:
                    raise ValueError("missing player_id")

                problem = body.get("problem")
                if not isinstance(problem, str) or not problem.strip():
                    problem = body.get("question")
                if not isinstance(problem, str) or not problem.strip():
                    problem = body.get("message")
                if not isinstance(problem, str) or not problem.strip():
                    raise ValueError("missing problem")

                extra_ctx = body.get("context")
                ctx_txt = ""
                if isinstance(extra_ctx, str) and extra_ctx.strip():
                    ctx_txt = str(extra_ctx).strip()

                prov = str(os.environ.get("DT_DISCUSS_PROVIDER") or "openai").strip().lower() or "openai"
                model = str(os.environ.get("DT_DISCUSS_MODEL") or "").strip()
                if prov == "claude":
                    prov = "anthropic"
                if prov == "grok":
                    prov = "xai"
                if not model:
                    if prov == "anthropic":
                        model = "claude-sonnet-4-5-20250929"
                    elif prov == "xai":
                        model = "grok-4"
                    elif prov == "gemini":
                        model = "gemini-2.5-pro"
                    else:
                        model = "gpt-5.2"

                timeout_s = 60.0
                try:
                    timeout_s = float(os.environ.get("DT_DISCUSS_TIMEOUT_S", str(timeout_s)) or timeout_s)
                except Exception:
                    timeout_s = 60.0
                timeout_s = max(5.0, min(600.0, float(timeout_s)))

                max_tokens = 500
                try:
                    max_tokens = int(os.environ.get("DT_DISCUSS_MAX_TOKENS", str(max_tokens)) or max_tokens)
                except Exception:
                    max_tokens = 500
                max_tokens = max(64, min(2000, int(max_tokens)))

                user_prompt = "Player problem:\n" + str(problem).strip()
                if ctx_txt:
                    user_prompt = user_prompt + "\n\nAdditional context:\n" + str(ctx_txt)

                advice_raw = _discuss_llm_generate(
                    provider=str(prov),
                    model=str(model),
                    system_prompt=str(_DISCUSS_ADVISOR_SYSTEM_PROMPT),
                    user_prompt=str(user_prompt),
                    timeout_s=float(timeout_s),
                    max_tokens=int(max_tokens),
                )
                advice = _discuss_postprocess_advice(advice_raw)

                try:
                    _LOG.info(
                        "discuss player_id=%s provider=%s model=%s problem_chars=%d advice_chars=%d",
                        str(pid),
                        str(prov),
                        str(model),
                        int(len(str(problem or ""))),
                        int(len(str(advice or ""))),
                    )
                except Exception:
                    pass

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "player_id": str(pid),
                        "provider": str(prov),
                        "model": str(model),
                        "advice": str(advice or ""),
                    },
                )
                return

            if self.path == "/api/omics/analyze":
                body = self._read_json_body()
                run_id = body.get("run_id")
                instructions = body.get("instructions")
                files_req = body.get("files")
                file_ids_req = body.get("file_ids")
                player_id_req = body.get("player_id")
                provider_req = body.get("provider")
                model_req = body.get("model")
                memory_limit_req = body.get("memory_limit")
                max_tokens_req = body.get("max_tokens")
                auto_continue_req = body.get("auto_continue")
                max_continuations_req = body.get("max_continuations")

                if not isinstance(instructions, str) or not instructions.strip():
                    raise ValueError("missing instructions")

                def _mask_disease_term(text: str) -> str:
                    t = str(text or "")
                    if not t:
                        return ""
                    t = t.replace("tests_cancer_", "tests_disease_")
                    t = t.replace("healthy_organism", "healthy")
                    t = t.replace("cancer_organism", "disease")
                    t = t.replace("healthy_cell_culture", "cell_culture_healthy")
                    t = t.replace("cancer_cell_culture", "cell_culture_disease")
                    t = t.replace("/api/tests/cancer/", "/api/tests/disease/")
                    t = t.replace("/api/tests/cancer", "/api/tests/disease")
                    t = t.replace("/api/tests/hereditary_disease/", "/api/tests/disease/")
                    t = t.replace("/api/tests/hereditary_disease", "/api/tests/disease")
                    t = t.replace("/api/tests/aging/", "/api/tests/disease/")
                    t = t.replace("/api/tests/aging", "/api/tests/disease")
                    t = re.sub(r"\bcancerous\b", "diseased", t, flags=re.IGNORECASE)
                    t = re.sub(r"\bcancer\b", "disease", t, flags=re.IGNORECASE)
                    return t

                instructions_prefix = (
                    "You will be communicating with an LLM. Do not generate files, plots, or charts. "
                    "Describe your findings in as much detail as possible using pure text. "
                    "If helpful, you may include a small table in plain text. "
                    "IMPORTANT: Avoid strategic or normative language (e.g., 'candidate', 'recommend', 'therapy', 'inhibit', 'activate') "
                    "unless the user's instructions explicitly ask you to recommend next steps. Default to descriptive reporting of computed results only. "
                    "IMPORTANT: If you encounter any errors reading files or executing code, you must explicitly report the errors and stop; do not guess or approximate results. "
                    "IMPORTANT: Always use the actual column names from the attached files as identifiers; never assume protein_1..protein_N ordering by index. "
                    "IMPORTANT: The user's instructions may include additional context beyond the attachments (e.g., characterization or experimental setup). "
                    "Clearly separate (A) results computed from attached files vs (B) statements inferred from extra context, and never present inferred statements as computed results."
                )
                instructions_eff = _mask_disease_term(instructions_prefix + "\n\n" + str(instructions))

                prov = str(provider_req or "openai").strip().lower() or "openai"
                if prov == "claude":
                    prov = "anthropic"
                if prov == "grok":
                    prov = "xai"

                selected: list[Dict[str, Any]] = []
                if isinstance(file_ids_req, list) and file_ids_req:
                    fids = [str(x or "").strip() for x in file_ids_req]
                    fids = [x for x in fids if x]
                    if not fids:
                        raise ValueError("no file_ids selected")
                    pid = _sanitize_player_id(player_id_req)
                    if pid:
                        selected = _OMICS.resolve_player_file_ids(pid, fids)
                    else:
                        selected, ambiguous = _OMICS.resolve_file_ids(fids)
                        if ambiguous:
                            raise ValueError("ambiguous file_id(s): " + ", ".join([str(x) for x in ambiguous[:20]]))
                    if not selected:
                        raise ValueError(
                            "no files selected (0/%d file_ids matched inventory; pass file_id values from GET /api/omics/inventory?player_id=...)" % int(len(fids))
                        )
                    try:
                        idx_by_fid = {str(fid): int(i) for i, fid in enumerate(list(fids)) if str(fid)}
                        selected.sort(key=lambda d: idx_by_fid.get(str((d or {}).get("file_id") or ""), 10**9))
                    except Exception:
                        pass
                else:
                    if not isinstance(run_id, str) or not run_id.strip():
                        raise ValueError("missing run_id")
                    manifest = _OMICS.get_run(run_id)
                    all_files = _OMICS.list_files(run_id)
                    all_names = [str(f.get("name") or "") for f in all_files if isinstance(f, dict)]
                    all_names = [n for n in all_names if n]

                    if isinstance(files_req, list) and files_req:
                        req_names = [str(x or "") for x in files_req]
                        req_names = [n for n in req_names if n]
                        names = [n for n in req_names if n in set(all_names)]
                    else:
                        names = list(all_names)
                    if not names:
                        raise ValueError("no files selected")
                    for nm in names:
                        selected.append({"run_id": str(run_id), "name": str(nm)})

                manifest = None
                try:
                    if isinstance(run_id, str) and run_id.strip():
                        manifest = _OMICS.get_run(run_id)
                except Exception:
                    manifest = None

                analyze_manifest_entries: list[Dict[str, Any]] = []
                for i, ent in enumerate(list(selected)):
                    if not isinstance(ent, dict):
                        continue
                    label = "FILE_%02d" % int(i + 1)
                    ent["analyze_label"] = str(label)
                    analyze_manifest_entries.append(
                        {
                            "label": str(label),
                            "run_id": str(ent.get("run_id") or ""),
                            "file_id": str(ent.get("file_id") or ""),
                            "display_name": str(ent.get("display_name") or ent.get("name") or ""),
                            "name": str(ent.get("name") or ""),
                        }
                    )

                if analyze_manifest_entries:
                    analyze_manifest_text = (
                        "ATTACHMENT MANIFEST (authoritative):\n"
                        "The files you can read are listed below with stable labels.\n"
                        "When you refer to a file or dataset, use the label (e.g., FILE_01).\n"
                        "Do NOT infer which file is baseline/treated from filenames; follow the user's instructions and map roles to FILE_XX explicitly.\n\n"
                        + json.dumps(analyze_manifest_entries, indent=2)
                    )
                    instructions_eff = str(instructions_eff) + "\n\n" + _mask_disease_term(str(analyze_manifest_text))

                model = "gpt-5.2"
                if prov == "anthropic":
                    model = "claude-sonnet-4-5-20250929"
                if prov == "xai":
                    model = "grok-4"
                if prov == "gemini":
                    model = "gemini-2.5-pro"
                memory_limit = str(memory_limit_req or "4g")
                if memory_limit not in ("1g", "4g", "16g", "64g"):
                    memory_limit = "4g"

                if isinstance(model_req, str) and model_req.strip():
                    model = str(model_req).strip()

                stx_pool_k = 4
                try:
                    stx_pool_k = int(os.environ.get("DT_OMICS_ANALYZE_STX_POOL_K", str(stx_pool_k)) or stx_pool_k)
                except Exception:
                    stx_pool_k = 4
                stx_pool_k = max(1, int(stx_pool_k))

                ent_by_rid_name: Dict[tuple[str, str], Dict[str, Any]] = {}
                for ent in selected:
                    rid0 = str(ent.get("run_id") or "")
                    name0 = str(ent.get("name") or "")
                    if rid0 and name0:
                        ent_by_rid_name[(str(rid0), str(name0))] = ent

                pooled_bytes_by_rid_name: Dict[tuple[str, str], bytes] = {}
                if int(stx_pool_k) > 1:
                    for (rid0, name0), _ent0 in ent_by_rid_name.items():
                        if not str(name0).endswith("_matrix.csv"):
                            continue
                        meta_name0 = str(name0[:-len("_matrix.csv")] + "_cell_metadata.csv")
                        if (str(rid0), str(meta_name0)) not in ent_by_rid_name:
                            continue
                        try:
                            p_mat = _OMICS.file_path(str(rid0), str(name0))
                            p_meta = _OMICS.file_path(str(rid0), str(meta_name0))
                            mat_bytes = p_mat.read_bytes()
                            meta_bytes = p_meta.read_bytes()
                            pooled_mat, pooled_meta = _stx_pool_matrix_and_metadata_csv(
                                mat_bytes,
                                meta_bytes,
                                k=int(stx_pool_k),
                            )
                            pooled_bytes_by_rid_name[(str(rid0), str(name0))] = bytes(pooled_mat)
                            pooled_bytes_by_rid_name[(str(rid0), str(meta_name0))] = bytes(pooled_meta)
                        except Exception:
                            pass


                if prov in ("openai", "openai_compat"):
                    if OpenAI is None:
                        raise ValueError("openai sdk not installed")
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not isinstance(api_key, str) or not api_key.strip():
                        raise ValueError("missing OPENAI_API_KEY")

                    client = OpenAI(api_key=api_key)
                    file_ids: list[str] = []
                    uploaded_bytes_total = 0

                    used_files: list[Dict[str, Any]] = []
                    for ent in selected:
                        rid0 = str(ent.get("run_id") or "")
                        name0 = str(ent.get("name") or "")
                        if not rid0 or not name0:
                            continue
                        p = _OMICS.file_path(rid0, name0)

                        raw_bytes = b""
                        try:
                            raw_bytes = p.read_bytes()
                        except Exception:
                            raw_bytes = b""

                        pb = pooled_bytes_by_rid_name.get((str(rid0), str(name0)))
                        if pb is not None:
                            raw_bytes = bytes(pb)

                        label = str(ent.get("analyze_label") or "")
                        upload_name = str(p.name)
                        if label:
                            upload_name = str(label) + "__" + str(upload_name)
                        suf = str(p.suffix or "").lower()
                        if suf in (".csv", ".tsv", ".txt", ".json"):
                            try:
                                txt0 = raw_bytes.decode("utf-8", errors="replace")
                                txt1 = _mask_disease_term(txt0)
                                raw_bytes = txt1.encode("utf-8")
                                upload_name = _mask_disease_term(upload_name)
                            except Exception:
                                pass

                        try:
                            uploaded_bytes_total += int(len(raw_bytes or b""))
                        except Exception:
                            pass

                        bio = io.BytesIO(raw_bytes)
                        try:
                            bio.name = upload_name
                        except Exception:
                            pass
                        fo = client.files.create(file=bio, purpose="user_data")
                        fid = str(getattr(fo, "id", "") or "")
                        if fid:
                            file_ids.append(fid)
                        used_files.append({
                            "label": str(label),
                            "run_id": rid0,
                            "name": name0,
                            "file_id": str(ent.get("file_id") or ""),
                            "display_name": str(ent.get("display_name") or name0),
                        })

                    tools = [
                        {
                            "type": "code_interpreter",
                            "container": {"type": "auto", "file_ids": file_ids, "memory_limit": memory_limit},
                        }
                    ]

                    max_out_tokens = 120000
                    try:
                        if max_tokens_req is not None:
                            max_out_tokens = int(max_tokens_req)
                    except Exception:
                        max_out_tokens = 120000
                    max_out_tokens = max(256, min(120000, int(max_out_tokens)))

                    base_model, effort = _openai_base_model_and_effort(str(model))
                    model_call = str(base_model or model)
                    eff = str(effort or "medium")
                    reasoning: Dict[str, Any] = {"effort": str(eff), "summary": "auto"}
                    if str(model or "").strip() == "gpt-5.2":
                        reasoning = {"effort": "high", "summary": None}
                    max_judge_attempts = 3
                    try:
                        max_judge_attempts = int(os.environ.get("DT_OMICS_ANALYZE_MAX_JUDGE_ATTEMPTS", str(max_judge_attempts)) or max_judge_attempts)
                    except Exception:
                        max_judge_attempts = 3
                    max_judge_attempts = max(1, min(3, int(max_judge_attempts)))

                    judge_traces: list[Dict[str, Any]] = []
                    resp = None
                    resp_dump: Any = None
                    out_text_raw = ""
                    out_text = ""
                    diag: Dict[str, Any] = {}
                    retry_guidance = ""

                    for judge_attempt in range(int(max_judge_attempts)):
                        prompt_attempt = _apply_omics_analyze_retry_guidance(str(instructions_eff), str(retry_guidance))
                        try:
                            try:
                                resp = client.responses.create(
                                    model=str(model_call),
                                    input=str(prompt_attempt),
                                    text={"format": {"type": "text"}, "verbosity": "medium"},
                                    reasoning=reasoning,
                                    tools=tools,
                                    include=["code_interpreter_call.outputs"],
                                    max_output_tokens=int(max_out_tokens),
                                )
                            except Exception:
                                resp = client.responses.create(
                                    model=str(model_call),
                                    input=str(prompt_attempt),
                                    text={"format": {"type": "text"}, "verbosity": "medium"},
                                    reasoning=reasoning,
                                    tools=tools,
                                    include=["code_interpreter_call.outputs"],
                                )
                        except Exception:
                            raise

                        out_text_raw = ""
                        try:
                            out_text_raw = str(getattr(resp, "output_text", "") or "")
                        except Exception:
                            out_text_raw = ""

                        resp_dump = None
                        try:
                            resp_dump = resp.model_dump()
                        except Exception:
                            resp_dump = None

                        out_text, diag = _omics_analyze_diagnostics(out_text_raw, resp_dump)

                        judge = _omics_analyze_judge(
                            player_instructions=str(instructions),
                            analyze_manifest_entries=list(analyze_manifest_entries),
                            output_text=str(out_text_raw),
                            analysis_diagnostics=dict(diag),
                            provider="openai",
                            model=str(model_call),
                            attempt=int(judge_attempt) + 1,
                            max_attempts=int(max_judge_attempts),
                        )
                        judge_traces.append({"attempt": int(judge_attempt) + 1, **dict(judge)})

                        if str(judge.get("decision")) == "retry":
                            retry_guidance = _merge_omics_analyze_retry_guidance(
                                str(retry_guidance),
                                str(judge.get("retry_instructions") or ""),
                            )

                        if str(judge.get("decision")) != "retry" or judge_attempt >= (int(max_judge_attempts) - 1):
                            break

                    _maybe_log_truncation_alarm(
                        out_text,
                        provider="openai",
                        model=str(model_call),
                        path="/api/omics/analyze",
                        run_id=str(run_id or "") if run_id is not None else None,
                        player_id=str(player_id_req or "") if player_id_req is not None else None,
                    )

                    try:
                        if judge_traces and str(judge_traces[-1].get("decision")) == "retry":
                            summary = (
                                "\n\nAUTO_RETRY_EXHAUSTED: The analysis tool output still appeared incomplete or erroneous after "
                                + str(len(judge_traces))
                                + " attempt(s).\n"
                                + "Judge_reason: "
                                + str(judge_traces[-1].get("reason") or "")
                            )
                            out_text = str(out_text or "") + str(summary)
                    except Exception:
                        pass

                    if isinstance(diag, dict):
                        diag["judge"] = {"attempts": int(len(judge_traces)), "trace": list(judge_traces)}

                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "run_id": str(run_id or ""),
                            "manifest": manifest,
                            "files": used_files,
                            "provider": "openai",
                            "openai": {
                                "model": str(model),
                                "memory_limit": str(memory_limit),
                                "file_ids": file_ids,
                            },
                            "output_text": out_text,
                            "analysis_diagnostics": diag,
                            "response": resp_dump,
                        },
                    )
                    return

                if prov in ("gemini",):
                    api_key_g = os.environ.get("GEMINI_API_KEY")
                    if not isinstance(api_key_g, str) or not api_key_g.strip():
                        raise ValueError("missing GEMINI_API_KEY")
                    base_url_g = str(os.environ.get("GEMINI_BASE_URL") or "").strip() or "https://generativelanguage.googleapis.com/v1beta"

                    max_tokens_g = 8192
                    try:
                        if max_tokens_req is not None:
                            max_tokens_g = int(max_tokens_req)
                    except Exception:
                        max_tokens_g = 8192
                    max_tokens_g = max(256, min(65536, int(max_tokens_g)))

                    used_files: list[Dict[str, Any]] = []
                    parts: list[Dict[str, Any]] = [{"text": str(instructions_eff)}]

                    uploaded_bytes_total = 0
                    for ent in selected:
                        rid0 = str(ent.get("run_id") or "")
                        name0 = str(ent.get("name") or "")
                        if not rid0 or not name0:
                            continue
                        p = _OMICS.file_path(rid0, name0)

                        raw_bytes = b""
                        try:
                            raw_bytes = p.read_bytes()
                        except Exception:
                            raw_bytes = b""

                        pb = pooled_bytes_by_rid_name.get((str(rid0), str(name0)))
                        if pb is not None:
                            raw_bytes = bytes(pb)

                        try:
                            uploaded_bytes_total += int(len(raw_bytes or b""))
                        except Exception:
                            pass

                        suf = str(p.suffix or "").lower()
                        mime_type = "application/octet-stream"
                        if suf == ".csv":
                            mime_type = "text/csv"
                        elif suf == ".tsv":
                            mime_type = "text/tab-separated-values"
                        elif suf == ".txt":
                            mime_type = "text/plain"
                        elif suf == ".json":
                            mime_type = "application/json"

                        if suf in (".csv", ".tsv", ".txt", ".json"):
                            try:
                                txt0 = raw_bytes.decode("utf-8", errors="replace")
                                txt1 = _mask_disease_term(txt0)
                                raw_bytes = txt1.encode("utf-8")
                            except Exception:
                                pass

                        b64 = ""
                        try:
                            b64 = base64.b64encode(raw_bytes or b"").decode("ascii")
                        except Exception:
                            b64 = ""
                        parts.append({"inlineData": {"mimeType": str(mime_type), "data": str(b64)}})

                        used_files.append(
                            {
                                "label": str(ent.get("analyze_label") or ""),
                                "run_id": rid0,
                                "name": name0,
                                "file_id": str(ent.get("file_id") or ""),
                                "display_name": str(ent.get("display_name") or name0),
                            }
                        )

                    if int(uploaded_bytes_total) > 18 * 1024 * 1024:
                        raise ValueError("files too large for gemini inlineData")

                    url = str(base_url_g).rstrip("/") + "/models/" + str(model) + ":generateContent"
                    payload: Dict[str, Any] = {
                        "tools": [{"code_execution": {}}],
                        "contents": [{"role": "user", "parts": parts}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": int(max_tokens_g),
                        },
                    }
                    if str(model or "").strip().startswith("gemini-3-"):
                        payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "high"}
                    headers = {
                        "x-goog-api-key": str(api_key_g),
                        "content-type": "application/json",
                    }

                    resp_json: Dict[str, Any] = {}
                    last_err: Optional[str] = None
                    rate_limited = False
                    sleep_budget_s = 300.0
                    try:
                        sleep_budget_s = float(os.environ.get("DT_GEMINI_RATE_LIMIT_SLEEP_BUDGET_S", str(sleep_budget_s)) or sleep_budget_s)
                    except Exception:
                        sleep_budget_s = 300.0
                    sleep_budget_s = max(0.0, min(3600.0, float(sleep_budget_s)))

                    attempts = int(max(3, _gemini_retry_attempts()))
                    for attempt in range(int(attempts)):
                        rem_s = _gemini_cooldown_remaining_s(base_url=str(base_url_g), model=str(model))
                        if rem_s > 0.0:
                            if sleep_budget_s <= 0.0:
                                rate_limited = True
                                try:
                                    _gemini_debug_write_snapshot(
                                        event="cooldown_budget_exhausted",
                                        provider="gemini",
                                        model=str(model),
                                        base_url=str(base_url_g),
                                        url=str(url),
                                        headers=dict(headers),
                                        payload=dict(payload),
                                        error=None,
                                        attempt=int(attempt),
                                        attempts=int(attempts),
                                        sleep_budget_s=float(sleep_budget_s),
                                        cooldown_remaining_s=float(rem_s),
                                        path="/api/omics/analyze",
                                        run_id=str(run_id or "") if run_id is not None else None,
                                        player_id=str(player_id_req or "") if player_id_req is not None else None,
                                    )
                                except Exception:
                                    pass
                                break
                            try:
                                sleep_s = min(float(rem_s), float(sleep_budget_s))
                                time.sleep(float(max(0.0, sleep_s)))
                                sleep_budget_s -= float(sleep_s)
                            except Exception:
                                pass
                            continue

                        try:
                            t_s = float(600.0)
                            if attempt >= 1:
                                t_s = float(600.0) + float(300.0) * float(min(5, int(attempt)))
                            resp_json = _http_post_json(url=str(url), headers=headers, payload=payload, timeout_s=t_s)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = str(e)
                            if _is_rate_limited_error(last_err):
                                rate_limited = True
                                hint_s = _retry_delay_seconds_from_error(last_err)
                                wait_s = _gemini_set_cooldown(base_url=str(base_url_g), model=str(model), retry_after_s=hint_s)
                                try:
                                    _gemini_debug_write_snapshot(
                                        event="rate_limited",
                                        provider="gemini",
                                        model=str(model),
                                        base_url=str(base_url_g),
                                        url=str(url),
                                        headers=dict(headers),
                                        payload=dict(payload),
                                        error=str(last_err),
                                        attempt=int(attempt),
                                        attempts=int(attempts),
                                        sleep_budget_s=float(sleep_budget_s),
                                        cooldown_remaining_s=float(wait_s),
                                        path="/api/omics/analyze",
                                        run_id=str(run_id or "") if run_id is not None else None,
                                        player_id=str(player_id_req or "") if player_id_req is not None else None,
                                    )
                                except Exception:
                                    pass
                                if attempt >= (int(attempts) - 1) or sleep_budget_s <= 0.0:
                                    break
                                try:
                                    sleep_s = min(float(wait_s), float(sleep_budget_s))
                                    time.sleep(float(max(0.0, sleep_s)))
                                    sleep_budget_s -= float(sleep_s)
                                except Exception:
                                    pass
                                continue
                            if _should_retry_remote_http_error(last_err) and attempt < (int(attempts) - 1):
                                try:
                                    _LOG.warning("Gemini /generateContent retry attempt=%d error=%s", int(attempt + 1), str(last_err)[:300])
                                except Exception:
                                    pass
                                try:
                                    sleep_s = float(1.0 + (2.0 * float(attempt)))
                                    hint_s = _retry_delay_seconds_from_error(last_err)
                                    if hint_s is not None:
                                        sleep_s = max(float(sleep_s), float(hint_s))
                                    sleep_s = min(60.0, max(0.0, float(sleep_s)))
                                    time.sleep(float(sleep_s))
                                except Exception:
                                    pass
                                continue
                            raise
                    if last_err:
                        if rate_limited:
                            try:
                                _gemini_debug_write_snapshot(
                                    event="temporary_unavailable",
                                    provider="gemini",
                                    model=str(model),
                                    base_url=str(base_url_g),
                                    url=str(url),
                                    headers=dict(headers),
                                    payload=dict(payload),
                                    error=str(last_err),
                                    attempt=None,
                                    attempts=int(attempts),
                                    sleep_budget_s=float(sleep_budget_s),
                                    cooldown_remaining_s=_gemini_cooldown_remaining_s(base_url=str(base_url_g), model=str(model)),
                                    path="/api/omics/analyze",
                                    run_id=str(run_id or "") if run_id is not None else None,
                                    player_id=str(player_id_req or "") if player_id_req is not None else None,
                                )
                            except Exception:
                                pass
                            raise TemporaryUnavailableError("Gemini temporarily unavailable", provider="gemini", model=str(model))
                        raise ValueError(str(last_err))
                    if rate_limited and last_err is None:
                        try:
                            _gemini_debug_write_snapshot(
                                event="temporary_unavailable",
                                provider="gemini",
                                model=str(model),
                                base_url=str(base_url_g),
                                url=str(url),
                                headers=dict(headers),
                                payload=dict(payload),
                                error=None,
                                attempt=None,
                                attempts=int(attempts),
                                sleep_budget_s=float(sleep_budget_s),
                                cooldown_remaining_s=_gemini_cooldown_remaining_s(base_url=str(base_url_g), model=str(model)),
                                path="/api/omics/analyze",
                                run_id=str(run_id or "") if run_id is not None else None,
                                player_id=str(player_id_req or "") if player_id_req is not None else None,
                            )
                        except Exception:
                            pass
                        raise TemporaryUnavailableError("Gemini temporarily unavailable", provider="gemini", model=str(model))

                    max_judge_attempts = 3
                    try:
                        max_judge_attempts = int(os.environ.get("DT_OMICS_ANALYZE_MAX_JUDGE_ATTEMPTS", str(max_judge_attempts)) or max_judge_attempts)
                    except Exception:
                        max_judge_attempts = 3
                    max_judge_attempts = max(1, min(3, int(max_judge_attempts)))

                    judge_traces: list[Dict[str, Any]] = []
                    out_text_raw = ""
                    out_text = ""
                    diag: Dict[str, Any] = {}
                    retry_guidance = ""

                    for judge_attempt in range(int(max_judge_attempts)):
                        out_text_raw = ""
                        try:
                            candidates = resp_json.get("candidates") if isinstance(resp_json, dict) else None
                            if isinstance(candidates, list) and candidates:
                                c0 = candidates[0] if isinstance(candidates[0], dict) else {}
                                content0 = c0.get("content") if isinstance(c0, dict) else None
                                parts0 = content0.get("parts") if isinstance(content0, dict) else None
                                if isinstance(parts0, list) and parts0:
                                    out_chunks: list[str] = []
                                    for p0 in parts0:
                                        if isinstance(p0, dict) and isinstance(p0.get("text"), str):
                                            out_chunks.append(str(p0.get("text") or ""))
                                    out_text_raw = "".join(out_chunks).strip()
                        except Exception:
                            out_text_raw = ""

                        out_text, diag = _omics_analyze_diagnostics(out_text_raw, resp_json)

                        judge = _omics_analyze_judge(
                            player_instructions=str(instructions),
                            analyze_manifest_entries=list(analyze_manifest_entries),
                            output_text=str(out_text_raw),
                            analysis_diagnostics=dict(diag),
                            provider="gemini",
                            model=str(model),
                            attempt=int(judge_attempt) + 1,
                            max_attempts=int(max_judge_attempts),
                        )
                        judge_traces.append({"attempt": int(judge_attempt) + 1, **dict(judge)})

                        if str(judge.get("decision")) == "retry":
                            retry_guidance = _merge_omics_analyze_retry_guidance(
                                str(retry_guidance),
                                str(judge.get("retry_instructions") or ""),
                            )
                            try:
                                parts[0]["text"] = _apply_omics_analyze_retry_guidance(str(instructions_eff), str(retry_guidance))
                            except Exception:
                                pass

                        if str(judge.get("decision")) != "retry" or judge_attempt >= (int(max_judge_attempts) - 1):
                            break

                        last_err = None
                        rate_limited = False
                        sleep_budget_s = 300.0
                        try:
                            sleep_budget_s = float(os.environ.get("DT_GEMINI_RATE_LIMIT_SLEEP_BUDGET_S", str(sleep_budget_s)) or sleep_budget_s)
                        except Exception:
                            sleep_budget_s = 300.0
                        sleep_budget_s = max(0.0, min(3600.0, float(sleep_budget_s)))

                        attempts = int(max(3, _gemini_retry_attempts()))
                        for attempt in range(int(attempts)):
                            rem_s = _gemini_cooldown_remaining_s(base_url=str(base_url_g), model=str(model))
                            if rem_s > 0.0:
                                if sleep_budget_s <= 0.0:
                                    rate_limited = True
                                    break
                                try:
                                    sleep_s = min(float(rem_s), float(sleep_budget_s))
                                    time.sleep(float(max(0.0, sleep_s)))
                                    sleep_budget_s -= float(sleep_s)
                                except Exception:
                                    pass
                                continue
                            try:
                                t_s = float(600.0)
                                if attempt >= 1:
                                    t_s = float(600.0) + float(300.0) * float(min(5, int(attempt)))
                                resp_json = _http_post_json(url=str(url), headers=headers, payload=payload, timeout_s=t_s)
                                last_err = None
                                break
                            except Exception as e:
                                last_err = str(e)
                                if _is_rate_limited_error(last_err):
                                    rate_limited = True
                                    hint_s = _retry_delay_seconds_from_error(last_err)
                                    wait_s = _gemini_set_cooldown(base_url=str(base_url_g), model=str(model), retry_after_s=hint_s)
                                    if attempt >= (int(attempts) - 1) or sleep_budget_s <= 0.0:
                                        break
                                    try:
                                        sleep_s = min(float(wait_s), float(sleep_budget_s))
                                        time.sleep(float(max(0.0, sleep_s)))
                                        sleep_budget_s -= float(sleep_s)
                                    except Exception:
                                        pass
                                    continue
                                if _should_retry_remote_http_error(last_err) and attempt < (int(attempts) - 1):
                                    try:
                                        sleep_s = float(1.0 + (2.0 * float(attempt)))
                                        hint_s = _retry_delay_seconds_from_error(last_err)
                                        if hint_s is not None:
                                            sleep_s = max(float(sleep_s), float(hint_s))
                                        sleep_s = min(60.0, max(0.0, float(sleep_s)))
                                        time.sleep(float(sleep_s))
                                    except Exception:
                                        pass
                                    continue
                                raise
                        if last_err:
                            if rate_limited:
                                raise TemporaryUnavailableError("Gemini temporarily unavailable", provider="gemini", model=str(model))
                            raise ValueError(str(last_err))
                        if rate_limited and last_err is None:
                            raise TemporaryUnavailableError("Gemini temporarily unavailable", provider="gemini", model=str(model))

                    try:
                        if judge_traces and str(judge_traces[-1].get("decision")) == "retry":
                            summary = (
                                "\n\nAUTO_RETRY_EXHAUSTED: The analysis tool output still appeared incomplete or erroneous after "
                                + str(len(judge_traces))
                                + " attempt(s).\n"
                                + "Judge_reason: "
                                + str(judge_traces[-1].get("reason") or "")
                            )
                            out_text = str(out_text or "") + str(summary)
                    except Exception:
                        pass

                    if isinstance(diag, dict):
                        diag["judge"] = {"attempts": int(len(judge_traces)), "trace": list(judge_traces)}

                    _maybe_log_truncation_alarm(
                        out_text,
                        provider="gemini",
                        model=str(model),
                        path="/api/omics/analyze",
                        run_id=str(run_id or "") if run_id is not None else None,
                        player_id=str(player_id_req or "") if player_id_req is not None else None,
                    )

                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "run_id": str(run_id or ""),
                            "manifest": manifest,
                            "files": used_files,
                            "provider": "gemini",
                            "gemini": {
                                "base_url": str(base_url_g),
                                "model": str(model),
                                "max_tokens": int(max_tokens_g),
                            },
                            "output_text": str(out_text or ""),
                            "analysis_diagnostics": diag,
                            "response": resp_json,
                        },
                    )
                    return

                if prov in ("xai",):
                    if OpenAI is None:
                        raise ValueError("openai sdk not installed")
                    api_key = os.environ.get("XAI_API_KEY")
                    if not isinstance(api_key, str) or not api_key.strip():
                        raise ValueError("missing XAI_API_KEY")
                    base_url = str(os.environ.get("XAI_BASE_URL") or "").strip() or "https://api.x.ai/v1"

                    client = OpenAI(api_key=api_key, base_url=str(base_url))
                    model_call = _xai_canonical_model(str(model))
                    file_ids: list[str] = []
                    used_files: list[Dict[str, Any]] = []

                    def _xai_is_safety_check_bio(err: Exception) -> bool:
                        s = str(err or "")
                        if not s.strip():
                            return False
                        if "SAFETY_CHECK_TYPE_BIO" in s:
                            return True
                        if "usage guidelines" in s.lower() and "safety_check" in s.lower():
                            return True
                        if "Content violates usage guidelines" in s:
                            return True
                        return False

                    def _xai_neutralize_analyze_instructions(text: str, level: int) -> str:
                        t = str(text or "")
                        if not t:
                            return ""
                        try:
                            t = re.sub(r"\btherapy\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bcure\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bcures\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\btreating\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\btreatment\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\btreatments\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bdiagnose\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bdiagnosis\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bpatient\b", "sample", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bdrug\b", "perturbation", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bdrugs\b", "perturbations", t, flags=re.IGNORECASE)
                            t = re.sub(r"\s{2,}", " ", t)
                        except Exception:
                            pass
                        if int(level) >= 1:
                            t = re.sub(r"\bdisease\b", "condition", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bhealthy\b", "control", t, flags=re.IGNORECASE)
                            t = re.sub(r"\borganism\b", "sample", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bblood\b", "fluid", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bintervention\b", "variant", t, flags=re.IGNORECASE)
                        if int(level) >= 2:
                            t = re.sub(r"\bbiolog(y|ical)\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bclinical\b", "", t, flags=re.IGNORECASE)
                            t = re.sub(r"\bmedical\b", "", t, flags=re.IGNORECASE)
                        prefix = (
                            "Perform descriptive data analysis on the attached CSV files. "
                            "Do not provide advice or recommendations. "
                            "Use neutral language (e.g., dataset, group, condition, control). "
                            "When referencing attachments, use the FILE_XX labels from the manifest.\n\n"
                        )
                        return str(prefix) + str(t)

                    for ent in selected:
                        rid0 = str(ent.get("run_id") or "")
                        name0 = str(ent.get("name") or "")
                        if not rid0 or not name0:
                            continue
                        p = _OMICS.file_path(rid0, name0)

                        raw_bytes = b""
                        try:
                            raw_bytes = p.read_bytes()
                        except Exception:
                            raw_bytes = b""

                        pb = pooled_bytes_by_rid_name.get((str(rid0), str(name0)))
                        if pb is not None:
                            raw_bytes = bytes(pb)

                        label = str(ent.get("analyze_label") or "")
                        upload_name = str(p.name)
                        if label:
                            upload_name = str(label) + "__" + str(upload_name)
                        suf = str(p.suffix or "").lower()
                        if suf in (".csv", ".tsv", ".txt", ".json"):
                            try:
                                txt0 = raw_bytes.decode("utf-8", errors="replace")
                                txt1 = _mask_disease_term(txt0)
                                raw_bytes = txt1.encode("utf-8")
                                upload_name = _mask_disease_term(upload_name)
                            except Exception:
                                pass

                        bio = io.BytesIO(raw_bytes)
                        try:
                            bio.name = upload_name
                        except Exception:
                            pass
                        fo = client.files.create(file=bio, purpose="assistants")
                        fid = str(getattr(fo, "id", "") or "")
                        if fid:
                            file_ids.append(fid)
                        used_files.append(
                            {
                                "label": str(label),
                                "run_id": rid0,
                                "name": name0,
                                "file_id": str(ent.get("file_id") or ""),
                                "display_name": str(ent.get("display_name") or name0),
                            }
                        )

                    input_parts: list[Dict[str, Any]] = [
                        {"type": "input_text", "text": str(instructions_eff)}
                    ]
                    for fid in file_ids:
                        input_parts.append({"type": "input_file", "file_id": str(fid)})

                    tools = [{"type": "code_interpreter"}]

                    max_out_tokens = 120000
                    try:
                        if max_tokens_req is not None:
                            max_out_tokens = int(max_tokens_req)
                    except Exception:
                        max_out_tokens = 120000
                    max_out_tokens = max(256, min(120000, int(max_out_tokens)))

                    xai_policy_retries = 2
                    try:
                        xai_policy_retries = int(os.environ.get("DT_OMICS_ANALYZE_XAI_POLICY_RETRIES", str(xai_policy_retries)) or xai_policy_retries)
                    except Exception:
                        xai_policy_retries = 2
                    xai_policy_retries = max(0, min(5, int(xai_policy_retries)))

                    max_judge_attempts = 3
                    try:
                        max_judge_attempts = int(os.environ.get("DT_OMICS_ANALYZE_MAX_JUDGE_ATTEMPTS", str(max_judge_attempts)) or max_judge_attempts)
                    except Exception:
                        max_judge_attempts = 3
                    max_judge_attempts = max(1, min(3, int(max_judge_attempts)))

                    judge_traces: list[Dict[str, Any]] = []
                    resp = None
                    resp_dump: Any = None
                    out_text_raw = ""
                    out_text = ""
                    diag: Dict[str, Any] = {}
                    retry_guidance = ""

                    for judge_attempt in range(int(max_judge_attempts)):
                        prompt_attempt = _apply_omics_analyze_retry_guidance(str(instructions_eff), str(retry_guidance))
                        resp = None
                        last_err: Optional[Exception] = None
                        for pol_attempt in range(int(xai_policy_retries) + 1):
                            lvl = int(pol_attempt) + 1
                            instructions_xai = _xai_neutralize_analyze_instructions(str(prompt_attempt), level=lvl)
                            input_parts = [{"type": "input_text", "text": str(instructions_xai)}]
                            for fid in file_ids:
                                input_parts.append({"type": "input_file", "file_id": str(fid)})
                            try:
                                try:
                                    resp = client.responses.create(
                                        model=str(model_call),
                                        input=[{"role": "user", "content": input_parts}],
                                        tools=tools,
                                        include=["code_interpreter_call.outputs"],
                                        max_output_tokens=int(max_out_tokens),
                                    )
                                except Exception:
                                    resp = client.responses.create(
                                        model=str(model_call),
                                        input=[{"role": "user", "content": input_parts}],
                                        tools=tools,
                                        include=["code_interpreter_call.outputs"],
                                    )
                                last_err = None
                                break
                            except Exception as e:
                                last_err = e
                                if pol_attempt >= int(xai_policy_retries) or (not _xai_is_safety_check_bio(e)):
                                    raise
                                continue

                        if resp is None and last_err is not None:
                            raise last_err

                        out_text_raw = ""
                        try:
                            out_text_raw = str(getattr(resp, "output_text", "") or "")
                        except Exception:
                            out_text_raw = ""

                        resp_dump = None
                        try:
                            resp_dump = resp.model_dump()
                        except Exception:
                            resp_dump = None

                        out_text, diag = _omics_analyze_diagnostics(out_text_raw, resp_dump)

                        judge = _omics_analyze_judge(
                            player_instructions=str(instructions),
                            analyze_manifest_entries=list(analyze_manifest_entries),
                            output_text=str(out_text_raw),
                            analysis_diagnostics=dict(diag),
                            provider="xai",
                            model=str(model_call),
                            attempt=int(judge_attempt) + 1,
                            max_attempts=int(max_judge_attempts),
                        )
                        judge_traces.append({"attempt": int(judge_attempt) + 1, **dict(judge)})

                        if str(judge.get("decision")) == "retry":
                            retry_guidance = _merge_omics_analyze_retry_guidance(
                                str(retry_guidance),
                                str(judge.get("retry_instructions") or ""),
                            )

                        if str(judge.get("decision")) != "retry" or judge_attempt >= (int(max_judge_attempts) - 1):
                            break

                    _maybe_log_truncation_alarm(
                        out_text,
                        provider="xai",
                        model=str(model),
                        path="/api/omics/analyze",
                        run_id=str(run_id or "") if run_id is not None else None,
                        player_id=str(player_id_req or "") if player_id_req is not None else None,
                    )

                    try:
                        if judge_traces and str(judge_traces[-1].get("decision")) == "retry":
                            summary = (
                                "\n\nAUTO_RETRY_EXHAUSTED: The analysis tool output still appeared incomplete or erroneous after "
                                + str(len(judge_traces))
                                + " attempt(s).\n"
                                + "Judge_reason: "
                                + str(judge_traces[-1].get("reason") or "")
                            )
                            out_text = str(out_text or "") + str(summary)
                    except Exception:
                        pass

                    if isinstance(diag, dict):
                        diag["judge"] = {"attempts": int(len(judge_traces)), "trace": list(judge_traces)}

                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "run_id": str(run_id or ""),
                            "manifest": manifest,
                            "files": used_files,
                            "provider": "xai",
                            "xai": {
                                "base_url": str(base_url),
                                "model": str(model),
                                "file_ids": list(file_ids),
                            },
                            "output_text": str(out_text or ""),
                            "analysis_diagnostics": diag,
                            "response": resp_dump,
                        },
                    )
                    return

                if prov in ("anthropic",):
                    api_key_a = os.environ.get("ANTHROPIC_API_KEY")
                    if not isinstance(api_key_a, str) or not api_key_a.strip():
                        raise ValueError("missing ANTHROPIC_API_KEY")

                    max_tokens_a = 8192
                    try:
                        if max_tokens_req is not None:
                            max_tokens_a = int(max_tokens_req)
                    except Exception:
                        max_tokens_a = 8192
                    max_tokens_a = max(256, min(16384, int(max_tokens_a)))

                    used_files: list[Dict[str, Any]] = []
                    file_ids: list[str] = []
                    uploaded_bytes_total = 0

                    for ent in selected:
                        rid0 = str(ent.get("run_id") or "")
                        name0 = str(ent.get("name") or "")
                        if not rid0 or not name0:
                            continue
                        p = _OMICS.file_path(rid0, name0)

                        raw_bytes = b""
                        try:
                            raw_bytes = p.read_bytes()
                        except Exception:
                            raw_bytes = b""

                        pb = pooled_bytes_by_rid_name.get((str(rid0), str(name0)))
                        if pb is not None:
                            raw_bytes = bytes(pb)

                        label = str(ent.get("analyze_label") or "")
                        upload_name = str(p.name)
                        if label:
                            upload_name = str(label) + "__" + str(upload_name)
                        suf = str(p.suffix or "").lower()
                        if suf in (".csv", ".tsv", ".txt", ".json"):
                            try:
                                txt0 = raw_bytes.decode("utf-8", errors="replace")
                                txt1 = _mask_disease_term(txt0)
                                raw_bytes = txt1.encode("utf-8")
                                upload_name = _mask_disease_term(upload_name)
                            except Exception:
                                pass

                        try:
                            uploaded_bytes_total += int(len(raw_bytes or b""))
                        except Exception:
                            pass

                        up = _anthropic_upload_file(
                            api_key=str(api_key_a),
                            filename=str(upload_name),
                            file_bytes=(raw_bytes or b""),
                            timeout_s=120.0,
                        )
                        fid = str(up.get("id") or "")
                        if fid:
                            file_ids.append(fid)
                        used_files.append({
                            "label": str(label),
                            "run_id": rid0,
                            "name": name0,
                            "file_id": str(ent.get("file_id") or ""),
                            "display_name": str(ent.get("display_name") or name0),
                            "anthropic_file_id": fid,
                        })

                    if not file_ids:
                        raise ValueError("no files uploaded")

                    try:
                        need_tokens = int(_approx_token_count_text(str(instructions_eff)))
                        need_tokens += int(max(0, int(uploaded_bytes_total)) // 4)
                        need_tokens += 500
                        _anthropic_tpm_throttle(need_tokens=int(need_tokens))
                    except Exception:
                        pass

                    content_blocks0: list[Dict[str, Any]] = [{"type": "text", "text": str(instructions_eff)}]
                    for fid in list(file_ids):
                        if str(fid or "").strip():
                            content_blocks0.append({"type": "container_upload", "file_id": str(fid)})
                    messages0: list[Dict[str, Any]] = [{"role": "user", "content": content_blocks0}]

                    max_judge_attempts = 3
                    try:
                        max_judge_attempts = int(os.environ.get("DT_OMICS_ANALYZE_MAX_JUDGE_ATTEMPTS", str(max_judge_attempts)) or max_judge_attempts)
                    except Exception:
                        max_judge_attempts = 3
                    max_judge_attempts = max(1, min(3, int(max_judge_attempts)))

                    judge_traces: list[Dict[str, Any]] = []
                    resp_json: Dict[str, Any] = {}
                    out_text_raw = ""
                    out_text = ""
                    diag: Dict[str, Any] = {}
                    retry_guidance = ""

                    for judge_attempt in range(int(max_judge_attempts)):
                        prompt_attempt = _apply_omics_analyze_retry_guidance(str(instructions_eff), str(retry_guidance))
                        content_blocks = [{"type": "text", "text": str(prompt_attempt)}]
                        for fid in list(file_ids):
                            if str(fid or "").strip():
                                content_blocks.append({"type": "container_upload", "file_id": str(fid)})
                        messages_attempt: list[Dict[str, Any]] = [{"role": "user", "content": content_blocks}]
                        resp_json = _anthropic_messages_code_execution(
                            api_key=str(api_key_a),
                            model=str(model or "claude-sonnet-4-5"),
                            instructions=str(prompt_attempt),
                            file_ids=list(file_ids),
                            timeout_s=600.0,
                            max_tokens=int(max_tokens_a),
                            messages=list(messages_attempt),
                        )
                        out_text_raw = str(_anthropic_message_text(resp_json if isinstance(resp_json, dict) else {}) or "")

                        out_text, diag = _omics_analyze_diagnostics(out_text_raw, resp_json)

                        judge = _omics_analyze_judge(
                            player_instructions=str(instructions),
                            analyze_manifest_entries=list(analyze_manifest_entries),
                            output_text=str(out_text_raw),
                            analysis_diagnostics=dict(diag),
                            provider="anthropic",
                            model=str(model),
                            attempt=int(judge_attempt) + 1,
                            max_attempts=int(max_judge_attempts),
                        )
                        judge_traces.append({"attempt": int(judge_attempt) + 1, **dict(judge)})

                        if str(judge.get("decision")) == "retry":
                            retry_guidance = _merge_omics_analyze_retry_guidance(
                                str(retry_guidance),
                                str(judge.get("retry_instructions") or ""),
                            )

                        if str(judge.get("decision")) != "retry" or judge_attempt >= (int(max_judge_attempts) - 1):
                            break

                    try:
                        if judge_traces and str(judge_traces[-1].get("decision")) == "retry":
                            summary = (
                                "\n\nAUTO_RETRY_EXHAUSTED: The analysis tool output still appeared incomplete or erroneous after "
                                + str(len(judge_traces))
                                + " attempt(s).\n"
                                + "Judge_reason: "
                                + str(judge_traces[-1].get("reason") or "")
                            )
                            out_text = str(out_text or "") + str(summary)
                    except Exception:
                        pass

                    if isinstance(diag, dict):
                        diag["judge"] = {"attempts": int(len(judge_traces)), "trace": list(judge_traces)}

                    _maybe_log_truncation_alarm(
                        out_text,
                        provider="anthropic",
                        model=str(model),
                        path="/api/omics/analyze",
                        run_id=str(run_id or "") if run_id is not None else None,
                        player_id=str(player_id_req or "") if player_id_req is not None else None,
                    )

                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "run_id": str(run_id or ""),
                            "manifest": manifest,
                            "files": used_files,
                            "provider": "anthropic",
                            "anthropic": {
                                "model": str(model),
                                "file_ids": list(file_ids),
                                "max_tokens": int(max_tokens_a),
                                "auto_continue": False,
                                "max_continuations": 0,
                                "continuations": 0,
                            },
                            "output_text": str(out_text or ""),
                            "analysis_diagnostics": diag,
                            "response": resp_json,
                        },
                    )
                    return

                raise ValueError(f"unsupported provider: {prov}")

            if self.path == "/api/game/reset":
                body = self._read_json_body()
                self._send_json(200, {"ok": True, "game": _game_reset_player(body.get("player_id"))})
                return

            if self.path in ("/api/tests/cancer/estimate_cost", "/api/tests/hereditary_disease/estimate_cost", "/api/tests/aging/estimate_cost"):
                if self.path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif self.path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                body = self._read_json_body()
                player_id = body.get("player_id")
                model_key = body.get("model")
                exp = str(body.get("experiment") or "").strip().lower()
                ticks = body.get("ticks", 100)
                replicates = body.get("replicates", 1)
                interventions = _tests_validate_protein_interventions(body.get("interventions"))

                try:
                    ticks_i = max(0, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                iv_n = int(len(interventions)) if isinstance(interventions, list) else 0

                if exp in ("bulk", "bulk_omics"):
                    omics_set_req = body.get("omics_set")
                    omics_set_name, _ = _load_bulk_omics_set(str(omics_set_req or ""))
                    kind0 = _bulk_omics_kind_from_set_name(str(omics_set_name or ""))
                    unit = _tests_compute_unit_cost_cents(
                        challenge=challenge,
                        kind=str(kind0),
                        model_key=model_key,
                        ticks=int(ticks_i),
                        interventions_n=int(iv_n),
                    )
                    fixed = _tests_compute_fixed_cost_cents(
                        challenge=challenge,
                        kind=str(kind0),
                        model_key=model_key,
                        interventions_n=int(iv_n),
                    )
                    charge = _tests_make_charge(
                        kind=f"tests_{challenge}_{str(kind0)}",
                        samples=int(reps_i),
                        unit_cost_cents=int(unit),
                        fixed_cost_cents=int(fixed),
                        meta={"experiment": f"tests_{challenge}_bulk_omics_v1", "player_id": _sanitize_player_id(player_id)},
                    )
                    self._send_json(200, {"ok": True, "charge": charge})
                    return

                if exp in ("spatial", "spatial_tx", "spatial_omics"):
                    exp_label = f"tests_{challenge}_spatial_omics_v1" if exp == "spatial_omics" else f"tests_{challenge}_spatial_tx_v1"
                    charge_kind = f"tests_{challenge}_spatial_omics" if exp == "spatial_omics" else f"tests_{challenge}_spatial_transcriptomics"
                    unit = _tests_compute_unit_cost_cents(
                        challenge=challenge,
                        kind="spatial_transcriptomics",
                        model_key=model_key,
                        ticks=int(ticks_i),
                        interventions_n=int(iv_n),
                    )
                    fixed = _tests_compute_fixed_cost_cents(
                        challenge=challenge,
                        kind="spatial_transcriptomics",
                        model_key=model_key,
                        interventions_n=int(iv_n),
                    )
                    charge = _tests_make_charge(
                        kind=str(charge_kind),
                        samples=int(reps_i),
                        unit_cost_cents=int(unit),
                        fixed_cost_cents=int(fixed),
                        meta={"experiment": str(exp_label), "player_id": _sanitize_player_id(player_id)},
                    )
                    self._send_json(200, {"ok": True, "charge": charge})
                    return

                if exp in ("characterization", "char"):
                    unit = _tests_compute_unit_cost_cents(
                        challenge=challenge,
                        kind="characterization",
                        model_key=model_key,
                        ticks=int(ticks_i),
                        interventions_n=int(iv_n),
                    )
                    fixed = _tests_compute_fixed_cost_cents(
                        challenge=challenge,
                        kind="characterization",
                        model_key=model_key,
                        interventions_n=int(iv_n),
                    )
                    charge = _tests_make_charge(
                        kind=f"tests_{challenge}_characterization",
                        samples=int(reps_i),
                        unit_cost_cents=int(unit),
                        fixed_cost_cents=int(fixed),
                        meta={"experiment": f"tests_{challenge}_characterization_v1", "player_id": _sanitize_player_id(player_id)},
                    )
                    self._send_json(200, {"ok": True, "charge": charge})
                    return

                if exp in ("protein_screen", "screen"):
                    if not _tests_is_in_vitro_model(model_key):
                        raise ValueError("protein_screen is only allowed for in vitro cell_culture_* models")
                    payload0 = _tests_load_model_payload_for_challenge(challenge, model_key)
                    prot_layers = _protein_layer_names_from_payload(payload0)
                    samples_run = int(int(reps_i) * int(len(prot_layers) + 2))
                    unit = _tests_compute_unit_cost_cents(
                        challenge=challenge,
                        kind="protein_screen",
                        model_key=model_key,
                        ticks=int(ticks_i),
                        interventions_n=int(iv_n + 1),
                    )
                    fixed = _tests_compute_fixed_cost_cents(
                        challenge=challenge,
                        kind="protein_screen",
                        model_key=model_key,
                        interventions_n=int(iv_n + 1),
                    )
                    charge = _tests_make_charge(
                        kind=f"tests_{challenge}_protein_screen",
                        samples=int(samples_run),
                        unit_cost_cents=int(unit),
                        fixed_cost_cents=int(fixed),
                        meta={"experiment": f"tests_{challenge}_protein_screen_v1", "player_id": _sanitize_player_id(player_id)},
                    )
                    self._send_json(200, {"ok": True, "charge": charge})
                    return

                if exp in ("claim_cure", "cure"):
                    ticks_i = 1500 if str(challenge or "").strip().lower() == "aging" else 400
                    disease_key = _tests_claim_cure_disease_model_key_for_challenge(challenge)
                    if str(challenge or "").strip().lower() == "aging":
                        prev_min_reps = _game_get_player_int(player_id, "aging_claim_cure_min_reps", default=0)
                        reps_min_i = max(10, int(prev_min_reps))
                        reps_i = max(int(reps_i), int(reps_min_i))
                    unit = _tests_compute_unit_cost_cents(
                        challenge=challenge,
                        kind="claim_cure",
                        model_key=str(disease_key),
                        ticks=int(ticks_i),
                        interventions_n=int(iv_n),
                    )
                    fixed = _tests_compute_fixed_cost_cents(
                        challenge=challenge,
                        kind="claim_cure",
                        model_key=str(disease_key),
                        interventions_n=int(iv_n),
                    )
                    groups = 2
                    if str(challenge or "").strip().lower() in ("cancer", "hereditary_disease") and int(iv_n) > 0:
                        groups = 3
                    charge = _tests_make_charge(
                        kind=f"tests_{challenge}_claim_cure",
                        samples=int(int(groups) * int(reps_i)),
                        unit_cost_cents=int(unit),
                        fixed_cost_cents=int(fixed),
                        meta={"experiment": f"tests_{challenge}_claim_cure_v1", "player_id": _sanitize_player_id(player_id)},
                    )
                    self._send_json(200, {"ok": True, "charge": charge})
                    return

                raise ValueError("unknown experiment")

            if self.path in ("/api/tests/cancer/bulk_omics", "/api/tests/hereditary_disease/bulk_omics", "/api/tests/aging/bulk_omics"):
                if self.path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif self.path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = _tests_validate_protein_interventions(body.get("interventions"))
                model_key = body.get("model")

                ticks = body.get("ticks", 100)
                replicates = body.get("replicates", 1)
                seed = body.get("seed", 1)
                omics_set_req = body.get("omics_set")
                omics_set_name, features = _load_bulk_omics_set(str(omics_set_req or ""))

                if not features:
                    raise ValueError("no features selected")

                kind0 = _bulk_omics_kind_from_set_name(str(omics_set_name or ""))
                masked_features = _bulk_omics_mask_feature_headers(features, kind=str(kind0))

                z_target_arr = None
                z_target_in = body.get("z_target")
                if isinstance(z_target_in, list):
                    if len(z_target_in) != int(len(features)):
                        raise ValueError("z_target must have length == #features")
                    tmp: list[float] = []
                    for v in z_target_in:
                        try:
                            f = float(v)
                        except Exception:
                            f = float("nan")
                        tmp.append(f)
                    z_target_arr = np.asarray(tmp, dtype=np.float64)

                try:
                    ticks_i = max(0, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e

                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                payload0 = _tests_load_model_payload_for_challenge(challenge, model_key)
                payload = _deepcopy_payload(payload0)
                interventions_real = _tests_translate_interventions_masked_to_real(interventions, model_key=model_key, challenge=challenge)
                if interventions_real:
                    payload["_tick_interventions"] = list(interventions_real)

                syn_rng = np.random.default_rng(3)
                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]
                model_label0 = str(model_key or "")
                if model_label0.startswith("cell_culture_"):
                    model_label0 = model_label0[len("cell_culture_"):]
                if model_label0.startswith("tissue_"):
                    model_label0 = model_label0[len("tissue_"):]
                model_label = _omics_safe_label(model_label0, default="sample")

                replicate_deaths: list[Dict[str, Any]] = []

                assay = str(kind0 or "")
                if assay == "bulk_rnaseq":
                    assay = "Bulk transcriptomics"
                elif assay == "bulk_proteomics":
                    assay = "Bulk proteomics"
                elif assay == "bulk_metabolomics":
                    assay = "Bulk metabolomics"

                meta_header = [
                    "sample_id",
                    "assay",
                    "model",
                    "replicate",
                    "sample_age",
                ]
                meta_rows: list[list[Any]] = []
                mat_header = ["sample_id", *masked_features]

                run_ids: list[str] = []
                run_T: list[list[float]] = []
                out_runs: list[Dict[str, Any]] = []

                results_by_ri: list[Optional[tuple[int, int, Optional[Dict[str, Any]], Optional[list[float]], int, int]]] = [
                    None for _ in range(int(reps_i))
                ]
                if int(reps_i) <= 1:
                    for ri in range(int(reps_i)):
                        seed0 = int(seed_i) + (int(ri) * 97)
                        pf = _preflight_death_before_ticks(payload, ticks=int(ticks_i), seed0=int(seed0))
                        if isinstance(pf, dict):
                            results_by_ri[int(ri)] = (int(ri), int(seed0), dict(pf), None, 0, 0)
                            continue
                        p = _run_payload_ticks(payload, ticks=int(ticks_i), seed0=int(seed0))
                        H = int(p.get("H") or 0)
                        W = int(p.get("W") or 0)
                        if H <= 0 or W <= 0:
                            raise ValueError("payload invalid H/W")
                        expected_len = int(H * W)
                        layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
                        if not layers:
                            raise ValueError("payload has no float32 layers")

                        vv: list[float] = []
                        for ln in features:
                            arr = layers.get(ln)
                            if arr is None:
                                vv.append(0.0)
                                continue
                            try:
                                s = float(np.asarray(arr, dtype=np.float64).reshape(-1).sum())
                            except Exception:
                                s = 0.0
                            if not np.isfinite(s) or s < 0.0:
                                s = 0.0
                            vv.append(float(s))

                        results_by_ri[int(ri)] = (int(ri), int(seed0), None, vv, int(H), int(W))
                else:
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(reps_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=2.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(reps_i),
                            mp_context=ctx,
                            initializer=_bulk_omics_worker_init,
                            initargs=(payload, list(features), int(ticks_i), int(seed_i)),
                        ) as ex:
                            futs: list[concurrent.futures.Future] = []
                            for ri in range(int(reps_i)):
                                futs.append(ex.submit(_bulk_omics_worker_eval, int(ri)))
                            for fut in concurrent.futures.as_completed(futs):
                                ri0, seed0, pf, vv, H, W = fut.result()
                                if 0 <= int(ri0) < int(len(results_by_ri)):
                                    results_by_ri[int(ri0)] = (int(ri0), int(seed0), pf, vv, int(H), int(W))
                    finally:
                        _cm.__exit__(None, None, None)

                for ri in range(int(reps_i)):
                    ent = results_by_ri[int(ri)]
                    if ent is None:
                        continue
                    ri0, seed0, pf, vv, H, W = ent
                    if isinstance(pf, dict):
                        replicate_deaths.append(
                            {
                                "model": str(model_key or ""),
                                "replicate": int(ri0),
                                "seed": int(seed0),
                                "requested_ticks": int(ticks_i),
                                "death_tick": int(pf.get("death_tick") or 0),
                                "death_measurement": str(pf.get("death_measurement") or ""),
                                "death_names": pf.get("death_names") if isinstance(pf.get("death_names"), list) else [],
                            }
                        )
                        continue

                    if not isinstance(vv, list):
                        continue

                    sample_id = f"{model_label}_r{int(ri0)}_{run_tag}"
                    meta_rows.append(
                        [
                            sample_id,
                            str(assay),
                            str(model_key or ""),
                            int(ri0),
                            int(ticks_i),
                        ]
                    )
                    run_ids.append(sample_id)
                    run_T.append([float(x) for x in vv])

                    out_runs.append(
                        {
                            "model": str(model_key or ""),
                            "replicate": int(ri0),
                            "seed": int(seed0),
                            "ticks": int(ticks_i),
                            "age_days": int(ticks_i),
                            "H": int(H),
                            "W": int(W),
                        }
                    )

                if not run_ids:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": f"all replicates died before requested ticks={int(ticks_i)}",
                            "error_kind": "all_replicates_died",
                            "experiment": "tests_cancer_bulk_omics_v1",
                            "details": {
                                "model": str(model_key or ""),
                                "requested_ticks": int(ticks_i),
                                "replicates_requested": int(reps_i),
                                "replicate_deaths": list(replicate_deaths),
                            },
                        },
                    )
                    return

                noisy_mat_rows: list[list[Any]] = []
                if run_ids and run_T:
                    T_arr = np.asarray(run_T, dtype=np.float64)
                    Y = _bulk_synthetic_v1_noisy_counts(T_arr, rng=syn_rng, z_target=z_target_arr)
                    for ii, sid in enumerate(run_ids):
                        noisy_mat_rows.append([sid, *[int(x) for x in np.asarray(Y[ii], dtype=np.int64).tolist()]])

                matrix_noisy_csv = _csv_from_rows(mat_header, noisy_mat_rows)
                metadata_csv = _csv_from_rows(meta_header, meta_rows)

                # Persist files to disk (wide format: one CSV per replicate sample).
                files_text: Dict[str, Any] = {}
                files_text[f"metadata_{run_tag}.csv"] = str(metadata_csv)
                for row in noisy_mat_rows:
                    if not isinstance(row, list) or not row:
                        continue
                    sid = str(row[0] or "")
                    safe_sid = _omics_safe_label(sid, default="sample")
                    files_text[f"samples/{safe_sid}.csv"] = _csv_from_rows(mat_header, [row])

                manifest = {
                    "experiment": f"tests_{challenge}_bulk_omics_v1",
                    "kind": str(kind0),
                    "player_id": _sanitize_player_id(player_id),
                    "model": str(model_key or ""),
                    "ticks": int(ticks_i),
                    "age_days": int(ticks_i),
                    "replicates": int(reps_i),
                    "replicates_completed": int(len(out_runs)),
                    "replicate_deaths": list(replicate_deaths),
                    "omics_set": str(omics_set_name or ""),
                    "features": list(masked_features),
                    "runs": out_runs,
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                iv_n = int(len(interventions)) if isinstance(interventions, list) else 0
                unit = _tests_compute_unit_cost_cents(
                    challenge=challenge,
                    kind=str(kind0),
                    model_key=model_key,
                    ticks=int(ticks_i),
                    interventions_n=int(iv_n),
                )
                fixed = _tests_compute_fixed_cost_cents(
                    challenge=challenge,
                    kind=str(kind0),
                    model_key=model_key,
                    interventions_n=int(iv_n),
                )
                charge = _tests_make_charge(
                    kind=f"tests_{challenge}_{str(kind0)}",
                    samples=int(len(out_runs)),
                    unit_cost_cents=int(unit),
                    fixed_cost_cents=int(fixed),
                    meta={
                        "experiment": f"tests_{challenge}_bulk_omics_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "omics_set": str(omics_set_name or ""),
                        "model": str(model_key or ""),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": f"tests_{challenge}_bulk_omics_v1",
                        "model": str(model_key or ""),
                        "ticks": int(ticks_i),
                        "age_days": int(ticks_i),
                        "replicates": int(reps_i),
                        "replicates_completed": int(len(out_runs)),
                        "replicate_deaths": list(replicate_deaths),
                        "omics_set": str(omics_set_name or ""),
                        "genes": masked_features,
                        "run_id": str(omics_saved.get("run_id") or ""),
                        "files": omics_saved.get("files"),
                        "omics_inventory": {
                            "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                            "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                        },
                        "noise": {
                            "model": "bulk_synthetic_v1",
                            "sigma_sample": 0.35,
                            "theta": 50.0,
                            "ambient_frac": 0.001,
                            "ambient_sigma_sample": 0.25,
                            "rng_seed": 3,
                        },
                        "matrix_noisy_csv": matrix_noisy_csv,
                        "metadata_csv": metadata_csv,
                        "runs": out_runs,
                        "game": game,
                    },
                )
                return

            if self.path in (
                "/api/tests/cancer/spatial_tx",
                "/api/tests/hereditary_disease/spatial_tx",
                "/api/tests/aging/spatial_tx",
                "/api/tests/cancer/spatial_omics",
                "/api/tests/hereditary_disease/spatial_omics",
                "/api/tests/aging/spatial_omics",
            ):
                if self.path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif self.path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = _tests_validate_protein_interventions(body.get("interventions"))
                model_key = body.get("model")

                ticks = body.get("ticks", 100)
                replicates = body.get("replicates", 1)
                seed = body.get("seed", 1)
                gene_set_req = body.get("gene_set")
                gene_set_name, genes = _load_stx_gene_set(str(gene_set_req or ""))

                is_spatial_omics = bool(self.path.endswith("/spatial_omics"))
                exp_label = f"tests_{challenge}_spatial_omics_v1" if is_spatial_omics else f"tests_{challenge}_spatial_tx_v1"
                gene_set_out = str(gene_set_name or "")
                if is_spatial_omics:
                    gene_set_type = _normalize_spatial_omics_type(gene_set_req)
                    if not gene_set_type:
                        gene_set_type = _normalize_spatial_omics_type(gene_set_name)
                    gene_set_out = str(gene_set_type or gene_set_name)

                payload0 = _tests_load_model_payload_for_challenge(challenge, model_key)
                if not genes:
                    genes = _default_stx_gene_list(payload0, max_genes=8)
                if not genes:
                    raise ValueError("no genes selected")

                z_target_arr = None
                z_target_in = body.get("z_target")
                if isinstance(z_target_in, list):
                    if len(z_target_in) != int(len(genes)):
                        raise ValueError("z_target must have length == #genes")
                    tmp: list[float] = []
                    for v in z_target_in:
                        try:
                            f = float(v)
                        except Exception:
                            f = float("nan")
                        tmp.append(f)
                    z_target_arr = np.asarray(tmp, dtype=np.float64)

                try:
                    ticks_i = max(0, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                payload = _deepcopy_payload(payload0)
                interventions_real = _tests_translate_interventions_masked_to_real(interventions, model_key=model_key, challenge=challenge)
                if interventions_real:
                    payload["_tick_interventions"] = list(interventions_real)

                replicate_deaths: list[Dict[str, Any]] = []

                syn_rng = np.random.default_rng(3)

                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]
                model_label0 = str(model_key or "")
                if model_label0.startswith("cell_culture_"):
                    model_label0 = model_label0[len("cell_culture_"):]
                if model_label0.startswith("tissue_"):
                    model_label0 = model_label0[len("tissue_"):]
                model_label = _omics_safe_label(model_label0, default="sample")

                meta_header = [
                    "cell_id",
                    "assay",
                    "model",
                    "replicate",
                    "seed",
                    "sample_taken_at_day",
                    "x",
                    "y",
                    "grid_index",
                ]
                meta_rows: list[list[Any]] = []
                stx_kind = _stx_kind_from_gene_set_and_genes(gene_set_name, genes)
                assay = "Spatial transcriptomics"
                if str(stx_kind) == "bulk_proteomics":
                    assay = "Spatial proteomics"
                if is_spatial_omics:
                    assay = "Spatial protein" if str(stx_kind) == "bulk_proteomics" else "Spatial RNA"
                genes_masked = _bulk_omics_mask_feature_headers(genes, kind=str(stx_kind))
                mat_header = ["cell_id", *genes_masked]
                noisy_mat_rows: list[list[Any]] = []

                out_runs: list[Dict[str, Any]] = []
                files_text: Dict[str, Any] = {}

                results_by_ri: list[Optional[tuple[int, int, Optional[Dict[str, Any]], int, int, list[Dict[str, Any]]]]] = [
                    None for _ in range(int(reps_i))
                ]
                if int(reps_i) <= 1:
                    for ri in range(int(reps_i)):
                        seed0 = int(seed_i) + (int(ri) * 97)
                        pf = _preflight_death_before_ticks(payload, ticks=int(ticks_i), seed0=int(seed0))
                        if isinstance(pf, dict):
                            results_by_ri[int(ri)] = (int(ri), int(seed0), dict(pf), 0, 0, [])
                            continue
                        p = _run_payload_ticks(payload, ticks=int(ticks_i), seed0=int(seed0))
                        tx = _spatial_tx_rows(
                            p,
                            genes,
                            cell_layer="",
                            min_cell_value=0.0,
                            stride=1,
                            max_spots=None,
                            seed=int(seed0),
                        )
                        H = int(tx.get("H") or 0)
                        W = int(tx.get("W") or 0)
                        rows = tx.get("rows")
                        if not isinstance(rows, list):
                            rows = []
                        rows_out: list[Dict[str, Any]] = [r for r in rows if isinstance(r, dict)]
                        results_by_ri[int(ri)] = (int(ri), int(seed0), None, int(H), int(W), rows_out)
                else:
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(reps_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=3.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(reps_i),
                            mp_context=ctx,
                            initializer=_stx_tests_worker_init,
                            initargs=(payload, list(genes), int(ticks_i), int(seed_i)),
                        ) as ex:
                            futs: list[concurrent.futures.Future] = []
                            for ri in range(int(reps_i)):
                                futs.append(ex.submit(_stx_tests_worker_eval, int(ri)))
                            for fut in concurrent.futures.as_completed(futs):
                                ri0, seed0, pf, H, W, rows = fut.result()
                                if 0 <= int(ri0) < int(len(results_by_ri)):
                                    rows_out: list[Dict[str, Any]] = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
                                    results_by_ri[int(ri0)] = (int(ri0), int(seed0), pf, int(H), int(W), rows_out)
                    finally:
                        _cm.__exit__(None, None, None)

                for ri in range(int(reps_i)):
                    ent = results_by_ri[int(ri)]
                    if ent is None:
                        continue
                    ri0, seed0, pf, H, W, rows = ent
                    if isinstance(pf, dict):
                        replicate_deaths.append(
                            {
                                "model": str(model_key or ""),
                                "replicate": int(ri0),
                                "seed": int(seed0),
                                "requested_ticks": int(ticks_i),
                                "death_tick": int(pf.get("death_tick") or 0),
                                "death_measurement": str(pf.get("death_measurement") or ""),
                                "death_names": pf.get("death_names") if isinstance(pf.get("death_names"), list) else [],
                            }
                        )
                        continue

                    run_cell_ids: list[str] = []
                    run_x: list[int] = []
                    run_y: list[int] = []
                    run_T: list[list[float]] = []
                    rep_meta_rows: list[list[Any]] = []

                    for si, row in enumerate(rows):
                        if not isinstance(row, dict):
                            continue
                        x = row.get("x")
                        y = row.get("y")
                        try:
                            xi = int(x)
                        except Exception:
                            continue
                        try:
                            yi = int(y)
                        except Exception:
                            continue
                        grid_index = (yi * int(W) + xi) if (int(W) > 0) else int(yi)
                        cell_id = f"{model_label}_r{int(ri0)}_{run_tag}_s{int(seed0)}_{int(si)}"

                        meta_rows.append(
                            [
                                cell_id,
                                str(assay),
                                str(model_key or ""),
                                int(ri0),
                                int(seed0),
                                int(ticks_i),
                                int(xi),
                                int(yi),
                                int(grid_index),
                            ]
                        )
                        rep_meta_rows.append(
                            [
                                cell_id,
                                str(assay),
                                str(model_key or ""),
                                int(ri0),
                                int(seed0),
                                int(ticks_i),
                                int(xi),
                                int(yi),
                                int(grid_index),
                            ]
                        )

                        vv: list[float] = []
                        for g in genes:
                            try:
                                f0 = float(row.get(g) or 0.0)
                            except Exception:
                                f0 = 0.0
                            if not np.isfinite(f0) or f0 < 0.0:
                                f0 = 0.0
                            vv.append(float(f0))

                        run_cell_ids.append(cell_id)
                        run_x.append(int(xi))
                        run_y.append(int(yi))
                        run_T.append(vv)

                    if run_cell_ids and run_T:
                        T_arr = np.asarray(run_T, dtype=np.float64)
                        x_arr = np.asarray(run_x, dtype=np.int64)
                        y_arr = np.asarray(run_y, dtype=np.int64)
                        Y = _stx_synthetic_v3_noisy_counts(
                            T_arr,
                            x_arr,
                            y_arr,
                            H=int(H),
                            W=int(W),
                            rng=syn_rng,
                            z_target=z_target_arr,
                        )
                        for ii, cid in enumerate(run_cell_ids):
                            noisy_mat_rows.append([cid, *[int(x) for x in np.asarray(Y[ii], dtype=np.int64).tolist()]])

                        rep_prefix = f"replicates/{model_label}_r{int(ri0)}_{run_tag}"
                        rep_rows = [
                            [run_cell_ids[ii], *[int(x) for x in np.asarray(Y[ii], dtype=np.int64).tolist()]]
                            for ii in range(len(run_cell_ids))
                        ]
                        files_text[f"{rep_prefix}_matrix.csv"] = _csv_from_rows(mat_header, rep_rows)
                        files_text[f"{rep_prefix}_cell_metadata.csv"] = _csv_from_rows(meta_header, rep_meta_rows)

                    out_runs.append(
                        {
                            "model": str(model_key or ""),
                            "replicate": int(ri0),
                            "seed": int(seed0),
                            "ticks": int(ticks_i),
                            "cells": int(len(rows)),
                            "H": int(H),
                            "W": int(W),
                        }
                    )

                if not out_runs:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": f"all replicates died before requested ticks={int(ticks_i)}",
                            "error_kind": "all_replicates_died",
                            "experiment": str(exp_label),
                            "details": {
                                "model": str(model_key or ""),
                                "requested_ticks": int(ticks_i),
                                "replicates_requested": int(reps_i),
                                "replicate_deaths": list(replicate_deaths),
                            },
                        },
                    )
                    return

                matrix_noisy_csv = _csv_from_rows(mat_header, noisy_mat_rows)
                metadata_csv = _csv_from_rows(meta_header, meta_rows)

                files_text[f"metadata_{run_tag}.csv"] = str(metadata_csv)
                manifest = {
                    "experiment": str(exp_label),
                    "kind": str(stx_kind),
                    "player_id": _sanitize_player_id(player_id),
                    "model": str(model_key or ""),
                    "ticks": int(ticks_i),
                    "replicates": int(reps_i),
                    "replicates_completed": int(len(out_runs)),
                    "replicate_deaths": list(replicate_deaths),
                    "gene_set": str(gene_set_out or ""),
                    "genes": list(genes_masked),
                    "runs": out_runs,
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                iv_n = int(len(interventions)) if isinstance(interventions, list) else 0
                unit = _tests_compute_unit_cost_cents(
                    challenge=challenge,
                    kind="spatial_transcriptomics",
                    model_key=model_key,
                    ticks=int(ticks_i),
                    interventions_n=int(iv_n),
                )
                fixed = _tests_compute_fixed_cost_cents(
                    challenge=challenge,
                    kind="spatial_transcriptomics",
                    model_key=model_key,
                    interventions_n=int(iv_n),
                )
                charge_kind = f"tests_{challenge}_spatial_omics" if is_spatial_omics else f"tests_{challenge}_spatial_transcriptomics"
                charge = _tests_make_charge(
                    kind=str(charge_kind),
                    samples=int(len(out_runs)),
                    unit_cost_cents=int(unit),
                    fixed_cost_cents=int(fixed),
                    meta={
                        "experiment": str(exp_label),
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "gene_set": str(gene_set_out or ""),
                        "model": str(model_key or ""),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": str(exp_label),
                        "model": str(model_key or ""),
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "replicates_completed": int(len(out_runs)),
                        "replicate_deaths": list(replicate_deaths),
                        "gene_set": str(gene_set_out or ""),
                        "genes": genes_masked,
                        "run_id": str(omics_saved.get("run_id") or ""),
                        "files": omics_saved.get("files"),
                        "omics_inventory": {
                            "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                            "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                        },
                        "noise": {
                            "model": "synthetic_v3",
                            "sigma_cell": 0.35,
                            "theta": 50.0,
                            "eps": 0.08,
                            "target_median_umi": 2000.0,
                            "ambient_total_umi": 0.05,
                            "ambient_sigma_cell": 0.25,
                            "rng_seed": 3,
                        },
                        "matrix_noisy_csv": matrix_noisy_csv,
                        "metadata_csv": metadata_csv,
                        "runs": out_runs,
                        "game": game,
                    },
                )
                return

            if self.path in ("/api/tests/cancer/characterization", "/api/tests/hereditary_disease/characterization", "/api/tests/aging/characterization"):
                if self.path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif self.path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = _tests_validate_protein_interventions(body.get("interventions"))
                model_key = body.get("model")

                ticks = body.get("ticks", 100)
                replicates = body.get("replicates", 1)
                seed = body.get("seed", 1)
                include_replicates_raw = body.get("include_replicates", True)
                include_replicates = bool(include_replicates_raw)

                try:
                    ticks_i = max(0, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                payload0 = _tests_load_model_payload_for_challenge(challenge, model_key)
                payload = _deepcopy_payload(payload0)
                interventions_real = _tests_translate_interventions_masked_to_real(interventions, model_key=model_key, challenge=challenge)
                if interventions_real:
                    payload["_tick_interventions"] = list(interventions_real)

                names = _measurement_names_from_payload(payload)
                if not names:
                    raise ValueError("no measurements configured (missing measurements_config)")

                if _tests_is_in_vitro_model(model_key):
                    # In vitro characterization always returns per-replicate files.
                    include_replicates = True

                    rep_series: list[Dict[str, list[float]]] = []
                    rep_seeds: list[int] = []
                    rep_series_out: list[Optional[Dict[str, list[float]]]] = [None for _ in range(int(reps_i))]
                    rep_seeds_out: list[Optional[int]] = [None for _ in range(int(reps_i))]
                    if int(reps_i) <= 1:
                        for ri in range(int(reps_i)):
                            seed0 = int(seed_i) + (int(ri) * 97)
                            s0, _m0 = _run_cell_culture_measurement_series_and_metrics(
                                payload,
                                ticks=int(ticks_i),
                                seed0=int(seed0),
                                names=names,
                            )
                            rep_series_out[int(ri)] = s0
                            rep_seeds_out[int(ri)] = int(seed0)
                    else:
                        ctx = mp.get_context("spawn")
                        cpu_req = max(1, min(int(reps_i), int(_RESOURCE_SCHED.cpu_total)))
                        _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=2.0)
                        _cm.__enter__()
                        try:
                            with concurrent.futures.ProcessPoolExecutor(
                                max_workers=int(reps_i),
                                mp_context=ctx,
                                initializer=_char_worker_init,
                                initargs=(payload, int(ticks_i), int(seed_i), list(names), [], "vitro"),
                            ) as ex:
                                futs: list[concurrent.futures.Future] = []
                                for ri in range(int(reps_i)):
                                    futs.append(ex.submit(_char_worker_eval, int(ri)))
                                for fut in concurrent.futures.as_completed(futs):
                                    ri0, seed0, s0, _dt0, _dm0 = fut.result()
                                    if 0 <= int(ri0) < int(len(rep_series_out)):
                                        rep_series_out[int(ri0)] = s0
                                        rep_seeds_out[int(ri0)] = int(seed0)
                        finally:
                            _cm.__exit__(None, None, None)

                    for ri in range(int(reps_i)):
                        s0 = rep_series_out[int(ri)]
                        sd = rep_seeds_out[int(ri)]
                        if not isinstance(sd, int):
                            sd = int(seed_i) + (int(ri) * 97)
                        if not isinstance(s0, dict):
                            s0 = {}
                        rep_series.append(s0)
                        rep_seeds.append(int(sd))

                    # Persist files to disk.
                    files_text: Dict[str, Any] = {}

                    run_id0 = uuid.uuid4().hex
                    run_tag = str(run_id0)[:12]
                    model_label0 = str(model_key or "")
                    if model_label0.startswith("cell_culture_"):
                        model_label0 = model_label0[len("cell_culture_"):]
                    if model_label0.startswith("tissue_"):
                        model_label0 = model_label0[len("tissue_"):]
                    model_label = _omics_safe_label(model_label0, default="sample")

                    meta_header = [
                        "run_id",
                        "sample_id",
                        "assay",
                        "model",
                        "replicate",
                        "seed",
                        "study_ran_for_days",
                        "timecourse_filename",
                        "timecourse_relpath",
                        "timecourse_file_id",
                        "timecourse_url",
                    ]
                    meta_rows: list[list[Any]] = []
                    tc_names: list[str] = []
                    for ri, sd in enumerate(rep_seeds):
                        sample_id = f"{model_label}_r{int(ri)}_{run_tag}"
                        tc_name = f"replicates/{sample_id}_timecourse.csv"
                        tc_names.append(tc_name)
                        tc_fid = _OMICS._file_id(run_id0, tc_name)
                        tc_url = f"/api/omics/file?run_id={run_id0}&name={tc_name}"
                        meta_rows.append(
                            [
                                str(run_id0),
                                str(sample_id),
                                "Characterization experiment",
                                str(model_key or ""),
                                int(ri),
                                int(sd),
                                int(ticks_i),
                                str(Path(tc_name).name),
                                str(tc_name),
                                str(tc_fid),
                                str(tc_url),
                            ]
                        )

                    metadata_name = f"metadata_{run_tag}.csv"
                    files_text[metadata_name] = _csv_from_rows(meta_header, meta_rows)

                    tc_header = ["day", *[str(x) for x in names]]
                    for ri, s0 in enumerate(rep_series):
                        tr: list[list[Any]] = []
                        for ti in range(int(ticks_i)):
                            row: list[Any] = [int(ti)]
                            for nm in names:
                                vals = s0.get(nm) if isinstance(s0.get(nm), list) else []
                                row.append(vals[int(ti)] if int(ti) < int(len(vals)) else "")
                            tr.append(row)
                        tc_name = tc_names[int(ri)] if int(ri) < int(len(tc_names)) else f"replicates/{model_label}_r{int(ri)}_{run_tag}_timecourse.csv"
                        files_text[str(tc_name)] = _csv_from_rows(tc_header, tr)

                    manifest = {
                        "experiment": "tests_cancer_characterization_v1",
                        "kind": "characterization",
                        "player_id": _sanitize_player_id(player_id),
                        "model": str(model_key or ""),
                        "ticks": int(ticks_i),
                        "age_days": int(ticks_i),
                        "replicates": int(reps_i),
                        "seed": int(seed_i),
                        "interventions": interventions_real,
                        "measurements": list(names),
                    }
                    omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                    iv_n = int(len(interventions)) if isinstance(interventions, list) else 0
                    unit = _tests_compute_unit_cost_cents(
                        challenge="cancer",
                        kind="characterization",
                        model_key=model_key,
                        ticks=int(ticks_i),
                        interventions_n=int(iv_n),
                    )
                    fixed = _tests_compute_fixed_cost_cents(
                        challenge="cancer",
                        kind="characterization",
                        model_key=model_key,
                        interventions_n=int(iv_n),
                    )
                    charge = _tests_make_charge(
                        kind="tests_cancer_characterization",
                        samples=int(reps_i),
                        unit_cost_cents=int(unit),
                        fixed_cost_cents=int(fixed),
                        meta={
                            "experiment": "tests_cancer_characterization_v1",
                            "ticks": int(ticks_i),
                            "replicates": int(reps_i),
                            "model": str(model_key or ""),
                        },
                    )
                    game = _game_apply_charge(player_id, charge)

                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "experiment": "tests_cancer_characterization_v1",
                            "model": str(model_key or ""),
                            "ticks": int(ticks_i),
                            "replicates": int(reps_i),
                            "measurements": names,
                            "run_id": str(omics_saved.get("run_id") or ""),
                            "files": omics_saved.get("files"),
                            "omics_inventory": {
                                "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                                "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                            },
                            "game": game,
                        },
                    )
                    return

                death_names = _death_measurement_names_from_payload(payload)

                # In vivo characterization also persists per-replicate timecourses and replicate metadata only.
                include_replicates = True

                rep_series: list[Dict[str, list[float]]] = []
                rep_seeds: list[int] = []
                death_days: list[int] = []
                death_meas: list[str] = []

                rep_series_out: list[Optional[Dict[str, list[float]]]] = [None for _ in range(int(reps_i))]
                rep_seeds_out: list[Optional[int]] = [None for _ in range(int(reps_i))]
                death_days_out: list[Optional[int]] = [None for _ in range(int(reps_i))]
                death_meas_out: list[Optional[str]] = [None for _ in range(int(reps_i))]
                if int(reps_i) <= 1:
                    for ri in range(int(reps_i)):
                        seed0 = int(seed_i) + (int(ri) * 97)
                        s0, dt0, dm0 = _run_in_vivo_measurement_series_until_death(
                            payload,
                            ticks=int(ticks_i),
                            seed0=int(seed0),
                            death_names=death_names,
                        )
                        rep_series_out[int(ri)] = s0
                        rep_seeds_out[int(ri)] = int(seed0)
                        death_days_out[int(ri)] = int(dt0)
                        death_meas_out[int(ri)] = str(dm0)
                else:
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(reps_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=4.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(reps_i),
                            mp_context=ctx,
                            initializer=_char_worker_init,
                            initargs=(payload, int(ticks_i), int(seed_i), list(names), list(death_names), "invivo"),
                        ) as ex:
                            futs: list[concurrent.futures.Future] = []
                            for ri in range(int(reps_i)):
                                futs.append(ex.submit(_char_worker_eval, int(ri)))
                            for fut in concurrent.futures.as_completed(futs):
                                ri0, seed0, s0, dt0, dm0 = fut.result()
                                if 0 <= int(ri0) < int(len(rep_series_out)):
                                    rep_series_out[int(ri0)] = s0
                                    rep_seeds_out[int(ri0)] = int(seed0)
                                    death_days_out[int(ri0)] = int(dt0)
                                    death_meas_out[int(ri0)] = str(dm0)
                    finally:
                        _cm.__exit__(None, None, None)

                for ri in range(int(reps_i)):
                    sd = rep_seeds_out[int(ri)]
                    if not isinstance(sd, int):
                        sd = int(seed_i) + (int(ri) * 97)
                    s0 = rep_series_out[int(ri)]
                    if not isinstance(s0, dict):
                        s0 = {}
                    dt0 = death_days_out[int(ri)]
                    if not isinstance(dt0, int):
                        dt0 = int(ticks_i)
                    dm0 = death_meas_out[int(ri)]
                    if not isinstance(dm0, str):
                        dm0 = ""

                    rep_series.append(s0)
                    rep_seeds.append(int(sd))
                    death_days.append(int(dt0))
                    death_meas.append(str(dm0))

                # Persist files to disk.
                files_text: Dict[str, Any] = {}

                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]
                model_label0 = str(model_key or "")
                if model_label0.startswith("cell_culture_"):
                    model_label0 = model_label0[len("cell_culture_"):]
                if model_label0.startswith("tissue_"):
                    model_label0 = model_label0[len("tissue_"):]
                model_label = _omics_safe_label(model_label0, default="sample")

                meta_header = [
                    "run_id",
                    "sample_id",
                    "assay",
                    "model",
                    "replicate",
                    "seed",
                    "study_ran_for_days",
                    "death_observed",
                    "death_occurred_on_day",
                    "death_measurement",
                    "timecourse_filename",
                    "timecourse_relpath",
                    "timecourse_file_id",
                    "timecourse_url",
                ]
                meta_rows: list[list[Any]] = []
                tc_names: list[str] = []
                for ri, sd in enumerate(rep_seeds):
                    sample_id = f"{model_label}_r{int(ri)}_{run_tag}"
                    tc_name = f"replicates/{sample_id}_timecourse.csv"
                    tc_names.append(tc_name)
                    dd0 = int(death_days[int(ri)]) if int(ri) < int(len(death_days)) else int(ticks_i)
                    dm0 = str(death_meas[int(ri)]) if int(ri) < int(len(death_meas)) else ""
                    death_observed = 1 if int(dd0) < int(ticks_i) else 0
                    dd = int(dd0) if int(death_observed) == 1 else None
                    dm = str(dm0) if int(death_observed) == 1 else ""
                    tc_fid = _OMICS._file_id(run_id0, tc_name)
                    tc_url = f"/api/omics/file?run_id={run_id0}&name={tc_name}"
                    meta_rows.append(
                        [
                            str(run_id0),
                            str(sample_id),
                            "Characterization experiment",
                            str(model_key or ""),
                            int(ri),
                            int(sd),
                            int(ticks_i),
                            int(death_observed),
                            dd,
                            str(dm),
                            str(Path(tc_name).name),
                            str(tc_name),
                            str(tc_fid),
                            str(tc_url),
                        ]
                    )

                metadata_name = f"metadata_{run_tag}.csv"
                files_text[metadata_name] = _csv_from_rows(meta_header, meta_rows)

                tc_header = ["day", *[str(x) for x in names]]
                for ri, s0 in enumerate(rep_series):
                    tr: list[list[Any]] = []
                    for ti in range(int(ticks_i)):
                        row: list[Any] = [int(ti)]
                        for nm in names:
                            vals = s0.get(nm) if isinstance(s0.get(nm), list) else []
                            row.append(vals[int(ti)] if int(ti) < int(len(vals)) else "")
                        tr.append(row)
                    tc_name = tc_names[int(ri)] if int(ri) < int(len(tc_names)) else f"replicates/{model_label}_r{int(ri)}_{run_tag}_timecourse.csv"
                    files_text[str(tc_name)] = _csv_from_rows(tc_header, tr)

                manifest = {
                    "experiment": f"tests_{challenge}_characterization_v1",
                    "kind": "characterization",
                    "player_id": _sanitize_player_id(player_id),
                    "model": str(model_key or ""),
                    "ticks": int(ticks_i),
                    "age_days": int(ticks_i),
                    "replicates": int(reps_i),
                    "seed": int(seed_i),
                    "interventions": interventions_real,
                    "measurements": list(names),
                    "death_names": list(death_names),
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                iv_n = int(len(interventions)) if isinstance(interventions, list) else 0
                unit = _tests_compute_unit_cost_cents(
                    challenge=challenge,
                    kind="characterization",
                    model_key=model_key,
                    ticks=int(ticks_i),
                    interventions_n=int(iv_n),
                )
                fixed = _tests_compute_fixed_cost_cents(
                    challenge=challenge,
                    kind="characterization",
                    model_key=model_key,
                    interventions_n=int(iv_n),
                )
                charge = _tests_make_charge(
                    kind=f"tests_{challenge}_characterization",
                    samples=int(reps_i),
                    unit_cost_cents=int(unit),
                    fixed_cost_cents=int(fixed),
                    meta={
                        "experiment": f"tests_{challenge}_characterization_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "model": str(model_key or ""),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": f"tests_{challenge}_characterization_v1",
                        "model": str(model_key or ""),
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "measurements": names,
                        "run_id": str(omics_saved.get("run_id") or ""),
                        "files": omics_saved.get("files"),
                        "omics_inventory": {
                            "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                            "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                        },
                        "game": game,
                    },
                )
                return

            if self.path in ("/api/tests/cancer/protein_screen", "/api/tests/hereditary_disease/protein_screen", "/api/tests/aging/protein_screen"):
                if self.path.startswith("/api/tests/cancer/"):
                    challenge = "cancer"
                elif self.path.startswith("/api/tests/hereditary_disease/"):
                    challenge = "hereditary_disease"
                else:
                    challenge = "aging"
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = _tests_validate_protein_interventions(body.get("interventions"))
                model_key = body.get("model")

                if not _tests_is_in_vitro_model(model_key):
                    raise ValueError("protein_screen is only allowed for in vitro cell_culture_* models")

                ticks = body.get("ticks", 200)
                replicates = body.get("replicates", 10)
                direction = body.get("direction", "up")
                dose = body.get("dose", 1)

                payload0 = _tests_load_model_payload_for_challenge(challenge, model_key)
                baseline_payload = _deepcopy_payload(payload0)
                payload = _deepcopy_payload(payload0)
                interventions_real = _tests_translate_interventions_masked_to_real(interventions, model_key=model_key, challenge=challenge)
                if interventions_real:
                    payload["_tick_interventions"] = list(interventions_real)

                try:
                    ticks_i = max(1, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e

                seed_i = 1

                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]
                model_label0 = str(model_key or "")
                if model_label0.startswith("cell_culture_"):
                    model_label0 = model_label0[len("cell_culture_"):]
                if model_label0.startswith("tissue_"):
                    model_label0 = model_label0[len("tissue_"):]
                model_label = _omics_safe_label(model_label0, default="sample")

                try:
                    dose_f = float(dose)
                except Exception:
                    dose_f = 0.0
                if not np.isfinite(dose_f) or dose_f < 0.0:
                    dose_f = 0.0

                direction_s = str(direction or "").strip().lower()
                if direction_s in ("+", "inc", "increase", "up", "pos", "positive"):
                    direction_s = "up"
                elif direction_s in ("-", "dec", "decrease", "down", "neg", "negative"):
                    direction_s = "down"
                else:
                    raise ValueError("invalid direction")

                worker_mode = "process"
                workers_i = 35

                prot_layers = _protein_layer_names_from_payload(payload0)
                real_to_mask, _ = _tests_get_protein_mask_maps(model_key, challenge=challenge)
                if not prot_layers:
                    raise ValueError("no protein_* float32 layers found")

                meas_names = _measurement_names_from_payload(payload0)
                if not meas_names:
                    raise ValueError("model has no measurements_config measurements")

                def _eval_layer(li: int, nm: str) -> Dict[str, Any]:
                    p0 = _deepcopy_payload(payload)
                    tiv0 = p0.get("_tick_interventions")
                    tiv = list(tiv0) if isinstance(tiv0, list) else []
                    tiv.append({"layer": str(nm), "direction": str(direction_s), "dose": float(dose_f)})
                    p0["_tick_interventions"] = tiv
                    return {
                        "layer": str(nm),
                        "layer_index": int(li),
                        "measurements_end_sample": _cell_culture_measurements_end_sample_from_payload(
                            p0,
                            ticks=int(ticks_i),
                            seed=int(seed_i),
                            replicates=int(reps_i),
                            selected_names=meas_names,
                            condition_index=int(li + 2),
                        ).get("measurements_end_sample"),
                    }

                results_out: list[Optional[Dict[str, Any]]] = [None for _ in range(int(len(prot_layers)))]
                if int(len(prot_layers)) <= 1:
                    for li, nm in enumerate(prot_layers):
                        results_out[int(li)] = _eval_layer(int(li), str(nm))
                else:
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(workers_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=8.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(workers_i),
                            mp_context=ctx,
                            initializer=_vitro_screen_worker_init,
                            initargs=(payload, int(ticks_i), int(seed_i), int(reps_i), str(direction_s), float(dose_f), list(meas_names)),
                        ) as ex:
                            pending: set[concurrent.futures.Future] = set()
                            it = iter(list(enumerate(prot_layers)))

                            def _submit_one() -> None:
                                try:
                                    li0, nm0 = next(it)
                                except StopIteration:
                                    return
                                pending.add(ex.submit(_vitro_screen_worker_eval, int(li0), str(nm0)))

                            for _ in range(min(int(workers_i), int(len(prot_layers)))):
                                _submit_one()

                            while pending:
                                done, pending = concurrent.futures.wait(
                                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for fut in done:
                                    r = fut.result()
                                    li = int(r.get("layer_index") or 0)
                                    if 0 <= li < int(len(results_out)):
                                        results_out[li] = r
                                    _submit_one()
                    finally:
                        _cm.__exit__(None, None, None)

                results: list[Dict[str, Any]] = [r for r in results_out if isinstance(r, dict)]

                for r in results:
                    try:
                        ln = str(r.get("layer") or "")
                    except Exception:
                        ln = ""
                    if ln:
                        m = real_to_mask.get(ln)
                        if m:
                            r["layer"] = str(m)

                results_out2: list[Dict[str, Any]] = []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    results_out2.append(
                        {
                            "layer": r.get("layer"),
                            "layer_index": r.get("layer_index"),
                            "measurements_end_sample": r.get("measurements_end_sample"),
                        }
                    )

                baseline_out = _cell_culture_measurements_end_sample_from_payload(
                    baseline_payload,
                    ticks=int(ticks_i),
                    seed=int(seed_i),
                    replicates=int(reps_i),
                    selected_names=meas_names,
                    condition_index=0,
                )
                control_out = _cell_culture_measurements_end_sample_from_payload(
                    payload,
                    ticks=int(ticks_i),
                    seed=int(seed_i),
                    replicates=int(reps_i),
                    selected_names=meas_names,
                    condition_index=1,
                )

                # Persist files to disk.
                files_text: Dict[str, Any] = {}

                meta_header = [
                    "run_id",
                    "sample_id",
                    "assay",
                    "model",
                    "direction",
                    "dose",
                    "study_ran_for_days",
                    "replicates",
                    "replicate",
                    "seed",
                    "protein_layers",
                    "measurements_filename",
                    "measurements_relpath",
                    "measurements_file_id",
                    "measurements_url",
                ]
                meta_rows: list[list[Any]] = []

                rows_by_rep: Dict[int, list[list[Any]]] = {}
                seed_by_rep: Dict[int, int] = {}

                def _add_row(*, rep: int, seed: Any, condition: str, protein: str, meas: Any) -> None:
                    try:
                        rep_i = int(rep)
                    except Exception:
                        return
                    if rep_i < 0:
                        return
                    if rep_i not in rows_by_rep:
                        rows_by_rep[rep_i] = []
                    if rep_i not in seed_by_rep:
                        try:
                            seed_by_rep[rep_i] = int(seed)
                        except Exception:
                            seed_by_rep[rep_i] = int(seed_i)
                    m = meas if isinstance(meas, dict) else {}
                    row: list[Any] = [str(condition or ""), str(protein or "")]
                    for nm in meas_names:
                        v = m.get(nm)
                        if v is None:
                            row.append("")
                            continue
                        try:
                            vf = float(v)
                        except Exception:
                            row.append("")
                            continue
                        if not np.isfinite(vf):
                            row.append("")
                            continue
                        row.append(float(vf))
                    rows_by_rep[rep_i].append(row)

                b0 = baseline_out.get("measurements_end_sample")
                if not isinstance(b0, list):
                    b0 = []
                for ent in b0:
                    if not isinstance(ent, dict):
                        continue
                    rep = ent.get("replicate")
                    sd = ent.get("seed")
                    m = ent.get("measurements_end")
                    _add_row(rep=int(rep) if rep is not None else 0, seed=sd, condition="baseline", protein="", meas=m)

                c0 = control_out.get("measurements_end_sample")
                if not isinstance(c0, list):
                    c0 = []
                for ent in c0:
                    if not isinstance(ent, dict):
                        continue
                    rep = ent.get("replicate")
                    sd = ent.get("seed")
                    m = ent.get("measurements_end")
                    _add_row(rep=int(rep) if rep is not None else 0, seed=sd, condition="control", protein="", meas=m)

                for r in results_out2:
                    if not isinstance(r, dict):
                        continue
                    ln = str(r.get("layer") or "")
                    rows0 = r.get("measurements_end_sample")
                    if not isinstance(rows0, list):
                        rows0 = []
                    for ent in rows0:
                        if not isinstance(ent, dict):
                            continue
                        rep = ent.get("replicate")
                        sd = ent.get("seed")
                        m = ent.get("measurements_end")
                        _add_row(rep=int(rep) if rep is not None else 0, seed=sd, condition="perturb", protein=str(ln), meas=m)

                base_sid = f"{model_label}_{run_tag}"
                data_header = ["condition", "protein", *[str(x) for x in meas_names]]
                for rep_i in range(int(reps_i)):
                    sid = f"{base_sid}_r{int(rep_i)}"
                    safe_sid = _omics_safe_label(sid, default="sample")
                    fn = f"results/{safe_sid}_protein_screen_measurements.csv"
                    fid = _OMICS._file_id(run_id0, fn)
                    url = f"/api/omics/file?player_id={_sanitize_player_id(player_id)}&file_id={fid}"
                    out_rows = rows_by_rep.get(int(rep_i))
                    if not isinstance(out_rows, list):
                        out_rows = []
                    files_text[fn] = _csv_from_rows(data_header, out_rows)
                    sd_i = seed_by_rep.get(int(rep_i))
                    if sd_i is None:
                        sd_i = int(seed_i)
                    meta_rows.append(
                        [
                            str(run_id0),
                            str(sid),
                            "Drug screen",
                            str(model_key or ""),
                            str(direction_s),
                            float(dose_f),
                            int(ticks_i),
                            int(reps_i),
                            int(rep_i),
                            int(sd_i),
                            int(len(prot_layers)),
                            str(Path(fn).name),
                            str(fn),
                            str(fid),
                            str(url),
                        ]
                    )

                metadata_name = f"metadata_{run_tag}.csv"
                files_text[metadata_name] = _csv_from_rows(meta_header, meta_rows)

                manifest = {
                    "experiment": f"tests_{challenge}_protein_screen_v1",
                    "kind": "protein_screen",
                    "player_id": _sanitize_player_id(player_id),
                    "model": str(model_key or ""),
                    "ticks": int(ticks_i),
                    "replicates": int(reps_i),
                    "seed": int(seed_i),
                    "direction": str(direction_s),
                    "dose": float(dose_f),
                    "protein_layers": int(len(prot_layers)),
                    "measurements": list(meas_names),
                    "interventions": interventions_real,
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                samples_run = int(int(reps_i) * int(len(prot_layers) + 2))
                iv_n = int(len(interventions)) if isinstance(interventions, list) else 0
                unit = _tests_compute_unit_cost_cents(
                    challenge=challenge,
                    kind="protein_screen",
                    model_key=model_key,
                    ticks=int(ticks_i),
                    interventions_n=int(iv_n + 1),
                )
                fixed = _tests_compute_fixed_cost_cents(
                    challenge=challenge,
                    kind="protein_screen",
                    model_key=model_key,
                    interventions_n=int(iv_n + 1),
                )
                charge = _tests_make_charge(
                    kind=f"tests_{challenge}_protein_screen",
                    samples=int(samples_run),
                    unit_cost_cents=int(unit),
                    fixed_cost_cents=int(fixed),
                    meta={
                        "experiment": f"tests_{challenge}_protein_screen_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "protein_layers": int(len(prot_layers)),
                        "direction": str(direction_s),
                        "dose": float(dose_f),
                        "model": str(model_key or ""),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": f"tests_{challenge}_protein_screen_v1",
                        "model": str(model_key or ""),
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "workers": int(workers_i),
                        "worker_mode": str(worker_mode),
                        "direction": str(direction_s),
                        "dose": float(dose_f),
                        "protein_layers": int(len(prot_layers)),
                        "measurements": list(meas_names),
                        "interventions": interventions,
                        "run_id": str(omics_saved.get("run_id") or ""),
                        "files": omics_saved.get("files"),
                        "omics_inventory": {
                            "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                            "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                        },
                        "baseline": dict(baseline_out),
                        "control": dict(control_out),
                        "results": results_out2,
                        "game": game,
                    },
                )
                return

            if self.path == "/api/tests/aging/claim_cure":
                challenge = "aging"
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = _tests_validate_protein_interventions(body.get("interventions"))

                ticks = body.get("ticks", 200)
                replicates = body.get("replicates", 10)
                seed = body.get("seed", 1)

                ticks_cap = 1500
                try:
                    reps_req_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                prev_min_reps = _game_get_player_int(player_id, "aging_claim_cure_min_reps", default=0)
                reps_min_i = max(10, int(prev_min_reps))
                reps_i = max(int(reps_req_i), int(reps_min_i))
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]

                healthy0 = _tests_load_model_payload_for_challenge(challenge, "healthy")
                healthy_base = _deepcopy_payload(healthy0)
                healthy = _deepcopy_payload(healthy0)
                interventions_real = _tests_translate_interventions_masked_to_real(interventions, model_key="healthy", challenge=challenge)
                if interventions_real:
                    healthy["_tick_interventions"] = list(interventions_real)

                death_names = _death_measurement_names_from_payload(healthy)
                if not death_names:
                    raise ValueError("no death measurements found (measurement name must contain 'death')")

                death_ticks_base: list[int] = []
                death_ticks_treated: list[int] = []
                death_meas_base: list[str] = []
                death_meas_treated: list[str] = []
                death_ticks_base = [int(ticks_cap) for _ in range(int(reps_i))]
                death_ticks_treated = [int(ticks_cap) for _ in range(int(reps_i))]
                death_meas_base = ["" for _ in range(int(reps_i))]
                death_meas_treated = ["" for _ in range(int(reps_i))]
                if int(reps_i) <= 1:
                    for ri in range(int(reps_i)):
                        seed0_b = int(seed_i) + (0 * 1000003) + (int(ri) * 97)
                        seed0_t = int(seed_i) + (1 * 1000003) + (int(ri) * 97)
                        rb = _run_lifespan_death_tick(
                            healthy_base, ticks=int(ticks_cap), seed0=int(seed0_b), death_names=death_names
                        )
                        rt = _run_lifespan_death_tick(
                            healthy, ticks=int(ticks_cap), seed0=int(seed0_t), death_names=death_names
                        )
                        try:
                            death_ticks_base[int(ri)] = int(rb.get("death_tick"))
                        except Exception:
                            death_ticks_base[int(ri)] = int(ticks_cap)
                        try:
                            death_ticks_treated[int(ri)] = int(rt.get("death_tick"))
                        except Exception:
                            death_ticks_treated[int(ri)] = int(ticks_cap)
                        death_meas_base[int(ri)] = str(rb.get("death_measurement") or "")
                        death_meas_treated[int(ri)] = str(rt.get("death_measurement") or "")
                else:
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(reps_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=6.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(cpu_req),
                            mp_context=ctx,
                            initializer=_aging_claim_worker_init,
                            initargs=(healthy_base, healthy, list(death_names), int(ticks_cap), int(seed_i)),
                        ) as ex:
                            futs: list[concurrent.futures.Future] = []
                            for ri in range(int(reps_i)):
                                futs.append(ex.submit(_aging_claim_worker_eval, int(ri)))
                            for fut in concurrent.futures.as_completed(futs):
                                ri0, dt_b, dm_b, dt_t, dm_t = fut.result()
                                if 0 <= int(ri0) < int(reps_i):
                                    death_ticks_base[int(ri0)] = int(dt_b)
                                    death_meas_base[int(ri0)] = str(dm_b)
                                    death_ticks_treated[int(ri0)] = int(dt_t)
                                    death_meas_treated[int(ri0)] = str(dm_t)
                    finally:
                        _cm.__exit__(None, None, None)

                observed_until = int(ticks_cap)
                try:
                    max_dt = int(max([int(x) for x in (death_ticks_base + death_ticks_treated)]))
                    if int(max_dt) >= int(ticks_cap):
                        observed_until = int(ticks_cap)
                    else:
                        observed_until = int(min(int(ticks_cap), int(max_dt) + 1))
                except Exception:
                    observed_until = int(ticks_cap)
                if int(observed_until) < 1:
                    observed_until = 1

                stats_base = _tests_lifespan_stats(death_ticks_base, ticks=int(observed_until))
                stats_treated = _tests_lifespan_stats(death_ticks_treated, ticks=int(observed_until))
                curve_base = _lifespan_survival_curve(death_ticks_base, ticks=int(observed_until))
                curve_treated = _lifespan_survival_curve(death_ticks_treated, ticks=int(observed_until))
                try:
                    med_b = float(stats_base.get("median_lifespan_tick") or 0.0)
                except Exception:
                    med_b = float(0.0)
                try:
                    med_t = float(stats_treated.get("median_lifespan_tick") or 0.0)
                except Exception:
                    med_t = float(0.0)
                delta_med = float(med_t) - float(med_b)
                extra_days = float(delta_med)
                win = False
                try:
                    win = bool(int(max([int(x) for x in death_ticks_treated])) >= int(ticks_cap))
                except Exception:
                    win = False
                lifespan_recovery_pct = None
                try:
                    if float(med_b) > 0.0:
                        lifespan_recovery_pct = float(med_t) / float(med_b) * 100.0
                except Exception:
                    lifespan_recovery_pct = None

                files_text: Dict[str, Any] = {}

                meta_header = [
                    "run_id",
                    "sample_id",
                    "assay",
                    "study_observed_until_days",
                    "replicates_per_group",
                    "interventions_in_healthy_model",
                    "interventions",
                    "win",
                ]
                meta_rows: list[list[Any]] = []

                sid_sum = f"summarized_results_{run_tag}"
                fn_sum = f"results/summarized_results_{run_tag}.csv"

                sum_header = [
                    "sample_type",
                    "study_observed_until_days",
                    "age_at_death_days",
                    "cause_of_death",
                ]
                sum_rows: list[list[Any]] = []

                for ri in range(int(reps_i)):
                    dt = int(death_ticks_base[int(ri)]) if int(ri) < int(len(death_ticks_base)) else int(ticks_cap)
                    dm = str(death_meas_base[int(ri)]) if int(ri) < int(len(death_meas_base)) else ""
                    cause = str(dm or "")
                    if int(dt) >= int(ticks_cap):
                        cause = "survived_to_end"
                    sum_rows.append(["baseline_healthy", int(observed_until), int(dt), str(cause)])

                for ri in range(int(reps_i)):
                    dt = int(death_ticks_treated[int(ri)]) if int(ri) < int(len(death_ticks_treated)) else int(ticks_cap)
                    dm = str(death_meas_treated[int(ri)]) if int(ri) < int(len(death_meas_treated)) else ""
                    cause = str(dm or "")
                    if int(dt) >= int(ticks_cap):
                        cause = "survived_to_end"
                    sum_rows.append(["treated_healthy", int(observed_until), int(dt), str(cause)])

                files_text[fn_sum] = _csv_from_rows(sum_header, sum_rows)

                meta_rows.append(
                    [
                        str(run_id0),
                        str(sid_sum),
                        "in vivo lifespan study (aging claim)",
                        int(observed_until),
                        int(reps_i),
                        bool(bool(interventions_real)),
                        json.dumps(interventions_real, ensure_ascii=False),
                        bool(win),
                    ]
                )

                metadata_name = f"metadata_{run_tag}.csv"
                files_text[metadata_name] = _csv_from_rows(meta_header, meta_rows)

                iv_n = int(len(interventions)) if isinstance(interventions, list) else 0
                unit = _tests_compute_unit_cost_cents(
                    challenge=challenge,
                    kind="claim_cure",
                    model_key="healthy",
                    ticks=int(ticks_cap),
                    interventions_n=int(iv_n),
                )
                fixed = _tests_compute_fixed_cost_cents(
                    challenge=challenge,
                    kind="claim_cure",
                    model_key="healthy",
                    interventions_n=int(iv_n),
                )
                charge = _tests_make_charge(
                    kind=f"tests_{challenge}_claim_cure",
                    samples=int(2 * reps_i),
                    unit_cost_cents=int(unit),
                    fixed_cost_cents=int(fixed),
                    meta={
                        "experiment": f"tests_{challenge}_claim_cure_v1",
                        "ticks": int(ticks_cap),
                        "ticks_cap": int(ticks_cap),
                        "study_observed_until_days": int(observed_until),
                        "replicates": int(reps_i),
                    },
                )
                game = _game_apply_charge(player_id, charge)
                _game_set_player_int(player_id, "aging_claim_cure_min_reps", int(reps_i))

                score_lifedays_per_usd = None
                try:
                    msu = float(game.get("money_spent_usd") or 0.0) if isinstance(game, dict) else 0.0
                except Exception:
                    msu = 0.0
                if msu > 0.0:
                    try:
                        score_lifedays_per_usd = float(extra_days) / float(msu)
                    except Exception:
                        score_lifedays_per_usd = None

                score = None
                if score_lifedays_per_usd is not None:
                    try:
                        score = float(score_lifedays_per_usd) * 10000.0
                    except Exception:
                        score = None

                manifest = {
                    "experiment": f"tests_{challenge}_claim_cure_v1",
                    "kind": "claim_cure",
                    "player_id": _sanitize_player_id(player_id),
                    "ticks": int(ticks_cap),
                    "ticks_cap": int(ticks_cap),
                    "study_observed_until_days": int(observed_until),
                    "replicates": int(reps_i),
                    "replicates_requested": int(reps_req_i),
                    "replicates_min_enforced": int(reps_min_i),
                    "seed": int(seed_i),
                    "interventions": interventions_real,
                    "baseline_healthy_median_tick": float(med_b),
                    "treated_healthy_median_tick": float(med_t),
                    "extra_days": float(extra_days),
                    "score_lifedays_per_usd": score_lifedays_per_usd,
                    "score": score,
                    "delta_median_ticks": float(delta_med),
                    "lifespan_recovery_pct": lifespan_recovery_pct,
                    "win": bool(win),
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": f"tests_{challenge}_claim_cure_v1",
                        "ticks": int(ticks_cap),
                        "ticks_cap": int(ticks_cap),
                        "study_observed_until_days": int(observed_until),
                        "replicates": int(reps_i),
                        "replicates_requested": int(reps_req_i),
                        "replicates_min_enforced": int(reps_min_i),
                        "death_measurements": list(death_names),
                        "run_id": str(omics_saved.get("run_id") or ""),
                        "files": omics_saved.get("files"),
                        "omics_inventory": {
                            "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                            "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                        },
                        "baseline_healthy": {
                            **stats_base,
                            "death_ticks": [int(x) for x in death_ticks_base],
                            "death_measurements": [str(x) for x in death_meas_base],
                            "curve": curve_base,
                        },
                        "healthy": {
                            **stats_treated,
                            "death_ticks": [int(x) for x in death_ticks_treated],
                            "death_measurements": [str(x) for x in death_meas_treated],
                            "curve": curve_treated,
                        },
                        "baseline_healthy_median_tick": float(med_b),
                        "treated_healthy_median_tick": float(med_t),
                        "extra_days": float(extra_days),
                        "score_lifedays_per_usd": score_lifedays_per_usd,
                        "score": score,
                        "delta_median_ticks": float(delta_med),
                        "lifespan_recovery_pct": lifespan_recovery_pct,
                        "win": bool(win),
                        "game": game,
                    },
                )
                return

            if self.path in ("/api/tests/cancer/claim_cure", "/api/tests/hereditary_disease/claim_cure"):
                challenge = "cancer" if self.path.startswith("/api/tests/cancer/") else "hereditary_disease"
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = _tests_validate_protein_interventions(body.get("interventions"))

                ticks = body.get("ticks", 200)
                replicates = body.get("replicates", 10)
                seed = body.get("seed", 1)

                ticks_i = 400
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]

                disease_key = _tests_claim_cure_disease_model_key_for_challenge(challenge)
                healthy_key = "healthy_organism" if str(challenge) == "cancer" else "healthy"
                healthy = _tests_load_model_payload_for_challenge(challenge, healthy_key)
                sick0 = _tests_load_model_payload_for_challenge(challenge, str(disease_key))
                sick = _deepcopy_payload(sick0)
                interventions_real = _tests_translate_interventions_masked_to_real(interventions, model_key=str(disease_key), challenge=challenge)
                run_treated = bool(interventions_real)
                if run_treated:
                    sick["_tick_interventions"] = list(interventions_real)

                death_names = _death_measurement_names_from_payload(healthy)
                dn2 = _death_measurement_names_from_payload(sick)
                seen_dn = set(death_names)
                for nm in dn2:
                    if nm in seen_dn:
                        continue
                    death_names.append(nm)
                    seen_dn.add(nm)

                if not death_names:
                    raise ValueError("no death measurements found (measurement name must contain 'death')")

                death_ticks_healthy: list[int] = []
                death_ticks_sick_base: list[int] = []
                death_ticks_sick: list[int] = []
                death_meas_healthy: list[str] = []
                death_meas_sick_base: list[str] = []
                death_meas_sick: list[str] = []
                death_ticks_healthy = [int(ticks_i) for _ in range(int(reps_i))]
                death_ticks_sick_base = [int(ticks_i) for _ in range(int(reps_i))]
                death_ticks_sick = [int(ticks_i) for _ in range(int(reps_i))]
                death_meas_healthy = ["" for _ in range(int(reps_i))]
                death_meas_sick_base = ["" for _ in range(int(reps_i))]
                death_meas_sick = ["" for _ in range(int(reps_i))]
                if int(reps_i) <= 1:
                    for ri in range(int(reps_i)):
                        seed0_h = int(seed_i) + (0 * 1000003) + (int(ri) * 97)
                        seed0_s = int(seed_i) + (1 * 1000003) + (int(ri) * 97)
                        rh = _run_lifespan_death_tick(
                            healthy, ticks=int(ticks_i), seed0=int(seed0_h), death_names=death_names
                        )
                        rs0 = _run_lifespan_death_tick(
                            sick0, ticks=int(ticks_i), seed0=int(seed0_s), death_names=death_names
                        )
                        rs = (
                            _run_lifespan_death_tick(sick, ticks=int(ticks_i), seed0=int(seed0_s), death_names=death_names)
                            if run_treated
                            else rs0
                        )
                        try:
                            death_ticks_healthy[int(ri)] = int(rh.get("death_tick"))
                        except Exception:
                            death_ticks_healthy[int(ri)] = int(ticks_i)
                        try:
                            death_ticks_sick_base[int(ri)] = int(rs0.get("death_tick"))
                        except Exception:
                            death_ticks_sick_base[int(ri)] = int(ticks_i)
                        try:
                            death_ticks_sick[int(ri)] = int(rs.get("death_tick"))
                        except Exception:
                            death_ticks_sick[int(ri)] = int(ticks_i)
                        death_meas_healthy[int(ri)] = str(rh.get("death_measurement") or "")
                        death_meas_sick_base[int(ri)] = str(rs0.get("death_measurement") or "")
                        death_meas_sick[int(ri)] = str(rs.get("death_measurement") or "")
                else:
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(reps_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=6.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(reps_i),
                            mp_context=ctx,
                            initializer=_disease_claim_worker_init,
                            initargs=(healthy, sick0, sick, list(death_names), int(ticks_i), int(seed_i), bool(run_treated)),
                        ) as ex:
                            futs: list[concurrent.futures.Future] = []
                            for ri in range(int(reps_i)):
                                futs.append(ex.submit(_disease_claim_worker_eval, int(ri)))
                            for fut in concurrent.futures.as_completed(futs):
                                ri0, dt_h, dm_h, dt_s0, dm_s0, dt_s, dm_s = fut.result()
                                if 0 <= int(ri0) < int(reps_i):
                                    death_ticks_healthy[int(ri0)] = int(dt_h)
                                    death_meas_healthy[int(ri0)] = str(dm_h)
                                    death_ticks_sick_base[int(ri0)] = int(dt_s0)
                                    death_meas_sick_base[int(ri0)] = str(dm_s0)
                                    death_ticks_sick[int(ri0)] = int(dt_s)
                                    death_meas_sick[int(ri0)] = str(dm_s)
                    finally:
                        _cm.__exit__(None, None, None)

                stats_healthy = _tests_lifespan_stats(death_ticks_healthy, ticks=int(ticks_i))
                stats_sick_base = _tests_lifespan_stats(death_ticks_sick_base, ticks=int(ticks_i))
                stats_sick = _tests_lifespan_stats(death_ticks_sick, ticks=int(ticks_i))
                curve_healthy = _lifespan_survival_curve(death_ticks_healthy, ticks=int(ticks_i))
                curve_sick_base = _lifespan_survival_curve(death_ticks_sick_base, ticks=int(ticks_i))
                curve_sick = _lifespan_survival_curve(death_ticks_sick, ticks=int(ticks_i))
                try:
                    med_h = float(stats_healthy.get("median_lifespan_tick" ) or 0.0)
                except Exception:
                    med_h = float(0.0)
                try:
                    med_s0 = float(stats_sick_base.get("median_lifespan_tick") or 0.0)
                except Exception:
                    med_s0 = float(0.0)
                try:
                    med_s = float(stats_sick.get("median_lifespan_tick") or 0.0)
                except Exception:
                    med_s = float(0.0)
                delta_med = float(med_s) - float(med_h)
                extra_days = float(med_s) - float(med_s0)
                lifespan_recovery_pct = None
                try:
                    if float(med_h) > 0.0:
                        lifespan_recovery_pct = float(med_s) / float(med_h) * 100.0
                except Exception:
                    lifespan_recovery_pct = None

                win = False
                try:
                    if lifespan_recovery_pct is not None and float(lifespan_recovery_pct) >= 90.0:
                        win = True
                except Exception:
                    win = False

                files_text: Dict[str, Any] = {}

                meta_header = [
                    "run_id",
                    "sample_id",
                    "assay",
                    "study_observed_until_days",
                    "replicates_per_group",
                    "interventions_in_cancer_model",
                    "interventions",
                    "win",
                ]
                meta_rows: list[list[Any]] = []

                sid_sum = f"summarized_results_{run_tag}"
                fn_sum = f"results/summarized_results_{run_tag}.csv"

                sum_header = [
                    "sample_type",
                    "study_observed_until_days",
                    "age_at_death_days",
                    "cause_of_death",
                ]
                sum_rows: list[list[Any]] = []

                # One row per individual. The simulation produces `reps_i` healthy and `reps_i` sick.
                for ri in range(int(reps_i)):
                    dt = int(death_ticks_healthy[int(ri)]) if int(ri) < int(len(death_ticks_healthy)) else int(ticks_i)
                    dm = str(death_meas_healthy[int(ri)]) if int(ri) < int(len(death_meas_healthy)) else ""
                    cause = str(dm or "")
                    if int(dt) >= int(ticks_i):
                        cause = "survived_to_end"
                    sum_rows.append(["healthy", int(ticks_i), int(dt), str(cause)])

                if run_treated:
                    for ri in range(int(reps_i)):
                        dt = int(death_ticks_sick_base[int(ri)]) if int(ri) < int(len(death_ticks_sick_base)) else int(ticks_i)
                        dm = str(death_meas_sick_base[int(ri)]) if int(ri) < int(len(death_meas_sick_base)) else ""
                        cause = str(dm or "")
                        if int(dt) >= int(ticks_i):
                            cause = "survived_to_end"
                        sum_rows.append([f"baseline_{str(disease_key)}", int(ticks_i), int(dt), str(cause)])

                for ri in range(int(reps_i)):
                    dt = int(death_ticks_sick[int(ri)]) if int(ri) < int(len(death_ticks_sick)) else int(ticks_i)
                    dm = str(death_meas_sick[int(ri)]) if int(ri) < int(len(death_meas_sick)) else ""
                    cause = str(dm or "")
                    if int(dt) >= int(ticks_i):
                        cause = "survived_to_end"
                    sum_type = f"treated_{str(disease_key)}" if run_treated else str(disease_key)
                    sum_rows.append([sum_type, int(ticks_i), int(dt), str(cause)])

                files_text[fn_sum] = _csv_from_rows(sum_header, sum_rows)

                meta_rows.append(
                    [
                        str(run_id0),
                        str(sid_sum),
                        "in vivo lifespan study (win claim)",
                        int(ticks_i),
                        int(reps_i),
                        bool(bool(interventions_real)),
                        json.dumps(interventions_real, ensure_ascii=False),
                        bool(win),
                    ]
                )

                metadata_name = f"metadata_{run_tag}.csv"
                files_text[metadata_name] = _csv_from_rows(meta_header, meta_rows)

                iv_n = int(len(interventions)) if isinstance(interventions, list) else 0
                unit = _tests_compute_unit_cost_cents(
                    challenge=challenge,
                    kind="claim_cure",
                    model_key=str(disease_key),
                    ticks=int(ticks_i),
                    interventions_n=int(iv_n),
                )
                fixed = _tests_compute_fixed_cost_cents(
                    challenge=challenge,
                    kind="claim_cure",
                    model_key=str(disease_key),
                    interventions_n=int(iv_n),
                )
                charge = _tests_make_charge(
                    kind=f"tests_{challenge}_claim_cure",
                    samples=int(int((3 if run_treated else 2)) * int(reps_i)),
                    unit_cost_cents=int(unit),
                    fixed_cost_cents=int(fixed),
                    meta={
                        "experiment": f"tests_{challenge}_claim_cure_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                score_lifedays_per_usd = None
                try:
                    msu = float(game.get("money_spent_usd") or 0.0) if isinstance(game, dict) else 0.0
                except Exception:
                    msu = 0.0
                if msu > 0.0:
                    try:
                        score_lifedays_per_usd = float(extra_days) / float(msu)
                    except Exception:
                        score_lifedays_per_usd = None

                score = None
                if score_lifedays_per_usd is not None:
                    try:
                        score = float(score_lifedays_per_usd) * 10000.0
                    except Exception:
                        score = None

                manifest = {
                    "experiment": f"tests_{challenge}_claim_cure_v1",
                    "kind": "claim_cure",
                    "player_id": _sanitize_player_id(player_id),
                    "ticks": int(ticks_i),
                    "replicates": int(reps_i),
                    "seed": int(seed_i),
                    "interventions": interventions_real,
                    "baseline_disease_median_tick": float(med_s0),
                    "treated_disease_median_tick": float(med_s),
                    "extra_days": float(extra_days),
                    "score_lifedays_per_usd": score_lifedays_per_usd,
                    "score": score,
                    "delta_median_ticks": float(delta_med),
                    "lifespan_recovery_pct": lifespan_recovery_pct,
                    "win": bool(win),
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                out_json = {
                    "ok": True,
                    "experiment": f"tests_{challenge}_claim_cure_v1",
                    "ticks": int(ticks_i),
                    "replicates": int(reps_i),
                    "death_measurements": list(death_names),
                    "run_id": str(omics_saved.get("run_id") or ""),
                    "files": omics_saved.get("files"),
                    "omics_inventory": {
                        "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                        "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                    },
                    "healthy": {
                        **stats_healthy,
                        "death_ticks": [int(x) for x in death_ticks_healthy],
                        "death_measurements": [str(x) for x in death_meas_healthy],
                        "curve": curve_healthy,
                    },
                    "sick": {
                        **stats_sick,
                        "death_ticks": [int(x) for x in death_ticks_sick],
                        "death_measurements": [str(x) for x in death_meas_sick],
                        "curve": curve_sick,
                    },
                    "baseline_disease_median_tick": float(med_s0),
                    "treated_disease_median_tick": float(med_s),
                    "extra_days": float(extra_days),
                    "score_lifedays_per_usd": score_lifedays_per_usd,
                    "score": score,
                    "delta_median_ticks": float(delta_med),
                    "lifespan_recovery_pct": lifespan_recovery_pct,
                    "win": bool(win),
                    "game": game,
                }
                if run_treated:
                    out_json["baseline_sick"] = {
                        **stats_sick_base,
                        "death_ticks": [int(x) for x in death_ticks_sick_base],
                        "death_measurements": [str(x) for x in death_meas_sick_base],
                        "curve": curve_sick_base,
                    }
                self._send_json(200, out_json)
                return

            if self.path == "/api/experiments/spatial_tx":
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = body.get("interventions")

                ticks = body.get("ticks", 100)
                replicates = body.get("replicates", 1)
                seed = body.get("seed", 1)
                gene_set_req = body.get("gene_set")
                gene_set_name, genes = _load_stx_gene_set(str(gene_set_req or ""))

                conds = body.get("conditions")
                cond_list: list[Dict[str, Any]] = []
                if isinstance(conds, list):
                    for c in conds:
                        if not isinstance(c, dict):
                            continue
                        nm = str(c.get("name") or "").strip()
                        payload = c.get("payload")
                        if not nm or not isinstance(payload, dict):
                            continue
                        cond_list.append({"name": nm, "payload": payload})

                if cond_list:
                    for ent in cond_list:
                        try:
                            nm = str(ent.get("name") or "").strip().lower()
                        except Exception:
                            nm = ""
                        if nm != "sick":
                            continue
                        p0 = ent.get("payload")
                        if not isinstance(p0, dict):
                            continue
                        p1 = _deepcopy_payload(p0)
                        if isinstance(interventions, list) and interventions:
                            p1["_tick_interventions"] = list(interventions)
                        ent["payload"] = p1

                if not cond_list:
                    healthy = body.get("healthy")
                    sick = body.get("sick")
                    if isinstance(healthy, dict):
                        cond_list.append({"name": "healthy", "payload": healthy})
                    if isinstance(sick, dict):
                        sick2 = _deepcopy_payload(sick)
                        if isinstance(interventions, list) and interventions:
                            sick2["_tick_interventions"] = list(interventions)
                        cond_list.append({"name": "sick", "payload": sick2})

                if not cond_list:
                    raise ValueError("missing conditions (provide conditions[] or healthy/sick payloads)")

                if not genes:
                    first_payload = cond_list[0].get("payload") if cond_list else None
                    if isinstance(first_payload, dict):
                        genes = _default_stx_gene_list(first_payload, max_genes=8)
                if not genes:
                    raise ValueError("no genes selected")

                z_target_arr = None
                z_target_in = body.get("z_target")
                if isinstance(z_target_in, list):
                    if len(z_target_in) != int(len(genes)):
                        raise ValueError("z_target must have length == #genes")
                    tmp: list[float] = []
                    for v in z_target_in:
                        try:
                            f = float(v)
                        except Exception:
                            f = float("nan")
                        tmp.append(f)
                    z_target_arr = np.asarray(tmp, dtype=np.float64)

                try:
                    ticks_i = max(0, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e
                syn_rng = np.random.default_rng(3)

                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]

                # Death-aware preflight: spatial/bulk assays are snapshots at `ticks`, so
                # cancel if any replicate dies before the requested tick.
                replicate_deaths: list[Dict[str, Any]] = []

                meta_header = [
                    "cell_id",
                    "sample",
                    "condition",
                    "assay",
                    "replicate",
                    "seed",
                    "sample_taken_at_day",
                    "x",
                    "y",
                    "grid_index",
                ]
                meta_rows: list[list[Any]] = []
                stx_kind = _stx_kind_from_gene_set_and_genes(gene_set_name, genes)
                assay = "Spatial transcriptomics"
                if str(stx_kind) == "bulk_proteomics":
                    assay = "Spatial proteomics"
                genes_masked = _bulk_omics_mask_feature_headers(genes, kind=str(stx_kind))
                mat_header = ["cell_id", *genes_masked]
                truth_mat_rows: list[list[Any]] = []
                noisy_mat_rows: list[list[Any]] = []

                out_runs: list[Dict[str, Any]] = []
                files_text: Dict[str, Any] = {}
                for ci, c in enumerate(cond_list):
                    nm = str(c.get("name") or "").strip()
                    payload = c.get("payload")
                    if not nm or not isinstance(payload, dict):
                        continue

                    safe_cond = _omics_safe_label(nm, default="condition")

                    for ri in range(reps_i):
                        seed0 = int(seed_i) + (int(ci) * 1000003) + (int(ri) * 97)
                        pf = _preflight_death_before_ticks(payload, ticks=int(ticks_i), seed0=int(seed0))
                        if isinstance(pf, dict):
                            replicate_deaths.append(
                                {
                                    "condition": str(nm),
                                    "replicate": int(ri),
                                    "seed": int(seed0),
                                    "requested_ticks": int(ticks_i),
                                    "death_tick": int(pf.get("death_tick") or 0),
                                    "death_measurement": str(pf.get("death_measurement") or ""),
                                    "death_names": pf.get("death_names") if isinstance(pf.get("death_names"), list) else [],
                                }
                            )
                            continue
                        p = _run_payload_ticks(payload, ticks=ticks_i, seed0=seed0)
                        tx = _spatial_tx_rows(
                            p,
                            genes,
                            cell_layer="",
                            min_cell_value=0.0,
                            stride=1,
                            max_spots=None,
                            seed=seed0,
                        )

                        H = int(tx.get("H") or 0)
                        W = int(tx.get("W") or 0)
                        rows = tx.get("rows")
                        if not isinstance(rows, list):
                            rows = []

                        run_cell_ids: list[str] = []
                        run_x: list[int] = []
                        run_y: list[int] = []
                        run_T: list[list[float]] = []
                        rep_meta_rows: list[list[Any]] = []

                        for si, row in enumerate(rows):
                            if not isinstance(row, dict):
                                continue
                            x = row.get("x")
                            y = row.get("y")
                            try:
                                xi = int(x)
                            except Exception:
                                continue
                            try:
                                yi = int(y)
                            except Exception:
                                continue
                            grid_index = (yi * int(W) + xi) if (W > 0) else int(yi)

                            cell_id = f"{safe_cond}_r{int(ri)}_{run_tag}_s{int(seed0)}_{int(si)}"

                            meta_rows.append(
                                [
                                    cell_id,
                                    nm,
                                    nm,
                                    str(assay),
                                    int(ri),
                                    int(seed0),
                                    int(ticks_i),
                                    int(xi),
                                    int(yi),
                                    int(grid_index),
                                ]
                            )
                            rep_meta_rows.append(
                                [
                                    cell_id,
                                    nm,
                                    nm,
                                    str(assay),
                                    int(ri),
                                    int(seed0),
                                    int(ticks_i),
                                    int(xi),
                                    int(yi),
                                    int(grid_index),
                                ]
                            )
                            vals: list[Any] = []
                            for g in genes:
                                vals.append(row.get(g))
                            vv: list[float] = []
                            for x0 in vals:
                                try:
                                    f0 = float(x0)
                                except Exception:
                                    f0 = 0.0
                                if not np.isfinite(f0) or f0 < 0.0:
                                    f0 = 0.0
                                vv.append(float(f0))

                            truth_mat_rows.append([cell_id, *vv])
                            run_cell_ids.append(cell_id)
                            run_x.append(int(xi))
                            run_y.append(int(yi))
                            run_T.append(vv)

                        if run_cell_ids and run_T:
                            T_arr = np.asarray(run_T, dtype=np.float64)
                            x_arr = np.asarray(run_x, dtype=np.int64)
                            y_arr = np.asarray(run_y, dtype=np.int64)
                            Y = _stx_synthetic_v3_noisy_counts(T_arr, x_arr, y_arr, H=int(H), W=int(W), rng=syn_rng, z_target=z_target_arr)
                            for ii, cid in enumerate(run_cell_ids):
                                noisy_mat_rows.append([cid, *[int(x) for x in np.asarray(Y[ii], dtype=np.int64).tolist()]])

                            rep_prefix = f"replicates/{safe_cond}_r{int(ri)}_{run_tag}"
                            rep_rows = [[run_cell_ids[ii], *[int(x) for x in np.asarray(Y[ii], dtype=np.int64).tolist()]] for ii in range(len(run_cell_ids))]
                            files_text[f"{rep_prefix}_matrix.csv"] = _csv_from_rows(mat_header, rep_rows)
                            files_text[f"{rep_prefix}_cell_metadata.csv"] = _csv_from_rows(meta_header, rep_meta_rows)

                        out_runs.append(
                            {
                                "condition": nm,
                                "replicate": int(ri),
                                "seed": int(seed0),
                                "ticks": int(ticks_i),
                                "cells": int(len(rows)),
                                "H": int(H),
                                "W": int(W),
                            }
                        )

                if not out_runs:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": f"all replicates died before requested ticks={int(ticks_i)}",
                            "error_kind": "all_replicates_died",
                            "experiment": "spatial_tx_v1",
                            "details": {
                                "requested_ticks": int(ticks_i),
                                "replicates_requested": int(reps_i),
                                "replicate_deaths": list(replicate_deaths),
                            },
                        },
                    )
                    return

                matrix_truth_csv = _csv_from_rows(mat_header, truth_mat_rows)
                matrix_noisy_csv = _csv_from_rows(mat_header, noisy_mat_rows)
                metadata_csv = _csv_from_rows(meta_header, meta_rows)

                files_text[f"metadata_{run_tag}.csv"] = str(metadata_csv)
                manifest = {
                    "experiment": "spatial_tx_v1",
                    "kind": str(stx_kind),
                    "player_id": _sanitize_player_id(player_id),
                    "ticks": int(ticks_i),
                    "replicates": int(reps_i),
                    "replicates_completed": int(len(out_runs)),
                    "replicate_deaths": list(replicate_deaths),
                    "gene_set": str(gene_set_name or ""),
                    "genes": list(genes_masked),
                    "runs": out_runs,
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                samples_run = int(len(out_runs))
                charge = _game_compute_charge(
                    "spatial_transcriptomics",
                    samples_run,
                    meta={
                        "experiment": "spatial_tx_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "gene_set": str(gene_set_name or ""),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": "spatial_tx_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "replicates_completed": int(len(out_runs)),
                        "replicate_deaths": list(replicate_deaths),
                        "gene_set": str(gene_set_name or ""),
                        "genes": genes_masked,
                        "run_id": str(omics_saved.get("run_id") or ""),
                        "files": omics_saved.get("files"),
                        "noise": {
                            "model": "synthetic_v3",
                            "sigma_cell": 0.35,
                            "theta": 50.0,
                            "eps": 0.08,
                            "target_median_umi": 2000.0,
                            "ambient_total_umi": 0.05,
                            "ambient_sigma_cell": 0.25,
                            "rng_seed": 3,
                        },
                        "matrix_csv": matrix_noisy_csv,
                        "matrix_truth_csv": matrix_truth_csv,
                        "matrix_noisy_csv": matrix_noisy_csv,
                        "metadata_csv": metadata_csv,
                        "runs": out_runs,
                        "game": game,
                    },
                )
                return

            if self.path == "/api/experiments/in_vivo_trial":
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = body.get("interventions")

                include_replicates_raw = body.get("include_replicates", False)
                include_replicates = bool(include_replicates_raw)

                ticks = body.get("ticks", 100)
                replicates = body.get("replicates", 1)
                seed = body.get("seed", 1)
                workers_raw = body.get("workers", None)
                worker_mode_raw = body.get("worker_mode", "process")
                healthy = body.get("healthy")
                sick = body.get("sick")
                if not isinstance(healthy, dict) or not isinstance(sick, dict):
                    raise ValueError("missing healthy/sick payloads")

                sick2 = _deepcopy_payload(sick)
                if isinstance(interventions, list) and interventions:
                    sick2["_tick_interventions"] = list(interventions)

                try:
                    ticks_i = max(0, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                requested_ticks = int(ticks_i)
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                worker_mode = str(worker_mode_raw or "process").strip().lower()
                if worker_mode not in ("thread", "process"):
                    worker_mode = "process"

                workers_missing = "workers" not in body
                if workers_missing:
                    workers_i = 1
                else:
                    try:
                        workers_i = int(workers_raw)
                    except Exception:
                        workers_i = 0
                if workers_i <= 0:
                    workers_i = max(1, int(min(4, os.cpu_count() or 1)))
                workers_i = max(1, min(int(workers_i), int(reps_i), 32))

                names = _measurement_names_union(healthy, sick)
                if not names:
                    raise ValueError("no measurements configured (missing measurements_config)")

                death_names = _death_measurement_names_from_payload(healthy)
                dn2 = _death_measurement_names_from_payload(sick2)
                seen_dn = set(death_names)
                for nm in dn2:
                    if nm in seen_dn:
                        continue
                    death_names.append(nm)
                    seen_dn.add(nm)

                # Auto-extend ticks until all individuals in (at least) one group are dead,
                # with a safety cap so runs can't explode.
                ticks_cap = int(max(0, max(int(requested_ticks), int(max(200, int(5 * int(requested_ticks or 1)))))))
                ticks_cap = int(min(int(ticks_cap), 5000))
                ticks_probe = int(max(1, int(requested_ticks)))
                ticks_probe = int(min(int(ticks_probe), int(ticks_cap)))

                probe_death_ticks_h: list[int] = [int(ticks_probe) for _ in range(int(reps_i))]
                probe_death_ticks_s: list[int] = [int(ticks_probe) for _ in range(int(reps_i))]
                probe_death_meas_h: list[str] = ["" for _ in range(int(reps_i))]
                probe_death_meas_s: list[str] = ["" for _ in range(int(reps_i))]

                def _run_death_probe(ticks_run: int) -> None:
                    nonlocal probe_death_ticks_h
                    nonlocal probe_death_ticks_s
                    nonlocal probe_death_meas_h
                    nonlocal probe_death_meas_s

                    probe_death_ticks_h = [int(ticks_run) for _ in range(int(reps_i))]
                    probe_death_ticks_s = [int(ticks_run) for _ in range(int(reps_i))]
                    probe_death_meas_h = ["" for _ in range(int(reps_i))]
                    probe_death_meas_s = ["" for _ in range(int(reps_i))]

                    if int(reps_i) <= 1 or int(workers_i) <= 1:
                        for ri in range(int(reps_i)):
                            seed0_h = int(seed_i) + (0 * 1000003) + (int(ri) * 97)
                            seed0_s = int(seed_i) + (1 * 1000003) + (int(ri) * 97)
                            rh = _run_lifespan_death_tick(healthy, ticks=int(ticks_run), seed0=int(seed0_h), death_names=death_names)
                            rs = _run_lifespan_death_tick(sick2, ticks=int(ticks_run), seed0=int(seed0_s), death_names=death_names)
                            try:
                                probe_death_ticks_h[int(ri)] = int(rh.get("death_tick"))
                            except Exception:
                                probe_death_ticks_h[int(ri)] = int(ticks_run)
                            try:
                                probe_death_ticks_s[int(ri)] = int(rs.get("death_tick"))
                            except Exception:
                                probe_death_ticks_s[int(ri)] = int(ticks_run)
                            probe_death_meas_h[int(ri)] = str(rh.get("death_measurement") or "")
                            probe_death_meas_s[int(ri)] = str(rs.get("death_measurement") or "")
                        return

                    if worker_mode == "process":
                        ctx = mp.get_context("spawn")
                        cpu_req = max(1, min(int(workers_i), int(_RESOURCE_SCHED.cpu_total)))
                        _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=6.0)
                        _cm.__enter__()
                        try:
                            with concurrent.futures.ProcessPoolExecutor(
                                max_workers=int(workers_i),
                                mp_context=ctx,
                                initializer=_invivo_worker_init_v2,
                                initargs=(healthy, sick2, death_names),
                            ) as ex:
                                futs: list[concurrent.futures.Future] = []
                                for ci in (0, 1):
                                    for ri in range(int(reps_i)):
                                        futs.append(ex.submit(_invivo_worker_death_eval, int(ci), int(ri), int(ticks_run), int(seed_i)))
                                for fut in concurrent.futures.as_completed(futs):
                                    ci, ri, dt, dm = fut.result()
                                    if 0 <= int(ri) < int(reps_i):
                                        if int(ci) == 0:
                                            probe_death_ticks_h[int(ri)] = int(dt)
                                            probe_death_meas_h[int(ri)] = str(dm)
                                        else:
                                            probe_death_ticks_s[int(ri)] = int(dt)
                                            probe_death_meas_s[int(ri)] = str(dm)
                        finally:
                            _cm.__exit__(None, None, None)
                        return

                    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers_i)) as ex:
                        futs2: list[concurrent.futures.Future] = []
                        for ci in (0, 1):
                            for ri in range(int(reps_i)):
                                futs2.append(ex.submit(_invivo_worker_death_eval, int(ci), int(ri), int(ticks_run), int(seed_i)))
                        for fut in concurrent.futures.as_completed(futs2):
                            ci, ri, dt, dm = fut.result()
                            if 0 <= int(ri) < int(reps_i):
                                if int(ci) == 0:
                                    probe_death_ticks_h[int(ri)] = int(dt)
                                    probe_death_meas_h[int(ri)] = str(dm)
                                else:
                                    probe_death_ticks_s[int(ri)] = int(dt)
                                    probe_death_meas_s[int(ri)] = str(dm)

                while True:
                    _run_death_probe(int(ticks_probe))
                    try:
                        max_h_probe = int(max(int(x) for x in probe_death_ticks_h)) if probe_death_ticks_h else int(ticks_probe)
                    except Exception:
                        max_h_probe = int(ticks_probe)
                    try:
                        max_s_probe = int(max(int(x) for x in probe_death_ticks_s)) if probe_death_ticks_s else int(ticks_probe)
                    except Exception:
                        max_s_probe = int(ticks_probe)

                    healthy_all_dead = bool(int(max_h_probe) < int(ticks_probe))
                    sick_all_dead = bool(int(max_s_probe) < int(ticks_probe))
                    if healthy_all_dead or sick_all_dead:
                        break
                    if int(ticks_probe) >= int(ticks_cap):
                        break
                    ticks_probe = int(min(int(ticks_cap), max(int(ticks_probe) + 1, int(ticks_probe) * 2)))

                # Determine how many ticks to actually compute full measurement series for.
                # If a group fully died during the probe, we stop at the earlier group's extinction time (+1 tick).
                # Otherwise we run to ticks_cap and warn.
                try:
                    max_h_probe2 = int(max(int(x) for x in probe_death_ticks_h)) if probe_death_ticks_h else int(ticks_probe)
                except Exception:
                    max_h_probe2 = int(ticks_probe)
                try:
                    max_s_probe2 = int(max(int(x) for x in probe_death_ticks_s)) if probe_death_ticks_s else int(ticks_probe)
                except Exception:
                    max_s_probe2 = int(ticks_probe)

                group_dead_found = bool(int(max_h_probe2) < int(ticks_probe) or int(max_s_probe2) < int(ticks_probe))
                if group_dead_found:
                    end_tick = int(min(int(max_h_probe2), int(max_s_probe2)))
                    ticks_used = int(max(int(requested_ticks), int(end_tick) + 1))
                    ticks_used = int(min(int(ticks_used), int(ticks_probe)))
                else:
                    ticks_used = int(min(int(ticks_cap), int(ticks_probe)))

                # Rebind ticks_i to the actual run length for the rest of the handler.
                ticks_i = int(ticks_used)

                rep_series_healthy: list[Dict[str, list[float]]] = []
                rep_series_sick: list[Dict[str, list[float]]] = []

                death_ticks_healthy: list[int] = [int(ticks_i) for _ in range(int(reps_i))]
                death_ticks_sick: list[int] = [int(ticks_i) for _ in range(int(reps_i))]
                death_meas_healthy: list[str] = ["" for _ in range(int(reps_i))]
                death_meas_sick: list[str] = ["" for _ in range(int(reps_i))]

                def _run_one(ri: int) -> tuple[int, Dict[str, list[float]], Dict[str, list[float]], int, str, int, str]:
                    seed0_h = int(seed_i) + (0 * 1000003) + (int(ri) * 97)
                    seed0_s = int(seed_i) + (1 * 1000003) + (int(ri) * 97)
                    sh, dt_h, dm_h = _run_in_vivo_measurement_series_until_death(
                        healthy, ticks=ticks_i, seed0=seed0_h, death_names=death_names
                    )
                    ss, dt_s, dm_s = _run_in_vivo_measurement_series_until_death(
                        sick2, ticks=ticks_i, seed0=seed0_s, death_names=death_names
                    )
                    return int(ri), sh, ss, int(dt_h), str(dm_h), int(dt_s), str(dm_s)

                if int(reps_i) <= 1 or int(workers_i) <= 1:
                    for ri in range(reps_i):
                        _, sh, ss, dt_h, dm_h, dt_s, dm_s = _run_one(int(ri))
                        rep_series_healthy.append(sh)
                        rep_series_sick.append(ss)
                        if 0 <= int(ri) < int(reps_i):
                            death_ticks_healthy[int(ri)] = int(dt_h)
                            death_ticks_sick[int(ri)] = int(dt_s)
                            death_meas_healthy[int(ri)] = str(dm_h)
                            death_meas_sick[int(ri)] = str(dm_s)
                elif worker_mode == "process":
                    ctx = mp.get_context("spawn")
                    out_h: list[Optional[Dict[str, list[float]]]] = [None for _ in range(int(reps_i))]
                    out_s: list[Optional[Dict[str, list[float]]]] = [None for _ in range(int(reps_i))]
                    out_dt_h: list[int] = [int(ticks_i) for _ in range(int(reps_i))]
                    out_dt_s: list[int] = [int(ticks_i) for _ in range(int(reps_i))]
                    out_dm_h: list[str] = ["" for _ in range(int(reps_i))]
                    out_dm_s: list[str] = ["" for _ in range(int(reps_i))]

                    cpu_req = max(1, min(int(workers_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=6.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(workers_i),
                            mp_context=ctx,
                            initializer=_invivo_worker_init_v2,
                            initargs=(healthy, sick2, death_names),
                        ) as ex:
                            pending: set[concurrent.futures.Future] = set()
                            it = iter(range(reps_i))

                            def _submit_one() -> None:
                                try:
                                    ri0 = next(it)
                                except StopIteration:
                                    return
                                pending.add(ex.submit(_invivo_worker_eval, int(ri0), int(ticks_i), int(seed_i)))

                            for _ in range(min(int(workers_i), int(reps_i))):
                                _submit_one()

                            while pending:
                                done, pending = concurrent.futures.wait(
                                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for fut in done:
                                    ri, sh, ss, dt_h, dm_h, dt_s, dm_s = fut.result()
                                    if 0 <= int(ri) < int(reps_i):
                                        out_h[int(ri)] = sh
                                        out_s[int(ri)] = ss
                                        out_dt_h[int(ri)] = int(dt_h)
                                        out_dt_s[int(ri)] = int(dt_s)
                                        out_dm_h[int(ri)] = str(dm_h)
                                        out_dm_s[int(ri)] = str(dm_s)
                                    _submit_one()
                    finally:
                        _cm.__exit__(None, None, None)

                    rep_series_healthy = [x if isinstance(x, dict) else {} for x in out_h]
                    rep_series_sick = [x if isinstance(x, dict) else {} for x in out_s]
                    death_ticks_healthy = [int(x) for x in out_dt_h]
                    death_ticks_sick = [int(x) for x in out_dt_s]
                    death_meas_healthy = [str(x) for x in out_dm_h]
                    death_meas_sick = [str(x) for x in out_dm_s]
                else:
                    out_h2: list[Optional[Dict[str, list[float]]]] = [None for _ in range(int(reps_i))]
                    out_s2: list[Optional[Dict[str, list[float]]]] = [None for _ in range(int(reps_i))]
                    out_dt_h2: list[int] = [int(ticks_i) for _ in range(int(reps_i))]
                    out_dt_s2: list[int] = [int(ticks_i) for _ in range(int(reps_i))]
                    out_dm_h2: list[str] = ["" for _ in range(int(reps_i))]
                    out_dm_s2: list[str] = ["" for _ in range(int(reps_i))]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers_i)) as ex:
                        pending: set[concurrent.futures.Future] = set()
                        it = iter(range(reps_i))

                        def _submit_one() -> None:
                            try:
                                ri0 = next(it)
                            except StopIteration:
                                return
                            pending.add(ex.submit(_run_one, int(ri0)))

                        for _ in range(min(int(workers_i), int(reps_i))):
                            _submit_one()

                        while pending:
                            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                            for fut in done:
                                ri, sh, ss, dt_h, dm_h, dt_s, dm_s = fut.result()
                                if 0 <= int(ri) < int(reps_i):
                                    out_h2[int(ri)] = sh
                                    out_s2[int(ri)] = ss
                                    out_dt_h2[int(ri)] = int(dt_h)
                                    out_dt_s2[int(ri)] = int(dt_s)
                                    out_dm_h2[int(ri)] = str(dm_h)
                                    out_dm_s2[int(ri)] = str(dm_s)
                                _submit_one()

                    rep_series_healthy = [x if isinstance(x, dict) else {} for x in out_h2]
                    rep_series_sick = [x if isinstance(x, dict) else {} for x in out_s2]
                    death_ticks_healthy = [int(x) for x in out_dt_h2]
                    death_ticks_sick = [int(x) for x in out_dt_s2]
                    death_meas_healthy = [str(x) for x in out_dm_h2]
                    death_meas_sick = [str(x) for x in out_dm_s2]

                mean_healthy, mean_healthy_n = _mean_measurement_series_ragged(rep_series_healthy, ticks=ticks_i, names=names)
                mean_sick, mean_sick_n = _mean_measurement_series_ragged(rep_series_sick, ticks=ticks_i, names=names)

                series_reps: Optional[Dict[str, Any]] = None
                if bool(include_replicates):
                    reps_h = [_pad_measurement_series_to_ticks(s, ticks=int(ticks_i), names=names) for s in rep_series_healthy]
                    reps_s = [_pad_measurement_series_to_ticks(s, ticks=int(ticks_i), names=names) for s in rep_series_sick]
                    series_reps = {"healthy": reps_h, "sick": reps_s}

                alive_n_healthy = _alive_n_from_death_ticks(death_ticks_healthy, ticks=int(ticks_i))
                alive_n_sick = _alive_n_from_death_ticks(death_ticks_sick, ticks=int(ticks_i))

                cure = _invivo_cure_score(mean_healthy, mean_sick, names=names, ticks=ticks_i)

                warnings: list[Dict[str, Any]] = []
                if int(ticks_i) > int(requested_ticks):
                    warnings.append(
                        {
                            "kind": "auto_extended_ticks",
                            "message": "The run was extended beyond requested ticks to reach group extinction (all dead in at least one group), subject to a safety cap.",
                            "details": {
                                "requested_ticks": int(requested_ticks),
                                "ticks_used": int(ticks_i),
                                "ticks_cap": int(ticks_cap),
                                "ticks_probe": int(ticks_probe),
                            },
                        }
                    )
                if not group_dead_found:
                    warnings.append(
                        {
                            "kind": "group_extinction_not_reached",
                            "message": "Neither group reached extinction (all dead) within the safety cap.",
                            "details": {
                                "requested_ticks": int(requested_ticks),
                                "ticks_used": int(ticks_i),
                                "ticks_cap": int(ticks_cap),
                            },
                        }
                    )
                try:
                    min_h = int(min(int(x) for x in death_ticks_healthy)) if death_ticks_healthy else int(ticks_i)
                except Exception:
                    min_h = int(ticks_i)
                try:
                    min_s = int(min(int(x) for x in death_ticks_sick)) if death_ticks_sick else int(ticks_i)
                except Exception:
                    min_s = int(ticks_i)
                if int(min_h) < int(ticks_i) or int(min_s) < int(ticks_i):
                    warnings.append(
                        {
                            "kind": "series_truncated_by_death",
                            "message": "One or more replicates died before the requested tick count; series values after death are omitted (null).",
                            "details": {
                                "requested_ticks": int(ticks_i),
                                "healthy_min_death_tick": int(min_h),
                                "sick_min_death_tick": int(min_s),
                            },
                        }
                    )

                samples_run = int(2 * reps_i)
                charge = _game_compute_charge(
                    "in_vivo_trial",
                    samples_run,
                    meta={
                        "experiment": "in_vivo_trial_v1",
                        "ticks": int(ticks_i),
                        "requested_ticks": int(requested_ticks),
                        "replicates": int(reps_i),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": "in_vivo_trial_v1",
                        "ticks": int(ticks_i),
                        "requested_ticks": int(requested_ticks),
                        "ticks_cap": int(ticks_cap),
                        "replicates": int(reps_i),
                        "workers": int(workers_i),
                        "worker_mode": str(worker_mode),
                        "measurements": names,
                        "series": {
                            "healthy": mean_healthy,
                            "sick": mean_sick,
                        },
                        "series_replicates": series_reps,
                        "series_n": {
                            "healthy": mean_healthy_n,
                            "sick": mean_sick_n,
                        },
                        "death": {
                            "healthy": {
                                "death_ticks": [int(x) for x in death_ticks_healthy],
                                "death_measurements": [str(x) for x in death_meas_healthy],
                                "alive_n": alive_n_healthy,
                            },
                            "sick": {
                                "death_ticks": [int(x) for x in death_ticks_sick],
                                "death_measurements": [str(x) for x in death_meas_sick],
                                "alive_n": alive_n_sick,
                            },
                            "death_names": list(death_names),
                        },
                        "warnings": warnings,
                        "cure": cure,
                        "game": game,
                    },
                )
                return

            if self.path == "/api/experiments/in_vivo_screen":
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = body.get("interventions")

                ticks = body.get("ticks", 200)
                replicates = body.get("replicates", 10)
                seed = body.get("seed", 1)
                workers_raw = body.get("workers", None)
                worker_mode_raw = body.get("worker_mode", "process")
                direction = body.get("direction", "up")
                dose = body.get("dose", 1)

                sick = body.get("sick")
                if not isinstance(sick, dict):
                    raise ValueError("missing sick payload")

                sick2 = _deepcopy_payload(sick)
                if isinstance(interventions, list) and interventions:
                    sick2["_tick_interventions"] = list(interventions)

                try:
                    ticks_i = max(1, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                try:
                    dose_f = float(dose)
                except Exception:
                    dose_f = 0.0
                if not np.isfinite(dose_f) or dose_f < 0.0:
                    dose_f = 0.0

                worker_mode = str(worker_mode_raw or "process").strip().lower()
                if worker_mode not in ("thread", "process"):
                    worker_mode = "process"

                try:
                    workers_i = int(workers_raw) if ("workers" in body) else 0
                except Exception:
                    workers_i = 0
                if workers_i <= 0:
                    workers_i = max(1, int(min(4, os.cpu_count() or 1)))

                death_names = _death_measurement_names_from_payload(sick2)

                prot_layers = _protein_layer_names_from_payload(sick2)
                if not prot_layers:
                    raise ValueError("no protein_* float32 layers found")

                def _lifespan_stats(dts: list[int]) -> Dict[str, Any]:
                    arr = np.asarray([int(x) for x in dts], dtype=np.float64) if dts else np.asarray([], dtype=np.float64)
                    if arr.size:
                        med = float(np.median(arr))
                        mean = float(np.mean(arr))
                        try:
                            p25 = float(np.quantile(arr, 0.25))
                        except Exception:
                            p25 = float(med)
                        try:
                            p75 = float(np.quantile(arr, 0.75))
                        except Exception:
                            p75 = float(med)
                        mn = float(np.min(arr))
                        mx = float(np.max(arr))
                    else:
                        med = float(ticks_i)
                        mean = float(ticks_i)
                        p25 = float(ticks_i)
                        p75 = float(ticks_i)
                        mn = float(ticks_i)
                        mx = float(ticks_i)
                    deaths = int(sum(1 for dt in dts if int(dt) < int(ticks_i)))
                    return {
                        "n": int(len(dts)),
                        "ticks": int(ticks_i),
                        "median_lifespan_tick": float(med),
                        "mean_lifespan_tick": float(mean),
                        "p25_lifespan_tick": float(p25),
                        "p75_lifespan_tick": float(p75),
                        "min_lifespan_tick": float(mn),
                        "max_lifespan_tick": float(mx),
                        "deaths": int(deaths),
                        "survivors": int(len(dts) - deaths),
                    }

                # Baseline (sick + interventions)
                base_death_ticks: list[int] = []
                for ri in range(int(reps_i)):
                    seed0 = int(seed_i) + (0 * 1000003) + (int(ri) * 97)
                    r0 = _run_lifespan_death_tick(sick2, ticks=int(ticks_i), seed0=int(seed0), death_names=death_names)
                    try:
                        dt0 = int(r0.get("death_tick"))
                    except Exception:
                        dt0 = int(ticks_i)
                    base_death_ticks.append(int(dt0))
                base_stats = _lifespan_stats(base_death_ticks)

                workers_i = max(1, min(int(workers_i), int(len(prot_layers)), 32))

                def _eval_layer(li: int, nm: str) -> Dict[str, Any]:
                    p0 = _deepcopy_payload(sick2)
                    tiv0 = p0.get("_tick_interventions")
                    tiv = list(tiv0) if isinstance(tiv0, list) else []
                    tiv.append({"layer": str(nm), "direction": direction, "dose": float(dose_f)})
                    p0["_tick_interventions"] = tiv
                    dts: list[int] = []
                    for ri in range(int(reps_i)):
                        seed0 = int(seed_i) + (int(li + 1) * 1000003) + (int(ri) * 97)
                        r = _run_lifespan_death_tick(p0, ticks=int(ticks_i), seed0=int(seed0), death_names=death_names)
                        try:
                            dt = int(r.get("death_tick"))
                        except Exception:
                            dt = int(ticks_i)
                        dts.append(int(dt))
                    stats = _lifespan_stats(dts)
                    return {
                        "layer": str(nm),
                        "layer_index": int(li),
                        **stats,
                    }

                results_out: list[Optional[Dict[str, Any]]] = [None for _ in range(int(len(prot_layers)))]
                if int(len(prot_layers)) <= 1 or int(workers_i) <= 1:
                    for li, nm in enumerate(prot_layers):
                        results_out[int(li)] = _eval_layer(int(li), str(nm))
                elif worker_mode == "process":
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(workers_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=6.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(workers_i),
                            mp_context=ctx,
                            initializer=_invivo_screen_worker_init,
                            initargs=(sick2, death_names, int(ticks_i), int(seed_i), int(reps_i), str(direction), float(dose_f)),
                        ) as ex:
                            pending: set[concurrent.futures.Future] = set()
                            it = iter(list(enumerate(prot_layers)))

                            def _submit_one() -> None:
                                try:
                                    li0, nm0 = next(it)
                                except StopIteration:
                                    return
                                pending.add(ex.submit(_invivo_screen_worker_eval, int(li0), str(nm0)))

                            for _ in range(min(int(workers_i), int(len(prot_layers)))):
                                _submit_one()

                            while pending:
                                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                                for fut in done:
                                    r = fut.result()
                                    li = int(r.get("layer_index") or 0)
                                    if 0 <= li < int(len(results_out)):
                                        results_out[li] = r
                                    _submit_one()
                    finally:
                        _cm.__exit__(None, None, None)
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers_i)) as ex:
                        pending: set[concurrent.futures.Future] = set()
                        it = iter(list(enumerate(prot_layers)))

                        def _submit_one() -> None:
                            try:
                                li0, nm0 = next(it)
                            except StopIteration:
                                return
                            pending.add(ex.submit(_eval_layer, int(li0), str(nm0)))

                        for _ in range(min(int(workers_i), int(len(prot_layers)))):
                            _submit_one()

                        while pending:
                            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                            for fut in done:
                                r = fut.result()
                                li = int(r.get("layer_index") or 0)
                                if 0 <= li < int(len(results_out)):
                                    results_out[li] = r
                                _submit_one()

                results: list[Dict[str, Any]] = [r for r in results_out if isinstance(r, dict)]
                results.sort(key=lambda d: float(d.get("median_lifespan_tick") or 0.0), reverse=True)

                samples_run = int(int(reps_i) * int(len(prot_layers) + 1))
                charge = _game_compute_charge(
                    "in_vivo_trial",
                    samples_run,
                    meta={
                        "experiment": "in_vivo_screen_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "protein_layers": int(len(prot_layers)),
                        "direction": str(direction),
                        "dose": float(dose_f),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": "in_vivo_screen_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "workers": int(workers_i),
                        "worker_mode": str(worker_mode),
                        "direction": str(direction),
                        "dose": float(dose_f),
                        "protein_layers": int(len(prot_layers)),
                        "baseline": {
                            **base_stats,
                        },
                        "death_names": list(death_names),
                        "results": results,
                        "game": game,
                    },
                )
                return

            if self.path == "/api/experiments/bulk_omics":
                body = self._read_json_body()
                player_id = body.get("player_id")
                interventions = body.get("interventions")

                ticks = body.get("ticks", 100)
                replicates = body.get("replicates", 1)
                seed = body.get("seed", 1)
                omics_set_req = body.get("omics_set")
                omics_set_name, features = _load_bulk_omics_set(str(omics_set_req or ""))

                conds = body.get("conditions")
                cond_list: list[Dict[str, Any]] = []
                if isinstance(conds, list):
                    for c in conds:
                        if not isinstance(c, dict):
                            continue
                        nm = str(c.get("name") or "").strip()
                        payload = c.get("payload")
                        if not nm or not isinstance(payload, dict):
                            continue
                        cond_list.append({"name": nm, "payload": payload})

                if cond_list:
                    for ent in cond_list:
                        try:
                            nm = str(ent.get("name") or "").strip().lower()
                        except Exception:
                            nm = ""
                        if nm != "sick":
                            continue
                        p0 = ent.get("payload")
                        if not isinstance(p0, dict):
                            continue
                        p1 = _deepcopy_payload(p0)
                        if isinstance(interventions, list) and interventions:
                            p1["_tick_interventions"] = list(interventions)
                        ent["payload"] = p1

                if not cond_list:
                    healthy = body.get("healthy")
                    sick = body.get("sick")
                    if isinstance(healthy, dict):
                        cond_list.append({"name": "healthy", "payload": healthy})
                    if isinstance(sick, dict):
                        sick2 = _deepcopy_payload(sick)
                        if isinstance(interventions, list) and interventions:
                            sick2["_tick_interventions"] = list(interventions)
                        cond_list.append({"name": "sick", "payload": sick2})

                if not cond_list:
                    raise ValueError("missing conditions (provide conditions[] or healthy/sick payloads)")

                if not features:
                    raise ValueError("no features selected")

                kind = _bulk_omics_kind_from_set_name(str(omics_set_name or ""))
                masked_features = _bulk_omics_mask_feature_headers(features, kind=str(kind))

                z_target_arr = None
                z_target_in = body.get("z_target")
                if isinstance(z_target_in, list):
                    if len(z_target_in) != int(len(features)):
                        raise ValueError("z_target must have length == #features")
                    tmp: list[float] = []
                    for v in z_target_in:
                        try:
                            f = float(v)
                        except Exception:
                            f = float("nan")
                        tmp.append(f)
                    z_target_arr = np.asarray(tmp, dtype=np.float64)

                try:
                    ticks_i = max(0, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                syn_rng = np.random.default_rng(3)

                run_id0 = uuid.uuid4().hex
                run_tag = str(run_id0)[:12]

                # Death-aware preflight: bulk assays are snapshots at `ticks`.
                replicate_deaths: list[Dict[str, Any]] = []

                assay = str(kind or "")
                if assay == "bulk_rnaseq":
                    assay = "Bulk transcriptomics"
                elif assay == "bulk_proteomics":
                    assay = "Bulk proteomics"
                elif assay == "bulk_metabolomics":
                    assay = "Bulk metabolomics"

                meta_header = [
                    "sample_id",
                    "assay",
                    "condition",
                    "replicate",
                    "study_ran_for_days",
                ]
                meta_rows: list[list[Any]] = []
                mat_header = ["sample_id", *masked_features]
                truth_mat_rows: list[list[Any]] = []

                run_ids: list[str] = []
                run_T: list[list[float]] = []

                out_runs: list[Dict[str, Any]] = []
                for ci, c in enumerate(cond_list):
                    nm = str(c.get("name") or "").strip()
                    payload = c.get("payload")
                    if not nm or not isinstance(payload, dict):
                        continue

                    for ri in range(reps_i):
                        seed0 = int(seed_i) + (int(ci) * 1000003) + (int(ri) * 97)
                        pf = _preflight_death_before_ticks(payload, ticks=int(ticks_i), seed0=int(seed0))
                        if isinstance(pf, dict):
                            replicate_deaths.append(
                                {
                                    "condition": str(nm),
                                    "replicate": int(ri),
                                    "seed": int(seed0),
                                    "requested_ticks": int(ticks_i),
                                    "death_tick": int(pf.get("death_tick") or 0),
                                    "death_measurement": str(pf.get("death_measurement") or ""),
                                    "death_names": pf.get("death_names") if isinstance(pf.get("death_names"), list) else [],
                                }
                            )
                            continue
                        p = _run_payload_ticks(payload, ticks=ticks_i, seed0=seed0)
                        H = int(p.get("H") or 0)
                        W = int(p.get("W") or 0)
                        if H <= 0 or W <= 0:
                            raise ValueError("payload invalid H/W")
                        expected_len = int(H * W)
                        layers = _layers_dict_from_payload_data(p, expected_len=expected_len)
                        if not layers:
                            raise ValueError("payload has no float32 layers")

                        sample_id = f"{nm}_c{int(ci)}_r{int(ri)}"
                        vv: list[float] = []
                        for ln in features:
                            arr = layers.get(ln)
                            if arr is None:
                                vv.append(0.0)
                                continue
                            try:
                                s = float(np.asarray(arr, dtype=np.float64).reshape(-1).sum())
                            except Exception:
                                s = 0.0
                            if not np.isfinite(s) or s < 0.0:
                                s = 0.0
                            vv.append(float(s))

                        meta_rows.append([
                            sample_id,
                            str(assay),
                            nm,
                            int(ri),
                            int(ticks_i),
                        ])
                        truth_mat_rows.append([sample_id, *vv])
                        run_ids.append(sample_id)
                        run_T.append(vv)

                        out_runs.append(
                            {
                                "condition": nm,
                                "replicate": int(ri),
                                "seed": int(seed0),
                                "ticks": int(ticks_i),
                                "age_days": int(ticks_i),
                                "H": int(H),
                                "W": int(W),
                            }
                        )

                if not run_ids:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": f"all replicates died before requested ticks={int(ticks_i)}",
                            "error_kind": "all_replicates_died",
                            "experiment": "bulk_omics_v1",
                            "details": {
                                "requested_ticks": int(ticks_i),
                                "replicates_requested": int(reps_i),
                                "replicate_deaths": list(replicate_deaths),
                            },
                        },
                    )
                    return

                noisy_mat_rows: list[list[Any]] = []
                if run_ids and run_T:
                    T_arr = np.asarray(run_T, dtype=np.float64)
                    Y = _bulk_synthetic_v1_noisy_counts(T_arr, rng=syn_rng, z_target=z_target_arr)
                    for ii, sid in enumerate(run_ids):
                        noisy_mat_rows.append([sid, *[int(x) for x in np.asarray(Y[ii], dtype=np.int64).tolist()]])

                matrix_truth_csv = _csv_from_rows(mat_header, truth_mat_rows)
                matrix_noisy_csv = _csv_from_rows(mat_header, noisy_mat_rows)
                metadata_csv = _csv_from_rows(meta_header, meta_rows)

                files_text: Dict[str, Any] = {}
                files_text[f"metadata_{run_tag}.csv"] = str(metadata_csv)
                for row in noisy_mat_rows:
                    if not isinstance(row, list) or not row:
                        continue
                    sid = str(row[0] or "")
                    safe_sid = _omics_safe_label(sid, default="sample")
                    files_text[f"samples/{safe_sid}.csv"] = _csv_from_rows(mat_header, [row])

                manifest = {
                    "experiment": "bulk_omics_v1",
                    "kind": str(kind),
                    "player_id": _sanitize_player_id(player_id),
                    "ticks": int(ticks_i),
                    "age_days": int(ticks_i),
                    "replicates": int(reps_i),
                    "replicates_completed": int(len(out_runs)),
                    "replicate_deaths": list(replicate_deaths),
                    "omics_set": str(omics_set_name or ""),
                    "features": list(masked_features),
                    "runs": out_runs,
                }
                omics_saved = _OMICS.create_run({**manifest, "run_id": str(run_id0)}, files_text)

                samples_run = int(len(out_runs))
                charge = _game_compute_charge(
                    kind,
                    samples_run,
                    meta={
                        "experiment": "bulk_omics_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "omics_set": str(omics_set_name or ""),
                    },
                )
                game = _game_apply_charge(player_id, charge)

                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": "bulk_omics_v1",
                        "ticks": int(ticks_i),
                        "age_days": int(ticks_i),
                        "replicates": int(reps_i),
                        "replicates_completed": int(len(out_runs)),
                        "replicate_deaths": list(replicate_deaths),
                        "omics_set": str(omics_set_name or ""),
                        "genes": masked_features,
                        "run_id": str(omics_saved.get("run_id") or ""),
                        "files": omics_saved.get("files"),
                        "omics_inventory": {
                            "inventory_url": f"/api/omics/inventory?player_id={_sanitize_player_id(player_id)}",
                            "llm_message": str(_OMICS.inventory(player_id).get("llm_message") or ""),
                        },
                        "noise": {
                            "model": "bulk_synthetic_v1",
                            "sigma_sample": 0.35,
                            "theta": 50.0,
                            "ambient_frac": 0.001,
                            "ambient_sigma_sample": 0.25,
                            "rng_seed": 3,
                        },
                        "matrix_csv": matrix_noisy_csv,
                        "matrix_truth_csv": matrix_truth_csv,
                        "matrix_noisy_csv": matrix_noisy_csv,
                        "metadata_csv": metadata_csv,
                        "runs": out_runs,
                        "game": game,
                    },
                )
                return

            if self.path == "/api/doc/get":
                self._send_json(200, {"ok": True, "payload_text": _DOC.get_payload_text(), **_DOC.status()})
                return

            if self.path == "/api/doc/autosave":
                body = self._read_json_body()
                payload_text = body.get("payload_text")
                path = body.get("path")
                if not isinstance(payload_text, str):
                    raise ValueError("payload_text must be string")
                out = _DOC.set_payload_from_text(payload_text, path=str(path or "") if isinstance(path, str) else "")
                self._send_json(200, out)
                return

            if self.path == "/api/doc/open":
                body = self._read_json_body()
                name = body.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("missing name")
                out = _DOC.open_doc(name)
                self._send_json(200, out)
                return

            if self.path == "/api/doc/save":
                body = self._read_json_body()
                name = body.get("name")
                if name is not None and (not isinstance(name, str) or not name.strip()):
                    raise ValueError("bad name")
                out = _DOC.save_doc(name if isinstance(name, str) else None)
                self._send_json(200, out)
                return

            if self.path == "/api/doc/delete":
                body = self._read_json_body()
                name = body.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("missing name")
                out = _DOC.delete_doc(name)
                self._send_json(200, out)
                return

            if self.path == "/api/doc/recover":
                out = _DOC.recover_autosave()
                self._send_json(200, out)
                return

            if self.path == "/api/evolution/start":
                body = self._read_json_body()
                payload = body.get("payload")
                cfg = body.get("config")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                if not isinstance(cfg, dict):
                    raise ValueError("missing config")
                job_id = _evo_start_runner(payload, cfg)
                self._send_json(200, {"ok": True, "job_id": job_id})
                return

            if self.path == "/api/evolution/stop":
                _evo_stop_runner()
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/evolution/status":
                self._send_json(200, _evo_status_from_disk())
                return

            if self.path == "/api/evolution/candidate":
                body = self._read_json_body()
                cid = body.get("id")
                if not isinstance(cid, str) or not cid:
                    raise ValueError("missing id")
                base = _safe_read_json(_evo_base_payload_path())
                cfg = _safe_read_json(_evo_cfg_path())
                cands = _safe_read_json(_evo_candidates_path())
                if not isinstance(base, dict) or not isinstance(cfg, dict) or not isinstance(cands, dict):
                    raise ValueError("candidate store missing")
                c = cands.get(cid)
                if not isinstance(c, dict):
                    raise ValueError("unknown candidate")

                # Auto-switch persists a stable "__winner__" snapshot that may already be
                # reflected in base_payload.json. In that case, return the base payload directly
                # so the latest winner can always be fetched even if candidates were cleared.
                if bool(c.get("payload_inline")) and isinstance(c.get("payload"), dict):
                    payload = c.get("payload")
                elif bool(c.get("payload_is_base")):
                    payload = base
                else:
                    payload = _evo_build_candidate_payload(base, cfg, c)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "id": cid,
                        "fitness": c.get("fitness"),
                        "metrics": c.get("metrics"),
                        "genome": c.get("genome"),
                        "payload": payload,
                    },
                )
                return

            if self.path == "/api/evolution/fitness-config":
                body = self._read_json_body()
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                
                measurements_cfg = payload.get("measurements_config")
                available_measurements = []
                if isinstance(measurements_cfg, dict) and int(measurements_cfg.get("version") or 0) == 3:
                    meas_list = measurements_cfg.get("measurements")
                    if isinstance(meas_list, list):
                        for m in meas_list:
                            if isinstance(m, dict):
                                name = str(m.get("name") or "").strip()
                                expr = str(m.get("expr") or "").strip()
                                if name and expr:
                                    available_measurements.append({
                                        "name": name,
                                        "expr": expr,
                                    })
                
                self._send_json(200, {
                    "ok": True,
                    "measurements": available_measurements,
                    "events": ["divisions", "starvation_deaths", "damage_deaths"],
                    "distribution_methods": [
                        {"value": "entropy", "label": "Entropy (normalized Shannon)"},
                        {"value": "cv", "label": "Coefficient of Variation"},
                        {"value": "spread", "label": "Spread (non-zero ticks)"},
                    ],
                })
                return

            if self.path == "/api/pathway/topology":
                body = self._read_json_body()
                step = body.get("step")
                if not isinstance(step, dict):
                    raise ValueError("missing step")
                out = _pathway_compute_topology(step)
                self._send_json(200, out)
                return

            if self.path == "/api/runtime/reset":
                body = self._read_json_body()
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                out = _RT.reset(payload)
                self._send_json(200, out)
                return

            if self.path == "/api/runtime/frame":
                body = self._read_json_body()
                layers = body.get("layers")
                if layers is not None and not isinstance(layers, list):
                    raise ValueError("layers must be a list")
                out = _RT.frame(layers)
                self._send_json(200, out)
                return

            if self.path == "/api/runtime/step":
                body = self._read_json_body()
                layers = body.get("layers")
                if layers is not None and not isinstance(layers, list):
                    raise ValueError("layers must be a list")
                out = _RT.step(layers)
                self._send_json(200, out)
                return

            if self.path == "/api/runtime/export":
                out = _RT.export()
                self._send_json(200, out)
                return

            if self.path == "/api/profile/run":
                body = self._read_json_body()
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")
                ticks = body.get("ticks", 50)
                warmup = body.get("warmup", 5)
                repeats = body.get("repeats", 1)
                do_estimate = bool(body.get("estimate", True))
                do_breakdown = bool(body.get("breakdown", True))
                out = _profile_run_payload(
                    payload,
                    ticks=int(ticks) if str(ticks).strip() != "" else 50,
                    warmup=int(warmup) if str(warmup).strip() != "" else 0,
                    repeats=int(repeats) if str(repeats).strip() != "" else 1,
                    do_estimate=do_estimate,
                    do_breakdown=do_breakdown,
                )
                self._send_json(200, out)
                return

            if self.path == "/api/lifespan/run":
                body = self._read_json_body()
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("missing payload")

                ticks = body.get("ticks", 200)
                replicates = body.get("replicates", 10)
                seed = body.get("seed", 1)
                try:
                    ticks_i = max(1, int(ticks))
                except Exception as e:
                    raise ValueError(f"ticks must be an int: {e}") from e
                try:
                    reps_i = max(1, int(replicates))
                except Exception as e:
                    raise ValueError(f"replicates must be an int: {e}") from e
                try:
                    seed_i = int(seed)
                except Exception as e:
                    raise ValueError(f"seed must be an int: {e}") from e

                include_series = bool(body.get("include_series") or False)

                worker_mode_raw = body.get("worker_mode", "thread")
                worker_mode = str(worker_mode_raw or "thread").strip().lower()
                if worker_mode not in ("thread", "process"):
                    worker_mode = "thread"

                workers_missing = "workers" not in body
                workers_raw = body.get("workers", None)
                if workers_missing:
                    workers_i = 1
                else:
                    try:
                        workers_i = int(workers_raw)
                    except Exception:
                        workers_i = 0

                death_names = _death_measurement_names_from_payload(payload)
                if not death_names:
                    raise ValueError("no death measurements found (measurement name must contain 'death')")

                series_names: list[str] = []
                if include_series:
                    all_names = _measurement_names_from_payload(payload)
                    dn = set(death_names)
                    series_names = [nm for nm in all_names if nm not in dn]

                if workers_i <= 0:
                    workers_i = max(1, int(min(4, os.cpu_count() or 1)))
                workers_i = max(1, min(int(workers_i), int(reps_i), 32))

                reps_out: list[Dict[str, Any]] = [{} for _ in range(reps_i)]
                death_ticks: list[int] = [int(ticks_i) for _ in range(reps_i)]
                reps_series_out: list[Optional[Dict[str, list[Optional[float]]]]] = [None for _ in range(reps_i)]

                series_sum: Dict[str, list[float]] = {nm: [0.0] * int(ticks_i) for nm in series_names}
                series_n: Dict[str, list[int]] = {nm: [0] * int(ticks_i) for nm in series_names}

                def _accum_series(series: Optional[Dict[str, list[Optional[float]]]]) -> None:
                    if not series_names or not isinstance(series, dict):
                        return
                    for nm in series_names:
                        arr = series.get(nm)
                        if not isinstance(arr, list) or not arr:
                            continue
                        ss = series_sum.get(nm)
                        nn = series_n.get(nm)
                        if not isinstance(ss, list) or not isinstance(nn, list):
                            continue
                        m = min(int(ticks_i), len(arr), len(ss), len(nn))
                        for i in range(m):
                            v0 = arr[i]
                            if v0 is None:
                                continue
                            try:
                                v = float(v0)
                            except Exception:
                                continue
                            if not np.isfinite(v):
                                continue
                            ss[i] += float(v)
                            nn[i] += 1

                def _run_one(ri: int) -> tuple[int, Dict[str, Any], Optional[Dict[str, list[Optional[float]]]]]:
                    seed0 = int(seed_i) + (int(ri) * 97)
                    r, series = _run_lifespan_rep(
                        payload,
                        ticks=ticks_i,
                        seed0=seed0,
                        death_names=death_names,
                        series_names=series_names,
                    )
                    r["seed0"] = int(seed0)
                    return int(ri), r, series

                if int(reps_i) <= 1 or int(workers_i) <= 1:
                    for ri in range(reps_i):
                        _, r, series = _run_one(int(ri))
                        reps_out[int(ri)] = r
                        if 0 <= int(ri) < int(reps_i):
                            reps_series_out[int(ri)] = series
                        _accum_series(series)
                        try:
                            death_ticks[int(ri)] = int(r.get("death_tick"))
                        except Exception:
                            death_ticks[int(ri)] = int(ticks_i)
                elif worker_mode == "process":
                    ctx = mp.get_context("spawn")
                    cpu_req = max(1, min(int(workers_i), int(_RESOURCE_SCHED.cpu_total)))
                    _cm = _RESOURCE_SCHED.acquire(cpu=int(cpu_req), mem_gb=4.0)
                    _cm.__enter__()
                    try:
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=int(workers_i),
                            mp_context=ctx,
                            initializer=_lifespan_worker_init,
                            initargs=(payload, death_names, series_names),
                        ) as ex:
                            pending: set[concurrent.futures.Future] = set()
                            it = iter(range(reps_i))

                            def _submit_one() -> None:
                                try:
                                    ri0 = next(it)
                                except StopIteration:
                                    return
                                pending.add(ex.submit(_lifespan_worker_eval, int(ri0), int(ticks_i), int(seed_i)))

                            for _ in range(min(int(workers_i), int(reps_i))):
                                _submit_one()

                            while pending:
                                done, pending = concurrent.futures.wait(
                                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for fut in done:
                                    ri, r, series = fut.result()
                                    if 0 <= int(ri) < int(reps_i):
                                        reps_out[int(ri)] = r
                                        reps_series_out[int(ri)] = series
                                        _accum_series(series)
                                        try:
                                            death_ticks[int(ri)] = int(r.get("death_tick"))
                                        except Exception:
                                            death_ticks[int(ri)] = int(ticks_i)
                                    _submit_one()
                    finally:
                        _cm.__exit__(None, None, None)
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers_i)) as ex:
                        pending: set[concurrent.futures.Future] = set()
                        it = iter(range(reps_i))

                        def _submit_one() -> None:
                            try:
                                ri0 = next(it)
                            except StopIteration:
                                return
                            pending.add(ex.submit(_run_one, int(ri0)))

                        for _ in range(min(int(workers_i), int(reps_i))):
                            _submit_one()

                        while pending:
                            done, pending = concurrent.futures.wait(
                                pending, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for fut in done:
                                ri, r, series = fut.result()
                                if 0 <= int(ri) < int(reps_i):
                                    reps_out[int(ri)] = r
                                    reps_series_out[int(ri)] = series
                                    _accum_series(series)
                                    try:
                                        death_ticks[int(ri)] = int(r.get("death_tick"))
                                    except Exception:
                                        death_ticks[int(ri)] = int(ticks_i)
                                _submit_one()

                mean_series: Optional[Dict[str, Any]] = None
                if series_names:
                    mean: Dict[str, list[Optional[float]]] = {}
                    for nm in series_names:
                        ss = series_sum.get(nm)
                        nn = series_n.get(nm)
                        if not isinstance(ss, list) or not isinstance(nn, list) or len(ss) != int(ticks_i) or len(nn) != int(ticks_i):
                            continue
                        out_arr: list[Optional[float]] = []
                        for i in range(int(ticks_i)):
                            c = int(nn[i])
                            if c <= 0:
                                out_arr.append(None)
                            else:
                                out_arr.append(float(ss[i]) / float(c))
                        mean[nm] = out_arr
                    mean_series = {
                        "ticks": list(range(int(ticks_i))),
                        "names": series_names,
                        "mean": mean,
                        "replicates": reps_series_out,
                    }

                curve = _lifespan_survival_curve(death_ticks, ticks=ticks_i)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "experiment": "lifespan_v1",
                        "ticks": int(ticks_i),
                        "replicates": int(reps_i),
                        "seed": int(seed_i),
                        "workers": int(workers_i),
                        "worker_mode": str(worker_mode),
                        "death_measurements": death_names,
                        "measurements_series": mean_series,
                        "replicates_out": reps_out,
                        "curve": curve,
                    },
                )
                return

            self._send_json(404, {"ok": False, "error": "not found"})
        except TemporaryUnavailableError as e:
            err_id = uuid.uuid4().hex[:10]
            try:
                _LOG.warning(
                    "POST %s temporarily unavailable (error_id=%s provider=%s model=%s)",
                    str(self.path),
                    err_id,
                    str(getattr(e, "provider", "") or ""),
                    str(getattr(e, "model", "") or ""),
                )
            except Exception:
                pass

            self._send_json(
                503,
                {
                    "ok": False,
                    "error": "temporarily unavailable",
                    "error_id": err_id,
                    "provider": str(getattr(e, "provider", "") or ""),
                    "model": str(getattr(e, "model", "") or ""),
                },
            )
        except RateLimitError as e:
            err_id = uuid.uuid4().hex[:10]
            retry_after_s = None
            try:
                retry_after_s = float(e.retry_after_s) if e.retry_after_s is not None else None
            except Exception:
                retry_after_s = None

            extra_headers: Dict[str, str] = {}
            if retry_after_s is not None:
                try:
                    extra_headers["Retry-After"] = str(int(max(1.0, float(retry_after_s))))
                except Exception:
                    extra_headers = {}

            try:
                _LOG.warning(
                    "POST %s rate limited (error_id=%s provider=%s model=%s retry_after_s=%s)",
                    str(self.path),
                    err_id,
                    str(getattr(e, "provider", "") or ""),
                    str(getattr(e, "model", "") or ""),
                    str(retry_after_s),
                )
            except Exception:
                pass

            self._send_json(
                429,
                {
                    "ok": False,
                    "error": str(e),
                    "error_id": err_id,
                    "provider": str(getattr(e, "provider", "") or ""),
                    "model": str(getattr(e, "model", "") or ""),
                    "retry_after_s": retry_after_s,
                },
                extra_headers=extra_headers if extra_headers else None,
            )
        except ValueError as e:
            err_id = uuid.uuid4().hex[:10]
            try:
                _LOG.exception("POST %s failed (error_id=%s)", str(self.path), err_id)
            except Exception:
                pass
            self._send_json(400, {"ok": False, "error": str(e), "error_id": err_id})
        except Exception as e:
            err_id = uuid.uuid4().hex[:10]
            try:
                _LOG.exception("POST %s failed (error_id=%s)", str(self.path), err_id)
            except Exception:
                pass
            self._send_json(500, {"ok": False, "error": str(e), "error_id": err_id})
        finally:
            try:
                dt_ms = (time.time() - t0) * 1000.0
                _LOG.info("POST %s %.1fms", str(self.path), float(dt_ms))
            except Exception:
                pass


def main() -> int:
    _setup_logging()
    _install_exception_hooks()
    _ensure_dirs()
    if not _WEB_DIR.exists():
        try:
            _LOG.error("web editor dir not found: %s", str(_WEB_DIR))
        except Exception:
            pass
        return 2

    port = 8000
    try:
        if len(sys.argv) >= 2:
            port = int(sys.argv[1])
    except Exception:
        port = 8000

    # Avoid caching during dev.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("DT_RUNTIME_PORT", str(port))

    srv = ThreadingHTTPServer(("0.0.0.0", port), RuntimeHandler)
    try:
        srv.daemon_threads = True
    except Exception:
        pass
    try:
        _LOG.info("Runtime server: http://0.0.0.0:%s/", str(port))
        _LOG.info("Open Functions → Runtime")
    except Exception:
        pass
    try:
        _LOG.info("Runtime server starting on port %s", str(port))
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        srv.server_close()
        try:
            _LOG.info("Runtime server stopped")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
