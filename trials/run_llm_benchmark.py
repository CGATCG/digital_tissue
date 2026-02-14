import argparse
import csv
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
import platform
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from backend.env_keys import apply_keys_to_environ
except Exception:
    apply_keys_to_environ = None  # type: ignore


def _http_post_json(*, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
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
        raise RuntimeError(f"HTTP {status}: {txt[:2000]}") from e

    obj: Dict[str, Any] = {}
    try:
        parsed = json.loads(txt) if isinstance(txt, str) and txt.strip() else None
        obj = parsed if isinstance(parsed, dict) else {}
    except Exception:
        obj = {}
    return obj


_BENCH_CHALLENGE = "cancer"
_BENCH_PLAYER_ID = ""
_BENCH_PROVIDER = ""
_BENCH_MODEL = ""
_BENCH_PROMPT_TEXT = ""
_BENCH_PROMPT_FILE = ""
_BENCH_STEP: Optional[int] = None
_BENCH_MAX_STEPS: Optional[int] = None


def _llm_step_budget_enabled() -> bool:
    v = str(os.environ.get("DT_LLM_STEP_BUDGET") or "").strip().lower()
    if not v:
        return True
    return v in ("1", "true", "yes")


def _llm_step_budget_text(*, step: Optional[int], max_steps: Optional[int]) -> str:
    if not _llm_step_budget_enabled():
        return ""
    if step is None or max_steps is None:
        return ""
    try:
        cur = int(step) + 1
        tot = int(max_steps)
    except Exception:
        return ""
    if tot <= 0:
        return ""
    cur = max(1, min(cur, tot))
    remaining = max(0, tot - cur)
    return (
        f"STEP_BUDGET: This is step {cur} of {tot} ({remaining} remaining). "
        "Plan accordingly so you can finish within the remaining steps."
    )


def _llm_messages_with_step_budget(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    txt = _llm_step_budget_text(step=_BENCH_STEP, max_steps=_BENCH_MAX_STEPS)
    if not txt:
        return messages
    msgs: List[Dict[str, str]] = []
    for m in (messages or []):
        if isinstance(m, dict):
            msgs.append(dict(m))
    msgs.append({"role": "user", "content": str(txt)})
    return msgs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prompts_dir() -> Path:
    return _repo_root() / "assets" / "prompts"


def _read_prompt_text(prompt_file: Optional[str]) -> Tuple[str, str]:
    pf_in = str(prompt_file or "").strip()
    explicit = bool(pf_in)
    if not pf_in:
        pf_in = "default.txt"

    p = Path(pf_in).expanduser()
    if not p.is_absolute():
        p = _prompts_dir() / pf_in

    if not (p.exists() and p.is_file()):
        if explicit:
            raise FileNotFoundError(f"Prompt file not found: {pf_in}")
        return _PROMPT, ""

    with open(str(p), "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    label = pf_in if not Path(pf_in).is_absolute() else str(p)
    return str(txt).strip(), str(label)


_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"


_ANTHROPIC_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["call_api"]},
        "method": {"type": "string"},
        "path": {"type": "string"},
        "query": {"type": "object"},
        "body": {"type": "object"},
        "last_result_summary": {"type": "string"},
        "next_step_rationale": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": True,
}


_ANTHROPIC_INPUT_TPM_LIMIT = 450_000
_ANTHROPIC_TPM_WINDOW_S = 60.0
_ANTHROPIC_TPM_SAFETY_FRAC = 0.95
_ANTHROPIC_TPM_USAGE: List[Tuple[float, int]] = []
_ANTHROPIC_CACHE_PREFIX_SIGS: Set[str] = set()


def _approx_token_count_text(text: str) -> int:
    s = str(text or "")
    if not s:
        return 0
    return max(1, int(math.ceil(len(s) / 4.0)))


def _anthropic_estimate_input_tokens(*, system_blocks: List[Dict[str, Any]], messages: List[Dict[str, Any]]) -> int:
    total = 0
    for b in system_blocks or []:
        if not isinstance(b, dict):
            continue
        if str(b.get("type") or "").strip().lower() != "text":
            continue
        total += _approx_token_count_text(str(b.get("text") or "")) + 8

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        total += 10
        total += _approx_token_count_text(str(m.get("role") or ""))
        c = m.get("content")
        if isinstance(c, str):
            total += _approx_token_count_text(c)
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                if str(blk.get("type") or "").strip().lower() != "text":
                    continue
                total += _approx_token_count_text(str(blk.get("text") or ""))
        else:
            total += _approx_token_count_text(str(c or ""))
    return int(total)


def _anthropic_cache_prefix_sig(*, system_blocks: List[Dict[str, Any]], messages: List[Dict[str, Any]], breakpoint_msg_idx: Optional[int]) -> str:
    h = hashlib.sha256()
    for b in system_blocks or []:
        if not isinstance(b, dict):
            continue
        if str(b.get("type") or "").strip().lower() != "text":
            continue
        h.update(b"system\n")
        h.update(str(b.get("text") or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")

    if breakpoint_msg_idx is None:
        return h.hexdigest()

    lim = int(breakpoint_msg_idx) + 1
    for m in (messages or [])[:lim]:
        if not isinstance(m, dict):
            continue
        h.update(b"msg\n")
        h.update(str(m.get("role") or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
        c = m.get("content")
        if isinstance(c, str):
            h.update(c.encode("utf-8", errors="ignore"))
            h.update(b"\n")
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                if str(blk.get("type") or "").strip().lower() != "text":
                    continue
                h.update(str(blk.get("text") or "").encode("utf-8", errors="ignore"))
                h.update(b"\n")
        else:
            h.update(str(c or "").encode("utf-8", errors="ignore"))
            h.update(b"\n")
    return h.hexdigest()


def _anthropic_tpm_throttle(*, need_tokens: int) -> None:
    global _ANTHROPIC_TPM_USAGE
    need = int(need_tokens or 0)
    if need <= 0:
        return

    limit = int(int(_ANTHROPIC_INPUT_TPM_LIMIT) * float(_ANTHROPIC_TPM_SAFETY_FRAC))
    if limit <= 0:
        return

    while True:
        now = float(time.time())
        _ANTHROPIC_TPM_USAGE = [(t, n) for (t, n) in (_ANTHROPIC_TPM_USAGE or []) if (now - float(t)) < float(_ANTHROPIC_TPM_WINDOW_S)]
        used = int(sum(int(n) for (_, n) in _ANTHROPIC_TPM_USAGE))
        if used + need <= limit:
            _ANTHROPIC_TPM_USAGE.append((now, need))
            return
        if not _ANTHROPIC_TPM_USAGE:
            _ANTHROPIC_TPM_USAGE.append((now, need))
            return
        oldest_t = float(_ANTHROPIC_TPM_USAGE[0][0])
        sleep_s = max(0.0, float(_ANTHROPIC_TPM_WINDOW_S) - (now - oldest_t) + 1.0)
        sleep_s = min(75.0, sleep_s)
        try:
            print(
                f"Anthropic TPM throttle: used={used} need~{need} limit={limit}. Sleeping {sleep_s:.1f}s...",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        time.sleep(float(sleep_s))


def _anthropic_headers(*, api_key: str, betas: Optional[List[str]] = None, content_type_json: bool = True) -> Dict[str, str]:
    h = {
        "x-api-key": str(api_key),
        "anthropic-version": "2023-06-01",
    }
    if content_type_json:
        h["content-type"] = "application/json"
    if betas:
        h["anthropic-beta"] = ",".join([str(x) for x in betas if str(x).strip()])
    return h


def _anthropic_assistant_text(resp_json: Dict[str, Any]) -> str:
    parts: List[str] = []
    blocks = resp_json.get("content")
    if isinstance(blocks, list):
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
    return "".join(parts)


def _anthropic_messages_generate(
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    dump_path: Optional[str] = None,
) -> str:
    api_key = str(os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing ANTHROPIC_API_KEY")

    system_blocks: List[Dict[str, Any]] = []
    msgs_out: List[Dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role0 = str(m.get("role") or "user").strip().lower() or "user"
        content0 = str(m.get("content") or "")
        if role0 == "system":
            if content0.strip():
                system_blocks.append({"type": "text", "text": content0})
            continue
        if role0 not in ("user", "assistant"):
            role0 = "user"
        msgs_out.append({"role": role0, "content": [{"type": "text", "text": content0}]})

    cache_breakpoint_msg_idx: Optional[int] = None
    for i, m in enumerate(msgs_out):
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "").strip().lower() != "user":
            continue
        c = m.get("content")
        txt = ""
        if isinstance(c, str):
            txt = c
        elif isinstance(c, list) and c and isinstance(c[0], dict):
            txt = str(c[0].get("text") or "")
        if str(txt).strip().startswith("Harness info:"):
            cache_breakpoint_msg_idx = int(i)
            break

    payload: Dict[str, Any] = {
        "model": str(model),
        "max_tokens": int(max_tokens),
        "messages": msgs_out,
    }
    if str(model or "").strip() == "claude-opus-4-5-20251101":
        budget = 63_999
        try:
            budget = int(os.environ.get("DT_ANTHROPIC_THINKING_BUDGET", str(budget)) or budget)
        except Exception:
            budget = 63_999
        budget = max(1, min(200_000, int(budget)))
        max_out = max(1, int(max_tokens))
        if max_out > 1:
            budget = max(1, min(int(budget), int(max_out) - 1))
            payload["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
    if system_blocks:
        payload["system"] = system_blocks

    cache_ttl = str(os.environ.get("ANTHROPIC_CACHE_TTL") or "").strip().lower()
    cache_control_sys: Dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl in ("5m", "1h"):
        cache_control_sys["ttl"] = cache_ttl
    if system_blocks:
        system_blocks[-1]["cache_control"] = cache_control_sys

    if cache_breakpoint_msg_idx is not None:
        try:
            blk = payload["messages"][int(cache_breakpoint_msg_idx)]["content"]
            if isinstance(blk, list) and blk:
                cache_control_msg: Dict[str, Any] = {"type": "ephemeral"}
                if cache_ttl in ("5m", "1h"):
                    cache_control_msg["ttl"] = cache_ttl
                blk[-1]["cache_control"] = cache_control_msg
        except Exception:
            pass

    try:
        if isinstance(payload.get("messages"), list) and payload["messages"]:
            blk = payload["messages"][-1].get("content")
            if isinstance(blk, list) and blk:
                cache_control_end: Dict[str, Any] = {"type": "ephemeral"}
                if cache_ttl in ("5m", "1h"):
                    cache_control_end["ttl"] = cache_ttl
                blk[-1]["cache_control"] = cache_control_end
    except Exception:
        pass

    try:
        payload["temperature"] = float(temperature)
    except Exception:
        payload["temperature"] = 0.0

    if str(model or "").strip() == "claude-opus-4-5-20251101" and isinstance(payload.get("thinking"), dict):
        payload["temperature"] = 1.0

    if str(dump_path or "").strip():
        try:
            _write_json_file(str(dump_path), payload)
        except Exception:
            pass

    betas: List[str] = []

    try:
        cache_sig = _anthropic_cache_prefix_sig(system_blocks=system_blocks, messages=msgs_out, breakpoint_msg_idx=cache_breakpoint_msg_idx)
        need_full = _anthropic_estimate_input_tokens(system_blocks=system_blocks, messages=msgs_out)
        tail = msgs_out[(int(cache_breakpoint_msg_idx) + 1) :] if cache_breakpoint_msg_idx is not None else msgs_out
        need_uncached = _anthropic_estimate_input_tokens(system_blocks=[], messages=tail)
        need = int(need_uncached if (cache_sig in _ANTHROPIC_CACHE_PREFIX_SIGS) else need_full)
        need += 250
        _anthropic_tpm_throttle(need_tokens=int(need))
    except Exception:
        pass

    resp_json = _http_post_json(
        url=f"{_ANTHROPIC_BASE_URL}/messages",
        headers=_anthropic_headers(api_key=api_key, betas=betas, content_type_json=True),
        payload=payload,
        timeout_s=float(timeout_s),
    )
    try:
        usage = resp_json.get("usage") if isinstance(resp_json, dict) else None
        if isinstance(usage, dict):
            cr = float(usage.get("cache_read_input_tokens") or 0.0)
            cc = float(usage.get("cache_creation_input_tokens") or 0.0)
            it = float(usage.get("input_tokens") or 0.0)
            if str(os.environ.get("ANTHROPIC_CACHE_DEBUG") or "").strip() in ("1", "true", "yes"):
                try:
                    tot = cr + cc + it
                    frac = (cr / tot) if tot > 0 else 0.0
                    print(
                        f"Anthropic cache: read={int(cr)} create={int(cc)} input={int(it)} total={int(tot)} hit_frac={frac:.3f}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:
                    pass
            if cr > 0.0 or cc > 0.0:
                _ANTHROPIC_CACHE_PREFIX_SIGS.add(
                    _anthropic_cache_prefix_sig(system_blocks=system_blocks, messages=msgs_out, breakpoint_msg_idx=cache_breakpoint_msg_idx)
                )
    except Exception:
        pass
    if isinstance(resp_json, dict) and str(resp_json.get("stop_reason") or "").strip().lower() == "refusal":
        raise RuntimeError("Anthropic refusal")
    txt = _anthropic_assistant_text(resp_json if isinstance(resp_json, dict) else {})
    return str(txt or "")


def _llm_retry_attempts() -> int:
    attempts = 3
    try:
        attempts = int(os.environ.get("DT_LLM_RETRY_ATTEMPTS", str(attempts)) or attempts)
    except Exception:
        attempts = 3
    return max(1, min(10, int(attempts)))


def _llm_should_retry_error(err: str) -> bool:
    s = str(err or "").lower()
    if not s.strip():
        return False
    needles = (
        "http 429",
        "429",
        "rate limit",
        "rate_limit",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "http 529",
        "overloaded",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "remote end closed connection",
        "service unavailable",
    )
    return any(n in s for n in needles)


def _llm_call_with_retries(fn: Callable[[], str], *, attempts: int, base_sleep_s: float) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(int(attempts)):
        try:
            return str(fn() or "")
        except Exception as e:
            last_err = e
            if attempt >= (int(attempts) - 1) or (not _llm_should_retry_error(str(e))):
                raise
            sleep_s = float(base_sleep_s) * float(2**attempt)
            sleep_s = min(60.0, max(0.0, sleep_s))
            time.sleep(float(sleep_s))
    if last_err is not None:
        raise last_err
    raise RuntimeError("LLM request failed")


def _openai_base_model_and_effort(model: str) -> Tuple[str, Optional[str]]:
    m = str(model or "").strip()
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


_PROMPT = r"""
You are an autonomous agent being evaluated in a benchmark.

You are playing a biology strategy game (a scientific puzzle).

# 1) Story & biology (what’s being simulated)
You are studying a novel organism with an unknown internal disease. Your job is to discover an effective treatment.

Important: you are NOT told what the disease is.
- You must do your own R&D: form hypotheses, design experiments, and infer mechanisms from data.
- This organism is NOT human and is poorly studied.
- Molecular feature names are intentionally masked (e.g., “protein_7”), so you must reason from data, not from known biology.

In the simulator, time is measured in ticks (roughly days). Some endpoints report age_days, which is the same as ticks.

# 2) Objective / how to win
Your primary objective in this benchmark is to maximize cost-effective lifespan extension.

Your score is based on:
POST /api/tests/disease/claim_cure

The claim_cure response includes:
- extra_days: treated_disease_median_tick - baseline_disease_median_tick
- lifespan_recovery_pct ("Lifespan Recovery"): treated_disease_median_tick / healthy_median_tick * 100
- score: (extra_days / game.money_spent_usd) * 10000

Score is computed from lifespan improvement and money_spent. Use the returned fields (extra_days, lifespan_recovery_pct, score, win) as ground truth.

Win is a ceiling condition computed server-side:
- win: boolean
- delta_median_ticks: treated_disease_median_tick - healthy_median_tick

If win=true, the run will stop automatically (the disease cannot be improved beyond matching healthy controls).

# 3) The catch: budget & experimental design
Every experiment costs money. Efficient strategy matters.
The more replicates you use the more certain you can be of your outcomes but the more expensive it is
Just like in real life there is natural variation between individuals.

# 4) What you can do (scientific actions)
You can run experiments that reveal different views of the disease and intervention effects:

A) Characterization
- Purpose: It is an in vivo study that can be used with an intervention or by itself. It provides information about biomarkers and lifespan.
- If it is run in an in vitro model it will provide biomarkers for the cells. If it is used in an in vivo model the biomarkers come from the blood. 
- Note: death channels end with "_death" and are 0/1. For a replicate timecourse, the death time is the first day where any "*_death" becomes 1; later rows may be empty.

B) Bulk omics (snapshots)
- You can perform bulk transcriptomics, proteomics or metabolomics from samples in cell culture or from the organism.
- It allows you to see the levels of RNA, protein or metabolites in cell culture or in vivo at a given timepoint with or without an intervention.
- Purpose: identify signatures of diseased vs healthy or intervention vs control.
- Output: CSV(s) (samples x features) plus metadata CSV describing samples/replicates/age_days.

C) Spatial omics
- You can perform spatial RNA or spatial protein.
- Identify where features are localized in the tissue and how they change between two conditions.
- You can run it with or without an intervention.
- You can run it in cell culture or in vivo.

D) Protein perturbation screen (in vitro only)
- Purpose: rapidly discover how altering protein activity changes cellular behavior.
- Interpretation: in vitro readouts like confluency / n_cells / births_per_tick are context-dependent. Do not assume “higher is better”; interpret them relative to controls and the stated objective.
- This is an experiment in cells in culture, you cannot use it to derive specific outcomes for the in vivo study, it is just a cheaper way to do experiments to understand how cells behave under different perturbations

E) Final validation (claim_cure)
- Purpose: run the head-to-head survival test: healthy vs treated-disease.

# 5) Interventions (what treatments look like)
An intervention is a list of protein-target perturbations. Each perturbation is a dict:
- layer: a masked protein id like "protein_17"
- direction: "up" or "down"
- dose: a dose of 1 up increases protein activity by 10%, 10 up increases protein activity by 100%, etc. 10 down decreases activity by 100%, etc. each dose equals 10% change in activity. Dose has to be a positive integer.

# 6) General guidance
Use any experimental and analytic strategy you deem appropriate. The environment provides tools and costs but does not prescribe a particular workflow.

# 7) Data handling: inventory + analysis (omics)
Omics outputs are saved as files on the server. Prefer inventory + analysis over tiny inline previews.

Inventory:
GET /api/omics/inventory

To perform an analysis:
1) Choose the file_id(s) for the data you want to analyze (typically counts matrices).
2) Include the matching metadata file_id(s) shown in the mapping above. (For spatial: include BOTH run metadata and per-replicate cell_metadata when present.)
3) Think of the instructions for the analysis you want to run. IMPORTANT: /api/omics/analyze is stateless (it has no memory of prior steps or goals), so your instructions must include the necessary context: what the selected files are, which groups/conditions to compare, and what question to answer.
4) If you need more samples to run the analysis, run more experiments before asking for analysis.
5) Call POST /api/omics/analyze with JSON like:
{"file_ids":["<data_file_id>","<metadata_file_id>","..."],"instructions":"..."}

6) You can use /api/omics/analyze to run any type of analysis on any type of file or metadata. or even just things you tell it. It is a coding agent that will run python to analyze anything in the way you tell it to.

# 8) Minimal “how to operate the simulator” (API you can call)
You can ONLY interact with the world by calling these HTTP APIs.
Base URL is provided by the harness.

General rules (read this once and follow it every time):
- player_id is managed automatically by the harness/backstage for the duration of a run; omit it.
- Use GET /api/tests/disease/models to learn valid model keys (e.g. healthy, disease, cell_culture_healthy, cell_culture_disease).
- ticks is the study duration (roughly “days”). If you request too many ticks and the organism/culture dies earlier, the API can return an error telling you the death tick; reduce ticks or change the intervention.
- replicates is the number of independent repeats (different random seeds). Cost scales roughly linearly with replicates.
- interventions is a list of perturbations; an empty list means a baseline/control run.

Cost model (you MUST manage budget):
- Money spent is tracked server-side per run (see GET /api/game/state).
- Before running expensive studies, you can request a quote (exact price) with:
  - POST /api/tests/disease/estimate_cost
    body: {
      "experiment":"bulk_omics|spatial_omics|characterization|protein_screen|claim_cure",
      "model":"...",
      "ticks":<int>,
      "replicates":<int>,
      "omics_set":"...",
      "gene_set":"...",
      "interventions":[...]
    }
  - Notes:
    - model is required for all experiments except claim_cure.
    - omics_set is required for bulk_omics.
    - gene_set is required for spatial_omics.
    - claim_cure ignores ticks (it always uses a long study).
- Rough unit cost per sample (in vivo). In vitro (cell_culture_*) is ~4x cheaper.
  - Bulk RNA (omics_set="rna/Bulk RNAseq"): ~$200 + $0.50*ticks + $20*(#interventions)
  - Bulk proteomics (omics_set="protein/Bulk Proteomics"): ~$800 + $0.50*ticks + $20*(#interventions)
  - Bulk metabolomics (omics_set="metabolite/targeted_metabolomics"): ~$500 + $0.50*ticks + $20*(#interventions)
  - Spatial omics (spatial_omics): ~$2500 + $1.00*ticks + $40*(#interventions)
  - Characterization: ~$3000 + $2.50*ticks + $80*(#interventions)
  - Protein screen (in vitro only): ~$7500 + $3.00*ticks + $80*(#interventions), but it runs many samples internally (roughly replicates*(#proteins+2)), so total cost can be very large.
  - claim_cure: very expensive; unit cost per sample is ~$10000 + $5.00*ticks + $100*(#interventions) with ticks fixed at ~400, and total samples = 2*replicates (healthy control + treated disease).

Game / housekeeping:
- GET /api/health
- GET /api/game/state
  - Use this to track money_spent and verify your budget is not exhausted.
- POST /api/game/reset   body: {}
  - Use sparingly; it resets your player state/budget.

Discovery helpers (learn what you can target + what panels exist):
- GET /api/tests/disease/models
  - Returns the available model keys (organism vs cell culture; healthy vs disease).
- GET /api/tests/disease/proteins?model=...
  - Returns the masked proteins you can perturb in that model (protein_1, protein_2, ...).
- GET /api/bulk_omics/sets
  - Returns available bulk_omics panels (omics_set strings). Current panels include:
    - rna/Bulk RNAseq (104 RNA features)
    - protein/Bulk Proteomics (107 protein features)
    - metabolite/targeted_metabolomics (12 metabolites/toxins)
- GET /api/spatial_omics/type
  - Returns available spatial panels (gene_set strings). Current panels include:
    - spatial transcriptomics (104 RNA features)
    - spatial proteomics (107 protein features)

Core experiments (what to call, what you get back, and how to use it):

A) Characterization (biomarkers + timecourses)
- POST /api/tests/disease/characterization
  body: {"model":"...","ticks":<int>,"replicates":<int>,"interventions":[...]}
- What it is:
  - A general phenotype readout. In vivo: blood biomarkers + survival/lifespan-related signals. In vitro: cellular biomarkers.
- What you use it for:
  - Establish baselines (healthy vs disease) and learn what is abnormal.
  - Test whether an intervention moves key measurements toward the healthy baseline.
- What it returns:
  - A compact TOOL_RESULT summary plus persisted per-replicate timecourse files in the omics inventory.

B) Bulk omics (snapshots: samples x features)
- POST /api/tests/disease/bulk_omics
  body: {"model":"...","ticks":<int>,"replicates":<int>,"omics_set":"...","interventions":[...]}
- How to choose omics_set:
  - RNA panel ("rna/Bulk RNAseq"): transcript abundance-like signals.
  - Protein panel ("protein/Bulk Proteomics"): protein abundance-like signals.
  - Metabolite panel ("metabolite/targeted_metabolomics"): small molecules/toxin-like signals.
- What it returns:
  - A counts/abundance matrix (one row per replicate sample) and a metadata table describing each row (model, replicate, sample_id, study duration, assay).
  - Large arrays are usually omitted from TOOL_RESULT; use the omics inventory message to retrieve saved CSV files.
- Interpretation notes:
  - Use comparisons and controls appropriate to your question.
  - Use /api/omics/analyze on saved files to compute summaries (e.g., fold-changes, feature rankings).

C) Spatial omics (maps: per-spot features + coordinates)
- POST /api/tests/disease/spatial_omics
  body: {"model":"...","ticks":<int>,"replicates":<int>,"gene_set":"...","interventions":[...]}
- Spatial RNA vs spatial protein:
  - This endpoint supports both. You choose the modality by choosing gene_set:
    - gene_set="spatial_rna" => RNA-like spatial panel
    - gene_set="spatial_protein" => protein-like spatial panel
  - The returned metadata includes an assay label indicating which modality was run.
- What it returns (conceptually):
  - A set of spatial “spots/cells”, each with:
    - coordinates (x, y) on a grid
    - a feature vector (measured RNA/protein panel for that spot)
- What it returns (files you should analyze):
  - metadata_*.csv (combined metadata across all replicates) with columns like:
    - cell_id, assay, model, replicate, seed, sample_taken_at_day, x, y, grid_index
  - Per-replicate files (for each replicate):
    - replicates/*_matrix.csv (rows are cell_id; columns are the measured features)
    - replicates/*_cell_metadata.csv (one row per cell_id with the coordinate fields)
- How to use it well:
  - Use it to localize dysregulation: identify spatial gradients, hotspots, or regions where features differ.
  - Run healthy vs disease first, then disease+intervention to see whether spatial patterns “normalize”.
  - When you call /api/omics/analyze, explicitly describe:
    - which files are matrix vs metadata
    - which groups to compare (healthy vs disease, control vs intervention)
    - which spatial question to answer (regional differences, gradients, clustering, etc.)

D) Protein perturbation screen (in vitro only; many targets)
- POST /api/tests/disease/protein_screen   (in vitro models only)
  body: {"model":"cell_culture_*","ticks":<int>,"replicates":<int>,"direction":"up|down","dose":<int>,"interventions":[...]}
- What it is:
  - A systematic screen that perturbs many masked proteins and measures cell phenotypes.


Win check (final validation):
- POST /api/tests/disease/claim_cure
  body: {"replicates":<int>,"interventions":[...]}
- What it does:
  - Runs a head-to-head survival test used for scoring and win.

# 9) Output format (BENCHMARK CRITICAL)
You MUST respond with EXACTLY ONE JSON object per turn. No prose.

Choose ONE action:

1) Call an API
{
  "action": "call_api",
  "method": "GET" | "POST",
  "path": "/api/...",
  "query": {"k": "v", ...},          // optional, only for GET
  "body": {...},                        // optional, only for POST
  "last_result_summary": "brief summary of the previous TOOL_RESULT (or 'none' if first step)",
  "next_step_rationale": "brief reason why you are making this API call and what you expect to learn"
}

Tool execution semantics:
- When you emit a call_api action, the harness will execute it and return a TOOL_RESULT message containing http_status and response_json/response_text.
- Use those results; do NOT invent data.

Stopping rules:
- You cannot end the run explicitly.
- The harness stops automatically when a claim_cure result reports win=true, or when the step limit is reached.

Now begin. Budget is 5 million dollars; you may go into debt if needed.
""".strip()

_BENCH_PROMPT_TEXT = _PROMPT


@dataclass
class ApiResult:
    http_status: int
    response_json: Optional[Dict[str, Any]]
    response_text: str
    seconds: float


@dataclass
class BenchMetrics:
    llm_calls: int = 0
    tool_calls: int = 0
    api_calls: int = 0
    experiment_calls: int = 0
    start_ts: float = 0.0
    end_ts: float = 0.0
    win: bool = False
    final_delta_median_ticks: Optional[float] = None
    best_extra_days: Optional[float] = None
    best_lifespan_recovery_pct: Optional[float] = None
    best_score: Optional[float] = None
    best_score_lifedays_per_usd: Optional[float] = None
    best_score_seq: Optional[int] = None
    money_spent_cents: Optional[int] = None
    money_spent_usd: Optional[float] = None
    experiments: List[str] = None

    def __post_init__(self) -> None:
        if self.experiments is None:
            self.experiments = []


def _msg(role: str, content: str) -> Dict[str, str]:
    return {
        "role": str(role or "").strip() or "user",
        "content": str(content or ""),
    }


def _new_player_id() -> str:
    t = int(time.time())
    r = secrets.token_hex(4)
    return f"bench_{t}_{r}"


def call_local_api(
    *,
    base_url: str,
    method: str,
    path: str,
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 900.0,
) -> ApiResult:
    t0 = time.time()
    try:
        base = str(base_url or "").strip().rstrip("/")
        p = str(path or "").strip()
        if not base or not p.startswith("/"):
            return ApiResult(http_status=0, response_json=None, response_text="Bad base_url or path", seconds=time.time() - t0)

        url = base + p
        if isinstance(query, dict) and query:
            q2: Dict[str, Any] = {}
            for k, v in query.items():
                if v is None:
                    continue
                q2[str(k)] = str(v)
            if q2:
                url = url + "?" + urllib.parse.urlencode(q2, doseq=True)

        m = str(method or "").strip().upper() or "GET"
        data: Optional[bytes] = None
        headers: Dict[str, str] = {}
        if m == "POST":
            if body is None:
                data = b"{}"
            else:
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, method=m, headers=headers)
        status = 0
        text = ""
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                raw = resp.read()
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        except urllib.error.HTTPError as e:
            status = int(getattr(e, "code", 0) or 0)
            try:
                raw = e.read()
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
            except Exception:
                text = str(e)
        except Exception as e:
            return ApiResult(http_status=0, response_json=None, response_text=str(e), seconds=time.time() - t0)

        obj: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(text) if isinstance(text, str) and text.strip() else None
            obj = parsed if isinstance(parsed, dict) else None
        except Exception:
            obj = None

        return ApiResult(http_status=int(status), response_json=obj, response_text=str(text or ""), seconds=time.time() - t0)
    except Exception as e:
        return ApiResult(http_status=0, response_json=None, response_text=str(e), seconds=time.time() - t0)


def _is_safe_retry_request(method: str, path: str) -> bool:
    m = str(method or "").strip().upper()
    p = str(path or "").strip()
    if m == "GET":
        return True
    if p == "/api/discuss" or p.startswith("/api/omics/"):
        return True
    if p.endswith("/estimate_cost"):
        return True
    return False


def call_local_api_retrying(
    *,
    base_url: str,
    method: str,
    path: str,
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 900.0,
    max_attempts: int = 3,
) -> ApiResult:
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        timeout_eff = float(timeout_s)
        try:
            p0 = str(path or "").strip()
            if p0 == "/api/discuss" or p0.startswith("/api/omics/"):
                timeout_eff = min(float(timeout_eff), 900.0)
        except Exception:
            timeout_eff = float(timeout_s)

        res = call_local_api(
            base_url=base_url,
            method=method,
            path=path,
            query=query,
            body=body,
            timeout_s=timeout_eff,
        )
        st = int(getattr(res, "http_status", 0) or 0)
        if attempt >= (attempts - 1):
            return res
        if not _is_safe_retry_request(method=str(method), path=str(path)):
            return res
        if not (st == 0 or st == 429 or st == 503 or st >= 500):
            return res

        retry_after_s = None
        try:
            if isinstance(res.response_json, dict):
                ra = res.response_json.get("retry_after_s")
                if ra is not None:
                    retry_after_s = float(ra)
        except Exception:
            retry_after_s = None

        sleep_s = float(1.0 + 2.0 * float(attempt))
        if retry_after_s is not None:
            sleep_s = max(float(sleep_s), float(retry_after_s))
        sleep_s = min(60.0, max(0.25, float(sleep_s)))
        try:
            time.sleep(float(sleep_s))
        except Exception:
            pass
    return res


def llm_generate(
    *,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    dump_path: Optional[str] = None,
) -> str:
    prov = str(provider or "").strip().lower()
    if prov == "grok":
        prov = "xai"
    attempts = _llm_retry_attempts()
    messages = _llm_messages_with_step_budget(list(messages or []))
    if prov in ("openai", "openai_compat"):
        from openai import OpenAI

        client = OpenAI(api_key=str(os.environ.get("OPENAI_API_KEY") or "").strip() or None)

        base_model, effort = _openai_base_model_and_effort(str(model))
        if effort is not None:
            ctx = [{"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")} for m in (messages or [])]
            req2: Dict[str, Any] = {
                "model": str(base_model),
                "input": ctx,
                "reasoning": {"effort": str(effort)},
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "timeout": float(timeout_s),
            }
            try:
                req2["temperature"] = float(temperature)
            except Exception:
                pass
            if str(dump_path or "").strip():
                try:
                    _write_json_file(str(dump_path), {"provider": "openai", "request": req2, "max_tokens": int(max_tokens)})
                except Exception:
                    pass

            def _call_once() -> str:
                try:
                    try:
                        resp = client.responses.create(**req2, max_output_tokens=int(max_tokens))
                    except Exception:
                        resp = client.responses.create(**req2)
                except Exception as e:
                    msg = str(e)
                    if ("temperature" in msg) and ("Unsupported parameter" in msg or "not supported" in msg):
                        req3 = dict(req2)
                        req3.pop("temperature", None)
                        try:
                            resp = client.responses.create(**req3, max_output_tokens=int(max_tokens))
                        except Exception:
                            resp = client.responses.create(**req3)
                    else:
                        raise
                out_text = ""
                try:
                    out_text = str(getattr(resp, "output_text", "") or "")
                except Exception:
                    out_text = ""
                if str(out_text or "").strip():
                    return str(out_text or "")
                try:
                    dump = resp.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    dump = None
                if isinstance(dump, dict) and isinstance(dump.get("output"), list):
                    chunks: List[str] = []
                    for item in (dump.get("output") or []):
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") != "message":
                            continue
                        content = item.get("content")
                        if not isinstance(content, list):
                            continue
                        for c in content:
                            if not isinstance(c, dict):
                                continue
                            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                                chunks.append(str(c.get("text") or ""))
                            elif c.get("type") == "text" and isinstance(c.get("text"), str):
                                chunks.append(str(c.get("text") or ""))
                    out_text = "".join(chunks).strip()
                return str(out_text or "")

            return _llm_call_with_retries(_call_once, attempts=int(attempts), base_sleep_s=0.8)

        req = {
            "model": str(base_model),
            "messages": [{"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")} for m in (messages or [])],
            "temperature": float(temperature),
            "timeout": float(timeout_s),
        }
        if str(dump_path or "").strip():
            try:
                _write_json_file(str(dump_path), {"provider": "openai", "request": req, "max_tokens": int(max_tokens)})
            except Exception:
                pass
        def _call_once() -> str:
            try:
                resp = client.chat.completions.create(**req, max_completion_tokens=int(max_tokens))
            except Exception as e:
                msg = str(e)
                if ("temperature" in msg) and ("Unsupported parameter" in msg or "not supported" in msg):
                    req3 = dict(req)
                    req3.pop("temperature", None)
                    try:
                        resp = client.chat.completions.create(**req3, max_completion_tokens=int(max_tokens))
                    except Exception:
                        resp = client.chat.completions.create(**req3, max_tokens=int(max_tokens))
                else:
                    resp = client.chat.completions.create(**req, max_tokens=int(max_tokens))
            try:
                return str(resp.choices[0].message.content or "")
            except Exception:
                return ""

        return _llm_call_with_retries(_call_once, attempts=int(attempts), base_sleep_s=0.8)

    if prov in ("gemini",):
        api_key = str(os.environ.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("missing GEMINI_API_KEY")
        base_url = str(os.environ.get("GEMINI_BASE_URL") or "").strip() or "https://generativelanguage.googleapis.com/v1beta"

        sys_txts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for m in (messages or []):
            r = str(m.get("role") or "user").strip().lower()
            c = str(m.get("content") or "")
            if r == "system":
                if c.strip():
                    sys_txts.append(c)
                continue
            role_out = "user"
            if r in ("assistant", "model"):
                role_out = "model"
            contents.append({"role": role_out, "parts": [{"text": c}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        if str(model or "").strip().startswith("gemini-3-"):
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "high"}
        if sys_txts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_txts)}]}

        url = str(base_url).rstrip("/") + "/models/" + str(model) + ":generateContent"
        headers = {
            "x-goog-api-key": str(api_key),
            "content-type": "application/json",
        }

        if str(dump_path or "").strip():
            try:
                _write_json_file(
                    str(dump_path),
                    {
                        "provider": "gemini",
                        "base_url": str(base_url),
                        "url": str(url),
                        "request": payload,
                        "max_tokens": int(max_tokens),
                    },
                )
            except Exception:
                pass

        def _call_once() -> str:
            resp_json = _http_post_json(url=str(url), headers=headers, payload=payload, timeout_s=float(timeout_s))
            out_text0 = ""
            diag: Dict[str, Any] = {}
            try:
                candidates = resp_json.get("candidates") if isinstance(resp_json, dict) else None
                if isinstance(candidates, list) and candidates:
                    c0 = candidates[0] if isinstance(candidates[0], dict) else {}
                    content = c0.get("content") if isinstance(c0, dict) else None
                    parts = content.get("parts") if isinstance(content, dict) else None
                    if isinstance(parts, list) and parts:
                        out_chunks: List[str] = []
                        for p in parts:
                            if isinstance(p, dict) and isinstance(p.get("text"), str):
                                out_chunks.append(str(p.get("text") or ""))
                        out_text0 = "".join(out_chunks).strip()
                    if isinstance(resp_json, dict) and isinstance(c0, dict):
                        try:
                            diag["finishReason"] = c0.get("finishReason")
                            diag["finishMessage"] = c0.get("finishMessage")
                            diag["safetyRatings"] = c0.get("safetyRatings")
                        except Exception:
                            pass
            except Exception:
                out_text0 = ""
            if isinstance(resp_json, dict):
                try:
                    pf = resp_json.get("promptFeedback")
                    if isinstance(pf, dict) and pf:
                        diag["promptFeedback"] = pf
                except Exception:
                    pass
            if not str(out_text0 or "").strip():
                diag_s = ""
                try:
                    diag_s = json.dumps(diag, ensure_ascii=False)
                except Exception:
                    diag_s = str(diag)
                raise RuntimeError("Gemini returned empty text" + (": " + diag_s[:1000] if diag_s else ""))
            return str(out_text0 or "")

        return _llm_call_with_retries(_call_once, attempts=int(attempts), base_sleep_s=0.8)

    if prov in ("xai", "grok"):
        from openai import OpenAI

        api_key = str(os.environ.get("XAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("missing XAI_API_KEY")
        base_url = str(os.environ.get("XAI_BASE_URL") or "").strip() or "https://api.x.ai/v1"

        client = OpenAI(api_key=api_key, base_url=str(base_url))
        model_call = _xai_canonical_model(str(model))
        req = {
            "model": str(model_call),
            "messages": [{"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")} for m in (messages or [])],
            "temperature": float(temperature),
            "timeout": float(timeout_s),
        }
        if str(dump_path or "").strip():
            try:
                _write_json_file(
                    str(dump_path),
                    {
                        "provider": "xai",
                        "base_url": str(base_url),
                        "request": req,
                        "max_tokens": int(max_tokens),
                    },
                )
            except Exception:
                pass

        def _call_once() -> str:
            try:
                resp = client.chat.completions.create(**req, max_completion_tokens=int(max_tokens))
            except Exception:
                resp = client.chat.completions.create(**req, max_tokens=int(max_tokens))
            try:
                return str(resp.choices[0].message.content or "")
            except Exception:
                return ""

        return _llm_call_with_retries(_call_once, attempts=int(attempts), base_sleep_s=0.8)

    if prov in ("anthropic", "claude"):
        def _call_once() -> str:
            return _anthropic_messages_generate(
                model=str(model),
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                timeout_s=float(timeout_s),
                dump_path=str(dump_path or "") or None,
            )

        return _llm_call_with_retries(_call_once, attempts=int(attempts), base_sleep_s=0.8)

    raise RuntimeError(f"unsupported provider: {provider!r}")


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        p = str(path or "").strip()
        if not p:
            return None
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json_file(path: str, obj: Dict[str, Any]) -> None:
    p = str(path or "").strip()
    if not p:
        return
    out_dir = os.path.dirname(os.path.abspath(p))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        f.write("\n")
    os.replace(tmp, p)


def _write_json_any_file(path: str, obj: Any) -> None:
    p = str(path or "").strip()
    if not p:
        return
    out_dir = os.path.dirname(os.path.abspath(p))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        f.write("\n")
    os.replace(tmp, p)


def _parse_first_json(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "")
    i0 = s.find("{")
    if i0 < 0:
        return None
    i1 = s.rfind("}")
    if i1 < 0 or i1 <= i0:
        return None
    try:
        obj = json.loads(s[i0 : i1 + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _issues_from_text(*, text: str, source: str, max_items: int = 200) -> List[Dict[str, Any]]:
    s = str(text or "")
    if not s.strip():
        return []
    out: List[Dict[str, Any]] = []
    pats: List[Tuple[str, str, str]] = [
        (r"\btraceback\b", "error", "traceback"),
        (r"\b(exception|assertionerror|runtimeerror|valueerror|typeerror|keyerror)\b", "error", "exception"),
        (r"\b(llm_call_failed|http\s+\d{3})\b", "error", "llm_or_http_error"),
        (r"\b(429|rate\s*limit|quota|retry[-\s]?after|retrydelay)\b", "warn", "rate_limit_or_retry"),
        (r"\b(max[_\s]?tokens|max[_\s]?output[_\s]?tokens|stop_reason\s*[:=]\s*max_tokens)\b", "warn", "possible_truncation"),
        (r"\b(timeout|timed\s*out)\b", "warn", "timeout"),
    ]
    for ln in s.splitlines():
        if len(out) >= int(max_items):
            break
        line = str(ln or "")
        if not line.strip():
            continue
        hit: Optional[Tuple[str, str]] = None
        for pat, sev, kind in pats:
            try:
                if re.search(pat, line, flags=re.IGNORECASE):
                    hit = (sev, kind)
                    break
            except Exception:
                continue
        if hit is None:
            continue
        sev, kind = hit
        out.append(
            {
                "severity": str(sev),
                "kind": str(kind),
                "source": str(source),
                "seq": None,
                "ts": None,
                "summary": line[:3200],
                "details": line,
            }
        )
    return out


def _detect_issues_from_events_path(*, events_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        if not (events_path.exists() and events_path.is_file()):
            return []
        with events_path.open("r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                s = str(ln or "").strip()
                if not s:
                    continue
                try:
                    ev = json.loads(s)
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                t = str(ev.get("type") or "")
                seq = ev.get("seq")
                ts = ev.get("ts")

                if t == "llm_error":
                    msg = str(ev.get("error") or "")
                    out.append({"severity": "error", "kind": "llm_error", "source": "events", "seq": seq, "ts": ts, "summary": (msg[:3200] if msg else "llm_error"), "details": ev})
                    continue

                if t == "player_id_mismatch":
                    details = ev.get("details") if isinstance(ev.get("details"), dict) else {}
                    found = details.get("found")
                    exp = details.get("expected")
                    out.append({"severity": "warn", "kind": "player_id_mismatch", "source": "events", "seq": seq, "ts": ts, "summary": f"player_id mismatch (found={found} expected={exp})", "details": ev})
                    continue

                if t == "final_rejected":
                    out.append({"severity": "warn", "kind": "final_rejected", "source": "events", "seq": seq, "ts": ts, "summary": "Final action was rejected (format/validation issue)", "details": ev})
                    continue

                if t == "llm":
                    txt = str(ev.get("text") or "")
                    obj = _parse_first_json(txt)
                    if obj is None:
                        out.append({"severity": "warn", "kind": "llm_malformed_output", "source": "events", "seq": seq, "ts": ts, "summary": "LLM output did not parse as JSON object", "details": txt[:12000]})
                    else:
                        act = str(obj.get("action") or "").strip().lower()
                        if act and act not in ("call_api",):
                            out.append({"severity": "warn", "kind": "llm_unexpected_action", "source": "events", "seq": seq, "ts": ts, "summary": f"Unexpected action='{act}'", "details": obj})
                    continue

                if t == "api":
                    status = ev.get("http_status")
                    path = str(ev.get("path") or "")
                    method = str(ev.get("method") or "")
                    seconds = ev.get("seconds")
                    try:
                        st_i = int(status)
                    except Exception:
                        st_i = None
                    if st_i == 0:
                        out.append({"severity": "warn", "kind": "api_unreachable", "source": "events", "seq": seq, "ts": ts, "summary": f"API {method} {path} failed (http_status=0)", "details": ev})
                    if st_i is not None and st_i >= 400:
                        sev = "error" if st_i >= 500 else "warn"
                        kind = "api_rate_limited" if st_i == 429 else "api_http_error"
                        out.append({"severity": sev, "kind": kind, "source": "events", "seq": seq, "ts": ts, "summary": f"API {method} {path} -> HTTP {st_i}", "details": ev})
                    try:
                        if seconds is not None and float(seconds) >= 120.0:
                            out.append({"severity": "warn", "kind": "api_slow", "source": "events", "seq": seq, "ts": ts, "summary": f"Slow API call: {method} {path} ({float(seconds):.1f}s)", "details": ev})
                    except Exception:
                        pass
                    continue

                if t == "tool_result":
                    payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                    status = payload.get("http_status")
                    path = str(ev.get("path") or payload.get("path") or "")
                    try:
                        st_i = int(status)
                    except Exception:
                        st_i = None
                    if st_i == 0:
                        out.append({"severity": "warn", "kind": "tool_unreachable", "source": "events", "seq": seq, "ts": ts, "summary": f"Tool result failed for {path} (http_status=0)", "details": ev})
                    if st_i is not None and st_i >= 400:
                        sev = "error" if st_i >= 500 else "warn"
                        kind = "tool_rate_limited" if st_i == 429 else "tool_http_error"
                        out.append({"severity": sev, "kind": kind, "source": "events", "seq": seq, "ts": ts, "summary": f"Tool result issue for {path} (HTTP {st_i})", "details": ev})
                    continue
    except Exception:
        return out

    def _sort_key(x: Dict[str, Any]) -> Tuple[int, float]:
        seq0 = x.get("seq")
        try:
            s0 = int(seq0)
        except Exception:
            s0 = -1
        ts0 = x.get("ts")
        try:
            t0 = float(ts0)
        except Exception:
            t0 = 0.0
        return (s0, t0)

    out.sort(key=_sort_key)
    return out


def _story_markdown_from_events_path(*, events_path: Path, max_steps: int = 300) -> str:
    try:
        if not (events_path.exists() and events_path.is_file()):
            return ""
    except Exception:
        return ""

    steps: List[Dict[str, Any]] = []
    llm_errors: List[Dict[str, Any]] = []
    final_rejected: List[Dict[str, Any]] = []
    end_ev: Optional[Dict[str, Any]] = None

    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                s = str(ln or "").strip()
                if not s:
                    continue
                try:
                    ev = json.loads(s)
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                t = str(ev.get("type") or "")
                if t == "llm":
                    try:
                        step_i = int(ev.get("step") or 0)
                    except Exception:
                        step_i = 0
                    if step_i > int(max_steps):
                        continue
                    lrs = ev.get("last_result_summary")
                    nsr = ev.get("next_step_rationale")
                    if not (isinstance(lrs, str) and lrs.strip()) and not (isinstance(nsr, str) and nsr.strip()):
                        continue
                    steps.append({"step": step_i, "seq": ev.get("seq"), "ts": ev.get("ts"), "last_result_summary": lrs, "next_step_rationale": nsr})
                elif t == "llm_error":
                    llm_errors.append({"seq": ev.get("seq"), "ts": ev.get("ts"), "error": ev.get("error")})
                elif t == "final_rejected":
                    final_rejected.append({"seq": ev.get("seq"), "ts": ev.get("ts"), "payload": ev.get("payload")})
                elif t == "end":
                    end_ev = ev
    except Exception:
        return ""

    steps.sort(key=lambda x: int(x.get("step") or 0))

    lines: List[str] = []
    lines.append("# Run story")
    lines.append("")

    if steps:
        for stp in steps:
            lines.append(f"## Step {int(stp.get('step') or 0)}")
            lines.append("")
            lrs = stp.get("last_result_summary")
            nsr = stp.get("next_step_rationale")
            if isinstance(lrs, str) and lrs.strip():
                lines.append("### Last result summary")
                lines.append("")
                lines.append(str(lrs).strip())
                lines.append("")
            if isinstance(nsr, str) and nsr.strip():
                lines.append("### Next step rationale")
                lines.append("")
                lines.append(str(nsr).strip())
                lines.append("")

    if llm_errors:
        lines.append("## LLM errors")
        lines.append("")
        for e in llm_errors[-50:]:
            lines.append(f"- seq={e.get('seq')} {str(e.get('error') or '')[:6000]}")
        lines.append("")

    if final_rejected:
        lines.append("## Final rejected")
        lines.append("")
        for e in final_rejected[-50:]:
            lines.append(f"- seq={e.get('seq')} payload={json.dumps(e.get('payload'), ensure_ascii=False)[:4000]}")
        lines.append("")

    if isinstance(end_ev, dict):
        lines.append("## End")
        lines.append("")
        try:
            end_compact = dict(end_ev)
            end_compact.pop("type", None)
            end_compact.pop("ts", None)
            end_compact.pop("seq", None)
            lines.append("```json")
            lines.append(json.dumps(end_compact, ensure_ascii=False, indent=2)[:12000])
            lines.append("```")
        except Exception:
            pass
        lines.append("")

    out = "\n".join(lines).strip() + "\n"
    return out if out.strip() else ""


def _human_mode_dir(*, state_out: Optional[str], events_out: Optional[str], files_dir: Optional[str]) -> Optional[Path]:
    cand: Optional[str] = None
    for p in (state_out, events_out, files_dir):
        if str(p or "").strip():
            cand = str(p)
            break
    if not cand:
        return None
    try:
        d = Path(str(cand)).resolve()
        if d.is_dir():
            return d / "human_mode"
        return d.parent / "human_mode"
    except Exception:
        return None


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _human_write_pending(
    *,
    human_dir: Path,
    step: int,
    player_id: str,
    base_url: str,
    error: Optional[str],
) -> None:
    try:
        human_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    pending = {
        "ok": True,
        "waiting": True,
        "step": int(step),
        "player_id": str(player_id),
        "base_url": str(base_url),
        "expecting": f"input_step_{int(step):06d}.json",
        "error": str(error or "").strip(),
        "ts": float(time.time()),
        "instructions": "Provide a directive for the next action. You may provide either plain text (key 'text') or a full Action JSON (key 'action_json').",
    }
    _write_json_file(str(human_dir / "pending.json"), pending)


def _human_wait_for_input(
    *,
    human_dir: Path,
    step: int,
    player_id: str,
    base_url: str,
    poll_s: float,
    error: Optional[str],
) -> Dict[str, Any]:
    poll = float(poll_s)
    if poll <= 0.05:
        poll = 0.25
    if poll > 5.0:
        poll = 5.0

    _human_write_pending(human_dir=human_dir, step=int(step), player_id=str(player_id), base_url=str(base_url), error=error)

    inp = human_dir / f"input_step_{int(step):06d}.json"
    stop_path = human_dir / "stop.json"
    processed_dir = human_dir / "processed"
    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    while True:
        if stop_path.exists():
            stop_obj = _read_json_if_exists(stop_path) or {}
            return {"ok": True, "stop": True, **stop_obj}

        obj = _read_json_if_exists(inp)
        if isinstance(obj, dict):
            try:
                ts = int(time.time() * 1000)
                outp = processed_dir / f"input_step_{int(step):06d}_{int(ts)}.json"
                os.replace(str(inp), str(outp))
            except Exception:
                try:
                    inp.unlink()
                except Exception:
                    pass
            try:
                (human_dir / "pending.json").unlink()
            except Exception:
                pass
            return obj

        time.sleep(poll)


def _is_csv_field(k: str) -> bool:
    ks = str(k or "")
    return ks.endswith("_csv") or ks in ("matrix_csv", "metadata_csv")


def _csv_preview(text: str, *, max_lines: int = 30, max_chars: int = 6000) -> str:
    s = str(text or "")
    if not s:
        return ""
    lines = s.splitlines()
    out = "\n".join(lines[: max(1, int(max_lines))])
    if len(out) > int(max_chars):
        out = out[: int(max_chars)]
    return out


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    n = int(len(ys))
    if n <= 0:
        return None
    if (n % 2) == 1:
        return float(ys[n // 2])
    return float(0.5 * (ys[(n // 2) - 1] + ys[n // 2]))


def _series_stats(values: Any) -> Dict[str, Any]:
    if not isinstance(values, list) or not values:
        return {}
    xs: List[float] = []
    start_v: Optional[float] = None
    end_v: Optional[float] = None
    for x in values:
        try:
            f = float(x)
        except Exception:
            continue
        if not math.isfinite(f):
            continue
        xs.append(float(f))
        if start_v is None:
            start_v = float(f)
        end_v = float(f)
    if not xs or start_v is None or end_v is None:
        return {}
    n = int(len(xs))
    mn = float(min(xs))
    mx = float(max(xs))
    mean = float(sum(xs) / float(n)) if n > 0 else None
    med = _median(xs)
    out: Dict[str, Any] = {
        "n": int(n),
        "start": float(start_v),
        "end": float(end_v),
        "delta": float(end_v - start_v),
        "min": float(mn),
        "max": float(mx),
        "mean": float(mean) if mean is not None else None,
        "median": float(med) if med is not None else None,
    }
    return out


def _is_death_like_measurement(name: str, *, death_names: Optional[List[str]] = None) -> bool:
    s = str(name or "")
    if not s:
        return False
    if isinstance(death_names, list) and s in set(str(x) for x in death_names):
        return True
    s2 = s.lower()
    # Only treat true in-vivo death channels as death-like.
    # In vitro models include valid metrics like "deaths_per_tick" and "n_deaths".
    if s2.startswith("death_"):
        return True
    if s2.endswith("_death"):
        return True
    return False


def _measurement_timecourse_stats(
    series: Any,
    *,
    death_names: Optional[List[str]] = None,
    max_measurements: int = 50,
) -> Dict[str, Any]:
    if not isinstance(series, dict):
        return {}
    sample = series.get("sample")
    if not isinstance(sample, dict):
        return {}

    stats_all: List[Dict[str, Any]] = []
    n_days: Optional[int] = None
    for k, v in list(sample.items())[: int(max_measurements)]:
        nm = str(k)
        if _is_death_like_measurement(nm, death_names=death_names):
            continue
        st = _series_stats(v)
        if not st:
            continue
        if n_days is None and isinstance(v, list):
            n_days = int(len(v))
        stats_all.append({"measurement": nm, **st})

    top_abs_delta = sorted(
        stats_all,
        key=lambda d: abs(float(d.get("delta") or 0.0)),
        reverse=True,
    )[:8]

    out: Dict[str, Any] = {
        "measurements_n": int(len(stats_all)),
        "stats": stats_all,
        "top_abs_delta": top_abs_delta,
    }
    if n_days is not None:
        out["days_n"] = int(n_days)
    return out


def _is_in_vitro_model_key(model_key: Any) -> bool:
    s = str(model_key or "").strip()
    return bool(s == "cell_culture" or s.startswith("cell_culture_") or s.endswith("_cell_culture"))


def _safe_file_filename(name: str) -> str:
    s = str(name or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "file"


def _safe_artifact_filename(name: str) -> str:
    # Backward compatible alias.
    return _safe_file_filename(name)


def _maybe_write_file(
    *,
    files_dir: Optional[str],
    seq: int,
    field: str,
    content: str,
) -> Optional[str]:
    if not files_dir:
        return None
    try:
        d = Path(str(files_dir)).resolve()
        d.mkdir(parents=True, exist_ok=True)
        fn = f"{int(seq):06d}_{_safe_file_filename(field)}.csv"
        p = d / fn
        p.write_text(str(content), encoding="utf-8")
        return str(p)
    except Exception:
        return None


def _maybe_write_artifact(*, artifacts_dir: Optional[str], seq: int, field: str, content: str) -> Optional[str]:
    # Backward compatible alias.
    return _maybe_write_file(files_dir=artifacts_dir, seq=seq, field=field, content=content)


def _summarize_response_json_for_events(
    resp_json: Optional[Dict[str, Any]],
    *,
    seq: int,
    files_dir: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not resp_json or not isinstance(resp_json, dict):
        return resp_json, []
    use_dir = files_dir if files_dir is not None else artifacts_dir
    files: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {}
    for k, v in resp_json.items():
        if isinstance(v, str) and _is_csv_field(str(k)):
            file_path = _maybe_write_file(files_dir=use_dir, seq=seq, field=str(k), content=v)
            files.append(
                {
                    "field": str(k),
                    "mime": "text/csv",
                    "bytes": len(v.encode("utf-8", errors="replace")),
                    "preview": _csv_preview(v),
                    "path": file_path,
                }
            )
            out[str(k)] = {
                "_file": True,
                "mime": "text/csv",
                "bytes": files[-1]["bytes"],
                "path": file_path,
            }
        else:
            out[str(k)] = v
    return out, files


def _write_event_line(fp: Optional[Any], event: Dict[str, Any]) -> None:
    if fp is None:
        return
    try:
        ev = event
        if not isinstance(ev, dict):
            ev = {}

        ts_s: Optional[float] = None
        try:
            v = ev.get("ts")
            if v is not None:
                ts_s = float(v)
        except Exception:
            ts_s = None
        if ts_s is None:
            ts_s = float(time.time())
            try:
                if "ts" not in ev:
                    ev["ts"] = float(ts_s)
            except Exception:
                pass

        try:
            if "ts_ms" not in ev:
                ev["ts_ms"] = int(round(float(ts_s) * 1000.0))
        except Exception:
            pass
        try:
            if "iso" not in ev:
                ev["iso"] = datetime.fromtimestamp(float(ts_s), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            pass

        fp.write(json.dumps(ev, ensure_ascii=False) + "\n")
        fp.flush()
    except Exception:
        pass


def _notebook_append(notebook: str, line: str, *, max_chars: int = 9000, max_lines: int = 200) -> str:
    nb = str(notebook or "").strip("\n")
    ln = str(line or "").strip()
    if not ln:
        return nb
    parts = [p for p in nb.splitlines() if p.strip()] if nb else []
    parts.append(ln)
    if len(parts) > int(max_lines):
        parts = parts[-int(max_lines) :]
    out = "\n".join(parts)
    if len(out) > int(max_chars):
        out = out[-int(max_chars) :]
        cut = out.find("\n")
        if cut >= 0:
            out = out[cut + 1 :]
    return out


def _prune_messages(messages: List[Dict[str, str]], *, pinned: List[Dict[str, str]], keep_last: int = 10) -> List[Dict[str, str]]:
    pin = list(pinned)
    tail = list(messages[-int(keep_last) :]) if messages else []
    out: List[Dict[str, str]] = []
    for m in pin:
        if m not in out:
            out.append(m)
    for m in tail:
        if m in out:
            continue
        out.append(m)
    return out


def _estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        total += 10
        total += _approx_token_count_text(str(m.get("role") or ""))
        total += _approx_token_count_text(str(m.get("content") or ""))
    return int(total)


def _messages_to_summary_text(messages: List[Dict[str, str]], *, max_chars: int) -> str:
    parts: List[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")
        if not role and not content:
            continue
        parts.append(f"[{role}]\n{content}\n")
    txt = "\n".join(parts).strip()
    if int(max_chars) > 0 and len(txt) > int(max_chars):
        txt = txt[-int(max_chars) :]
        cut = txt.find("\n")
        if cut >= 0:
            txt = txt[cut + 1 :]
    return txt


def _is_context_summary_message(m: Dict[str, str]) -> bool:
    if not isinstance(m, dict):
        return False
    if str(m.get("role") or "") != "user":
        return False
    c = str(m.get("content") or "")
    return c.lstrip().startswith("CONTEXT_SUMMARY_ENTRY:")


def _extract_context_summary_text(content: str) -> str:
    s = str(content or "")
    while True:
        s2 = s.lstrip()
        if not s2.startswith("CONTEXT_SUMMARY_ENTRY:"):
            break
        s = s2[len("CONTEXT_SUMMARY_ENTRY:") :]
    return str(s).strip()


def _maybe_prune_messages_with_summary(
    *,
    messages: List[Dict[str, str]],
    pinned: List[Dict[str, str]],
    provider: str,
    model: str,
    llm_timeout_s: float,
    step: int,
    max_tokens_per_call: int,
    notebook: str,
) -> Tuple[List[Dict[str, str]], str]:
    context_tokens = 64_000
    try:
        context_tokens = int(os.environ.get("DT_LLM_CONTEXT_TOKENS", str(context_tokens)) or context_tokens)
    except Exception:
        context_tokens = 64_000
    context_tokens = max(20_000, int(context_tokens))

    trim_at = int(0.92 * float(context_tokens))
    try:
        trim_at = int(os.environ.get("DT_PROMPT_TRIM_AT_TOKENS", str(trim_at)) or trim_at)
    except Exception:
        trim_at = int(0.92 * float(context_tokens))
    trim_at = max(10_000, min(int(context_tokens), int(trim_at)))

    target = int(0.80 * float(context_tokens))
    try:
        target = int(os.environ.get("DT_PROMPT_TARGET_TOKENS", str(target)) or target)
    except Exception:
        target = int(0.80 * float(context_tokens))
    target = max(10_000, min(int(context_tokens), int(target)))

    keep_recent = 60
    try:
        keep_recent = int(os.environ.get("DT_PROMPT_KEEP_RECENT", str(keep_recent)) or keep_recent)
    except Exception:
        keep_recent = 60
    keep_recent = max(8, int(keep_recent))

    summary_out_tokens = 1200
    try:
        summary_out_tokens = int(os.environ.get("DT_PROMPT_SUMMARY_OUT_TOKENS", str(summary_out_tokens)) or summary_out_tokens)
    except Exception:
        summary_out_tokens = 1200
    summary_out_tokens = max(200, min(4000, int(summary_out_tokens)))

    chunk_max_chars = 240_000
    try:
        chunk_max_chars = int(os.environ.get("DT_PROMPT_SUMMARY_CHUNK_MAX_CHARS", str(chunk_max_chars)) or chunk_max_chars)
    except Exception:
        chunk_max_chars = 240_000
    chunk_max_chars = max(20_000, int(chunk_max_chars))

    msgs = list(messages or [])
    pin = list(pinned or [])
    pin_len0 = int(len(pin))
    pin_ids = {id(x) for x in pin}
    msgs = [m for m in msgs if isinstance(m, dict)]
    if not msgs:
        return [], str(notebook or "")

    summary_msg: Optional[Dict[str, str]] = None
    rolling_summary = ""
    summary_msgs: List[Dict[str, str]] = []
    summary_texts: List[str] = []
    for m in msgs:
        if id(m) in pin_ids:
            continue
        if _is_context_summary_message(m):
            summary_msgs.append(m)
            summary_texts.append(_extract_context_summary_text(str(m.get("content") or "")))
    if summary_msgs:
        summary_msg = summary_msgs[-1]
        try:
            summary_msg["role"] = "user"
        except Exception:
            pass

        if len(summary_texts) == 1:
            rolling_summary = str(summary_texts[0] or "").strip()
        else:
            summaries_txt = "\n\n---\n\n".join([str(x or "").strip() for x in summary_texts if str(x or "").strip()])
            summary_prompt0 = (
                "Consolidate the following context summaries into ONE coherent rolling summary. Preserve all distinct facts, "
                "decisions, hypotheses, experiment results (keep numbers), constraints/rules, and open questions. "
                "Keep continuity and avoid duplication. Return plain text only.\n\n" + summaries_txt
            )
            merged = ""
            try:
                merged = llm_generate(
                    provider=str(provider),
                    model=str(model),
                    messages=[
                        _msg("system", "You maintain a rolling memory summary for continuity."),
                        _msg("user", summary_prompt0),
                    ],
                    temperature=0.0,
                    max_tokens=int(summary_out_tokens),
                    timeout_s=float(llm_timeout_s),
                    dump_path=None,
                )
            except Exception:
                merged = ""
            merged = str(merged or "").strip()
            if merged:
                rolling_summary = merged
            else:
                rolling_summary = "\n\n".join([str(x or "").strip() for x in summary_texts if str(x or "").strip()]).strip()

        rm_ids0 = {id(x) for x in summary_msgs}
        msgs = [m for m in msgs if id(m) not in rm_ids0]

        if rolling_summary:
            try:
                summary_msg["content"] = "CONTEXT_SUMMARY_ENTRY: " + str(rolling_summary)
            except Exception:
                pass
            try:
                msgs.insert(int(pin_len0), summary_msg)
            except Exception:
                msgs.append(summary_msg)
            pin.append(summary_msg)
            pin_ids.add(id(summary_msg))

    est = _estimate_prompt_tokens(msgs)
    reserve = int(max(0, int(max_tokens_per_call)) + 1500)
    limit_eff = max(10_000, int(context_tokens) - int(reserve))
    if est <= min(int(trim_at), int(limit_eff)):
        return msgs, str(notebook or "")

    while True:
        est = _estimate_prompt_tokens(msgs)
        if est <= min(int(target), int(limit_eff)):
            break

        pin_len = int(len(pin))
        cut_end = max(pin_len, int(len(msgs) - int(keep_recent)))
        prunable = [m for m in msgs[pin_len:cut_end] if id(m) not in pin_ids]
        if not prunable:
            if keep_recent > 8:
                keep_recent = max(8, int(keep_recent) - 8)
                continue
            msgs = _prune_messages(msgs, pinned=pin, keep_last=12)
            break

        chunk: List[Dict[str, str]] = []
        chunk_tokens = 0
        for m in prunable:
            mt = 10 + _approx_token_count_text(str(m.get("role") or "")) + _approx_token_count_text(str(m.get("content") or ""))
            if chunk and (chunk_tokens + mt) > 18_000:
                break
            chunk.append(m)
            chunk_tokens += int(mt)

        if not chunk:
            msgs = _prune_messages(msgs, pinned=pin, keep_last=12)
            break

        chunk_txt = _messages_to_summary_text(chunk, max_chars=int(chunk_max_chars))
        prev_summary = str(rolling_summary or "").strip()
        if prev_summary:
            summary_prompt = (
                "You maintain a single rolling context summary for an LLM agent. Update the existing summary by incorporating "
                "the NEW EVENTS below. Preserve continuity and avoid duplication. Preserve key facts, decisions, hypotheses, "
                "experiment results (include numbers), constraints/rules, and open questions. Return plain text only.\n\n"
                "EXISTING SUMMARY:\n"
                + prev_summary
                + "\n\nNEW EVENTS:\n"
                + chunk_txt
            )
        else:
            summary_prompt = (
                "Summarize the following earlier conversation history into a compact context that preserves key facts, "
                "decisions, hypotheses, experiment results (include numbers if present), constraints/rules, and open questions. "
                "Be concise but do not omit critical details. Return plain text.\n\n"
                + chunk_txt
            )

        summary_txt = ""
        try:
            summary_txt = llm_generate(
                provider=str(provider),
                model=str(model),
                messages=[_msg("system", "You summarize conversations for future continuity."), _msg("user", summary_prompt)],
                temperature=0.0,
                max_tokens=int(summary_out_tokens),
                timeout_s=float(llm_timeout_s),
                dump_path=None,
            )
        except Exception:
            summary_txt = ""

        summary_txt = str(summary_txt or "").strip()
        if summary_txt:
            rolling_summary = str(summary_txt)
            nb_line = f"step={int(step)} context_summary={_llm_sanitize_text(summary_txt)[:240]}"
            notebook = _notebook_append(str(notebook or ""), nb_line)
            if summary_msg is None:
                summary_msg = _msg("user", "CONTEXT_SUMMARY_ENTRY: " + summary_txt)
                try:
                    msgs.insert(int(pin_len0), summary_msg)
                except Exception:
                    msgs.append(summary_msg)
                pin.append(summary_msg)
                pin_ids.add(id(summary_msg))
            else:
                try:
                    summary_msg["content"] = "CONTEXT_SUMMARY_ENTRY: " + summary_txt
                except Exception:
                    pass

        rm_ids = {id(x) for x in chunk}
        kept: List[Dict[str, str]] = []
        for m in msgs:
            if id(m) in pin_ids:
                kept.append(m)
                continue
            if id(m) in rm_ids:
                continue
            kept.append(m)
        msgs = kept

    out2: List[Dict[str, str]] = []
    for m in pin:
        if m not in out2:
            out2.append(m)
    for m in msgs:
        if m in out2:
            continue
        out2.append(m)
    return out2, str(notebook or "")


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    s0 = str(text or "")
    if not s0.strip():
        return {}

    lines = []
    for ln in s0.splitlines():
        if str(ln).strip().startswith("```"):
            continue
        lines.append(ln)
    s = "\n".join(lines)

    n = int(len(s))
    for start in range(n):
        if s[start] != "{":
            continue

        depth = 0
        in_str = False
        esc = False
        end: Optional[int] = None
        for i in range(start, n):
            ch = s[i]
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
                    end = i + 1
                    break

        if end is None:
            continue

        cand = s[start:end]
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        return obj if isinstance(obj, dict) else {}

    return {}


def _action_has_required_fields(action: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    act = str(action.get("action") or "").strip().lower()
    if act == "call_api":
        if not isinstance(action.get("method"), str) or not str(action.get("method") or "").strip():
            return False
        if not isinstance(action.get("path"), str) or not str(action.get("path") or "").strip():
            return False
        if not isinstance(action.get("query"), dict):
            return False
        if not isinstance(action.get("body"), dict):
            return False
        if not isinstance(action.get("last_result_summary"), str):
            return False
        if not isinstance(action.get("next_step_rationale"), str):
            return False
        return True
    return False


def _repair_action_json_with_llm(
    *,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    bad_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    dump_path: Optional[str],
) -> str:
    bad0 = str(bad_text or "")
    if len(bad0) > 12000:
        bad0 = bad0[:12000]
    repair_prompt = (
        "Your previous assistant message was malformed or incomplete JSON. "
        "Return exactly one valid JSON object matching the Action schema. "
        "Do not use markdown fences. Do not add extra text.\n\n"

        "You must output action=call_api and include: action, method, path, query (object), body (object), "
        "last_result_summary (string), next_step_rationale (string).\n\n"

        "Malformed message to fix:\n"
        + bad0
    )

    msgs2 = list(messages or [])
    msgs2.append(_msg("user", repair_prompt))
    try:
        return llm_generate(
            provider=str(provider),
            model=str(model),
            messages=msgs2,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            timeout_s=float(timeout_s),
            dump_path=str(dump_path or "") or None,
        )
    except Exception:
        return ""


def _parse_counts_csv_mean(csv_text: str, *, max_rows: int = 2000) -> Tuple[List[str], List[float]]:
    s = str(csv_text or "").strip()
    if not s:
        return [], []
    buf = io.StringIO(s)
    reader = csv.reader(buf)
    try:
        hdr = next(reader)
    except Exception:
        return [], []
    if not hdr or len(hdr) < 2:
        return [], []
    genes = [str(x) for x in hdr[1:]]
    sums = [0.0 for _ in genes]
    n = 0
    for row in reader:
        if not row:
            continue
        n += 1
        if n > int(max_rows):
            break
        for i in range(len(genes)):
            try:
                v = float(row[i + 1]) if (i + 1) < len(row) else 0.0
            except Exception:
                v = 0.0
            sums[i] += v
    if n <= 0:
        return genes, [0.0 for _ in genes]
    return genes, [float(x) / float(n) for x in sums]


def _top_k_genes_by_score(genes: List[str], scores: List[float], *, k: int = 10, reverse: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not genes or not scores or len(genes) != len(scores):
        return out
    idx = list(range(len(genes)))
    idx.sort(key=lambda i: float(scores[i]), reverse=bool(reverse))
    for i in idx[: int(k)]:
        out.append({"gene": str(genes[i]), "score": float(scores[i])})
    return out


def _llm_tool_result_compact(
    path: str,
    llm_json: Optional[Dict[str, Any]],
    *,
    omics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if llm_json is None or not isinstance(llm_json, dict):
        return out

    if path == "/api/health":
        ok = bool(llm_json.get("ok") is True)
        return {
            "ok": bool(ok),
            "status": "ok" if ok else "error",
        }

    if path == "/api/discuss":
        out2: Dict[str, Any] = {
            "ok": bool(llm_json.get("ok") is True),
        }
        if "provider" in llm_json:
            out2["provider"] = llm_json.get("provider")
        if "model" in llm_json:
            out2["model"] = llm_json.get("model")
        advice = llm_json.get("advice")
        if isinstance(advice, str) and advice.strip():
            out2["advice"] = str(advice).strip()[:4000]
        out2["note"] = "Advisor returned concise scientific guidance. This endpoint is optional."
        return out2

    if path == "/api/bulk_omics/sets":
        sets0 = llm_json.get("sets")
        sets_out: List[str] = []
        if isinstance(sets0, list):
            sets_out = [str(x) for x in sets0 if str(x).strip()]
        return {
            "ok": bool(llm_json.get("ok") is True),
            "sets": sets_out,
            "note": "Use one of these omics_set strings when calling POST /api/tests/disease/bulk_omics.",
        }

    if path in ("/api/spatial_tx/gene_sets", "/api/spatial_omics/type"):
        gs0 = llm_json.get("types") if path == "/api/spatial_omics/type" else llm_json.get("gene_sets")
        gs_out: List[str] = []
        if isinstance(gs0, list):
            gs_out = [str(x) for x in gs0 if str(x).strip()]
        key = "types" if path == "/api/spatial_omics/type" else "gene_sets"
        return {
            "ok": bool(llm_json.get("ok") is True),
            str(key): gs_out,
            "note": "Use one of these type strings as gene_set when calling POST /api/tests/disease/spatial_omics.",
        }

    if path in ("/api/tests/cancer/models", "/api/tests/hereditary_disease/models", "/api/tests/aging/models"):
        models0 = llm_json.get("models")
        models: Dict[str, str] = {}

        desc = {
            "healthy_organism": "Whole organism (healthy baseline).",
            "healthy_cell_culture": "Cell culture derived from a healthy organism.",
            "cell_culture": "Cell culture derived from a healthy organism.",
            "cancer_organism": "Whole organism (diseased condition).",
            "cancer_cell_culture": "Cell culture derived from a diseased organism.",

            "healthy": "Whole organism (healthy baseline).",
            "cell_culture_healthy": "Cell culture derived from a healthy organism.",
            "cancer": "Whole organism (diseased condition).",
            "cell_culture_cancer": "Cell culture derived from a diseased organism.",
            "disease": "Whole organism (diseased condition).",
            "cell_culture_disease": "Cell culture derived from a diseased organism.",
        }

        if isinstance(models0, list):
            for ent in models0:
                if not isinstance(ent, dict):
                    continue
                k = str(ent.get("key") or "").strip()
                if not k:
                    continue
                k_out = _llm_model_key_to_llm(k)
                models[k_out] = str(desc.get(k) or ent.get("label") or "")

        return {
            "ok": bool(llm_json.get("ok") is True),
            "challenge": llm_json.get("challenge"),
            "models": models,
            "note": "Use the model keys above in API calls.",
        }

    for k in (
        "ok",
        "error",
        "error_kind",
        "experiment",
        "model",
        "days",
        "day",
        "replicates",
        "replicates_completed",
        "replicate_deaths",
        "omics_set",
        "gene_set",
        "win",
        "extra_days",
        "lifespan_recovery_pct",
        "score",
        "score_lifedays_per_usd",
        "delta_median_days",
        "delta_median_ticks",
        "direction",
        "dose",
    ):
        if k in llm_json:
            out[k] = llm_json.get(k)

    if path == "/api/game/state":
        g = llm_json.get("game")
        if isinstance(g, dict):
            out["money_spent_usd"] = g.get("money_spent_usd")
            out["ledger_n"] = len(g.get("ledger")) if isinstance(g.get("ledger"), list) else None
            out["note"] = "Game state: only money spent is shown. Full ledger omitted."
        return out

    if path == "/api/omics/inventory":
        inv_out: Dict[str, Any] = {
            "ok": bool(llm_json.get("ok") is True),
            "player_id": llm_json.get("player_id"),
        }

        inv_msg0 = llm_json.get("llm_message")
        if isinstance(inv_msg0, str) and inv_msg0.strip():
            inv_out["llm_message"] = str(inv_msg0).strip()[:40000]

        files0 = llm_json.get("files")
        if isinstance(files0, list):
            inv_out["files_n"] = int(len(files0))
            files_out: List[Dict[str, Any]] = []
            for ent in files0[:80]:
                if not isinstance(ent, dict):
                    continue
                files_out.append(
                    {
                        "file_id": ent.get("file_id"),
                        "display_name": ent.get("display_name"),
                        "llm_filename": ent.get("llm_filename"),
                        "role": ent.get("role"),
                        "kind": ent.get("kind"),
                        "experiment": ent.get("experiment"),
                        "condition": ent.get("condition"),
                        "replicate": ent.get("replicate"),
                        "run_id": ent.get("run_id"),
                        "bytes": ent.get("bytes"),
                        "download_url": ent.get("download_url"),
                    }
                )
            if files_out:
                inv_out["files"] = files_out
            if len(files0) > 80:
                inv_out["more_files"] = int(len(files0) - 80)

        dsets0 = llm_json.get("datasets")
        if isinstance(dsets0, list):
            inv_out["datasets_n"] = int(len(dsets0))
            dsets_out: List[Dict[str, Any]] = []
            for ent in dsets0[:40]:
                if not isinstance(ent, dict):
                    continue

                file_ids0 = ent.get("file_ids")
                data_ids0 = ent.get("data_file_ids")
                meta_ids0 = ent.get("metadata_file_ids")
                file_ids_n = int(len(file_ids0)) if isinstance(file_ids0, list) else None
                data_file_ids_n = int(len(data_ids0)) if isinstance(data_ids0, list) else None
                metadata_file_ids_n = int(len(meta_ids0)) if isinstance(meta_ids0, list) else None

                dsets_out.append(
                    {
                        "display_name": ent.get("display_name"),
                        "run_id": ent.get("run_id"),
                        "experiment": ent.get("experiment"),
                        "kind": ent.get("kind"),
                        "model": ent.get("model"),
                        "ticks": ent.get("ticks"),
                        "replicates": ent.get("replicates"),
                        "omics_set": ent.get("omics_set"),
                        "gene_set": ent.get("gene_set"),
                        "file_ids_n": file_ids_n,
                        "data_file_ids_n": data_file_ids_n,
                        "metadata_file_ids_n": metadata_file_ids_n,
                    }
                )
            if dsets_out:
                inv_out["datasets"] = dsets_out
            if len(dsets0) > 40:
                inv_out["more_datasets"] = int(len(dsets0) - 40)

        inv_out["note"] = "Use file_id with GET /api/omics/file (query: player_id, file_id). For older datasets, use run_id with GET /api/omics/run to see file_ids, then analyze via POST /api/omics/analyze with file_ids + instructions."
        return inv_out

    if path in ("/api/tests/cancer/estimate_cost", "/api/tests/hereditary_disease/estimate_cost", "/api/tests/aging/estimate_cost"):
        ch = llm_json.get("charge")
        if isinstance(ch, dict):
            kind = ch.get("kind")
            samples = ch.get("samples")
            unit_usd = ch.get("unit_cost_usd")
            total_usd = ch.get("total_cost_usd")
            out["kind"] = kind
            out["samples"] = samples
            out["unit_cost_usd"] = unit_usd
            out["total_cost_usd"] = total_usd
            out["summary"] = f"Estimated cost for {kind}: total ${total_usd} (${unit_usd} per sample) for {samples} samples."
        return out

    if path in ("/api/tests/cancer/proteins", "/api/tests/hereditary_disease/proteins", "/api/tests/aging/proteins"):
        prots = llm_json.get("proteins")
        if isinstance(prots, list):
            out["proteins"] = [str(x) for x in prots[:400]]
        return out

    if path == "/api/omics/analyze":
        if "run_id" in llm_json:
            out["run_id"] = llm_json.get("run_id")

        diag0 = llm_json.get("analysis_diagnostics")
        if isinstance(diag0, dict):
            hits0 = diag0.get("error_hits")
            hits_out: List[Dict[str, Any]] = []
            if isinstance(hits0, list):
                for ent in hits0[:6]:
                    if not isinstance(ent, dict):
                        continue
                    txt0 = ent.get("text")
                    if isinstance(txt0, str) and len(txt0) > 800:
                        txt0 = txt0[:800]
                    hits_out.append({"kind": ent.get("kind"), "text": txt0})
            out["analysis_diagnostics"] = {
                "ok": bool(diag0.get("ok") is True),
                "has_code_execution_errors": bool(diag0.get("has_code_execution_errors") is True),
                "error_hit_count": diag0.get("error_hit_count"),
                "error_hits": hits_out,
            }

        ot = llm_json.get("output_text")
        if isinstance(ot, str) and ot.strip():
            ot2 = str(ot)
            out["output_text_chars"] = int(len(ot2))
            out["output_text"] = ot2
            out["output_text_truncated"] = False

        used = llm_json.get("files")
        if isinstance(used, list) and used:
            used_out: List[Dict[str, Any]] = []
            for ent in used[:60]:
                if not isinstance(ent, dict):
                    continue
                used_out.append(
                    {
                        "display_name": ent.get("display_name"),
                        "file_id": ent.get("file_id"),
                        "name": ent.get("name"),
                        "run_id": ent.get("run_id"),
                    }
                )
            if used_out:
                out["files"] = used_out

        oa = llm_json.get("openai")
        if isinstance(oa, dict):
            out["openai"] = {
                "model": oa.get("model"),
                "memory_limit": oa.get("memory_limit"),
            }

        if "output_text" not in out:
            resp_dump = llm_json.get("response")
            if isinstance(resp_dump, dict):
                out["response"] = resp_dump

        out["note"] = "Analysis response from code interpreter."
        return out

    if path in ("/api/tests/cancer/bulk_omics", "/api/tests/hereditary_disease/bulk_omics", "/api/tests/aging/bulk_omics"):
        if "run_id" in llm_json:
            out["run_id"] = llm_json.get("run_id")
        if "experiment" in llm_json:
            out["experiment"] = llm_json.get("experiment")
        if "model" in llm_json:
            out["model"] = llm_json.get("model")
        if "ticks" in llm_json:
            out["ticks"] = llm_json.get("ticks")
        if "age_days" in llm_json and "days" not in out:
            out["days"] = llm_json.get("age_days")
        if "replicates" in llm_json:
            out["replicates"] = llm_json.get("replicates")
        if "omics_set" in llm_json:
            out["omics_set"] = llm_json.get("omics_set")
        genes = llm_json.get("genes")
        if isinstance(genes, list):
            out["features_n"] = int(len(genes))

        deaths0 = llm_json.get("replicate_deaths")
        if isinstance(deaths0, list) and deaths0:
            out["replicate_deaths_n"] = int(len(deaths0))
            deaths_out: List[Dict[str, Any]] = []
            for ent in deaths0[:50]:
                if not isinstance(ent, dict):
                    continue
                deaths_out.append(
                    {
                        "condition": ent.get("condition"),
                        "model": ent.get("model"),
                        "replicate": ent.get("replicate"),
                        "seed": ent.get("seed"),
                        "requested_ticks": ent.get("requested_ticks"),
                        "death_tick": ent.get("death_tick"),
                        "death_measurement": ent.get("death_measurement"),
                    }
                )
            if deaths_out:
                out["replicate_deaths"] = deaths_out

        inv = llm_json.get("omics_inventory")
        if isinstance(inv, dict):
            inv_url = inv.get("inventory_url")
            inv_msg = str(inv.get("llm_message") or "").strip()
            if inv_url:
                out["omics_inventory_url"] = inv_url
            if inv_msg:
                out["omics_inventory_llm_message"] = inv_msg[:40000]

        out["note"] = "Bulk omics matrices are saved as files. If some replicates died early, see replicate_deaths / replicates_completed. Use omics_inventory_llm_message to find file_id(s), then download via /api/omics/file and analyze via /api/omics/analyze."
        return out

    if path in ("/api/tests/cancer/protein_screen", "/api/tests/hereditary_disease/protein_screen", "/api/tests/aging/protein_screen"):
        if "run_id" in llm_json:
            out["run_id"] = llm_json.get("run_id")
        if "experiment" in llm_json:
            out["experiment"] = llm_json.get("experiment")
        if "model" in llm_json:
            out["model"] = llm_json.get("model")
        if "ticks" in llm_json:
            out["ticks"] = llm_json.get("ticks")
        if "replicates" in llm_json:
            out["replicates"] = llm_json.get("replicates")
        if "interventions" in llm_json:
            out["interventions"] = llm_json.get("interventions")
        inv = llm_json.get("omics_inventory")
        if isinstance(inv, dict):
            inv_url = inv.get("inventory_url")
            inv_msg = str(inv.get("llm_message") or "").strip()
            if inv_url:
                out["omics_inventory_url"] = inv_url
            if inv_msg:
                out["omics_inventory_llm_message"] = inv_msg[:40000]

        out["screened_proteins_n"] = llm_json.get("protein_layers") if isinstance(llm_json.get("protein_layers"), int) else None
        if isinstance(llm_json.get("measurements"), list):
            out["measurements"] = llm_json.get("measurements")
        out["note"] = "Protein screen results are saved as files. Use omics_inventory_llm_message to find file_id(s), then analyze via /api/omics/analyze."
        return out

    if path in ("/api/tests/cancer/claim_cure", "/api/tests/hereditary_disease/claim_cure", "/api/tests/aging/claim_cure"):
        if "run_id" in llm_json:
            out["run_id"] = llm_json.get("run_id")
        if "experiment" in llm_json:
            out["experiment"] = llm_json.get("experiment")
        if "ticks" in llm_json:
            out["ticks"] = llm_json.get("ticks")
        if "replicates" in llm_json:
            out["replicates"] = llm_json.get("replicates")
        if "win" in llm_json:
            out["win"] = llm_json.get("win")
        if "delta_median_ticks" in llm_json:
            out["delta_median_ticks"] = llm_json.get("delta_median_ticks")
        inv = llm_json.get("omics_inventory")
        if isinstance(inv, dict):
            inv_url = inv.get("inventory_url")
            inv_msg = str(inv.get("llm_message") or "").strip()
            if inv_url:
                out["omics_inventory_url"] = inv_url
            if inv_msg:
                out["omics_inventory_llm_message"] = inv_msg[:40000]

        out["note"] = "Claim-cure study outputs are saved as files. Use omics_inventory_llm_message to locate summarized_results + metadata and analyze if needed."
        return out

    return out


_CANCER_WORD_RE = re.compile(r"\bcancer\b", re.IGNORECASE)
_CANCEROUS_WORD_RE = re.compile(r"\bcancerous\b", re.IGNORECASE)


def _llm_sanitize_text(s: str) -> str:
    t = str(s or "")
    if not t:
        return ""
    t = t.replace("SAFETY_CHECK_TYPE_BIO", "SAFETY_CHECK")
    t = t.replace("hereditary_disease", "disease")
    t = t.replace("tests/hereditary_disease", "tests/disease")
    t = t.replace("tests_hereditary_disease", "tests_disease")
    t = t.replace("/hereditary_disease", "/disease")
    t = re.sub(r"\baging\b", "disease", t, flags=re.IGNORECASE)
    t = t.replace("spatial transcriptomics", "spatial_rna")
    t = t.replace("spatial proteomics", "spatial_protein")
    t = t.replace("Spatial transcriptomics", "Spatial RNA")
    t = t.replace("Spatial proteomics", "Spatial protein")
    t = t.replace("tests_cancer_", "tests_disease_")
    t = t.replace("tests_hereditary_disease_", "tests_disease_")
    t = t.replace("tests_aging_", "tests_disease_")
    t = t.replace("healthy_organism", "healthy")
    t = t.replace("cancer_organism", "disease")
    t = t.replace("healthy_cell_culture", "cell_culture_healthy")
    t = t.replace("cancer_cell_culture", "cell_culture_disease")
    t = t.replace("cell_culture_cancer", "cell_culture_disease")
    t = t.replace("/api/spatial_tx/gene_sets", "/api/spatial_omics/type")
    t = t.replace("/api/tests/cancer/spatial_tx", "/api/tests/disease/spatial_omics")
    t = t.replace("/api/tests/cancer/", "/api/tests/disease/")
    t = t.replace("/api/tests/cancer", "/api/tests/disease")
    t = t.replace("/api/tests/hereditary_disease/", "/api/tests/disease/")
    t = t.replace("/api/tests/hereditary_disease", "/api/tests/disease")
    t = t.replace("/api/tests/aging/", "/api/tests/disease/")
    t = t.replace("/api/tests/aging", "/api/tests/disease")
    t = _CANCEROUS_WORD_RE.sub("diseased", t)
    t = _CANCER_WORD_RE.sub("disease", t)
    return t


def _llm_path_to_server(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return p
    if p.startswith("/api/tests/disease/"):
        return _tests_prefix_for_challenge(_BENCH_CHALLENGE) + "/" + p[len("/api/tests/disease/") :]
    if p == "/api/tests/disease":
        return _tests_prefix_for_challenge(_BENCH_CHALLENGE)
    return p


def _llm_body_to_server(server_path: str, body: Any) -> Optional[Dict[str, Any]]:
    if body is None:
        return None
    if not isinstance(body, dict):
        return None

    out = dict(body)

    # Many endpoints require player_id; allow the LLM to omit it.
    pid = str(out.get("player_id") or "").strip()
    if not pid:
        pid2 = str(_BENCH_PLAYER_ID or "").strip()
        if pid2:
            out["player_id"] = pid2

    if server_path.startswith("/api/tests/") and ("model" in out):
        out["model"] = _llm_model_key_to_server(out.get("model"))

    if server_path == "/api/omics/analyze":
        prov = str(out.get("provider") or "").strip()
        if not prov:
            p2 = str(_BENCH_PROVIDER or "").strip()
            if p2:
                out["provider"] = p2
        model0 = str(out.get("model") or "").strip()
        if not model0:
            m2 = str(_BENCH_MODEL or "").strip()
            if m2:
                out["model"] = m2

    return out


def _llm_query_to_server(server_path: str, query: Any) -> Optional[Dict[str, Any]]:
    if query is None:
        out: Dict[str, Any] = {}
    else:
        if not isinstance(query, dict):
            return None
        out = dict(query)

    pid = str(out.get("player_id") or "").strip()
    if not pid:
        pid2 = str(_BENCH_PLAYER_ID or "").strip()
        if pid2:
            out["player_id"] = pid2

    if server_path.startswith("/api/tests/") and ("model" in out):
        out["model"] = _llm_model_key_to_server(out.get("model"))

    return out if out else None


def _server_path_to_llm(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return p
    if p.endswith("/spatial_tx"):
        return _llm_sanitize_text(p[: -len("/spatial_tx")] + "/spatial_omics")
    if p.startswith("/api/tests/cancer/"):
        return "/api/tests/disease/" + p[len("/api/tests/cancer/") :]
    if p.startswith("/api/tests/hereditary_disease/"):
        return "/api/tests/disease/" + p[len("/api/tests/hereditary_disease/") :]
    if p.startswith("/api/tests/aging/"):
        return "/api/tests/disease/" + p[len("/api/tests/aging/") :]
    if p == "/api/tests/cancer":
        return "/api/tests/disease"
    if p == "/api/tests/hereditary_disease":
        return "/api/tests/disease"
    if p == "/api/tests/aging":
        return "/api/tests/disease"
    return _llm_sanitize_text(p)


def _llm_model_key_to_server(model_key: Any) -> str:
    s = str(model_key or "").strip()
    if not s:
        return ""
    if _normalize_challenge(_BENCH_CHALLENGE) == "cancer":
        if s == "healthy":
            return "healthy_organism"
        if s == "cancer" or s == "disease":
            return "cancer_organism"
        if s == "cell_culture_healthy":
            return "healthy_cell_culture"
        if s == "cell_culture_cancer" or s == "cell_culture_disease":
            return "cancer_cell_culture"
        if s == "disease":
            return "cancer_organism"
        if s == "cell_culture_disease":
            return "cancer_cell_culture"
    return s


def _llm_model_key_to_llm(model_key: Any) -> str:
    s = str(model_key or "").strip()
    if not s:
        return ""
    if _normalize_challenge(_BENCH_CHALLENGE) == "cancer":
        if s == "healthy_organism":
            return "healthy"
        if s == "cancer_organism":
            return "disease"
        if s == "healthy_cell_culture":
            return "cell_culture_healthy"
        if s == "cancer_cell_culture":
            return "cell_culture_disease"
        if s == "cancer":
            return "disease"
        if s == "cell_culture_cancer":
            return "cell_culture_disease"
    return _llm_sanitize_text(s)


def _llm_translate_response_obj(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_llm_translate_response_obj(x) for x in obj]
    if isinstance(obj, str):
        return _llm_sanitize_text(obj)
    if not isinstance(obj, dict):
        return obj

    out: Dict[str, Any] = {}
    for k0, v0 in obj.items():
        k = _llm_sanitize_text(str(k0))
        out[k] = _llm_translate_response_obj(v0)
    return out


def _llm_response_json(path: str, response_json: Any) -> Any:
    # Keep runner metrics based on raw server JSON; only translate what the LLM sees.
    try:
        return _llm_translate_response_obj(json.loads(json.dumps(response_json)))
    except Exception:
        return response_json


def _normalize_challenge(challenge: Any) -> str:
    ch = str(challenge or "").strip().lower()
    if ch in ("cancer", "hereditary_disease", "aging"):
        return ch
    return "cancer"


def _tests_prefix_for_challenge(challenge: Any) -> str:
    ch = _normalize_challenge(challenge)
    if ch == "cancer":
        return "/api/tests/cancer"
    if ch == "hereditary_disease":
        return "/api/tests/hereditary_disease"
    return "/api/tests/aging"


def _is_experiment_path(path: str) -> bool:
    p = str(path or "")
    return p in (
        "/api/tests/cancer/bulk_omics",
        "/api/tests/cancer/spatial_tx",
        "/api/tests/cancer/spatial_omics",
        "/api/tests/cancer/characterization",
        "/api/tests/cancer/protein_screen",
        "/api/tests/cancer/claim_cure",
        "/api/tests/hereditary_disease/bulk_omics",
        "/api/tests/hereditary_disease/spatial_tx",
        "/api/tests/hereditary_disease/spatial_omics",
        "/api/tests/hereditary_disease/characterization",
        "/api/tests/hereditary_disease/protein_screen",
        "/api/tests/hereditary_disease/claim_cure",
        "/api/tests/aging/bulk_omics",
        "/api/tests/aging/spatial_tx",
        "/api/tests/aging/spatial_omics",
        "/api/tests/aging/characterization",
        "/api/tests/aging/protein_screen",
        "/api/tests/aging/claim_cure",
    )


def run_benchmark(
    *,
    challenge: str,
    base_url: str,
    provider: str,
    model: str,
    executor_provider: Optional[str] = None,
    executor_model: Optional[str] = None,
    human_poll_s: float = 0.5,
    events_out_path: Optional[str] = None,
    player_id: Optional[str],
    events_fp: Optional[Any],
    files_dir: Optional[str],
    state_out: Optional[str],
    resume_state: Optional[str],
    max_steps: int,
    temperature: float,
    max_tokens: int,
    api_timeout_s: float,
    llm_timeout_s: float,
    reset_first: bool,
    prompt_file: Optional[str] = None,
) -> Tuple[str, BenchMetrics, List[Dict[str, Any]]]:
    resumed = False
    loaded_state: Optional[Dict[str, Any]] = None
    if str(resume_state or "").strip():
        loaded_state = _load_json_file(str(resume_state))
        if isinstance(loaded_state, dict) and loaded_state.get("ok") is True:
            resumed = True

    ch = _normalize_challenge(challenge)

    prompt_text = _PROMPT
    prompt_file_used = ""
    pf_arg = str(prompt_file or "").strip()
    if (not pf_arg) and (not resumed):
        if str(ch) == "aging":
            pf_arg = "aging.txt"
    if resumed and isinstance(loaded_state, dict):
        st_pf = loaded_state.get("prompt_file")
        if isinstance(st_pf, str) and st_pf.strip():
            prompt_file_used = st_pf.strip()
            if pf_arg and pf_arg != prompt_file_used:
                raise RuntimeError("resume_state prompt_file does not match")
        elif pf_arg:
            prompt_file_used = pf_arg
        if isinstance(loaded_state.get("prompt"), str) and str(loaded_state.get("prompt") or "").strip():
            prompt_text = str(loaded_state.get("prompt") or "")
    if not resumed:
        try:
            prompt_text, prompt_file_used = _read_prompt_text(pf_arg)
        except FileNotFoundError:
            raise
    if resumed and (not str(prompt_text or "").strip()):
        try:
            if isinstance(loaded_state, dict) and isinstance(loaded_state.get("messages"), list):
                msgs0 = loaded_state.get("messages")
                if msgs0 and isinstance(msgs0[0], dict) and str(msgs0[0].get("role") or "") == "system":
                    prompt_text = str(msgs0[0].get("content") or "")
        except Exception:
            prompt_text = _PROMPT

    global _BENCH_PROMPT_TEXT
    _BENCH_PROMPT_TEXT = str(prompt_text or "")
    global _BENCH_PROMPT_FILE
    _BENCH_PROMPT_FILE = str(prompt_file_used or "")

    global _BENCH_CHALLENGE
    _BENCH_CHALLENGE = str(ch)

    if resumed:
        st_player = loaded_state.get("player_id")
        if isinstance(st_player, str) and st_player.strip():
            player_id = st_player.strip()
        st_base = loaded_state.get("base_url")
        st_prov = loaded_state.get("provider")
        st_model = loaded_state.get("model")
        st_ch = loaded_state.get("challenge")
        if isinstance(st_base, str) and st_base.strip() and str(base_url).strip() != st_base.strip():
            raise RuntimeError("resume_state base_url does not match")
        if isinstance(st_prov, str) and st_prov.strip() and str(provider).strip() != st_prov.strip():
            raise RuntimeError("resume_state provider does not match")
        if isinstance(st_model, str) and st_model.strip() and str(model).strip() != st_model.strip():
            raise RuntimeError("resume_state model does not match")
        if isinstance(st_ch, str) and st_ch.strip() and _normalize_challenge(st_ch) != _normalize_challenge(ch):
            raise RuntimeError("resume_state challenge does not match")

    if not player_id:
        player_id = _new_player_id()

    global _BENCH_PLAYER_ID
    _BENCH_PLAYER_ID = str(player_id)

    if str(provider or "").strip().lower() in ("openai", "openai_compat") and (not str(model or "").strip()):
        model = "gpt-5.2"

    prov0 = str(provider or "").strip().lower()
    human_mode = bool(prov0 == "human")
    exec_provider = str(executor_provider or os.environ.get("DT_HUMAN_EXECUTOR_PROVIDER") or "").strip().lower()
    if exec_provider == "anthropic":
        exec_provider = "claude"
    if not exec_provider:
        exec_provider = "claude"
    exec_model = str(executor_model or os.environ.get("DT_HUMAN_EXECUTOR_MODEL") or "").strip()
    if not exec_model:
        exec_model = str(model or "").strip()

    # The runner uses these globals when auto-filling /api/omics/analyze params.
    global _BENCH_PROVIDER
    _BENCH_PROVIDER = str(exec_provider if human_mode else provider or "").strip().lower()
    global _BENCH_MODEL
    _BENCH_MODEL = str(exec_model if human_mode else model or "").strip()

    transcript: List[Dict[str, Any]] = []
    notebook = ""
    omics_state: Dict[str, Any] = {}
    messages: List[Dict[str, str]] = []
    seq = 0
    step_start = 0

    human_dir: Optional[Path] = None
    if human_mode:
        human_dir = _human_mode_dir(state_out=state_out, events_out=events_out_path, files_dir=files_dir)
        if human_dir is None:
            raise RuntimeError("provider=human requires --state-out or --events-out or --files-dir to derive a run directory")

    llm_payload_dir: Optional[Path] = None
    try:
        if str(state_out or "").strip():
            llm_payload_dir = Path(str(state_out)).resolve().parent / "llm_payloads"
            llm_payload_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        llm_payload_dir = None

    if resumed:
        try:
            st_metrics = loaded_state.get("metrics") if isinstance(loaded_state.get("metrics"), dict) else {}
            metrics = BenchMetrics(start_ts=float(st_metrics.get("start_ts") or time.time()))
            metrics.llm_calls = int(st_metrics.get("llm_calls") or 0)
            metrics.tool_calls = int(st_metrics.get("tool_calls") or 0)
            metrics.api_calls = int(st_metrics.get("api_calls") or 0)
            metrics.experiment_calls = int(st_metrics.get("experiment_calls") or 0)
            metrics.win = bool(st_metrics.get("win") is True)
            try:
                metrics.final_delta_median_ticks = float(st_metrics.get("final_delta_median_ticks"))
            except Exception:
                metrics.final_delta_median_ticks = None
            try:
                metrics.best_extra_days = float(st_metrics.get("best_extra_days"))
            except Exception:
                metrics.best_extra_days = None
            try:
                metrics.best_lifespan_recovery_pct = float(st_metrics.get("best_lifespan_recovery_pct"))
            except Exception:
                metrics.best_lifespan_recovery_pct = None
            try:
                metrics.best_score = float(st_metrics.get("best_score"))
            except Exception:
                metrics.best_score = None
            try:
                metrics.best_score_lifedays_per_usd = float(st_metrics.get("best_score_lifedays_per_usd"))
            except Exception:
                metrics.best_score_lifedays_per_usd = None
            try:
                metrics.best_score_seq = int(st_metrics.get("best_score_seq"))
            except Exception:
                metrics.best_score_seq = None
            try:
                metrics.money_spent_cents = int(st_metrics.get("money_spent_cents"))
            except Exception:
                metrics.money_spent_cents = None
            try:
                metrics.money_spent_usd = float(st_metrics.get("money_spent_usd"))
            except Exception:
                metrics.money_spent_usd = None
            metrics.experiments = list(st_metrics.get("experiments") or []) if isinstance(st_metrics.get("experiments"), list) else []
        except Exception:
            metrics = BenchMetrics(start_ts=time.time())

        st_messages = loaded_state.get("messages")
        if isinstance(st_messages, list):
            msgs2: List[Dict[str, str]] = []
            for m in st_messages:
                if isinstance(m, dict):
                    r = str(m.get("role") or "")
                    c = _llm_sanitize_text(str(m.get("content") or ""))
                    if r and c is not None:
                        msgs2.append({"role": r, "content": c})
            if msgs2:
                messages = msgs2

        notebook = _llm_sanitize_text(str(loaded_state.get("notebook") or ""))
        st_omics = loaded_state.get("omics_state")
        if isinstance(st_omics, dict):
            omics_state = dict(st_omics)
        try:
            seq = int(loaded_state.get("seq") or 0)
        except Exception:
            seq = 0
        try:
            step_start = int(loaded_state.get("next_step") or 0)
        except Exception:
            step_start = 0

        if not messages:
            messages.append(_msg("system", prompt_text))
            messages.append(_msg("user", f"Harness info: base_url={base_url}"))
            notebook_msg = _msg("user", "LAB_NOTEBOOK:")
            messages.append(notebook_msg)
            if notebook.strip():
                messages.append(_msg("user", notebook.strip()))
        else:
            if len(messages) >= 3 and str(messages[2].get("role")) == "user" and str(messages[2].get("content") or "").startswith("LAB_NOTEBOOK:"):
                notebook_msg = messages[2]
            else:
                notebook_msg = _msg("user", "LAB_NOTEBOOK:")
                messages.insert(2, notebook_msg)

        pinned = [messages[0], messages[1], notebook_msg]
        if human_mode:
            exec_msg = None
            for m in messages:
                if isinstance(m, dict) and str(m.get("role") or "") == "user" and str(m.get("content") or "").startswith("HUMAN_EXECUTOR_MODE:"):
                    exec_msg = m
                    break
            if exec_msg is None:
                exec_msg = _msg(
                    "user",
                    "HUMAN_EXECUTOR_MODE: You are an API executor in human-in-the-loop mode. "
                    "The human provides directives in messages starting with 'HUMAN_DIRECTIVE:'. "
                    "Your job is to translate the latest directive into exactly one valid Action JSON object (call_api) "
                    "matching the schema in the system prompt. Do not add extra strategy beyond the directive. "
                    "If the directive is ambiguous, make one low-cost clarifying API call (e.g., GET /api/game/state or GET /api/tests/disease/proteins).",
                )
                try:
                    messages.insert(3, exec_msg)
                except Exception:
                    messages.append(exec_msg)
            pinned.append(exec_msg)
        seq += 1
        _write_event_line(events_fp, {"seq": seq, "ts": time.time(), "type": "resume", "player_id": player_id, "next_step": step_start})
    else:
        metrics = BenchMetrics(start_ts=time.time())
        # Use a system message for OpenAI-compatible providers.
        messages.append(_msg("system", prompt_text))
        messages.append(_msg("user", f"Harness info: base_url={base_url}"))

        notebook_msg = _msg("user", "LAB_NOTEBOOK:")
        messages.append(notebook_msg)
        pinned = [messages[0], messages[1], notebook_msg]
        if human_mode:
            exec_msg = _msg(
                "user",
                "HUMAN_EXECUTOR_MODE: You are an API executor in human-in-the-loop mode. "
                "The human provides directives in messages starting with 'HUMAN_DIRECTIVE:'. "
                "Your job is to translate the latest directive into exactly one valid Action JSON object (call_api) "
                "matching the schema in the system prompt. Do not add extra strategy beyond the directive. "
                "If the directive is ambiguous, make one low-cost clarifying API call (e.g., GET /api/game/state or GET /api/tests/disease/proteins).",
            )
            try:
                messages.insert(3, exec_msg)
            except Exception:
                messages.append(exec_msg)
            pinned.append(exec_msg)

        seq = 0
        _write_event_line(
            events_fp,
            {
                "seq": seq,
                "ts": time.time(),
                "type": "start",
                "challenge": str(ch),
                "base_url": base_url,
                "provider": provider,
                "model": model,
                "executor_provider": exec_provider if human_mode else None,
                "executor_model": exec_model if human_mode else None,
                "player_id": player_id,
                "prompt": prompt_text,
                "prompt_file": prompt_file_used,
            },
        )
        if str(state_out or "").strip():
            st_metrics_out = {
                "start_ts": float(metrics.start_ts),
                "llm_calls": int(metrics.llm_calls),
                "tool_calls": int(metrics.tool_calls),
                "api_calls": int(metrics.api_calls),
                "experiment_calls": int(metrics.experiment_calls),
                "win": bool(metrics.win),
                "final_delta_median_ticks": metrics.final_delta_median_ticks,
                "best_extra_days": metrics.best_extra_days,
                "best_lifespan_recovery_pct": metrics.best_lifespan_recovery_pct,
                "best_score": metrics.best_score,
                "best_score_lifedays_per_usd": metrics.best_score_lifedays_per_usd,
                "best_score_seq": metrics.best_score_seq,
                "money_spent_cents": metrics.money_spent_cents,
                "money_spent_usd": metrics.money_spent_usd,
                "experiments": metrics.experiments,
            }
            state_obj = {
                "ok": True,
                "challenge": str(ch),
                "prompt_file": str(prompt_file_used or ""),
                "base_url": str(base_url),
                "provider": str(provider),
                "model": str(model),
                "executor_provider": str(exec_provider) if human_mode else "",
                "executor_model": str(exec_model) if human_mode else "",
                "player_id": str(player_id),
                "seq": int(seq),
                "next_step": int(step_start),
                "notebook": str(notebook),
                "omics_state": omics_state,
                "metrics": st_metrics_out,
                "messages": messages,
            }
            _write_json_file(str(state_out), state_obj)

    if reset_first and (not resumed):
        r0 = call_local_api(base_url=base_url, method="POST", path="/api/game/reset", body={"player_id": player_id}, timeout_s=api_timeout_s)
        seq += 1
        transcript.append({"seq": seq, "type": "api", "method": "POST", "path": "/api/game/reset", "result": r0.__dict__})
        _write_event_line(events_fp, {"seq": seq, "ts": time.time(), "type": "api", "method": "POST", "path": "/api/game/reset", "query": None, "body": {"player_id": player_id}, "http_status": r0.http_status, "seconds": r0.seconds, "response_json": r0.response_json})

    if not resumed:
        st0 = call_local_api(
            base_url=base_url,
            method="GET",
            path="/api/game/state",
            query={"player_id": player_id},
            timeout_s=api_timeout_s,
        )
        seq += 1
        transcript.append({"seq": seq, "type": "api", "method": "GET", "path": "/api/game/state", "query": {"player_id": player_id}, "result": st0.__dict__})
        _write_event_line(
            events_fp,
            {
                "seq": seq,
                "ts": time.time(),
                "type": "api",
                "method": "GET",
                "path": "/api/game/state",
                "query": {"player_id": player_id},
                "body": None,
                "http_status": st0.http_status,
                "seconds": st0.seconds,
                "response_json": st0.response_json,
            },
        )

    for step in range(int(step_start), int(max_steps)):
        global _BENCH_STEP
        global _BENCH_MAX_STEPS
        _BENCH_STEP = int(step)
        _BENCH_MAX_STEPS = int(max_steps)
        dump_path: Optional[str] = None
        try:
            if isinstance(llm_payload_dir, Path):
                dump_path = str(llm_payload_dir / f"llm_payload_step_{int(step):06d}.json")
        except Exception:
            dump_path = None

        used_provider = str(provider)
        used_model = str(model)

        if int(step) == (int(max_steps) - 1):
            note = (
                "FINAL_STEP_NOTICE: This is your last move. You must make exactly one final best-guess attempt by calling "
                "POST /api/tests/disease/claim_cure. Do not call any other endpoints."
            )
            try:
                already = any(
                    isinstance(m, dict)
                    and str(m.get("role") or "") == "user"
                    and str(m.get("content") or "").startswith("FINAL_STEP_NOTICE:")
                    for m in (messages or [])
                )
            except Exception:
                already = False
            if not already:
                messages.append(_msg("user", note))
        human_in = None
        if human_mode and isinstance(human_dir, Path):
            seq += 1
            _write_event_line(
                events_fp,
                {
                    "seq": seq,
                    "ts": time.time(),
                    "type": "human_wait",
                    "step": int(step),
                    "player_id": str(player_id),
                },
            )
            try:
                human_in = _human_wait_for_input(
                    human_dir=human_dir,
                    step=int(step),
                    player_id=str(player_id),
                    base_url=str(base_url),
                    poll_s=float(human_poll_s),
                    error=None,
                )
            except Exception as e:
                human_in = {"ok": False, "error": str(e)}

            if isinstance(human_in, dict) and human_in.get("stop") is True:
                seq += 1
                _write_event_line(events_fp, {"seq": seq, "ts": time.time(), "type": "human_stop", "step": int(step)})
                transcript.append({"type": "human_stop", "step": int(step), "payload": human_in})
                break

            seq += 1
            _write_event_line(
                events_fp,
                {
                    "seq": seq,
                    "ts": time.time(),
                    "type": "human_input",
                    "step": int(step),
                    "text": str((human_in or {}).get("text") or "")[:8000],
                    "has_action_json": bool(isinstance((human_in or {}).get("action_json"), dict)),
                },
            )

        try:
            if human_mode and isinstance(human_in, dict) and isinstance(human_in.get("action_json"), dict):
                # Treat as if the assistant produced this JSON (keeps downstream parsing/validation uniform).
                try:
                    if isinstance(human_dir, Path):
                        try:
                            (human_dir / "decisions").mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass
                        _write_json_file(
                            str((human_dir / "decisions" / f"human_action_step_{int(step):06d}.json")),
                            {"step": int(step), "action_json": human_in.get("action_json"), "ts": float(time.time())},
                        )
                except Exception:
                    pass
                out = json.dumps(human_in.get("action_json"), ensure_ascii=False)
                used_provider = "human"
                used_model = "human"
                if str(dump_path or "").strip():
                    try:
                        _write_json_file(
                            str(dump_path),
                            {
                                "provider": "human",
                                "step": int(step),
                                "action_json": human_in.get("action_json"),
                                "human_text": str(human_in.get("text") or "")[:12000],
                            },
                        )
                    except Exception:
                        pass
            else:
                if human_mode:
                    used_provider = str(exec_provider)
                    used_model = str(exec_model)
                    txt = str((human_in or {}).get("text") or "").strip()
                    if not txt:
                        txt = "(no human directive provided; request clarification)"
                    messages.append(_msg("user", "HUMAN_DIRECTIVE: " + txt))
                    nb_line = f"step={int(step)} human_directive={_llm_sanitize_text(txt)[:240]}"
                    notebook = _notebook_append(notebook, nb_line)
                    messages.append(_msg("user", "LAB_NOTEBOOK_ENTRY: " + nb_line))

                metrics.llm_calls += 1
                out = llm_generate(
                    provider=str(used_provider),
                    model=str(used_model),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=llm_timeout_s,
                    dump_path=dump_path,
                )
        except Exception as e:
            err = _llm_sanitize_text(str(e))
            messages.append(_msg("user", f"TOOL_RESULT: {{\"error\": \"LLM_CALL_FAILED: {err}\"}}"))
            transcript.append({"type": "llm_error", "error": str(e)})
            seq += 1
            _write_event_line(events_fp, {"seq": seq, "ts": time.time(), "type": "llm_error", "error": str(e)})
            break

        out_raw = out
        out_effective = out

        action0 = _extract_first_json_object(out_effective)
        if not _action_has_required_fields(action0 if isinstance(action0, dict) else {}):
            dump_path_repair = None
            try:
                if str(dump_path or "").strip():
                    dump_path_repair = str(Path(str(dump_path)).with_name(f"llm_repair_step_{int(step):06d}.json"))
            except Exception:
                dump_path_repair = None
            repaired = _repair_action_json_with_llm(
                provider=str(used_provider),
                model=str(used_model),
                messages=messages,
                bad_text=str(out_raw or ""),
                temperature=0.0,
                max_tokens=max(256, min(1200, int(max_tokens))),
                timeout_s=float(llm_timeout_s),
                dump_path=dump_path_repair,
            )
            if str(repaired or "").strip():
                action1 = _extract_first_json_object(repaired)
                if _action_has_required_fields(action1 if isinstance(action1, dict) else {}):
                    out_effective = str(repaired)

        seq += 1
        transcript.append({"seq": seq, "type": "llm", "step": step, "text": out_effective})
        action_preview = _extract_first_json_object(out_effective)
        lrs_preview = action_preview.get("last_result_summary") if isinstance(action_preview, dict) else None
        nsr_preview = action_preview.get("next_step_rationale") if isinstance(action_preview, dict) else None
        ev_obj = {
            "seq": seq,
            "ts": time.time(),
            "type": "llm",
            "step": step,
            "text": out_effective,
            "provider_used": str(used_provider),
            "model_used": str(used_model),
            "last_result_summary": lrs_preview,
            "next_step_rationale": nsr_preview,
        }
        if str(out_effective or "") != str(out_raw or ""):
            ev_obj["text_raw"] = str(out_raw or "")[:8000]
        _write_event_line(events_fp, ev_obj)

        out_llm_safe = _llm_sanitize_text(out_effective)
        messages.append(_msg("assistant", out_llm_safe))

        action = _extract_first_json_object(out_effective)
        if not action or "action" not in action:
            messages.append(
                _msg(
                    "user",
                    "TOOL_RESULT: {\"error\": \"Invalid response. You must output exactly one JSON object matching the Action schema.\"}",
                )
            )
            continue

        act = str(action.get("action") or "").strip().lower()
        if act != "call_api":
            if act == "final":
                messages.append(_msg("user", "TOOL_RESULT: {\"error\": \"FINAL_ACTION_NOT_SUPPORTED: do not output action=final. The harness stops automatically on win=true or step limit. Use call_api.\"}"))
            else:
                messages.append(_msg("user", "TOOL_RESULT: {\"error\": \"Unknown action. Use call_api.\"}"))
            continue

        lrs = action.get("last_result_summary")
        nsr = action.get("next_step_rationale")
        if not isinstance(lrs, str) or not isinstance(nsr, str):
            messages.append(
                _msg(
                    "user",
                    "TOOL_RESULT: {\"error\": \"Missing required fields: last_result_summary (string) and next_step_rationale (string).\"}",
                )
            )
            continue

        nb_line = f"step={int(step)} agent_summary={_llm_sanitize_text(str(lrs).strip())[:240]} agent_plan={_llm_sanitize_text(str(nsr).strip())[:240]}"
        notebook = _notebook_append(notebook, nb_line)
        messages.append(_msg("user", "LAB_NOTEBOOK_ENTRY: " + nb_line))

        method = str(action.get("method") or "").strip().upper()
        llm_path = str(action.get("path") or "").strip()
        query = action.get("query") if isinstance(action.get("query"), dict) else None
        body = action.get("body") if isinstance(action.get("body"), dict) else None

        if llm_path == "/api/game/state" and method != "GET":
            messages.append(_msg("user", "TOOL_RESULT: {\"error\": \"/api/game/state must be called with method=GET.\"}"))
            continue

        if llm_path in ("/api/tests/disease/protein_layers", "/api/tests/cancer/protein_layers"):
            messages.append(
                _msg(
                    "user",
                    "TOOL_RESULT: {\"error\": "
                    "\"Use /api/tests/disease/proteins (biological terminology) instead of /api/tests/disease/protein_layers.\"}",
                )
            )
            continue

        server_path = _llm_path_to_server(llm_path)
        body = _llm_body_to_server(server_path, body)
        query = _llm_query_to_server(server_path, query)

        if method == "POST":
            if isinstance(query, dict) and query:
                if body is None:
                    body = {}
                try:
                    for k, v in query.items():
                        if k not in body:
                            body[k] = v
                except Exception:
                    pass
            query = None

        expected_pid = str(player_id or "").strip()
        pid_mismatch: Optional[Dict[str, Any]] = None
        try:
            if isinstance(query, dict) and ("player_id" in query):
                qpid = str(query.get("player_id") or "").strip()
                if qpid and expected_pid and qpid != expected_pid:
                    pid_mismatch = {"source": "query", "found": qpid, "expected": expected_pid}
                    query = dict(query)
                    query["player_id"] = expected_pid
            if isinstance(body, dict) and ("player_id" in body):
                bpid = str(body.get("player_id") or "").strip()
                if bpid and expected_pid and bpid != expected_pid:
                    pid_mismatch = {"source": "body", "found": bpid, "expected": expected_pid}
                    body = dict(body)
                    body["player_id"] = expected_pid
        except Exception:
            pid_mismatch = None

        if isinstance(pid_mismatch, dict):
            seq += 1
            _write_event_line(
                events_fp,
                {
                    "seq": seq,
                    "ts": time.time(),
                    "type": "player_id_mismatch",
                    "step": int(step),
                    "path": str(server_path),
                    "method": str(method),
                    "details": dict(pid_mismatch),
                },
            )

        if method == "POST" and server_path in ("/api/tests/cancer/claim_cure", "/api/tests/hereditary_disease/claim_cure", "/api/tests/aging/claim_cure"):
            if body is None:
                body = {}
            body["ticks"] = 400

        if method == "POST" and server_path in ("/api/tests/cancer/protein_screen", "/api/tests/hereditary_disease/protein_screen", "/api/tests/aging/protein_screen"):
            if body is None:
                body = {}
            body["worker_mode"] = "process"
            try:
                workers_i = int(body.get("workers") or 35)
            except Exception:
                workers_i = 35
            if workers_i < 35:
                body["workers"] = 35

        if method == "POST" and server_path in ("/api/tests/cancer/estimate_cost", "/api/tests/hereditary_disease/estimate_cost", "/api/tests/aging/estimate_cost"):
            if isinstance(body, dict):
                exp0 = str(body.get("experiment") or "").strip().lower()
                if exp0 in ("claim_cure", "cure"):
                    body["ticks"] = 400

        if method not in ("GET", "POST") or not server_path.startswith("/api/"):
            messages.append(_msg("user", "TOOL_RESULT: {\"error\": \"Bad method or path.\"}"))
            continue

        metrics.tool_calls += 1
        metrics.api_calls += 1
        if _is_experiment_path(server_path):
            metrics.experiment_calls += 1
        res = call_local_api_retrying(
            base_url=base_url,
            method=method,
            path=server_path,
            query=query,
            body=body,
            timeout_s=api_timeout_s,
            max_attempts=3,
        )


        # Track experiments and win if claim_cure was called.
        if res.response_json and isinstance(res.response_json, dict):
            exp = res.response_json.get("experiment")
            if isinstance(exp, str) and exp:
                metrics.experiments.append(exp)
            g0 = res.response_json.get("game")
            if isinstance(g0, dict):
                try:
                    metrics.money_spent_cents = int(g0.get("money_spent_cents"))
                except Exception:
                    pass
                try:
                    metrics.money_spent_usd = float(g0.get("money_spent_usd"))
                except Exception:
                    pass
            if server_path in ("/api/tests/cancer/claim_cure", "/api/tests/hereditary_disease/claim_cure", "/api/tests/aging/claim_cure"):
                try:
                    metrics.final_delta_median_ticks = float(res.response_json.get("delta_median_ticks"))
                except Exception:
                    metrics.final_delta_median_ticks = None
                metrics.win = bool(res.response_json.get("win") is True)
                try:
                    extra = float(res.response_json.get("extra_days"))
                except Exception:
                    extra = None
                if extra is not None:
                    if metrics.best_extra_days is None or float(extra) > float(metrics.best_extra_days):
                        metrics.best_extra_days = float(extra)
                try:
                    rec = res.response_json.get("lifespan_recovery_pct")
                    rec_f = float(rec) if rec is not None else None
                except Exception:
                    rec_f = None
                if rec_f is not None and (metrics.best_lifespan_recovery_pct is None or float(rec_f) > float(metrics.best_lifespan_recovery_pct)):
                    metrics.best_lifespan_recovery_pct = float(rec_f)
                try:
                    score = res.response_json.get("score")
                    score_f = float(score) if score is not None else None
                except Exception:
                    score_f = None
                if score_f is None:
                    try:
                        s2 = res.response_json.get("score_lifedays_per_usd")
                        s2f = float(s2) if s2 is not None else None
                    except Exception:
                        s2f = None
                    if s2f is not None:
                        score_f = float(s2f) * 10000.0
                if score_f is not None and (metrics.best_score is None or float(score_f) > float(metrics.best_score)):
                    metrics.best_score = float(score_f)
                    metrics.best_score_seq = int(seq)
                try:
                    score_lpu = res.response_json.get("score_lifedays_per_usd")
                    score_lpu_f = float(score_lpu) if score_lpu is not None else None
                except Exception:
                    score_lpu_f = None
                if score_lpu_f is not None and (metrics.best_score_lifedays_per_usd is None or float(score_lpu_f) > float(metrics.best_score_lifedays_per_usd)):
                    metrics.best_score_lifedays_per_usd = float(score_lpu_f)

        seq += 1
        resp_summary, files = _summarize_response_json_for_events(res.response_json, seq=seq, files_dir=files_dir)
        res_for_transcript = {
            "http_status": res.http_status,
            "response_json": resp_summary if files_dir else res.response_json,
            "response_text": res.response_text,
            "seconds": res.seconds,
        }
        transcript.append({"seq": seq, "type": "api", "method": method, "path": server_path, "query": query, "body": body, "result": res_for_transcript, "files": files})
        if files:
            for a in files:
                a["seq"] = seq

        _write_event_line(
            events_fp,
            {
                "seq": seq,
                "ts": time.time(),
                "type": "api",
                "method": method,
                "path": server_path,
                "query": query,
                "body": body,
                "http_status": res.http_status,
                "seconds": res.seconds,
                "response_json": resp_summary,
                "files": files,
            },
        )

        messages.append(_msg("user", "LAB_NOTEBOOK_ENTRY: " + nb_line))

        compact = resp_summary
        if compact is None:
            compact = res.response_json

        llm_compact = None
        try:
            llm_compact = _llm_tool_result_compact(str(server_path), res.response_json if isinstance(res.response_json, dict) else None, omics_state=omics_state)
        except Exception:
            llm_compact = None
        if not llm_compact:
            llm_compact = compact

        tool_payload = {
            "http_status": res.http_status,
            "seconds": res.seconds,
            "response_json": llm_compact,
        }
        if res.response_json is None:
            tool_payload["response_text"] = _llm_sanitize_text(res.response_text[:400])
        messages.append(_msg("user", "TOOL_RESULT: " + json.dumps(tool_payload, ensure_ascii=False)))
        messages, notebook = _maybe_prune_messages_with_summary(
            messages=messages,
            pinned=pinned,
            provider=str(exec_provider if human_mode else provider),
            model=str(exec_model if human_mode else model),
            llm_timeout_s=float(llm_timeout_s),
            step=int(step),
            max_tokens_per_call=int(max_tokens),
            notebook=str(notebook),
        )
        seq += 1
        tool_payload_event = {
            "http_status": res.http_status,
            "seconds": res.seconds,
            "response_json": compact,
            "api_response_json_summary": resp_summary,
        }
        if res.response_json is None:
            tool_payload_event["response_text"] = res.response_text[:2000]
        _write_event_line(
            events_fp,
            {
                "seq": seq,
                "ts": time.time(),
                "type": "tool_result",
                "tool": "call_api",
                "path": server_path,
                "payload": tool_payload_event,
            },
        )
        if str(state_out or "").strip():
            st_metrics_out = {
                "start_ts": float(metrics.start_ts),
                "llm_calls": int(metrics.llm_calls),
                "tool_calls": int(metrics.tool_calls),
                "api_calls": int(metrics.api_calls),
                "experiment_calls": int(metrics.experiment_calls),
                "win": bool(metrics.win),
                "final_delta_median_ticks": metrics.final_delta_median_ticks,
                "best_extra_days": metrics.best_extra_days,
                "best_lifespan_recovery_pct": metrics.best_lifespan_recovery_pct,
                "best_score": metrics.best_score,
                "best_score_lifedays_per_usd": metrics.best_score_lifedays_per_usd,
                "best_score_seq": metrics.best_score_seq,
                "money_spent_cents": metrics.money_spent_cents,
                "money_spent_usd": metrics.money_spent_usd,
                "experiments": metrics.experiments,
            }
            state_obj = {
                "ok": True,
                "challenge": str(ch),
                "base_url": str(base_url),
                "provider": str(provider),
                "model": str(model),
                "executor_provider": str(exec_provider) if human_mode else "",
                "executor_model": str(exec_model) if human_mode else "",
                "player_id": str(player_id),
                "seq": int(seq),
                "next_step": int(step) + 1,
                "notebook": str(notebook),
                "omics_state": omics_state,
                "metrics": st_metrics_out,
                "messages": messages,
            }
            _write_json_file(str(state_out), state_obj)

        if (
            server_path
            in (
                "/api/tests/cancer/claim_cure",
                "/api/tests/hereditary_disease/claim_cure",
                "/api/tests/aging/claim_cure",
            )
            and metrics.win
        ):
            transcript.append({"type": "auto_stop", "reason": "win_true"})
            break

    st1 = call_local_api(
        base_url=base_url,
        method="GET",
        path="/api/game/state",
        query={"player_id": player_id},
        timeout_s=api_timeout_s,
    )
    seq += 1
    transcript.append({"seq": seq, "type": "api", "method": "GET", "path": "/api/game/state", "query": {"player_id": player_id}, "result": st1.__dict__})
    _write_event_line(events_fp, {"seq": seq, "ts": time.time(), "type": "api", "method": "GET", "path": "/api/game/state", "query": {"player_id": player_id}, "body": None, "http_status": st1.http_status, "seconds": st1.seconds, "response_json": st1.response_json})
    if st1.response_json and isinstance(st1.response_json.get("game"), dict):
        g = st1.response_json.get("game")
        try:
            metrics.money_spent_cents = int(g.get("money_spent_cents"))
        except Exception:
            metrics.money_spent_cents = None
        try:
            metrics.money_spent_usd = float(g.get("money_spent_usd"))
        except Exception:
            metrics.money_spent_usd = None

    metrics.end_ts = time.time()
    seq += 1
    _write_event_line(
        events_fp,
        {
            "seq": seq,
            "ts": time.time(),
            "type": "end",
            "win": bool(metrics.win),
            "final_delta_median_ticks": metrics.final_delta_median_ticks,
            "best_extra_days": metrics.best_extra_days,
            "best_lifespan_recovery_pct": metrics.best_lifespan_recovery_pct,
            "best_score": metrics.best_score,
            "best_score_lifedays_per_usd": metrics.best_score_lifedays_per_usd,
            "best_score_seq": metrics.best_score_seq,
            "money_spent_cents": metrics.money_spent_cents,
            "money_spent_usd": metrics.money_spent_usd,
            "llm_calls": metrics.llm_calls,
            "tool_calls": metrics.tool_calls,
            "api_calls": metrics.api_calls,
            "experiment_calls": metrics.experiment_calls,
        },
    )
    return str(player_id), metrics, transcript


def main() -> int:
    try:
        if apply_keys_to_environ is not None:
            apply_keys_to_environ()
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="LLM benchmark runner for the disease challenge (tool-using biology).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="Runtime server base URL.")
    ap.add_argument(
        "--provider",
        choices=["openai", "anthropic", "claude", "human", "xai", "gemini"],
        help="LLM provider. Use 'human' for human-in-the-loop mode.",
    )
    ap.add_argument("--model", default="", help="Model name (provider-specific).")
    ap.add_argument("--executor-provider", default="", help="When --provider=human, use this LLM provider to translate directives into Action JSON.")
    ap.add_argument("--executor-model", default="", help="When --provider=human, use this model for the executor LLM (defaults to --model).")
    ap.add_argument("--human-poll", type=float, default=0.5, help="When --provider=human, poll interval (seconds) while waiting for human input.")
    ap.add_argument("--challenge", default="cancer", choices=["cancer", "hereditary_disease", "aging"], help="Which test challenge to run (server-side).")
    ap.add_argument("--player-id", default="", help="Optional fixed player_id (for reproducible runs).")
    ap.add_argument("--max-steps", type=int, default=40, help="Max tool-using steps.")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature.")
    ap.add_argument("--max-tokens", type=int, default=8000, help="Max tokens to generate per LLM call (provider-dependent).")
    ap.add_argument("--api-timeout", type=float, default=5000.0, help="Timeout for local API calls.")
    ap.add_argument("--llm-timeout", type=float, default=5000.0, help="Timeout for LLM API calls.")
    ap.add_argument("--reset-first", action="store_true", help="Reset game state for the benchmark player_id before starting.")
    ap.add_argument("--out", default="", help="Write JSON report to this file.")
    ap.add_argument("--events-out", default="", help="Optional JSONL stream of live events for monitoring.")
    ap.add_argument("--files-dir", default="", help="Optional directory to write CSV files for monitoring/UI.")
    ap.add_argument("--artifacts-dir", default="", help="(legacy) Alias for --files-dir.")
    ap.add_argument("--state-out", default="", help="Optional JSON checkpoint file for resuming a run.")
    ap.add_argument("--resume-state", default="", help="Resume a run from a previous --state-out checkpoint.")
    ap.add_argument("--prompt-file", default="", help="Optional prompt file name (relative to assets/prompts/) or absolute path.")
    ap.add_argument("--print-prompt", action="store_true", help="Print the benchmark prompt and exit.")
    ap.add_argument("--run-id", default="", help="Optional run_id label for tracking (e.g. run_...).")
    ap.add_argument("--suite-id", default="", help="Optional suite_id label for tracking (e.g. suite_...).")

    args = ap.parse_args()

    if (not args.print_prompt) and (not str(args.provider or "").strip()):
        ap.error("the following arguments are required: --provider")
    if (not args.print_prompt) and (not str(args.model or "").strip()):
        ap.error("the following arguments are required: --model")

    if args.print_prompt:
        pf0 = str(args.prompt_file or "").strip()
        if not pf0:
            try:
                if _normalize_challenge(getattr(args, "challenge", "")) == "aging":
                    pf0 = "aging.txt"
            except Exception:
                pf0 = ""
        p_txt, _ = _read_prompt_text(str(pf0))
        sys.stdout.write(str(p_txt) + "\n")
        return 0

    events_fp = None
    if str(args.events_out or "").strip():
        ev_dir = os.path.dirname(os.path.abspath(str(args.events_out)))
        if ev_dir:
            os.makedirs(ev_dir, exist_ok=True)
        events_fp = open(str(args.events_out), "a", encoding="utf-8")

    files_dir = str(args.files_dir or "").strip() or None
    if not files_dir:
        files_dir = str(args.artifacts_dir or "").strip() or None

    used_player_id, metrics, transcript = run_benchmark(
        challenge=str(args.challenge),
        base_url=str(args.base_url),
        provider=str(args.provider),
        model=str(args.model),
        executor_provider=str(args.executor_provider or "").strip() or None,
        executor_model=str(args.executor_model or "").strip() or None,
        human_poll_s=float(args.human_poll),
        events_out_path=str(args.events_out or "").strip() or None,
        player_id=str(args.player_id or "").strip() or None,
        events_fp=events_fp,
        files_dir=files_dir,
        state_out=str(args.state_out or "").strip() or None,
        resume_state=str(args.resume_state or "").strip() or None,
        max_steps=int(args.max_steps),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        api_timeout_s=float(args.api_timeout),
        llm_timeout_s=float(args.llm_timeout),
        reset_first=bool(args.reset_first),
        prompt_file=str(args.prompt_file or "").strip() or None,
    )

    if events_fp is not None:
        try:
            events_fp.close()
        except Exception:
            pass

    def _git_cmd(args2: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args2, cwd=str(_repo_root()), stderr=subprocess.STDOUT)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    def _pkg_ver(name: str) -> Optional[str]:
        try:
            import importlib.metadata as _imd

            return str(_imd.version(str(name)))
        except Exception:
            return None

    git_commit = _git_cmd(["git", "rev-parse", "HEAD"])
    git_dirty = None
    try:
        st = _git_cmd(["git", "status", "--porcelain"])
        if st is not None:
            git_dirty = bool(str(st).strip())
    except Exception:
        git_dirty = None

    run_id = str(args.run_id or "").strip()
    if (not run_id) and str(args.out or "").strip():
        try:
            run_id = Path(str(args.out)).resolve().parent.name
        except Exception:
            run_id = ""

    suite_id = str(args.suite_id or "").strip()
    env_keys_present = {
        "OPENAI_API_KEY": bool(str(os.environ.get("OPENAI_API_KEY") or "").strip()),
        "ANTHROPIC_API_KEY": bool(str(os.environ.get("ANTHROPIC_API_KEY") or "").strip()),
        "GEMINI_API_KEY": bool(str(os.environ.get("GEMINI_API_KEY") or "").strip()),
        "XAI_API_KEY": bool(str(os.environ.get("XAI_API_KEY") or "").strip()),
    }

    report = {
        "schema_version": 1,
        "ok": True,
        "run_id": run_id,
        "suite_id": suite_id,
        "prompt": str(_BENCH_PROMPT_TEXT or _PROMPT),
        "prompt_file": str(_BENCH_PROMPT_FILE or ""),
        "challenge": str(args.challenge),
        "provider": str(args.provider),
        "model": str(args.model),
        "executor_provider": str(args.executor_provider or "").strip(),
        "executor_model": str(args.executor_model or "").strip(),
        "human_poll": float(args.human_poll),
        "base_url": str(args.base_url),
        "player_id": str(used_player_id),
        "files_dir": files_dir,
        "artifacts_dir": files_dir,
        "metrics": {
            "llm_calls": metrics.llm_calls,
            "tool_calls": metrics.tool_calls,
            "api_calls": metrics.api_calls,
            "experiment_calls": metrics.experiment_calls,
            "seconds_total": float(metrics.end_ts - metrics.start_ts) if metrics.end_ts and metrics.start_ts else None,
            "win": bool(metrics.win),
            "final_delta_median_ticks": metrics.final_delta_median_ticks,
            "best_extra_days": metrics.best_extra_days,
            "best_lifespan_recovery_pct": metrics.best_lifespan_recovery_pct,
            "best_score": metrics.best_score,
            "best_score_lifedays_per_usd": metrics.best_score_lifedays_per_usd,
            "best_score_seq": metrics.best_score_seq,
            "money_spent_cents": metrics.money_spent_cents,
            "money_spent_usd": metrics.money_spent_usd,
            "experiments": metrics.experiments,
        },
        "meta": {
            "created_ts": float(time.time()),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "pid": int(os.getpid()),
            "python_executable": str(sys.executable),
            "python_version": str(sys.version),
            "platform": str(platform.platform()),
            "machine": str(platform.machine()),
            "processor": str(platform.processor()),
            "cpu_count": int(os.cpu_count() or 0),
            "git": {"commit": git_commit, "dirty": git_dirty},
            "env_keys_present": env_keys_present,
            "packages": {
                "openai": _pkg_ver("openai"),
                "anthropic": _pkg_ver("anthropic"),
            },
        },
        "transcript": transcript,
    }

    had_error_issue = False

    try:
        evp = str(args.events_out or "").strip()
        if evp:
            run_dir = Path(str(evp)).resolve().parent
            issues_path = run_dir / "issues.json"
            story_path = run_dir / "story.md"
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            issues: List[Dict[str, Any]] = []
            issues.extend(_detect_issues_from_events_path(events_path=Path(str(evp)).resolve()))
            try:
                if stderr_path.exists() and stderr_path.is_file():
                    issues.extend(_issues_from_text(text=stderr_path.read_text(encoding="utf-8", errors="replace"), source="runner_stderr", max_items=200))
            except Exception:
                pass
            try:
                if stdout_path.exists() and stdout_path.is_file():
                    issues.extend(_issues_from_text(text=stdout_path.read_text(encoding="utf-8", errors="replace"), source="runner_stdout", max_items=200))
            except Exception:
                pass

            seen = set()
            deduped: List[Dict[str, Any]] = []
            for it in issues:
                if not isinstance(it, dict):
                    continue
                k = (str(it.get("severity") or ""), str(it.get("kind") or ""), str(it.get("source") or ""), str(it.get("seq") or ""), str(it.get("summary") or ""))
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(it)
            try:
                had_error_issue = any(str((it or {}).get("severity") or "").strip().lower() == "error" for it in list(deduped))
            except Exception:
                had_error_issue = False
            _write_json_any_file(str(issues_path), deduped)

            try:
                story_txt = _story_markdown_from_events_path(events_path=Path(str(evp)).resolve())
                if story_txt.strip():
                    story_path.write_text(story_txt, encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass

    report["ok"] = bool(report.get("ok") is True and (not had_error_issue))

    raw = json.dumps(report, indent=2)
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(str(args.out)))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(str(args.out), "w", encoding="utf-8") as f:
            f.write(raw)

    sys.stdout.write(raw + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
