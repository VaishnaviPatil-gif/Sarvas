"""
backend/app/qualcomm/service.py
Qualcomm Cloud AI Playground integration: plain-language explanation of
vessel attribution. Legal text is templated in code, never LLM-generated.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx

# Fill all three from the Playground docs/account. No guessed defaults.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

API_URL = os.getenv("QUALCOMM_API_URL", "https://aisuite.cirrascale.com/apis/v2/chat/completions")
API_KEY = os.getenv("QUALCOMM_API_KEY", "")
MODEL = os.getenv("QUALCOMM_MODEL", "Llama-3.3-70B")

CACHE_DIR = Path(os.getenv("QUALCOMM_CACHE_DIR", "data/qualcomm_cache"))

# Neutral, legally accurate statutory reference without unverified section claims
LEGAL_BASIS = os.getenv(
    "SARVAS_LEGAL_BASIS",
    "Applicable Indian merchant shipping and marine pollution statutes / UNCLOS Part XII"
)

SYSTEM_PROMPT = (
    "You write short attribution briefs for a maritime oil-spill investigation tool. "
    "Rules: use ONLY the data in the message. Never mention a vessel that is not listed. "
    "Never invent or estimate numbers. Do not expand abbreviations. "
    "Do not cite laws or penalties. Do not say a vessel is guilty or that any act was "
    "deliberate or illegal. Plain text only, no markdown symbols."
)


def _lead_strength(total_score: float) -> str:
    if total_score >= 70:
        return "strong"
    if total_score >= 40:
        return "moderate"
    return "weak"


def _clean(d: dict) -> dict:
    """Drop empty fields so the model cannot speculate about them."""
    return {k: v for k, v in d.items() if v is not None and v != ""}


def _build_prompt(spill: dict, candidates: list[dict]) -> str:
    """
    Each candidate must carry "factors": the same six rows the dashboard shows,
    each as {"name", "weight_percent", "score_0_to_100", "contribution", "evidence"}.
    """
    top = candidates[0]
    sorted_factors = sorted(top.get("factors", []), key=lambda f: f.get("contribution", 0.0), reverse=True)
    payload = {
        "spill": _clean(spill),
        "candidate_count": len(candidates),
        "lead_strength": _lead_strength(top["total_score"]),
        "top_two_by_contribution": sorted_factors[:2],
        "weakest_two_by_contribution": sorted_factors[-2:],
        "candidates": [_clean(c) for c in candidates],
    }
    only_one = len(candidates) == 1
    comparison = (
        "2. State that this is the only candidate, so no comparison is possible."
        if only_one
        else "2. One sentence per other listed candidate on why it ranks lower."
    )
    return (
        "Data:\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Write four short numbered sections:\n"
        "1. Start with 'This is a <lead_strength> lead.' Then name the two factors in "
        "top_two_by_contribution, with their scores and contributions.\n"
        f"{comparison}\n"
        "3. Two or three inspection checks, each tied to a factor that scored above 0.\n"
        "4. Name the two factors in weakest_two_by_contribution and say they weaken the attribution."
    )


def _cache_path(spill_id: str) -> Path:
    return CACHE_DIR / f"{spill_id}.json"


def _save_cache(spill_id: str, result: Dict[str, Any]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(spill_id).write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Failed to write Qualcomm cache: {e}")


def _load_cache(spill_id: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(spill_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


async def explain_attribution(spill: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a dict whose "source" is strictly truthful:
      "qualcomm_cloud_ai" : live response from Qualcomm Cloud AI Playground API
      "cached"            : earlier live response, retaining original timestamp & model
      "unavailable"       : API not configured or unreachable, no cached copy
    """
    spill_id = str(spill.get("id") or spill.get("name") or "unknown")
    error = "Qualcomm Cloud AI Playground API not configured in environment"

    if API_URL and API_KEY and MODEL:
        try:
            started = time.perf_counter()
            # 20s timeout so live calls complete swiftly without stalling the demo
            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": _build_prompt(spill, candidates)},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 300,
                    },
                )
            res.raise_for_status()
            payload = res.json()

            text = payload["choices"][0]["message"]["content"]
            latency_ms = round((time.perf_counter() - started) * 1000)

            result = {
                "source": "qualcomm_cloud_ai",
                "model": MODEL,
                "explanation": text,
                "latency_ms": latency_ms,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": None
            }
            _save_cache(spill_id, result)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] Qualcomm API call failed: {error}")

    cached = _load_cache(spill_id)
    if cached:
        return {
            **cached,
            "source": "cached",
            "is_stale_warning": "Serving previously validated response from local cache"
        }

    return {
        "source": "unavailable",
        "model": None,
        "explanation": None,
        "latency_ms": None,
        "generated_at": None,
        "error": error
    }
