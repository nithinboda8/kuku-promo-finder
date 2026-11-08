"""Optional LLM re-scoring layer. Provider-agnostic and stdlib-only.

The rule-based scorer finds candidates cheaply; an LLM (if configured) re-judges only the
shortlisted clips for the things rules are bad at: is the hook *actually* shocking, does the
cut leave a real open question, does it make sense to a cold viewer.

Configure with environment variables (nothing is hard-coded to one vendor):
    PROMO_LLM_PROVIDER = openai | anthropic       (openai = any OpenAI-compatible endpoint:
                                                   OpenAI, Gemini's compat endpoint, OpenRouter, Groq, local)
    PROMO_LLM_MODEL    = model name
    PROMO_LLM_API_KEY  = key
    PROMO_LLM_BASE_URL = optional, e.g. https://generativelanguage.googleapis.com/v1beta/openai
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .parser import Episode
from .pipeline import Promo, Weights

SYSTEM = (
    "You are a short-form video editor choosing promos for vertical microdrama (TikTok/Reels/Shorts). "
    "A good promo is 20-60s, OPENS on a hook in the first 3 seconds (slap, reveal, shocking line), "
    "ENDS on an unresolved cliffhanger, and makes sense to someone who has never seen the show. "
    "Judge ONLY from the transcript given. Reply with JSON only."
)
USER_TMPL = """Show logline for this episode: {logline}

Clip transcript (episode {ep}, {t0}-{t1}):
{transcript}

Return JSON with keys:
  "hook": 0-10 (strength of the first 3 seconds),
  "cliffhanger": 0-10 (does the cut leave a real unanswered question?),
  "standalone": 0-10 (clear to a cold viewer?),
  "hook_line": the exact line/action that hooks,
  "cliffhanger_line": the exact final line/action,
  "why": one sentence on why it works or why it doesn't."""


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        key = os.getenv("PROMO_LLM_API_KEY", "")
        model = os.getenv("PROMO_LLM_MODEL", "")
        if not (key and model):
            return None
        return cls(os.getenv("PROMO_LLM_PROVIDER", "openai").lower(), model, key,
                   os.getenv("PROMO_LLM_BASE_URL", ""))


def _post(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_llm(cfg: LLMConfig, system: str, user: str) -> str:
    if cfg.provider == "anthropic":
        url = (cfg.base_url or "https://api.anthropic.com") + "/v1/messages"
        out = _post(url, {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"},
                    {"model": cfg.model, "max_tokens": 500, "system": system,
                     "messages": [{"role": "user", "content": user}]})
        return "".join(b.get("text", "") for b in out.get("content", []))
    url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    out = _post(url, {"Authorization": f"Bearer {cfg.api_key}"},
                {"model": cfg.model, "temperature": 0,
                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    return out["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _num(x, default: float) -> float:
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return default


def rerank(promos: list[Promo], episodes: list[Episode], cfg: LLMConfig | None = None,
           weights: Weights | None = None, caller: Callable[[str, str], str] | None = None,
           blend: float = 0.6) -> int:
    """Blend LLM judgement into each promo (in place). Returns how many were updated.

    `caller(system, user) -> str` can be injected for tests or another SDK.
    blend = weight of the LLM score vs the rule score. Failures leave the rule score untouched.
    """
    w = (weights or Weights()).normalised()
    by_ep = {e.number: e for e in episodes}
    if caller is None:
        if cfg is None:
            return 0
        caller = lambda s, u: call_llm(cfg, s, u)  # noqa: E731
    done = 0
    for p in promos:
        seg = p.segments[0]
        prompt = USER_TMPL.format(logline=by_ep[seg.episode].logline, ep=seg.episode,
                                  t0=round(seg.start, 1), t1=round(seg.end, 1), transcript=p.transcript[:3500])
        try:
            data = parse_json(caller(SYSTEM, prompt))
        except Exception:  # network/auth/rate-limit: keep the rule-based score
            continue
        if not data:
            continue
        h = blend * _num(data.get("hook"), p.hook_score) + (1 - blend) * p.hook_score
        c = blend * _num(data.get("cliffhanger"), p.cliff_score) + (1 - blend) * p.cliff_score
        a = blend * _num(data.get("standalone"), p.standalone_score) + (1 - blend) * p.standalone_score
        p.hook_score, p.cliff_score, p.standalone_score = round(h, 1), round(c, 1), round(a, 1)
        p.composite = round(h * w.hook + c * w.cliff + a * w.standalone + p.pacing_score * w.pacing
                            + p.duration_fit * w.duration, 2)
        p.hook_line = str(data.get("hook_line") or p.hook_line)
        p.cliff_line = str(data.get("cliffhanger_line") or p.cliff_line)
        p.why = str(data.get("why") or p.why)
        p.scorer = "heuristic+llm"
        done += 1
    promos.sort(key=lambda x: -x.composite)
    for i, p in enumerate(promos, 1):
        p.rank = i
    return done
