"""Rule-based, explainable promo scoring (no API needed).

Every feature is a transparent heuristic over *timed* screenplay events, so each
score can be traced to a specific line. Lexicons are matched on word stems with
word boundaries (the old version matched substrings, so "no" fired on "know").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import Event

HOOK_WINDOW = 3.0       # seconds: the brief's "first 3 seconds"
CLIFF_WINDOW = 6.0      # seconds examined at the end of a clip


def _rx(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\w*", re.I)


# --- lexicons (genre: historical romance / revenge microdrama) -------------
SHOCK = _rx(["slap", "strike", "struck", "blood", "bleed", "dead", "death", "die", "kill", "murder",
             "poison", "dagger", "sword", "blade", "arrest", "seiz", "drag", "collaps", "scream",
             "shriek", "gasp", "betray", "traitor", "liar", "lie[sd]?", "lied", "fraud", "impost",
             "secret", "reveal", "truth", "duke", "king", "prince", "heir", "royal", "pregnan",
             "ruin", "destroy", "banish", "disgrace", "curse", "thief", "steal", "stole", "forge",
             "execut", "hang", "burn", "fire", "explod", "shatter", "crown"])
STAKES = _rx(["marry", "marriage", "wed", "bride", "groom", "altar", "divorce", "annul", "child", "baby",
              "son", "daughter", "mother", "father", "sister", "wife", "husband", "love", "heart",
              "life", "family", "inherit", "fortune", "title", "estate", "oath", "vow", "promise"])
THREAT = _rx(["or else", "or you", "i will", "i'll", "you will", "you'll", "never", "last chance", "choose",
              "ultimatum", "regret", "pay for", "beg", "kneel", "tonight", "by dawn", "before", "unless",
              "if you", "consequence", "mercy", "no choice"])
INTENSE_TONE = _rx(["scream", "shout", "furious", "venom", "threat", "cruel", "fierce", "outrag", "accus",
                    "gloat", "taunt", "mock", "spiteful", "scorn", "incredul", "shock", "horrif", "panick",
                    "desperat", "sobb", "weep", "defiant", "chilling", "cold", "sinister", "enrag", "seething",
                    "hysteric", "triumphant", "menac", "stunned", "gasp"])
SOFT_TONE = _rx(["gentle", "tender", "soft", "calm", "reassur", "sincere", "warm", "relieved", "grateful",
                 "content", "peaceful", "loving", "serene"])
RESOLVE = _rx(["thank you", "thanks", "i forgive", "forgive", "goodbye", "farewell", "at last", "finally",
               "it is over", "it's over", "safe now", "all is well", "together", "happily", "peace"])
CLIFF_SFX = _rx(["cliffhanger", "cut to black", "freeze", "froze", "sting", "abrupt", "unresolved", "silence",
                 "ominous", "sudden", "thunder", "swell", "hold", "snap", "gasp", "crash", "scream"])
CLIFF_ACT = _rx(["freez", "stare", "stunned", "pale", "gasp", "horrif", "enter", "burst", "appear", "reveal",
                 "step", "door", "unseen", "lock", "eyes", "ring", "letter", "blood", "dagger", "gun",
                 "sword", "collaps", "drop", "turn"])
BACKREF = _rx(["as i said", "as before", "again", "still", "earlier", "last night", "back then", "remember",
               "that day", "i told you", "once more", "anymore", "any more"])
RELATION = _rx(["my (?:wife|husband|sister|brother|mother|father|daughter|son|bride|fianc\\w+|lord|lady)",
                "your (?:wife|husband|sister|brother|mother|father|daughter|son|bride)", "marry", "wedding",
                "altar", "betroth", "engage"])
CONNECTOR = re.compile(r"^(\.{2,}|…|and |but |then |so |because |which |or |yet |instead )", re.I)
FILLER_START = re.compile(r"^(yes|no|why|what|who|how|please|stop|wait|oh|ah|hm+|well|very well|i see|"
                          r"indeed|perhaps|very|then|so)\W*$", re.I)


@dataclass
class Scored:
    hook: float
    cliff: float
    pacing: float
    standalone: float
    duration_fit: float
    hook_line: str
    cliff_line: str
    reasons: list[str] = field(default_factory=list)


def clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def _content(events: list[Event]) -> list[Event]:
    return [e for e in events if e.kind in ("dialogue", "action")]


def _norm(sc: float) -> float:
    return round(clamp(sc), 1)


def score_hook(window: list[Event], t0: float) -> tuple[float, str, list[str]]:
    early = [e for e in window if e.kind in ("dialogue", "action", "sfx") and e.start < t0 + HOOK_WINDOW]
    cont = _content(early)
    reasons: list[str] = []
    if not cont:
        return 1.0, "", ["no spoken or visible action in the first 3s"]
    first = cont[0]
    score = 2.5
    text = " ".join(e.text for e in cont)
    if first.kind == "dialogue":
        score += 1.5
        reasons.append("~opens on a spoken line")
        if FILLER_START.match(first.text.strip()) or len(first.text.split()) <= 1:
            score -= 2.0
            reasons.append("first line is a one-word reply (weak cold open)")
    n_shock = len(set(m.group(0).lower() for m in SHOCK.finditer(text)))
    if n_shock:
        score += min(3.0, 1.2 * n_shock)
        reasons.append("shock vocabulary in first 3s")
    if STAKES.search(text):
        score += 0.8
    if THREAT.search(text):
        score += 1.0
        reasons.append("threat/ultimatum")
    if any(INTENSE_TONE.search(e.tone) for e in cont if e.kind == "dialogue"):
        score += 1.2
        reasons.append("high-intensity delivery")
    if "!" in text:
        score += 0.8
    if "?" in first.text and first.kind == "dialogue" and len(first.text.split()) >= 4:
        score += 0.6
    if first.kind == "action" and SHOCK.search(first.text):
        score += 1.0
    return _norm(score), first.text, reasons


def score_cliff(window: list[Event], t1: float, ep_end: float, next_ev: Event | None) -> tuple[float, str, list[str]]:
    tail = [e for e in window if e.end > t1 - CLIFF_WINDOW and e.kind != "camera"]
    cont = _content(tail)
    reasons: list[str] = []
    if not cont:
        return 1.0, "", ["no content at the end of the clip"]
    last = cont[-1]
    score = 2.5
    ttext = " ".join(e.text for e in tail)
    sfx_txt = " ".join(e.text for e in tail if e.kind == "sfx")
    if CLIFF_SFX.search(sfx_txt) or re.search(r"cliffhanger|cut to black|freezes", ttext, re.I):
        score += 2.5
        reasons.append("scripted cliffhanger/freeze/sting cue")
    if last.kind == "dialogue":
        t = last.text.strip()
        if t.endswith("?"):
            score += 2.0
            reasons.append("ends on an unanswered question")
        if t.endswith(("...", "—", "-", "…")):
            score += 1.5
            reasons.append("line cut off mid-thought")
        if THREAT.search(t):
            score += 1.5
            reasons.append("ends on a threat/ultimatum")
        if SHOCK.search(t):
            score += 1.2
        if INTENSE_TONE.search(last.tone):
            score += 1.0
    else:
        if CLIFF_ACT.search(last.text):
            score += 1.8
            reasons.append("ends on a visual beat (entrance/reaction/object)")
    if abs(ep_end - t1) < 1.5:
        score += 1.5
        reasons.append("lands on the episode's own cliffhanger")
    if RESOLVE.search(ttext):
        score -= 2.5
        reasons.append("resolution language near the end (weakens cliffhanger)")
    if last.kind == "dialogue" and SOFT_TONE.search(last.tone):
        score -= 1.5
        reasons.append("soft/closing tone at the cut")
    if next_ev is None and abs(ep_end - t1) >= 1.5:
        score -= 0.5
    return _norm(score), last.text if last.kind == "dialogue" else f"[{last.text}]", reasons


def score_pacing(window: list[Event], t0: float, t1: float) -> tuple[float, list[str]]:
    dur = max(1.0, t1 - t0)
    dlg = [e for e in window if e.kind == "dialogue"]
    talk = sum(e.dur for e in dlg) / dur
    speakers = {e.speaker for e in dlg}
    turns = sum(1 for a, b in zip(dlg, dlg[1:]) if a.speaker != b.speaker)
    c = _content(window)
    gaps = [b.start - a.end for a, b in zip(c, c[1:])]
    max_gap = max(gaps) if gaps else 0.0
    s = 2.0 + 5.0 * min(talk / 0.5, 1.0)
    s += min(2.0, turns * 0.35)
    if len(speakers) < 2:
        s -= 1.5
    if max_gap > 5:
        s -= min(3.0, (max_gap - 5) * 0.5)
    r = [f"{turns} speaker turns, {int(talk*100)}% of runtime is speech"]
    return _norm(s), r


def score_standalone(window: list[Event], ep_continuation: bool, t0: float, scene_starts: set[float],
                     ep_number: int) -> tuple[float, list[str]]:
    reasons: list[str] = []
    s = 5.0
    text = " ".join(e.text for e in window if e.kind in ("dialogue", "action"))
    spk = {e.speaker for e in window if e.kind == "dialogue"}
    if min((abs(t0 - x) for x in scene_starts), default=9) < 1.0:
        s += 1.5
        reasons.append("starts at a scene boundary (self-contained setup)")
    elif ep_continuation and t0 < 8:
        s -= 1.5
        reasons.append("opens mid-continuation of the previous episode")
    if len(spk) >= 2:
        s += 1.0
    first_txt = next((e.text for e in window if e.kind == "dialogue"), "")
    if CONNECTOR.match(first_txt.strip()):
        s -= 1.5
        reasons.append("first line starts mid-sentence/mid-argument (needs prior context)")
    if RELATION.search(text):
        s += 1.5
        reasons.append("states the relationship/stakes explicitly")
    if BACKREF.search(text):
        s -= 1.5
        reasons.append("refers back to earlier events")
    if ep_number == 1 and t0 < 30:
        s += 1.0
        reasons.append("series opener: premise is established on screen")
    return _norm(s), reasons


def duration_fit(d: float) -> float:
    """Best 25-45s (short-form sweet spot); legal range 20-60s."""
    if 25 <= d <= 45:
        return 10.0
    if d < 25:
        return clamp(10 - (25 - d) * 1.2)
    return clamp(10 - (d - 45) * 0.5)
