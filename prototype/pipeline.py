"""Candidate generation, selection and ranking."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .parser import Episode, Event, fmt_time
from . import scoring as S

MIN_DUR, MAX_DUR = 20.0, 60.0


@dataclass
class Weights:
    hook: float = 0.30
    cliff: float = 0.35
    standalone: float = 0.15
    pacing: float = 0.15
    duration: float = 0.05

    def normalised(self) -> "Weights":
        t = self.hook + self.cliff + self.standalone + self.pacing + self.duration
        t = t or 1.0
        return Weights(*(getattr(self, k) / t for k in ("hook", "cliff", "standalone", "pacing", "duration")))


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Promo:
    segments: list[Segment]
    hook_score: float
    cliff_score: float
    standalone_score: float
    pacing_score: float
    duration_fit: float
    composite: float
    hook_line: str
    cliff_line: str
    why: str
    reasons: list[str] = field(default_factory=list)
    transcript: str = ""
    scorer: str = "heuristic"
    rank: int = 0

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def episode(self) -> int:
        return self.segments[0].episode

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 1)
        return d


def candidates_for_episode(ep: Episode, w: Weights) -> list[Promo]:
    ev = ep.events
    content = [e for e in ev if e.kind in ("dialogue", "action")]
    scene_starts = {s.start for s in ep.scenes}
    ep_end = ep.duration
    out: list[Promo] = []
    ends = sorted({e.end for e in ev})
    for st in content:
        t0 = st.start
        for t1 in ends:
            d = t1 - t0
            if d < MIN_DUR:
                continue
            if d > MAX_DUR:
                break
            win = [e for e in ev if e.end > t0 and e.start < t1]
            # the last block must be content or the cue that sits on it, not a bare camera move
            tail = [e for e in win if e.end >= t1 - 0.01]
            if not any(e.kind != "camera" for e in tail):
                continue
            nxt = next((e for e in content if e.start >= t1 - 0.01), None)
            h, hl, hr = S.score_hook(win, t0)
            c, cl, cr = S.score_cliff(win, t1, ep_end, nxt)
            p, pr = S.score_pacing(win, t0, t1)
            a, ar = S.score_standalone(win, ep.is_continuation, t0, scene_starts, ep.number)
            df = S.duration_fit(d)
            comp = h * w.hook + c * w.cliff + a * w.standalone + p * w.pacing + df * w.duration
            reasons = [r.lstrip("~") for r in hr + cr + ar + pr]
            out.append(Promo([Segment(ep.number, t0, t1)], h, c, a, p, df, round(comp, 2), hl, cl,
                             _why(h, c, a, p, hr, cr), reasons,
                             " ".join(f"{_nice(e.speaker)}: {e.text}" if e.kind == "dialogue" else f"[{e.text}]"
                                      for e in win if e.kind in ("dialogue", "action"))))
    return out


def _nice(name: str) -> str:
    return " ".join(w.capitalize() if "'" not in w else w[0] + w[1:].lower() for w in name.split())


def _why(h, c, a, p, hr, cr) -> str:
    def best(rs):
        rs = sorted(rs, key=lambda r: r.startswith("~"))  # generic reasons last
        return rs[0].lstrip("~") if rs else ""

    bits = []
    if best(hr):
        bits.append(f"Hook {h:.1f}: {best(hr)}")
    if best(cr):
        bits.append(f"cliffhanger {c:.1f}: {best(cr)}")
    if not bits:
        bits.append("balanced pacing and setup")
    return "; ".join(bits)


def _overlaps(a: Promo, b: Promo, pad: float = 0.0) -> bool:
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode and x.start < y.end + pad and y.start < x.end + pad:
                return True
    return False


def select_per_episode(cands: list[Promo], per_ep: int, min_comp: float, min_hook: float,
                       min_cliff: float) -> list[Promo]:
    pool = [c for c in cands if c.composite >= min_comp and c.hook_score >= min_hook and c.cliff_score >= min_cliff]
    pool.sort(key=lambda c: -c.composite)
    chosen: list[Promo] = []
    for c in pool:
        if len(chosen) >= per_ep:
            break
        if all(not _overlaps(c, k, pad=5.0) for k in chosen):
            chosen.append(c)
    return chosen


def find_promos(episodes: list[Episode], weights: Weights | None = None, per_ep: int = 2,
                min_comp: float = 6.5, min_hook: float = 6.0, min_cliff: float = 6.0,
                ) -> tuple[list[Promo], dict]:
    w = (weights or Weights()).normalised()
    promos: list[Promo] = []
    stats = {"episodes": len(episodes), "candidates": 0, "episodes_with_promo": 0}
    for ep in episodes:
        cands = candidates_for_episode(ep, w)
        stats["candidates"] += len(cands)
        picked = select_per_episode(cands, per_ep, min_comp, min_hook, min_cliff)
        if picked:
            stats["episodes_with_promo"] += 1
        promos.extend(picked)
    promos.sort(key=lambda p: -p.composite)
    for i, p in enumerate(promos, 1):
        p.rank = i
    return promos, stats


def promos_table(promos: list[Promo]) -> list[dict]:
    rows = []
    for p in promos:
        s = p.segments
        rows.append({
            "Rank": p.rank,
            "Episode": s[0].episode if len(s) == 1 else ",".join(str(x.episode) for x in s),
            "Segments (start-end)": " + ".join(f"{fmt_time(x.start)}-{fmt_time(x.end)}" for x in s),
            "Duration (s)": round(p.duration, 1),
            "Hook": p.hook_line,
            "Cliffhanger": p.cliff_line,
            "Why it works": p.why,
            "Hook score": p.hook_score,
            "Cliffhanger score": p.cliff_score,
            "Standalone score": p.standalone_score,
            "Pacing score": p.pacing_score,
            "Composite": p.composite,
            "Scorer": p.scorer,
        })
    return rows


def tension_curve(ep: Episode, w: Weights | None = None) -> list[tuple[float, float]]:
    """(start_time, best composite of any 20-60s clip starting there) for plotting one episode."""
    w = (w or Weights()).normalised()
    best: dict[float, float] = {}
    for c in candidates_for_episode(ep, w):
        t = c.segments[0].start
        best[t] = max(best.get(t, 0.0), c.composite)
    return sorted(best.items())
