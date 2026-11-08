"""Parse Kuku-format screenplays into timed events.

Format (per episode .txt):
    EPISODE NUMBER / EPISODE LOGLINE / CONTINUITY NOTES header
    SCENE nn / Scene heading / Scene start / Scene end
    blocks:  **CAMERA:** | **ACTION:** | **SOUND FX:** | **SPEAKER (tone):**
             Start: (mm:ss.ss) | End: (mm:ss.ss)
             text...
All times are relative to the start of the episode.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

TIME = r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?"
_RANGE_RE = re.compile(rf"Start:\s*\(?{TIME}\)?\s*\|\s*End:\s*\(?{TIME}\)?", re.I)
_BLOCK_RE = re.compile(r"^\*\*(.+?):?\*\*\s*$")
_SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
_SPEAKER_RE = re.compile(r"^(?P<name>[^(]+?)\s*(?:\((?P<tone>.*)\))?\s*:?$")
NON_SPEECH = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC"}


def to_seconds(mm: str, ss: str, frac: str | None) -> float:
    return int(mm) * 60 + int(ss) + (float(f"0.{frac}") if frac else 0.0)


def fmt_time(sec: float) -> str:
    """Seconds -> mm:ss.s (matches the source files, sans hundredths noise)."""
    sec = max(0.0, sec)
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:04.1f}"


@dataclass
class Event:
    kind: str            # dialogue | action | camera | sfx
    start: float
    end: float
    text: str
    speaker: str = ""
    tone: str = ""
    scene: int = 0

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    source: str = ""

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)

    @property
    def is_continuation(self) -> bool:
        return self.number > 1 and bool(re.search(r"pick|follow|continu|resum", self.continuity, re.I))


def _kind(label: str) -> str:
    up = label.upper()
    if up == "CAMERA":
        return "camera"
    if up == "ACTION":
        return "action"
    if up in {"SOUND FX", "SFX", "MUSIC"}:
        return "sfx"
    return "dialogue"


def parse_episode(text: str, source: str = "") -> Episode:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    num_m = re.search(r"EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", text, re.I)
    log_m = re.search(r"EPISODE LOGLINE:\s*(.+)", text, re.I)
    con_m = re.search(r"CONTINUITY NOTES:\s*(.+)", text, re.I)
    if not num_m:
        fm = re.search(r"episode[-_ ]?(\d+)", source, re.I)
        number = int(fm.group(1)) if fm else 0
    else:
        number = int(num_m.group(1))
    ep = Episode(number, log_m.group(1).strip() if log_m else "",
                 con_m.group(1).strip() if con_m else "", source=source)

    scene_no, cur = 0, None
    i, n = 0, len(lines)
    in_script = False
    while i < n:
        ln = lines[i].strip()
        if re.match(r"^---\s*SCRIPT START\s*---", ln, re.I):
            in_script = True
        elif re.match(r"^---\s*SCRIPT END\s*---", ln, re.I):
            break
        if not in_script:
            i += 1
            continue
        sm = _SCENE_RE.match(ln)
        if sm:
            scene_no = int(sm.group(1))
            heading, s0, s1 = "", 0.0, 0.0
            j = i + 1
            while j < n and j < i + 8:
                l2 = lines[j].strip()
                if _BLOCK_RE.match(l2):  # first content block: stop, don't swallow it
                    break
                hm = re.match(r"Scene heading:\s*(.+)", l2, re.I)
                st = re.match(rf"Scene start:\s*{TIME}", l2, re.I)
                en = re.match(rf"Scene end:\s*{TIME}", l2, re.I)
                if hm:
                    heading = hm.group(1).strip()
                if st:
                    s0 = to_seconds(*st.groups())
                if en:
                    s1 = to_seconds(*en.groups())
                j += 1
            ep.scenes.append(Scene(scene_no, heading, s0, s1))
            i = j
            continue
        bm = _BLOCK_RE.match(ln)
        if bm and i + 1 < n:
            rm = _RANGE_RE.search(lines[i + 1])
            if rm:
                g = rm.groups()
                start, end = to_seconds(*g[0:3]), to_seconds(*g[3:6])
                j = i + 2
                body = []
                while j < n and lines[j].strip() and not _BLOCK_RE.match(lines[j].strip()):
                    body.append(lines[j].strip())
                    j += 1
                label = bm.group(1).strip()
                kind = _kind(label.split("(")[0].strip())
                speaker = tone = ""
                if kind == "dialogue":
                    sp = _SPEAKER_RE.match(label)
                    speaker = (sp.group("name") if sp else label).strip().upper()
                    tone = (sp.group("tone") or "").strip() if sp else ""
                ep.events.append(Event(kind, start, end, " ".join(body).strip().strip('"'),
                                       speaker, tone, scene_no))
                i = j
                continue
        i += 1
    ep.events.sort(key=lambda e: (e.start, e.end))
    return ep


def load_pack(path: str | Path) -> list[Episode]:
    """Load every episode from a folder of .txt files or a .zip of them."""
    path = Path(path)
    items: list[tuple[str, str]] = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith(".txt") and not name.startswith("__MACOSX"):
                    items.append((name, z.read(name).decode("utf-8", "replace")))
    else:
        for f in sorted(path.rglob("*.txt")):
            items.append((f.name, f.read_text(encoding="utf-8", errors="replace")))
    return finish(items)


def finish(items: list[tuple[str, str]]) -> list[Episode]:
    eps = [parse_episode(t, s) for s, t in items]
    eps = [e for e in eps if e.events]
    eps.sort(key=lambda e: e.number)
    return eps
