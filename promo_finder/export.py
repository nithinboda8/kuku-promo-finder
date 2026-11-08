"""Write the clip sheet (CSV / JSON / XLSX)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .pipeline import Promo, promos_table


def to_json(promos: list[Promo]) -> str:
    return json.dumps([p.to_dict() for p in promos], indent=2, ensure_ascii=False, default=str)


def to_csv(promos: list[Promo]) -> str:
    return pd.DataFrame(promos_table(promos)).to_csv(index=False)


def to_xlsx(promos: list[Promo], path: str | Path, stats: dict | None = None, config: dict | None = None) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    df = pd.DataFrame(promos_table(promos))
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Ranked Promos", index=False)
        ws = xw.sheets["Ranked Promos"]
        widths = {"Rank": 6, "Episode": 9, "Segments (start-end)": 20, "Duration (s)": 11, "Hook": 42,
                  "Cliffhanger": 42, "Why it works": 55}
        for i, col in enumerate(df.columns, 1):
            ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 14)
            c = ws.cell(1, i)
            c.font, c.fill = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1E293B")
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.font = Font(name="Arial", size=10)
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        tx = pd.DataFrame([{"Rank": p.rank, "Episode": p.episode, "Start": round(p.segments[0].start, 1),
                            "End": round(p.segments[0].end, 1), "Transcript": p.transcript} for p in promos])
        tx.to_excel(xw, sheet_name="Clip Transcripts", index=False)
        w2 = xw.sheets["Clip Transcripts"]
        w2.column_dimensions["E"].width = 120
        for c in w2[1]:
            c.font = Font(name="Arial", bold=True)
        meta = [("Item", "Value")] + [(k, v) for k, v in (stats or {}).items()] + [(k, v) for k, v in (config or {}).items()]
        pd.DataFrame(meta[1:], columns=meta[0]).to_excel(xw, sheet_name="Run Summary", index=False)
        xw.sheets["Run Summary"].column_dimensions["A"].width = 32
        xw.sheets["Run Summary"].column_dimensions["B"].width = 60
