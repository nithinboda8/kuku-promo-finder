#!/usr/bin/env python3
"""CLI: python run_pipeline.py PACK_DIR_OR_ZIP [--out outputs] [--llm]"""
import argparse
from pathlib import Path

from promo_finder.export import to_csv, to_json, to_xlsx
from promo_finder.llm import LLMConfig, rerank
from promo_finder.parser import load_pack
from promo_finder.pipeline import Weights, find_promos


def main() -> None:
    ap = argparse.ArgumentParser(description="Find 20-60s promo clips in timestamped microdrama screenplays.")
    ap.add_argument("pack", help="folder of episode .txt files, or a .zip")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--per-episode", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=6.5)
    ap.add_argument("--min-hook", type=float, default=6.0)
    ap.add_argument("--min-cliff", type=float, default=6.0)
    ap.add_argument("--llm", action="store_true", help="re-score the shortlist with an LLM (see promo_finder/llm.py)")
    a = ap.parse_args()

    episodes = load_pack(a.pack)
    w = Weights()
    promos, stats = find_promos(episodes, w, per_ep=a.per_episode, min_comp=a.min_score,
                              min_hook=a.min_hook, min_cliff=a.min_cliff)
    if a.llm:
        cfg = LLMConfig.from_env()
        if cfg is None:
            raise SystemExit("Set PROMO_LLM_API_KEY and PROMO_LLM_MODEL to use --llm.")
        stats["llm_rescored"] = rerank(promos, episodes, cfg, w)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(to_json(promos), encoding="utf-8")
    (out / "promos.csv").write_text(to_csv(promos), encoding="utf-8")
    to_xlsx(promos, out / "promos.xlsx", stats, {"per_episode": a.per_episode, "min_score": a.min_score, "min_hook": a.min_hook, "min_cliff": a.min_cliff,
                                                  "weights": str(w.normalised())})
    print(f"{stats['episodes']} episodes, {stats['candidates']} candidate clips scored, "
          f"{len(promos)} promos selected -> {out}/")


if __name__ == "__main__":
    main()
