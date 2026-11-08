"""Kuku promo finder - Streamlit UI.  Run:  streamlit run app.py"""
import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from promo_finder.export import to_csv, to_json, to_xlsx
from promo_finder.llm import LLMConfig, rerank
from promo_finder.parser import finish, load_pack
from promo_finder.pipeline import Weights, find_promos, promos_table, tension_curve
from promo_finder.parser import fmt_time

st.set_page_config(page_title="Kuku Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Kuku Promo Finder")
st.caption("Reads timestamped microdrama screenplays and proposes 20-60s promo clips: hook in the first 3s, "
           "cliffhanger at the cut, understandable cold. Rule-based by default; optional LLM re-scoring.")

# ---------------------------------------------------------------- input
st.sidebar.header("1. Input")
up = st.sidebar.file_uploader("Upload the episode .txt files or the .zip pack", type=["txt", "zip"],
                              accept_multiple_files=True)
folder = st.sidebar.text_input("...or a local folder / .zip path", value="")

st.sidebar.header("2. Scoring weights")
w = Weights(
    hook=st.sidebar.slider("Hook (first 3s)", 0.0, 1.0, 0.30, 0.05),
    cliff=st.sidebar.slider("Cliffhanger (the cut)", 0.0, 1.0, 0.35, 0.05),
    standalone=st.sidebar.slider("Standalone clarity", 0.0, 1.0, 0.15, 0.05),
    pacing=st.sidebar.slider("Pacing / dialogue density", 0.0, 1.0, 0.15, 0.05),
    duration=st.sidebar.slider("Duration fit (25-45s)", 0.0, 1.0, 0.05, 0.05),
)
st.sidebar.caption("Weights are normalised to sum to 1.")

st.sidebar.header("3. Selection")
per_ep = st.sidebar.number_input("Max promos per episode", 1, 5, 2)
min_comp = st.sidebar.slider("Minimum composite score", 4.0, 9.0, 6.5, 0.1)
min_hook = st.sidebar.slider("Minimum hook score", 0.0, 9.0, 6.0, 0.5)
min_cliff = st.sidebar.slider("Minimum cliffhanger score", 0.0, 9.0, 6.0, 0.5)

st.sidebar.header("4. Optional LLM re-scoring")
use_llm = st.sidebar.checkbox("Re-score shortlist with an LLM", value=False)
cfg = None
if use_llm:
    prov = st.sidebar.selectbox("Provider", ["openai", "anthropic"],
                                help="'openai' = any OpenAI-compatible endpoint (OpenAI, Gemini compat, OpenRouter, local).")
    model = st.sidebar.text_input("Model", os.getenv("PROMO_LLM_MODEL", ""))
    base = st.sidebar.text_input("Base URL (optional)", os.getenv("PROMO_LLM_BASE_URL", ""))
    key = st.sidebar.text_input("API key", os.getenv("PROMO_LLM_API_KEY", ""), type="password")
    if key and model:
        cfg = LLMConfig(prov, model, key, base)
    else:
        st.sidebar.warning("Enter a model and key, or untick the box.")


@st.cache_data(show_spinner=False)
def _load_uploaded(files: tuple) -> list:
    import tempfile
    import zipfile
    items = []
    for name, data in files:
        if name.lower().endswith(".zip"):
            with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
                tmp.write(data)
                tmp.flush()
                with zipfile.ZipFile(tmp.name) as z:
                    for n in sorted(z.namelist()):
                        if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                            items.append((n, z.read(n).decode("utf-8", "replace")))
        else:
            items.append((name, data.decode("utf-8", "replace")))
    return finish(items)


episodes = []
try:
    if up:
        episodes = _load_uploaded(tuple((f.name, f.getvalue()) for f in up))
    elif folder.strip():
        episodes = load_pack(folder.strip())
except Exception as exc:  # unreadable zip, bad path ...
    st.error(f"Could not read the input: {exc}")

if not episodes:
    st.info("Upload the screenplay pack (or enter a folder path) in the sidebar to begin.")
    st.stop()

n_events = sum(len(e.events) for e in episodes)
st.success(f"Parsed {len(episodes)} episodes, {n_events:,} timed blocks "
           f"(episode numbers {episodes[0].number}-{episodes[-1].number}).")

if st.button("⚡ Find promos", type="primary"):
    with st.spinner("Scoring candidate clips..."):
        promos, stats = find_promos(episodes, w, per_ep=int(per_ep), min_comp=min_comp,
                                    min_hook=min_hook, min_cliff=min_cliff)
        if use_llm and cfg:
            with st.spinner("Asking the LLM to re-judge the shortlist..."):
                stats["llm_rescored"] = rerank(promos, episodes, cfg, w)
    st.session_state["res"] = (promos, stats)

if "res" not in st.session_state:
    st.stop()

promos, stats = st.session_state["res"]
if not promos:
    st.warning("No clip met the thresholds. Lower the minimum composite / hook / cliffhanger scores.")
    st.stop()

df = pd.DataFrame(promos_table(promos))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Episodes", stats["episodes"])
c2.metric("Candidate clips scored", f"{stats['candidates']:,}")
c3.metric("Promos selected", len(promos))
c4.metric("Episodes with a promo", f"{stats['episodes_with_promo']} / {stats['episodes']}")

top = promos[0]
s0 = top.segments[0]
st.success(f"🏆 Post first: Episode {s0.episode}, {fmt_time(s0.start)}-{fmt_time(s0.end)} "
           f"({top.duration:.0f}s), composite {top.composite}/10")
st.write(f"**Hook:** {top.hook_line}")
st.write(f"**Cliffhanger:** {top.cliff_line}")
st.write(f"**Why:** {top.why}")

st.subheader("Ranked promos")
st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Read the clips (transcripts) before you trust the scores"):
    for p in promos[:20]:
        s = p.segments[0]
        st.markdown(f"**#{p.rank} - Ep {s.episode} {fmt_time(s.start)}-{fmt_time(s.end)}** (score {p.composite})")
        st.caption(p.transcript)

st.subheader("Tension curve for one episode")
ep_no = st.selectbox("Episode", [e.number for e in episodes])
ep = next(e for e in episodes if e.number == ep_no)
curve = tension_curve(ep, w)
fig, ax = plt.subplots(figsize=(10, 3.2))
if curve:
    ax.plot([t for t, _ in curve], [v for _, v in curve], marker="o", color="#2C3E50", linewidth=1.5)
ax.axhline(min_comp, color="r", linestyle="--", label=f"threshold ({min_comp})")
for p in promos:
    if p.episode == ep_no:
        ax.axvspan(p.segments[0].start, p.segments[0].end, color="#2ecc71", alpha=0.2)
ax.set_xlabel("Clip start time in episode (s)")
ax.set_ylabel("Best composite score")
ax.set_title(f"Episode {ep_no}: best clip score by start time (green = selected promo)")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Export")
x = io.BytesIO()
to_xlsx(promos, x, stats, {"per_episode": int(per_ep), "min_score": min_comp, "min_hook": min_hook,
                    "min_cliff": min_cliff, "weights": str(w.normalised())})
d1, d2, d3 = st.columns(3)
d1.download_button("Clip sheet (.xlsx)", x.getvalue(), "promos.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
d2.download_button("Clip sheet (.csv)", to_csv(promos).encode("utf-8"), "promos.csv", "text/csv")
d3.download_button("Metadata (.json)", to_json(promos).encode("utf-8"), "promos.json", "application/json")
