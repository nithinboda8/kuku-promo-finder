# Kuku Promo Finder (Problem 1)

Reads timestamped microdrama screenplays and proposes **20-60 s promo clips**: a hook in the first
3 s, a cliffhanger at the cut, understandable to a cold viewer. For each promo it outputs the episode,
start/end timestamps of every segment, the hook line, the cliffhanger line, a one-line "why it works",
and a global rank.

## Run it
```bash
pip install -r requirements.txt
python run_pipeline.py "path/to/pack_folder_or_zip" --out outputs   # CLI -> promos.xlsx / .csv / .json
streamlit run app.py                                                # UI (upload the zip or .txt files)
python -m unittest discover -s tests -v                             # 9 tests
```
Runs offline in about 2 s for all 50 episodes (19,244 candidate clips scored). No API key needed.

## How it works
1. **Parse** (`parser.py`): reads the real file format (scene headers, `**SPEAKER (tone):**` blocks, `Start/End`
   timestamps) into timed events. Verified: all 3,727 blocks in the 50 episodes are captured.
2. **Generate candidates** (`pipeline.py`): every clip that starts on a spoken line or action beat and ends at a
   block boundary, 20-60 s long, inside one episode.
3. **Score** (`scoring.py`, rules): hook (first 3 s), cliffhanger (last ~6 s), standalone clarity, pacing,
   duration fit. Every score has a recorded reason.
4. **Select**: best non-overlapping clips per episode above thresholds, then rank globally by composite.
5. **Optional LLM re-score** (`llm.py`): only the shortlist is sent to a model (about 27-100 calls for 50
   episodes). Provider-agnostic via env vars `PROMO_LLM_PROVIDER / MODEL / API_KEY / BASE_URL`, so a stronger model
   is a config change. **This path is unit-tested with a fake model but has not been run against a live API.**

## Known limits (observed on the 50-episode run)
* **Rules cannot judge meaning.** Cliffhanger and hook are keyword/structure heuristics. Rank 1 (Ep 12) opens on
  "What do you want?"; the stronger hook ("I want to see you crawl, sister.") is a few seconds later.
* **Soft endings can slip through**, e.g. Ep 16 ends on a lingering look between the leads, which is atmospheric
  rather than an open question. Ep 12 ends on a scripted freeze, which scores well as a cliffhanger.
* **Single contiguous clips only.** The data model supports multi-segment / cross-episode promos
  (`Promo.segments` is a list) but the finder does not build them yet.
* **Text only.** No video, audio, or acting is seen; timestamps come from the screenplay, not the final edit.
* **Thresholds are judgement calls** (defaults: composite >= 6.5, hook >= 6, cliffhanger >= 6). 26 of 50 episodes get
  no promo at the defaults; lower the thresholds in the UI/CLI to see more.
* **Not validated against editor choices.** There is no ground truth here, so a human should read the clip
  transcripts (Excel sheet "Clip Transcripts") before anything is posted.
