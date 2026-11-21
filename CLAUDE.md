# CLAUDE.md — team context for Claude Code

## Project
EDU AI Engineering Hackathon 2026, Phase 1: "AI for Urban Dengue Outbreak".
Build window 21 Nov 2026 2:00 PM → 22 Nov 2026 2:00 PM (Asia/Dhaka). **Submit by 1:30 PM.**
A web app (JSON API + HTML/CSS/JS frontend) that forecasts **ward-level dengue cases and risk tier 2, 3 and 4 weeks ahead**
from lagged weather and ward attributes, so public health officers can target high-risk wards early.

## Team (each person runs their own Claude Code session, on their own branch)
- **A (Nazifa): ML.** Owns `src/features.py`, `src/train.py`, models, metrics, and the ARCHITECTURE.md ML sections.
- **B (Alif): repo and efficiency.** Owns requirements, README, AI disclosure, tests, timings, the git audit and the clean-clone test.
- **C (Srijita): frontend engineer.** Owns `web/` (marketing page `web/index.html`, dashboard `web/app/`), the design, screenshots and the UI part of the video.
  B and A share `api/`, ARCHITECTURE.md and the rest of the video.

Only edit files outside your own area if their owner agrees. Branches: `a/<task>`, `b/<task>`, `c/<task>`; merge via PR.

## Layout
```
hackathon_preli_dataset.xlsx      raw data (committed)
data/dataset_cache.csv            generated CSV cache (gitignored)
src/features.py                   load_raw(), validate(), add_lag_features(), feature_columns(), build_dataset()
src/train.py                      python -m src.train  -> models/dengue_model.joblib + reports/metrics.json
src/predict.py                    build_features(), predict_risk(), predict_all(), predict(), load_bundle()
models/dengue_model.joblib        one bundle: model per horizon, tier cuts, conformal quantiles
reports/metrics.json              validation + test metrics per horizon (source of every number shown)
reports/test_predictions_h{2,3,4}.csv   per-row test predictions (for the error tab)
tests/                            pytest
api/service.py                    JSON-ready views of the model (pure Python, tested)
api/main.py                       Flask app: /api/* routes + serves web/
web/index.html                    marketing page (C)
web/app/index.html                dashboard (C)
web/assets/                       css, js, fonts, images (C) — no CDN, no build step
```

## Web API (the frontend talks ONLY to these; run `python -m api`)
`GET /api/meta` · `/api/forecast?ward_id&as_of&horizon` · `/api/ranking?as_of&horizon[&city]` ·
`/api/metrics` · `/api/history?ward_id&horizon` · `/api/health` · `POST /api/what_if` (JSON: ward_id, as_of,
horizon, weeks 1–8, any of rainfall_mm [mm per week], avg_temp_c, relative_humidity_pct → baseline vs scenario). Errors: HTTP 400 `{"error": "<friendly message>"}`.
Dates are ISO `YYYY-MM-DD` Mondays; horizon 2|3|4. Shapes are documented in docs/frontend/ (guide PDF).

## Working here
- Remaining work, owners and checks: `docs/TASKS.md`. Tick tasks in the commit that finishes them.
- Frontend: use the `build-ui` skill. Design = warm paper "Jute" direction (`web/assets/css/tokens.css`, locked
  22 Nov; prototypes in `web/proto/`), which replaces the palette in the frontend brief PDF. The PDF's API reference
  and task order still apply. Every UI string goes through i18n (`web/assets/i18n/en.json` + `bn.json`).
- Before every commit: `verify` skill, or delegate it to the `verifier` subagent to save context.

## Python interface (used by api/service.py)
```python
from src.predict import build_features, predict_risk, predict_all
feat = build_features()                                   # cached in api/service.py
predict_risk(feat, ward_id, as_of_date, horizon) -> dict  # one ward
predict_all(feat, as_of_date, horizon) -> DataFrame       # all wards, sorted by predicted_cases
```
Keys/columns: `ward_id, city, forecast_week, predicted_cases, range_low, range_high,
risk_tier ("Low"|"Medium"|"High"|"Critical"), top_drivers [(plain-language label, contribution)]`.
In the API, drivers are exact grouped-Shapley contributions from src/explain.py (they add up from
`driver_reference_cases`, the average week, to the prediction). > 0 raises predicted cases, < 0 lowers them.
Bad input raises `ValueError` with a user-friendly message; the API returns it as a 400 and the UI shows it inline, never a traceback.
`load_bundle()` is cached and retrains automatically if the model file can't be loaded.

## Challenge brief (Hackathon_preli_nov21.pdf) — scoring
Setup 10 · Lag pipeline 15 · Prediction engine 15 · UI 10 (needs driver breakdown) · ARCHITECTURE.md 10
(1–2 pages, diagram + split + model rationale) · Compute/API cost 10 (documented caching/cost controls)
· 3-min uncut video 20 · Git history + AI disclosure 10 · Bonus +5 (ward heatmap, confidence intervals,
MAE/RMSE validation dashboard). Treat the code as Stage 1 of a production system (reused in the onsite final).

## Verified facts
- Data: 31,320 rows = 60 wards × 522 weekly rows, 2016-01-04 → 2025-12-29, no gaps or missing values.
  Cities: Dhaka North 20, Dhaka South 20, Chattogram 10, Gazipur 5, Narayanganj 5. Cases 0–41, 58% zeros, peak Jul–Oct.
  There are no coordinates, so no geographic map. Use a ranked grid as the "heatmap".
- Split, by **target week**: train ≤ 2021-12-31, validation 2022–2023, test 2024–2025 (used once).
- Model: one deployed model per horizon, chosen by time-ordered cross-validation over 12 candidates and two
  baselines (seasonal average, Poisson GLM), with rules fixed before running: a weighted ensemble at 2 weeks,
  a HistGradientBoosting + extra-trees blend at 3 weeks, HistGradientBoosting (Poisson) at 4 weeks.
  Season-aware 80% ranges; test years locked by a SHA-256 fingerprint (src/test_lock.py).
- scikit-learn is pinned to **1.6.1** because the committed model was saved with it.

## Rules
1. Never invent numbers. Every metric in the UI, README, ARCHITECTURE.md or the video is read from `reports/metrics.json`,
   `reports/efficiency.json` or printed by code the human ran.
2. No mocked or hard-coded predictions (they score 0).
3. Relative paths only, no secrets/.env in the repo, and no external API calls at runtime.
4. Small commits from the person's own GitHub account. No force-push and no bulk final commit.
5. Every AI tool used goes in the README section "AI Assistants and External APIs".
6. Keep requirements minimal. Frontend = plain HTML/CSS/vanilla JS (ES modules), no npm/build step, no CDN at runtime;
   charts are hand-built with HTML/CSS/SVG. Fonts and images are committed under web/assets/.
7. Explain in simple language, one step at a time, and wait for confirmation.

## Commands
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train          # retrain (time-ordered CV + ensemble; re-measure, see docs/TASKS.md)
pytest -q
python -m api                # http://127.0.0.1:8000  (dashboard: /app/)
python -m scripts.measure_efficiency --train   # -> reports/efficiency.json
```
