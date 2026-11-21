# CLAUDE.md — team context for Claude Code

## Project
EDU AI Engineering Hackathon 2026, Phase 1: "AI for Urban Dengue Outbreak".
Build window 21 Nov 2026 2:00 PM → 22 Nov 2026 2:00 PM (Asia/Dhaka). **Submit by 1:30 PM.**
A Streamlit app that forecasts **ward-level dengue cases and risk tier 2, 3 and 4 weeks ahead**
from lagged weather and ward attributes, so public health officers can target high-risk wards early.

## Team (each person runs their own Claude Code session, on their own branch)
- **A (Nazifa): ML.** Owns `src/features.py`, `src/train.py`, models, metrics, and the ARCHITECTURE.md ML sections.
- **B (Azman): repo and efficiency.** Owns requirements, README, AI disclosure, tests, timings, the git audit and the clean-clone test.
- **C: dashboard and docs.** Owns `app.py`, `.streamlit/`, the ARCHITECTURE.md diagram/UI/cost sections, screenshots and the video.

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
app.py                            Streamlit UI (C)
```

## Interface (the UI must use only these functions)
```python
from src.predict import build_features, predict_risk, predict_all
feat = build_features()                                   # cache with st.cache_data
predict_risk(feat, ward_id, as_of_date, horizon) -> dict  # one ward
predict_all(feat, as_of_date, horizon) -> DataFrame       # all wards, sorted by predicted_cases
```
Keys/columns: `ward_id, city, forecast_week, predicted_cases, range_low, range_high,
risk_tier ("Low"|"Medium"|"High"|"Critical"), top_drivers [(plain-language label, contribution)]`.
Contribution > 0 means the factor raises predicted cases, and < 0 means it lowers them (what-if occlusion against training averages).
Bad input raises `ValueError` with a user-friendly message. Show it with `st.warning`, never a traceback.
`load_bundle()` is cached and retrains automatically if the model file can't be loaded.

## Verified facts
- Data: 31,320 rows = 60 wards × 522 weekly rows, 2016-01-04 → 2025-12-29, no gaps or missing values.
  Cities: Dhaka North 20, Dhaka South 20, Chattogram 10, Gazipur 5, Narayanganj 5. Cases 0–41, 58% zeros, peak Jul–Oct.
  There are no coordinates, so no geographic map. Use a ranked grid as the "heatmap".
- Split, by **target week**: train ≤ 2021-12-31, validation 2022–2023, test 2024–2025 (used once).
- Model: HistGradientBoostingRegressor (Poisson loss), one per horizon, chosen on validation Poisson deviance
  over seasonal-naive and Poisson GLM baselines.
- scikit-learn is pinned to **1.6.1** because the committed model was saved with it.

## Rules
1. Never invent numbers. Every metric in the UI, README, ARCHITECTURE.md or the video is read from `reports/metrics.json`,
   `reports/efficiency.json` or printed by code the human ran.
2. No mocked or hard-coded predictions (they score 0).
3. Relative paths only, no secrets/.env in the repo, and no external API calls at runtime.
4. Small commits from the person's own GitHub account. No force-push and no bulk final commit.
5. Every AI tool used goes in the README section "AI Assistants and External APIs".
6. Keep requirements minimal. Charts use Streamlit built-ins or Altair (ships with Streamlit). Don't add plotly etc.
7. Explain in simple language, one step at a time, and wait for confirmation.

## Commands
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train          # retrain (~5 s)
pytest -q
streamlit run app.py
```
