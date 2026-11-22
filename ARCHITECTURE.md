# Architecture

> ML sections (features, split, model selection) by A; data flow, serving, cost and Phase 2 by B/C.

![aerosol-all architecture: offline training pipeline and online serving path](docs/architecture.svg)

**Offline, once** (`python -m src.train`, about 100 s): data → lag features → chronological split → model
selection → one serialized bundle plus `reports/`. **Online, per click:** the browser calls the local Flask API,
which reuses the model and lag features loaded at startup. No step calls an external service.

## Features and leakage control
A forecast is made at origin week *t* for week *t + h* (h = 2, 3, 4), using only data up to *t*.
Features are built per ward with `groupby("ward_id").shift(k)`:

- rainfall lags 0–8 weeks, covering the 2–4 week breeding window. In the data, above-normal rain is followed
  by above-normal cases 3 weeks later (r = 0.47, every city; `/api/lag_profile`);
- temperature and humidity lags, season (week-of-year), ward water-logging, density and sanitation;
- 4-week model only: the city-wide average of reported cases per ward (this week, last week, 4- and 8-week
  means, 3-week change), a steadier early outbreak signal than one ward's count (58% zero weeks).

Tests prove no feature changes when future rows are removed. Feature selection dropped only near-duplicates
whose removal improved CV (`reports/feature_selection_log.json`).

## Chronological validation split
| Set | Target weeks | Purpose |
|---|---|---|
| Train | 2016–2021 | fitting; **time-ordered CV** (5 expanding yearly folds) chooses settings and features |
| Validation | 2022–2023 | confirming the choice; calibrating tiers and 80% intervals |
| Test | 2024–2025 | reported once, never used for any choice; rows hash-locked (`reports/test_set_lock.json`) |

No random shuffle anywhere; a test fails if any decision script reads the test rows.

## Model selection
12 models on identical folds (`reports/model_comparison.json`), Poisson deviance, CV mean (validation):

| | h=2 | h=3 | h=4 |
|---|---|---|---|
| Best baseline (seasonal mean / Ridge) | 1.91 (1.74) | 1.98 (1.96) | 1.98 (1.96) |
| Extra trees | 1.03 (0.98) | 1.16 (1.10) | 1.44 (1.57) |
| HistGradientBoosting, Poisson loss | 0.96 (1.03) | 1.14 (1.20) | 1.31 (1.70) |
| **Deployed** | **0.95 (0.97)** | **1.09 (1.11)** | **1.24 (1.57)** |

Changes were adopted only under rules fixed in advance (random search, 75 trials, `reports/tuning_log.json`):

- **2 weeks:** weighted ensemble (boosting 0.65, random forest 0.22, Poisson GLM 0.13; weights fitted on
  out-of-fold training predictions), 2.2% better on CV; stacking never helped (`reports/ensemble.json`).
- **3 weeks:** boosting + extra-trees average with outbreak-week weighting.
- **4 weeks:** regularised boosting plus the city-wide case average (CV deviance 1.275 → 1.243, validation −0.6%).
  At 2 and 3 weeks it made validation worse, so those stay weather-only.
- Rejected: recent-year weights, isotonic recalibration (`reports/improvements_round2.json`).

**Test (2024–25):** MAE 1.12 / 1.12 / 1.24 (seasonal baseline 1.26). Truly Critical weeks flagged High or above:
88% / 78% / 67%. Outbreak weeks are still under-predicted by 6–8 cases (`reports/outbreak_errors.json`).
At 4 weeks the model barely beats the baseline because the main driver, rain 3 weeks earlier, hasn't fallen yet:
**2–3 weeks is the dependable window.** Removing weather raises CV MAE by 39% / 26% / 3.5%, removing ward
attributes by 13% / 10% / 8.5% (`reports/ablation.json`). An unseen-city hold-out matches seen cities (MAE 1.08 vs 1.08 at h=2).

**Tiers:** 2 / 9 / 15 or more cases (training p70/p90/p97), cut points calibrated on validation per horizon.
**Intervals:** conformal per prediction level and season (Jul–Oct coverage 0.77–0.78).
**Drivers:** exact grouped Shapley values that add up from the average week to the forecast (`reports/explainability.json`).

## Serving, compute and API cost efficiency
- **No LLMs, no external APIs.** No per-request fees, tokens or rate limits. `scripts/measure_efficiency.py`
  blocks outbound network while exercising every endpoint and records 0 external and 0 LLM calls.
- **Loaded once.** One joblib bundle holds all three horizons; lag features for all wards are built at startup.
- **One model pass per week and horizon.** `predict_all` scores all 60 wards at once (Shapley drivers: 32 batched
  calls). That result is cached and reused by every view: a ward's forecast, the ranking, early warnings, the other horizons.
- **Four cache layers, cheapest first:** browser memory (`api.js`), HTTP `Cache-Control: max-age=3600`, a bounded
  LRU cache per question on the server, then the shared model pass. `/api/efficiency` shows live hit counts in the app.
- **What-if** (`POST /api/what_if`) replaces 1–8 weeks of recorded weather and reruns the same lag pipeline; repeats are cached.
- **Small payloads:** JSON over 1 KB is gzip-compressed. **Minimal stack:** Python (scikit-learn, pandas, Flask),
  plain HTML/CSS/JS, no Node build, no GPU.

Measured with `python scripts/measure_efficiency.py --train --train-repeats 3` (MacBook, Apple silicon, Python 3.9.6,
21 Nov 2026; medians of 50 calls; `reports/efficiency.json`):

| Measure | Value |
|---|---|
| Model file (3 horizons, incl. the 2-week ensemble) | 11.4 MB |
| Startup: model load / lag features / app ready | 0.43 s / 1.0 s / 2.7 s |
| One-ward forecast, uncached: median / 95th percentile | 0.76 s / 0.88 s |
| Cached: 60-ward ranking / one-ward forecast | 0.002 s / 0.0006 s |
| What-if, new / repeated | 0.53 s / 0.016 s |
| Ranking payload, raw → gzip | 31.7 KB → 3.3 KB |
| Peak Python memory | 229 MB |
| Full retrain, 3 horizons: median (min–max of 3) | 100 s (97–106 s) |
| External API calls / LLM calls | 0 / 0 |

## Path to a live system (Phase 2)
- **Live data:** append weekly weather and case rows, re-run `validate()`, refresh caches; the lag pipeline and API contract stay the same.
- **Scale out:** the API is stateless, so several workers can run behind a load balancer, with `web/` on a CDN.
- **Retraining:** `python -m src.train` rebuilds every artifact in under 2 minutes; metrics are versioned next to the model in `reports/`.
