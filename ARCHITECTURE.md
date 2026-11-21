# Architecture

> ML sections (A). The data-flow diagram, serving and cost sections are maintained by B/C.

## Features and leakage control
Each forecast is made at origin week *t* for week *t + h*, where h = 2, 3 or 4, using only data up
to *t*. All features are computed per ward with `groupby("ward_id").shift(k)`:

- rainfall lags 0–8 weeks, including the 2–4 week breeding window (rain at a 3-week lag has the
  highest deseasonalised correlation with cases, r = 0.47);
- temperature and humidity lags;
- season (week-of-year);
- ward water-logging, density and sanitation.

Tests prove that no feature changes when future rows are removed. **Feature selection** dropped only
near-duplicates whose removal improved CV (the 4-week temperature mean, r = 0.98 with season); see
`reports/feature_selection_log.json` and `reports/feature_list.json`.

## Chronological validation split
| Set | Target weeks | Purpose |
|---|---|---|
| Train | 2016–2021 | fitting; **time-ordered CV**: 5 expanding yearly folds choose settings and features |
| Validation | 2022–2023 | confirming the choice; calibrating tiers and 80% intervals |
| Test | 2024–2025 | reported once, never used for any choice |

There is no random shuffle anywhere (sklearn's random early stopping is off).

## Model selection
| Poisson deviance, CV mean (validation) | h=2 | h=3 | h=4 |
|---|---|---|---|
| Seasonal naive | 1.98 (1.96) | 1.98 (1.96) | 1.98 (1.96) |
| Poisson GLM | 1.11 (1.26) | 1.35 (1.51) | 1.72 (2.04) |
| **HistGradientBoosting, Poisson loss** | **0.96 (1.03)** | **1.14 (1.20)** | **1.31 (1.70)** |

Poisson-loss boosting fits zero-heavy counts and learns rain × ward interactions. It trains in about
70 s and predicts 60 wards in about 40 ms. Recent case counts didn't improve CV, so the model stays
weather-only.

**Test (2024–25):**
- MAE 1.07 / 1.11 / 1.28, against 1.26 for the seasonal baseline.
- Truly Critical weeks flagged High or above: 85% / 83% / 68%.
- 80% interval coverage: 0.79.

At h=4 the model matches the baseline, because its main driver (rain 3 weeks earlier) hasn't fallen
yet, so **2–3 weeks is the dependable window**. A city hold-out (train on 4 cities, test on the 5th)
gives the same MAE as seen cities (1.00).

**Tiers:** 2 / 9 / 15 or more cases (training p70/p90/p97), with cut points on predictions
calibrated on validation. **Intervals:** conformal, per prediction level. **Drivers:** exact grouped
Shapley values (`src/explain.py`). The tier and interval issues were first seen on test, then
confirmed on validation alone.
