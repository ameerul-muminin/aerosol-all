# Architecture

> These sections cover the weather lag pipeline, the chronological validation split, leakage
> controls and the model selection rationale. Teammates: add the data flow diagram, UI and cost sections.

## Lag pipeline (`src/features.py`)

Each prediction is made at a **forecast origin** week *t* for week *t + h*, where h ∈ {2, 3, 4} weeks.
Features may only use data from weeks ≤ *t*. All lags are computed in code from the raw file with
`groupby("ward_id").shift(k)`, so one ward's history never bleeds into another's.
`validate()` raises `ValueError` if a column is missing, a value is missing, a ward/date row is
duplicated, any gap isn't exactly 7 days (so `shift(k)` = *k weeks*), or a ward's attributes change.

| Feature group | Columns |
|---|---|
| Rainfall | lags 0–8 weeks; sums over weeks 0–1, **2–4** (breeding window), 0–3, 0–8; rain anomaly (h=2 only, see below) |
| Temperature, humidity | lags 0–4 weeks and 4-week means |
| Season | week-of-year sin/cos |
| Ward vulnerability | `water_logging_risk_index`, `population_density_sqkm`, `sanitation_score` |
| Recent cases (optional mode only) | reported cases, lags 0–4 weeks and 4-week sum |

**The data supports the lag choice.** After removing ward × week-of-year seasonality, case counts
correlate with rain at a lag of 2 weeks (r = 0.23), 3 weeks (r = 0.47, the peak) and 4 weeks (r = 0.33),
and at roughly 0 beyond 5 weeks. Weather is effectively **one regional series**: average weekly rainfall
correlates 1.00 between every pair of cities. Wards differ mainly through their attributes; average cases
per ward correlate with water-logging risk at r = 0.82.

## Leakage and the chronological split

**What leakage is:** information the model couldn't have at prediction time gets into training,
for example future weather or neighbouring weeks of the same season. The model then scores well
in testing and fails in real use. That's why the data is split **first**, before anything is
learned, and split **by time**: neighbouring weeks share weather, so a random split would train on
14 July and test on 21 July.

| Set | Target weeks | Rows (h=2) | Used for |
|---|---|---|---|
| Train | 2016 → 2021 | 18,180 | fitting; time-ordered cross-validation inside it |
| Validation | 2022 → 2023 | 6,240 | confirming the choice, tier calibration, interval calibration |
| Test | 2024 → 2025 | 6,300 | **reported once**; never used to choose anything |

- **Time-ordered cross-validation.** Within train there are 5 expanding folds: train on years before
  *Y*, validate on year *Y*, for *Y* = 2017…2021. The folds are cut on dates, not row positions,
  because rows are stacked ward by ward. sklearn's default early stopping (a **random** 10% holdout)
  is switched off; the number of trees is chosen by these folds instead.
- **What is fitted where (C6):**
  - Train only: tier definitions (percentiles of cases), the GLM scaler (inside its pipeline), all
    model parameters, and the reference point for drivers.
  - Validation only: tier cut points on the prediction scale and the conformal interval widths.
  - Fixed transforms, nothing learned: lags, week-of-year, and the rain anomaly (each ward's mean
    over *previous years only*).
- `tests/test_train.py` checks that train/validation/test target dates never overlap and that no
  fold trains on its own year or later. `tests/test_features.py` checks that truncating the data at
  a cutoff date leaves earlier features unchanged.
- **Test-set discipline (C5):** two problems (the tier calibration and in-season interval coverage,
  below) were first noticed in test output. Both were then confirmed **on validation alone**, and
  the fixes are fitted on validation. The final model was never tuned again after its test numbers
  were seen.

## Model selection rationale (`src/train.py`)

The target is a weekly count: 58% zeros, peaking Jul–Oct. Candidates were compared on Poisson deviance (lower is better):

| Poisson deviance: CV mean (validation) | h=2 | h=3 | h=4 |
|---|---|---|---|
| Seasonal naive (ward × week-of-year mean) | 1.98 (1.96) | 1.98 (1.96) | 1.98 (1.96) |
| Poisson GLM (log features) | 1.11 (1.26) | 1.35 (1.51) | 1.72 (2.04) |
| **HistGradientBoosting, Poisson loss** | **0.97 (1.04)** | **1.14 (1.22)** | **1.34 (1.71)** |

Gradient boosting with Poisson loss was chosen because:

- the target is a count and can't go negative;
- it learns interactions like "heavy rain *and* poor drainage" without hand-built terms;
- it needs only scikit-learn, trains in about 1.5 minutes, and predicts all 60 wards in about 40 ms.

Cross-validation chose **7-leaf trees** (300 trees for h=2 and h=3, 150 for h=4). The **rain anomaly**
was kept by the rule "keep only if cross-validation improves": it was used for h=2 (0.976 → 0.973,
a marginal gain) and dropped for h=3 and h=4. The model that ships is refit on train + validation;
test numbers come from the train-only model. Everything is serialized in `models/dengue_model.joblib`.

**Test results (2024–2025):**

| | h=2 | h=3 | h=4 |
|---|---|---|---|
| MAE (seasonal naive: 1.26) | 1.07 | 1.12 | 1.28 |
| RMSE | 2.18 | 2.25 | 2.55 |
| Truly Critical weeks flagged High or Critical | 85% | 81% | 66% |
| Tier within one level | 98% | 98% | 97% |
| 80% interval coverage, all weeks / Jul–Oct | 0.79 / 0.77 | 0.79 / 0.74 | 0.79 / 0.70 |

**h=4 is no better than the seasonal baseline, and the physics explains why:** the strongest driver,
rain 3 weeks before the target week, falls after the forecast origin. Permutation importance on
validation agrees. At h=2, rainfall dominates (deviance +8.7 when shuffled, against +1.9 for season).
At h=4, season dominates (+13.4, against +1.1 for rainfall). **2–3 weeks is the dependable warning window.**
Validation errors (2022–23) are higher than test errors because those were peak outbreak years
(about 10,000 cases per year, against about 7,000 in 2024–25).

**Optional weather + recent-cases mode** (`mode="weather_cases"`) uses the same settings plus
reported cases up to the origin week. Cross-validation changes are negligible: 0.973→0.963, 1.141→1.143,
1.337→1.323. In this dataset, cases follow weather and ward attributes with little week-to-week
carry-over, so the default stays **weather-only**, matching the brief.

**City hold-out (C4, `src/region_holdout.py`):** train on 4 cities up to 2023, test on the 5th city
in 2024–25, using the hyperparameters chosen by cross-validation. Mean MAE for an unseen city is
1.00 / 1.03 / 1.21 for h = 2/3/4, against 1.00 / 1.03 / 1.21 for a model that has seen the city and
1.33 for a regional week-of-year baseline. Truly Critical weeks flagged High or above: 75–100%.
Because weather is shared across cities, this mainly shows that **ward attributes carry over to
unseen wards**, not that the model handles a different climate.

## Risk tiers, intervals and drivers

- **Tiers:**
  - Defined in actual cases per ward-week from training percentiles: Medium ≥ 2 (p70), High ≥ 9
    (p90), Critical ≥ 15 (p97).
  - Predicted means are shrunk toward the centre, so those cut points applied directly caught only
    65 of 350 Critical validation weeks.
  - Cut points on the prediction scale are therefore set by **quantile matching on validation**
    (about 1.1 / 7.2 / 11): each tier is predicted about as often as it really occurs.
  - The cost is more false alarms, which is the right trade for early warning.
- **80% intervals:**
  - Split-conformal, asymmetric, scaled by √(prediction + 1), and calibrated separately for
    predictions below 1, 1–5, and 5 or more cases.
  - Quantile gradient boosting was rejected because its lower bound stayed near 0 at the season peak.
  - A single global calibration was rejected because it covered only 0.57 of Jul–Oct on validation.
- **Drivers (`src/explain.py`):**
  - Exact grouped Shapley values for each prediction: Rainfall, Temperature, Humidity, Season,
    Ward vulnerability, plus Recent cases in the optional mode.
  - Contributions are measured against an "average week" reference taken from the training period,
    and add up exactly to the prediction.
  - Example: *Dhaka South Ward 04, 12.2 cases, Critical: Rainfall +4.2, Humidity +2.9, Ward
    vulnerability +2.1, Season +1.0, Temperature +0.7, against an average week of 1.35*.
- **Dashboard data:**
  - `reports/validation_errors.csv`: MAE and RMSE by source (each CV fold, validation, test) ×
    horizon × city × month, with sums so any slice can be re-aggregated.
  - `reports/feature_importance.json`, `reports/region_holdout.json` and `reports/metrics.json`.
