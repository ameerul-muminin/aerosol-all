# Phase 2 Operational Architecture (DRAFT)

Phase 2 turns the Phase 1 ward-level dengue forecast (2, 3, and 4 weeks ahead; validation and test metrics `<from reports/metrics.json>`) into an offline-ready allocation, hospital routing, intervention, and dispatch pipeline across all 60 urban wards.

## Data Flow Diagram

```text
[Phase 1 Forecast: predict_all(as_of, h=2,3,4)]
   (predicted_cases, range_high, risk_tier)
                     |
                     v
   [Case->Supply Lookup: src/supplies.py]
   (saline, kits, beds from onsite dataset)
      |              |                 \
      v              v                  v
[1. Allocation]  [2. Overflow]    [3. Vector-Control]
+ ASSUMPTION:    + ASSUMPTION:    + ASSUMPTION:
  stock, crews     bed capacity     crews, crew effect
      |              |            -> perturbs lag/ward inputs
      |              |            -> re-runs Phase 1 models
      \              |                 /
       v             v                v
      [4. Multi-Channel Dispatch Notices]
      (SMS, WhatsApp, HTML/PDF in EN + BN)
                     |
                     v
      [5. Offline Resilience & Fallback]
      (local joblib, zero network/LLMs, deterministic sort)
```

## Phase 1 Forecast to Phase 2 Modules

First, `src/supplies.py` converts each ward's `predicted_cases` (or optional `range_high`) into `saline_bags_needed`, `diagnostic_kits_needed`, and `hospital_beds_required` via the deterministic case-to-supply lookup learned from the onsite dataset.

1. **Constrained Resource Allocation (`src/allocation.py`, `POST /api/allocate`):** Combines ward demand, `risk_tier`, and `population_density_sqkm` with two configurable parameters in `config/operations.json`: **ASSUMPTION** regional available stock (interactive 30%–100% scarcity slider) and **ASSUMPTION** total spraying crews. Wards are ranked by `risk_tier × population_density_sqkm` (tie-broken by `ward_id`) and filled greedily, outputting `fulfilled_pct`, `deficit`, assigned crews, and critical deficit flags when demand is under 100% fulfilled.
2. **Hospital Overflow Routing (`src/overflow.py`, `GET /api/overflow`):** Compares `hospital_beds_required` against an **ASSUMPTION** per-ward bed capacity (absent from the dataset). Excess patients route strictly within the same city pool (Dhaka North, Dhaka South, Chattogram, Gazipur, Narayanganj) to lowest-burden wards with headroom, reporting intra-city transfers, unroutable net deficits, and city bed utilization.
3. **Vector-Control Simulator (`src/intervention.py`, `POST /api/intervention`):** Takes officer-selected wards and **ASSUMPTION** deployed crews, applies an **ASSUMPTION** crew effect to Phase 1 model inputs (such as lagged cases or effective water-logging), and re-runs the Phase 1 ML models across horizons 2–4 to project reductions in cases, beds, and saline, ranking wards by ROI (cases prevented per crew).
4. **Dispatch Notices (`src/dispatch.py`, `exports/`):** Formats forecast cases, allocated versus required supplies, deficits, overflow routes, and crew assignments into deterministic bilingual (EN and বাংলা) templates for SMS, WhatsApp, and printable HTML/PDF commissioner briefs.
5. **Offline Resilience & Deterministic Fallback:** Every calculation and template renders locally using cached features, serialized `joblib` bundles, vendored assets, zero external APIs or LLMs, and deterministic `ward_id` tie-breaking.
