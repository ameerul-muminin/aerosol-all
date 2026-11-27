# Phase 2 Technical Q&A Prep (DRAFT)

15 likely judge questions covering the five Phase 2 modules, with concise, honest answers and explicit method limits. Note: Phase 1 forecast metrics are referenced as `<from reports/metrics.json>` rather than hardcoded.

---

## Constrained Resource Allocation

### 1. How do you convert predicted dengue cases into saline bags, diagnostic kits, and hospital beds?
**Answer:** In the onsite dataset, `saline_bags_needed`, `diagnostic_kits_needed`, and `hospital_beds_required` are exact deterministic functions of `dengue_cases` (zero when cases equal zero). `src/supplies.py` learns this lookup table directly from the dataset and maps Phase 1 `predicted_cases` into ward-level supply and bed demand.
**Limit:** A static point-wise lookup ignores age mix, clinical severity shifts during severe outbreaks, and multi-week patient carry-over; any Phase 1 forecast error (`<from reports/metrics.json>`) propagates linearly into supply estimates.

### 2. Why use greedy priority (`risk_tier × population_density_sqkm`) instead of proportional sharing or linear programming?
**Answer:** Under acute scarcity (30%–100% stock), proportional sharing leaves high-risk, densely populated wards under-supplied to protect low-risk wards. Greedy priority with deterministic `ward_id` tie-breaking is fast, transparent to health officers, and concentrates scarce stock where transmission and severe caseloads cluster.
**Limit:** Greedy allocation can completely starve lower-priority wards at low stock levels, treats density linearly without accounting for local clinic access, and does not optimize multi-period stockpiling.

### 3. How are spraying crews assigned under scarcity, and what triggers a critical deficit alert?
**Answer:** Crew assignments are derived from ward priority and predicted caseload up to the configured **ASSUMPTION** total crew cap, so assigned crews never exceed available teams. Any ward whose allocated saline, kits, or beds fall below 100% of required demand is flagged `critical` with its exact `deficit` and `fulfilled_pct`.
**Limit:** Crew assignment assumes every crew has equal capacity regardless of ward area or road congestion, and greedy assignment does not account for travel time between wards.

### 4. Where did regional stock, crew counts, and bed capacities come from if they are not in the dataset?
**Answer:** They are not in `bangladesh_dengue_large_dataset_onsite.xlsx`. Every operational capacity parameter—regional stock baseline, total spraying crews, per-ward bed capacity, and crew effect size—is stored as an explicit, documented **ASSUMPTION** in `config/operations.json` so city officials can swap in real inventory counts without touching code.
**Limit:** Because these values are configured assumptions rather than audited facility records, absolute deficit counts and utilization percentages are illustrative until calibrated with live DGHS and city corporation registries.

### 5. Should allocation use `predicted_cases` or the 80% upper bound (`range_high`)?
**Answer:** By default, allocation plans against `predicted_cases` so baseline stock matches expected demand, with an option to plan against `range_high` for conservative surge preparedness.
**Limit:** Phase 1 models under-predict sharp outbreak peaks (`<from reports/metrics.json>`), so point-forecast planning risks under-ordering during surges, while planning on `range_high` everywhere can exhaust regional stock prematurely.

---

## Hospital Overflow Routing

### 6. Why does overflow routing restrict patient transfers strictly within the same city?
**Answer:** City hospital networks (Dhaka North, Dhaka South, Chattogram, Gazipur, Narayanganj) operate under distinct municipal health administrations and physical travel boundaries. Keeping transfers strictly intra-city avoids unrealistic cross-city routing (e.g., Dhaka to Chattogram) and ensures city-level utilization reflects true local pressure.
**Limit:** Intra-city pooling still assumes any two wards within the same city can transfer patients freely, ignoring ward-to-ward ambulance distance and traffic because the dataset contains no geographic coordinates.

### 7. How does `src/overflow.py` choose receiving wards, and what happens if an entire city overflows?
**Answer:** Over-capacity wards send excess `hospital_beds_required` to wards in the same city with remaining headroom, prioritizing lowest-burden wards first and conserving total patient counts. When total city demand exceeds total city capacity, remaining patients are reported explicitly as unroutable net ward deficits alongside city-wide utilization.
**Limit:** The router balances single-week bed counts and does not model length-of-stay carry-over from prior weeks, ICU versus general ward specialization, or private-versus-public hospital eligibility.

---

## Vector-Control Simulator

### 8. Why is the simulator "model-driven" instead of applying a fixed percentage discount to cases?
**Answer:** `src/intervention.py` maps deployed crews on selected wards into changes to Phase 1 model inputs (e.g., reducing lagged case momentum or effective water-logging) using a configurable **ASSUMPTION** crew effect, then re-runs the actual Phase 1 ML models across horizons 2, 3, and 4. Downstream reductions in beds and saline come from passing those re-predicted cases through `src/supplies.py`.
**Limit:** The ML models respond nonlinearly to ward features, which is more realistic than a flat multiplier, but the magnitude of input reduction per crew remains an unverified **ASSUMPTION**.

### 9. Can the Phase 1 ML model genuinely estimate the causal effect of larvicide or fogging?
**Answer:** No—and we state that honestly. The dataset contains observational weather, ward attributes, and case counts, with no historical records of spraying campaigns. The simulator performs a counterfactual sensitivity run under an explicit assumption, not causal inference from a randomized trial.
**Limit:** If an officer expects true causal effectiveness, the assumed crew effect in `config/operations.json` must first be calibrated against field surveillance data from past vector-control drives.

### 10. How is intervention ROI computed, and where can ROI ranking mislead decision-makers?
**Answer:** ROI is defined as total predicted cases prevented over horizons 2–4 divided by the crews deployed to that ward, allowing all 60 wards to be ranked consistently.
**Limit:** Ward-level ROI evaluates each ward independently, ignoring mosquito spillover between adjacent wards, saturation effects when too many crews target one ward, and equity considerations for smaller high-risk pockets.

### 11. How trustworthy are 4-week intervention and allocation plans compared to 2- or 3-week plans?
**Answer:** Across Phase 1 evaluation (`<from reports/metrics.json>`), 2–3 weeks is the dependable weather-driven window because dengue follows rainfall by roughly 3 weeks. At 4 weeks, the key rainfall event has often not occurred yet, so forecasts rely more on seasonal and city-wide case trends.
**Limit:** 4-week supply orders, overflow estimates, and simulator projections carry wider uncertainty and should be treated as early procurement signals to be updated at the 2-week horizon.

---

## Dispatch Notices & Offline Resilience

### 12. Why use template-based dispatch notices instead of an LLM?
**Answer:** `src/dispatch.py` generates SMS (≤160 characters), WhatsApp alerts, and printable HTML/PDF commissioner briefs using deterministic bilingual (EN + বাংলা) templates populated directly from allocation, overflow, and crew outputs. Every number matches the engine output exactly with zero hallucination risk.
**Limit:** Templates have fixed wording and cannot answer ad-hoc follow-up questions or adapt tone dynamically for unusual local events.

### 13. What happens if internet access and external APIs are disconnected during live judging?
**Answer:** Every endpoint, simulation, allocation, overflow calculation, and dispatch export runs 100% locally with zero outbound network calls, zero external APIs, serialized `joblib` models, and vendored fonts and frontend assets—verified by `scripts/offline_check.py`.
**Limit:** Offline execution relies on the locally cached dataset and model bundle; new weekly weather or case reports cannot be ingested until connectivity is restored.

### 14. How do you guarantee deterministic behavior across runs?
**Answer:** Inference uses no random sampling, sorting ties are broken deterministically by `ward_id`, lookups are exact tables, and dispatch notices use static templates. Identical inputs (`as_of`, `horizon`, stock %, crews, capacity config) always yield identical allocations, transfers, and exports.
**Limit:** Determinism applies given a fixed `config/operations.json` and model bundle; changing any assumption or retraining the model changes downstream outputs cleanly and reproducibly.

### 15. What is missing before a municipal health team could deploy this in production?
**Answer:** The algorithmic pipeline is modular and stateless, ready to run behind a standard WSGI server with weekly data appends. However, production rollout requires replacing configured assumptions with live facility registries and surveillance feeds.
**Limit:** Specifically, production deployment needs (1) verified ward hospital bed and warehouse stock APIs, (2) ward geo-coordinates for distance-aware ambulance routing, (3) empirical calibration of spraying crew effectiveness, and (4) automated weekly case/weather ingestion.
