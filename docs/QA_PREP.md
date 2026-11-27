# Phase 2 technical Q&A preparation

25 likely questions for the 30-point Q&A (judges ask 5–6). Each answer can be **said in about 30 seconds**,
then **Proof** (where to point on screen) and **Limit** (the honest caveat, say it before they ask).

Numbers come from `reports/*.json` or from `python -m scripts.phase2_numbers` (run 27 Nov 2026; read-only,
about a minute). Anything not checked is marked **VERIFY**.

---

## 1. Constrained resource allocation

### Q1. How do you turn predicted cases into saline, kits and beds?
In the onsite data, each supply column is a fixed function of `dengue_cases`: every row with the same case count
has the same values. We learn that table (38 distinct counts, 0–41). For counts never seen we use one ratio per
item fitted on the table: 2.386 saline, 1.162 kits and 0.298 beds per case. The forecast is rounded half up to
whole patients first.
**Proof:** `src/supplies.py:supplies_for`, `learn_table`, `learn_ratios`; `tests/test_supplies.py::test_reproduces_every_row_of_the_data` (31,320 of 31,320 rows).
**Limit:** above 41 cases the ratio is an extrapolation, and the lookup ignores severity and age.

### Q2. Why greedy priority and not linear programming?
For this problem greedy *is* the LP optimum. The objective is "maximise the sum of priority × units delivered"
with one stock limit per item and each ward capped at its need. That is a fractional knapsack where every unit
weighs the same, and filling in priority order is provably optimal. An LP solver would return the same answer,
slower and harder to explain to an officer.
**Proof:** `src/allocation.py:_greedy_fill` (one cumulative sum); `tests/test_allocation.py::test_scarcity_hits_only_the_lowest_priority_wards`.
**Limit:** this holds because items are allocated independently. Joint constraints (a truck that carries both saline and kits) would need a real LP.

### Q3. Why priority = risk tier × population density?
The brief asks for high-risk, high-density wards first, so the score uses exactly those two fields. Multiplying
means a ward needs both to rank high. A Critical ward counts 4× a Low ward of the same density. Ties go to the
lower `ward_id`, so the order never depends on input order.
**Proof:** `src/allocation.py:allocate`; weights in `config/operations.json` → `priority.tier_weight` (1/2/3/4).
**Limit:** the equal tier steps (1–4) are an assumption. Density spans roughly 4× across wards (**VERIFY:** min/max of `population_density_sqkm`), so density and tier carry similar weight.

### Q4. How are spraying crews assigned, and can you exceed the crews available?
Crews go out one at a time, each to the ward with the highest (priority × predicted cases) ÷ (crews already
there + 1). The second crew in a ward counts half, which spreads crews. There are at most 3 per ward, and none go
to wards forecast below 1 case. The loop stops at the number of crews given, so the total can never be exceeded.
**Proof:** `src/allocation.py:assign_crews`; `tests/test_allocation.py::test_crews_never_exceed_the_total_or_the_ward_cap`.
**Limit:** every crew is treated as equal; travel time and ward area are not modelled.

### Q5. Where do stock levels and crew numbers come from? They aren't in the data.
They aren't, and we say so. "100% stock" means exactly enough for every ward's forecast this week, and the slider
scales that down to 30%. Crews default to 20. Every such value sits in `config/operations.json` with a reason,
marked "assumption", so real warehouse counts can replace it without code changes.
**Proof:** `config/operations.json` → `stock.*`, `crews.*`; `src/allocation.py:available_stock`.
**Limit:** absolute deficits are illustrative until real stock counts replace the rule.
*Example (scripts/phase2_numbers): 2025-07-14, 3 weeks ahead, 50% stock: 654 of 1,309 saline bags and 82 of 165 beds allocated; 23 wards fully served, 37 critical; 20 crews placed.*

---

## 2. Hospital overflow routing

### Q6. How did you set bed capacity, with no capacity in the data?
Each ward's capacity is the 95th percentile of its own weekly beds needed in the training years (2016–2021),
rounded up, with a minimum of 2. A ward is then over capacity only in its worst ~5% of weeks. Training years only,
so validation and test years don't leak into the rule. That gives 2–6 beds per ward, 223 in total.
**Proof:** `src/overflow.py:bed_capacity`; `config/operations.json` → `bed_capacity.rule` (the one "data rule").
**Limit:** it is a rule, not a hospital census; real bed counts would replace it.

### Q7. How does routing choose where patients go?
Patients move only between wards of the same city. The ward with the largest overflow is helped first, and each
patient goes to the ward with the most free beds, ties by `ward_id`. Anything that cannot be placed is reported
per ward as `net_deficit` and per city as `unroutable`, so no patient disappears.
**Proof:** `src/overflow.py:route_overflow`; `tests/test_overflow.py::test_transfers_stay_within_city`, `test_patients_are_conserved_with_fewer_beds`.
**Limit:** no coordinates in the data, so distance and traffic between wards are ignored.

### Q8. On 2025-07-14 at 50% beds, why are there no transfers?
Because every ward is full. 58 of 60 wards are over capacity and none has a free bed, so there is nowhere to send
anyone. Dengue peaks hit every ward of a city in the same weeks, so cities sit at 153–186% use. We report all of
it as unroutable, with city utilisation, so the officer sees the real gap and the need for outside help.
**Proof:** `tests/test_overflow.py::test_2025_07_14_at_50_pct_cannot_route_because_every_ward_is_full`; `python -m scripts.phase2_numbers`.
**Limit:** routing only helps when some neighbours still have room; it cannot create beds.

### Q9. What does the "beds available %" control do, and when does routing happen?
Beds are shared with non-dengue patients, so officers set the share free for dengue (30–100%). Capacity used =
max(1, floor(capacity × pct/100)). Over 78 forecast weeks (2024-06-03 to 2025-11-24, 2 weeks ahead), transfers
happen in 1 week at 100%, 13 at 70% and 4 at 50%. Fewer at 50% because whole cities go over. Demo:
2025-09-15 at 70%, 5 patients moved, none stranded.
**Proof:** `src/overflow.py:usable_capacity`, `overflow_plan(..., beds_available_pct=)`; `python -m scripts.phase2_numbers`.
**Limit:** the share is the same for every ward; real hospitals differ.

---

## 3. Vector-control simulator

### Q10. How is the simulator model-driven and not a toggle?
Crews change a real model input, the ward's `water_logging_risk_index`, by 15% per crew with a 50% cap. We then
re-run the deployed 2-, 3- and 4-week models. Cases prevented = forecast before − forecast after, and beds and
saline follow through the supply lookup. Example: 2 crews in each of the top 5 ROI wards cut the 2-week regional
forecast from 564.3 to 552.4 cases, freeing 5 beds and 30 saline bags.
**Proof:** `src/intervention.py:simulate`, `_before_after`; `python -m scripts.phase2_numbers`.
**Limit:** the 15% per crew is an assumption; there is no intervention data.

### Q11. Why water-logging and not recent cases?
The 2- and 3-week models use no case history at all, so cutting past cases would change nothing at those
horizons. Water-logging is used by all three models, and it is what larvicide crews remove: standing water where
Aedes breed. The 4-week model's city case rates are cases already reported, which a crew cannot undo.
**Proof:** `config/operations.json` → `crew_effect.model_input`; feature lists via `src/predict.py:_features_of`.
**Limit:** see Q12. It is the right lever for the model, but not a measured causal effect.

### Q12. Is the crew effect causal?
No, and we say so on screen. `water_logging_risk_index` is fixed per ward (one value each), so the model learned its
link to cases by comparing wards, not by watching one ward change over time. The simulator answers "what does the
model expect for a ward that looks less water-logged", under an assumed 15% per crew.
**Proof:** `python -m scripts.phase2_numbers` ("water_logging distinct values per ward (max): 1"); ward attributes add 8.5–12.9% to CV MAE when removed (`reports/ablation.json`).
**Limit:** calibrating it needs field data from past spraying campaigns.

### Q13. Does lowering water-logging ever *raise* a forecast?
Almost never, and we measured it. On two 2025 peak weeks (14 July, 1 September), with 15/30/50% cuts at all three
horizons, 1,077 of 1,080 ward forecasts went down or stayed the same. 3 rose by 0.01 cases, one rounding step, all
at 4 weeks with a 15% cut. That is tree-model noise, not a real effect.
**Proof:** `python -m scripts.phase2_numbers` ("forecasts that rose 3 of 1080, biggest rise 0.01 cases").
**Limit:** checked on two weeks, not all of them.

### Q14. How is ROI defined, and what does the ranking show?
ROI = cases prevented over the three forecast weeks (+2, +3, +4) ÷ crews sent. A crew changes only its own ward's
forecast, so all 60 wards are scored in one model run per horizon, in under 0.2 s. On 2025-07-14 the best ward,
DSCC_Ward_12, saves 5.6 cases per crew. With 2 crews it is 3.5 per crew, which shows diminishing returns.
Off-season (2025-02-03) no ward saves more than 0.01 cases.
**Proof:** `src/intervention.py:roi_ranking`; `tests/test_intervention.py::test_one_wards_crew_does_not_change_another_ward`, `test_more_crews_prevent_more_but_each_extra_crew_less`.
**Limit:** the three weeks are separate forecasts from one date, not a week-by-week epidemic simulation, and spillover between neighbouring wards is ignored.

### Q15. Why never push water-logging below a minimum?
Below the lowest value the models were trained on (0.22, read from the training years), a tree model just repeats
its edge value, so any extra "effect" would be invented. The cut stops there, and the output flags `floor_hit` so
the UI can say why a ward shows little or no effect. With 1 crew each, 5 wards hit the floor on 2025-07-14.
**Proof:** `src/intervention.py:input_floor`; `tests/test_intervention.py::test_input_never_goes_below_the_training_minimum`.
**Limit:** wards already near the floor will always look like poor crew investments in this model.

---

## 4. Dispatch notices

### Q16. Why templates and not an LLM for the notices?
Every number in a notice is copied from the allocation and overflow results by fixed templates, in English and
Bangla. That means no invented figures, no API key, no cost per message, and it works offline. Same inputs give
the same text.
**Proof:** `src/dispatch.py:build_notices`, `sms`, `whatsapp`, `brief_html`; `tests/test_dispatch.py::test_numbers_match_planners`, `test_localized_output_is_offline_and_deterministic`.
**Limit:** wording is fixed; it cannot answer follow-up questions.

### Q17. Do the SMS fit in one message?
English SMS are cut to 160 characters, and the "CRITICAL DEFICIT" alert is kept at the front. Bangla SMS are not
shortened.
**Proof:** `src/dispatch.py:sms`; `tests/test_dispatch.py::test_sms_lengths_and_critical_alerts`; samples in `exports/` (`python -m scripts.export_dispatch`).
**Limit:** Bangla uses Unicode SMS (about 70 characters per part; **VERIFY** with the carrier), so a Bangla notice is several parts.

---

## 5. Offline resilience

### Q18. What runs if the Wi-Fi is cut?
Everything: forecasting, allocation, routing, the simulator and the notices all run in one local Python process
from committed files (data, model, config, fonts). There are no LLM or external API calls. The Phase 1
measurement blocks the network and counts 0 external calls, 0 network calls and 0 LLM calls.
**Proof:** `reports/efficiency.json` (`external_api_calls` 0, `external_network_calls` 0, `llm_calls` 0); `scripts/measure_efficiency.py`.
**Limit:** the Phase 2 kill-switch script (`scripts/offline_check.py`, F2) and the Phase 2 API routes (PR #36) are not on `main` yet (**VERIFY** before judging).

### Q19. What if the model file is missing or won't load?
`load_bundle()` catches the error and retrains from the committed data, then carries on. A full retrain of the
three horizons took about 100 s on a MacBook. The app slows down once but does not crash.
**Proof:** `src/predict.py:load_bundle`; `reports/efficiency.json` → `full_retrain_3_horizons` 99.9 s.
**Limit:** a retrain also rewrites `reports/`, and needs scikit-learn 1.6.1 (pinned) to match the committed numbers.

### Q20. How is determinism guaranteed?
There is no random sampling at prediction time. Every sort breaks ties by `ward_id`, the supply lookup is an exact
table, and crews and routing are fixed rules. A test calls allocation, routing, the simulator and the ROI ranking
twice, and again with shuffled input rows, and the outputs are identical.
**Proof:** `tests/test_determinism.py` (`test_twice_gives_identical_output`, `test_shuffled_rows_give_identical_output`).
**Limit:** determinism holds for a fixed config and model file; change either and results change (reproducibly).

---

## 6. The forecasting model

### Q21. Why forecast 2, 3 and 4 weeks, and which model runs at each?
Dengue follows rain by a few weeks, so 2–3 weeks ahead is the dependable window. 4 weeks is an early warning.
Each horizon has its own model, chosen by time-ordered cross-validation with rules fixed in advance: a weighted
ensemble at 2 weeks (boosting 0.647, random forest 0.225, Poisson GLM 0.128), a boosting + extra-trees blend at 3,
and Poisson boosting at 4. Test MAE (2024–25) is 1.12 / 1.12 / 1.24, against 1.26 for a seasonal baseline.
**Proof:** `reports/metrics.json` → `horizons.{2,3,4}.test.MAE` and `baseline_seasonal_naive`; bundle `kind` via `src/predict.py:load_bundle`.
**Limit:** removing weather raises CV MAE by 38.6% / 26.2% but only 3.5% at 4 weeks (`reports/ablation.json`), so the 4-week model leans on season and city trends.

### Q22. `metrics.json` says `hgb_poisson`. Is that the model you serve?
`hgb_poisson` is the internal name of the chosen family. At 2 and 3 weeks that family is built as the ensemble or
blend (the bundle's `kind` says so). The test numbers come from the model trained on 2016–2021 only, used once on
2024–25. The served model has the same settings refit on 2016–2023; it never sees the test years.
**Proof:** `src/train.py` (deployment refit "on train+val"; test numbers from "train-only model").
**Limit:** the served model has 2 more years of data than the one scored, so its own test error is not the reported number.

### Q23. How good is it in outbreak weeks?
It is weakest there. In truly Critical weeks it under-predicts by 5.8 / 6.4 / 8.2 cases on average (2 / 3 / 4
weeks), and it flags those weeks as High or above 88% / 78% / 67% of the time. The 80% range covers about 78%
of outcomes overall, which is why `range_high` exists as the cautious planning basis.
**Proof:** `reports/outbreak_errors.json` (test, `outbreak_critical`); `reports/metrics.json` → `critical_recall`, `interval80_coverage`.
**Limit:** supplies planned on the central forecast will be short in the worst weeks.

---

## 7. "Why not X?"

### Q24. Why not share stock proportionally, so every ward gets something?
Proportional sharing spreads a shortage over every ward, including the highest-risk, densest ones. The brief asks
us to prioritise those. Greedy priority keeps them at 100% and flags exactly which lower-priority wards are short,
and by how much, so officers can act on a clear list.
**Proof:** `tests/test_allocation.py::test_scarcity_hits_only_the_lowest_priority_wards`.
**Limit:** at low stock, the lowest-priority wards can get nothing; a minimum share per ward would be a policy change in the config.

### Q25. Why no map, and why not route by distance?
The data has no coordinates, only ward and city names. So "same city" is the only distance we can defend, and the
heatmap is a ranked grid rather than a map.
**Proof:** dataset columns (`src/features.py:validate`); `CLAUDE.md` "Verified facts".
**Limit:** with coordinates, routing should prefer the nearest ward with room.

---

## If you only remember this
1. Supplies = exact table from the data (38 counts) + ratio beyond: 2.386 / 1.162 / 0.298 per case; all 31,320 rows match.
2. Priority = tier (1–4) × density, ties by ward_id. Greedy *is* the LP optimum here (fractional knapsack).
3. 100% stock = this week's forecast need; stock, crews and capacity are labelled assumptions in `config/operations.json`.
4. Bed capacity = each ward's 95th-percentile week in 2016–21, min 2 → 223 beds. Routing stays within the city, biggest overflow first.
5. 2025-07-14 at 50%: no ward has a free bed → 0 transfers, all reported unroutable. Demo week: 2025-09-15 at 70%, 5 moved, 0 stranded.
6. Crews lower water-logging (the only lever all 3 models use), 15% each, cap 50%, floor 0.22. It is an assumption, not causal.
7. 3 of 1,080 forecasts rose by 0.01 when water-logging was cut: rounding noise, measured and stated.
8. ROI = cases prevented over weeks +2..+4 per crew; best ward 5.6, with 2 crews 3.5 (diminishing returns).
9. Offline: one local process, 0 external/LLM calls; missing model → `load_bundle` retrains (~100 s); outputs identical on repeat.
10. Model: test MAE 1.12 / 1.12 / 1.24 vs baseline 1.26; weakest in outbreaks (−5.8 / −6.4 / −8.2 cases in Critical weeks).
