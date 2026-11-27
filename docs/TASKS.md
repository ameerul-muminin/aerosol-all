# Phase 2 tasks: 10-hour onsite final

Source: `Hackathon_Phase2_Problemset.pdf` + `bangladesh_dengue_large_dataset_onsite.xlsx`.
Owner, check. Tick `[x]` in the commit that finishes a task. Phase 1 tasks are in git history.

## What the new data tells us (checked by code, 27 Nov)
- Same 31,320 rows, 60 wards, 2016-01-04 → 2025-12-29. Every Phase 1 column is **identical** to
  `hackathon_preli_dataset.xlsx`, so the Phase 1 model and metrics stay valid.
- 3 new columns: `saline_bags_needed`, `diagnostic_kits_needed`, `hospital_beds_required`.
  Each is a **fixed function of `dengue_cases`** (one value per case count, 0 when cases = 0),
  about saline ≈ 2.4×, kits ≈ 1.15×, beds ≈ 0.3× cases. So: forecast cases → look up supplies.
- **Not in the data:** bed capacity per ward, stock levels, number of crews, intervention effect.
  These must be stated, documented assumptions (configurable), never presented as data.

## Scoring (100 + up to 10 bonus)
Allocation 15 · Overflow routing 10 · Vector-control simulator 10 · Offline 15 ·
Repo + live URL + /exports 10 · 7-min uncut video 10 · Technical Q&A 30.

## 0. Setup (first 30 min)
- [x] S1 Load new xlsx in `src/features.py` (keep Phase 1 file working); supply columns kept as targets, not features. (A)
- [x] S2 `src/supplies.py`: case → supply lookup learned from data, with tests that it reproduces the table exactly. (A)
- [x] S3 `config/operations.json`: bed capacity rule, stock per item, crew count, crew effect — every assumption in one place,
      with the reason written next to it. (A + B agree)
- [ ] S4 Ask organisers: what does "follow version-onsite-final for each commit" mean (tag? branch? message prefix?). (B)

## 1. Constrained allocation engine (15)
- [x] R1 `src/allocation.py`: demand per ward = supplies(predicted cases, upper range optional);
      stock slider 30–100%, crews total; priority score = risk tier × population density;
      greedy fill in priority order, `fulfilled_pct` and `deficit` per item, `critical` flag when < 100%. (A)
- [x] R2 Crew assignment derived from priority + cases; never more crews than available. (A)
- [x] R3 Tests: 100% stock → no deficits; 30% → deficits only in lowest-priority wards; totals never exceed stock. (B)
      Covered in `tests/test_allocation.py`: `test_full_stock_means_no_deficit_and_nothing_critical`,
      `test_scarcity_hits_only_the_lowest_priority_wards`, `test_totals_never_exceed_stock` (+ `test_plan_on_real_forecasts`).
- [x] R4 API `POST /api/allocate` (as_of, horizon, stock_pct, crews, city?). (B)
- [ ] R5 UI: sliders, live table, red deficit alerts. (C)

## 2. Hospital overflow routing (10)
- [x] O1 `src/overflow.py`: per ward beds needed vs capacity; overflow sent to wards with headroom
      **in the same city only**, lowest burden first; output transfers list, net deficit per ward,
      city utilisation %, city headroom. (A)
- [x] O2 Tests: no cross-city transfer; patients conserved; unroutable overflow reported. (B)
      Covered in `tests/test_overflow.py`: `test_transfers_stay_within_city`, `test_fewer_beds_produces_intra_city_transfers`,
      `test_patients_are_conserved_*` (2), `test_unroutable_overflow_is_reported_per_ward_and_city`.
- [x] O3 API `GET /api/overflow?as_of&horizon`. (B)
- [ ] O4 UI: per-city utilisation bars + transfer list (from → to, patients). (C)

## 3. Vector-control simulator (10) — must be model-driven, not a toggle
- [x] V1 `src/intervention.py`: crews on selected wards change the **model inputs** (e.g. lower lagged
      cases / effective water-logging), re-run the Phase 1 model at h = 2, 3, 4, report reduction in
      cases, beds, saline. Effect size is an assumption in S3; say so in the UI. (A)
- [x] V2 ROI = cases prevented per crew; rank all wards. (A)
- [x] V3 Tests + API `POST /api/intervention`. (B)
- [ ] V4 UI: pick wards + crews, before/after per horizon, ROI ranking. (C)

## 4. Dispatch notices + /exports (part of 10)
- [x] D1 `src/dispatch.py`: template-only (no LLM) notices per ward: SMS (≤160 chars), WhatsApp, printable HTML brief
      with predicted cases, allocated vs required, deficits, assigned crews. EN + বাংলা. (B)
- [x] D2 `python -m scripts.export_dispatch` → `exports/` sample SMS/WhatsApp/HTML (committed). (B)
- [ ] D3 UI: "Dispatch" view with copy buttons and print-to-PDF. (C)

## 5. Offline resilience (15)
- [ ] F1 Audit: no network calls anywhere (grep for http/requests/CDN); fonts vendored. (B)
- [ ] F2 Kill-switch test: Wi-Fi off → run every endpoint + UI, zero crashes. Script it: `scripts/offline_check.py`. (B)
- [x] F3 Deterministic: same inputs → same allocation/routing/notices (seeded, stable sort ties by ward_id). Test it. (A)
      Done for allocation, overflow, simulator and ROI ranking (`tests/test_determinism.py`); add notices when D1 lands.

## 6. Repo, live URL, docs (10)
- [ ] G1 Deploy a live URL (free host that runs Flask); app must still run fully locally. (B)
- [ ] G2 ARCHITECTURE.md: add Phase 2 modules, diagram, every assumption from S3 and why. (A + B)
  - [x] A's half: Phase 2 diagram, one subsection per module, assumptions table from the config, limits, in-progress list.
  - [ ] B's half: API routes, dispatch and offline sections once R4/O3/V3, D1 and F2 land.
- [ ] G3 README: new run steps, exports, AI disclosure table updated with every tool each member used. (B)
- [ ] G4 Small commits from each person's account, commit convention from S4. (all)

## 7. Video (10) + Q&A (30)
- [ ] Q1 7-min script covering all 5 modules in order; rehearse twice; one uncut take on the committed hash. (all)
- [ ] Q2 Q&A prep: why greedy priority, why density, how capacity was set, how the simulator changes model inputs,
      what happens offline, limits of the supply formula. Each member must explain their own module. (all)

## Bonus (+10) — only after 1–6 work
- [ ] B1 Multi-week supply plan (order now for weeks +2..+4).
- [x] B2 Optimal crew placement (choose wards to maximise cases prevented). `src/placement.py`: exact knapsack, `GET /api/crew_plan`. (A)
