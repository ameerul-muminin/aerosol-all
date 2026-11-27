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
- [x] S5 `/api/selfcheck` (`src/selfcheck.py`): every Phase 2 math check re-run live, with the network blocked. (A)

## 1. Constrained allocation engine (15)
- [x] R1 `src/allocation.py`: demand per ward = supplies(predicted cases, upper range optional);
      stock slider 30–100%, crews total; priority score = risk tier × population density;
      greedy fill in priority order, `fulfilled_pct` and `deficit` per item, `critical` flag when < 100%. (A)
- [x] R2 Crew assignment derived from priority + cases; never more crews than available. (A)
- [x] R3 Tests: 100% stock → no deficits; 30% → deficits only in lowest-priority wards; totals never exceed stock. (B)
      Covered in `tests/test_allocation.py`: `test_full_stock_means_no_deficit_and_nothing_critical`,
      `test_scarcity_hits_only_the_lowest_priority_wards`, `test_totals_never_exceed_stock` (+ `test_plan_on_real_forecasts`).
- [x] R4 API `POST /api/allocate` (as_of, horizon, stock_pct, crews, city?). (B)
- [x] R5 UI: sliders, live table, red deficit alerts. (C) Full spec: section 8, R5.

## 2. Hospital overflow routing (10)
- [x] O1 `src/overflow.py`: per ward beds needed vs capacity; overflow sent to wards with headroom
      **in the same city only**, lowest burden first; output transfers list, net deficit per ward,
      city utilisation %, city headroom. (A)
- [x] O2 Tests: no cross-city transfer; patients conserved; unroutable overflow reported. (B)
      Covered in `tests/test_overflow.py`: `test_transfers_stay_within_city`, `test_fewer_beds_produces_intra_city_transfers`,
      `test_patients_are_conserved_*` (2), `test_unroutable_overflow_is_reported_per_ward_and_city`.
- [x] O3 API `GET /api/overflow?as_of&horizon`. (B)
- [x] O4 UI: per-city utilisation bars + transfer list (from → to, patients). (C) Full spec: section 8, O4.

## 3. Vector-control simulator (10) — must be model-driven, not a toggle
- [x] V1 `src/intervention.py`: crews on selected wards change the **model inputs** (e.g. lower lagged
      cases / effective water-logging), re-run the Phase 1 model at h = 2, 3, 4, report reduction in
      cases, beds, saline. Effect size is an assumption in S3; say so in the UI. (A)
- [x] V2 ROI = cases prevented per crew; rank all wards. (A)
- [x] V3 Tests + API `POST /api/intervention`. (B)
- [x] V4 UI: pick wards + crews, before/after per horizon, ROI ranking. (C) Full spec: section 8, V4.

## 4. Dispatch notices + /exports (part of 10)
- [x] D1 `src/dispatch.py`: template-only (no LLM) notices per ward: SMS (≤160 chars), WhatsApp, printable HTML brief
      with predicted cases, allocated vs required, deficits, assigned crews. EN + বাংলা. (B)
- [x] D2 `python -m scripts.export_dispatch` → `exports/` sample SMS/WhatsApp/HTML (committed). (B)
- [x] D4 API for D3: `GET /api/dispatch` (notices as JSON with SMS + WhatsApp text) and `GET /api/dispatch/brief`
      (printable HTML). Reproduces `exports/` byte for byte (`tests/test_api_dispatch.py`). (B)
- [x] D3 UI: "Dispatch" view with copy buttons and print-to-PDF. (C) Full spec: section 8, D3.

## 5. Offline resilience (15)
- [x] F1 Audit: no network calls anywhere (grep for http/requests/CDN); fonts vendored. (B)
      Automated in `scripts/offline_check.py`: no network client imported in src/, api/ or scripts/ (`network_imports()`),
      no remote URL a browser would load in web/ (`remote_urls()`); fonts, images, CSS and JS are all in `web/assets/`.
- [x] F2 Kill-switch test: Wi-Fi off → run every endpoint + UI, zero crashes. Script it: `scripts/offline_check.py`. (B)
      Done by A: `python -m scripts.offline_check` blocks outbound network, calls every /api route and every web file the two pages load, and scans web/ for remote loads.
- [x] F3 Deterministic: same inputs → same allocation/routing/notices (seeded, stable sort ties by ward_id). Test it. (A)
      Done for allocation, overflow, simulator and ROI ranking (`tests/test_determinism.py`); add notices when D1 lands.

## 6. Repo, live URL, docs (10)
- [ ] G1 Deploy a live URL (free host that runs Flask); app must still run fully locally. (B)
- [x] G2 ARCHITECTURE.md: add Phase 2 modules, diagram, every assumption from S3 and why. (A + B)
  - [x] A's half: Phase 2 diagram, one subsection per module, assumptions table from the config, limits, in-progress list.
  - [x] B's half: API routes, dispatch and offline sections once R4/O3/V3, D1 and F2 land.
- [x] G3 README: new run steps, exports, AI disclosure table updated with every tool each member used. (B)
- [ ] G4 Small commits from each person's account, commit convention from S4. (all)

## 7. Video (10) + Q&A (30)
- [ ] Q1 7-min script covering all 5 modules in order; rehearse twice; one uncut take on the committed hash. (all)
- [ ] Q2 Q&A prep: why greedy priority, why density, how capacity was set, how the simulator changes model inputs,
      what happens offline, limits of the supply formula. Each member must explain their own module. (all)

## Bonus (+10) — only after 1–6 work
- [x] B1 Multi-week supply plan (order now for weeks +2..+4). `GET /api/supply_plan`: per ward, city and region,
      expected and cautious; matches allocation and overflow (`tests/test_api_supply_plan.py`). (B)
- [x] B2 Optimal crew placement (choose wards to maximise cases prevented). `src/placement.py`: exact knapsack, `GET /api/crew_plan`. (A)

## Judges' feedback (27 Nov)
- [x] J1 "Public API on public Wi-Fi: add application rate limiting." Per-client + global token buckets,
      HTTP 429 + `Retry-After`, health/static/local never limited (`api/ratelimit.py`, `tests/test_rate_limit.py`). (B)
- [ ] J2 "ML model: deployed where?" Show it live: `GET /api/version` (model file, fingerprint, versions, commit). (B)
- [ ] J3 "Hospital: what if the patient is sent and it's full?" A backup ward per transfer, same city. (B)

## 8. Phase 2 screens: full spec for C (R5, O4, V4, D3)
**For Srijita's agent.** Tell it: *"Implement section 8 of docs/TASKS.md in web/, in the order C0, R5, O4, V4, D3.
Follow the build-ui skill. Do not change api/ or src/; if a field you need is missing, stop and ask Alif."*
Every field named below was checked against the running API. The judges score these screens directly: allocation
needs "live scarcity controls & deficit alerts", overflow a "clear deficit/headroom breakdown", the simulator must be
"interactive" and "model-driven" with an ROI ranking, and dispatch must show "predicted cases, allocated vs required
supplies, deficit alerts, and assigned spraying teams".

### C0 Shared setup (do this first, one commit)
1. **`web/assets/js/api.js`:** add these to `api` (GETs go through the cached `get`, POSTs through `post`):
   ```js
   allocate: (body) => post("/api/allocate", body),           // {as_of, horizon, stock_pct, crews?, demand_basis?, city?}
   overflow: (as_of, horizon, beds_available_pct, demand_basis) => get("/api/overflow", { as_of, horizon, beds_available_pct, demand_basis }),
   roi: (as_of, crews_per_ward) => get("/api/intervention/roi", { as_of, crews_per_ward }),
   intervention: (body) => post("/api/intervention", body),   // {as_of, deployment: {ward_id: crews}, total_crews?}
   crewPlan: (as_of, crews) => get("/api/crew_plan", { as_of, crews }),
   dispatch: (q) => get("/api/dispatch", q),                  // q = {as_of, horizon, stock_pct, crews, beds_available_pct, lang, city}
   briefUrl: (q) => `/api/dispatch/brief?${new URLSearchParams(q)}`,   // a page to open, not JSON to fetch
   supplyPlan: (as_of, city) => get("/api/supply_plan", { as_of, city }),
   ```
2. **New views:** `web/assets/js/views/allocation.js`, `overflow.js`, `simulator.js`, `dispatch.js`, each exporting
   `render(root, { state, meta, set, retry })` like `views/wards.js`. Register them in `VIEWS` in `app.js` and add
   four tabs to `web/app/index.html` after "All wards": `#allocation`, `#overflow`, `#simulator`, `#dispatch`
   (`data-i18n="nav.allocation"` etc.).
3. **Shared settings in the URL** (so Dispatch uses what the officer set on the other screens): add `stock` (default
   60), `crews` (empty = the server's default), `beds` (empty = the server's default) and `basis` (empty =
   `predicted_cases`) to `readState()`/`writeUrl()` in `app.js`. Write them only when set. Hide the sidebar Ward
   field on all four views (they are region-wide): extend the `f-ward` condition in `syncSidebar()`.
4. **Defaults come from the API, never typed in.** Slider ranges come from each answer's `assumptions` (below).
   On first load send no `crews`/`beds_available_pct` and show the value the answer returns (`crews` = 20 and
   `beds_available_pct` = 100 today, from `config/operations.json`).
5. **Live controls:** sliders update on `input`, debounced about 200 ms, and use `ticket(root)` so only the latest
   answer paints. Keep the previous numbers on screen (`aria-busy="true"`) while the next answer loads; no
   skeleton flash on every drag. API errors on a control change show inline (`<p class="inline-error"
   role="alert">` with the API's message), not a full-page error. A full-page `showError()` is only for the first load.
6. **CSS (`components.css`, tokens only):** a range slider (`input[type=range]`, `--accent` thumb, 44 px touch
   target, visible focus ring); `.deficit` (text in `--tier-critical`); an alert banner (`--tier-critical` text on
   `--paper`, 1 px `--line` border, icon + words, never colour alone); horizontal bars that grow with
   `transform: scaleX()`; `.table-wrap { overflow-x: auto }` so wide tables scroll inside their card at 360 px.
7. **i18n:** every new string in `en.json` and `bn.json` (Bangla numbers through `num0/num1`). Suggested groups:
   `nav.*`, `alloc.*`, `flow.*`, `sim.*`, `disp.*`. The `why` text inside `assumptions` is data (English); show
   it as is inside a `<details>`.

### R5 Supply allocation (`#allocation`): 15 points
**Call:** `api.allocate({as_of: state.as_of, horizon: state.h, stock_pct: state.stock, crews, demand_basis, city})`.
**Answer:** `forecast_week, stock_pct, crews, demand_basis, totals, wards[], assumptions`.
- `totals.items.{saline|kits|beds}` = `{required, available, allocated, deficit, wards_short}`;
  `totals.critical_wards`, `totals.wards` (60), `totals.crews = {available, assigned, unused}`. Totals are always
  region-wide; `city` only filters `wards[]`.
- `wards[]` in priority order: `priority_rank, ward_id, city, risk_tier, population_density_sqkm, priority_score,
  predicted_cases, range_low, range_high, demand_cases`, and for each of saline/kits/beds `<item>_required,
  <item>_allocated, <item>_fulfilled_pct, <item>_deficit`, then `fulfilled_pct` (worst item), `critical`,
  `crews_assigned`.
- Ranges: `assumptions.stock.stock_pct_range.value` (`[30, 100]`), `assumptions.demand.basis.allowed`,
  `assumptions.crews.max_per_ward.value`.

**Layout:**
1. Head: title "Supply allocation", subtitle "Week of {forecast_week} · {h} weeks ahead".
2. Toolbar (the live scarcity controls): horizon segmented control (copy `wards.js`); **"Available regional stock"
   slider** 30–100, step 5, showing "60%"; **"Total spraying crews"** number input (min 0, step 1); **demand**
   segmented control "Expected" (`predicted_cases`) / "Cautious" (`range_high`); city select with the caption
   "Totals cover all 60 wards".
3. **Deficit alert banner** (`role="alert"`) when `totals.critical_wards > 0`: "{n} of 60 wards cannot be fully
   supplied: saline short {deficit}, kits short {deficit}, beds short {deficit}." At 100% stock, a calm line:
   "Every ward fully supplied."
4. Four tiles (`span-3`): Saline, Kits, Beds = big `allocated`, "of {required}", red "{deficit} short in
   {wards_short} wards" when > 0. Crews = `assigned` "of {available}", "{unused} unused".
5. Ward table (`span-12`, in `.table-wrap`): Rank · Ward (`wardName` + city) · Tier (`tierHTML`) · Density
   (/km²) · Forecast cases (`num1(predicted_cases)`; with Cautious also show `demand_cases`) · Saline, Kits and Beds
   as "allocated / required" (red when deficit > 0) · Fulfilled (worst item %, small bar) · Crews · Status (a
   red "Critical" badge with an icon and the word when `critical`). Show 20 rows and a "Show all 60" button, plus
   a CSV download like `wards.js`.
6. **Order plan card (bonus B1)**, from `api.supplyPlan(as_of, city)`: rows +2, +3, +4 weeks (`weeks[i]
   .forecast_week`) × columns Saline · Kits · Beds, for `expected` or `cautious` (follows the demand toggle), and an
   **Order now** row from `order.{expected|cautious}` = `saline`, `kits` (3-week sums) and `beds_peak`. One line
   from `rule`: "Saline and kits: total for 3 weeks. Beds: the busiest week."
7. `<details>` "How this is decided": the `why` of `priority.tier_weight`, `priority.fill_rule`,
   `stock.full_stock_basis`, `crews.assignment_rule`.

**Done when:** dragging the stock slider changes deficits and the alert live; at 100% there are no deficits;
crews and the demand toggle change the table; the ranking shows tier and density. Demo values: 14 Jul 2025, +3,
50% stock → 37 critical wards.

### O4 Hospital overflow (`#overflow`): 10 points
**Call:** `api.overflow(state.as_of, state.h, state.beds, demandBasis)`.
**Answer:** `forecast_week, beds_available_pct, transfers[], wards[], cities[], assumptions`.
- `transfers[]` = `{from_ward, to_ward, city, patients}`: always inside one city.
- `cities[]` = `{city, demand, capacity, available_capacity, full_capacity, utilisation_pct, headroom, unroutable}`.
  `capacity` = `available_capacity` (after the slider); `utilisation_pct` = demand ÷ capacity.
- `wards[]` = `{ward_id, city, beds_required, capacity, available_capacity, full_capacity, routed_out, routed_in,
  net_deficit}`.
- Slider range: `assumptions.bed_capacity.beds_available_pct.range` (`[30, 100]`).

**Layout:**
1. Toolbar: horizon control; **"Beds free for dengue"** slider 30–100, step 5; the demand toggle (shared with R5).
2. Banner: red `role="alert"` for each city with `unroutable > 0`: "{n} patients in {city} cannot be placed: every
   ward in the city is full." Otherwise a neutral line: "{sum of patients} patients moved between wards, always
   within their city", or "No ward is over capacity this week."
3. **Five city cards** (`span-4`, wrapping): city name; a utilisation bar (width capped at 100%, the % number
   always shown, red above 100%); "Beds needed {demand} · available {available_capacity} of {full_capacity}";
   **"Headroom {headroom} beds"**; "Unroutable {unroutable}" in red when > 0.
4. **Transfers** (`span-6`): one row per transfer, "{wardName(from)} → {wardName(to)} · {city} · {patients}
   patients"; empty state "No transfers needed".
5. **Ward table** (`span-6`): Ward · City · Beds needed · Capacity "available / full" · Out · In · **Net deficit**
   (red when > 0). By default show only wards with out, in or net deficit > 0, with a "Show all wards" toggle.
6. A small "Same city only" note from `assumptions.overflow.same_city_only.why`.

**Done when:** the slider changes routing live, every transfer stays in its city, and city headroom and ward net
deficits are visible. Demo values: 5 Aug 2024, +2, beds 80% → 6 transfers in 4 cities.

### V4 Crew simulator (`#simulator`): 10 points
**Calls:** ranking `api.roi(as_of, crewsPerWard)`; run `api.intervention({as_of, deployment, total_crews})`;
suggestion `api.crewPlan(as_of, total_crews)`.
- ROI answer: `wards[]` (best first) = `{rank, ward_id, city, crews, cases_prevented, roi, roi_low, roi_high,
  cases_prevented_low, cases_prevented_high, floor_hit, input_before, input_after, by_horizon}`.
- Run answer: `crews_used, total_crews, assumption, horizons[], wards[], floor_hit_wards, assumptions`.
  `horizons[]` (+2, +3, +4) = `{horizon, forecast_week, cases_before, cases_after, cases_prevented,
  cases_prevented_low, cases_prevented_high, beds_before, beds_after, beds_freed, beds_freed_low, beds_freed_high,
  saline_before, saline_after, saline_saved, saline_saved_low, saline_saved_high}`. `wards[]` = `{ward_id, city,
  crews, by_horizon.{2|3|4}.{cases_before, cases_after, cases_prevented}, cases_prevented, roi, roi_low, roi_high,
  floor_hit}`.
- `assumption` = `{model_input, reduction_per_crew, max_total_reduction, sensitivity_per_crew, input_floor, note}`.
- Suggestion answer: `crews_available, crews_used, plan {ward_id: crews}, cases_prevented, wards[]`, and
  `compare.allocation_rule.cases_prevented`.
- Limits: crews per ward 1..`assumptions.crews.max_per_ward.value` (3); default total = `crews_available` from the
  suggestion answer.

**Layout:**
1. Toolbar: **"Total crews available"** number input; "crews per ward for the ranking" segmented control 1 / 2 / 3.
2. **ROI ranking** (`span-7`): Rank · Ward · City · Cases prevented (+2 to +4) · **Cases prevented per crew**, with
   "low–high" in small text · an "at limit" badge when `floor_hit` · an **add/remove crew stepper** per row
   (0..max) that builds the deployment.
3. **Deployment panel** (`span-5`): the chosen wards with steppers; "Crews used {n} of {total}" (red, and Run
   disabled, when over); **Run simulation** button (and an auto-run about 400 ms after changes); **Suggest the best
   placement** button: fill the deployment from `plan` and say "Prevents {cases_prevented} cases vs
   {compare.allocation_rule.cases_prevented} with the allocation rule".
4. **Results** (`span-12`): three cards, +2 / +3 / +4 weeks (`forecast_week`): cases before → after (big),
   "{cases_prevented} prevented (range {low}–{high})", beds freed, saline saved; before/after bars. Under them,
   per-ward rows: crews, cases prevented at +2/+3/+4, ROI.
5. **Honesty line (required):** "Model-driven: each crew lowers the ward's water-logging index by
   {reduction_per_crew × 100}% (at most {max_total_reduction × 100}%), and the forecast models are re-run at 2, 3
   and 4 weeks. The effect size is a planning assumption; low–high uses {sensitivity_per_crew}." All values come from
   `assumption`.
6. An API 400 (e.g. "That plan uses 6 crews but only 4 are available.") shows inline next to the Run button.

**Done when:** choosing wards and crews and running shows before/after at +2/+3/+4 for cases, beds and saline;
the ROI ranking is visible; the suggestion fills the plan. Demo values: 14 Jul 2025, 20 crews, suggest → prevents
77.39 cases vs 51.62 with the allocation rule.

### D3 Dispatch (`#dispatch`)
**Call:** `api.dispatch({as_of, horizon: state.h, stock_pct: state.stock, crews, beds_available_pct, lang, city})`.
The settings come from the shared URL state, so notices match what was set on Allocation and Overflow.
**Answer:** `forecast_week, lang, stock_pct, crews, beds_available_pct, summary, notices[], assumptions`.
- `summary` = `{wards, critical, unroutable_patients, max_sms_segments}`.
- `notices[]` in priority order = all of R5's ward fields, plus `sms, sms_chars, sms_segments, whatsapp, unroutable,
  allocation_critical, overflow {beds_required, capacity, routed_out, routed_in, net_deficit, …}, transfers[]`.
  `critical` = stock deficit or unroutable patients.
- Brief: `api.briefUrl(sameQuery)` is a printable HTML page (critical wards first).

**Layout:**
1. A line showing the settings in use: "60% stock · 20 crews · 70% beds free · +2 weeks", with a link "Change on
   Supply allocation". Notice language toggle **English / বাংলা** (default = UI language), city select and a
   **"Critical only"** checkbox (filters on the client).
2. Tiles: wards, critical (red), unroutable patients, max SMS length ("{max_sms_segments} SMS").
3. Buttons: **Open printable brief** (`window.open(api.briefUrl(q))`); **Print / Save as PDF** (open it, then call
   `print()` on that window after it loads); **Copy all SMS** (the SMS texts joined by blank lines).
4. One card per ward: ward + city + tier + a red "Critical" badge; a facts row (forecast {num1} cases, range,
   saline/kits/beds as allocated/required with deficits in red, crews, beds in/out); the **SMS** in a monospace box
   with "{sms_chars} characters · {sms_segments} SMS" and a **Copy** button; the **WhatsApp** text (`white-space:
   pre-wrap`) with a **Copy** button. Copy uses `navigator.clipboard.writeText`, falling back to select + `execCommand`,
   and shows "Copied" through an `aria-live` region.

**Done when:** each notice shows predicted cases, allocated vs required supplies, deficits and crews; copy and print
work in English and Bangla. Demo values: 15 Sep 2025, +2, 60% stock, 20 crews, 70% beds = the `exports/` sample.

### Checks before each PR (C)
- `python -m scripts.offline_check` passes (it follows the new JS modules automatically and fails on any CDN).
- The `verify` skill (pytest, API smoke, UI at 360 / 768 / 1440 px, keyboard, server stopped → error panel).
- Screenshots `docs/screenshots/allocation.png`, `overflow.png`, `simulator.png`, `dispatch.png` (1440 px), and
  tell Alif so he can add the views to the README table.
- Commits: `version-onsite-final: feat(web): <what>` (first line ≤ 72 characters), branch `c/<task>`, one PR per view.
