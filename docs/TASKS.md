# Remaining tasks (as of 22 Nov, 1:00 AM). Submit by 12:30 PM (hard limit 2:00 PM)

Owner, check. Tick `[x]` in the commit that finishes a task.

## Done (backend)
- [x] ML pipeline, 12-model comparison, ensembles, season-aware ranges, test lock, ablation (Nazifa, PRs #4–#19)
- [x] API incl. `POST /api/what_if`, efficiency script with tests (Alif + Nazifa)
- [x] PR #13 closed (duplicate of #12)

## Critical path: frontend (warm paper "Jute" direction, `build-ui` skill, prototypes in `web/proto/`)
Nothing in `web/` is real yet. UI = 10 points + 5 bonus, and the video (20) shows it.
Order matters; each step is demo-able on its own.
- [x] **F1 Shell (Alif):** `web/app/index.html`, `assets/css/base.css` + `components.css`, `assets/js/api.js`,
      `format.js`, `app.js` (hash router `#outlook #wards #accuracy`, URL state, inputs from `/api/meta`).
- [x] **F2 Ward outlook (Alif):** verdict card (cases, tier glyph+colour+word, likely range), rain-to-risk bars,
      2/3/4 segmented control, drivers with "average week X → this ward Y", model inputs.
- [x] **F3 What-if panel (Alif, on the outlook view):** spec from Nazifa:
      three number inputs (rainfall "mm per week", average temperature °C, humidity %), weeks selector (1–8,
      default 4), "Run what-if" button → `POST /api/what_if`. Baseline vs scenario side by side: tier badge
      (colour AND text), predicted cases, range, change (`change.cases`, `tier_changed`). Show `warnings[]`
      and any `error` message from the API inline. No new packages. Demo line: 0 → 250 mm.
- [x] **F4 All wards (Alif):** tier counts, ranking with +2/+3/+4 heat strip (bonus: heatmap), by-city bars, CSV.
- [ ] **F5 Model accuracy (Nazifa):** MAE/RMSE per horizon vs seasonal baseline, by city, `/api/history`
      chart with 80% band (bonus: confidence intervals + validation dashboard).
- [ ] **F6 Marketing page (Srijita):** `web/index.html` + `landing.css` + `landing.js` from `web/proto/hero-a.html`.
      Live card from `/api/forecast`, accuracy from `/api/metrics`. No login/register pages: the app has no
      accounts, and a page that doesn't work reads as mock UI.
- [ ] **F7 Screenshots (Srijita):** fonts are vendored (Anton + Hind Siliguri); screenshots of
      each view into `docs/screenshots/`, linked from README.
- [ ] **F8 Polish (Alif):** widths 360/768/1440, keyboard, error states (stop the server and reload),
      no console errors, `verify` skill.

## Docs (morning)
- [ ] ARCHITECTURE.md to 1–2 pages: 1,240 words now, target about 900. Nazifa trims ML sections; the diagram,
      split and model rationale stay.
- [ ] `/api/metrics`: add the deployed model type next to `selected_model` (ensemble / blend / hgb) (Nazifa).
- [ ] Deep-learning study: commit it under `experiments/deep_learning/` with its own requirements file (not
      the main one), so the documentation's claims can be checked. Or drop the claims (Nazifa).
- [ ] README: screenshots, confirm AI table per member (Nazifa: Claude Code; Srijita: Antigravity + model).
- [ ] Re-run `python -m scripts.measure_efficiency --train` after the last model change; sync the cost table.

## Submission (Alif)
- [ ] Git check: Srijita has 0 commits. F6/F7 must land from her own account, as her own work.
- [ ] Clean-clone test on Windows: new folder + venv, README only, under 3 minutes.
- [ ] Video: script (Nazifa drafts the ML part), two rehearsals, one uncut take of 3:00 or less, on localhost,
      voiceover, no slides.
- [ ] 11:30 AM: repo back to **public**; open every link in incognito; submit the form by 12:30 PM.
