# Remaining tasks (as of 21 Nov, 8:30 PM). Submit by 12:30 PM 22 Nov (hard limit 2:00 PM)

Each task: owner, depends on, check. Tick `[x]` in the same commit that finishes it.

## Now
- [ ] **PR #13 (Nazifa).** Duplicate of the already-merged #12, made 2 minutes later with a different
      implementation (`src/improvements.py`). #14 (ensemble) is built on #12, so merging #13 would undo it.
      Nazifa confirms nothing in #13 is missing from main, then closes it. Check: no open PRs with conflicts.
- [ ] **Re-verify main on Alif's laptop** (Nazifa merged #11, #12, #14; model is now an ensemble, ~11 MB).
      `git pull`, `python -m pytest -q`, `python -m api`, open `/api/forecast?ward_id=CCC_Ward_01&horizon=3`.
      Check: all tests pass, forecast JSON returns.
- [ ] **Re-measure efficiency** (Alif). The ARCHITECTURE.md table still shows the old 0.56 MB model.
      `python -m scripts.measure_efficiency --train`, then update the table and the retrain time in
      ARCHITECTURE.md, README.md and CLAUDE.md. Then `git checkout -- models reports` (keep efficiency.json).

## Frontend (night direction; prototype canvas artboards Night A/B/C)
Split so two people never edit the same file:
- [ ] **F1 Shell (Alif):** `web/app/index.html`, `assets/css/base.css` + `components.css` (on tokens.css),
      `assets/js/api.js`, `format.js`, `app.js` (hash router, URL state, inputs from `/api/meta`).
      Check: switching city updates wards; refresh keeps choices.
- [ ] **F2 Ward outlook (Alif):** verdict card, rain-to-risk bars, horizon segmented control, drivers with
      average-week line, model inputs. Check: 2→3→4 weeks moves the window and the forecast bar.
- [ ] **F3 All wards (Alif or Srijita):** tier counts, ranking with +2/+3/+4 heat strip, by-city bars, CSV.
- [ ] **F4 Model accuracy (Nazifa, optional):** MAE/RMSE per horizon vs baseline, by city, history chart
      with 80% band from `/api/history`.
- [ ] **F5 Marketing page (Srijita):** `web/index.html` + `landing.css` + `landing.js` from Night A.
      Live snapshot card from `/api/forecast`; accuracy numbers from `/api/metrics`.
- [ ] **F6 Fonts:** download Outfit 400/500/600 woff2 into `web/assets/fonts/`, `@font-face` in base.css.
- [ ] **F7 Polish pass:** Emil's prompt per page (widths, touch, throttled network), break-ui data,
      screenshots to `docs/screenshots/`, linked from README.

## Docs and submission (Alif)
- [ ] README: screenshots section; confirm AI table (add Antigravity/Gemini if Srijita or Nazifa used them).
- [ ] ARCHITECTURE.md: still 1–2 pages after Nazifa's additions (~1,050 words now; diagram must stay).
- [ ] Midnight git audit: all three have commits. Srijita has none yet: F5 or F6 must land from her account.
- [ ] Clean-clone test twice (tonight, morning): new folder + venv, README only, under 3 minutes.
- [ ] Video: script, two rehearsals, one uncut take ≤ 3:00 on localhost, voiceover, no slides.
- [ ] 11:30 AM: repo back to **public**, test every link in incognito, submit by 12:30 PM.
