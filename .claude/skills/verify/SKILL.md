---
name: verify
description: Prove a change works before committing. Boots the app, hits every API endpoint, runs the test suite, and checks the web pages load. Use before every commit and before every PR.
---

# Verify

"Done" means someone ran it and saw it work. Run these steps and report the evidence.

## 1. Tests (always)
```
python -m pytest -q
```
Must end with `N passed`. A failure: stop, show the last 20 lines, fix the cause (max 3 attempts, then ask the human).

## 2. Repo hygiene (always)
```
git status --short
```
`models/` and `reports/` must NOT show as modified unless the task was retraining. If they are, the scikit-learn
version is wrong or a script retrained by accident: `git checkout -- models reports`.

## 3. API smoke test (when api/, src/ or web/ changed)
Start the server in the background, wait for "ready", then:
```
curl -s localhost:8000/api/health
curl -s "localhost:8000/api/forecast?ward_id=CCC_Ward_01&as_of=2025-07-14&horizon=3"
curl -s "localhost:8000/api/ranking?as_of=2025-07-14&horizon=3"
curl -s localhost:8000/api/metrics
curl -s -o /dev/null -w "%{http_code}\n" "localhost:8000/api/forecast?ward_id=NOPE&horizon=3"   # expect 400
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/        # expect 200
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/app/    # expect 200
```
Check: `risk_tier` is one of Low/Medium/High/Critical; `driver_reference_cases + sum(top_drivers.effect)` ≈
`predicted_cases` (±0.02). Stop the server afterwards.

## 4. UI (when web/ changed)
Open http://127.0.0.1:8000/ and /app/ in the browser (or Playwright if installed):
- no console errors; every number on screen also appears in an /api response (no hard-coded numbers)
- widths 360, 768, 1440: nothing overflows, tables scroll inside their panel
- keyboard: Tab reaches every control, focus ring visible
- loading, empty and error states exist (stop the server and reload to see the error state)
Save a screenshot per page to `docs/screenshots/` when the page is final.

## 5. Report
End with one line for the commit/PR: `Verified: pytest N passed; API smoke OK; UI checked at 360/1440` or
`Verified: none (reason)`.
