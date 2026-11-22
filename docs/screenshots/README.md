# Screenshots

Taken from the running app (`python -m api`, http://127.0.0.1:8000) on 22 Nov 2026. Every number on
screen comes from the local API. Dashboard shots: Chattogram Ward 1, as of 14 Jul 2025, 3 weeks ahead.

| File | View |
|---|---|
| [landing.png](landing.png) | Marketing page, desktop (1440 px): live chips from `/api/ranking` and `/api/metrics` |
| [landing-phone.png](landing-phone.png) | Marketing page on a phone (390 px) |
| [landing-bn.png](landing-bn.png) | Marketing page in বাংলা |
| [outlook.png](outlook.png) | Ward outlook: expected cases, risk tier, 80% range, rain to risk, "Why this forecast" drivers |
| [what-if-250mm.png](what-if-250mm.png) | What-if panel after the Heavy rain (250 mm) preset: actual weather vs scenario |
| [all-wards.png](all-wards.png) | All wards: tier counts, ranking with the +2/+3/+4 heat strip, by-city bars |
| [early-warnings.png](early-warnings.png) | All wards, as of 18 Aug 2025: early warnings, wards whose risk rises between +2 and +4 weeks, with the action to take |
| [accuracy.png](accuracy.png) | Model accuracy: test-year MAE/RMSE vs the seasonal average, by city, one ward's forecast vs reported cases with the 80% band, and the breeding clock (rain 0–12 weeks earlier vs cases now) |
| [cost.png](cost.png) | Cost & speed: model size, cached vs fresh times, live cache savings, response size (from `reports/efficiency.json` and the running server) |
| [check-a-ward.png](check-a-ward.png) | Landing page "Check a ward": Dhaka North Ward 5, week of 14 Jul 2025, with 250 mm of rain typed in, next to the recorded-weather forecast |
