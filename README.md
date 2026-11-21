# Dengue Early-Warning for Urban Wards

Forecasts weekly dengue cases and a **risk tier (Low / Medium / High / Critical)** for 60 city
wards in Dhaka North, Dhaka South, Chattogram, Gazipur and Narayanganj, **2, 3 and 4 weeks ahead**.
It uses lagged rainfall, temperature and humidity, plus each ward's water-logging, density and
sanitation profile. Public health officers can use it to send larvicide teams and hospital
resources to high-risk wards before cases peak.

> Decision support on the provided hackathon dataset, not a medical diagnosis.

## Quick start (about 3 minutes)
```bash
git clone https://github.com/ameerul-muminin/aerosol-all.git
cd aerosol-all
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```
Tested with Python 3.11–3.13. The trained model is committed, so no training is needed to launch.

## Retrain (one command)
```bash
python -m src.train      # rebuilds models/dengue_model.joblib and reports/metrics.json
pytest -q                # lag pipeline, leakage and interface tests
```

## Project structure
```
app.py                     Streamlit dashboard
src/features.py            data validation + per-ward lag features
src/train.py               chronological split, model selection, tiers, conformal intervals
src/predict.py             inference API used by the app (predict_risk / predict_all)
models/dengue_model.joblib serialized models (one per horizon)
reports/metrics.json       validation and test metrics
tests/                     pytest suite
ARCHITECTURE.md            data flow, validation split, model rationale, efficiency
```

## Dataset
`hackathon_preli_dataset.xlsx` (provided by the organizers): 31,320 weekly rows for 60 wards,
2016-01-04 to 2025-12-29. On first load it is validated and cached as `data/dataset_cache.csv`.

## Results
See `ARCHITECTURE.md` and `reports/metrics.json`. Every number shown in the app is read from
`reports/metrics.json`.

## AI Assistants and External APIs
| Tool | Used by | What for |
|---|---|---|
| Claude (claude.ai chat, Claude Opus) | B | Planning, dataset profiling, code review of the ML pipeline, interface layer (`predict_risk`/`predict_all`), tests, README/CLAUDE.md drafts |
| Claude Code | A, B, C | Writing and refactoring code in the repo under each member's direction |
| _add any other tool here (ChatGPT, Copilot, Gemini…)_ | | |

**External APIs at runtime: none.** All inference runs locally from the serialized model, so
there are no API calls, tokens or usage costs.

## Team
- A: ML pipeline and model
- B: repo, reproducibility, efficiency
- C: dashboard, architecture doc, video
