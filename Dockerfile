# One container: JSON API + web frontend + model, fully offline at runtime.
# Works on any Docker host (Hugging Face Spaces, Render "Docker", Railway, Koyeb, Cloud Run).
#   docker build -t aerosol .
#   docker run -p 7860:7860 aerosol        ->  http://localhost:7860  (dashboard: /app/)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Hugging Face Spaces runs containers as user 1000; the app must own its folder (CSV cache, retrain fallback)
RUN useradd -m -u 1000 app
WORKDIR /home/app/aerosol
RUN chown app:app /home/app/aerosol   # WORKDIR creates it as root; the app writes data/ here

# dependencies first, so code changes don't reinstall them
COPY --chown=app:app requirements.txt .
RUN pip install -r requirements.txt gunicorn==23.0.0

COPY --chown=app:app . .
USER app

# Build steps that would otherwise run on every start (slow on small servers):
# 1) write data/dataset_cache.csv once, so startup never parses the xlsx;
# 2) load the model, failing the build if it does not unpickle (scikit-learn is pinned to 1.6.1).
RUN python -c "from src.features import load_raw; load_raw()" \
 && python -c "import joblib; from src.train import BUNDLE; joblib.load(BUNDLE)"

EXPOSE 7860
# gunicorn instead of Flask's development server. --preload loads the model and features once,
# before the workers fork, so both workers share that memory; threads serve requests in parallel.
CMD ["sh", "-c", "exec gunicorn 'api.main:create_app()' --bind 0.0.0.0:${PORT} --preload --workers 2 --threads 4 --timeout 120 --access-logfile -"]
