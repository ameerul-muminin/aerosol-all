# Deploying the app (Docker)

The `Dockerfile` runs the JSON API, the web frontend and the model in one container, with no network
access needed at runtime. It is faster than `python -m api` on a free host because:

- **gunicorn** (a production server) replaces Flask's development server: 2 worker processes x 4 threads;
- **`--preload`**: the model and lag features load once, before the workers fork, so they share that memory;
- the **CSV cache is built into the image**, so a cold start never parses the xlsx;
- the image build **fails if the model file does not load**, instead of retraining at startup.

Measured locally with the image's exact server command (`gunicorn 'api.main:create_app()' --preload --workers 2
--threads 4`, MacBook, 27 Nov 2026): ready in about 3 s, `/api/selfcheck` passed, `/api/crew_plan` 0.35 s.
Speed on a host depends on its CPU; a free Render instance (0.1 CPU, sleeps after 15 idle minutes) is
the main reason the current deployment is slow.

## Option A (recommended): Hugging Face Spaces, free
A free CPU Space has far more CPU and memory than a free Render instance and sleeps only after a long
idle period (check the current numbers on huggingface.co/pricing).

1. On huggingface.co: **New Space** → SDK **Docker** → template **Blank** → hardware **CPU basic (free)**.
   The Space gets its own git repo with a `README.md` holding settings like:
   ```yaml
   ---
   title: Aerosol Dengue
   sdk: docker
   app_port: 7860
   ---
   ```
   Keep that README in the Space (do not overwrite it with the project README).
2. Copy the project into the Space repo, everything except `README.md` and `.git`:
   ```sh
   git clone https://huggingface.co/spaces/<user>/<space> space && cd space
   git lfs install && git lfs track "*.joblib" "*.xlsx" "*.webp" "*.png" "*.woff2"
   rsync -a --exclude .git --exclude README.md ../AI_Hackathon-2026/ ./
   git add -A && git commit -m "Deploy aerosol" && git push
   ```
   Files over 10 MB (the 11.4 MB `models/dengue_model.joblib`) must go through Git LFS, hence `git lfs track`.
3. The Space builds the image and starts it. Open `https://<user>-<space>.hf.space/app/`.
4. Check it: `https://<user>-<space>.hf.space/api/selfcheck?as_of=2025-07-14&horizon=3` must say `"passed": true`.

## Option B: keep Render, switch it to Docker
Render → the service → **Settings** → runtime **Docker** (it finds the `Dockerfile` at the repo root);
remove any custom start command. Render sets `PORT` itself and the image listens on it. The free plan
still sleeps and has 0.1 CPU; a paid instance removes both.

## Run it locally (any machine with Docker)
```sh
docker build -t aerosol .
docker run --rm -p 7860:7860 aerosol        # http://localhost:7860  (dashboard: /app/)
```
Inside the container, `python -m scripts.offline_check` runs the offline kill-switch check.
