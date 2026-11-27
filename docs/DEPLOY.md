# Deploying the app (Docker)

The `Dockerfile` runs the JSON API, the web frontend and the model in one container, with no network
access needed at runtime. It is faster than `python -m api` on a free host because:

- **gunicorn** (a production server) replaces Flask's development server: `WEB_CONCURRENCY` worker processes
  (default 2) x 4 threads;
- **`--preload`**: the model and lag features load once, before the workers fork, so they share that memory;
- the **CSV cache is built into the image**, so a cold start never parses the xlsx;
- the image build **fails if the model file does not load**, instead of retraining at startup.

Measured locally with the image's exact server command (`gunicorn 'api.main:create_app()' --preload --workers 2
--threads 4`, MacBook, 27 Nov 2026): ready in about 3 s, `/api/selfcheck` passed, `/api/crew_plan` 0.35 s.
Speed on a host depends on its CPU: a free Render instance has 0.1 CPU and sleeps after 15 idle minutes.

## Option A (used for the live URL): Render, free plan, Docker
Render's free plan has 512 MB of memory and 0.1 CPU, and the service sleeps after 15 idle minutes.
Two settings make the app fit:
- `WEB_CONCURRENCY=1`: one worker process (the image defaults to 2). Measured in a container limited to
  0.1 CPU and 512 MB: memory 267 MB after every heavy route; with 2 workers it reached 469 MB, too close to the limit.
- The case → supplies table is read once at startup (`api/service.py: warm_up()`), so no visitor waits for the xlsx.

Steps (dashboard.render.com):
1. **New** → **Web Service** → **Public Git Repository** → `https://github.com/ameerul-muminin/aerosol-all` → **Connect**.
2. Name `aerosol-all`, **Language: Docker**, branch `main`, region **Singapore** (closest to Bangladesh),
   instance type **Free**.
3. **Environment Variables:** `WEB_CONCURRENCY` = `1`. Render sets `PORT` itself and the image listens on it.
4. **Advanced → Health Check Path:** `/api/health`. Then **Deploy Web Service**.
5. Check it: `https://<service>.onrender.com/api/selfcheck` must say `"passed": true`, then open `/app/`.

To reuse an existing Render service instead (keeping its URL): **Settings** → language **Docker**, remove any
custom start command, add `WEB_CONCURRENCY=1`, then deploy.

In the same 0.1 CPU / 512 MB container: about 2 minutes from start to the first answer (paid at each deploy and each
wake-up after sleep), then every route answers in under 6 s and cached answers at once. Open the URL a few minutes
before a live demo so it is awake.

## Option B: Hugging Face Spaces (needs a PRO account for Docker)
On 27 Nov 2026 Hugging Face refused to create a Docker Space on the free `cpu-basic` hardware without a PRO
subscription (free accounts can only host static Spaces, and this app needs its Python server). With PRO:
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

## Run it locally (any machine with Docker)
```sh
docker build -t aerosol .
docker run --rm -p 7860:7860 aerosol        # http://localhost:7860  (dashboard: /app/)
```
Inside the container, `python -m scripts.offline_check` runs the offline kill-switch check.
