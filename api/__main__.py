"""python -m api  ->  serves the dashboard and JSON API on http://127.0.0.1:8000"""
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Open http://127.0.0.1:{port}  (dashboard: /app/, API: /api/meta)")
    uvicorn.run("api.main:app", host=os.environ.get("HOST", "127.0.0.1"), port=port)
