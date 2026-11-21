"""python -m api  ->  serves the marketing page, dashboard and JSON API on http://127.0.0.1:8000"""
import os

from api.main import create_app

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app = create_app()
    print(f"Open http://{host}:{port}   (dashboard: /app/   API: /api/meta)")
    app.run(host=host, port=port, debug=False, threaded=True)
