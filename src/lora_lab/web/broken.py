"""Degraded dashboard for an invalid configuration.

A config error used to kill the process, which in Docker means a restart loop
and no way to `docker exec` into the container to fix it. The web UI now comes
up anyway and explains what is wrong, so the container stays alive.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

PAGE = """<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lora-lab · config error</title>
<style>
 body {{ margin:0; background:#0f1115; color:#e6e8ee;
        font:14px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
 main {{ max-width:900px; margin:0 auto; padding:40px 24px; }}
 h1 {{ font-size:18px; margin:0 0 6px; }}
 .bad {{ color:#ff7875; }}
 pre {{ background:#171a21; border:1px solid #262b36; border-radius:8px;
        padding:14px 16px; overflow:auto; white-space:pre-wrap; }}
 code {{ background:#11141a; padding:1px 5px; border-radius:4px; }}
 .muted {{ color:#8b93a7; }}
 li {{ margin:4px 0; }}
</style></head>
<body><main>
<h1>lora-lab <span class="bad">config error</span></h1>
<p class="muted">Der Container läuft weiter — die Konfiguration muss korrigiert werden.</p>
<pre>{error}</pre>
<h2>Projekt</h2>
<pre>root: {root}
config: {config}
local: {local}</pre>
<h2>Reihenfolge</h2>
<ol>
  <li><code>config/local/&lt;name&gt;.yaml</code> — gewinnt immer</li>
  <li><code>config/&lt;name&gt;.yaml</code> — beim Start gesätes Beispiel</li>
  <li><code>/app/defaults/&lt;name&gt;.yaml</code> — read-only Fallback im Image</li>
</ol>
<h2>Reparieren</h2>
<pre>docker exec -it lora-lab lora-lab --root {root} discover \\
  --datasets /datasets --loras /models/loras --apply
docker exec -it lora-lab lora-lab --root {root} doctor</pre>
<p class="muted">Mehr in <code>docs/PATHS.md</code> im Repository.</p>
</main></body></html>
"""


def create_broken_app(root: Path, error: Exception) -> FastAPI:
    app = FastAPI(title="lora-lab (config error)")
    root = Path(root)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "degraded", "error": str(error)}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(
            PAGE.format(
                error=escape(str(error)),
                root=escape(str(root)),
                config=escape(str(root / "config")),
                local=escape(str(root / "config" / "local")),
            )
        )

    return app


def serve_broken(root: Path, error: Exception, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_broken_app(root, error), host=host, port=port, log_level="warning")
