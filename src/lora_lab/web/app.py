"""Read-only web dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from ..config import ProjectConfig
from ..db import Store

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app(project: ProjectConfig) -> FastAPI:
    app = FastAPI(title="lora-lab", version="0.1.0")

    def store() -> Store:
        return Store(project.db_path)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        db = store()
        try:
            counts = db.counts()
            runs = [dict(r) for r in db.runs()]
            recent = [dict(r) for r in db.recent_images(24)]
            images = db.image_count()
        finally:
            db.close()
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "project": project,
                "counts": counts,
                "runs": runs,
                "recent": recent,
                "images": images,
                "grid": project.grid,
            },
        )

    @app.get("/report", response_class=HTMLResponse)
    def report(request: Request, run: str | None = None) -> HTMLResponse:
        from ..analyze.report import build_report

        db = store()
        try:
            run_id = None
            if run:
                row = db.run_by_name(run)
                run_id = int(row["id"]) if row else None
            payload = build_report(db, run_id=run_id, bootstrap=800, stability_draws=300)
            runs = [dict(r) for r in db.runs()]
        finally:
            db.close()
        return TEMPLATES.TemplateResponse(
            request,
            "report.html",
            {"project": project, "payload": payload, "runs": runs, "selected": run},
        )

    @app.get("/api/report")
    def api_report(run: str | None = None) -> dict[str, object]:
        from ..analyze.report import build_report

        db = store()
        try:
            run_id = None
            if run:
                row = db.run_by_name(run)
                run_id = int(row["id"]) if row else None
            return build_report(db, run_id=run_id, bootstrap=800, stability_draws=300)
        finally:
            db.close()

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        db = store()
        try:
            return {
                "grid": project.grid.name,
                "profile": project.grid.profile,
                "counts": db.counts(),
                "images": db.image_count(),
                "runs": [dict(r) for r in db.runs()],
            }
        finally:
            db.close()

    @app.get("/images/{path:path}")
    def image(path: str) -> FileResponse:
        root = (project.data_dir / "images").resolve()
        target = (project.data_dir / path).resolve()
        if not str(target).startswith(str(root)) or not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target)

    return app


def serve(project: ProjectConfig, host: str = "127.0.0.1", port: int = 8099) -> None:
    import uvicorn

    uvicorn.run(create_app(project), host=host, port=port, log_level="info")
