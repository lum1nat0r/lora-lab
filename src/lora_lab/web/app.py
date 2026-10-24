"""FastAPI application: dashboard, report, images and the setup/run routers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import ProjectConfig, load_project
from ..db import Store
from .routes_loras import router as loras_router
from .routes_runs import router as runs_router
from .routes_setup import router as setup_router
from .state import AppState

HERE = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))


def create_app(
    project: ProjectConfig,
    reload_fn=None,
) -> FastAPI:
    app = FastAPI(title="lora-lab", version="0.2.0")
    state = AppState(project, reload_fn)
    app.state.lora = state

    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    app.include_router(setup_router)
    app.include_router(loras_router)
    app.include_router(runs_router)

    def store() -> Store:
        return Store(state.project.db_path)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        project = state.project
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
                "loras": len(project.grid.loras),
                "scenarios": len(project.scenario_list()),
                "snapshot": state.runner.snapshot(),
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
            {"project": state.project, "payload": payload, "runs": runs, "selected": run},
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

    @app.post("/report/delete-lora")
    async def report_delete_lora(request: Request) -> RedirectResponse:
        if state.runner.running():
            return RedirectResponse(
                "/report?" + urlencode({"err": "Lauf aktiv – bitte erst stoppen"}),
                status_code=303,
            )
        form = await request.form()
        lora_id = str(form.get("lora_id", "")).strip()
        run_name = str(form.get("run", "")).strip() or None
        raw_epoch = str(form.get("epoch", "")).strip()
        raw_strength = str(form.get("strength", "")).strip()
        epoch = int(raw_epoch) if raw_epoch else None
        strength = float(raw_strength) if raw_strength else None

        if not lora_id:
            return RedirectResponse("/report?" + urlencode({"err": "lora_id fehlt"}), status_code=303)

        query_id = "" if lora_id.endswith(":base") else lora_id
        db = store()
        try:
            run_id = None
            if run_name:
                row = db.run_by_name(run_name)
                run_id = int(row["id"]) if row else None
            stats = db.delete_lora(
                state.project.data_dir,
                query_id,
                epoch=epoch,
                strength=strength,
                run_id=run_id,
            )
        finally:
            db.close()

        params: dict[str, str] = {
            "ok": f"{lora_id} gelöscht ({stats['files']} Bilder, {stats['jobs']} Jobs entfernt)"
        }
        if run_name:
            params["run"] = run_name
        return RedirectResponse("/report?" + urlencode(params), status_code=303)

    @app.post("/report/delete-run")
    async def report_delete_run(request: Request) -> RedirectResponse:
        if state.runner.running():
            return RedirectResponse(
                "/report?" + urlencode({"err": "Lauf aktiv – bitte erst stoppen"}),
                status_code=303,
            )
        form = await request.form()
        run_name = str(form.get("run", "")).strip() or None

        db = store()
        try:
            run_id = None
            if run_name:
                row = db.run_by_name(run_name)
                if not row:
                    return RedirectResponse(
                        "/report?" + urlencode({"err": f"Run '{run_name}' nicht gefunden"}),
                        status_code=303,
                    )
                run_id = int(row["id"])
            stats = db.delete_run(state.project.data_dir, run_id=run_id)
        finally:
            db.close()

        label = f"Run '{run_name}'" if run_name else "Alle Runs"
        params = {"ok": f"{label} gelöscht ({stats['files']} Bilder, {stats['jobs']} Jobs entfernt)"}
        return RedirectResponse("/report?" + urlencode(params), status_code=303)

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        db = store()
        try:
            return {
                "grid": state.project.grid.name,
                "profile": state.project.grid.profile,
                "counts": db.counts(),
                "images": db.image_count(),
                "runs": [dict(r) for r in db.runs()],
            }
        finally:
            db.close()

    @app.get("/images/{path:path}")
    def image(path: str) -> FileResponse:
        root = (state.project.data_dir / "images").resolve()
        target = (state.project.data_dir / path).resolve()
        if not str(target).startswith(str(root)) or not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target)

    return app


def serve(project: ProjectConfig, host: str = "127.0.0.1", port: int = 8099) -> None:
    import uvicorn

    app = create_app(project, reload_fn=lambda: load_project(project.root))
    uvicorn.run(app, host=host, port=port, log_level="info")
