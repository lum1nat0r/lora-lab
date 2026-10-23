"""Run page: plan, generate, score, stop."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..grid import expand, summary

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _state(request: Request):
    return request.app.state.lora


def _flash(url: str, *, ok: str | None = None, err: str | None = None) -> RedirectResponse:
    params = {}
    if ok:
        params["ok"] = ok
    if err:
        params["err"] = err
    return RedirectResponse(url + ("?" + urlencode(params) if params else ""), status_code=303)


def _plan(state) -> dict:
    project = state.project
    try:
        jobs = expand(project.grid, project.scenario_list(), project.subjects)
    except Exception as exc:  # noqa: BLE001 - shown on the page
        return {"error": str(exc), "total": 0, "counts": {}, "minutes": 0}
    counts = summary(jobs)
    return {
        "error": None,
        "total": counts.get("total", 0),
        "counts": counts,
        "minutes": round(counts.get("total", 0) * 15 / 60),
    }


@router.get("/run", response_class=HTMLResponse)
def run_page(request: Request) -> HTMLResponse:
    state = _state(request)
    project = state.project
    refs = {
        subject.id: bool(
            subject.ref_dir
            and project.path(subject.ref_dir).is_dir()
            and any(project.path(subject.ref_dir).iterdir())
        )
        for subject in project.subjects.values()
    }
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "project": project,
            "plan": _plan(state),
            "snapshot": state.runner.snapshot(),
            "loras": len(project.grid.loras),
            "scenarios": len(project.scenario_list()),
            "refs": refs,
            "refs_ok": sum(1 for value in refs.values() if value),
            "refs_total": len(refs),
            "vlm": bool(os.environ.get("VLM_JUDGE_URL") and os.environ.get("VLM_JUDGE_MODEL")),
            "default_limit": 8,
        },
    )


@router.post("/run/generate")
async def run_generate(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    raw_limit = str(form.get("limit", "")).strip()
    limit = None
    if raw_limit:
        try:
            limit = max(1, int(raw_limit))
        except ValueError:
            return _flash("/run", err="limit is not a number")
    if not state.project.grid.loras:
        return _flash("/run", err="no LoRAs selected — see the LoRAs page")
    try:
        state.runner.start("generate", state.runner.generate, limit)
    except RuntimeError as exc:
        return _flash("/run", err=str(exc))
    return _flash("/run", ok="generation started")


@router.post("/run/score")
def run_score(request: Request) -> RedirectResponse:
    state = _state(request)
    try:
        state.runner.start("score", state.runner.score)
    except RuntimeError as exc:
        return _flash("/run", err=str(exc))
    return _flash("/run", ok="scoring started")


@router.post("/run/stop")
def run_stop(request: Request) -> RedirectResponse:
    state = _state(request)
    state.runner.stop()
    return _flash("/run", ok="stop requested")


@router.get("/api/run")
def run_status(request: Request) -> dict:
    return _state(request).runner.snapshot()
