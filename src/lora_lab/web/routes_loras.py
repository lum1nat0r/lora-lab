"""LoRA curation UI: scan the models mount, pick candidates, assign subjects."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..config import LoraEntry
from ..discover import discover_loras
from .store import (
    ConfigError,
    load_all_loras,
    save_all_loras,
    save_grid,
    save_loras,
)

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


def _all_items(state) -> list[dict]:
    items = load_all_loras(state.root)
    if items:
        return items
    return [
        {
            "id": lora.id,
            "subject": lora.subject,
            "file": lora.file,
            "epoch": lora.epoch,
        }
        for lora in state.project.grid.loras
    ]


@router.get("/loras", response_class=HTMLResponse)
def loras_page(request: Request) -> HTMLResponse:
    state = _state(request)
    project = state.project
    items = _all_items(state)
    selected = {lora.id for lora in project.grid.loras}
    subjects = sorted(project.subjects)
    grouped: dict[str, list[dict]] = {}
    by_folder: dict[str, list[dict]] = {}
    for index, item in enumerate(items):
        rel = str(item.get("file") or "")
        parts = [p for p in rel.split("/") if p]
        folder = "/".join(parts[:-1]) if len(parts) > 1 else "(root)"
        family = parts[0] if len(parts) > 1 else "(root)"
        item["index"] = index
        item["folder"] = folder
        item["family"] = family
        item["filename"] = parts[-1] if parts else rel
        item["selected"] = item.get("id") in selected
        grouped.setdefault(str(item.get("subject") or "?"), []).append(item)
        by_folder.setdefault(folder, []).append(item)

    groups = []
    for folder, group_items in sorted(by_folder.items()):
        groups.append(
            {
                "folder": folder,
                "family": group_items[0]["family"],
                "subject": group_items[0].get("subject") or (subjects[0] if subjects else ""),
                "rank": next((it.get("rank") for it in group_items if it.get("rank")), None),
                "selected_count": sum(1 for it in group_items if it["selected"]),
                "entries": group_items,
            }
        )
    families = sorted({g["family"] for g in groups})

    from ..grid import expand

    try:
        jobs = len(expand(project.grid, project.scenario_list(), project.subjects))
    except Exception:
        jobs = 0

    return templates.TemplateResponse(
        request,
        "loras.html",
        {
            "project": project,
            "items": items,
            "grouped": sorted(grouped.items()),
            "groups": groups,
            "families": families,
            "subjects": subjects,
            "selected_count": len(selected),
            "jobs": jobs,
            "scan_path": os.environ.get("LORA_LAB_LORA_ROOT", "/models/loras"),
            "estimate_minutes": round(jobs * 15 / 60),
        },
    )


@router.post("/loras/scan")
async def loras_scan(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    default_root = Path(os.environ.get("LORA_LAB_LORA_ROOT", "/models/loras"))
    path = Path(str(form.get("path", "")).strip() or default_root)
    if not path.is_dir():
        return _flash("/loras", err=f"not a directory: {path}")

    skipped: list[str] = []
    found = discover_loras(
        path,
        subjects=list(state.project.subjects),
        skipped=skipped,
        base_root=default_root,
    )
    if not found:
        return _flash(
            "/loras",
            err=f"no LoRAs matched a subject under {path} — add subjects first",
        )

    # keep the current selection when re-scanning
    selected_ids = {lora.id for lora in state.project.grid.loras}
    items = []
    for item in found:
        item["selected"] = item["id"] in selected_ids
        items.append(item)
    save_all_loras(state.root, items)

    message = f"{len(items)} LoRAs found under {path}"
    if skipped:
        message += f", {len(skipped)} skipped (no matching subject)"
    return _flash("/loras", ok=message)


@router.post("/loras/save")
async def loras_save(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    items = _all_items(state)
    picked: list[LoraEntry] = []

    try:
        for index, item in enumerate(items):
            if not form.get(f"pick_{index}"):
                continue
            subject = str(form.get(f"subject_{index}", item.get("subject", ""))).strip()
            epoch_raw = str(form.get(f"epoch_{index}", item.get("epoch", 0))).strip() or "0"
            try:
                epoch = int(epoch_raw)
            except ValueError as exc:
                raise ConfigError(f"{item.get('id')}: epoch is not a number") from exc
            picked.append(
                LoraEntry(
                    id=str(item.get("id")),
                    subject=subject,
                    file=str(item.get("file")),
                    epoch=epoch,
                )
            )
        save_loras(state.root, picked)

        grid = state.project.grid
        data = grid.model_dump()
        data["loras"] = []
        data["loras_file"] = "config/local/loras.yaml"
        save_grid(state.root, type(grid)(**data))
        state.reload()
    except (ConfigError, ValueError) as exc:
        return _flash("/loras", err=str(exc))

    return _flash("/loras", ok=f"{len(picked)} LoRAs selected")
