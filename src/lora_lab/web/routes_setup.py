"""Setup UI: scenarios, subjects, references and grid settings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..config import Attribute, GridConfig, Scenario, Subject
from .store import (
    SLUG,
    ConfigError,
    default_ref_dir,
    list_ref_images,
    parse_attributes,
    parse_float_list,
    parse_int_list,
    resolve_ref_dir,
    safe_child,
    save_grid,
    save_project_settings,
    save_scenarios,
    save_subjects,
    writable_dir,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_UPLOAD = 40 * 1024 * 1024


def _state(request: Request):
    return request.app.state.lora


def _flash(url: str, *, ok: str | None = None, err: str | None = None) -> RedirectResponse:
    params = {}
    if ok:
        params["ok"] = ok
    if err:
        params["err"] = err
    return RedirectResponse(url + ("?" + urlencode(params) if params else ""), status_code=303)


def _allowed_roots(root: Path) -> list[Path]:
    roots = []
    for candidate in (Path("/datasets"), Path(root)):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


@router.get("/setup", response_class=HTMLResponse)
def setup(request: Request) -> HTMLResponse:
    state = _state(request)
    project = state.project
    ref_counts = {}
    for subject in project.subjects.values():
        directory = resolve_ref_dir(state.root, subject.ref_dir, subject.id)
        ref_counts[subject.id] = len(list_ref_images(directory))
    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "project": project,
            "scenarios": sorted(project.scenarios.values(), key=lambda s: s.id),
            "subjects": sorted(project.subjects.values(), key=lambda s: s.id),
            "loras": project.grid.loras,
            "ref_counts": ref_counts,
            "jobs": _job_count(project),
        },
    )


def _job_count(project) -> int:
    from ..grid import expand

    try:
        return len(expand(project.grid, project.scenario_list(), project.subjects))
    except Exception:
        return 0


# -- scenarios --------------------------------------------------------------


@router.get("/setup/scenarios/new", response_class=HTMLResponse)
def scenario_new(request: Request) -> HTMLResponse:
    state = _state(request)
    return templates.TemplateResponse(
        request,
        "scenario_form.html",
        {
            "project": state.project,
            "scenario": Scenario(
                id="",
                category="core",
                prompt="{trigger}, ",
                attributes=[Attribute(id="subject_present", question="Is the subject visible?")],
            ),
            "is_new": True,
        },
    )


@router.get("/setup/scenarios/{scenario_id}/edit", response_class=HTMLResponse)
def scenario_edit(request: Request, scenario_id: str) -> HTMLResponse:
    state = _state(request)
    scenario = state.project.scenarios.get(scenario_id)
    if scenario is None:
        return _flash("/setup", err=f"unknown scenario: {scenario_id}")
    return templates.TemplateResponse(
        request,
        "scenario_form.html",
        {"project": state.project, "scenario": scenario, "is_new": False},
    )


@router.post("/setup/scenarios/save")
async def scenario_save(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    is_new = str(form.get("is_new", "")).lower() == "true"
    scenario_id = str(form.get("id", "")).strip()

    try:
        if not SLUG.match(scenario_id):
            raise ConfigError("id: use letters, digits, dot, dash, underscore")
        attributes = parse_attributes(
            [str(x) for x in form.getlist("attr_id")],
            [str(x) for x in form.getlist("attr_question")],
            [str(x) for x in form.getlist("attr_weight")],
        )
        width = str(form.get("width", "")).strip()
        height = str(form.get("height", "")).strip()
        scenario = Scenario(
            id=scenario_id,
            category=str(form.get("category", "core")),
            prompt=str(form.get("prompt", "")).strip(),
            attributes=attributes,
            width=int(width) if width else None,
            height=int(height) if height else None,
        )

        existing = {item.id: item for item in state.project.scenarios.values()}
        if is_new and scenario.id in existing:
            raise ConfigError(f"scenario {scenario.id} already exists")
        if not is_new and scenario.id not in existing:
            raise ConfigError(f"scenario {scenario.id} does not exist")
        existing[scenario.id] = scenario

        save_scenarios(state.root, list(existing.values()))
        state.reload()
    except (ConfigError, ValueError) as exc:
        target = "/setup/scenarios/new" if is_new else f"/setup/scenarios/{scenario_id}/edit"
        return _flash(target, err=str(exc))

    return _flash("/setup", ok=f"scenario {scenario_id} saved")


@router.post("/setup/scenarios/import")
async def scenario_import(request: Request) -> RedirectResponse:
    """Load a scenario pack (YAML/JSON) without touching the filesystem by hand."""

    state = _state(request)
    form = await request.form()
    upload = form.get("file")
    if upload is None or not getattr(upload, "filename", ""):
        return _flash("/setup", err="no file selected")
    try:
        import yaml as _yaml

        raw = await upload.read()
        data = _yaml.safe_load(raw.decode("utf-8"))
        items = data.get("scenarios") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ConfigError("expected a list of scenarios or {scenarios: [...]}")
        imported = {}
        for item in items:
            scenario = Scenario(**item)
            imported[scenario.id] = scenario
        if not imported:
            raise ConfigError("the file contains no scenarios")
        existing = {item.id: item for item in state.project.scenarios.values()}
        existing.update(imported)
        save_scenarios(state.root, list(existing.values()))
        state.reload()
    except ConfigError as exc:
        return _flash("/setup", err=str(exc))
    except Exception as exc:  # noqa: BLE001 - shown in the UI
        return _flash("/setup", err=f"import failed: {exc}")
    return _flash("/setup", ok=f"{len(imported)} scenario(s) imported")


@router.post("/setup/subjects/cleanup")
def subjects_cleanup(request: Request) -> RedirectResponse:
    """Remove subjects that no selected LoRA references."""

    state = _state(request)
    used = {lora.subject for lora in state.project.grid.loras}
    existing = {item.id: item for item in state.project.subjects.values()}
    removed = [key for key in existing if key not in used]
    for key in removed:
        del existing[key]
    if not removed:
        return _flash("/setup", ok="no unused subjects")
    try:
        save_subjects(state.root, list(existing.values()))
        state.reload()
    except ConfigError as exc:
        return _flash("/setup", err=str(exc))
    return _flash("/setup", ok=f"{len(removed)} unused subject(s) removed")


@router.post("/setup/scenarios/{scenario_id}/delete")
def scenario_delete(request: Request, scenario_id: str) -> RedirectResponse:
    state = _state(request)
    existing = {item.id: item for item in state.project.scenarios.values()}
    if scenario_id not in existing:
        return _flash("/setup", err=f"unknown scenario: {scenario_id}")
    del existing[scenario_id]
    try:
        save_scenarios(state.root, list(existing.values()))
        state.reload()
    except ConfigError as exc:
        return _flash("/setup", err=str(exc))
    return _flash("/setup", ok=f"scenario {scenario_id} deleted")


# -- subjects ---------------------------------------------------------------


@router.get("/setup/subjects/new", response_class=HTMLResponse)
def subject_new(request: Request) -> HTMLResponse:
    state = _state(request)
    subject = Subject(id="", trigger="")
    return templates.TemplateResponse(
        request,
        "subject_form.html",
        {"project": state.project, "subject": subject, "is_new": True},
    )


@router.get("/setup/subjects/{subject_id}/edit", response_class=HTMLResponse)
def subject_edit(request: Request, subject_id: str) -> HTMLResponse:
    state = _state(request)
    subject = state.project.subjects.get(subject_id)
    if subject is None:
        return _flash("/setup", err=f"unknown subject: {subject_id}")
    return templates.TemplateResponse(
        request,
        "subject_form.html",
        {"project": state.project, "subject": subject, "is_new": False},
    )


@router.post("/setup/subjects/save")
async def subject_save(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    is_new = str(form.get("is_new", "")).lower() == "true"
    subject_id = str(form.get("id", "")).strip()

    try:
        if not SLUG.match(subject_id):
            raise ConfigError("id: use letters, digits, dot, dash, underscore")
        trigger = str(form.get("trigger", "")).strip()
        ref_dir = str(form.get("ref_dir", "")).strip() or default_ref_dir(state.root, subject_id)
        train_dir = str(form.get("train_dir", "")).strip() or None
        subject = Subject(
            id=subject_id,
            trigger=trigger,
            ref_dir=ref_dir,
            train_dir=train_dir,
        )
        existing = {item.id: item for item in state.project.subjects.values()}
        if is_new and subject.id in existing:
            raise ConfigError(f"subject {subject.id} already exists")
        if not is_new and subject.id not in existing:
            raise ConfigError(f"subject {subject.id} does not exist")
        existing[subject.id] = subject
        save_subjects(state.root, list(existing.values()))
        state.reload()
    except (ConfigError, ValueError) as exc:
        target = "/setup/subjects/new" if is_new else f"/setup/subjects/{subject_id}/edit"
        return _flash(target, err=str(exc))

    return _flash("/setup", ok=f"subject {subject_id} saved")


@router.post("/setup/subjects/{subject_id}/delete")
def subject_delete(request: Request, subject_id: str) -> RedirectResponse:
    state = _state(request)
    existing = {item.id: item for item in state.project.subjects.values()}
    if subject_id not in existing:
        return _flash("/setup", err=f"unknown subject: {subject_id}")
    if any(lora.subject == subject_id for lora in state.project.grid.loras):
        return _flash(
            "/setup",
            err=f"subject {subject_id} still has LoRAs assigned — reassign them first",
        )
    del existing[subject_id]
    try:
        save_subjects(state.root, list(existing.values()))
        state.reload()
    except ConfigError as exc:
        return _flash("/setup", err=str(exc))
    return _flash("/setup", ok=f"subject {subject_id} deleted")


# -- references -------------------------------------------------------------


@router.get("/setup/subjects/{subject_id}/refs", response_class=HTMLResponse)
def refs_page(request: Request, subject_id: str) -> HTMLResponse:
    state = _state(request)
    subject = state.project.subjects.get(subject_id)
    if subject is None:
        return _flash("/setup", err=f"unknown subject: {subject_id}")
    directory = resolve_ref_dir(state.root, subject.ref_dir, subject_id)
    images = list_ref_images(directory)
    return templates.TemplateResponse(
        request,
        "refs.html",
        {
            "project": state.project,
            "subject": subject,
            "directory": str(directory),
            "images": [{"name": p.name, "size": p.stat().st_size} for p in images],
            "exists": directory.is_dir(),
            "writable": writable_dir(directory),
        },
    )


@router.post("/setup/subjects/{subject_id}/refs/upload")
async def refs_upload(request: Request, subject_id: str) -> RedirectResponse:
    state = _state(request)
    subject = state.project.subjects.get(subject_id)
    if subject is None:
        return _flash("/setup", err=f"unknown subject: {subject_id}")

    directory = resolve_ref_dir(state.root, subject.ref_dir, subject_id)
    url = f"/setup/subjects/{subject_id}/refs"

    # Read the multipart body before touching the filesystem. Bailing out
    # earlier leaves the browser streaming into a closed socket, which surfaces
    # as "connection was reset" instead of an error message.
    form = await request.form()
    uploads: list[tuple[str, str, Any]] = []
    for upload in form.getlist("files"):
        filename = getattr(upload, "filename", "") or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIX:
            continue
        uploads.append((Path(filename).stem, suffix, upload))

    if not uploads:
        return _flash(url, err="Keine Bilddatei im Upload (png, jpg, jpeg, webp, bmp).")

    if not writable_dir(directory):
        return _flash(url, err=_not_writable(directory))

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _flash(url, err=_not_writable(directory, exc))

    saved = 0
    try:
        for source_stem, suffix, upload in uploads:
            data = await upload.read()
            if not data or len(data) > MAX_UPLOAD:
                continue
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem)[:80] or "ref"
            target = directory / f"{stem}{suffix}"
            counter = 1
            while target.exists():
                target = directory / f"{stem}-{counter}{suffix}"
                counter += 1
            target.write_bytes(data)
            saved += 1
    except OSError as exc:
        return _flash(url, err=f"Upload fehlgeschlagen: {exc}")

    if not saved:
        return _flash(url, err="Keine der Dateien war lesbar oder unter 40 MB.")

    if not subject.ref_dir:
        subject.ref_dir = str(directory)
        existing = {item.id: item for item in state.project.subjects.values()}
        existing[subject.id] = subject
        save_subjects(state.root, list(existing.values()))
        state.reload()

    return _flash(url, ok=f"{saved} reference image(s) uploaded to {directory}")


def _not_writable(directory: Path, exc: OSError | None = None) -> str:
    reason = f" ({exc.strerror})" if exc is not None and exc.strerror else ""
    return (
        f"Referenzordner {directory} ist nicht beschreibbar{reason}. "
        "Meist liegt es am ro-Mount: /datasets ist read-only, /datasets/_refs "
        "braucht einen eigenen :rw-Mount. Alternativ im Subjekt einen "
        "beschreibbaren Ordner eintragen."
    )



@router.post("/setup/subjects/{subject_id}/refs/delete")
async def refs_delete(request: Request, subject_id: str) -> RedirectResponse:
    state = _state(request)
    subject = state.project.subjects.get(subject_id)
    if subject is None:
        return _flash("/setup", err=f"unknown subject: {subject_id}")
    form = await request.form()
    filename = str(form.get("filename", ""))
    directory = resolve_ref_dir(state.root, subject.ref_dir, subject_id)
    try:
        target = safe_child(directory, filename)
        target.unlink(missing_ok=True)
    except (ConfigError, OSError) as exc:
        return _flash(f"/setup/subjects/{subject_id}/refs", err=str(exc))
    return _flash(f"/setup/subjects/{subject_id}/refs", ok=f"{filename} deleted")


@router.get("/refs/{subject_id}/{filename}")
def ref_image(request: Request, subject_id: str, filename: str) -> FileResponse:
    state = _state(request)
    subject = state.project.subjects.get(subject_id)
    if subject is None:
        return _flash("/setup", err=f"unknown subject: {subject_id}")
    directory = resolve_ref_dir(state.root, subject.ref_dir, subject_id)
    target = safe_child(directory, filename)
    if not target.is_file():
        return _flash(f"/setup/subjects/{subject_id}/refs", err="image not found")
    return FileResponse(target)


# -- folder browser ---------------------------------------------------------


@router.get("/api/browse")
def browse(request: Request, path: str = "") -> dict[str, Any]:
    state = _state(request)
    roots = _allowed_roots(state.root)

    if not path:
        return {
            "path": "",
            "roots": [{"name": str(root), "path": str(root)} for root in roots],
            "dirs": [],
            "parent": None,
            "images": 0,
        }

    target = Path(path).resolve()
    if not any(target == root or root in target.parents for root in roots):
        return {"error": "path is outside the allowed mounts", "dirs": [], "parent": None}
    if not target.is_dir():
        return {"error": "not a directory", "dirs": [], "parent": None}

    dirs = []
    images = 0
    for child in sorted(target.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            # cap the folder list only; the image count stays complete
            if len(dirs) < 300:
                dirs.append({"name": child.name, "path": str(child)})
        elif child.suffix.lower() in IMAGE_SUFFIX:
            images += 1

    parent = str(target.parent) if any(root in target.parents for root in roots) else None
    return {"path": str(target), "dirs": dirs, "parent": parent, "images": images, "roots": []}


# -- grid -------------------------------------------------------------------


@router.get("/setup/grid", response_class=HTMLResponse)
def grid_page(request: Request) -> HTMLResponse:
    state = _state(request)
    project = state.project
    return templates.TemplateResponse(
        request,
        "grid.html",
        {
            "project": project,
            "grid": project.grid,
            "scenarios": sorted(project.scenarios.values(), key=lambda s: s.id),
            "selected": {s.id for s in project.scenario_list()},
            "profiles": sorted(set(project.grid.profiles) | {project.grid.profile}),
            "loras": len(project.grid.loras),
            "jobs": _job_count(project),
        },
    )


@router.post("/setup/grid/save")
async def grid_save(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    grid = state.project.grid
    try:
        strengths = parse_float_list(str(form.get("strengths", "")), "strengths")
        seeds = parse_int_list(str(form.get("seeds", "")), "seeds")
        scenarios = [str(x) for x in form.getlist("scenarios")]
        if not scenarios:
            raise ConfigError("select at least one scenario")
        extra_conditions = [
            str(x) for x in form.getlist("conditions") if x in ("base_only", "no_trigger")
        ]
        conditions = ["lora", *extra_conditions]
        name = str(form.get("name", "")).strip() or grid.name
        profile = str(form.get("profile", grid.profile))
        if profile not in grid.profiles and profile != grid.profile:
            raise ConfigError(f"unknown profile: {profile}")
        data = grid.model_dump()
        data.update(
            {
                "name": name,
                "strengths": strengths,
                "seeds": seeds,
                "scenarios": scenarios,
                "conditions": conditions,
                "profile": profile,
                "negative_prompt": str(form.get("negative_prompt", grid.negative_prompt)),
                "width": int(str(form.get("width", grid.width))),
                "height": int(str(form.get("height", grid.height))),
            }
        )
        updated = GridConfig(**data)
        save_grid(state.root, updated)
        state.reload()
    except (ConfigError, ValueError) as exc:
        return _flash("/setup/grid", err=str(exc))
    return _flash("/setup/grid", ok="grid saved")


# -- vlm settings -----------------------------------------------------------


@router.post("/setup/vlm/save")
async def vlm_save(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    vlm_url = str(form.get("vlm_url", "")).strip() or None
    vlm_model = str(form.get("vlm_model", "")).strip() or None
    vlm_api_key = str(form.get("vlm_api_key", "")).strip() or None
    save_project_settings(
        state.root,
        vlm_url=vlm_url,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
    )
    state.reload()
    return _flash("/setup", ok="VLM-Einstellungen gespeichert")


@router.post("/setup/vlm/test")
async def vlm_test(request: Request) -> RedirectResponse:
    from ..scorers.vlm import VLMJudge, VLMSettings

    state = _state(request)
    form = await request.form()
    vlm_url = str(form.get("vlm_url", "")).strip() or state.project.vlm_url or ""
    vlm_model = str(form.get("vlm_model", "")).strip() or state.project.vlm_model or ""
    vlm_api_key = str(form.get("vlm_api_key", "")).strip() or state.project.vlm_api_key or None

    if not vlm_url or not vlm_model:
        return _flash("/setup", err="VLM-Test: Bitte URL und Modellname eintragen.")

    save_project_settings(
        state.root,
        vlm_url=vlm_url,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
    )
    state.reload()

    with VLMJudge(
        VLMSettings(base_url=vlm_url, model=vlm_model, api_key=vlm_api_key, timeout=10.0)
    ) as judge:
        result = judge.ping()

    if not result["ok"]:
        return _flash("/setup", err=f"VLM nicht erreichbar ({vlm_url}): {result['error']}")
    if not result["matched"]:
        available = ", ".join(result["models"][:8]) or "keine"
        return _flash(
            "/setup",
            err=f"VLM erreichbar, aber Modell '{vlm_model}' nicht gefunden (verfügbar: {available})",
        )
    return _flash("/setup", ok=f"VLM-Verbindung OK: {vlm_model} unter {vlm_url} antwortet.")
