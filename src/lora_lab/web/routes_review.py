"""Human review: gold labels for judge calibration and the blind A/B tournament."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..analyze.agreement import DEFECT_PREFIX, MIN_LABELS, REJECT_KAPPA, TRUST_KAPPA, calibration
from ..analyze.report import samples_from_store
from ..analyze.stats import analyse_all
from ..analyze.tournament import op_of, pick_pair, standings
from ..db import Store
from ..scorers.defects import DEFECT_QUESTIONS, PROMPT_DEFECT_QUESTIONS

# the label page shows the prompt, so a human can tick the prompt defects too
LABEL_DEFECTS = {**DEFECT_QUESTIONS, **PROMPT_DEFECT_QUESTIONS}

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

FINALISTS = 6


def _state(request: Request):
    return request.app.state.lora


def _run_id(store: Store, name: str | None) -> int | None:
    if not name:
        return None
    row = store.run_by_name(name)
    return int(row["id"]) if row else None


def _back(url: str, run: str | None, **params: str) -> RedirectResponse:
    if run:
        params["run"] = run
    return RedirectResponse(url + ("?" + urlencode(params) if params else ""), status_code=303)


def _votes(store: Store, run_id: int | None) -> list[tuple[str, str, str]]:
    return [
        (
            op_of(r["left_lora"], r["subject"], r["left_epoch"], r["left_strength"]),
            op_of(r["right_lora"], r["subject"], r["right_epoch"], r["right_strength"]),
            str(r["winner"]),
        )
        for r in store.comparisons(run_id)
    ]


# -- gold labels -----------------------------------------------------------
@router.get("/label", response_class=HTMLResponse)
def label_page(request: Request, run: str | None = None) -> HTMLResponse:
    state = _state(request)
    store = Store(state.project.db_path)
    try:
        image = store.next_label_image(_run_id(store, run))
        image = dict(image) if image else None
        result = calibration(store)
        labelled = store.label_count()
        runs = [dict(r) for r in store.runs()]
    finally:
        store.close()

    scenario = state.project.scenarios.get(image["scenario"]) if image else None
    return templates.TemplateResponse(
        request,
        "label.html",
        {
            "project": state.project,
            "image": image,
            "scenario": scenario,
            "defects": LABEL_DEFECTS,
            "calibration": result,
            "labelled": labelled,
            "runs": runs,
            "selected": run,
            "min_labels": MIN_LABELS,
            "trust_kappa": TRUST_KAPPA,
            "reject_kappa": REJECT_KAPPA,
        },
    )


@router.post("/label")
async def label_save(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    run = str(form.get("run", "")).strip() or None
    try:
        image_id = int(str(form.get("image_id", "")))
    except ValueError:
        return _back("/label", run, err="image_id fehlt")

    labels: dict[str, bool] = {}
    for key, value in form.items():
        if key.startswith("attr:") and value in ("yes", "no"):
            labels[key.removeprefix("attr:")] = value == "yes"
    # the defect list is a checklist: what is not ticked was looked at and absent
    for key in LABEL_DEFECTS:
        labels[DEFECT_PREFIX + key] = form.get(f"defect:{key}") == "on"

    store = Store(state.project.db_path)
    try:
        if store.conn.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone() is None:
            return _back("/label", run, err="Bild nicht gefunden")
        store.set_labels(image_id, labels)
    finally:
        store.close()
    return _back("/label", run)


# -- blind A/B --------------------------------------------------------------
@router.get("/duel", response_class=HTMLResponse)
def duel_page(request: Request, run: str | None = None) -> HTMLResponse:
    state = _state(request)
    store = Store(state.project.db_path)
    try:
        run_id = _run_id(store, run)
        rows = [dict(r) for r in store.images_for_run(run_id)]
        votes = _votes(store, run_id)
        results = analyse_all(samples_from_store(store, run_id), bootstrap=100)
        runs = [dict(r) for r in store.runs()]
    finally:
        store.close()

    # acceptance is for the finalists of the automated ranking; without
    # scores every operating point takes part
    finalists = {
        op_of(r.lora_id, r.subject, r.epoch, r.strength) for r in results[:FINALISTS]
    } or None
    pair = pick_pair(rows, votes, finalists)
    scenario = state.project.scenarios.get(pair[0]["scenario"]) if pair else None
    return templates.TemplateResponse(
        request,
        "duel.html",
        {
            "project": state.project,
            "pair": pair,
            "scenario": scenario,
            "standings": standings(votes) if votes else [],
            "votes": len(votes),
            "finalists": sorted(finalists or ()),
            "runs": runs,
            "selected": run,
        },
    )


@router.post("/duel")
async def duel_vote(request: Request) -> RedirectResponse:
    state = _state(request)
    form = await request.form()
    run = str(form.get("run", "")).strip() or None
    winner = str(form.get("winner", ""))
    try:
        left_id = int(str(form.get("left_id", "")))
        right_id = int(str(form.get("right_id", "")))
    except ValueError:
        return _back("/duel", run, err="Bild-IDs fehlen")
    if winner not in ("left", "right", "tie"):
        return _back("/duel", run, err="ungültige Stimme")

    store = Store(state.project.db_path)
    try:
        found = store.conn.execute(
            "SELECT COUNT(*) AS n FROM images WHERE id IN (?, ?)", (left_id, right_id)
        ).fetchone()
        if left_id == right_id or int(found["n"]) != 2:
            return _back("/duel", run, err="Bild nicht gefunden")
        store.add_comparison(left_id, right_id, winner)
    finally:
        store.close()
    return _back("/duel", run)
