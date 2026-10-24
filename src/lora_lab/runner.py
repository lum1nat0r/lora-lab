"""Job runner: build prompts, talk to ComfyUI, persist results."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .comfy.client import ComfyClient, ComfyError
from .comfy.workflow import (
    bypass_lora,
    drop_nodes,
    is_ui_format,
    load_workflow,
    prompt_hash,
    render_prompt,
    required_nodes,
    ui_to_api,
    validate_prompt,
)
from .config import ProjectConfig
from .db import Store
from .grid import Job


@dataclass
class RunStats:
    total: int = 0
    done: int = 0
    skipped: int = 0
    failed: int = 0


def build_prompt(
    job: Job,
    project: ProjectConfig,
    api: dict[str, Any],
    *,
    object_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Render one job into an API prompt plus its content hash."""

    scenario = project.scenarios[job.scenario]
    subject = project.subjects[job.subject]
    bindings = project.grid.bindings
    profile = project.grid.profiles.get(job.profile)

    if job.condition == "no_trigger":
        positive = scenario.strip_trigger(subject.trigger)
    else:
        positive = scenario.render(subject.trigger)

    prompt = render_prompt(
        api,
        bindings=bindings,
        positive=positive,
        negative=project.grid.negative_prompt,
        seed=job.seed,
        filename_prefix=job.prefix(),
        lora_file=job.lora_file or None,
        lora_strength=job.strength,
        width=scenario.width or project.grid.width,
        height=scenario.height or project.grid.height,
        overrides=profile.overrides if profile else None,
    )

    if job.is_baseline:
        prompt = bypass_lora(prompt, bindings.lora_node)

    prompt = drop_nodes(prompt, bindings.drop_nodes)

    required = required_nodes(bindings)
    if job.is_baseline:
        required.discard(str(bindings.lora_node))
    validate_prompt(prompt, required)

    return prompt, prompt_hash(prompt, bindings)


def run_jobs(
    project: ProjectConfig,
    store: Store,
    jobs: Iterable[Job],
    client: ComfyClient,
    *,
    run_id: int,
    limit: int | None = None,
    dry_run: bool = False,
    retries: int = 2,
    on_progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    object_info: dict[str, Any] | None = None,
) -> RunStats:
    stats = RunStats()
    planned = 0
    workflow_path = project.path(project.grid.workflow_for(project.grid.profile))
    raw = load_workflow(workflow_path)
    api = ui_to_api(raw) if is_ui_format(raw) else raw

    images_root = project.data_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        if should_cancel is not None and should_cancel():
            break
        stats.total += 1
        prompt, job_hash = build_prompt(job, project, api, object_info=object_info)

        if not dry_run and store.reuse_completed_job(run_id, job_hash, job, prompt):
            stats.skipped += 1
            _progress(on_progress, f"skip  {job.label()}")
            continue
        if dry_run and store.job_status(job_hash, run_id=run_id) == "done":
            stats.skipped += 1
            _progress(on_progress, f"skip  {job.label()}")
            continue

        if dry_run:
            _progress(on_progress, f"plan  {job.label()}  {job_hash[:10]}")
            planned += 1
            if limit is not None and planned >= limit:
                break
            continue

        job_id = store.add_job(run_id, job_hash, job, prompt)
        store.set_status(job_id, "running")

        attempt = 0
        while True:
            attempt += 1
            try:
                prompt_id = client.queue_prompt(prompt)
                entry = (
                    client.wait(prompt_id, should_cancel=should_cancel)
                    if should_cancel is not None
                    else client.wait(prompt_id)
                )
                _save_outputs(project, store, job_id, job, entry, images_root, client)
                store.set_status(job_id, "done", prompt_id=prompt_id)
                stats.done += 1
                _progress(on_progress, f"done  {job.label()}")
                break
            except (ComfyError, OSError) as exc:
                if should_cancel is not None and should_cancel():
                    store.set_status(job_id, "pending", error="cancelled")
                    return stats
                if attempt > retries:
                    store.set_status(job_id, "failed", error=str(exc))
                    stats.failed += 1
                    _progress(on_progress, f"FAIL  {job.label()}: {exc}")
                    break
                time.sleep(2.0 * attempt)

        if limit is not None and stats.done + stats.failed >= limit:
            break

    return stats


def _save_outputs(
    project: ProjectConfig,
    store: Store,
    job_id: int,
    job: Job,
    entry: dict[str, Any],
    images_root: Path,
    client: ComfyClient,
) -> None:
    outputs = entry.get("outputs") or {}
    refs: list[dict[str, Any]] = []
    for node_output in outputs.values():
        for image in node_output.get("images") or []:
            refs.append(image)
    if not refs:
        raise ComfyError("ComfyUI returned no images")

    target_dir = images_root / job.prefix()
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, ref in enumerate(refs):
        filename = ref.get("filename") or f"image_{idx}.png"
        subfolder = ref.get("subfolder") or ""
        type_ = ref.get("type") or "output"
        data = client_view(project, client, filename, subfolder, type_)
        path = target_dir / filename
        path.write_bytes(data)

        width = height = None
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception:  # pragma: no cover - corrupt file, keep going
            pass

        store.add_image(
            job_id=job_id,
            idx=idx,
            filename=filename,
            subfolder=subfolder,
            type_=type_,
            path=str(path.relative_to(project.data_dir)),
            sha256=hashlib.sha256(data).hexdigest(),
            width=width,
            height=height,
            size=len(data),
        )


def client_view(
    project: ProjectConfig,
    client: ComfyClient,
    filename: str,
    subfolder: str,
    type_: str,
) -> bytes:
    """Download a produced image.

    Uses the shared output volume when it is configured (fast, no HTTP), and
    falls back to ComfyUI's ``/view`` endpoint otherwise.
    """

    if project.comfy_output_dir:
        candidate = project.comfy_output_dir / subfolder / filename
        if candidate.exists():
            return candidate.read_bytes()

    return client.view(filename, subfolder, type_)


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
