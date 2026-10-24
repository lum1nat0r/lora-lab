"""Apply scorers to stored images."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..db import Store
from .adherence import AdherenceScorer
from .base import ImageRef, Scorer
from .defects import FaceCountScorer, HealthScorer
from .identity import (
    ArcFaceEmbedder,
    IdentityScorer,
    build_references,
    build_training_embeddings,
    calibrate_from_references,
)


def build_default_scorers(
    project: ProjectConfig,
    *,
    log: Callable[[str], None] | None = None,
    judge: Any | None = None,
    identity: bool = True,
    target_megapixels: float = 1.0,
) -> tuple[list[Scorer], RuntimeError | None]:
    """Health + adherence + identity stack shared by CLI and web UI.

    Returns the scorers and the error that stopped the identity stack, if any.
    Scoring still works without identity; the caller decides whether to abort
    (CLI does, the web UI just logs it).
    """

    def note(message: str) -> None:
        if log is not None:
            log(message)

    scorers: list[Scorer] = [HealthScorer(target_megapixels=target_megapixels)]
    if judge is not None:
        attributes = {s.id: s.attributes for s in project.scenarios.values()}
        scorers.append(AdherenceScorer(judge, attributes))

    if not identity:
        return scorers, None

    try:
        embedder = ArcFaceEmbedder()
    except RuntimeError as exc:
        return scorers, exc

    # objective multi-face detection comes for free with the detector
    scorers.append(FaceCountScorer(embedder))

    directories = {
        subject.id: project.path(subject.ref_dir)
        for subject in project.subjects.values()
        if subject.ref_dir and project.path(subject.ref_dir).is_dir()
    }
    if not directories:
        note(
            "identity skipped: needs a subject with a reference folder;"
            " face counting still runs"
        )
        return scorers, None

    train_dirs = {
        subject.id: project.path(subject.train_dir)
        for subject in project.subjects.values()
        if subject.train_dir and project.path(subject.train_dir).is_dir()
    }
    cache_path = project.data_dir / "ref_embeddings.json"
    note(f"reference embeddings for {len(directories)} subject(s)")
    references = build_references(directories, embedder, cache_path=cache_path)
    training = (
        build_training_embeddings(train_dirs, embedder, cache_path=cache_path)
        if train_dirs
        else {}
    )
    calibration = calibrate_from_references(references)
    note(
        f"calibration slope={calibration.slope:.2f} intercept={calibration.intercept:.2f}"
        f" genuine={calibration.genuine} impostor={calibration.impostor}"
    )
    scorers.append(IdentityScorer(references, calibration, embedder, training=training))
    return scorers, None


def score_run(
    project: ProjectConfig,
    store: Store,
    scorers: Iterable[Scorer],
    *,
    run_id: int | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    on_progress: Callable[[str], None] | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Score stored images; model work happens in threads, DB writes do not.

    ``workers`` > 1 runs the scorers (VLM requests above all) in a thread pool.
    The ONNX face stack stays serialised by a lock inside ``ArcFaceEmbedder``;
    SQLite writes stay in this thread, so no extra locking is needed.
    """

    scorer_list = list(scorers)
    rows = store.images_for_run(run_id)
    counts: dict[str, int] = {}
    seen = 0
    batch = max(workers * 4, 1)

    def score_image(job) -> list:
        row = job["row"]
        path = Path(project.data_dir) / row["path"]
        image = ImageRef(
            id=int(row["id"]),
            path=str(path),
            subject=str(row["subject"]),
            lora_id=str(row["lora_id"]),
            epoch=int(row["epoch"]),
            strength=float(row["strength"]),
            scenario=str(row["scenario"]),
            seed=int(row["seed"]),
            condition=str(row["condition"]),
        )
        out = []
        for scorer in job["active"]:
            out.extend(scorer.score(image))
        return out

    for start in range(0, len(rows), batch):
        jobs = []
        for row in rows[start : start + batch]:
            active = [
                scorer
                for scorer in scorer_list
                if not (
                    skip_existing
                    and store.has_axes(
                        int(row["id"]), scorer.name, scorer.version, getattr(scorer, "axes", ())
                    )
                )
            ]
            jobs.append({"row": row, "active": active})

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(score_image, jobs))
        else:
            results = [score_image(job) for job in jobs]

        for job, scores in zip(jobs, results, strict=True):
            for score in scores:
                store.add_score(
                    int(job["row"]["id"]),
                    score.scorer,
                    score.version,
                    score.axis,
                    score.value,
                    score.meta,
                )
                counts[score.axis] = counts.get(score.axis, 0) + 1
            seen += 1
            if on_progress:
                on_progress(f"scored {seen}/{len(rows)}  {job['row']['path']}")
        # one commit per batch: the SQLite write lock is only held briefly
        store.commit()
        if limit is not None and seen >= limit:
            break

    counts["images"] = seen
    return counts
