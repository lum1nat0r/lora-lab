"""Apply scorers to stored images."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from ..config import ProjectConfig
from ..db import Store
from .base import ImageRef, Scorer


def score_run(
    project: ProjectConfig,
    store: Store,
    scorers: Iterable[Scorer],
    *,
    run_id: int | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    scorer_list = list(scorers)
    rows = store.images_for_run(run_id)
    counts: dict[str, int] = {}
    seen = 0

    for row in rows:
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
        for scorer in scorer_list:
            if skip_existing and store.has_axes(
                image.id, scorer.name, scorer.version, getattr(scorer, "axes", ())
            ):
                continue
            for score in scorer.score(image):
                store.add_score(
                    image.id,
                    score.scorer,
                    score.version,
                    score.axis,
                    score.value,
                    score.meta,
                )
                counts[score.axis] = counts.get(score.axis, 0) + 1
        seen += 1
        if on_progress:
            on_progress(f"scored {seen}/{len(rows)}  {row['path']}")
        if limit is not None and seen >= limit:
            break

    counts["images"] = seen
    return counts
