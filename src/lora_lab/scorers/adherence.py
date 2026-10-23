"""Prompt adherence as a weighted attribute checklist.

The judge returns a yes/no per attribute; the score is the weighted share of
satisfied attributes.  Keeping the prompt out of the score is deliberate: a
long prompt with ten clauses cannot be graded as one number, and a failure
should be attributable to a specific attribute.

``JsonVerdicts`` reads verdicts from a JSON file so the scoring pass can run
offline (or from a human review round) without a model in the loop:

    {
      "fast/subject_a/lora/ep20/s0.9/still_life/11/0001.png": {
        "object_present": true,
        "background_plain": false
      }
    }

Keys may be the full relative path, the basename, or the image id as a string.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .base import ImageRef, Score


class Judge(Protocol):
    name: str
    version: str

    def judge(self, image: ImageRef, attributes: Sequence[Any]) -> dict[str, bool | None]: ...


def weighted_adherence(
    verdicts: dict[str, bool | None], attributes: Sequence[Any]
) -> tuple[float | None, dict]:
    """Weighted share of satisfied attributes.

    ``None`` verdicts are treated as missing, not as failures; if nothing could
    be judged the result is ``None`` rather than a misleading zero.
    """

    hit = 0.0
    total = 0.0
    missing: list[str] = []
    failed: list[str] = []
    for attribute in attributes:
        weight = float(getattr(attribute, "weight", 1.0))
        verdict = verdicts.get(attribute.id)
        if verdict is None:
            missing.append(attribute.id)
            continue
        total += weight
        if verdict:
            hit += weight
        else:
            failed.append(attribute.id)
    if total == 0:
        return None, {"missing": missing, "failed": failed, "judged": 0}
    return hit / total, {
        "missing": missing,
        "failed": failed,
        "judged": int(len(attributes) - len(missing)),
    }


class AdherenceScorer:
    name = "adherence"
    version = "v1"
    axes = ("adherence",)

    def __init__(self, judge: Judge, attributes_by_scenario: dict[str, list[Any]]) -> None:
        self.judge = judge
        self.attributes_by_scenario = attributes_by_scenario

    def score(self, image: ImageRef) -> list[Score]:
        attributes = self.attributes_by_scenario.get(image.scenario)
        if not attributes:
            return [
                Score(
                    axis="adherence",
                    value=None,
                    scorer=self.name,
                    version=self.version,
                    meta={"reason": "no_attributes_for_scenario"},
                )
            ]
        verdicts = self.judge.judge(image, attributes)
        value, meta = weighted_adherence(verdicts, attributes)
        meta["judge"] = self.judge.name
        return [
            Score(
                axis="adherence",
                value=value,
                scorer=self.name,
                version=self.version,
                meta=meta,
            )
        ]


@dataclass
class JsonVerdicts:
    """Offline verdict source backed by a JSON file."""

    path: Path
    name: str = "json-verdicts"
    version: str = "v1"
    _data: dict[str, dict[str, bool | None]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self._data = {str(k): dict(v) for k, v in raw.items()}

    def judge(self, image: ImageRef, attributes: Sequence[Any]) -> dict[str, bool | None]:
        candidates = [
            image.path,
            Path(image.path).name,
            str(image.id),
        ]
        for key in candidates:
            if key in self._data:
                row = self._data[key]
                return {attr.id: row.get(attr.id) for attr in attributes}
        return {attr.id: None for attr in attributes}
