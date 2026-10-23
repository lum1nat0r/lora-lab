"""Scoring primitives shared by all scorers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ImageRef:
    id: int
    path: str
    subject: str
    lora_id: str
    epoch: int
    strength: float
    scenario: str
    seed: int
    condition: str = "lora"


@dataclass(frozen=True)
class Score:
    """One axis value for one image."""

    axis: str
    value: float | None
    scorer: str
    version: str = "v1"
    meta: dict = field(default_factory=dict)


class Scorer(Protocol):
    name: str
    version: str
    axes: tuple[str, ...]

    def score(self, image: ImageRef) -> list[Score]: ...


class Embedder(Protocol):
    """Turns an image into a unit-length embedding, or None if not applicable."""

    def embed(self, path: str) -> list[float] | None: ...
