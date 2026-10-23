"""Read and write the uncommitted config files behind the setup UI.

Everything the UI touches lives in ``config/local/``. The committed examples
are never modified, so a bad edit can always be undone by deleting the local
file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..config import Attribute, GridConfig, LoraEntry, Scenario, Subject

SCENARIO_HEADER = "# Edited in the lora-lab UI. Not committed.\n"
SUBJECT_HEADER = "# Edited in the lora-lab setup UI. Not committed.\n"
LORA_HEADER = "# Curated in the lora-lab UI. Not committed.\n"
GRID_HEADER = "# Edited in the lora-lab UI. Not committed.\n"

SLUG = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


class ConfigError(ValueError):
    pass


def local_dir(root: Path) -> Path:
    path = Path(root) / "config" / "local"
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_path(root: Path, name: str) -> Path:
    return local_dir(root) / name


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write(path: Path, payload: dict[str, Any], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = header + yaml.safe_dump(
        payload, sort_keys=False, allow_unicode=True, width=1000
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# -- scenarios --------------------------------------------------------------


def save_scenarios(root: Path, scenarios: list[Scenario]) -> None:
    seen: set[str] = set()
    for scenario in scenarios:
        if not SLUG.match(scenario.id):
            raise ConfigError(
                f"scenario id {scenario.id!r}: use letters, digits, dot, dash, underscore"
            )
        if scenario.id in seen:
            raise ConfigError(f"duplicate scenario id: {scenario.id}")
        if "{trigger}" not in scenario.prompt:
            raise ConfigError(
                f"scenario {scenario.id}: prompt must contain {{trigger}}"
            )
        if not scenario.attributes:
            raise ConfigError(f"scenario {scenario.id}: add at least one attribute")
        for attribute in scenario.attributes:
            if not attribute.id or not attribute.question:
                raise ConfigError(
                    f"scenario {scenario.id}: every attribute needs an id and a question"
                )
        seen.add(scenario.id)

    _write(
        local_path(root, "scenarios.yaml"),
        {"scenarios": [scenario.model_dump() for scenario in scenarios]},
        SCENARIO_HEADER,
    )


def parse_attributes(ids: list[str], questions: list[str], weights: list[str]) -> list[Attribute]:
    attributes = []
    for index, raw_id in enumerate(ids):
        attr_id = (raw_id or "").strip()
        if not attr_id:
            continue
        question = (questions[index] if index < len(questions) else "").strip()
        raw_weight = (weights[index] if index < len(weights) else "") or "1"
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ConfigError(f"attribute {attr_id}: weight is not a number") from exc
        attributes.append(Attribute(id=attr_id, question=question, weight=weight))
    return attributes


# -- subjects ---------------------------------------------------------------


def save_subjects(root: Path, subjects: list[Subject]) -> None:
    seen: set[str] = set()
    for subject in subjects:
        if not SLUG.match(subject.id):
            raise ConfigError(f"subject id {subject.id!r}: invalid characters")
        if subject.id in seen:
            raise ConfigError(f"duplicate subject id: {subject.id}")
        if not subject.trigger.strip():
            raise ConfigError(f"subject {subject.id}: trigger is empty")
        seen.add(subject.id)
    _write(
        local_path(root, "subjects.yaml"),
        {"subjects": [subject.model_dump(exclude_none=True) for subject in subjects]},
        SUBJECT_HEADER,
    )


# -- loras ------------------------------------------------------------------


def save_loras(root: Path, loras: list[LoraEntry]) -> None:
    seen: set[str] = set()
    for lora in loras:
        if not SLUG.match(lora.id):
            raise ConfigError(f"LoRA id {lora.id!r}: invalid characters")
        if lora.id in seen:
            raise ConfigError(f"duplicate LoRA id: {lora.id}")
        if not lora.subject:
            raise ConfigError(f"LoRA {lora.id}: assign a subject")
        if not lora.file:
            raise ConfigError(f"LoRA {lora.id}: file is empty")
        seen.add(lora.id)
    if not loras:
        raise ConfigError("select at least one LoRA")
    _write(
        local_path(root, "loras.yaml"),
        {"loras": [lora.model_dump(exclude_none=True) for lora in loras]},
        LORA_HEADER,
    )


def save_all_loras(root: Path, items: list[dict[str, Any]]) -> None:
    _write(local_path(root, "loras.all.yaml"), {"loras": items}, LORA_HEADER)


def load_all_loras(root: Path) -> list[dict[str, Any]]:
    data = _read(local_path(root, "loras.all.yaml"))
    items = data.get("loras") or []
    return [item for item in items if isinstance(item, dict)]


# -- grid -------------------------------------------------------------------


def save_grid(root: Path, grid: GridConfig) -> None:
    if not grid.strengths:
        raise ConfigError("at least one strength is required")
    if not grid.seeds:
        raise ConfigError("at least one seed is required")
    payload = grid.model_dump()
    if local_path(root, "loras.yaml").exists():
        payload["loras"] = []
        payload["loras_file"] = "config/local/loras.yaml"
    _write(local_path(root, "grid.yaml"), payload, GRID_HEADER)


def parse_float_list(text: str, field: str) -> list[float]:
    values: list[float] = []
    for chunk in re.split(r"[,;\s]+", text.strip()):
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise ConfigError(f"{field}: {chunk!r} is not a number") from exc
    return values


def parse_int_list(text: str, field: str) -> list[int]:
    values: list[int] = []
    for chunk in re.split(r"[,;\s]+", text.strip()):
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError as exc:
            raise ConfigError(f"{field}: {chunk!r} is not a whole number") from exc
    return values


# -- references -------------------------------------------------------------


def default_ref_dir(root: Path, subject_id: str) -> str:
    datasets = Path("/datasets")
    base = datasets / "_refs" if datasets.is_dir() else Path(root) / "datasets" / "_refs"
    return str(base / subject_id)


def resolve_ref_dir(root: Path, value: str | None, subject_id: str) -> Path:
    if value and value.strip():
        path = Path(value.strip())
    else:
        path = Path(default_ref_dir(root, subject_id))
    return path if path.is_absolute() else (Path(root) / path).resolve()


def list_ref_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    suffix = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in suffix)


def safe_child(directory: Path, filename: str) -> Path:
    target = (directory / filename).resolve()
    if directory.resolve() not in target.parents:
        raise ConfigError("path escapes the reference folder")
    return target
