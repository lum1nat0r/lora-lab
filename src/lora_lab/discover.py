"""Discover subjects and LoRA files on a mounted volume.

Point this at the container paths (``/datasets``, ``/models/loras``) and it
prints config snippets in the shape ``config/local/`` expects.  With ``--apply``
it writes them, so nobody has to copy dozens of file names by hand.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .config import Subject
from .heldout import list_images
from .importers.fizgig import epoch_from_filename, read_metadata

REF_NAMES = ("refs", "ref", "reference", "references", "heldout", "held-out", "held_out")
TRAIN_NAMES = ("train", "training", "dataset", "images", "raw", "src")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _match_subject(candidates: list[str], known: set[str]) -> str | None:
    lookup = {_normalise(item): item for item in known}
    for candidate in candidates:
        if not candidate:
            continue
        direct = lookup.get(_normalise(candidate))
        if direct:
            return direct
        normalised = _normalise(candidate)
        for key, original in lookup.items():
            if key and (key in normalised or normalised in key):
                return original
    return None


def slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned.strip("_") or "subject"


def _find_child(parent: Path, names: tuple[str, ...]) -> Path | None:
    lowered = {name.lower() for name in names}
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.lower() in lowered:
            return child
    return None


def discover_subjects(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue

        refs = _find_child(child, REF_NAMES)
        train = _find_child(child, TRAIN_NAMES)
        if train is None and refs is None and list_images(child):
            train = child

        trigger = child.name
        trigger_file = child / "trigger.txt"
        if trigger_file.is_file():
            trigger = trigger_file.read_text(encoding="utf-8").strip() or child.name

        out.append(
            {
                "id": slug(child.name),
                "trigger": trigger,
                "ref_dir": str(refs) if refs else None,
                "train_dir": str(train) if train else None,
                "ref_count": len(list_images(refs)) if refs else 0,
                "train_count": len(list_images(train)) if train else 0,
            }
        )
    return out


def discover_loras(lora_root: Path, *, subjects: list[str] | None = None) -> list[dict[str, Any]]:
    """Walk a ComfyUI ``models/loras`` tree and describe every LoRA file."""

    lora_root = Path(lora_root)
    if not lora_root.is_dir():
        raise NotADirectoryError(lora_root)

    known = set(subjects or [])
    out: list[dict[str, Any]] = []
    for file in sorted(lora_root.rglob("*.safetensors")):
        if file.name.startswith("."):
            continue
        epoch = epoch_from_filename(file)
        try:
            meta = read_metadata(file)
        except Exception:
            meta = {}
        trigger = meta.get("ss_trigger_word") or ""
        candidates = [trigger, file.parent.name, *reversed(file.parent.parts[-3:])]
        matched = _match_subject(candidates, known)
        if matched:
            subject = matched
        else:
            subject = slug(trigger) if trigger else slug(file.parent.name)

        stem = re.sub(r"-\d{6}(-state)?$", "", file.stem)
        lora_id = f"{slug(stem)}_ep{epoch}" if epoch else slug(stem)

        out.append(
            {
                "id": lora_id,
                "subject": subject,
                "file": file.relative_to(lora_root).as_posix(),
                "epoch": epoch or 0,
                "rank": _int(meta.get("ss_network_dim")),
                "learning_rate": _float(meta.get("ss_learning_rate") or meta.get("ss_unet_lr")),
                "trigger": trigger or None,
                "size_mb": round(file.stat().st_size / 1024 / 1024, 1),
            }
        )
    return out


def _int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def subjects_yaml(subjects: list[dict[str, Any]]) -> str:
    entries = []
    for item in subjects:
        entry = {
            "id": item["id"],
            "trigger": item["trigger"],
            "ref_dir": item["ref_dir"],
            "train_dir": item["train_dir"],
        }
        entries.append(Subject(**entry).model_dump(exclude_none=True))
    return yaml.safe_dump({"subjects": entries}, sort_keys=False, allow_unicode=True)


def loras_yaml(loras: list[dict[str, Any]]) -> str:
    entries = [
        {
            "id": item["id"],
            "subject": item["subject"],
            "file": item["file"],
            "epoch": item["epoch"],
        }
        for item in loras
    ]
    return yaml.safe_dump({"loras": entries}, sort_keys=False, allow_unicode=True)


GRID_HEADER = "# generated by `lora-lab discover --apply`\n"


def grid_source(root: Path, defaults: Path | None = None) -> Path | None:
    candidates = [
        root / "config" / "local" / "grid.yaml",
        root / "config" / "grid.yaml",
    ]
    if defaults is not None:
        candidates.append(defaults / "grid.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def apply_grid(path: Path, source: Path) -> Path:
    """Create a local grid from an existing one, pointing it at the LoRA list."""

    raw = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    raw["loras"] = []
    raw["loras_file"] = "config/local/loras.yaml"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        GRID_HEADER + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def apply_subjects(path: Path, subjects: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(subjects_yaml(subjects), encoding="utf-8")


def apply_loras(path: Path, loras: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(loras_yaml(loras), encoding="utf-8")
