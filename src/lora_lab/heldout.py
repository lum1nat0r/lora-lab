"""Held-out check: are reference images part of the training set?

Identity scores are only meaningful if the references were not trained on.
Exact duplicates are cheap to detect by content hash; near-duplicates need
embeddings and are reported separately once a scorer is available.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def count_images(path: Path) -> int:
    return len(list_images(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlap(ref_dir: Path, train_dir: Path) -> dict:
    """Exact content duplicates between two folders (recursive)."""

    ref_files = list_images(ref_dir)
    train_files = list_images(train_dir)

    ref_hashes: dict[str, Path] = {}
    for file in ref_files:
        ref_hashes.setdefault(_sha256(file), file)

    duplicates: list[dict[str, str]] = []
    seen_train: set[str] = set()
    for file in train_files:
        digest = _sha256(file)
        if digest in seen_train:
            continue
        seen_train.add(digest)
        if digest in ref_hashes:
            duplicates.append(
                {
                    "ref": str(ref_hashes[digest].relative_to(ref_dir)),
                    "train": str(file.relative_to(train_dir)),
                }
            )

    ref_count = len(ref_files)
    return {
        "ref_dir": str(ref_dir),
        "train_dir": str(train_dir),
        "ref_images": ref_count,
        "train_images": len(train_files),
        "duplicates": duplicates,
        "duplicate_rate": round(len(duplicates) / ref_count, 4) if ref_count else 0.0,
        "clean": not duplicates,
    }


def contains(ref_dir: Path, train_dir: Path) -> bool:
    """True when one folder is inside the other (a structural leak)."""

    try:
        ref = Path(ref_dir).resolve()
        train = Path(train_dir).resolve()
    except OSError:
        return False
    return ref == train or ref in train.parents or train in ref.parents
