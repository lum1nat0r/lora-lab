"""Basic dataset audit: counts, resolution buckets, captions, duplicates.

Face/body composition needs the optional scoring extras (InsightFace) and is
deliberately left to a later pass; this module answers the questions that need
no model.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CAPTION_EXTENSION = ".txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    resolutions: Counter[tuple[int, int]] = Counter()
    aspect_buckets: Counter[str] = Counter()
    sizes: list[int] = []
    hashes: dict[str, list[str]] = {}
    missing_captions: list[str] = []
    caption_lengths: list[int] = []
    unreadable: list[str] = []

    for file in files:
        sizes.append(file.stat().st_size)
        digest = _sha256(file)
        hashes.setdefault(digest, []).append(file.name)

        try:
            with Image.open(file) as im:
                width, height = im.size
        except Exception:
            unreadable.append(file.name)
            continue

        resolutions[(width, height)] += 1
        aspect_buckets[_aspect(width, height)] += 1

        caption = file.with_suffix(CAPTION_EXTENSION)
        if caption.exists():
            caption_lengths.append(len(caption.read_text(encoding="utf-8", errors="ignore").split()))
        else:
            missing_captions.append(file.name)

    duplicates = {h: names for h, names in hashes.items() if len(names) > 1}
    return {
        "path": str(path),
        "images": len(files),
        "unreadable": unreadable,
        "duplicates": list(duplicates.values()),
        "resolution_top": [
            {"width": w, "height": h, "count": n}
            for (w, h), n in resolutions.most_common(10)
        ],
        "aspect_distribution": dict(aspect_buckets),
        "megapixels": _megapixel_stats(resolutions),
        "missing_captions": missing_captions,
        "caption_word_stats": _word_stats(caption_lengths),
        "total_megabytes": round(sum(sizes) / (1024 * 1024), 1),
    }


def _aspect(width: int, height: int) -> str:
    ratio = width / height
    if ratio > 1.15:
        return "landscape"
    if ratio < 0.87:
        return "portrait"
    return "square"


def _megapixel_stats(resolutions: Counter[tuple[int, int]]) -> dict[str, float]:
    total = sum(resolutions.values())
    if not total:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    area = sum(w * h * n for (w, h), n in resolutions.items()) / total
    return {
        "mean": round(area / 1_000_000, 2),
        "min": round(min(w * h for w, h in resolutions) / 1_000_000, 2),
        "max": round(max(w * h for w, h in resolutions) / 1_000_000, 2),
    }


def _word_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(lengths),
        "mean": round(sum(lengths) / len(lengths), 1),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }
