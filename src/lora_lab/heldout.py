"""Held-out check: are reference images part of the training set?

Identity scores are only meaningful if the references were not trained on.
Exact duplicates are cheap to detect by content hash. Near-duplicates (the
same photo re-encoded, resized or cropped) need face embeddings; they are
found with the identity embedder when the score extras are installed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
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


# ArcFace: two photos of one person sit around 0.5-0.8; the same photo after
# resizing, re-encoding or a light crop stays above ~0.9
NEAR_DUPLICATE = 0.92


def near_duplicates(
    ref_dir: Path,
    train_dir: Path,
    embedder,
    *,
    threshold: float = NEAR_DUPLICATE,
    cache_path: Path | None = None,
    exact: Iterable[dict] = (),
) -> list[dict]:
    """Reference/training pairs whose face embeddings are nearly identical.

    Only images with a detectable face take part. Pass the ``duplicates`` of
    :func:`overlap` as ``exact`` to leave those pairs out.
    """

    from .scorers.identity import (
        _load_embedding_cache,
        _save_embedding_cache,
        cosine,
        embed_files,
    )

    cache = _load_embedding_cache(cache_path)
    refs, dirty_refs = embed_files(list_images(ref_dir), embedder, cache)
    train, dirty_train = embed_files(list_images(train_dir), embedder, cache)
    if dirty_refs or dirty_train:
        _save_embedding_cache(cache_path, cache)

    known = {(d["ref"], d["train"]) for d in exact}
    found = []
    for ref_file, ref_vec in refs:
        best_file, best = None, -1.0
        for train_file, train_vec in train:
            similarity = cosine(ref_vec, train_vec)
            if similarity > best:
                best_file, best = train_file, similarity
        if best_file is None or best < threshold:
            continue
        pair = (str(ref_file.relative_to(ref_dir)), str(best_file.relative_to(train_dir)))
        if pair not in known:
            found.append({"ref": pair[0], "train": pair[1], "similarity": round(best, 4)})
    return found


def contains(ref_dir: Path, train_dir: Path) -> bool:
    """True when one folder is inside the other (a structural leak)."""

    try:
        ref = Path(ref_dir).resolve()
        train = Path(train_dir).resolve()
    except OSError:
        return False
    return ref == train or ref in train.parents or train in ref.parents
