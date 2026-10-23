"""Subject fidelity from face embeddings, with explicit calibration.

Raw cosine similarity is not a score: what counts as "the same subject"
depends on the embedding model and on the population.  We therefore fit a
one-dimensional logistic calibration on genuine and impostor pairs and report
P(same subject).  That makes values comparable across subjects and runs.

The embedder is pluggable.  ``ArcFaceEmbedder`` uses InsightFace when the
optional ``score`` extras are installed; tests and offline runs can pass any
object implementing ``embed(path) -> list[float] | None``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .base import ImageRef, Score


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("empty vector list")
    dim = len(vectors[0])
    totals = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("dimension mismatch")
        for i, value in enumerate(vec):
            totals[i] += value
    return [value / len(vectors) for value in totals]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Calibration:
    """Logistic map from cosine similarity to P(same subject)."""

    slope: float = 12.0
    intercept: float = -5.0
    genuine: int = 0
    impostor: int = 0

    def probability(self, similarity: float) -> float:
        return _sigmoid(self.slope * similarity + self.intercept)

    @classmethod
    def fit(
        cls,
        genuine_scores: list[float],
        impostor_scores: list[float],
        *,
        iterations: int = 3000,
        learning_rate: float = 0.5,
        l2: float = 1e-4,
    ) -> Calibration:
        if not genuine_scores or not impostor_scores:
            raise ValueError("need genuine and impostor scores")
        samples = [(s, 1.0) for s in genuine_scores] + [(s, 0.0) for s in impostor_scores]
        n = len(samples)
        slope, intercept = 12.0, -5.0
        for _ in range(iterations):
            grad_slope = 0.0
            grad_intercept = 0.0
            for x, y in samples:
                p = _sigmoid(slope * x + intercept)
                error = p - y
                grad_slope += error * x
                grad_intercept += error
            slope -= learning_rate * (grad_slope / n + l2 * slope)
            intercept -= learning_rate * (grad_intercept / n)
        return cls(
            slope=slope,
            intercept=intercept,
            genuine=len(genuine_scores),
            impostor=len(impostor_scores),
        )

    def to_dict(self) -> dict:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "genuine": self.genuine,
            "impostor": self.impostor,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: data[k] for k in ("slope", "intercept", "genuine", "impostor")})


@dataclass
class SubjectReferences:
    """Reference embeddings per subject, plus the impostor pool."""

    embeddings: dict[str, list[list[float]]] = field(default_factory=dict)

    def add(self, subject: str, embedding: list[float]) -> None:
        self.embeddings.setdefault(subject, []).append(embedding)

    def centroids(self) -> dict[str, list[float]]:
        return {name: centroid(vecs) for name, vecs in self.embeddings.items() if vecs}

    def genuine_scores(self) -> list[float]:
        """Best-match score of each reference against its own subject centroid."""

        out: list[float] = []
        for vectors in self.embeddings.values():
            if len(vectors) < 2:
                continue
            center = centroid(vectors)
            out.extend(cosine(vec, center) for vec in vectors)
        return out

    def impostor_scores(self) -> list[float]:
        """Every reference against every *other* subject's centroid."""

        centers = self.centroids()
        out: list[float] = []
        for name, vectors in self.embeddings.items():
            for other, center in centers.items():
                if other == name:
                    continue
                out.extend(cosine(vec, center) for vec in vectors)
        return out


class IdentityScorer:
    name = "identity"
    version = "v1"

    def __init__(
        self,
        references: SubjectReferences,
        calibration: Calibration,
        embedder,
    ) -> None:
        self.references = references
        self.calibration = calibration
        self.embedder = embedder
        self._centroids = references.centroids()

    def score(self, image: ImageRef) -> list[Score]:
        center = self._centroids.get(image.subject)
        if center is None:
            return [
                Score(
                    axis="identity",
                    value=None,
                    scorer=self.name,
                    version=self.version,
                    meta={"reason": "no_reference_for_subject"},
                )
            ]
        embedding = self.embedder.embed(image.path)
        if embedding is None:
            return [
                Score(
                    axis="identity",
                    value=None,
                    scorer=self.name,
                    version=self.version,
                    meta={"reason": "no_face_detected"},
                )
            ]
        similarity = cosine(embedding, center)
        return [
            Score(
                axis="identity",
                value=self.calibration.probability(similarity),
                scorer=self.name,
                version=self.version,
                meta={"cosine": round(similarity, 4)},
            )
        ]


class ArcFaceEmbedder:
    """InsightFace ArcFace embeddings. Imported lazily to keep the core light."""

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640) -> None:
        try:
            import cv2  # noqa: F401
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "identity scoring needs the optional extras: pip install 'lora-lab[score]'"
            ) from exc

        self._app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def embed(self, path: str) -> list[float] | None:  # pragma: no cover - needs models
        import cv2

        image = cv2.imread(path)
        if image is None:
            return None
        faces = self._app.get(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return [float(x) for x in face.normed_embedding]


def build_references(
    subject_dirs: dict[str, Path], embedder, *, limit: int = 40
) -> SubjectReferences:
    """Embed reference images per subject."""

    refs = SubjectReferences()
    for subject, directory in subject_dirs.items():
        files = sorted(
            p
            for p in Path(directory).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )[:limit]
        for file in files:
            embedding = embedder.embed(str(file))
            if embedding is not None:
                refs.add(subject, embedding)
    return refs


def calibrate_from_references(references: SubjectReferences) -> Calibration:
    return Calibration.fit(references.genuine_scores(), references.impostor_scores())
