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
    axes = ("identity",)

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


ARCFACE_DST = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
]

DEFAULT_ARCFACE = "/opt/face/w600k_r50.onnx"
DEFAULT_YUNET = "/opt/face/yunet.onnx"


def l2_normalize(vector) -> list[float] | None:
    values = [float(x) for x in vector]
    norm = math.sqrt(math.fsum(x * x for x in values))
    if norm == 0:
        return None
    return [x / norm for x in values]


def align_face(image, landmarks, size: int = 112):
    """Align a face to the ArcFace template using YuNet's five points."""

    import cv2
    import numpy as np

    source = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    target = np.asarray(ARCFACE_DST, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if matrix is None:
        matrix, _ = cv2.estimateAffine2D(source, target, method=cv2.RANSAC)
    if matrix is None:
        return None
    return cv2.warpAffine(image, matrix, (size, size), borderValue=0.0)


def preprocess(aligned):
    import cv2

    return cv2.dnn.blobFromImage(
        aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )


@dataclass
class Face:
    box: tuple[float, float, float, float]
    landmarks: list[tuple[float, float]]
    score: float

    @property
    def area(self) -> float:
        return max(self.box[2], 0.0) * max(self.box[3], 0.0)


class ArcFaceEmbedder:
    """YuNet face detection plus an ArcFace ONNX embedding model.

    Deliberately no InsightFace dependency: that package is source-only and
    makes the container build fragile. Detection and recognition are two ONNX
    models loaded directly with onnxruntime.
    """

    def __init__(
        self,
        model_path: str | None = None,
        detector_path: str | None = None,
    ) -> None:
        try:
            import cv2
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - optional extras
            raise RuntimeError(
                "identity scoring needs the score extras: pip install 'lora-lab[score]'"
            ) from exc

        import os
        from pathlib import Path

        arcface = Path(model_path or os.environ.get("ARCFACE_MODEL", DEFAULT_ARCFACE))
        yunet = Path(detector_path or os.environ.get("YUNET_MODEL", DEFAULT_YUNET))
        for path, label in ((arcface, "ArcFace"), (yunet, "YuNet")):
            if not path.is_file():
                raise RuntimeError(f"{label} model not found: {path}")

        self._cv2 = cv2
        self._session = onnxruntime.InferenceSession(
            str(arcface), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        self._detector = cv2.FaceDetectorYN.create(str(yunet), "", (640, 640), 0.6, 0.3, 5000)

    def _read(self, path: str):
        image = self._cv2.imread(path)
        if image is None:
            return None, []
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None:
            return image, []
        parsed = []
        for row in faces:
            parsed.append(
                Face(
                    box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                    landmarks=[
                        (float(row[4 + i * 2]), float(row[5 + i * 2])) for i in range(5)
                    ],
                    score=float(row[14]),
                )
            )
        return image, parsed

    def detect(self, path: str) -> list[Face]:
        _image, faces = self._read(path)
        return faces

    def embed(self, path: str) -> list[float] | None:
        image, faces = self._read(path)
        if image is None or not faces:
            return None
        face = max(faces, key=lambda item: item.area)
        aligned = align_face(image, face.landmarks)
        if aligned is None:
            return None
        blob = preprocess(aligned)
        output = self._session.run([self._output], {self._input: blob})[0][0]
        return l2_normalize(output)


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
