"""Defects and image health.

Two different things, kept apart:

* **defects** are semantic failures (two faces, extra limb, text, ignored
  clothing).  They come from a detector or a judge and are severity weighted.
* **health** is what can be measured without a model: resolution, aspect and
  sharpness.  It is cheap, deterministic and catches broken renders early.

Critical defects make an image unusable regardless of how good it looks
otherwise, which is why they are tracked separately from the index.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageStat

from .base import ImageRef, Score

# severity 1..3; 3 = unusable
CATALOG: dict[str, int] = {
    "multi_face": 3,
    "extra_limb": 3,
    "extra_hand": 2,
    "hand_artifact": 2,
    "face_artifact": 2,
    "text_watermark": 2,
    "clothing_ignored": 2,
    "anatomy_error": 3,
    "background_leak": 2,
    "multi_image": 3,
    "expression_mismatch": 1,
    "blur_lowres": 1,
    "duplicate_subject": 3,
}

CRITICAL = {"multi_face", "extra_limb", "anatomy_error", "multi_image", "duplicate_subject"}

# What a judge can decide from the image alone.
DEFECT_QUESTIONS: dict[str, str] = {
    "multi_face": "Does the image show more than one face of the main subject?",
    "extra_limb": "Does any person or animal have an extra arm or leg?",
    "extra_hand": "Is there an extra hand that belongs to no arm?",
    "hand_artifact": "Are there malformed hands (wrong finger count, fused or melted fingers)?",
    "face_artifact": "Is a face distorted, melted or asymmetric in an unnatural way?",
    "text_watermark": "Is there visible text, a watermark, a signature or a logo?",
    "anatomy_error": "Is there a clear anatomy error (twisted body, impossible joints)?",
    "multi_image": "Is the image split into several panels or a collage?",
    "duplicate_subject": "Does the main subject appear twice or more?",
    "blur_lowres": "Is the image noticeably blurry, low resolution or out of focus?",
}

# Only decidable against the scenario prompt; asked when one is known.
PROMPT_DEFECT_QUESTIONS: dict[str, str] = {
    "clothing_ignored": (
        "Does the clothing contradict the prompt (wrong garment, wrong colour or missing)?"
        " Answer null if the prompt names no clothing."
    ),
    "expression_mismatch": (
        "Does the facial expression clearly differ from the one the prompt asks for?"
        " Answer null if the prompt names no expression."
    ),
    "background_leak": (
        "Does the background or setting contradict the prompt, for example a"
        " training-photo backdrop instead of the requested scene?"
    ),
}


def defect_index(detected: dict[str, int] | None) -> float:
    """Severity-weighted defect index in [0, 1].

    ``detected`` maps a catalogue key to its observed severity (1..3).
    Unknown keys are ignored; the maximum possible is every category at
    severity 3.
    """

    if not detected:
        return 0.0
    total = 0.0
    for key, severity in detected.items():
        if key not in CATALOG:
            continue
        total += min(max(float(severity), 0.0), 3.0)
    maximum = 3.0 * len(CATALOG)
    return min(total / maximum, 1.0)


def has_critical(detected: dict[str, int] | None) -> bool:
    return bool(detected) and any(key in CRITICAL for key in detected)


def health(path: str, *, target_megapixels: float = 1.0, target_sharpness: float = 6.0) -> dict:
    """Resolution and sharpness, both normalised to [0, 1]."""

    with Image.open(path) as im:
        width, height = im.size
        gray = im.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = float(ImageStat.Stat(edges).stddev[0])

    megapixels = width * height / 1_000_000
    resolution_factor = min(megapixels / target_megapixels, 1.0) if target_megapixels else 1.0
    sharpness_factor = min(sharpness / target_sharpness, 1.0) if target_sharpness else 1.0
    return {
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 3),
        "aspect": round(width / height, 4) if height else 0.0,
        "sharpness": round(sharpness, 3),
        "resolution_factor": round(resolution_factor, 4),
        "sharpness_factor": round(sharpness_factor, 4),
        "quality": round(resolution_factor * sharpness_factor, 4),
    }


class HealthScorer:
    """Model-free quality proxy: enough to catch OOM-truncated and mushy renders."""

    name = "health"
    version = "v1"
    axes = ("quality",)

    def __init__(
        self, *, target_megapixels: float = 1.0, target_sharpness: float = 6.0
    ) -> None:
        self.target_megapixels = target_megapixels
        self.target_sharpness = target_sharpness

    def score(self, image: ImageRef) -> list[Score]:
        try:
            data = health(
                image.path,
                target_megapixels=self.target_megapixels,
                target_sharpness=self.target_sharpness,
            )
        except (OSError, ValueError) as exc:
            return [
                Score(
                    axis="quality",
                    value=None,
                    scorer=self.name,
                    version=self.version,
                    meta={"reason": f"unreadable: {exc}"},
                )
            ]
        return [
            Score(
                axis="quality",
                value=data["quality"],
                scorer=self.name,
                version=self.version,
                meta=data,
            )
        ]


def pairwise_diversity(embeddings: list[list[float]]) -> float | None:
    """Mean 1 - cosine over all pairs. Low values indicate mode collapse."""

    from .identity import cosine

    if len(embeddings) < 2:
        return None
    total = 0.0
    pairs = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            total += 1.0 - cosine(embeddings[i], embeddings[j])
            pairs += 1
    return total / pairs


def leakage(embedding: list[float], training_embeddings: list[list[float]]) -> float | None:
    """Max similarity to the training set. High values indicate memorisation."""

    from .identity import cosine

    if not training_embeddings:
        return None
    return max(cosine(embedding, other) for other in training_embeddings)


__all__ = [
    "CATALOG",
    "CRITICAL",
    "DEFECT_QUESTIONS",
    "PROMPT_DEFECT_QUESTIONS",
    "DefectScorer",
    "HealthScorer",
    "defect_index",
    "has_critical",
    "health",
    "leakage",
    "pairwise_diversity",
]



class FaceCountScorer:
    """Objective multi-face detection, no judge needed.

    More than one detected face is a critical defect for a single-subject
    scenario; the count is also recorded in the metadata.
    """

    name = "face_count"
    version = "v1"
    axes = ("critical",)

    def __init__(self, detector) -> None:
        self.detector = detector

    def score(self, image: ImageRef) -> list[Score]:
        try:
            faces = self.detector.detect(image.path)
        except Exception as exc:  # noqa: BLE001 - one unreadable image must not stop a run
            return [
                Score(
                    axis="critical",
                    value=None,
                    scorer=self.name,
                    version=self.version,
                    meta={"reason": f"detect failed: {exc}"},
                )
            ]
        count = len(faces)
        detected = {"multi_face": 3} if count > 1 else {}
        return [
            Score(
                axis="critical",
                value=1.0 if has_critical(detected) else 0.0,
                scorer=self.name,
                version=self.version,
                meta={"faces": count, "defect_index": round(defect_index(detected), 4)},
            )
        ]


@dataclass(frozen=True)
class DefectQuestion:
    id: str
    question: str
    weight: float = 1.0


class DefectScorer:
    """Semantic defects from the same judge that answers the attributes.

    Each catalogue key is a yes/no question where *yes* means the defect is
    present. The axis value is the severity-weighted :func:`defect_index`;
    ``meta`` keeps which keys were detected, judged absent or left open, so
    the report can derive critical failures and the gold set can check the
    judge per key.

    With ``prompts_by_scenario`` the scenario prompt goes into the request and
    the prompt-dependent keys are asked as well.
    """

    name = "defects"
    version = "v1"
    axes = ("defects",)

    def __init__(self, judge, prompts_by_scenario: dict[str, str] | None = None) -> None:
        self.judge = judge
        self.prompts_by_scenario = prompts_by_scenario or {}
        self.questions = [DefectQuestion(key, text) for key, text in DEFECT_QUESTIONS.items()]
        self.prompt_questions = [
            DefectQuestion(key, text) for key, text in PROMPT_DEFECT_QUESTIONS.items()
        ]

    def score(self, image: ImageRef) -> list[Score]:
        prompt = self.prompts_by_scenario.get(image.scenario)
        questions = self.questions + self.prompt_questions if prompt else self.questions
        verdicts = self.judge.judge(image, questions, prompt=prompt)
        detected = [q.id for q in questions if verdicts.get(q.id) is True]
        absent = [q.id for q in questions if verdicts.get(q.id) is False]
        missing = [q.id for q in questions if verdicts.get(q.id) is None]
        judged = bool(detected or absent)
        return [
            Score(
                axis="defects",
                value=defect_index({key: CATALOG[key] for key in detected}) if judged else None,
                scorer=self.name,
                version=self.version,
                meta={
                    "detected": detected,
                    "absent": absent,
                    "missing": missing,
                    "judge": getattr(self.judge, "name", "?"),
                },
            )
        ]
