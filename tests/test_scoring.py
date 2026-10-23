import json
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from lora_lab.scorers.adherence import JsonVerdicts, weighted_adherence
from lora_lab.scorers.base import ImageRef
from lora_lab.scorers.defects import (
    HealthScorer,
    defect_index,
    has_critical,
    health,
    leakage,
    pairwise_diversity,
)
from lora_lab.scorers.identity import (
    Calibration,
    IdentityScorer,
    SubjectReferences,
    centroid,
    cosine,
)


def image_ref(path: str | Path, subject: str = "a", scenario: str = "s") -> ImageRef:
    return ImageRef(
        id=1,
        path=str(path),
        subject=subject,
        lora_id="lora",
        epoch=1,
        strength=1.0,
        scenario=scenario,
        seed=1,
    )


class FakeAttribute:
    def __init__(self, id: str, weight: float = 1.0) -> None:
        self.id = id
        self.weight = weight


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float] | None]) -> None:
        self.mapping = mapping

    def embed(self, path: str) -> list[float] | None:
        return self.mapping.get(path)


# -- identity ---------------------------------------------------------------


def test_cosine_and_centroid() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert centroid([[1, 0], [0, 1]]) == pytest.approx([0.5, 0.5])


def test_calibration_separates_genuine_and_impostor() -> None:
    genuine = [0.70, 0.75, 0.80, 0.85, 0.90]
    impostor = [0.10, 0.15, 0.20, 0.25, 0.30]
    calibration = Calibration.fit(genuine, impostor)
    assert calibration.probability(0.85) > 0.8
    assert calibration.probability(0.15) < 0.2
    assert calibration.probability(0.85) > calibration.probability(0.55)


def test_reference_scores() -> None:
    refs = SubjectReferences()
    refs.add("a", [1.0, 0.0])
    refs.add("a", [0.9, 0.1])
    refs.add("b", [0.0, 1.0])
    refs.add("b", [0.1, 0.9])
    genuine = refs.genuine_scores()
    impostor = refs.impostor_scores()
    assert genuine and impostor
    assert min(genuine) > max(impostor)
    calibration = Calibration.fit(genuine, impostor)
    assert calibration.probability(min(genuine)) > calibration.probability(max(impostor))


def test_identity_scorer_handles_missing_face() -> None:
    refs = SubjectReferences()
    refs.add("a", [1.0, 0.0])
    refs.add("b", [0.0, 1.0])
    calibration = Calibration(slope=10, intercept=-5)
    scorer = IdentityScorer(refs, calibration, FakeEmbedder({"x.png": None}))
    score = scorer.score(image_ref("x.png", subject="a"))[0]
    assert score.value is None
    assert score.meta["reason"] == "no_face_detected"


def test_identity_scorer_reports_probability() -> None:
    refs = SubjectReferences()
    refs.add("a", [1.0, 0.0])
    refs.add("b", [0.0, 1.0])
    scorer = IdentityScorer(
        refs,
        Calibration(slope=10, intercept=-5),
        FakeEmbedder({"x.png": [1.0, 0.0]}),
    )
    score = scorer.score(image_ref("x.png", subject="a"))[0]
    assert score.value is not None
    assert score.value > 0.9
    assert score.meta["cosine"] == pytest.approx(1.0)


# -- adherence --------------------------------------------------------------


def test_weighted_adherence() -> None:
    attributes = [FakeAttribute("a"), FakeAttribute("b", 2.0), FakeAttribute("c")]
    value, meta = weighted_adherence({"a": True, "b": False, "c": True}, attributes)
    assert value == pytest.approx(2.0 / 4.0)
    assert meta["failed"] == ["b"]
    assert meta["judged"] == 3


def test_missing_verdicts_are_not_failures() -> None:
    attributes = [FakeAttribute("a"), FakeAttribute("b")]
    value, meta = weighted_adherence({"a": True}, attributes)
    assert value == pytest.approx(1.0)
    assert meta["missing"] == ["b"]


def test_no_verdicts_returns_none() -> None:
    value, _ = weighted_adherence({}, [FakeAttribute("a")])
    assert value is None


def test_json_verdicts_by_basename(tmp_path: Path) -> None:
    path = tmp_path / "v.json"
    path.write_text(
        json.dumps({"img.png": {"a": True, "b": False}}), encoding="utf-8"
    )
    judge = JsonVerdicts(path)
    verdicts = judge.judge(image_ref("/deep/dir/img.png"), [FakeAttribute("a"), FakeAttribute("b")])
    assert verdicts == {"a": True, "b": False}


# -- defects ----------------------------------------------------------------


def test_defect_index_and_critical() -> None:
    assert defect_index(None) == 0.0
    assert defect_index({"multi_face": 3}) == pytest.approx(3 / (3 * 13))
    assert defect_index({"unknown": 3}) == 0.0
    assert has_critical({"multi_face": 1})
    assert not has_critical({"hand_artifact": 1})


def _noise(size=(256, 256)) -> Image.Image:
    return Image.effect_noise(size, 110).convert("RGB")


def test_health_prefers_sharp_images(tmp_path: Path) -> None:
    sharp = tmp_path / "sharp.png"
    soft = tmp_path / "soft.png"
    _noise().save(sharp)
    _noise().filter(ImageFilter.GaussianBlur(6)).save(soft)

    sharp_stats = health(str(sharp))
    soft_stats = health(str(soft))
    assert sharp_stats["sharpness"] > soft_stats["sharpness"]
    assert sharp_stats["megapixels"] == pytest.approx(0.0655, abs=1e-3)
    assert HealthScorer().score(image_ref(sharp))[0].value is not None


def test_health_reports_unreadable(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    score = HealthScorer().score(image_ref(broken))[0]
    assert score.value is None
    assert score.meta["reason"].startswith("unreadable")


def test_diversity_and_leakage() -> None:
    assert pairwise_diversity([[1, 0], [0, 1]]) == pytest.approx(1.0)
    assert pairwise_diversity([[1, 0], [1, 0]]) == pytest.approx(0.0)
    assert leakage([1, 0], [[1, 0], [0, 1]]) == pytest.approx(1.0)
    assert leakage([1, 0], []) is None
