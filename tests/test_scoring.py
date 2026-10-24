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


def test_impostor_scores_skip_subjects_of_the_same_person() -> None:
    refs = SubjectReferences()
    for subject in ("a_v1", "a_v2"):  # two datasets, one person
        refs.add(subject, [1.0, 0.0])
        refs.add(subject, [0.9, 0.1])
    refs.add("b", [0.0, 1.0])
    refs.add("b", [0.1, 0.9])
    impostor = refs.impostor_scores()
    assert impostor and max(impostor) < 0.5
    calibration = Calibration.fit(refs.genuine_scores(), impostor)
    assert calibration.probability(0.9) > 0.8


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
        training={"a": [[0.8, 0.6]]},
    )
    score = scorer.score(image_ref("x.png", subject="a"))[0]
    assert score.value is not None
    assert score.value > 0.9
    assert score.meta["cosine"] == pytest.approx(1.0)
    assert score.meta["leakage"] == pytest.approx(0.8)
    assert score.meta["embedding"] == [1.0, 0.0]


def test_single_subject_calibration_fallback_and_embedding_cache(tmp_path: Path) -> None:
    from lora_lab.scorers.identity import build_references, calibrate_from_references

    ref_dir = tmp_path / "refs_a"
    ref_dir.mkdir()
    img = ref_dir / "1.png"
    Image.new("RGB", (16, 16), (50, 60, 70)).save(img)

    calls = 0

    class CountingEmbedder:
        def embed(self, path: str):
            nonlocal calls
            calls += 1
            return [1.0, 0.0]

    cache_file = tmp_path / "cache.json"
    refs1 = build_references({"a": ref_dir}, CountingEmbedder(), cache_path=cache_file)
    assert calls == 1
    assert cache_file.is_file()

    # Second call hits the cache without invoking the embedder
    refs2 = build_references({"a": ref_dir}, CountingEmbedder(), cache_path=cache_file)
    assert calls == 1
    assert refs2.embeddings == refs1.embeddings

    # Single subject falls back to default calibration instead of raising ValueError
    cal = calibrate_from_references(refs2)
    assert cal.slope == pytest.approx(12.0)
    assert cal.intercept == pytest.approx(-5.0)


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


def test_l2_normalize() -> None:
    from lora_lab.scorers.identity import l2_normalize

    vector = l2_normalize([3.0, 4.0])
    assert vector == pytest.approx([0.6, 0.8])
    assert l2_normalize([0.0, 0.0]) is None


def test_arcface_embedder_reports_missing_models(tmp_path: Path) -> None:
    from lora_lab.scorers.identity import ArcFaceEmbedder

    pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    with pytest.raises(RuntimeError, match="model not found"):
        ArcFaceEmbedder(
            model_path=str(tmp_path / "nope.onnx"),
            detector_path=str(tmp_path / "nope2.onnx"),
        )


def test_face_count_scorer_marks_multi_face() -> None:
    from lora_lab.scorers.defects import FaceCountScorer

    class Detector:
        def __init__(self, count: int) -> None:
            self.count = count

        def detect(self, path: str):
            return [object()] * self.count

    one = FaceCountScorer(Detector(1)).score(image_ref("x.png"))[0]
    assert one.value == 0.0
    assert one.meta["faces"] == 1

    two = FaceCountScorer(Detector(2)).score(image_ref("x.png"))[0]
    assert two.value == 1.0
    assert two.meta["faces"] == 2


def test_face_count_scorer_survives_detector_error() -> None:
    from lora_lab.scorers.defects import FaceCountScorer

    class Broken:
        def detect(self, path: str):
            raise OSError("boom")

    score = FaceCountScorer(Broken()).score(image_ref("x.png"))[0]
    assert score.value is None
    assert score.meta["reason"].startswith("detect failed")


def test_align_face_shape() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from lora_lab.scorers.identity import ARCFACE_DST, align_face

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    aligned = align_face(image, ARCFACE_DST)
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)


def test_arcface_pipeline_with_fake_models(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from lora_lab.scorers.identity import ArcFaceEmbedder

    class FakeTensor:
        name = "input"

    class FakeSession:
        def get_inputs(self):
            return [FakeTensor()]

        def get_outputs(self):
            return [FakeTensor()]

        def run(self, outputs, feed):
            blob = feed["input"]
            assert blob.shape == (1, 3, 112, 112)
            return [np.array([[3.0, 4.0]], dtype=np.float32)]

    class FakeDetector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, image):
            row = np.array(
                [[10, 10, 50, 50, 40, 30, 70, 30, 55, 45, 42, 60, 68, 60, 0.95]],
                dtype=np.float32,
            )
            return 1, row

    image = tmp_path / "face.png"
    Image.new("RGB", (100, 100), (120, 90, 80)).save(image)

    embedder = ArcFaceEmbedder(session=FakeSession(), detector=FakeDetector())
    vector = embedder.embed(str(image))
    assert vector == pytest.approx([0.6, 0.8])
    assert len(embedder.detect(str(image))) == 1


def test_build_default_scorers_shared_stack(root: Path) -> None:
    from lora_lab.config import load_project
    from lora_lab.scorers.run import build_default_scorers

    class FakeJudge:
        name = "fake"
        version = "v1"

        def judge(self, image, attributes, prompt=None):
            return {a.id: True for a in attributes}

    project = load_project(root, use_local=False)
    scorers, identity_error = build_default_scorers(
        project, judge=FakeJudge(), identity=False
    )
    assert identity_error is None
    assert [scorer.name for scorer in scorers] == ["health", "adherence", "defects"]
    assert set(scorers[1].attributes_by_scenario) == set(project.scenarios)
