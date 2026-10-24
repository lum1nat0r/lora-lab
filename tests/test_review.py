import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lora_lab.analyze.agreement import MIN_LABELS, agreement, cohen_kappa
from lora_lab.analyze.report import gate_by_calibration
from lora_lab.analyze.stats import Sample, is_fail, is_pass, paired_difference
from lora_lab.analyze.tournament import bradley_terry, pick_pair, standings
from lora_lab.config import load_project
from lora_lab.db import Store
from lora_lab.grid import Job
from lora_lab.heldout import near_duplicates
from lora_lab.scorers.base import ImageRef
from lora_lab.scorers.defects import DefectScorer
from lora_lab.web.app import create_app

# -- calibration --------------------------------------------------------------


def test_cohen_kappa() -> None:
    assert cohen_kappa([(True, True), (False, False)] * 5) == pytest.approx(1.0)
    assert cohen_kappa([(True, False), (False, True)] * 5) == pytest.approx(-1.0)
    # chance agreement already perfect: kappa is undefined, not 1
    assert cohen_kappa([(True, True)] * 4) is None
    assert cohen_kappa([]) is None


def test_agreement_status_and_gate() -> None:
    labels = {i: {"vase": i % 2 == 0, "defect:text_watermark": False} for i in range(MIN_LABELS)}
    # judge answers the opposite of the human on the attribute
    verdicts = {i: {"vase": i % 2 == 1, "defect:text_watermark": False} for i in range(MIN_LABELS)}
    result = agreement(labels, verdicts)
    assert result["adherence"]["status"] == "rejected"
    assert result["adherence"]["kappa"] == pytest.approx(-1.0)
    # defects all "absent" on both sides: no evidence either way
    assert result["defects"]["status"] == "uncalibrated"

    sample = Sample("l", "s", "sc", 1, identity=0.9, adherence=0.2, quality=0.5, defects=("x",))
    gated = gate_by_calibration([sample], result)
    assert gated[0].adherence is None
    assert gated[0].defects == ("x",)


def test_too_few_labels_is_uncalibrated() -> None:
    labels = {1: {"vase": True}, 2: {"vase": False}}
    result = agreement(labels, {1: {"vase": True}, 2: {"vase": False}})
    assert result["adherence"]["status"] == "uncalibrated"
    assert result["adherence"]["agreement"] == 1.0


# -- defect judge -------------------------------------------------------------


class FakeJudge:
    name = "fake"
    version = "v1"

    def __init__(self, answers: dict) -> None:
        self.answers = answers

    def judge(self, image, attributes, prompt=None):
        return {a.id: self.answers.get(a.id) for a in attributes}


IMAGE = ImageRef(id=1, path="x.png", subject="s", lora_id="l", epoch=1, strength=1.0,
                 scenario="sc", seed=1)


def test_defect_scorer_records_detected_absent_missing() -> None:
    judge = FakeJudge({"extra_limb": True, "text_watermark": False})
    (score,) = DefectScorer(judge).score(IMAGE)
    assert score.axis == "defects"
    assert score.value > 0
    assert score.meta["detected"] == ["extra_limb"]
    assert score.meta["absent"] == ["text_watermark"]
    assert "blur_lowres" in score.meta["missing"]


def test_defect_scorer_asks_prompt_defects_only_with_prompt() -> None:
    class Recorder(FakeJudge):
        def judge(self, image, attributes, prompt=None):
            self.seen = (prompt, [a.id for a in attributes])
            return super().judge(image, attributes)

    judge = Recorder({"clothing_ignored": True})
    (score,) = DefectScorer(judge, {"sc": "red dress"}).score(IMAGE)
    assert judge.seen[0] == "red dress"
    assert "clothing_ignored" in judge.seen[1]
    assert score.meta["detected"] == ["clothing_ignored"]

    DefectScorer(judge, {"other": "x"}).score(IMAGE)
    assert judge.seen[0] is None
    assert "clothing_ignored" not in judge.seen[1]


def test_defect_scorer_without_answers_is_none() -> None:
    (score,) = DefectScorer(FakeJudge({})).score(IMAGE)
    assert score.value is None


def test_defects_drive_pass_and_fail() -> None:
    base = dict(lora_id="l", subject="s", scenario="sc", seed=1,
                identity=0.9, adherence=0.9, quality=0.9)
    assert is_pass(Sample(**base))
    assert is_pass(Sample(**base, defects=("blur_lowres",)))  # severity 1 tolerated
    assert not is_pass(Sample(**base, defects=("text_watermark",)))
    assert not is_fail(Sample(**base, defects=("text_watermark",)))
    assert is_fail(Sample(**base, defects=("extra_limb",)))  # critical


# -- paired comparison ----------------------------------------------------------


def _grid(lora: str, offset: float) -> list[Sample]:
    out = []
    for s_index, scenario in enumerate(("a", "b", "c")):
        for seed in (1, 2, 3):
            value = 0.3 + 0.1 * s_index + 0.05 * seed + offset
            out.append(Sample(lora, "s", scenario, seed, identity=value, adherence=value,
                              quality=value))
    return out


def test_paired_difference_detects_consistent_gain() -> None:
    result = paired_difference(_grid("new", 0.1), _grid("old", 0.0), n=500)
    assert result["pairs"] == 9
    assert result["mean_diff"] == pytest.approx(0.1)
    assert result["lcb"] > 0
    assert result["verdict"] == "a"
    assert result["wins"] == 9


def test_paired_difference_only_uses_shared_cells() -> None:
    a = _grid("new", 0.0)
    b = [s for s in _grid("old", 0.0) if s.scenario == "a"]
    result = paired_difference(a, b, n=200)
    assert result["pairs"] == 3
    assert result["verdict"] == "none"
    assert paired_difference(a, [], n=10) is None


# -- tournament ---------------------------------------------------------------


def test_bradley_terry_orders_by_wins() -> None:
    votes = [("a", "b", "left")] * 8 + [("b", "c", "left")] * 8 + [("a", "c", "tie")] * 2
    strength = bradley_terry(votes)
    assert strength["a"] > strength["b"] > strength["c"]
    rows = standings(votes, bootstrap=50)
    assert [r["op"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["wins"] == 8 and rows[0]["ties"] == 2
    assert rows[0]["low"] <= rows[0]["score"] <= rows[0]["high"]


def test_pick_pair_shares_cell_and_prefers_unseen_pairs() -> None:
    rows = [
        {"id": i, "lora_id": lora, "subject": "s", "epoch": 1, "strength": 1.0,
         "scenario": "sc", "seed": seed}
        for i, (lora, seed) in enumerate(
            [("a", 1), ("b", 1), ("c", 1), ("a", 2), ("b", 2)]
        )
    ]
    votes = [("a@ep1@s1", "b@ep1@s1", "left")]
    for _ in range(20):
        left, right = pick_pair(rows, votes, rng=random.Random(_))
        assert left["seed"] == right["seed"]
        assert {left["lora_id"], right["lora_id"]} != {"a", "b"}
    assert pick_pair(rows[:1], []) is None


# -- near duplicates ------------------------------------------------------------


class FileEmbedder:
    """Embeds by image colour, so a re-saved copy lands on the same vector."""

    def embed(self, path: str):
        with Image.open(path) as im:
            r, g, b = im.convert("RGB").getpixel((0, 0))
        return [r + 1.0, g + 1.0, b + 1.0]


def test_near_duplicates_finds_reencoded_copy(tmp_path: Path) -> None:
    refs, train = tmp_path / "refs", tmp_path / "train"
    refs.mkdir()
    train.mkdir()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(refs / "r1.png")
    Image.new("RGB", (8, 8), (10, 10, 200)).save(refs / "r2.png")
    Image.new("RGB", (8, 8), (200, 10, 10)).save(train / "t1.jpg", quality=100)
    Image.new("RGB", (8, 8), (10, 200, 10)).save(train / "t2.png")

    found = near_duplicates(refs, train, FileEmbedder())
    assert [(d["ref"], d["train"]) for d in found] == [("r1.png", "t1.jpg")]
    assert near_duplicates(refs, train, FileEmbedder(), exact=[{"ref": "r1.png", "train": "t1.jpg"}]) == []


# -- web ------------------------------------------------------------------------


def _seed(root: Path, tmp: Path):
    project = load_project(root, use_local=False).model_copy(deep=True)
    project.data_dir = tmp / "data"
    project.db_path = tmp / "db.sqlite"

    store = Store(project.db_path)
    run_id = store.upsert_run("smoke", "hash", "fast", {})
    for lora in ("lora_a", "lora_b"):
        for seed in (1, 2):
            job = Job(subject="subject_a", lora_id=lora, epoch=20, strength=1.0,
                      scenario="still_life", seed=seed, lora_file=f"{lora}.safetensors")
            job_id = store.add_job(run_id, f"{lora}-{seed}", job, {})
            relative = f"images/{job.prefix()}/frame.png"
            target = project.data_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.effect_noise((64, 64), 120).convert("RGB").save(target)
            store.add_image(job_id, 0, "frame.png", "", "output", relative, "sha", 64, 64, 1)
    rows = store.images_for_run(run_id)
    for row in rows:
        store.add_score(row["id"], "health", "v1", "quality", 0.5, {})
        # the judge always claims the vase is there
        store.add_score(row["id"], "adherence", "v1", "adherence", 1.0,
                        {"passed": ["object_present"], "failed": [], "missing": []})
    store.commit()
    store.close()
    return project, [int(r["id"]) for r in rows]


def test_label_and_duel_pages(root: Path, tmp_path: Path) -> None:
    project, image_ids = _seed(root, tmp_path)
    client = TestClient(create_app(project))

    page = client.get("/label")
    assert page.status_code == 200
    assert "Gold-Set labeln" in page.text
    assert "Is a ceramic vase visible?" in page.text

    res = client.post(
        "/label",
        data={"image_id": str(image_ids[0]), "attr:object_present": "no",
              "defect:text_watermark": "on"},
        follow_redirects=False,
    )
    assert res.status_code == 303
    store = Store(project.db_path)
    labels = store.labels()[image_ids[0]]
    store.close()
    assert labels["object_present"] is False
    assert labels["defect:text_watermark"] is True
    assert labels["defect:extra_limb"] is False

    duel = client.get("/duel")
    assert duel.status_code == 200
    assert "Links besser" in duel.text
    assert client.post("/duel", data={"left_id": str(image_ids[0]), "right_id": str(image_ids[2]),
                                      "winner": "left"}, follow_redirects=False).status_code == 303
    assert client.post("/duel", data={"left_id": "1", "right_id": "1", "winner": "left"},
                       follow_redirects=False).headers["location"].startswith("/duel?err=")
    standings_page = client.get("/duel")
    assert "1 Stimmen" in standings_page.text
    assert "lora_a@ep20@s1" in standings_page.text


def test_report_gates_rejected_judge_and_compares_paired(root: Path, tmp_path: Path) -> None:
    project, image_ids = _seed(root, tmp_path)
    client = TestClient(create_app(project))
    assert client.get("/api/report").json()["results"][0]["axes"]["adherence"] == 1.0

    # humans disagree with the judge on half the images
    store = Store(project.db_path)
    for index, image_id in enumerate(image_ids):
        store.set_labels(image_id, {"object_present": index % 2 == 0})
    store.close()

    payload = client.get("/api/report").json()  # labels invalidate the cache
    calibration = payload["calibration"]["adherence"]
    assert calibration["n"] == 4
    assert calibration["agreement"] == 0.5
    # 4 answers is below MIN_LABELS: the gate must not fire yet
    assert calibration["status"] == "uncalibrated"
    assert payload["results"][0]["axes"]["adherence"] == 1.0

    paired = client.get("/report", params={"a": "lora_a@ep20@s1", "b": "lora_b@ep20@s1"})
    assert paired.status_code == 200
    assert "Gepaarter Vergleich" in paired.text
    assert "Kein belastbarer Unterschied" in paired.text
    missing = client.get("/report", params={"a": "lora_a@ep20@s1", "b": "nope@ep1@s1"})
    assert "keine gemeinsamen" in missing.text
