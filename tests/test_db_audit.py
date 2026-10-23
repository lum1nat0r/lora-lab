from pathlib import Path

from PIL import Image

from lora_lab.audit import audit_dataset
from lora_lab.db import Store
from lora_lab.grid import Job


def make_image(path: Path, size=(640, 480), color=(120, 30, 30)) -> None:
    Image.new("RGB", size, color).save(path)


def test_store_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite")
    run_id = store.upsert_run("test", "abc", "fast", {"x": 1})
    assert run_id > 0

    job = Job(
        subject="c",
        lora_id="l",
        epoch=1,
        strength=1.0,
        scenario="s",
        seed=1,
        lora_file="f.safetensors",
    )
    job_id = store.add_job(run_id, "hash1", job, {"7": {"class_type": "KSampler", "inputs": {}}})
    assert job_id > 0
    # idempotent insert returns the same row
    assert store.add_job(run_id, "hash1", job, {}) == job_id

    assert store.job_status("hash1") == "pending"
    store.set_status(job_id, "done", prompt_id="p1")
    assert store.job_status("hash1") == "done"

    store.add_image(job_id, 0, "a.png", "", "output", "images/a.png", "sha", 64, 48, 10)
    assert store.image_count() == 1
    counts = store.counts()
    assert counts["done"] == 1
    assert counts["total"] == 1
    store.close()


def test_audit_counts_and_duplicates(tmp_path: Path) -> None:
    make_image(tmp_path / "a.png", color=(1, 2, 3))
    make_image(tmp_path / "b.png", color=(1, 2, 3))
    make_image(tmp_path / "c.png", color=(9, 9, 9))
    (tmp_path / "a.txt").write_text("trigger, a red vase, studio", encoding="utf-8")

    result = audit_dataset(tmp_path)
    assert result["images"] == 3
    assert result["unreadable"] == []
    assert len(result["duplicates"]) == 1
    assert len(result["duplicates"][0]) == 2
    assert result["missing_captions"] == ["b.png", "c.png"]
    assert result["caption_word_stats"]["count"] == 1
    assert result["aspect_distribution"]["landscape"] == 3
    assert result["megapixels"]["mean"] > 0


def test_audit_missing_dir(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(NotADirectoryError):
        audit_dataset(tmp_path / "nope")
