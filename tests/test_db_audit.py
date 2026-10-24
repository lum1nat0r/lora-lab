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
    store.add_score(1, "health", "v1", "quality", 0.85, {"sharpness": 5.0})
    assert store.image_count() == 1
    counts = store.counts()
    assert counts["done"] == 1
    assert counts["total"] == 1

    # Reusing the same job_hash in a second run links the image & score to the new run
    run2_id = store.upsert_run("test2", "abc", "fast", {})
    assert store.reuse_completed_job(run2_id, "hash1", job, {}) is True
    assert len(store.scored_images(run2_id)) == 1
    assert store.scored_images(run2_id)[0]["quality"] == 0.85
    store.close()


def _shared_image_store(tmp_path: Path) -> tuple[Store, Path, Path, int, int]:
    """Two runs whose job is reused, so both point at the same file."""

    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    image = data / "images" / "shared.png"
    image.write_bytes(b"png")

    store = Store(tmp_path / "db.sqlite")
    run_one = store.upsert_run("one", "abc", "fast", {})
    job = Job(
        subject="c",
        lora_id="l",
        epoch=1,
        strength=1.0,
        scenario="s",
        seed=1,
        lora_file="f.safetensors",
    )
    job_id = store.add_job(run_one, "hash", job, {})
    store.add_image(job_id, 0, "shared.png", "", "output", "images/shared.png", "sha", 8, 4, 4)
    store.set_status(job_id, "done", prompt_id="p")
    store.commit()
    run_two = store.upsert_run("two", "abc", "fast", {})
    assert store.reuse_completed_job(run_two, "hash", job, {})
    store.commit()
    return store, data, image, run_one, run_two


def test_delete_run_keeps_files_shared_with_other_runs(tmp_path: Path) -> None:
    store, data, image, run_one, run_two = _shared_image_store(tmp_path)

    # deleting the source run must not destroy the file the reuse still uses
    stats = store.delete_run(data, run_id=run_one)
    assert stats == {"jobs": 1, "files": 0}
    assert image.exists()
    assert store.image_count() == 1

    # the last referencing run takes the file with it
    stats = store.delete_run(data, run_id=run_two)
    assert stats == {"jobs": 1, "files": 1}
    assert not image.exists()
    store.close()


def test_delete_lora_keeps_files_shared_with_other_runs(tmp_path: Path) -> None:
    store, data, image, run_one, run_two = _shared_image_store(tmp_path)

    stats = store.delete_lora(data, "l", run_id=run_one)
    assert stats == {"jobs": 1, "files": 0}
    assert image.exists()

    stats = store.delete_lora(data, "l", run_id=run_two)
    assert stats == {"jobs": 1, "files": 1}
    assert not image.exists()
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
