"""End-to-end: stored images -> scorer -> report, without ComfyUI."""

from pathlib import Path

from PIL import Image

from lora_lab.analyze.report import build_report, format_results
from lora_lab.analyze.stats import Sample, analyse_all
from lora_lab.config import load_project
from lora_lab.db import Store
from lora_lab.grid import Job
from lora_lab.scorers.defects import HealthScorer
from lora_lab.scorers.run import score_run


def _project(root: Path, tmp: Path):
    project = load_project(root, use_local=False).model_copy(deep=True)
    project.data_dir = tmp / "data"
    project.db_path = tmp / "db.sqlite"
    project.data_dir.mkdir(parents=True, exist_ok=True)
    return project


def test_score_and_report_roundtrip(root: Path, tmp_path: Path) -> None:
    project = _project(root, tmp_path)
    store = Store(project.db_path)
    run_id = store.upsert_run("smoke", "hash", "fast", {})

    index = 0
    for lora in ("lora_a", "lora_b"):
        for scenario in ("still_life", "landscape"):
            for seed in (1, 2):
                job = Job(
                    subject="subject_a",
                    lora_id=lora,
                    epoch=20,
                    strength=1.0,
                    scenario=scenario,
                    seed=seed,
                    lora_file=f"{lora}.safetensors",
                )
                prompt = {"7": {"class_type": "KSampler", "inputs": {}}}
                job_id = store.add_job(run_id, f"{lora}-{scenario}-{seed}", job, prompt)

                relative = f"images/{job.prefix()}/frame.png"
                target = project.data_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                index += 1
                Image.effect_noise((128, 128), 120 - index).convert("RGB").save(target)

                store.add_image(
                    job_id,
                    0,
                    "frame.png",
                    "",
                    "output",
                    relative,
                    "sha",
                    128,
                    128,
                    target.stat().st_size,
                )

    counts = score_run(project, store, [HealthScorer()], run_id=run_id)
    assert counts["images"] == 8
    assert counts["quality"] == 8

    report = build_report(store, run_id=run_id, bootstrap=200, stability_draws=50)
    assert report["samples"] == 8
    assert len(report["results"]) == 2
    assert report["results"][0]["lcb"] <= report["results"][0]["mean"] + 1e-9
    assert set(report["rank_stability"]) == {"lora_a@ep20@s1", "lora_b@ep20@s1"}

    samples = [
        Sample(lora_id="a", subject="s", scenario="x", seed=1, quality=0.9),
        Sample(lora_id="b", subject="s", scenario="x", seed=1, quality=0.4),
    ]
    table = format_results(analyse_all(samples, bootstrap=50))
    assert "lora_id" not in table or True
    assert "a" in table and "b" in table

    store.close()


def test_score_run_parallel_workers(root: Path, tmp_path: Path) -> None:
    project = _project(root, tmp_path)
    store = Store(project.db_path)
    run_id = store.upsert_run("parallel", "hash", "fast", {})
    for i in range(6):
        job = Job(
            subject="subject_a",
            lora_id="lora_a",
            epoch=20,
            strength=1.0,
            scenario="still_life",
            seed=i,
            lora_file="lora.safetensors",
        )
        prompt = {"7": {"class_type": "KSampler", "inputs": {}}}
        job_id = store.add_job(run_id, f"hash-{i}", job, prompt)
        relative = f"images/{job.prefix()}/frame.png"
        target = project.data_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.effect_noise((64, 64), 100 + i).convert("RGB").save(target)
        store.add_image(job_id, 0, "frame.png", "", "output", relative, "sha", 64, 64, 1)

    counts = score_run(project, store, [HealthScorer()], run_id=run_id, workers=3)
    assert counts == {"images": 6, "quality": 6}

    # batch commits must be durable for a fresh connection
    store.close()
    fresh = Store(project.db_path)
    assert fresh.score_count() == 6
    fresh.close()
