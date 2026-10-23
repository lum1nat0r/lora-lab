from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from lora_lab.config import load_project
from lora_lab.db import Store
from lora_lab.grid import Job
from lora_lab.scorers.defects import HealthScorer
from lora_lab.scorers.run import score_run
from lora_lab.web.app import create_app


def _seed(root: Path, tmp: Path):
    project = load_project(root, use_local=False).model_copy(deep=True)
    project.data_dir = tmp / "data"
    project.db_path = tmp / "db.sqlite"
    project.data_dir.mkdir(parents=True, exist_ok=True)

    store = Store(project.db_path)
    run_id = store.upsert_run("smoke", "hash", "fast", {})
    for lora in ("lora_a", "lora_b"):
        for scenario in ("still_life", "landscape"):
            job = Job(
                subject="subject_a",
                lora_id=lora,
                epoch=20,
                strength=1.0,
                scenario=scenario,
                seed=1,
                lora_file=f"{lora}.safetensors",
            )
            job_id = store.add_job(run_id, f"{lora}-{scenario}", job, {})
            relative = f"images/{job.prefix()}/frame.png"
            target = project.data_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.effect_noise((96, 96), 120).convert("RGB").save(target)
            store.add_image(job_id, 0, "frame.png", "", "output", relative, "sha", 96, 96, 1)
    score_run(project, store, [HealthScorer()], run_id=run_id)
    store.close()
    return project


def test_dashboard_and_report_render(root: Path, tmp_path: Path) -> None:
    project = _seed(root, tmp_path)
    client = TestClient(create_app(project))

    assert client.get("/healthz").json() == {"status": "ok"}
    index = client.get("/")
    assert index.status_code == 200
    assert "Dashboard" in index.text

    report = client.get("/report")
    assert report.status_code == 200
    assert "Ranking" in report.text
    assert "Szenario-Matrix" in report.text

    api = client.get("/api/report").json()
    assert api["samples"] == 4
    assert len(api["results"]) == 2
