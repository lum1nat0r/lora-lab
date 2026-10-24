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
    assert "Side-by-Side Bildvergleich" in report.text
    assert "/images/images/" in report.text

    api = client.get("/api/report").json()
    assert api["samples"] == 4
    assert len(api["results"]) == 2
    assert len(api["results"][0]["images"]) == 2
    assert api["results"][0]["axes"]["quality"] is not None
    assert len(api["compare"]) == 2

    # Delete a single LoRA operating point and verify files + DB records are gone
    res = client.post(
        "/report/delete-lora",
        data={"lora_id": "lora_a", "epoch": "20", "strength": "1.0", "run": ""},
        follow_redirects=False,
    )
    assert res.status_code == 303
    api_after_lora = client.get("/api/report").json()
    assert api_after_lora["samples"] == 2
    assert [r["lora_id"] for r in api_after_lora["results"]] == ["lora_b"]
    assert not any((project.data_dir / "images").rglob("*lora_a*"))

    # Delete the entire run and verify everything is cleaned up
    res_run = client.post(
        "/report/delete-run",
        data={"run": "smoke"},
        follow_redirects=False,
    )
    assert res_run.status_code == 303
    api_empty = client.get("/api/report").json()
    assert api_empty["samples"] == 0
    assert api_empty["results"] == []
    assert list((project.data_dir / "images").rglob("*.png")) == []


def test_report_payload_is_cached_until_data_changes(root, tmp_path, monkeypatch) -> None:
    import lora_lab.analyze.report as report_mod

    project = _seed(root, tmp_path)
    calls = {"n": 0}
    original = report_mod.build_report

    def counting(store, **kwargs):
        calls["n"] += 1
        return original(store, **kwargs)

    monkeypatch.setattr(report_mod, "build_report", counting)
    client = TestClient(create_app(project))
    first = client.get("/api/report").json()
    second = client.get("/api/report").json()
    assert calls["n"] == 1
    assert second == first

    # a new score invalidates the cache
    store = Store(project.db_path)
    store.add_score(1, "face_count", "v1", "critical", 1.0, {})
    store.commit()
    store.close()
    third = client.get("/api/report").json()
    assert calls["n"] == 2
    assert third["total_images"] == first["total_images"]
