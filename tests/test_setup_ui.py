import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from lora_lab.config import load_project
from lora_lab.web.app import create_app


@pytest.fixture()
def web(root: Path, tmp_path: Path):
    target = tmp_path / "project"
    shutil.copytree(
        root / "config", target / "config", ignore=shutil.ignore_patterns("local")
    )
    shutil.copytree(root / "workflows", target / "workflows")

    def reload():
        project = load_project(target)
        project.data_dir = tmp_path / "data"
        project.db_path = tmp_path / "data" / "db.sqlite"
        project.data_dir.mkdir(parents=True, exist_ok=True)
        return project

    project = reload()
    client = TestClient(create_app(project, reload_fn=reload))
    return client, target, tmp_path


def test_setup_page_renders(web) -> None:
    client, target, _ = web
    response = client.get("/setup")
    assert response.status_code == 200
    assert "Szenarien" in response.text
    assert "still_life" in response.text


def test_create_and_edit_scenario(web) -> None:
    client, target, _ = web
    response = client.post(
        "/setup/scenarios/save",
        data={
            "is_new": "true",
            "id": "studio_test",
            "category": "core",
            "prompt": "{trigger}, a red chair in a white room, 50mm lens.",
            "width": "",
            "height": "",
            "attr_id": ["chair", "white_room"],
            "attr_question": ["Is a chair visible?", "Is the wall white?"],
            "attr_weight": ["1", "2"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    data = yaml.safe_load((target / "config" / "local" / "scenarios.yaml").read_text())
    entry = next(item for item in data["scenarios"] if item["id"] == "studio_test")
    assert entry["attributes"][1]["weight"] == 2.0

    page = client.get("/setup")
    assert "studio_test" in page.text

    response = client.post(
        "/setup/scenarios/save",
        data={
            "is_new": "false",
            "id": "studio_test",
            "category": "optional",
            "prompt": "{trigger}, a blue chair, 35mm lens.",
            "attr_id": ["chair"],
            "attr_question": ["Is a chair visible?"],
            "attr_weight": ["1"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    data = yaml.safe_load((target / "config" / "local" / "scenarios.yaml").read_text())
    entry = next(item for item in data["scenarios"] if item["id"] == "studio_test")
    assert entry["category"] == "optional"


def test_scenario_requires_trigger(web) -> None:
    client, target, _ = web
    response = client.post(
        "/setup/scenarios/save",
        data={
            "is_new": "true",
            "id": "broken",
            "category": "core",
            "prompt": "no trigger here",
            "attr_id": ["x"],
            "attr_question": ["Is x visible?"],
            "attr_weight": ["1"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "err=" in response.headers["location"]
    assert not (target / "config" / "local" / "scenarios.yaml").exists()


def test_subject_and_reference_upload(web) -> None:
    client, target, tmp_path = web
    refs = tmp_path / "refs" / "subject_c"
    response = client.post(
        "/setup/subjects/save",
        data={
            "is_new": "true",
            "id": "subject_c",
            "trigger": "subjc",
            "ref_dir": str(refs),
            "train_dir": "/datasets/subject_c",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    data = yaml.safe_load((target / "config" / "local" / "subjects.yaml").read_text())
    assert any(item["id"] == "subject_c" for item in data["subjects"])

    image = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), (10, 20, 30)).save(image)
    with image.open("rb") as fh:
        response = client.post(
            "/setup/subjects/subject_c/refs/upload",
            files={"files": ("sample.png", fh.read(), "image/png")},
            follow_redirects=False,
        )
    assert response.status_code == 303
    assert (refs / "sample.png").exists()

    page = client.get("/setup/subjects/subject_c/refs")
    assert page.status_code == 200
    assert "sample.png" in page.text

    served = client.get("/refs/subject_c/sample.png")
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/")


def test_upload_to_unwritable_folder_reports_error(web) -> None:
    """A read-only mount must not kill the request — the body is consumed first."""
    client, _, tmp_path = web
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    refs = blocker / "refs"

    response = client.post(
        "/setup/subjects/save",
        data={
            "is_new": "true",
            "id": "subject_d",
            "trigger": "subjd",
            "ref_dir": str(refs),
            "train_dir": "/datasets/subject_d",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    image = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), (10, 20, 30)).save(image)
    with image.open("rb") as fh:
        response = client.post(
            "/setup/subjects/subject_d/refs/upload",
            files={"files": ("sample.png", fh.read(), "image/png")},
            follow_redirects=False,
        )
    assert response.status_code == 303
    assert "err=" in response.headers["location"]
    assert "beschreibbar" in response.headers["location"]
    assert not refs.exists()

    page = client.get("/setup/subjects/subject_d/refs")
    assert page.status_code == 200
    assert "nicht beschreibbar" in page.text


def test_upload_without_image_files_reports_error(web) -> None:
    client, _, tmp_path = web
    response = client.post(
        "/setup/subjects/save",
        data={
            "is_new": "true",
            "id": "subject_e",
            "trigger": "subje",
            "ref_dir": str(tmp_path / "refs" / "subject_e"),
            "train_dir": "/datasets/subject_e",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    response = client.post(
        "/setup/subjects/subject_e/refs/upload",
        files={"files": ("notes.txt", b"not an image", "text/plain")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "err=" in response.headers["location"]


def test_lora_selection_and_grid(web) -> None:
    client, target, _ = web
    local = target / "config" / "local"
    local.mkdir(exist_ok=True)
    (local / "subjects.yaml").write_text(
        yaml.safe_dump(
            {
                "subjects": [
                    {"id": "subject_a", "trigger": "subja"},
                    {"id": "subject_b", "trigger": "subjb"},
                ]
            }
        )
    )
    (local / "loras.all.yaml").write_text(
        yaml.safe_dump(
            {
                "loras": [
                    {"id": "a_ep10", "subject": "subject_a", "file": "a/10.safetensors", "epoch": 10},
                    {"id": "a_ep20", "subject": "subject_a", "file": "a/20.safetensors", "epoch": 20},
                    {"id": "b_ep10", "subject": "subject_b", "file": "b/10.safetensors", "epoch": 10},
                ]
            }
        )
    )

    page = client.get("/loras")
    assert page.status_code == 200
    assert "a_ep10" in page.text
    assert 'data-folder="a"' in page.text
    assert 'data-family="b"' in page.text
    assert "selectGroupFinal" in page.text

    response = client.post(
        "/loras/save",
        data={
            "pick_0": "on",
            "subject_0": "subject_a",
            "epoch_0": "10",
            "subject_2": "subject_b",
            "epoch_2": "10",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    loras = yaml.safe_load((local / "loras.yaml").read_text())["loras"]
    assert [item["id"] for item in loras] == ["a_ep10"]

    grid = client.get("/setup/grid")
    assert grid.status_code == 200
    response = client.post(
        "/setup/grid/save",
        data={
            "name": "test-grid",
            "strengths": "0.8, 1.0",
            "seeds": "1, 2",
            "scenarios": ["still_life", "landscape"],
            "profile": "fast",
            "negative_prompt": "bad",
            "width": "1024",
            "height": "1024",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    saved = yaml.safe_load((local / "grid.yaml").read_text())
    assert saved["name"] == "test-grid"
    assert saved["loras_file"] == "config/local/loras.yaml"
    assert saved["strengths"] == [0.8, 1.0]


def test_run_page(web) -> None:
    client, _, _ = web
    response = client.get("/run")
    assert response.status_code == 200
    assert "Generieren" in response.text
    status = client.get("/api/run").json()
    assert status["running"] is False


def test_import_scenario_pack(web) -> None:
    client, target, _ = web
    pack = """
scenarios:
  - id: imported
    category: core
    prompt: "{trigger}, a brass lamp on a desk, 50mm lens."
    attributes:
      - id: lamp
        question: Is a lamp visible?
        weight: 1
"""
    response = client.post(
        "/setup/scenarios/import",
        files={"file": ("pack.yaml", pack.encode(), "application/x-yaml")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    data = yaml.safe_load((target / "config" / "local" / "scenarios.yaml").read_text())
    assert any(item["id"] == "imported" for item in data["scenarios"])


def test_deleting_planned_scenarios_warns_once_without_crashing(web) -> None:
    client, target, _ = web
    response = client.post(
        "/setup/grid/save",
        data={
            "name": "test-grid",
            "strengths": "0.8, 1.0",
            "seeds": "1, 2",
            "scenarios": ["still_life", "landscape", "texture_macro"],
            "profile": "fast",
            "negative_prompt": "bad",
            "width": "1024",
            "height": "1024",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    for sid in ("landscape", "still_life", "texture_macro"):
        res = client.post(f"/setup/scenarios/{sid}/delete", follow_redirects=False)
        assert res.status_code == 303

    index = client.get("/")
    assert index.status_code == 200
    assert "Gelöschte Szenarien wurden aus dem geplanten Lauf entfernt" in index.text
    for sid in ("landscape", "still_life", "texture_macro"):
        assert sid in index.text

    # Notice is shown only once.
    index_again = client.get("/")
    assert index_again.status_code == 200
    assert "Gelöschte Szenarien wurden aus dem geplanten Lauf entfernt" not in index_again.text

    for url in ("/setup", "/setup/grid", "/run", "/loras"):
        assert client.get(url).status_code == 200

    saved_grid = yaml.safe_load((target / "config" / "local" / "grid.yaml").read_text())
    assert "scenarios" not in saved_grid

