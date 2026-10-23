import json
import struct
from pathlib import Path

from PIL import Image

from lora_lab.config import load_project
from lora_lab.discover import discover_loras, discover_subjects


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (80, 40, 20)).save(path)


def fake_lora(path: Path, metadata: dict[str, str]) -> None:
    header = {"__metadata__": metadata, "w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00\x00")


def test_discover_subjects(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    make_image(root / "subject_a" / "refs" / "1.png")
    make_image(root / "subject_a" / "refs" / "2.png")
    make_image(root / "subject_a" / "train" / "1.png")
    (root / "subject_a" / "trigger.txt").write_text("tri_a\n")

    make_image(root / "subject_b" / "references" / "1.png")
    make_image(root / "subject_b" / "images" / "1.png")

    subjects = discover_subjects(root)
    assert [s["id"] for s in subjects] == ["subject_a", "subject_b"]
    assert subjects[0]["trigger"] == "tri_a"
    assert subjects[0]["ref_count"] == 2
    assert subjects[0]["train_count"] == 1
    assert subjects[1]["ref_dir"].endswith("references")


def test_discover_loras_matches_subject_and_epoch(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    fake_lora(root / "moss" / "lisalu.le" / "lisalule-000066.safetensors", {"ss_trigger_word": "lisalule"})
    fake_lora(root / "moss" / "lisalu.le" / "lisalule-000067.safetensors", {"ss_trigger_word": "lisalule"})
    fake_lora(root / "moss" / "other" / "mystery.safetensors", {})

    loras = discover_loras(root, subjects=["lisalule"])
    by_id = {item["id"]: item for item in loras}
    assert "lisalule_ep66" in by_id
    assert by_id["lisalule_ep66"]["subject"] == "lisalule"
    assert by_id["lisalule_ep66"]["file"] == "moss/lisalu.le/lisalule-000066.safetensors"
    assert by_id["lisalule_ep66"]["epoch"] == 66
    assert by_id["mystery"]["subject"] == "other"


def test_loras_file_is_merged(tmp_path: Path, root: Path) -> None:
    import shutil

    import yaml

    target = tmp_path / "project"
    shutil.copytree(
        root / "config", target / "config", ignore=shutil.ignore_patterns("local")
    )
    shutil.copytree(root / "workflows", target / "workflows")

    (target / "config" / "local").mkdir(exist_ok=True)
    (target / "config" / "local" / "loras.yaml").write_text(
        yaml.safe_dump(
            {"loras": [{"id": "x_ep1", "subject": "subject_a", "file": "x/1.safetensors", "epoch": 1}]}
        )
    )
    grid_path = target / "config" / "grid.yaml"
    grid = yaml.safe_load(grid_path.read_text())
    grid["loras"] = []
    grid_path.write_text(yaml.safe_dump(grid))

    project = load_project(target)
    assert [lora.id for lora in project.grid.loras] == ["x_ep1"]


def test_apply_grid_points_at_loras_file(tmp_path: Path, root: Path) -> None:
    from lora_lab.discover import apply_grid, grid_source

    source = grid_source(root)
    assert source is not None
    target = tmp_path / "local" / "grid.yaml"
    apply_grid(target, source)

    import yaml

    data = yaml.safe_load(target.read_text())
    assert data["loras"] == []
    assert data["loras_file"] == "config/local/loras.yaml"
    assert data["bindings"]["lora_node"] == 11


def _copy_project(root: Path, target: Path) -> None:
    import shutil

    shutil.copytree(
        root / "config", target / "config", ignore=shutil.ignore_patterns("local")
    )
    shutil.copytree(root / "workflows", target / "workflows")


def test_inline_loras_with_unknown_subjects_fall_back_to_local(tmp_path: Path, root: Path) -> None:
    import yaml

    target = tmp_path / "project"
    _copy_project(root, target)
    (target / "config" / "local").mkdir()

    (target / "config" / "local" / "subjects.yaml").write_text(
        yaml.safe_dump({"subjects": [{"id": "real", "trigger": "real"}]})
    )
    (target / "config" / "local" / "loras.yaml").write_text(
        yaml.safe_dump(
            {"loras": [{"id": "real_ep1", "subject": "real", "file": "r/1.safetensors", "epoch": 1}]}
        )
    )

    project = load_project(target)
    assert [lora.id for lora in project.grid.loras] == ["real_ep1"]


def test_apply_grid_repairs_inconsistent_existing_grid(tmp_path: Path, root: Path) -> None:
    import shutil

    import yaml

    from lora_lab.discover import apply_grid

    target = tmp_path / "local" / "grid.yaml"
    target.parent.mkdir(parents=True)
    shutil.copy(root / "config" / "grid.yaml", target)

    apply_grid(target, root / "config" / "grid.yaml", {"real"})

    data = yaml.safe_load(target.read_text())
    assert data["loras"] == []
    assert data["loras_file"] == "config/local/loras.yaml"
