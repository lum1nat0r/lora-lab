import shutil
from pathlib import Path

import yaml

from lora_lab.cli import check_paths
from lora_lab.config import load_project


def _project(tmp_path: Path, root: Path, subjects: list[dict]) -> object:
    target = tmp_path / "project"
    shutil.copytree(
        root / "config", target / "config", ignore=shutil.ignore_patterns("local")
    )
    shutil.copytree(root / "workflows", target / "workflows")
    (target / "config" / "local").mkdir()
    # the sample grid references subject_a and subject_b, keep them resolvable
    entries = [
        {"id": "subject_a", "trigger": "subja"},
        {"id": "subject_b", "trigger": "subjb"},
        *subjects,
    ]
    (target / "config" / "local" / "subjects.yaml").write_text(
        yaml.safe_dump({"subjects": entries})
    )
    return load_project(target)


def test_check_paths_warns_about_unwritable_reference_folder(
    tmp_path: Path, root: Path, capsys
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    project = _project(
        tmp_path,
        root,
        [
            {
                "id": "subj",
                "trigger": "subj",
                "ref_dir": str(blocker / "refs"),
                "train_dir": str(tmp_path / "train"),
            }
        ],
    )

    check_paths(project)
    out = capsys.readouterr().out
    assert "refs        subj" in out
    assert "MISSING" in out
    assert "not writable" in out


def test_check_paths_accepts_writable_reference_folder(
    tmp_path: Path, root: Path, capsys
) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "a.png").write_bytes(b"not really a png")
    project = _project(
        tmp_path,
        root,
        [
            {
                "id": "subj",
                "trigger": "subj",
                "ref_dir": str(refs),
                "train_dir": str(tmp_path / "train"),
            }
        ],
    )

    check_paths(project)
    out = capsys.readouterr().out
    assert "refs        subj" in out
    assert "1 images" in out
    assert "not writable" not in out
