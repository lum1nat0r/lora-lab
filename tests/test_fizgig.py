import json
import struct
from pathlib import Path

import pytest

from lora_lab.importers.fizgig import (
    SafetensorsError,
    epoch_from_filename,
    import_lora,
    normalise,
    read_header,
)


def write_fake_safetensors(path: Path, metadata: dict[str, str]) -> Path:
    header = {"__metadata__": metadata, "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    blob = json.dumps(header).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00\x00")
    return path


def test_read_header(tmp_path: Path) -> None:
    path = write_fake_safetensors(tmp_path / "a.safetensors", {"ss_network_dim": "16"})
    header = read_header(path)
    assert header["__metadata__"]["ss_network_dim"] == "16"


def test_rejects_short_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"123")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_normalise_maps_kohya_keys() -> None:
    meta = {
        "ss_network_dim": "16",
        "ss_network_alpha": "16",
        "ss_learning_rate": "0.0001",
        "ss_optimizer": "adamw",
        "ss_epoch": "70",
        "ss_resolution": "(1024, 1024)",
        "ss_trigger_word": "subja",
        "ss_num_train_images": "62",
        "ss_bucket_info": '{"buckets": {}}',
        "ss_network_args": '["lownoise_pct=60:7:30:15:2"]',
        "ss_base_model_version": "moss_image",
    }
    out = normalise(meta)
    assert out["network_dim"] == 16
    assert out["learning_rate"] == pytest.approx(1e-4)
    assert out["epochs"] == 70
    assert out["num_train_images"] == 62
    assert out["bucket_info"] == {"buckets": {}}
    assert out["network_args"] == ["lownoise_pct=60:7:30:15:2"]
    assert out["trigger_word"] == "subja"


def test_import_includes_filename_epoch(tmp_path: Path) -> None:
    path = write_fake_safetensors(
        tmp_path / "lora-000066.safetensors",
        {"ss_network_dim": "16", "ss_optimizer": "adamw"},
    )
    record = import_lora(path)
    assert record["epoch_from_filename"] == 66
    assert record["network_dim"] == 16
    assert record["file"] == "lora-000066.safetensors"


def test_epoch_from_filename_patterns() -> None:
    assert epoch_from_filename(Path("x-000066.safetensors")) == 66
    assert epoch_from_filename(Path("x-000066-state.safetensors")) == 66
    assert epoch_from_filename(Path("x.safetensors")) is None


def test_read_fizgig_runs(tmp_path: Path) -> None:
    import json

    from lora_lab.importers.fizgig import normalise_run, read_runs

    payload = [
        {
            "id": "abc123",
            "family": "moss_image",
            "lora": "lora_a",
            "output_dir": "/out/lora_a",
            "image_folder": "/datasets/subject_a/train",
            "values": {"network_dim": 16, "learning_rate": 1e-4, "max_train_epochs": 70},
            "started_at": 1792775892.0,
            "ended_at": 1792780000.0,
            "result": "finished",
            "notes": "slide default",
        }
    ]
    (tmp_path / "runs.json").write_text(json.dumps(payload), encoding="utf-8")

    runs = read_runs(tmp_path)
    assert len(runs) == 1
    record = normalise_run(runs[0])
    assert record["lora"] == "lora_a"
    assert record["settings"]["network_dim"] == 16
    assert record["notes"] == "slide default"


def test_read_runs_missing(tmp_path: Path) -> None:
    from lora_lab.importers.fizgig import read_runs

    with pytest.raises(FileNotFoundError):
        read_runs(tmp_path)
