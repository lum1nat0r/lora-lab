from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_contract() -> None:
    config = yaml.safe_load((ROOT / "docker" / "compose.yaml").read_text())
    service = config["services"]["lora-lab"]

    assert service["image"].startswith("ghcr.io/")
    assert service["ports"] == ["${LORA_LAB_PORT:-8199}:8099"]

    volumes = "\n".join(service["volumes"])
    for container_path in (
        ":/app/config",
        ":/app/data",
        ":/datasets:ro",
        ":/datasets/_refs:rw",
        ":/models:ro",
        ":/fizgig:ro",
        ":/comfy-output:ro",
    ):
        assert container_path in volumes

    env = service["environment"]
    assert env["COMFY_OUTPUT_DIR"] == "/comfy-output"
    assert "LORA_LAB_PORT" not in env  # host-side only, not consumed by the app
