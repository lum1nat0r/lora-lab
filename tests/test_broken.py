from pathlib import Path

from fastapi.testclient import TestClient

from lora_lab.web.broken import create_broken_app


def test_degraded_app_explains_the_error(tmp_path: Path) -> None:
    error = ValueError("lora 'x' references unknown subject 'y'")
    client = TestClient(create_broken_app(tmp_path, error))

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"

    page = client.get("/")
    assert page.status_code == 200
    assert "config error" in page.text
    assert "unknown subject" in page.text
    assert "discover" in page.text
