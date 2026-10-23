from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def project(root: Path):
    from lora_lab.config import load_project

    return load_project(root, use_local=False)
