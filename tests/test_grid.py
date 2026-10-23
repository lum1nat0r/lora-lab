from pathlib import Path

import pytest

from lora_lab.config import load_project
from lora_lab.grid import Job, expand, summary


def test_project_loads(project) -> None:
    assert project.grid.name == "example-grid"
    assert len(project.subjects) == 2
    assert len(project.scenario_list()) == 3
    assert all(s.category == "core" for s in project.scenario_list())


def test_optional_scenarios_are_opt_in(project) -> None:
    ids = {s.id for s in project.scenario_list()}
    optional = {s.id for s in project.scenarios.values() if s.category != "core"}
    assert ids.isdisjoint(optional)


def test_trigger_and_strip(project) -> None:
    scenario = project.scenario_list()[0]
    assert scenario.render("subja").startswith("subja")
    stripped = scenario.strip_trigger("subja")
    assert "subja" not in stripped


def test_expand_counts(project) -> None:
    jobs = expand(project.grid, project.scenario_list(), project.subjects)
    expected = (
        len(project.grid.loras)
        * len(project.grid.strengths)
        * len(project.scenario_list())
        * len(project.grid.seeds)
    )
    assert len(jobs) == expected
    assert summary(jobs)["total"] == expected
    # deterministic ordering
    assert jobs == expand(project.grid, project.scenario_list(), project.subjects)


def test_every_lora_sees_every_seed(project) -> None:
    jobs = expand(project.grid, project.scenario_list(), project.subjects)
    for lora in project.grid.loras:
        seeds = {j.seed for j in jobs if j.lora_id == lora.id}
        assert seeds == set(project.grid.seeds)


def test_seeds_are_shared_across_loras(project) -> None:
    jobs = expand(project.grid, project.scenario_list(), project.subjects)
    first = min(jobs, key=lambda j: (j.lora_id, j.seed))
    same_seed = [j for j in jobs if j.seed == first.seed]
    assert len({j.lora_id for j in same_seed}) == len(project.grid.loras)


def test_prefix_layout() -> None:
    job = Job(
        subject="subject_a",
        lora_id="lora_a_ep20",
        epoch=20,
        strength=0.9,
        scenario="still_life",
        seed=101,
        lora_file="x.safetensors",
    )
    assert job.prefix() == "fast/subject_a/lora_a_ep20/ep20/s0.9/still_life/101"


def test_baseline_jobs(project) -> None:
    grid = project.grid.model_copy(update={"conditions": ["base_only"]})
    jobs = expand(grid, project.scenario_list(), project.subjects)
    subjects = {j.subject for j in jobs}
    assert subjects == {lora.subject for lora in project.grid.loras}
    assert all(j.is_baseline for j in jobs)
    assert len(jobs) == len(subjects) * len(project.scenario_list()) * len(project.grid.seeds)


def test_duplicate_lora_ids_rejected() -> None:
    from lora_lab.config import GridConfig

    raw = {
        "name": "x",
        "workflow": "w.json",
        "bindings": {
            "lora_node": 1,
            "positive": {"id": 2, "input": "text"},
            "negative": {"id": 3, "input": "text"},
            "seed": {"id": 4, "input": "seed"},
            "save": {"id": 5, "input": "filename_prefix"},
        },
        "loras": [
            {"id": "a", "subject": "c", "file": "f", "epoch": 1},
            {"id": "a", "subject": "c", "file": "g", "epoch": 2},
        ],
    }
    with pytest.raises(ValueError):
        GridConfig(**raw)


def test_missing_config_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path)
