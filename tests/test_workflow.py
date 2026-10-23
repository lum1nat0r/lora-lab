from pathlib import Path

import pytest

from lora_lab.comfy.workflow import (
    WorkflowError,
    bypass_lora,
    drop_nodes,
    load_workflow,
    prompt_hash,
    render_prompt,
    ui_to_api,
    validate_prompt,
)


@pytest.fixture()
def ui(root: Path) -> dict:
    return load_workflow(root / "workflows" / "example-sft-dmd.json")


def test_ui_to_api_maps_widgets_by_position(ui: dict) -> None:
    api = ui_to_api(ui)

    ks = api["7"]
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["steps"] == 8
    assert ks["inputs"]["cfg"] == 1
    assert ks["inputs"]["sampler_name"] == "er_sde"
    assert ks["inputs"]["scheduler"] == "simple"
    assert ks["inputs"]["seed"] == 689534332298784
    # the frontend-only control_after_generate value must not survive
    assert "randomize" not in ks["inputs"].values()

    assert api["1"]["inputs"]["unet_name"].endswith(".safetensors")
    assert api["6"]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
    assert api["9"]["inputs"]["filename_prefix"].startswith("lora-lab/")
    assert api["11"]["inputs"]["strength_model"] == 0.9


def test_note_nodes_are_not_converted(ui: dict) -> None:
    api = ui_to_api(ui)
    assert "12" not in api


def test_links_are_resolved(ui: dict) -> None:
    api = ui_to_api(ui)
    assert api["7"]["inputs"]["model"] == ["11", 0]
    assert api["11"]["inputs"]["model"] == ["10", 0]
    assert api["7"]["inputs"]["positive"] == ["4", 0]


def test_drop_nodes_rejects_dangling_reference(ui: dict) -> None:
    api = ui_to_api(ui)
    with pytest.raises(WorkflowError):
        drop_nodes(api, [13])


def test_render_and_hash_are_deterministic(project, ui: dict) -> None:
    from lora_lab.grid import expand

    api = ui_to_api(ui)
    job = expand(project.grid, project.scenario_list(), project.subjects)[0]
    kwargs = dict(
        bindings=project.grid.bindings,
        positive="hello",
        negative="bad",
        seed=job.seed,
        filename_prefix=job.prefix(),
        lora_file=job.lora_file,
        lora_strength=job.strength,
        width=1024,
        height=1024,
    )
    prompt = render_prompt(api, **kwargs)
    digest = prompt_hash(prompt, project.grid.bindings)
    digest2 = prompt_hash(render_prompt(api, **kwargs), project.grid.bindings)

    assert prompt["4"]["inputs"]["text"] == "hello"
    assert prompt["7"]["inputs"]["seed"] == job.seed
    assert prompt["11"]["inputs"]["lora_name"] == job.lora_file
    assert prompt["11"]["inputs"]["strength_model"] == pytest.approx(job.strength)
    assert digest == digest2
    assert len(digest) == 64


def test_hash_ignores_output_prefix(project, ui: dict) -> None:
    api = ui_to_api(ui)
    base = dict(
        bindings=project.grid.bindings,
        positive="hello",
        negative="bad",
        seed=1,
        lora_file="x.safetensors",
        lora_strength=1.0,
        width=1024,
        height=1024,
    )
    a = render_prompt(api, filename_prefix="one", **base)
    b = render_prompt(api, filename_prefix="two", **base)
    assert prompt_hash(a, project.grid.bindings) == prompt_hash(b, project.grid.bindings)


def test_hash_changes_with_seed(project, ui: dict) -> None:
    api = ui_to_api(ui)
    kwargs = dict(
        bindings=project.grid.bindings,
        positive="hello",
        negative="bad",
        lora_file="x.safetensors",
        lora_strength=1.0,
        filename_prefix="p",
        width=1024,
        height=1024,
    )
    a = render_prompt(api, seed=1, **kwargs)
    b = render_prompt(api, seed=2, **kwargs)
    assert prompt_hash(a, project.grid.bindings) != prompt_hash(b, project.grid.bindings)


def test_bypass_lora_reconnects_model(project, ui: dict) -> None:
    api = ui_to_api(ui)
    prompt = render_prompt(
        api,
        bindings=project.grid.bindings,
        positive="hello",
        negative="bad",
        seed=1,
        filename_prefix="p",
        lora_file="x.safetensors",
        lora_strength=1.0,
    )
    bypassed = bypass_lora(prompt, project.grid.bindings.lora_node)
    assert "11" not in bypassed
    assert bypassed["7"]["inputs"]["model"] == ["10", 0]
    validate_prompt(bypassed, {"7", "9"})


def test_build_prompt_base_only(project, ui: dict) -> None:
    from lora_lab.grid import Job
    from lora_lab.runner import build_prompt

    subject = next(iter(project.subjects))
    scenario = project.scenario_list()[0].id
    job = Job(
        subject=subject,
        lora_id="",
        epoch=0,
        strength=0.0,
        scenario=scenario,
        seed=101,
        condition="base_only",
        profile="fast",
    )
    prompt, _ = build_prompt(job, project, ui_to_api(ui))
    assert "11" not in prompt
    assert prompt["7"]["inputs"]["model"] == ["10", 0]


def test_build_prompt_no_trigger(project, ui: dict) -> None:
    from lora_lab.grid import Job
    from lora_lab.runner import build_prompt

    subject = next(iter(project.subjects))
    scenario = project.scenario_list()[0].id
    trigger = project.subjects[subject].trigger
    job = Job(
        subject=subject,
        lora_id="lora_a_ep20",
        epoch=61,
        strength=1.0,
        scenario=scenario,
        seed=101,
        condition="no_trigger",
        profile="fast",
        lora_file="x.safetensors",
    )
    prompt, _ = build_prompt(job, project, ui_to_api(ui))
    text = prompt["4"]["inputs"]["text"]
    assert trigger not in text
    assert not text.startswith(trigger)
