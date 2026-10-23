"""Configuration models and loaders.

All paths in the YAML files are relative to the project root (the directory
that contains ``config/``), not to the YAML file itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Binding(BaseModel):
    """A node input the runner is allowed to overwrite."""

    id: int
    input: str


class WorkflowBindings(BaseModel):
    lora_node: int
    lora_file_input: str = "lora_name"
    lora_strength_input: str = "strength_model"
    positive: Binding
    negative: Binding
    seed: Binding
    save: Binding
    filename_prefix_input: str = "filename_prefix"
    size_node: int | None = None
    size_inputs: tuple[str, str] = ("width", "height")
    drop_nodes: list[int] = Field(default_factory=list)


class ProfileConfig(BaseModel):
    """Per-render-profile overrides.

    ``workflow`` may point to a different workflow file (e.g. the 30-step
    reference profile). ``overrides`` patches arbitrary node inputs, keyed by
    node id, and is applied last.
    """

    workflow: str | None = None
    overrides: dict[int, dict[str, Any]] = Field(default_factory=dict)


class LoraEntry(BaseModel):
    id: str
    subject: str
    file: str
    epoch: int
    strengths: list[float] | None = None
    notes: str | None = None


class GridConfig(BaseModel):
    name: str
    workflow: str
    profile: str = "fast"
    bindings: WorkflowBindings
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    strengths: list[float] = Field(default_factory=lambda: [0.8, 0.9, 1.0])
    seeds: list[int] = Field(default_factory=lambda: [101, 202, 303, 404])
    scenarios: list[str] | None = None
    conditions: list[Literal["lora", "base_only", "no_trigger"]] = Field(
        default_factory=lambda: ["lora"]
    )
    negative_prompt: str = "worst quality, bad anatomy, watermark, jpeg artifact"
    width: int = 1024
    height: int = 1024
    loras: list[LoraEntry] = Field(default_factory=list)
    loras_file: str | None = None

    @model_validator(mode="after")
    def _check(self) -> GridConfig:
        ids = [lora.id for lora in self.loras]
        if len(ids) != len(set(ids)):
            raise ValueError("grid.loras contains duplicate ids")
        if not self.strengths:
            raise ValueError("grid.strengths is empty")
        if not self.seeds:
            raise ValueError("grid.seeds is empty")
        for lora in self.loras:
            if lora.strengths is not None and not lora.strengths:
                raise ValueError(f"{lora.id}: strengths override is empty")
        return self

    def workflow_for(self, profile: str) -> str:
        override = self.profiles.get(profile)
        if override and override.workflow:
            return override.workflow
        return self.workflow


class Subject(BaseModel):
    id: str
    trigger: str
    ref_dir: str | None = None
    train_dir: str | None = None
    identity_prompt: str | None = None


class Attribute(BaseModel):
    id: str
    question: str
    weight: float = 1.0


class Scenario(BaseModel):
    id: str
    category: Literal["core", "optional", "control", "explicit"] = "core"
    prompt: str
    attributes: list[Attribute] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None
    notes: str | None = None

    def render(self, trigger: str) -> str:
        """Fill the trigger word.

        ``str.format`` is deliberately avoided: prompts contain literal
        braces in some templates (JSON layout prompts).
        """

        return self.prompt.replace("{trigger}", trigger)

    def strip_trigger(self, trigger: str) -> str:
        """Prompt without the trigger word, for the ``no_trigger`` control."""

        if not trigger:
            return self.prompt
        return self.render(trigger).replace(f"{trigger}, ", "").replace(trigger, "")


class ProjectConfig(BaseModel):
    root: Path
    data_dir: Path
    comfyui_url: str = "http://127.0.0.1:8188"
    comfy_output_dir: Path | None = None
    db_path: Path
    subjects: dict[str, Subject]
    scenarios: dict[str, Scenario]
    grid: GridConfig

    def path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def scenario_list(self) -> list[Scenario]:
        wanted = self.grid.scenarios
        if wanted is None:
            return [s for s in self.scenarios.values() if s.category == "core"]
        missing = [sid for sid in wanted if sid not in self.scenarios]
        if missing:
            raise ValueError(f"grid references unknown scenarios: {', '.join(missing)}")
        return [self.scenarios[sid] for sid in wanted]


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _index(items: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key = item.get("id")
        if not key:
            raise ValueError(f"{kind} entry without id: {item!r}")
        if key in out:
            raise ValueError(f"duplicate {kind} id: {key}")
        out[key] = item
    return out


def _merge_loras(
    root: Path,
    cfg_dir: Path,
    grid: GridConfig,
    use_local: bool,
    valid_subjects: set[str] | None = None,
) -> GridConfig:
    """Load LoRA entries from ``loras_file`` or ``config/local/loras.yaml``.

    The generated list is authoritative when the inline entries are missing or
    reference subjects that no longer exist. That makes a half-migrated config
    (example grid + discovered subjects) self-healing instead of a hard error.
    """

    source: Path | None = None
    if grid.loras_file:
        source = Path(grid.loras_file)
        if not source.is_absolute():
            source = root / source
    elif use_local:
        inline_valid = bool(grid.loras) and (
            valid_subjects is None
            or all(lora.subject in valid_subjects for lora in grid.loras)
        )
        if not inline_valid:
            candidate = cfg_dir / "local" / "loras.yaml"
            if candidate.exists():
                source = candidate

    if source is None or not source.exists():
        return grid

    raw = _read_yaml(source) or {}
    entries = [LoraEntry(**item) for item in raw.get("loras", [])]
    return grid.model_copy(update={"loras": entries})


def load_project(root: Path, *, use_local: bool = True) -> ProjectConfig:
    """Load a project.

    ``config/local/`` holds the real, uncommitted configuration. When a file is
    present there it wins over the committed example, so a fresh clone runs
    with the examples while a working installation uses its own data.
    """

    root = Path(root).resolve()
    cfg_dir = root / "config"

    defaults_dir = os.environ.get("LORA_LAB_DEFAULTS")
    defaults = Path(defaults_dir) if defaults_dir else None

    def cfg(name: str) -> Path:
        if use_local:
            local = cfg_dir / "local" / name
            if local.exists():
                return local
        primary = cfg_dir / name
        if primary.exists():
            return primary
        if defaults is not None and (defaults / name).exists():
            return defaults / name
        return primary

    project = _read_yaml(cfg("project.yaml")) or {}
    subjects_raw = _read_yaml(cfg("subjects.yaml")) or {}
    scenarios_raw = _read_yaml(cfg("scenarios.yaml")) or {}
    grid_raw = _read_yaml(cfg("grid.yaml")) or {}

    subjects = {
        key: Subject(**value)
        for key, value in _index(subjects_raw.get("subjects", []), "subject").items()
    }
    scenarios = {
        key: Scenario(**value)
        for key, value in _index(scenarios_raw.get("scenarios", []), "scenario").items()
    }
    grid = GridConfig(**grid_raw)
    grid = _merge_loras(root, cfg_dir, grid, use_local, set(subjects))

    if not grid.loras:
        raise ValueError("grid.loras is empty — add entries or point loras_file at a file")

    for lora in grid.loras:
        if lora.subject not in subjects:
            raise ValueError(
                f"lora {lora.id!r} references unknown subject {lora.subject!r} — "
                "run `lora-lab discover --apply` or fix config/local/subjects.yaml"
            )

    data_dir = Path(os.environ.get("LORA_LAB_DATA_DIR") or project.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    comfy_output = os.environ.get("COMFY_OUTPUT_DIR") or project.get("comfy_output_dir")
    comfy_output_dir = None
    if comfy_output:
        comfy_output_dir = Path(comfy_output)
        if not comfy_output_dir.is_absolute():
            comfy_output_dir = root / comfy_output_dir

    db_path = Path(os.environ.get("LORA_LAB_DB_PATH") or project.get("db_path", "data/lora-lab.sqlite"))
    if not db_path.is_absolute():
        db_path = root / db_path

    return ProjectConfig(
        root=root,
        data_dir=data_dir,
        comfyui_url=os.environ.get("COMFYUI_URL")
        or project.get("comfyui_url", "http://127.0.0.1:8188"),
        comfy_output_dir=comfy_output_dir,
        db_path=db_path,
        subjects=subjects,
        scenarios=scenarios,
        grid=grid,
    )
