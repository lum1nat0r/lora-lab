"""ComfyUI workflow handling: UI -> API conversion, bindings, hashing.

The ComfyUI UI format stores links as a flat list and widget values
positionally.  The API format (what ``POST /prompt`` expects) is a mapping of
node id to ``{class_type, inputs}``.  This module does that conversion without
requiring the ComfyUI source tree, using ``/object_info`` when available and a
small fallback table otherwise.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .objectinfo import CONTROL_VALUES, map_widget_values, widget_columns

NON_NODE_TYPES = {"Note", "MarkdownNote"}


class WorkflowError(ValueError):
    pass


def load_workflow(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise WorkflowError(f"{path}: not a workflow object")
    return data


def is_ui_format(data: dict[str, Any]) -> bool:
    return "nodes" in data and isinstance(data.get("nodes"), list)


# --------------------------------------------------------------------------
# UI graph normalisation
# --------------------------------------------------------------------------


def _link_table(ui: dict[str, Any]) -> dict[int, tuple[int, int]]:
    table: dict[int, tuple[int, int]] = {}
    for link in ui.get("links") or []:
        if isinstance(link, dict):
            table[int(link["id"])] = (int(link["origin_id"]), int(link["origin_slot"]))
        else:
            table[int(link[0])] = (int(link[1]), int(link[2]))
    return table


def _bypass_mapping(
    node: dict[str, Any], links: dict[int, tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    """For a bypassed node, map output slots to the upstream link of the same type."""

    upstream: dict[str, tuple[int, int]] = {}
    for inp in node.get("inputs") or []:
        link_id = inp.get("link")
        kind = inp.get("type")
        if link_id is not None and kind and kind != "*" and int(link_id) in links:
            upstream.setdefault(kind, links[int(link_id)])

    mapping: dict[int, tuple[int, int]] = {}
    for slot, out in enumerate(node.get("outputs") or []):
        kind = out.get("type")
        if kind in upstream:
            mapping[slot] = upstream[kind]
    return mapping


def normalise_ui(ui: dict[str, Any]) -> dict[str, Any]:
    """Resolve bypassed/muted nodes in the UI graph.

    mode 0 = always, 2 = never (mute), 4 = bypass.
    Bypassed nodes are removed and their consumers are re-linked to the
    upstream source. Muted nodes are removed; a dependent link is an error.
    """

    ui = copy.deepcopy(ui)
    nodes = {int(n["id"]): n for n in ui.get("nodes") or []}
    links = _link_table(ui)

    bypassed: dict[int, dict[int, tuple[int, int]]] = {}
    bypassed_ids: set[int] = set()
    muted_ids: set[int] = set()
    for nid, node in nodes.items():
        mode = int(node.get("mode") or 0)
        if mode == 4:
            bypassed[nid] = _bypass_mapping(node, links)
            bypassed_ids.add(nid)
        elif mode == 2:
            muted_ids.add(nid)

    if not bypassed and not muted_ids:
        return ui

    # Rewrite every link whose origin was bypassed.
    for link in ui.get("links") or []:
        if isinstance(link, dict):
            origin = int(link["origin_id"])
            slot = int(link["origin_slot"])
            if origin in bypassed:
                if slot not in bypassed[origin]:
                    raise WorkflowError(
                        f"cannot bypass node {origin}: output {slot} has no matching input"
                    )
                link["origin_id"], link["origin_slot"] = bypassed[origin][slot]
        else:
            origin = int(link[1])
            slot = int(link[2])
            if origin in bypassed:
                if slot not in bypassed[origin]:
                    raise WorkflowError(
                        f"cannot bypass node {origin}: output {slot} has no matching input"
                    )
                link[1], link[2] = bypassed[origin][slot]

    ui["nodes"] = [
        n for n in ui["nodes"] if int(n["id"]) not in bypassed_ids | muted_ids
    ]
    return ui


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def ui_to_api(
    ui: dict[str, Any], object_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convert a ComfyUI UI workflow into API prompt format."""

    ui = normalise_ui(ui)
    links = _link_table(ui)
    api: dict[str, Any] = {}

    for node in ui.get("nodes") or []:
        class_type = node.get("type")
        if not class_type or class_type in NON_NODE_TYPES:
            continue
        nid = str(node["id"])
        inputs: dict[str, Any] = {}

        values = node.get("widgets_values")
        if isinstance(values, dict):
            inputs.update(values)
        elif isinstance(values, list):
            inputs.update(map_widget_values(class_type, values, object_info))

        for inp in node.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is None:
                continue
            if int(link_id) not in links:
                raise WorkflowError(f"node {nid}: dangling link {link_id}")
            origin, origin_slot = links[int(link_id)]
            inputs[inp["name"]] = [str(origin), origin_slot]

        api[nid] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": node.get("title") or class_type},
        }

    return api


def validate_prompt(api: dict[str, Any], require: set[str] | None = None) -> None:
    """Structural validation before a prompt is sent to ComfyUI."""

    if not api:
        raise WorkflowError("prompt is empty")
    if require:
        missing = sorted(require - set(api))
        if missing:
            raise WorkflowError(f"prompt is missing bound nodes: {', '.join(missing)}")
    for nid, node in api.items():
        for name, value in (node.get("inputs") or {}).items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and value[0].isdigit()
            ):
                target = value[0]
                if target not in api:
                    raise WorkflowError(
                        f"node {nid}.{name} references missing node {target}"
                    )
            elif value is None:
                raise WorkflowError(f"node {nid}.{name} has no value")


def drop_nodes(api: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    """Remove nodes after bindings have replaced every reference to them."""

    if not ids:
        return api
    removed = {str(i) for i in ids}
    out = {k: v for k, v in api.items() if k not in removed}
    for nid, node in out.items():
        for name, value in (node.get("inputs") or {}).items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and value[0].isdigit()
            ):
                if value[0] in removed:
                    raise WorkflowError(
                        f"node {nid}.{name} still references dropped node {value[0]}"
                    )
    return out


def bypass_lora(api: dict[str, Any], node_id: int, model_input: str = "model") -> dict[str, Any]:
    """Remove the LoRA node and reconnect its model input to its consumers."""

    key = str(node_id)
    if key not in api:
        raise WorkflowError(f"lora node {node_id} not found in prompt")
    upstream = api[key]["inputs"].get(model_input)
    if not (isinstance(upstream, list) and len(upstream) == 2):
        raise WorkflowError(f"lora node {node_id}.{model_input} is not a link")

    out: dict[str, Any] = {}
    for nid, node in api.items():
        if nid == key:
            continue
        node = copy.deepcopy(node)
        for name, value in (node.get("inputs") or {}).items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and value[0].isdigit()
                and value[0] == key
            ):
                node["inputs"][name] = list(upstream)
        out[nid] = node
    return out


def render_prompt(
    api: dict[str, Any],
    *,
    bindings: Any,
    positive: str,
    negative: str,
    seed: int,
    filename_prefix: str,
    lora_file: str | None,
    lora_strength: float | None,
    width: int | None = None,
    height: int | None = None,
    overrides: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply per-job values to a converted workflow."""

    prompt = copy.deepcopy(api)

    def set_input(node_id: int, input_name: str, value: Any) -> None:
        key = str(node_id)
        if key not in prompt:
            raise WorkflowError(f"bound node {node_id} is not in the prompt")
        prompt[key].setdefault("inputs", {})[input_name] = value

    set_input(bindings.positive.id, bindings.positive.input, positive)
    set_input(bindings.negative.id, bindings.negative.input, negative)
    set_input(bindings.seed.id, bindings.seed.input, int(seed))
    set_input(bindings.save.id, bindings.filename_prefix_input, filename_prefix)

    if lora_file is not None:
        set_input(bindings.lora_node, bindings.lora_file_input, lora_file)
        if lora_strength is not None:
            set_input(bindings.lora_node, bindings.lora_strength_input, float(lora_strength))

    if bindings.size_node is not None:
        w_name, h_name = bindings.size_inputs
        if width is not None:
            set_input(bindings.size_node, w_name, int(width))
        if height is not None:
            set_input(bindings.size_node, h_name, int(height))

    for node_id, patch in (overrides or {}).items():
        for name, value in patch.items():
            set_input(int(node_id), name, value)

    return prompt


def prompt_hash(api: dict[str, Any], bindings: Any) -> str:
    """Content hash of everything that affects the rendered image.

    The save prefix is metadata and is normalised away, so re-running the same
    generation with a different output folder is still recognised.
    """

    canonical = copy.deepcopy(api)
    canonical.pop(str(bindings.save.id), None)
    for node in canonical.values():
        node.pop("_meta", None)
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def required_nodes(bindings: Any) -> set[str]:
    return {
        str(bindings.lora_node),
        str(bindings.positive.id),
        str(bindings.negative.id),
        str(bindings.seed.id),
        str(bindings.save.id),
    }


def describe(api: dict[str, Any]) -> list[str]:
    """Human-readable node summary, used by ``doctor``."""

    lines = []
    for nid, node in sorted(api.items(), key=lambda kv: int(kv[0])):
        inputs = node.get("inputs") or {}
        linked = sum(1 for v in inputs.values() if isinstance(v, list))
        lines.append(
            f"{nid:>4}  {node['class_type']:<28} {len(inputs) - linked:>2} values, "
            f"{linked:>2} links"
        )
    return lines


# Small helper used by tests and doctor output.
def widget_names(class_type: str, object_info: dict[str, Any] | None = None) -> list[str]:
    return [name for name, _ in widget_columns(class_type, object_info)]


__all__ = [
    "CONTROL_VALUES",
    "WorkflowError",
    "bypass_lora",
    "describe",
    "drop_nodes",
    "is_ui_format",
    "load_workflow",
    "normalise_ui",
    "prompt_hash",
    "render_prompt",
    "required_nodes",
    "ui_to_api",
    "validate_prompt",
    "widget_names",
]
