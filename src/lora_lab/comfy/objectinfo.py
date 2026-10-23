"""Widget-column resolution for ComfyUI nodes.

ComfyUI's UI workflow format stores widget values positionally
(``widgets_values``) without names.  The authoritative name order lives in
``/object_info``.  When no server is reachable we fall back to a small table
covering the node classes used by the MOSS workflow.
"""

from __future__ import annotations

from typing import Any

WidgetColumn = tuple[str, str]

# Values that the ComfyUI frontend inserts after seed widgets and that are not
# node inputs in the API format.
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}

WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

FALLBACK: dict[str, list[WidgetColumn]] = {
    "UNETLoader": [("unet_name", "COMBO"), ("weight_dtype", "COMBO")],
    "CLIPLoader": [("clip_name", "COMBO"), ("type", "COMBO"), ("device", "COMBO")],
    "VAELoader": [("vae_name", "COMBO")],
    "LoraLoaderModelOnly": [("lora_name", "COMBO"), ("strength_model", "FLOAT")],
    "LoraLoader": [
        ("lora_name", "COMBO"),
        ("strength_model", "FLOAT"),
        ("strength_clip", "FLOAT"),
    ],
    "CLIPTextEncode": [("text", "STRING")],
    "EmptyFlux2LatentImage": [("width", "INT"), ("height", "INT"), ("batch_size", "INT")],
    "EmptyLatentImage": [("width", "INT"), ("height", "INT"), ("batch_size", "INT")],
    "KSampler": [
        ("seed", "INT"),
        ("steps", "INT"),
        ("cfg", "FLOAT"),
        ("sampler_name", "COMBO"),
        ("scheduler", "COMBO"),
        ("denoise", "FLOAT"),
    ],
    "KSamplerAdvanced": [
        ("add_noise", "COMBO"),
        ("noise_seed", "INT"),
        ("steps", "INT"),
        ("cfg", "FLOAT"),
        ("sampler_name", "COMBO"),
        ("scheduler", "COMBO"),
        ("start_at_step", "INT"),
        ("end_at_step", "INT"),
        ("return_with_leftover_noise", "COMBO"),
    ],
    "SaveImage": [("filename_prefix", "STRING")],
    "CheckpointLoaderSimple": [("ckpt_name", "COMBO")],
    "ImageScale": [
        ("upscale_method", "COMBO"),
        ("width", "INT"),
        ("height", "INT"),
        ("crop", "COMBO"),
    ],
}


def _decl_type(decl: Any) -> str | None:
    """Normalise a ComfyUI input declaration to a widget type."""

    if isinstance(decl, (list, tuple)):
        if not decl:
            return None
        head = decl[0]
        if isinstance(head, list):
            return "COMBO"
        if isinstance(head, str):
            return head
        if isinstance(head, dict):
            return head.get("type")
        return None
    if isinstance(decl, str):
        return decl
    if isinstance(decl, dict):
        return decl.get("type")
    return None


def widget_columns(class_type: str, object_info: dict[str, Any] | None) -> list[WidgetColumn]:
    """Ordered widget inputs for a node class."""

    spec = (object_info or {}).get(class_type)
    if spec:
        columns: list[WidgetColumn] = []
        inputs = spec.get("input") or {}
        for section in ("required", "optional"):
            for name, decl in (inputs.get(section) or {}).items():
                kind = _decl_type(decl)
                if kind in WIDGET_SCALARS:
                    columns.append((name, kind))
        if columns:
            return columns
    return FALLBACK.get(class_type, [])


def map_widget_values(
    class_type: str, values: list[Any], object_info: dict[str, Any] | None
) -> dict[str, Any]:
    """Map positional ``widgets_values`` onto named inputs."""

    columns = widget_columns(class_type, object_info)
    out: dict[str, Any] = {}
    index = 0
    for name, _kind in columns:
        if index >= len(values):
            break
        value = values[index]
        out[name] = value
        index += 1
        # The frontend adds a "control_after_generate" dropdown right after
        # seed/noise_seed. It is not an API input.
        if name in {"seed", "noise_seed"} and isinstance(value, int):
            if index < len(values) and values[index] in CONTROL_VALUES:
                index += 1
    return out
