"""Read training metadata out of ``.safetensors`` files.

Fizgig (like kohya and most other trainers) stores the training parameters in
the safetensors JSON header under ``__metadata__`` as ``ss_*`` keys.  No torch
or safetensors package is needed: the format is an 8-byte little-endian
header length followed by a JSON document.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

# Friendly name -> list of keys trainers have used.
FIELDS: dict[str, tuple[str, ...]] = {
    "network_module": ("ss_network_module",),
    "network_dim": ("ss_network_dim",),
    "network_alpha": ("ss_network_alpha",),
    "learning_rate": ("ss_learning_rate", "ss_unet_lr"),
    "text_encoder_lr": ("ss_text_encoder_lr",),
    "optimizer": ("ss_optimizer", "ss_optimizer_type"),
    "lr_scheduler": ("ss_lr_scheduler",),
    "epochs": ("ss_epoch", "ss_num_epochs"),
    "max_train_steps": ("ss_max_train_steps",),
    "batch_size": ("ss_batch_size_per_device",),
    "gradient_accumulation": ("ss_gradient_accumulation_steps",),
    "resolution": ("ss_resolution",),
    "seed": ("ss_seed",),
    "trigger_word": ("ss_trigger_word",),
    "base_model": ("ss_base_model_version", "ss_sd_model_name"),
    "num_train_images": ("ss_num_train_images",),
    "num_reg_images": ("ss_num_reg_images",),
    "num_batches_per_epoch": ("ss_num_batches_per_epoch",),
    "bucket_info": ("ss_bucket_info",),
    "min_bucket_reso": ("ss_min_bucket_reso",),
    "max_bucket_reso": ("ss_max_bucket_reso",),
    "dataset_dirs": ("ss_dataset_dirs",),
    "tag_frequency": ("ss_tag_frequency",),
    "network_args": ("ss_network_args",),
    "mixed_precision": ("ss_mixed_precision",),
    "noise_offset": ("ss_noise_offset",),
    "clip_skip": ("ss_clip_skip",),
    "keep_tokens": ("ss_keep_tokens",),
}

JSON_FIELDS = {"bucket_info", "dataset_dirs", "tag_frequency", "network_args"}


class SafetensorsError(ValueError):
    pass


def read_header(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise SafetensorsError(f"{path}: file is too short")
        (length,) = struct.unpack("<Q", prefix)
        if length <= 0 or length > 200 * 1024 * 1024:
            raise SafetensorsError(f"{path}: implausible header length {length}")
        raw = fh.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SafetensorsError(f"{path}: header is not valid JSON") from exc


def read_metadata(path: Path) -> dict[str, str]:
    header = read_header(path)
    meta = header.get("__metadata__") or {}
    return {str(k): str(v) for k, v in meta.items()}


def _coerce(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return value


def normalise(meta: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, keys in FIELDS.items():
        for key in keys:
            if key in meta and meta[key] != "":
                raw = meta[key]
                if name in JSON_FIELDS:
                    try:
                        out[name] = json.loads(raw)
                    except json.JSONDecodeError:
                        out[name] = raw
                else:
                    out[name] = _coerce(raw)
                break
    return out


def epoch_from_filename(path: Path) -> int | None:
    match = re.search(r"-(\d{6})(?:-state)?\.safetensors$", Path(path).name)
    return int(match.group(1)) if match else None


def import_lora(path: Path) -> dict[str, Any]:
    path = Path(path)
    meta = read_metadata(path)
    record = normalise(meta)
    record["file"] = path.name
    record["epoch_from_filename"] = epoch_from_filename(path)
    record["metadata"] = meta
    return record
