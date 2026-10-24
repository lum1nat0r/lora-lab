"""Attribute judging through a local vision-language model.

The judge speaks the OpenAI chat-completions dialect, which covers vLLM,
llama.cpp server, Ollama, LM Studio and most local runtimes:

    lora-lab score --vlm-url http://gpu-box:8000/v1 --vlm-model qwen3-vl-8b

One request per image, all attributes in one JSON object.  Malformed answers
and refusals are counted, not raised: a single bad image should not abort a
nightly scoring run.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .base import ImageRef

SYSTEM = (
    "You are a strict visual inspector. You answer only with a JSON object and "
    "never explain, apologise or add prose."
)

INSTRUCTIONS = """Inspect the image and answer each question with true, false or null.
Return only a JSON object with exactly these keys. Use null when the image does
not let you decide.

Questions:
{questions}
"""


@dataclass
class VLMSettings:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 120.0
    max_tokens: int = 400
    temperature: float = 0.0
    max_side: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def models_endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        return f"{base}/models"


class VLMJudge:
    name = "vlm"
    version = "v1"

    def __init__(self, settings: VLMSettings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = client or httpx.Client(timeout=settings.timeout, headers=headers)
        self.stats = {"requests": 0, "parsed": 0, "refused": 0, "errors": 0}
        # scoring runs in a thread pool; plain += can lose counts
        self._stats_lock = threading.Lock()

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] += 1

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VLMJudge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ping(self) -> dict[str, Any]:
        """Verify the VLM endpoint responds and check whether the configured model is listed."""

        try:
            response = self._client.get(self.settings.models_endpoint())
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc), "models": [], "matched": False}

        raw_models = payload.get("data") if isinstance(payload, dict) else None
        models: list[str] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
        wanted = self.settings.model.strip()
        matched = not models or any(
            m == wanted or m.startswith(wanted) or wanted in m for m in models
        )
        return {
            "ok": True,
            "model": wanted,
            "models": models,
            "matched": matched,
        }

    # -- protocol ----------------------------------------------------------
    def judge(self, image: ImageRef, attributes: Sequence[Any]) -> dict[str, bool | None]:
        if not attributes:
            return {}
        questions = "\n".join(f"- {a.id}: {a.question}" for a in attributes)
        body = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTIONS.format(questions=questions)},
                        {"type": "image_url", "image_url": {"url": self._data_url(image.path)}},
                    ],
                },
            ],
            **self.settings.extra_body,
        }

        self._bump("requests")
        payload = None
        for attempt in (0, 1):
            try:
                response = self._client.post(self.settings.endpoint(), json=body)
                response.raise_for_status()
                payload = response.json()
                break
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                # one retry on transient faults; a 4xx is the server's verdict
                transient = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code >= 500
                )
                if attempt == 0 and transient:
                    time.sleep(1.0)
                    continue
                self._bump("errors")
                return {a.id: None for a in attributes}

        content = _message_text(payload)
        parsed = _parse_json(content)
        if parsed is None:
            if _looks_like_refusal(content):
                self._bump("refused")
            else:
                self._bump("errors")
            return {a.id: None for a in attributes}

        self._bump("parsed")
        return {a.id: _coerce(parsed.get(a.id)) for a in attributes}

    def _data_url(self, path: str) -> str:
        data = Path(path).read_bytes()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        if self.settings.max_side:
            data, mime = _resize(data, self.settings.max_side)
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _message_text(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def _parse_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    markers = ("i can't", "i cannot", "i'm sorry", "i am sorry", "unable to", "not able to")
    return any(marker in lowered for marker in markers)


def _coerce(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _resize(data: bytes, max_side: int) -> tuple[bytes, str]:  # pragma: no cover - optional
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buffer = BytesIO()
        im.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), "image/jpeg"
