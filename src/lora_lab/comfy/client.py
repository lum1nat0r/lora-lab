"""Minimal ComfyUI HTTP client (queue, history, view, free)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ComfyClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- endpoints ---------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        try:
            response = self._client.get("/system_stats")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyError(f"ComfyUI not reachable at {self.base_url}: {exc}") from exc
        return response.json()

    def object_info(self) -> dict[str, Any]:
        response = self._client.get("/object_info")
        response.raise_for_status()
        return response.json()

    def queue_prompt(self, prompt: dict[str, Any]) -> str:
        payload = {"prompt": prompt, "client_id": self.client_id}
        response = self._client.post("/prompt", json=payload)
        if response.status_code >= 400:
            raise ComfyError(_format_error(response))
        data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI did not return a prompt_id: {data}")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        response = self._client.get(f"/history/{prompt_id}")
        response.raise_for_status()
        data = response.json()
        entry = data.get(prompt_id)
        return entry if entry else None

    def wait(
        self,
        prompt_id: str,
        *,
        poll: float = 1.5,
        timeout: float = 1800.0,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if should_cancel is not None and should_cancel():
                self.interrupt()
                raise ComfyError("cancelled")
            entry = self.history(prompt_id)
            if entry:
                status = entry.get("status") or {}
                state = status.get("status_str")
                if state == "error":
                    raise ComfyError(f"prompt {prompt_id} failed: {_status_messages(status)}")
                if state == "success" or entry.get("outputs"):
                    return entry
            if should_cancel is not None and should_cancel():
                self.interrupt()
                raise ComfyError("cancelled")
            time.sleep(poll)
        raise ComfyError(f"prompt {prompt_id} timed out after {timeout:.0f}s")

    def interrupt(self) -> None:
        """Ask ComfyUI to abort the currently running prompt."""

        try:
            self._client.post("/interrupt")
        except httpx.HTTPError:
            pass

    def view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": type_}
        response = self._client.get("/view", params=params)
        response.raise_for_status()
        return response.content

    def free(self, unload_models: bool = True) -> None:
        """Ask ComfyUI to release VRAM before the next phase."""

        self._client.post(
            "/free",
            json={"unload_models": unload_models, "free_memory": True},
        )


def _format_error(response: httpx.Response) -> str:
    try:
        data = response.json()
    except json.JSONDecodeError:
        return f"HTTP {response.status_code}: {response.text[:400]}"
    parts = [f"HTTP {response.status_code}"]
    if data.get("error"):
        err = data["error"]
        if isinstance(err, dict):
            parts.append(str(err.get("message") or err))
        else:
            parts.append(str(err))
    node_errors = data.get("node_errors") or {}
    for node_id, info in node_errors.items():
        errors = info.get("errors") if isinstance(info, dict) else None
        if errors:
            for item in errors:
                parts.append(f"node {node_id}: {item.get('message')} {item.get('details') or ''}".strip())
    return " | ".join(parts)


def _status_messages(status: dict[str, Any]) -> str:
    messages = []
    for item in status.get("messages") or []:
        if isinstance(item, list) and len(item) == 2 and item[0] == "execution_error":
            detail = item[1] or {}
            messages.append(
                f"node {detail.get('node_id')} {detail.get('node_type')}: "
                f"{detail.get('exception_message')}"
            )
    return "; ".join(messages) or "unknown error"
