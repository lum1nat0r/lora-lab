"""Background generation and scoring started from the web UI."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any

from ..comfy.client import ComfyClient
from ..config import load_project
from ..db import Store
from ..grid import expand
from ..runner import run_jobs


class Cancelled(Exception):
    pass


class RunManager:
    """One background job at a time; the web UI polls :meth:`snapshot`."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.cancel = threading.Event()
        self.kind: str | None = None
        self.error: str | None = None
        self.log: deque[str] = deque(maxlen=400)
        self.progress = {"done": 0, "total": 0}
        self.started_at: float | None = None
        self.finished_at: float | None = None

    # -- lifecycle ---------------------------------------------------------
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, kind: str, target, *args) -> None:
        with self._lock:
            if self.running():
                raise RuntimeError("a run is already in progress")
            self.cancel.clear()
            self.kind = kind
            self.error = None
            self.log.clear()
            self.progress = {"done": 0, "total": 0}
            self.started_at = time.time()
            self.finished_at = None
            self._thread = threading.Thread(
                target=self._wrap, args=(target, *args), daemon=True, name=f"lora-lab-{kind}"
            )
            self._thread.start()

    def stop(self) -> None:
        self.cancel.set()
        self._append("stop requested")

    def _wrap(self, target, *args) -> None:
        try:
            target(*args)
        except Cancelled:
            self.error = "cancelled"
            self._append("cancelled")
        except Exception as exc:  # surface any failure in the UI
            self.error = str(exc)
            self._append(f"ERROR {exc}")
        finally:
            self.finished_at = time.time()

    def _append(self, message: str) -> None:
        self.log.append(message)

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.running(),
            "kind": self.kind,
            "error": self.error,
            "progress": dict(self.progress),
            "log": list(self.log)[-120:],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    # -- jobs --------------------------------------------------------------
    def generate(self, limit: int | None = None) -> None:
        project = load_project(self.state.root)
        jobs = expand(project.grid, project.scenario_list(), project.subjects)
        if limit:
            jobs = jobs[:limit]
        self.progress["total"] = len(jobs)
        self._append(f"generating {len(jobs)} images via {project.comfyui_url}")

        store = Store(project.db_path)
        run_id = store.upsert_run(
            project.grid.name,
            "ui",
            project.grid.profile,
            {"jobs": len(jobs), "source": "ui"},
        )
        client = ComfyClient(project.comfyui_url)
        try:
            object_info = client.object_info()

            def on_progress(message: str) -> None:
                self.log.append(message)
                if message.startswith(("done", "skip", "FAIL")):
                    self.progress["done"] += 1
                self._check_cancel()

            stats = run_jobs(
                project,
                store,
                jobs,
                client,
                run_id=run_id,
                limit=None,
                on_progress=on_progress,
                should_cancel=self.cancel.is_set,
                object_info=object_info,
            )
            self._check_cancel()
            self._append(
                f"finished: {stats.done} done · {stats.skipped} skipped · {stats.failed} failed"
            )
        finally:
            try:
                client.free()
            except Exception:
                pass
            client.close()
            store.close()

    def score(
        self,
        only_missing: bool = True,
        lora_id: str | None = None,
        epoch: int | None = None,
        strength: float | None = None,
        run_id: int | None = None,
    ) -> None:
        from ..scorers.run import build_default_scorers, score_run

        project = load_project(self.state.root)
        vlm_url = project.vlm_url or os.environ.get("VLM_JUDGE_URL")
        vlm_model = project.vlm_model or os.environ.get("VLM_JUDGE_MODEL")
        vlm_api_key = project.vlm_api_key or os.environ.get("VLM_API_KEY") or None
        judge = None
        if vlm_url and vlm_model:
            from ..scorers.vlm import VLMJudge, VLMSettings

            judge = VLMJudge(
                VLMSettings(base_url=vlm_url, model=vlm_model, api_key=vlm_api_key)
            )
            self._append(f"adherence judge: {vlm_model} at {vlm_url}")
        else:
            self._append("no VLM configured (VLM_JUDGE_URL / VLM_JUDGE_MODEL) — quality only")

        scorers, identity_error = build_default_scorers(project, judge=judge, log=self._append)
        if identity_error is not None:
            self._append(f"identity skipped: {identity_error}")

        def on_progress(message: str) -> None:
            self.log.append(message)
            self.progress["done"] += 1
            self._check_cancel()

        store = Store(project.db_path)
        try:
            self.progress["total"] = len(
                store.images_for_run(
                    run_id, lora_id=lora_id, epoch=epoch, strength=strength
                )
            )
            counts = score_run(
                project,
                store,
                scorers,
                run_id=run_id,
                lora_id=lora_id,
                epoch=epoch,
                strength=strength,
                skip_existing=only_missing,
                on_progress=on_progress,
                workers=4,
            )
            self._append(" · ".join(f"{axis} {value}" for axis, value in sorted(counts.items())))
        finally:
            store.close()
            if judge is not None:
                stats = judge.stats
                self._append(
                    f"vlm: {stats['parsed']}/{stats['requests']} parsed · "
                    f"{stats['refused']} refused · {stats['errors']} errors"
                )
                judge.close()
