"""Shared web application state."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config import ProjectConfig, load_project
from .runner import RunManager


class AppState:
    def __init__(
        self,
        project: ProjectConfig,
        reload_fn: Callable[[], ProjectConfig] | None = None,
    ) -> None:
        self.root: Path = project.root
        self.project: ProjectConfig = project
        self._reload_fn = reload_fn or (lambda: load_project(self.root))
        self.notices: list[str] = []
        self._notified_scenarios: set[str] = set()
        self._collect_notices(project)
        self.runner = RunManager(self)

    def reload(self) -> None:
        self.project = self._reload_fn()
        self._collect_notices(self.project)

    def _collect_notices(self, project: ProjectConfig) -> None:
        self._notified_scenarios &= set(project.removed_scenarios)
        fresh = [s for s in project.removed_scenarios if s not in self._notified_scenarios]
        if fresh:
            self._notified_scenarios.update(fresh)
            names = ", ".join(fresh)
            self.notices.append(
                f"Gelöschte Szenarien wurden aus dem geplanten Lauf entfernt: {names}"
            )

    def pop_notices(self) -> list[str]:
        out = list(self.notices)
        self.notices.clear()
        return out
