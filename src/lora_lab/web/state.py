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
        self.runner = RunManager(self)

    def reload(self) -> None:
        self.project = self._reload_fn()
