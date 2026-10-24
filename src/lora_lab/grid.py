"""Grid expansion: turn the config into concrete render jobs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .config import GridConfig, Scenario, Subject


@dataclass(frozen=True, order=True)
class Job:
    subject: str
    lora_id: str
    epoch: int
    strength: float
    scenario: str
    seed: int
    condition: str = "lora"
    profile: str = "fast"
    lora_file: str = ""

    @property
    def is_baseline(self) -> bool:
        return self.condition == "base_only"

    def prefix(self) -> str:
        """SaveImage filename prefix; also the on-disk path below images/."""

        strength = f"{self.strength:g}"
        if self.is_baseline:
            return f"{self.profile}/{self.subject}/_base/s{strength}/{self.scenario}/{self.seed}"
        return (
            f"{self.profile}/{self.subject}/{self.lora_id}/ep{self.epoch}"
            f"/s{strength}/{self.scenario}/{self.seed}"
        )

    def label(self) -> str:
        if self.is_baseline:
            return f"{self.subject} · baseline · {self.scenario} · seed {self.seed}"
        return (
            f"{self.subject} · {self.lora_id} · ep{self.epoch} · s{self.strength:g}"
            f" · {self.scenario} · seed {self.seed}"
        )


def expand(
    grid: GridConfig,
    scenarios: Iterable[Scenario],
    subjects: dict[str, Subject],
) -> list[Job]:
    scenario_ids = [s.id for s in scenarios]
    jobs: list[Job] = []

    for condition in grid.conditions:
        if condition == "base_only":
            for subject in sorted({lora.subject for lora in grid.loras}):
                if subject not in subjects:
                    continue
                for scenario in scenario_ids:
                    for seed in grid.seeds:
                        jobs.append(
                            Job(
                                subject=subject,
                                lora_id="",
                                epoch=0,
                                strength=0.0,
                                scenario=scenario,
                                seed=seed,
                                condition=condition,
                                profile=grid.profile,
                            )
                        )
            continue

        for lora in grid.loras:
            strengths = lora.strengths if lora.strengths is not None else grid.strengths
            for strength in strengths:
                for scenario in scenario_ids:
                    for seed in grid.seeds:
                        jobs.append(
                            Job(
                                subject=lora.subject,
                                lora_id=lora.id,
                                epoch=lora.epoch,
                                strength=float(strength),
                                scenario=scenario,
                                seed=int(seed),
                                condition=condition,
                                profile=grid.profile,
                                lora_file=lora.file,
                            )
                        )

    jobs.sort()
    return jobs


def summary(jobs: Iterable[Job]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.condition] = counts.get(job.condition, 0) + 1
    counts["total"] = sum(counts.values())
    return counts
