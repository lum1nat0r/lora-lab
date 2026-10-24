"""Aggregation and ranking statistics.

Everything here is pure Python so it can be tested without models or a GPU.

The important choices:

* a per-image **utility** is a weighted mean over the axes that exist; missing
  axes are excluded and the weights renormalised, never treated as zero
* a LoRA is scored **per scenario first**, then averaged over scenarios with
  equal weight, so a heavily sampled scenario cannot dominate
* the ranking uses a **cluster bootstrap lower bound**, resampling scenarios
  and, inside them, seeds
* **robustness** is the 20th percentile of scenario scores, which penalises a
  LoRA that only works in one setting
* **rank stability** perturbs the axis weights and reports how often a LoRA
  keeps its rank; a 0.03-point win over a flat distribution is not a win
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

AXES = ("identity", "adherence", "quality")

DEFAULT_WEIGHTS = {"identity": 0.45, "adherence": 0.35, "quality": 0.20}

DEFAULT_THRESHOLDS = {
    "pass_identity": 0.60,
    "pass_adherence": 0.80,
    "pass_quality": 0.30,
    "fail_identity": 0.20,
}


@dataclass(frozen=True)
class Sample:
    lora_id: str
    subject: str
    scenario: str
    seed: int
    identity: float | None = None
    adherence: float | None = None
    quality: float | None = None
    critical: bool = False
    epoch: int = 0
    strength: float = 0.0
    image_id: int | None = None
    path: str = ""
    adherence_passed: tuple[str, ...] = ()
    adherence_failed: tuple[str, ...] = ()
    adherence_missing: tuple[str, ...] = ()
    leakage: float | None = None
    embedding: tuple[float, ...] | None = None
    defect_index: float | None = None


@dataclass
class LoraResult:
    lora_id: str
    subject: str
    epoch: int
    strength: float
    n: int
    mean: float
    lcb: float
    ucb: float
    robustness: float
    stability: float | None
    yield_rate: float
    fail_rate: float
    scenarios: int
    per_scenario: dict[str, float] = field(default_factory=dict)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return math.fsum(items) / len(items) if items else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def weighted(values: dict[str, float], weights: dict[str, float]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for axis, weight in weights.items():
        value = values.get(axis)
        if value is None:
            continue
        total += weight * value
        weight_sum += weight
    if weight_sum == 0:
        return None
    return total / weight_sum


def image_utility(sample: Sample, weights: dict[str, float] | None = None) -> float | None:
    return weighted(
        {"identity": sample.identity, "adherence": sample.adherence, "quality": sample.quality},
        weights or DEFAULT_WEIGHTS,
    )


def is_pass(sample: Sample, thresholds: dict[str, float] | None = None) -> bool:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if sample.critical:
        return False
    judged = False
    if sample.identity is not None:
        judged = True
        if sample.identity < thresholds["pass_identity"]:
            return False
    if sample.adherence is not None:
        judged = True
        if sample.adherence < thresholds["pass_adherence"]:
            return False
    if sample.quality is not None:
        judged = True
        if sample.quality < thresholds["pass_quality"]:
            return False
    return judged


def is_fail(sample: Sample, thresholds: dict[str, float] | None = None) -> bool:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if sample.critical:
        return True
    return sample.identity is not None and sample.identity < thresholds["fail_identity"]


def cells(samples: Iterable[Sample]) -> dict[tuple[str, int], list[float]]:
    out: dict[tuple[str, int], list[float]] = {}
    for sample in samples:
        utility = image_utility(sample)
        if utility is None:
            continue
        out.setdefault((sample.scenario, sample.seed), []).append(utility)
    return out


def scenario_scores(samples: Iterable[Sample]) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for (scenario, _seed), values in cells(samples).items():
        per.setdefault(scenario, []).append(mean(values))
    return {scenario: mean(values) for scenario, values in per.items()}


def stability(samples: Iterable[Sample]) -> float | None:
    """Seed-to-seed consistency, averaged over scenarios.

    The spread is taken across seed means inside a scenario: a LoRA that only
    works on two of four seeds is unstable even if its average looks fine.
    """

    per_scenario: dict[str, list[float]] = {}
    for (scenario, _seed), values in cells(samples).items():
        per_scenario.setdefault(scenario, []).append(mean(values))

    spreads = []
    for values in per_scenario.values():
        if len(values) < 2:
            continue
        avg = mean(values)
        variance = math.fsum((v - avg) ** 2 for v in values) / (len(values) - 1)
        spreads.append(math.sqrt(variance))
    if not spreads:
        return None
    return max(0.0, 1.0 - mean(min(sd / 0.5, 1.0) for sd in spreads))


def cluster_bootstrap_interval(
    samples: list[Sample],
    *,
    n: int = 2000,
    seed: int = 7,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Two-level bootstrap: scenarios first, then seeds inside each scenario."""

    cell_map = cells(samples)
    if not cell_map:
        return (0.0, 0.0)

    by_scenario: dict[str, list[list[float]]] = {}
    for (scenario, _seed), values in cell_map.items():
        by_scenario.setdefault(scenario, []).append(values)
    scenario_names = sorted(by_scenario)
    if len(scenario_names) < 2:
        flat = [mean(v) for v in cell_map.values()]
        return (percentile(flat, alpha / 2), percentile(flat, 1 - alpha / 2))

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n):
        chosen = [rng.choice(scenario_names) for _ in scenario_names]
        per_draw = []
        for name in chosen:
            group = by_scenario[name]
            per_draw.append(mean(rng.choice(group)))
        draws.append(mean(per_draw))

    return (percentile(draws, alpha / 2), percentile(draws, 1 - alpha / 2))


def analyse_lora(
    samples: list[Sample],
    *,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    bootstrap: int = 2000,
    seed: int = 7,
) -> LoraResult | None:
    usable = [s for s in samples if image_utility(s, weights) is not None]
    if not usable:
        return None

    per_scenario = scenario_scores(usable)
    ordered = sorted(per_scenario.values())
    overall = mean(ordered)
    robustness = percentile(ordered, 0.20)
    low, high = cluster_bootstrap_interval(usable, n=bootstrap, seed=seed)

    first = usable[0]
    return LoraResult(
        lora_id=first.lora_id,
        subject=first.subject,
        epoch=first.epoch,
        strength=first.strength,
        n=len(usable),
        mean=overall,
        lcb=low,
        ucb=high,
        robustness=robustness,
        stability=stability(usable),
        yield_rate=mean(1.0 if is_pass(s, thresholds) else 0.0 for s in usable),
        fail_rate=mean(1.0 if is_fail(s, thresholds) else 0.0 for s in usable),
        scenarios=len(per_scenario),
        per_scenario=per_scenario,
    )


def analyse_all(
    samples: list[Sample],
    *,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    bootstrap: int = 2000,
    seed: int = 7,
) -> list[LoraResult]:
    grouped: dict[tuple[str, int, float], list[Sample]] = {}
    for sample in samples:
        key = (sample.lora_id, sample.epoch, sample.strength)
        grouped.setdefault(key, []).append(sample)

    results = []
    for index, key in enumerate(sorted(grouped)):
        group = grouped[key]
        result = analyse_lora(
            group,
            weights=weights,
            thresholds=thresholds,
            bootstrap=bootstrap,
            seed=seed + index,
        )
        if result is not None:
            results.append(result)
    results.sort(key=lambda r: (-r.lcb, -r.mean))
    return results


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Non-dominated points, maximising both coordinates."""

    frontier = []
    for name, (x, y) in points.items():
        dominated = any(
            (ox >= x and oy >= y) and (ox > x or oy > y)
            for other, (ox, oy) in points.items()
            if other != name
        )
        if not dominated:
            frontier.append(name)
    return sorted(frontier, key=lambda name: points[name][0], reverse=True)


def rank_stability(
    components: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
    *,
    draws: int = 1000,
    concentration: float = 40.0,
    seed: int = 11,
) -> dict[str, dict[float, float]]:
    """Rank distribution under Dirichlet-perturbed weights."""

    base = dict(weights or DEFAULT_WEIGHTS)
    names = sorted(components)
    if not names:
        return {}

    rng = random.Random(seed)
    counts: dict[str, dict[float, float]] = {name: {} for name in names}

    for _ in range(draws):
        alphas = [max(base.get(axis, 0.0), 1e-9) * concentration for axis in AXES]
        raw = [rng.gammavariate(a, 1.0) for a in alphas]
        total = sum(raw)
        perturbed = {axis: value / total for axis, value in zip(AXES, raw, strict=True)}

        scored = []
        for name in names:
            value = weighted(components[name], perturbed)
            if value is not None:
                scored.append((name, value))
        scored.sort(key=lambda item: -item[1])

        position = 0
        while position < len(scored):
            end = position
            while end + 1 < len(scored) and math.isclose(
                scored[end + 1][1], scored[position][1], rel_tol=1e-12, abs_tol=1e-12
            ):
                end += 1
            rank = (position + end) / 2 + 1
            for index in range(position, end + 1):
                entry = counts[scored[index][0]]
                entry[rank] = entry.get(rank, 0.0) + 1.0
            position = end + 1

    return {
        name: {float(rank): count / draws for rank, count in sorted(ranks.items())}
        for name, ranks in counts.items()
    }
