"""Blind A/B acceptance: pair selection and Bradley-Terry strengths.

Two images are only compared when they share subject, scenario and seed, so a
vote measures the operating point and not the seed. Pairs of operating points
with the fewest votes are asked first.

Strengths are fitted with the MM algorithm (Hunter 2004). Every operating
point plays ``prior`` virtual wins and losses against an average opponent,
which keeps unbeaten or winless points finite and pulls thin evidence towards
the middle. Ties count as half a win for each side.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Any

from .stats import percentile


def op_of(lora_id: str, subject: str, epoch: int, strength: float) -> str:
    return f"{lora_id or f'{subject}:base'}@ep{epoch}@s{float(strength):g}"


def pick_pair(
    rows: Iterable[Any],
    votes: Sequence[tuple[str, str, str]],
    allowed: set[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[Any, Any] | None:
    """Two image rows of the least-compared pair of operating points, in random order."""

    rng = rng or random.Random()
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        op = op_of(row["lora_id"], row["subject"], row["epoch"], row["strength"])
        if allowed is not None and op not in allowed:
            continue
        cells.setdefault((row["subject"], row["scenario"], int(row["seed"])), {}).setdefault(op, row)

    counts: dict[frozenset[str], int] = {}
    for a, b, _ in votes:
        key = frozenset((a, b))
        counts[key] = counts.get(key, 0) + 1

    best: list[tuple[Any, Any]] = []
    fewest = math.inf
    for by_op in cells.values():
        ops = sorted(by_op)
        for i, a in enumerate(ops):
            for b in ops[i + 1 :]:
                seen = counts.get(frozenset((a, b)), 0)
                if seen < fewest:
                    fewest, best = seen, []
                if seen == fewest:
                    best.append((by_op[a], by_op[b]))
    if not best:
        return None
    left, right = rng.choice(best)
    return (left, right) if rng.random() < 0.5 else (right, left)


def bradley_terry(
    votes: Sequence[tuple[str, str, str]],
    *,
    prior: float = 1.0,
    iterations: int = 200,
    tol: float = 1e-7,
) -> dict[str, float]:
    """Strength per operating point, normalised to a geometric mean of 1.

    ``votes`` are ``(left_op, right_op, winner)`` with winner in
    ``left``/``right``/``tie``.
    """

    wins: dict[str, float] = {}
    games: dict[tuple[str, str], int] = {}
    for a, b, winner in votes:
        if a == b:
            continue
        wins.setdefault(a, 0.0)
        wins.setdefault(b, 0.0)
        if winner == "left":
            wins[a] += 1.0
        elif winner == "right":
            wins[b] += 1.0
        else:
            wins[a] += 0.5
            wins[b] += 0.5
        key = (a, b) if a < b else (b, a)
        games[key] = games.get(key, 0) + 1
    if not wins:
        return {}

    opponents: dict[str, list[tuple[str, int]]] = {name: [] for name in wins}
    for (a, b), n in games.items():
        opponents[a].append((b, n))
        opponents[b].append((a, n))

    strength = dict.fromkeys(wins, 1.0)
    for _ in range(iterations):
        updated = {}
        for name, p in strength.items():
            denom = sum(n / (p + strength[other]) for other, n in opponents[name])
            denom += 2 * prior / (p + 1.0)
            updated[name] = (wins[name] + prior) / denom
        change = max(abs(updated[k] - strength[k]) for k in strength)
        strength = updated
        if change < tol:
            break

    log_mean = sum(math.log(p) for p in strength.values()) / len(strength)
    scale = math.exp(log_mean)
    return {name: p / scale for name, p in strength.items()}


def standings(
    votes: Sequence[tuple[str, str, str]],
    *,
    bootstrap: int = 200,
    seed: int = 13,
    alpha: float = 0.1,
) -> list[dict]:
    """Ranked operating points with P(beats an average point) and a bootstrap interval."""

    fitted = bradley_terry(votes)
    if not fitted:
        return []

    tally = {name: {"n": 0, "wins": 0, "ties": 0, "losses": 0} for name in fitted}
    for a, b, winner in votes:
        if a == b:
            continue
        for side, name in (("left", a), ("right", b)):
            tally[name]["n"] += 1
            if winner == "tie":
                tally[name]["ties"] += 1
            elif winner == side:
                tally[name]["wins"] += 1
            else:
                tally[name]["losses"] += 1

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in fitted}
    for _ in range(bootstrap):
        resampled = bradley_terry([rng.choice(votes) for _ in votes], iterations=60, tol=1e-5)
        for name in fitted:
            p = resampled.get(name, 1.0)
            draws[name].append(p / (1 + p))

    rows = []
    for name, p in fitted.items():
        rows.append(
            {
                "op": name,
                **tally[name],
                "score": p / (1 + p),
                "low": percentile(draws[name], alpha / 2) if draws[name] else None,
                "high": percentile(draws[name], 1 - alpha / 2) if draws[name] else None,
            }
        )
    rows.sort(key=lambda r: -r["score"])
    return rows
