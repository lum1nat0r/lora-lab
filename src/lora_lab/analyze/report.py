"""Build samples from the database and format a ranking."""

from __future__ import annotations

import json

from ..db import Store
from .stats import AXES, LoraResult, Sample, analyse_all, rank_stability

HEADER = (
    f"{'#':>2}  {'LoRA':<28} {'subj':<12} {'ep':>4} {'s':>4} {'n':>5} "
    f"{'mean':>6} {'lcb':>6} {'rob':>6} {'stab':>6} {'pass':>6} {'fail':>6} {'scen':>4}"
)


def samples_from_store(store: Store, run_id: int | None = None) -> list[Sample]:
    samples = []
    for row in store.scored_images(run_id):
        critical = row["critical"]
        samples.append(
            Sample(
                lora_id=str(row["lora_id"]) or f"{row['subject']}:base",
                subject=str(row["subject"]),
                scenario=str(row["scenario"]),
                seed=int(row["seed"]),
                identity=row["identity"],
                adherence=row["adherence"],
                quality=row["quality"],
                critical=bool(critical) if critical is not None else False,
                epoch=int(row["epoch"]),
                strength=float(row["strength"]),
            )
        )
    return samples


def format_results(results: list[LoraResult]) -> str:
    lines = [HEADER, "-" * len(HEADER)]
    for index, result in enumerate(results, start=1):
        stability = f"{result.stability:.2f}" if result.stability is not None else "  - "
        lines.append(
            f"{index:>2}  {result.lora_id[:28]:<28} {result.subject[:12]:<12} "
            f"{result.epoch:>4} {result.strength:>4g} {result.n:>5} "
            f"{result.mean:>6.3f} {result.lcb:>6.3f} {result.robustness:>6.3f} "
            f"{stability:>6} {result.yield_rate:>5.0%} {result.fail_rate:>5.0%} "
            f"{result.scenarios:>4}"
        )
    return "\n".join(lines)


def build_report(
    store: Store,
    *,
    run_id: int | None = None,
    bootstrap: int = 2000,
    weights: dict[str, float] | None = None,
    stability_draws: int = 500,
) -> dict:
    samples = samples_from_store(store, run_id)
    results = analyse_all(samples, weights=weights, bootstrap=bootstrap)
    sums = {r.lora_id: {axis: 0.0 for axis in AXES} for r in results}
    seen = {r.lora_id: {axis: 0 for axis in AXES} for r in results}
    for sample in samples:
        if sample.lora_id not in sums:
            continue
        for axis in AXES:
            value = getattr(sample, axis)
            if value is not None:
                sums[sample.lora_id][axis] += value
                seen[sample.lora_id][axis] += 1
    components = {
        name: {
            axis: (sums[name][axis] / seen[name][axis] if seen[name][axis] else None)
            for axis in AXES
        }
        for name in sums
    }

    return {
        "samples": len(samples),
        "results": [
            {
                "lora_id": r.lora_id,
                "subject": r.subject,
                "epoch": r.epoch,
                "strength": r.strength,
                "n": r.n,
                "mean": r.mean,
                "lcb": r.lcb,
                "ucb": r.ucb,
                "robustness": r.robustness,
                "stability": r.stability,
                "yield_rate": r.yield_rate,
                "fail_rate": r.fail_rate,
                "scenarios": r.scenarios,
                "per_scenario": r.per_scenario,
            }
            for r in results
        ],
        "rank_stability": rank_stability(components, draws=stability_draws) if results else {},
    }


def report_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
