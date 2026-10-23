"""Build samples from the database and format a ranking."""

from __future__ import annotations

import json

from ..db import Store
from .stats import AXES, LoraResult, Sample, analyse_all, image_utility, mean, rank_stability

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


def operating_point(result: LoraResult) -> str:
    return f"{result.lora_id}@ep{result.epoch}@s{result.strength:g}"


def scenario_breakdown(
    samples: list[Sample], weights: dict[str, float] | None = None
) -> list[dict]:
    acc: dict[tuple[str, int, float, str], list[float]] = {}
    for sample in samples:
        utility = image_utility(sample, weights)
        if utility is None:
            continue
        key = (sample.lora_id, sample.epoch, sample.strength, sample.scenario)
        acc.setdefault(key, []).append(utility)
    rows = []
    for (lora_id, epoch, strength, scenario), values in sorted(acc.items()):
        rows.append(
            {
                "op": f"{lora_id}@ep{epoch}@s{strength:g}",
                "lora_id": lora_id,
                "epoch": epoch,
                "strength": strength,
                "scenario": scenario,
                "mean": mean(values),
                "n": len(values),
            }
        )
    return rows


def _shade(value: float) -> str:
    value = max(0.0, min(1.0, value))
    return f"hsl({int(value * 120)}, 42%, {22 + value * 14:.0f}%)"


def _series(rows: list[dict], field: str) -> dict[str, list[dict]]:
    acc: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        acc.setdefault(row["lora_id"], {}).setdefault(row[field], []).append(row["mean"])
    out: dict[str, list[dict]] = {}
    for lora_id, buckets in acc.items():
        out[lora_id] = [
            {field: key, "mean": mean(values), "n": len(values)}
            for key, values in sorted(buckets.items())
        ]
    return out


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
    keys = {operating_point(r): r for r in results}
    sums = {key: {axis: 0.0 for axis in AXES} for key in keys}
    seen = {key: {axis: 0 for axis in AXES} for key in keys}
    for sample in samples:
        key = f"{sample.lora_id}@ep{sample.epoch}@s{sample.strength:g}"
        if key not in sums:
            continue
        for axis in AXES:
            value = getattr(sample, axis)
            if value is not None:
                sums[key][axis] += value
                seen[key][axis] += 1
    components = {
        name: {
            axis: (sums[name][axis] / seen[name][axis] if seen[name][axis] else None)
            for axis in AXES
        }
        for name in sums
    }

    breakdown = scenario_breakdown(samples, weights)
    scenarios = sorted({row["scenario"] for row in breakdown})
    matrix: dict[str, dict[str, dict]] = {}
    for row in breakdown:
        matrix.setdefault(row["op"], {})[row["scenario"]] = {
            "mean": row["mean"],
            "n": row["n"],
            "shade": _shade(row["mean"]),
        }

    op_order = [operating_point(r) for r in results]
    ranks = rank_stability(components, draws=stability_draws) if results else {}
    top_rank = {
        name: sum(p for rank, p in values.items() if rank <= 1.5)
        for name, values in ranks.items()
    }
    return {
        "samples": len(samples),
        "operating_points": op_order,
        "scenario_names": scenarios,
        "matrix": matrix,
        "breakdown": breakdown,
        "strength_series": _series(breakdown, "strength"),
        "epoch_series": _series(breakdown, "epoch"),
        "results": [
            {
                "op": operating_point(r),
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
        "rank_stability": ranks,
        "top_rank": top_rank,
    }


def report_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
