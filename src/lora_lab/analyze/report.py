"""Build samples from the database and format a ranking."""

from __future__ import annotations

import json
from dataclasses import replace

from ..db import Store
from ..scorers.defects import pairwise_diversity
from .agreement import calibration as judge_calibration
from .stats import (
    AXES,
    LoraResult,
    Sample,
    analyse_all,
    image_utility,
    is_fail,
    is_pass,
    mean,
    paired_difference,
    pareto_front,
    rank_stability,
)

HEADER = (
    f"{'#':>2}  {'LoRA':<28} {'subj':<12} {'ep':>4} {'s':>4} {'n':>5} "
    f"{'mean':>6} {'lcb':>6} {'rob':>6} {'stab':>6} {'pass':>6} {'fail':>6} {'scen':>4}"
)


def _row_meta(row, column: str) -> dict:
    if column not in row.keys() or not row[column]:
        return {}
    try:
        data = json.loads(row[column])
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _format_adherence_details(
    passed: tuple[str, ...], failed: tuple[str, ...], missing: tuple[str, ...]
) -> str:
    parts = []
    if passed:
        parts.append("✓ " + ", ".join(passed))
    if failed:
        parts.append("✗ " + ", ".join(failed))
    if missing:
        parts.append("? " + ", ".join(missing))
    return " · ".join(parts)


def samples_from_store(store: Store, run_id: int | None = None) -> list[Sample]:
    samples = []
    for row in store.scored_images(run_id):
        critical = row["critical"]
        id_meta = _row_meta(row, "identity_meta")
        adh_meta = _row_meta(row, "adherence_meta")
        crit_meta = _row_meta(row, "critical_meta")
        raw_emb = id_meta.get("embedding")
        embedding = (
            tuple(float(x) for x in raw_emb)
            if isinstance(raw_emb, list) and raw_emb
            else None
        )
        raw_leak = id_meta.get("leakage")
        def_meta = _row_meta(row, "defects_meta")
        raw_def = row["defects"] if row["defects"] is not None else crit_meta.get("defect_index")
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
                image_id=int(row["id"]) if "id" in row.keys() and row["id"] is not None else None,
                path=str(row["path"]) if "path" in row.keys() and row["path"] else "",
                adherence_passed=tuple(str(x) for x in (adh_meta.get("passed") or ())),
                adherence_failed=tuple(str(x) for x in (adh_meta.get("failed") or ())),
                adherence_missing=tuple(str(x) for x in (adh_meta.get("missing") or ())),
                leakage=float(raw_leak) if raw_leak is not None else None,
                embedding=embedding,
                defect_index=float(raw_def) if raw_def is not None else None,
                defects=tuple(str(x) for x in (def_meta.get("detected") or ())),
            )
        )
    return samples


def gate_by_calibration(samples: list[Sample], calibration: dict[str, dict]) -> list[Sample]:
    """Drop the judge axes whose agreement with the gold set is at chance level."""

    drop_adherence = calibration.get("adherence", {}).get("status") == "rejected"
    drop_defects = calibration.get("defects", {}).get("status") == "rejected"
    if not (drop_adherence or drop_defects):
        return samples
    out = []
    for sample in samples:
        changes: dict = {}
        if drop_adherence:
            changes["adherence"] = None
        if drop_defects:
            changes.update(defects=(), defect_index=None)
        out.append(replace(sample, **changes))
    return out


def paired_report(
    samples_a: list[Sample],
    samples_b: list[Sample],
    op_a: str,
    op_b: str,
    *,
    bootstrap: int = 2000,
) -> dict | None:
    """Paired difference of two operating points (``lora@epN@sX``)."""

    def pick(samples: list[Sample], op: str) -> list[Sample]:
        return [s for s in samples if f"{s.lora_id}@ep{s.epoch}@s{s.strength:g}" == op]

    result = paired_difference(pick(samples_a, op_a), pick(samples_b, op_b), n=bootstrap)
    if result is not None:
        result.update(a=op_a, b=op_b)
    return result


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
    calibration = judge_calibration(store)
    samples = gate_by_calibration(samples_from_store(store, run_id), calibration)
    results = analyse_all(samples, weights=weights, bootstrap=bootstrap)
    keys = {operating_point(r): r for r in results}
    sums = {key: {axis: 0.0 for axis in AXES} for key in keys}
    seen = {key: {axis: 0 for axis in AXES} for key in keys}

    images_by_op: dict[str, list[dict]] = {}
    images_by_cell: dict[tuple[str, str], list[dict]] = {}
    compare_slots: dict[tuple[str, int], dict[str, dict]] = {}
    embeddings_by_op: dict[str, list[list[float]]] = {}
    leakages_by_op: dict[str, list[float]] = {}
    defects_by_op: dict[str, list[bool]] = {}
    unscored: list[dict] = []

    for sample in samples:
        key = f"{sample.lora_id}@ep{sample.epoch}@s{sample.strength:g}"
        utility = image_utility(sample, weights)
        adh_summary = _format_adherence_details(
            sample.adherence_passed, sample.adherence_failed, sample.adherence_missing
        )
        item = {
            "id": sample.image_id,
            "path": sample.path,
            "op": key,
            "lora_id": sample.lora_id,
            "subject": sample.subject,
            "epoch": sample.epoch,
            "strength": sample.strength,
            "scenario": sample.scenario,
            "seed": sample.seed,
            "identity": sample.identity,
            "adherence": sample.adherence,
            "adherence_passed": list(sample.adherence_passed),
            "adherence_failed": list(sample.adherence_failed),
            "adherence_missing": list(sample.adherence_missing),
            "adherence_details": adh_summary,
            "quality": sample.quality,
            "critical": sample.critical,
            "leakage": sample.leakage,
            "defect_index": sample.defect_index,
            "defects": list(sample.defects),
            "utility": utility,
            "passed": is_pass(sample) if utility is not None else False,
            "failed": is_fail(sample) if utility is not None else False,
        }
        if sample.embedding is not None:
            embeddings_by_op.setdefault(key, []).append(list(sample.embedding))
        if sample.leakage is not None:
            leakages_by_op.setdefault(key, []).append(sample.leakage)
        if sample.defect_index is not None:
            defects_by_op.setdefault(key, []).append(bool(sample.defects))
        if sample.path:
            compare_slots.setdefault((sample.scenario, sample.seed), {})[key] = item
        if utility is None:
            if sample.path:
                unscored.append(item)
            continue
        if sample.path:
            images_by_op.setdefault(key, []).append(item)
            images_by_cell.setdefault((key, sample.scenario), []).append(item)
        if key not in sums:
            continue
        for axis in AXES:
            value = getattr(sample, axis)
            if value is not None:
                sums[key][axis] += value
                seen[key][axis] += 1

    for items in images_by_op.values():
        items.sort(key=lambda x: -(x["utility"] or 0.0))
    for items in images_by_cell.values():
        items.sort(key=lambda x: x["seed"])

    components = {
        name: {
            axis: (sums[name][axis] / seen[name][axis] if seen[name][axis] else None)
            for axis in AXES
        }
        for name in sums
    }

    breakdown = scenario_breakdown(samples, weights)
    scenarios = sorted({row["scenario"] for row in breakdown} | {s.scenario for s in samples})
    seeds = sorted({s.seed for s in samples})
    matrix: dict[str, dict[str, dict]] = {}
    for row in breakdown:
        cell_images = images_by_cell.get((row["op"], row["scenario"]), [])
        matrix.setdefault(row["op"], {})[row["scenario"]] = {
            "mean": row["mean"],
            "n": row["n"],
            "shade": _shade(row["mean"]),
            "images": cell_images,
        }

    op_order = [operating_point(r) for r in results]
    if not op_order:
        op_order = sorted({f"{s.lora_id}@ep{s.epoch}@s{s.strength:g}" for s in samples})
    ranks = rank_stability(components, draws=stability_draws) if results else {}
    top_rank = {
        name: sum(p for rank, p in values.items() if rank <= 1.5)
        for name, values in ranks.items()
    }
    pareto_ops = (
        pareto_front({operating_point(r): (r.mean, r.robustness) for r in results})
        if results
        else []
    )
    pareto_set = set(pareto_ops)
    return {
        "samples": sum(r.n for r in results),
        "total_images": len(samples),
        "unscored": unscored,
        "operating_points": op_order,
        "pareto_front": pareto_ops,
        "scenario_names": scenarios,
        "seeds": seeds,
        "matrix": matrix,
        "breakdown": breakdown,
        "strength_series": _series(breakdown, "strength"),
        "epoch_series": _series(breakdown, "epoch"),
        "compare": [
            {
                "scenario": scenario,
                "seed": seed,
                "by_op": slots,
            }
            for (scenario, seed), slots in sorted(compare_slots.items())
        ],
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
                "pareto": operating_point(r) in pareto_set,
                "diversity": pairwise_diversity(embeddings_by_op.get(operating_point(r), [])),
                "max_leakage": (
                    max(leakages_by_op[operating_point(r)])
                    if leakages_by_op.get(operating_point(r))
                    else None
                ),
                "defect_rate": (
                    mean(1.0 if d else 0.0 for d in defects_by_op[operating_point(r)])
                    if defects_by_op.get(operating_point(r))
                    else None
                ),
                "axes": components.get(operating_point(r), {}),
                "images": images_by_op.get(operating_point(r), []),
            }
            for r in results
        ],
        "rank_stability": ranks,
        "top_rank": top_rank,
        "calibration": calibration,
    }


def report_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
