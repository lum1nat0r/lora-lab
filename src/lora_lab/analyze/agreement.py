"""Judge calibration: agreement of the stored judge verdicts with human labels.

Gold labels are stored per image as ``{key: bool}``; scenario attributes use
their attribute id, defects use ``defect:<key>``. The judge's verdicts are
read back from the score metadata the adherence and defect scorers write, so
no extra model call is needed.

Agreement is reported as Cohen's kappa per judge kind. The report gates on it:
a judge whose kappa is below ``REJECT_KAPPA`` is dropped from the ranking,
because a ranking built on chance-level answers is worse than none.
"""

from __future__ import annotations

import json

from ..db import Store

MIN_LABELS = 30  # judged pairs per kind before a verdict is drawn
TRUST_KAPPA = 0.6
REJECT_KAPPA = 0.4
DEFECT_PREFIX = "defect:"


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Binary Cohen's kappa; None when chance agreement is already perfect."""

    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    yes_a = sum(1 for a, _ in pairs if a) / n
    yes_b = sum(1 for _, b in pairs if b) / n
    expected = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def judge_verdicts(store: Store) -> dict[int, dict[str, bool]]:
    """Judge answers of every labelled image, keyed like the labels."""

    out: dict[int, dict[str, bool]] = {}
    for row in store.judge_meta():
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except json.JSONDecodeError:
            continue
        verdicts = out.setdefault(int(row["image_id"]), {})
        if row["scorer"] == "adherence":
            verdicts.update(dict.fromkeys(meta.get("passed") or (), True))
            verdicts.update(dict.fromkeys(meta.get("failed") or (), False))
        else:
            for key in meta.get("detected") or ():
                verdicts[DEFECT_PREFIX + key] = True
            for key in meta.get("absent") or ():
                verdicts[DEFECT_PREFIX + key] = False
    return out


def _status(n: int, kappa: float | None) -> str:
    if n < MIN_LABELS or kappa is None:
        return "uncalibrated"
    if kappa >= TRUST_KAPPA:
        return "ok"
    return "weak" if kappa >= REJECT_KAPPA else "rejected"


def agreement(
    labels: dict[int, dict[str, bool]], verdicts: dict[int, dict[str, bool]]
) -> dict[str, dict]:
    """Per judge kind (``adherence``, ``defects``): n, agreement, kappa, status, per key."""

    pairs: dict[str, dict[str, list[tuple[bool, bool]]]] = {"adherence": {}, "defects": {}}
    for image_id, gold in labels.items():
        judged = verdicts.get(image_id, {})
        for key, human in gold.items():
            if key not in judged:
                continue
            kind = "defects" if key.startswith(DEFECT_PREFIX) else "adherence"
            pairs[kind].setdefault(key, []).append((human, judged[key]))

    out: dict[str, dict] = {}
    for kind, by_key in pairs.items():
        flat = [pair for items in by_key.values() for pair in items]
        kappa = cohen_kappa(flat)
        out[kind] = {
            "n": len(flat),
            "agreement": (sum(1 for a, b in flat if a == b) / len(flat)) if flat else None,
            "kappa": kappa,
            "status": _status(len(flat), kappa),
            "needed": MIN_LABELS,
            "keys": [
                {
                    "key": key.removeprefix(DEFECT_PREFIX),
                    "n": len(items),
                    "agreement": sum(1 for a, b in items if a == b) / len(items),
                    "kappa": cohen_kappa(items),
                }
                for key, items in sorted(by_key.items())
            ],
        }
    return out


def calibration(store: Store) -> dict[str, dict]:
    return agreement(store.labels(), judge_verdicts(store))
