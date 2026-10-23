import pytest

from lora_lab.analyze.stats import (
    DEFAULT_WEIGHTS,
    Sample,
    analyse_all,
    analyse_lora,
    cluster_bootstrap_interval,
    image_utility,
    is_fail,
    is_pass,
    pareto_front,
    percentile,
    rank_stability,
    stability,
    weighted,
)


def sample(scenario: str, seed: int, value: float, **kwargs) -> Sample:
    return Sample(
        lora_id=kwargs.pop("lora_id", "lora"),
        subject=kwargs.pop("subject", "a"),
        scenario=scenario,
        seed=seed,
        identity=value,
        adherence=value,
        quality=value,
        **kwargs,
    )


def test_weighted_renormalises_missing_axes() -> None:
    assert weighted({"identity": 1.0}, {"identity": 0.5, "adherence": 0.5}) == pytest.approx(1.0)
    assert weighted({"identity": None, "quality": 1.0}, {"identity": 0.9, "quality": 0.1}) == pytest.approx(1.0)
    assert weighted({"identity": None}, {"identity": 1.0}) is None


def test_image_utility_excludes_missing_axes() -> None:
    only_identity = Sample(lora_id="l", subject="a", scenario="s", seed=1, identity=0.8)
    assert image_utility(only_identity) == pytest.approx(0.8)


def test_percentile_interpolates() -> None:
    assert percentile([0, 10], 0.5) == pytest.approx(5.0)
    assert percentile([0, 10, 20], 0.2) == pytest.approx(4.0)
    assert percentile([7], 0.9) == 7


def test_scenarios_are_weighted_equally() -> None:
    samples = [sample("easy", seed, 1.0) for seed in range(10)]
    samples += [sample("hard", seed, 0.0) for seed in range(2)]
    result = analyse_lora(samples, bootstrap=200)
    assert result is not None
    # pooled mean would be 10/12 = 0.83; scenario weighting gives 0.5
    assert result.mean == pytest.approx(0.5)
    assert result.per_scenario == {"easy": pytest.approx(1.0), "hard": pytest.approx(0.0)}


def test_robustness_is_lower_percentile() -> None:
    samples = [sample(f"s{i}", 1, value) for i, value in enumerate([0.1, 0.5, 0.9, 1.0, 1.0])]
    result = analyse_lora(samples, bootstrap=100)
    assert result is not None
    assert result.robustness < result.mean
    assert result.robustness == pytest.approx(0.5, abs=0.2)


def test_stability_penalises_seed_spread() -> None:
    tight = [sample("s", seed, 0.8) for seed in range(4)]
    wide = [sample("s", seed, value) for seed, value in enumerate([0.0, 0.4, 1.0, 0.9])]
    assert stability(tight) == pytest.approx(1.0)
    assert stability(wide) < 0.5


def test_bootstrap_is_deterministic_and_brackets_mean() -> None:
    samples = [sample("a", 1, 0.9), sample("a", 2, 0.7), sample("b", 1, 0.5), sample("b", 2, 0.6)]
    low, high = cluster_bootstrap_interval(samples, n=500, seed=3)
    assert (low, high) == cluster_bootstrap_interval(samples, n=500, seed=3)
    assert low <= high


def test_pass_and_fail_thresholds() -> None:
    good = Sample(lora_id="l", subject="a", scenario="s", seed=1, identity=0.9, adherence=0.9)
    assert is_pass(good)
    assert not is_fail(good)

    bad = Sample(lora_id="l", subject="a", scenario="s", seed=1, identity=0.1, adherence=0.9)
    assert not is_pass(bad)
    assert is_fail(bad)

    critical = Sample(
        lora_id="l", subject="a", scenario="s", seed=1, identity=0.9, adherence=0.9, critical=True
    )
    assert not is_pass(critical)
    assert is_fail(critical)
    assert is_pass(good)


def test_strengths_are_separate_operating_points() -> None:
    samples = [
        sample("a", 1, 0.9, lora_id="lora", strength=0.8),
        sample("a", 1, 0.5, lora_id="lora", strength=1.0),
    ]
    results = analyse_all(samples, bootstrap=100)
    assert len(results) == 2
    assert {r.strength for r in results} == {0.8, 1.0}


def test_analyse_all_orders_by_lcb() -> None:
    strong = [sample("a", seed, 0.9, lora_id="strong") for seed in range(4)]
    weak = [sample("a", seed, 0.3, lora_id="weak") for seed in range(4)]
    results = analyse_all(strong + weak, bootstrap=200)
    assert [r.lora_id for r in results] == ["strong", "weak"]
    assert results[0].lcb > results[1].lcb


def test_pareto_front() -> None:
    points = {"a": (0.9, 0.5), "b": (0.8, 0.8), "c": (0.5, 0.5), "d": (0.95, 0.4)}
    assert pareto_front(points) == ["d", "a", "b"]


def test_rank_stability_normalises_and_keeps_dominant() -> None:
    components = {
        "good": {"identity": 0.9, "adherence": 0.9, "quality": 0.9},
        "bad": {"identity": 0.3, "adherence": 0.3, "quality": 0.3},
    }
    ranks = rank_stability(components, DEFAULT_WEIGHTS, draws=200)
    assert sum(ranks["good"].values()) == pytest.approx(1.0)
    assert ranks["good"].get(1, 0) == pytest.approx(1.0)
    assert ranks["bad"].get(2, 0) == pytest.approx(1.0)


def test_rank_stability_declares_ties() -> None:
    components = {
        "a": {"identity": 0.5, "adherence": 0.5, "quality": 0.5},
        "b": {"identity": 0.5, "adherence": 0.5, "quality": 0.5},
    }
    ranks = rank_stability(components, draws=20)
    assert ranks["a"].get(1.5, 0) == pytest.approx(1.0)
    assert ranks["b"].get(1.5, 0) == pytest.approx(1.0)
