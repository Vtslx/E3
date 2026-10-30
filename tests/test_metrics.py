from __future__ import annotations

import math

import pytest

from sprm.eval.metrics import (
    extract_code,
    mcnemar_one_sided,
    rank_candidates,
    wilson_interval,
)


def test_rank_candidates_selects_the_highest_logprob() -> None:
    score = rank_candidates(
        "73914",
        ["73914", "48120", "20877", "91536"],
        {"73914": -0.2, "48120": -1.0, "20877": -2.0, "91536": -3.0},
    )
    assert score.correct is True
    assert score.predicted_code == "73914"
    assert score.gold_rank == 1
    assert score.gold_probability > 0.5
    assert math.isclose(sum(score.probabilities.values()), 1.0)


def test_rank_candidates_reports_gold_rank() -> None:
    score = rank_candidates(
        "73914",
        ["73914", "48120", "20877", "91536"],
        {"73914": -3.0, "48120": -0.1, "20877": -2.0, "91536": -4.0},
    )
    assert score.correct is False
    assert score.gold_rank == 3
    assert score.predicted_code == "48120"


def test_rank_candidates_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="unique"):
        rank_candidates("a", ["a", "a"], {"a": 0.0})
    with pytest.raises(ValueError, match="not among"):
        rank_candidates("z", ["a"], {"a": 0.0})
    with pytest.raises(ValueError, match="missing logprobs"):
        rank_candidates("a", ["a", "b"], {"a": 0.0})
    with pytest.raises(ValueError, match="must not be empty"):
        rank_candidates("a", [], {})


def test_rank_candidates_handles_all_negative_infinity() -> None:
    score = rank_candidates("a", ["a", "b"], {"a": float("-inf"), "b": float("-inf")})
    assert math.isclose(score.probabilities["a"], 0.5)
    assert score.gold_rank == 1


def test_extract_code() -> None:
    assert extract_code("The passkey is 73914.") == "73914"
    assert extract_code(" 7391") == "7391"
    assert extract_code("no digits here") is None
    assert extract_code("123") is None


def test_mcnemar_known_values() -> None:
    assert mcnemar_one_sided(0, 0) == 1.0
    assert math.isclose(mcnemar_one_sided(5, 0), 1.0 / 32.0)
    assert mcnemar_one_sided(0, 5) == 1.0
    assert math.isclose(mcnemar_one_sided(1, 1), 0.75)


def test_wilson_interval() -> None:
    low, high = wilson_interval(5, 10)
    assert 0.0 <= low < 0.5 < high <= 1.0
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)
