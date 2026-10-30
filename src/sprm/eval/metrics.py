"""Metrics for passkey retrieval.

The primary score is a candidate ranking in answer space: the prompt asks for the passkey
without listing candidates, and the scorer compares the model's teacher-forced log-probability
of each candidate code. Nothing about option order or letter tokenization can bias it, and a
model that has not retrieved the passkey is at chance.

Why not multiple-choice letters: with this tokenizer " A" and " C" are single tokens while
" B" and " D" split into a generic space plus the letter, and the model has a strong
first-option bias. Both effects swamped the retrieval signal.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateScore:
    logprobs: dict[str, float]
    probabilities: dict[str, float]
    predicted_code: str
    gold_code: str
    gold_logprob: float
    gold_probability: float
    gold_rank: int
    correct: bool


def rank_candidates(
    gold_code: str,
    candidates: Sequence[str],
    logprobs: dict[str, float],
) -> CandidateScore:
    """Rank candidate codes by total sequence log-probability."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if gold_code not in candidates:
        raise ValueError(f"gold code {gold_code!r} not among candidates")
    missing = [code for code in candidates if code not in logprobs]
    if missing:
        raise ValueError(f"missing logprobs for {missing}")
    ordered = sorted(candidates, key=lambda code: (-logprobs[code], candidates.index(code)))
    predicted = ordered[0]
    gold_logprob = float(logprobs[gold_code])
    gold_rank = 1 + sum(1 for code in candidates if logprobs[code] > gold_logprob)
    finite = [value for value in logprobs.values() if math.isfinite(value)]
    if finite:
        maximum = max(finite)
        weights = {code: math.exp(logprobs[code] - maximum) for code in candidates}
        total = sum(weights.values())
        probabilities = {code: weights[code] / total for code in candidates}
    else:
        probabilities = {code: 1.0 / len(candidates) for code in candidates}
    return CandidateScore(
        logprobs={code: float(logprobs[code]) for code in candidates},
        probabilities=probabilities,
        predicted_code=predicted,
        gold_code=gold_code,
        gold_logprob=gold_logprob,
        gold_probability=probabilities[gold_code],
        gold_rank=gold_rank,
        correct=predicted == gold_code,
    )


_CODE_PATTERN = re.compile(r"\b(\d{4,6})\b")


def extract_code(text: str) -> str | None:
    """First numeric passkey in generated text, or ``None``."""
    match = _CODE_PATTERN.search(text)
    return match.group(1) if match else None


def mcnemar_one_sided(improved: int, degraded: int) -> float:
    """Exact one-sided p-value for ``improved > degraded`` under the null.

    ``improved`` counts pairs where the candidate is right and the control wrong;
    ``degraded`` is the reverse. Ties carry no information and are excluded.
    """
    if improved < 0 or degraded < 0:
        raise ValueError("counts must be non-negative")
    n = improved + degraded
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(improved, n + 1))
    return float(tail / (2**n))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centre - margin), min(1.0, centre + margin))
