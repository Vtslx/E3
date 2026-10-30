from __future__ import annotations

import pytest

from sprm.config.schema import TaskConfig
from sprm.eval.niah import build_samples, describe_sample
from tests.fakes import CharTokenizer


def _task() -> TaskConfig:
    return TaskConfig(
        kinds=("single_needle", "multi_needle", "conditional"),
        context_lengths=(512, 1024),
        samples_per_cell=2,
        depths=(0.1, 0.5, 0.9),
        filler_seed=7101,
        needle_seed=9107,
        use_chat_template=False,
        max_answer_tokens=8,
        long_form_records=2,
    )


def test_samples_are_deterministic() -> None:
    tokenizer = CharTokenizer()
    first = build_samples(tokenizer, _task())
    second = build_samples(tokenizer, _task())
    assert len(first) == len(second) > 0
    for left, right in zip(first, second, strict=True):
        assert left.sample_id == right.sample_id
        assert left.token_ids == right.token_ids
        assert left.question_token_ids == right.question_token_ids
        assert left.metadata["options"] == right.metadata["options"]
        assert left.metadata["code"] == right.metadata["code"]


def test_context_length_is_exact_and_prompt_is_concatenation() -> None:
    tokenizer = CharTokenizer()
    for sample in build_samples(tokenizer, _task(), max_samples=12):
        assert len(sample.token_ids) == sample.context_length
        assert sample.full_token_ids == sample.token_ids + sample.question_token_ids
        assert tokenizer.decode(sample.question_token_ids).endswith("Answer: ")


def test_needle_is_present_once_and_gold_is_a_candidate() -> None:
    tokenizer = CharTokenizer()
    for sample in build_samples(tokenizer, _task(), max_samples=12):
        context_text = tokenizer.decode(sample.token_ids)
        code = sample.metadata["code"]
        assert context_text.count(code) == 1
        options = sample.metadata["options"]
        assert len(options) == len(set(options)) == 4
        assert code in options
        assert sample.gold_answer == code
        assert sample.metadata["correct_index"] == options.index(code)
        assert sample.gold_token_ids == tuple(tokenizer.encode(code))
        start, end = sample.metadata["needle_positions"][0]
        assert 0 <= start < end <= sample.context_length


def test_depth_moves_the_target_needle() -> None:
    tokenizer = CharTokenizer()
    samples = build_samples(
        tokenizer,
        _task(),
        kinds=["single_needle"],
        context_lengths=[1024],
        depths=[0.1, 0.9],
        samples_per_cell=1,
    )
    positions = [sample.metadata["needle_position"] for sample in samples]
    assert positions[0] < positions[1]
    for sample in samples:
        expected = round(sample.depth * sample.context_length)
        assert abs(sample.metadata["needle_position"] - expected) <= 1


def test_multi_needle_contains_three_records() -> None:
    tokenizer = CharTokenizer()
    sample = build_samples(
        tokenizer, _task(), kinds=["multi_needle"], context_lengths=[1024], depths=[0.1], samples_per_cell=1
    )[0]
    assert len(sample.metadata["needle_positions"]) == 3
    assert len(sample.metadata["sentences"]) == 3


def test_conditional_contains_both_attributes() -> None:
    tokenizer = CharTokenizer()
    sample = build_samples(
        tokenizer, _task(), kinds=["conditional"], context_lengths=[1024], depths=[0.1], samples_per_cell=1
    )[0]
    text = tokenizer.decode(sample.token_ids)
    attribute = sample.metadata["attribute"]
    assert f"{attribute} passkey" in text
    assert any(other in text for other in ("red passkey", "blue passkey", "north passkey", "south passkey"))


def test_max_samples_is_respected() -> None:
    tokenizer = CharTokenizer()
    samples = build_samples(tokenizer, _task(), max_samples=3)
    assert len(samples) == 3


def test_invalid_filters_are_rejected() -> None:
    tokenizer = CharTokenizer()
    with pytest.raises(ValueError, match="unknown task kind"):
        build_samples(tokenizer, _task(), kinds=["mystery"])
    with pytest.raises(ValueError, match="depth"):
        build_samples(tokenizer, _task(), depths=[1.0])


def test_describe_sample_reports_candidates() -> None:
    tokenizer = CharTokenizer()
    sample = build_samples(tokenizer, _task(), max_samples=1)[0]
    description = describe_sample(sample)
    assert description["gold_answer"] == sample.metadata["code"]
    assert description["options"] == sample.metadata["options"]
    assert description["prompt_tokens"] == len(sample.full_token_ids)
