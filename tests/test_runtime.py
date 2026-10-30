from __future__ import annotations

import pytest
import torch

from sprm.contracts.types import KVSlice
from sprm.errors import MemoryInvariantError
from sprm.runtime.form_a import ExactWindow


def _slice(start: int = 3, tokens: int = 4) -> KVSlice:
    return KVSlice(
        keys=torch.zeros(1, 2, tokens, 8),
        values=torch.ones(1, 2, tokens, 8),
        positions=torch.arange(start, start + tokens),
        start=start,
    )


def test_kvslice_basics() -> None:
    slice_ = _slice()
    assert slice_.num_tokens == 4
    assert slice_.end == 7
    narrowed = slice_.narrow(4, 2)
    assert narrowed.start == 4
    assert narrowed.num_tokens == 2


def test_kvslice_rejects_bad_positions() -> None:
    with pytest.raises(ValueError, match="positions\\[0\\]"):
        KVSlice(
            keys=torch.zeros(1, 2, 4, 8),
            values=torch.zeros(1, 2, 4, 8),
            positions=torch.arange(0, 4),
            start=3,
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        KVSlice(
            keys=torch.zeros(1, 2, 4, 8),
            values=torch.zeros(1, 2, 3, 8),
            positions=torch.arange(0, 4),
            start=0,
        )


def test_exact_window_keeps_one_position_per_token() -> None:
    window = ExactWindow(
        keys={0: torch.zeros(1, 2, 2, 4), 1: torch.zeros(1, 2, 2, 4)},
        values={0: torch.zeros(1, 2, 2, 4), 1: torch.zeros(1, 2, 2, 4)},
        positions=torch.arange(0, 2),
        capacity=4,
    )
    step_keys = {0: torch.ones(1, 2, 1, 4), 1: torch.ones(1, 2, 1, 4)}
    step_values = {0: torch.ones(1, 2, 1, 4), 1: torch.ones(1, 2, 1, 4)}
    window.append_step(step_keys, step_values, 2)
    assert window.positions.tolist() == [0, 1, 2]
    for layer in (0, 1):
        assert window.keys[layer].shape[-2] == 3
        assert window.values[layer].shape[-2] == 3


def test_exact_window_rolls_at_capacity() -> None:
    window = ExactWindow(
        keys={0: torch.zeros(1, 2, 2, 4)},
        values={0: torch.zeros(1, 2, 2, 4)},
        positions=torch.arange(0, 2),
        capacity=2,
    )
    window.append_step(
        {0: torch.ones(1, 2, 1, 4)},
        {0: torch.ones(1, 2, 1, 4)},
        2,
    )
    assert window.positions.tolist() == [1, 2]
    assert window.keys[0].shape[-2] == 2


def test_exact_window_rejects_incomplete_layer_set() -> None:
    window = ExactWindow(
        keys={0: torch.zeros(1, 2, 2, 4), 1: torch.zeros(1, 2, 2, 4)},
        values={0: torch.zeros(1, 2, 2, 4), 1: torch.zeros(1, 2, 2, 4)},
        positions=torch.arange(0, 2),
        capacity=4,
    )
    with pytest.raises(MemoryInvariantError, match="exactly"):
        window.append_step(
            {0: torch.ones(1, 2, 1, 4)},
            {0: torch.ones(1, 2, 1, 4)},
            2,
        )
