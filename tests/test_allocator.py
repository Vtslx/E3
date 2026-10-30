from __future__ import annotations

import pytest

from sprm.errors import ConfigError
from sprm.memory.allocator import allocate_slots
from sprm.runtime.form_a import _online_slot_plan


@pytest.mark.parametrize("blocks,slots", [(1, 8), (4, 32), (8, 16), (17, 512)])
def test_uniform_spends_exactly_the_budget(blocks: int, slots: int) -> None:
    allocation = allocate_slots(
        num_blocks=blocks, num_slots=slots, mode="uniform", min_slots_per_block=1, max_slot_multiplier=4.0
    )
    assert len(allocation) == blocks
    assert sum(allocation) == slots
    assert min(allocation) >= 1


def test_effective_rank_is_monotone_in_importance() -> None:
    allocation = allocate_slots(
        num_blocks=4,
        num_slots=32,
        mode="effective_rank",
        importances=[1.0, 2.0, 4.0, 8.0],
        min_slots_per_block=1,
        max_slot_multiplier=4.0,
    )
    assert sum(allocation) == 32
    assert allocation[-1] > allocation[0]


def test_budget_below_minimum_is_rejected() -> None:
    with pytest.raises(ConfigError, match="at least"):
        allocate_slots(
            num_blocks=8, num_slots=4, mode="uniform", min_slots_per_block=1, max_slot_multiplier=4.0
        )


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown allocation mode"):
        allocate_slots(
            num_blocks=2, num_slots=4, mode="mystery", min_slots_per_block=1, max_slot_multiplier=4.0
        )


def test_online_plan_clamps_short_block() -> None:
    # A short final block cannot receive more slots than it has tokens.
    allocation = _online_slot_plan(
        [10, 2], 8, min_slots_per_block=1, max_slot_multiplier=4.0
    )
    assert sum(allocation) == 8
    assert allocation[1] <= 2
    assert allocation[0] <= 10


def test_online_plan_rejects_oversized_budget() -> None:
    with pytest.raises(ConfigError, match="cannot give"):
        _online_slot_plan([3, 3], 10, min_slots_per_block=1, max_slot_multiplier=4.0)


def test_online_plan_is_deterministic() -> None:
    first = _online_slot_plan([100, 200, 300], 64, min_slots_per_block=1, max_slot_multiplier=4.0)
    second = _online_slot_plan([100, 200, 300], 64, min_slots_per_block=1, max_slot_multiplier=4.0)
    assert first == second
