from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sprm.contracts.memory import UnsupportedOperation
from sprm.contracts.types import KVSlice
from sprm.errors import ConfigError
from sprm.memory.prefix import (
    CompressedPrefixConfig,
    CompressedPrefixMemory,
    plan_prefix,
)
from sprm.runtime.form_n import build_prefix_scaffold, describe_prefix


def _info(num_layers: int = 6, kv_width: int = 512, dtype: str = "bfloat16"):
    return SimpleNamespace(
        num_layers=num_layers,
        kv_width=kv_width,
        dtype=dtype,
        num_kv_heads=4,
        head_dim=128,
    )


def _experiment():
    return SimpleNamespace(
        tier1=SimpleNamespace(sink_tokens=128, window_tokens=2048),
        tier2=SimpleNamespace(
            block_tokens=512,
            num_slots=512,
            min_slots_per_block=1,
            max_slot_multiplier=4.0,
        ),
    )


def test_plan_prefix_spends_exactly_the_budget() -> None:
    plan = plan_prefix(block_ranges=[(128, 640), (640, 1152), (1152, 1552)], budget=32)
    assert plan.total_soft_tokens == 32
    assert len(plan.soft_tokens_per_block) == 3
    assert sum(plan.soft_tokens_per_block) == 32
    assert min(plan.soft_tokens_per_block) >= 1


def test_plan_prefix_rejects_oversized_budget() -> None:
    with pytest.raises(ConfigError):
        plan_prefix(block_ranges=[(0, 2)], budget=513)


def test_plan_prefix_uses_importances() -> None:
    plan = plan_prefix(
        block_ranges=[(0, 512), (512, 1024)],
        budget=16,
        importances=[1.0, 8.0],
    )
    assert sum(plan.soft_tokens_per_block) == 16
    assert plan.soft_tokens_per_block[1] > plan.soft_tokens_per_block[0]


def _block(tokens: int = 4) -> KVSlice:
    return KVSlice(
        keys=torch.zeros(1, 4, tokens, 128),
        values=torch.zeros(1, 4, tokens, 128),
        positions=torch.arange(tokens),
        start=0,
    )


def test_form_n_memory_never_fabricates_data() -> None:
    config = CompressedPrefixConfig(num_layers=6, kv_width=512, dtype_bytes=2, soft_token_budget=64)
    plan = plan_prefix(block_ranges=[(128, 640)], budget=64)
    memory = CompressedPrefixMemory(config, plan)
    block = _block()
    for operation in (
        lambda: memory.write(block),
        lambda: memory.absorb(block),
        lambda: memory.read(torch.zeros(1, 512)),
        lambda: memory.recon_error(block),
        lambda: memory.condition(),
    ):
        with pytest.raises(UnsupportedOperation):
            operation()
    assert memory.stats.extra["form"] == "N"
    assert memory.memory_bytes() == 64 * 6 * 2 * 512 * 2


def test_build_prefix_scaffold_plans_evicted_blocks() -> None:
    build = build_prefix_scaffold(
        _info(), _experiment(), context_length=8192, full_length=8208
    )
    assert build is not None
    assert build.evict_start == 128
    assert build.evict_end == 8208 - 2048
    assert build.total_soft_tokens == 512
    assert sum(build.plan.soft_tokens_per_block) == 512
    assert build.memory.memory_bytes() == 512 * 6 * 2 * 512 * 2
    description = describe_prefix(build)
    assert description["status"] == "planned-not-verified"
    assert "No compressor checkpoint" in str(description["note"])


def test_build_prefix_scaffold_returns_none_without_eviction() -> None:
    assert build_prefix_scaffold(_info(), _experiment(), context_length=512, full_length=600) is None


def test_compressed_prefix_config_rejects_oversized_budget() -> None:
    with pytest.raises(ConfigError):
        CompressedPrefixConfig(num_layers=6, kv_width=512, dtype_bytes=2, soft_token_budget=1024)
