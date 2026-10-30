"""Form N scaffold: compressed-prefix planning without a trained compressor.

Form N replaces the far context with ``k`` soft tokens prepended to the frozen model input.
This round implements the *contract and budget planner* only. It deliberately raises
``UnsupportedOperation`` for every data-dependent operation: no compressor weights exist, and
returning a fabricated reconstruction error or read would violate the verification rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sprm.contracts.memory import UnsupportedOperation
from sprm.contracts.types import KVSlice, MemoryStats, WriteReport
from sprm.errors import ConfigError
from sprm.memory.allocator import allocate_slots

MAX_SOFT_TOKEN_BUDGET = 512


@dataclass(frozen=True)
class CompressedPrefixConfig:
    num_layers: int
    kv_width: int
    dtype_bytes: int
    soft_token_budget: int = MAX_SOFT_TOKEN_BUDGET
    min_soft_tokens_per_block: int = 1
    max_slot_multiplier: float = 4.0

    def __post_init__(self) -> None:
        if self.num_layers < 1 or self.kv_width < 1 or self.dtype_bytes < 1:
            raise ConfigError("num_layers, kv_width and dtype_bytes must be positive")
        if not 1 <= self.soft_token_budget <= MAX_SOFT_TOKEN_BUDGET:
            raise ConfigError(
                f"soft_token_budget must lie in [1,{MAX_SOFT_TOKEN_BUDGET}], got {self.soft_token_budget}"
            )


@dataclass(frozen=True)
class PrefixPlan:
    block_ranges: tuple[tuple[int, int], ...]
    soft_tokens_per_block: tuple[int, ...]
    importances: tuple[float, ...]
    total_soft_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_ranges": [list(range_) for range_ in self.block_ranges],
            "soft_tokens_per_block": list(self.soft_tokens_per_block),
            "importances": list(self.importances),
            "total_soft_tokens": self.total_soft_tokens,
        }


def plan_prefix(
    *,
    block_ranges: Sequence[tuple[int, int]],
    budget: int,
    importances: Sequence[float] | None = None,
    min_soft_tokens_per_block: int = 1,
    max_slot_multiplier: float = 4.0,
) -> PrefixPlan:
    """Distribute a fixed soft-token budget over evicted blocks."""
    if budget < 1 or budget > MAX_SOFT_TOKEN_BUDGET:
        raise ConfigError(f"budget must lie in [1,{MAX_SOFT_TOKEN_BUDGET}], got {budget}")
    if len(block_ranges) < 1:
        raise ConfigError("block_ranges must not be empty")
    mode = "effective_rank" if importances is not None else "uniform"
    allocation = allocate_slots(
        num_blocks=len(block_ranges),
        num_slots=budget,
        mode=mode,
        importances=importances,
        min_slots_per_block=min_soft_tokens_per_block,
        max_slot_multiplier=max_slot_multiplier,
    )
    values = tuple(float(x) for x in (importances or [0.0] * len(block_ranges)))
    return PrefixPlan(
        block_ranges=tuple((int(a), int(b)) for a, b in block_ranges),
        soft_tokens_per_block=allocation,
        importances=values,
        total_soft_tokens=int(sum(allocation)),
    )


class CompressedPrefixMemory:
    """Contract-compatible placeholder for Form N.

    The class exists so the interface is exercised and the budget is auditable. It never
    fabricates a read, a write, or a reconstruction error.
    """

    def __init__(self, config: CompressedPrefixConfig, plan: PrefixPlan):
        self.config = config
        self.plan = plan
        self._stats = MemoryStats(
            name="form_n_prefix",
            num_slots=plan.total_soft_tokens,
            key_dim=config.kv_width,
            value_dim=config.kv_width,
            extra={"form": "N", "compressor": "absent"},
        )

    @property
    def stats(self) -> MemoryStats:
        return self._stats

    def reset(self) -> None:
        self._stats.writes = 0
        self._stats.tokens_written = 0

    def write(self, block: KVSlice, *, gain: float | None = None) -> WriteReport:
        raise UnsupportedOperation("Form N has no trained compressor in this round")

    def absorb(self, block: KVSlice, *, gain: float | None = None) -> WriteReport:
        raise UnsupportedOperation("Form N has no trained compressor in this round")

    def read(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise UnsupportedOperation("Form N reads through the frozen model's prefix attention")

    def recon_error(self, exact: KVSlice) -> float:
        raise UnsupportedOperation("Form N has no free validation set; it needs a periodic dual forward")

    def condition(self) -> float:
        raise UnsupportedOperation("Form N has no key matrix to condition")

    def memory_bytes(self) -> int:
        """KV bytes occupied by the planned soft tokens in the frozen model."""
        return int(
            self.plan.total_soft_tokens
            * self.config.num_layers
            * 2
            * self.config.kv_width
            * self.config.dtype_bytes
        )
