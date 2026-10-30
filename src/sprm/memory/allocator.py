"""Slot budget allocation across eviction blocks.

The only online-available signal in this round is a predictability proxy: the effective
rank of a block's key matrix. A block whose keys span many directions needs more slots to
keep reconstruction error low; a low-rank block can share a mean slot. The design's
future-attention term requires the proposer (not in this round) and is recorded as an open
limitation, not simulated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from sprm.errors import ConfigError


def effective_rank(keys: torch.Tensor) -> float:
    """Participation ratio of the singular value spectrum of ``keys`` [T,D]."""
    if keys.ndim != 2 or keys.shape[0] == 0:
        return 0.0
    matrix = keys.detach().float().cpu()
    try:
        singular = torch.linalg.svdvals(matrix)
    except RuntimeError:
        return float(matrix.shape[0])
    # Participation ratio of the squared spectrum: (Σs²)² / Σs⁴.
    squared = singular.square()
    numerator = squared.sum().square()
    denominator = squared.square().sum()
    if denominator <= 0:
        return 1.0
    return float((numerator / denominator).clamp_min(1.0).item())


def allocate_slots(
    *,
    num_blocks: int,
    num_slots: int,
    mode: str,
    importances: Sequence[float] | None = None,
    min_slots_per_block: int = 1,
    max_slot_multiplier: float = 4.0,
) -> tuple[int, ...]:
    if num_blocks < 1 or num_slots < 1:
        raise ConfigError("num_blocks and num_slots must be positive")
    if num_blocks * min_slots_per_block > num_slots:
        raise ConfigError(
            f"cannot give every one of {num_blocks} blocks at least {min_slots_per_block} slots "
            f"from a budget of {num_slots}"
        )
    if mode == "none":
        raise ConfigError("allocation mode 'none' has no memory slots to allocate")
    if mode == "uniform" or mode == "attention":
        base = num_slots // num_blocks
        remainder = num_slots % num_blocks
        result = [base + (1 if i < remainder else 0) for i in range(num_blocks)]
        return tuple(result)
    if mode != "effective_rank":
        raise ConfigError(f"unknown allocation mode {mode!r}")
    if importances is None or len(importances) != num_blocks:
        raise ConfigError("effective_rank allocation requires one importance per block")
    values = [max(float(x), 0.0) for x in importances]
    total = sum(values)
    if total <= 0:
        values = [1.0] * num_blocks
        total = float(num_blocks)
    weights = [x / total for x in values]
    average = num_slots / num_blocks
    cap = max(min_slots_per_block, math.ceil(max_slot_multiplier * average))
    raw = [num_slots * w for w in weights]
    alloc = [max(min_slots_per_block, min(cap, math.floor(x))) for x in raw]
    allocated = sum(alloc)
    if allocated < num_slots:
        # Hand out the remainder to the largest fractional entitlements first.
        order = sorted(range(num_blocks), key=lambda i: (raw[i] - math.floor(raw[i])), reverse=True)
        i = 0
        while allocated < num_slots:
            idx = order[i % num_blocks]
            if alloc[idx] < cap:
                alloc[idx] += 1
                allocated += 1
            i += 1
            if i > 10000:
                raise ConfigError("allocation did not converge while distributing slots")
    elif allocated > num_slots:
        order = sorted(range(num_blocks), key=lambda i: alloc[i], reverse=True)
        i = 0
        while allocated > num_slots:
            idx = order[i % num_blocks]
            if alloc[idx] > min_slots_per_block:
                alloc[idx] -= 1
                allocated -= 1
            i += 1
            if i > 10000:
                raise ConfigError("allocation did not converge while trimming slots")
    if sum(alloc) != num_slots:
        raise ConfigError(f"allocation sum {sum(alloc)} != budget {num_slots}")
    return tuple(alloc)
