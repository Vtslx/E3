"""Memory contract.

Form A (associative memory) and Form N (compressed prefix) both implement this surface.
Form N raises ``UnsupportedOperation`` for the free-validation methods it cannot provide;
it must never return a fabricated number.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from sprm.contracts.types import KVSlice, MemoryStats, WriteReport


class UnsupportedOperation(NotImplementedError):
    """The memory form cannot provide this operation by construction."""


@runtime_checkable
class MemoryModule(Protocol):
    """Layer-local or model-local memory state.

    Keys and values are ``[B,H,T,D]`` exact KV at construction time. Implementations own
    their internal representation (flattened, per-head, quantized, ...).
    """

    @property
    def stats(self) -> MemoryStats: ...

    def reset(self) -> None: ...

    def write(self, block: KVSlice, *, gain: float | None = None) -> WriteReport:
        """W1: closed-form correction from exact KV (no gradients)."""
        ...

    def absorb(self, block: KVSlice, *, gain: float | None = None) -> WriteReport:
        """W2: compressed absorption of an evicted block (allocation happens here)."""
        ...

    def read(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(values_hat, attention_weights)`` for query keys ``[B,H,T,D]``."""
        ...

    def recon_error(self, exact: KVSlice) -> float:
        """Free validation set: relative squared error of reconstructing ``exact``."""
        ...

    def condition(self) -> float:
        """Condition number of the key matrix, or NaN when not applicable."""
        ...

    def memory_bytes(self) -> int:
        """Resident bytes of the memory state, excluding the exact Tier-1 cache."""
        ...


@runtime_checkable
class SlotAllocator(Protocol):
    def allocate(self, *, num_blocks: int, num_slots: int) -> tuple[int, ...]:
        """Return slots per block; must sum to ``num_slots``."""
        ...
