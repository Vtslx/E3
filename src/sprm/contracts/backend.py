"""Backend contract.

A backend exposes only what the memory and evaluation layers need: exact KV, hidden
states when requested, and a rebuildable cache. No layer of SPRM imports a model class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from sprm.contracts.types import BackendInfo, KVSlice


@runtime_checkable
class ModelBackend(Protocol):
    @property
    def info(self) -> BackendInfo: ...

    def encode(self, token_ids: list[int], *, start_position: int = 0) -> KVSlice:
        """Teacher-forced forward of a contiguous chunk starting at ``start_position``.

        The backend keeps its native cache (conv state, sliding KV) between calls. The
        returned slice contains the chunk's exact post-RoPE KV for global and sliding
        attention layers, indexed by layer in ``info`` order.
        """
        ...

    def kv_for_layer(self, chunk: KVSlice, layer_idx: int) -> KVSlice: ...

    def decode_next(self, token_id: int, *, position: int) -> torch.Tensor:
        """Return next-token logits ``[vocab]`` for a single token."""
        ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class CacheBackend(Protocol):
    """Rebuildable cache surface used by the Form A runtime."""

    def rebuild(self, *, global_kv: dict[int, KVSlice], sliding_kv: dict[int, KVSlice]) -> None: ...

    def effective_length(self) -> int: ...
