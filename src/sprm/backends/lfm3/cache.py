"""Rebuildable KV cache for LFM3.

We never reimplement attention. The model keeps its native ``DynamicCache``; we only
replace the *global-attention* layers' K/V with an effective set ``[sink | memory slots |
window]``. Sliding-window layers and short-conv states are untouched, because they are
already bounded and exact.

Invariants:
- All global layers have the same effective length (the cache reports it to the mask
  builder, and the builder sizes the causal mask from it).
- Sliding layers keep their native ``cumulative_length`` and their native last
  ``sliding_window-1`` keys.
- Rebuilding never changes conv or sliding state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from sprm.contracts.types import BackendInfo, KVSlice
from sprm.errors import CacheInvariantError


@dataclass
class CacheController:
    """Owns a native ``DynamicCache`` and exposes only the operations the harness needs."""

    cache: DynamicCache
    info: BackendInfo

    @classmethod
    def new(cls, model: Any, info: BackendInfo) -> CacheController:
        return cls(cache=DynamicCache(config=model.config), info=info)

    @property
    def layers(self) -> Any:
        """Inner cache layers.

        transformers types them as a union that includes non-KV layers; the harness has
        already checked the layer plan, so it accesses them dynamically.
        """
        return self.cache.layers

    # ------------------------------------------------------------------ cloning
    def clone(self) -> CacheController:
        return CacheController(cache=copy.deepcopy(self.cache), info=self.info)

    # ------------------------------------------------------------------ reads
    def global_effective_length(self) -> int:
        lengths = [int(self.layers[i].keys.shape[-2]) for i in self.info.global_layers]
        if not lengths:
            return 0
        if len(set(lengths)) != 1:
            raise CacheInvariantError(f"global layers have different effective lengths: {lengths}")
        return lengths[0]

    def layer_length(self, layer_idx: int) -> int:
        layer = self.layers[layer_idx]
        if layer_idx in self.info.conv_layers:
            return 0
        return int(layer.keys.shape[-2])

    def full_kv(self, layer_idx: int) -> KVSlice:
        """Exact, chronologically ordered KV of a global layer (no memory injection)."""
        if layer_idx not in self.info.global_layers:
            raise CacheInvariantError(f"layer {layer_idx} is not a global-attention layer")
        layer = self.layers[layer_idx]
        if not getattr(layer, "is_initialized", False):
            raise CacheInvariantError(f"layer {layer_idx} is not initialized")
        n = int(layer.keys.shape[-2])
        positions = torch.arange(n, device=layer.keys.device)
        return KVSlice(keys=layer.keys, values=layer.values, positions=positions, start=0)

    def chunk_kv(self, layer_idx: int, *, start: int, length: int) -> KVSlice:
        """Last ``length`` entries of a global layer, labelled with logical ``start``."""
        if layer_idx not in self.info.global_layers:
            raise CacheInvariantError(f"layer {layer_idx} is not a global-attention layer")
        layer = self.layers[layer_idx]
        n = int(layer.keys.shape[-2])
        if length > n:
            raise CacheInvariantError(f"chunk length {length} exceeds cache length {n} at layer {layer_idx}")
        keys = layer.keys[..., -length:, :]
        values = layer.values[..., -length:, :]
        positions = torch.arange(start, start + length, device=keys.device)
        return KVSlice(keys=keys, values=values, positions=positions, start=start)

    # ------------------------------------------------------------------ writes
    def rebuild_global(self, effective: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Replace the global layers' K/V with the effective set.

        ``effective[layer_idx] = (keys, values)`` with shape ``[1,H,E,D]``. The sliding and
        conv layers are not touched.
        """
        if set(effective) != set(self.info.global_layers):
            missing = sorted(set(self.info.global_layers) - set(effective))
            extra = sorted(set(effective) - set(self.info.global_layers))
            raise CacheInvariantError(f"rebuild must cover exactly the global layers; missing={missing} extra={extra}")
        lengths: dict[int, int] = {}
        for layer_idx, (keys, values) in effective.items():
            if keys.shape != values.shape:
                raise CacheInvariantError(f"layer {layer_idx}: K/V shape mismatch")
            if keys.ndim != 4:
                raise CacheInvariantError(f"layer {layer_idx}: expected [1,H,E,D], got {tuple(keys.shape)}")
            if keys.shape[0] != 1:
                raise CacheInvariantError(f"layer {layer_idx}: batch must be 1, got {keys.shape[0]}")
            if keys.dtype != self.info.dtype_torch:
                raise CacheInvariantError(f"layer {layer_idx}: dtype {keys.dtype} != model dtype")
            layer = self.layers[layer_idx]
            layer.keys = keys.contiguous()
            layer.values = values.contiguous()
            layer.is_initialized = True
            lengths[layer_idx] = int(keys.shape[-2])
        if len(set(lengths.values())) != 1:
            raise CacheInvariantError(f"rebuilt global layers have different lengths: {lengths}")

    # ------------------------------------------------------------------ validation
    def assert_invariants(self, *, expected_global_length: int | None = None) -> None:
        if expected_global_length is not None:
            actual = self.global_effective_length()
            if actual != expected_global_length:
                raise CacheInvariantError(
                    f"global effective length {actual} != expected {expected_global_length}"
                )
        for layer_idx in self.info.global_layers:
            layer = self.layers[layer_idx]
            if not getattr(layer, "is_initialized", False):
                raise CacheInvariantError(f"global layer {layer_idx} not initialized")
            if layer.keys.shape != layer.values.shape:
                raise CacheInvariantError(f"global layer {layer_idx} K/V mismatch")
        for layer_idx in self.info.sliding_layers:
            layer = self.layers[layer_idx]
            if getattr(layer, "is_initialized", False):
                window = int(self.info.sliding_window or 0)
                if window and int(layer.keys.shape[-2]) > window:
                    raise CacheInvariantError(
                        f"sliding layer {layer_idx} holds {layer.keys.shape[-2]} keys, window={window}"
                    )
