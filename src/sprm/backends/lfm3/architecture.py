"""LFM3 architecture inspection and validation.

The model is a tri-hybrid: short-conv, sliding-window attention, and global attention.
Form A only replaces the KV of the global-attention layers; conv and sliding layers stay
native. This module is the single place that decides which layers those are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sprm.errors import BackendError

_ALLOWED = {"conv", "sliding_attention", "full_attention"}


@dataclass(frozen=True)
class LayerPlan:
    kinds: tuple[str, ...]
    global_layers: tuple[int, ...]
    sliding_layers: tuple[int, ...]
    conv_layers: tuple[int, ...]

    @property
    def num_layers(self) -> int:
        return len(self.kinds)


def build_layer_plan(config: Any) -> LayerPlan:
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or not layer_types:
        raise BackendError("LFM3 config has no layer_types; this checkpoint is not supported")
    kinds = tuple(str(t) for t in layer_types)
    unknown = sorted(set(kinds) - _ALLOWED)
    if unknown:
        raise BackendError(f"LFM3 layer_types contain unknown kinds: {unknown}")
    num_layers = int(getattr(config, "num_hidden_layers", len(kinds)))
    if num_layers != len(kinds):
        raise BackendError(
            f"num_hidden_layers={num_layers} does not match len(layer_types)={len(kinds)}"
        )
    globals_ = tuple(i for i, t in enumerate(kinds) if t == "full_attention")
    sliding = tuple(i for i, t in enumerate(kinds) if t == "sliding_attention")
    conv = tuple(i for i, t in enumerate(kinds) if t == "conv")
    if not globals_:
        raise BackendError("LFM3 config has no full_attention layers; Form A has nothing to attach to")
    if not hasattr(config, "num_key_value_heads") or not hasattr(config, "head_dim"):
        raise BackendError("LFM3 config is missing num_key_value_heads/head_dim")
    if int(config.num_attention_heads) % int(config.num_key_value_heads) != 0:
        raise BackendError("num_attention_heads must be divisible by num_key_value_heads")
    if sliding and getattr(config, "sliding_window", None) is None:
        raise BackendError("config has sliding_attention layers but no sliding_window")
    return LayerPlan(kinds=kinds, global_layers=globals_, sliding_layers=sliding, conv_layers=conv)


def plan_as_dict(plan: LayerPlan, config: Any) -> dict[str, Any]:
    return {
        "layer_kinds": list(plan.kinds),
        "global_layers": list(plan.global_layers),
        "sliding_layers": list(plan.sliding_layers),
        "conv_layers": list(plan.conv_layers),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "hidden_size": int(config.hidden_size),
        "num_attention_heads": int(config.num_attention_heads),
        "sliding_window": getattr(config, "sliding_window", None),
        "max_position_embeddings": int(getattr(config, "max_position_embeddings", 0)),
        "vocab_size": int(config.vocab_size),
    }
