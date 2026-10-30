"""RoPE codec for the LFM3 global-attention layers.

The KV cache stores post-RoPE keys. An associative memory should store the *content* key,
not a key rotated by an arbitrary absolute position: a slot mixes keys from many positions,
and a mixture of rotations is not a rotation of the mixture. We therefore unrotate exact
keys before writing them into memory, and rotate each slot key with its representative
position when injecting it back into the attention cache.

The transform is the exact inverse of LFM3's ``apply_rotary_pos_emb`` for its zero-tail
partial RoPE: cos/sin are taken from the model's own rotary module (the same bf16 values the
forward pass uses), and the inverse is ``k * cos - rotate_half(k) * sin``.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryCodec:
    """Bidirectional RoPE for one layer type (``full_attention``)."""

    def __init__(self, rotary_module: torch.nn.Module, *, layer_type: str, hidden_size: int, dtype: torch.dtype):
        self.rotary = rotary_module
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.dtype = dtype

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``cos``/``sin`` of shape ``[1, T, head_dim]`` for ``positions`` ``[T]``."""
        if positions.ndim != 1:
            raise ValueError(f"positions must be [T], got {tuple(positions.shape)}")
        device = next(self.rotary.parameters()).device if any(True for _ in self.rotary.parameters()) else positions.device
        pos = positions.to(device=device).unsqueeze(0)
        dummy = torch.zeros(1, 1, self.hidden_size, device=device, dtype=self.dtype)
        cos, sin = self.rotary(dummy, pos, self.layer_type)
        if cos.shape[0] != 1 or cos.shape[1] != positions.numel():
            raise RuntimeError(f"unexpected rotary output shape {tuple(cos.shape)} for {positions.numel()} positions")
        return cos, sin

    def unrotate(self, keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Invert RoPE on ``keys`` ``[B,H,T,D]`` at ``positions`` ``[T]``."""
        cos, sin = self.cos_sin(positions)
        cos = cos.float().unsqueeze(1)
        sin = sin.float().unsqueeze(1)
        out = keys.float() * cos - rotate_half(keys.float()) * sin
        return out.to(keys.dtype)

    def rotate(self, keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to ``keys`` ``[B,H,T,D]`` at ``positions`` ``[T]``."""
        cos, sin = self.cos_sin(positions)
        cos = cos.float().unsqueeze(1)
        sin = sin.float().unsqueeze(1)
        out = keys.float() * cos + rotate_half(keys.float()) * sin
        return out.to(keys.dtype)
