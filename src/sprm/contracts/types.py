"""Shared value objects for the SPRM contracts.

The harness never passes raw dicts between layers: every boundary has a typed object so a
shape/position mismatch is caught at construction, not after a 32k-token prefill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

LayerKind = Literal["conv", "sliding_attention", "full_attention"]
AllocationMode = Literal["none", "uniform", "effective_rank", "attention"]


@dataclass(frozen=True)
class BackendInfo:
    kind: str
    model_path: str
    device: str
    dtype: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int | None
    max_position_embeddings: int
    layer_kinds: tuple[LayerKind, ...]
    global_layers: tuple[int, ...]
    sliding_layers: tuple[int, ...]
    conv_layers: tuple[int, ...]
    model_sha256: str | None = None
    tokenizer_sha256: str | None = None
    remote_code_sha256: str | None = None

    @property
    def kv_width(self) -> int:
        """Flattened K (or V) width per token per attention layer."""
        return self.num_kv_heads * self.head_dim

    @property
    def dtype_torch(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            return mapping[self.dtype]
        except KeyError as exc:
            raise ValueError(f"unsupported dtype {self.dtype!r}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "sliding_window": self.sliding_window,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_kinds": list(self.layer_kinds),
            "global_layers": list(self.global_layers),
            "sliding_layers": list(self.sliding_layers),
            "conv_layers": list(self.conv_layers),
            "model_sha256": self.model_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "remote_code_sha256": self.remote_code_sha256,
        }


@dataclass
class KVSlice:
    """Exact KV for a contiguous, chronologically ordered token range.

    Shapes follow the HuggingFace cache convention: ``[batch, kv_heads, tokens, head_dim]``.
    ``positions`` is ``[tokens]`` and always starts at ``start``.
    """

    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    start: int

    def __post_init__(self) -> None:
        if self.keys.shape != self.values.shape:
            raise ValueError(f"K/V shape mismatch: {tuple(self.keys.shape)} vs {tuple(self.values.shape)}")
        if self.keys.ndim != 4:
            raise ValueError(f"expected [B,H,T,D] KV, got {tuple(self.keys.shape)}")
        if self.positions.ndim != 1 or self.positions.numel() != self.keys.shape[-2]:
            raise ValueError("positions must be [tokens] and match the KV token axis")
        if int(self.positions[0].item()) != self.start:
            raise ValueError(f"positions[0]={int(self.positions[0].item())} != start={self.start}")

    @property
    def num_tokens(self) -> int:
        return int(self.keys.shape[-2])

    @property
    def end(self) -> int:
        return self.start + self.num_tokens

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> KVSlice:
        return KVSlice(
            keys=self.keys.to(device=device, dtype=dtype or self.keys.dtype),
            values=self.values.to(device=device, dtype=dtype or self.values.dtype),
            positions=self.positions.to(device=device),
            start=self.start,
        )

    def narrow(self, start: int, length: int) -> KVSlice:
        if start < self.start or start + length > self.end:
            raise ValueError(f"narrow({start},{length}) outside [{self.start},{self.end})")
        lo = start - self.start
        return KVSlice(
            keys=self.keys[..., lo : lo + length, :],
            values=self.values[..., lo : lo + length, :],
            positions=self.positions[lo : lo + length],
            start=start,
        )


@dataclass
class WriteReport:
    tokens: int
    slots_touched: int
    delta_norm: float
    state_norm: float
    trust_scaled: bool
    gain: float
    mean_abs_error: float


@dataclass
class MemoryStats:
    name: str
    num_slots: int
    key_dim: int
    value_dim: int
    writes: int = 0
    tokens_written: int = 0
    last_recon_error: float = float("nan")
    recon_error_ema: float = float("nan")
    condition_number: float = float("nan")
    trust_scaled_count: int = 0
    forgetting_events: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_slots": self.num_slots,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "writes": self.writes,
            "tokens_written": self.tokens_written,
            "last_recon_error": self.last_recon_error,
            "recon_error_ema": self.recon_error_ema,
            "condition_number": self.condition_number,
            "trust_scaled_count": self.trust_scaled_count,
            "forgetting_events": self.forgetting_events,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class Sample:
    sample_id: str
    task_kind: str
    context_length: int
    depth: float
    token_ids: tuple[int, ...]
    question_token_ids: tuple[int, ...]
    gold_answer: str
    gold_token_ids: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_token_ids(self) -> tuple[int, ...]:
        return self.token_ids + self.question_token_ids


@dataclass
class DecodeStep:
    step: int
    token_id: int
    token_text: str
    logprob: float
    entropy: float


@dataclass
class ArmResult:
    arm_id: str
    sample_id: str
    task_kind: str
    context_length: int
    depth: float
    exact_match: bool
    predicted_answer: str
    gold_answer: str
    gold_first_token_logprob: float
    gold_first_token_rank: int
    generated_token_ids: tuple[int, ...]
    generated_text: str
    recon_error_trajectory: tuple[float, ...]
    recon_error_final: float
    condition_number_final: float
    trust_scaled_count: int
    memory_bytes: int
    kv_bytes_per_token: int
    peak_memory_bytes: int
    elapsed_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
