"""High-level LFM3 backend used by the Form A runtime and the evaluation harness."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from sprm.backends.lfm3.cache import CacheController
from sprm.backends.lfm3.loader import build_info, load_model
from sprm.backends.lfm3.rotary import RotaryCodec
from sprm.config.schema import BackendConfig
from sprm.contracts.types import BackendInfo, KVSlice
from sprm.errors import BackendError, CacheInvariantError


@dataclass
class ForwardStats:
    calls: int = 0
    tokens: int = 0
    elapsed_s: float = 0.0

    def add(self, tokens: int, elapsed: float) -> None:
        self.calls += 1
        self.tokens += tokens
        self.elapsed_s += elapsed

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.elapsed_s if self.elapsed_s > 0 else float("nan")


class Lfm3Backend:
    def __init__(self, config: BackendConfig, *, hash_files: bool = False, seed: int | None = None):
        self.config = config
        if seed is not None:
            torch.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
        self.model, self.tokenizer, self.plan = load_model(config)
        self.info: BackendInfo = build_info(config, self.model, self.tokenizer, self.plan, hash_files=hash_files)
        self.rotary = RotaryCodec(
            self.model.model.rotary_emb,
            layer_type="full_attention",
            hidden_size=self.info.hidden_size,
            dtype=self.info.dtype_torch,
        )
        self.stats = ForwardStats()
        self._closed = False
        self._verify_native_path()

    # ------------------------------------------------------------------ setup
    def _verify_native_path(self) -> None:
        cfg = self.model.config
        if getattr(cfg, "_attn_implementation", None) != self.config.attn_implementation:
            raise BackendError(
                f"attention implementation is {getattr(cfg, '_attn_implementation', None)!r}, "
                f"requested {self.config.attn_implementation!r}"
            )
        if self.info.global_layers and self.info.sliding_window is None:
            raise BackendError("sliding layers exist but config.sliding_window is unset")

    def new_cache(self) -> CacheController:
        return CacheController.new(self.model, self.info)

    # ------------------------------------------------------------------ forward
    def _check_open(self) -> None:
        if self._closed:
            raise BackendError("backend is closed")

    def _forward(
        self,
        cache: CacheController,
        token_ids: list[int],
        start_position: int,
        *,
        logits_to_keep: int = 1,
    ) -> torch.Tensor:
        self._check_open()
        if not token_ids:
            raise BackendError("cannot forward an empty token list")
        if start_position < 0:
            raise BackendError(f"start_position must be >= 0, got {start_position}")
        device = self.info.device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        positions = torch.arange(
            start_position, start_position + len(token_ids), dtype=torch.long, device=device
        ).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                position_ids=positions,
                past_key_values=cache.cache,
                use_cache=True,
                logits_to_keep=logits_to_keep,
            )
        elapsed = time.perf_counter() - t0
        self.stats.add(len(token_ids), elapsed)
        return cast(torch.Tensor, out.logits)

    def prefill_chunk(
        self,
        cache: CacheController,
        token_ids: list[int],
        start_position: int,
    ) -> tuple[dict[int, KVSlice], torch.Tensor]:
        """Forward one chunk and return its exact global-layer KV plus the last logits."""
        logits = self._forward(cache, token_ids, start_position, logits_to_keep=1)
        chunks: dict[int, KVSlice] = {}
        for layer_idx in self.info.global_layers:
            chunks[layer_idx] = cache.chunk_kv(layer_idx, start=start_position, length=len(token_ids))
        return chunks, logits[0, -1].float()

    def prefill_exact(
        self,
        token_ids: list[int],
        *,
        chunk_tokens: int,
    ) -> tuple[CacheController, torch.Tensor, list[float]]:
        """Exact prefilling, optionally chunked. No memory is injected."""
        if chunk_tokens <= 0:
            raise BackendError("chunk_tokens must be positive")
        cache = self.new_cache()
        step_times: list[float] = []
        last_logits: torch.Tensor | None = None
        pos = 0
        while pos < len(token_ids):
            chunk = token_ids[pos : pos + chunk_tokens]
            t0 = time.perf_counter()
            _, last_logits = self.prefill_chunk(cache, chunk, pos)
            step_times.append(time.perf_counter() - t0)
            pos += len(chunk)
        if last_logits is None:
            raise BackendError("prefill_exact received no tokens")
        cache.assert_invariants(expected_global_length=len(token_ids))
        return cache, last_logits, step_times

    def decode(self, cache: CacheController, token_id: int, position: int) -> torch.Tensor:
        logits = self._forward(cache, [token_id], position, logits_to_keep=1)
        return logits[0, -1].float()

    # ------------------------------------------------------------------ RoPE
    def unrotate_layer(self, layer_idx: int, keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if layer_idx not in self.info.global_layers:
            raise CacheInvariantError(f"layer {layer_idx} is not a global-attention layer")
        return self.rotary.unrotate(keys, positions)

    def rotate_slots(self, slot_keys: torch.Tensor, slot_positions: torch.Tensor) -> torch.Tensor:
        """``slot_keys`` ``[B, H*D]`` unrotated -> ``[1, H, B, D]`` rotated at ``B`` positions.

        Memory keys are unit-norm over the flattened ``H*D`` vector, while the model's
        ``k_norm`` produces per-head RMS 1 (flattened norm ``sqrt(H*D)``). Slots are therefore
        multiplied by ``sqrt(H*D)`` before injection: a sharp cluster (mean direction close to
        one token) gets a native-scale key, and a diffuse cluster keeps a proportionally
        weaker key. Per-slot renormalization would erase that confidence signal.
        """
        if slot_keys.ndim != 2:
            raise BackendError(f"slot_keys must be [B,H*D], got {tuple(slot_keys.shape)}")
        b, width = slot_keys.shape
        if width != self.info.kv_width:
            raise BackendError(f"slot width {width} != kv_width {self.info.kv_width}")
        shaped = slot_keys.reshape(b, self.info.num_kv_heads, self.info.head_dim).transpose(0, 1).unsqueeze(0)
        shaped = shaped.to(dtype=self.info.dtype_torch)
        rotated = self.rotary.rotate(shaped, slot_positions).float()
        rotated = rotated * (self.info.kv_width**0.5)
        return cast(torch.Tensor, rotated.to(self.info.dtype_torch))

    # ------------------------------------------------------------------ helpers
    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode_tokens(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def chat_template_available(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def render_chat(self, user_content: str) -> str:
        if not self.chat_template_available():
            raise BackendError("tokenizer has no chat template")
        messages = [{"role": "user", "content": user_content}]
        return str(
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    def file_hashes(self) -> dict[str, str]:
        path = Path(self.config.model_path)
        out: dict[str, str] = {}
        for name in ("model.safetensors", "tokenizer.json", "modeling_lfm3.py", "configuration_lfm3.py", "lfm3_conv.py"):
            target = path / name
            if target.is_file():
                digest = hashlib_sha256(target)
                out[name] = digest
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        del self.model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __enter__(self) -> Lfm3Backend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def hashlib_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
