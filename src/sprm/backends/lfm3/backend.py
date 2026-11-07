"""High-level LFM3 backend used by the Form A runtime and the evaluation harness."""

from __future__ import annotations

import time
from collections.abc import Sequence
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


class QueryProbe:
    """Capture the question-token queries during an exact prefill.

    The queries are taken after ``q_norm`` and before RoPE. They are rotated at their absolute
    positions before being compared with cached post-RoPE keys.
    """

    def __init__(self, backend: Lfm3Backend, layer_indices: tuple[int, ...], n_question: int):
        self.backend = backend
        self.n_question = max(1, int(n_question))
        self.current_start = 0
        self._captured: dict[int, tuple[torch.Tensor, int]] = {}
        self._handles: list[Any] = []
        for layer_idx in layer_indices:
            module = backend.model.model.layers[layer_idx].self_attn.q_norm
            self._handles.append(module.register_forward_hook(self._make_hook(layer_idx)))

    def _make_hook(self, layer_idx: int) -> Any:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            self._captured[layer_idx] = (output.detach(), int(self.current_start))

        return hook

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def queries(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer ``([1,H,n,D] post-RoPE query, positions [n])``."""
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx, (query, start) in self._captured.items():
            tokens = int(query.shape[1])
            count = min(self.n_question, tokens)
            last = query[:, tokens - count :, :, :].transpose(1, 2)
            positions = torch.arange(
                start + tokens - count, start + tokens, device=query.device, dtype=torch.long
            )
            rotated = self.backend.rotary.rotate(last, positions)
            out[layer_idx] = (rotated, positions)
        return out


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
        self.query_probe: QueryProbe | None = None
        self.last_hidden: torch.Tensor | None = None
        self._hidden_handle = self.model.lm_head.register_forward_pre_hook(self._capture_hidden)
        self._closed = False
        self._verify_native_path()

    def _capture_hidden(self, module: Any, inputs: Any) -> None:
        """Keep the last position's pre-LM-head hidden state for draft heads."""
        hidden = inputs[0]
        if hidden.ndim == 3 and hidden.shape[1] > 0:
            self.last_hidden = hidden[:, -1, :].detach().float()

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
        if self.query_probe is not None:
            self.query_probe.current_start = start_position
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

    def decode_many(
        self, cache: CacheController, token_ids: Sequence[int], position: int
    ) -> torch.Tensor:
        """Teacher-forced forward of several tokens; returns logits ``[tokens, vocab]``."""
        if not token_ids:
            raise BackendError("decode_many needs at least one token")
        logits = self._forward(cache, [int(t) for t in token_ids], position, logits_to_keep=len(token_ids))
        return logits[0].float()

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

    def block_attention_importances(
        self,
        cache: CacheController,
        block_ranges: Sequence[tuple[int, int]],
        queries: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> list[float]:
        """Attention mass from the last question query to each evicted block.

        Averaged over heads and global layers. The allocation uses these masses as the
        future-query importance term; they replace the uniform W2 budget.
        """
        if not block_ranges or not queries:
            return []
        scale = float(self.info.head_dim) ** 0.5
        per_layer: list[list[float]] = []
        for layer_idx, (query, _) in queries.items():
            keys = cache.layers[layer_idx].keys
            last = query[:, :, -1, :].float()
            # GQA: average the query-head group that maps to each KV head.
            group = int(self.info.num_attention_heads) // int(self.info.num_kv_heads)
            if group > 1:
                last = last.reshape(
                    last.shape[0], self.info.num_kv_heads, group, last.shape[-1]
                ).mean(dim=2)
            logits = torch.einsum("bhd,bhld->bhl", last, keys.float()) / scale
            attention = torch.softmax(logits, dim=-1)[0]
            masses = [
                float(attention[:, start:end].sum(dim=-1).mean().item())
                for start, end in block_ranges
            ]
            per_layer.append(masses)
        if not per_layer:
            return []
        length = len(block_ranges)
        return [
            float(sum(layer[index] for layer in per_layer) / len(per_layer))
            for index in range(length)
        ]

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
        self._hidden_handle.remove()
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
