"""Form A runtime: online compressed prefill and compressed decode.

Protocol implemented here (see docs/FORM_A.md):

1. Prefill the prompt in chunks. Once the prefix exceeds sink+window, each forwarded chunk
   is preceded by absorbing the outgoing block into Tier-2 and rebuilding the global K/V as
   ``[sink | memory slots | window]``. Every cached state is therefore conditioned on the
   compressed context the arm will actually see.
2. Sink and the last window stay exact. Tier-2 covers only the evicted region.
3. Decode greedily. In the W1 arm, every generated token's exact KV is unrotated and
   written back with gain ``eta*g``. The exact window is the free validation set.

The post-hoc ``build_memory`` path (full exact prefill, then build) is retained only as a
diagnostic: it cannot produce valid hidden states for a compressed arm, because the layers
above the first dropped global layer already saw the dropped tokens.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.backends.lfm3.cache import CacheController
from sprm.config.schema import ArmConfig, ExperimentConfig
from sprm.contracts.types import DecodeStep
from sprm.errors import ConfigError, MemoryInvariantError
from sprm.memory.allocator import allocate_slots, effective_rank
from sprm.memory.associative import AssociativeMemory, AssociativeMemoryConfig


# ---------------------------------------------------------------------- build
@dataclass
class MemoryBuild:
    memories: dict[int, AssociativeMemory]
    block_ranges: list[tuple[int, int]]
    slots_per_block: tuple[int, ...]
    slot_offsets: tuple[int, ...]
    importances: tuple[float, ...]
    build_seconds: float
    condition_checks: dict[int, list[float]] = field(default_factory=dict)
    forgetting_events: int = 0

    @property
    def total_slots(self) -> int:
        return int(sum(self.slots_per_block))

    def memory_bytes(self) -> int:
        return int(sum(m.memory_bytes() for m in self.memories.values()))

    def clone(self) -> MemoryBuild:
        """Copy the memory state but share the immutable block plan."""
        return MemoryBuild(
            memories={layer: memory.clone() for layer, memory in self.memories.items()},
            block_ranges=list(self.block_ranges),
            slots_per_block=self.slots_per_block,
            slot_offsets=self.slot_offsets,
            importances=self.importances,
            build_seconds=self.build_seconds,
            condition_checks={layer: list(values) for layer, values in self.condition_checks.items()},
            forgetting_events=self.forgetting_events,
        )


@dataclass
class ExactWindow:
    """Rolling exact post-RoPE window, used for the free validation set and W1."""

    keys: dict[int, torch.Tensor]
    values: dict[int, torch.Tensor]
    positions: torch.Tensor
    capacity: int

    def append_step(
        self,
        keys: dict[int, torch.Tensor],
        values: dict[int, torch.Tensor],
        position: int,
    ) -> None:
        """Append one token's exact KV for every global layer, sharing one position entry.

        The position vector is per token, not per layer; appending inside the layer loop
        duplicates it and misaligns the reconstruction-error probe.
        """
        if set(keys) != set(self.keys) or set(values) != set(self.values):
            raise MemoryInvariantError("append_step must cover exactly the stored global layers")
        for layer_idx in keys:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], keys[layer_idx]], dim=-2)[
                ..., -self.capacity :, :
            ]
            self.values[layer_idx] = torch.cat([self.values[layer_idx], values[layer_idx]], dim=-2)[
                ..., -self.capacity :, :
            ]
        self.positions = torch.cat(
            [self.positions, torch.tensor([position], device=self.positions.device, dtype=self.positions.dtype)]
        )[-self.capacity :]


def _flatten_keys(keys: torch.Tensor, info: Any) -> torch.Tensor:
    # [1,H,T,D] -> [T,H*D]
    return keys.transpose(1, 2).reshape(keys.shape[-2], info.num_kv_heads * info.head_dim)


def _flatten_values(values: torch.Tensor, info: Any) -> torch.Tensor:
    return values.transpose(1, 2).reshape(values.shape[-2], info.num_kv_heads * info.head_dim)


def _unrotated_flat(
    backend: Lfm3Backend, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    unrotated = backend.unrotate_layer(layer_idx, keys, positions)
    return _flatten_keys(unrotated, backend.info), _flatten_values(values, backend.info)


def build_memory(
    backend: Lfm3Backend,
    full_cache: CacheController,
    experiment: ExperimentConfig,
    *,
    context_length: int,
    full_length: int,
    allocation: str | None = None,
) -> MemoryBuild | None:
    """Build one memory per global layer from the evicted context region.

    Returns ``None`` when the prompt is short enough that nothing is evicted.
    """
    info = backend.info
    cfg = experiment.tier2
    sink = experiment.tier1.sink_tokens
    window = experiment.tier1.window_tokens
    evict_start = sink
    evict_end = full_length - window
    if evict_end <= evict_start:
        return None
    block = cfg.block_tokens
    ranges = [(start, min(start + block, evict_end)) for start in range(evict_start, evict_end, block)]

    t0 = time.perf_counter()
    # Importance proxy: mean effective rank of the post-RoPE key matrix over global layers.
    importances: list[float] = []
    for start, end in ranges:
        per_layer: list[float] = []
        for layer_idx in info.global_layers:
            layer = full_cache.layers[layer_idx]
            keys = layer.keys[..., start:end, :]
            per_layer.append(effective_rank(_flatten_keys(keys, info)))
        importances.append(float(sum(per_layer) / len(per_layer)))

    allocation_mode = allocation or cfg.allocation
    if allocation_mode == "none":
        allocation_mode = "uniform"
    per_block = allocate_slots(
        num_blocks=len(ranges),
        num_slots=cfg.num_slots,
        mode=allocation_mode,
        importances=importances,
        min_slots_per_block=cfg.min_slots_per_block,
        max_slot_multiplier=cfg.max_slot_multiplier,
    )
    offsets: list[int] = []
    cursor = 0
    for count in per_block:
        offsets.append(cursor)
        cursor += count
    if cursor != cfg.num_slots:
        raise MemoryInvariantError(f"slot plan consumed {cursor} slots, expected {cfg.num_slots}")

    memories: dict[int, AssociativeMemory] = {}
    for layer_idx in info.global_layers:
        memories[layer_idx] = AssociativeMemory(
            AssociativeMemoryConfig(
                name=f"form_a_layer_{layer_idx}",
                num_slots=cfg.num_slots,
                key_dim=info.kv_width,
                value_dim=info.kv_width,
                read_temperature=cfg.read_temperature,
                write_gain=cfg.write_gain,
                gate=cfg.gate,
                forgetting=cfg.forgetting,
                trust_region_eps=cfg.trust_region_eps,
                key_l2_normalize=cfg.key_l2_normalize,
                kmeans_iters=cfg.kmeans_iters,
                refine_passes=cfg.refine_passes,
            ),
            device=info.device,
            seed=experiment.seed + layer_idx,
        )

    condition_checks: dict[int, list[float]] = {i: [] for i in info.global_layers}
    forgetting_events = 0
    for block_id, (start, end) in enumerate(ranges):
        positions = torch.arange(start, end, device=info.device, dtype=torch.long)
        for layer_idx in info.global_layers:
            layer = full_cache.layers[layer_idx]
            keys_post = layer.keys[..., start:end, :]
            values = layer.values[..., start:end, :]
            keys, flat_values = _unrotated_flat(backend, layer_idx, keys_post, values, positions)
            memories[layer_idx].write_slots(
                keys,
                flat_values,
                slot_start=offsets[block_id],
                slot_count=per_block[block_id],
                positions=positions,
                block_id=block_id,
            )
        if (block_id + 1) % cfg.condition_every_blocks == 0:
            for layer_idx, memory in memories.items():
                cond = memory.condition()
                condition_checks[layer_idx].append(cond)
                if cond > 1.0e6:
                    memory.apply_forgetting(cfg.forgetting)
                    forgetting_events += 1

    elapsed = time.perf_counter() - t0
    return MemoryBuild(
        memories=memories,
        block_ranges=ranges,
        slots_per_block=per_block,
        slot_offsets=tuple(offsets),
        importances=tuple(importances),
        build_seconds=elapsed,
        condition_checks=condition_checks,
        forgetting_events=forgetting_events,
    )


# ---------------------------------------------------------------------- online prefill
@dataclass
class OnlinePrefill:
    cache: CacheController
    logits: torch.Tensor
    build: MemoryBuild | None
    prefill_seconds: float
    evictions: int
    block_ranges: list[tuple[int, int]]


def _online_slot_plan(
    counts: Sequence[int],
    budget: int,
    *,
    min_slots_per_block: int,
    max_slot_multiplier: float,
) -> tuple[int, ...]:
    """Slots per block, never exceeding the block's token count.

    The online prefill evicts the same chunks it forwards, so the last block may be short.
    Uniform allocation can over-assign slots to that block; we clamp and redistribute the
    surplus to blocks that still have room.
    """
    if not counts or any(count < 1 for count in counts):
        raise ConfigError("every evicted block must contain at least one token")
    if sum(counts) < budget:
        raise ConfigError(f"cannot give {budget} slots to {sum(counts)} evicted tokens")
    base = allocate_slots(
        num_blocks=len(counts),
        num_slots=budget,
        mode="uniform",
        importances=None,
        min_slots_per_block=min_slots_per_block,
        max_slot_multiplier=max_slot_multiplier,
    )
    allocation = [min(base[index], counts[index]) for index in range(len(counts))]
    allocated = sum(allocation)
    if allocated < budget:
        order = sorted(range(len(counts)), key=lambda index: counts[index] - allocation[index], reverse=True)
        cursor = 0
        while allocated < budget:
            index = order[cursor % len(order)]
            if allocation[index] < counts[index]:
                allocation[index] += 1
                allocated += 1
            cursor += 1
            if cursor > 1_000_000:
                raise ConfigError("online slot allocation did not converge")
    if any(value < 1 for value in allocation):
        raise ConfigError("every block must receive at least one slot")
    return tuple(int(value) for value in allocation)


def prefill_online(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    token_ids: Sequence[int],
    *,
    use_memory: bool,
    allocation: str = "uniform",
) -> OnlinePrefill:
    """Prefill a prompt while applying Tier-1 eviction and Tier-2 writes online.

    This is the faithful protocol. Prefilling the whole prompt and then slicing the KV is
    not equivalent: every layer above the first dropped global layer computed its window
    states with the dropped tokens visible, so the sliced cache carries their influence.
    Here each chunk is forwarded after the evicted blocks have been removed and the memory
    slots injected, so every cached state is conditioned on exactly the context the arm
    will see at decode time.

    ``block_tokens`` must equal ``prefill_chunk_tokens``: the online loop evicts exactly
    the chunk it just forwarded. The last block may be shorter when the prompt does not
    divide evenly; its slot budget is clamped to its token count.
    """
    info = backend.info
    cfg = experiment.tier2
    sink = experiment.tier1.sink_tokens
    window = experiment.tier1.window_tokens
    chunk = experiment.tier1.prefill_chunk_tokens
    block = cfg.block_tokens
    tokens = [int(t) for t in token_ids]
    n = len(tokens)
    if n < 2:
        raise ConfigError("online prefill needs at least two tokens")
    if chunk != block:
        raise ConfigError(
            f"online prefill requires block_tokens == prefill_chunk_tokens, got {block} != {chunk}"
        )
    if window % block != 0:
        raise ConfigError(f"window {window} must be a multiple of block {block}")
    if use_memory and allocation != "uniform":
        raise ConfigError(
            "online prefill requires allocation='uniform'; importance-based allocation needs a "
            "first pass and is only available through the post-hoc `form-a build` diagnostic"
        )

    t0 = time.perf_counter()
    cache = backend.new_cache()
    initial = min(n, sink + window)
    pos = 0
    logits: torch.Tensor | None = None
    while pos < initial:
        step = min(chunk, initial - pos)
        _, logits = backend.prefill_chunk(cache, tokens[pos : pos + step], pos)
        pos += step
    if logits is None:
        raise ConfigError("online prefill forwarded no tokens")
    if n <= sink + window:
        return OnlinePrefill(
            cache=cache,
            logits=logits,
            build=None,
            prefill_seconds=time.perf_counter() - t0,
            evictions=0,
            block_ranges=[],
        )

    # The block evicted before forwarding the chunk at ``p`` is exactly [p-window, p-window+step).
    ranges: list[tuple[int, int]] = []
    p = initial
    while p < n:
        step = min(chunk, n - p)
        ranges.append((p - window, min(p - window + step, n - window)))
        p += step
    counts = [end - start for start, end in ranges]
    # A short prompt can evict fewer tokens than the configured budget; spending more slots
    # than tokens would make every slot a single token and is not the intended regime.
    slot_budget = min(cfg.num_slots, sum(counts)) if use_memory else 0
    slots = (
        _online_slot_plan(
            counts,
            slot_budget,
            min_slots_per_block=cfg.min_slots_per_block,
            max_slot_multiplier=cfg.max_slot_multiplier,
        )
        if use_memory
        else tuple()
    )
    offsets: list[int] = []
    cursor = 0
    for value in slots:
        offsets.append(cursor)
        cursor += value
    if use_memory and cursor != slot_budget:
        raise MemoryInvariantError(f"online slot plan consumed {cursor}, expected {slot_budget}")

    build: MemoryBuild | None = None
    if use_memory:
        memories: dict[int, AssociativeMemory] = {}
        for layer_idx in info.global_layers:
            memories[layer_idx] = AssociativeMemory(
                AssociativeMemoryConfig(
                    name=f"form_a_layer_{layer_idx}",
                    num_slots=cfg.num_slots,
                    key_dim=info.kv_width,
                    value_dim=info.kv_width,
                    read_temperature=cfg.read_temperature,
                    write_gain=cfg.write_gain,
                    gate=cfg.gate,
                    forgetting=cfg.forgetting,
                    trust_region_eps=cfg.trust_region_eps,
                    key_l2_normalize=cfg.key_l2_normalize,
                    kmeans_iters=cfg.kmeans_iters,
                    refine_passes=cfg.refine_passes,
                ),
                device=info.device,
                seed=experiment.seed + layer_idx,
            )
        build = MemoryBuild(
            memories=memories,
            block_ranges=list(ranges),
            slots_per_block=tuple(slots),
            slot_offsets=tuple(offsets),
            importances=tuple(0.0 for _ in ranges),
            build_seconds=0.0,
            condition_checks={layer: [] for layer in info.global_layers},
            forgetting_events=0,
        )

    position = initial
    block_index = 0
    importances: list[float] = [0.0] * len(ranges)
    while position < n:
        step = min(chunk, n - position)
        if position >= sink + window:
            start, end = ranges[block_index]
            count = end - start
            if count != step:
                raise MemoryInvariantError(f"block {block_index} size {count} != forward chunk {step}")
            initialized = offsets[block_index] if use_memory else 0
            window_start = sink + initialized
            rank_values: list[float] = []
            for layer_idx in info.global_layers:
                layer = cache.layers[layer_idx]
                keys_post = layer.keys[..., window_start : window_start + count, :]
                values = layer.values[..., window_start : window_start + count, :]
                if int(keys_post.shape[-2]) != count:
                    raise MemoryInvariantError(
                        f"window holds {keys_post.shape[-2]} tokens at layer {layer_idx}, expected {count}"
                    )
                positions = torch.arange(start, end, device=info.device, dtype=torch.long)
                flat_keys, flat_values = _unrotated_flat(backend, layer_idx, keys_post, values, positions)
                rank_values.append(effective_rank(flat_keys))
                if build is not None:
                    build.memories[layer_idx].write_slots(
                        flat_keys,
                        flat_values,
                        slot_start=offsets[block_index],
                        slot_count=slots[block_index],
                        positions=positions,
                        block_id=block_index,
                    )
            importances[block_index] = float(sum(rank_values) / len(rank_values))
            if build is not None:
                build.importances = tuple(importances)
                initialized += slots[block_index]
                if (block_index + 1) % cfg.condition_every_blocks == 0:
                    for layer_idx, memory in build.memories.items():
                        condition = memory.condition()
                        build.condition_checks[layer_idx].append(condition)
                        if condition > 1.0e6:
                            memory.apply_forgetting(cfg.forgetting)
                            build.forgetting_events += 1
            effective = assemble_effective_global(
                backend,
                cache,
                build,
                experiment,
                full_length=int(cache.global_effective_length()),
                compressed=True,
                force_evict=True,
                inject_slots=initialized if use_memory else 0,
                drop_head=count,
            )
            cache.rebuild_global(effective)
            block_index += 1
        _, logits = backend.prefill_chunk(cache, tokens[position : position + step], position)
        position += step

    if block_index != len(ranges):
        raise MemoryInvariantError(f"absorbed {block_index} blocks, expected {len(ranges)}")
    if build is not None:
        build.build_seconds = time.perf_counter() - t0
    return OnlinePrefill(
        cache=cache,
        logits=logits,
        build=build,
        prefill_seconds=time.perf_counter() - t0,
        evictions=block_index,
        block_ranges=list(ranges),
    )


# ---------------------------------------------------------------------- cache assembly
def assemble_effective_global(
    backend: Lfm3Backend,
    full_cache: CacheController,
    build: MemoryBuild | None,
    experiment: ExperimentConfig,
    *,
    full_length: int,
    compressed: bool,
    force_evict: bool = False,
    inject_slots: int | None = None,
    drop_head: int = 0,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Build the effective global KV set from the current cache.

    ``compressed`` says whether this arm applies Tier-1 eviction at all. ``build`` says
    whether the evicted region is represented by memory slots. The two are independent:
    the streaming arm is compressed with ``build=None`` and keeps only sink + window.

    ``force_evict`` is for the online prefill, where the cache length is already the
    effective length and the usual ``full_length > sink+window`` test would not fire.
    ``inject_slots`` limits injection to the memory slots initialized so far; unwritten
    slots are not injected, so they cannot distract attention. ``drop_head`` removes the
    oldest ``drop_head`` window tokens, which are the block just absorbed.
    """
    info = backend.info
    sink = experiment.tier1.sink_tokens
    window = experiment.tier1.window_tokens
    if drop_head < 0 or drop_head > window:
        raise MemoryInvariantError(f"drop_head {drop_head} outside [0,{window}]")
    effective: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in info.global_layers:
        layer = full_cache.layers[layer_idx]
        cached = int(layer.keys.shape[-2])
        if full_length > cached:
            raise MemoryInvariantError(
                f"full_length {full_length} exceeds cached length {cached} at layer {layer_idx}"
            )
        evicts = compressed and (force_evict or full_length > sink + window)
        if not evicts:
            # Nothing leaves the exact set (C0, or a prompt shorter than sink+window).
            effective[layer_idx] = (
                layer.keys[..., :full_length, :].contiguous(),
                layer.values[..., :full_length, :].contiguous(),
            )
            continue
        key_parts = [layer.keys[..., :sink, :]]
        value_parts = [layer.values[..., :sink, :]]
        if build is not None:
            memory = build.memories[layer_idx]
            count = memory.num_slots if inject_slots is None else max(0, min(inject_slots, memory.num_slots))
            if count > 0:
                slot_keys = backend.rotate_slots(memory.M_k[:count], memory.slot_positions[:count])
                slot_values = (
                    memory.M_v[:count]
                    .reshape(count, info.num_kv_heads, info.head_dim)
                    .transpose(0, 1)
                    .unsqueeze(0)
                    .to(dtype=info.dtype_torch)
                )
                key_parts.append(slot_keys)
                value_parts.append(slot_values)
        tail_len = window - drop_head
        if tail_len > 0:
            key_parts.append(layer.keys[..., -tail_len:, :])
            value_parts.append(layer.values[..., -tail_len:, :])
        effective[layer_idx] = (
            torch.cat(key_parts, dim=-2).contiguous(),
            torch.cat(value_parts, dim=-2).contiguous(),
        )
    return effective


def refresh_memory_slots(
    backend: Lfm3Backend,
    cache: CacheController,
    build: MemoryBuild,
    *,
    sink: int,
    count: int | None = None,
) -> None:
    """Rewrite the injected slot rows from the current memory state.

    W1 updates ``M_k``/``M_v``, but the model reads the memory through the KV cache. Without
    this refresh the write is invisible to the next forward, so C3 and C4 would be identical
    by construction. The slot range is ``[sink, sink+count)`` in every global layer.
    """
    info = backend.info
    total = build.total_slots if count is None else min(count, build.total_slots)
    if total <= 0:
        return
    for layer_idx in info.global_layers:
        memory = build.memories[layer_idx]
        slot_keys = backend.rotate_slots(memory.M_k[:total], memory.slot_positions[:total])
        slot_values = (
            memory.M_v[:total]
            .reshape(total, info.num_kv_heads, info.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(dtype=info.dtype_torch)
        )
        layer = cache.layers[layer_idx]
        if int(layer.keys.shape[-2]) < sink + total:
            raise MemoryInvariantError(
                f"cache holds {layer.keys.shape[-2]} tokens at layer {layer_idx}, "
                f"cannot refresh slots [{sink},{sink + total})"
            )
        layer.keys[..., sink : sink + total, :] = slot_keys
        layer.values[..., sink : sink + total, :] = slot_values


def write_last_token_to_memory(
    backend: Lfm3Backend,
    cache: CacheController,
    build: MemoryBuild,
    *,
    position: int,
    gain: float,
) -> None:
    """W1 without the slot refresh: write the token just appended to ``cache``."""
    info = backend.info
    step_keys = {i: cache.layers[i].keys[..., -1:, :] for i in info.global_layers}
    step_values = {i: cache.layers[i].values[..., -1:, :] for i in info.global_layers}
    positions = torch.tensor([position], device=info.device, dtype=torch.long)
    for layer_idx in info.global_layers:
        flat_key, flat_value = _unrotated_flat(
            backend, layer_idx, step_keys[layer_idx], step_values[layer_idx], positions
        )
        build.memories[layer_idx].write(flat_key, flat_value, gain=gain)


def write_token_to_memory(
    backend: Lfm3Backend,
    cache: CacheController,
    build: MemoryBuild,
    *,
    position: int,
    gain: float,
    sink: int,
) -> None:
    """W1 plus refresh: the next forward sees the corrected slots."""
    write_last_token_to_memory(backend, cache, build, position=position, gain=gain)
    refresh_memory_slots(backend, cache, build, sink=sink)


# ---------------------------------------------------------------------- decode
@dataclass
class DecodeOutcome:
    steps: list[DecodeStep]
    first_logits: torch.Tensor
    recon_error_trajectory: list[float]
    condition_final: float
    trust_scaled_count: int
    elapsed_s: float
    peak_memory_bytes: int


def decode_arm(
    backend: Lfm3Backend,
    cache: CacheController,
    experiment: ExperimentConfig,
    arm: ArmConfig,
    *,
    first_logits: torch.Tensor,
    start_position: int,
    max_new_tokens: int,
    exact_window: ExactWindow,
    build: MemoryBuild | None,
    eostoken_id: int | None,
) -> DecodeOutcome:
    info = backend.info
    t0 = time.perf_counter()
    steps: list[DecodeStep] = []
    recon_trajectory: list[float] = []
    logits = first_logits
    generated: list[int] = []
    gain = experiment.tier2.write_gain * experiment.tier2.gate
    peak = _mps_peak_bytes()

    for step in range(max_new_tokens):
        probs = torch.softmax(logits, dim=-1)
        prob, token = torch.topk(probs, k=1)
        token_id = int(token.item())
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        steps.append(
            DecodeStep(
                step=step,
                token_id=token_id,
                token_text=backend.decode_tokens([token_id]),
                logprob=float(torch.log(prob.clamp_min(1e-12)).item()),
                entropy=entropy,
            )
        )
        generated.append(token_id)
        if eostoken_id is not None and token_id == eostoken_id:
            break
        if step == max_new_tokens - 1:
            break

        position = start_position + step
        new_logits = backend.decode(cache, token_id, position)
        # The forward appended this token's exact KV at the end of every global layer.
        if build is not None:
            step_keys = {i: cache.layers[i].keys[..., -1:, :] for i in info.global_layers}
            step_values = {i: cache.layers[i].values[..., -1:, :] for i in info.global_layers}
            exact_window.append_step(step_keys, step_values, position)
            if arm.writeback:
                write_token_to_memory(
                    backend,
                    cache,
                    build,
                    position=position,
                    gain=gain,
                    sink=experiment.tier1.sink_tokens,
                )
            recon_trajectory.append(_recon_error(backend, build, exact_window))
        logits = new_logits
        peak = max(peak, _mps_peak_bytes())

    if build is not None and not recon_trajectory:
        recon_trajectory.append(_recon_error(backend, build, exact_window))
    condition = float("nan")
    trust_scaled = 0
    if build is not None:
        condition = max((m.condition() for m in build.memories.values()), default=float("nan"))
        trust_scaled = int(sum(m.stats.trust_scaled_count for m in build.memories.values()))
    return DecodeOutcome(
        steps=steps,
        first_logits=first_logits,
        recon_error_trajectory=recon_trajectory,
        condition_final=condition,
        trust_scaled_count=trust_scaled,
        elapsed_s=time.perf_counter() - t0,
        peak_memory_bytes=peak,
    )


def _recon_error(backend: Lfm3Backend, build: MemoryBuild, exact_window: ExactWindow) -> float:
    info = backend.info
    errors: list[float] = []
    for layer_idx in info.global_layers:
        keys_post = exact_window.keys[layer_idx]
        values = exact_window.values[layer_idx]
        positions = exact_window.positions[-keys_post.shape[-2] :]
        keys, flat_values = _unrotated_flat(backend, layer_idx, keys_post, values, positions)
        errors.append(build.memories[layer_idx].recon_error(keys, flat_values))
    return float(sum(errors) / len(errors)) if errors else float("nan")


def _mps_peak_bytes() -> int:
    if not torch.backends.mps.is_available():
        return 0
    try:
        return int(torch.mps.driver_allocated_memory())
    except Exception:
        return 0


def exact_window_from_cache(
    backend: Lfm3Backend, cache: CacheController, *, capacity: int, full_length: int
) -> ExactWindow:
    info = backend.info
    keys: dict[int, torch.Tensor] = {}
    values: dict[int, torch.Tensor] = {}
    for layer_idx in info.global_layers:
        layer = cache.layers[layer_idx]
        keys[layer_idx] = layer.keys[..., -capacity:, :].clone()
        values[layer_idx] = layer.values[..., -capacity:, :].clone()
    start = full_length - int(next(iter(keys.values())).shape[-2])
    positions = torch.arange(start, full_length, device=info.device, dtype=torch.long)
    return ExactWindow(keys=keys, values=values, positions=positions, capacity=capacity)


def cache_footprint_bytes(
    backend: Lfm3Backend,
    cache: CacheController,
    build: MemoryBuild | None,
    *,
    context_length: int,
) -> tuple[int, int]:
    """Return ``(cache_bytes, per_token_bytes)`` for the resident attention state."""
    info = backend.info
    total = 0
    element = 2 if info.dtype in {"bfloat16", "float16"} else 4
    for layer_idx in info.global_layers + info.sliding_layers:
        layer = cache.layers[layer_idx]
        if getattr(layer, "is_initialized", False):
            total += int(layer.keys.numel() + layer.values.numel()) * element
    for layer_idx in info.conv_layers:
        layer = cache.layers[layer_idx]
        for state in getattr(layer, "conv_states", {}).values():
            if state is not None:
                total += int(state.numel()) * element
    if build is not None:
        total += build.memory_bytes()
    per_token = int(total / max(context_length, 1))
    return total, per_token
