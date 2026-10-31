"""Prompt-lookup speculative decoding with an all-or-nothing commit.

The draft is deterministic: find the last earlier occurrence of the final ``n`` tokens and
propose the following ``gamma`` tokens. The target verifies the whole proposal in one
forward. If every proposed token matches the target's greedy token, the proposal is kept
and one bonus token is fed; otherwise the cache is rolled back to the pre-proposal state
(global/sliding crop + conv-state restore) and the normal greedy token is decoded.

All-or-nothing avoids a partial-prefix conv-state rollback, which the model cannot express:
the short-conv layers update from a whole chunk at once. The generated sequence is still
exactly the target's greedy sequence; only full proposals speed up.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.backends.lfm3.cache import CacheController
from sprm.config.schema import ArmConfig, ExperimentConfig
from sprm.contracts.types import DecodeStep
from sprm.runtime.form_a import (
    ExactWindow,
    MemoryBuild,
    _unrotated_flat,
    refresh_memory_slots,
)


@dataclass
class PromptLookup:
    """Deterministic n-gram draft over prompt + generated history."""

    history: list[int]
    n: int = 3
    gamma: int = 4
    proposed: int = 0

    def propose(self) -> list[int]:
        if self.gamma < 1 or len(self.history) < self.n + 1:
            return []
        suffix = self.history[-self.n :]
        start = len(self.history) - self.n - 1
        for index in range(start, -1, -1):
            if self.history[index : index + self.n] == suffix:
                proposal = self.history[index + self.n : index + self.n + self.gamma]
                if proposal:
                    self.proposed += len(proposal)
                    return list(proposal)
        return []

    def update(self, tokens: list[int]) -> None:
        self.history.extend(int(token) for token in tokens)


@dataclass
class SpeculativeOutcome:
    steps: list[DecodeStep]
    generated: list[int]
    proposed_tokens: int
    accepted_tokens: int
    full_accepts: int
    verify_calls: int
    fallback_calls: int
    elapsed_s: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


def _record_token(
    backend: Lfm3Backend,
    cache: CacheController,
    experiment: ExperimentConfig,
    exact_window: ExactWindow,
    build: MemoryBuild | None,
    arm: ArmConfig,
    *,
    index: int,
    position: int,
) -> None:
    info = backend.info
    step_keys = {i: cache.layers[i].keys[..., index : index + 1, :] for i in info.global_layers}
    step_values = {i: cache.layers[i].values[..., index : index + 1, :] for i in info.global_layers}
    exact_window.append_step(step_keys, step_values, position)
    if build is None or not arm.writeback:
        return
    gain = experiment.tier2.write_gain * experiment.tier2.gate
    positions = torch.tensor([position], device=info.device, dtype=torch.long)
    for layer_idx in info.global_layers:
        flat_key, flat_value = _unrotated_flat(
            backend, layer_idx, step_keys[layer_idx], step_values[layer_idx], positions
        )
        build.memories[layer_idx].write(flat_key, flat_value, gain=gain)
    refresh_memory_slots(
        backend, cache, build, sink=experiment.tier1.sink_tokens
    )


def _decode_step(backend: Lfm3Backend, logits: torch.Tensor, token: int, step: int) -> DecodeStep:
    probabilities = torch.softmax(logits.float(), dim=-1)
    probability = float(probabilities[token].item())
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1.0e-12))).sum().item())
    return DecodeStep(
        step=step,
        token_id=int(token),
        token_text=backend.decode_tokens([int(token)]),
        logprob=math.log(max(probability, 1.0e-12)),
        entropy=entropy,
    )


def decode_speculative(
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
    proposer: PromptLookup,
) -> SpeculativeOutcome:
    """Greedy speculative decode. Output tokens are identical to greedy decoding."""
    t0 = time.perf_counter()
    steps: list[DecodeStep] = []
    generated: list[int] = []
    current = first_logits
    position = start_position
    proposed_tokens = 0
    accepted_tokens = 0
    full_accepts = 0
    verify_calls = 0
    fallback_calls = 0

    def emit(token: int, logits: torch.Tensor) -> bool:
        nonlocal position
        steps.append(_decode_step(backend, logits, token, len(steps)))
        generated.append(int(token))
        proposer.update([int(token)])
        return not (eostoken_id is not None and int(token) == eostoken_id)

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        proposal = proposer.propose()[:remaining]
        if proposal:
            verify_calls += 1
            snapshot = cache.snapshot_conv()
            cache.activate_sliding_past()
            verify = backend.decode_many(cache, proposal, position)
            logits_sequence = [current] + [verify[index] for index in range(len(proposal) - 1)]
            matches = 0
            for index in range(len(proposal)):
                target = int(torch.argmax(logits_sequence[index]).item())
                if target != proposal[index]:
                    break
                matches += 1
            if matches != len(proposal):
                # Reject the whole batch: crop KV, restore conv state, resume normal decoding.
                cache.crop_tail(len(proposal))
                cache.restore_conv(snapshot)
                cache.finish_speculation()
                fallback_calls += 1
                token = int(torch.argmax(current).item())
                if not emit(token, current):
                    break
                current = backend.decode(cache, token, position)
                if build is not None:
                    _record_token(
                        backend, cache, experiment, exact_window, build, arm, index=-1, position=position
                    )
                position += 1
                continue

            # Full proposal accepted.
            cache.finish_speculation()
            for index, token in enumerate(proposal):
                if not emit(token, logits_sequence[index]):
                    break
                cache_index = -(len(proposal) - index)
                if build is not None:
                    _record_token(
                        backend,
                        cache,
                        experiment,
                        exact_window,
                        build,
                        arm,
                        index=cache_index,
                        position=position + index,
                    )
            accepted_tokens += len(proposal)
            proposed_tokens += len(proposal)
            full_accepts += 1
            position += len(proposal)
            if len(generated) >= max_new_tokens:
                break
            if eostoken_id is not None and generated and generated[-1] == eostoken_id:
                break
            bonus = int(torch.argmax(verify[-1]).item())
            if not emit(bonus, verify[-1]):
                break
            current = backend.decode(cache, bonus, position)
            if build is not None:
                _record_token(
                    backend, cache, experiment, exact_window, build, arm, index=-1, position=position
                )
            position += 1
            continue

        # No proposal: one normal greedy token.
        proposed_tokens += 0
        token = int(torch.argmax(current).item())
        if not emit(token, current):
            break
        current = backend.decode(cache, token, position)
        if build is not None:
            _record_token(
                backend, cache, experiment, exact_window, build, arm, index=-1, position=position
            )
        position += 1

    return SpeculativeOutcome(
        steps=steps,
        generated=generated,
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        full_accepts=full_accepts,
        verify_calls=verify_calls,
        fallback_calls=fallback_calls,
        elapsed_s=time.perf_counter() - t0,
        diagnostics={
            "acceptance_rate": (
                accepted_tokens / proposed_tokens if proposed_tokens else float("nan")
            ),
            "mean_accepted": (
                accepted_tokens / full_accepts if full_accepts else float("nan")
            ),
            "tokens": float(len(generated)),
        },
    )
