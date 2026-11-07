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
from sprm.runtime.draft_heads import DraftHeadProposer
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
class ReservoirLookup:
    """Frequency-ranked n-gram draft over prompt + generated history.

    Prompt-lookup takes the continuation after the *most recent* occurrence of the current
    suffix. The reservoir collects up to ``max_occurrences`` earlier occurrences and proposes
    the majority continuation at each step. On natural text, where one suffix has several
    continuations, the majority is a cheaper estimate than the last one; on repetitive text
    the two proposers coincide.
    """

    history: list[int]
    n: int = 3
    gamma: int = 4
    max_occurrences: int = 64
    proposed: int = 0

    def _matches(self) -> list[int]:
        if self.gamma < 1 or len(self.history) < self.n + 1:
            return []
        suffix = self.history[-self.n :]
        matches: list[int] = []
        for index in range(len(self.history) - self.n - 1, -1, -1):
            if self.history[index : index + self.n] == suffix:
                matches.append(index)
                if len(matches) >= self.max_occurrences:
                    break
        return matches

    def propose(self) -> list[int]:
        positions = self._matches()
        if not positions:
            return []
        proposal: list[int] = []
        length = len(self.history)
        for step in range(self.gamma):
            candidates = [
                self.history[pos + self.n + step]
                for pos in positions
                if pos + self.n + step < length
            ]
            if not candidates:
                break
            best = candidates[0]
            best_count = 0
            for token in candidates:
                count = candidates.count(token)
                if count > best_count:
                    best = token
                    best_count = count
            proposal.append(best)
        if proposal:
            self.proposed += len(proposal)
        return proposal

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
    partial_accepts: int = 0
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
    proposer: PromptLookup | ReservoirLookup | DraftHeadProposer,
    partial_accept: bool = False,
) -> SpeculativeOutcome:
    """Greedy speculative decode. Output tokens are identical to greedy decoding.

    ``partial_accept=False`` keeps the original all-or-nothing commit. When enabled, a
    mismatching proposal still commits its matching prefix: the KV for the whole proposal is
    cropped, the conv state is restored from the pre-verify snapshot, and the accepted prefix
    is re-fed as one chunk so the conv state after ``k`` tokens is rebuilt exactly. The next
    token is then the target's argmax at the last accepted position (a free bonus), so the
    output remains the greedy sequence.
    """
    t0 = time.perf_counter()
    steps: list[DecodeStep] = []
    generated: list[int] = []
    current = first_logits
    position = start_position
    proposed_tokens = 0
    accepted_tokens = 0
    full_accepts = 0
    partial_accepts = 0
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
            proposed_tokens += len(proposal)
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
            if partial_accept and 0 < matches < len(proposal):
                # Commit the matching prefix. The conv state after k tokens is rebuilt by
                # re-feeding the prefix as one chunk from the pre-verify snapshot.
                cache.crop_tail(len(proposal))
                cache.restore_conv(snapshot)
                cache.finish_speculation()
                accepted = proposal[:matches]
                refeed = backend.decode_many(cache, accepted, position)
                committed = 0
                for index, token in enumerate(accepted):
                    if not emit(token, logits_sequence[index]):
                        break
                    cache_index = -(matches - index)
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
                    committed += 1
                accepted_tokens += committed
                partial_accepts += 1
                position += committed
                if committed < len(accepted) or len(generated) >= max_new_tokens:
                    break
                if eostoken_id is not None and generated and generated[-1] == eostoken_id:
                    break
                bonus = int(torch.argmax(refeed[-1]).item())
                if not emit(bonus, refeed[-1]):
                    break
                current = backend.decode(cache, bonus, position)
                if build is not None:
                    _record_token(
                        backend, cache, experiment, exact_window, build, arm, index=-1, position=position
                    )
                position += 1
                continue

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
        partial_accepts=partial_accepts,
        elapsed_s=time.perf_counter() - t0,
        diagnostics={
            "acceptance_rate": (
                accepted_tokens / proposed_tokens if proposed_tokens else float("nan")
            ),
            "mean_accepted": (
                accepted_tokens / full_accepts if full_accepts else float("nan")
            ),
            "partial_accepts": float(partial_accepts),
            "tokens": float(len(generated)),
        },
    )
