"""Experiment runner: one online compressed prefill per arm family.

Arms that drop context must be prefilled with that context already dropped. A single exact
prefill followed by KV slicing is invalid: every layer above the first dropped global layer
already saw the dropped tokens, so its window states carry their influence. The runner
therefore prefills each family separately:

* exact (C0): the full native cache;
* streaming (C1): online sink+window prefill;
* memory (C3/C4): online sink+slots+window prefill, shared by both arms and cloned.

The prefill's own last-token logits are the first honest next-token distribution for that
arm; no replay is needed.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.config.schema import ExperimentConfig
from sprm.contracts.types import ArmResult, Sample
from sprm.eval.metrics import CandidateScore, extract_code, rank_candidates
from sprm.eval.niah import sample_prompt_token_ids
from sprm.logging import JsonLogger
from sprm.runtime.form_a import (
    MemoryBuild,
    cache_footprint_bytes,
    decode_arm,
    exact_window_from_cache,
    prefill_online,
    refresh_memory_slots,
    write_last_token_to_memory,
)


@dataclass
class ArmRun:
    arm_id: str
    result: ArmResult
    score: CandidateScore | None
    generated_code: str | None
    generated_correct: bool | None
    effective_length: int
    prefill_s: float
    build_s: float
    score_s: float
    decode_s: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleRun:
    sample: Sample
    arms: dict[str, ArmRun]
    prefill_s: float
    build_s: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample": {
                "sample_id": self.sample.sample_id,
                "task_kind": self.sample.task_kind,
                "context_length": self.sample.context_length,
                "depth": self.sample.depth,
                "gold_answer": self.sample.gold_answer,
                "prompt_tokens": len(self.sample.full_token_ids),
                "metadata": self.sample.metadata,
            },
            "arms": {arm_id: run.as_dict() for arm_id, run in self.arms.items()},
            "prefill_s": self.prefill_s,
            "build_s": self.build_s,
            "error": self.error,
        }


def _error_result(
    *,
    arm_id: str,
    sample: Sample,
    message: str,
    prefill_s: float = 0.0,
    build_s: float = 0.0,
    score_s: float = 0.0,
    decode_s: float = 0.0,
) -> ArmRun:
    result = ArmResult(
        arm_id=arm_id,
        sample_id=sample.sample_id,
        task_kind=sample.task_kind,
        context_length=sample.context_length,
        depth=sample.depth,
        exact_match=False,
        predicted_answer="",
        gold_answer=sample.gold_answer,
        gold_first_token_logprob=float("nan"),
        gold_first_token_rank=0,
        generated_token_ids=(),
        generated_text="",
        recon_error_trajectory=(),
        recon_error_final=float("nan"),
        condition_number_final=float("nan"),
        trust_scaled_count=0,
        memory_bytes=0,
        kv_bytes_per_token=0,
        peak_memory_bytes=0,
        elapsed_s=prefill_s + build_s + score_s + decode_s,
        diagnostics={"error": message},
    )
    return ArmRun(
        arm_id=arm_id,
        result=result,
        score=None,
        generated_code=None,
        generated_correct=False,
        effective_length=0,
        prefill_s=prefill_s,
        build_s=build_s,
        score_s=score_s,
        decode_s=decode_s,
        error=message,
    )


def _score_candidates(
    backend: Lfm3Backend,
    cache: Any,
    logits: torch.Tensor,
    sample: Sample,
    *,
    experiment: ExperimentConfig,
    build: MemoryBuild | None,
    writeback: bool,
) -> CandidateScore:
    """Teacher-forced log-probability of every candidate code.

    The prompt ends after "Answer: "; candidates are scored as continuations from cloned
    caches, so the model never sees them as options and their order cannot bias the result.

    For a W1 arm the fed candidate token is written into a **clone** of the memory and the
    slot rows are refreshed before the next token is scored. This isolates W1: without the
    write and refresh, C3 and C4 would have identical scores by construction.
    """
    candidates = list(sample.metadata["options"])
    gold = str(sample.metadata["code"])
    start = len(sample.full_token_ids)
    gain = experiment.tier2.write_gain * experiment.tier2.gate
    sink = experiment.tier1.sink_tokens
    logprobs: dict[str, float] = {}
    for code in candidates:
        token_ids = backend.tokenize(code, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"candidate {code!r} tokenizes to nothing")
        clone = cache.clone()
        candidate_build = build.clone() if (writeback and build is not None) else None
        current = logits
        total = 0.0
        position = start
        for token_id in token_ids:
            total += float(torch.log_softmax(current.float(), dim=-1)[token_id].item())
            if candidate_build is not None:
                # W1 needs the token's exact KV, which only a forward produces. Use a scratch
                # clone to produce it, write it, refresh the scoring cache's slots, then feed
                # the token again so the next token is scored under the corrected memory.
                scratch = clone.clone()
                backend.decode(scratch, token_id, position)
                write_last_token_to_memory(
                    backend, scratch, candidate_build, position=position, gain=gain
                )
                refresh_memory_slots(backend, clone, candidate_build, sink=sink)
                current = backend.decode(clone, token_id, position)
                del scratch
            else:
                current = backend.decode(clone, token_id, position)
            position += 1
        logprobs[code] = total
        del clone, candidate_build
    return rank_candidates(gold, candidates, logprobs)


def _finish_arm(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    sample: Sample,
    arm_id: str,
    cache: Any,
    logits: torch.Tensor,
    build: MemoryBuild | None,
    *,
    prefill_s: float,
    build_s: float,
    eos_token_id: int | None,
    new_tokens: int,
) -> ArmRun:
    """Decode one arm from an already-prepared cache and score its first distribution."""
    arm = experiment.arms[arm_id]
    try:
        window = exact_window_from_cache(
            backend,
            cache,
            capacity=experiment.tier1.window_tokens,
            full_length=len(sample.full_token_ids),
        )
        # Score candidates before generation: the clone must see the prompt, not the
        # model's own answer tokens appended to the cache by decode_arm.
        t_score = time.perf_counter()
        score = _score_candidates(
            backend,
            cache,
            logits,
            sample,
            experiment=experiment,
            build=build,
            writeback=arm.writeback,
        )
        score_s = time.perf_counter() - t_score
        t0 = time.perf_counter()
        outcome = decode_arm(
            backend,
            cache,
            experiment,
            arm,
            first_logits=logits,
            start_position=len(sample.full_token_ids),
            max_new_tokens=new_tokens,
            exact_window=window,
            build=build,
            eostoken_id=eos_token_id,
        )
        decode_s = time.perf_counter() - t0
        generated_text = backend.decode_tokens([step.token_id for step in outcome.steps])
        generated_code = extract_code(generated_text)
        cache_bytes, per_token = cache_footprint_bytes(
            backend, cache, build, context_length=sample.context_length
        )
        memory_bytes = build.memory_bytes() if build is not None else 0
        recon_final = (
            float(outcome.recon_error_trajectory[-1]) if outcome.recon_error_trajectory else float("nan")
        )
        build_diag: dict[str, Any] = {}
        if build is not None:
            build_diag = {
                "block_ranges": [list(pair) for pair in build.block_ranges],
                "slots_per_block": list(build.slots_per_block),
                "slot_offsets": list(build.slot_offsets),
                "importances": list(build.importances),
                "condition_checks": {str(k): list(v) for k, v in build.condition_checks.items()},
                "forgetting_events": int(build.forgetting_events),
                "build_seconds": float(build.build_seconds),
            }
        result = ArmResult(
            arm_id=arm_id,
            sample_id=sample.sample_id,
            task_kind=sample.task_kind,
            context_length=sample.context_length,
            depth=sample.depth,
            exact_match=bool(score.correct),
            predicted_answer=score.predicted_code,
            gold_answer=sample.gold_answer,
            gold_first_token_logprob=float(score.gold_logprob),
            gold_first_token_rank=int(score.gold_rank),
            generated_token_ids=tuple(int(step.token_id) for step in outcome.steps),
            generated_text=generated_text,
            recon_error_trajectory=tuple(float(x) for x in outcome.recon_error_trajectory),
            recon_error_final=recon_final,
            condition_number_final=float(outcome.condition_final),
            trust_scaled_count=int(outcome.trust_scaled_count),
            memory_bytes=int(memory_bytes),
            kv_bytes_per_token=int(per_token),
            peak_memory_bytes=int(outcome.peak_memory_bytes),
            elapsed_s=float(prefill_s + build_s + score_s + decode_s),
            diagnostics={
                "candidate_logprobs": dict(score.logprobs),
                "candidate_probabilities": dict(score.probabilities),
                "gold_candidate_rank": int(score.gold_rank),
                "gold_candidate_probability": float(score.gold_probability),
                "generated_code": generated_code,
                "generated_correct": bool(generated_code == sample.gold_answer),
                "expected_answer_text": sample.metadata.get("code"),
                "options": sample.metadata.get("options"),
                "correct_index": sample.metadata.get("correct_index"),
                "owner": sample.metadata.get("owner"),
                "attribute": sample.metadata.get("attribute"),
                "context_token_sha256": sample.metadata.get("context_token_sha256"),
                "needle_position": sample.metadata.get("needle_position"),
                "distance_from_end": sample.metadata.get("distance_from_end"),
                "effective_length": int(cache.global_effective_length()),
                "cache_bytes": int(cache_bytes),
                "prefill_s": float(prefill_s),
                "build_s": float(build_s),
                "score_s": float(score_s),
                "decode_s": float(decode_s),
                "step_entropy_mean": (
                    float(sum(step.entropy for step in outcome.steps) / len(outcome.steps))
                    if outcome.steps
                    else float("nan")
                ),
                "build": build_diag,
            },
        )
        return ArmRun(
            arm_id=arm_id,
            result=result,
            score=score,
            generated_code=generated_code,
            generated_correct=(generated_code == sample.gold_answer),
            effective_length=int(cache.global_effective_length()),
            prefill_s=prefill_s,
            build_s=build_s,
            score_s=score_s,
            decode_s=decode_s,
        )
    except Exception as exc:
        return _error_result(
            arm_id=arm_id,
            sample=sample,
            message=f"{type(exc).__name__}: {exc}",
            prefill_s=prefill_s,
            build_s=build_s,
        )


def run_sample(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    sample: Sample,
    arm_ids: Sequence[str],
    *,
    max_new_tokens: int | None = None,
) -> SampleRun:
    """Prefill each arm family with its own context, then decode every arm."""
    prompt = sample_prompt_token_ids(sample)
    if len(prompt) < 2:
        raise ValueError("prompt must contain at least two tokens")
    eos_token_id = getattr(backend.tokenizer, "eos_token_id", None)
    new_tokens = experiment.runtime.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
    if new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    exact_arms = [arm_id for arm_id in arm_ids if not experiment.arms[arm_id].compressed]
    stream_arms = [
        arm_id
        for arm_id in arm_ids
        if experiment.arms[arm_id].compressed and not experiment.arms[arm_id].memory
    ]
    memory_arms = [arm_id for arm_id in arm_ids if experiment.arms[arm_id].memory]
    allocations = {experiment.arms[arm_id].allocation for arm_id in memory_arms}
    if len(allocations) > 1:
        raise ValueError(f"all memory arms must share one allocation mode, got {sorted(allocations)}")

    arms: dict[str, ArmRun] = {}
    prefill_total = 0.0
    build_total = 0.0

    if exact_arms:
        t0 = time.perf_counter()
        cache, logits, _ = backend.prefill_exact(prompt, chunk_tokens=experiment.tier1.prefill_chunk_tokens)
        prefill_s = time.perf_counter() - t0
        prefill_total += prefill_s
        for arm_id in exact_arms:
            arm_cache = cache.clone()
            arms[arm_id] = _finish_arm(
                backend,
                experiment,
                sample,
                arm_id,
                arm_cache,
                logits,
                None,
                prefill_s=prefill_s,
                build_s=0.0,
                eos_token_id=eos_token_id,
                new_tokens=new_tokens,
            )
            del arm_cache
        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if stream_arms:
        online = prefill_online(backend, experiment, prompt, use_memory=False)
        prefill_total += online.prefill_seconds
        for arm_id in stream_arms:
            arm_cache = online.cache.clone()
            arms[arm_id] = _finish_arm(
                backend,
                experiment,
                sample,
                arm_id,
                arm_cache,
                online.logits,
                None,
                prefill_s=online.prefill_seconds,
                build_s=0.0,
                eos_token_id=eos_token_id,
                new_tokens=new_tokens,
            )
            del arm_cache
        del online
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if memory_arms:
        online = prefill_online(
            backend,
            experiment,
            prompt,
            use_memory=True,
            allocation=allocations.pop() if allocations else "uniform",
        )
        build_s = float(online.build.build_seconds) if online.build is not None else 0.0
        prefill_s = max(0.0, online.prefill_seconds - build_s)
        prefill_total += prefill_s
        build_total += build_s
        for arm_id in memory_arms:
            arm_cache = online.cache.clone()
            arm_build = online.build.clone() if online.build is not None else None
            arms[arm_id] = _finish_arm(
                backend,
                experiment,
                sample,
                arm_id,
                arm_cache,
                online.logits,
                arm_build,
                prefill_s=prefill_s,
                build_s=build_s,
                eos_token_id=eos_token_id,
                new_tokens=new_tokens,
            )
            del arm_cache, arm_build
        del online
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    missing = [arm_id for arm_id in arm_ids if arm_id not in arms]
    if missing:
        raise ValueError(f"arms were not executed: {missing}")
    return SampleRun(sample=sample, arms=arms, prefill_s=prefill_total, build_s=build_total)


def run_experiment(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    samples: Sequence[Sample],
    arm_ids: Sequence[str],
    *,
    logger: JsonLogger,
    max_new_tokens: int | None = None,
    archive: Any | None = None,
) -> list[SampleRun]:
    unknown = [arm_id for arm_id in arm_ids if arm_id not in experiment.arms]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; configured={sorted(experiment.arms)}")
    runs: list[SampleRun] = []
    for sample in samples:
        started = time.perf_counter()
        run = run_sample(
            backend,
            experiment,
            sample,
            arm_ids,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.perf_counter() - started
        runs.append(run)
        if archive is not None:
            archive.write_json(f"artifacts/samples/{sample.sample_id}.json", run.as_dict())
        logger.info(
            "sample_done",
            sample_id=sample.sample_id,
            kind=sample.task_kind,
            context_length=sample.context_length,
            depth=sample.depth,
            elapsed_s=round(elapsed, 2),
            arms={
                arm_id: {
                    "correct": arm.result.exact_match,
                    "generated": arm.generated_code,
                    "error": arm.error,
                }
                for arm_id, arm in run.arms.items()
            },
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return runs


def summarize_accuracy(runs: Sequence[SampleRun], arm_id: str) -> dict[str, Any]:
    """Accuracy overall, by context length, by kind, and by kind at a distance bucket."""
    usable = [run for run in runs if arm_id in run.arms and run.arms[arm_id].error is None]
    total = len(usable)
    correct = sum(1 for run in usable if run.arms[arm_id].result.exact_match)
    by_length: dict[str, dict[str, float | int]] = {}
    for run in usable:
        key = str(run.sample.context_length)
        bucket = by_length.setdefault(key, {"correct": 0, "total": 0})
        bucket["total"] = int(bucket["total"]) + 1
        bucket["correct"] = int(bucket["correct"]) + (1 if run.arms[arm_id].result.exact_match else 0)
    for bucket in by_length.values():
        bucket["accuracy"] = float(bucket["correct"]) / int(bucket["total"])
    by_kind: dict[str, dict[str, float | int]] = {}
    for run in usable:
        key = run.sample.task_kind
        bucket = by_kind.setdefault(key, {"correct": 0, "total": 0})
        bucket["total"] = int(bucket["total"]) + 1
        bucket["correct"] = int(bucket["correct"]) + (1 if run.arms[arm_id].result.exact_match else 0)
    for bucket in by_kind.values():
        bucket["accuracy"] = float(bucket["correct"]) / int(bucket["total"])
    gold_logprobs = [
        float(run.arms[arm_id].result.gold_first_token_logprob)
        for run in usable
        if math.isfinite(float(run.arms[arm_id].result.gold_first_token_logprob))
    ]
    gold_probabilities = [
        float(run.arms[arm_id].result.diagnostics.get("gold_candidate_probability", float("nan")))
        for run in usable
    ]
    gold_probabilities = [value for value in gold_probabilities if math.isfinite(value)]
    gold_ranks = [float(run.arms[arm_id].result.gold_first_token_rank) for run in usable]
    return {
        "arm_id": arm_id,
        "correct": correct,
        "total": total,
        "accuracy": (float(correct) / total) if total else float("nan"),
        "mean_gold_logprob": (sum(gold_logprobs) / len(gold_logprobs)) if gold_logprobs else float("nan"),
        "mean_gold_probability": (
            sum(gold_probabilities) / len(gold_probabilities) if gold_probabilities else float("nan")
        ),
        "mean_gold_rank": (sum(gold_ranks) / len(gold_ranks)) if gold_ranks else float("nan"),
        "by_context_length": by_length,
        "by_task_kind": by_kind,
    }
