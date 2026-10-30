from __future__ import annotations

from pathlib import Path

import pytest

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.config.schema import load_experiment
from sprm.eval.niah import build_samples
from sprm.eval.runner import run_sample

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "d1_niah.yaml"

pytestmark = pytest.mark.integration


def test_one_sample_all_arms() -> None:
    config = load_experiment(CONFIG)
    backend = Lfm3Backend(config.backend, seed=config.seed)
    try:
        sample = build_samples(
            backend.tokenizer,
            config.tasks,
            kinds=["single_needle"],
            context_lengths=[8192],
            depths=[0.1],
            samples_per_cell=1,
            max_samples=1,
        )[0]
        run = run_sample(
            backend,
            config,
            sample,
            ["C0_full", "C1_stream", "C3_memory", "C4_memory_w1"],
            max_new_tokens=2,
        )
        prompt_tokens = len(sample.full_token_ids)
        budget = min(
            config.tier2.num_slots,
            prompt_tokens - config.tier1.window_tokens - config.tier1.sink_tokens,
        )
        # effective_length is measured after decoding, so it may include one generated
        # token that was fed back into the cache; the prefill base is exact.
        slack = 1
        assert run.arms["C0_full"].effective_length in (prompt_tokens, prompt_tokens + slack)
        assert run.arms["C1_stream"].effective_length in (
            config.tier1.sink_tokens + config.tier1.window_tokens,
            config.tier1.sink_tokens + config.tier1.window_tokens + slack,
        )
        base = config.tier1.sink_tokens + budget + config.tier1.window_tokens
        assert run.arms["C3_memory"].effective_length in (base, base + slack)
        assert run.arms["C4_memory_w1"].effective_length in (base, base + slack)
        for arm_id, arm in run.arms.items():
            assert arm.error is None, f"{arm_id}: {arm.error}"
            assert arm.score is not None
            assert set(arm.score.logprobs) == set(sample.metadata["options"])
        # Exact KV is the ceiling; this deterministic sample is answerable.
        assert run.arms["C0_full"].result.exact_match is True
        assert run.arms["C0_full"].score.gold_rank == 1
        # W1 must reach the scored continuation, otherwise C4 is a duplicate of C3.
        assert run.arms["C3_memory"].score.logprobs != run.arms["C4_memory_w1"].score.logprobs
    finally:
        backend.close()
