from __future__ import annotations

from pathlib import Path

import pytest

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.config.schema import load_experiment
from sprm.runtime.form_a import decode_arm, exact_window_from_cache
from sprm.runtime.speculative import PromptLookup, decode_speculative

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "d1_niah.yaml"

pytestmark = pytest.mark.integration


def test_speculative_output_matches_greedy() -> None:
    config = load_experiment(CONFIG)
    backend = Lfm3Backend(config.backend, seed=config.seed)
    try:
        sentence = " The archive room is quiet and the shelves are labelled by year."
        prompt: list[int] = []
        while len(prompt) < 512:
            prompt.extend(backend.tokenize(sentence))
        prompt = prompt[:512]
        cache, logits, _ = backend.prefill_exact(prompt, chunk_tokens=config.tier1.prefill_chunk_tokens)
        arm = config.arms["C0_full"]
        baseline_cache = cache.clone()
        baseline_window = exact_window_from_cache(
            backend, baseline_cache, capacity=config.tier1.window_tokens, full_length=len(prompt)
        )
        baseline = decode_arm(
            backend,
            baseline_cache,
            config,
            arm,
            first_logits=logits,
            start_position=len(prompt),
            max_new_tokens=16,
            exact_window=baseline_window,
            build=None,
            eostoken_id=None,
        )
        speculative_cache = cache.clone()
        speculative_window = exact_window_from_cache(
            backend, speculative_cache, capacity=config.tier1.window_tokens, full_length=len(prompt)
        )
        speculative = decode_speculative(
            backend,
            speculative_cache,
            config,
            arm,
            first_logits=logits,
            start_position=len(prompt),
            max_new_tokens=16,
            exact_window=speculative_window,
            build=None,
            eostoken_id=None,
            proposer=PromptLookup(prompt, n=3, gamma=4),
        )
        assert speculative.generated == [step.token_id for step in baseline.steps]
        assert speculative.proposed_tokens > 0
    finally:
        backend.close()
