from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.config.schema import load_experiment
from sprm.eval.gates import gate_v0, gate_v2
from sprm.logging import JsonLogger
from sprm.runtime.form_a import prefill_online

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "d1_niah.yaml"

pytestmark = pytest.mark.integration


def test_backend_contract_and_online_prefill() -> None:
    config = load_experiment(CONFIG)
    logger = JsonLogger(component="integration")
    logger._echo = False
    backend = Lfm3Backend(config.backend, seed=config.seed)
    try:
        outcome = gate_v0(backend, config, logger, hash_files=False)
        assert outcome.status == "PASS", outcome.report

        tokens = []
        while len(tokens) < 3200:
            tokens.extend(backend.tokenize(" The reading room keeps a ledger of every request and visit."))
        tokens = tokens[:3200]
        online = prefill_online(backend, config, tokens, use_memory=True, allocation="uniform")
        assert online.build is not None
        budget = min(config.tier2.num_slots, len(tokens) - config.tier1.window_tokens - config.tier1.sink_tokens)
        assert online.cache.global_effective_length() == (
            config.tier1.sink_tokens + budget + config.tier1.window_tokens
        )
        assert bool(torch.isfinite(online.logits).all())
        assert online.evictions == len(online.block_ranges)
    finally:
        backend.close()
        logger.close()


def test_v2_gate_on_model() -> None:
    config = load_experiment(CONFIG)
    logger = JsonLogger(component="integration")
    logger._echo = False
    backend = Lfm3Backend(config.backend, seed=config.seed)
    try:
        outcome = gate_v2(backend, config, logger)
        assert outcome.status == "PASS", outcome.report
    finally:
        backend.close()
        logger.close()
