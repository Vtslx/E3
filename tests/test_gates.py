from __future__ import annotations

from pathlib import Path

from sprm.config.schema import load_experiment
from sprm.eval.gates import gate_v1
from sprm.logging import JsonLogger

ROOT = Path(__file__).resolve().parents[1]


def test_v1_gate_passes_on_synthetic_properties() -> None:
    config = load_experiment(ROOT / "configs" / "experiments" / "d1_niah.yaml")
    logger = JsonLogger(component="v1_test")
    logger._echo = False
    try:
        outcome = gate_v1(config, logger)
    finally:
        logger.close()
    assert outcome.status == "PASS"
    assert outcome.checks
    assert all(check.passed for check in outcome.checks)
