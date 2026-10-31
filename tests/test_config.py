from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from sprm.config.schema import load_experiment
from sprm.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "d1_niah.yaml"


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_repository_config_loads() -> None:
    config = load_experiment(CONFIG)
    assert config.name == "d1_niah_lfm3"
    assert config.arms["C0_full"].memory is False
    assert config.arms["C0_full"].compressed is False
    assert config.arms["C1_stream"].compressed is True
    assert config.arms["C3_memory"].allocation == "attention"
    assert config.gates.d1_min_far_distance == 32768
    assert config.tier1.window_tokens % config.tier1.prefill_chunk_tokens == 0
    assert config.tier2.block_tokens == config.tier1.prefill_chunk_tokens
    assert len(config.__dict__["config_sha256"]) == 64


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["tier1"]["window_token"] = 512
    with pytest.raises(ConfigError, match="unknown keys"):
        load_experiment(_write(tmp_path, raw))


def test_window_must_be_multiple_of_chunk(tmp_path: Path) -> None:
    raw = _raw()
    raw["tier1"]["window_tokens"] = 2000
    with pytest.raises(ConfigError, match="multiple"):
        load_experiment(_write(tmp_path, raw))


def test_c0_must_not_have_memory(tmp_path: Path) -> None:
    raw = _raw()
    raw["arms"]["C0_full"]["memory"] = True
    with pytest.raises(ConfigError, match="C0_full"):
        load_experiment(_write(tmp_path, raw))


def test_writeback_requires_memory(tmp_path: Path) -> None:
    raw = _raw()
    raw["arms"]["C1_stream"]["writeback"] = True
    with pytest.raises(ConfigError, match="writeback"):
        load_experiment(_write(tmp_path, raw))


def test_numeric_string_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["gates"]["d5_max_condition"] = "1.0e6"
    with pytest.raises(ConfigError, match="must be a number"):
        load_experiment(_write(tmp_path, raw))


def test_unknown_task_kind_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["tasks"]["kinds"] = ["single_needle", "needle_in_a_haystack"]
    with pytest.raises(ConfigError, match="unknown task kinds"):
        load_experiment(_write(tmp_path, raw))


def test_depths_must_be_interior(tmp_path: Path) -> None:
    raw = _raw()
    raw["tasks"]["depths"] = [0.0, 0.5]
    with pytest.raises(ConfigError, match="strictly between"):
        load_experiment(_write(tmp_path, raw))


def test_config_snapshot_is_independent(tmp_path: Path) -> None:
    raw = _raw()
    config = load_experiment(_write(tmp_path, raw))
    mutated = copy.deepcopy(raw)
    mutated["tier2"]["num_slots"] = 999
    assert config.tier2.num_slots == 512
