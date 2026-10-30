from __future__ import annotations

import math

import pytest
import torch

from sprm.errors import MemoryInvariantError
from sprm.memory.associative import AssociativeMemory, AssociativeMemoryConfig


def _memory(slots: int = 16) -> AssociativeMemory:
    return AssociativeMemory(
        AssociativeMemoryConfig(
            name="test",
            num_slots=slots,
            key_dim=32,
            value_dim=32,
            read_temperature=8.0,
            write_gain=0.5,
            gate=1.0,
            forgetting=0.99,
            trust_region_eps=1.0e-3,
            key_l2_normalize=True,
            kmeans_iters=8,
            refine_passes=2,
        ),
        device="cpu",
        seed=7,
    )


def _clustered(n: int = 32, dim: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    center = torch.randn(dim, generator=generator)
    center = center / center.norm()
    keys = center + 0.05 * torch.randn(n, dim, generator=generator)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    values = center.unsqueeze(0) + 0.02 * torch.randn(n, dim, generator=generator)
    return keys, values


def test_write_slots_reduces_reconstruction_error() -> None:
    memory = _memory()
    keys, values = _clustered()
    baseline = _memory().recon_error(keys, values)
    memory.write_slots(
        keys,
        values,
        slot_start=0,
        slot_count=16,
        positions=torch.arange(32),
        block_id=0,
    )
    after = memory.recon_error(keys, values)
    assert after < baseline
    assert math.isfinite(after)
    assert memory.stats.writes == 1
    assert memory.stats.tokens_written == 32


def test_trust_region_scales_a_huge_write() -> None:
    memory = _memory()
    keys, values = _clustered()
    report = memory.write(keys, values * 1.0e6, gain=1.0)
    assert report.trust_scaled is True
    assert report.delta_norm > 0.0
    assert memory.stats.trust_scaled_count == 1


def test_forgetting_scales_values_not_keys() -> None:
    memory = _memory()
    keys, values = _clustered()
    memory.write_slots(keys, values, slot_start=0, slot_count=16, positions=torch.arange(32), block_id=0)
    before_v = float(memory.M_v.norm())
    before_k = float(memory.M_k.norm())
    memory.apply_forgetting(0.5)
    assert abs(float(memory.M_v.norm()) / before_v - 0.5) < 1.0e-5
    assert abs(float(memory.M_k.norm()) - before_k) < 1.0e-5
    assert memory.stats.forgetting_events == 1


def test_clone_is_independent() -> None:
    memory = _memory()
    keys, values = _clustered()
    memory.write_slots(keys, values, slot_start=0, slot_count=16, positions=torch.arange(32), block_id=0)
    clone = memory.clone()
    clone.M_v.add_(10.0)
    assert not torch.allclose(clone.M_v, memory.M_v)
    assert clone.stats.name == memory.stats.name
    clone.apply_forgetting(0.5)
    assert memory.stats.forgetting_events == 0


def test_recon_error_rejects_shape_mismatch() -> None:
    memory = _memory()
    with pytest.raises(MemoryInvariantError):
        memory.recon_error(torch.zeros(4, 31), torch.zeros(4, 32))
    with pytest.raises(MemoryInvariantError):
        memory.recon_error(torch.zeros(4, 32), torch.zeros(3, 32))


def test_write_slots_rejects_slot_overflow() -> None:
    memory = _memory(slots=8)
    keys, values = _clustered()
    with pytest.raises(MemoryInvariantError):
        memory.write_slots(keys, values, slot_start=6, slot_count=4, positions=torch.arange(32), block_id=0)


def test_condition_is_finite_for_well_conditioned_keys() -> None:
    memory = _memory()
    assert math.isfinite(memory.condition())
