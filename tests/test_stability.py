from __future__ import annotations

import torch

from sprm.memory.associative import AssociativeMemory, AssociativeMemoryConfig


def _memory() -> AssociativeMemory:
    return AssociativeMemory(
        AssociativeMemoryConfig(
            name="stability",
            num_slots=8,
            key_dim=16,
            value_dim=16,
            read_temperature=8.0,
            write_gain=0.5,
            gate=1.0,
            forgetting=0.99,
            trust_region_eps=1.0e-3,
            key_l2_normalize=True,
            kmeans_iters=4,
            refine_passes=1,
            stability_buffers=True,
            anchor_interval=2,
            anchor_blend=0.1,
            replay_blocks=1,
            replay_interval=2,
        ),
        device="cpu",
        seed=3,
    )


def _block() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    keys = torch.randn(16, 16, generator=generator)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    values = torch.randn(16, 16, generator=generator)
    return keys, values


def test_replay_buffer_and_anchor_events() -> None:
    memory = _memory()
    keys, values = _block()
    memory.write_slots(keys, values, slot_start=0, slot_count=8, positions=torch.arange(16), block_id=0)
    assert memory.stats.extra["replay_tokens"] == 16.0
    memory.write(keys[:2], values[:2], gain=0.5)
    assert memory.stats.extra["anchor_updates"] >= 1.0
    memory.write(keys[:2], values[:2], gain=0.5)
    assert memory.stats.extra["replay_events"] >= 1.0
    assert memory.stats.extra["anchor_updates"] >= 1.0


def test_clone_preserves_stability_state() -> None:
    memory = _memory()
    keys, values = _block()
    memory.write_slots(keys, values, slot_start=0, slot_count=8, positions=torch.arange(16), block_id=0)
    clone = memory.clone()
    assert clone.stats.extra["replay_tokens"] == memory.stats.extra["replay_tokens"]
    assert torch.equal(clone.M_k_ref, memory.M_k_ref)
    clone.M_v.add_(1.0)
    assert not torch.allclose(clone.M_v, memory.M_v)
