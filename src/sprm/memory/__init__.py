from __future__ import annotations

from sprm.memory.allocator import allocate_slots, effective_rank
from sprm.memory.associative import AssociativeMemory, AssociativeMemoryConfig
from sprm.memory.prefix import CompressedPrefixConfig, CompressedPrefixMemory, PrefixPlan, plan_prefix

__all__ = [
    "AssociativeMemory",
    "AssociativeMemoryConfig",
    "CompressedPrefixConfig",
    "CompressedPrefixMemory",
    "PrefixPlan",
    "allocate_slots",
    "effective_rank",
    "plan_prefix",
]
