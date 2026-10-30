"""Form N planning harness.

Form N prepends a fixed number of soft tokens per evicted block to the frozen model input.
This module only plans the budget and the block ranges. It never claims a quality result:
there is no compressor checkpoint, so the runtime path is intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass

from sprm.config.schema import ExperimentConfig
from sprm.contracts.types import BackendInfo
from sprm.memory.prefix import CompressedPrefixConfig, CompressedPrefixMemory, PrefixPlan, plan_prefix


@dataclass
class PrefixBuild:
    plan: PrefixPlan
    evict_start: int
    evict_end: int
    memory: CompressedPrefixMemory

    @property
    def total_soft_tokens(self) -> int:
        return self.plan.total_soft_tokens


def build_prefix_scaffold(
    info: BackendInfo,
    experiment: ExperimentConfig,
    *,
    context_length: int,
    full_length: int,
) -> PrefixBuild | None:
    """Plan the compressed prefix for one prompt, or ``None`` when nothing is evicted."""
    sink = experiment.tier1.sink_tokens
    window = experiment.tier1.window_tokens
    evict_start = sink
    evict_end = full_length - window
    if evict_end <= evict_start:
        return None
    block = max(1, experiment.tier2.block_tokens)
    ranges = [(start, min(start + block, evict_end)) for start in range(evict_start, evict_end, block)]
    # Form N has no online importance estimator in this round; the budget is uniform.
    plan = plan_prefix(
        block_ranges=ranges,
        budget=experiment.tier2.num_slots,
        importances=None,
        min_soft_tokens_per_block=max(1, experiment.tier2.min_slots_per_block),
        max_slot_multiplier=experiment.tier2.max_slot_multiplier,
    )
    dtype_bytes = 2 if info.dtype in {"bfloat16", "float16"} else 4
    config = CompressedPrefixConfig(
        num_layers=info.num_layers,
        kv_width=info.kv_width,
        dtype_bytes=dtype_bytes,
        soft_token_budget=experiment.tier2.num_slots,
    )
    return PrefixBuild(
        plan=plan,
        evict_start=evict_start,
        evict_end=evict_end,
        memory=CompressedPrefixMemory(config, plan),
    )


def describe_prefix(build: PrefixBuild) -> dict[str, object]:
    return {
        "status": "planned-not-verified",
        "evict_start": build.evict_start,
        "evict_end": build.evict_end,
        "plan": build.plan.as_dict(),
        "soft_token_bytes": build.memory.memory_bytes(),
        "note": (
            "No compressor checkpoint exists in this round. Form N is a contract and budget "
            "scaffold only; it makes no quality claim."
        ),
    }
