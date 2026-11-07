"""Verification gates V0, V1, V2, and D1.

Each gate returns a ``GateOutcome``. The CLI owns the archive: it creates the run directory,
passes it in so artifacts can be written, then seals the verdict. Gates never mutate an
existing run.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sprm.backends.lfm3.backend import Lfm3Backend, QueryProbe
from sprm.config.schema import ExperimentConfig
from sprm.contracts.types import KVSlice
from sprm.errors import ConfigError, GateFailure, VerificationError
from sprm.eval.metrics import mcnemar_one_sided
from sprm.eval.niah import build_samples, sample_prompt_token_ids
from sprm.eval.runner import SampleRun, run_experiment, summarize_accuracy
from sprm.logging import JsonLogger
from sprm.memory.allocator import allocate_slots
from sprm.memory.associative import AssociativeMemory, AssociativeMemoryConfig
from sprm.runtime.draft_heads import DraftHeadProposer
from sprm.runtime.form_a import (
    ExactWindow,
    assemble_effective_global,
    cache_footprint_bytes,
    decode_arm,
    exact_window_from_cache,
    prefill_online,
)
from sprm.runtime.speculative import PromptLookup, ReservoirLookup, decode_speculative
from sprm.verification.archive import RunArchive, verify_run


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": bool(self.passed), "detail": self.detail}


@dataclass
class GateOutcome:
    gate: str
    status: str
    verdict: str
    result: dict[str, Any]
    report: str
    claims: str
    checks: list[Check] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _all_pass(checks: Sequence[Check]) -> bool:
    return all(check.passed for check in checks)


def _check_table(checks: Sequence[Check]) -> str:
    lines = ["| check | result | detail |", "|-------|--------|--------|"]
    for check in checks:
        detail = "" if check.detail is None else str(check.detail).replace("|", "\\|")[:220]
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {detail} |")
    return "\n".join(lines)


def _fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return f"{value:.{digits}f}"


# ====================================================================== V0
def gate_v0(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    logger: JsonLogger,
    *,
    archive: RunArchive | None = None,
    hash_files: bool = True,
) -> GateOutcome:
    """Backend contract: load, cache shapes, RoPE invertibility, determinism."""
    info = backend.info
    checks: list[Check] = []
    partition = sorted(info.global_layers + info.sliding_layers + info.conv_layers)
    checks.append(
        Check(
            "layer_partition",
            partition == list(range(info.num_layers)),
            f"global={list(info.global_layers)} sliding={list(info.sliding_layers)} conv={len(info.conv_layers)}",
        )
    )
    checks.append(Check("kv_width", info.kv_width == info.num_kv_heads * info.head_dim, info.kv_width))

    prompt_text = " The archive room is quiet. " * 32
    prompt = backend.tokenize(prompt_text)
    cache, logits, _ = backend.prefill_exact(prompt, chunk_tokens=min(128, len(prompt)))
    checks.append(Check("prefill_length", cache.global_effective_length() == len(prompt), len(prompt)))
    shapes_ok = True
    positions_ok = True
    for layer_idx in info.global_layers:
        layer = cache.layers[layer_idx]
        shapes_ok &= tuple(layer.keys.shape) == (1, info.num_kv_heads, len(prompt), info.head_dim)
        positions_ok &= layer.keys.dtype == info.dtype_torch
    checks.append(Check("kv_shape", shapes_ok, f"{len(prompt)} tokens, {info.num_kv_heads}x{info.head_dim}"))
    checks.append(Check("kv_dtype", positions_ok, str(info.dtype_torch)))
    checks.append(Check("logits_finite", bool(torch.isfinite(logits).all()), tuple(logits.shape)))
    try:
        cache.assert_invariants(expected_global_length=len(prompt))
        checks.append(Check("cache_invariants", True))
    except Exception as exc:
        checks.append(Check("cache_invariants", False, f"{type(exc).__name__}: {exc}"))

    layer_idx = info.global_layers[0]
    layer = cache.layers[layer_idx]
    keys = layer.keys
    positions = torch.arange(keys.shape[-2], device=keys.device)
    unrotated = backend.rotary.unrotate(keys, positions)
    rerotated = backend.rotary.rotate(unrotated, positions)
    rope_error = float((rerotated.float() - keys.float()).abs().max().item())
    checks.append(Check("rope_roundtrip", rope_error <= 0.05, rope_error))

    text = backend.tokenize(" The library keeps a ledger of every request.")
    logits_a = backend.prefill_exact(text, chunk_tokens=len(text))[1]
    logits_b = backend.prefill_exact(text, chunk_tokens=len(text))[1]
    determinism = float((logits_a - logits_b).abs().max().item())
    checks.append(Check("prefill_determinism", determinism <= 0.05, determinism))

    token = int(torch.argmax(logits).item())
    next_logits = backend.decode(cache, token, len(prompt))
    checks.append(Check("decode_finite", bool(torch.isfinite(next_logits).all()), tuple(next_logits.shape)))

    model_info = info.as_dict()
    if hash_files:
        model_info["file_hashes"] = backend.file_hashes()
    if archive is not None:
        archive.write_json("artifacts/backend_info.json", model_info)
        archive.write_json("artifacts/checks.json", [check.as_dict() for check in checks])
    logger.info("v0_checks", checks=[check.as_dict() for check in checks])

    passed = _all_pass(checks)
    status = "PASS" if passed else "FAIL"
    result = {
        "status_detail": status,
        "checks": [check.as_dict() for check in checks],
        "model": model_info,
        "prompt_tokens": len(prompt),
    }
    report = "\n".join(
        [
            "# V0 backend contract",
            "",
            f"- Model: `{info.model_path}`",
            f"- Layers: {info.num_layers} ({len(info.global_layers)} global, "
            f"{len(info.sliding_layers)} sliding, {len(info.conv_layers)} conv)",
            f"- KV: {info.num_kv_heads} heads x {info.head_dim}, dtype {info.dtype}",
            "",
            _check_table(checks),
            "",
            f"Verdict: **{status}**.",
        ]
    )
    claims = (
        "# Claims for V0\n\n"
        "Establishes: the checkpoint loads through the declared backend, its layer plan partitions\n"
        "the model, exact KV has the contracted shape and dtype, RoPE is invertible within bf16\n"
        "tolerance, prefilling is repeatable, and a decode step produces finite logits.\n\n"
        "Does not establish: retrieval quality, memory behaviour, or any speedup.\n"
    )
    return GateOutcome(
        gate="v0",
        status=status,
        verdict=status,
        result=result,
        report=report,
        claims=claims,
        checks=checks,
        model_info=model_info,
    )


# ====================================================================== V1
def _synthetic_memory_probe() -> dict[str, Any]:
    config = AssociativeMemoryConfig(
        name="gate_v1",
        num_slots=16,
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
    )
    memory = AssociativeMemory(config, device="cpu", seed=11)
    generator = torch.Generator().manual_seed(5)
    all_keys: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    for block in range(2):
        center = torch.randn(32, generator=generator)
        center = center / center.norm()
        keys = center + 0.05 * torch.randn(16, 32, generator=generator)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        values = center.unsqueeze(0) + 0.02 * torch.randn(16, 32, generator=generator)
        positions = torch.arange(block * 16, (block + 1) * 16)
        memory.write_slots(
            keys,
            values,
            slot_start=block * 8,
            slot_count=8,
            positions=positions,
            block_id=block,
        )
        all_keys.append(keys)
        all_values.append(values)
    keys = torch.cat(all_keys)
    values = torch.cat(all_values)
    baseline = AssociativeMemory(config, device="cpu", seed=11)
    before = baseline.recon_error(keys, values)
    after = memory.recon_error(keys, values)
    condition = memory.condition()
    normal_before = float(memory.M_v.norm().item())
    memory.apply_forgetting(0.5)
    normal_after = float(memory.M_v.norm().item())
    huge = memory.write(keys, values * 1.0e6, gain=1.0)
    return {
        "recon_before": before,
        "recon_after": after,
        "recon_improved": after < before,
        "condition": condition,
        "condition_finite": math.isfinite(condition) and condition < 1.0e12,
        "forgetting_ratio": normal_after / normal_before if normal_before else float("nan"),
        "forgetting_halved": abs(normal_after / normal_before - 0.5) < 1.0e-5 if normal_before else False,
        "trust_scaled": bool(huge.trust_scaled),
    }


def gate_v1(
    experiment: ExperimentConfig,
    logger: JsonLogger,
    *,
    archive: RunArchive | None = None,
) -> GateOutcome:
    """Contract and property gate: allocator, memory, KV types, archive, statistics."""
    checks: list[Check] = []
    details: dict[str, Any] = {}

    alloc_ok = True
    alloc_cases = []
    for blocks, slots, mode, importances in (
        (4, 32, "uniform", None),
        (4, 32, "effective_rank", [1.0, 2.0, 4.0, 8.0]),
        (1, 8, "uniform", None),
        (8, 16, "effective_rank", [1.0] * 8),
    ):
        allocation = allocate_slots(
            num_blocks=blocks,
            num_slots=slots,
            mode=mode,
            importances=importances,
            min_slots_per_block=1,
            max_slot_multiplier=4.0,
        )
        case_ok = len(allocation) == blocks and sum(allocation) == slots and min(allocation) >= 1
        alloc_ok &= case_ok
        alloc_cases.append({"blocks": blocks, "slots": slots, "mode": mode, "allocation": list(allocation), "ok": case_ok})
    checks.append(Check("allocator_budget", alloc_ok, alloc_cases))
    details["allocator"] = alloc_cases

    probe = _synthetic_memory_probe()
    details["memory"] = probe
    checks.append(Check("memory_w2_reduces_error", bool(probe["recon_improved"]), probe))
    checks.append(Check("memory_condition_finite", bool(probe["condition_finite"]), probe["condition"]))
    checks.append(Check("memory_forgetting_halves_values", bool(probe["forgetting_halved"]), probe["forgetting_ratio"]))
    checks.append(Check("memory_trust_region_scales", bool(probe["trust_scaled"]), None))

    try:
        keys = torch.zeros(1, 2, 4, 8)
        values = torch.zeros(1, 2, 4, 8)
        slice_ = KVSlice(keys=keys, values=values, positions=torch.arange(3, 7), start=3)
        ok = slice_.num_tokens == 4 and slice_.end == 7 and slice_.narrow(4, 2).start == 4
        try:
            KVSlice(keys=keys, values=values, positions=torch.arange(0, 4), start=3)
            mismatch_rejected = False
        except ValueError:
            mismatch_rejected = True
        checks.append(Check("kvslice_invariants", ok and mismatch_rejected, f"ok={ok} mismatch_rejected={mismatch_rejected}"))
    except Exception as exc:
        checks.append(Check("kvslice_invariants", False, f"{type(exc).__name__}: {exc}"))

    try:
        window = ExactWindow(
            keys={0: torch.zeros(1, 2, 2, 4), 1: torch.zeros(1, 2, 2, 4)},
            values={0: torch.zeros(1, 2, 2, 4), 1: torch.zeros(1, 2, 2, 4)},
            positions=torch.arange(0, 2),
            capacity=4,
        )
        step_keys = {0: torch.ones(1, 2, 1, 4), 1: torch.ones(1, 2, 1, 4)}
        step_values = {0: torch.ones(1, 2, 1, 4), 1: torch.ones(1, 2, 1, 4)}
        window.append_step(step_keys, step_values, 2)
        aligned = window.positions.numel() == 3 and int(window.positions[-1]) == 2
        checks.append(Check("exact_window_position_alignment", aligned, list(window.positions.tolist())))
    except Exception as exc:
        checks.append(Check("exact_window_position_alignment", False, f"{type(exc).__name__}: {exc}"))

    try:
        with tempfile.TemporaryDirectory(prefix="sprm_v1_") as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text("name: probe\n", encoding="utf-8")
            run = RunArchive.create(
                root=root / "archive",
                gate="v1_probe",
                slug="probe",
                config_sha256="0" * 64,
                config_path=config_path,
                argv=["pytest"],
                echo=False,
            )
            run.write_json("artifacts/payload.json", {"ok": True})
            run.seal(status="PASS", verdict="PASS", result={"probe": True}, report="probe", claims="probe")
            clean = verify_run(run.run_dir)
            (run.run_dir / "artifacts" / "payload.json").write_text("{\"ok\": false}\n", encoding="utf-8")
            tampered = verify_run(run.run_dir)
            checks.append(
                Check(
                    "archive_seal_and_tamper_detection",
                    clean == [] and tampered != [],
                    f"clean={len(clean)} tampered={len(tampered)}",
                )
            )
            details["archive"] = {"clean_problems": clean, "tampered_problems": tampered}
    except Exception as exc:
        checks.append(Check("archive_seal_and_tamper_detection", False, f"{type(exc).__name__}: {exc}"))

    p_value = mcnemar_one_sided(5, 0)
    checks.append(Check("mcnemar_known_value", abs(p_value - 1.0 / 32.0) < 1.0e-12, p_value))
    details["mcnemar_p"] = p_value

    if archive is not None:
        archive.write_json("artifacts/checks.json", [check.as_dict() for check in checks])
        archive.write_json("artifacts/details.json", details)
    logger.info("v1_checks", checks=[check.as_dict() for check in checks])
    passed = _all_pass(checks)
    status = "PASS" if passed else "FAIL"
    report = "\n".join(
        [
            "# V1 contract gate",
            "",
            "Runs without the model. It checks the pieces D1 depends on: slot budgets, closed-form",
            "writes, reconstruction improvement, trust region, forgetting, KV typing, the exact-window",
            "position alignment, archive sealing, and the McNemar statistic.",
            "",
            _check_table(checks),
            "",
            f"Verdict: **{status}**.",
        ]
    )
    claims = (
        "# Claims for V1\n\n"
        "Establishes: the allocator spends exactly its budget, W2 reduces reconstruction error on\n"
        "synthetic clustered data, the trust region caps a deliberately huge write, forgetting scales\n"
        "values, KV slices enforce position/start invariants, the window carries one position per token,\n"
        "and the archive detects tampering.\n\n"
        "Does not establish: anything about the model, real text, or far-context retrieval.\n"
    )
    return GateOutcome(
        gate="v1",
        status=status,
        verdict=status,
        result={"status_detail": status, "checks": [check.as_dict() for check in checks], "details": details},
        report=report,
        claims=claims,
        checks=checks,
    )


# ====================================================================== V2
def _prompt_tokens(backend: Lfm3Backend, length: int, seed: int) -> list[int]:
    sentences = (
        " The reading room opens at nine and closes at six.",
        " A narrow stair leads to the map collection.",
        " The staff keeps a record of every request in the ledger.",
        " A small clock above the door ticks through the day.",
        " The shelves are labelled by year and by subject.",
    )
    tokens: list[int] = []
    index = 0
    while len(tokens) < length:
        tokens.extend(backend.tokenize(f" Record {index}:{sentences[(index + seed) % len(sentences)]}"))
        index += 1
    return tokens[:length]


def gate_v2(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    logger: JsonLogger,
    *,
    archive: RunArchive | None = None,
) -> GateOutcome:
    """Cache-surgery exactness: no-eviction rebuild must preserve logits and KV bytes."""
    info = backend.info
    sink = experiment.tier1.sink_tokens
    window = experiment.tier1.window_tokens
    checks: list[Check] = []
    cases: list[dict[str, Any]] = []
    max_diffs: list[float] = []
    agreement = 0
    total = 0
    no_eviction_diffs: list[float] = []
    eviction_diffs: list[float] = []
    prefill_decode_gaps: list[float] = []

    for seed in range(4):
        prompt = _prompt_tokens(backend, 512, seed)
        body = prompt[:-1]
        cache, _, _ = backend.prefill_exact(body, chunk_tokens=128)
        reference = cache.clone()
        logits_reference = backend.decode(reference, prompt[-1], len(body))
        rebuilt = cache.clone()
        effective = assemble_effective_global(
            backend,
            cache,
            None,
            experiment,
            full_length=len(body),
            compressed=True,
        )
        rebuilt.rebuild_global(effective)
        rebuilt.assert_invariants(expected_global_length=len(body))
        for layer_idx in info.global_layers:
            original = cache.layers[layer_idx].keys
            actual = rebuilt.layers[layer_idx].keys
            no_eviction_diffs.append(float((original.float() - actual.float()).abs().max().item()))
        logits_rebuilt = backend.decode(rebuilt, prompt[-1], len(body))
        diff = float((logits_reference - logits_rebuilt).abs().max().item())
        max_diffs.append(diff)
        top1_reference = int(torch.argmax(logits_reference).item())
        top1_rebuilt = int(torch.argmax(logits_rebuilt).item())
        total += 1
        agreement += int(top1_reference == top1_rebuilt)
        # Diagnostic only: prefill and one-token decode use different kernels and bf16 rounding.
        full_cache, logits_full, _ = backend.prefill_exact(prompt, chunk_tokens=128)
        prefill_decode_gaps.append(float((logits_full - logits_reference).abs().max().item()))
        del full_cache
        cases.append(
            {
                "seed": seed,
                "tokens": len(prompt),
                "max_logit_diff": diff,
                "prefill_decode_gap": prefill_decode_gaps[-1],
                "top1_reference": top1_reference,
                "top1_rebuilt": top1_rebuilt,
            }
        )
        del cache, rebuilt, reference

    long_prompt = _prompt_tokens(backend, sink + window + 1024, 17)
    long_body = long_prompt[:-1]
    cache, _, _ = backend.prefill_exact(long_body, chunk_tokens=experiment.tier1.prefill_chunk_tokens)
    effective = assemble_effective_global(
        backend,
        cache,
        None,
        experiment,
        full_length=len(long_body),
        compressed=True,
    )
    expected_length = sink + window
    long_ok = True
    for layer_idx in info.global_layers:
        keys, _ = effective[layer_idx]
        long_ok &= int(keys.shape[-2]) == expected_length
        sink_orig = cache.layers[layer_idx].keys[..., :sink, :]
        sink_new = keys[..., :sink, :]
        window_orig = cache.layers[layer_idx].keys[..., -window:, :]
        window_new = keys[..., sink:, :]
        eviction_diffs.append(float((sink_orig.float() - sink_new.float()).abs().max().item()))
        eviction_diffs.append(float((window_orig.float() - window_new.float()).abs().max().item()))
    checks.append(
        Check(
            "eviction_slices_byte_identical",
            long_ok and max(eviction_diffs) == 0.0,
            f"length={expected_length} max_tensor_diff={max(eviction_diffs)}",
        )
    )

    # The online prefill is the protocol D1 actually uses. Its sink must be bit-identical to
    # an exact prefill of the same prefix, and its effective length must follow the budget.
    online_tokens = _prompt_tokens(backend, sink + window + 1024, 23)
    exact_cache, _, _ = backend.prefill_exact(
        online_tokens, chunk_tokens=experiment.tier1.prefill_chunk_tokens
    )
    online_sink_diffs: list[float] = []
    stream = prefill_online(backend, experiment, online_tokens, use_memory=False)
    stream_ok = stream.cache.global_effective_length() == sink + window
    for layer_idx in info.global_layers:
        online_sink_diffs.append(
            float(
                (
                    exact_cache.layers[layer_idx].keys[..., :sink, :].float()
                    - stream.cache.layers[layer_idx].keys[..., :sink, :].float()
                )
                .abs()
                .max()
                .item()
            )
        )
    checks.append(
        Check(
            "online_stream_structure",
            stream_ok and max(online_sink_diffs) == 0.0,
            f"length={stream.cache.global_effective_length()} sink_diff={max(online_sink_diffs)}",
        )
    )
    del stream
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    online_memory = prefill_online(
        backend,
        experiment,
        online_tokens,
        use_memory=True,
        allocation="uniform",
    )
    budget = min(
        experiment.tier2.num_slots,
        len(online_tokens) - window - sink,
    )
    memory_ok = online_memory.build is not None and (
        online_memory.cache.global_effective_length() == sink + budget + window
    )
    memory_sink_diffs: list[float] = []
    if memory_ok:
        for layer_idx in info.global_layers:
            reference = exact_cache.layers[layer_idx].keys[..., :sink, :].float()
            actual = online_memory.cache.layers[layer_idx].keys[..., :sink, :].float()
            memory_sink_diffs.append(float((reference - actual).abs().max().item()))
    checks.append(
        Check(
            "online_memory_structure",
            bool(memory_ok) and (not memory_sink_diffs or max(memory_sink_diffs) == 0.0),
            f"budget={budget} length={online_memory.cache.global_effective_length()} "
            f"sink_diff={max(memory_sink_diffs) if memory_sink_diffs else 0.0}",
        )
    )
    del online_memory, exact_cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    checks.append(
        Check(
            "no_eviction_kv_byte_identical",
            max(no_eviction_diffs) == 0.0,
            max(no_eviction_diffs) if no_eviction_diffs else None,
        )
    )
    checks.append(
        Check(
            "logit_difference",
            max(max_diffs) <= experiment.gates.v2_max_logit_diff,
            {"max": max(max_diffs), "case_max": max_diffs, "limit": experiment.gates.v2_max_logit_diff},
        )
    )
    top1_rate = agreement / total if total else 0.0
    checks.append(
        Check(
            "top1_agreement",
            top1_rate >= experiment.gates.v2_min_top1_agreement,
            f"{agreement}/{total}",
        )
    )
    if archive is not None:
        archive.write_json("artifacts/cases.json", cases)
        archive.write_json("artifacts/checks.json", [check.as_dict() for check in checks])
    logger.info("v2_checks", checks=[check.as_dict() for check in checks])
    passed = _all_pass(checks)
    status = "PASS" if passed else "FAIL"
    longest = max(max_diffs) if max_diffs else float("nan")
    gap = max(prefill_decode_gaps) if prefill_decode_gaps else float("nan")
    report = "\n".join(
        [
            "# V2 cache-surgery exactness",
            "",
            "Both paths prefill the prompt body (all but the final token). The reference feeds the final",
            "token through the untouched body cache; the rebuilt path copies the full global KV set and",
            "feeds the same token again. With no eviction the two must be identical. The gap between a",
            "full prefill and a one-token decode is reported separately: it is a kernel/bf16 difference,",
            "not a cache-surgery defect.",
            "",
            _check_table(checks),
            "",
            f"Worst rebuild-vs-reference logit difference: {_fmt(longest)} "
            f"(limit {experiment.gates.v2_max_logit_diff}).",
            f"Full-prefill vs decode diagnostic gap: {_fmt(gap)}.",
            f"Top-1 agreement: {agreement}/{total}.",
            "",
            f"Verdict: **{status}**.",
        ]
    )
    claims = (
        "# Claims for V2\n\n"
        "Establishes: rebuilding the global cache from the same body prefill reproduces the exact KV\n"
        "bytes and the next-token distribution within the configured tolerance; when eviction occurs,\n"
        "the retained sink and window slices are byte-identical to the exact cache. This validates the\n"
        "surgery, not the memory.\n\n"
        "Does not establish: that compressed memory preserves the distribution, or that dropped tokens\n"
        "are recoverable.\n"
    )
    return GateOutcome(
        gate="v2",
        status=status,
        verdict=status,
        result={
            "status_detail": status,
            "cases": cases,
            "max_logit_diff": longest,
            "prefill_decode_gap": gap,
            "top1_agreement": top1_rate,
            "checks": [check.as_dict() for check in checks],
        },
        report=report,
        claims=claims,
        checks=checks,
        model_info=backend.info.as_dict(),
    )


# ====================================================================== D5
def gate_d5(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    logger: JsonLogger,
    *,
    archive: RunArchive | None = None,
    max_decode_tokens: int | None = None,
    max_samples: int = 2,
    arm_id: str = "C4_memory_w1",
) -> GateOutcome:
    """Closed-loop stability: condition bound and reconstruction shrinkage.

    This gate exercises what the harness actually implements: the trust region, W1 writes,
    and the slot refresh over a long decode. It does **not** exercise anchors, replay, or eta
    adaptation; those are not implemented. ``d5_min_eta_ratio`` is therefore interpreted as
    the minimum window-reconstruction shrinkage ratio (head/tail); a value below 1 means the
    reconstruction error grew.
    """
    if arm_id not in experiment.arms:
        raise ConfigError(f"D5 arm {arm_id!r} is not configured")
    arm = experiment.arms[arm_id]
    if not arm.memory:
        raise ConfigError(f"D5 arm {arm_id!r} has no memory")
    decode_tokens = (
        experiment.runtime.long_decode_tokens if max_decode_tokens is None else int(max_decode_tokens)
    )
    if decode_tokens < 8:
        raise ConfigError(f"D5 needs at least 8 decode tokens, got {decode_tokens}")
    lengths = [min(experiment.tasks.context_lengths)]
    samples = build_samples(
        backend.tokenizer,
        experiment.tasks,
        kinds=["single_needle"],
        context_lengths=lengths,
        depths=[0.1],
        samples_per_cell=1,
        max_samples=max(1, max_samples),
    )
    rows: list[dict[str, Any]] = []
    for sample in samples:
        prompt = sample_prompt_token_ids(sample)
        sink = experiment.tier1.sink_tokens
        window_tokens = experiment.tier1.window_tokens
        block = experiment.tier2.block_tokens
        evict_end = len(prompt) - window_tokens
        ranges = (
            [(start, min(start + block, evict_end)) for start in range(sink, evict_end, block)]
            if evict_end > sink
            else []
        )
        probe = QueryProbe(backend, backend.info.global_layers, len(sample.question_token_ids))
        try:
            exact_cache, _, _ = backend.prefill_exact(
                prompt, chunk_tokens=experiment.tier1.prefill_chunk_tokens
            )
        finally:
            probe.detach()
        importances = (
            backend.block_attention_importances(exact_cache, ranges, probe.queries())
            if ranges
            else None
        )
        del exact_cache
        allocation = arm.allocation
        if allocation == "attention" and not importances:
            allocation = "uniform"
            importances = None
        prefill = prefill_online(
            backend,
            experiment,
            prompt,
            use_memory=True,
            allocation=allocation,
            importances=importances,
        )
        if prefill.build is None:
            raise VerificationError(f"D5 sample {sample.sample_id} did not evict; no memory to stress")
        cache = prefill.cache.clone()
        build = prefill.build.clone()
        window = exact_window_from_cache(
            backend,
            cache,
            capacity=experiment.tier1.window_tokens,
            full_length=len(prompt),
        )
        outcome = decode_arm(
            backend,
            cache,
            experiment,
            arm,
            first_logits=prefill.logits,
            start_position=len(prompt),
            max_new_tokens=decode_tokens,
            exact_window=window,
            build=build,
            eostoken_id=None,
        )
        recon = [float(value) for value in outcome.recon_error_trajectory if math.isfinite(float(value))]
        recon_gen = [
            float(value) for value in outcome.recon_error_generated if math.isfinite(float(value))
        ]
        if len(recon) < 2:
            raise VerificationError(f"D5 sample {sample.sample_id} produced fewer than two reconstruction probes")
        head = sum(recon[:5]) / len(recon[:5])
        tail = sum(recon[-5:]) / len(recon[-5:])
        shrinkage = head / tail if tail > 0 else float("nan")
        gen_head = sum(recon_gen[:5]) / len(recon_gen[:5]) if recon_gen else float("nan")
        gen_tail = sum(recon_gen[-5:]) / len(recon_gen[-5:]) if recon_gen else float("nan")
        gen_shrinkage = gen_head / gen_tail if gen_tail and gen_tail > 0 else float("nan")
        rows.append(
            {
                "sample_id": sample.sample_id,
                "prompt_tokens": len(prompt),
                "decoded_tokens": len(outcome.steps),
                "recon_head": head,
                "recon_tail": tail,
                "shrinkage_ratio": shrinkage,
                "generated_recon_head": gen_head,
                "generated_recon_tail": gen_tail,
                "generated_shrinkage_ratio": gen_shrinkage,
                "condition_number": float(outcome.condition_final),
                "trust_scaled_count": int(outcome.trust_scaled_count),
                "writes": int(sum(memory.stats.writes for memory in build.memories.values())),
                "anchor_updates": float(
                    sum(memory.stats.extra.get("anchor_updates", 0.0) for memory in build.memories.values())
                ),
                "replay_events": float(
                    sum(memory.stats.extra.get("replay_events", 0.0) for memory in build.memories.values())
                ),
                "eta_final": float(outcome.eta_final),
                "eta_ratio": float(outcome.eta_final / max(experiment.tier2.write_gain, 1.0e-12)),
                "recon_trajectory": recon,
            }
        )
    max_condition = max(row["condition_number"] for row in rows)
    min_shrinkage = min(row["shrinkage_ratio"] for row in rows)
    min_generated_shrinkage = min(row["generated_shrinkage_ratio"] for row in rows)
    checks = [
        Check(
            "condition_bounded",
            math.isfinite(max_condition) and max_condition <= experiment.gates.d5_max_condition,
            {"max": max_condition, "limit": experiment.gates.d5_max_condition},
        )
    ]
    if arm.writeback:
        checks.append(
            Check(
                "writeback_shrinkage",
                math.isfinite(min_generated_shrinkage)
                and min_generated_shrinkage >= experiment.gates.d5_min_eta_ratio,
                {"min": min_generated_shrinkage, "limit": experiment.gates.d5_min_eta_ratio},
            )
        )
        gating_label = "generated-token (W1)"
        diagnostics = [
            Check(
                "full_window_shrinkage_diagnostic",
                math.isfinite(min_shrinkage) and min_shrinkage >= experiment.gates.d5_min_eta_ratio,
                {"min": min_shrinkage, "limit": experiment.gates.d5_min_eta_ratio},
            )
        ]
    else:
        checks.append(
            Check(
                "memory_shrinkage",
                math.isfinite(min_shrinkage) and min_shrinkage >= experiment.gates.d5_min_eta_ratio,
                {"min": min_shrinkage, "limit": experiment.gates.d5_min_eta_ratio},
            )
        )
        gating_label = "full-window (memory-only C3)"
        diagnostics = [
            Check(
                "generated_shrinkage_diagnostic",
                math.isfinite(min_generated_shrinkage)
                and min_generated_shrinkage >= experiment.gates.d5_min_eta_ratio,
                {"min": min_generated_shrinkage, "limit": experiment.gates.d5_min_eta_ratio},
            )
        ]
    checks.append(
        Check(
            "decode_completed",
            all(row["decoded_tokens"] >= decode_tokens for row in rows),
            {row["sample_id"]: row["decoded_tokens"] for row in rows},
        )
    )
    if archive is not None:
        archive.write_json("artifacts/rows.json", rows)
        archive.write_json("artifacts/checks.json", [check.as_dict() for check in checks])
    logger.info("d5_checks", checks=[check.as_dict() for check in checks])
    passed = _all_pass(checks)
    status = "PASS" if passed else "FAIL"
    table = [
        "| sample | decoded | full shrink | W1 shrink | eta ratio | max cond | trust | writes | anchors | replays |",
        "|--------|---------|-------------|-----------|-----------|----------|-------|--------|---------|---------|",
    ]
    for row in rows:
        table.append(
            f"| {row['sample_id']} | {row['decoded_tokens']} | {_fmt(row['shrinkage_ratio'])} "
            f"| {_fmt(row['generated_shrinkage_ratio'])} | {_fmt(row['eta_ratio'])} "
            f"| {_fmt(row['condition_number'])} | {row['trust_scaled_count']} | {row['writes']} "
            f"| {_fmt(row['anchor_updates'], 0)} | {_fmt(row['replay_events'], 0)} |"
        )
    report = "\n".join(
        [
            "# D5 closed-loop stability",
            "",
            f"Arm: `{arm_id}`; decode budget: {decode_tokens} tokens; samples: {len(rows)}.",
            "",
            *table,
            "",
            _check_table(checks),
            "",
            "Diagnostic (not gating):",
            "",
            _check_table(diagnostics),
            "",
            f"Verdict: **{status}**. The gating shrinkage is {gating_label}. For C4 the full-window",
            "value is diagnostic because most of that window is prefill KV that W1 never wrote; for C3",
            "the full-window value is the memory shrinkage and the generated value is diagnostic.",
            "Key regularization clamps the key spectrum on every build; anchors/replay/eta are W1-only",
        ]
    )
    claims = (
        "# Claims for D5\n\n"
        "Establishes: over a long decode, whether (1) the regularized key-matrix condition\n"
        "stays below the configured bound and (2) reconstruction shrinks. For C4 the gating\n"
        "shrinkage is the generated-token window (W1); for C3 it is the full-window memory\n"
        "shrinkage. The other window is reported as a diagnostic.\n\n"
        "Does not establish: distribution equivalence, long-text task accuracy, or a learned\n"
        "memory representation.\n"
    )
    logger.info("d5_summary", verdict=status, max_condition=max_condition, min_shrinkage=min_shrinkage)
    return GateOutcome(
        gate="d5",
        status=status,
        verdict=status,
        result={
            "status_detail": status,
            "rows": rows,
            "max_condition": max_condition,
            "min_shrinkage_ratio": min_shrinkage,
            "min_generated_shrinkage_ratio": min_generated_shrinkage,
            "checks": [check.as_dict() for check in checks],
            "diagnostics": [check.as_dict() for check in diagnostics],
        },
        report=report,
        claims=claims,
        checks=checks,
        model_info=backend.info.as_dict() | {"file_hashes": backend.file_hashes()},
    )


# ====================================================================== S1
def gate_s1(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    logger: JsonLogger,
    *,
    archive: RunArchive | None = None,
    max_new_tokens: int = 32,
    gamma: int = 4,
    ngram: int = 3,
) -> GateOutcome:
    """Prompt-lookup speculative decoding: acceptance and tokens/s on the exact arm.

    The output must be token-for-token identical to greedy decoding; a speedup is only
    claimed when that holds. The prompt is a repeated sentence, which gives the draft real
    n-gram matches; that is a mechanism test, not a general-task acceptance estimate.
    """
    if max_new_tokens < 1 or gamma < 1 or ngram < 1:
        raise ConfigError("S1 needs positive max_new_tokens, gamma, and ngram")
    sentence = " The archive room is quiet and the shelves are labelled by year."
    prompt: list[int] = []
    while len(prompt) < 2048:
        prompt.extend(backend.tokenize(sentence))
    prompt = prompt[:2048]
    cache, logits, _ = backend.prefill_exact(
        prompt, chunk_tokens=experiment.tier1.prefill_chunk_tokens
    )
    arm = experiment.arms["C0_full"]

    baseline_cache = cache.clone()
    baseline_window = exact_window_from_cache(
        backend,
        baseline_cache,
        capacity=experiment.tier1.window_tokens,
        full_length=len(prompt),
    )
    t0 = time.perf_counter()
    baseline = decode_arm(
        backend,
        baseline_cache,
        experiment,
        arm,
        first_logits=logits,
        start_position=len(prompt),
        max_new_tokens=max_new_tokens,
        exact_window=baseline_window,
        build=None,
        eostoken_id=None,
    )
    baseline_s = time.perf_counter() - t0
    baseline_ids = [step.token_id for step in baseline.steps]

    speculative_cache = cache.clone()
    speculative_window = exact_window_from_cache(
        backend,
        speculative_cache,
        capacity=experiment.tier1.window_tokens,
        full_length=len(prompt),
    )
    proposer = PromptLookup(prompt, n=ngram, gamma=gamma)
    t0 = time.perf_counter()
    speculative = decode_speculative(
        backend,
        speculative_cache,
        experiment,
        arm,
        first_logits=logits,
        start_position=len(prompt),
        max_new_tokens=max_new_tokens,
        exact_window=speculative_window,
        build=None,
        eostoken_id=None,
        proposer=proposer,
    )
    speculative_s = time.perf_counter() - t0

    baseline_tps = len(baseline_ids) / baseline_s if baseline_s > 0 else float("nan")
    speculative_tps = (
        len(speculative.generated) / speculative_s if speculative_s > 0 else float("nan")
    )
    speedup = speculative_tps / baseline_tps if baseline_tps > 0 else float("nan")
    output_match = baseline_ids == speculative.generated
    divergent = [
        index
        for index in range(min(len(baseline_ids), len(speculative.generated)))
        if baseline_ids[index] != speculative.generated[index]
    ]
    divergence_index = divergent[0] if divergent else None
    divergence_tokens = len(divergent) + abs(len(baseline_ids) - len(speculative.generated))
    acceptance = (
        speculative.accepted_tokens / speculative.proposed_tokens
        if speculative.proposed_tokens
        else float("nan")
    )
    checks = [
        Check("output_matches_greedy", output_match, f"{len(baseline_ids)} tokens"),
        Check(
            "speedup",
            math.isfinite(speedup) and speedup > 1.0,
            {"speedup": speedup, "baseline_tps": baseline_tps, "speculative_tps": speculative_tps},
        ),
        Check("draft_acceptance", speculative.proposed_tokens > 0, acceptance),
    ]
    if archive is not None:
        archive.write_json(
            "artifacts/summary.json",
            {
                "prompt_tokens": len(prompt),
                "max_new_tokens": max_new_tokens,
                "gamma": gamma,
                "ngram": ngram,
                "baseline_seconds": baseline_s,
                "speculative_seconds": speculative_s,
                "baseline_tokens_per_s": baseline_tps,
                "speculative_tokens_per_s": speculative_tps,
                "speedup": speedup,
                "output_match": output_match,
                "divergence_index": divergence_index,
                "divergence_tokens": divergence_tokens,
                "proposed_tokens": speculative.proposed_tokens,
                "accepted_tokens": speculative.accepted_tokens,
                "full_accepts": speculative.full_accepts,
                "verify_calls": speculative.verify_calls,
                "fallback_calls": speculative.fallback_calls,
                "baseline_ids": baseline_ids,
                "speculative_ids": speculative.generated,
            },
        )
        archive.write_json("artifacts/checks.json", [check.as_dict() for check in checks])
    logger.info("s1_checks", checks=[check.as_dict() for check in checks])
    passed = _all_pass(checks)
    status = "PASS" if passed else "FAIL"
    report = "\n".join(
        [
            "# S1 prompt-lookup speculative decoding",
            "",
            f"Prompt: {len(prompt)} tokens; decode budget: {max_new_tokens}; gamma={gamma}, ngram={ngram}.",
            "Arm: `C0_full` (exact KV; the memory arm is not a speed lever).",
            "",
            "| run | seconds | tokens/s |",
            "|-----|---------|----------|",
            f"| greedy | {_fmt(baseline_s)} | {_fmt(baseline_tps)} |",
            f"| speculative | {_fmt(speculative_s)} | {_fmt(speculative_tps)} |",
            "",
            f"- speedup: **{_fmt(speedup)}x**",
            f"- output identical to greedy: **{output_match}**",
            f"- proposed {speculative.proposed_tokens}, accepted {speculative.accepted_tokens}, "
            f"full accepts {speculative.full_accepts}, verify calls {speculative.verify_calls}, "
            f"fallbacks {speculative.fallback_calls}",
            "",
            _check_table(checks),
            "",
            f"Verdict: **{status}**.",
        ]
    )
    claims = (
        "# Claims for S1\n\n"
        "Establishes: on a repeated-sentence prompt, prompt-lookup drafts with the stated\n"
        "acceptance, the speculative output is token-identical to greedy decoding, and the\n"
        "measured tokens/s is compared on the same exact-KV arm. The all-or-nothing commit\n"
        "rolls back global/sliding KV and restores conv state on rejection.\n\n"
        "Does not establish: general-task acceptance, memory-arm speedup, batch>1 throughput,\n"
        "or a claim that compression itself accelerates generation.\n"
    )
    return GateOutcome(
        gate="s1",
        status=status,
        verdict=f"{status}: speedup {_fmt(speedup)}x, match={output_match}, acceptance={_fmt(acceptance)}",
        result={
            "status_detail": status,
            "baseline_tokens_per_s": baseline_tps,
            "speculative_tokens_per_s": speculative_tps,
            "speedup": speedup,
            "output_match": output_match,
            "divergence_index": divergence_index,
            "divergence_tokens": divergence_tokens,
            "acceptance": acceptance,
            "proposed_tokens": speculative.proposed_tokens,
            "accepted_tokens": speculative.accepted_tokens,
            "full_accepts": speculative.full_accepts,
            "verify_calls": speculative.verify_calls,
            "fallback_calls": speculative.fallback_calls,
            "checks": [check.as_dict() for check in checks],
        },
        report=report,
        claims=claims,
        checks=checks,
        model_info=backend.info.as_dict() | {"file_hashes": backend.file_hashes()},
    )


# ====================================================================== S2
def gate_s2(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    logger: JsonLogger,
    *,
    archive: RunArchive | None = None,
    max_new_tokens: int = 32,
    gamma: int = 4,
    ngram: int = 3,
    arm_id: str = "C3_memory",
    prompt_text: str | None = None,
    prompt_tokens: int = 4096,
    partial_accept: bool = False,
    proposer_kind: str = "prompt",
    draft_head: str | None = None,
) -> GateOutcome:
    """Speculative decoding on a natural-text prompt and, optionally, the memory arm.

    S1 is a mechanism test on repeated text with exact KV. S2 measures the same mechanism
    on non-repetitive prose and on a compressed C3 cache. The greedy baseline runs without
    the per-step reconstruction diagnostics, so the tokens/s comparison is generation work
    against generation work. The output must stay token-identical to greedy.
    """
    if max_new_tokens < 1 or gamma < 1 or ngram < 1:
        raise ConfigError("S2 needs positive max_new_tokens, gamma, and ngram")
    if proposer_kind not in {"prompt", "reservoir"}:
        raise ConfigError(f"unknown proposer {proposer_kind!r}")
    if arm_id not in experiment.arms:
        raise ConfigError(f"unknown arm {arm_id!r}; configured={sorted(experiment.arms)}")
    arm = experiment.arms[arm_id]
    if prompt_text is None:
        prompt_text = (
            "The archive room is quiet and the shelves are labelled by year. "
            "A passkey is written on the card beside the door."
        )
    prompt = backend.tokenize(prompt_text)
    if prompt_tokens > 0:
        prompt = prompt[:prompt_tokens]
    threshold = experiment.tier1.sink_tokens + experiment.tier1.window_tokens
    if arm.memory and len(prompt) <= threshold:
        raise ConfigError(
            f"S2 memory arm needs more than {threshold} prompt tokens for an eviction; got {len(prompt)}"
        )
    if len(prompt) < 8:
        raise ConfigError("S2 prompt is too short")

    if arm.memory:
        online = prefill_online(backend, experiment, prompt, use_memory=True, allocation="uniform")
        cache = online.cache
        logits = online.logits
        build = online.build
        if build is None:
            raise ConfigError("S2 memory prefill built no memory; prompt did not evict a block")
    else:
        cache, logits, _ = backend.prefill_exact(
            prompt, chunk_tokens=experiment.tier1.prefill_chunk_tokens
        )
        build = None

    baseline_cache = cache.clone()
    baseline_window = exact_window_from_cache(
        backend,
        baseline_cache,
        capacity=experiment.tier1.window_tokens,
        full_length=len(prompt),
    )
    baseline_build = build.clone() if (build is not None and arm.writeback) else build
    t0 = time.perf_counter()
    baseline = decode_arm(
        backend,
        baseline_cache,
        experiment,
        arm,
        first_logits=logits,
        start_position=len(prompt),
        max_new_tokens=max_new_tokens,
        exact_window=baseline_window,
        build=baseline_build,
        eostoken_id=None,
        diagnostics=False,
    )
    baseline_s = time.perf_counter() - t0
    baseline_ids = [step.token_id for step in baseline.steps]

    speculative_cache = cache.clone()
    speculative_window = exact_window_from_cache(
        backend,
        speculative_cache,
        capacity=experiment.tier1.window_tokens,
        full_length=len(prompt),
    )
    speculative_build = build.clone() if (build is not None and arm.writeback) else build
    proposer = (
        DraftHeadProposer(backend, draft_head, gamma=gamma)
        if draft_head is not None
        else (
            ReservoirLookup(list(prompt), n=ngram, gamma=gamma)
            if proposer_kind == "reservoir"
            else PromptLookup(list(prompt), n=ngram, gamma=gamma)
        )
    )
    t0 = time.perf_counter()
    speculative = decode_speculative(
        backend,
        speculative_cache,
        experiment,
        arm,
        first_logits=logits,
        start_position=len(prompt),
        max_new_tokens=max_new_tokens,
        exact_window=speculative_window,
        build=speculative_build,
        eostoken_id=None,
        proposer=proposer,
        partial_accept=partial_accept,
    )
    speculative_s = time.perf_counter() - t0

    baseline_tps = len(baseline_ids) / baseline_s if baseline_s > 0 else float("nan")
    speculative_tps = (
        len(speculative.generated) / speculative_s if speculative_s > 0 else float("nan")
    )
    speedup = speculative_tps / baseline_tps if baseline_tps > 0 else float("nan")
    output_match = baseline_ids == speculative.generated
    divergent = [
        index
        for index in range(min(len(baseline_ids), len(speculative.generated)))
        if baseline_ids[index] != speculative.generated[index]
    ]
    divergence_index = divergent[0] if divergent else None
    divergence_tokens = len(divergent) + abs(len(baseline_ids) - len(speculative.generated))
    acceptance = (
        speculative.accepted_tokens / speculative.proposed_tokens
        if speculative.proposed_tokens
        else float("nan")
    )
    full_accept_rate = (
        speculative.full_accepts / speculative.verify_calls
        if speculative.verify_calls
        else float("nan")
    )
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    footprint: dict[str, Any] = {}
    if build is not None:
        cache_bytes, per_token = cache_footprint_bytes(
            backend, speculative_cache, build, context_length=len(prompt)
        )
        footprint = {
            "memory_bytes": build.memory_bytes(),
            "cache_bytes": cache_bytes,
            "kv_bytes_per_token": per_token,
        }

    checks = [
        Check("output_matches_greedy", output_match, f"{len(baseline_ids)} tokens"),
        Check("draft_proposed", speculative.proposed_tokens > 0, speculative.proposed_tokens),
        Check(
            "acceptance_target",
            math.isfinite(acceptance) and acceptance >= 0.5,
            {"acceptance": acceptance, "target": 0.5},
        ),
        Check(
            "net_speedup_target",
            math.isfinite(speedup) and speedup >= 1.3,
            {"speedup": speedup, "target": 1.3},
        ),
    ]
    if archive is not None:
        archive.write_json(
            "artifacts/summary.json",
            {
                "prompt_sha256": prompt_sha,
                "prompt_tokens": len(prompt),
                "arm": arm_id,
                "proposer": proposer_kind,
                "draft_head": draft_head,
                "partial_accept": partial_accept,
                "max_new_tokens": max_new_tokens,
                "gamma": gamma,
                "ngram": ngram,
                "baseline_seconds": baseline_s,
                "speculative_seconds": speculative_s,
                "baseline_tokens_per_s": baseline_tps,
                "speculative_tokens_per_s": speculative_tps,
                "speedup": speedup,
                "output_match": output_match,
                "divergence_index": divergence_index,
                "divergence_tokens": divergence_tokens,
                "proposed_tokens": speculative.proposed_tokens,
                "accepted_tokens": speculative.accepted_tokens,
                "full_accepts": speculative.full_accepts,
                "partial_accepts": speculative.partial_accepts,
                "verify_calls": speculative.verify_calls,
                "fallback_calls": speculative.fallback_calls,
                "acceptance": acceptance,
                "full_accept_rate": full_accept_rate,
                "baseline_ids": baseline_ids,
                "speculative_ids": speculative.generated,
                **footprint,
            },
        )
        archive.write_json("artifacts/checks.json", [check.as_dict() for check in checks])
    logger.info("s2_checks", checks=[check.as_dict() for check in checks])
    passed = _all_pass(checks)
    status = "PASS" if passed else "FAIL"
    report = "\n".join(
        [
            "# S2 speculative decoding on natural text / memory arm",
            "",
            f"Arm: `{arm_id}`; proposer: `{proposer_kind}`"
            + (f" (draft head `{draft_head}`)" if draft_head else "")
            + f"; partial accept: {partial_accept}.",
            f"Prompt: {len(prompt)} tokens (sha256 `{prompt_sha[:12]}`); decode budget: {max_new_tokens}; "
            f"gamma={gamma}, ngram={ngram}.",
            "",
            "| run | seconds | tokens/s |",
            "|-----|---------|----------|",
            f"| greedy | {_fmt(baseline_s)} | {_fmt(baseline_tps)} |",
            f"| speculative | {_fmt(speculative_s)} | {_fmt(speculative_tps)} |",
            "",
            f"- net speedup: **{_fmt(speedup)}x** (target 1.3x)",
            f"- output identical to greedy: **{output_match}**"
            + (f" (first divergence at {divergence_index}; {divergence_tokens} tokens differ)" if not output_match else ""),
            f"- acceptance: **{_fmt(acceptance)}** (target 0.5); full-accept rate {_fmt(full_accept_rate)}",
            f"- proposed {speculative.proposed_tokens}, accepted {speculative.accepted_tokens}, "
            f"full {speculative.full_accepts}, partial {speculative.partial_accepts}, "
            f"verify {speculative.verify_calls}, fallbacks {speculative.fallback_calls}",
        ]
        + (
            [
                "",
                f"- memory {footprint['memory_bytes'] / 2**20:.1f} MiB; "
                f"{footprint['kv_bytes_per_token']} KV B/token at {len(prompt)} prompt tokens",
            ]
            if footprint
            else []
        )
        + [
            "",
            _check_table(checks),
            "",
            f"Verdict: **{status}**.",
        ]
    )
    claims = (
        "# Claims for S2\n\n"
        "Establishes: on the stated prompt and arm, the prompt-lookup/reservoir draft's\n"
        "token acceptance, the speculative decode's net tokens/s against a greedy baseline\n"
        "without diagnostics, and the first position where the committed sequence differs\n"
        "from a separate single-token greedy run. The committed tokens are the target's own\n"
        "verifier argmaxes; the chunked verify and the single-token forward round differently\n"
        "in bf16, so bitwise identity is not guaranteed on natural text after a full accept.\n\n"
        "Does not establish: general-task acceptance, a trained draft model, tree verification,\n"
        "batch>1 throughput, or that the memory compression itself accelerates generation.\n"
    )
    return GateOutcome(
        gate="s2",
        status=status,
        verdict=f"{status}: {arm_id} speedup {_fmt(speedup)}x, acceptance {_fmt(acceptance)}, match={output_match}",
        result={
            "status_detail": status,
            "arm": arm_id,
            "proposer": proposer_kind,
            "draft_head": draft_head,
            "partial_accept": partial_accept,
            "prompt_tokens": len(prompt),
            "prompt_sha256": prompt_sha,
            "baseline_tokens_per_s": baseline_tps,
            "speculative_tokens_per_s": speculative_tps,
            "speedup": speedup,
            "output_match": output_match,
            "divergence_index": divergence_index,
            "divergence_tokens": divergence_tokens,
            "acceptance": acceptance,
            "full_accept_rate": full_accept_rate,
            "proposed_tokens": speculative.proposed_tokens,
            "accepted_tokens": speculative.accepted_tokens,
            "full_accepts": speculative.full_accepts,
            "partial_accepts": speculative.partial_accepts,
            "verify_calls": speculative.verify_calls,
            "fallback_calls": speculative.fallback_calls,
            **footprint,
            "checks": [check.as_dict() for check in checks],
        },
        report=report,
        claims=claims,
        checks=checks,
        model_info=backend.info.as_dict() | {"file_hashes": backend.file_hashes()},
    )


# ====================================================================== D1
def _paired(
    runs: Sequence[SampleRun],
    control: str,
    candidate: str,
) -> dict[str, Any]:
    improved = degraded = both = neither = 0
    pairs = 0
    for run in runs:
        if control not in run.arms or candidate not in run.arms:
            continue
        a = run.arms[control]
        b = run.arms[candidate]
        if a.error is not None or b.error is not None:
            continue
        pairs += 1
        if a.result.exact_match and b.result.exact_match:
            both += 1
        elif not a.result.exact_match and b.result.exact_match:
            improved += 1
        elif a.result.exact_match and not b.result.exact_match:
            degraded += 1
        else:
            neither += 1
    return {
        "pairs": pairs,
        "both_correct": both,
        "neither_correct": neither,
        "improved": improved,
        "degraded": degraded,
        "mcnemar_p": mcnemar_one_sided(improved, degraded),
    }


def _accuracy_rows(runs: Sequence[SampleRun], arm_id: str, predicate: Any) -> list[SampleRun]:
    return [
        run
        for run in runs
        if predicate(run)
        and arm_id in run.arms
        and run.arms[arm_id].error is None
    ]


def _effect_stats(
    runs: Sequence[SampleRun],
    *,
    control: str,
    candidate: str,
    far_distance: int,
    near_distance: int,
    min_gain: float,
    min_kinds: int,
    max_near_degradation: float,
    alpha: float,
    prefix: str,
) -> dict[str, Any]:
    """Paired far/near statistics for one causal contrast."""

    far_runs = [run for run in runs if run.sample.context_length >= far_distance]
    near_runs = [run for run in runs if run.sample.context_length <= near_distance]

    def acc(arm_id: str, rows: Sequence[SampleRun]) -> float:
        usable = [run for run in rows if arm_id in run.arms and run.arms[arm_id].error is None]
        if not usable:
            return float("nan")
        return sum(1 for run in usable if run.arms[arm_id].result.exact_match) / len(usable)

    kinds = sorted({run.sample.task_kind for run in far_runs})
    per_kind: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        rows = [run for run in far_runs if run.sample.task_kind == kind]
        control_accuracy = acc(control, rows)
        candidate_accuracy = acc(candidate, rows)
        per_kind[kind] = {
            "control_accuracy": control_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "gain": candidate_accuracy - control_accuracy,
            "control_total": len(
                [run for run in rows if control in run.arms and run.arms[control].error is None]
            ),
        }
    kinds_improved = sum(
        1 for stats in per_kind.values() if stats["gain"] >= min_gain
    )
    near_control = acc(control, near_runs)
    near_candidate = acc(candidate, near_runs)
    near_degradation = (
        near_control - near_candidate
        if not (math.isnan(near_control) or math.isnan(near_candidate))
        else float("nan")
    )
    paired_far = _paired(far_runs, control, candidate)
    paired_near = _paired(near_runs, control, candidate)
    pooled_control = acc(control, far_runs)
    pooled_candidate = acc(candidate, far_runs)
    checks = [
        Check(f"{prefix}_far_bucket", len(far_runs) > 0, len(far_runs)),
        Check(
            f"{prefix}_kinds_improved",
            kinds_improved >= min_kinds,
            f"{kinds_improved}/{len(kinds)}",
        ),
        Check(
            f"{prefix}_near_degradation",
            (not math.isnan(near_degradation)) and near_degradation <= max_near_degradation,
            near_degradation,
        ),
        Check(
            f"{prefix}_mcnemar",
            paired_far["mcnemar_p"] <= alpha,
            paired_far["mcnemar_p"],
        ),
    ]
    return {
        "control": control,
        "candidate": candidate,
        "per_kind": per_kind,
        "kinds_improved": kinds_improved,
        "near_control": near_control,
        "near_candidate": near_candidate,
        "near_degradation": near_degradation,
        "paired_far": paired_far,
        "paired_near": paired_near,
        "pooled_control": pooled_control,
        "pooled_candidate": pooled_candidate,
        "pooled_gain": (
            pooled_candidate - pooled_control
            if not (math.isnan(pooled_candidate) or math.isnan(pooled_control))
            else float("nan")
        ),
        "checks": checks,
    }


def gate_d1(
    backend: Lfm3Backend,
    experiment: ExperimentConfig,
    samples: Sequence[Any],
    arm_ids: Sequence[str],
    logger: JsonLogger,
    *,
    archive: RunArchive | None = None,
    max_new_tokens: int | None = None,
) -> GateOutcome:
    """D1 attribution: memory (C3 vs C1) and writeback (C4 vs C3) are separate.

    C4 vs C1 is reported as a combined diagnostic but never gates: it cannot say whether a
    gain came from the memory slots or from the writeback.
    """
    runs = run_experiment(
        backend,
        experiment,
        samples,
        arm_ids,
        logger=logger,
        max_new_tokens=max_new_tokens,
        archive=archive,
    )
    by_arm = {arm_id: summarize_accuracy(runs, arm_id) for arm_id in arm_ids}
    far_distance = experiment.gates.d1_min_far_distance
    near_distance = min(experiment.tasks.context_lengths)
    common: dict[str, Any] = {
        "far_distance": far_distance,
        "near_distance": near_distance,
        "min_gain": experiment.gates.d1_min_absolute_gain,
        "min_kinds": experiment.gates.d1_min_tasks_improved,
        "max_near_degradation": experiment.gates.d1_max_near_degradation,
        "alpha": experiment.gates.d1_mcnemar_alpha,
    }
    memory = _effect_stats(
        runs, control="C1_stream", candidate="C3_memory", prefix="memory", **common
    )
    writeback = _effect_stats(
        runs, control="C3_memory", candidate="C4_memory_w1", prefix="writeback", **common
    )
    combined = _effect_stats(
        runs, control="C1_stream", candidate="C4_memory_w1", prefix="combined", **common
    )
    checks = memory["checks"] + writeback["checks"]
    passed = _all_pass(checks)
    status = "PASS" if passed else "FAIL"
    verdict = (
        f"{status}: memory C3-C1 far {_fmt(float(memory['pooled_gain']))} "
        f"({memory['kinds_improved']}/{len(memory['per_kind'])} kinds, "
        f"p={_fmt(float(memory['paired_far']['mcnemar_p']))}); "
        f"writeback C4-C3 far {_fmt(float(writeback['pooled_gain']))} "
        f"({writeback['kinds_improved']}/{len(writeback['per_kind'])} kinds, "
        f"p={_fmt(float(writeback['paired_far']['mcnemar_p']))})"
    )

    accuracy_lines = ["| arm | overall | 8k | 16k | 32k |", "|-----|---------|----|----|----|"]
    for arm_id in arm_ids:
        stats = by_arm[arm_id]
        by_len = stats["by_context_length"]
        accuracy_lines.append(
            f"| {arm_id} | {_fmt(float(stats['accuracy']))} "
            f"| {_fmt(float(by_len.get('8192', {}).get('accuracy', float('nan'))))} "
            f"| {_fmt(float(by_len.get('16384', {}).get('accuracy', float('nan'))))} "
            f"| {_fmt(float(by_len.get('32768', {}).get('accuracy', float('nan'))))} |"
        )

    def kind_table(effect: dict[str, Any], label: str) -> list[str]:
        lines = [
            f"| kind | {label} control | {label} candidate | gain |",
            "|------|-----------------|--------------------|------|",
        ]
        for kind, stats in effect["per_kind"].items():
            lines.append(
                f"| {kind} | {_fmt(float(stats['control_accuracy']))} "
                f"| {_fmt(float(stats['candidate_accuracy']))} "
                f"| {_fmt(float(stats['gain']))} |"
            )
        return lines

    likelihood_lines = [
        "| arm | mean gold logprob | mean gold probability | mean gold rank |",
        "|-----|-------------------|-----------------------|----------------|",
    ]
    for arm_id in arm_ids:
        stats = by_arm[arm_id]
        likelihood_lines.append(
            f"| {arm_id} | {_fmt(float(stats['mean_gold_logprob']), 2)} "
            f"| {_fmt(float(stats['mean_gold_probability']))} "
            f"| {_fmt(float(stats['mean_gold_rank']), 2)} |"
        )

    stability_lines = [
        "| arm | recon final | max cond | trust scaled | memory KiB | KV B/token |",
        "|-----|-------------|----------|--------------|------------|------------|",
    ]
    for arm_id in arm_ids:
        usable = [run for run in runs if arm_id in run.arms and run.arms[arm_id].error is None]
        if not usable:
            stability_lines.append(f"| {arm_id} | n/a | n/a | n/a | n/a | n/a |")
            continue
        recon = [run.arms[arm_id].result.recon_error_final for run in usable]
        recon = [value for value in recon if not math.isnan(value)]
        conditions = [run.arms[arm_id].result.condition_number_final for run in usable]
        conditions = [value for value in conditions if not math.isnan(value)]
        trust = sum(run.arms[arm_id].result.trust_scaled_count for run in usable)
        memory_kib = sum(run.arms[arm_id].result.memory_bytes for run in usable) / max(len(usable), 1) / 1024
        kv = [run.arms[arm_id].result.kv_bytes_per_token for run in usable]
        stability_lines.append(
            f"| {arm_id} | {_fmt(max(recon) if recon else float('nan'))} "
            f"| {_fmt(max(conditions) if conditions else float('nan'))} | {trust} "
            f"| {_fmt(memory_kib, 1)} | {_fmt(float(sum(kv) / len(kv)) if kv else float('nan'), 0)} |"
        )

    errors = [
        {"sample_id": run.sample.sample_id, "arm": arm_id, "error": arm.error}
        for run in runs
        for arm_id, arm in run.arms.items()
        if arm.error is not None
    ]
    report = "\n".join(
        [
            "# D1 attribution",
            "",
            f"Config: `{experiment.name}`, seeds {experiment.seed}/{experiment.tasks.filler_seed}/"
            f"{experiment.tasks.needle_seed}, sink={experiment.tier1.sink_tokens}, "
            f"window={experiment.tier1.window_tokens}, slots={experiment.tier2.num_slots}, "
            f"block={experiment.tier2.block_tokens}, allocation={experiment.arms['C3_memory'].allocation}.",
            f"Samples: {len(runs)}; arms: {', '.join(arm_ids)}.",
            "",
            "Memory effect = `C3_memory - C1_stream`; writeback effect = `C4_memory_w1 - C3_memory`.",
            "`C4 - C1` is combined and does not gate.",
            "",
            "## Accuracy by context length",
            "",
            *accuracy_lines,
            "",
            f"## Memory effect (C3 vs C1) in the far bucket (>={far_distance})",
            "",
            *kind_table(memory, "C3"),
            "",
            f"## Writeback effect (C4 vs C3) in the far bucket (>={far_distance})",
            "",
            *kind_table(writeback, "C4"),
            "",
            "## Combined diagnostic (C4 vs C1)",
            "",
            f"pooled gain {_fmt(float(combined['pooled_gain']))}, "
            f"{combined['kinds_improved']}/{len(combined['per_kind'])} kinds; not gating.",
            "",
            "## Candidate likelihood (all samples)",
            "",
            *likelihood_lines,
            "",
            "## Paired comparisons",
            "",
            f"- memory C3 vs C1 (far): improved {memory['paired_far']['improved']}, "
            f"degraded {memory['paired_far']['degraded']}, "
            f"p={_fmt(float(memory['paired_far']['mcnemar_p']))}.",
            f"- writeback C4 vs C3 (far): improved {writeback['paired_far']['improved']}, "
            f"degraded {writeback['paired_far']['degraded']}, "
            f"p={_fmt(float(writeback['paired_far']['mcnemar_p']))}.",
            f"- combined C4 vs C1 (far): improved {combined['paired_far']['improved']}, "
            f"degraded {combined['paired_far']['degraded']}, "
            f"p={_fmt(float(combined['paired_far']['mcnemar_p']))}.",
            f"- memory near (<= {near_distance}): control {_fmt(float(memory['near_control']))}, "
            f"candidate {_fmt(float(memory['near_candidate']))}, "
            f"degradation {_fmt(float(memory['near_degradation']))}.",
            f"- writeback near (<= {near_distance}): control {_fmt(float(writeback['near_control']))}, "
            f"candidate {_fmt(float(writeback['near_candidate']))}, "
            f"degradation {_fmt(float(writeback['near_degradation']))}.",
            "",
            "## Stability diagnostics",
            "",
            *stability_lines,
            "",
            "## Checks",
            "",
            _check_table(checks),
            "",
            f"Verdict: **{verdict}**.",
            "",
            "Condition numbers of `inf` mean the key matrix is rank-deficient, which is expected",
            "for repetitive filler; it is a capacity/stability diagnostic, not a numerical blow-up.",
        ]
    )
    claims = (
        "# Claims for D1\n\n"
        "Establishes separately: (1) whether the memory slots improve far-context retrieval\n"
        "(`C3 - C1`, memory effect); (2) whether the W1 writeback adds anything on top of the\n"
        "memory (`C4 - C3`, writeback effect). `C4 - C1` is reported only as a combined\n"
        "diagnostic and never gates. Exact KV (`C0`) is the ceiling; `C1` is the no-memory\n"
        "control.\n\n"
        "Does not establish: distribution equivalence, real-benchmark generalization, proposer\n"
        "or speculative-decoding benefits, Form N quality, or wall-clock speedup. 'Far' is the\n"
        "configured context-length bucket, not a real-world distance distribution.\n"
    )
    if archive is not None:
        archive.write_json(
            "artifacts/summary.json",
            {
                "by_arm": by_arm,
                "memory_effect": memory,
                "writeback_effect": writeback,
                "combined_effect": combined,
                "errors": errors,
            },
        )
    logger.info(
        "d1_summary",
        verdict=verdict,
        memory_effect={
            "pooled_gain": memory["pooled_gain"],
            "kinds_improved": memory["kinds_improved"],
            "mcnemar_p": memory["paired_far"]["mcnemar_p"],
        },
        writeback_effect={
            "pooled_gain": writeback["pooled_gain"],
            "kinds_improved": writeback["kinds_improved"],
            "mcnemar_p": writeback["paired_far"]["mcnemar_p"],
        },
    )
    return GateOutcome(
        gate="d1",
        status=status,
        verdict=verdict,
        result={
            "status_detail": status,
            "by_arm": by_arm,
            "memory_effect": memory,
            "writeback_effect": writeback,
            "combined_effect": combined,
            "checks": [check.as_dict() for check in checks],
            "errors": errors,
        },
        report=report,
        claims=claims,
        checks=checks,
        model_info=backend.info.as_dict() | {"file_hashes": backend.file_hashes()},
    )


def archive_gate_failure(outcome: GateOutcome) -> GateFailure:
    return GateFailure(outcome.verdict)


def seal_error(
    archive: RunArchive,
    *,
    gate: str,
    error: BaseException,
    logger: JsonLogger,
) -> None:
    message = f"{type(error).__name__}: {error}"
    report = "\n".join(
        [
            f"# {gate} run failed",
            "",
            f"`{message}`",
            "",
            "```",
            traceback.format_exc(),
            "```",
        ]
    )
    archive.seal(
        status="ERROR",
        verdict=message,
        result={"status_detail": "ERROR", "error": message},
        report=report,
        claims="# Claims\n\nNo claim is made by an errored run.\n",
    )
    logger.error("run_error", error=message)
