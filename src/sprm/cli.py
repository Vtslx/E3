"""SPRM command line.

Routes: ``backend check``, ``form-a build``, ``form-n build``, ``verify v0|v1|v2|d1``,
and ``archive list|verify``. Every ``verify`` run is written to a new immutable directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.config.schema import ExperimentConfig, load_experiment
from sprm.contracts.types import Sample
from sprm.errors import ArchiveError, ConfigError, SprmError, VerificationError
from sprm.eval.gates import (
    GateOutcome,
    gate_d1,
    gate_d5,
    gate_s1,
    gate_s2,
    gate_v0,
    gate_v1,
    gate_v2,
    seal_error,
)
from sprm.eval.niah import build_samples
from sprm.logging import JsonLogger
from sprm.runtime.form_a import (
    ExactWindow,
    MemoryBuild,
    build_memory,
    cache_footprint_bytes,
    exact_window_from_cache,
)
from sprm.runtime.form_n import build_prefix_scaffold, describe_prefix
from sprm.verification.archive import RunArchive, list_runs, verify_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sprm", description="SPRM verification harness")
    sub = parser.add_subparsers(dest="command", required=True)

    backend = sub.add_parser("backend", help="backend operations")
    backend_sub = backend.add_subparsers(dest="backend_command", required=True)
    backend_check = backend_sub.add_parser("check", help="load the model and run contract checks")
    backend_check.add_argument("--config", required=True)
    backend_check.add_argument("--quiet", action="store_true")
    backend_check.set_defaults(func=_cmd_backend_check)

    forma = sub.add_parser("form-a", help="Form A associative memory")
    forma_sub = forma.add_subparsers(dest="form_a_command", required=True)
    forma_build = forma_sub.add_parser("build", help="build memory for one sample and report diagnostics")
    forma_build.add_argument("--config", required=True)
    forma_build.add_argument("--kind", default="single_needle")
    forma_build.add_argument("--context-length", type=int, default=None)
    forma_build.add_argument("--depth", type=float, default=0.1)
    forma_build.add_argument("--quiet", action="store_true")
    forma_build.set_defaults(func=_cmd_form_a_build)

    formn = sub.add_parser("form-n", help="Form N compressed prefix")
    formn_sub = formn.add_subparsers(dest="form_n_command", required=True)
    formn_build = formn_sub.add_parser("build", help="plan the soft-token budget (no compressor)")
    formn_build.add_argument("--config", required=True)
    formn_build.add_argument("--kind", default="single_needle")
    formn_build.add_argument("--context-length", type=int, default=None)
    formn_build.add_argument("--depth", type=float, default=0.1)
    formn_build.add_argument("--quiet", action="store_true")
    formn_build.set_defaults(func=_cmd_form_n_build)

    verify = sub.add_parser("verify", help="run a verification gate")
    verify_sub = verify.add_subparsers(dest="gate", required=True)
    for gate in ("v0", "v1", "v2", "d1", "d5", "s1", "s2"):
        gate_parser = verify_sub.add_parser(gate)
        gate_parser.add_argument("--config", required=True)
        gate_parser.add_argument("--quiet", action="store_true")
        gate_parser.add_argument("--no-archive", action="store_true")
        gate_parser.add_argument("--archive-root", default=None)
        if gate == "d1":
            gate_parser.add_argument("--slots", type=int, default=None, help="override tier2.num_slots (512/2048)")
            gate_parser.add_argument("--kinds", default=None, help="comma separated task kinds")
            gate_parser.add_argument("--lengths", default=None, help="comma separated context lengths")
            gate_parser.add_argument("--depths", default=None, help="comma separated needle depths")
            gate_parser.add_argument("--samples-per-cell", type=int, default=None)
            gate_parser.add_argument("--max-samples", type=int, default=None)
            gate_parser.add_argument("--arms", default=None, help="comma separated arm ids")
            gate_parser.add_argument("--max-new-tokens", type=int, default=None)
        if gate == "d5":
            gate_parser.add_argument("--slots", type=int, default=None, help="override tier2.num_slots")
            gate_parser.add_argument("--arm", default="C4_memory_w1", help="memory arm for D5")
        if gate == "s1":
            gate_parser.add_argument("--max-new-tokens", type=int, default=32)
            gate_parser.add_argument("--gamma", type=int, default=4)
            gate_parser.add_argument("--ngram", type=int, default=3)
        if gate == "s2":
            gate_parser.add_argument("--arm", default="C3_memory", help="arm to speculate on")
            gate_parser.add_argument("--slots", type=int, default=None, help="override tier2.num_slots")
            gate_parser.add_argument("--max-new-tokens", type=int, default=32)
            gate_parser.add_argument("--gamma", type=int, default=4)
            gate_parser.add_argument("--ngram", type=int, default=3)
            gate_parser.add_argument("--prompt", default=None, help="path to a natural-text prompt")
            gate_parser.add_argument("--prompt-tokens", type=int, default=4096)
            gate_parser.add_argument("--partial", action="store_true", help="commit a matching prefix")
            gate_parser.add_argument(
                "--reservoir", action="store_true", help="use the frequency-ranked reservoir proposer"
            )
            gate_parser.add_argument("--draft-head", default=None, help="trained draft-head weights")
        gate_parser.set_defaults(func=_cmd_verify)

    archive = sub.add_parser("archive", help="inspect the verification archive")
    archive_sub = archive.add_subparsers(dest="archive_command", required=True)
    archive_list = archive_sub.add_parser("list", help="list sealed runs")
    archive_list.add_argument("--root", default="/Volumes/Net/E3/verification")
    archive_list.set_defaults(func=_cmd_archive_list)
    archive_verify = archive_sub.add_parser("verify", help="re-hash one sealed run")
    archive_verify.add_argument("run")
    archive_verify.set_defaults(func=_cmd_archive_verify)
    return parser


# ---------------------------------------------------------------------- helpers
def _load(path: str) -> ExperimentConfig:
    return load_experiment(path)


_NATURAL_DOCS = (
    "docs/FORM_A.md",
    "docs/VERIFICATION_PROTOCOL.md",
    "README.md",
    "docs/ADR-005-future-query-allocation.md",
    "docs/ADR-006-stability-anchors-replay-eta.md",
    "docs/ADR-007-prompt-lookup-speculative.md",
    "docs/ADR-008-w1-cut.md",
    "docs/ADR-009-c3-stability-regularization.md",
)


def _natural_prompt(path: str | None) -> str:
    """A deterministic, non-repetitive prose prompt for S2.

    The default concatenates the design and verification documents in the repository, so the
    prompt is versioned with the source tree. ``--prompt`` overrides it with a file.
    """
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    parts = [
        (root / name).read_text(encoding="utf-8")
        for name in _NATURAL_DOCS
        if (root / name).exists()
    ]
    if not parts:
        raise ConfigError("no natural-text prompt found; pass --prompt")
    return "\n\n".join(parts)


def _backend(config: ExperimentConfig, *, quiet: bool = False) -> Lfm3Backend:
    return Lfm3Backend(config.backend, seed=config.seed)


def _logger(gate: str, quiet: bool) -> JsonLogger:
    logger = JsonLogger(component=gate)
    logger._echo = not quiet
    return logger


def _split(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _verify_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kinds": _split(getattr(args, "kinds", None)),
        "lengths": [int(x) for x in _split(getattr(args, "lengths", None)) or []] or None,
        "depths": [float(x) for x in _split(getattr(args, "depths", None)) or []] or None,
        "samples_per_cell": getattr(args, "samples_per_cell", None),
        "max_samples": getattr(args, "max_samples", None),
        "arms": _split(getattr(args, "arms", None)),
        "max_new_tokens": getattr(args, "max_new_tokens", None),
        "slots": getattr(args, "slots", None),
    }


def _reduced(overrides: dict[str, Any], config: ExperimentConfig) -> bool:
    return any(
        overrides.get(key) for key in ("kinds", "lengths", "depths", "samples_per_cell", "max_samples", "arms")
    )


def _print_outcome(outcome: GateOutcome, run_dir: Path | None) -> None:
    print(outcome.report)
    print()
    print(outcome.claims)
    if run_dir is not None:
        print(f"\nrun directory: {run_dir}")


def _terminal_status(status: str) -> int:
    if status == "PASS":
        return 0
    if status == "FAIL":
        return 8
    return 7


# ---------------------------------------------------------------------- backend
def _cmd_backend_check(args: argparse.Namespace) -> int:
    config = _load(args.config)
    logger = _logger("v0_contract", args.quiet)
    with _backend(config, quiet=args.quiet) as backend:
        outcome = gate_v0(backend, config, logger, archive=None, hash_files=True)
    _print_outcome(outcome, None)
    return _terminal_status(outcome.status)


# ---------------------------------------------------------------------- form A
def _one_sample(
    backend: Lfm3Backend, config: ExperimentConfig, kind: str, length: int | None, depth: float
) -> Sample:
    lengths = [length] if length is not None else [config.tasks.context_lengths[0]]
    return build_samples(
        backend.tokenizer,
        config.tasks,
        kinds=[kind],
        context_lengths=lengths,
        depths=[depth],
        samples_per_cell=1,
        max_samples=1,
    )[0]


def _recon_for_window(backend: Lfm3Backend, build: MemoryBuild, window: ExactWindow) -> float:
    info = backend.info
    errors: list[float] = []
    for layer_idx in info.global_layers:
        keys_post = window.keys[layer_idx]
        values = window.values[layer_idx]
        positions = window.positions[-keys_post.shape[-2] :]
        unrotated = backend.unrotate_layer(layer_idx, keys_post, positions)
        flat_keys = unrotated.transpose(1, 2).reshape(keys_post.shape[-2], info.kv_width)
        flat_values = values.transpose(1, 2).reshape(values.shape[-2], info.kv_width)
        errors.append(build.memories[layer_idx].recon_error(flat_keys, flat_values))
    return float(sum(errors) / len(errors)) if errors else float("nan")


def _cmd_form_a_build(args: argparse.Namespace) -> int:
    config = _load(args.config)
    with _backend(config, quiet=args.quiet) as backend:
        sample = _one_sample(backend, config, args.kind, args.context_length, args.depth)
        prompt = list(sample.full_token_ids)
        cache, _, _ = backend.prefill_exact(prompt, chunk_tokens=config.tier1.prefill_chunk_tokens)
        build = build_memory(
            backend,
            cache,
            config,
            context_length=sample.context_length,
            full_length=len(prompt),
            allocation="effective_rank",
        )
        window = exact_window_from_cache(
            backend, cache, capacity=config.tier1.window_tokens, full_length=len(prompt)
        )
        recon = _recon_for_window(backend, build, window) if build is not None else float("nan")
        payload = {
            "sample": {
                "sample_id": sample.sample_id,
                "kind": sample.task_kind,
                "context_length": sample.context_length,
                "depth": sample.depth,
                "prompt_tokens": len(prompt),
                "gold_answer": sample.gold_answer,
                "correct_index": sample.metadata.get("correct_index"),
            },
            "memory": None,
            "reconstruction_error_window": recon,
        }
        if build is not None:
            payload["memory"] = {
                "blocks": len(build.block_ranges),
                "block_ranges": [list(pair) for pair in build.block_ranges],
                "slots_per_block": list(build.slots_per_block),
                "importances": list(build.importances),
                "total_slots": build.total_slots,
                "build_seconds": build.build_seconds,
                "memory_bytes": build.memory_bytes(),
                "forgetting_events": build.forgetting_events,
                "condition_checks": {str(k): v for k, v in build.condition_checks.items()},
            }
            bytes_total, per_token = cache_footprint_bytes(
                backend, cache, build, context_length=sample.context_length
            )
            payload["cache_bytes"] = bytes_total
            payload["kv_bytes_per_token"] = per_token
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


# ---------------------------------------------------------------------- form N
def _cmd_form_n_build(args: argparse.Namespace) -> int:
    config = _load(args.config)
    with _backend(config, quiet=args.quiet) as backend:
        sample = _one_sample(backend, config, args.kind, args.context_length, args.depth)
        build = build_prefix_scaffold(
            backend.info,
            config,
            context_length=sample.context_length,
            full_length=len(sample.full_token_ids),
        )
        payload = {
            "sample": {
                "sample_id": sample.sample_id,
                "context_length": sample.context_length,
                "prompt_tokens": len(sample.full_token_ids),
            },
            "prefix": None if build is None else describe_prefix(build),
        }
        if build is None:
            payload["prefix"] = {"status": "no-eviction", "note": "prompt fits in sink+window"}
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


# ---------------------------------------------------------------------- verify
def _cmd_verify(args: argparse.Namespace) -> int:
    config = _load(args.config)
    config_sha256 = getattr(config, "config_sha256", "0" * 64)
    gate = args.gate
    slots = getattr(args, "slots", None)
    if slots is not None:
        config = replace(config, tier2=replace(config.tier2, num_slots=int(slots)))
    overrides = _verify_overrides(args) if gate == "d1" else {}
    reduced = _reduced(overrides, config) if gate == "d1" else False
    slug = config.name
    if slots is not None:
        slug = f"{slug}_slots{slots}"
    arm_for_slug = str(getattr(args, "arm", "")) if gate == "d5" else ""
    if arm_for_slug and arm_for_slug != "C4_memory_w1":
        slug = f"{slug}_{arm_for_slug.split('_')[0].lower()}"
    if gate == "s2":
        slug = f"{slug}_{str(getattr(args, 'arm', 'C3_memory')).split('_')[0].lower()}"
        if bool(getattr(args, "partial", False)):
            slug = f"{slug}_partial"
        if bool(getattr(args, "reservoir", False)):
            slug = f"{slug}_reservoir"
        if getattr(args, "draft_head", None):
            slug = f"{slug}_draft"
    if reduced:
        slug = f"{slug}_reduced"
    archive: RunArchive | None = None
    if not args.no_archive:
        root = args.archive_root or config.archive.root
        archive = RunArchive.create(
            root=root,
            gate=gate,
            slug=slug,
            config_sha256=config_sha256,
            config_path=config.source_path,
            argv=sys.argv,
            echo=not args.quiet,
        )
    logger = archive.logger if archive is not None else _logger(gate, args.quiet)
    outcome: GateOutcome | None = None
    backend: Lfm3Backend | None = None
    try:
        if gate == "v1":
            outcome = gate_v1(config, logger, archive=archive)
        else:
            backend = _backend(config, quiet=args.quiet)
            if gate == "v0":
                outcome = gate_v0(backend, config, logger, archive=archive, hash_files=True)
            elif gate == "v2":
                outcome = gate_v2(backend, config, logger, archive=archive)
            elif gate == "d5":
                arm_id = str(getattr(args, "arm", "C4_memory_w1"))
                outcome = gate_d5(backend, config, logger, archive=archive, arm_id=arm_id)
                outcome.extra = {
                    "slot_budget": slots if slots is not None else config.tier2.num_slots,
                    "arm": arm_id,
                }
            elif gate == "s1":
                outcome = gate_s1(
                    backend,
                    config,
                    logger,
                    archive=archive,
                    max_new_tokens=int(getattr(args, "max_new_tokens", 32)),
                    gamma=int(getattr(args, "gamma", 4)),
                    ngram=int(getattr(args, "ngram", 3)),
                )
                outcome.extra = {
                    "max_new_tokens": int(getattr(args, "max_new_tokens", 32)),
                    "gamma": int(getattr(args, "gamma", 4)),
                    "ngram": int(getattr(args, "ngram", 3)),
                }
            elif gate == "s2":
                outcome = gate_s2(
                    backend,
                    config,
                    logger,
                    archive=archive,
                    max_new_tokens=int(getattr(args, "max_new_tokens", 32)),
                    gamma=int(getattr(args, "gamma", 4)),
                    ngram=int(getattr(args, "ngram", 3)),
                    arm_id=str(getattr(args, "arm", "C3_memory")),
                    prompt_text=_natural_prompt(getattr(args, "prompt", None)),
                    prompt_tokens=int(getattr(args, "prompt_tokens", 4096)),
                    partial_accept=bool(getattr(args, "partial", False)),
                    proposer_kind=(
                        "reservoir" if bool(getattr(args, "reservoir", False)) else "prompt"
                    ),
                    draft_head=getattr(args, "draft_head", None),
                )
                outcome.extra = {
                    "max_new_tokens": int(getattr(args, "max_new_tokens", 32)),
                    "gamma": int(getattr(args, "gamma", 4)),
                    "ngram": int(getattr(args, "ngram", 3)),
                    "arm": str(getattr(args, "arm", "C3_memory")),
                    "prompt_tokens": int(getattr(args, "prompt_tokens", 4096)),
                    "partial": bool(getattr(args, "partial", False)),
                    "proposer": str(getattr(args, "reservoir", False)),
                    "draft_head": getattr(args, "draft_head", None),
                }
            elif gate == "d1":
                kinds = overrides.get("kinds") or list(config.tasks.kinds)
                lengths = overrides.get("lengths") or list(config.tasks.context_lengths)
                depths = overrides.get("depths") or list(config.tasks.depths)
                samples = build_samples(
                    backend.tokenizer,
                    config.tasks,
                    kinds=kinds,
                    context_lengths=lengths,
                    depths=depths,
                    samples_per_cell=overrides.get("samples_per_cell"),
                    max_samples=overrides.get("max_samples"),
                )
                arm_ids = overrides.get("arms") or list(config.arms)
                unknown = [arm_id for arm_id in arm_ids if arm_id not in config.arms]
                if unknown:
                    raise ConfigError(f"unknown arms in --arms: {unknown}; configured={sorted(config.arms)}")
                logger.info(
                    "d1_plan",
                    samples=len(samples),
                    arms=arm_ids,
                    kinds=kinds,
                    lengths=lengths,
                    depths=depths,
                    reduced=reduced,
                )
                outcome = gate_d1(
                    backend,
                    config,
                    samples,
                    arm_ids,
                    logger,
                    archive=archive,
                    max_new_tokens=overrides.get("max_new_tokens"),
                )
                outcome.extra = {
                    "overrides": overrides,
                    "sample_count": len(samples),
                    "arms": arm_ids,
                    "reduced": reduced,
                    "slot_budget": slots if slots is not None else config.tier2.num_slots,
                }
            else:  # pragma: no cover - argparse constrains this
                raise VerificationError(f"unknown gate {gate!r}")
    except Exception as exc:
        if archive is not None and not archive._sealed:
            seal_error(archive, gate=gate, error=exc, logger=logger)
            print(f"errored run: {archive.run_dir}", file=sys.stderr)
        if isinstance(exc, SprmError):
            raise
        raise VerificationError(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}") from exc
    finally:
        if backend is not None:
            backend.close()
    assert outcome is not None
    if archive is not None:
        archive.seal(
            status=outcome.status,
            verdict=outcome.verdict,
            result=outcome.result,
            report=outcome.report,
            claims=outcome.claims,
            model_info=outcome.model_info,
            extra=outcome.extra,
        )
    _print_outcome(outcome, archive.run_dir if archive is not None else None)
    return _terminal_status(outcome.status)


# ---------------------------------------------------------------------- archive
def _cmd_archive_list(args: argparse.Namespace) -> int:
    entries = list_runs(args.root)
    if not entries:
        print(f"no runs under {args.root}")
        return 0
    print(f"{'run_id':<70} {'gate':<5} {'status':<6} verdict")
    for entry in entries:
        print(
            f"{entry.get('run_id', ''):<70} {entry.get('gate', ''):<5} "
            f"{entry.get('status', ''):<6} {entry.get('verdict', '')}"
        )
    return 0


def _cmd_archive_verify(args: argparse.Namespace) -> int:
    problems = verify_run(args.run)
    if not problems:
        print(f"OK: {args.run}")
        return 0
    print(f"{len(problems)} problem(s) in {args.run}:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return ArchiveError.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SprmError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
