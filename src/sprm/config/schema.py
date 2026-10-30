"""Strict configuration schema.

YAML is loaded once and validated into frozen dataclasses. Unknown keys are errors: a
misspelled `window_tokens` must not silently fall back to a default. Every range check
that protects an experiment from being meaningless lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sprm.errors import ConfigError


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}; allowed={sorted(allowed)}")


def _int(value: Any, where: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{where} must be <= {maximum}, got {value}")
    return value


def _float(value: Any, where: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {type(value).__name__}")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{where} must be <= {maximum}, got {result}")
    return result


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a bool, got {type(value).__name__}")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty string")
    return value


_DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


@dataclass(frozen=True)
class BackendConfig:
    kind: str
    model_path: str
    device: str = "mps"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    local_files_only: bool = True

    @classmethod
    def from_dict(cls, raw: Any, where: str = "backend") -> BackendConfig:
        d = _mapping(raw, where)
        _reject_unknown(
            d,
            {
                "kind",
                "model_path",
                "device",
                "dtype",
                "attn_implementation",
                "trust_remote_code",
                "local_files_only",
            },
            where,
        )
        kind = _str(d.get("kind"), f"{where}.kind")
        if kind != "lfm3":
            raise ConfigError(f"{where}.kind: only 'lfm3' is supported in this round, got {kind!r}")
        dtype = _str(d.get("dtype", "bfloat16"), f"{where}.dtype")
        if dtype not in _DTYPES:
            raise ConfigError(f"{where}.dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        device = _str(d.get("device", "mps"), f"{where}.device")
        if device not in {"mps", "cpu", "cuda"}:
            raise ConfigError(f"{where}.device must be mps/cpu/cuda, got {device!r}")
        model_path = _str(d.get("model_path"), f"{where}.model_path")
        if not Path(model_path).is_dir():
            raise ConfigError(f"{where}.model_path is not a directory: {model_path}")
        attn = _str(d.get("attn_implementation", "sdpa"), f"{where}.attn_implementation")
        if attn not in {"sdpa", "eager"}:
            raise ConfigError(f"{where}.attn_implementation must be sdpa or eager, got {attn!r}")
        return cls(
            kind=kind,
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn,
            trust_remote_code=_bool(d.get("trust_remote_code", True), f"{where}.trust_remote_code"),
            local_files_only=_bool(d.get("local_files_only", True), f"{where}.local_files_only"),
        )


@dataclass(frozen=True)
class Tier1Config:
    sink_tokens: int
    window_tokens: int
    prefill_chunk_tokens: int

    @classmethod
    def from_dict(cls, raw: Any, where: str = "tier1") -> Tier1Config:
        d = _mapping(raw, where)
        _reject_unknown(d, {"sink_tokens", "window_tokens", "prefill_chunk_tokens"}, where)
        sink = _int(d.get("sink_tokens"), f"{where}.sink_tokens", minimum=0)
        window = _int(d.get("window_tokens"), f"{where}.window_tokens", minimum=1)
        chunk = _int(d.get("prefill_chunk_tokens"), f"{where}.prefill_chunk_tokens", minimum=1)
        if window % chunk != 0:
            raise ConfigError(
                f"{where}: window_tokens ({window}) must be a multiple of prefill_chunk_tokens ({chunk}) "
                "so that every eviction block is exactly one chunk"
            )
        return cls(sink_tokens=sink, window_tokens=window, prefill_chunk_tokens=chunk)


@dataclass(frozen=True)
class Tier2Config:
    form: Literal["A", "N"]
    num_slots: int
    read_temperature: float
    write_gain: float
    gate: float
    forgetting: float
    trust_region_eps: float
    key_l2_normalize: bool
    allocation: Literal["none", "uniform", "effective_rank", "attention"]
    block_tokens: int
    kmeans_iters: int
    refine_passes: int
    condition_every_blocks: int
    min_slots_per_block: int
    max_slot_multiplier: float

    @classmethod
    def from_dict(cls, raw: Any, where: str = "tier2") -> Tier2Config:
        d = _mapping(raw, where)
        _reject_unknown(
            d,
            {
                "form",
                "num_slots",
                "read_temperature",
                "write_gain",
                "gate",
                "forgetting",
                "trust_region_eps",
                "key_l2_normalize",
                "allocation",
                "block_tokens",
                "kmeans_iters",
                "refine_passes",
                "condition_every_blocks",
                "min_slots_per_block",
                "max_slot_multiplier",
            },
            where,
        )
        form = _str(d.get("form"), f"{where}.form")
        if form not in {"A", "N"}:
            raise ConfigError(f"{where}.form must be 'A' or 'N', got {form!r}")
        allocation = _str(d.get("allocation"), f"{where}.allocation")
        if allocation not in {"none", "uniform", "effective_rank", "attention"}:
            raise ConfigError(f"{where}.allocation is invalid: {allocation!r}")
        return cls(
            form=form,  # type: ignore[arg-type]
            num_slots=_int(d.get("num_slots"), f"{where}.num_slots", minimum=1),
            read_temperature=_float(d.get("read_temperature"), f"{where}.read_temperature", minimum=1e-3),
            write_gain=_float(d.get("write_gain"), f"{where}.write_gain", minimum=0.0, maximum=1.0),
            gate=_float(d.get("gate"), f"{where}.gate", minimum=0.0, maximum=1.0),
            forgetting=_float(d.get("forgetting"), f"{where}.forgetting", minimum=0.0, maximum=1.0),
            trust_region_eps=_float(d.get("trust_region_eps"), f"{where}.trust_region_eps", minimum=0.0),
            key_l2_normalize=_bool(d.get("key_l2_normalize"), f"{where}.key_l2_normalize"),
            allocation=allocation,  # type: ignore[arg-type]
            block_tokens=_int(d.get("block_tokens"), f"{where}.block_tokens", minimum=1),
            kmeans_iters=_int(d.get("kmeans_iters"), f"{where}.kmeans_iters", minimum=1),
            refine_passes=_int(d.get("refine_passes"), f"{where}.refine_passes", minimum=0),
            condition_every_blocks=_int(
                d.get("condition_every_blocks"), f"{where}.condition_every_blocks", minimum=1
            ),
            min_slots_per_block=_int(d.get("min_slots_per_block"), f"{where}.min_slots_per_block", minimum=0),
            max_slot_multiplier=_float(
                d.get("max_slot_multiplier"), f"{where}.max_slot_multiplier", minimum=1.0
            ),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    max_new_tokens: int
    long_decode_tokens: int
    deterministic: bool
    log_every_steps: int
    probe_tokens: int

    @classmethod
    def from_dict(cls, raw: Any, where: str = "runtime") -> RuntimeConfig:
        d = _mapping(raw, where)
        _reject_unknown(
            d,
            {"max_new_tokens", "long_decode_tokens", "deterministic", "log_every_steps", "probe_tokens"},
            where,
        )
        return cls(
            max_new_tokens=_int(d.get("max_new_tokens"), f"{where}.max_new_tokens", minimum=1),
            long_decode_tokens=_int(d.get("long_decode_tokens"), f"{where}.long_decode_tokens", minimum=0),
            deterministic=_bool(d.get("deterministic"), f"{where}.deterministic"),
            log_every_steps=_int(d.get("log_every_steps"), f"{where}.log_every_steps", minimum=1),
            probe_tokens=_int(d.get("probe_tokens"), f"{where}.probe_tokens", minimum=0),
        )


_TASK_KINDS = {"single_needle", "multi_needle", "conditional"}


@dataclass(frozen=True)
class TaskConfig:
    kinds: tuple[str, ...]
    context_lengths: tuple[int, ...]
    samples_per_cell: int
    depths: tuple[float, ...]
    filler_seed: int
    needle_seed: int
    use_chat_template: bool
    max_answer_tokens: int
    long_form_records: int

    @classmethod
    def from_dict(cls, raw: Any, where: str = "tasks") -> TaskConfig:
        d = _mapping(raw, where)
        _reject_unknown(
            d,
            {
                "kinds",
                "context_lengths",
                "samples_per_cell",
                "depths",
                "filler_seed",
                "needle_seed",
                "use_chat_template",
                "max_answer_tokens",
                "long_form_records",
            },
            where,
        )
        kinds_raw = d.get("kinds")
        if not isinstance(kinds_raw, list) or not kinds_raw:
            raise ConfigError(f"{where}.kinds must be a non-empty list")
        kinds = tuple(_str(k, f"{where}.kinds[]") for k in kinds_raw)
        unknown = sorted(set(kinds) - _TASK_KINDS)
        if unknown:
            raise ConfigError(f"{where}.kinds: unknown task kinds {unknown}")
        lengths_raw = d.get("context_lengths")
        if not isinstance(lengths_raw, list) or not lengths_raw:
            raise ConfigError(f"{where}.context_lengths must be a non-empty list")
        lengths = tuple(_int(x, f"{where}.context_lengths[]", minimum=2) for x in lengths_raw)
        if tuple(sorted(set(lengths))) != lengths:
            raise ConfigError(f"{where}.context_lengths must be strictly increasing and unique")
        depths_raw = d.get("depths")
        if not isinstance(depths_raw, list) or not depths_raw:
            raise ConfigError(f"{where}.depths must be a non-empty list")
        depths = tuple(_float(x, f"{where}.depths[]", minimum=0.0, maximum=1.0) for x in depths_raw)
        if any(x <= 0.0 or x >= 1.0 for x in depths):
            raise ConfigError(f"{where}.depths entries must lie strictly between 0 and 1")
        return cls(
            kinds=kinds,
            context_lengths=lengths,
            samples_per_cell=_int(d.get("samples_per_cell"), f"{where}.samples_per_cell", minimum=1),
            depths=depths,
            filler_seed=_int(d.get("filler_seed"), f"{where}.filler_seed"),
            needle_seed=_int(d.get("needle_seed"), f"{where}.needle_seed"),
            use_chat_template=_bool(d.get("use_chat_template"), f"{where}.use_chat_template"),
            max_answer_tokens=_int(d.get("max_answer_tokens"), f"{where}.max_answer_tokens", minimum=1),
            long_form_records=_int(d.get("long_form_records"), f"{where}.long_form_records", minimum=0),
        )


@dataclass(frozen=True)
class ArmConfig:
    id: str
    description: str
    memory: bool
    writeback: bool
    compressed: bool
    allocation: Literal["none", "uniform", "effective_rank", "attention"]

    @classmethod
    def from_dict(cls, arm_id: str, raw: Any, where: str) -> ArmConfig:
        d = _mapping(raw, where)
        _reject_unknown(d, {"description", "memory", "writeback", "compressed", "allocation"}, where)
        allocation = _str(d.get("allocation"), f"{where}.allocation")
        if allocation not in {"none", "uniform", "effective_rank", "attention"}:
            raise ConfigError(f"{where}.allocation is invalid: {allocation!r}")
        memory = _bool(d.get("memory"), f"{where}.memory")
        writeback = _bool(d.get("writeback"), f"{where}.writeback")
        compressed = _bool(d.get("compressed", memory), f"{where}.compressed")
        if writeback and not memory:
            raise ConfigError(f"{where}: writeback=true requires memory=true")
        if memory and not compressed:
            raise ConfigError(f"{where}: memory=true requires compressed=true")
        if not memory and allocation != "none":
            raise ConfigError(f"{where}: memory=false requires allocation='none'")
        return cls(
            id=arm_id,
            description=_str(d.get("description"), f"{where}.description"),
            memory=memory,
            writeback=writeback,
            compressed=compressed,
            allocation=allocation,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class GateConfig:
    v2_max_logit_diff: float
    v2_min_top1_agreement: float
    d1_min_absolute_gain: float
    d1_min_tasks_improved: int
    d1_min_far_distance: int
    d1_max_near_degradation: float
    d1_mcnemar_alpha: float
    d5_max_condition: float
    d5_min_eta_ratio: float

    @classmethod
    def from_dict(cls, raw: Any, where: str = "gates") -> GateConfig:
        d = _mapping(raw, where)
        _reject_unknown(
            d,
            {
                "v2_max_logit_diff",
                "v2_min_top1_agreement",
                "d1_min_absolute_gain",
                "d1_min_tasks_improved",
                "d1_min_far_distance",
                "d1_max_near_degradation",
                "d1_mcnemar_alpha",
                "d5_max_condition",
                "d5_min_eta_ratio",
            },
            where,
        )
        return cls(
            v2_max_logit_diff=_float(d.get("v2_max_logit_diff"), f"{where}.v2_max_logit_diff", minimum=0.0),
            v2_min_top1_agreement=_float(
                d.get("v2_min_top1_agreement"), f"{where}.v2_min_top1_agreement", minimum=0.0, maximum=1.0
            ),
            d1_min_absolute_gain=_float(
                d.get("d1_min_absolute_gain"), f"{where}.d1_min_absolute_gain", minimum=0.0
            ),
            d1_min_tasks_improved=_int(
                d.get("d1_min_tasks_improved"), f"{where}.d1_min_tasks_improved", minimum=1
            ),
            d1_min_far_distance=_int(
                d.get("d1_min_far_distance"), f"{where}.d1_min_far_distance", minimum=1
            ),
            d1_max_near_degradation=_float(
                d.get("d1_max_near_degradation"), f"{where}.d1_max_near_degradation", minimum=0.0
            ),
            d1_mcnemar_alpha=_float(
                d.get("d1_mcnemar_alpha"), f"{where}.d1_mcnemar_alpha", minimum=0.0, maximum=1.0
            ),
            d5_max_condition=_float(d.get("d5_max_condition"), f"{where}.d5_max_condition", minimum=1.0),
            d5_min_eta_ratio=_float(d.get("d5_min_eta_ratio"), f"{where}.d5_min_eta_ratio", minimum=0.0),
        )


@dataclass(frozen=True)
class ArchiveConfig:
    root: str
    keep_artifacts: bool

    @classmethod
    def from_dict(cls, raw: Any, where: str = "archive") -> ArchiveConfig:
        d = _mapping(raw, where)
        _reject_unknown(d, {"root", "keep_artifacts"}, where)
        return cls(
            root=_str(d.get("root"), f"{where}.root"),
            keep_artifacts=_bool(d.get("keep_artifacts"), f"{where}.keep_artifacts"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    backend: BackendConfig
    tier1: Tier1Config
    tier2: Tier2Config
    runtime: RuntimeConfig
    tasks: TaskConfig
    arms: dict[str, ArmConfig]
    gates: GateConfig
    archive: ArchiveConfig
    source_path: Path = field(default_factory=Path, compare=False)

    @classmethod
    def from_dict(cls, raw: Any, source_path: Path | None = None) -> ExperimentConfig:
        d = _mapping(raw, "experiment")
        _reject_unknown(
            d,
            {
                "name",
                "seed",
                "backend",
                "tier1",
                "tier2",
                "runtime",
                "tasks",
                "arms",
                "gates",
                "archive",
            },
            "experiment",
        )
        arms_raw = _mapping(d.get("arms"), "arms")
        if not arms_raw:
            raise ConfigError("arms must not be empty")
        arms = {
            arm_id: ArmConfig.from_dict(arm_id, arm_raw, f"arms.{arm_id}")
            for arm_id, arm_raw in arms_raw.items()
        }
        if "C0_full" not in arms:
            raise ConfigError("arms must contain C0_full as the exact-KV control")
        if arms["C0_full"].memory:
            raise ConfigError("arms.C0_full must have memory=false")
        return cls(
            name=_str(d.get("name"), "name"),
            seed=_int(d.get("seed"), "seed"),
            backend=BackendConfig.from_dict(d.get("backend")),
            tier1=Tier1Config.from_dict(d.get("tier1")),
            tier2=Tier2Config.from_dict(d.get("tier2")),
            runtime=RuntimeConfig.from_dict(d.get("runtime")),
            tasks=TaskConfig.from_dict(d.get("tasks")),
            arms=arms,
            gates=GateConfig.from_dict(d.get("gates")),
            archive=ArchiveConfig.from_dict(d.get("archive")),
            source_path=source_path or Path(""),
        )


def load_experiment(path: str | Path) -> ExperimentConfig:
    import hashlib

    import yaml

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"config file not found: {source}")
    raw_bytes = source.read_bytes()
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must contain a top-level mapping")
    config = ExperimentConfig.from_dict(raw, source_path=source)
    # Keep the digest available without a second file read.
    object.__setattr__(config, "config_sha256", hashlib.sha256(raw_bytes).hexdigest())
    return config
