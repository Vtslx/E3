"""LFM3 checkpoint loading.

The local checkpoint was fetched by a downloader that renamed several remote-code files to
``*.js`` descriptors. Loading therefore fails unless the real Python files are present. We
detect the exact missing files and print the two commands that repair the checkpoint instead
of letting ``trust_remote_code`` fail with an opaque import error.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sprm.backends.lfm3.architecture import LayerPlan, build_layer_plan
from sprm.config.schema import BackendConfig
from sprm.contracts.types import BackendInfo
from sprm.errors import BackendError

REQUIRED_REMOTE_CODE = (
    "modeling_lfm3.py",
    "configuration_lfm3.py",
    "lfm3_conv.py",
    "tokenizer_config.json",
)


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_checkpoint(model_path: Path) -> None:
    if not model_path.is_dir():
        raise BackendError(f"model path is not a directory: {model_path}")
    missing = [name for name in REQUIRED_REMOTE_CODE if not (model_path / name).is_file()]
    if missing:
        repair = "\n".join(
            f'curl -fsSL "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/{name}" '
            f'-o "{model_path / name}"'
            for name in missing
        )
        raise BackendError(
            "LFM3 checkpoint is missing remote-code files "
            f"{missing}. The downloader renamed them to *.js descriptors. Repair with:\n{repair}"
        )
    if not (model_path / "config.json").is_file():
        raise BackendError(f"missing config.json in {model_path}")
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise BackendError(f"missing *.safetensors weights in {model_path}")


def load_tokenizer(config: BackendConfig) -> Any:
    path = Path(config.model_path)
    _check_checkpoint(path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
    except Exception as exc:
        raise BackendError(f"failed to load LFM3 tokenizer from {path}: {exc}") from exc
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise BackendError("tokenizer has neither a pad token nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: BackendConfig) -> tuple[Any, Any, LayerPlan]:
    path = Path(config.model_path)
    _check_checkpoint(path)
    tokenizer = load_tokenizer(config)
    dtype = _torch_dtype(config.dtype)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise BackendError("device=mps requested but MPS is not available")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise BackendError("device=cuda requested but CUDA is not available")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            dtype=dtype,
            device_map={"": config.device},
            attn_implementation=config.attn_implementation,
        )
    except Exception as exc:
        raise BackendError(
            f"failed to load LFM3 model from {path} (dtype={config.dtype}, device={config.device}, "
            f"attn={config.attn_implementation}): {exc}"
        ) from exc
    model.eval()  # type: ignore[no-untyped-call]
    plan = build_layer_plan(model.config)
    return model, tokenizer, plan


def build_info(
    config: BackendConfig,
    model: Any,
    tokenizer: Any,
    plan: LayerPlan,
    *,
    hash_files: bool = False,
) -> BackendInfo:
    path = Path(config.model_path)
    model_sha = tokenizer_sha = remote_sha = None
    if hash_files:
        weights = sorted(path.glob("*.safetensors"))
        if len(weights) != 1:
            raise BackendError(f"expected exactly one safetensors shard, found {len(weights)}")
        model_sha = _sha256_file(weights[0])
        tokenizer_sha = _sha256_file(path / "tokenizer.json")
        digest = hashlib.sha256()
        for name in ("modeling_lfm3.py", "configuration_lfm3.py", "lfm3_conv.py"):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(path / name).encode("ascii"))
            digest.update(b"\n")
        remote_sha = digest.hexdigest()
    return BackendInfo(
        kind="lfm3",
        model_path=str(path),
        device=config.device,
        dtype=config.dtype,
        num_layers=plan.num_layers,
        hidden_size=int(model.config.hidden_size),
        num_attention_heads=int(model.config.num_attention_heads),
        num_kv_heads=int(model.config.num_key_value_heads),
        head_dim=int(model.config.head_dim),
        vocab_size=int(model.config.vocab_size),
        sliding_window=getattr(model.config, "sliding_window", None),
        max_position_embeddings=int(getattr(model.config, "max_position_embeddings", 0)),
        layer_kinds=plan.kinds,  # type: ignore[arg-type]
        global_layers=plan.global_layers,
        sliding_layers=plan.sliding_layers,
        conv_layers=plan.conv_layers,
        model_sha256=model_sha,
        tokenizer_sha256=tokenizer_sha,
        remote_code_sha256=remote_sha,
    )
