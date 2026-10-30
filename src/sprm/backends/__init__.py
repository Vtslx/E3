from __future__ import annotations

from sprm.backends.lfm3.backend import Lfm3Backend
from sprm.config.schema import BackendConfig

_BACKENDS = {"lfm3": Lfm3Backend}


def load_backend(config: BackendConfig, *, hash_files: bool = False, seed: int | None = None) -> Lfm3Backend:
    try:
        cls = _BACKENDS[config.kind]
    except KeyError as exc:
        raise ValueError(f"unsupported backend kind {config.kind!r}") from exc
    return cls(config, hash_files=hash_files, seed=seed)


__all__ = ["load_backend"]
