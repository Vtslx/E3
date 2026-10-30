"""Immutable verification archive.

A run directory is created once, written to completion, then sealed with a manifest of
SHA-256 digests. Re-running a gate creates a new directory; it never mutates an old one.
``sprm archive verify`` re-hashes a sealed run and reports any mismatch.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from sprm.errors import ArchiveError
from sprm.logging import JsonLogger

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, suffixes: tuple[str, ...] = (".py", ".yaml", ".toml", ".md")) -> str:
    """Order-independent digest of source files under ``root``."""
    digest = hashlib.sha256()
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in suffixes
        and "__pycache__" not in p.parts
        and ".venv" not in p.parts
    )
    for path in files:
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (slug or "run")[:48]


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def environment_snapshot() -> dict[str, Any]:
    import torch
    import transformers

    info: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
    }
    if torch.backends.mps.is_available():
        info["mps_recommended_max_memory_bytes"] = int(torch.mps.recommended_max_memory())
    return info


@dataclass
class RunArchive:
    run_dir: Path
    run_id: str
    gate: str
    config_sha256: str
    logger: JsonLogger
    started_at: str
    argv: tuple[str, ...]
    _sealed: bool = False

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(
        cls,
        *,
        root: str | Path,
        gate: str,
        slug: str,
        config_sha256: str,
        config_path: str | Path | None,
        argv: Iterable[str],
        echo: bool = True,
    ) -> RunArchive:
        archive_root = Path(root).expanduser().resolve()
        archive_root.mkdir(parents=True, exist_ok=True)
        run_id = f"{utc_stamp()}_{_slugify(gate)}_{_slugify(slug)}_{config_sha256[:8]}"
        run_dir = archive_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArchiveError(f"run directory already exists, refusing to overwrite: {run_dir}") from exc
        (run_dir / "artifacts").mkdir()
        (run_dir / "logs").mkdir()

        if config_path is not None:
            source = Path(config_path).expanduser().resolve()
            if source.is_file():
                shutil.copyfile(source, run_dir / "config.yaml")
        logger = JsonLogger(component=gate)
        logger.add_sink((run_dir / "logs" / "stdout.log").open("a", encoding="utf-8"))
        logger.add_sink((run_dir / "logs" / "events.jsonl").open("a", encoding="utf-8"))
        logger._echo = echo
        archive = cls(
            run_dir=run_dir,
            run_id=run_id,
            gate=gate,
            config_sha256=config_sha256,
            logger=logger,
            started_at=utc_now(),
            argv=tuple(str(a) for a in argv),
        )
        archive.logger.info("run_created", run_id=run_id, run_dir=str(run_dir), gate=gate)
        return archive

    # ------------------------------------------------------------------ writers
    def write_json(self, relative: str, payload: Any) -> Path:
        self._assert_open()
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, payload)
        return path

    def write_text(self, relative: str, text: str) -> Path:
        self._assert_open()
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def append_jsonl(self, relative: str, record: dict[str, Any]) -> None:
        self._assert_open()
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------ sealing
    def seal(
        self,
        *,
        status: str,
        verdict: str,
        result: dict[str, Any],
        report: str,
        claims: str,
        model_info: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        self._assert_open()
        if status not in {"PASS", "FAIL", "ERROR"}:
            raise ArchiveError(f"invalid status {status!r}")
        finished_at = utc_now()
        self.write_json("RESULT.json", {"schema_version": SCHEMA_VERSION, "status": status, "verdict": verdict, **result})
        self.write_text("REPORT.md", report)
        self.write_text("CLAIMS.md", claims)
        self.logger.info("run_sealed", status=status, verdict=verdict)
        self.logger.close()

        artifact_hashes: dict[str, dict[str, Any]] = {}
        for path in sorted(self.run_dir.rglob("*")):
            if not path.is_file() or path.name in {"MANIFEST.json", "MANIFEST.sha256"}:
                continue
            if path.name.endswith(".tmp"):
                continue
            rel = path.relative_to(self.run_dir).as_posix()
            artifact_hashes[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "gate": self.gate,
            "status": status,
            "verdict": verdict,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_s": round(time.time() - _parse_ts(self.started_at), 3),
            "argv": list(self.argv),
            "config": {"sha256": self.config_sha256, "snapshot": "config.yaml"},
            "environment": environment_snapshot(),
            "model": model_info or {},
            "source_tree": {
                "src_sprm_sha256": sha256_tree(Path(__file__).resolve().parents[1]),
                "files": len(list((Path(__file__).resolve().parents[1]).rglob("*.py"))),
            },
            "artifacts": artifact_hashes,
            "extra": extra or {},
        }
        _write_json_atomic(self.run_dir / "MANIFEST.json", manifest)
        digest = sha256_file(self.run_dir / "MANIFEST.json")
        (self.run_dir / "MANIFEST.sha256").write_text(digest + "\n", encoding="ascii")
        (self.run_dir / ".sealed").write_text(finished_at + "\n", encoding="ascii")
        self._append_index(status=status, verdict=verdict, finished_at=finished_at)
        self._sealed = True
        return self.run_dir

    def _append_index(self, *, status: str, verdict: str, finished_at: str) -> None:
        root = self.run_dir.parent
        lock_path = root / ".index.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            index_path = root / "index.json"
            if index_path.is_file():
                try:
                    entries = json.loads(index_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ArchiveError(f"corrupt index.json: {exc}") from exc
            else:
                entries = []
            entries.append(
                {
                    "run_id": self.run_id,
                    "gate": self.gate,
                    "status": status,
                    "verdict": verdict,
                    "started_at": self.started_at,
                    "finished_at": finished_at,
                    "config_sha256": self.config_sha256,
                    "path": self.run_dir.name,
                }
            )
            _write_json_atomic(index_path, entries)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _assert_open(self) -> None:
        if self._sealed:
            raise ArchiveError(f"run {self.run_id} is already sealed")


def _parse_ts(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def verify_run(run_dir: str | Path) -> list[str]:
    """Return a list of mismatches; empty means the archive is intact."""
    path = Path(run_dir).expanduser().resolve()
    manifest_path = path / "MANIFEST.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    expected_manifest_digest = (path / "MANIFEST.sha256")
    if not expected_manifest_digest.is_file():
        problems.append("missing MANIFEST.sha256")
    else:
        actual = sha256_file(manifest_path)
        recorded = expected_manifest_digest.read_text(encoding="ascii").strip()
        if actual != recorded:
            problems.append(f"manifest digest mismatch: {actual} != {recorded}")
    for rel, meta in manifest.get("artifacts", {}).items():
        file_path = path / rel
        if not file_path.is_file():
            problems.append(f"missing artifact: {rel}")
            continue
        actual = sha256_file(file_path)
        if actual != meta.get("sha256"):
            problems.append(f"artifact digest mismatch: {rel}")
        if file_path.stat().st_size != meta.get("size"):
            problems.append(f"artifact size mismatch: {rel}")
    return problems


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    index_path = Path(root).expanduser().resolve() / "index.json"
    if not index_path.is_file():
        return []
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ArchiveError("index.json must contain a list")
    return entries
