"""Structured JSON logging.

One JSON object per line. The archive copies stdout.log verbatim and also writes the same
events to events.jsonl, so a run can be inspected without parsing prose.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class JsonLogger:
    """Minimal structured logger with optional file sinks."""

    component: str
    _sinks: list[TextIO] = field(default_factory=list)
    _echo: bool = True

    def add_sink(self, handle: TextIO) -> None:
        self._sinks.append(handle)

    def close(self) -> None:
        for handle in self._sinks:
            handle.flush()
            handle.close()
        self._sinks.clear()

    def log(self, event: str, level: str = "INFO", **fields: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level,
            "component": self.component,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if self._echo:
            print(line, file=sys.stderr, flush=True)
        for handle in self._sinks:
            handle.write(line + "\n")
            handle.flush()

    def info(self, event: str, **fields: Any) -> None:
        self.log(event, level="INFO", **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self.log(event, level="WARNING", **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.log(event, level="ERROR", **fields)


def log_path(summary: Any) -> dict[str, Any]:
    """Serialize a dataclass or mapping for manifest inclusion."""
    if hasattr(summary, "__dataclass_fields__"):
        return asdict(summary)
    if isinstance(summary, dict):
        return summary
    raise TypeError(f"cannot serialize {type(summary)!r}")


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
