"""Typed error taxonomy.

Every failure that can surface to an operator is one of these types. The CLI maps them to
exit codes; the verification archive records the type and message in RESULT.json.
"""

from __future__ import annotations


class SprmError(Exception):
    """Base class for all SPRM errors."""

    exit_code = 2


class ConfigError(SprmError):
    """The configuration is missing, malformed, or internally inconsistent."""

    exit_code = 3


class BackendError(SprmError):
    """The model backend could not be loaded or does not satisfy the contract."""

    exit_code = 4


class CacheInvariantError(SprmError):
    """A KV-cache invariant was violated (lengths, positions, or layer kinds)."""

    exit_code = 5


class MemoryInvariantError(SprmError):
    """An associative-memory invariant was violated (shape, normalization, trust region)."""

    exit_code = 6


class VerificationError(SprmError):
    """A verification gate could not be executed to a verdict."""

    exit_code = 7


class GateFailure(SprmError):
    """A verification gate executed and failed its acceptance criteria."""

    exit_code = 8


class ArchiveError(SprmError):
    """The verification archive is inconsistent or a run would be overwritten."""

    exit_code = 9
