"""Deterministic synthetic passkey retrieval tasks.

The tasks are deliberately narrow: a context of filler records, one or more passkey
sentences, and a question asking for one passkey. Candidates are not shown to the model; the
scorer ranks them by teacher-forced likelihood. Free-form generation was observed to stop
after a prefix of a multi-token code, which would make exact match insensitive to real
retrieval improvements.

Every token sequence is generated from explicit seeds and hashed. Contexts are built at the
token level so ``context_length`` is exact, not approximate.
"""

from __future__ import annotations

import hashlib
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sprm.config.schema import TaskConfig
from sprm.contracts.types import Sample

_FILLER = (
    "The archive room is quiet in the afternoon and the shelves are labelled by year.",
    "A narrow corridor connects the reading hall to the old storage wing.",
    "Visitors must sign the ledger before entering the manuscript section.",
    "The building keeps a steady temperature to protect the paper records.",
    "Every folder is checked twice before it is returned to its cabinet.",
    "The staff rotate between the front desk and the basement stacks.",
    "A small clock above the door ticks through the working day.",
    "Most requests arrive by email and are answered in order.",
    "The catalogue uses a fixed numbering scheme that has not changed in decades.",
    "A service lift at the back carries boxes between the floors.",
    "Reading lamps are switched off at the end of each shift.",
    "The index cards are stored in drawers ordered by subject.",
)

_OWNERS = ("courier", "archivist", "librarian", "curator", "registrar", "steward")
_ATTRIBUTES = ("red", "blue", "north", "south")


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class Needle:
    owner: str
    attribute: str
    code: str

    def sentence(self) -> str:
        if self.attribute:
            return f" The {self.attribute} passkey for {self.owner} is {self.code}."
        return f" The passkey for the {self.owner} is {self.code}."


@dataclass(frozen=True)
class TaskPlan:
    kind: str
    context_length: int
    depth: float
    index: int
    filler_seed: int
    needle_seed: int


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = struct.pack(f"<{len(token_ids)}I", *(int(t) for t in token_ids))
    return hashlib.sha256(payload).hexdigest()


def _unique_codes(rng: random.Random, count: int) -> list[str]:
    """Codes in generation order.

    A set would make the order depend on Python's per-process string hash seed, which would
    change the target code and option order between runs.
    """
    codes: list[str] = []
    seen: set[str] = set()
    while len(codes) < count:
        code = f"{rng.randrange(10000, 100000)}"
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _filler_tokens(tokenizer: TokenizerLike, length: int, rng: random.Random) -> list[int]:
    """Deterministic filler long enough for ``length`` tokens.

    Each sentence carries a monotonically increasing record number. Repetition without a
    varying prefix makes the key matrix rank-deficient and is not representative of text.
    """
    tokens: list[int] = []
    index = 0
    while len(tokens) < length:
        sentence = f" Record {index}: {_FILLER[rng.randrange(len(_FILLER))]}"
        tokens.extend(int(t) for t in tokenizer.encode(" " + sentence, add_special_tokens=False))
        index += 1
    return tokens[:length]


def _insert_blocks(
    context: list[int],
    blocks: Sequence[tuple[int, list[int]]],
    *,
    limit: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Insert token blocks at positions, in descending order so positions stay valid."""
    result = list(context)
    placed: list[tuple[int, int]] = []
    for position, block in sorted(blocks, key=lambda item: item[0], reverse=True):
        start = max(0, min(int(position), max(0, limit - len(block))))
        result[start : start + len(block)] = block
        placed.append((start, start + len(block)))
    result = result[:limit]
    placed.sort()
    return result, placed


def _question(kind: str, needle: Needle) -> str:
    if kind == "conditional":
        stem = f"What is the {needle.attribute} passkey for the {needle.owner}?"
    else:
        stem = f"What is the passkey for the {needle.owner}?"
    # Candidates are never shown to the model. The scorer ranks them by teacher-forced
    # likelihood, which removes option-order and letter-tokenization bias.
    return f"\nQuestion: {stem}\nAnswer: "


def _make_sample(
    *,
    tokenizer: TokenizerLike,
    plan: TaskPlan,
) -> Sample:
    rng = random.Random(_stable_seed("needle", plan.needle_seed, plan.kind, plan.context_length, plan.depth, plan.index))
    codes = _unique_codes(rng, 8)
    owners = rng.sample(_OWNERS, k=3)
    attribute = _ATTRIBUTES[rng.randrange(len(_ATTRIBUTES))]
    owner = owners[0]
    target = Needle(owner=owner, attribute=attribute if plan.kind == "conditional" else "", code=codes[0])

    context = _filler_tokens(tokenizer, plan.context_length, random.Random(_stable_seed("filler", plan.filler_seed, plan.index)))
    target_position = round(plan.depth * plan.context_length)
    target_position = max(0, min(target_position, plan.context_length - 1))
    blocks: list[tuple[int, list[int]]] = []
    sentences: list[str] = []

    if plan.kind == "single_needle":
        sentences.append(target.sentence())
        blocks.append((target_position, tokenizer.encode(target.sentence(), add_special_tokens=False)))
    elif plan.kind == "multi_needle":
        # Three records with distinct owners. Only the target is asked about. Records are
        # placed before the target when it sits near the end, so none are clipped.
        spacing = max(64, plan.context_length // 8)
        records = [target]
        for offset, other_owner in enumerate(owners[1:], start=1):
            records.append(Needle(owner=other_owner, attribute="", code=codes[offset]))
        for offset, record in enumerate(records):
            position = target_position - offset * spacing
            if position < 0:
                position = target_position + offset * spacing
            blocks.append((position, tokenizer.encode(record.sentence(), add_special_tokens=False)))
            sentences.append(record.sentence())
    elif plan.kind == "conditional":
        # Two attributes for the same owner at one region; the question names the attribute.
        other_attribute = next(a for a in _ATTRIBUTES if a != attribute)
        other = Needle(owner=owner, attribute=other_attribute, code=codes[1])
        first = tokenizer.encode(target.sentence(), add_special_tokens=False)
        second = tokenizer.encode(other.sentence(), add_special_tokens=False)
        blocks.append((target_position, first + second))
        sentences.extend([target.sentence(), other.sentence()])
    else:
        raise ValueError(f"unknown task kind {plan.kind!r}")

    context, placed = _insert_blocks(context, blocks, limit=plan.context_length)
    if len(context) != plan.context_length:
        raise RuntimeError(f"context length {len(context)} != requested {plan.context_length}")

    # The answer options are correct + three unique foils. The correct position is shuffled.
    foil_pool = codes[1:5] if plan.kind == "conditional" else codes[2:6]
    options = [target.code, *foil_pool[:3]]
    rng.shuffle(options)
    correct_index = options.index(target.code)
    question = _question(plan.kind, target)
    question_ids = [int(t) for t in tokenizer.encode(question, add_special_tokens=False)]

    if target_position >= plan.context_length:
        raise RuntimeError("needle position outside the context")
    context_text = tokenizer.decode(context, skip_special_tokens=True)
    if target.code not in context_text:
        raise RuntimeError("needle code was truncated out of the context")
    if not any(context[start:end] for start, end in placed):
        raise RuntimeError("needle block was not placed")

    metadata: dict[str, Any] = {
        "owner": owner,
        "attribute": target.attribute,
        "code": target.code,
        "options": list(options),
        "correct_index": correct_index,
        "needle_positions": [[int(a), int(b)] for a, b in placed],
        "needle_position": int(target_position),
        "sentences": sentences,
        "filler_seed": plan.filler_seed,
        "needle_seed": plan.needle_seed,
        "task_index": plan.index,
        "context_token_sha256": _token_hash(context),
        "distance_from_end": int(plan.context_length - target_position),
    }
    sample_id = (
        f"{plan.kind}_L{plan.context_length}_d{str(plan.depth).replace('.', 'p')}"
        f"_i{plan.index}_{metadata['context_token_sha256'][:8]}"
    )
    return Sample(
        sample_id=sample_id,
        task_kind=plan.kind,
        context_length=plan.context_length,
        depth=plan.depth,
        token_ids=tuple(context),
        question_token_ids=tuple(question_ids),
        gold_answer=target.code,
        gold_token_ids=tuple(int(t) for t in tokenizer.encode(target.code, add_special_tokens=False)),
        metadata=metadata,
    )


def build_samples(
    tokenizer: TokenizerLike,
    task: TaskConfig,
    *,
    kinds: Sequence[str] | None = None,
    context_lengths: Sequence[int] | None = None,
    depths: Sequence[float] | None = None,
    samples_per_cell: int | None = None,
    max_samples: int | None = None,
) -> list[Sample]:
    """Build the deterministic sample grid in a stable order."""
    selected_kinds = tuple(kinds) if kinds is not None else task.kinds
    selected_lengths = tuple(context_lengths) if context_lengths is not None else task.context_lengths
    selected_depths = tuple(depths) if depths is not None else task.depths
    per_cell = task.samples_per_cell if samples_per_cell is None else int(samples_per_cell)
    if per_cell < 1:
        raise ValueError("samples_per_cell must be positive")
    for kind in selected_kinds:
        if kind not in {"single_needle", "multi_needle", "conditional"}:
            raise ValueError(f"unknown task kind {kind!r}")
    samples: list[Sample] = []
    for kind in selected_kinds:
        for length in selected_lengths:
            if length < 2:
                raise ValueError(f"context length must be >= 2, got {length}")
            for depth in selected_depths:
                if not 0.0 < depth < 1.0:
                    raise ValueError(f"depth must lie strictly between 0 and 1, got {depth}")
                for index in range(per_cell):
                    plan = TaskPlan(
                        kind=kind,
                        context_length=int(length),
                        depth=float(depth),
                        index=index,
                        filler_seed=task.filler_seed,
                        needle_seed=task.needle_seed,
                    )
                    samples.append(_make_sample(tokenizer=tokenizer, plan=plan))
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
    return samples


def sample_prompt_token_ids(sample: Sample) -> list[int]:
    """The exact model input: context tokens followed by question tokens."""
    return list(sample.token_ids) + list(sample.question_token_ids)


def describe_sample(sample: Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "task_kind": sample.task_kind,
        "context_length": sample.context_length,
        "depth": sample.depth,
        "gold_answer": sample.gold_answer,
        "prompt_tokens": len(sample.full_token_ids),
        **{k: sample.metadata.get(k) for k in ("owner", "attribute", "code", "options", "correct_index")},
    }
