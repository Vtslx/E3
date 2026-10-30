from __future__ import annotations

from collections.abc import Sequence


class CharTokenizer:
    """Deterministic byte-per-character tokenizer for task and metric tests."""

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) + 1 for character in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(int(token_id) - 1) for token_id in token_ids if int(token_id) > 0)
