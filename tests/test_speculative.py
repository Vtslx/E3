from __future__ import annotations

from sprm.runtime.speculative import PromptLookup


def test_prompt_lookup_finds_the_last_match() -> None:
    history = [1, 2, 3, 4, 5, 1, 2, 3]
    proposer = PromptLookup(history, n=3, gamma=2)
    assert proposer.propose() == [4, 5]


def test_prompt_lookup_returns_empty_without_match() -> None:
    proposer = PromptLookup([1, 2, 3, 4], n=3, gamma=2)
    assert proposer.propose() == []


def test_prompt_lookup_updates_history() -> None:
    proposer = PromptLookup([1, 2, 3, 4, 1, 2], n=2, gamma=1)
    assert proposer.propose() == [3]
    proposer.update([7])
    assert proposer.history[-1] == 7
