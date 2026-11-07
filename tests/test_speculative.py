from __future__ import annotations

from sprm.runtime.speculative import PromptLookup, ReservoirLookup


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


def test_reservoir_prefers_the_majority_continuation() -> None:
    history = [1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3]
    proposer = ReservoirLookup(history, n=3, gamma=1)
    assert proposer.propose() == [9]
    assert proposer.proposed == 1


def test_reservoir_ties_use_the_most_recent_occurrence() -> None:
    history = [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3]
    proposer = ReservoirLookup(history, n=3, gamma=1)
    assert proposer.propose() == [8]


def test_reservoir_stops_when_a_continuation_runs_out() -> None:
    history = [9, 8, 7, 4, 5, 6, 1, 2, 3]
    proposer = ReservoirLookup(history, n=3, gamma=4)
    assert proposer.propose() == []


def test_reservoir_matches_prompt_lookup_on_repetition() -> None:
    history = [1, 2, 3, 4, 5, 1, 2, 3]
    reservoir = ReservoirLookup(history, n=3, gamma=2)
    prompt = PromptLookup(history, n=3, gamma=2)
    assert reservoir.propose() == prompt.propose() == [4, 5]
