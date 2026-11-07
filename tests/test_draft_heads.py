from __future__ import annotations

from pathlib import Path

import torch

from sprm.runtime.draft_heads import DraftHeadProposer


class _Info:
    device = "cpu"


class _Backend:
    info = _Info()
    last_hidden: torch.Tensor | None = torch.tensor([[1.0, 0.0]])


def test_draft_heads_propose_from_the_captured_hidden_state(tmp_path: Path) -> None:
    down = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    up = torch.zeros(5, 2)
    up[3, 0] = 1.0
    up[1, 1] = 1.0
    up_second = torch.zeros(5, 2)
    up_second[1, 0] = 1.0
    path = tmp_path / "heads.pt"
    torch.save(
        {
            "rank": 2,
            "heads": 2,
            "hidden": 2,
            "vocab": 5,
            "down": [down, down],
            "up": [up, up_second],
            "accuracy": [0.0, 0.0],
        },
        path,
    )
    proposer = DraftHeadProposer(_Backend(), path)
    assert proposer.gamma == 2
    assert proposer.propose() == [3, 1]
    assert proposer.proposed == 2
    _Backend.last_hidden = None
    assert proposer.propose() == []
