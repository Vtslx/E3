"""Trained low-rank draft heads as a speculative proposer.

The heads are trained by ``tools/train_draft_heads.py`` on the frozen target's final hidden
state (the tied LM head input). Each head predicts one token offset; a proposal is the
concatenation of the head argmaxes. Inference is a few low-rank matrix products per token,
so the draft is effectively free compared with a 1.7B forward.

The backend captures the last position's hidden state on every forward (``last_hidden``),
so the proposer always drafts from the state after the last committed token.
"""

from __future__ import annotations

from pathlib import Path

import torch

from sprm.backends.lfm3.backend import Lfm3Backend


class DraftHeadProposer:
    def __init__(
        self,
        backend: Lfm3Backend,
        path: str | Path,
        *,
        gamma: int | None = None,
    ) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.down = [tensor.to(backend.info.device) for tensor in state["down"]]
        self.up = [tensor.to(backend.info.device) for tensor in state["up"]]
        self.gamma = min(gamma or len(self.down), len(self.down))
        self.backend = backend
        self.proposed = 0
        if self.gamma < 1:
            raise ValueError("draft head needs at least one head")

    def propose(self) -> list[int]:
        hidden = self.backend.last_hidden
        if hidden is None or hidden.numel() == 0:
            return []
        h = hidden.reshape(1, -1).float()
        tokens: list[int] = []
        with torch.no_grad():
            for index in range(self.gamma):
                low = h @ self.down[index].transpose(0, 1)
                logits = low @ self.up[index].transpose(0, 1)
                tokens.append(int(logits.argmax(dim=-1).item()))
        self.proposed += len(tokens)
        return tokens

    def update(self, tokens: list[int]) -> None:
        # The hidden state is captured by the backend; no history is needed.
        del tokens
