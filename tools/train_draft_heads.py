"""Train lightweight Medusa-style draft heads on the frozen LFM3-1.7B.

The target's final hidden state (the input to the tied LM head) is captured for every
position of a natural-text chunk. K low-rank heads learn to predict token offsets 1..K from
that hidden state. The trunk stays frozen; only the heads train. The heads are then used as
a speculative proposer (see runtime/draft_heads.py), so a proposal costs a few low-rank
matrix products instead of a forward pass.

Usage:
    python tools/train_draft_heads.py --corpus /tmp/sprm_draft/wikitext2_train.parquet \
        --out /tmp/sprm_draft/heads.pt --tokens 1000000 --rank 128 --heads 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

MODEL = "/Users/yikewang/Model/LFM3/LFM3-1.7B"


def load_corpus(path: Path, limit_tokens: int, tokenizer) -> list[int]:
    """Pack non-empty text rows into one token stream, truncated to ``limit_tokens``."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        rows = [str(value) for value in table["text"].to_pylist()]
    else:
        rows = path.read_text(encoding="utf-8").splitlines()
    tokens: list[int] = []
    for row in rows:
        text = row.strip()
        if not text:
            continue
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(tokens) >= limit_tokens:
            break
    return tokens[:limit_tokens]


class DraftHeads(nn.Module):
    def __init__(self, hidden: int, vocab: int, rank: int, count: int) -> None:
        super().__init__()
        self.down = nn.ModuleList([nn.Linear(hidden, rank, bias=False) for _ in range(count)])
        self.up = nn.ModuleList([nn.Linear(rank, vocab, bias=False) for _ in range(count)])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return [up(down(hidden)) for down, up in zip(self.down, self.up, strict=True)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1_000_000)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs):
        captured["hidden"] = inputs[0].detach()

    handle = model.lm_head.register_forward_pre_hook(capture)

    tokens = load_corpus(args.corpus, args.tokens, tokenizer)
    if len(tokens) < args.chunk + args.heads + 1:
        raise SystemExit("corpus too short")
    chunks = [
        tokens[start : start + args.chunk + 1]
        for start in range(0, len(tokens) - args.chunk, args.chunk)
    ]
    val_count = max(1, min(64, len(chunks) // 10))
    train_chunks, val_chunks = chunks[:-val_count], chunks[-val_count:]
    print(json.dumps({"tokens": len(tokens), "chunks": len(chunks), "val": val_count, "rank": args.rank}))

    hidden_size = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    heads = DraftHeads(hidden_size, vocab, args.rank, args.heads).to(args.device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.0)
    start = time.perf_counter()
    step = 0
    for chunk in train_chunks:
        if args.steps is not None and step >= args.steps:
            break
        input_ids = torch.tensor([chunk], dtype=torch.long, device=args.device)
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        hidden = captured["hidden"].to(torch.float32)[0]  # [L, H]
        labels = torch.tensor(chunk, dtype=torch.long, device=args.device)
        losses = []
        for offset in range(1, args.heads + 1):
            logits = heads.up[offset - 1](heads.down[offset - 1](hidden[: len(chunk) - offset]))
            target = labels[offset:]
            losses.append(nn.functional.cross_entropy(logits, target))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % 50 == 0:
            elapsed = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "tokens": step * args.chunk,
                        "tok_per_s": round(step * args.chunk / elapsed, 1),
                    }
                ),
                flush=True,
            )

    # Report per-offset top-1 accuracy on the last chunks.
    correct = [0] * args.heads
    total = 0
    with torch.no_grad():
        for chunk in val_chunks:
            input_ids = torch.tensor([chunk], dtype=torch.long, device=args.device)
            model(input_ids=input_ids, use_cache=False)
            hidden = captured["hidden"].to(torch.float32)[0]
            labels = torch.tensor(chunk, dtype=torch.long, device=args.device)
            for offset in range(1, args.heads + 1):
                logits = heads.up[offset - 1](heads.down[offset - 1](hidden[: len(chunk) - offset]))
                correct[offset - 1] += int((logits.argmax(dim=-1) == labels[offset:]).sum().item())
            total += len(chunk) - args.heads
    accuracy = [value / max(total, 1) for value in correct]
    print(json.dumps({"top1": accuracy, "steps": step, "tokens": step * args.chunk}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rank": args.rank,
            "heads": args.heads,
            "hidden": hidden_size,
            "vocab": vocab,
            "down": [down.weight.detach().cpu() for down in heads.down],
            "up": [up.weight.detach().cpu() for up in heads.up],
            "accuracy": accuracy,
        },
        args.out,
    )
    handle.remove()
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
