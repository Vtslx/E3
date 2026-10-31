# ADR-005: future-query attention allocation for W2

Status: accepted, implemented.

## Context

The first D1 used a uniform slot budget over evicted blocks. That is the low-spec baseline:
a block containing the needle gets the same number of slots as any filler block, so at 32k
the needle block is compressed ~15 tokens per slot at 512 slots and retrieval collapses.
The design's importance term is future attention over predictability,
`I_j = E_{future ~ q}[attn(future -> j)] / (1 + predictability_j)`. The future-query part was
missing (`allocator.py` said it requires the proposer, not in this round).

## Decision

Use the question itself as the first future query. During the `C0` exact prefill, a forward
hook on each global layer's `q_norm` captures the question-token queries. They are:

1. rotated at their absolute positions (the cached keys are post-RoPE);
2. grouped to KV heads by averaging each GQA query group (16 query heads -> 4 KV heads);
3. compared with every context key at that layer;
4. reduced to a per-block attention mass, averaged over heads and global layers.

The per-block masses are passed to `allocate_slots` in the proportional mode with the
configured floor and `max_slot_multiplier` cap. No extra prefill is needed: the `C0` arm
already runs the exact prefill.

## Evidence (preview, single samples)

At 8k, the needle block received attention mass `0.334` versus `0.02–0.047` for filler
blocks; with 512 slots it received 171 slots (the 4x cap) and `C3` went from 1/3 to 3/3 over
depths 0.1/0.5/0.9. At 32k with 512 slots it still failed 0/3: 512/59 blocks gives ~9 slots
per block and the 4x cap gives the needle block only ~34 slots for 512 tokens. At 32k with
2048 slots the needle block received 137 slots and `C3` went 0/3 -> 3/3.

## Limits

- The question query is a strong but single future-query proxy. The full design would use
  the proposer's future distribution; that is not implemented.
- Block-level allocation only; a needle inside a mixed block can still be diluted.
- The `max_slot_multiplier` cap limits how much a single block can take. At 512 slots/32k
  the budget itself is the binding constraint.
- The future-query probe is computed from the exact `C0` cache; if `C0` is not in the arm
  set, the memory arm falls back to uniform.
