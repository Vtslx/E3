# ADR-009: C3 key regularization and the G0 stability result

Status: accepted. Key regularization is in; G0 fails on memory shrinkage; C3 is an
accuracy-only memory, not a stability-claim memory.

## Context

The 2048 D1 passed the memory effect (+43.06 pp far), but C3 `condition` was `inf` and the
D5 2048 pass belonged to C4's W1 anchors. Phase 0 asked for memory-only stability: bound the
condition, then evaluate C3's own shrinkage.

## Decision

1. **Key regularization** after every memory build: SVD of `M_k`, clamp singular values to
   `max(smax * key_singular_floor, eps)` with `key_singular_floor=2e-6`, reconstruct, and
   renormalize rows. This lifts only the null directions; the dominant retrieval directions
   are preserved. It is memory-only and does not depend on W1.
2. **Stability buffers off by default** (`stability_buffers=false`). The anchor refs and
   replay buffer are W1-only; C3 never used them. Keeping them allocated inflated the 2048
   memory by ~67 MiB.

## Evidence

C3 D5, 8k sample, 192 decode tokens:

| slots | condition | memory shrinkage | generated diag | result |
|-------|-----------|------------------|----------------|--------|
| 1024 | 501,935 | 0.958 | 1.002 | FAIL on shrinkage |
| 2048 | 501,691 | 0.956 | 1.098 | FAIL on shrinkage |

The condition is fixed (`inf` -> ~5e5 < 1e6): the key regularization works. The full-window
memory reconstruction still grows ~4.2-4.4% over 192 tokens; the static C3 memory has no
adaptation during decode. Removing the W1 buffers leaves the C3 D5 numbers exactly unchanged
(the buffers were dead weight), while dropping the resident memory by ~67 MiB at 2048.

## Consequence

G0 fails on shrinkage. Per the plan's fallback, **A is accuracy-only**: do not claim memory
stability for C3. The conditions that pass are the condition bound and the memory effect;
the missing piece is a memory-only adapt mechanism (e.g., a periodic write from the exact
Tier-1 window, not W1 generated tokens). That is a separate experiment; it is not the
current W1 and not a reason to add D1 samples.

Phase 1 step 1 (buffers off) is measured by the reduced 2048 D1; the expected resident is
~3.8x with the same far accuracy. Steps 2/3 (bf16 memory core, direct associative read) are
the remaining compression engineering.
