# ADR-008: cut the current W1 from the A-route frontier

Status: accepted. C4 remains archived as a null result; the frontier uses C3.

## Evidence

- D1 2048: writeback effect `C4 - C3` = 0.0 pp in the far bucket, 0/3 kinds, 1 improved /
  1 degraded, McNemar p=0.75. C3 and C4 both score 50/72 at 32k. C4 is +0.46 pp overall,
  entirely from one 16k sample.
- D1 512: writeback 0/0; C3=C4 at 32k.
- The W1 was tested with the slot refresh and a W1-aware candidate scorer (write the fed
  token into a memory clone, refresh the slot rows, re-feed before scoring the next token).
  The candidate log-probabilities move by ~0.03, but the argmax does not.
- The W1 stability mechanisms cost real memory: at 2048 slots, `memory_bytes` is 117.6 MiB,
  of which ~50 MiB is `M_k/M_v`, ~50 MiB is the frozen anchor refs, and ~25 MiB is the
  bounded replay buffer. The 32k resident compression is 2.28x; without the W1-only
  refs/replay it is ~3.8x, and with a bf16 memory core ~5x.

## Decision

Cut the current W1 from the A-route frontier. Keep C3 (sink + attention-allocated slots +
window); the C4 code path and its archives stay as a null result. The frontier uses C3.

A W1 redo is a separate experiment, not more samples of the current one. The concrete
redesign is: sparse top-k write (only the attention support), a lag-free path (read the
updated M before the next forward, or an associative branch instead of static slot KV),
bf16 memory, and a batch write of the accepted prefix. Those change the mechanism; the
current single-token all-slot delta with gain 0.025 and a one-step lag has no measurable
benefit.

## Consequences

- The frontier's memory overhead can drop the W1-only refs/replay; the resident point moves
  from 2.28x toward ~3.8x without touching C3 accuracy.
- The C3 key matrix is still rank-deficient (`cond=inf`) on repetitive filler; the D5 2048
  pass was C4-specific (anchors). C3 stability is a separate caveat, not solved by cutting W1.
- Do not claim that the A route meets the design's 8x target at 32k. The measured resident
  points are 4.84x at 512 and 2.28x at 2048 for 27.8% and 69.4% far accuracy; the global
  attention/logical projections are ~7.8x/7.4x at 2048, but they are not the current
  resident implementation.
