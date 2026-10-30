# ADR-002: normalized keys, read temperature, and a trust-region floor

Status: accepted, implemented.

## Context

The design's read is `a = softmax(q M_k^T / sqrt(d_m)); v_hat = a M_v`. With unit-normalized
keys of dimension 512, dot products concentrate near zero (standard deviation about
`1/sqrt(512)`), so dividing by `sqrt(d_m)` produces a near-uniform softmax over 512 slots.
That memory cannot retrieve anything: every query reads the same average.

## Decision

1. L2-normalize keys before writes, reads, and delta updates. Use cosine k-means for the W2
   initialization so cluster centers are comparable to query keys.
2. Use an explicit read temperature `tau` (default 8.0):
   `a = softmax(tau * q M_k^T)`. For a query matching its own cluster center, the dot
   product is high and the softmax is sharp; for random directions it stays diffuse.
3. The delta update moves slots toward the residual with gain `eta * g`; keys are
   re-normalized after each update.
4. The trust region has a **norm floor**: `cap = eps * max(||M_v||_F, floor)`. Without the
   floor, the first write to a zero-initialized `M_v` is scaled to zero.
5. Forgetting decays only `M_v`. Unit-normalized keys are invariant to a global scalar, so
   decaying `M_k` would be a no-op after renormalization: the horizon is a value horizon.

## Consequences

- The memory can retrieve on synthetic clustered data: V1 measures reconstruction error
  falling from 1.0 (zero values) to 0.010 after one 32-token block.
- Condition numbers can be `inf` on real filler: many slots encode nearly the same
  direction, so the key matrix is rank-deficient. This is a capacity observation. It is
  reported, not hidden, and D5 (not run this round) is where it becomes a gate.
- `read_temperature` is a real hyperparameter: too low is uniform, too high overfits one
  slot. It is set in the experiment config and recorded in the run.
