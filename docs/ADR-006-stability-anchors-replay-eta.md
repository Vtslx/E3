# ADR-006: anchors, bounded replay, and eta adaptation

Status: accepted, implemented.

## Context

D5 at 512 slots showed `cond(M_k) = inf` and window reconstruction shrinkage below 1. The
condition was rank deficiency on repetitive filler; the reconstruction error grew over a
192-token decode. The design's stability section asks for a trust region, anchors, replay,
and gain adaptation. The trust region and condition monitor existed; the other three did
not.

## Decision

**Anchors.** The memory keeps `M_k_ref` / `M_v_ref`. The reference is set on the first W1
write and refreshed every `anchor_interval` writes; between refreshes the memory is blended
toward it by `anchor_blend`. Anchors are applied only in `write` (decode-time W1), not during
the block prefill, so the reference is not an arbitrary early block.

**Bounded replay.** `write_slots` keeps the last `replay_blocks` exact blocks (unrotated
keys, values, positions) in a CPU buffer. Every `replay_interval` W1 writes, those blocks
are re-written with a half-gain delta. This is a **delta replay**, not the design's batch
least-squares refit; the refit is not implemented. The buffer bytes are included in
`memory_bytes()`.

**Eta adaptation.** In `decode_arm`, the W1 gain starts at `write_gain` and, every
`eta_adapt_interval` steps, is scaled by `exp(eta_adapt_rate * relative_error_change)` and
clamped to `[eta_min, eta_max]`. The final eta and ratio are recorded.

## Evidence

- 512 slots: condition remained `inf` and the generated-token shrinkage failed; D5 512 is
  archived FAIL.
- 2048 slots: condition dropped from `inf` to **80** with anchors; the generated-token
  reconstruction shrinkage passed; D5 2048 is archived PASS.
- The full-window shrinkage (0.783 at 2048) is a **diagnostic** and still fails: most of that
  window is prefill KV that W1 never wrote. It measures out-of-sample generalization, not the
  writeback.
- Eta barely moved (final 0.04996 vs base 0.05); the error changes per interval were small.
  The adaptation is implemented but not yet a meaningful lever.

## Limits

- Replay is a delta re-write, not a least-squares refit.
- The anchor reference is a frozen snapshot, not a learned prior.
- Full-window reconstruction is not a fair W1 shrinkage test in this runtime; the generated
  window is.
- D5 still does not measure long-text task accuracy or distribution equivalence.
