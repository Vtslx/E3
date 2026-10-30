# ADR-001: store keys unrotated, rotate slots at injection

Status: accepted, implemented.

## Context

The native cache stores **post-RoPE** keys. RoPE depends on the absolute position, and a
memory slot is a mixture of keys from many positions. A mixture of rotations is not a
rotation of the mixture, so writing post-RoPE keys directly into a slot would make the slot
key meaningless for retrieval.

## Decision

1. Before a key enters Tier-2, remove RoPE with the model's own cos/sin tables. The inverse
   of `k * cos + rotate_half(k) * sin` is `k * cos - rotate_half(k) * sin`; LFM3's zero-tail
   partial RoPE keeps `cos^2 + sin^2 = 1` on the rotated prefix and `cos=1, sin=0` on the
   tail, so the inverse is exact up to bf16 rounding.
2. Each slot stores a representative position (the rounded mean position of its assigned
   tokens). At injection, the slot key is rotated at that position.
3. Values are not rotated and are stored as-is.

This is an approximation for a diffuse slot: one position cannot represent a cluster that
spans many positions. For short clusters the phase error is small; for a 512-token block
with a handful of slots, the high-frequency RoPE dimensions decorrelate. This is a known
limitation, not a hidden one.

## The axis bug

The first implementation reshaped the flattened slot keys `[B, H*D]` to `[B, H, D]` and
unsqueezed batch, producing `[1, B, H, D]`: the head and token axes were swapped. The
rotation code then saw `H` as the token axis and failed as soon as `B != H`. Fixed by
transposing to `[1, H, B, D]` before rotation. The same swap existed for slot values; fixed
with the same transpose.

## Injection scale

Memory keys are L2-normalized over the flattened `H*D` vector. The model's `k_norm`
produces per-head RMS 1, i.e. flattened norm `sqrt(H*D)`. Injecting a unit vector unscaled
would give slots per-head norms of `1/sqrt(H)` and make them nearly invisible next to exact
keys.

The code multiplies the unit slot keys by `sqrt(H*D)`. A sharp cluster (mean direction close
to one token) then has native scale; a diffuse cluster (mean of many unit vectors) keeps a
proportionally weaker key. We deliberately do **not** renormalize each slot to a fixed norm:
that would erase the confidence signal and promote noise directions.

## Consequences

- RoPE round-trip error is checked in V0 (`0.03125` in bf16 at the 224-token probe).
- The memory is position-aware at slot granularity, not token granularity.
- A slot's key norm encodes cluster sharpness, which is a useful attention prior.
