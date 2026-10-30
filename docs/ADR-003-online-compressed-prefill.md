# ADR-003: online compressed prefill, not post-hoc KV slicing

Status: accepted, implemented. This is the most important correctness decision in the
harness.

## Context

The obvious way to evaluate a compressed arm is to prefill the full prompt exactly, build
memory from the evicted region, slice the global cache to `[sink | slots | window]`, and
decode. That protocol is wrong, and the error is large enough to invert the experiment.

Each layer's hidden state depends on the layers below it. During a full prefill, the first
global layer attends to the entire context, including the tokens that will later be
evicted. Its output feeds the conv and sliding layers above it, and their cached window
states are computed from hidden states that already contain the dropped tokens. The
contamination propagates upward: by the last global layer, the "exact window" keys differ
from what the same tokens would produce under the compressed context.

Measured on a 32k passkey prompt, replacing the needle sentence with equal-length filler
while keeping the cache slice identical changed:

- the first global layer's window keys by `0.0` (it is below the first attention layer);
- higher global layers' window keys by up to `4.88`;
- sliding-layer keys by up to `5.98`;
- conv states by up to `0.29`;
- the next-token logits by `6.4`.

The post-hoc "streaming" arm therefore still saw the needle and answered at `0.9995`
confidence, while a true streaming control was at chance. Any D1 comparison under that
protocol would have been meaningless.

## Decision

Prefill each arm family with the context it will actually see, online:

1. Prefill `sink + window` tokens exactly, in chunks.
2. For each subsequent chunk:
   - take the outgoing block from the head of the exact window;
   - unrotate its keys and absorb it into Tier-2 (`write_slots`) if the arm has memory;
   - rebuild the global cache to `[sink | initialized slots | window minus the block]`;
   - forward the chunk under that cache.
3. The prefill's own last-token logits are the first honest next-token distribution. No
   hold-out token or replay is needed.

`block_tokens` must equal `prefill_chunk_tokens` so the evicted block is exactly the chunk
just forwarded. The last block may be shorter; its slot budget is clamped to its token count.

The exact arm (`C0`) uses the native full prefill. The streaming arm (`C1`) runs the same
online loop with no memory. The memory arms (`C3`/`C4`) share one online prefill and clone
the final cache and memory state, so `C4`'s W1 writes cannot affect `C3`.

## Cost

Three prefills per sample instead of one. On M4/MPS at 32k this is roughly 2–2.5 minutes per
sample; the full configured D1 grid is a multi-hour run, which is why reduced smoke runs are
archived separately.

## Consequences

- Every cached window state is conditioned on the compressed context, including conv and
  sliding states, because the forward pass itself produced them.
- Memory is built from exact KV that was computed under the arm's own compressed context.
- The post-hoc `build_memory` path remains only as a `form-a build` diagnostic. It must not
  be used to produce an arm's decode cache.
- Online memory allocation is uniform: importance-based allocation needs a first pass over
  the context and is therefore a post-hoc diagnostic, not part of D1.
