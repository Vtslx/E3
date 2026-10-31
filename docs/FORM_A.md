# Form A: associative memory runtime

Form A is the per-layer associative memory from DESIGN.md §4.2, adapted to LFM3's tri-hybrid
stack. Only the six global-attention layers are modified; the 13 short-conv layers and five
sliding-window layers keep their native caches.

## Components

- **Tier-1 exact memory.** The first `sink_tokens` (128) and the last `window_tokens` (2048)
  global-layer K/V, at full precision. Sliding-window layers additionally keep their native
  255-token window; conv layers keep their 7-token state.
- **Tier-2 associative memory.** One `B x d` key matrix and `B x d` value matrix per global
  layer, `B=512`, `d = num_kv_heads * head_dim = 512`. Read:
  `a = softmax(tau * k M_k^T); v_hat = a M_v`, with `tau=8`, unit-normalized keys.
- **W1 writeback.** After each generated token, its exact K/V is unrotated and written into
  the whole memory with gain `eta * g` (0.05 * 0.5 = 0.025). The trust region caps the delta.
  The injected slot rows in the cache are then refreshed from `M_k`/`M_v`
  (`refresh_memory_slots`); without that refresh the write is invisible to the next forward.
  The live decode has a one-step lag: the logits for `t+1` are computed before the write of
  `t`. Candidate scoring removes the lag by re-feeding each token after the refresh.
- **Anchors / replay / eta.** The first W1 write establishes `M_ref`; it is refreshed every
  `anchor_interval` writes and the memory is blended toward it between refreshes. The last
  `replay_blocks` absorbed blocks are kept and re-written as a half-gain delta every
  `replay_interval` W1 writes. The W1 gain adapts every `eta_adapt_interval` steps from the
  window error trend, clamped to `[eta_min, eta_max]` (ADR-006).
- **W2 absorption.** When a block leaves the exact window, its keys are unrotated, clustered
  with cosine k-means, and written into that block's dedicated slot range. Two delta
  refinement passes follow.
- **Free validation set.** The exact window is known, so `recon_error` measures how well the
  memory reconstructs it. The value is logged per decode step and summarized in `RESULT.json`.

## Online prefill

The runtime never prefills the full context and then slices the cache. It processes chunks
in order and, once the prefix exceeds `sink + window`, before forwarding each chunk it:

1. takes the outgoing block from the head of the window;
2. absorbs it into Tier-2 (unless the arm has no memory);
3. rebuilds the global cache to `[sink | initialized slots | window minus the block]`;
4. forwards the chunk.

`block_tokens == prefill_chunk_tokens == 512`; the last block may be shorter. The prefill's
last-token logits are the arm's first next-token distribution. See ADR-003 for why post-hoc
slicing is invalid and the measured contamination it produces.

Slot allocation uses **future-query attention** online. During the `C0` exact prefill, the
`q_norm` hook captures the question-token queries; after RoPE and GQA head grouping, their
attention mass to each evicted block is the importance. The budget is allocated
proportionally with a floor and a `max_slot_multiplier` cap. `uniform` remains available for
the post-hoc diagnostic. The 512/2048 runs compare budgets (ADR-005).

## Arms

| arm | cache | memory | W1 |
|-----|-------|--------|----|
| `C0_full` | full exact | no | no |
| `C1_stream` | sink + window | no | no |
| `C3_memory` | sink + slots + window | W2 | no |
| `C4_memory_w1` | sink + slots + window | W2 + W1 | yes |

`C0` is the ceiling. `C1` is the StreamingLLM baseline. `C3` isolates W2. `C4` adds W1.
`C3` and `C4` share one online prefill and clone the cache and memory before decoding.

## Diagnostics

`ArmResult.effective_length` and `cache_bytes` are measured **after** decoding, so they
include the generated token KV that was fed back into the cache. The prefill-only lengths are
`sink + window` for `C1` and `sink + slots + window` for `C3`/`C4`; V2 checks those directly.
Likewise, `recon_error_final` is the last out-of-sample window reconstruction after the final
generated token, not the prefill value.

## Invariants

- All global layers have the same effective length.
- Sink, slots, window, and generated tokens are in chronological order in the cache; the
  causal mask is correct without custom masks.
- Rebuilding never touches conv or sliding state.
- Keys are stored unrotated; slot keys are rotated at their representative positions with a
  fixed native-scale factor (ADR-001).
- Values are fp32 in memory, cast to the model dtype at injection.
- The trust region never scales a zero state to zero (norm floor).
- `condition()` and `recon_error()` are diagnostics; neither is used to fabricate a quality
  claim.

## Commands

```bash
# Build memory for one sample and print the block plan, slot allocation, and window error.
sprm form-a build --config configs/experiments/d1_niah.yaml --context-length 8192 --depth 0.1

# The full retrieval gate (online prefill, all arms).
sprm verify d1 --config configs/experiments/d1_niah.yaml

# Reduced smoke (recorded as reduced in the manifest).
sprm verify d1 --config configs/experiments/d1_niah.yaml \
  --samples-per-cell 1 --lengths 8192,32768 --depths 0.1,0.5,0.9
```

## D5 stability

The archived D5 run decodes 192 tokens with `C4_memory_w1` on one 8k sample. It checks the
key-matrix condition (`<= 1e6`) and the window-reconstruction shrinkage (`head/tail >= 1.0`).
The run fails both: the condition is `inf` because the key matrix is rank-deficient on
repetitive filler, and the reconstruction error grew about 4.6% over the decode (shrinkage
0.956). Anchors, replay, and adaptive eta are not implemented; D5 tests only the trust region
and the W1 slot refresh.

## S1 speed half

`decode_speculative` uses a prompt-lookup draft (last `n`-token suffix match) and an
all-or-nothing commit: one target forward verifies the whole proposal; a full match keeps it
plus a bonus token, any mismatch crops global/sliding KV and restores the short-conv state
before the normal greedy token. The output is token-identical to greedy. On a repeated
sentence, S1 measured 1.85–2.34x tokens/s with 26/26 draft acceptance. This is a mechanism
test, not a general-task acceptance estimate, and it does not claim memory-arm speedup.

## Known limitations

- The memory is trained only by closed-form writes; the model was not trained to read it.
  Retrieval is partial and capacity-sensitive. At 32k/512 slots it raises the gold candidate
  probability but rarely wins the argmax.
- Tier-3 residual codes and the proposer are not implemented.
- The condition number of the key matrix is often `inf` on repetitive filler because many
  slots encode the same direction. This is reported, not hidden.
- No throughput or latency claim is made. The harness measures memory quality and cache
  correctness.
