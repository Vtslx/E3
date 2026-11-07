# ADR-007: prompt-lookup speculative decoding with an all-or-nothing commit

Status: accepted, implemented.

## Context

The design's speed lever is speculative decoding: propose tokens, verify them with one
target forward, commit the accepted prefix. The memory compression itself reduces KV bytes,
not latency; at batch=1 the decode is dominated by weight reads. The A route needs a speed
half to test the original dual objective.

## Decision

Implement a training-free prompt-lookup draft and an all-or-nothing commit.

- **Draft.** Find the last earlier occurrence of the final `n` tokens in the prompt +
  generated history; propose the following `gamma` tokens. No match means a normal greedy
  step.
- **Verify.** One `decode_many` forward over the whole proposal; the target's greedy tokens
  at each position are compared with the proposal.
- **Commit.** If every proposed token matches, the proposal stays in the cache plus one bonus
  token. If any token mismatches, the whole batch is rejected: global and sliding KV are
  cropped, the short-conv state is restored from a pre-forward snapshot, and the normal greedy
  token is decoded.

All-or-nothing is deliberate. The conv layers update from a whole chunk at once; a
partial-prefix rollback would need the conv state after `k < gamma` tokens, which a single
parallel forward does not expose. Rejecting the whole batch keeps the cache exact and the
output token-identical to greedy decoding.

## Evidence (S1)

On a repeated-sentence 2048-token prompt, 32 decode tokens, `gamma=4`, `n=3`:

- greedy 20.62 tok/s; speculative 38.15 tok/s; **1.85x**; output token-identical;
- proposed 26, accepted 26, 7 full accepts, 0 fallbacks;
- a second run measured 2.34x; `gamma=13` also ~2.2x; `gamma=8` was ~1.0x because partial
  proposals caused fallbacks.

S1 is archived PASS. It is a mechanism test on a repetitive prompt, not a general-task
acceptance estimate.

## Limits

- Prompt-lookup acceptance on natural, non-repetitive text is unknown; no trained draft head.
- The speed run uses `C0_full` exact KV. It does not claim memory-arm speedup, batch>1
  throughput, or that compression accelerates generation.
- All-or-nothing discards partially correct proposals; a prefix commit with conv-state
  reconstruction would be faster but is not implemented.
- The live memory arm's W1 is processed after the accepted batch; its one-step lag is
  extended to `gamma` for the tokens inside the batch.

## Update (2026-10-31): partial commit, reservoir draft, and acceptance accounting

- `proposed_tokens` now counts **every** proposed token, including proposals that are
  rejected. The original S1 run reported 26/26 because it counted only full accepts; that
  number is a full-accept count, not a token acceptance rate.
- `partial_accept=True` commits the matching prefix instead of rejecting the whole proposal.
  The unaccepted rows are cropped, the pre-verify conv snapshot is restored, and the accepted
  prefix is re-fed as one chunk so the conv state after `k` tokens is rebuilt exactly. The
  next token is the target's argmax at the last accepted position, so the output stays the
  greedy sequence. This is the conv-state reconstruction the original ADR said was missing.
- `ReservoirLookup` generalizes the draft: collect up to 64 earlier occurrences of the suffix
  and propose the majority continuation at each step. It coincides with prompt lookup on
  repetitive text and is measured on natural text in S2.
- S2 (natural text, C3 arm) is the operating-point gate; see ADR-010.
