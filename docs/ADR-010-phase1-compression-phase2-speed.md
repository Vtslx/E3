# ADR-010: Phase 1 compression engineering and the Phase 2 speed gate

Status: accepted. Step 1 (buffers off) and step 2 (bf16 memory) are in; the direct-read step
is deferred. Phase 2 fails its targets on natural text: the prompt-lookup draft needs a
trained head.

## Context

Phase 0 left A accuracy-only (ADR-009): the condition is bounded, the memory effect is
large at 1024/2048, but the static C3 memory does not shrink its reconstruction error over a
long decode. ADR-008 cut the current W1, leaving C3 as the frontier arm. The measured
resident compression at 2048 was 2.28x: the memory core was fp32 and the W1-only anchor
references and replay buffer were still allocated.

Phase 1 asked for 2048 far >= 65% at resident >= 3.8x and 1024 far >= 35% at >= 6x, then a
natural-text / C3 speculative measurement (Phase 2 targets: acceptance >= 0.5, net speedup
>= 1.3x).

## Decisions

1. **Stability buffers off by default.** `stability_buffers=false`. The anchors and replay
   buffer were used only by W1; C3 never touched them. The C3 D5 numbers are unchanged
   (same condition and shrinkage) while the 2048 memory drops by ~67 MiB.
2. **Optional bf16 memory storage.** `memory_dtype=bfloat16` stores `M_k`/`M_v` (and the
   reference buffers, when enabled) in bf16; reads, writes, the SVD key regularization, and
   the condition diagnostic upcast to fp32 and round on store. The core bytes halve.
3. **S2 speed gate.** A natural-text prompt (the repository's design/ADR corpus, 4096
   tokens), a selectable arm, a greedy baseline with the per-step reconstruction diagnostics
   disabled, and two drafts: prompt lookup and the frequency-ranked reservoir. `--partial`
   commits a matching prefix (crop the unaccepted rows, restore the pre-verify conv snapshot,
   re-feed the accepted prefix in one chunk, then emit the target's bonus token).
4. **Acceptance accounting.** `proposed_tokens` counts every proposed token, including
   proposals that are rejected. S1's original 26/26 counted only fully accepted proposals.

## Evidence

### Phase 1 screens (27 samples, 1 per cell, 8 decode tokens; far bucket has 9 samples)

| slots | memory dtype | far C1 | far C3 | resident | C3 memory |
|-------|--------------|--------|--------|----------|-----------|
| 2048 | fp32 | 3/9 (33.3%) | 8/9 (88.9%) | **3.85x** | 48.2 MiB |
| 1024 | fp32 | 3/9 (33.3%) | 5/9 (55.6%) | **6.00x** | 24.1 MiB |
| 2048 | bf16 | 3/9 (33.3%) | 8/9 (88.9%) | **5.05x** | 24.2 MiB |

Runs: `20261031T125403Z_d1_..._slots2048_reduced_64d20d45`,
`20261031T133823Z_d1_..._slots1024_reduced_64d20d45`,
`20261031T144448Z_d1_..._slots2048_reduced_3efc1c20`.

The bf16 screen matches the fp32 far accuracy exactly (8/9) and halves the 32k memory
(2,450 vs 3,219 B/token at 32k).

**Full 2048 confirmation** (216 samples, 16 decode tokens, bf16 core): far C3 **51/72
(70.83%)** vs C1 19/72 (26.39%), +44.44 pp, 3/3 kinds, McNemar p=2.2e-7; resident **5.05x**
(C3 2,452 B/token vs C0 12,387), memory 24.2 MiB, condition max 506,933. The writeback
checks still fail (C4 = C3), as expected after ADR-008. This is the authoritative Phase 1
point.

### Phase 2 (natural text, 4096 prompt tokens, 32 decode tokens)

| arm | proposer | partial | acceptance | net tokens/s | output vs greedy |
|-----|----------|---------|------------|--------------|------------------|
| C0_full | prompt | no | 0.387 | 1.128x | match |
| C0_full | reservoir | no | 0.387 | 1.063x | match |
| C3 (512) | prompt | no | 0.082 | 0.770x | diverges at 26 |
| C3 (2048) | prompt | no | 0.030 | 0.778x | match |
| C3 (2048) | prompt | yes | **0.294** | 1.060x | match |
| C3 (2048) | reservoir | no | 0.030 | 0.782x | match |

The C3 continuation rarely follows the literal prompt: at 512 slots acceptance is 8.2%, at
2048 it is 3.0%. Prefix commit raises the 2048 acceptance to 29.4% and turns the 0.78x into
1.06x, but the conv-state refeed costs a forward, so the net speed stays below the 1.3x
target. The reservoir proposer is identical to prompt lookup on this corpus: the matched
suffixes have a single continuation, so the majority vote has nothing to choose.

### A numeric caveat on bitwise identity

S1 claimed output token-identical to greedy on a repeated-sentence prompt. On natural text
that identity can fail after a full accept: the chunked verify forward and the single-token
forward round differently in bf16, so the KV for the accepted tokens differs by ~0.05-0.1
and a later argmax can flip (observed once, at generated token 26). The committed tokens are
still the target's own verifier argmaxes; they are not guaranteed to be bit-identical to a
separate sequential greedy run. S2 reports the first divergence position. The rollback after
a rejection was checked in isolation: after crop + conv restore, the cache keys and the next
logits are bit-identical to the pre-verify state, so the drift comes from the accepted chunk,
not from the rollback.

## Consequences

- **Phase 1 is met.** The full 2048 bf16 run confirms **70.83% far at 5.05x resident**
  (+44.4 pp over C1, 3/3 kinds, p=2.2e-7); the fp32 buffers-off screen gives 3.85x and the
  1024 screen 6.00x. bf16 keeps the full-grid far accuracy and halves the core.
- **Direct associative read is deferred.** Removing the duplicated slot KV (or the core)
  would take 2048 to ~7.36x, but it needs either an attention branch that reads `M_k/M_v`
  directly or a reconstruction of the memory from the slot rows, and the Phase 1 targets are
  already met. It is the next compression step, not a blocker.
- **Phase 2 fails on natural text.** Prompt lookup is a repetition detector: 39% acceptance
  on exact KV, 3% on the 2048-slot memory, net 1.13x/0.78x. Partial commit helps acceptance
  but not net speed. A trained draft head (the Phase 3 MTP/N head) is the path to a natural
  text speed half; a stronger n-gram proposer is not.
- The W1 stays cut (ADR-008). Its anchor/replay buffers stay off (ADR-009).


## Follow-up experiments (2026-11-01, negative)

### Trained low-rank draft heads

Four Medusa-style low-rank heads (rank 128) were trained on 967k WikiText-2 tokens from the
frozen target's final hidden state (the tied LM-head input). Held-out top-1 per offset:
**0.271 / 0.126 / 0.063 / 0.059**. On the S2 docs prompt the draft accepted **0/122**
proposed tokens and netted **0.369x** (`20261031T211305Z_s2_d1_niah_lfm3_c0_draft_3efc1c20`).

The arithmetic of the 2x target: with all-or-nothing commit and gamma=4, a loop costs
2 forwards (verify + bonus) and yields 1+4p tokens, so 2x needs p >= 0.75, i.e. ~0.93
per-token acceptance. With prefix commit the refeed makes a loop cost 3 forwards, so gamma
must be >=6 and the accepted prefix ~5 tokens (~0.8 per token). The prototype is far below
either bar. A natural-text 2x needs an in-domain distilled draft with tree verification,
i.e. Phase 3; prompt lookup and low-rank heads cannot reach it.

### Capacity

4096 slots at 2048 (`20261031T211343Z_d1_..._slots4096_reduced_3efc1c20`): far **8/9**
(unchanged), resident **3.10x** (down from 5.05x). Capacity is not the binding constraint at
this sample size; the doubled memory and slot KV cost 1.95x residency for no measured far
gain.

### Question-query read token

Injecting one question-conditioned read row per global layer (key = GQA-averaged question
query, value = `a M_v`) leaves the reduced far at **8/9** (conditional 2/3), resident 5.05x
(`20261031T215913Z_d1_..._slots2048_reduced_57c7da0c`). Without a trained projection the row
does not add usable binding information; the conditional bucket is the remaining gap and
needs a trained read/write, not a single untrained row.

### Status against the speed/memory targets

- Natural-text speed >= 2x: **not met**. Best measured is 1.128x on exact KV and 1.060x on
  C3 with prefix commit.
- Memory enhancement: no far gain from capacity or the read token. The delivered memory
  result is the compression point (70.83% far at 5.05x resident) and the +44.4 pp memory
  effect; the stability claim still fails (D5 shrinkage 0.974 < 1).
