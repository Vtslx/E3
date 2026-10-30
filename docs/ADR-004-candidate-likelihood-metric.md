# ADR-004: rank candidate codes by likelihood, not option letters

Status: accepted, implemented.

## Context

The first task design listed four codes as multiple-choice options and scored the next-token
probability of the option letter. Two tokenizer properties made that metric unusable:

1. **Asymmetric tokenization.** With the LFM3 tokenizer, `" A"` and `" C"` are single tokens,
   but `" B"` and `" D"` split into a generic space token followed by the letter. A scorer
   that takes `encode(" " + letter)[0]` therefore reads the shared space token for B and D
   and the letter token for A and C.
2. **Option-position bias.** After the prompt `"Answer ("`, the model put `0.82–0.94` mass on
   option A regardless of the correct code; after `"Answer:"` the bias was `0.6–0.95` toward
   A and D. Randomizing the correct position turns this into noise at the aggregate level,
   but a single-sample run is unreadable and the memory signal is swamped.

Generation-based grading was also fragile: the model sometimes stops after a prefix of a
five-digit code (`739` of `73914`), so exact match understates retrieval.

## Decision

Do not show candidates to the model. Ask the question and end the prompt with
`"Answer: "`. Score each candidate code by the **teacher-forced sum of token
log-probabilities** of that code as a continuation, from a clone of the arm's cache. The
highest total is the prediction; the gold code's rank and probability are reported.

- Four candidates: the gold plus three unique foils. Chance is 25%.
- The candidate list is in the sample metadata and artifact, but never in the prompt.
- Candidate order is irrelevant to the argmax; it is recorded for reproducibility.
- Cost: four cache clones and roughly three decode steps each per arm, negligible next to
  the prefill.

## Consequences

- On exact KV at 8k and 32k, the gold candidate gets probability `~1.0` (rank 1), confirming
  the task is answerable and the scoring path is correct.
- Streaming without the needle is at chance (`0.14–0.20` gold probability, rank 2–3).
- Memory can raise the gold probability without winning the argmax (e.g. `0.138 -> 0.219` at
  32k with 2048 slots). That is reported as a partial retrieval signal, not a pass.
- For a writeback arm the scorer writes each fed token into a clone of the memory, refreshes
  the injected slot rows, and re-feeds the token before scoring the next one. This makes W1
  visible to the continuation; without it, C3 and C4 would be identical by construction.
- Generated text is still recorded as a secondary diagnostic, but it is not the primary
  metric.
