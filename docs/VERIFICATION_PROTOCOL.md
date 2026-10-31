# SPRM verification protocol

Version 1, 2026-10-30. This document is the contract for every `sprm verify` run: what each
gate measures, what a PASS means, and what it does not mean. If a claim is not listed here,
the harness does not make it.

## 1. Scope

The harness verifies the **memory half** of SPRM on one local checkpoint
(`LFM3-1.7B`, a tri-hybrid with 13 conv, 5 sliding-attention, and 6 global-attention
layers) on Apple MPS. It does not implement or verify the proposer, tree verification,
speculative acceptance, or any wall-clock speedup.

Verified:

- exact sink+window KV, Form A associative memory, online W2 absorption, W1 writeback;
- free reconstruction-error probe from the exact window;
- cache surgery and the online compressed-prefill protocol;
- deterministic synthetic passkey retrieval at 8k / 16k / 32k;
- archive integrity and reproducibility metadata.

Not verified:

- distribution equivalence with the uncompressed model;
- real benchmarks (LongBench, RULER, InfinityBench, NIAH variants beyond this protocol);
- Form N quality (no compressor checkpoint exists);
- proposer / acceptance-rate gains;
- throughput or latency claims of any kind.

## 2. Reproducibility

- Every run is a new directory under `verification/`; existing runs are never modified.
- `MANIFEST.json` records the config hash, source-tree hash, environment, argv, and a
  SHA-256 for every artifact. `MANIFEST.sha256` seals the manifest itself.
- `sprm archive verify <run>` re-hashes the run and reports mismatches.
- Task token IDs are generated from explicit seeds and hashed per sample; the hash is stored
  in the sample metadata and in the artifact.
- MPS kernels are not guaranteed bit-exact across processes. V0 checks same-process prefill
  determinism; cross-process bit-exactness is not claimed.
- The source-tree hash in `MANIFEST.json` is computed **at seal time**. Do not edit
  `src/sprm` while a run is in progress; a long run should be started from a frozen tree.
- Greedy decoding is used; `generate`-style sampling is never involved.

## 3. Gates

### V0 — backend contract

Checks: layer partition (conv / sliding / full), KV width and dtype, prefill length, KV
shape, finite logits, cache invariants, RoPE round-trip error, same-process prefill
determinism, and a finite decode step. Also records model file hashes.

PASS: every check passes. `rope_roundtrip <= 0.05` in bf16; prefill determinism `<= 0.05`.

Establishes: the checkpoint loads through the declared backend and the exact KV surface
behaves as the rest of the harness assumes.

Does not establish: retrieval quality, memory behaviour, or speed.

### V1 — contract and property gate

Runs without the model. Checks: allocator budget and monotonicity; W2 reduces
reconstruction error on synthetic clustered data; the trust region caps a huge write;
forgetting halves value norms; KV-slice invariants; the exact window carries one position per
token; archive sealing and tamper detection; the McNemar statistic.

PASS: every check passes.

Establishes: the numerical pieces D1 depends on are internally consistent.

Does not establish: anything about the model, real text, or far-context retrieval.

### V2 — cache-surgery and online-structure gate

Two independent checks:

1. **Surgery exactness.** Prefill a short body (all but the final token) twice. The reference
   feeds the final token through the untouched body cache; the rebuilt path copies the full
   global KV set and feeds the same token. With no eviction, the rebuilt KV bytes must be
   identical and the next-token logits must agree.
2. **Eviction slices and online structure.** On a longer prompt, the retained sink and window
   slices from `assemble_effective_global` must be byte-identical to the source cache. The
   online streaming prefill must have effective length `sink + window`; the online memory
   prefill must have `sink + budget + window`, and both must reproduce the exact sink bytes.

PASS: `max_logit_diff <= v2_max_logit_diff` (0.05), top-1 agreement
`>= v2_min_top1_agreement` (1.0), all tensor diffs exactly zero, and both online lengths as
expected.

The report also prints the gap between a full prefill and a one-token decode. That gap is a
bf16/kernel difference, not a surgery defect; it is diagnostic only.

Does not establish: that compressed memory preserves the distribution.

### D1 — far-context retrieval

**Task.** A context of deterministic filler records with one or more passkey sentences.
The question asks for one passkey; no candidates are shown to the model. Four candidate codes
are scored by teacher-forced sequence log-probability, and the highest-probability candidate
is the prediction. Chance is 25%. This avoids option-order bias and the tokenizer's
letter-tokenization asymmetry (see ADR-004).

Kinds:

- `single_needle`: one record.
- `multi_needle`: three records with distinct owners; only one is asked about.
- `conditional`: two attributes for the same owner; the question names the attribute.

Depths: needle position as a fraction of the context from the start (0.1, 0.5, 0.9).
Context lengths: 8192, 16384, 32768 tokens (exact, token-level).

**Arms.**

| arm | compression | memory | W1 |
|-----|-------------|--------|----|
| `C0_full` | none (exact full KV) | no | no |
| `C1_stream` | sink + window | no | no |
| `C3_memory` | sink + slots + window | W2 | no |
| `C4_memory_w1` | sink + slots + window | W2 | yes |

Every arm is prefilled with the context it will actually see. The compressed arms use the
online prefill: each forwarded chunk is preceded by absorbing the outgoing block and
rebuilding the global cache. Prefilling the full context and then slicing the KV is invalid
because the layers above the first dropped global layer already saw the dropped tokens
(ADR-003).

**Future-query allocation.** `C3`/`C4` do not use uniform slots. During the `C0` exact
prefill, a `q_norm` hook captures the question-token queries; they are rotated at their
absolute positions, grouped to KV heads (GQA average), and their attention mass to each
evicted block is the block importance. The slot budget is allocated proportionally with the
configured floor and cap (`max_slot_multiplier`). The 512- and 2048-slot runs compare the two
budgets; `--slots` overrides `tier2.num_slots` and is recorded in the manifest, not in the
config snapshot.

**Primary metric.** Exact match of the candidate-ranking argmax against the gold code.

**Secondary metrics.** Mean gold candidate log-probability, mean gold probability, mean gold
rank, generated code, reconstruction error, condition number, trust-region count, memory
bytes, and KV bytes per token.

**W1 isolation.** Candidate scoring is W1-aware: for a writeback arm, each fed candidate
token is written into a clone of the memory, the injected slot rows are refreshed, and the
token is fed again so the next candidate token is scored under the corrected memory. The
runtime does the same after every generated token. W1 has a one-step lag in the live decode
(the logits for `t+1` are computed before the write of `t`); the scorer's re-feed removes that
lag for the scored continuation. `C3` and `C4` therefore differ on the second and later code
tokens.

**Buckets.** Far: `context_length >= d1_min_far_distance` (32768). Near:
`context_length <= min(configured lengths)` (8192).

**Attribution.** Two independent contrasts, never a single `C4 - C1`:

- **memory effect** = `C3_memory - C1_stream` (does the slot memory help?);
- **writeback effect** = `C4_memory_w1 - C3_memory` (does W1 add anything on top?);
- `C4 - C1` is reported as a combined diagnostic and never gates.

Each effect is checked separately against all four conditions:

1. at least one far-bucket sample;
2. `kinds_improved >= d1_min_tasks_improved` (2), where a kind improves when
   `candidate - control >= d1_min_absolute_gain` (0.02) in the far bucket;
3. near-bucket degradation `control - candidate <= d1_max_near_degradation` (0.0);
4. pooled far McNemar one-sided p `<= d1_mcnemar_alpha` (0.10), pairing the same samples.

D1 PASS requires **both** effects to pass. If memory passes and writeback fails, the archive
says so explicitly: the memory half is viable, the W1 half is not.

A FAIL is a result, not an error. It means the configured memory budget did not beat
sink+window on this task; the archive records it. Per the design's M2 rule, if the 32k far
bucket is still only a few points above chance after the allocation and stability work, stop
A or move to SparSpec rather than adding samples.

**Reduced runs.** `--samples-per-cell`, `--max-samples`, `--kinds`, `--lengths`, `--depths`,
and `--arms` reduce the grid. The run's `MANIFEST.json` records `extra.reduced = true` and
the overrides. A reduced run is a smoke test, not the pre-registered D1. Only the full
config's verdict is a D1 verdict.

### D5 — stability (implemented)

One 8k sample, `C4_memory_w1`, 192 W1 decode tokens with attention allocation, anchors,
bounded replay, eta adaptation, and the slot refresh. Checks:

1. `max condition <= d5_max_condition` (1e6);
2. **generated-token** reconstruction shrinkage `head/tail >= d5_min_eta_ratio` (1.0);
3. the decode ran the full budget.

The generated-token window is the part W1 actually writes. The full-window shrinkage is
reported as a **diagnostic**, not a gate: most of that window is prefill KV that W1 never
wrote, so it tests the memory's out-of-sample generalization, not the writeback.

Anchors: the reference is set on the first W1 write and refreshed every
`anchor_interval` writes; otherwise the memory is blended toward it by `anchor_blend`.
Replay: `write_slots` keeps the last `replay_blocks` exact blocks; every
`replay_interval` W1 writes they are re-written with a half-gain delta (this is a delta
replay, not the design's batch least-squares refit). Eta: every `eta_adapt_interval` steps
the gain is scaled by `exp(rate * relative_error_change)`, clamped to
`[eta_min, eta_max]`.

D5 2048 passes (condition 80, generated-token shrinkage >= 1); D5 512 fails on the configured
checks and is archived as a budget-limited negative result. D5 does not measure long-text
task accuracy or distribution equivalence.

## 4. Archive layout

```
verification/<UTC timestamp>_<gate>_<slug>_<config hash>/
  MANIFEST.json        hashes, environment, argv, model info
  MANIFEST.sha256
  config.yaml          exact config snapshot
  logs/stdout.log      structured JSON log
  logs/events.jsonl    same events, machine-readable
  artifacts/           per-sample results, summaries, checks
  RESULT.json          status, verdict, measured values
  REPORT.md            human-readable report
  CLAIMS.md            claim boundary for this run
```

`verification/index.json` is an append-only registry.

### S1 — speculative decoding speed (implemented)

Prompt-lookup draft (`n`-gram suffix match) with an **all-or-nothing** commit: the target
verifies the whole proposal in one forward; if every token matches greedy, the batch plus one
bonus token is kept; otherwise global/sliding KV is cropped and conv state restored, and the
normal greedy token is decoded. Checks:

1. output is token-identical to greedy decoding;
2. `speculative tokens/s > greedy tokens/s`;
3. the draft proposed at least one token.

The stored run uses `C0_full` (exact KV); the memory arm is not a speed lever. It does not
claim general-task acceptance, memory-arm speed, batch>1 throughput, or that compression
itself accelerates generation.

## 4.1 Current archive (2026-10-30, A route)

All runs are from source/config `b165bff0`; `sprm archive verify` passes on each.

| gate | run id | status | note |
|------|--------|--------|------|
| V0 | `20261030T125922Z_v0_d1_niah_lfm3_b165bff0` | PASS | backend contract |
| V1 | `20261030T125935Z_v1_d1_niah_lfm3_b165bff0` | PASS | contracts/properties |
| V2 | `20261030T125936Z_v2_d1_niah_lfm3_b165bff0` | PASS | surgery + online structure |
| D5 | `20261030T130006Z_d5_d1_niah_lfm3_slots512_b165bff0` | FAIL | 512 slots: condition/shrinkage fail |
| D5 | `20261030T130102Z_d5_d1_niah_lfm3_slots2048_b165bff0` | PASS | condition 80; generated-W1 shrinkage pass |
| S1 | `20261030T130205Z_s1_d1_niah_lfm3_b165bff0` | PASS | 2.34x, output match, acceptance 1.0 |
| D1 | `20261030T130237Z_d1_d1_niah_lfm3_slots512_b165bff0` | FAIL | memory far +1.39pp, p=0.50; writeback 0/0 |
| D1 | `20261030T175745Z_d1_d1_niah_lfm3_slots2048_b165bff0` | FAIL | memory PASS (+43.06pp, p=3.9e-7); writeback FAIL (0.0, p=0.75) |
| D1 | `20261031T052302Z_d1_d1_niah_lfm3_slots1024_b165bff0` | FAIL | memory PASS (+13.89pp, p=0.021); writeback FAIL (0.0) |

**D1 512:** memory overall 59.3% vs 42.6% (8k 79.2/47.2, 16k 70.8/54.2), but the 32k far
gain is only +1.39pp (27.8% vs 26.4%), paired 4 improved / 3 degraded, p=0.50. Writeback is
0/0. Per M2, this budget is a stop: 32k is within a few points of chance.

**D1 2048:** memory far 26.4% -> 69.4% (**+43.06pp**), 3/3 kinds (single 29.2 -> 100,
multi 37.5 -> 66.7, conditional 12.5 -> 41.7), near degradation -40.3pp, 36 improved /
5 degraded, p=3.9e-7: **memory effect PASS**. Writeback is 0.0pp, 0/3 kinds, 1 improved /
1 degraded, p=0.75: **writeback effect FAIL**; C3 and C4 both score 50/72 at 32k.

**D1 1024:** memory far 26.4% -> 40.3% (+13.89pp), 2/3 kinds, near -34.7pp, 15 improved /
5 degraded, p=0.021: memory PASS. Writeback 0.0, 0/3, p=1.0: FAIL.

**Frontier (32k, resident bytes):** 512 = 4.84x / 27.8%; 1024 = 3.52x / 40.3%;
2048 = 2.28x / 69.4%. The 8x design target appears only in the no-W1/bf16/logical
projections, not in the measured resident implementation. The working point is 2048 at
2.28x resident. See `docs/frontier.md` and `docs/frontier.svg`.

**Decision (M2):** at 512, stop A or move to SparSpec; at 2048, keep A's memory
(W2 + future-query allocation) and drop or redesign W1. N remains deferred. W1 is cut from
the frontier (ADR-008).

## 5. Interpretation rules

- Report accuracy with the candidate count (4-way, chance 25%) and the bucket. Do not
  compare this accuracy to a free-form NIAH number.
- A gold probability above chance without an argmax win is a partial retrieval signal. Report
  it, but do not count it as a D1 pass.
- S1 is the only speed gate. It uses the exact `C0` arm and a repeated-sentence prompt; it
  does not claim general-task acceptance, memory-arm speed, or batch>1 throughput.
- A D1 FAIL at 512 slots does not prove the design cannot work; it falsifies the configured
  budget. The 2048 run shows the memory effect at 4x the budget; that is a budget change, not
  a free win, and the writeback is still null at both budgets.
- Form N is not verified and must not be described as implemented beyond its budget planner.
