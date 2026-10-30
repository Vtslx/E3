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

**Pass condition** (all four):

1. at least one far-bucket sample;
2. `kinds_improved >= d1_min_tasks_improved` (2), where a kind improves when
   `acc(C4) - acc(C1) >= d1_min_absolute_gain` (0.02) in the far bucket;
3. near-bucket degradation `acc(C1) - acc(C4) <= d1_max_near_degradation` (0.0);
4. pooled far McNemar one-sided p `<= d1_mcnemar_alpha` (0.10), pairing C1 and C4 on the
   same samples.

A FAIL is a result, not an error. It means the configured memory budget did not beat
sink+window on this task; the archive records it.

**Reduced runs.** `--samples-per-cell`, `--max-samples`, `--kinds`, `--lengths`, `--depths`,
and `--arms` reduce the grid. The run's `MANIFEST.json` records `extra.reduced = true` and
the overrides. A reduced run is a smoke test, not the pre-registered D1. Only the full
config's verdict is a D1 verdict.

### D5 — stability (implemented)

One 8k sample, `C4_memory_w1`, 192 W1 decode tokens with the slot refresh. Checks:

1. `max condition <= d5_max_condition` (1e6);
2. window reconstruction shrinkage `head/tail >= d5_min_eta_ratio` (1.0);
3. the decode ran the full budget.

`d5_min_eta_ratio` is interpreted as the minimum shrinkage ratio because the design's eta
adaptation is not implemented. The archived run
`20261030T113511Z_d5_d1_niah_lfm3_e72d25ba` failed both substantive checks: condition `inf`
(rank-deficient key matrix on repetitive filler) and shrinkage 0.956 (the window
reconstruction error grew about 4.6% over 192 tokens). This is a stability finding, not a
numerical blow-up.

D5 does not test anchors, replay, or adaptive eta; those are not implemented. It also does
not measure long-text task accuracy or distribution equivalence.

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

## 4.1 Current archive (2026-10-30)

| gate | run id | status | note |
|------|--------|--------|------|
| V0 | `20261030T064802Z_v0_d1_niah_lfm3_e72d25ba` | PASS | model hashes recorded |
| V1 | `20261030T064813Z_v1_d1_niah_lfm3_e72d25ba` | PASS | no model load |
| V2 | `20261030T064814Z_v2_d1_niah_lfm3_e72d25ba` | PASS | surgery + online structure |
| D1 | `20261030T064851Z_d1_d1_niah_lfm3_e72d25ba` | FAIL | full 216-sample grid, `reduced=false` |
| D5 | `20261030T113511Z_d5_d1_niah_lfm3_e72d25ba` | FAIL | 192-token W1 decode; condition `inf`, shrinkage 0.956 |

The full D1 ran from source `ab1c0bc2ce9b` on 216 samples. `C0` was 97.2% overall
(94.4% at 32k), `C1` 42.6% (26.4% at 32k), `C3`/`C4` 44.9% (30.6% at 32k). The far-bucket
gain was +4.17 pp (single +4.17, multi +8.33, conditional 0.0; 2/3 kinds), near degradation
0.0, and the paired far result was 3 improved / 0 degraded / p=0.125. The gate fails only on
the McNemar threshold. W1 was 0 improved / 0 degraded; C3 and C4 are indistinguishable in
accuracy and nearly so in candidate probability.

D5 was added after the D1 seal, so its source hash differs (`da069ddd2815`). It ran one 8k
sample for 192 W1 decode tokens. The key-matrix condition was `inf` (rank-deficient) and the
window reconstruction shrinkage was 0.956 (error grew ~4.6%); both configured checks failed.
This tests the implemented trust region and slot refresh only; anchors, replay, and eta
adaptation are not implemented.

## 5. Interpretation rules

- Report accuracy with the candidate count (4-way, chance 25%) and the bucket. Do not
  compare this accuracy to a free-form NIAH number.
- A gold probability above chance without an argmax win is a partial retrieval signal. Report
  it, but do not count it as a D1 pass.
- No speed claim may be derived from this harness: it measures memory quality and cache
  correctness, not throughput.
- A D1 FAIL at 512 slots does not prove the design cannot work; it falsifies the configured
  budget on this task. Capacity sweeps are diagnostics, not D1 redefinitions.
- Form N is not verified and must not be described as implemented beyond its budget planner.
