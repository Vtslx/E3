# SPRM verification workspace

This repository is the verification harness for **SPRM** (stochastic predictive regression
memory). The research design lives in [`DESIGN.md`](DESIGN.md); the claim contract for every
gate lives in [`docs/VERIFICATION_PROTOCOL.md`](docs/VERIFICATION_PROTOCOL.md).

| Form | Description | Status |
|------|-------------|--------|
| **A** | Per-layer global associative memory, delta-rule writes, exact Tier-1 sink+window | implemented; V0/V1/V2 PASS; S1 PASS; D1 512/1024/2048 done; W1 cut; C3 condition bounded but G0 fails on memory shrinkage (accuracy-only). Phase 1: full 2048 bf16 far **70.8%** at resident **5.05x** (fp32 buffers-off screen 3.85x). Phase 2: natural-text acceptance 0.39 exact / 0.03 C3; speed targets missed |
| **N** | Compressed prefix of soft tokens, frozen main model | contract + budget planner only; not verified |

The verification model is the local `LFM3-1.7B` checkpoint at
`/Users/yikewang/Model/LFM3/LFM3-1.7B`. It is a tri-hybrid (13 short-conv, 5 sliding-window,
6 global-attention layers). Form A only touches the six global-attention layers; conv and
sliding layers stay native. The runs in `verification/` were produced on an Apple M4 with MPS
and 16 GB unified memory.

## Current archive (2026-10-30, A route)

| gate | run | status | note |
|------|-----|--------|------|
| V0 | `20261030T125922Z_v0_d1_niah_lfm3_b165bff0` | PASS | backend contract |
| V1 | `20261030T125935Z_v1_d1_niah_lfm3_b165bff0` | PASS | contracts/properties |
| V2 | `20261030T125936Z_v2_d1_niah_lfm3_b165bff0` | PASS | cache surgery/online structure |
| D5 | `20261030T130006Z_d5_d1_niah_lfm3_slots512_b165bff0` | FAIL | 512 slots: condition/shrinkage fail |
| D5 | `20261030T130102Z_d5_d1_niah_lfm3_slots2048_b165bff0` | PASS | condition 80, generated-W1 shrinkage pass |
| S1 | `20261030T130205Z_s1_d1_niah_lfm3_b165bff0` | PASS | prompt-lookup 2.34x, output match |
| D1 | `20261030T130237Z_d1_d1_niah_lfm3_slots512_b165bff0` | **FAIL** | memory far +1.39pp, p=0.50; writeback 0/0; M2 stop at 512 |
| D1 | `20261030T175745Z_d1_d1_niah_lfm3_slots2048_b165bff0` | **FAIL** | memory PASS (+43.06pp far, p=3.9e-7); writeback FAIL (0.0, p=0.75) |
| D1 | `20261031T052302Z_d1_d1_niah_lfm3_slots1024_b165bff0` | **FAIL** | memory PASS (+13.89pp far, p=0.021); writeback FAIL (0.0) |
| D5 | `20261031T120925Z_d5_d1_niah_lfm3_slots2048_c3_c32e32d2` | **FAIL** | C3: condition 501,691 pass; memory shrinkage 0.956 fail |
| D5 | `20261031T124832Z_d5_d1_niah_lfm3_slots1024_c3_c32e32d2` | **FAIL** | C3: condition 501,935 pass; memory shrinkage 0.958 fail |
| D5 | `20261031T125249Z_d5_d1_niah_lfm3_slots2048_c3_64d20d45` | **FAIL** | C3 with W1 buffers off: same condition/shrinkage |
| D1 | `20261031T125403Z_d1_d1_niah_lfm3_slots2048_reduced_64d20d45` | **FAIL** (screen) | far C3 8/9 vs C1 3/9; resident **3.85x**; memory 48.2 MiB |
| D1 | `20261031T133823Z_d1_d1_niah_lfm3_slots1024_reduced_64d20d45` | **FAIL** (screen) | far C3 5/9 vs C1 3/9; resident **6.00x**; memory 24.1 MiB |
| D1 | `20261031T144448Z_d1_d1_niah_lfm3_slots2048_reduced_3efc1c20` | **FAIL** (screen) | bf16 core: far C3 8/9 vs C1 3/9; resident **5.05x**; memory 24.2 MiB |
| S2 | `20261031T143340Z_s2_d1_niah_lfm3_c0_64d20d45` | **FAIL** | C0 natural: acceptance 0.387, 1.128x, output match |
| S2 | `20261031T143420Z_s2_d1_niah_lfm3_c3_64d20d45` | **FAIL** | C3 512 natural: acceptance 0.082, 0.770x, diverges at 26 |
| S2 | `20261031T144154Z_s2_d1_niah_lfm3_slots2048_c3_64d20d45` | **FAIL** | C3 2048 natural: acceptance 0.030, 0.778x, output match |
| S2 | `20261031T144316Z_s2_d1_niah_lfm3_slots2048_c3_partial_64d20d45` | **FAIL** | C3 2048 partial: acceptance 0.294, 1.060x, output match |
| S2 | `20261031T144357Z_s2_d1_niah_lfm3_slots2048_c3_reservoir_64d20d45` | **FAIL** | C3 2048 reservoir: acceptance 0.030, 0.782x, output match |
| S2 | `20261031T144240Z_s2_d1_niah_lfm3_c0_reservoir_64d20d45` | **FAIL** | C0 reservoir: acceptance 0.387, 1.063x, output match |
| D1 | `20261031T153124Z_d1_d1_niah_lfm3_slots2048_3efc1c20` | **FAIL** (writeback only) | **full 2048 bf16**: far C3 **70.83%** vs C1 26.39% (+44.4pp, 3/3, p=2.2e-7); resident **5.05x**; memory 24.2 MiB |
| D1 | `20261031T211343Z_d1_d1_niah_lfm3_slots4096_reduced_3efc1c20` | **FAIL** (screen) | 4096 slots: far 8/9, resident 3.10x (no gain) |
| D1 | `20261031T215913Z_d1_d1_niah_lfm3_slots2048_reduced_57c7da0c` | **FAIL** (screen) | query-read row: far 8/9, resident 5.05x (no gain) |
| S2 | `20261031T211305Z_s2_d1_niah_lfm3_c0_draft_3efc1c20` | **FAIL** | low-rank draft heads: 0/122 accepted, 0.369x |

The D1 gate now reports **memory effect (`C3 - C1`)** and **writeback effect
(`C4 - C3`)** separately; `C4 - C1` is a combined diagnostic and never gates. The previous
uniform-allocation D1 is superseded: uniform gave +4.17 pp far but McNemar p=0.125 and no
writeback effect. The new runs use future-query attention allocation and the stability
mechanisms.

**512-sealed result:** the memory effect is large outside the far bucket (overall 59.3% vs
42.6%; 8k 79.2% vs 47.2%; 16k 70.8% vs 54.2%) but only +1.39pp at 32k (27.8% vs 26.4%,
paired 4 improved / 3 degraded, p=0.50). Writeback is exactly 0/0. The decision tool returns
**STOP A / switch to SparSpec at 512** under the M2 rule: 32k is within a few points of chance.
**2048-sealed result (the capacity test):** memory effect **PASS** — far C1 26.4% ->
C3 69.4% (**+43.06pp**), 3/3 kinds, near degradation -40.3pp (C3 better), 36 improved /
5 degraded, McNemar p=3.9e-7. By kind: single 29.2% -> 100%, multi 37.5% -> 66.7%,
conditional 12.5% -> 41.7%. C0 97.2% overall. Writeback effect **FAIL**: 0.0 pp, 0/3 kinds,
1 improved / 1 degraded, p=0.75; C3 and C4 have the same 32k far result (50/72).

**Decision (M2):**
- **512 slots: STOP A or switch to SparSpec at this budget.** 32k is only 2.8pp above chance
  and the memory effect is not significant. Do not add samples at 512.
- **2048 slots: KEEP A MEMORY (W2 + future-query allocation); DROP OR REDESIGN W1.** The
  memory effect is large and significant; the writeback is null. The next iteration should
  test a stronger/different W1 (gain, per-step refresh, or batch write) or remove C4 and keep
  C3. N remains deferred until the A writeback decision.

## Compression vs far-accuracy frontier

W1 decision: **cut the current W1** from the A-route frontier (ADR-008). It is 0/3 kinds and
p=0.75 at 2048 even with the slot refresh and W1-aware scorer; C3=C4 at 32k. The current W1
also pays for fp32 memory, anchor refs, and a replay buffer, which is why the resident
compression is only 2.28x at the 2048 working point.

Measured at 32k (`tools/frontier.py`, `docs/frontier.svg`):

| slots | dtype / buffers | far C1 | far C3 | resident | C3 MiB |
|-------|-----------------|--------|--------|----------|--------|
| 512 (full) | fp32, on | 26.4% | 27.8% | 4.84x | 117.6 |
| 1024 (full) | fp32, on | 26.4% | 40.3% | 3.52x | 58.8 |
| 2048 (full) | fp32, on | 26.4% | 69.4% | 2.28x | 117.6 |
| 2048 (reduced) | fp32, off | 33.3% | 88.9% | **3.85x** | 48.2 |
| 1024 (reduced) | fp32, off | 33.3% | 55.6% | **6.00x** | 24.1 |
| 2048 (reduced) | bf16, off | 33.3% | 88.9% | **5.05x** | 24.2 |
| 2048 (full) | bf16, off | 26.4% | **70.8%** | **5.05x** | 24.2 |

The reduced rows use 9 far samples (1 per cell), so the accuracies carry sampling error; their
resident numbers are exact for the configured memory. The full 512/1024/2048 rows are the
original 72-far-sample runs with the W1 buffers allocated.

The measured frontier is the resident line: 512 buys 4.84x but stays at chance (27.8%);
1024 gives 3.52x and 40.3%; 2048 gives 69.4% far accuracy at **2.28x resident**. The 8x
target appears only in the no-W1/bf16/logical projections; the gap is the fp32 memory
duplicate, the anchor refs, and the replay buffer. Cutting W1 removes the last two; a bf16
memory and a direct associative read (no duplicated slot cache) are the remaining
engineering steps. Do not claim 8x at the measured 2048 point. The 8x-at-512 point is 27.8%
far: near chance. The frontier is monotone in the wrong direction for the design's
8x-at-mached-quality claim.

## Phase 0 result: C3 stability

Key regularization (SVD floor `2e-6`) fixes the `inf` condition: **501,691 (2048)** and
**501,935 (1024)**, both below `1e6`. But the full-window memory shrinkage is **0.956
(2048)** and **0.958 (1024)**, below 1: the static C3 memory does not shrink the window
error over a 192-token decode. **G0 fails on shrinkage; A is accuracy-only — do not claim
C3 stability.** The W1-only anchor/replay buffers were dead weight for C3: turning them off
leaves the C3 D5 numbers unchanged and drops ~67 MiB at 2048. A memory-only adapt mechanism
(periodic write from the exact Tier-1 window) is a separate experiment.

## Phase 1 result: resident compression

The W1-only anchor/replay buffers were dead weight for C3. With `stability_buffers=false`
the C3 D5 condition/shrinkage are unchanged and the 2048 memory drops by ~67 MiB. A reduced
2048 D1 then measures **far 8/9 vs 3/9, resident 3.85x**; a reduced 1024 D1 measures
**5/9 vs 3/9, resident 6.00x**. Both Phase 1 gates (2048 >= 65%/3.8x, 1024 >= 35%/6x) are
met by the fp32 buffers-off build.

The bf16 memory core (`memory_dtype=bfloat16`; fp32 reads/writes, bf16 storage) halves the
core and moves 2048 to **5.05x** (24.2 MiB). The full 2048 confirmation: far C3 **70.83%**
vs C1 26.39% (+44.4pp, 3/3 kinds, p=2.2e-7), resident **5.05x**, condition max 506,933.
Direct associative read (removing the duplicated slot KV) would reach ~7.36x; it is deferred
because the resident targets are already met (ADR-010).

## Phase 2 result: speed half on natural text

S2 runs the prompt-lookup draft on a 4096-token natural-text prompt, exact KV and the C3
arm. Acceptance and net tokens/s:

| arm | acceptance | net tokens/s | output |
|-----|------------|--------------|--------|
| C0 exact | 0.387 | 1.128x | match |
| C3 2048 | 0.030 | 0.778x | match |
| C3 2048 + partial commit | **0.294** | 1.060x | match |
| C3 2048 + reservoir draft | 0.030 | 0.782x | match |

The Phase 2 targets (acceptance >= 0.5, net >= 1.3x) are **missed**. Prompt lookup is a
repetition detector: the C3 memory's continuation rarely follows the literal prompt n-grams.
Prefix commit recovers acceptance but the conv-state refeed costs a forward, so net speed
stays near 1.06x. The natural-text speed half needs the trained draft head of Phase 3, not a
stronger n-gram proposer.

A caveat: on natural text the chunked verify can round differently from single-token greedy
in bf16; the C3 512 run diverged at token 26. The rollback after a rejection was verified to
be bit-exact; the drift comes from the accepted chunk (ADR-010).

Follow-up negatives (2026-11-01): low-rank Medusa heads trained on 967k WikiText tokens give
held-out top-1 0.271/0.126/0.063/0.059 and accept **0/122** on the docs prompt (0.37x);
4096 slots and a question-query read row do not move the reduced far (8/9). The 2x
natural-text target requires a trained in-domain draft with tree verification (Phase 3),
and the conditional far gap needs a trained read/write; neither is reachable by capacity or
an untrained read row.

## What is verified

- Exact sink + window, Tier-2 associative memory, W1 writeback, W2 online absorption.
- The free reconstruction-error probe from the exact window.
- **Online compressed prefill**: each arm is prefilled with the context it actually sees.
  Post-hoc KV slicing is invalid because higher layers' window states already contain the
  dropped tokens; see ADR-003.
- **Candidate-likelihood retrieval metric**: the prompt asks for the passkey; four candidate
  codes are ranked by teacher-forced log-probability. Chance is 25%. This avoids the
  first-option and letter-tokenization biases that made multiple-choice letters unusable;
  see ADR-004.
- **Future-query attention allocation** for W2: the question queries from the exact `C0`
  prefill score each evicted block; the slot budget replaces uniform (ADR-005).
- **Stability**: anchors, bounded replay, eta adaptation, and D5 at 512/2048 slots
  (ADR-006).
- **D1 attribution**: memory (`C3 - C1`) and writeback (`C4 - C3`) are separate checks.
- **S1 speed**: prompt-lookup speculative decoding with all-or-nothing commit, output
  token-identical to greedy, tokens/s measured (ADR-007).
- **S2 speed**: natural-text and C3-arm acceptance/net tokens/s, prefix commit, reservoir
  draft, and the first greedy-divergence position (ADR-010).
- Stability diagnostics: reconstruction error, condition number, trust region, memory bytes,
  KV bytes per token, and the gold candidate's probability and rank.
- D5 long-decode stability gate: condition bound and window-reconstruction shrinkage.

Not claimed: distribution equivalence, real-benchmark generalization, proposer benefits,
Form N quality, or any wall-clock speedup.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

# Backend contract without archiving.
.venv/bin/sprm backend check --config configs/experiments/d1_niah.yaml

# Archived gates.
.venv/bin/sprm verify v0 --config configs/experiments/d1_niah.yaml
.venv/bin/sprm verify v1 --config configs/experiments/d1_niah.yaml
.venv/bin/sprm verify v2 --config configs/experiments/d1_niah.yaml

# Reduced D1 smoke (recorded as reduced; ~70 min on M4/MPS).
.venv/bin/sprm verify d1 --config configs/experiments/d1_niah.yaml \
  --samples-per-cell 2 --max-new-tokens 8

# Full pre-registered D1 grid (8 samples per cell; multi-hour).
.venv/bin/sprm verify d1 --config configs/experiments/d1_niah.yaml

# Inspect and audit the archive.
.venv/bin/sprm archive list --root verification
.venv/bin/sprm archive verify verification/<run-id>
```

A D1 FAIL exits with code 8 and still seals a complete run. Exit codes: 0 PASS, 8 gate
failure, 7 verification error, 3 config error, 9 archive error.

Every `verify` command creates a **new immutable directory** under `verification/`
(`<UTC timestamp>_<gate>_<slug>_<config-hash>`). A run cannot be overwritten; the command
fails if the directory exists. Each run contains:

```
MANIFEST.json       hashes, environment, seeds, model info, status
MANIFEST.sha256
config.yaml         exact configuration snapshot
logs/stdout.log     structured JSON log
logs/events.jsonl   machine-readable event stream
artifacts/          per-sample results, gate checks, summaries
RESULT.json         gate verdict + measured values
REPORT.md           human-readable report
CLAIMS.md           what this run does and does not establish
```

`verification/index.json` is an append-only registry of every run.

## Layout

```
src/sprm/
  config/          strict YAML -> dataclass loading
  contracts/       Memory / Proposer / ModelBackend protocols
  backends/lfm3/   loader, layer plan, rotary codec, rebuildable KV cache
  memory/          Form A associative memory, allocator, Form N planner
  runtime/         Form A online prefill/decode, Form N scaffold
  eval/            NIAH tasks, candidate metric, runner, gates
  verification/    immutable run archive, manifests, index
  cli.py           backend / form-a / form-n / verify / archive
tests/             unit, property, integration (--run-integration)
configs/           backend, experiment, gate configs
docs/              protocol, ADRs, Form A/N notes
verification/      one directory per executed gate (generated)
```

## Reproducibility contract

- Seeds are explicit in the config and recorded in the manifest.
- Task token IDs are generated deterministically and hashed; the hash is stored per sample.
- The model file hashes are recorded in V0 and D1 manifests.
- `sprm archive verify <run>` re-hashes every sealed artifact.
- MPS kernels are not guaranteed bit-exact across processes. V0 checks same-process
  determinism; the manifest records the device. Cross-process bit-exactness is not claimed.

## Selected design notes

- [ADR-001](docs/ADR-001-rope-unrotation-and-slot-scale.md): store keys unrotated, rotate
  slots at their representative positions, and restore native key scale at injection.
- [ADR-002](docs/ADR-002-normalized-keys-and-read-temperature.md): unit-normalized keys,
  explicit read temperature, trust-region norm floor, value-only forgetting.
- [ADR-003](docs/ADR-003-online-compressed-prefill.md): why post-hoc KV slicing is invalid
  and how the online prefill works.
- [ADR-004](docs/ADR-004-candidate-likelihood-metric.md): why the primary metric ranks
  candidate codes by likelihood instead of scoring option letters.
- [Form A runtime](docs/FORM_A.md) and [Form N scaffold](docs/FORM_N.md).
