# SPRM verification workspace

This repository is the verification harness for **SPRM** (stochastic predictive regression
memory). The research design lives in [`DESIGN.md`](DESIGN.md); the claim contract for every
gate lives in [`docs/VERIFICATION_PROTOCOL.md`](docs/VERIFICATION_PROTOCOL.md).

| Form | Description | Status |
|------|-------------|--------|
| **A** | Per-layer global associative memory, delta-rule writes, exact Tier-1 sink+window | implemented; V0/V1/V2 PASS; D5 2048 PASS; D5 512 FAIL; S1 PASS; D1 512/1024/2048 done; W1 cut |
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

| slots | far C1 | far C3 | resident | no W1 buffers | bf16 core | logical | global attn |
|-------|--------|--------|----------|---------------|-----------|---------|-------------|
| 512 | 26.4% | 27.8% | 4.84x | 8.33x | 9.56x | 11.19x | 12.19x |
| 1024 | 26.4% | 40.3% | 3.52x | 6.00x | 7.38x | 9.53x | 10.24x |
| 2048 | 26.4% | 69.4% | **2.28x** | 3.85x | 5.06x | 7.36x | 7.76x |

The measured frontier is the resident line: 512 buys 4.84x but stays at chance (27.8%);
1024 gives 3.52x and 40.3%; 2048 gives 69.4% far accuracy at **2.28x resident**. The 8x
target appears only in the no-W1/bf16/logical projections; the gap is the fp32 memory
duplicate, the anchor refs, and the replay buffer. Cutting W1 removes the last two; a bf16
memory and a direct associative read (no duplicated slot cache) are the remaining
engineering steps. Do not claim 8x at the measured 2048 point. The 8x-at-512 point is 27.8%
far: near chance. The frontier is monotone in the wrong direction for the design's
8x-at-mached-quality claim.

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
