# SPRM verification workspace

This repository is the verification harness for **SPRM** (stochastic predictive regression
memory). The research design lives in [`DESIGN.md`](DESIGN.md); the claim contract for every
gate lives in [`docs/VERIFICATION_PROTOCOL.md`](docs/VERIFICATION_PROTOCOL.md).

| Form | Description | Status |
|------|-------------|--------|
| **A** | Per-layer global associative memory, delta-rule writes, exact Tier-1 sink+window | implemented; V0/V1/V2 PASS; full D1 FAIL (McNemar p=0.125); D5 FAIL |
| **N** | Compressed prefix of soft tokens, frozen main model | contract + budget planner only; not verified |

The verification model is the local `LFM3-1.7B` checkpoint at
`/Users/yikewang/Model/LFM3/LFM3-1.7B`. It is a tri-hybrid (13 short-conv, 5 sliding-window,
6 global-attention layers). Form A only touches the six global-attention layers; conv and
sliding layers stay native. The runs in `verification/` were produced on an Apple M4 with MPS
and 16 GB unified memory.

## Current archive (2026-10-30)

| gate | run | status | what it establishes |
|------|-----|--------|---------------------|
| V0 | `20261030T064802Z_v0_d1_niah_lfm3_e72d25ba` | PASS | backend contract, RoPE round-trip, cache invariants |
| V1 | `20261030T064813Z_v1_d1_niah_lfm3_e72d25ba` | PASS | allocator, memory writes, trust region, archive integrity |
| V2 | `20261030T064814Z_v2_d1_niah_lfm3_e72d25ba` | PASS | cache surgery exact, online prefill structure |
| D1 | `20261030T064851Z_d1_d1_niah_lfm3_e72d25ba` | **FAIL** | full 216-sample grid (8 samples/cell), `reduced=false` |
| D5 | `20261030T113511Z_d5_d1_niah_lfm3_e72d25ba` | **FAIL** | 192-token W1 decode; condition `inf`, shrinkage 0.956 |

The full D1 ran with source `ab1c0bc2ce9b` on 216 samples (8 per cell, all three kinds,
lengths and depths) and sealed with `reduced=false`. It supersedes the earlier reduced run.

- `C0` exact KV: **97.2%** overall, 94.4% at 32k (the task ceiling is high but not perfect);
- `C1` streaming: 42.6% overall, **26.4% at 32k**;
- `C3`/`C4` memory: 44.9% overall, **30.6% at 32k**;
- far-bucket pooled gain: **+4.17 pp**; by kind: single +4.17, multi +8.33, conditional 0.0
  (**2/3 kinds** improved);
- near-bucket degradation: **0.0**;
- paired far: 3 improved, 0 degraded, 19 both, 50 neither; **McNemar p=0.125**.

The gate fails only on the McNemar threshold (`alpha=0.10`): the +4.17 pp gain is real but
rests on three discordant pairs. W1 adds nothing measurable: C3 and C4 have identical
accuracy and 0 improved/0 degraded pairs; mean gold candidate probability is 0.4267 (C3)
versus 0.4264 (C4). The exact ceiling and the above-chance far numbers confirm the task and
scoring; the 512-slot memory effect is small and not statistically significant at this
sample size.

D5 (one 8k sample, 192-token W1 decode) also fails: the key-matrix condition is `inf`
(rank-deficient keys) and the window reconstruction shrinkage is 0.956, i.e. the error grew
about 4.6% over the decode. Anchors, replay, and eta adaptation are not implemented and are
not tested.

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
- D1-style experiment on synthetic passkey retrieval at 8k / 16k / 32k.
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
