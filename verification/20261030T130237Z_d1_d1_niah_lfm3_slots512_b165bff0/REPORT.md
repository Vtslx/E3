# D1 attribution

Config: `d1_niah_lfm3`, seeds 20261030/7101/9107, sink=128, window=2048, slots=512, block=512, allocation=attention.
Samples: 216; arms: C0_full, C1_stream, C3_memory, C4_memory_w1.

Memory effect = `C3_memory - C1_stream`; writeback effect = `C4_memory_w1 - C3_memory`.
`C4 - C1` is combined and does not gate.

## Accuracy by context length

| arm | overall | 8k | 16k | 32k |
|-----|---------|----|----|----|
| C0_full | 0.9722 | 0.9861 | 0.9861 | 0.9444 |
| C1_stream | 0.4259 | 0.4722 | 0.5417 | 0.2639 |
| C3_memory | 0.5926 | 0.7917 | 0.7083 | 0.2778 |
| C4_memory_w1 | 0.5972 | 0.7917 | 0.7222 | 0.2778 |

## Memory effect (C3 vs C1) in the far bucket (>=32768)

| kind | C3 control | C3 candidate | gain |
|------|-----------------|--------------------|------|
| conditional | 0.1250 | 0.0833 | -0.0417 |
| multi_needle | 0.3750 | 0.4167 | 0.0417 |
| single_needle | 0.2917 | 0.3333 | 0.0417 |

## Writeback effect (C4 vs C3) in the far bucket (>=32768)

| kind | C4 control | C4 candidate | gain |
|------|-----------------|--------------------|------|
| conditional | 0.0833 | 0.0833 | 0.0000 |
| multi_needle | 0.4167 | 0.4167 | 0.0000 |
| single_needle | 0.3333 | 0.3333 | 0.0000 |

## Combined diagnostic (C4 vs C1)

pooled gain 0.0139, 2/3 kinds; not gating.

## Candidate likelihood (all samples)

| arm | mean gold logprob | mean gold probability | mean gold rank |
|-----|-------------------|-----------------------|----------------|
| C0_full | -0.08 | 0.9723 | 1.03 |
| C1_stream | -11.65 | 0.4107 | 2.19 |
| C3_memory | -10.53 | 0.5553 | 1.75 |
| C4_memory_w1 | -10.53 | 0.5552 | 1.76 |

## Paired comparisons

- memory C3 vs C1 (far): improved 4, degraded 3, p=0.5000.
- writeback C4 vs C3 (far): improved 0, degraded 0, p=1.0000.
- combined C4 vs C1 (far): improved 4, degraded 3, p=0.5000.
- memory near (<= 8192): control 0.4722, candidate 0.7917, degradation -0.3194.
- writeback near (<= 8192): control 0.7917, candidate 0.7917, degradation 0.0000.

## Stability diagnostics

| arm | recon final | max cond | trust scaled | memory KiB | KV B/token |
|-----|-------------|----------|--------------|------------|------------|
| C0_full | nan | nan | 0 | 0.0 | 12520 |
| C1_stream | nan | nan | 0 | 0.0 | 2126 |
| C3_memory | 0.7041 | inf | 51617 | 46558.2 | 5968 |
| C4_memory_w1 | 0.7074 | 1341613107.7481 | 51619 | 46558.2 | 5968 |

## Checks

| check | result | detail |
|-------|--------|--------|
| memory_far_bucket | PASS | 72 |
| memory_kinds_improved | PASS | 2/3 |
| memory_near_degradation | PASS | -0.3194444444444444 |
| memory_mcnemar | FAIL | 0.5 |
| writeback_far_bucket | PASS | 72 |
| writeback_kinds_improved | FAIL | 0/3 |
| writeback_near_degradation | PASS | 0.0 |
| writeback_mcnemar | FAIL | 1.0 |

Verdict: **FAIL: memory C3-C1 far 0.0139 (2/3 kinds, p=0.5000); writeback C4-C3 far 0.0000 (0/3 kinds, p=1.0000)**.

Condition numbers of `inf` mean the key matrix is rank-deficient, which is expected
for repetitive filler; it is a capacity/stability diagnostic, not a numerical blow-up.