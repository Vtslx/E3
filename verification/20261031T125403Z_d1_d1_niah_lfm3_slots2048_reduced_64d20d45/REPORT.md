# D1 attribution

Config: `d1_niah_lfm3`, seeds 20261030/7101/9107, sink=128, window=2048, slots=2048, block=512, allocation=attention.
Samples: 27; arms: C0_full, C1_stream, C3_memory, C4_memory_w1.

Memory effect = `C3_memory - C1_stream`; writeback effect = `C4_memory_w1 - C3_memory`.
`C4 - C1` is combined and does not gate.

## Accuracy by context length

| arm | overall | 8k | 16k | 32k |
|-----|---------|----|----|----|
| C0_full | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C1_stream | 0.4074 | 0.3333 | 0.5556 | 0.3333 |
| C3_memory | 0.9630 | 1.0000 | 1.0000 | 0.8889 |
| C4_memory_w1 | 0.9630 | 1.0000 | 1.0000 | 0.8889 |

## Memory effect (C3 vs C1) in the far bucket (>=32768)

| kind | C3 control | C3 candidate | gain |
|------|-----------------|--------------------|------|
| conditional | 0.0000 | 0.6667 | 0.6667 |
| multi_needle | 1.0000 | 1.0000 | 0.0000 |
| single_needle | 0.0000 | 1.0000 | 1.0000 |

## Writeback effect (C4 vs C3) in the far bucket (>=32768)

| kind | C4 control | C4 candidate | gain |
|------|-----------------|--------------------|------|
| conditional | 0.6667 | 0.6667 | 0.0000 |
| multi_needle | 1.0000 | 1.0000 | 0.0000 |
| single_needle | 1.0000 | 1.0000 | 0.0000 |

## Combined diagnostic (C4 vs C1)

pooled gain 0.5556, 2/3 kinds; not gating.

## Candidate likelihood (all samples)

| arm | mean gold logprob | mean gold probability | mean gold rank |
|-----|-------------------|-----------------------|----------------|
| C0_full | -0.03 | 0.9904 | 1.00 |
| C1_stream | -11.53 | 0.4390 | 2.22 |
| C3_memory | -6.73 | 0.8531 | 1.04 |
| C4_memory_w1 | -6.73 | 0.8522 | 1.04 |

## Paired comparisons

- memory C3 vs C1 (far): improved 5, degraded 0, p=0.0312.
- writeback C4 vs C3 (far): improved 0, degraded 0, p=1.0000.
- combined C4 vs C1 (far): improved 5, degraded 0, p=0.0312.
- memory near (<= 8192): control 0.3333, candidate 1.0000, degradation -0.6667.
- writeback near (<= 8192): control 1.0000, candidate 1.0000, degradation 0.0000.

## Stability diagnostics

| arm | recon final | max cond | trust scaled | memory KiB | KV B/token |
|-----|-------------|----------|--------------|------------|------------|
| C0_full | nan | nan | 0 | 0.0 | 12517 |
| C1_stream | nan | nan | 0 | 0.0 | 2121 |
| C3_memory | 0.6215 | 507926.8785 | 6 | 49344.0 | 7510 |
| C4_memory_w1 | 0.6215 | 509104.6420 | 6 | 49344.0 | 7510 |

## Checks

| check | result | detail |
|-------|--------|--------|
| memory_far_bucket | PASS | 9 |
| memory_kinds_improved | PASS | 2/3 |
| memory_near_degradation | PASS | -0.6666666666666667 |
| memory_mcnemar | PASS | 0.03125 |
| writeback_far_bucket | PASS | 9 |
| writeback_kinds_improved | FAIL | 0/3 |
| writeback_near_degradation | PASS | 0.0 |
| writeback_mcnemar | FAIL | 1.0 |

Verdict: **FAIL: memory C3-C1 far 0.5556 (2/3 kinds, p=0.0312); writeback C4-C3 far 0.0000 (0/3 kinds, p=1.0000)**.

Condition numbers of `inf` mean the key matrix is rank-deficient, which is expected
for repetitive filler; it is a capacity/stability diagnostic, not a numerical blow-up.