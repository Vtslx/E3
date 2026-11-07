# D1 attribution

Config: `d1_niah_lfm3`, seeds 20261030/7101/9107, sink=128, window=2048, slots=2048, block=512, allocation=attention.
Samples: 3; arms: C0_full, C1_stream, C3_memory, C4_memory_w1.

Memory effect = `C3_memory - C1_stream`; writeback effect = `C4_memory_w1 - C3_memory`.
`C4 - C1` is combined and does not gate.

## Accuracy by context length

| arm | overall | 8k | 16k | 32k |
|-----|---------|----|----|----|
| C0_full | 1.0000 | nan | nan | 1.0000 |
| C1_stream | 0.0000 | nan | nan | 0.0000 |
| C3_memory | 0.6667 | nan | nan | 0.6667 |
| C4_memory_w1 | 0.6667 | nan | nan | 0.6667 |

## Memory effect (C3 vs C1) in the far bucket (>=32768)

| kind | C3 control | C3 candidate | gain |
|------|-----------------|--------------------|------|
| conditional | 0.0000 | 0.6667 | 0.6667 |

## Writeback effect (C4 vs C3) in the far bucket (>=32768)

| kind | C4 control | C4 candidate | gain |
|------|-----------------|--------------------|------|
| conditional | 0.6667 | 0.6667 | 0.0000 |

## Combined diagnostic (C4 vs C1)

pooled gain 0.6667, 1/1 kinds; not gating.

## Candidate likelihood (all samples)

| arm | mean gold logprob | mean gold probability | mean gold rank |
|-----|-------------------|-----------------------|----------------|
| C0_full | -0.06 | 0.9561 | 1.00 |
| C1_stream | -14.81 | 0.1981 | 2.67 |
| C3_memory | -12.37 | 0.4667 | 1.33 |
| C4_memory_w1 | -12.38 | 0.4732 | 1.33 |

## Paired comparisons

- memory C3 vs C1 (far): improved 2, degraded 0, p=0.2500.
- writeback C4 vs C3 (far): improved 0, degraded 0, p=1.0000.
- combined C4 vs C1 (far): improved 2, degraded 0, p=0.2500.
- memory near (<= 8192): control nan, candidate nan, degradation nan.
- writeback near (<= 8192): control nan, candidate nan, degradation nan.

## Stability diagnostics

| arm | recon final | max cond | trust scaled | memory KiB | KV B/token |
|-----|-------------|----------|--------------|------------|------------|
| C0_full | nan | nan | 0 | 0.0 | 12386 |
| C1_stream | nan | nan | 0 | 0.0 | 909 |
| C3_memory | 0.6219 | 502664.8198 | 0 | 24768.0 | 2449 |
| C4_memory_w1 | 0.6219 | 503525.6517 | 0 | 24768.0 | 2449 |

## Checks

| check | result | detail |
|-------|--------|--------|
| memory_far_bucket | PASS | 3 |
| memory_kinds_improved | FAIL | 1/1 |
| memory_near_degradation | FAIL | nan |
| memory_mcnemar | FAIL | 0.25 |
| writeback_far_bucket | PASS | 3 |
| writeback_kinds_improved | FAIL | 0/1 |
| writeback_near_degradation | FAIL | nan |
| writeback_mcnemar | FAIL | 1.0 |

Verdict: **FAIL: memory C3-C1 far 0.6667 (1/1 kinds, p=0.2500); writeback C4-C3 far 0.0000 (0/1 kinds, p=1.0000)**.

Condition numbers of `inf` mean the key matrix is rank-deficient, which is expected
for repetitive filler; it is a capacity/stability diagnostic, not a numerical blow-up.