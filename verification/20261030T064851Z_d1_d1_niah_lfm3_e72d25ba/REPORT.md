# D1 far-context retrieval

Config: `d1_niah_lfm3`, seeds 20261030/7101/9107, sink=128, window=2048, slots=512, block=512.
Samples: 216; arms: C0_full, C1_stream, C3_memory, C4_memory_w1.

## Accuracy by context length

| arm | overall | 8k | 16k | 32k |
|-----|---------|----|----|----|
| C0_full | 0.9722 | 0.9861 | 0.9861 | 0.9444 |
| C1_stream | 0.4259 | 0.4722 | 0.5417 | 0.2639 |
| C3_memory | 0.4491 | 0.4722 | 0.5694 | 0.3056 |
| C4_memory_w1 | 0.4491 | 0.4722 | 0.5694 | 0.3056 |

## Far bucket (>=32768) by task kind

| kind | C1_stream | C4_memory_w1 | gain |
|------|-----------|--------------|------|
| conditional | 0.1250 | 0.1250 | 0.0000 |
| multi_needle | 0.3750 | 0.4583 | 0.0833 |
| single_needle | 0.2917 | 0.3333 | 0.0417 |

## Candidate likelihood (all samples)

`C0` is the exact-KV ceiling; `C1` is the no-memory control. A gold probability near
1 means the model assigns the retrieved code almost all candidate mass; near 0.25 is
chance. The rank is the gold code's position among the four candidates.

| arm | mean gold logprob | mean gold probability | mean gold rank |
|-----|-------------------|-----------------------|----------------|
| C0_full | -0.08 | 0.9723 | 1.03 |
| C1_stream | -11.65 | 0.4107 | 2.19 |
| C3_memory | -11.56 | 0.4267 | 2.12 |
| C4_memory_w1 | -11.56 | 0.4264 | 2.13 |

## Paired comparisons (far bucket)

- C4_memory_w1 vs C1_stream: improved 3, degraded 0, p=0.1250.
- C4_memory_w1 vs C3_memory: improved 0, degraded 0, p=1.0000.
- Near bucket (<= 8192): control 0.4722, candidate 0.4722, degradation 0.0000; paired improved 1, degraded 1, p=0.7500.

## Stability diagnostics

| arm | recon final | max cond | trust scaled | memory KiB | KV B/token |
|-----|-------------|----------|--------------|------------|------------|
| C0_full | nan | nan | 0 | 0.0 | 12520 |
| C1_stream | nan | nan | 0 | 0.0 | 2126 |
| C3_memory | 0.7069 | inf | 49958 | 12336.0 | 3474 |
| C4_memory_w1 | 0.7068 | inf | 49959 | 12336.0 | 3474 |

## Checks

| check | result | detail |
|-------|--------|--------|
| far_bucket_present | PASS | 72 |
| kinds_improved | PASS | 2/3 |
| near_degradation | PASS | 0.0 |
| far_mcnemar | FAIL | 0.125 |

Verdict: **FAIL: C4_memory_w1 vs C1_stream at >=32768 tokens (0.0417 pooled gain, 2/3 kinds improved, McNemar p=0.1250)**.

Condition numbers of `inf` mean the key matrix is rank-deficient, which is expected for
repetitive filler: many slots encode the same direction. It is a capacity observation, not a
numerical blow-up. D1 does not gate on it; D5 is the dedicated stability gate.