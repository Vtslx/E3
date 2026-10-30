# D5 closed-loop stability

Arm: `C4_memory_w1`; decode budget: 192 tokens; samples: 1.

| sample | decoded | recon head | recon tail | shrinkage | max cond | trust scaled | writes |
|--------|---------|------------|------------|-----------|----------|--------------|--------|
| single_needle_L8192_d0p1_i0_0ed14387 | 192 | 0.6170 | 0.6456 | 0.9558 | inf | 0 | 1218 |

| check | result | detail |
|-------|--------|--------|
| condition_bounded | FAIL | {'max': inf, 'limit': 1000000.0} |
| window_shrinkage | FAIL | {'min': 0.9558269931801934, 'limit': 1.0} |
| decode_completed | PASS | {'single_needle_L8192_d0p1_i0_0ed14387': 192} |

Verdict: **FAIL**. Condition `inf` means the key matrix is rank-deficient; it is a
stability finding, not a numerical blow-up. Anchors, replay, and eta adaptation are not
implemented and are not tested here.