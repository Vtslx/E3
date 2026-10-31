# D5 closed-loop stability

Arm: `C4_memory_w1`; decode budget: 192 tokens; samples: 1.

| sample | decoded | full shrink | W1 shrink | eta ratio | max cond | trust | writes | anchors | replays |
|--------|---------|-------------|-----------|-----------|----------|-------|--------|---------|---------|
| single_needle_L8192_d0p1_i0_0ed14387 | 192 | 0.8160 | 0.9438 | 0.9994 | 152690.3535 | 0 | 1218 | 18 | 36 |

| check | result | detail |
|-------|--------|--------|
| condition_bounded | PASS | {'max': 152690.3534696188, 'limit': 1000000.0} |
| writeback_shrinkage | FAIL | {'min': 0.9437620462722567, 'limit': 1.0} |
| decode_completed | PASS | {'single_needle_L8192_d0p1_i0_0ed14387': 192} |

Diagnostic (not gating):

| check | result | detail |
|-------|--------|--------|
| full_window_shrinkage_diagnostic | FAIL | {'min': 0.8159785977725799, 'limit': 1.0} |

Verdict: **FAIL**. The gating shrinkage uses the generated-token window, the part W1
actually writes. The full-window shrinkage is diagnostic: most of that window is prefill
KV that W1 never wrote. Anchors, bounded replay, and eta adaptation are implemented and
reported above.