# D5 closed-loop stability

Arm: `C4_memory_w1`; decode budget: 192 tokens; samples: 1.

| sample | decoded | full shrink | W1 shrink | eta ratio | max cond | trust | writes | anchors | replays |
|--------|---------|-------------|-----------|-----------|----------|-------|--------|---------|---------|
| single_needle_L8192_d0p1_i0_0ed14387 | 192 | 0.7832 | 1.0105 | 0.9992 | 79.9578 | 0 | 1218 | 18 | 36 |

| check | result | detail |
|-------|--------|--------|
| condition_bounded | PASS | {'max': 79.9577789358876, 'limit': 1000000.0} |
| writeback_shrinkage | PASS | {'min': 1.0105389012566623, 'limit': 1.0} |
| decode_completed | PASS | {'single_needle_L8192_d0p1_i0_0ed14387': 192} |

Diagnostic (not gating):

| check | result | detail |
|-------|--------|--------|
| full_window_shrinkage_diagnostic | FAIL | {'min': 0.7832032263906201, 'limit': 1.0} |

Verdict: **PASS**. The gating shrinkage uses the generated-token window, the part W1
actually writes. The full-window shrinkage is diagnostic: most of that window is prefill
KV that W1 never wrote. Anchors, bounded replay, and eta adaptation are implemented and
reported above.