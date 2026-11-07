# D5 closed-loop stability

Arm: `C3_memory`; decode budget: 192 tokens; samples: 1.

| sample | decoded | full shrink | W1 shrink | eta ratio | max cond | trust | writes | anchors | replays |
|--------|---------|-------------|-----------|-----------|----------|-------|--------|---------|---------|
| single_needle_L8192_d0p1_i0_0ed14387 | 192 | 0.9557 | 1.0981 | 0.9998 | 501691.3673 | 0 | 72 | 0 | 0 |

| check | result | detail |
|-------|--------|--------|
| condition_bounded | PASS | {'max': 501691.36731183867, 'limit': 1000000.0} |
| memory_shrinkage | FAIL | {'min': 0.9556711696255701, 'limit': 1.0} |
| decode_completed | PASS | {'single_needle_L8192_d0p1_i0_0ed14387': 192} |

Diagnostic (not gating):

| check | result | detail |
|-------|--------|--------|
| generated_shrinkage_diagnostic | PASS | {'min': 1.0980600753281977, 'limit': 1.0} |

Verdict: **FAIL**. The gating shrinkage is full-window (memory-only C3). For C4 the full-window
value is diagnostic because most of that window is prefill KV that W1 never wrote; for C3
the full-window value is the memory shrinkage and the generated value is diagnostic.
Key regularization clamps the key spectrum on every build; anchors/replay/eta are W1-only