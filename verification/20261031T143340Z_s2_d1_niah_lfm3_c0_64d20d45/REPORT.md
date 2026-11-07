# S2 speculative decoding on natural text / memory arm

Arm: `C0_full`; proposer: `prompt`; partial accept: False.
Prompt: 4128 tokens (sha256 `bf407000ad9c`); decode budget: 32; gamma=4, ngram=3.

| run | seconds | tokens/s |
|-----|---------|----------|
| greedy | 1.8483 | 17.3134 |
| speculative | 1.6392 | 19.5223 |

- net speedup: **1.1276x** (target 1.3x)
- output identical to greedy: **True**
- acceptance: **0.3871** (target 0.5); full-accept rate 0.3333
- proposed 31, accepted 12, full 3, partial 0, verify 9, fallbacks 6

| check | result | detail |
|-------|--------|--------|
| output_matches_greedy | PASS | 32 tokens |
| draft_proposed | PASS | 31 |
| acceptance_target | FAIL | {'acceptance': 0.3870967741935484, 'target': 0.5} |
| net_speedup_target | FAIL | {'speedup': 1.1275843989974568, 'target': 1.3} |

Verdict: **FAIL**.