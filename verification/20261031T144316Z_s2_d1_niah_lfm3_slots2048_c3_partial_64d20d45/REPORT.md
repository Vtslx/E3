# S2 speculative decoding on natural text / memory arm

Arm: `C3_memory`; proposer: `prompt`; partial accept: True.
Prompt: 4096 tokens (sha256 `bf407000ad9c`); decode budget: 32; gamma=4, ngram=3.

| run | seconds | tokens/s |
|-----|---------|----------|
| greedy | 2.1687 | 14.7554 |
| speculative | 2.0454 | 15.6448 |

- net speedup: **1.0603x** (target 1.3x)
- output identical to greedy: **True**
- acceptance: **0.2941** (target 0.5); full-accept rate 0.2000
- proposed 17, accepted 5, full 1, partial 2, verify 5, fallbacks 2

- memory 48.2 MiB; 25448 KV B/token at 4096 prompt tokens

| check | result | detail |
|-------|--------|--------|
| output_matches_greedy | PASS | 32 tokens |
| draft_proposed | PASS | 17 |
| acceptance_target | FAIL | {'acceptance': 0.29411764705882354, 'target': 0.5} |
| net_speedup_target | FAIL | {'speedup': 1.060275432030324, 'target': 1.3} |

Verdict: **FAIL**.