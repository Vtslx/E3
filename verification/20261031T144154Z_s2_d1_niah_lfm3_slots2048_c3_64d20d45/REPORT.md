# S2 speculative decoding on natural text / memory arm

Arm: `C3_memory`; proposer: `prompt`; partial accept: False.
Prompt: 4096 tokens (sha256 `bf407000ad9c`); decode budget: 32; gamma=4, ngram=3.

| run | seconds | tokens/s |
|-----|---------|----------|
| greedy | 1.8407 | 17.3844 |
| speculative | 2.3658 | 13.5262 |

- net speedup: **0.7781x** (target 1.3x)
- output identical to greedy: **True**
- acceptance: **0.0303** (target 0.5); full-accept rate 0.1111
- proposed 33, accepted 1, full 1, partial 0, verify 9, fallbacks 8

- memory 48.2 MiB; 25448 KV B/token at 4096 prompt tokens

| check | result | detail |
|-------|--------|--------|
| output_matches_greedy | PASS | 32 tokens |
| draft_proposed | PASS | 33 |
| acceptance_target | FAIL | {'acceptance': 0.030303030303030304, 'target': 0.5} |
| net_speedup_target | FAIL | {'speedup': 0.7780652584444371, 'target': 1.3} |

Verdict: **FAIL**.