# S2 speculative decoding on natural text / memory arm

Arm: `C3_memory`; proposer: `prompt`; partial accept: False.
Prompt: 4128 tokens (sha256 `bf407000ad9c`); decode budget: 32; gamma=4, ngram=3.

| run | seconds | tokens/s |
|-----|---------|----------|
| greedy | 1.8252 | 17.5325 |
| speculative | 2.3710 | 13.4967 |

- net speedup: **0.7698x** (target 1.3x)
- output identical to greedy: **False**
- acceptance: **0.0816** (target 0.5); full-accept rate 0.0769
- proposed 49, accepted 4, full 1, partial 0, verify 13, fallbacks 12

- memory 12.0 MiB; 11879 KV B/token at 4128 prompt tokens

| check | result | detail |
|-------|--------|--------|
| output_matches_greedy | FAIL | 32 tokens |
| draft_proposed | PASS | 49 |
| acceptance_target | FAIL | {'acceptance': 0.08163265306122448, 'target': 0.5} |
| net_speedup_target | FAIL | {'speedup': 0.7698114393320515, 'target': 1.3} |

Verdict: **FAIL**.