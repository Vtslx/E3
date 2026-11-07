# S2 speculative decoding on natural text / memory arm

Arm: `C0_full`; proposer: `prompt` (draft head `/tmp/sprm_draft/heads_wt2_r128.pt`); partial accept: False.
Prompt: 4096 tokens (sha256 `e7b99e892443`); decode budget: 32; gamma=4, ngram=3.

| run | seconds | tokens/s |
|-----|---------|----------|
| greedy | 1.4413 | 22.2023 |
| speculative | 3.9051 | 8.1944 |

- net speedup: **0.3691x** (target 1.3x)
- output identical to greedy: **True**
- acceptance: **0.0000** (target 0.5); full-accept rate 0.0000
- proposed 122, accepted 0, full 0, partial 0, verify 32, fallbacks 32

| check | result | detail |
|-------|--------|--------|
| output_matches_greedy | PASS | 32 tokens |
| draft_proposed | PASS | 122 |
| acceptance_target | FAIL | {'acceptance': 0.0, 'target': 0.5} |
| net_speedup_target | FAIL | {'speedup': 0.36908061301055944, 'target': 1.3} |

Verdict: **FAIL**.