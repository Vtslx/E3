# S1 prompt-lookup speculative decoding

Prompt: 2080 tokens; decode budget: 32; gamma=4, ngram=3.
Arm: `C0_full` (exact KV; the memory arm is not a speed lever).

| run | seconds | tokens/s |
|-----|---------|----------|
| greedy | 1.6048 | 19.9401 |
| speculative | 0.6846 | 46.7407 |

- speedup: **2.3441x**
- output identical to greedy: **True**
- proposed 26, accepted 26, full accepts 7, verify calls 7, fallbacks 0

| check | result | detail |
|-------|--------|--------|
| output_matches_greedy | PASS | 32 tokens |
| speedup | PASS | {'speedup': 2.3440547128258147, 'baseline_tps': 19.940117335004356, 'speculative_tps': 46.74072601341668} |
| draft_acceptance | PASS | 1.0 |

Verdict: **PASS**.