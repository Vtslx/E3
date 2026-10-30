# V0 backend contract

- Model: `/Users/yikewang/Model/LFM3/LFM3-1.7B`
- Layers: 24 (6 global, 5 sliding, 13 conv)
- KV: 4 heads x 128, dtype bfloat16

| check | result | detail |
|-------|--------|--------|
| layer_partition | PASS | global=[2, 6, 10, 14, 18, 22] sliding=[4, 8, 12, 16, 20] conv=13 |
| kv_width | PASS | 512 |
| prefill_length | PASS | 224 |
| kv_shape | PASS | 224 tokens, 4x128 |
| kv_dtype | PASS | torch.bfloat16 |
| logits_finite | PASS | (128000,) |
| cache_invariants | PASS |  |
| rope_roundtrip | PASS | 0.03125 |
| prefill_determinism | PASS | 0.0 |
| decode_finite | PASS | (128000,) |

Verdict: **PASS**.