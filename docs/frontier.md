# Compression vs 32k far accuracy

Measured resident bytes at 32k; logical assumes a bf16 memory slot is the only copy and
the W1-only anchor/replay buffers are removed. The global-attention ratio is the
sink+slots+window cache alone, not the resident footprint.

| slots | far C1 | far C3 | resident | no W1 buffers | bf16 core | logical | global attn | C3 MiB | core |
|-------|--------|--------|----------|---------------|-----------|---------|-------------|--------|------|
| 512 | 26.4% | 27.8% | 4.84x | 8.33x | 9.56x | 11.19x | 12.19x | 45.5 | float32 |
| 1024 | 26.4% | 40.3% | 3.52x | 6.00x | 7.38x | 9.53x | 10.24x | 69.5 | float32 |
| 2048 | 26.4% | 69.4% | 2.28x | 3.85x | 5.06x | 7.36x | 7.76x | 117.6 | float32 |
| 2048 | 26.4% | 70.8% | 5.05x | 5.06x | 5.06x | 7.36x | 7.76x | 24.2 | bfloat16 |

Each row is one measured resident point. The 2048 working point moved from 2.28x
(fp32, W1 buffers on) to 3.85x (buffers off) and 5.05x (bf16 core); the reduced
screens keep the 8/9 far accuracy. The 8x design target is the single-memory-copy
projection (logical / direct read), not the resident implementation: the memory is
currently stored twice, as associative matrices and as injected slot KV. A direct
associative read removes that duplicate; it is deferred (ADR-010).

![frontier](frontier.svg)
