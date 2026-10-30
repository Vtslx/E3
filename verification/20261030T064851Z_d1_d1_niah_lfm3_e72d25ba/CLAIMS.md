# Claims for D1

Establishes: on this synthetic passkey retrieval task with candidate-likelihood ranking,
with this checkpoint and budget, whether the memory arms improve far-context retrieval over sink+window, and whether W1
writeback changes that result. Exact-KV (C0) is the ceiling; StreamingLLM (C1) is the
no-memory control; C3 isolates W2; C4 adds W1.

Does not establish: distribution equivalence, gains on real benchmarks (LongBench/RULER/etc.),
proposer or speculative-decoding benefits, Form N quality, or wall-clock speedup. 'Far' means
the configured context-length bucket at or above 32768 tokens, not a real-world
distance distribution.
