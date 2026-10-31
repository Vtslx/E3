# Claims for D1

Establishes separately: (1) whether the memory slots improve far-context retrieval
(`C3 - C1`, memory effect); (2) whether the W1 writeback adds anything on top of the
memory (`C4 - C3`, writeback effect). `C4 - C1` is reported only as a combined
diagnostic and never gates. Exact KV (`C0`) is the ceiling; `C1` is the no-memory
control.

Does not establish: distribution equivalence, real-benchmark generalization, proposer
or speculative-decoding benefits, Form N quality, or wall-clock speedup. 'Far' is the
configured context-length bucket, not a real-world distance distribution.
