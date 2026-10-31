# Claims for V0

Establishes: the checkpoint loads through the declared backend, its layer plan partitions
the model, exact KV has the contracted shape and dtype, RoPE is invertible within bf16
tolerance, prefilling is repeatable, and a decode step produces finite logits.

Does not establish: retrieval quality, memory behaviour, or any speedup.
