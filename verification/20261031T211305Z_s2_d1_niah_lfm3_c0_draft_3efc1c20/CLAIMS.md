# Claims for S2

Establishes: on the stated prompt and arm, the prompt-lookup/reservoir draft's
token acceptance, the speculative decode's net tokens/s against a greedy baseline
without diagnostics, and the first position where the committed sequence differs
from a separate single-token greedy run. The committed tokens are the target's own
verifier argmaxes; the chunked verify and the single-token forward round differently
in bf16, so bitwise identity is not guaranteed on natural text after a full accept.

Does not establish: general-task acceptance, a trained draft model, tree verification,
batch>1 throughput, or that the memory compression itself accelerates generation.
