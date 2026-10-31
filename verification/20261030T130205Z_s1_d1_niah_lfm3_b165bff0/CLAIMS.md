# Claims for S1

Establishes: on a repeated-sentence prompt, prompt-lookup drafts with the stated
acceptance, the speculative output is token-identical to greedy decoding, and the
measured tokens/s is compared on the same exact-KV arm. The all-or-nothing commit
rolls back global/sliding KV and restores conv state on rejection.

Does not establish: general-task acceptance, memory-arm speedup, batch>1 throughput,
or a claim that compression itself accelerates generation.
