# Claims for V2

Establishes: rebuilding the global cache from the same body prefill reproduces the exact KV
bytes and the next-token distribution within the configured tolerance; when eviction occurs,
the retained sink and window slices are byte-identical to the exact cache. This validates the
surgery, not the memory.

Does not establish: that compressed memory preserves the distribution, or that dropped tokens
are recoverable.
