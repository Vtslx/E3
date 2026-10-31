# V2 cache-surgery exactness

Both paths prefill the prompt body (all but the final token). The reference feeds the final
token through the untouched body cache; the rebuilt path copies the full global KV set and
feeds the same token again. With no eviction the two must be identical. The gap between a
full prefill and a one-token decode is reported separately: it is a kernel/bf16 difference,
not a cache-surgery defect.

| check | result | detail |
|-------|--------|--------|
| eviction_slices_byte_identical | PASS | length=2176 max_tensor_diff=0.0 |
| online_stream_structure | PASS | length=2176 sink_diff=0.0 |
| online_memory_structure | PASS | budget=512 length=2688 sink_diff=0.0 |
| no_eviction_kv_byte_identical | PASS | 0.0 |
| logit_difference | PASS | {'max': 0.0, 'case_max': [0.0, 0.0, 0.0, 0.0], 'limit': 0.05} |
| top1_agreement | PASS | 4/4 |

Worst rebuild-vs-reference logit difference: 0.0000 (limit 0.05).
Full-prefill vs decode diagnostic gap: 0.1875.
Top-1 agreement: 4/4.

Verdict: **PASS**.