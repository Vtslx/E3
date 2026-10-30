# V1 contract gate

Runs without the model. It checks the pieces D1 depends on: slot budgets, closed-form
writes, reconstruction improvement, trust region, forgetting, KV typing, the exact-window
position alignment, archive sealing, and the McNemar statistic.

| check | result | detail |
|-------|--------|--------|
| allocator_budget | PASS | [{'blocks': 4, 'slots': 32, 'mode': 'uniform', 'allocation': [8, 8, 8, 8], 'ok': True}, {'blocks': 4, 'slots': 32, 'mode': 'effective_rank', 'allocation': [2, 4, 9, 17], 'ok': True}, {'blocks': 1, 'slots': 8, 'mode': 'un |
| memory_w2_reduces_error | PASS | {'recon_before': 1.0, 'recon_after': 0.010111900046467781, 'recon_improved': True, 'condition': 36.9675816100685, 'condition_finite': True, 'forgetting_ratio': 0.5, 'forgetting_halved': True, 'trust_scaled': True} |
| memory_condition_finite | PASS | 36.9675816100685 |
| memory_forgetting_halves_values | PASS | 0.5 |
| memory_trust_region_scales | PASS |  |
| kvslice_invariants | PASS | ok=True mismatch_rejected=True |
| exact_window_position_alignment | PASS | [0, 1, 2] |
| archive_seal_and_tamper_detection | PASS | clean=0 tampered=2 |
| mcnemar_known_value | PASS | 0.03125 |

Verdict: **PASS**.