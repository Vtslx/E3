# Claims for D5

Establishes: over a long W1 decode with the implemented trust region, anchors, bounded
replay, eta adaptation, and slot refresh, whether (1) the key-matrix condition stays
below the configured bound and (2) the generated-token reconstruction error shrinks.
The full-window shrinkage is reported separately as a diagnostic because the prefill
window is Tier-1 exact and was never W1-written.

Does not establish: distribution equivalence, long-text task accuracy, or a learned
memory representation.
