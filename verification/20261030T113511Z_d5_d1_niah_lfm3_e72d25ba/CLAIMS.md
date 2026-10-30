# Claims for D5

Establishes: over a long W1 decode with the implemented trust region and slot refresh,
whether the key-matrix condition stays below the configured bound and whether the exact
window reconstruction error shrinks rather than grows. `d5_min_eta_ratio` is used as the
minimum head/tail shrinkage ratio because eta adaptation is not implemented.

Does not establish: anchor/replay stability, adaptive-eta behaviour, distribution
equivalence, or long-text task accuracy.
