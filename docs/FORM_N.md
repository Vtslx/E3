# Form N: compressed-prefix scaffold

Form N prepends a fixed number of soft tokens per evicted block to the frozen model input.
This round implements the **contract, budget planner, and byte accounting only**. There is no
compressor checkpoint, no training loop, and no prefix injection. Nothing here is verified for
quality.

## What exists

- `CompressedPrefixConfig`: validates the soft-token budget (`1..512`), layer count, KV width,
  and dtype size.
- `plan_prefix`: distributes the budget over evicted block ranges with the same allocator as
  Form A. Uniform by default; importance-based allocation is available when importances are
  supplied.
- `CompressedPrefixMemory`: satisfies the memory interface, but every data-dependent method
  raises `UnsupportedOperation`. It never fabricates a read or a reconstruction error.
- `build_prefix_scaffold`: computes the evicted ranges for one prompt and returns the plan
  plus the KV bytes the soft tokens would occupy.
- `sprm form-n build`: prints the plan and explicitly labels it `planned-not-verified`.

## What does not exist

- A trained compressor (4-layer Transformer or cross-attention encoder).
- `L_LM + L_MTP + L_rec + L_align` training code.
- Prefix-token injection into the model input, or prefix-cache management.
- Any quality, stability, or speed measurement.

The free-validation probe is unavailable by construction: Form N has no exact window
reconstruction to compare against. Its validation would require a periodic dual forward
(compressed prefix vs exact context) and a KL or NLL delta. That is not implemented, and the
class raises rather than returning a number.

## Budget

The recommended total soft-token budget is at most 512, matching the design. At
`num_slots=512`, 24 layers, KV width 512, and bf16, the planned soft tokens occupy:

```
512 * 24 * 2 * 512 * 2 bytes = 25,165,824 bytes = 24 MiB
```

`memory_bytes()` computes this from the plan, not from a real allocation.

## Failure modes to test once a compressor exists

From DESIGN.md §15.6:

- position aliasing (all blocks share fixed early positions);
- weak exact recall (needle-in-a-haystack) versus associative memory;
- coarse correction (whole-block re-encoding only);
- per-model compressor retraining;
- attention dilution from too many soft tokens;
- prefix-cache invalidation if old slots are rewritten (append-only correction is required);
- trust calibration: a frozen model may ignore or over-trust the prefix.

## Next steps

1. Train a small compressor on the frozen model with the multi-token-prediction objective.
2. Add append-only prefix-token injection and per-block KV caching.
3. Implement the periodic dual-forward validation probe.
4. Run a Form N variant of D1 with the same task, arms, and candidate metric, so the two
   forms are comparable.
