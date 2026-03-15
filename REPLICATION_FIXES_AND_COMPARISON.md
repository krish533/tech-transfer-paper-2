# Replication Fixes and Comparison

## Files added or updated

- `replication.py`: standalone script version of the notebook pipeline
- `README.md`: run instructions and baseline analysis definition
- `replication_output.txt`: full rerun log from the fixed script

## Changes made

1. Added a stable script entry point so replication no longer depends on notebook execution.
2. Added automatic data-path resolution for either `merged_autm.csv` or `merged_autm (3).csv`.
3. Fixed numeric coercion for pandas string-typed columns, including `Royalty Share`.
4. Fixed the randomization-inference permutation step so it runs on current pandas versions.
5. Replaced the hard-coded RI note in Table 4 with the value computed in the current run.
6. Kept the paper's baseline specification unchanged: panel entity remains `[ID]` renamed to `institution_id`.

## Full rerun results

The fixed script completed successfully and produced the following core results:

- `Ln(New Patent Applications)`: `b = 1.236`, `SE = 0.555`, `p = 0.027`
- Event-study pre-trend for patent applications: `chi2(3) = 2.53`, `p = 0.469`
- Randomization inference: `p = 0.0015`, which rounds to `0.002`
- Conversion rate: `b = 1.481`, `SE = 0.572`, `p = 0.010`
- Mundlak within estimate for patent applications: `b = 1.367`, `p = 0.012`

## Comparison with manuscript text

The rerun matches the current manuscript on all headline numbers checked:

- Abstract main effect: matches `b = 1.236`, `p = 0.027`, RI `p = 0.002` after rounding
- Event-study pre-trend: matches `chi2(3) = 2.53`, `p = 0.469`
- Main Table 2 coefficients: match
- Mundlak decomposition values reported in the text: match
- Table 4 channel-analysis values: match
- Table 1 `Royalty Share` summary now matches the manuscript after the coercion fix

## Remaining note

The script intentionally preserves the manuscript's current panel-entity definition based on `[ID]`. This reproduces the paper's baseline results, but users should be aware that `[ID]` is not the same as `Institution_std` in the raw CSV. That is a data-design issue rather than a runtime bug, and it was not changed here so that the package continues to replicate the paper as written.
