# Paper 2 Replication Package

This package reproduces the Paper 2 analysis from the base AUTM panel and the
frozen Paper 1 policy-index panel. It contains the complete data chain,
cross-paper consistency checks, analysis code, and verified outputs.

## Package structure

### Core data chain

- `data/raw/merged_autm_base.csv`
- `data/external/paper1_policy_level_indices_institution_year.csv`
- `data/derived/merged_autm.csv`

### Replication pipeline

- `pipeline/01_build_merged_panel.py`
- `pipeline/02_validate_paper1_consistency.py`
- `pipeline/03_run_analysis.py`
- `pipeline/run_all.py`

### Matching documentation

- `pipeline/institution_aliases.json`
- `pipeline/institution_match_decisions.csv`
- `data/derived/institution_name_mapping.csv`
- `data/derived/merged_autm_match_audit.csv`

### Paper outputs

- figure: `paper_outputs/figures/figure1_event_study.png`
- full console output: `paper_outputs/logs/replication_output.txt`
- cross-paper validation: `validation/paper1_paper2_consistency.json`

## Current data counts

- Paper 2 panel rows: `3,507`
- Paper 2 institution names: `149`
- Paper 2 panel years: `1991-2023`
- rows with a Paper 1 NLP score: `2,564`
- rows without a Paper 1 NLP score: `943`
- Paper 2 institutions with at least one score: `139`
- frozen Paper 1 panel rows: `4,296`
- frozen Paper 1 institutions: `150`
- frozen Paper 1 panel years: `1925-2025`

The Paper 2 panel contains 253 AUTM unit identifiers. Several units map to the
same standardized institution name, which is why the unit count differs from
the 149 institution-name count.

## How to run

Python 3.11 is recommended. From `P2_replication_package/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline/run_all.py
```

`run_all.py` performs the following steps in order:

1. rebuilds `data/derived/merged_autm.csv` from the raw AUTM panel and frozen
   Paper 1 indices;
2. verifies all overlapping Paper 1-derived values;
3. runs the full Paper 2 analysis with 2,000 randomization-inference
   permutations;
4. refreshes the event-study figure and full analysis log.

Individual stages can also be run separately:

```powershell
python pipeline/01_build_merged_panel.py
python pipeline/02_validate_paper1_consistency.py
python pipeline/03_run_analysis.py
```

All scripts resolve paths relative to the package, so they can be launched
from any working directory.

## What each data file contains

### 1. Base AUTM panel

File:

- `data/raw/merged_autm_base.csv`

This is the pre-update Paper 2 panel containing AUTM technology-transfer
outcomes, institutional characteristics, royalty-sharing variables, and the
older policy-index merge. The build pipeline preserves its AUTM variables and
replaces all Paper 1-derived NLP fields with the frozen current Paper 1 values.

### 2. Frozen Paper 1 policy indices

File:

- `data/external/paper1_policy_level_indices_institution_year.csv`

Source:

- repository: `https://github.com/krish533/Tech-transfer-1`
- commit: `aa735a40a754f73496fffdaa1b0b771f606e713e`
- source path:
  `P1_replication_package/data/derived/policy_level_indices_institution_year.csv`
- SHA-256:
  `d0050621c013ccce9568812134424bf4eb79771648a0f0ae77d0892c91273cea`

Complete provenance is recorded in
`data/external/paper1_source_provenance.json`.

### 3. Final Paper 2 analysis panel

File:

- `data/derived/merged_autm.csv`

This is the canonical Paper 2 analysis file. Policy values are joined by
reviewed canonical institution name and calendar year. A policy observation is
carried forward according to the Paper 1 panel until the next observed policy
revision. Missing scores remain missing; the build does not reuse stale values
from the base panel.

Paper 1-derived fields include:

- `Mean_Tone_Score`
- `Median_Tone_Score`
- `Tone_Index`
- `Clarity_Index`
- `Legal_Load_Index`
- `n_sentences`
- `n_words`
- `Source_Year`
- `Is_Carried_Forward`

## Paper 1 / Paper 2 consistency check

The validator compares all nine Paper 1-derived fields on every overlapping
institution-year row.

| Check | Result |
|---|---:|
| Comparable Paper 2 rows | 2,564 |
| Paper 1 keys not found | 0 |
| Total field mismatches | 0 |
| Maximum absolute numeric difference | 0 |
| Duplicate institution-year keys | 0 |

The main NLP score (`Mean_Tone_Score`), median score, all three sub-indices,
sentence/word counts, source year, and carry-forward flag are therefore exactly
consistent with the frozen Paper 1 data.

Three Paper 2 institutions have no Paper 1 name match: Cornell University,
MIT, and Northwestern University. Seven additional institutions have a name
match but no overlapping AUTM/policy year. Full coverage details are in
`data/derived/institution_name_mapping.csv`.

The validator exits with a nonzero status if it finds a duplicate key, missing
upstream key, or value mismatch. It writes:

- `validation/paper1_paper2_consistency.json`
- `validation/paper1_paper2_mismatches.csv`

## Analysis definition

The primary specification uses:

- lagged `Mean_Tone_Score` as the Policy Communication Index;
- institution and year fixed effects;
- institution-clustered standard errors;
- lagged log research expenditures;
- lagged log licensing FTEs;
- lagged inventor royalty share;
- lagged TLO age.

The verified baseline estimate for log new patent applications is `0.524`
(`SE = 0.501`, `p = 0.296`, `N = 2,289`). The 2,000-permutation
randomization-inference p-value is `0.1105`.

The analysis script also produces descriptive statistics, seven outcome
models, lag checks, heterogeneity estimates, a Mundlak decomposition, event
studies, multiple-testing adjustments, leave-one-out estimates, and negative
binomial robustness results.
