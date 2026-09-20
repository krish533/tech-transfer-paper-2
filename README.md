# Technology Transfer Paper 2 Replication Package

This repository contains the data and code used to reproduce the Paper 2
analysis. The canonical analysis file is `merged_autm.csv`.

## Data release status

`merged_autm.csv` combines the AUTM panel with the current Paper 1 policy
indices. The packaged Paper 1 snapshot is taken from
[`krish533/Tech-transfer-1`](https://github.com/krish533/Tech-transfer-1) at
commit `aa735a40a754f73496fffdaa1b0b771f606e713e` (2026-04-09).

The upstream file and the frozen copy in this repository have SHA-256:

```text
d0050621c013ccce9568812134424bf4eb79771648a0f0ae77d0892c91273cea
```

Detailed provenance is recorded in `data/paper1_source_provenance.json`.

## Package contents

- `merged_autm.csv`: current Paper 2 analysis panel (3,507 institution-years,
  149 institutions, 1991-2023)
- `merged_autm (3).csv`: legacy AUTM merge used as the non-NLP base input
- `data/paper1_policy_level_indices_institution_year.csv`: frozen Paper 1
  institution-year policy panel (4,296 rows, 150 institutions, 1925-2025)
- `build_updated_merged_autm.py`: deterministic merge and audit pipeline
- `replication.py`: tables, robustness analyses, randomization inference, and
  event-study figure
- `validate_paper1_consistency.py`: cross-repository value-level validation
- `institution_name_mapping.csv`: reviewed Paper 2 to Paper 1 name mapping
- `merged_autm_match_audit.csv`: institution-level match coverage
- `merged_autm_update_audit.json`: input/output hashes and merge counts
- `validation/`: machine-readable cross-paper consistency results
- `name_match_artifacts/`: reviewed non-exact name matches and decisions
- `outputs/replication_output.txt`: verified full-run console output
- `figure1_event_study.png`: verified event-study figure

## Paper 1 / Paper 2 consistency result

The validation compares every Paper 1-derived field on all overlapping
institution-year rows in Paper 2.

| Check | Result |
|---|---:|
| Comparable Paper 2 rows | 2,564 |
| Paper 1 keys not found | 0 |
| `Mean_Tone_Score` mismatches | 0 |
| `Median_Tone_Score` mismatches | 0 |
| `Tone_Index` mismatches | 0 |
| `Clarity_Index` mismatches | 0 |
| `Legal_Load_Index` mismatches | 0 |
| Sentence/word count mismatches | 0 |
| Source-year/carry-forward mismatches | 0 |
| Maximum absolute numeric difference | 0 |

Paper 2 has 2,564 rows with an NLP score and 943 rows without one. Missing
scores are retained as missing rather than filled with a stale value. In total,
139 of the 149 Paper 2 institutions have at least one matched score. Three
institutions have no Paper 1 name match (Cornell, MIT, and Northwestern), and
seven additional institutions have a Paper 1 name match but no overlapping
AUTM/policy year. See `institution_name_mapping.csv` for the full audit.

## Reproduction

Python 3.11 is recommended. From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

1. Rebuild the current merged panel and merge audits:

```powershell
python build_updated_merged_autm.py
```

2. Verify all Paper 1-derived values:

```powershell
python validate_paper1_consistency.py
```

The validator exits nonzero if it finds a duplicate key, missing upstream key,
or value mismatch. It writes:

- `validation/paper1_paper2_consistency.json`
- `validation/paper1_paper2_mismatches.csv`

3. Reproduce the Paper 2 estimates and event-study figure:

```powershell
python replication.py | Tee-Object -FilePath outputs/replication_output.txt
```

The main script uses institution and year fixed effects, institution-clustered
standard errors, and 2,000 within-institution randomization-inference
permutations (seed 42). It writes `figure1_event_study.png` and prints all
tables to the console/output file.

## Merge rules

Policy values are merged by canonical institution name and calendar year.
Exact normalized names are used where possible; reviewed aliases cover known
formatting and campus-name differences. Each observed policy value begins in
its source year and is carried forward by Paper 1 until the next observed
revision. The build fails on duplicate keys, normalized-name collisions, or a
row-changing merge.

The current baseline specification uses lagged `Mean_Tone_Score` (PCI), lagged
log research expenditures, lagged log licensing FTEs, lagged inventor royalty
share, and lagged TLO age.
