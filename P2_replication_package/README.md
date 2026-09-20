# Paper 2 Replication Package

This package contains the final analysis dataset, replication code, and
verified outputs for Paper 2.

## Package structure

- `data/merged_autm.csv`: final analysis dataset
- `code/replication.py`: complete statistical analysis
- `code/run_all.py`: one-command replication runner
- `paper_outputs/figures/figure1_event_study.png`: event-study figure
- `paper_outputs/logs/replication_output.txt`: complete verified results
- `requirements.txt`: Python dependencies

## Run the replication

Python 3.11 is recommended. From `P2_replication_package/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code/run_all.py
```

The runner reads the final dataset, executes all analyses, refreshes the
event-study figure, and writes the complete console output to
`paper_outputs/logs/replication_output.txt`.

The analysis can also be run directly:

```powershell
python code/replication.py
```

All paths are resolved relative to the package, so the scripts can be launched
from any working directory.

## Final dataset

`data/merged_autm.csv` is the only dataset required for replication.

- rows: `3,507`
- standardized institution names: `149`
- AUTM unit identifiers: `253`
- years: `1991-2023`
- rows with an NLP score: `2,564`
- SHA-256: `d0776a261103f828bbdf59af301fd871f460c82781d621f8217f54fb36dc4a86`

The Paper 1-derived NLP fields in the final dataset were previously checked
against the Paper 1 institution-year panel. All 2,564 comparable rows matched
exactly, including `Mean_Tone_Score`, the median score, three sub-indices,
sentence and word counts, source year, and carry-forward status.

## Analysis

The primary specification uses lagged `Mean_Tone_Score` as the Policy
Communication Index, institution and year fixed effects, institution-clustered
standard errors, and lagged institutional controls.

The verified baseline estimate for log new patent applications is `0.524`
(`SE = 0.501`, `p = 0.296`, `N = 2,289`). The 2,000-permutation
randomization-inference p-value is `0.1105`.

The code also reproduces descriptive statistics, alternative outcomes, lag
checks, heterogeneity estimates, a Mundlak decomposition, the event study,
multiple-testing adjustments, leave-one-out estimates, and count-model
robustness results.
