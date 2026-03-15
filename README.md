## Tech Transfer Paper 2 Replication

This repository contains a stable script version of the original notebook replication pipeline for:

`Policy Communication and Technology Transfer: Evidence from University Intellectual Property Governance Documents`

### Run

From this directory:

```powershell
python replication.py | Tee-Object -FilePath replication_output.txt
```

The script will automatically look for either:

- `merged_autm.csv`
- `merged_autm (3).csv`

and will save the event-study figure to:

- `figure1_event_study.png`

### What was fixed

- Added a standalone `replication.py` entry point so the package no longer depends on notebook execution.
- Fixed numeric coercion for string-typed columns such as `Royalty Share`.
- Fixed the randomization-inference permutation step so it runs on current pandas versions.
- Replaced the hard-coded randomization-inference note in Table 4 with the computed value from the current run.
- Added automatic dataset path resolution for the repository's shipped CSV.

### Baseline analysis definition

The script preserves the paper's current baseline definition:

- panel entity: `[ID]` renamed to `institution_id`
- year fixed effects
- institution fixed effects
- lagged PCI treatment
- lagged log research expenditure and lagged log licensing FTE controls

No changes were made to the paper's baseline regression specification.
