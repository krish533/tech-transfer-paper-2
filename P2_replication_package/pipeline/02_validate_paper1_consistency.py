from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPARE_COLUMNS = [
    "Mean_Tone_Score",
    "Median_Tone_Score",
    "Tone_Index",
    "Clarity_Index",
    "Legal_Load_Index",
    "n_sentences",
    "n_words",
    "Source_Year",
    "Is_Carried_Forward",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Check Paper 2 NLP values against the frozen Paper 1 panel."
    )
    parser.add_argument(
        "--paper1",
        type=Path,
        default=package_root / "data" / "external" / "paper1_policy_level_indices_institution_year.csv",
    )
    parser.add_argument(
        "--paper2",
        type=Path,
        default=package_root / "data" / "derived" / "merged_autm.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=package_root / "validation" / "paper1_paper2_consistency.json",
    )
    parser.add_argument(
        "--mismatches",
        type=Path,
        default=package_root / "validation" / "paper1_paper2_mismatches.csv",
    )
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper1_path = args.paper1.resolve()
    paper2_path = args.paper2.resolve()
    for path in (paper1_path, paper2_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    paper1 = pd.read_csv(paper1_path, low_memory=False)
    paper2 = pd.read_csv(paper2_path, low_memory=False)
    required_p1 = {"Institution", "Year", *COMPARE_COLUMNS}
    required_p2 = {
        "Institution_std",
        "Institution_pci",
        "Year",
        *COMPARE_COLUMNS,
    }
    if missing := sorted(required_p1 - set(paper1.columns)):
        raise ValueError(f"Paper 1 data is missing columns: {missing}")
    if missing := sorted(required_p2 - set(paper2.columns)):
        raise ValueError(f"Paper 2 data is missing columns: {missing}")

    p1_duplicate_keys = int(paper1.duplicated(["Institution", "Year"]).sum())
    p2_duplicate_keys = int(paper2.duplicated(["Institution_std", "Year"]).sum())

    comparable = paper2.loc[
        paper2["Institution_pci"].notna(),
        ["Institution_std", "Institution_pci", "Year", *COMPARE_COLUMNS],
    ].merge(
        paper1[["Institution", "Year", *COMPARE_COLUMNS]],
        left_on=["Institution_pci", "Year"],
        right_on=["Institution", "Year"],
        how="left",
        suffixes=("_paper2", "_paper1"),
        validate="many_to_one",
    )

    unmatched_keys = int(comparable["Institution"].isna().sum())
    field_results: dict[str, dict[str, float | int]] = {}
    mismatch_frames = []
    total_mismatches = 0
    for column in COMPARE_COLUMNS:
        left = pd.to_numeric(comparable[f"{column}_paper2"], errors="coerce")
        right = pd.to_numeric(comparable[f"{column}_paper1"], errors="coerce")
        both = left.notna() & right.notna()
        mismatch = (left.isna() != right.isna()) | (
            both & ~np.isclose(left, right, rtol=0.0, atol=args.atol)
        )
        mismatch_count = int(mismatch.sum())
        total_mismatches += mismatch_count
        max_abs_diff = float((left[both] - right[both]).abs().max()) if both.any() else None
        field_results[column] = {
            "compared_rows": int(both.sum()),
            "mismatch_rows": mismatch_count,
            "max_absolute_difference": max_abs_diff,
        }
        if mismatch_count:
            details = comparable.loc[
                mismatch,
                ["Institution_std", "Institution_pci", "Year"],
            ].copy()
            details["field"] = column
            details["paper2_value"] = left.loc[mismatch].to_numpy()
            details["paper1_value"] = right.loc[mismatch].to_numpy()
            details["absolute_difference"] = (
                left.loc[mismatch] - right.loc[mismatch]
            ).abs().to_numpy()
            mismatch_frames.append(details)

    mismatch_columns = [
        "Institution_std",
        "Institution_pci",
        "Year",
        "field",
        "paper2_value",
        "paper1_value",
        "absolute_difference",
    ]
    mismatches = (
        pd.concat(mismatch_frames, ignore_index=True)
        if mismatch_frames
        else pd.DataFrame(columns=mismatch_columns)
    )
    args.mismatches.parent.mkdir(parents=True, exist_ok=True)
    mismatches.to_csv(args.mismatches, index=False)

    score_by_institution = paper2.groupby("Institution_std")["Mean_Tone_Score"].apply(
        lambda values: values.notna().any()
    )
    passed = (
        p1_duplicate_keys == 0
        and p2_duplicate_keys == 0
        and unmatched_keys == 0
        and total_mismatches == 0
    )
    report = {
        "passed": passed,
        "absolute_tolerance": args.atol,
        "paper1": {
            "file": paper1_path.name,
            "sha256": sha256(paper1_path),
            "rows": int(len(paper1)),
            "institutions": int(paper1["Institution"].nunique()),
            "year_min": int(paper1["Year"].min()),
            "year_max": int(paper1["Year"].max()),
            "duplicate_institution_year_keys": p1_duplicate_keys,
        },
        "paper2": {
            "file": paper2_path.name,
            "sha256": sha256(paper2_path),
            "rows": int(len(paper2)),
            "institutions": int(paper2["Institution_std"].nunique()),
            "year_min": int(paper2["Year"].min()),
            "year_max": int(paper2["Year"].max()),
            "duplicate_institution_year_keys": p2_duplicate_keys,
            "rows_with_nlp_score": int(paper2["Mean_Tone_Score"].notna().sum()),
            "rows_without_nlp_score": int(paper2["Mean_Tone_Score"].isna().sum()),
            "institutions_with_any_nlp_score": int(score_by_institution.sum()),
            "institutions_without_any_nlp_score": sorted(
                score_by_institution.loc[lambda values: ~values].index.tolist()
            ),
        },
        "comparison": {
            "comparable_rows": int(len(comparable)),
            "paper1_keys_not_found": unmatched_keys,
            "total_field_mismatches": total_mismatches,
            "fields": field_results,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"{'PASS' if passed else 'FAIL'}: {len(comparable):,} comparable rows; "
        f"{total_mismatches:,} field mismatches; {unmatched_keys:,} missing Paper 1 keys."
    )
    try:
        report_display = args.report.resolve().relative_to(
            Path(__file__).resolve().parents[1]
        ).as_posix()
    except ValueError:
        report_display = str(args.report)
    print(f"Report: {report_display}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
