from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


NLP_COLUMNS = [
    "Mean_Tone_Score",
    "Median_Tone_Score",
    "Tone_Index",
    "Clarity_Index",
    "Legal_Load_Index",
    "n_sentences",
    "n_words",
]

MANUAL_ALIAS_MAP = {
    "Miami University (Ohio)": "Miami University",
    "University of Texas At Austin": "University of Texas at Austin",
    "Virginia Commonwealth University (VCU)": "Virginia Commonwealth University",
    "Washington State University (WSU)": "Washington State University",
    "University of Wisconsin Madison": "University of Wisconsin-Madison",
    "University of Oklahoma (All Campuses)": "University of Oklahoma",
    "University of Maryland, Baltimore County (UMBC)": "University of Maryland Baltimore County",
    "University of Massachusetts Medical Center (UMass Chan Medical School)": "University of Massachusetts Medical Center",
}


def norm_name(value: str) -> str:
    text = str(value).lower().replace("&", "and")
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Merge the frozen Paper 1 policy indices into the AUTM panel."
    )
    parser.add_argument(
        "--autm-input",
        type=Path,
        default=repo_dir / "merged_autm (3).csv",
        help="Base AUTM panel (default: merged_autm (3).csv).",
    )
    parser.add_argument(
        "--paper1-input",
        type=Path,
        default=repo_dir / "data" / "paper1_policy_level_indices_institution_year.csv",
        help="Frozen Paper 1 institution-year indices included in this package.",
    )
    parser.add_argument(
        "--alias-map",
        type=Path,
        default=repo_dir / "name_match_artifacts" / "paper2_pci_name_alias_map.json",
        help="Reviewed aliases for institution names that do not normalize exactly.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_dir / "merged_autm.csv",
        help="Updated Paper 2 analysis panel to create.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    autm_path = args.autm_input.resolve()
    pci_path = args.paper1_input.resolve()
    alias_path = args.alias_map.resolve()
    output_path = args.output.resolve()

    for path in (autm_path, pci_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    autm = pd.read_csv(autm_path, low_memory=False)
    pci = pd.read_csv(pci_path, low_memory=False)
    reviewed_aliases = (
        json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.is_file() else {}
    )
    alias_map = {**reviewed_aliases, **MANUAL_ALIAS_MAP}

    required_autm = {"Institution_std", "Year"}
    required_pci = {
        "Institution",
        "Year",
        "Source_Year",
        "Is_Carried_Forward",
        *NLP_COLUMNS,
    }
    if missing := sorted(required_autm - set(autm.columns)):
        raise ValueError(f"AUTM input is missing columns: {missing}")
    if missing := sorted(required_pci - set(pci.columns)):
        raise ValueError(f"Paper 1 input is missing columns: {missing}")
    if autm.duplicated(["Institution_std", "Year"]).any():
        raise ValueError("AUTM input has duplicate Institution_std-Year keys")
    if pci.duplicated(["Institution", "Year"]).any():
        raise ValueError("Paper 1 input has duplicate Institution-Year keys")

    pci["pci_norm_key"] = pci["Institution"].map(norm_name)
    norm_name_counts = pci.groupby("pci_norm_key")["Institution"].nunique()
    collisions = norm_name_counts[norm_name_counts > 1]
    if not collisions.empty:
        raise ValueError(
            "Paper 1 institution names collide after normalization: "
            + ", ".join(collisions.index.tolist())
        )

    canonical_by_norm = (
        pci[["pci_norm_key", "Institution"]]
        .drop_duplicates()
        .set_index("pci_norm_key")["Institution"]
        .to_dict()
    )
    autm_names = sorted(autm["Institution_std"].dropna().unique().tolist())

    mapping_rows = []
    resolved_names: dict[str, str | None] = {}
    for autm_name in autm_names:
        proposed_name = alias_map.get(autm_name, autm_name)
        canonical_name = canonical_by_norm.get(norm_name(proposed_name))
        resolved_names[autm_name] = canonical_name
        if canonical_name is None:
            source = "no_match"
        elif autm_name in MANUAL_ALIAS_MAP:
            source = "manual_alias"
        elif autm_name in reviewed_aliases:
            source = "reviewed_alias"
        else:
            source = "exact_normalized"
        mapping_rows.append(
            {
                "autm_institution": autm_name,
                "pci_match_name": canonical_name or "",
                "match_source": source,
                "autm_norm_key": norm_name(autm_name),
                "pci_norm_key": norm_name(canonical_name) if canonical_name else "",
            }
        )

    autm["pci_match_name"] = autm["Institution_std"].map(resolved_names)
    pci_cols = [
        "Institution",
        "Year",
        "Source_Year",
        "Is_Carried_Forward",
        *NLP_COLUMNS,
    ]
    merged = autm.merge(
        pci[pci_cols],
        left_on=["pci_match_name", "Year"],
        right_on=["Institution", "Year"],
        how="left",
        suffixes=("", "_pci"),
        validate="many_to_one",
    )
    if len(merged) != len(autm):
        raise RuntimeError("Merge changed the number of AUTM rows")

    for column in NLP_COLUMNS:
        merged[column] = merged[f"{column}_pci"]
        merged.drop(columns=[f"{column}_pci"], inplace=True)

    autm_years = (
        autm.groupby("Institution_std")
        .agg(
            autm_year_min=("Year", "min"),
            autm_year_max=("Year", "max"),
            autm_rows=("Year", "count"),
        )
        .reset_index()
    )
    pci_years = (
        pci.groupby("Institution")
        .agg(
            pci_year_min=("Year", "min"),
            pci_year_max=("Year", "max"),
            pci_rows=("Year", "count"),
        )
        .reset_index()
    )
    mapping_df = pd.DataFrame(mapping_rows).merge(
        autm_years,
        left_on="autm_institution",
        right_on="Institution_std",
        how="left",
    )
    mapping_df = mapping_df.merge(
        pci_years,
        left_on="pci_match_name",
        right_on="Institution",
        how="left",
    )
    mapping_df["has_year_overlap"] = (
        mapping_df["pci_year_min"].notna()
        & (
            mapping_df[["autm_year_min", "pci_year_min"]].max(axis=1)
            <= mapping_df[["autm_year_max", "pci_year_max"]].min(axis=1)
        )
    )
    mapping_df.drop(columns=["Institution_std", "Institution"], inplace=True)
    mapping_df.sort_values(
        ["match_source", "autm_institution"], inplace=True, ignore_index=True
    )

    match_audit = (
        merged.assign(has_pci=merged["Mean_Tone_Score"].notna())
        .groupby(["Institution_std", "pci_match_name"], dropna=False)
        .agg(
            autm_years=("Year", "count"),
            matched_years=("has_pci", "sum"),
            first_autm_year=("Year", "min"),
            last_autm_year=("Year", "max"),
            first_pci_source_year=("Source_Year", "min"),
            last_pci_source_year=("Source_Year", "max"),
        )
        .reset_index()
        .sort_values(["matched_years", "Institution_std"], ascending=[False, True])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(merged.to_csv(index=False), encoding="utf-8", newline="")

    audit = {
        "autm_input": autm_path.name,
        "autm_input_sha256": sha256(autm_path),
        "paper1_input": (
            str(pci_path.relative_to(repo_dir))
            if pci_path.is_relative_to(repo_dir)
            else str(pci_path)
        ),
        "paper1_input_sha256": sha256(pci_path),
        "output": output_path.name,
        "output_sha256": sha256(output_path),
        "autm_rows": int(len(autm)),
        "autm_institutions": int(autm["Institution_std"].nunique()),
        "pci_panel_rows": int(len(pci)),
        "pci_panel_institutions": int(pci["Institution"].nunique()),
        "matched_rows": int(merged["Mean_Tone_Score"].notna().sum()),
        "matched_institutions_any_year": int(
            merged.groupby("Institution_std")["Mean_Tone_Score"]
            .apply(lambda values: values.notna().any())
            .sum()
        ),
        "unmatched_institutions_no_year_overlap": sorted(
            merged.groupby("Institution_std")["Mean_Tone_Score"]
            .apply(lambda values: values.notna().any())
            .loc[lambda values: ~values]
            .index.tolist()
        ),
    }

    audit_path = repo_dir / "merged_autm_update_audit.json"
    match_path = repo_dir / "merged_autm_match_audit.csv"
    mapping_csv_path = repo_dir / "institution_name_mapping.csv"
    mapping_json_path = repo_dir / "institution_name_mapping.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    match_audit.to_csv(match_path, index=False, encoding="utf-8-sig")
    mapping_df.to_csv(mapping_csv_path, index=False, encoding="utf-8-sig")
    mapping_json_path.write_text(
        json.dumps(mapping_df.to_dict(orient="records"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
