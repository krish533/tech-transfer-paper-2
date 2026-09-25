from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
P2_DATA = PACKAGE_ROOT / "data" / "merged_autm.csv"
OUT_DATA = PACKAGE_ROOT / "data" / "verified_replication_dataset.csv"
OUT_REPORT = PACKAGE_ROOT / "paper_outputs" / "logs" / "cross_repo_verification.md"
OUT_JSON = PACKAGE_ROOT / "paper_outputs" / "logs" / "cross_repo_verification.json"
P1_COMMIT = "25a9472b34334825b6d6c6a334f5b88eb00695b5"
P2_BASE_COMMIT = "88cf4a1c02540b136adb8beaa35e212625ac755e"
P1_URL = (
    "https://raw.githubusercontent.com/krish533/Tech-transfer-1/"
    f"{P1_COMMIT}/P1_replication_package/data/derived/"
    "policy_level_indices_institution_year.csv"
)
EXPECTED_P2_SHA256 = "d0776a261103f828bbdf59af301fd871f460c82781d621f8217f54fb36dc4a86"
EXPECTED_CROSS_MATCH = 2564
EXPECTED_BASELINE = {
    "beta": 0.6159,
    "se": 0.523,
    "p": 0.241,
    "n": 2115,
}
P1_COMPARE_COLUMNS = [
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )


def import_replication_module():
    path = PACKAGE_ROOT / "code" / "replication.py"
    spec = importlib.util.spec_from_file_location("paper2_replication", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_p2_institution_column(p2: pd.DataFrame, p1: pd.DataFrame) -> tuple[str, dict[str, int]]:
    if "Year" not in p2.columns:
        raise KeyError("Paper 2 data has no Year column")

    explicit = [
        "Institution",
        "Institution_std",
        "Standardized_Institution",
        "Institution_Standardized",
        "University",
        "University Name",
        "Institution Name",
    ]
    candidates = [c for c in explicit if c in p2.columns]
    candidates += [
        c for c in p2.columns
        if c not in candidates
        and ("institution" in c.casefold() or "university" in c.casefold())
        and c != "[ID]"
    ]
    if not candidates:
        raise KeyError("Could not find a plausible standardized institution-name column in Paper 2 data")

    p1_keys = p1[["Institution", "Year"]].copy()
    p1_keys["_inst_norm"] = normalize_name(p1_keys["Institution"])
    p1_keys["Year"] = pd.to_numeric(p1_keys["Year"], errors="coerce")
    p1_keys = p1_keys[["_inst_norm", "Year"]].drop_duplicates()

    scored = {}
    pci_mask = p2["Mean_Tone_Score"].notna() if "Mean_Tone_Score" in p2.columns else pd.Series(True, index=p2.index)
    for c in candidates:
        sub = p2.loc[pci_mask, [c, "Year"]].copy()
        sub["_inst_norm"] = normalize_name(sub[c])
        sub["Year"] = pd.to_numeric(sub["Year"], errors="coerce")
        merged = sub.merge(p1_keys, on=["_inst_norm", "Year"], how="left", indicator=True)
        scored[c] = int((merged["_merge"] == "both").sum())

    best = max(scored, key=scored.get)
    return best, scored


def compare_cross_repo(p2: pd.DataFrame, p1: pd.DataFrame, inst_col: str):
    p2x = p2.copy()
    p1x = p1.copy()
    p2x["_inst_norm"] = normalize_name(p2x[inst_col])
    p1x["_inst_norm"] = normalize_name(p1x["Institution"])
    p2x["Year"] = pd.to_numeric(p2x["Year"], errors="coerce")
    p1x["Year"] = pd.to_numeric(p1x["Year"], errors="coerce")

    p1_keep = ["_inst_norm", "Year", "Institution"] + [c for c in P1_COMPARE_COLUMNS if c in p1x.columns]
    p1x = p1x[p1_keep].drop_duplicates(subset=["_inst_norm", "Year"], keep="last")

    pci_mask = p2x["Mean_Tone_Score"].notna()
    left = p2x.loc[pci_mask].copy()
    merged = left.merge(
        p1x,
        on=["_inst_norm", "Year"],
        how="left",
        suffixes=("_p2", "_p1"),
        indicator=True,
        validate="many_to_one",
    )
    matched = merged["_merge"] == "both"
    matched_n = int(matched.sum())

    mismatches: dict[str, int] = {}
    compared: list[str] = []
    missing_columns: list[str] = []
    for c in P1_COMPARE_COLUMNS:
        p2c = f"{c}_p2" if f"{c}_p2" in merged.columns else c if c in merged.columns and c in p2x.columns else None
        p1c = f"{c}_p1" if f"{c}_p1" in merged.columns else None
        if p2c is None or p1c is None:
            missing_columns.append(c)
            continue
        compared.append(c)
        a = merged.loc[matched, p2c]
        b = merged.loc[matched, p1c]
        an = pd.to_numeric(a, errors="coerce")
        bn = pd.to_numeric(b, errors="coerce")
        numeric_fraction = max(an.notna().mean(), bn.notna().mean())
        if numeric_fraction > 0.95:
            both_na = an.isna() & bn.isna()
            equal = np.isclose(an.fillna(0).to_numpy(float), bn.fillna(0).to_numpy(float), rtol=1e-10, atol=1e-12)
            equal = equal | both_na.to_numpy()
        else:
            aa = a.astype("string").fillna("<NA>")
            bb = b.astype("string").fillna("<NA>")
            equal = (aa == bb).to_numpy()
        mismatches[c] = int((~equal).sum())

    return merged, matched_n, compared, missing_columns, mismatches


def build_verified_dataset(rep, raw: pd.DataFrame, inst_col: str, cross_merged: pd.DataFrame) -> pd.DataFrame:
    clean = rep.load_and_prepare(P2_DATA).copy()
    if inst_col in clean.columns:
        clean["institution_standardized"] = clean[inst_col].astype("string")
    else:
        clean["institution_standardized"] = raw[inst_col].astype("string").to_numpy()

    keyset = set(
        zip(
            cross_merged.loc[cross_merged["_merge"] == "both", "_inst_norm"].astype(str),
            cross_merged.loc[cross_merged["_merge"] == "both", "Year"].astype(float),
        )
    )
    norm = normalize_name(clean["institution_standardized"]).astype(str)
    years = pd.to_numeric(clean["year"], errors="coerce").astype(float)
    clean["paper1_key_match_verified"] = [(n, y) in keyset for n, y in zip(norm, years)]
    clean["paper1_commit"] = P1_COMMIT
    clean["paper2_base_commit"] = P2_BASE_COMMIT

    ordered = [
        "institution_id", "institution_standardized", "year",
        "paper1_key_match_verified", "paper1_commit", "paper2_base_commit",
        "pci", "pci_median", "tone_index", "clarity_index", "legal_load_index",
        "Source_Year", "Is_Carried_Forward", "n_sentences", "n_words",
        "new_patent_apps", "total_patent_apps", "patents_issued", "disclosures",
        "licenses", "startups", "license_income",
        "research_exp", "licensing_ftes", "royalty_share", "tlo_age",
        "private", "carnegie_r1", "med_school", "land_grant",
        "ln_new_patent_apps", "ln_total_patent_apps", "ln_patents_issued",
        "ln_disclosures", "ln_licenses", "ln_startups", "ln_license_income",
        "conv_rate", "pci_l1", "pci_l2", "pci_l3", "pci_l4", "pci_l5",
        "ln_research_exp_l1", "ln_licensing_ftes_l1", "royalty_share_l1",
    ]
    cols = [c for c in ordered if c in clean.columns]
    return clean[cols].copy()


def main() -> None:
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    p2_sha = sha256(P2_DATA)
    if p2_sha != EXPECTED_P2_SHA256:
        raise AssertionError(f"Paper 2 merged data SHA-256 changed: {p2_sha}")

    p1_path = PACKAGE_ROOT / "data" / "_paper1_policy_panel_for_verification.csv"
    urllib.request.urlretrieve(P1_URL, p1_path)
    p1_sha = sha256(p1_path)

    p1 = pd.read_csv(p1_path, low_memory=False)
    p2 = pd.read_csv(P2_DATA, low_memory=False)

    p1_rows = len(p1)
    p1_inst = p1["Institution"].nunique()
    p1_primary = p1[pd.to_numeric(p1["Year"], errors="coerce").between(1944, 2025)]
    p1_observed = p1_primary[pd.to_numeric(p1_primary["Is_Carried_Forward"], errors="coerce") == 0]
    if (p1_rows, p1_inst, len(p1_primary), p1_primary["Institution"].nunique(), len(p1_observed)) != (4296, 150, 4277, 150, 480):
        raise AssertionError(
            "Paper 1 structural benchmarks changed: "
            f"rows={p1_rows}, inst={p1_inst}, primary={len(p1_primary)}, "
            f"primary_inst={p1_primary['Institution'].nunique()}, observed={len(p1_observed)}"
        )

    inst_col, candidate_matches = choose_p2_institution_column(p2, p1)
    cross_merged, matched_n, compared, missing_columns, mismatches = compare_cross_repo(p2, p1, inst_col)
    if matched_n != EXPECTED_CROSS_MATCH:
        raise AssertionError(
            f"Expected {EXPECTED_CROSS_MATCH} Paper 2 rows with PCI to match Paper 1 keys; found {matched_n}. "
            f"Institution candidate diagnostics: {candidate_matches}"
        )
    nonzero_mismatch = {k: v for k, v in mismatches.items() if v != 0}
    if nonzero_mismatch:
        raise AssertionError(f"Cross-repository P1/P2 field mismatches found: {nonzero_mismatch}")

    rep = import_replication_module()
    prepared = rep.load_and_prepare(P2_DATA)
    baseline = rep.twfe(
        prepared,
        "ln_new_patent_apps",
        treatment="pci_l1",
        controls=rep.BASE_CONTROLS,
    )
    if baseline["n"] != EXPECTED_BASELINE["n"]:
        raise AssertionError(f"Baseline N changed: {baseline['n']}")
    for key, tol in [("beta", 0.001), ("se", 0.001), ("p", 0.003)]:
        if not math.isclose(float(baseline[key]), EXPECTED_BASELINE[key], abs_tol=tol):
            raise AssertionError(
                f"Baseline {key} changed: got {baseline[key]}, expected about {EXPECTED_BASELINE[key]}"
            )

    verified = build_verified_dataset(rep, p2, inst_col, cross_merged)
    verified.to_csv(OUT_DATA, index=False)
    out_sha = sha256(OUT_DATA)

    result = {
        "paper1_commit": P1_COMMIT,
        "paper2_base_commit": P2_BASE_COMMIT,
        "paper1_panel_sha256": p1_sha,
        "paper2_merged_autm_sha256": p2_sha,
        "verified_dataset_sha256": out_sha,
        "paper1_full_rows": p1_rows,
        "paper1_institutions": int(p1_inst),
        "paper1_primary_rows_1944_2025": int(len(p1_primary)),
        "paper1_direct_observed_rows_1944_2025": int(len(p1_observed)),
        "paper2_rows": int(len(p2)),
        "paper2_rows_with_pci": int(p2["Mean_Tone_Score"].notna().sum()),
        "paper2_institution_name_column": inst_col,
        "institution_candidate_match_counts": candidate_matches,
        "cross_repo_matches": matched_n,
        "cross_repo_columns_compared": compared,
        "cross_repo_columns_missing_from_comparison": missing_columns,
        "cross_repo_mismatches": mismatches,
        "baseline": {k: float(baseline[k]) if k != "n" else int(baseline[k]) for k in ["beta", "se", "p", "n"]},
        "verified_dataset_rows": int(len(verified)),
        "verified_dataset_columns": int(len(verified.columns)),
        "verified_dataset_paper1_match_rows": int(verified["paper1_key_match_verified"].sum()),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    report = "# Cross-repository replication verification\n\n"
    report += f"- Paper 1 source commit: `{P1_COMMIT}`\n"
    report += f"- Paper 2 base commit: `{P2_BASE_COMMIT}`\n"
    report += f"- Paper 1 canonical panel SHA-256: `{p1_sha}`\n"
    report += f"- Paper 2 merged dataset SHA-256: `{p2_sha}`\n"
    report += f"- Verified replication dataset SHA-256: `{out_sha}`\n\n"
    report += "## Paper 1 checks\n\n"
    report += f"- Full canonical panel: **{p1_rows:,} rows, {p1_inst} institutions**.\n"
    report += f"- Primary 1944-2025 panel: **{len(p1_primary):,} rows, {p1_primary['Institution'].nunique()} institutions**.\n"
    report += f"- Directly observed 1944-2025 records: **{len(p1_observed):,}**.\n\n"
    report += "## Cross-repository merge checks\n\n"
    report += f"- Paper 2 institution-name key selected: `{inst_col}`.\n"
    report += f"- Paper 2 rows with PCI: **{int(p2['Mean_Tone_Score'].notna().sum()):,}**.\n"
    report += f"- Rows matched to Paper 1 on standardized institution + calendar year: **{matched_n:,}**.\n"
    report += f"- Compared fields: {', '.join(compared)}.\n"
    report += f"- Field mismatch counts: `{json.dumps(mismatches, sort_keys=True)}`.\n"
    if missing_columns:
        report += f"- Requested comparison fields not available on both sides: {', '.join(missing_columns)}.\n"
    report += "\n## Paper 2 baseline re-estimation\n\n"
    report += (
        f"- Lag-1 PCSI coefficient on ln(1 + new patent applications): **{baseline['beta']:.6f}**.\n"
        f"- Clustered SE: **{baseline['se']:.6f}**.\n"
        f"- p-value: **{baseline['p']:.6f}**.\n"
        f"- N: **{baseline['n']:,}**.\n\n"
    )
    report += "## New analysis-ready dataset\n\n"
    report += f"`data/{OUT_DATA.name}` contains **{len(verified):,} rows and {len(verified.columns)} columns**. "
    report += "It is rebuilt from the original Paper 2 merge, uses calendar-year lags from the Paper 2 replication code, "
    report += "and includes commit provenance plus a row-level Paper 1 key-verification flag.\n"
    OUT_REPORT.write_text(report, encoding="utf-8")

    p1_path.unlink(missing_ok=True)
    print(report)
    print(f"Wrote {OUT_DATA.relative_to(PACKAGE_ROOT)}")
    print(f"Wrote {OUT_REPORT.relative_to(PACKAGE_ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(PACKAGE_ROOT)}")


if __name__ == "__main__":
    main()
