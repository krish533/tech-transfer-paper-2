"""
extended_analysis.py
====================
Extension analyses for Paper 2:
  1. Data quality fix: flag ECU sentence-length outliers (mean_sentence_length > 500)
  2. Direction-split event study (±4 year window, upward vs downward revisions)
  3. Targeted DiD: post window restricted to et ∈ {2, 3}
  4. Mechanism analysis: 15 granular text features on conversion rate
  5. LaTeX table fragments saved to paper_outputs/tables/

Run from any directory:
    python code/extended_analysis.py

Requires the base replication package to be installed (same dependencies).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats, linalg
import warnings
warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

from replication import load_and_prepare, twfe, _stars, BASE_CONTROLS, REVISION_THRESHOLD

TABLES_DIR = PACKAGE_ROOT / "paper_outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

EVENT_WINDOW = 4          # ±4 years around revision
MIN_PRE_PERIODS = 2       # require at least 2 clean pre-periods per event
SENT_LEN_CUTOFF = 500     # mean_sentence_length > this → parsing error


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _coerce(series):
    if (series.dtype == object
            or str(series.dtype) == "str"
            or "string" in str(series.dtype)):
        series = (series.astype(str)
                  .str.replace("%", "", regex=False)
                  .str.strip()
                  .replace({"": np.nan, "nan": np.nan, "None": np.nan}))
    return pd.to_numeric(series, errors="coerce")


def load_extended():
    """Load merged_autm with all granular text features retained."""
    raw = pd.read_csv(PACKAGE_ROOT / "data" / "merged_autm.csv")

    # Standard renames
    from replication import RENAME_MAP
    df = raw.rename(columns=RENAME_MAP)
    df = df.sort_values(["institution_id", "year"]).reset_index(drop=True)

    # Coerce standard columns
    coerce_cols = [
        "pci", "pci_median", "tone_index", "clarity_index", "legal_load_index",
        "new_patent_apps", "total_patent_apps", "patents_issued",
        "disclosures", "licenses", "startups", "license_income",
        "research_exp", "licensing_ftes", "royalty_share", "tlo_age",
        "private", "carnegie_r1", "med_school", "land_grant",
    ]
    for c in coerce_cols:
        if c in df.columns:
            df[c] = _coerce(df[c])

    if "research_exp" in df.columns and df["research_exp"].median(skipna=True) > 1e6:
        df["research_exp"] = df["research_exp"] / 1e6

    # Granular text features (already lower-case in raw)
    text_features = [
        "supportive_per_1000w", "restrictive_per_1000w", "sanction_per_1000w",
        "second_person_per_1000w", "inclusive_we_per_1000w",
        "obligation_modal_share", "mean_sentence_length", "long_sentence_share",
        "procedural_share", "legalese_per_1000w", "iptech_per_1000w",
        "n_sentences", "n_words",
    ]
    for c in text_features:
        if c in df.columns:
            df[c] = _coerce(df[c])

    # Institution name for event study
    if "Institution" in raw.columns:
        df["institution_name"] = raw["Institution"].values

    # Log outcomes
    for v in ["new_patent_apps", "total_patent_apps", "patents_issued",
              "disclosures", "licenses", "startups", "license_income",
              "research_exp", "licensing_ftes"]:
        if v in df.columns:
            df[f"ln_{v}"] = np.log1p(pd.to_numeric(df[v], errors="coerce"))

    if "ln_new_patent_apps" in df.columns and "ln_disclosures" in df.columns:
        df["conv_rate"] = df["ln_new_patent_apps"] - df["ln_disclosures"]

    # Calendar-year lags
    lag_src = (
        ["pci", "pci_median", "tone_index", "clarity_index", "legal_load_index",
         "ln_research_exp", "ln_licensing_ftes", "royalty_share", "tlo_age"]
        + text_features
    )
    lookup_index = df.set_index(["institution_id", "year"])
    for v in lag_src:
        if v not in df.columns:
            continue
        lkp = lookup_index[v]
        for k in [1, 2, 3, 4, 5]:
            lag_idx = pd.MultiIndex.from_arrays([df["institution_id"], df["year"] - k])
            df[f"{v}_l{k}"] = lkp.reindex(lag_idx).to_numpy()

    # Forward research exp for falsification
    if "ln_research_exp" in df.columns:
        lkp = lookup_index["ln_research_exp"]
        lead_idx = pd.MultiIndex.from_arrays([df["institution_id"], df["year"] + 1])
        df["ln_research_exp_fwd"] = lkp.reindex(lead_idx).to_numpy()

    df["pci_change"] = df["pci"] - df.groupby("institution_id")["pci"].shift(1)

    return df


def flag_ecу_outliers(df):
    """
    Mark rows with mean_sentence_length > SENT_LEN_CUTOFF as data errors.
    These rows are excluded from mechanism regressions involving text features
    (East Carolina University 2021-2023, parsing artifact).
    """
    if "mean_sentence_length" not in df.columns:
        df["ecу_flag"] = False
        return df
    df["ecу_flag"] = df["mean_sentence_length"] > SENT_LEN_CUTOFF
    n_flagged = df["ecу_flag"].sum()
    insts = df.loc[df["ecу_flag"], "institution_name"].unique() if "institution_name" in df.columns else []
    print(f"\n  Data quality: {n_flagged} rows flagged (mean_sentence_length > {SENT_LEN_CUTOFF})")
    if len(insts):
        print(f"  Affected institutions: {', '.join(insts)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM TABLE
# ─────────────────────────────────────────────────────────────────────────────

TEXT_FEATURE_LABELS = {
    "supportive_per_1000w_l1":    "Supportive terms (per 1,000w)",
    "restrictive_per_1000w_l1":   "Restrictive terms (per 1,000w)",
    "sanction_per_1000w_l1":      "Sanction terms (per 1,000w)",
    "second_person_per_1000w_l1": "Second-person pronouns (per 1,000w)",
    "inclusive_we_per_1000w_l1":  "Inclusive 'we' (per 1,000w)",
    "obligation_modal_share_l1":  "Obligation modal share",
    "mean_sentence_length_l1":    "Mean sentence length (words)",
    "long_sentence_share_l1":     "Long-sentence share",
    "procedural_share_l1":        "Procedural cue share",
    "legalese_per_1000w_l1":      "Legalese terms (per 1,000w)",
    "iptech_per_1000w_l1":        "IP/tech terms (per 1,000w)",
}


def mechanism_table(df):
    """
    For each granular text feature, run a separate TWFE regression on
    conv_rate and ln_new_patent_apps.  ECU outlier rows are excluded for
    features involving sentence length; all other rows included for all
    other features.
    """
    print("\n" + "=" * 74)
    print("APPENDIX C1: Mechanism Analysis — Granular Text Features")
    print("=" * 74)

    df_clean = df[~df["ecу_flag"]].copy()   # ECU rows excluded

    rows_conv = []
    rows_pats = []

    for feat, label in TEXT_FEATURE_LABELS.items():
        if feat not in df.columns:
            continue
        # For sentence-length features use ECU-excluded sample
        if "sentence_length" in feat or "long_sentence" in feat:
            src = df_clean
        else:
            src = df

        rc = twfe(src, "conv_rate",          treatment=feat)
        rp = twfe(src, "ln_new_patent_apps", treatment=feat)
        rows_conv.append((label, rc))
        rows_pats.append((label, rp))

    print(f"\n  {'Feature':<40} {'Conv Rate b':>11}  {'SE':>7} {'p':>7} {'N':>6}")
    print("  " + "-" * 74)
    for label, r in rows_conv:
        print(f"  {label:<40} {r['beta']:>+11.3f}{_stars(r['p']):<3} "
              f"{r['se']:>7.3f} {r['p']:>7.3f} {r['n']:>6,}")

    print(f"\n  {'Feature':<40} {'PatApps b':>11}  {'SE':>7} {'p':>7} {'N':>6}")
    print("  " + "-" * 74)
    for label, r in rows_pats:
        print(f"  {label:<40} {r['beta']:>+11.3f}{_stars(r['p']):<3} "
              f"{r['se']:>7.3f} {r['p']:>7.3f} {r['n']:>6,}")

    # Save LaTeX table
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Mechanism Analysis: Granular Text Features}",
        r"\label{tab:mechanism}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Feature & \multicolumn{2}{c}{Conversion Rate} & \multicolumn{2}{c}{Ln(Patent Apps)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & $\beta$ & $p$ & $\beta$ & $p$ \\",
        r"\midrule",
    ]
    for (label, rc), (_, rp) in zip(rows_conv, rows_pats):
        b_c = f"{rc['beta']:+.3f}{_stars(rc['p'])}"
        b_p = f"{rp['beta']:+.3f}{_stars(rp['p'])}"
        lines.append(
            f"  {label} & {b_c} & {rc['p']:.3f} & {b_p} & {rp['p']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize All specifications include institution and year FE, clustered SEs.}\\",
        r"\multicolumn{5}{l}{\footnotesize Sentence-length rows exclude ECU 2021--23 parsing artifact.}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = TABLES_DIR / "table_mechanism.tex"
    tex_path.write_text("\n".join(lines))
    print(f"\n  LaTeX saved → {tex_path.relative_to(PACKAGE_ROOT)}")

    return rows_conv, rows_pats


# ─────────────────────────────────────────────────────────────────────────────
# EVENT STUDY
# ─────────────────────────────────────────────────────────────────────────────

def _build_event_data(df):
    """
    For each institution with |dPCI| > REVISION_THRESHOLD, collect
    event-time conversion rate observations from -EVENT_WINDOW to +EVENT_WINDOW.
    Only includes events with at least MIN_PRE_PERIODS clean pre-event obs.
    Returns a DataFrame with columns: institution_id, event_id, et, conv_rate, direction.
    """
    df = df.sort_values(["institution_id", "year"]).copy()
    ev_rows = []
    event_id = 0

    for inst_id, grp in df.groupby("institution_id"):
        grp = grp.sort_values("year").reset_index(drop=True)
        if "pci" not in grp.columns or grp["pci"].isna().all():
            continue

        years = grp["year"].values
        pci = grp["pci"].values

        for i in range(1, len(grp)):
            if pd.isna(pci[i]) or pd.isna(pci[i - 1]):
                continue
            delta = pci[i] - pci[i - 1]
            if abs(delta) <= REVISION_THRESHOLD:
                continue

            event_year = years[i]
            direction = "up" if delta > 0 else "down"

            # Check pre-event data availability
            pre_obs = grp[
                (grp["year"] >= event_year - EVENT_WINDOW) &
                (grp["year"] < event_year) &
                grp["conv_rate"].notna()
            ]
            if len(pre_obs) < MIN_PRE_PERIODS:
                continue

            for _, row in grp.iterrows():
                et = row["year"] - event_year
                if abs(et) > EVENT_WINDOW:
                    continue
                if pd.notna(row.get("conv_rate")):
                    ev_rows.append({
                        "institution_id": inst_id,
                        "event_id": event_id,
                        "event_year": event_year,
                        "et": et,
                        "conv_rate": row["conv_rate"],
                        "direction": direction,
                        "delta_pci": delta,
                    })
            event_id += 1

    return pd.DataFrame(ev_rows) if ev_rows else pd.DataFrame(
        columns=["institution_id", "event_id", "event_year", "et",
                 "conv_rate", "direction", "delta_pci"]
    )


def event_study(df):
    """
    Direction-split event study: upward vs downward PCI revisions.
    Prints event-time table and targeted DiD (post = et ∈ {2, 3}).
    """
    print("\n" + "=" * 74)
    print("APPENDIX C2: Direction-Split Event Study")
    print("=" * 74)

    ev = _build_event_data(df)
    if ev.empty:
        print("  No qualifying events found.")
        return None

    n_events = ev["event_id"].nunique()
    n_up = ev[ev["direction"] == "up"]["event_id"].nunique()
    n_dn = ev[ev["direction"] == "down"]["event_id"].nunique()
    print(f"\n  Events: {n_events} total ({n_up} upward, {n_dn} downward)")
    print(f"  Window: ±{EVENT_WINDOW} years; requires ≥{MIN_PRE_PERIODS} pre-event obs")

    # Event-time table: mean conv_rate by (et, direction)
    et_table = (ev.groupby(["et", "direction"])["conv_rate"]
                  .agg(["mean", "sem", "count"])
                  .reset_index())
    et_table.columns = ["et", "direction", "mean", "sem", "n"]

    print(f"\n  {'et':>4}  {'Up mean':>9}  {'Up SE':>7}  {'Up N':>5}  "
          f"{'Dn mean':>9}  {'Dn SE':>7}  {'Dn N':>5}  {'Diff':>8}  p")
    print("  " + "-" * 72)

    pre_diffs = []
    et_rows = []  # for targeted DiD

    for et in range(-EVENT_WINDOW, EVENT_WINDOW + 1):
        up_row = et_table[(et_table["et"] == et) & (et_table["direction"] == "up")]
        dn_row = et_table[(et_table["et"] == et) & (et_table["direction"] == "down")]
        if up_row.empty or dn_row.empty:
            continue
        mu_up, se_up, n_up_t = up_row[["mean", "sem", "n"]].values[0]
        mu_dn, se_dn, n_dn_t = dn_row[["mean", "sem", "n"]].values[0]
        diff = mu_up - mu_dn
        se_diff = np.sqrt(se_up ** 2 + se_dn ** 2)
        if se_diff > 0:
            t_stat = diff / se_diff
            dof = (se_up**2 / n_up_t + se_dn**2 / n_dn_t)**2 / (
                (se_up**2 / n_up_t)**2 / (n_up_t - 1) +
                (se_dn**2 / n_dn_t)**2 / (n_dn_t - 1)
            ) if n_up_t > 1 and n_dn_t > 1 else 1
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(dof, 1)))
        else:
            p_val = float("nan")

        star = _stars(p_val)
        print(f"  {et:>4}  {mu_up:>9.3f}  {se_up:>7.4f}  {int(n_up_t):>5}  "
              f"{mu_dn:>9.3f}  {se_dn:>7.4f}  {int(n_dn_t):>5}  "
              f"{diff:>+8.3f}{star}  {p_val:.3f}")

        if et < 0:
            pre_diffs.append(diff)
        et_rows.append(dict(et=et, diff=diff, se_diff=se_diff, p=p_val,
                            mu_up=mu_up, mu_dn=mu_dn, n=n_up_t + n_dn_t))

    # Pre-trend Wald test
    if len(pre_diffs) > 1:
        pre_var = np.var(pre_diffs, ddof=1)
        wald_stat = np.sum(np.array(pre_diffs) ** 2) / pre_var if pre_var > 0 else 0
        pre_p = 1 - stats.chi2.cdf(wald_stat, df=len(pre_diffs))
        print(f"\n  Pre-trend Wald test: stat = {wald_stat:.3f}, "
              f"df = {len(pre_diffs)}, p = {pre_p:.3f}  "
              f"({'CLEAN' if pre_p > 0.10 else 'CONCERN'})")
    else:
        pre_p = float("nan")
        print("\n  Pre-trend test: insufficient pre-periods.")

    # Targeted DiD: post window restricted to et ∈ {2, 3}
    print("\n  --- Targeted DiD: post window et ∈ {2, 3} vs. pre window et ∈ {-4, -3} ---")
    pre_ev  = ev[ev["et"].isin([-4, -3, -2, -1])].copy()
    post_ev = ev[ev["et"].isin([2, 3])].copy()

    if not pre_ev.empty and not post_ev.empty:
        up_pre  = pre_ev[pre_ev["direction"] == "up"]["conv_rate"].mean()
        dn_pre  = pre_ev[pre_ev["direction"] == "down"]["conv_rate"].mean()
        up_post = post_ev[post_ev["direction"] == "up"]["conv_rate"].mean()
        dn_post = post_ev[post_ev["direction"] == "down"]["conv_rate"].mean()

        did_up = up_post - up_pre
        did_dn = dn_post - dn_pre
        did_diff = did_up - did_dn

        # SE via bootstrap
        rng = np.random.default_rng(42)
        event_ids = ev["event_id"].unique()
        n_boot = 2000
        boot_dids = []
        for _ in range(n_boot):
            ids = rng.choice(event_ids, size=len(event_ids), replace=True)
            b_ev = ev[ev["event_id"].isin(ids)]
            b_pre  = b_ev[b_ev["et"].isin([-4, -3, -2, -1])]
            b_post = b_ev[b_ev["et"].isin([2, 3])]
            try:
                b_diff = (
                    (b_post[b_post["direction"] == "up"]["conv_rate"].mean() -
                     b_pre[b_pre["direction"] == "up"]["conv_rate"].mean()) -
                    (b_post[b_post["direction"] == "down"]["conv_rate"].mean() -
                     b_pre[b_pre["direction"] == "down"]["conv_rate"].mean())
                )
                boot_dids.append(b_diff)
            except Exception:
                continue

        boot_arr = np.array([x for x in boot_dids if not np.isnan(x)])
        se_boot = np.std(boot_arr) if len(boot_arr) > 10 else float("nan")
        t_stat = did_diff / se_boot if se_boot > 0 else float("nan")
        p_targeted = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_events - 1)) if se_boot > 0 else float("nan")

        print(f"  Up DiD (post−pre):   {did_up:+.3f}")
        print(f"  Down DiD (post−pre): {did_dn:+.3f}")
        print(f"  Difference-in-DiD:  {did_diff:+.3f}  SE = {se_boot:.3f}  "
              f"p = {p_targeted:.3f}{_stars(p_targeted)}")
    else:
        p_targeted = float("nan")
        did_diff = float("nan")
        se_boot = float("nan")
        print("  Insufficient events for targeted DiD.")

    # Save event-time LaTeX table
    et_lines = [
        r"\begin{table}[htbp]",
        r"\caption{Event Study: Conversion Rate Around Policy Revisions}",
        r"\label{tab:eventstudy}",
        r"\small",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$\tau$ & Up mean & Up SE & Up $N$ & Down mean & Down SE & Down $N$ & Diff ($p$) \\",
        r"\midrule",
    ]
    for row in et_rows:
        et_val = int(row["et"])
        diff_str = f"{row['diff']:+.3f}{_stars(row['p'])}"
        et_lines.append(
            f"  {et_val} & {row['mu_up']:.3f} & {row['se_diff']/np.sqrt(2):.4f} & "
            f"{int(row['n']//2)} & {row['mu_dn']:.3f} & {row['se_diff']/np.sqrt(2):.4f} & "
            f"{int(row['n']//2)} & {diff_str} ({row['p']:.3f}) \\\\"
        )
    et_lines += [
        r"\bottomrule",
        r"\multicolumn{8}{l}{\footnotesize Pre-trend Wald test " +
        f"$p={pre_p:.3f}$.}}\\\\" if not np.isnan(pre_p) else r"\end{tabular}",
        r"\multicolumn{8}{l}{\footnotesize Targeted DiD (post = $\tau \in \{2,3\}$):" +
        (f" $\\Delta\\Delta = {did_diff:+.3f}$, SE = {se_boot:.3f}, $p = {p_targeted:.3f}${_stars(p_targeted)}.}}\\\\"
         if not np.isnan(p_targeted) else r""),
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = TABLES_DIR / "table_eventstudy.tex"
    tex_path.write_text("\n".join(et_lines))
    print(f"\n  LaTeX saved → {tex_path.relative_to(PACKAGE_ROOT)}")

    return ev, et_rows, pre_p, did_diff, se_boot, p_targeted


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 74)
    print("EXTENDED ANALYSIS - Policy Communication and Technology Transfer")
    print("Sharma, Wang, Basnet, Cossco (2026)")
    print("=" * 74)

    df = load_extended()
    df = flag_ecу_outliers(df)

    print(f"\nSample: {df['institution_id'].nunique()} institutions, "
          f"{df['year'].min()}–{df['year'].max()}, {len(df):,} obs")
    print(f"  Obs with PCI: {df['pci'].notna().sum():,}")
    print(f"  ECU outlier rows: {df['ecу_flag'].sum()}")

    # Mechanism table
    rows_conv, rows_pats = mechanism_table(df)

    # Event study
    result = event_study(df)

    print("\n" + "=" * 74)
    print("Extended analysis complete.")
    print("=" * 74)
