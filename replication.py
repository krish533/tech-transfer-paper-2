"""
replication.py
==============
Policy Communication and Technology Transfer:
Evidence from University Intellectual Property Governance Documents
Sharma, Wang, Basnet, Cossco (2026)

Produces ALL tables and figures in the paper:

  TABLE 1   — Descriptive Statistics
  TABLE 2   — Main TWFE Results + Lag Sensitivity
  TABLE 3   — Heterogeneity (split-sample) + Mundlak Decomposition
  TABLE 4   — Channel Analysis + Multiple Testing
  FIGURE 1  — Event Study: Pre/Post Coefficients for Patent Apps vs Disclosures
  APP A1    — Sub-Index Correlation Matrix
  APP B1    — Robustness Suite (winsorize, balanced, R1, HC3, Nickell, quad trend,
               falsification, leave-one-out)
  APP B2    — Lag Sensitivity across all outcomes

Requirements:
  pip install pandas numpy scipy statsmodels matplotlib

Data file:  merged_autm.csv  (place in same directory)

Column mapping (raw AUTM/PCI names → analysis names) is handled automatically.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SEED        = 42
WIN_TRIM    = 0.01             # winsorize at 1st / 99th percentile
EVENT_WIN   = 4                # years before/after revision to show in event study
# Revision threshold: abs(ΔPCI) must exceed this to count as a policy revision
# Set to 75th-percentile of non-trivial absolute changes ≈ 0.054
REVISION_THRESHOLD = 0.05
LAST_RI_P   = None

# ─────────────────────────────────────────────────────────────────────────────
# 0.  COLUMN RENAME MAP
# ─────────────────────────────────────────────────────────────────────────────

RENAME_MAP = {
    "[ID]":                   "institution_id",
    "Year":                   "year",
    "Mean_Tone_Score":        "pci",
    "Median_Tone_Score":      "pci_median",
    "Tone_Index":             "tone_index",
    "Clarity_Index":          "clarity_index",
    "Legal_Load_Index":       "legal_load_index",
    "New Pat App Fld":        "new_patent_apps",
    "Tot Pat App Fld":        "total_patent_apps",
    "Iss US Pat":             "patents_issued",
    "Inv Dis Rec":            "disclosures",
    "Tot Lic/Opt Exe":        "licenses",
    "St-Ups Formed":          "startups",
    "Gross Lic Inc":          "license_income",
    "Tot Res Exp":            "research_exp",
    "Lic FTEs":               "licensing_ftes",
    "Royalty Share":          "royalty_share",
    "TLO Age":                "tlo_age",
    "Private":                "private",
    "Carnegie R1":            "carnegie_r1",
    "MEDSCHOOL":              "med_school",
    "Land-Grant Institution": "land_grant",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING AND PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_data_path(path="merged_autm.csv"):
    """
    Resolve the shipped data file in the repository directory.
    """
    repo_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates = [repo_dir / path, repo_dir / "merged_autm.csv", repo_dir / "merged_autm (3).csv", Path(path)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No AUTM/PCI merged CSV found. Expected merged_autm.csv or merged_autm (3).csv."
    )


def _coerce_numeric_series(series):
    """
    Convert mixed string/numeric columns into numeric values.
    Handles current pandas string dtypes as well as object columns.
    """
    if (
        series.dtype == object
        or str(series.dtype) == "str"
        or "string" in str(series.dtype)
    ):
        series = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
    return pd.to_numeric(series, errors="coerce")


def load_and_prepare(path="merged_autm.csv"):
    """
    Load raw CSV, rename columns, build all derived variables.
    Returns a clean analysis-ready DataFrame.
    """
    df = pd.read_csv(_resolve_data_path(path))
    df = df.rename(columns=RENAME_MAP)
    df = df.sort_values(["institution_id", "year"]).reset_index(drop=True)

    # ── coerce to numeric (handles "33.00%" strings, mixed types) ──────────
    coerce_cols = [
        "pci", "pci_median", "tone_index", "clarity_index", "legal_load_index",
        "new_patent_apps", "total_patent_apps", "patents_issued",
        "disclosures", "licenses", "startups", "license_income",
        "research_exp", "licensing_ftes", "royalty_share", "tlo_age",
        "private", "carnegie_r1", "med_school", "land_grant",
    ]
    for c in coerce_cols:
        if c not in df.columns:
            continue
        df[c] = _coerce_numeric_series(df[c])

    # Scale research expenditure from dollars to millions if needed
    if "research_exp" in df.columns:
        if df["research_exp"].median(skipna=True) > 1e6:
            df["research_exp"] = df["research_exp"] / 1e6

    # ── log-transformed outcomes ───────────────────────────────────────────
    outcomes_raw = [
        "new_patent_apps", "total_patent_apps", "patents_issued",
        "disclosures", "licenses", "startups", "license_income",
        "research_exp", "licensing_ftes",
    ]
    for v in outcomes_raw:
        if v in df.columns:
            df[f"ln_{v}"] = np.log1p(pd.to_numeric(df[v], errors="coerce"))

    # Conversion rate: ln(PatApp) − ln(Discl)
    if "ln_new_patent_apps" in df.columns and "ln_disclosures" in df.columns:
        df["conv_rate"] = df["ln_new_patent_apps"] - df["ln_disclosures"]

    # ── lagged variables (k = 1, 2, 3) ────────────────────────────────────
    lag_src = [
        "pci", "pci_median",
        "tone_index", "clarity_index", "legal_load_index",
        "ln_research_exp", "ln_licensing_ftes", "royalty_share",
    ]
    for v in lag_src:
        if v not in df.columns:
            continue
        for k in [1, 2, 3]:
            df[f"{v}_l{k}"] = df.groupby("institution_id")[v].transform(
                lambda s, k=k: s.shift(k)
            )

    # ── Mundlak components ─────────────────────────────────────────────────
    if "pci" in df.columns:
        df["pci_mean"]   = df.groupby("institution_id")["pci"].transform("mean")
        df["pci_within"] = df["pci_l1"] - df["pci_mean"]

    # ── additional robustness variables ───────────────────────────────────
    if "ln_new_patent_apps" in df.columns:
        df["ln_new_patent_apps_lag"] = df.groupby("institution_id")[
            "ln_new_patent_apps"
        ].transform(lambda s: s.shift(1))

    if "ln_research_exp" in df.columns:
        df["ln_research_exp_fwd"] = df.groupby("institution_id")[
            "ln_research_exp"
        ].transform(lambda s: s.shift(-1))

    # Quadratic time trend (demeaned within institution)
    df["year2"]    = df["year"] ** 2
    df["year_dm"]  = (df["year"]
                      - df.groupby("institution_id")["year"].transform("mean"))
    df["year2_dm"] = (df["year2"]
                      - df.groupby("institution_id")["year2"].transform("mean"))

    # ── event study variables ──────────────────────────────────────────────
    df = _build_event_study_vars(df)

    return df


def _build_event_study_vars(df):
    """
    Identify policy revision events and compute event-time dummies.

    A 'large positive revision' is defined as the FIRST year in which an
    institution's PCI increases by more than REVISION_THRESHOLD in a single
    year.  We use first-revision only to avoid stacking multiple events on
    the same institution.

    Returns df with columns:
      revision_year   — year of first large positive revision (NaN if none)
      event_time      — year − revision_year (NaN if not treated)
      event_{k}       — indicator for event_time == k, k ∈ [−WIN, +WIN]
                        event_{-1} is the omitted baseline
    """
    df = df.copy()
    df["pci_change"] = df.groupby("institution_id")["pci"].diff()

    # First year with a large positive PCI change
    mask = df["pci_change"] > REVISION_THRESHOLD
    first_rev = (df[mask]
                 .groupby("institution_id")["year"]
                 .min()
                 .rename("revision_year"))

    df = df.merge(first_rev, on="institution_id", how="left")
    df["event_time"] = df["year"] - df["revision_year"]

    # Create event-time dummies (omit k = −1 as baseline)
    for k in range(-EVENT_WIN, EVENT_WIN + 1):
        if k == -1:
            continue
        df[f"event_{k}"] = ((df["event_time"] == k) &
                             df["revision_year"].notna()).astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CORE ESTIMATOR: TWO-WAY FE WITH CLUSTERED OR HC3 STANDARD ERRORS
# ─────────────────────────────────────────────────────────────────────────────

def twfe(df, outcome, treatment="pci_l1",
         controls=("ln_research_exp_l1", "ln_licensing_ftes_l1"),
         entity="institution_id", time="year",
         se_type="cluster",
         extra_regressors=()):
    """
    Estimate a two-way FE model via within-transformation.

    Parameters
    ----------
    df            : DataFrame
    outcome       : str — dependent variable (log-transformed)
    treatment     : str — main regressor of interest
    controls      : tuple of str — additional controls
    entity, time  : str — panel identifiers
    se_type       : 'cluster' (default) | 'hc3'
    extra_regressors : tuple of str — additional right-hand side vars
                       (used for joint sub-index specs and event study)

    Returns
    -------
    dict with keys: beta, se, p, n, ci_lo, ci_hi, t
    For multi-regressor calls (extra_regressors non-empty), returns
    additionally 'all_betas' and 'all_ses' indexed by variable name.
    """
    all_vars = (
        [outcome, treatment]
        + list(controls)
        + list(extra_regressors)
        + [entity, time]
    )
    sub = df[[c for c in all_vars if c in df.columns]].dropna().copy()
    if len(sub) < 20:
        nan = float("nan")
        return dict(beta=nan, se=nan, p=nan, n=0,
                    ci_lo=nan, ci_hi=nan, t=nan,
                    all_betas={}, all_ses={})

    reg_vars = (
        [outcome, treatment]
        + list(controls)
        + list(extra_regressors)
    )

    # Within-transformation: entity-demean + time-demean + grand mean back
    for col in reg_vars:
        sub[f"{col}_w"] = (
            sub[col]
            - sub.groupby(entity)[col].transform("mean")
            - sub.groupby(time)[col].transform("mean")
            + sub[col].mean()
        )

    y     = sub[f"{outcome}_w"].values
    x_names = [treatment] + list(controls) + list(extra_regressors)
    Xw    = [sub[f"{v}_w"].values for v in x_names]
    X     = np.column_stack([np.ones(len(sub))] + Xw)
    n, k  = X.shape

    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat

    # ── variance estimator ─────────────────────────────────────────────────
    if se_type == "hc3":
        # HC3: leave-one-out leverage correction
        H_diag = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
        e_adj  = resid / (1.0 - np.clip(H_diag, None, 0.9999))
        meat   = (X * e_adj[:, None]).T @ (X * e_adj[:, None])
        V      = np.linalg.inv(X.T @ X) @ meat @ np.linalg.inv(X.T @ X)
        df_t   = n - k
    else:
        # Clustered SE (Cameron & Miller 2015)
        clusters = sub[entity].values
        uniq     = np.unique(clusters)
        G        = len(uniq)
        bread    = np.linalg.inv(X.T @ X)
        meat     = np.zeros((k, k))
        for g in uniq:
            idx = clusters == g
            sg  = X[idx].T @ resid[idx]
            meat += np.outer(sg, sg)
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        V  = correction * bread @ meat @ bread
        df_t = G - 1

    se_vec = np.sqrt(np.diag(V).clip(0))

    # Treatment coefficient (index 1 in beta_hat)
    b, se_b = beta_hat[1], se_vec[1]
    t_stat  = b / se_b if se_b > 0 else float("nan")
    p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_t))

    # All coefficients (for event study / joint specs)
    all_betas = {v: beta_hat[i + 1] for i, v in enumerate(x_names)}
    all_ses   = {v: se_vec[i + 1]   for i, v in enumerate(x_names)}

    return dict(
        beta=b, se=se_b, p=p_val, n=n,
        ci_lo=b - 1.96 * se_b,
        ci_hi=b + 1.96 * se_b,
        t=t_stat,
        all_betas=all_betas,
        all_ses=all_ses,
    )


def _stars(p):
    if   p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TABLE 1 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def table1(df):
    print("\n" + "=" * 74)
    print("TABLE 1: Descriptive Statistics")
    print("=" * 74)

    panels = {
        "Panel A: Policy Communication Index": [
            ("pci",              "PCI (Mean Tone Score)"),
            ("tone_index",       "Tone Index"),
            ("clarity_index",    "Clarity Index"),
            ("legal_load_index", "Legal Load Index"),
        ],
        "Panel B: Technology Transfer Outcomes": [
            ("new_patent_apps", "New Patent Applications"),
            ("disclosures",     "Invention Disclosures"),
            ("licenses",        "Licenses Executed"),
            ("startups",        "Startups Formed"),
            ("license_income",  "License Income ($000)"),
        ],
        "Panel C: Institutional Characteristics": [
            ("research_exp",   "Total Research Exp ($M)"),
            ("licensing_ftes", "Licensing FTEs"),
            ("royalty_share",  "Royalty Share (%)"),
            ("tlo_age",        "TLO Age (years)"),
        ],
    }

    print(f"\n  {'Variable':<38} {'Mean':>8} {'SD':>8} "
          f"{'Min':>8} {'Max':>8} {'N':>7}")
    print("  " + "-" * 80)

    for panel, varlist in panels.items():
        print(f"\n  {panel}")
        for var, label in varlist:
            if var not in df.columns:
                continue
            s = pd.to_numeric(df[var], errors="coerce").dropna()
            print(f"    {label:<36} {s.mean():>8.1f} {s.std():>8.1f}"
                  f" {s.min():>8.1f} {s.max():>8.1f} {len(s):>7,}")

    # Revision summary
    if "pci_change" in df.columns:
        n_rev = (df["pci_change"].abs() > REVISION_THRESHOLD).sum()
        pct   = 100 * n_rev / len(df.dropna(subset=["pci_change"]))
        n_inst_rev = df.loc[
            df["pci_change"].abs() > REVISION_THRESHOLD, "institution_id"
        ].nunique()
        print(f"\n  Policy revision events (|ΔPCI| > {REVISION_THRESHOLD:.3f}): "
              f"{n_rev} obs ({pct:.1f}%)  across {n_inst_rev} institutions")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TABLE 2 — MAIN RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def table2(df):
    print("\n" + "=" * 74)
    print("TABLE 2: Main Regression Results")
    print("=" * 74)

    OUTCOMES = [
        ("ln_new_patent_apps",   "Ln(New Patent Applications)"),
        ("ln_total_patent_apps", "Ln(Total Patent Applications)"),
        ("ln_patents_issued",    "Ln(Patents Issued)"),
        ("ln_disclosures",       "Ln(Disclosures)"),
        ("ln_licenses",          "Ln(Licenses)"),
        ("ln_startups",          "Ln(Startups)"),
        ("ln_license_income",    "Ln(License Income)"),
    ]

    print("\n  Panel A: TWFE, PCI lagged one year, seven independent outcomes")
    print(f"  {'Outcome':<38} {'b':>9}  {'SE':>7} {'p':>7} "
          f"{'95% CI':>18} {'N':>6}")
    print("  " + "-" * 88)

    results = {}
    for var, label in OUTCOMES:
        if var not in df.columns:
            continue
        r = twfe(df, var)
        results[var] = r
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(f"  {label:<38} {r['beta']:>+9.3f}{_stars(r['p']):<3} "
              f"{r['se']:>7.3f} {r['p']:>7.3f} {ci:>18} {r['n']:>6,}")

    print("\n  Panel B: Specification checks — Ln(New Patent Applications)")
    print(f"  {'Specification':<38} {'b':>9}  {'SE':>7} {'p':>7} {'N':>6}")
    print("  " + "-" * 72)

    specs = [
        ("pci",           "Contemporaneous PCI"),
        ("pci_l1",        "Baseline: lag-1 PCI"),
        ("pci_l2",        "Lag-2 PCI"),
        ("pci_l3",        "Lag-3 PCI"),
        ("pci_median_l1", "Median PCI (lagged)"),
    ]
    for tvar, label in specs:
        if tvar not in df.columns:
            continue
        r = twfe(df, "ln_new_patent_apps", treatment=tvar)
        print(f"  {label:<38} {r['beta']:>+9.3f}{_stars(r['p']):<3} "
              f"{r['se']:>7.3f} {r['p']:>7.3f} {r['n']:>6,}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVENT STUDY
# ─────────────────────────────────────────────────────────────────────────────

def event_study(df, outcomes=None, window=EVENT_WIN, plot=True,
                out_path="figure1_event_study.png"):
    """
    Estimate and (optionally) plot the event-study model around large positive
    policy revisions.

    Model:
        y_it = α_i + γ_t + Σ_{k≠-1} β_k event_{k} + X_it δ + ε_it

    where event_{k} = 1[year − revision_year == k] for treated institutions,
    and event_{-1} = omitted baseline.

    Parameters
    ----------
    df       : prepared DataFrame (from load_and_prepare)
    outcomes : list of (col, label) pairs; defaults to patent apps + disclosures
    window   : int — number of years before/after to show
    plot     : bool — save figure to out_path
    out_path : str — file path for figure

    Returns
    -------
    dict keyed by outcome column, each containing lists of betas and SEs
    by event time.
    """
    if outcomes is None:
        outcomes = [
            ("ln_new_patent_apps", "Ln(New Patent Applications)"),
            ("ln_disclosures",     "Ln(Disclosures)"),
        ]

    event_dummies = [f"event_{k}" for k in range(-window, window + 1)
                     if k != -1 and f"event_{k}" in df.columns]

    if not event_dummies:
        print("No event dummies found. Check REVISION_THRESHOLD.")
        return {}

    # Print counts
    treated = df["revision_year"].notna().sum()
    n_inst  = df.loc[df["revision_year"].notna(), "institution_id"].nunique()
    print("\n" + "=" * 74)
    print("EVENT STUDY: Pre/Post Coefficients Around Policy Revisions")
    print("=" * 74)
    print(f"  Treated obs: {treated:,}  |  Treated institutions: {n_inst}")
    print(f"  Revision threshold: |ΔPCI| > {REVISION_THRESHOLD:.3f}  "
          f"(positive revisions only)")
    print(f"  Window: −{window} to +{window} years  |  Baseline: year −1\n")

    all_results = {}

    for var, label in outcomes:
        if var not in df.columns:
            continue

        r = twfe(df, var,
                 treatment=event_dummies[0],
                 controls=("ln_research_exp_l1", "ln_licensing_ftes_l1"),
                 extra_regressors=tuple(event_dummies[1:]))

        ks     = []
        betas  = []
        ses    = []
        p_vals = []

        for k in range(-window, window + 1):
            ks.append(k)
            if k == -1:
                betas.append(0.0)
                ses.append(0.0)
                p_vals.append(1.0)
            else:
                key = f"event_{k}"
                b   = r["all_betas"].get(key, float("nan"))
                se  = r["all_ses"].get(key, float("nan"))
                betas.append(b)
                ses.append(se)
                p_val = (2 * (1 - stats.t.cdf(abs(b / se), df=n_inst - 1))
                         if se > 0 else float("nan"))
                p_vals.append(p_val)

        all_results[var] = dict(ks=ks, betas=betas, ses=ses, p_vals=p_vals,
                                label=label, n=r["n"])

        # Print table
        print(f"  {label}  (N = {r['n']:,})")
        print(f"  {'k':>5} {'beta':>9} {'SE':>8} {'p':>8}  CI")
        print("  " + "-" * 52)
        for k, b, se, pv in zip(ks, betas, ses, p_vals):
            if k == -1:
                print(f"  {k:>5}   (baseline = 0)")
                continue
            ci = f"[{b-1.96*se:+.3f}, {b+1.96*se:+.3f}]"
            print(f"  {k:>5} {b:>+9.3f}{_stars(pv):<3} {se:>8.3f} "
                  f"{pv:>8.3f}  {ci}")

        # Pre-trend joint test (k = −window … −2)
        pre_ks    = [k for k in range(-window, -1)]
        pre_betas = [betas[k + window] for k in pre_ks]
        pre_ses   = [ses[k + window]   for k in pre_ks]
        if all(not np.isnan(b) and s > 0 for b, s in zip(pre_betas, pre_ses)):
            chi2 = sum((b / s) ** 2 for b, s in zip(pre_betas, pre_ses))
            pt   = 1 - stats.chi2.cdf(chi2, df=len(pre_ks))
            print(f"\n  Pre-trend joint test (k={pre_ks[0]}...{pre_ks[-1]}): "
                  f"χ²({len(pre_ks)}) = {chi2:.2f},  p = {pt:.3f}"
                  f"  {'[PASS ✓]' if pt > 0.10 else '[FAIL ✗]'}")
        print()

    # ── Figure ──────────────────────────────────────────────────────────────
    if plot and all_results:
        _plot_event_study(all_results, window, out_path)

    return all_results


def _plot_event_study(results, window, out_path):
    """
    Two-panel figure: left = patent applications, right = disclosures.
    Each panel shows point estimates with 95% CIs.
    Shaded region = pre-period. Vertical dashed line at k = 0.
    """
    outcomes = list(results.keys())
    n_panels = len(outcomes)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    colors = {"ln_new_patent_apps": "#1a5276",
              "ln_disclosures":     "#922b21"}
    default_color = "#2c3e50"

    for ax, var in zip(axes, outcomes):
        res   = results[var]
        ks    = np.array(res["ks"])
        betas = np.array(res["betas"])
        ses   = np.array(res["ses"])

        color = colors.get(var, default_color)

        # Shaded pre-period
        ax.axvspan(-window - 0.5, -0.5, alpha=0.07, color="grey",
                   label="Pre-revision")

        # Zero line
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Vertical line at revision
        ax.axvline(0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)

        # Plot CIs and points (skip baseline k = −1 from errorbars)
        for k, b, se in zip(ks, betas, ses):
            if k == -1:
                ax.plot(k, 0, "s", color="grey", markersize=6, zorder=5)
                continue
            ax.errorbar(k, b, yerr=1.96 * se, fmt="o",
                        color=color, markersize=5, linewidth=1.4,
                        capsize=3, capthick=1.2, zorder=5)

        # Connect points with a thin line (excluding baseline)
        mask = ks != -1
        sorted_idx = np.argsort(ks[mask])
        ax.plot(ks[mask][sorted_idx], betas[mask][sorted_idx],
                "-", color=color, linewidth=1.1, alpha=0.7, zorder=4)

        ax.set_xlabel("Years relative to policy revision", fontsize=10)
        ax.set_ylabel("Coefficient (log outcome)", fontsize=10)
        ax.set_title(res["label"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(range(-window, window + 1))
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Event Study: Patent Applications and Disclosures\n"
        f"(window ±{window} years around first large positive PCI revision; "
        f"baseline = year −1)",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RANDOMIZATION INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def randomization_inference(df, outcome="ln_new_patent_apps",
                             treatment="pci_l1", n_perms=2000):
    """
    Within-institution permutation test.
    Permutes the PCI time series within each institution, destroying temporal
    ordering while preserving distributional properties.
    """
    global LAST_RI_P
    print("\n" + "=" * 74)
    print(f"RANDOMIZATION INFERENCE  (seed={SEED}, n_perms={n_perms})")
    print("=" * 74)
    rng  = np.random.default_rng(SEED)
    obs  = twfe(df, outcome, treatment)["beta"]

    cols = [outcome, treatment, "ln_research_exp_l1", "ln_licensing_ftes_l1",
            "institution_id", "year"]
    base = df[[c for c in cols if c in df.columns]].dropna().copy()

    perm_betas = []
    for i in range(n_perms):
        perm = base.copy()
        perm[treatment] = perm.groupby("institution_id")[treatment].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        b = twfe(perm, outcome, treatment)["beta"]
        if not np.isnan(b):
            perm_betas.append(b)
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1} permutations")

    perm_betas = np.array(perm_betas)
    ri_p = np.mean(np.abs(perm_betas) >= np.abs(obs))
    LAST_RI_P = ri_p

    print(f"\n  Observed beta:       {obs:+.4f}")
    print(f"  Permutations:        {len(perm_betas)}")
    print(f"  RI p-value:          {ri_p:.4f}")
    print(f"  Pct perm |beta| >= obs: {100 * ri_p:.2f}%")
    return ri_p


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TABLE 3 — HETEROGENEITY + MUNDLAK
# ─────────────────────────────────────────────────────────────────────────────

def table3(df):
    print("\n" + "=" * 74)
    print("TABLE 3: Heterogeneity and Mundlak Decomposition")
    print("=" * 74)

    SUBGROUPS = [
        ("private == 1",     "Private"),
        ("private == 0",     "Public"),
        ("carnegie_r1 == 1", "Carnegie R1"),
        ("carnegie_r1 == 0", "Non-R1"),
        ("med_school == 1",  "Medical School"),
        ("med_school == 0",  "No Medical School"),
        ("land_grant == 1",  "Land Grant"),
        ("land_grant == 0",  "Non-Land-Grant"),
    ]

    print("\n  Panel A: Split-sample heterogeneity")
    print(f"  {'Subgroup':<24}  "
          f"{'PatApps b':>9} {'SE':>7} {'p':>6}  "
          f"{'ConvRate b':>10} {'SE':>7} {'p':>6}")
    print("  " + "-" * 84)

    for query, label in SUBGROUPS:
        try:
            sub = df.query(query)
        except Exception:
            continue
        r1 = twfe(sub, "ln_new_patent_apps")
        r2 = (twfe(sub, "conv_rate")
              if "conv_rate" in df.columns
              else {"beta": float("nan"), "se": float("nan"),
                    "p": float("nan")})
        print(f"  {label:<24}  "
              f"{r1['beta']:>+9.3f}{_stars(r1['p']):<3} "
              f"{r1['se']:>7.3f} {r1['p']:>6.3f}  "
              f"{r2['beta']:>+10.3f}{_stars(r2['p']):<3} "
              f"{r2['se']:>7.3f} {r2['p']:>6.3f}")

    MUNDLAK_OUTCOMES = [
        ("ln_new_patent_apps", "Ln(Patent Applications)"),
        ("ln_disclosures",     "Ln(Disclosures)"),
        ("conv_rate",          "Conv. Rate"),
        ("ln_licenses",        "Ln(Licenses)"),
        ("ln_startups",        "Ln(Startups)"),
        ("ln_license_income",  "Ln(License Income)"),
    ]

    print("\n  Panel B: Mundlak decomposition")
    print(f"  {'Outcome':<30} {'Within b':>10} {'p':>6}  "
          f"{'Between b':>10} {'p':>6}")
    print("  " + "-" * 70)
    for var, label in MUNDLAK_OUTCOMES:
        if var not in df.columns:
            continue
        rw = twfe(df, var, treatment="pci_within")
        rb = twfe(df, var, treatment="pci_mean")
        print(f"  {label:<30} "
              f"{rw['beta']:>+10.3f}{_stars(rw['p']):<3} {rw['p']:>6.3f}  "
              f"{rb['beta']:>+10.3f}{_stars(rb['p']):<3} {rb['p']:>6.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TABLE 4 — CHANNEL ANALYSIS + MULTIPLE TESTING
# ─────────────────────────────────────────────────────────────────────────────

def table4(df):
    print("\n" + "=" * 74)
    print("TABLE 4: Channel Analysis and Multiple Testing")
    print("=" * 74)

    # Panel A: conversion rate
    print("\n  Panel A: Conversion rate (Ln PatApp − Ln Discl)")
    if "conv_rate" in df.columns:
        r = twfe(df, "conv_rate")
        print(f"  PCI lag-1:  b = {r['beta']:+.3f}{_stars(r['p'])}"
              f"  SE = {r['se']:.3f}  p = {r['p']:.3f}  N = {r['n']:,}")

    SUBIDX = [
        ("tone_index_l1",       "Tone Index"),
        ("clarity_index_l1",    "Clarity Index"),
        ("legal_load_index_l1", "Legal Load Index"),
    ]
    avail = [t for t, _ in SUBIDX if t in df.columns]

    for out_var, out_label in [
        ("ln_new_patent_apps", "Ln(New Patent Applications)"),
        ("conv_rate",          "Conversion Rate"),
    ]:
        if out_var not in df.columns:
            continue

        print(f"\n  Panel B — sequential sub-index, outcome: {out_label}")
        print(f"  {'Sub-index':<28} {'b':>9}  {'SE':>7} {'p':>7} {'N':>6}")
        print("  " + "-" * 62)
        for tvar, tlabel in SUBIDX:
            if tvar not in df.columns:
                continue
            r = twfe(df, out_var, treatment=tvar)
            print(f"  {tlabel:<28} {r['beta']:>+9.3f}{_stars(r['p']):<3} "
                  f"{r['se']:>7.3f} {r['p']:>7.3f} {r['n']:>6,}")

        print(f"\n  Panel C — joint sub-index, outcome: {out_label}")
        print(f"  {'Sub-index':<28} {'b':>9}  {'SE':>7} {'p':>7}")
        print("  " + "-" * 54)
        base_ctrl = ("ln_research_exp_l1", "ln_licensing_ftes_l1")
        for tvar, tlabel in SUBIDX:
            if tvar not in avail:
                continue
            others = tuple(t for t in avail if t != tvar)
            r = twfe(df, out_var, treatment=tvar,
                     controls=base_ctrl + others)
            print(f"  {tlabel:<28} {r['beta']:>+9.3f}{_stars(r['p']):<3} "
                  f"{r['se']:>7.3f} {r['p']:>7.3f}")

    # Panel D: multiple testing
    OUTCOMES_MT = [
        ("ln_new_patent_apps",   "Ln(New Patent Applications)"),
        ("ln_total_patent_apps", "Ln(Total Patent Applications)"),
        ("ln_patents_issued",    "Ln(Patents Issued)"),
        ("ln_disclosures",       "Ln(Disclosures)"),
        ("ln_licenses",          "Ln(Licenses)"),
        ("ln_startups",          "Ln(Startups)"),
        ("ln_license_income",    "Ln(License Income)"),
    ]
    raw_ps, labels = [], []
    for var, label in OUTCOMES_MT:
        if var not in df.columns:
            continue
        raw_ps.append(twfe(df, var)["p"])
        labels.append(label)

    if raw_ps:
        _, bonf, _, _ = multipletests(raw_ps, method="bonferroni")
        _, bh,   _, _ = multipletests(raw_ps, method="fdr_bh")
        print(f"\n  Panel D: Multiple testing corrections")
        print(f"  {'Outcome':<38} {'Raw p':>7} {'Bonf.':>8} {'BH':>7}")
        print("  " + "-" * 64)
        for lbl, rp, bp, bhp in zip(labels, raw_ps, bonf, bh):
            print(f"  {lbl:<38} {rp:>7.3f} {bp:>8.3f} {bhp:>7.3f}")
        if LAST_RI_P is not None:
            print(f"  Note: RI p-value for Ln(New Patent Applications) = {LAST_RI_P:.3f} "
                  "(not subject to multiple comparison adjustment)")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  APPENDIX TABLE A1 — CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def appendix_a1(df):
    print("\n" + "=" * 74)
    print("APPENDIX TABLE A1: Sub-Index Correlation Matrix (lagged one year)")
    print("=" * 74)
    vv = ["pci_l1", "tone_index_l1", "clarity_index_l1", "legal_load_index_l1"]
    ll = ["PCI", "Tone", "Clarity", "Legal Load"]
    avail = [(v, l) for v, l in zip(vv, ll) if v in df.columns]
    if not avail:
        print("  Sub-index columns not found.")
        return
    vs, ls = zip(*avail)
    C = df[list(vs)].corr()
    C.index = ls
    C.columns = ls
    print(f"\n  {'':16}" + "".join(f"{l:>12}" for l in ls))
    for row in ls:
        print(f"  {row:<16}" + "".join(f"{C.loc[row, c]:>12.3f}" for c in ls))


# ─────────────────────────────────────────────────────────────────────────────
# 10. APPENDIX TABLE B1 — ROBUSTNESS SUITE
# ─────────────────────────────────────────────────────────────────────────────

def appendix_b1(df):
    print("\n" + "=" * 74)
    print("APPENDIX TABLE B1: Robustness Suite")
    print("=" * 74)

    hdr = (f"  {'Specification':<46} {'b':>9}  {'SE':>7} "
           f"{'p':>7} {'N':>6}")
    sep = "  " + "-" * 76

    def _row(label, r):
        print(f"  {label:<46} {r['beta']:>+9.3f}{_stars(r['p']):<3} "
              f"{r['se']:>7.3f} {r['p']:>7.3f} {r['n']:>6,}")

    base = twfe(df, "ln_new_patent_apps")

    print(f"\n  Panel A: Ln(New Patent Applications)")
    print(hdr); print(sep)
    _row("Baseline (lag-1 PCI)", base)

    # Winsorized
    df_w = df.copy()
    for col in ["ln_new_patent_apps", "pci_l1",
                "ln_research_exp_l1", "ln_licensing_ftes_l1"]:
        if col not in df_w.columns:
            continue
        lo, hi = df_w[col].quantile([WIN_TRIM, 1 - WIN_TRIM])
        df_w[col] = df_w[col].clip(lo, hi)
    _row(f"Winsorized {int(100*WIN_TRIM)}st/99th percentiles",
         twfe(df_w, "ln_new_patent_apps"))

    # Balanced panel
    cnt  = df.groupby("institution_id")["year"].count()
    keep = cnt[cnt >= 15].index
    df_b = df[df["institution_id"].isin(keep)]
    _row("Balanced panel (≥15 obs)", twfe(df_b, "ln_new_patent_apps"))

    # R1 only
    if "carnegie_r1" in df.columns:
        _row("R1 universities only",
             twfe(df[df["carnegie_r1"] == 1], "ln_new_patent_apps"))

    # HC3
    _row("HC3 standard errors",
         twfe(df, "ln_new_patent_apps", se_type="hc3"))

    # Lagged outcome (Nickell-biased lower bound)
    if "ln_new_patent_apps_lag" in df.columns:
        _row("With lagged outcome (Nickell-biased LB)",
             twfe(df, "ln_new_patent_apps",
                  controls=("ln_research_exp_l1", "ln_licensing_ftes_l1",
                             "ln_new_patent_apps_lag")))

    # Quadratic time trend
    _row("Common quadratic time trend",
         twfe(df, "ln_new_patent_apps",
              controls=("ln_research_exp_l1", "ln_licensing_ftes_l1",
                        "year_dm", "year2_dm")))

    # Falsification: PCI → future research expenditure
    if "ln_research_exp_fwd" in df.columns:
        rf = twfe(df, "ln_research_exp_fwd")
        print(f"  {'Falsification: PCI → Ln(Future Res Exp)':<46} "
              f"{rf['beta']:>+9.3f}{_stars(rf['p']):<3} "
              f"{rf['se']:>7.3f} {rf['p']:>7.3f} {rf['n']:>6,}")

    # Leave-one-out
    insts   = df["institution_id"].unique()
    loo     = []
    for inst in insts:
        b = twfe(df[df["institution_id"] != inst],
                 "ln_new_patent_apps")["beta"]
        if not np.isnan(b):
            loo.append(b)
    if loo:
        loo = np.array(loo)
        print(f"\n  Leave-one-out range:   [{loo.min():.3f}, {loo.max():.3f}]"
              f"  (N = {len(loo)} institutions)")
        print(f"  Pct positive:          {100 * np.mean(loo > 0):.1f}%")

    # Panel B: Conversion Rate
    if "conv_rate" not in df.columns:
        return
    print(f"\n  Panel B: Conversion Rate")
    print(hdr); print(sep)
    _row("Baseline (lag-1 PCI)", twfe(df, "conv_rate"))
    _row(f"Winsorized {int(100*WIN_TRIM)}st/99th percentiles",
         twfe(df_w, "conv_rate"))
    _row("Balanced panel (≥15 obs)", twfe(df_b, "conv_rate"))
    if "carnegie_r1" in df.columns:
        _row("R1 universities only",
             twfe(df[df["carnegie_r1"] == 1], "conv_rate"))
    if "pci_l2" in df.columns:
        _row("Lag-2 PCI", twfe(df, "conv_rate", treatment="pci_l2"))


# ─────────────────────────────────────────────────────────────────────────────
# 11. APPENDIX TABLE B2 — LAG SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def appendix_b2(df):
    print("\n" + "=" * 74)
    print("APPENDIX TABLE B2: Lag Sensitivity — All Outcomes")
    print("=" * 74)

    OUTCOMES = [
        ("ln_new_patent_apps",   "Ln(New Patent Apps)"),
        ("ln_total_patent_apps", "Ln(Total Patent Apps)"),
        ("ln_patents_issued",    "Ln(Patents Issued)"),
        ("ln_disclosures",       "Ln(Disclosures)"),
        ("conv_rate",            "Conv. Rate"),
        ("ln_licenses",          "Ln(Licenses)"),
        ("ln_startups",          "Ln(Startups)"),
        ("ln_license_income",    "Ln(License Income)"),
    ]
    LAGS = [("pci",    "Contemp."),
            ("pci_l1", "Lag 1"),
            ("pci_l2", "Lag 2"),
            ("pci_l3", "Lag 3")]

    hdr = f"  {'Outcome':<28}" + "".join(f"  {ln:>14}" for _, ln in LAGS)
    print(hdr)
    print("  " + "-" * 86)

    for var, label in OUTCOMES:
        if var not in df.columns:
            continue
        row = f"  {label:<28}"
        for tvar, _ in LAGS:
            if tvar not in df.columns:
                row += f"  {'---':>14}"
                continue
            r    = twfe(df, var, treatment=tvar)
            cell = f"{r['beta']:>+8.3f}{_stars(r['p'])}"
            row += f"  {cell:>14}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 74)
    print("REPLICATION — Policy Communication and Technology Transfer")
    print("Sharma, Wang, Basnet, Cossco (2026)")
    print("=" * 74)

    try:
        df = load_and_prepare("merged_autm.csv")
    except FileNotFoundError:
        sys.exit(
            "\nERROR: merged_autm.csv not found.\n"
            "Place the file in the working directory and re-run.\n"
        )

    print(f"\nSample summary:")
    print(f"  Institutions  :  {df['institution_id'].nunique()}")
    print(f"  Year range    :  {df['year'].min()} – {df['year'].max()}")
    print(f"  Raw obs       :  {len(df):,}")
    if "pci" in df.columns:
        print(f"  Obs with PCI  :  {df['pci'].notna().sum():,}")
    if "revision_year" in df.columns:
        n_treated = df["revision_year"].notna().sum()
        n_inst_t  = df.loc[
            df["revision_year"].notna(), "institution_id"
        ].nunique()
        print(f"  Treated obs   :  {n_treated:,}  "
              f"({n_inst_t} institutions with large positive revision)")

    # ── Tables ───────────────────────────────────────────────────────────────
    table1(df)
    table2(df)
    table3(df)

    # ── Event Study ──────────────────────────────────────────────────────────
    es_results = event_study(
        df,
        outcomes=[
            ("ln_new_patent_apps", "Ln(New Patent Applications)"),
            ("ln_disclosures",     "Ln(Disclosures)"),
        ],
        window=EVENT_WIN,
        plot=True,
        out_path="figure1_event_study.png",
    )

    # ── Randomization Inference ───────────────────────────────────────────────
    # Set n_perms=200 for a quick check; use 2000 for paper
    print("\n[Tip: set n_perms=200 for a quick check, 2000 for paper-quality RI]")
    ri_p = randomization_inference(df, n_perms=2000)

    # ── Channel + Multiple Testing ────────────────────────────────────────────
    table4(df)

    # ── Appendices ────────────────────────────────────────────────────────────
    appendix_a1(df)
    appendix_b1(df)
    appendix_b2(df)

    print("\n" + "=" * 74)
    print("Replication complete.")
    print(f"Event study figure → figure1_event_study.png")
    print("=" * 74)

