"""
replication.py
==============
Policy Communication and Technology Transfer:
Evidence from University Intellectual Property Governance Documents
Sharma, Wang, Basnet, Cossco (2026)

Produces all tables and analyses in the paper:

  TABLE 1   — Descriptive Statistics
  TABLE 2   — Main TWFE Results + Lag Sensitivity
  TABLE 3   — Heterogeneity (split-sample)
  TABLE 4   — Channel Analysis + Multiple Testing
  APP A1    — Sub-Index Correlation Matrix
  APP B1    — Robustness Suite (winsorize, balanced, R1, HC3, Nickell, quad trend,
               falsification, leave-one-out)
  APP B2    — Negative-binomial count-model robustness
  APP B3    — Lag Sensitivity across all outcomes

Requirements:
  pip install pandas numpy scipy statsmodels

Data file:  data/merged_autm.csv

Column mapping (raw AUTM/PCI names → analysis names) is handled automatically.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats, linalg
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson
import warnings
warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PACKAGE_ROOT / "data" / "merged_autm.csv"

SEED        = 42
WIN_TRIM    = 0.01             # winsorize at 1st / 99th percentile
# Revision threshold: abs(dPCI) must exceed this to count as a policy revision
# Set to 75th-percentile of non-trivial absolute changes ≈ 0.054
REVISION_THRESHOLD = 0.05
LAST_PLACEBO_P = None
BASE_CONTROLS = (
    "ln_research_exp_l1",
    "ln_licensing_ftes_l1",
    "royalty_share_l1",
)

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

def _resolve_data_path(path=None):
    """
    Resolve the analysis data from the replication package.
    """
    candidates = [DEFAULT_DATA_PATH]
    if path is not None:
        candidates.insert(0, Path(path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No AUTM/PCI merged CSV found. Expected {DEFAULT_DATA_PATH}."
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


def load_and_prepare(path=None):
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

    # ── calendar-year lags (k = 1, ..., 5) ────────────────────────────────────
    lag_src = [
        "pci", "pci_median",
        "tone_index", "clarity_index", "legal_load_index",
        "ln_research_exp", "ln_licensing_ftes", "royalty_share", "tlo_age",
    ]
    if df.duplicated(["institution_id", "year"]).any():
        raise ValueError("institution_id-year keys must be unique to build lags")
    for v in lag_src:
        if v not in df.columns:
            continue
        lookup = df.set_index(["institution_id", "year"])[v]
        for k in [1, 2, 3, 4, 5]:
            lag_index = pd.MultiIndex.from_arrays([
                df["institution_id"],
                df["year"] - k,
            ])
            df[f"{v}_l{k}"] = lookup.reindex(lag_index).to_numpy()

    # ── additional robustness variables ───────────────────────────────────
    if "ln_new_patent_apps" in df.columns:
        lookup = df.set_index(["institution_id", "year"])["ln_new_patent_apps"]
        lag_index = pd.MultiIndex.from_arrays([
            df["institution_id"],
            df["year"] - 1,
        ])
        df["ln_new_patent_apps_lag"] = lookup.reindex(lag_index).to_numpy()

    if "ln_research_exp" in df.columns:
        lookup = df.set_index(["institution_id", "year"])["ln_research_exp"]
        lead_index = pd.MultiIndex.from_arrays([
            df["institution_id"],
            df["year"] + 1,
        ])
        df["ln_research_exp_fwd"] = lookup.reindex(lead_index).to_numpy()

    # Consecutive-year policy-score changes used for descriptive revision counts.
    df["pci_change"] = df["pci"] - df["pci_l1"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CORE ESTIMATOR: TWO-WAY FE WITH CLUSTERED OR HC3 STANDARD ERRORS
# ─────────────────────────────────────────────────────────────────────────────

def _iterative_within_transform(sub, columns, entity, time,
                                tol=1e-10, max_iter=10000):
    """
    Absorb entity and time fixed effects by alternating projections.

    Unlike one-pass double demeaning, this is exact for unbalanced panels.
    """
    values = sub[list(columns)].to_numpy(dtype=float)
    entity_codes, entities = pd.factorize(sub[entity], sort=False)
    time_codes, periods = pd.factorize(sub[time], sort=False)

    def demean(x, codes, n_groups):
        sums = np.zeros((n_groups, x.shape[1]))
        np.add.at(sums, codes, x)
        counts = np.bincount(codes, minlength=n_groups).astype(float)
        return x - sums[codes] / counts[codes, None]

    for _ in range(max_iter):
        previous = values.copy()
        values = demean(values, entity_codes, len(entities))
        values = demean(values, time_codes, len(periods))
        if np.max(np.abs(values - previous)) < tol:
            return values

    raise RuntimeError("Fixed-effect absorption did not converge")


def twfe(df, outcome, treatment="pci_l1",
         controls=BASE_CONTROLS,
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
    extra_regressors : tuple of str — additional right-hand side variables
                       used for joint specifications

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

    x_names = [treatment] + list(controls) + list(extra_regressors)
    transformed = _iterative_within_transform(
        sub, reg_vars, entity=entity, time=time
    )
    y = transformed[:, 0]
    X_full = transformed[:, 1:]
    norms = np.linalg.norm(X_full, axis=0)
    rank_tol = max(1e-10, norms.max(initial=0) * 1e-10)
    active = norms > rank_tol
    if not active[0]:
        nan = float("nan")
        return dict(beta=nan, se=nan, p=nan, n=len(sub),
                    ci_lo=nan, ci_hi=nan, t=nan,
                    all_betas={}, all_ses={})
    active_names = [name for name, keep in zip(x_names, active) if keep]
    X = X_full[:, active]
    n, k  = X.shape

    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat

    # ── variance estimator ─────────────────────────────────────────────────
    if se_type == "hc3":
        # HC3: leave-one-out leverage correction
        bread = np.linalg.pinv(X.T @ X)
        H_diag = np.einsum("ij,jk,ik->i", X, bread, X)
        e_adj  = resid / (1.0 - np.clip(H_diag, None, 0.9999))
        meat   = (X * e_adj[:, None]).T @ (X * e_adj[:, None])
        V      = bread @ meat @ bread
        df_t   = n - k
    else:
        # Clustered SE (Cameron & Miller 2015)
        clusters = sub[entity].values
        uniq     = np.unique(clusters)
        G        = len(uniq)
        bread    = np.linalg.pinv(X.T @ X)
        meat     = np.zeros((k, k))
        for g in uniq:
            idx = clusters == g
            sg  = X[idx].T @ resid[idx]
            meat += np.outer(sg, sg)
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        V  = correction * bread @ meat @ bread
        df_t = G - 1

    se_vec = np.sqrt(np.diag(V).clip(0))

    # Treatment is the first transformed regressor.
    b, se_b = beta_hat[active_names.index(treatment)], se_vec[active_names.index(treatment)]
    t_stat  = b / se_b if se_b > 0 else float("nan")
    p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_t))

    # All coefficients for joint specifications
    all_betas = {v: float("nan") for v in x_names}
    all_ses   = {v: float("nan") for v in x_names}
    all_betas.update({v: beta_hat[i] for i, v in enumerate(active_names)})
    all_ses.update({v: se_vec[i] for i, v in enumerate(active_names)})
    all_ps    = {}
    all_ci_lo = {}
    all_ci_hi = {}
    for v in x_names:
        b_v = all_betas[v]
        se_v = all_ses[v]
        t_v = b_v / se_v if se_v > 0 else float("nan")
        p_v = 2 * (1 - stats.t.cdf(abs(t_v), df=df_t)) if se_v > 0 else float("nan")
        all_ps[v] = p_v
        all_ci_lo[v] = b_v - 1.96 * se_v if se_v > 0 else float("nan")
        all_ci_hi[v] = b_v + 1.96 * se_v if se_v > 0 else float("nan")

    tss = np.sum((y - y.mean()) ** 2)
    rss = np.sum(resid ** 2)
    r2 = 1 - rss / tss if tss > 0 else float("nan")

    return dict(
        beta=b, se=se_b, p=p_val, n=n,
        ci_lo=b - 1.96 * se_b,
        ci_hi=b + 1.96 * se_b,
        t=t_stat,
        r2=r2,
        all_betas=all_betas,
        all_ses=all_ses,
        all_ps=all_ps,
        all_ci_lo=all_ci_lo,
        all_ci_hi=all_ci_hi,
    )


def poisson_fe(df, outcome, treatment="pci_l1",
               controls=BASE_CONTROLS,
               entity="institution_id", time="year",
               maxiter=200):
    """
    Estimate a Poisson pseudo-maximum-likelihood model with institution and
    year fixed effects implemented via dummies and cluster-robust SEs.

    Parameters
    ----------
    outcome   : str - raw count outcome, not log transformed
    treatment : str - main regressor
    controls  : tuple of str - controls to include

    Returns
    -------
    dict with beta, se, p, n, ci_lo, ci_hi, converged
    """
    keep = [outcome, treatment, *controls, entity, time]
    keep = [c for c in keep if c in df.columns]
    sub = df[keep].dropna().copy()
    if len(sub) < 20:
        nan = float("nan")
        return dict(beta=nan, se=nan, p=nan, n=0,
                    ci_lo=nan, ci_hi=nan, converged=False)

    rhs = [treatment, *controls]
    X = sub[rhs].copy()
    X = pd.concat([
        X,
        pd.get_dummies(sub[entity], prefix="inst", drop_first=True, dtype=float),
        pd.get_dummies(sub[time].astype(int), prefix="yr", drop_first=True, dtype=float),
    ], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(sub[outcome], errors="coerce").astype(float)

    res = sm.GLM(y, X, family=sm.families.Poisson()).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub[entity]},
        maxiter=maxiter,
    )
    b = res.params[treatment]
    se_b = res.bse[treatment]
    p_val = res.pvalues[treatment]
    return dict(
        beta=b, se=se_b, p=p_val, n=len(sub),
        ci_lo=b - 1.96 * se_b,
        ci_hi=b + 1.96 * se_b,
        converged=bool(getattr(res, "converged", True)),
    )


def nb_fe(df, outcome, treatment="pci_l1",
          controls=BASE_CONTROLS,
          entity="institution_id", time="year",
          maxiter=300):
    """
    Estimate a negative binomial FE model via entity and year dummies with
    cluster-robust SEs.
    """
    keep = [outcome, treatment, *controls, entity, time]
    keep = [c for c in keep if c in df.columns]
    sub = df[keep].dropna().copy()
    if len(sub) < 20:
        nan = float("nan")
        return dict(beta=nan, se=nan, p=nan, n=0,
                    ci_lo=nan, ci_hi=nan, converged=False, alpha=nan)

    rhs = [treatment, *controls]
    X = sub[rhs].copy()
    X = pd.concat([
        X,
        pd.get_dummies(sub[entity], prefix="inst", drop_first=True, dtype=float),
        pd.get_dummies(sub[time].astype(int), prefix="yr", drop_first=True, dtype=float),
    ], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(sub[outcome], errors="coerce").astype(float)

    res = NegativeBinomial(y, X).fit(
        disp=0,
        maxiter=maxiter,
        cov_type="cluster",
        cov_kwds={"groups": sub[entity]},
    )
    b = res.params[treatment]
    se_b = res.bse[treatment]
    p_val = res.pvalues[treatment]
    return dict(
        beta=b, se=se_b, p=p_val, n=len(sub),
        ci_lo=b - 1.96 * se_b,
        ci_hi=b + 1.96 * se_b,
        converged=bool(getattr(res, "mle_retvals", {}).get("converged", True)),
        alpha=res.params.get("alpha", float("nan")),
    )


def zip_fe(df, outcome, treatment="pci_l1",
           controls=BASE_CONTROLS,
           entity="institution_id", time="year",
           maxiter=300):
    """
    Estimate a zero-inflated Poisson model with an intercept-only inflation
    equation. This is included as a sensitivity check, not a preferred model,
    because the main patent outcome has very little zero mass.
    """
    keep = [outcome, treatment, *controls, entity, time]
    keep = [c for c in keep if c in df.columns]
    sub = df[keep].dropna().copy()
    if len(sub) < 20:
        nan = float("nan")
        return dict(beta=nan, se=nan, p=nan, n=0,
                    ci_lo=nan, ci_hi=nan, converged=False,
                    inflate_const=nan)

    rhs = [treatment, *controls]
    X = sub[rhs].copy()
    X = pd.concat([
        X,
        pd.get_dummies(sub[entity], prefix="inst", drop_first=True, dtype=float),
        pd.get_dummies(sub[time].astype(int), prefix="yr", drop_first=True, dtype=float),
    ], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(sub[outcome], errors="coerce").astype(float)
    infl = pd.DataFrame({"inflate_const": 1.0}, index=sub.index)

    res = ZeroInflatedPoisson(
        endog=y,
        exog=X,
        exog_infl=infl,
        inflation="logit",
    ).fit(
        method="bfgs",
        maxiter=maxiter,
        disp=0,
        cov_type="cluster",
        cov_kwds={"groups": sub[entity]},
    )
    b = res.params[treatment]
    se_b = res.bse[treatment]
    p_val = res.pvalues[treatment]
    return dict(
        beta=b, se=se_b, p=p_val, n=len(sub),
        ci_lo=b - 1.96 * se_b,
        ci_hi=b + 1.96 * se_b,
        converged=bool(getattr(res, "mle_retvals", {}).get("converged", True)),
        inflate_const=res.params.get("inflate_const", float("nan")),
    )


def zero_share_table(df):
    """
    Compute zero shares for the baseline lag-1 estimation sample of each
    outcome. This directly addresses sensitivity of ln(1+y) transforms to
    mass at zero.
    """
    outcomes = [
        ("new_patent_apps",   "New Patent Applications"),
        ("total_patent_apps", "Total Patent Applications"),
        ("patents_issued",    "Patents Issued"),
        ("disclosures",       "Invention Disclosures"),
        ("licenses",          "Licenses Executed"),
        ("startups",          "Startups Formed"),
        ("license_income",    "License Income"),
    ]
    rows = []
    required = ["pci_l1", *BASE_CONTROLS, "institution_id", "year"]
    for var, label in outcomes:
        cols = [var, *required]
        cols = [c for c in cols if c in df.columns]
        sub = df[cols].dropna().copy()
        if len(sub) == 0:
            continue
        zeros = int((pd.to_numeric(sub[var], errors="coerce") == 0).sum())
        rows.append(dict(
            var=var,
            label=label,
            n=len(sub),
            zeros=zeros,
            zero_share=zeros / len(sub),
        ))
    return rows


def table2_baseline_eq1(df):
    """
    Print a sequential Equation (1) baseline regression table for
    ln(1 + new patent applications), using the common Equation (1)
    sample across all specifications for comparability.
    """
    print("\n" + "=" * 74)
    print("TABLE 2: Baseline Equation (1) - Ln(New Patent Applications)")
    print("=" * 74)

    common_cols = ["ln_new_patent_apps", "pci_l1", *BASE_CONTROLS, "institution_id", "year"]
    sub = df[common_cols].dropna().copy()

    specs = [
        tuple(),
        ("ln_research_exp_l1",),
        ("ln_research_exp_l1", "ln_licensing_ftes_l1"),
        BASE_CONTROLS,
    ]
    results = [twfe(sub, "ln_new_patent_apps", controls=s) for s in specs]

    row_vars = [
        ("pci_l1", "PCI$_{L1}$"),
        ("ln_research_exp_l1", "Lagged ln(Research Exp.)"),
        ("ln_licensing_ftes_l1", "Lagged ln(Licensing FTEs)"),
        ("royalty_share_l1", "Lagged Inventor Royalty Share"),
    ]

    print("\n  Specification".ljust(34)
          + "".join(f"{f'({i})':>11}" for i in range(1, len(specs) + 1)))
    print("  " + "-" * (32 + 11 * len(specs)))
    for var, label in row_vars:
        coef_line = f"  {label:<32}"
        se_line = "  " + " " * 32
        for r in results:
            b = r["all_betas"].get(var, float("nan"))
            se = r["all_ses"].get(var, float("nan"))
            p = r["all_ps"].get(var, float("nan"))
            if np.isnan(b):
                coef_line += f"{'':>11}"
                se_line += f"{'':>11}"
            else:
                coef_line += f"{(f'{b:+.3f}{_stars(p)}'):>11}"
                se_line += f"{(f'({se:.3f})'):>11}"
        print(coef_line)
        print(se_line)

    yesno = lambda included: "Yes" if included else "No"
    print("  " + "-" * (32 + 11 * len(specs)))
    print(f"  {'Lagged ln(Research Exp.)':<32}" + "".join(f"{yesno('ln_research_exp_l1' in s):>11}" for s in specs))
    print(f"  {'Lagged ln(Licensing FTEs)':<32}" + "".join(f"{yesno('ln_licensing_ftes_l1' in s):>11}" for s in specs))
    print(f"  {'Lagged Royalty Share':<32}" + "".join(f"{yesno('royalty_share_l1' in s):>11}" for s in specs))
    print(f"  {'Institution FE':<32}" + "".join(f"{'Yes':>11}" for _ in specs))
    print(f"  {'Year FE':<32}" + "".join(f"{'Yes':>11}" for _ in specs))
    print(f"  {'Observations':<32}" + "".join(f"{r['n']:>11,}" for r in results))
    print(f"  {'R-squared':<32}" + "".join(f"{r['r2']:>11.3f}" for r in results))
    print("\n  Note: TLO age is omitted because it is collinear with institution and year fixed effects.")


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
        print(f"\n  Policy revision events (|dPCI| > {REVISION_THRESHOLD:.3f}): "
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

    print("\n  Panel B: Specification checks - Ln(New Patent Applications)")
    print(f"  {'Specification':<38} {'b':>9}  {'SE':>7} {'p':>7} {'N':>6}")
    print("  " + "-" * 72)

    specs = [
        ("pci",           "Contemporaneous PCI"),
        ("pci_l1",        "Baseline: lag-1 PCI"),
        ("pci_l2",        "Lag-2 PCI"),
        ("pci_l3",        "Lag-3 PCI"),
        ("pci_l4",        "Lag-4 PCI"),
        ("pci_l5",        "Lag-5 PCI"),
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
# 5.  RANDOMIZATION INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def circular_shift_placebo(df, outcome="ln_new_patent_apps",
                           treatment="pci_l1", n_perms=2000):
    """
    Within-institution circular-shift placebo test.

    Circular shifts preserve each institution's PCI sequence and serial
    dependence while breaking its calendar alignment with the outcome.
    """
    global LAST_PLACEBO_P
    print("\n" + "=" * 74)
    print(f"CIRCULAR-SHIFT PLACEBO INFERENCE  (seed={SEED}, n={n_perms})")
    print("=" * 74)
    cols = [outcome, treatment, *BASE_CONTROLS, "institution_id", "year"]
    base = (df[[c for c in cols if c in df.columns]]
            .dropna()
            .sort_values(["institution_id", "year"])
            .reset_index(drop=True))
    rng = np.random.default_rng(SEED)

    # Partial out controls and exact entity/year fixed effects once.  Batched
    # matrix residualization then makes 2,000 permutations fast and exact.
    nuisance = pd.concat([
        base[list(BASE_CONTROLS)].astype(float),
        pd.get_dummies(base["institution_id"].astype(str),
                       prefix="inst", drop_first=True, dtype=float),
        pd.get_dummies(base["year"].astype(int),
                       prefix="yr", drop_first=True, dtype=float),
    ], axis=1)
    nuisance = sm.add_constant(nuisance, has_constant="add")
    z = nuisance.to_numpy(dtype=float)
    q, r, _ = linalg.qr(z, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    tol = np.finfo(float).eps * max(z.shape) * (diag.max() if len(diag) else 0)
    rank = int(np.sum(diag > tol))
    q = q[:, :rank]

    def residualize(values):
        return values - q @ (q.T @ values)

    y_res = residualize(base[outcome].to_numpy(dtype=float))
    observed_x = residualize(base[treatment].to_numpy(dtype=float))
    obs = float(observed_x @ y_res / (observed_x @ observed_x))

    x = base[treatment].to_numpy(dtype=float)
    group_indices = [
        np.asarray(idx)
        for idx in base.groupby("institution_id", sort=False).indices.values()
    ]
    perm_betas = []
    batch_size = min(100, n_perms)
    completed = 0
    while completed < n_perms:
        width = min(batch_size, n_perms - completed)
        x_perm = np.empty((len(base), width), dtype=float)
        for j in range(width):
            for idx in group_indices:
                if len(idx) <= 1:
                    x_perm[idx, j] = x[idx]
                else:
                    shift = int(rng.integers(1, len(idx)))
                    x_perm[idx, j] = np.roll(x[idx], shift)
        x_res = residualize(x_perm)
        numer = y_res @ x_res
        denom = np.sum(x_res * x_res, axis=0)
        perm_betas.extend((numer / denom).tolist())
        completed += width
        if completed % 500 == 0 or completed == n_perms:
            print(f"  ...{completed} permutations")

    perm_betas = np.asarray(perm_betas)
    ri_p = ((np.sum(np.abs(perm_betas) >= np.abs(obs)) + 1)
            / (len(perm_betas) + 1))
    LAST_PLACEBO_P = ri_p

    print(f"\n  Observed beta:          {obs:+.4f}")
    print(f"  Circular shifts:       {len(perm_betas)}")
    print(f"  Placebo p-value:       {ri_p:.4f}")
    print(f"  Pct shift |beta| >= obs: {100 * ri_p:.2f}%")
    return ri_p


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TABLE 3 — HETEROGENEITY
# ─────────────────────────────────────────────────────────────────────────────

def table3(df):
    print("\n" + "=" * 74)
    print("TABLE 3: Heterogeneity")
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

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TABLE 4 — CHANNEL ANALYSIS + MULTIPLE TESTING
# ─────────────────────────────────────────────────────────────────────────────

def table4(df):
    print("\n" + "=" * 74)
    print("TABLE 4: Channel Analysis and Multiple Testing")
    print("=" * 74)

    # Panel A: conversion rate
    print("\n  Panel A: Conversion rate (Ln PatApp - Ln Discl)")
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

        print(f"\n  Panel B - sequential sub-index, outcome: {out_label}")
        print(f"  {'Sub-index':<28} {'b':>9}  {'SE':>7} {'p':>7} {'N':>6}")
        print("  " + "-" * 62)
        for tvar, tlabel in SUBIDX:
            if tvar not in df.columns:
                continue
            r = twfe(df, out_var, treatment=tvar)
            print(f"  {tlabel:<28} {r['beta']:>+9.3f}{_stars(r['p']):<3} "
                  f"{r['se']:>7.3f} {r['p']:>7.3f} {r['n']:>6,}")

        print(f"\n  Panel C - joint sub-index, outcome: {out_label}")
        print(f"  {'Sub-index':<28} {'b':>9}  {'SE':>7} {'p':>7}")
        print("  " + "-" * 54)
        base_ctrl = BASE_CONTROLS
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
        ("conv_rate",            "Applications - Disclosures"),
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
        if LAST_PLACEBO_P is not None:
            print(f"  Note: circular-shift placebo p-value for Ln(New Patent Applications) = {LAST_PLACEBO_P:.3f} "
                  "(not subject to multiple comparison adjustment)")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  APPENDIX TABLE A1 — CORRELATION MATRIX
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
# 9.  APPENDIX TABLE B1 — ROBUSTNESS SUITE
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
                "ln_research_exp_l1", "ln_licensing_ftes_l1",
                "royalty_share_l1", "tlo_age_l1"]:
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
    _row("Balanced panel (>=15 obs)", twfe(df_b, "ln_new_patent_apps"))

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
                             "royalty_share_l1",
                             "ln_new_patent_apps_lag")))

    # Falsification: PCI → future research expenditure
    if "ln_research_exp_fwd" in df.columns:
        rf = twfe(df, "ln_research_exp_fwd")
        print(f"  {'Falsification: PCI -> Ln(Future Res Exp)':<46} "
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
    _row("Balanced panel (>=15 obs)", twfe(df_b, "conv_rate"))
    if "carnegie_r1" in df.columns:
        _row("R1 universities only",
             twfe(df[df["carnegie_r1"] == 1], "conv_rate"))
    if "pci_l2" in df.columns:
        _row("Lag-2 PCI", twfe(df, "conv_rate", treatment="pci_l2"))


# ─────────────────────────────────────────────────────────────────────────────
# 10. APPENDIX TABLE B2 — NEGATIVE-BINOMIAL ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────

def appendix_b2(df):
    print("\n" + "=" * 74)
    print("APPENDIX TABLE B2: Negative Binomial Robustness for New Patent Applications")
    print("=" * 74)

    no_ctrl = df[["new_patent_apps", "pci_l1", "institution_id", "year"]].dropna()
    with_ctrl = df[[
        "new_patent_apps", "pci_l1", *BASE_CONTROLS, "institution_id", "year"
    ]].dropna()
    print("\n  Outcome distribution (baseline samples for New Patent Applications)")
    print(f"  No-controls sample:    N = {len(no_ctrl):,}, mean = {no_ctrl['new_patent_apps'].mean():.2f}, "
          f"variance = {no_ctrl['new_patent_apps'].var():.2f}, "
          f"variance/mean = {no_ctrl['new_patent_apps'].var() / no_ctrl['new_patent_apps'].mean():.2f}")
    print(f"  With-controls sample:  N = {len(with_ctrl):,}, mean = {with_ctrl['new_patent_apps'].mean():.2f}, "
          f"variance = {with_ctrl['new_patent_apps'].var():.2f}, "
          f"variance/mean = {with_ctrl['new_patent_apps'].var() / with_ctrl['new_patent_apps'].mean():.2f}")

    print("\n  Panel A: PCI coefficient across negative binomial specifications")
    print(f"  {'Controls':<14} {'b':>9}  {'SE':>7} {'p':>7} {'N':>6}")
    print("  " + "-" * 54)
    for use_ctrl in [False, True]:
        ctrls = BASE_CONTROLS if use_ctrl else tuple()
        r = nb_fe(df, "new_patent_apps", controls=ctrls)
        ctrl_label = "With controls" if use_ctrl else "No controls"
        print(f"  {ctrl_label:<14} {r['beta']:>+9.3f}{_stars(r['p']):<3} "
              f"{r['se']:>7.3f} {r['p']:>7.3f} {r['n']:>6,}")


# ─────────────────────────────────────────────────────────────────────────
# 11. APPENDIX TABLE B3 — LAG SENSITIVITY
# ──────────────────────────────────────────────────────────────────────

def appendix_b3(df):
    print("\n" + "=" * 74)
    print("APPENDIX TABLE B3: Lag Sensitivity - All Outcomes")
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
            ("pci_l3", "Lag 3"),
            ("pci_l4", "Lag 4"),
            ("pci_l5", "Lag 5")]

    hdr = f"  {'Outcome':<28}" + "".join(f"  {ln:>14}" for _, ln in LAGS)
    print(hdr)
    print("  " + "-" * 122)

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
    print("REPLICATION - Policy Communication and Technology Transfer")
    print("Sharma, Wang, Basnet, Cossco (2026)")
    print("=" * 74)

    try:
        df = load_and_prepare()
    except FileNotFoundError:
        sys.exit(
            f"\nERROR: {DEFAULT_DATA_PATH} not found.\n"
            "Restore data/merged_autm.csv from the replication package.\n"
        )

    print(f"\nSample summary:")
    print(f"  Institutions  :  {df['institution_id'].nunique()}")
    print(f"  Year range    :  {df['year'].min()} - {df['year'].max()}")
    print(f"  Raw obs       :  {len(df):,}")
    if "pci" in df.columns:
        print(f"  Obs with PCI  :  {df['pci'].notna().sum():,}")

    # ── Tables ───────────────────────────────────────────────────────────────
    table1(df)
    table2_baseline_eq1(df)
    table2(df)
    table3(df)

    # ── Randomization Inference ───────────────────────────────────────────────
    # Set n_perms=200 for a quick check; use 2000 for the final package.
    print("\n[Tip: set n_perms=200 for a quick check, 2000 for the final placebo test]")
    placebo_p = circular_shift_placebo(df, n_perms=2000)

    # ── Channel + Multiple Testing ────────────────────────────────────────────
    table4(df)

    # ── Appendices ────────────────────────────────────────────────────────────
    appendix_a1(df)
    appendix_b1(df)
    appendix_b2(df)
    appendix_b3(df)

    print("\n" + "=" * 74)
    print("Replication complete.")
    print("=" * 74)

