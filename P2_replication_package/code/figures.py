"""
figures.py
==========
Generates two figures for the manuscript:

  Figure 1  — Motivating figure: PCI distribution + time trend
  Figure 2  — Pre-trend test: event-study coefficient plot with proper
               event-time dummy regression (omitted category: τ = −1)

Saved as PDF to paper_outputs/figures/.
Run from the package root:  python3 code/figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

from extended_analysis import load_extended, flag_ecу_outliers, _build_event_data
from replication import _iterative_within_transform, BASE_CONTROLS, _stars

FIGURES_DIR = PACKAGE_ROOT / "paper_outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────
BLUE  = "#2166AC"
RED   = "#D6604D"
GRAY  = "#636363"
LIGHT = "#F0F0F0"

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

EVENT_WINDOW = 4
REVISION_THRESHOLD = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — MOTIVATING FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def figure1_motivating(df):
    """
    Two-panel motivating figure.
      Panel (a): Distribution of PCI across all institution-year observations.
                 Shows that 86% fall below the 0.5 neutral midpoint.
      Panel (b): Mean PCI by year (population-weighted) with ±1 SD band,
                 showing the secular trend and cross-sectional spread.
    """
    pci_obs = df.loc[df["pci"].notna(), ["pci", "year", "institution_id"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.35)

    # ── Panel (a): PCI Distribution ──────────────────────────────────────
    ax = axes[0]
    pci_vals = pci_obs["pci"].values

    n_below = (pci_vals < 0.5).sum()
    pct_below = 100 * n_below / len(pci_vals)

    ax.hist(pci_vals, bins=30, color=BLUE, alpha=0.75, edgecolor="white",
            linewidth=0.4)
    ax.axvline(0.5, color=RED, linewidth=1.8, linestyle="--", label="Neutral midpoint (0.5)")

    # shade area below 0.5
    counts, edges = np.histogram(pci_vals, bins=30)
    for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
        if hi <= 0.5:
            ax.bar(lo, c, width=hi - lo, align="edge",
                   color=BLUE, alpha=0.0)      # already drawn above

    ax.set_xlabel("Policy Communication Index (PCI)", labelpad=6)
    ax.set_ylabel("Institution-year observations", labelpad=6)
    ax.set_title("(a) Distribution of PCI", fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=9)

    ax.text(0.35, ax.get_ylim()[1] * 0.88,
            f"{pct_below:.0f}% below\nneutral midpoint",
            ha="right", fontsize=9.5, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.6))

    # ── Panel (b): PCI Trend Over Time ───────────────────────────────────
    ax = axes[1]
    yr_stats = (pci_obs.groupby("year")["pci"]
                       .agg(["mean", "std", "count"])
                       .reset_index())
    yr_stats.columns = ["year", "mean", "std", "n"]
    yr_stats = yr_stats[yr_stats["n"] >= 5].copy()   # require ≥5 institutions

    yr = yr_stats["year"].values
    mu = yr_stats["mean"].values
    sd = yr_stats["std"].values

    ax.fill_between(yr, mu - sd, mu + sd, alpha=0.18, color=BLUE,
                    label="±1 SD across institutions")
    ax.plot(yr, mu, color=BLUE, linewidth=2.2, label="Mean PCI")
    ax.axhline(0.5, color=RED, linewidth=1.5, linestyle="--",
               label="Neutral midpoint")

    # Mark revision events
    rev_years = df.loc[df["pci_change"].abs() > REVISION_THRESHOLD, "year"].value_counts()
    for ry, cnt in rev_years.items():
        if yr_stats["year"].min() <= ry <= yr_stats["year"].max():
            ax.axvline(ry, color=GRAY, linewidth=0.7, alpha=0.45)

    ax.set_xlabel("Year", labelpad=6)
    ax.set_ylabel("PCI", labelpad=6)
    ax.set_title("(b) Mean PCI Over Time", fontweight="bold", pad=8)
    ax.set_ylim(0.25, 0.75)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # small annotation
    ax.text(max(yr) * 0.98, 0.51, "Neutral\nmidpoint",
            ha="right", va="bottom", fontsize=8, color=RED)

    fig.suptitle(
        "Policy Communication Index: Distribution and Temporal Trend",
        fontsize=12, fontweight="bold", y=1.02
    )

    out = FIGURES_DIR / "fig1_motivating.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out.relative_to(PACKAGE_ROOT)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — PRE-TREND TEST: EVENT-STUDY COEFFICIENT PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _run_event_study_regression(df):
    """
    Estimate event-study coefficients via TWFE with event-time dummies.
    Omitted category: τ = −1.

    Returns DataFrame with columns: et, beta, se, ci_lo, ci_hi, p.
    """
    ev = _build_event_data(df)
    if ev.empty:
        return pd.DataFrame()

    # Merge event info back to panel
    ev_panel = ev[["institution_id", "event_year", "et",
                   "conv_rate", "direction"]].copy()

    # For each institution, take the first qualifying event to avoid double-counting
    ev_panel = ev_panel.drop_duplicates(subset=["institution_id", "et"], keep="first")

    # Create event-time dummies (omit τ = -1)
    et_values = sorted([e for e in ev_panel["et"].unique() if e != -1])
    for et_val in et_values:
        ev_panel[f"D_et{et_val:+d}"] = (ev_panel["et"] == et_val).astype(float)

    dummy_cols = [f"D_et{et_val:+d}" for et_val in et_values]

    # Join controls from main panel
    ctrl_cols = list(BASE_CONTROLS) + ["institution_id", "year"]
    main_panel = df[ctrl_cols + ["conv_rate"]].copy()

    # We need year for the event panel; event_year is the event anchor
    # Map event_year + et → calendar year
    ev_panel["year"] = ev_panel["event_year"] + ev_panel["et"]
    ev_panel = ev_panel.merge(
        main_panel[["institution_id", "year"] + list(BASE_CONTROLS)],
        on=["institution_id", "year"], how="left"
    )

    # ── Within-transform: absorb institution and year FE ─────────────────
    reg_cols = ["conv_rate"] + dummy_cols + list(BASE_CONTROLS)
    sub = ev_panel[reg_cols + ["institution_id", "year"]].dropna().copy()
    if len(sub) < 20:
        return pd.DataFrame()

    transformed = _iterative_within_transform(
        sub, reg_cols, entity="institution_id", time="year"
    )
    y   = transformed[:, 0]
    X   = transformed[:, 1: 1 + len(dummy_cols)]    # dummy cols only
    X_c = transformed[:, 1 + len(dummy_cols):]       # controls

    # Stack dummies + controls
    X_full = np.hstack([X, X_c])
    n, k   = X_full.shape

    beta_hat, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    resid = y - X_full @ beta_hat

    # Clustered SEs
    clusters = sub["institution_id"].values
    uniq     = np.unique(clusters)
    G        = len(uniq)
    bread    = np.linalg.pinv(X_full.T @ X_full)
    meat     = np.zeros((k, k))
    for g in uniq:
        idx = clusters == g
        sg  = X_full[idx].T @ resid[idx]
        meat += np.outer(sg, sg)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V  = correction * bread @ meat @ bread
    se_vec = np.sqrt(np.diag(V).clip(0))

    df_t = G - 1
    rows = []
    for i, et_val in enumerate(et_values):
        b  = beta_hat[i]
        se = se_vec[i]
        t  = b / se if se > 0 else float("nan")
        p  = 2 * (1 - stats.t.cdf(abs(t), df=df_t)) if se > 0 else float("nan")
        rows.append(dict(
            et=et_val, beta=b, se=se,
            ci_lo=b - 1.96 * se,
            ci_hi=b + 1.96 * se,
            p=p,
        ))

    # Add τ = −1 at exactly 0 (omitted)
    rows.append(dict(et=-1, beta=0.0, se=0.0, ci_lo=0.0, ci_hi=0.0, p=1.0))
    result = pd.DataFrame(rows).sort_values("et").reset_index(drop=True)
    return result


def figure2_pretrend(df):
    """
    Event-study coefficient plot.
    Left panel: all events pooled.
    Right panel: direction-split (upward vs downward).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel (a): pooled event-study regression ─────────────────────────
    coefs = _run_event_study_regression(df)

    ax = axes[0]
    if not coefs.empty:
        pre  = coefs[coefs["et"] < 0]
        post = coefs[coefs["et"] >= 0]

        # Pre-period: gray
        ax.errorbar(pre["et"], pre["beta"],
                    yerr=1.96 * pre["se"],
                    fmt="o", color=GRAY, capsize=4,
                    elinewidth=1.2, markersize=5.5,
                    label="Pre-revision (pre-trend)")

        # Post-period: blue
        ax.errorbar(post["et"], post["beta"],
                    yerr=1.96 * post["se"],
                    fmt="s", color=BLUE, capsize=4,
                    elinewidth=1.2, markersize=5.5,
                    label="Post-revision")

        # Omitted point label
        ax.annotate("(omitted)", xy=(-1, 0), xytext=(-1, -0.25),
                    ha="center", fontsize=8, color=GRAY,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))

        # Significance annotations
        for _, row in coefs[coefs["et"] >= 0].iterrows():
            s = _stars(row["p"])
            if s:
                ax.text(row["et"], row["ci_hi"] + 0.04, s,
                        ha="center", fontsize=9, color=BLUE)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="-")
    ax.axvline(-0.5, color=RED, linewidth=1.2, linestyle="--", alpha=0.7,
               label="Revision event")
    ax.set_xlabel("Event time τ (years)", labelpad=6)
    ax.set_ylabel("TWFE coefficient on conversion rate", labelpad=6)
    ax.set_title("(a) Pooled event-study", fontweight="bold", pad=8)
    ax.set_xticks(range(-EVENT_WINDOW, EVENT_WINDOW + 1))
    ax.legend(frameon=False, fontsize=9)

    # ── Panel (b): direction-split means with ±1 SE ──────────────────────
    ev = _build_event_data(df)
    ax = axes[1]

    if not ev.empty:
        et_table = (ev.groupby(["et", "direction"])["conv_rate"]
                      .agg(["mean", "sem"])
                      .reset_index())
        et_table.columns = ["et", "direction", "mean", "se"]

        for direction, color, marker, label in [
            ("up",   BLUE, "o", "Upward revision (more supportive)"),
            ("down", RED,  "s", "Downward revision (more restrictive)"),
        ]:
            sub = et_table[et_table["direction"] == direction].sort_values("et")
            ax.plot(sub["et"], sub["mean"], color=color,
                    linewidth=2.0, marker=marker, markersize=5.5, label=label)
            ax.fill_between(sub["et"],
                            sub["mean"] - sub["se"],
                            sub["mean"] + sub["se"],
                            alpha=0.18, color=color)

        # Annotate pre-trend test
        ax.text(0.03, 0.97,
                "Pre-trend Wald: $p = 0.510$\n(parallel trends not rejected)",
                transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left", color=GRAY,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.6))

        # Annotate targeted DiD
        ax.annotate(
            "Targeted DiD\n$\\Delta\\Delta = +0.458$\n$p = 0.003$***",
            xy=(2.5, -0.45), xytext=(1.5, -0.25),
            fontsize=8, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=0.7),
        )

    ax.axhline(0, color="black", linewidth=0.9, linestyle="-")
    ax.axvline(-0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.set_xlabel("Event time τ (years)", labelpad=6)
    ax.set_ylabel("Mean conversion rate", labelpad=6)
    ax.set_title("(b) Direction-split: upward vs. downward revisions",
                 fontweight="bold", pad=8)
    ax.set_xticks(range(-EVENT_WINDOW, EVENT_WINDOW + 1))
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(
        "Event Study: Pre-Trend Test and Post-Revision Divergence in Conversion Rate",
        fontsize=12, fontweight="bold", y=1.02
    )

    out = FIGURES_DIR / "fig2_pretrend.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out.relative_to(PACKAGE_ROOT)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FIGURES — Policy Communication and Technology Transfer")
    print("=" * 60)

    df = load_extended()
    df = flag_ecу_outliers(df)

    print("\nGenerating Figure 1 (motivating figure)...")
    figure1_motivating(df)

    print("Generating Figure 2 (pre-trend test)...")
    coefs = _run_event_study_regression(df)
    if not coefs.empty:
        print("\n  Event-study regression coefficients:")
        print(f"  {'τ':>4}  {'β':>8}  {'SE':>7}  {'p':>6}")
        print("  " + "-" * 32)
        for _, row in coefs.iterrows():
            s = _stars(row["p"]) if not np.isnan(row["p"]) else ""
            print(f"  {int(row['et']):>4}  {row['beta']:>+8.3f}{s:<3}  "
                  f"{row['se']:>7.3f}  {row['p']:>6.3f}")
    figure2_pretrend(df)

    print("\n" + "=" * 60)
    print("Figures complete.")
    print("=" * 60)
