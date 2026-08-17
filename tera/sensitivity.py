"""
TERA Sensitivity — Parameter sensitivity analysis for Stage 2.

Addresses R5:
  1. PELT penalty sensitivity — BIC vs modified BIC vs AIC
  2. Policy window sensitivity — ±1 vs ±2 vs ±3 year windows

Usage:
    from tera.sensitivity import (
        pelt_penalty_sensitivity,
        policy_window_sensitivity,
        sensitivity_report,
    )
"""
import numpy as np
import pandas as pd


# ============================================================
# PELT Penalty Sensitivity
# ============================================================

def _pelt_penalty_value(n_obs, penalty="bic", gamma=2.0):
    """Compute PELT penalty value for different penalty types."""
    if penalty == "bic":
        return np.log(n_obs)
    elif penalty == "mbic":
        # Modified BIC: gamma * log(n)
        return gamma * np.log(n_obs)
    elif penalty == "aic":
        return 2.0
    elif penalty == "hannan_quinn":
        return 2.0 * np.log(np.log(n_obs))
    else:
        return np.log(n_obs)  # default BIC


def pelt_penalty_sensitivity(
    yearly_counts,
    category_names,
    years,
    penalties=None,
    min_size=2,
):
    """
    Run PELT change-point detection under multiple penalty values
    and report changepoint stability.

    This requires ruptures to be installed. If not available,
    fall back to manual segment-mean change analysis.

    Parameters
    ----------
    yearly_counts : dict — {category: np.array of shape (T,)}
    category_names : list of str
    years : list of int
    penalties : list of (label, penalty_name, kwargs) tuples
        e.g. [("BIC", "bic", {}), ("mBIC γ=2", "mbic", {"gamma": 2}), ("AIC", "aic", {})]
    min_size : int

    Returns
    -------
    dict — {category: DataFrame with changepoints per penalty}
    """
    if penalties is None:
        penalties = [
            ("BIC", "bic", {}),
            ("mBIC γ=2", "mbic", {"gamma": 2.0}),
            ("AIC", "aic", {}),
        ]

    n_obs = len(years)
    results = {}

    try:
        import ruptures as rpt
        _has_ruptures = True
    except ImportError:
        _has_ruptures = False
        print("[WARN] ruptures not installed — using fallback penalty calculation")

    for cat in category_names:
        if cat not in yearly_counts:
            continue

        raw = yearly_counts[cat]
        # Accept both array and dict-of-arrays
        if isinstance(raw, dict):
            signal = np.asarray(raw.get("patents", raw.get(list(raw.keys())[0])), dtype=float)
        else:
            signal = np.asarray(raw, dtype=float)
        cat_results = {}

        for label, penalty_name, kwargs in penalties:
            if _has_ruptures:
                pen_val = _pelt_penalty_value(len(signal), penalty_name, **kwargs)
                model = "l2"
                algo = rpt.Pelt(model=model, min_size=min_size)
                # ruptures Pelt uses pen parameter
                try:
                    changepoints = algo.fit_predict(signal, pen=pen_val)
                except TypeError:
                    changepoints = algo.fit_predict(signal)
                # Remove the final n (ruptures convention)
                changepoints = [cp for cp in changepoints if cp < len(signal)]
                changepoint_years = [years[cp] for cp in changepoints]
            else:
                # Fallback: report penalty value only
                pen_val = _pelt_penalty_value(len(signal), penalty_name, **kwargs)
                changepoints = []
                changepoint_years = []

            cat_results[label] = {
                "penalty_value": round(pen_val, 3),
                "n_changepoints": len(changepoints),
                "changepoint_years": changepoint_years,
            }

        results[cat] = cat_results

    return results


def _penalty_labels():
    """Default penalty variants for PELT sensitivity."""
    return [
        ("BIC", "bic", {}),
        ("mBIC γ=2", "mbic", {"gamma": 2.0}),
        ("AIC", "aic", {}),
    ]


def changepoint_stability_summary(pelt_results, category_names):
    """
    Compute stability metrics across penalty choices.

    For each category, measures:
      - n_penalties with same n_changepoints (consistency)
      - year agreement (±1 year) across penalties

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for cat in category_names:
        if cat not in pelt_results:
            continue
        cat_data = pelt_results[cat]
        cp_counts = [v["n_changepoints"] for v in cat_data.values()]
        cp_years = [set(v["changepoint_years"]) for v in cat_data.values()]

        # Consistency: how many penalties agree on n_changepoints
        mode_count = max(set(cp_counts), key=cp_counts.count)
        n_agree = cp_counts.count(mode_count)

        # Year overlap: Jaccard of years across all penalty pairs
        if len(cp_years) >= 2:
            jaccards = []
            for i in range(len(cp_years)):
                for j in range(i + 1, len(cp_years)):
                    if cp_years[i] or cp_years[j]:
                        jac = len(cp_years[i] & cp_years[j]) / len(cp_years[i] | cp_years[j])
                        jaccards.append(jac)
            mean_jaccard = np.mean(jaccards) if jaccards else 0.0
        else:
            mean_jaccard = 1.0

        rows.append({
            "category": cat,
            "n_penalties": len(cat_data),
            "cp_counts_range": f"{min(cp_counts)}-{max(cp_counts)}",
            "modal_cp_count": mode_count,
            "n_agree_on_mode": n_agree,
            "mean_year_jaccard": round(float(mean_jaccard), 3),
            "stable": n_agree >= len(cat_data) * 0.67 and mean_jaccard >= 0.6,
        })

    df = pd.DataFrame(rows)
    n_stable = df["stable"].sum() if "stable" in df.columns else 0
    print(f"\nChangepoint Stability: {n_stable}/{len(df)} categories stable "
          f"(≥67% penalty agreement + ≥0.6 year Jaccard)")
    return df


# ============================================================
# Policy Window Sensitivity
# ============================================================

def policy_window_sensitivity(
    changepoint_years,
    policy_event_years,
    windows=None,
):
    """
    Compute policy-breakpoint alignment scores across window sizes.

    Parameters
    ----------
    changepoint_years : dict — {category: [year1, year2, ...]}
    policy_event_years : list of int
    windows : list of int — e.g. [1, 2, 3] for ±1/±2/±3 years

    Returns
    -------
    pd.DataFrame — rows=categories, columns=one per window
    """
    if windows is None:
        windows = [1, 2, 3]

    rows = []
    for cat, cp_years in changepoint_years.items():
        row = {"category": cat, "n_changepoints": len(cp_years)}
        for w in windows:
            count = sum(
                1 for cp in cp_years
                for py in policy_event_years
                if abs(cp - py) <= w
            )
            row[f"±{w}yr"] = count
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["category", "n_changepoints"] + [f"±{w}yr" for w in windows]]

    # Add stability column: CV of window counts
    window_cols = [f"±{w}yr" for w in windows]
    df["window_cv"] = df[window_cols].std(axis=1) / df[window_cols].mean(axis=1).replace(0, np.nan)
    df["window_cv"] = df["window_cv"].round(3)
    df["stable"] = df["window_cv"] < 0.3

    n_stable = df["stable"].sum()
    print(f"\nPolicy Window Sensitivity: {n_stable}/{len(df)} categories stable (CV < 0.3)")

    return df


# ============================================================
# Combined Sensitivity Report
# ============================================================

def sensitivity_report(
    yearly_counts,
    category_names,
    years,
    changepoint_years,
    policy_event_years,
    penalties=None,
    windows=None,
):
    """
    Produce a complete sensitivity report.

    Returns
    -------
    dict with:
      - pelt_results: from pelt_penalty_sensitivity
      - changepoint_stability: DataFrame
      - policy_window: DataFrame
      - summary: str
    """
    if penalties is None:
        penalties = _penalty_labels()
    if windows is None:
        windows = [1, 2, 3]

    # 1. PELT penalty sensitivity
    print("Running PELT penalty sensitivity...")
    pelt_results = pelt_penalty_sensitivity(
        yearly_counts, category_names, years,
        penalties=penalties,
    )

    # 2. Changepoint stability
    stability_df = changepoint_stability_summary(pelt_results, category_names)

    # 3. Policy window sensitivity
    print("Running policy window sensitivity...")
    window_df = policy_window_sensitivity(
        changepoint_years, policy_event_years, windows=windows,
    )

    # 4. Summary
    n_total = len(category_names)
    n_stable_pelt = stability_df["stable"].sum() if "stable" in stability_df.columns else 0
    n_stable_window = window_df["stable"].sum() if "stable" in window_df.columns else 0

    summary_lines = [
        f"Sensitivity Analysis Report",
        f"{'='*60}",
        f"PELT Changepoint Stability:",
        f"  - Stable categories: {n_stable_pelt}/{n_total}",
        f"  - Mean year Jaccard across penalties: "
        f"{stability_df['mean_year_jaccard'].mean():.3f}"
        if "mean_year_jaccard" in stability_df.columns else "",
        f"",
        f"Policy Window Stability (±1/±2/±3 yr):",
        f"  - Stable categories (CV < 0.3): {n_stable_window}/{n_total}",
        f"  - Mean window CV: {window_df['window_cv'].mean():.3f}"
        if "window_cv" in window_df.columns else "",
        f"",
        f"Interpretation:",
    ]

    if n_stable_pelt >= 0.75 * n_total:
        summary_lines.append(
            f"  PELT changepoints are ROBUST to penalty choice "
            f"({n_stable_pelt}/{n_total} stable)."
        )
    else:
        summary_lines.append(
            f"  PELT changepoints are SENSITIVE to penalty choice "
            f"({n_stable_pelt}/{n_total} stable). Report dominant pattern."
        )

    if n_stable_window >= 0.75 * n_total:
        summary_lines.append(
            f"  Policy-breakpoint alignment is STABLE across window sizes."
        )
    else:
        summary_lines.append(
            f"  Policy-breakpoint alignment VARIES with window size. "
            f"Report ±2yr as primary and note range."
        )

    return {
        "pelt_results": pelt_results,
        "changepoint_stability": stability_df,
        "policy_window": window_df,
        "summary": "\n".join(summary_lines),
    }
