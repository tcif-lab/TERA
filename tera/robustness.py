"""
TERA Robustness — Statistical validation for Stage 2 temporal analysis.

Addresses R3 (Granger robustness):
  1. Permutation test — establishes whether observed Granger results
     exceed random baseline
  2. Observation-to-parameter ratio — quantifies statistical power
  3. Multiple comparison correction — Bonferroni + Benjamini-Hochberg

Usage:
    from tera.robustness import granger_permutation_test, granger_robustness_report

    results = granger_permutation_test(
        patent_series, standards_series, n_permutations=1000
    )
    report = granger_robustness_report(results)
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from scipy import stats

import warnings
warnings.filterwarnings("ignore", message="verbose is deprecated")


# ============================================================
# Observation-to-Parameter Ratio
# ============================================================

def compute_var_obs_to_param_ratio(n_obs, max_lag, n_vars=2):
    """
    Compute observation-to-parameter ratio for VAR(p) model.

    A VAR(p) with n_vars variables estimates:
      - n_vars * (n_vars * p + 1) parameters per equation
    For bivariate VAR(p): 2 * (2p + 1) = 4p + 2 per equation.

    Conventional threshold: >= 10:1 for reliable inference.
    Below 5:1 is severely underpowered.

    Parameters
    ----------
    n_obs : int — number of time-series observations
    max_lag : int — VAR lag order p
    n_vars : int — number of variables (default=2 for bivariate)

    Returns
    -------
    dict with: n_obs, max_lag, n_params_per_eq, n_eff_obs, ratio, status
    """
    n_params_per_eq = n_vars * max_lag + 1  # +1 for intercept
    n_eff_obs = n_obs - max_lag
    ratio = n_eff_obs / n_params_per_eq if n_params_per_eq > 0 else float("inf")

    if ratio >= 10:
        status = "adequate (>= 10:1)"
    elif ratio >= 5:
        status = "marginal (5:1 - 10:1)"
    else:
        status = "underpowered (< 5:1)"

    return {
        "n_obs": n_obs,
        "max_lag": max_lag,
        "n_vars": n_vars,
        "n_params_per_eq": n_params_per_eq,
        "n_eff_obs": n_eff_obs,
        "ratio": round(ratio, 1),
        "status": status,
    }


def compute_all_var_ratios(yearly_data, category_names, max_lags):
    """
    Compute observation-to-parameter ratios for all tech categories.

    Parameters
    ----------
    yearly_data : dict — {category: {"patents": np.array, "standards": np.array}}
    category_names : list of str
    max_lags : int or dict — max lag per category

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for cat in category_names:
        if cat not in yearly_data:
            continue
        n_obs = len(yearly_data[cat].get("patents", []))
        lag = max_lags.get(cat, 3) if isinstance(max_lags, dict) else max_lags
        result = compute_var_obs_to_param_ratio(n_obs, lag)
        result["category"] = cat
        rows.append(result)

    df = pd.DataFrame(rows)
    df = df[["category", "n_obs", "max_lag", "n_params_per_eq",
             "n_eff_obs", "ratio", "status"]]
    return df


# ============================================================
# Permutation Test for Granger Causality
# ============================================================

def granger_permutation_test(
    patent_series,
    standards_series,
    max_lag=3,
    significance=0.1,
    n_permutations=1000,
    seed=42,
    use_log_diff=True,
):
    """
    Permutation test: is the number of significant Granger-causal
    categories greater than expected by chance?

    Procedure:
      1. (Optional) Log-difference both series for stationarity.
      2. Run Granger on original data; count n_significant (p < sig).
      3. Randomly permute the patent series n_permutations times.
      4. For each permutation, run Granger on all categories;
         count n_significant.
      5. Compute empirical p-value using (count+1)/(n_perm+1) correction.

    Parameters
    ----------
    patent_series : dict — {category: np.array of shape (T,)}
    standards_series : dict — {category: np.array of shape (T,)}
    max_lag : int
    significance : float
    n_permutations : int
    seed : int
    use_log_diff : bool — if True, apply log-differencing for stationarity

    Returns
    -------
    dict with permutation results
    """
    rng = np.random.RandomState(seed)
    categories = sorted(set(patent_series.keys()) & set(standards_series.keys()))

    def _log_diff(s):
        """Log-difference for stationarity; returns (n-1) length array.
        Uses max(x, 1) since count data: log(1)=0 is the natural zero baseline."""
        s = np.asarray(s, dtype=float)
        s_safe = np.maximum(s, 1.0)
        return np.diff(np.log(s_safe))

    # ---- Original data (with optional log-diff) ----
    original_results = {}
    n_sig_original = 0
    for cat in categories:
        p_series = np.asarray(patent_series[cat], dtype=float)
        s_series = np.asarray(standards_series[cat], dtype=float)

        if use_log_diff:
            p_series = _log_diff(p_series)
            s_series = _log_diff(s_series)

        n = min(len(p_series), len(s_series))

        if n <= max_lag + 3:
            original_results[cat] = {"p_value": None, "significant": False, "error": "too_short"}
            continue

        actual_lag = min(max_lag, n - 3)
        # Column order: [dependent, independent] = [standards, patents]
        # Tests: patents → standards
        data = np.column_stack([s_series[:n], p_series[:n]])

        try:
            result = grangercausalitytests(data, maxlag=actual_lag, verbose=False)
            min_p = min(
                result[lag][0]["ssr_ftest"][1]
                for lag in range(1, actual_lag + 1)
            )
            is_sig = min_p < significance
            original_results[cat] = {
                "p_value": round(min_p, 4),
                "significant": is_sig,
                "best_lag": min(
                    range(1, actual_lag + 1),
                    key=lambda lag: result[lag][0]["ssr_ftest"][1],
                ),
            }
            if is_sig:
                n_sig_original += 1
        except Exception as e:
            original_results[cat] = {"p_value": None, "significant": False, "error": str(e)}

    # ---- Permutations ----
    perm_counts = []
    for perm_i in range(n_permutations):
        n_sig_perm = 0
        for cat in categories:
            p_series = np.asarray(patent_series[cat], dtype=float)
            s_series = np.asarray(standards_series[cat], dtype=float)

            if use_log_diff:
                p_series = _log_diff(p_series)
                s_series = _log_diff(s_series)

            n = min(len(p_series), len(s_series))
            if n <= max_lag + 3:
                continue

            # Permute the patent series (break temporal structure)
            p_perm = rng.permutation(p_series[:n])
            actual_lag = min(max_lag, n - 3)
            # Column order: [dependent, independent] = [standards, permuted_patents]
            data = np.column_stack([s_series[:n], p_perm])

            try:
                result = grangercausalitytests(data, maxlag=actual_lag, verbose=False)
                min_p = min(
                    result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, actual_lag + 1)
                )
                if min_p < significance:
                    n_sig_perm += 1
            except Exception:
                pass

        perm_counts.append(n_sig_perm)

    perm_counts = np.array(perm_counts)
    # Continuity correction: (count + 1) / (n_perm + 1) avoids p=0.000
    n_extreme = int(np.sum(perm_counts >= n_sig_original))
    empirical_p = (n_extreme + 1) / (n_permutations + 1)

    return {
        "n_categories": len(categories),
        "n_permutations": n_permutations,
        "n_significant_original": n_sig_original,
        "perm_mean": round(float(np.mean(perm_counts)), 2),
        "perm_std": round(float(np.std(perm_counts)), 2),
        "perm_95th_percentile": round(float(np.percentile(perm_counts, 95)), 2),
        "empirical_p_value": round(float(empirical_p), 4),
        "significant": empirical_p < 0.05,
        "interpretation": (
            f"Observed {n_sig_original} significant categories. "
            f"Permutation null distribution: mean={np.mean(perm_counts):.1f}, "
            f"95th %ile={np.percentile(perm_counts, 95):.0f}. "
            f"Empirical p={empirical_p:.4f}. "
            + (
                "Result EXCEEDS random baseline."
                if empirical_p < 0.05
                else "Result does NOT exceed random baseline (p >= 0.05)."
            )
        ),
        "perm_counts": perm_counts.tolist(),
        "original_results": original_results,
    }


# ============================================================
# Multiple Comparison Correction
# ============================================================

def multiple_comparison_correction(p_values, method="bonferroni"):
    """
    Apply multiple comparison correction to Granger p-values.

    Parameters
    ----------
    p_values : dict — {category: p_value}
    method : str — "bonferroni" or "fdr_bh" (Benjamini-Hochberg)

    Returns
    -------
    dict — {category: {original_p, corrected_p, significant_after_correction}}
    """
    cats = list(p_values.keys())
    p_vals = np.array([p_values[c] for c in cats])

    if method == "bonferroni":
        corrected = np.minimum(p_vals * len(cats), 1.0)
    elif method == "fdr_bh":
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(p_vals, method="fdr_bh")
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        cat: {
            "original_p": round(p_vals[i], 4),
            f"corrected_{method}": round(corrected[i], 4),
            "significant": corrected[i] < 0.1,
        }
        for i, cat in enumerate(cats)
    }


# ============================================================
# Combined Robustness Report
# ============================================================

def granger_robustness_report(
    yearly_data,
    category_names,
    max_lag=3,
    significance=0.1,
    n_permutations=1000,
    use_log_diff=True,
):
    """
    Produce a complete Granger robustness report.

    Parameters
    ----------
    use_log_diff : bool — if True, apply log-differencing before Granger
        (recommended for non-stationary time series with strong trends)

    Returns
    -------
    dict with keys:
      - var_ratios: DataFrame of obs-to-param ratios
      - permutation: dict from granger_permutation_test
      - corrected: dict from multiple_comparison_correction
      - summary: str — plain-English summary
    """
    # Extract patent and standards series
    patent_series = {}
    standards_series = {}
    for cat in category_names:
        if cat in yearly_data:
            patent_series[cat] = np.asarray(yearly_data[cat].get("patents", []))
            standards_series[cat] = np.asarray(yearly_data[cat].get("standards", []))

    # 1. VAR ratios
    var_ratios = compute_all_var_ratios(yearly_data, category_names, max_lag)

    # 2. Permutation test (with log-differencing for stationarity)
    perm_results = granger_permutation_test(
        patent_series, standards_series,
        max_lag=max_lag, significance=significance,
        n_permutations=n_permutations,
        use_log_diff=use_log_diff,
    )

    # 3. Multiple comparison correction
    p_values = {
        cat: perm_results["original_results"][cat].get("p_value", 1.0)
        for cat in category_names
        if cat in perm_results["original_results"]
        and perm_results["original_results"][cat].get("p_value") is not None
    }
    corrected = multiple_comparison_correction(p_values, "bonferroni")

    # 4. Summary
    underpowered = var_ratios[var_ratios["ratio"] < 5]
    summary_lines = [
        f"Granger Robustness Report",
        f"{'='*60}",
        f"Categories tested: {len(category_names)}",
        f"Observation-to-parameter ratios:",
        f"  - Min: {var_ratios['ratio'].min():.1f}:1",
        f"  - Median: {var_ratios['ratio'].median():.1f}:1",
        f"  - Max: {var_ratios['ratio'].max():.1f}:1",
        f"  - Underpowered (< 5:1): {len(underpowered)}/{len(var_ratios)}",
        f"",
        f"Permutation test ({n_permutations} permutations):",
        f"  - Observed significant: {perm_results['n_significant_original']}",
        f"  - Null mean: {perm_results['perm_mean']:.2f}",
        f"  - Null 95th %ile: {perm_results['perm_95th_percentile']}",
        f"  - Empirical p-value: {perm_results['empirical_p_value']:.4f}",
        f"  - Signal: {'YES — exceeds random' if perm_results['significant'] else 'NO — does not exceed random'}",
        f"",
        f"Multiple comparison (Bonferroni):",
        f"  - Significant after correction: "
        f"{sum(1 for v in corrected.values() if v['significant'])}/{len(corrected)}",
    ]

    return {
        "var_ratios": var_ratios,
        "permutation": perm_results,
        "corrected": corrected,
        "summary": "\n".join(summary_lines),
    }
