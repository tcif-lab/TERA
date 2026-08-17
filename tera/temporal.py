"""
TERA Temporal Evolution Mining (Stage 2).

Mines the category-level representation produced by :mod:`tera.taxonomy`:

  - PELT change-point detection (ruptures, l2 cost, BIC penalty)
  - Johansen cointegration test
  - Granger causality on log-differenced (stationarity-corrected) series
  - Time-lagged cross-correlation (TLCC) on log-differenced series

The stationarity correction is the methodological core: raw patent and
standards series share a strong upward trend, so tests are run on
log-differenced series to avoid spurious correlation / causality.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint as johansen_coint, grangercausalitytests

import warnings
warnings.filterwarnings("ignore", message="verbose is deprecated")


def log_diff(x) -> np.ndarray:
    """Log-difference for stationarity: Δlog(x) = log(max(x,1)) - log(max(x,1))[-1]."""
    x = np.asarray(x, dtype=float)
    return np.diff(np.log(np.maximum(x, 1.0)))


# ------------------------------------------------------------
# Change-point detection
# ------------------------------------------------------------
def detect_changepoints(
    signal, years: Optional[List[int]] = None, min_size: int = 2, penalty: str = "bic"
) -> List[int]:
    """
    PELT change-point detection on a single series.

    Returns changepoint *years* if `years` is provided, otherwise indices.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.sum() <= 0:
        return []
    try:
        import ruptures as rpt
    except ImportError:
        return []

    pen = {"bic": np.log(len(signal)), "aic": 2.0}.get(penalty, np.log(len(signal)))
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
    cps = algo.predict(pen=pen)
    cps = [cp for cp in cps if cp < len(signal)]
    return [years[cp] for cp in cps] if years is not None else cps


# ------------------------------------------------------------
# Causal discovery (cointegration + Granger + TLCC)
# ------------------------------------------------------------
def cointegration_test(patents, standards, significance: float = 0.05) -> dict:
    """Johansen cointegration test between two level series."""
    try:
        _, pvalue, _ = johansen_coint(np.asarray(patents, float), np.asarray(standards, float))
        return {"cointegrated": bool(pvalue < significance), "p_value": round(float(pvalue), 4)}
    except Exception:
        return {"cointegrated": False, "p_value": None}


def granger_test(
    patents, standards, max_lag: int = 3, significance: float = 0.1
) -> dict:
    """
    Granger causality on log-differenced series (patents -> standards).

    Column order [dependent, independent] = [standards, patents].
    """
    p = log_diff(np.asarray(patents, float))
    s = log_diff(np.asarray(standards, float))
    n = min(len(p), len(s))
    actual_lag = min(max_lag, n - 3)
    if actual_lag < 1:
        return {"granger_causal": False, "p_value": None, "best_lag": None}

    data = np.column_stack([s[:n], p[:n]])
    try:
        result = grangercausalitytests(data, maxlag=actual_lag, verbose=False)
    except Exception:
        return {"granger_causal": False, "p_value": None, "best_lag": None}

    best_lag, best_p = 1, 1.0
    for lag in range(1, actual_lag + 1):
        pv = result[lag][0]["ssr_ftest"][1]
        if pv < best_p:
            best_p, best_lag = pv, lag
    return {
        "granger_causal": bool(best_p < significance),
        "p_value": round(float(best_p), 4),
        "best_lag": best_lag,
    }


def tlcc(patents, standards, max_lag: int = 6) -> dict:
    """Time-lagged cross-correlation on log-differenced series."""
    p = log_diff(np.asarray(patents, float))
    s = log_diff(np.asarray(standards, float))
    n = len(p)
    best_corr, best_lag = 0.0, 0
    for lag in range(0, min(max_lag, n // 3)):
        if lag == 0:
            c = stats.pearsonr(p, s)[0]
        elif lag < len(p):
            c = stats.pearsonr(p[lag:], s[:-lag])[0]
        else:
            c = 0.0
        if abs(c) > abs(best_corr):
            best_corr, best_lag = c, lag
    return {"tlcc_r": round(float(best_corr), 3), "tlcc_lag": best_lag}


def analyze_category(
    patents,
    standards,
    max_lag: int = 3,
    tlcc_max_lag: int = 6,
    coint_significance: float = 0.05,
    granger_significance: float = 0.1,
) -> dict:
    """Run the full temporal analysis for one category's (patents, standards)."""
    patents = np.asarray(patents, float)
    standards = np.asarray(standards, float)
    if patents.sum() == 0:
        return {
            "cointegrated": False, "coint_p": None,
            "granger_causal": False, "granger_p": None, "granger_lag": None,
            "tlcc_r": 0.0, "tlcc_lag": 0,
        }
    coint = cointegration_test(patents, standards, coint_significance)
    gc = granger_test(patents, standards, max_lag, granger_significance)
    tc = tlcc(patents, standards, tlcc_max_lag)
    return {
        "cointegrated": coint["cointegrated"], "coint_p": coint["p_value"],
        "granger_causal": gc["granger_causal"], "granger_p": gc["p_value"],
        "granger_lag": gc["best_lag"],
        "tlcc_r": tc["tlcc_r"], "tlcc_lag": tc["tlcc_lag"],
    }


# ------------------------------------------------------------
# Standards & policy data loading
# ------------------------------------------------------------
def load_standards_years(path: str) -> List[tuple]:
    """Load (standard, body, year) tuples from config/standards.csv."""
    if not path or not __import__("os").path.exists(path):
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    return [(r["standard"], r["body"], int(r["year"])) for _, r in df.iterrows()]


def load_policies(path: str) -> pd.DataFrame:
    """Load the policy-events timeline from config/policies.csv."""
    if not path or not __import__("os").path.exists(path):
        return pd.DataFrame(columns=["year", "jurisdiction", "name", "type"])
    return pd.read_csv(path, encoding="utf-8-sig")


def build_standards_yearly(
    standards_path: str, years: List[int], orig_std_years: Optional[List[int]] = None
) -> np.ndarray:
    """
    Build yearly standards-activity counts by combining ISO/IEEE publication
    years (from CSV) with any original standards years (e.g. parsed from the
    STIM baseline columns).
    """
    from collections import Counter

    all_years = list(orig_std_years or [])
    for _, _, yr in load_standards_years(standards_path):
        all_years.append(yr)
    counts = Counter(all_years)
    return np.array([counts.get(y, 0) for y in years], dtype=float)
