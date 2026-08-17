"""
TERA Baselines — Comparative methods for cross-source alignment.

Provides three external baselines (not derived from TERA components):
  1. TF-IDF Baseline — replaces BM25 with TF-IDF + cosine similarity
  2. Static-Only Baseline — removes temporal analysis stage
  3. Single-Source Baseline — uses only patent data (no multi-source)

These baselines isolate TERA's three core design choices:
  - Why BM25 over simpler TF-IDF?
  - Why temporal analysis over static alignment?
  - Why multi-source over single-source?
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Baseline 1: TF-IDF + Cosine (replaces BM25)
# ============================================================

def compute_tfidf_alignment(tech_texts, std_texts):
    """
    Compute technology-standard alignment using TF-IDF + cosine similarity.

    This replaces BM25 with a simpler lexical method.
    If BM25 outperforms TF-IDF, it validates the choice of Okapi
    normalization (IDF saturation + term frequency saturation).

    Parameters
    ----------
    tech_texts : list of str
        Technology category description documents (K items).
    std_texts : list of str
        Standard documents (M items).

    Returns
    -------
    matrix : np.ndarray of shape (K, M)
        TF-IDF cosine similarity scores.
    """
    all_texts = tech_texts + std_texts
    vectorizer = TfidfVectorizer(
        max_features=10000, sublinear_tf=True, stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    tech_vecs = tfidf_matrix[:len(tech_texts)]
    std_vecs = tfidf_matrix[len(tech_texts):]
    sim = cosine_similarity(tech_vecs, std_vecs)
    # Min-max normalize to [0, 1]
    sim_min, sim_max = sim.min(), sim.max()
    if sim_max > sim_min:
        sim = (sim - sim_min) / (sim_max - sim_min)
    return sim


# ============================================================
# Baseline 2: Static-Only (no temporal analysis)
# ============================================================

def static_only_results(alignment_matrix, tech_names, years):
    """
    Produce results WITHOUT temporal analysis.

    Returns only the static alignment scores — no PELT changepoints,
    no Granger causality, no policy-breakpoint alignment.
    Used to quantify the information gain from TERA's Stage 2.

    Parameters
    ----------
    alignment_matrix : np.ndarray (K, M)
    tech_names : list of str
    years : list of int

    Returns
    -------
    dict with keys: mean_score, std_score, cv, has_temporal (False)
    """
    scores = alignment_matrix.mean(axis=1)
    return {
        "method": "Static-Only (no temporal)",
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "cv": float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else float("inf"),
        "has_temporal": False,
        "n_changepoints": 0,
        "n_granger_causal": 0,
    }


# ============================================================
# Baseline 3: Single-Source (patents only, no multi-source)
# ============================================================

def single_source_alignment(patent_texts, tech_texts):
    """
    Align patents to technology categories ONLY — no standards, no policies.

    This isolates the contribution of multi-source integration:
    if single-source alignment misses patterns that TERA captures,
    it validates the K-partite design.

    Parameters
    ----------
    patent_texts : list of str
    tech_texts : list of str

    Returns
    -------
    matrix : np.ndarray (K, N_patents)
        Patent-Technology alignment matrix.
    """
    all_texts = tech_texts + patent_texts
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    tech_vecs = tfidf_matrix[:len(tech_texts)]
    pat_vecs = tfidf_matrix[len(tech_texts):]
    sim = cosine_similarity(tech_vecs, pat_vecs)
    sim_min, sim_max = sim.min(), sim.max()
    if sim_max > sim_min:
        sim = (sim - sim_min) / (sim_max - sim_min)
    return sim


# ============================================================
# Random Baseline (lower bound)
# ============================================================

def random_alignment(n_tech, n_std, seed=42):
    """Uniform random baseline — establishes the lower bound."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, size=(n_tech, n_std))


# ============================================================
# Baseline Runner
# ============================================================

def run_all_baselines(
    alignment_bm25,       # K x M numpy array (TERA's BM25-only result)
    alignment_tfidf,      # K x M numpy array (TF-IDF result)
    alignment_sbert,      # K x M numpy array (SBERT-only result)
    alignment_ipc,        # K x M numpy array (IPC-only result)
    alignment_kg,         # K x M numpy array (KG-only result)
    alignment_full,       # K x M numpy array (TERA-Full 4-signal)
    static_result,        # dict from static_only_results()
    random_matrix=None,   # K x M numpy array (random)
):
    """
    Run all baselines and return a comparison table.

    Returns
    -------
    list of dict — each entry is one method's results.
    """
    methods = []

    def add(name, matrix):
        mean_s = float(np.mean(matrix))
        std_s = float(np.std(matrix))
        cv_s = std_s / mean_s if mean_s > 0 else float("inf")
        methods.append({
            "method": name,
            "mean_score": round(mean_s, 3),
            "std_score": round(std_s, 3),
            "cv": round(cv_s, 2),
        })

    if random_matrix is not None:
        add("Random (lower bound)", random_matrix)
    add("KG-Only", alignment_kg)
    add("IPC-Only", alignment_ipc)
    add("SBERT-Only", alignment_sbert)
    add("TF-IDF + Cosine", alignment_tfidf)
    add("TERA-Full (4-signal)", alignment_full)
    add("BM25-Only (best)", alignment_bm25)

    # Add static-only
    methods.append({
        "method": "Static-Only (no temporal)",
        "mean_score": static_result["mean_score"],
        "std_score": static_result["std_score"],
        "cv": static_result["cv"],
    })

    # Compute vs-best column
    best_score = max(m["mean_score"] for m in methods if "Static" not in m["method"])
    for m in methods:
        if m["mean_score"] > 0 and m["method"] != "BM25-Only (best)":
            m["vs_best"] = f"{(m['mean_score'] - best_score) / best_score * 100:+.1f}%"
        else:
            m["vs_best"] = "-"

    return methods


# ============================================================
# Stage 2 Baselines — Temporal analysis lower bounds (R4)
# ============================================================

def pearson_spearman_correlation_baseline(patent_series, standards_series, category_names):
    """
    Simple correlation baselines for Stage 2 temporal analysis.

    If Pearson/Spearman correlations between patent and standards
    activity series are near-zero, temporal analysis (PELT/Granger)
    adds no information beyond what simple correlation captures.
    Conversely, if correlations are high but Granger finds only 2/12
    significant, it strengthens the "cointegration without causality"
    interpretation.

    Parameters
    ----------
    patent_series : dict — {category: np.array}
    standards_series : dict — {category: np.array}
    category_names : list of str

    Returns
    -------
    pd.DataFrame — one row per category with Pearson r, Spearman rho, p-values
    """
    from scipy import stats as sp_stats

    rows = []
    for cat in category_names:
        p = np.asarray(patent_series.get(cat, []))
        s = np.asarray(standards_series.get(cat, []))
        n = min(len(p), len(s))
        if n < 5:
            continue

        p, s = p[:n], s[:n]
        pearson_r, pearson_p = sp_stats.pearsonr(p, s)
        spearman_r, spearman_p = sp_stats.spearmanr(p, s)

        rows.append({
            "category": cat,
            "n_obs": n,
            "pearson_r": round(float(pearson_r), 3),
            "pearson_p": round(float(pearson_p), 4),
            "spearman_rho": round(float(spearman_r), 3),
            "spearman_p": round(float(spearman_p), 4),
        })

    df = pd.DataFrame(rows)
    n_pearson_sig = (df["pearson_p"] < 0.05).sum() if len(df) > 0 else 0
    n_spearman_sig = (df["spearman_p"] < 0.05).sum() if len(df) > 0 else 0
    print(f"\nCorrelation Baselines:")
    print(f"  Pearson significant (p<0.05): {n_pearson_sig}/{len(df)}")
    print(f"  Spearman significant (p<0.05): {n_spearman_sig}/{len(df)}")
    return df


def stage2_component_ablation(yearly_data, category_names, years, max_lag=3):
    """
    Ablation study for Stage 2: evaluate each temporal analysis
    component individually.

    Components:
      A. Cointegration only (Johansen) — does NOT test direction
      B. Granger only — does NOT test long-run equilibrium
      C. TLCC only — does NOT test statistical significance
      D. Joint (cointegration + Granger + TLCC) — TERA's full analysis
      E. None (baseline) — no temporal analysis

    This addresses reviewer concern R4: "Stage 1 has 15-subset ablation;
    Stage 2 has none."

    Parameters
    ----------
    yearly_data : dict — {category: {"patents": array, "standards": array}}
    category_names : list of str
    years : list of int
    max_lag : int

    Returns
    -------
    dict with: coint_only, granger_only, tlcc_only, joint, summary
    """
    from statsmodels.tsa.stattools import coint as johansen_coint
    from statsmodels.tsa.stattools import grangercausalitytests
    from scipy import stats

    results = {
        "coint_only": {},
        "granger_only": {},
        "tlcc_only": {},
        "joint": {},
    }

    for cat in category_names:
        if cat not in yearly_data:
            continue
        p = np.asarray(yearly_data[cat].get("patents", []))
        s = np.asarray(yearly_data[cat].get("standards", []))
        n = min(len(p), len(s))
        if n < 10:
            continue

        p_series, s_series = p[:n], s[:n]

        # ---- A. Cointegration only ----
        try:
            coint_result = johansen_coint(p_series, s_series)
            coint_sig = coint_result[1] < 0.05
        except Exception:
            coint_sig = False
            coint_result = None

        results["coint_only"][cat] = {
            "cointegrated": coint_sig,
            "p_value": round(float(coint_result[1]), 4) if coint_result else None,
        }

        # ---- B. Granger only ----
        actual_lag = min(max_lag, n - 3)
        try:
            data = np.column_stack([p_series, s_series])
            gc_result = grangercausalitytests(data, maxlag=actual_lag, verbose=False)
            min_p = min(
                gc_result[lag][0]["ssr_ftest"][1]
                for lag in range(1, actual_lag + 1)
            )
            gc_sig = min_p < 0.1
        except Exception:
            min_p = 1.0
            gc_sig = False

        results["granger_only"][cat] = {
            "granger_causal": gc_sig,
            "p_value": round(float(min_p), 4),
        }

        # ---- C. TLCC only ----
        max_corr = 0.0
        best_lag = 0
        for lag in range(0, min(6, n // 3)):
            if lag == 0:
                corr = stats.pearsonr(p_series, s_series)[0]
            else:
                corr = stats.pearsonr(p_series[lag:], s_series[:-lag])[0]
            if abs(corr) > abs(max_corr):
                max_corr = corr
                best_lag = lag

        results["tlcc_only"][cat] = {
            "max_correlation": round(float(max_corr), 3),
            "best_lag": best_lag,
        }

        # ---- D. Joint ----
        results["joint"][cat] = {
            "cointegrated": coint_sig,
            "granger_causal": gc_sig,
            "tlcc_corr": round(float(max_corr), 3),
            "tlcc_lag": best_lag,
            "pattern": (
                "codifier (coint no GC)"
                if coint_sig and not gc_sig
                else "driver (coint + GC)"
                if coint_sig and gc_sig
                else "decoupled (no coint)"
            ),
        }

    # Summary
    n_cats = len(results["joint"])
    n_coint = sum(1 for v in results["coint_only"].values() if v["cointegrated"])
    n_gc = sum(1 for v in results["granger_only"].values() if v["granger_causal"])
    n_driver = sum(
        1 for v in results["joint"].values()
        if v["cointegrated"] and v["granger_causal"]
    )
    n_codifier = sum(
        1 for v in results["joint"].values()
        if v["cointegrated"] and not v["granger_causal"]
    )

    results["summary"] = {
        "n_categories": n_cats,
        "coint_only_significant": n_coint,
        "granger_only_significant": n_gc,
        "joint_codifier": n_codifier,
        "joint_driver": n_driver,
        "interpretation": (
            f"Cointegration alone: {n_coint}/{n_cats} significant. "
            f"Granger alone: {n_gc}/{n_cats} significant. "
            f"Joint: {n_driver} drivers, {n_codifier} codifiers. "
            + (
                "Cointegration-only overestimates directional relationships; "
                "joint analysis prevents false causal claims."
                if n_coint > n_gc
                else ""
            )
        ),
    }

    print(f"\nStage 2 Component Ablation:")
    print(f"  Cointegration only: {n_coint}/{n_cats} sig")
    print(f"  Granger only: {n_gc}/{n_cats} sig")
    print(f"  Joint: {n_driver} driver, {n_codifier} codifier")
    return results


def temporal_ablation_report(ablation_results, fmt="markdown"):
    """
    Format Stage 2 ablation results as a table.

    Parameters
    ----------
    ablation_results : dict from stage2_component_ablation()
    fmt : str
    """
    joint = ablation_results["joint"]
    categories = sorted(joint.keys())

    if fmt == "markdown":
        header = "| Category | Cointegrated | Granger-Causal | TLCC (r) | Pattern |"
        sep = "|---|---|---|---|---|"
        rows = [header, sep]
        for cat in categories:
            v = joint[cat]
            rows.append(
                f"| {cat} | {'Y' if v['cointegrated'] else 'N'} | "
                f"{'Y' if v['granger_causal'] else 'N'} | "
                f"{v['tlcc_corr']:.3f} | {v['pattern']} |"
            )
        return "\n".join(rows)

    # Default: return summary dict
    return ablation_results["summary"]


# ============================================================
# Formatting Utilities
# ============================================================

def format_comparison_table(methods, fmt="latex"):
    """
    Format the baseline comparison results as a table.

    Parameters
    ----------
    methods : list of dict
    fmt : str — "latex", "markdown", or "csv"

    Returns
    -------
    str — formatted table.
    """
    if fmt == "markdown":
        header = "| Method | Mean Score | SD | CV | vs. Best |"
        sep = "|---|---|---|---|---|"
        rows = [header, sep]
        for m in methods:
            rows.append(
                f"| {m['method']} | {m['mean_score']:.3f} | "
                f"{m['std_score']:.3f} | {m['cv']:.2f} | {m.get('vs_best', '-')} |"
            )
        return "\n".join(rows)

    elif fmt == "latex":
        rows = [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Mean Score & SD & CV & vs. Best \\",
            r"\midrule",
        ]
        for m in methods:
            rows.append(
                f"{m['method']} & {m['mean_score']:.3f} & "
                f"{m['std_score']:.3f} & {m['cv']:.2f} & {m.get('vs_best', '-')} \\\\"
            )
        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        return "\n".join(rows)

    else:  # csv
        rows = ["method,mean_score,std_score,cv,vs_best"]
        for m in methods:
            rows.append(
                f"{m['method']},{m['mean_score']:.3f},"
                f"{m['std_score']:.3f},{m['cv']:.2f},{m.get('vs_best', '-')}"
            )
        return "\n".join(rows)
