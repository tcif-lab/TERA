"""
TERA Annotation — Ground truth annotation protocol and helpers.

Addresses R2 (Alignment quality ground truth):
  1. Stratified sampling of patent-standard pairs
  2. Annotation protocol (binary relevance judgment guidelines)
  3. IR metrics computation (precision@k, recall@k, nDCG)

Usage:
    from tera.annotation import (
        stratified_sample_pairs,
        print_annotation_protocol,
        compute_ir_metrics,
    )

    # 1. Sample pairs
    pairs = stratified_sample_pairs(stim_matrix, tech_categories, std_names, n_per_cat=5)

    # 2. Print protocol for annotators
    print_annotation_protocol()

    # 3. After annotation, compute metrics
    metrics = compute_ir_metrics(annotations, scores, k_values=[5, 10])
"""
import numpy as np
import pandas as pd


# ============================================================
# Stratified Sampling
# ============================================================

def stratified_sample_pairs(
    stim_matrix,
    tech_categories,
    std_names,
    n_per_category=5,
    n_high=3,
    n_low=2,
    seed=42,
):
    """
    Stratified sample of patent-standard pairs for expert annotation.

    Sampling strategy:
      - Per technology category, sample n_per_category standards
      - Within each category: n_high from top-scoring, n_low from bottom-scoring
      - This ensures coverage of both likely-relevant and likely-irrelevant pairs

    Parameters
    ----------
    stim_matrix : np.ndarray — (n_tech, n_std)
    tech_categories : list of str
    std_names : list of str
    n_per_category : int — total pairs per tech category
    n_high : int — high-score pairs per category
    n_low : int — low-score pairs per category
    seed : int

    Returns
    -------
    pd.DataFrame with columns:
      pair_id, tech_category, std_document, score, score_rank, stratum
    """
    rng = np.random.RandomState(seed)
    pairs = []

    for i, tech in enumerate(tech_categories):
        scores = stim_matrix[i, :]
        n_std = len(scores)

        # Rank standards by score (descending)
        ranked_idx = np.argsort(scores)[::-1]
        high_pool = ranked_idx[:max(n_std // 4, n_high)]
        low_pool = ranked_idx[-max(n_std // 4, n_low):]

        # Sample high-score pairs
        if len(high_pool) >= n_high:
            high_selected = rng.choice(high_pool, size=n_high, replace=False)
        else:
            high_selected = high_pool

        # Sample low-score pairs
        if len(low_pool) >= n_low:
            low_selected = rng.choice(low_pool, size=n_low, replace=False)
        else:
            low_selected = low_pool

        for j, std_idx in enumerate(high_selected):
            pairs.append({
                "pair_id": f"{tech[:3]}_high_{j+1}",
                "tech_category": tech,
                "std_document": std_names[std_idx] if std_idx < len(std_names) else f"std_{std_idx}",
                "bm25_score": round(float(scores[std_idx]), 3),
                "score_rank": int(np.where(ranked_idx == std_idx)[0][0]) + 1,
                "stratum": "high",
                "relevance": None,  # to be filled by annotator
                "annotator_notes": "",
            })

        for j, std_idx in enumerate(low_selected):
            pairs.append({
                "pair_id": f"{tech[:3]}_low_{j+1}",
                "tech_category": tech,
                "std_document": std_names[std_idx] if std_idx < len(std_names) else f"std_{std_idx}",
                "bm25_score": round(float(scores[std_idx]), 3),
                "score_rank": int(np.where(ranked_idx == std_idx)[0][0]) + 1,
                "stratum": "low",
                "relevance": None,
                "annotator_notes": "",
            })

    df = pd.DataFrame(pairs)
    print(f"Sampled {len(df)} pairs: "
          f"{len(df[df['stratum']=='high'])} high-score, "
          f"{len(df[df['stratum']=='low'])} low-score, "
          f"across {len(tech_categories)} categories")
    return df


# ============================================================
# Annotation Protocol
# ============================================================

ANNOTATION_PROTOCOL = """
================================================================================
               TERA Cross-Source Alignment Annotation Protocol
================================================================================

OVERVIEW
--------
You will evaluate pairs of (Technology Category, Standard Document).
For each pair, judge whether the standard is RELEVANT to the technology
category.

Technology categories are described by short bilingual paragraphs
(see tech_categories_robotics.yaml). Standards are full PDF documents;
you may read the title, scope section, and key clauses.

RELEVANCE CRITERIA (Binary Judgment)
-------------------------------------
RELEVANT (1):
  - The standard directly addresses requirements, test methods,
    safety specifications, or performance criteria for the technology
    category.
  - Example: ISO 10218 (Robot Safety) IS relevant to "Safety" category.

NOT RELEVANT (0):
  - The standard addresses a different technology domain, or mentions
    the technology only tangentially without substantive requirements.
  - Example: A standard on general electrical safety (IEC 60204) is
    NOT directly relevant to "HRI" (human-robot interaction).

BORDERLINE CASES:
  - If the standard addresses a sub-domain or related aspect (e.g.,
    EMC testing for robots), judge YES if substantive requirements
    exist, NO if only mentioned in scope.
  - When in doubt, mark as 0 and note in "annotator_notes".

PROCEDURE
---------
1. For each pair, read the technology category description.
2. Open the standard document and read the title, scope, and 1-2
   key clauses (usually the first few pages).
3. Judge relevance: 1 (relevant) or 0 (not relevant).
4. Add any notes (optional).

OUTPUT
------
The annotation DataFrame should have the 'relevance' column filled
with 0 or 1 for each pair.

After annotation, run compute_ir_metrics() to evaluate alignment quality.

================================================================================
"""


def print_annotation_protocol():
    print(ANNOTATION_PROTOCOL)


# ============================================================
# IR Metrics from Annotations
# ============================================================

def compute_ir_metrics(annotations_df, score_columns=None, k_values=None):
    """
    Compute IR metrics from annotated relevance judgments.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Must have columns: tech_category, relevance (0/1),
        and score columns (bm25_score, sbert_score, etc.)
    score_columns : list of str — which score columns to evaluate
        Default: ["bm25_score"]
    k_values : list of int

    Returns
    -------
    pd.DataFrame — one row per (method, k), columns for precision/recall/nDCG
    """
    if score_columns is None:
        score_columns = ["bm25_score"]
    if k_values is None:
        k_values = [5, 10, 20]

    df = annotations_df.dropna(subset=["relevance"])
    df = df[df["relevance"].isin([0, 1])]

    if len(df) == 0:
        print("[WARN] No annotated pairs with valid relevance labels")
        return None

    results = []
    for method in score_columns:
        if method not in df.columns:
            continue
        scores = df[method].values
        relevance = df["relevance"].values.astype(int)

        for k in k_values:
            if k > len(scores):
                continue

            # Rank by descending score
            ranked_idx = np.argsort(scores)[::-1][:k]
            top_k_relevance = relevance[ranked_idx]

            # Precision@k
            precision = top_k_relevance.sum() / k

            # Recall@k
            total_relevant = relevance.sum()
            recall = top_k_relevance.sum() / total_relevant if total_relevant > 0 else 0.0

            # nDCG@k
            dcg = sum(
                (2 ** int(rel) - 1) / np.log2(i + 2)
                for i, rel in enumerate(top_k_relevance)
            )
            ideal_order = sorted(relevance, reverse=True)[:k]
            idcg = sum(
                (2 ** int(rel) - 1) / np.log2(i + 2)
                for i, rel in enumerate(ideal_order)
            )
            ndcg = dcg / idcg if idcg > 0 else 0.0

            results.append({
                "method": method.replace("_score", ""),
                "k": k,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "nDCG": round(ndcg, 3),
                "n_annotated": len(df),
                "n_relevant": int(total_relevant),
            })

    result_df = pd.DataFrame(results)
    return result_df


def format_ir_metrics_table(metrics_df, fmt="markdown"):
    """Format IR metrics as a table."""
    if metrics_df is None or len(metrics_df) == 0:
        return "No metrics available."

    if fmt == "markdown":
        cols = ["method", "k", "precision", "recall", "nDCG"]
        table = metrics_df[cols].to_markdown(index=False)
        return table
    elif fmt == "latex":
        rows = [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & $k$ & Precision & Recall & nDCG \\",
            r"\midrule",
        ]
        for _, row in metrics_df.iterrows():
            rows.append(
                f"{row['method']} & {row['k']} & {row['precision']:.3f} & "
                f"{row['recall']:.3f} & {row['nDCG']:.3f} \\\\"
            )
        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        return "\n".join(rows)
    else:
        return metrics_df.to_csv(index=False)


# ============================================================
# Annotation Data Management
# ============================================================

def save_annotation_template(sample_df, path):
    """Save sampled pairs as CSV template for annotation."""
    sample_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Annotation template saved to: {path}")
    print(f"Fill in the 'relevance' column (0 or 1) for each pair.")


def load_annotations(path):
    """Load completed annotations from CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    n_done = df["relevance"].notna().sum()
    n_total = len(df)
    print(f"Loaded annotations: {n_done}/{n_total} pairs annotated")
    return df
