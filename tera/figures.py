"""
TERA Figures — matplotlib rendering of the paper's result figures.

Reproducible from the derived data in `data/`:

  - yearly patent trend per category
  - PELT change-point detection per category
  - cross-source correlation (Pearson / Spearman / TLCC)
  - STIM alignment heatmap
  - annotation precision by category

Note: the paper's overview (Fig. 1) and methodology (Fig. 2) diagrams are
hand-crafted vector figures and are not reproduced here.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "figure.dpi": 200,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _short(name: str) -> str:
    return name.replace(" & ", " &\n")


def plot_yearly_trends(ts_data: Dict[str, np.ndarray], years: List[int], save_path: str):
    """5x2 grid of yearly patent filings per category."""
    techs = list(ts_data.keys())
    fig, axes = plt.subplots(5, 2, figsize=(8, 10))
    axes = axes.flatten()
    for i, tech in enumerate(techs[:10]):
        ax = axes[i]
        arr = ts_data[tech]
        ax.fill_between(years, arr, alpha=0.3, color=_COLORS[i])
        ax.plot(years, arr, color=_COLORS[i], linewidth=1.2)
        ax.set_title(tech, fontsize=7.5, fontweight="bold")
        ax.set_xlim(years[0], years[-1])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    fig.suptitle(f"Yearly Patent Filings by Technology Category ({years[0]}–{years[-1]})",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_pelt_changepoints(ts_data: Dict[str, np.ndarray], years: List[int], save_path: str,
                           min_size: int = 2):
    """5x2 grid of PELT change-point detection per category."""
    from tera.temporal import detect_changepoints

    techs = list(ts_data.keys())
    fig, axes = plt.subplots(5, 2, figsize=(8, 10))
    axes = axes.flatten()
    for i, tech in enumerate(techs[:10]):
        ax = axes[i]
        arr = ts_data[tech]
        if arr.sum() == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        ax.plot(years, arr, color=_COLORS[i], linewidth=1.2)
        for cp_year in detect_changepoints(arr, years, min_size=min_size):
            ax.axvline(x=cp_year, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(tech, fontsize=7.5, fontweight="bold")
        ax.set_xlim(years[0], years[-1])
    fig.suptitle("PELT Change-Point Detection by Technology Category", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cross_source_correlation(
    ts_data: Dict[str, np.ndarray], std_yearly: np.ndarray, years: List[int], save_path: str
):
    """Grouped bar chart of Pearson r / Spearman rho / TLCC per category."""
    from tera.temporal import log_diff

    techs = [t for t in ts_data if ts_data[t].sum() > 0]
    pearson_r, spearman_r, tlcc_vals, labels = [], [], [], []
    for tech in techs:
        p = ts_data[tech]
        n = min(len(p), len(std_yearly))
        pr, _ = stats.pearsonr(p[:n], std_yearly[:n])
        sr, _ = stats.spearmanr(p[:n], std_yearly[:n])
        p_d, s_d = log_diff(p[:n]), log_diff(std_yearly[:n])
        tc, _ = stats.pearsonr(p_d, s_d) if len(p_d) > 2 else (0.0, 1.0)
        pearson_r.append(pr); spearman_r.append(sr); tlcc_vals.append(tc)
        labels.append(_short(tech))

    x = np.arange(len(labels)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 1.5 * w, pearson_r, w, color="#2196F3", label="Pearson r (raw)")
    ax.bar(x - 0.5 * w, spearman_r, w, color="#4CAF50", label="Spearman ρ (raw)")
    ax.bar(x + 0.5 * w, tlcc_vals, w, color="#FF9800", label="TLCC (log-diff)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Correlation coefficient")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Cross-Source Correlation: Patents vs. Standards", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_stim_heatmap(stim_df: pd.DataFrame, save_path: str, top_n: int = 25):
    """Heatmap of the STIM matrix (categories x top-N standards)."""
    top_cols = stim_df.sum(axis=0).sort_values(ascending=False).head(top_n).index
    stim_top = stim_df[top_cols]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(stim_top.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(top_cols)))
    ax.set_xticklabels([c[:30].replace("_", " ") for c in top_cols],
                       rotation=45, ha="right", fontsize=5)
    ax.set_yticks(range(len(stim_top.index)))
    ax.set_yticklabels(stim_top.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="BM25 Score")
    ax.set_title(f"STIM Matrix: Technology Categories × Standards (Top {top_n} by BM25)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_annotation_precision(ann_df: pd.DataFrame, save_path: str):
    """Per-category alignment precision (relevance rate) from expert annotations.

    Works with either a stratum-sampled annotation file (has a ``stratum``
    column) or a pooled annotation file (relevance judgments only).
    """
    if "stratum" in ann_df.columns:
        cat_precision = {
            cat: ann_df[(ann_df["tech_category"] == cat) & (ann_df["stratum"] == "high")][
                "relevance"
            ].mean()
            for cat in ann_df["tech_category"].unique()
        }
    else:
        cat_precision = {
            cat: ann_df[ann_df["tech_category"] == cat]["relevance"].mean()
            for cat in ann_df["tech_category"].unique()
        }
    cat_precision = {k: v for k, v in cat_precision.items() if pd.notna(v)}
    cats = list(cat_precision.keys()); vals = list(cat_precision.values())
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in vals]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(range(len(cats)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=7)
    ax.set_xlabel("Relevance rate (expert annotation)")
    ax.set_xlim(0, 1.05)
    ax.set_title("BM25 Alignment Precision by Category (Expert Annotation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def generate_result_figures(ts_data, std_yearly, years, stim_df, ann_df, out_dir: str):
    """Generate all reproducible result figures into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    plot_yearly_trends(ts_data, years, os.path.join(out_dir, "yearly_trends.png"))
    plot_pelt_changepoints(ts_data, years, os.path.join(out_dir, "pelt_changepoints.png"))
    plot_cross_source_correlation(ts_data, std_yearly, years, os.path.join(out_dir, "correlation.png"))
    if stim_df is not None:
        plot_stim_heatmap(stim_df, os.path.join(out_dir, "stim_heatmap.png"))
    if ann_df is not None:
        plot_annotation_precision(ann_df, os.path.join(out_dir, "annotation_precision.png"))
    print(f"Figures written to {out_dir}")
