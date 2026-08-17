"""
TERA Taxonomy — deterministic IPC-to-category routing.

The routing table is loaded from `config/ipc_routing.csv` (a fully auditable
mapping of IPC subclass/class codes to the 10 technology categories, traceable
to WIPO IPC-Technology Concordance). Classification is deterministic: each
patent is routed by its IPC codes; multi-category patents are assigned equal
weight (1 / n_matches) to avoid arbitrary single-category assignment.

See the paper's Appendix A for the complete routing table.
"""
import csv
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_ipc_routing(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load the IPC routing table.

    Returns
    -------
    (subclass_map, class_map) where `subclass_map` maps 4-char IPC subclasses
    (e.g. ``G05B``) and `class_map` maps 3-char IPC classes (e.g. ``G05``)
    to category names. 4-char subclass lookup takes precedence over 3-char.
    """
    subclass_map: Dict[str, str] = {}
    class_map: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("ipc") or "").strip().upper()
            cat = (row.get("category") or "").strip()
            if not code or not cat:
                continue
            if len(code) == 4:
                subclass_map[code] = cat
            elif len(code) == 3:
                class_map[code] = cat
    return subclass_map, class_map


def classify_patent(
    ipc_string,
    subclass_map: Dict[str, str],
    class_map: Dict[str, str],
    top_k: int = 3,
) -> List[str]:
    """
    Route a single patent to technology categories via its IPC codes.

    Returns a list of up to `top_k` category names, ranked by matched-IPC
    count. Returns an empty list when no IPC code matches (unclassified).
    """
    if ipc_string is None or not str(ipc_string).strip():
        return []
    ipcs = [s.strip() for s in str(ipc_string).split(";")]

    scores: Dict[str, float] = {}
    for ipc in ipcs:
        ipc = ipc.strip().upper().replace(" ", "")
        if not ipc:
            continue
        matched = False
        if len(ipc) >= 4:
            sub4 = ipc[:4]
            if sub4 in subclass_map:
                cat = subclass_map[sub4]
                scores[cat] = scores.get(cat, 0.0) + 1.0
                matched = True
        if not matched and len(ipc) >= 3:
            cls3 = ipc[:3]
            if cls3 in class_map:
                cat = class_map[cls3]
                scores[cat] = scores.get(cat, 0.0) + 0.5

    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def build_patent_time_series(
    patents_df: pd.DataFrame,
    subclass_map: Dict[str, str],
    class_map: Dict[str, str],
    years: List[int],
    priority_date_col: str = "Earliest Priority Date",
    ipc_col: str = "IPC",
    top_k: int = 3,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Build yearly patent-count time series per technology category.

    Parameters
    ----------
    patents_df : DataFrame with `priority_date_col` and `ipc_col` columns.
    subclass_map, class_map : routing tables from :func:`load_ipc_routing`.
    years : list of int — the years covered (e.g. 1978..2025).

    Returns
    -------
    (ts_data, n_unclassified) where ts_data maps category -> np.array of
    yearly counts (multi-category patents split equally across matches).
    """
    categories = sorted(set(subclass_map.values()) | set(class_map.values()))
    counts = {c: Counter() for c in categories}

    prio_years = pd.to_datetime(patents_df[priority_date_col], errors="coerce").dt.year
    n_unclassified = 0

    for i, yr in enumerate(prio_years):
        if pd.isna(yr) or yr < years[0] or yr > years[-1]:
            continue
        cats = classify_patent(patents_df.iloc[i][ipc_col], subclass_map, class_map, top_k=top_k)
        if not cats:
            n_unclassified += 1
            continue
        weight = 1.0 / len(cats)
        for c in cats:
            counts[c][int(yr)] += weight

    ts_data = {
        c: np.array([counts[c].get(y, 0) for y in years], dtype=float)
        for c in categories
    }
    return ts_data, n_unclassified
