#!/usr/bin/env python3
"""
Build the STIM matrix (BM25: category descriptions x standard documents).

Requires the OCR'd standard CSVs (see docs/data_preparation.md). Run from
the repo root:

    python scripts/rebuild_stim.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tera.config import Config
from tera.alignment import load_standard_texts, build_stim_matrix, stim_to_dataframe


def main() -> None:
    cfg = Config.from_yaml("config/robotics.yaml")
    std_dir = cfg.standards_path

    if not os.path.isdir(std_dir):
        print(f"[ERROR] Raw standards not found: {std_dir}")
        print("TERA ships a precomputed STIM matrix in data/stim_matrix.csv.")
        print("To rebuild from scratch, see docs/data_preparation.md.")
        sys.exit(1)

    std_texts = load_standard_texts(std_dir)
    print(f"Loaded {len(std_texts)} standard documents")

    matrix, cat_names, std_names = build_stim_matrix(
        cfg.tech_queries, std_texts, k1=cfg.bm25_k1, b=cfg.bm25_b
    )
    stim_df = stim_to_dataframe(matrix, cat_names, std_names)

    out = cfg.output_path
    os.makedirs(out, exist_ok=True)
    stim_df.to_csv(os.path.join(out, "stim_matrix.csv"), encoding="utf-8-sig")
    print(f"Saved {out}/stim_matrix.csv  ({stim_df.shape})")


if __name__ == "__main__":
    main()
