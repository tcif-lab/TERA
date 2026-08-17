#!/usr/bin/env python3
"""
Render TERA's result figures from the derived data in `data/`.

Run from the repository root:

    python scripts/generate_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tera.config import Config
from tera.pipeline import TERAPipeline
from tera.figures import generate_result_figures


def main() -> None:
    cfg = Config.from_yaml("config/robotics.yaml")
    cfg.ensure_dirs()

    pipe = TERAPipeline(cfg)
    ts_data = pipe.load_ts_data()
    years = pipe.years
    std_yearly = pd.read_csv(
        os.path.join(cfg.project_root, "data", "std_yearly.csv"), encoding="utf-8-sig"
    )["standards"].to_numpy(dtype=float)

    stim = pd.read_csv(os.path.join(cfg.project_root, "data", "stim_matrix.csv"),
                       index_col=0, encoding="utf-8-sig")
    ann = pd.read_csv(os.path.join(cfg.project_root, "data", "annotations.csv"),
                      encoding="utf-8-sig")

    generate_result_figures(ts_data, std_yearly, years, stim, ann, cfg.figures_path)


if __name__ == "__main__":
    main()
