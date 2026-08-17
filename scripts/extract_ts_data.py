#!/usr/bin/env python3
"""
Reconstruct yearly patent time series per technology category from raw data.

Requires the raw patent CSV (see docs/data_preparation.md). With raw data
absent, this script prints instructions and exits. Run from the repo root:

    python scripts/extract_ts_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tera.config import Config
from tera.taxonomy import load_ipc_routing, build_patent_time_series

YEARS = list(range(1978, 2026))


def main() -> None:
    cfg = Config.from_yaml("config/robotics.yaml")
    patent_path = cfg.patent_path

    if not os.path.exists(patent_path):
        print(f"[ERROR] Raw patent data not found: {patent_path}")
        print("TERA ships derived time series in data/ts_data_yearly.csv so the")
        print("analysis can run without raw data. To reconstruct from scratch,")
        print("see docs/data_preparation.md for the required CSV schema.")
        sys.exit(1)

    subclass_map, class_map = load_ipc_routing(cfg.ipc_routing_path)
    df = pd.read_csv(
        patent_path, encoding="utf-8-sig",
        usecols=["Earliest Priority Date", "IPC"],
    )
    print(f"Loaded {len(df):,} patents")

    ts_data, n_unclassified = build_patent_time_series(
        df, subclass_map, class_map, YEARS
    )

    rows = [
        {"tech": tech, "year": y, "patents": ts_data[tech][i]}
        for tech in ts_data
        for i, y in enumerate(YEARS)
    ]
    out = cfg.output_path
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "ts_data_yearly.csv"),
                              index=False, encoding="utf-8-sig")

    yield_ = 100 * (len(df) - n_unclassified) / len(df) if len(df) else 0.0
    print(f"Classification yield: {yield_:.1f}% "
          f"({len(df) - n_unclassified:,}/{len(df):,} patents)")
    print(f"Saved {out}/ts_data_yearly.csv")


if __name__ == "__main__":
    main()
