"""
TERA Pipeline — end-to-end orchestrator.

Ties together the three stages of the framework:

  1. Taxonomy routing  — IPC -> category -> yearly time series (tera.taxonomy)
  2. Alignment         — BM25 STIM matrix (tera.alignment)
  3. Temporal mining   — PELT / cointegration / Granger / TLCC (tera.temporal)
                          + robustness / sensitivity / baselines

The analysis stage can run from the *derived* data shipped in `data/`
(no raw, licensed data required). Full reconstruction from raw patents /
standards is driven by the scripts in `scripts/`.
"""
import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tera.config import Config
from tera import temporal


class TERAPipeline:
    """Orchestrates TERA's analysis over category-level time series."""

    def __init__(self, config: Config):
        self.config = config
        self.config.ensure_dirs()

    # ------------------------------------------------------------
    # Derived-data loading (no raw data required)
    # ------------------------------------------------------------
    def load_ts_data(self, csv_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load yearly patent counts per category from data/ts_data_yearly.csv."""
        path = csv_path or os.path.join(self.config.project_root, "data", "ts_data_yearly.csv")
        df = pd.read_csv(path, encoding="utf-8-sig")
        self.years = sorted(df["year"].unique().tolist())
        return {tech: g["patents"].to_numpy(dtype=float) for tech, g in df.groupby("tech")}

    def load_standards_yearly(self, orig_std_years: Optional[List[int]] = None) -> np.ndarray:
        """Build standards yearly activity from config/standards.csv (+ optional originals)."""
        return temporal.build_standards_yearly(
            self.config.standards_years_path, self.years, orig_std_years
        )

    def load_policies(self) -> pd.DataFrame:
        """Load the policy-events timeline."""
        return temporal.load_policies(self.config.policies_path)

    # ------------------------------------------------------------
    # Temporal analysis
    # ------------------------------------------------------------
    def run_temporal(
        self,
        ts_data: Dict[str, np.ndarray],
        std_yearly: np.ndarray,
    ) -> Dict[str, dict]:
        """Run PELT change-points + causal discovery for every category."""
        results: Dict[str, dict] = {}
        for tech, patents in ts_data.items():
            cp_years = temporal.detect_changepoints(patents, self.years, self.config.pelt_min_size)
            causal = temporal.analyze_category(
                patents, std_yearly,
                max_lag=self.config.granger_max_lag,
                tlcc_max_lag=self.config.tlcc_max_lag,
                coint_significance=self.config.coint_significance,
                granger_significance=self.config.granger_significance,
            )
            results[tech] = {"changepoint_years": cp_years, **causal}
        self.results = results
        return results

    def summary(self) -> dict:
        """Produce a compact pipeline summary (for logging / the paper)."""
        techs = list(self.results.keys())
        n_coint = sum(1 for t in techs if self.results[t]["cointegrated"])
        n_gc = sum(1 for t in techs if self.results[t]["granger_causal"])
        return {
            "domain": self.config.domain,
            "n_categories": len(techs),
            "years": [min(self.years), max(self.years)],
            "n_cointegrated": n_coint,
            "n_granger_causal": n_gc,
        }

    def save_results(self, output_dir: Optional[str] = None) -> str:
        """Persist per-category results as JSON."""
        out = output_dir or self.config.output_path
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "temporal_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)
        return path
