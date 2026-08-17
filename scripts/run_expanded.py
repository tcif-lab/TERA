#!/usr/bin/env python3
"""
Reproduce TERA's R2-R5 analysis from the derived data shipped in `data/`.

No raw (licensed) data is required. Run from the repository root:

    python scripts/run_expanded.py

Outputs (R3/R4/R5 + temporal results) are written to `output/robotics/`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tera.config import Config
from tera.pipeline import TERAPipeline
from tera.temporal import detect_changepoints, analyze_category
from tera.robustness import granger_robustness_report
from tera.baselines import (
    pearson_spearman_correlation_baseline,
    stage2_component_ablation,
    temporal_ablation_report,
)
from tera.sensitivity import sensitivity_report
from tera.annotation import stratified_sample_pairs


def main() -> None:
    cfg = Config.from_yaml("config/robotics.yaml")
    cfg.ensure_dirs()
    out = cfg.output_path
    os.makedirs(out, exist_ok=True)

    # ---- 1. Load derived data ----
    pipe = TERAPipeline(cfg)
    ts_data = pipe.load_ts_data()                          # {category: np.array}
    years = pipe.years
    std_yearly = pd.read_csv(
        os.path.join(cfg.project_root, "data", "std_yearly.csv"), encoding="utf-8-sig"
    )["standards"].to_numpy(dtype=float)
    policies = pd.read_csv(cfg.policies_path, encoding="utf-8-sig")
    policy_years = sorted(policies["year"].astype(int).tolist())
    techs = cfg.tech_names or list(ts_data.keys())
    yearly_data = {t: {"patents": ts_data[t], "standards": std_yearly} for t in techs}

    # ---- 2. Temporal mining: PELT + causal discovery ----
    changepoints, causal = {}, {}
    for tech in techs:
        changepoints[tech] = detect_changepoints(ts_data[tech], years, cfg.pelt_min_size)
        causal[tech] = analyze_category(
            ts_data[tech], std_yearly,
            max_lag=cfg.granger_max_lag, tlcc_max_lag=cfg.tlcc_max_lag,
            coint_significance=cfg.coint_significance,
            granger_significance=cfg.granger_significance,
        )

    n_coint = sum(1 for t in techs if causal[t]["cointegrated"])
    n_gc = sum(1 for t in techs if causal[t]["granger_causal"])
    print(f"\nCausal discovery: cointegrated {n_coint}/{len(techs)} | "
          f"Granger-causal {n_gc}/{len(techs)}")

    # ---- 3. R3: Granger robustness (permutation + VAR power) ----
    rob = granger_robustness_report(yearly_data, techs, n_permutations=1000)
    print("\n" + rob["summary"])
    rob["var_ratios"].to_csv(os.path.join(out, "r3_var_ratios.csv"), index=False)

    # ---- 4. R4: correlation baselines + Stage-2 ablation ----
    corr = pearson_spearman_correlation_baseline(
        {t: ts_data[t] for t in techs}, {t: std_yearly for t in techs}, techs
    )
    ablation = stage2_component_ablation(yearly_data, techs, years)
    print(temporal_ablation_report(ablation))
    corr.to_csv(os.path.join(out, "r4_correlation_baselines.csv"), index=False)

    # ---- 5. R5: PELT penalty + policy-window sensitivity ----
    sens = sensitivity_report(
        {t: ts_data[t] for t in techs}, techs, years, changepoints, policy_years
    )
    print("\n" + sens["summary"])
    sens["changepoint_stability"].to_csv(os.path.join(out, "r5_pelt_stability.csv"), index=False)
    sens["policy_window"].to_csv(os.path.join(out, "r5_policy_window.csv"), index=False)

    # ---- 6. R2: annotation template (for re-annotation) ----
    stim = pd.read_csv(os.path.join(cfg.project_root, "data", "stim_matrix.csv"),
                       index_col=0, encoding="utf-8-sig")
    pairs = stratified_sample_pairs(stim.values, techs, list(stim.columns), n_per_category=5)
    pairs.to_csv(os.path.join(out, "r2_annotation_template.csv"),
                 index=False, encoding="utf-8-sig")

    print(f"\nDone. Results written to {out}")


if __name__ == "__main__":
    main()
