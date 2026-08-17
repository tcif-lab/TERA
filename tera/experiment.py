"""
TERA Experiment Runner — One-command experiments + comparison tables.

Usage:
    from tera.config import Config
    from tera.experiment import Experiment

    cfg = Config.from_yaml("experiments/robotics.yaml")
    exp = Experiment(cfg)
    exp.run()
    exp.save_results()
    exp.print_comparison_table()
"""
import os, json, time
import numpy as np
import pandas as pd

from tera.baselines import (
    compute_tfidf_alignment,
    static_only_results,
    random_alignment,
    run_all_baselines,
    format_comparison_table,
)


class Experiment:
    """
    Orchestrates TERA experiments with baseline comparison.

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config):
        self.config = config
        self.config.ensure_dirs()
        self.results = {}
        self.baseline_table = None
        self._start_time = None

    def run(self, alignment_data=None):
        """
        Run all experiments.

        Parameters
        ----------
        alignment_data : dict or None
            If provided, should contain numpy arrays for each method.
            If None, loads from output directory.
            Keys: bm25, sbert, ipc, kg, full, tech_texts, std_texts
        """
        self._start_time = time.time()
        print("=" * 60)
        print(f"TERA Experiment: {self.config.domain}")
        print("=" * 60)

        # Load or use provided data
        if alignment_data is None:
            alignment_data = self._load_alignment_data()

        # Compute TF-IDF baseline
        print("\n[1/4] Computing TF-IDF baseline...")
        tfidf_matrix = compute_tfidf_alignment(
            alignment_data.get("tech_texts", []),
            alignment_data.get("std_texts", []),
        )
        if tfidf_matrix.shape == alignment_data.get("bm25", np.zeros((1,1))).shape:
            self.results["tfidf"] = tfidf_matrix
            print(f"  Shape: {tfidf_matrix.shape}, Mean: {tfidf_matrix.mean():.3f}")
        else:
            # Fallback: use random matrix of same shape
            print("  [WARN] TF-IDF shape mismatch — using placeholder")
            ref = alignment_data.get("bm25", np.zeros((12, 55)))
            self.results["tfidf"] = random_alignment(ref.shape[0], ref.shape[1], seed=1)

        # Compute Random baseline
        print("\n[2/4] Computing Random baseline...")
        ref_matrix = alignment_data.get("bm25", np.zeros((12, 55)))
        rand_matrix = random_alignment(ref_matrix.shape[0], ref_matrix.shape[1])
        self.results["random"] = rand_matrix
        print(f"  Shape: {rand_matrix.shape}, Mean: {rand_matrix.mean():.3f}")

        # Compute Static-Only baseline
        print("\n[3/4] Computing Static-Only baseline...")
        static = static_only_results(
            alignment_data.get("bm25", np.zeros((12, 55))),
            alignment_data.get("tech_names", []),
            list(range(1978, 2026)),
        )
        self.results["static"] = static
        print(f"  Mean: {static['mean_score']:.3f}, CV: {static['cv']:.2f}")

        # Compile comparison table
        print("\n[4/4] Compiling baseline comparison table...")
        bm25 = alignment_data.get("bm25", np.zeros((12, 55)))
        sbert = alignment_data.get("sbert", bm25)
        ipc = alignment_data.get("ipc", bm25)
        kg = alignment_data.get("kg", bm25)
        full = alignment_data.get("full", bm25)

        self.baseline_table = run_all_baselines(
            alignment_bm25=bm25,
            alignment_tfidf=self.results["tfidf"],
            alignment_sbert=sbert,
            alignment_ipc=ipc,
            alignment_kg=kg,
            alignment_full=full,
            static_result=static,
            random_matrix=rand_matrix,
        )

        elapsed = time.time() - self._start_time
        print(f"\nExperiment complete ({elapsed:.1f}s)")
        return self.baseline_table

    def _load_alignment_data(self):
        """Try to load alignment matrices from output directory."""
        data = {}
        output = self.config.output_path

        # Try loading CSV files
        for key, fname in [
            ("bm25", "STIM_baseline.csv"),
            ("sbert", "STIM_sbert_semantic.csv"),
            ("ipc", "moduleC_ipc_scores.csv"),
            ("kg", "moduleB_kg_scores.csv"),
            ("full", "STIM_ensemble_v4.csv"),
        ]:
            path = os.path.join(output, fname)
            if os.path.exists(path):
                try:
                    data[key] = pd.read_csv(path, index_col=0).values
                except Exception:
                    pass

        if not data:
            print("[WARN] No alignment data found — using placeholder matrices")
            data["bm25"] = np.random.rand(12, 55)
            data["tech_names"] = [f"Tech_{i}" for i in range(12)]
            data["tech_texts"] = [f"Technology category {i}" for i in range(12)]
            data["std_texts"] = [f"Standard document {i}" for i in range(55)]

        return data

    def save_results(self):
        """Save experiment results to output directory."""
        output = self.config.output_path

        # Save comparison table as CSV
        if self.baseline_table:
            table_path = os.path.join(output, "baseline_comparison.csv")
            pd.DataFrame(self.baseline_table).to_csv(table_path, index=False)
            print(f"[OK] Comparison table: {table_path}")

        # Save LaTeX table
        if self.baseline_table:
            tex_path = os.path.join(output, "baseline_comparison.tex")
            tex = format_comparison_table(self.baseline_table, fmt="latex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"[OK] LaTeX table: {tex_path}")

        # Save experiment metadata
        meta = {
            "domain": self.config.domain,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config.to_dict(),
            "n_baselines": len(self.baseline_table) if self.baseline_table else 0,
        }
        meta_path = os.path.join(output, "experiment_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[OK] Metadata: {meta_path}")

    def print_comparison_table(self):
        """Print the baseline comparison table to console."""
        if not self.baseline_table:
            print("[WARN] No results — run experiment first.")
            return
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)
        print(format_comparison_table(self.baseline_table, fmt="markdown"))
