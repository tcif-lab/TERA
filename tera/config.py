"""
TERA Configuration — YAML-driven, domain-agnostic.

All paths, parameters, and domain-specific settings are read from a YAML
file (or passed programmatically). Relative paths are resolved against the
repository root, which is auto-detected from the location of this module,
so the code is portable and contains no machine-specific absolute paths.

Example YAML (config/robotics.yaml):
    domain: industrial_robotics
    data:
      patents: data/raw/patents.csv
      standards_dir: data/raw/standards/
      tech_categories: config/tech_categories.yaml
      ipc_routing: config/ipc_routing.csv
      standards_years: config/standards.csv
      policies: config/policies.csv
    parameters: { bm25_k1: 1.5, bm25_b: 0.75, ... }
    output: { results_dir: output/robotics/, figures_dir: output/robotics/figures/ }
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

# Repository root = parent directory of this `tera/` package.
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    """Domain-agnostic TERA configuration."""

    # ---- Domain identity ----
    domain: str = "default"

    # ---- Data paths (relative to repo root) ----
    patent_file: str = ""
    standards_dir: str = ""
    tech_categories_file: str = ""
    ipc_routing_file: str = ""
    standards_years_file: str = ""
    policies_file: str = ""

    # ---- Output ----
    results_dir: str = "output/"
    figures_dir: str = "output/figures/"

    # ---- BM25 ----
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---- Bootstrap ----
    bootstrap_b: int = 200

    # ---- PELT ----
    pelt_min_size: int = 2
    pelt_model: str = "l2"

    # ---- Granger ----
    granger_max_lag: int = 3
    granger_significance: float = 0.1

    # ---- Cointegration ----
    coint_significance: float = 0.05

    # ---- TLCC ----
    tlcc_max_lag: int = 6

    # ---- Policy alignment ----
    policy_windows: List[int] = field(default_factory=lambda: [1, 2, 3])

    # ---- Technology categories (loaded from tech_categories.yaml) ----
    tech_names: List[str] = field(default_factory=list)
    tech_queries: Dict[str, str] = field(default_factory=dict)   # bilingual BM25 query per category
    tech_sources: Dict[str, str] = field(default_factory=dict)   # ISO/IFR/WIPO source per category

    # ---- Policy jurisdictions ----
    jurisdictions: List[str] = field(default_factory=lambda: ["China", "EU", "USA"])

    # ---- Project root (auto-detected) ----
    project_root: str = field(default_factory=lambda: str(REPO_ROOT))

    # ------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------
    def resolve(self, path: str) -> str:
        """Resolve a path against project_root; absolute paths pass through."""
        if not path:
            return path
        p = Path(path)
        return path if p.is_absolute() else str(Path(self.project_root) / p)

    @property
    def patent_path(self) -> str:
        return self.resolve(self.patent_file)

    @property
    def standards_path(self) -> str:
        return self.resolve(self.standards_dir)

    @property
    def tech_categories_path(self) -> str:
        return self.resolve(self.tech_categories_file)

    @property
    def ipc_routing_path(self) -> str:
        return self.resolve(self.ipc_routing_file)

    @property
    def standards_years_path(self) -> str:
        return self.resolve(self.standards_years_file)

    @property
    def policies_path(self) -> str:
        return self.resolve(self.policies_file)

    @property
    def output_path(self) -> str:
        return self.resolve(self.results_dir)

    @property
    def figures_path(self) -> str:
        return self.resolve(self.figures_dir)

    def ensure_dirs(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        cfg.domain = data.get("domain", "default")

        # Data paths
        d = data.get("data", {}) or {}
        cfg.patent_file = d.get("patents", "")
        cfg.standards_dir = d.get("standards_dir", "")
        cfg.tech_categories_file = d.get("tech_categories", "")
        cfg.ipc_routing_file = d.get("ipc_routing", "")
        cfg.standards_years_file = d.get("standards_years", "")
        cfg.policies_file = d.get("policies", "")

        # Output
        o = data.get("output", {}) or {}
        cfg.results_dir = o.get("results_dir", "output/")
        cfg.figures_dir = o.get("figures_dir", "output/figures/")

        # Parameters
        p = data.get("parameters", {}) or {}
        cfg.bm25_k1 = p.get("bm25_k1", 1.5)
        cfg.bm25_b = p.get("bm25_b", 0.75)
        cfg.bootstrap_b = p.get("bootstrap_b", 200)
        cfg.pelt_min_size = p.get("pelt_min_size", 2)
        cfg.pelt_model = p.get("pelt_model", "l2")
        cfg.granger_max_lag = p.get("granger_max_lag", 3)
        cfg.granger_significance = p.get("granger_significance", 0.1)
        cfg.coint_significance = p.get("coint_significance", 0.05)
        cfg.tlcc_max_lag = p.get("tlcc_max_lag", 6)
        cfg.policy_windows = list(p.get("policy_windows", [1, 2, 3]))
        cfg.jurisdictions = list(p.get("jurisdictions", ["China", "EU", "USA"]))

        # Technology categories
        cfg._load_tech_categories()

        cfg.ensure_dirs()
        return cfg

    def _load_tech_categories(self) -> None:
        """Load the 10-category taxonomy from tech_categories.yaml."""
        path = self.tech_categories_path
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            tc = yaml.safe_load(f) or {}
        for cat in tc.get("categories", []):
            name = cat["name"]
            self.tech_names.append(name)
            # Bilingual BM25 query = English + Chinese concatenated.
            en = cat.get("description_en", "").strip()
            zh = cat.get("description_zh", "").strip()
            self.tech_queries[name] = f"{en} {zh}".strip()
            self.tech_sources[name] = cat.get("source", "")

    def to_dict(self) -> dict:
        """Export config as a dict (for logging / reproducibility)."""
        return {
            "domain": self.domain,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "bootstrap_b": self.bootstrap_b,
            "pelt_min_size": self.pelt_min_size,
            "pelt_model": self.pelt_model,
            "granger_max_lag": self.granger_max_lag,
            "granger_significance": self.granger_significance,
            "coint_significance": self.coint_significance,
            "tlcc_max_lag": self.tlcc_max_lag,
            "policy_windows": self.policy_windows,
            "n_tech_categories": len(self.tech_names),
            "jurisdictions": self.jurisdictions,
        }
