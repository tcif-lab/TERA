"""
TERA: Temporal Evolution and Relationship Alignment
====================================================
A general framework for mining technology evolution from heterogeneous
knowledge sources (patents, policies, standards).

Modules
-------
config      — YAML-driven, domain-agnostic configuration
taxonomy    — deterministic IPC -> category routing
alignment   — BM25 cross-source alignment (STIM matrix)
temporal    — PELT / cointegration / Granger / TLCC mining
pipeline    — end-to-end orchestrator
robustness  — permutation tests + VAR power diagnostics (R3)
sensitivity — PELT penalty + policy window sensitivity (R5)
baselines   — comparative baselines (R4)
annotation  — annotation protocol + IR metrics (R2)
figures     — matplotlib rendering of result figures

Quick start
-----------
    from tera.config import Config
    from tera.pipeline import TERAPipeline

    cfg = Config.from_yaml("config/robotics.yaml")
    pipe = TERAPipeline(cfg)
    ts = pipe.load_ts_data()
    std = pipe.load_standards_yearly()
    pipe.run_temporal(ts, std)
"""
from tera.config import Config
from tera.pipeline import TERAPipeline

__version__ = "1.0.0"
__all__ = ["Config", "TERAPipeline"]
