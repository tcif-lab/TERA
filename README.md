# TERA: Temporal Evolution and Relationship Alignment for Heterogeneous Technology Knowledge Mining

**TERA** is a configurable, two-stage framework for mining how technologies evolve from
*heterogeneous* knowledge sources — **patents**, **policies**, and **standards**.

- **Stage 1 · Relationship Alignment** aligns documents across sources using an ensemble of
  lexical (BM25), semantic (multilingual SBERT), structural (IPC–ICS), and relational
  (knowledge-graph) signals, with systematic ablation.
- **Stage 2 · Temporal Evolution Mining** discovers structural breaks (PELT change-point
  detection), cross-source relationships (Johansen cointegration, stationarity-corrected
  Granger causality, TLCC), and policy–breakpoint alignment.

On a 48-year industrial-robotics corpus (45,982 patents, 85 standards, 28 policy events,
1978–2025), TERA finds a 2015–2017 innovation inflection cluster aligned with *Made in
China 2025*, and a policy–breakpoint alignment gradient that mirrors regulatory attention.

This repository accompanies the WSDM '27 submission:

> **TERA: Temporal Evolution and Relationship Alignment for Heterogeneous Technology Knowledge Mining**
> Usharani Hareesh Govindarajan, Yeh Geng-Hui (Sylas), Gagan Narang, Adriano Mancini. *WSDM 2027*.

---

## Repository structure

```
TERA/
├── config/              # taxonomy, IPC routing table, standards, policies, domain config
├── data/                # DERIVED data (shareable) — time series, STIM matrix, annotations
│   └── results/         #   R3/R4/R5 outputs
├── tera/                # the Python package
│   ├── config.py        #   YAML-driven configuration
│   ├── taxonomy.py      #   IPC -> category routing
│   ├── alignment.py     #   BM25 cross-source alignment
│   ├── temporal.py      #   PELT / cointegration / Granger / TLCC
│   ├── pipeline.py      #   end-to-end orchestrator
│   ├── robustness.py    #   permutation tests + VAR power diagnostics (R3)
│   ├── sensitivity.py   #   PELT penalty + policy-window sensitivity (R5)
│   ├── baselines.py     #   comparative baselines (R4)
│   ├── annotation.py    #   annotation protocol + IR metrics (R2)
│   └── figures.py       #   matplotlib result figures
├── scripts/             # thin CLI entry points
├── tests/               # pytest suite
└── docs/                # data preparation guide
```

## Installation

```bash
# Python >= 3.10
pip install -r requirements.txt
# optional: install the package in editable mode
pip install -e .
```

## Quick start — reproduce the analysis from derived data

The analysis stage runs entirely from the derived data shipped in `data/` (no raw,
licensed data required):

```bash
python scripts/run_expanded.py      # PELT + causal discovery + R3/R4/R5
python scripts/generate_figures.py  # render the result figures
python -m pytest tests/             # run the test suite
```

`run_expanded.py` reproduces the paper's headline results, including the
observation-to-parameter ratio (6.4:1) and the stationarity-corrected causal tests.

## Reproduce from raw data

Reconstructing the derived data from scratch requires the raw corpora (not distributed;
see [`docs/data_preparation.md`](docs/data_preparation.md)):

```bash
python scripts/extract_ts_data.py   # raw patents  -> data/ts_data_yearly.csv
python scripts/rebuild_stim.py      # raw standards -> data/stim_matrix.csv
python scripts/run_expanded.py      # then run the analysis as above
```

## Data & licensing

Only **derived** (aggregate) data is distributed in `data/`: yearly patent counts per
category, the BM25 STIM matrix, expert annotations, and the R3/R4/R5 result tables.
The **raw** sources are not redistributed:

- **Patents** (IncoPat, 45,982 records) — licensed, not redistributable.
- **Standards** (GB/T, ISO, IEC, IEEE) — copyrighted, not redistributable.
- **Policies** — public; the curated timeline is in `config/policies.csv`.

See [`docs/data_preparation.md`](docs/data_preparation.md) for the exact input schema and
how to source each corpus.

## Configuring a new domain

TERA is domain-agnostic. To apply it to another technology (autonomous vehicles, medical
devices, renewable energy, …):

1. Add a `config/tech_categories.yaml` with your categories + bilingual descriptions.
2. Add an `config/ipc_routing.csv` mapping IPC codes to those categories.
3. Copy `config/robotics.yaml` and point `data:` at your corpora.

All parameters (BM25 k1/b, PELT penalty, Granger lags, policy windows) are YAML-driven.

## Citation

```bibtex
@inproceedings{govindarajan2027tera,
  title     = {TERA: Temporal Evolution and Relationship Alignment for Heterogeneous Technology Knowledge Mining},
  author    = {Govindarajan, Usharani Hareesh and Yeh, Geng-Hui and Narang, Gagan and Mancini, Adriano},
  booktitle = {Proceedings of the 20th ACM International Conference on Web Search and Data Mining (WSDM '27)},
  year      = {2027}
}
```

## License

MIT — see [LICENSE](LICENSE).
