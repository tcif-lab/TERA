# Data preparation guide

TERA's analysis runs on **derived** data (already shipped in `data/`). This guide
documents how to *reconstruct* that derived data from the raw corpora, and the exact
input schemas the scripts expect.

> The raw corpora are **not** redistributed with this repository:
> - **Patents** are from IncoPat (a licensed commercial database).
> - **Standards** (GB/T, ISO, IEC, IEEE) are copyrighted documents.
>
> Place them under `data/raw/` (git-ignored) following the layout below.

## 1. Patents — `data/raw/patents.csv`

A single CSV with one row per patent. Only three columns are used:

| column                  | description                          | example            |
|-------------------------|--------------------------------------|--------------------|
| `Earliest Priority Date`| priority date (ISO or `YYYY-MM-DD`)  | `2015-03-12`       |
| `IPC`                   | IPC code(s), `;`-separated           | `B25J9/16;G05B19/04` |
| `Title (English)`       | (optional) bilingual title           | `Robot arm`        |

The paper uses 45,982 industrial-robotics patents (1978–2025) exported from IncoPat,
covering USPTO, EPO, and CNIPA.

## 2. Standards — `data/raw/standards/`

One CSV per standard document, named after the standard (e.g. `ISO_10218-1.csv`).
Each CSV holds the OCR'd text of the document; the first ~100 rows are read and
flattened into a single text field.

The paper uses 85 standards: 57 from the original STIM baseline (Chinese GB/T and DB),
plus 16 ISO/TC 299 and 12 IEEE Robotics & Automation standards. ISO/IEEE publication
years are recorded in `config/standards.csv`.

## 3. Policies — `config/policies.csv`

The policy-events timeline is already curated and shipped in the repository
(28 events: China 17, EU 7, US 4). Columns: `year`, `jurisdiction`, `name`, `type`.

## 4. Reconstruct the derived data

```bash
# patents  -> per-category yearly time series
python scripts/extract_ts_data.py

# standards -> STIM matrix (BM25)
python scripts/rebuild_stim.py

# then run the full analysis + figures
python scripts/run_expanded.py
python scripts/generate_figures.py
```

The derived outputs are written to `output/robotics/` and can be copied back into
`data/` to refresh the shipped files.

## 5. Outputs produced

| script                  | output                                          |
|-------------------------|-------------------------------------------------|
| `extract_ts_data.py`    | `output/robotics/ts_data_yearly.csv`            |
| `rebuild_stim.py`       | `output/robotics/stim_matrix.csv`               |
| `run_expanded.py`       | `r3_var_ratios.csv`, `r4_*.csv`, `r5_*.csv`, `r2_annotation_template.csv` |
| `generate_figures.py`   | `output/robotics/figures/*.png`                 |
