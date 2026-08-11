# dt-twinarc-windsense

> Wind turbine fault detection and causal discovery using Fuhrlander FL2500 SCADA data.

A Digital Twin research project under the **TwinARC** platform. Uses real-world SCADA data from five Fuhrlander FL2500 wind turbines to detect early signs of mechanical failure and uncover causal relationships between sensor signals.

---

## Dataset

**Nature Scientific Data (2024)** — Fuhrlander FL2500 SCADA dataset.

| | |
|---|---|
| Turbines | WT80 – WT84 |
| Sensors | 78 sensors × 4 stats (avg / max / min / sdv) = 312 analog signals |
| Resolution | 5-minute intervals, ~3 years |
| Ground truth | WT84 experienced a real gearbox bearing failure in 2014 |

Raw data: `.json.bz2` per turbine + `wind_plant_data.json` (alarm dictionary, 64 Transmission alarm types).

---

## What This Project Does

```
Raw SCADA  →  Clean  →  Feature Selection  →  Pre-Causal Checks  →  Causal Discovery
                                ↓
                     ELM Anomaly Detection on WT84
```

1. **Data prep** — multi-format loader, imputation, IQR smoothing, drift removal, z-score normalisation
2. **Feature selection** — variance filter → correlation drop → two-phase Random Forest
   - Phase 1: RF(all sensors → fault alarm signal) → finds best target sensor (Y)
   - Phase 2: RF(remaining sensors → Y) → finds top predictors (X)
3. **Pre-causal checks** — stationarity (ADF), non-Gaussianity (kurtosis), ACF, time bucketing
4. **Causal discovery** — LiNGAM on bucketed outputs
5. **Anomaly detection** — ELM trained on healthy turbines (WT80–83), applied to full WT84 timeline

---

## Key Result

The ELM model — trained **only** on WT80–83 with no knowledge of WT84 — flagged anomalous behaviour on WT84 approximately **5 months before** the recorded gearbox failure date (June 2014).

---

## Repo Structure

```
├── causal-data-engine/
│   ├── 01_data_prep.ipynb          # ingest, clean, standardise
│   ├── 02_feature_selection.ipynb  # variance, correlation, two-phase RF
│   ├── 03_pre_causal.ipynb         # stationarity, kurtosis, ACF, bucketing
│   └── requirements.txt
│
├── FUHRLANDER GITHUB REPO/
│   ├── python_pipeline.ipynb       # full R+MATLAB rip-off in Python + WT84 anomaly detection
│   ├── matlab/                     # original MATLAB feature selection + ELM
│   ├── r/                          # original R data prep scripts
│   └── dataset/                    # raw .json.bz2 files (git-ignored)
│
├── Fuhrlander_Research_Log.docx    # running research log
└── explore.py                      # quick dataset exploration script
```

---

## Setup

```bash
cd causal-data-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then run notebooks in order: `01` → `02` → `03`.

For the standalone Python pipeline:

```bash
cd "FUHRLANDER GITHUB REPO"
jupyter notebook python_pipeline.ipynb
```

> Set `DATASET_PATH` in the config cell to the absolute path of the `dataset/` folder.

---

## Tech Stack

- Python, Jupyter, scikit-learn, NumPy, pandas, matplotlib
- ELM (Extreme Learning Machine) — pseudoinverse solver
- Random Forest (two-phase importance-based feature selection)
- LiNGAM (causal discovery)
- Original reference implementation: R + MATLAB

---

*Part of the TwinARC Digital Twin platform — E-Minds.*
