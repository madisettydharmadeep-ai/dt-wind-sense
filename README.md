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
Raw SCADA
  │  python_pipeline — feature finding (two-phase RF, no pre-filter) + cross-turbine ELM
  ▼
Selected features @ raw 5-min
  │  00_data_checks       → clean + segmented (data gaps handled)
  │  02_causal_prep       → glitch-clipped, detrended, standardised
  │  03_window_bucketing  → overlapping time windows
  ▼
LiNGAM causal discovery            (+ ELM anomaly detection on WT84)
```

1. **Feature finding** (`python_pipeline.ipynb`) — on **pure raw data**, no pre-filtering. Phase 1 RF (all sensors → fault alarm) picks the target sensor Y; Phase 2 RF (remaining → Y) picks the predictors X. A cross-turbine ELM (train WT80–83, test WT84) validates. Saves **only the selected columns at raw 5-min** resolution.
2. **Data checks** (`00_data_checks.ipynb`) — detects the 233-day data gap + short blank runs; interpolates the short gaps and splits at the large gap into continuous **segments** (no fabricated data).
3. **Causal prep** (`02_causal_prep.ipynb`) — per segment: robust **MAD glitch clipping** → drift removal → light winsorise → zero-mean / unit-variance standardisation.
4. **Window bucketing** (`03_window_bucketing.ipynb`) — slices each segment into overlapping windows to track how the causal structure evolves over time.
5. **Causal discovery** — LiNGAM on the windowed (or static weekly) outputs.
6. **Anomaly detection** — the cross-turbine ELM flags WT84 ~5 months before the failure.

---

## Key Result

The ELM model — trained **only** on WT80–83 with no knowledge of WT84 — flagged anomalous behaviour on WT84 approximately **5 months before** the recorded gearbox failure date (June 2014).

---

## Repo Structure

```
├── causal-data-engine/
│   ├── pipeline/                     # ACTIVE pipeline
│   │   ├── 00_data_checks.ipynb      # gap detection + segmentation
│   │   ├── 02_causal_prep.ipynb      # MAD glitch clip, detrend, standardise
│   │   └── 03_window_bucketing.ipynb # sliding windows for LiNGAM
│   ├── 01_data_prep.ipynb            # legacy (retired)
│   ├── 02_feature_selection.ipynb    # legacy (retired)
│   ├── 03_pre_causal.ipynb           # legacy (retired)
│   └── requirements.txt
│
├── FUHRLANDER GITHUB REPO/
│   ├── python_pipeline.ipynb         # feature finding on raw data + cross-turbine ELM + saves 5-min features
│   ├── matlab/                       # original MATLAB feature selection + ELM
│   ├── r/                            # original R data prep scripts
│   └── dataset/                      # raw .json.bz2 files (git-ignored)
│
├── Fuhrlander_Research_Log.docx      # running research log
└── explore.py                        # quick dataset exploration script
```

---

## Setup

```bash
cd causal-data-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Run order:**

1. `FUHRLANDER GITHUB REPO/python_pipeline.ipynb` — feature finding + ELM; saves the selected features at raw 5-min. Set `DATASET_PATH` in the config cell to the absolute path of the `dataset/` folder.
2. `causal-data-engine/pipeline/00_data_checks.ipynb` — clean + segment.
3. `causal-data-engine/pipeline/02_causal_prep.ipynb` — glitch-clip, detrend, standardise → causal-ready data.
4. `causal-data-engine/pipeline/03_window_bucketing.ipynb` — slice into windows for LiNGAM.

> Set the `INPUT_DIR` / `OUTPUT_DIR` paths in each notebook's config cell so each stage reads the previous stage's output.

---

## Tech Stack

- Python, Jupyter, scikit-learn, NumPy, pandas, matplotlib
- ELM (Extreme Learning Machine) — pseudoinverse solver
- Random Forest (two-phase importance-based feature selection, on raw data)
- Robust preprocessing — MAD glitch clipping, drift removal, z-score standardisation
- LiNGAM (causal discovery)
- Original reference implementation: R + MATLAB

---

*Part of the TwinARC Digital Twin platform — E-Minds.*
