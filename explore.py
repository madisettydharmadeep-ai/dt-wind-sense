"""
explore.py  —  Python port of the Fuhrländer SCADA repo workflow
Mirrors: r/examples.R  +  matlab/example_elm.m  +  matlab/elm_classifier.m

Run from the repo root:
    python explore.py

Produces:
    1. Console printout of dataset structure (what the R/MATLAB loaders give you)
    2. ELM normality model trained on WT84 first-25% (reproduces paper Fig 6)
    3. Saved plot: elm_result_wt84.png
"""

import bz2, json, os, sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless; switch to 'TkAgg' if you want a live window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

DATASET = os.path.join(os.path.dirname(__file__), "FUHRLANDER GITHUB REPO", "dataset")
PLANT_JSON = os.path.join(DATASET, "wind_plant_data.json")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 – helpers  (Python port of turbine_data.R / turbine_data.m)
# ─────────────────────────────────────────────────────────────────────────────

def load_turbine_json(turbine_id):
    """Read bz2-compressed JSON for one turbine, return raw dict."""
    path = os.path.join(DATASET, f"turbine_{turbine_id}.json.bz2")
    print(f"  Loading {path} ...", end=" ", flush=True)
    with open(path, "rb") as f:
        raw = json.loads(bz2.decompress(f.read()).decode("utf-8"))
    print("done")
    return raw


def filtered_3sdv_mean(series: pd.Series) -> float:
    """Remove values outside mean±3σ, return mean of remainder. Mirrors filtered_3sdv_mean() in R."""
    u, s = series.mean(), series.std()
    kept = series[(series >= u - 3*s) & (series <= u + 3*s)]
    return kept.mean() if len(kept) else np.nan


def get_turbine_data(turbine_id, alarm_id_list=None,
                     frequency_seconds=300, combine_func="mean",
                     one_hot_encoding=False, verbose=True):
    """
    Python port of get_turbine_data() from turbine_data.R / turbine_data.m

    Returns a DataFrame with:
      - turbine_id, date_time columns first
      - all analog sensor columns
      - optionally: one binary column per alarm_id  (if alarm_id_list given)
    """
    raw = load_turbine_json(turbine_id)
    ad  = raw["analog_data"]
    native_freq = raw["analog_data_frequency_seconds"]   # always 300 (5 min)

    # ── Build DataFrame from the analog_data dict ──────────────────────────
    if verbose:
        print("  Building DataFrame ...", end=" ", flush=True)
    df = pd.DataFrame({k: v for k, v in ad.items() if k != "date_time"})
    df.insert(1, "date_time", pd.to_datetime(ad["date_time"]))
    df = df.sort_values("date_time").reset_index(drop=True)
    if verbose:
        print(f"done  ({len(df):,} rows × {len(df.columns)} cols)")

    # ── Optionally resample to coarser frequency ────────────────────────────
    if frequency_seconds > native_freq:
        if verbose:
            print(f"  Resampling {native_freq}s → {frequency_seconds}s using '{combine_func}' ...",
                  end=" ", flush=True)
        freq_str = f"{frequency_seconds}s"
        df = df.set_index("date_time")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "turbine_id"]

        if combine_func == "mean":
            resampled = df[numeric_cols].resample(freq_str).mean()
        elif combine_func == "median":
            resampled = df[numeric_cols].resample(freq_str).median()
        elif combine_func == "max":
            resampled = df[numeric_cols].resample(freq_str).max()
        elif combine_func == "min":
            resampled = df[numeric_cols].resample(freq_str).min()
        elif combine_func in ("filtered_3sdv_mean", "filtered_3sdv_median"):
            # Vectorised 3-sigma filter — much faster than per-cell apply()
            # Same logic as filtered_3sdv_mean() in R: remove >mean±3σ per block, then aggregate
            agg_fn = "mean" if combine_func.endswith("mean") else "median"
            data_only = df[numeric_cols]
            grp = data_only.resample(freq_str)
            mu    = grp.transform("mean")
            sigma = grp.transform("std").fillna(0)
            clipped = data_only.copy()
            clipped[(data_only < mu - 3*sigma) | (data_only > mu + 3*sigma)] = np.nan
            if agg_fn == "mean":
                resampled = clipped.resample(freq_str).mean()
            else:
                resampled = clipped.resample(freq_str).median()
        else:
            resampled = df[numeric_cols].resample(freq_str).mean()

        resampled = resampled.reset_index().rename(columns={"date_time": "date_time"})
        resampled.insert(0, "turbine_id", turbine_id)
        df = resampled
        if verbose:
            print(f"done  ({len(df):,} rows)")
    else:
        df = df.reset_index() if "date_time" not in df.columns else df

    # ── Optionally merge alarm flags ────────────────────────────────────────
    if alarm_id_list is not None and len(alarm_id_list) > 0:
        alarms = raw["alarms"]
        alarm_df = pd.DataFrame(alarms)
        alarm_df["date_time_ini"] = pd.to_datetime(alarm_df["date_time_ini"])
        alarm_df["date_time_end"] = pd.to_datetime(alarm_df["date_time_end"])
        alarm_df = alarm_df[alarm_df["alarm_id"].isin(alarm_id_list)]

        if verbose:
            print(f"  Merging {len(alarm_df)} alarm events ...", end=" ", flush=True)

        if one_hot_encoding:
            for aid in sorted(alarm_df["alarm_id"].unique()):
                col = f"alarm_{aid}"
                df[col] = 0
                rows = alarm_df[alarm_df["alarm_id"] == aid]
                for _, row in rows.iterrows():
                    mask = (df["date_time"] >= row["date_time_ini"]) & \
                           (df["date_time"] <= row["date_time_end"])
                    df.loc[mask, col] = 1
        else:
            df["alarms_active"] = np.nan
            for _, row in alarm_df.iterrows():
                mask = (df["date_time"] >= row["date_time_ini"]) & \
                       (df["date_time"] <= row["date_time_end"])
                df.loc[mask, "alarms_active"] = row["alarm_id"]

        if verbose:
            print("done")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Explore the plant metadata  (mirrors plant_data = fromJSON(...) in R)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 1 — Plant metadata (wind_plant_data.json)")
print("="*70)

with open(PLANT_JSON) as f:
    plant_data = json.load(f)

turbines_meta = plant_data["turbines"]
alarm_dict    = plant_data["alarm_dictionary"]

print(f"\nTurbines in this wind farm:")
for i, tid in enumerate(turbines_meta["turbine_id"]):
    print(f"  WT{tid}  →  {turbines_meta['compressed_filename'][i]}")

# Alarm breakdown by system
alarm_df_meta = pd.DataFrame(alarm_dict)
print(f"\nAlarm dictionary: {len(alarm_df_meta)} unique alarm types")
print("\nAlarm types by system:")
print(alarm_df_meta.groupby("alarm_system").size().sort_values(ascending=False).to_string())

# Gearbox (Transmission) alarm IDs — mirrors the R/MATLAB examples exactly
gearbox_alarm_ids = alarm_df_meta.loc[
    alarm_df_meta["alarm_system"] == "Transmission", "alarm_id"
].tolist()
print(f"\nGearbox (Transmission) alarm IDs: {len(gearbox_alarm_ids)} types")
print("  First 10:", gearbox_alarm_ids[:10])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Load WT80 data with gearbox alarms (mirrors r/examples.R exactly)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 2 — Load WT80 data (mirrors r/examples.R)")
print("="*70)
print("  get_turbine_data(turbine_80, alarm_id_list=gearbox_alarms,")
print("                   frequency_seconds=3600, combine_func='filtered_3sdv_mean',")
print("                   one_hot_encoding=True)")
print()

wt80 = get_turbine_data(
    turbine_id=80,
    alarm_id_list=gearbox_alarm_ids,
    frequency_seconds=3600,
    combine_func="filtered_3sdv_mean",
    one_hot_encoding=True,
    verbose=True
)

print(f"\nResult shape: {wt80.shape[0]:,} rows × {wt80.shape[1]} columns")
print(f"Date range:  {wt80['date_time'].min()}  →  {wt80['date_time'].max()}")

sensor_cols  = [c for c in wt80.columns if not c.startswith("alarm_") and c not in ("turbine_id","date_time")]
alarm_cols   = [c for c in wt80.columns if c.startswith("alarm_")]
print(f"\nSensor columns: {len(sensor_cols)}")
print(f"Alarm columns:  {len(alarm_cols)}  (one-hot, gearbox alarms only)")

# How often are gearbox alarms active?
if alarm_cols:
    any_alarm = (wt80[alarm_cols].sum(axis=1) > 0)
    print(f"\nTimesteps with ≥1 gearbox alarm active: {any_alarm.sum():,} / {len(wt80):,}  "
          f"({100*any_alarm.mean():.1f}%)")

print("\nSample of the combined table (first 3 rows, select cols):")
show = ["date_time", "wgdc_avg_TriGri_PwrAt", "wtrm_avg_TrmTmp_GbxBrg152",
        "wtrm_avg_TrmTmp_GbxOil", "wnac_avg_WSpd1"] + alarm_cols[:3]
show = [c for c in show if c in wt80.columns]
print(wt80[show].head(3).to_string(index=False))

print("\nBasic statistics for key gearbox sensors:")
key_sensors = [
    "wgdc_avg_TriGri_PwrAt",
    "wtrm_avg_TrmTmp_GbxBrg152",
    "wtrm_avg_TrmTmp_GbxOil",
    "wgen_avg_RtrSpd_WP2035",
    "wnac_avg_WSpd1",
    "wnac_avg_ExlTmp",
]
key_sensors = [c for c in key_sensors if c in wt80.columns]
print(wt80[key_sensors].describe().round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Feature selection via Random Forest  (mirrors feature_selection.m)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 3 — Feature selection (mirrors matlab/feature_selection.m)")
print("="*70)
print("  Using RandomForest (sklearn) — same concept as MATLAB fitrensemble")

sensor_cols_fs = [c for c in sensor_cols if c not in ("turbine_id",)]
alarm_cols_fs  = alarm_cols

# Drop rows with any NaN in the sensor cols
wt80_clean = wt80[sensor_cols_fs + alarm_cols_fs].dropna()
print(f"  Clean rows for RF: {len(wt80_clean):,}")

if alarm_cols_fs and len(wt80_clean) > 100:
    alarm_active = (wt80_clean[alarm_cols_fs].sum(axis=1) > 0).astype(int)
    X_fs = wt80_clean[sensor_cols_fs].values

    print("\n  Phase 1: which sensor best correlates with gearbox alarms?")
    rf1 = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=1)
    rf1.fit(X_fs, alarm_active)
    importances1 = pd.Series(rf1.feature_importances_, index=sensor_cols_fs).sort_values(ascending=False)
    print("  Top 8 sensors most predictive of gearbox alarm state:")
    print(importances1.head(8).round(4).to_string())

    best_target = importances1.index[0]
    print(f"\n  → Best response variable: '{best_target}'")

    print(f"\n  Phase 2: which sensors best predict '{best_target}'?")
    input_cols_fs = [c for c in sensor_cols_fs if c != best_target]
    X_fs2 = wt80_clean[input_cols_fs].fillna(0).values
    y_fs2 = wt80_clean[best_target].values
    rf2 = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=1)
    rf2.fit(X_fs2, y_fs2)
    importances2 = pd.Series(rf2.feature_importances_, index=input_cols_fs).sort_values(ascending=False)
    print("  Top 10 input sensors for predicting the response:")
    print(importances2.head(10).round(4).to_string())
    selected_inputs_rf = importances2.head(8).index.tolist()
    print(f"\n  → RF-selected inputs: {selected_inputs_rf}")
else:
    print("  (skipped — no alarm data or too few rows)")
    selected_inputs_rf = []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – ELM: reproduce paper Figure 6 on WT84
#           (mirrors elm_classifier.m  +  paper's illustrative example)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 4 — ELM normality model on WT84 (reproduces paper Fig 6)")
print("="*70)

# Variables from Table 5 in the paper (manually chosen, as the paper does)
X_VARS = [
    "wgdc_avg_TriGri_PwrAt",        # Active power
    "wgdc_avg_TriGri_PF",           # Power factor
    "wtrm_avg_TrmTmp_GbxOil",       # Gearbox oil temp
    "wgen_avg_RtrSpd_WP2035",       # Rotor speed (before gearbox)
    "wnac_avg_WSpd1",               # Wind speed
    "wnac_avg_ExlTmp",              # External temperature
    "wtrm_avg_TrmTmp_GbxBrg151",    # Gearbox bearing 151 temp
]
Y_VAR = "wtrm_avg_TrmTmp_GbxBrg152"  # Target: bearing 152 (high-speed shaft output)

print(f"\nInput variables (X): {X_VARS}")
print(f"Target variable  (Y): {Y_VAR}")
print(f"\nApproach: train on first 25% of WT84 (healthy), test on remaining 75%")
print(f"          If test predictions deviate from actual → early failure warning")

print("\nLoading WT84 (raw 5-min, no resampling for this example) ...")
wt84_raw = load_turbine_json(84)
ad84 = wt84_raw["analog_data"]

df84 = pd.DataFrame({k: v for k, v in ad84.items() if k != "date_time"})
df84.insert(1, "date_time", pd.to_datetime(ad84["date_time"]))
df84 = df84.sort_values("date_time").reset_index(drop=True)
print(f"  {len(df84):,} rows, {df84['date_time'].iloc[0]} → {df84['date_time'].iloc[-1]}")

# Keep only relevant columns, drop NaN
all_cols = [Y_VAR] + X_VARS
missing = [c for c in all_cols if c not in df84.columns]
if missing:
    print(f"  WARNING: missing columns: {missing}")
    all_cols = [c for c in all_cols if c in df84.columns]
    X_VARS   = [c for c in X_VARS if c in df84.columns]

df84_sel = df84[["date_time"] + all_cols].copy()
df84_sel = df84_sel.dropna()
print(f"  After dropping NaN: {len(df84_sel):,} rows")

# ── Train / test split (first 25% = training, mirrors paper exactly) ─────────
split_idx = int(len(df84_sel) * 0.25)
train = df84_sel.iloc[:split_idx].copy()
test  = df84_sel.iloc[split_idx:].copy()
print(f"\n  Train: {len(train):,} rows  ({train['date_time'].iloc[0].date()} → {train['date_time'].iloc[-1].date()})")
print(f"  Test:  {len(test):,} rows   ({test['date_time'].iloc[0].date()} → {test['date_time'].iloc[-1].date()})")

X_train = train[X_VARS].values
y_train = train[Y_VAR].values
X_test  = test[X_VARS].values
y_test  = test[Y_VAR].values


def elm_train(X, y, H=50, seed=1):
    """
    Python port of elm_classifier.m → elm()

    ELM: random input weights W (d×H), bias b (H,)
         sigmoid hidden layer
         pseudoinverse output weights B = pinv(H_mat) @ y_norm
    Returns dict with all model params needed for prediction.
    """
    rng = np.random.default_rng(seed)
    # Z-score normalize X
    X_mu    = X.mean(axis=0)
    X_sigma = X.std(axis=0) + 1e-8
    Xn = (X - X_mu) / X_sigma

    # Z-score normalize y
    y_mu    = y.mean()
    y_sigma = y.std() + 1e-8
    yn = (y - y_mu) / y_sigma

    # Random weights and biases
    d = Xn.shape[1]
    W = rng.standard_normal((d, H))   # shape (d, H)
    b = rng.standard_normal(H)        # shape (H,)

    # Hidden layer (sigmoid activation)
    Hmat = 1.0 / (1.0 + np.exp(-(Xn @ W + b)))  # (N, H)

    # Output weights via pseudoinverse  — mirrors: B = pinv(H) * y_norm
    B = np.linalg.pinv(Hmat) @ yn                # (H,)

    # Training predictions (for RMSE)
    yhat_norm = Hmat @ B
    yhat      = yhat_norm * y_sigma + y_mu
    rmse_train = np.sqrt(np.mean((y - yhat)**2))

    return dict(W=W, b=b, B=B, X_mu=X_mu, X_sigma=X_sigma,
                y_mu=y_mu, y_sigma=y_sigma, rmse_train=rmse_train)


def elm_predict(X, model):
    """Port of elm_classifier.m → elm_predict()"""
    Xn   = (X - model["X_mu"]) / model["X_sigma"]
    Hmat = 1.0 / (1.0 + np.exp(-(Xn @ model["W"] + model["b"])))
    yhat_norm = Hmat @ model["B"]
    yhat      = yhat_norm * model["y_sigma"] + model["y_mu"]
    return yhat


print(f"\nTraining ELM (H=50 hidden nodes, sigmoid, pseudoinverse output weights) ...")
model = elm_train(X_train, y_train, H=50)
print(f"  Training RMSE: {model['rmse_train']:.3f} °C")

print("Predicting on test set ...")
yhat_test = elm_predict(X_test, model)
test_rmse = np.sqrt(np.mean((y_test - yhat_test)**2))
print(f"  Test RMSE (full 75%): {test_rmse:.3f} °C")

# ── Find the failure region ────────────────────────────────────────────────
# Known from earlier analysis: failure starts at WT84 sample 158112 (absolute)
# Map to our filtered/split dataframe
FAIL_DATE = pd.Timestamp("2014-06-06 14:10:00")
fail_mask = test["date_time"] >= FAIL_DATE
fail_idx_in_test = test.index[fail_mask][0] - test.index[0] if fail_mask.any() else len(test)
print(f"\n  Known failure date: {FAIL_DATE}")
print(f"  Failure falls at test sample index: {fail_idx_in_test:,} / {len(test):,}")

# Residuals (model error)
residuals = np.abs(y_test - yhat_test)
pre_fail_rmse  = np.sqrt(np.mean((y_test[:fail_idx_in_test] - yhat_test[:fail_idx_in_test])**2))
post_fail_rmse = np.sqrt(np.mean((y_test[fail_idx_in_test:] - yhat_test[fail_idx_in_test:])**2))
print(f"\n  RMSE before failure: {pre_fail_rmse:.3f} °C  ← model tracks well")
print(f"  RMSE after  failure: {post_fail_rmse:.3f} °C  ← model diverges")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Plot  (reproduces paper Figures 4, 5, 6, 7)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 5 — Plotting results")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.patch.set_facecolor("#0D1219")
for ax in axes:
    ax.set_facecolor("#111820")
    ax.tick_params(colors="#5A7090", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#182030")

T_test = test["date_time"].values

# ── Panel A: full test set (like Fig 4 in paper — model tracks normally) ─────
ax = axes[0]
n_show = min(fail_idx_in_test, 15000)  # first chunk where model works
ax.plot(T_test[:n_show], y_test[:n_show],    color="#4A90D9", lw=0.8, alpha=0.85, label="Measured (actual)")
ax.plot(T_test[:n_show], yhat_test[:n_show], color="#F5A623", lw=0.9, alpha=0.9, label="ELM prediction")
ax.set_ylabel("GbxBrg152 (°C)", color="#B8CDE0", fontsize=9)
ax.set_title("Normal operation — ELM tracks the signal well  (paper Fig 4)", color="#B8CDE0", fontsize=10, pad=8)
ax.legend(fontsize=8, facecolor="#0D1219", edgecolor="#182030", labelcolor="#B8CDE0")
ax.grid(True, color="#182030", linewidth=0.4)

# ── Panel B: failure window — model diverges (Fig 6 in paper) ────────────────
ax = axes[1]
# Show ±3000 samples around failure
lo = max(0, fail_idx_in_test - 3000)
hi = min(len(T_test), fail_idx_in_test + 7000)
T_win = T_test[lo:hi]
y_win = y_test[lo:hi]
yh_win = yhat_test[lo:hi]

# Colour by phase: blue = pre-failure working, green = post-failure
pre_end  = fail_idx_in_test - lo
ax.plot(T_win[:pre_end],  y_win[:pre_end],  color="#4A90D9", lw=1.0, label="Measured — normal (blue in paper)")
ax.plot(T_win[pre_end:],  y_win[pre_end:],  color="#3DB88A", lw=1.0, label="Measured — post-failure (green in paper)")
ax.plot(T_win,            yh_win,           color="#F5A623", lw=1.0, ls="--", alpha=0.9, label="ELM prediction")
ax.axvline(T_test[fail_idx_in_test], color="#DC3C3C", lw=1.5, label=f"Failure: {FAIL_DATE.strftime('%d %b %Y')}")
ax.set_ylabel("GbxBrg152 (°C)", color="#B8CDE0", fontsize=9)
ax.set_title("Failure window — ELM diverges as gearbox fails  (paper Fig 6)", color="#B8CDE0", fontsize=10, pad=8)
ax.legend(fontsize=8, facecolor="#0D1219", edgecolor="#182030", labelcolor="#B8CDE0")
ax.grid(True, color="#182030", linewidth=0.4)

# ── Panel C: residual (absolute error) over time ──────────────────────────────
ax = axes[2]
# Smooth residuals with rolling window
resid_s = pd.Series(residuals)
rolling = resid_s.rolling(window=200, min_periods=1, center=True).mean()
ax.fill_between(range(len(residuals)), residuals,
                color="#4A90D9", alpha=0.25, label="|actual − predicted|")
ax.plot(rolling, color="#F5A623", lw=1.2, label="200-sample rolling mean")
ax.axvline(fail_idx_in_test, color="#DC3C3C", lw=1.5, label="Failure")
ax.set_xlabel("Test sample index  (each = 5 minutes)", color="#B8CDE0", fontsize=9)
ax.set_ylabel("Absolute error (°C)", color="#B8CDE0", fontsize=9)
ax.set_title("Prediction residual — divergence grows before & after failure", color="#B8CDE0", fontsize=10, pad=8)
ax.legend(fontsize=8, facecolor="#0D1219", edgecolor="#182030", labelcolor="#B8CDE0")
ax.grid(True, color="#182030", linewidth=0.4)

fig.suptitle("WT84 — ELM Normality Model  |  Fuhrländer FL2500 SCADA",
             color="#B8CDE0", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "elm_result_wt84.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1219")
print(f"\n  Plot saved → {out_path}")
print("\nDone. Open elm_result_wt84.png to see the results.")
print("="*70)
