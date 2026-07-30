"""
================================================================================
EEGOdyssey — Day 3 Study Guide
Tabular CSV to NumPy Array Conversion & Metadata Merging
================================================================================

This guide demonstrates how to convert a tabular CSV dataset (`eeg_mastersheet.csv`)
into multi-dimensional NumPy arrays and matrices for statistical analysis and machine learning.

Key Operations:
  1. Load Tabular Features CSV
  2. Convert CSV to 2D Machine Learning Matrix (Samples × Features: 8,640 × 6)
  3. Convert CSV to 5D Signal Tensor Array (Groups × Timepoints × Subjects × Channels × Features: 3 × 3 × 30 × 32 × 6)
  4. Preserve exact metadata index ordering using pd.Categorical

Run: python day_3/03_day3_study_guide.py
================================================================================
"""

import os
import pandas as pd
import numpy as np


import sys

def resolve_csv_path():
    """Resolve eeg_mastersheet.csv path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "data", "eeg_mastersheet.csv"),
        os.path.join(script_dir, "data", "eeg_mastersheet_enriched.csv"),
        "day_3/data/eeg_mastersheet.csv",
        "data/eeg_mastersheet.csv",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        sys.path.insert(0, os.path.join(script_dir, "generators"))
        import convert_to_mastersheet
        path, _ = convert_to_mastersheet.create_mastersheet()
    return path


def main():
    print("=" * 80)
    print(" DAY 3: TABULAR CSV TO NUMPY ARRAY CONVERSION & DELETION STRATEGIES")
    print("=" * 80)

    # 1. Load CSV Mastersheet & Perform Missing Data Audit
    csv_path = resolve_csv_path()
    print(f"\n[STEP 1] Loading CSV Mastersheet: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  • Total Rows count  : {len(df):,}")
    print(f"  • Total Columns     : {list(df.columns)}")

    # Missing Value Audit
    n_missing = df["power_value"].isnull().sum()
    pct_missing = (n_missing / len(df)) * 100
    print(f"  • Total NaNs found  : {n_missing:,} ({pct_missing:.2f}%)")

    # Extract unique dimension lists
    groups = list(df["group"].unique())
    timepoints = list(df["timepoint"].unique())
    subjects = list(df["subject_id"].unique())
    channels = list(df["channel"].unique())
    features = list(df["feature"].unique())

    # 2. Demonstrate Feature Matrix Deletion Strategies (N samples x P features)
    print("\n" + "-" * 80)
    print("[STEP 2] Feature Matrix Deletion Strategies (N samples × P features)")
    print("-" * 80)

    # Pivot to wide format (index = group, timepoint, subject_id, channel, region; columns = features)
    df_wide = df.pivot(
        index=["group", "timepoint", "subject_id", "channel", "region"],
        columns="feature",
        values="power_value",
    ).reset_index()

    feature_cols = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
    X_2d_raw = df_wide[feature_cols].to_numpy()
    N_samples, P_features = X_2d_raw.shape

    print(f"  • Feature Matrix Structure : X in R^({N_samples:,} x {P_features}) ({N_samples:,} samples × {P_features} features)")

    # Feature-wise Missingness Breakdown
    nan_pct_per_feature = np.isnan(X_2d_raw).mean(axis=0) * 100
    print("\n  [Feature Audit] Missingness Rate per Feature Column:")
    for feat_name, pct in zip(feature_cols, nan_pct_per_feature):
        print(f"    - Feature '{feat_name:10s}': {pct:.2f}% missing values")

    # Strategy A: Sample-wise (Listwise) Deletion
    # Drops sample row i if ANY feature x_ij is NaN: X[~np.isnan(X).any(axis=1)]
    valid_sample_mask = ~np.isnan(X_2d_raw).any(axis=1)
    X_listwise = X_2d_raw[valid_sample_mask]
    df_listwise = df_wide[valid_sample_mask]

    n_dropped_samples = N_samples - len(X_listwise)
    pct_dropped_samples = (n_dropped_samples / N_samples) * 100

    print(f"\n  [Strategy A] Sample-wise (Listwise) Deletion (Row Drop):")
    print(f"    - Formula           : X[~np.isnan(X).any(axis=1)]")
    print(f"    - Remaining Samples : {len(X_listwise):,} / {N_samples:,} rows")
    print(f"    - Dropped Samples   : {n_dropped_samples:,} ({pct_dropped_samples:.2f}% data loss)")
    print(f"    - Feature Matrix    : Retains ALL {P_features} features, but loses {n_dropped_samples:,} sample vectors.")
    print(f"    - Multi-Feature Cascade: While each individual feature is missing ~1.5–1.9%, compounding across {P_features} features drops {pct_dropped_samples:.2f}% of total samples!")

    # Strategy B: Feature-wise (Column) Deletion
    # Drops feature column j if missingness exceeds threshold (e.g. >5%)
    feature_keep_mask = nan_pct_per_feature < 5.0
    kept_features = [f for f, keep in zip(feature_cols, feature_keep_mask) if keep]
    dropped_features = [f for f, keep in zip(feature_cols, feature_keep_mask) if not keep]

    print(f"\n  [Strategy B] Feature-wise (Column) Deletion (Column Drop):")
    print(f"    - Formula           : X[:, nan_pct_per_feature < threshold]")
    print(f"    - Kept Features ({len(kept_features)}/{P_features}) : {kept_features}")
    print(f"    - Dropped Features  : {dropped_features if dropped_features else 'None (>5% threshold)'}")
    print(f"    - Feature Matrix    : Retains ALL {N_samples:,} sample rows, but reduces feature dimension P.")

    # Strategy C: Pairwise Feature Analysis
    # Computes feature covariance/correlation on valid pairs without row or column loss
    corr_pairwise = df_wide[feature_cols].corr(method="pearson")  # pd.corr uses pairwise deletion
    nan_means = np.nanmean(X_2d_raw, axis=0)

    print(f"\n  [Strategy C] Pairwise Feature Analysis (Available Pair Analysis):")
    print(f"    - Feature Means (computed via np.nanmean across available values per feature column):")
    for band, m_val in zip(feature_cols, nan_means):
        print(f"        • {band:10s}: {m_val:.4f}")
    print(f"    - Correlation Matrix computed across non-null pairs without discarding entire sample vectors.")

    # Strategy D: Subject / Visit-wise Deletion
    visit_missing = df.groupby(["group", "subject_id", "timepoint"])["power_value"].apply(lambda x: x.isnull().mean() * 100)
    unrecorded_visits = visit_missing[visit_missing > 50.0].index.tolist()
    print(f"\n  [Strategy D] Subject / Visit-wise Deletion:")
    print(f"    - Visits with >50% missing data (Unrecorded Visits): {unrecorded_visits}")

    # 3. Convert CSV to 2D Feature Matrix (Samples x Features)
    print("\n" + "-" * 80)
    print("[STEP 3] Reshaping CSV into 2D Feature Matrix (Samples × Features)")
    print("-" * 80)

    X_2d_raw = df_wide[feature_cols].to_numpy()
    metadata_array = df_wide[["group", "timepoint", "subject_id", "channel", "region"]].to_numpy()
    X_2d_complete = df_listwise[feature_cols].to_numpy()

    print(f"  • Raw 2D Feature Matrix (X_raw)      : shape = {X_2d_raw.shape} (contains {np.isnan(X_2d_raw).sum()} NaNs)")
    print(f"  • Complete-Case 2D Matrix (X_clean): shape = {X_2d_complete.shape} (0 NaNs)")

    # 4. Convert CSV to 5D Tensor Array (G x T x S x C x F)
    print("\n" + "-" * 80)
    print("[STEP 4] Reshaping CSV into 5D Signal Tensor (G × T × S × C × F)")
    print("-" * 80)

    # Enforce categorical ordering matching metadata dimension lists
    df_ordered = df.copy()
    df_ordered["group"] = pd.Categorical(df_ordered["group"], categories=groups, ordered=True)
    df_ordered["timepoint"] = pd.Categorical(df_ordered["timepoint"], categories=timepoints, ordered=True)
    df_ordered["subject_id"] = pd.Categorical(df_ordered["subject_id"], categories=subjects, ordered=True)
    df_ordered["channel"] = pd.Categorical(df_ordered["channel"], categories=channels, ordered=True)
    df_ordered["feature"] = pd.Categorical(df_ordered["feature"], categories=features, ordered=True)

    df_sorted = df_ordered.sort_values(
        by=["group", "timepoint", "subject_id", "channel", "feature"]
    )

    G, T, S, C, F = len(groups), len(timepoints), len(subjects), len(channels), len(features)
    arr_5d = df_sorted["power_value"].to_numpy().reshape(G, T, S, C, F)

    print(f"  • 5D NumPy Tensor Shape: {arr_5d.shape} (3 Groups × 3 Timepoints × 30 Subjects × 32 Channels × 6 Features)")
    print(f"  • Total NaNs in 5D Tensor: {np.isnan(arr_5d).sum():,} ({np.isnan(arr_5d).mean()*100:.2f}%)")

    # 5. Verify Indexing & NaN Preservation Alignment
    print("\n" + "-" * 80)
    print("[STEP 5] Verifying Index Alignment & NaN Preservation")
    print("-" * 80)
    g_idx, t_idx, s_idx, c_idx, f_idx = 1, 0, 14, 9, 2
    arr_val = arr_5d[g_idx, t_idx, s_idx, c_idx, f_idx]
    csv_val = df[
        (df["group"] == groups[g_idx])
        & (df["timepoint"] == timepoints[t_idx])
        & (df["subject_id"] == subjects[s_idx])
        & (df["channel"] == channels[c_idx])
        & (df["feature"] == features[f_idx])
    ]["power_value"].values[0]

    print(f"  Query: Patient · Baseline · Sub_15 · Cz · alpha")
    print(f"  • 5D Array Value [1, 0, 14, 9, 2]: {arr_val:.6f}")
    print(f"  • CSV Value                      : {csv_val:.6f}")
    assert np.isclose(arr_val, csv_val, equal_nan=True), "Value mismatch!"
    print("  [OK] Verification SUCCESS: Array and CSV values match perfectly!")

    print("\n" + "=" * 80)
    print(" TABULAR CSV TO NUMPY ARRAY CONVERSION & DELETION TUTORIAL COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
