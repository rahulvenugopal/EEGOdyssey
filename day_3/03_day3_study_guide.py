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
        import convert_to_mastersheet
        path, _ = convert_to_mastersheet.create_mastersheet()
    return path


def main():
    print("=" * 80)
    print(" DAY 3: TABULAR CSV TO NUMPY ARRAY CONVERSION")
    print("=" * 80)

    # 1. Load CSV Mastersheet
    csv_path = resolve_csv_path()
    print(f"\n[STEP 1] Loading CSV Mastersheet: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  • Rows count : {len(df):,}")
    print(f"  • Columns    : {list(df.columns)}")

    # Extract unique dimension lists
    groups = list(df["group"].unique())
    timepoints = list(df["timepoint"].unique())
    subjects = list(df["subject_id"].unique())
    channels = list(df["channel"].unique())
    features = list(df["feature"].unique())

    # 2. Convert CSV to 2D Matrix (Samples x Features)
    print("\n" + "-" * 80)
    print("[STEP 2] Converting CSV to 2D Feature Matrix (Samples × Features)")
    print("-" * 80)

    df_wide = df.pivot(
        index=["group", "timepoint", "subject_id", "channel", "region"],
        columns="feature",
        values="power_value",
    ).reset_index()

    feature_cols = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
    X_2d = df_wide[feature_cols].to_numpy()
    metadata_array = df_wide[["group", "timepoint", "subject_id", "channel", "region"]].to_numpy()

    print(f"  • Wide DataFrame Shape : {df_wide.shape}")
    print(f"  • 2D Feature Matrix (X): shape = {X_2d.shape}  (8,640 samples × 6 features)")
    print(f"  • Sample Metadata Array: shape = {metadata_array.shape}")

    # 3. Convert CSV to 5D Tensor Array (G x T x S x C x F)
    print("\n" + "-" * 80)
    print("[STEP 3] Converting CSV to 5D Tensor Array (G × T × S × C × F)")
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

    # 4. Verify indexing alignment
    print("\n" + "-" * 80)
    print("[STEP 4] Verifying Index Alignment")
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
    assert np.isclose(arr_val, csv_val), "Value mismatch!"
    print("  [OK] Verification SUCCESS: Array and CSV values match perfectly!")

    print("\n" + "=" * 80)
    print(" TABULAR CSV TO NUMPY ARRAY CONVERSION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
