"""
EEG Odyssey — Day 3: Convert 5D Neural Data Array to CSV Mastersheet

This script loads the 5-D NumPy array (`neural_data.npy`) and corresponding
metadata (`metadata.pkl`) from Day 1 data directory, maps all dimension indices
to their categorical variable names, and exports a flat CSV mastersheet.

Dimensions of input array: (G=3, T=3, S=30, C=32, F=6)
  Axis 0 (G): Groups (Control, Patient, Treatment)
  Axis 1 (T): Timepoints (baseline, 90 days, 180 days)
  Axis 2 (S): Subjects (Sub_01 .. Sub_30)
  Axis 3 (C): Channels (Fp1, Fp2, ..., PO4) & Region mapping
  Axis 4 (F): Features / Frequency bands (delta, theta, alpha, beta, gamma, broadband)
"""

import os
import pickle
import numpy as np
import pandas as pd


def resolve_input_paths():
    """Find neural_data.npy and metadata.pkl relative to script directory or CWD."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))

    data_candidates = [
        os.path.join(script_dir, "..", "..", "day_1", "data", "neural_data.npy"),
        os.path.join(script_dir, "..", "data", "neural_data.npy"),
        os.path.join(workspace_root, "day_1", "data", "neural_data.npy"),
        "day_1/data/neural_data.npy",
        "data/neural_data.npy",
    ]

    meta_candidates = [
        os.path.join(script_dir, "..", "..", "day_1", "data", "metadata.pkl"),
        os.path.join(script_dir, "..", "data", "metadata.pkl"),
        os.path.join(workspace_root, "day_1", "data", "metadata.pkl"),
        "day_1/data/metadata.pkl",
        "data/metadata.pkl",
    ]

    data_path = next((p for p in data_candidates if os.path.exists(p)), None)
    meta_path = next((p for p in meta_candidates if os.path.exists(p)), None)

    if not data_path or not meta_path:
        raise FileNotFoundError(
            f"Could not locate input files. Data path: {data_path}, Meta path: {meta_path}"
        )

    return data_path, meta_path


def create_mastersheet():
    data_path, meta_path = resolve_input_paths()

    print(f"Loading neural dataset from : {data_path}")
    print(f"Loading metadata pickle from: {meta_path}")

    # Load 5D numpy array and metadata dictionary
    data = np.load(data_path)
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)

    G, T, S, C, F = data.shape
    print(f"Dataset shape: ({G=}, {T=}, {S=}, {C=}, {F=}) -> Total {data.size:,} observations")

    # 1. Categorical variable lookups from metadata
    group_names = meta.get("group_names", ["Control", "Patient", "Treatment"])
    feature_names = meta.get(
        "feature_names", ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
    )
    channel_names = meta.get("channel_names", [f"Ch_{c}" for c in range(C)])
    regions_dict = meta.get("regions", {})

    # Timepoint names mapping (0 -> Baseline, 1 -> Task, 2 -> Rest)
    timepoint_names = ["Baseline", "Task", "Rest"]

    # Subject IDs (Sub_01 to Sub_30)
    subject_ids = [f"Sub_{s+1:02d}" for s in range(S)]

    # 2. Build channel index to cortical region mapping dictionary
    channel_to_region = {}
    for region_name, channel_indices in regions_dict.items():
        for idx in channel_indices:
            if idx < len(channel_names):
                channel_to_region[channel_names[idx]] = region_name

    # 3. Construct MultiIndex matching C-contiguous raveled array ordering
    multi_index = pd.MultiIndex.from_product(
        [group_names, timepoint_names, subject_ids, channel_names, feature_names],
        names=["group", "timepoint", "subject_id", "channel", "feature"],
    )

    # 4. Flatten the 5D numpy array into a DataFrame
    df = pd.DataFrame({"power_value": data.ravel()}, index=multi_index).reset_index()

    # 5. Add numerical index columns & cortical region column for rich metadata
    df["region"] = df["channel"].map(channel_to_region)

    # Reorder columns for optimal readability
    df = df[
        [
            "group",
            "timepoint",
            "subject_id",
            "channel",
            "region",
            "feature",
            "power_value",
        ]
    ]

    # 6. Inject realistic missing values (NaNs) for pedagogical & real-world training
    # Mechanism 1: Sensor/Channel Dropouts (loose electrode/high impedance during visit)
    mask_channel_dropout1 = (
        (df["group"] == "Patient")
        & (df["subject_id"] == "Sub_07")
        & (df["timepoint"] == "Task")
        & (df["channel"] == "O1")
    )
    mask_channel_dropout2 = (
        (df["group"] == "Treatment")
        & (df["subject_id"] == "Sub_12")
        & (df["timepoint"] == "Baseline")
        & (df["channel"] == "T7")
    )

    # Mechanism 2: Unrecorded Visit (participant missed visit follow-up)
    mask_unrecorded_visit = (
        (df["group"] == "Patient")
        & (df["subject_id"] == "Sub_28")
        & (df["timepoint"] == "Rest")
    )

    # Mechanism 3: Random Noise / Artifact Epoch Rejection (~1.5% random cells)
    rng = np.random.default_rng(2026)
    random_nan_indices = rng.choice(len(df), size=int(len(df) * 0.015), replace=False)

    # Apply missingness to power_value column
    df.loc[mask_channel_dropout1 | mask_channel_dropout2 | mask_unrecorded_visit, "power_value"] = np.nan
    df.loc[random_nan_indices, "power_value"] = np.nan

    # 7. Save output CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "eeg_mastersheet.csv")
    df.to_csv(output_csv, index=False)

    n_missing = df["power_value"].isnull().sum()
    pct_missing = (n_missing / len(df)) * 100

    print(f"\nSuccessfully generated mastersheet with {len(df):,} rows!")
    print(f"  • Total Missing Values (NaNs) : {n_missing:,} ({pct_missing:.2f}%)")
    print(f"  • Channel Dropouts Simulated : O1 (Sub_07/Patient/Task), T7 (Sub_12/Treatment/Baseline)")
    print(f"  • Unrecorded Visit Simulated : Sub_28/Patient/Rest")
    print(f"  • Artifact Rejections        : {len(random_nan_indices):,} random observations")
    print(f"Saved to: {output_csv}")
    print("\nPreview of first 10 rows:")
    print(df.head(10).to_string(index=False))

    return output_csv, df


if __name__ == "__main__":
    create_mastersheet()
