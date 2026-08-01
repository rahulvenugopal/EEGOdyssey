"""
================================================================================
EEGOdyssey — Day 3: Tabular CSV to 5D NumPy Array Conversion
================================================================================

This script converts a flat tabular CSV dataset (`eeg_mastersheet.csv`) loaded directly 
from `day_1/data` into a 5-dimensional NumPy signal array/tensor using 
explicit nested loops.

Shape: (3, 3, 30, 32, 6) = 51,840 total observations

Run: python day_3/01_csv_to_array.py
================================================================================
"""

import os
import pandas as pd
import numpy as np

# 1. Load CSV directly from day_1/data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "day_1", "data", "eeg_mastersheet.csv")
if not os.path.exists(csv_path):
    csv_path = "day_1/data/eeg_mastersheet.csv"

print(f"Loading CSV dataset: {csv_path}")
df = pd.read_csv(csv_path)

# 2. Extract dimension ordering for axes (Groups x Timepoints x Subjects x Channels x Features)
groups = list(df["group"].unique())
timepoints = list(df["timepoint"].unique())
subjects = list(df["subject_id"].unique())
channels = list(df["channel"].unique())
features = list(df["feature"].unique())

G, T, S, C, F = len(groups), len(timepoints), len(subjects), len(channels), len(features)

# 3. Allocate empty 5D NumPy array initialized with NaNs
arr_5d = np.full((G, T, S, C, F), fill_value=np.nan, dtype=np.float32)

# 4. Build coordinate lookup dictionary from CSV rows
value_lookup = {}
for row in df.itertuples(index=False):
    key = (row.group, row.timepoint, row.subject_id, row.channel, row.feature)
    value_lookup[key] = row.power_value

# 5. Populate 5D array via explicit 5-level nested loop
for g_idx, group_name in enumerate(groups):
    for t_idx, timepoint_name in enumerate(timepoints):
        for s_idx, subject_name in enumerate(subjects):
            for c_idx, channel_name in enumerate(channels):
                for f_idx, feature_name in enumerate(features):
                    cell_key = (group_name, timepoint_name, subject_name, channel_name, feature_name)
                    arr_5d[g_idx, t_idx, s_idx, c_idx, f_idx] = value_lookup.get(cell_key, np.nan)

# 6. Spot check verification
sample_arr_val = arr_5d[0, 0, 0, 0, 0]
sample_csv_val = df[
    (df["group"] == groups[0])
    & (df["timepoint"] == timepoints[0])
    & (df["subject_id"] == subjects[0])
    & (df["channel"] == channels[0])
    & (df["feature"] == features[0])
]["power_value"].values[0]
assert np.isclose(sample_arr_val, sample_csv_val, equal_nan=True), "Mismatch between CSV and Array!"

print(f"5D Array shape: {arr_5d.shape} | Total NaNs: {np.isnan(arr_5d).sum()}")

# 7. Export 5D NumPy Array to .npy file
output_dir = os.path.join(script_dir, "data")
os.makedirs(output_dir, exist_ok=True)
output_npy_path = os.path.join(output_dir, "eeg_5d_array.npy")

np.save(output_npy_path, arr_5d)
print(f"Exported 5D array to: {output_npy_path}")
