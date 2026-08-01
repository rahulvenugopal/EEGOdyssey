"""
================================================================================
EEGOdyssey — Day 3: Merge Sociodemographics with Features Mastersheet
================================================================================

Combines tabular EEG neural features (`eeg_mastersheet.csv`) with participant
sociodemographics (`sociodemographics.csv`) using composite keys.

Run: python day_3/02_merge_sociodemographics.py
================================================================================
"""

import os
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
gen_dir = os.path.join(script_dir, "generators")
if gen_dir not in sys.path:
    sys.path.insert(0, gen_dir)

features_path = os.path.join(script_dir, "data", "eeg_mastersheet.csv")
demo_path = os.path.join(script_dir, "data", "sociodemographics.csv")

if not os.path.exists(features_path):
    import convert_to_mastersheet
    features_path, _ = convert_to_mastersheet.create_mastersheet()

if not os.path.exists(demo_path):
    import importlib
    gen_demo = importlib.import_module("00_generate_sociodemographics")
    demo_path = gen_demo.generate_sociodemographics()

print(f"Loading features: {features_path}")
df_features = pd.read_csv(features_path)

print(f"Loading sociodemographics: {demo_path}")
df_demo = pd.read_csv(demo_path)

composite_key = ["group", "timepoint", "subject_id"]

# Assert no duplicate key records exist in demographics lookup table
assert not df_demo.duplicated(subset=composite_key).any(), "Duplicate demographic records found!"

# Merge features and demographics
df_merged = pd.merge(
    df_features,
    df_demo,
    on=composite_key,
    how="left",
    validate="many_to_one",
)

# Post-Merge Invariant Assertions
assert len(df_merged) == len(df_features), "Row count unexpectedly changed during join!"
assert df_merged[["age", "gender", "bmi"]].isnull().sum().sum() == 0, "Unmatched demographic entries detected!"

output_path = os.path.join(script_dir, "data", "eeg_mastersheet_enriched.csv")
df_merged.to_csv(output_path, index=False)
print(f"Merged dataframe shape: {df_merged.shape} -> Exported to: {output_path}")
