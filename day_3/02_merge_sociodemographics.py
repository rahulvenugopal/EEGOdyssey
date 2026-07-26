"""
================================================================================
EEGOdyssey — Day 3: Merge Sociodemographics with Features Mastersheet
================================================================================

In this script, we demonstrate how to combine tabular EEG neural features
(`eeg_mastersheet.csv`) with participant sociodemographics (`sociodemographics.csv`).

Real-World Research Rationale:
  1. Separation of Concerns : Electrophysiological signals (EEG) and clinical EHR
                               data (demographics) are acquired separately.
  2. Memory Efficiency (M:1): 192 feature rows per visit map to 1 demographic record.
  3. Controlling Confounders: Merging Age, Sex, BMI enables ANCOVA covariate controls
                               and Linear Mixed Models (LMMs).

Key Concept — Non-Unique Subject IDs & Defensive Sanity Checks:
  `subject_id` (e.g. `Sub_01`) is repeated across groups (`Control`, `Patient`, `Treatment`).
  Therefore, we specify the composite key `['group', 'timepoint', 'subject_id']` and enforce
  Pandas `validate="many_to_one"` assertions.

Run: python day_3/02_merge_sociodemographics.py
================================================================================
"""

import os
import pandas as pd


def resolve_file_paths():
    """Resolve paths for features CSV and sociodemographics CSV."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    features_path = os.path.join(script_dir, "data", "eeg_mastersheet.csv")
    demo_path = os.path.join(script_dir, "data", "sociodemographics.csv")

    if not os.path.exists(features_path):
        import convert_to_mastersheet
        features_path, _ = convert_to_mastersheet.create_mastersheet()

    if not os.path.exists(demo_path):
        import importlib
        gen_demo = importlib.import_module("00_generate_sociodemographics")
        demo_path = gen_demo.generate_sociodemographics()

    return features_path, demo_path, script_dir


def main():
    print("=" * 80)
    print(" DAY 3: MERGING SOCIODEMOGRAPHICS WITH FEATURES MASTERSHEET")
    print("=" * 80)

    features_path, demo_path, script_dir = resolve_file_paths()

    # 1. Load CSV Datasets
    print(f"\n[STEP 1] Loading Features CSV      : {features_path}")
    df_features = pd.read_csv(features_path)
    print(f"  • Shape : {df_features.shape} ({len(df_features):,} rows × {len(df_features.columns)} columns)")

    print(f"\n[STEP 2] Loading Demographics CSV    : {demo_path}")
    df_demo = pd.read_csv(demo_path)
    print(f"  • Shape : {df_demo.shape} ({len(df_demo):,} subject-visit records × {len(df_demo.columns)} columns)")

    # 2. Defensive Sanity Check on Key Lookup Table
    print("\n" + "-" * 80)
    print("[STEP 3] Defensive Sanity Check on Composite Key")
    print("-" * 80)
    composite_key = ["group", "timepoint", "subject_id"]

    # Assert no duplicate key records exist in demographics lookup table
    has_duplicates = df_demo.duplicated(subset=composite_key).any()
    print(f"  • Composite Key                     : {composite_key}")
    print(f"  • Duplicate Keys in Demographics    : {has_duplicates} (Expected: False)")
    assert not has_duplicates, "Sanity Check Error: Duplicate demographic records found!"

    # 3. Perform Relational Left Merge with explicit Many-to-One validation
    print("\n" + "-" * 80)
    print("[STEP 4] Merging with pandas.merge(..., validate='many_to_one')")
    print("-" * 80)

    df_merged = pd.merge(
        df_features,
        df_demo,
        on=composite_key,
        how="left",
        validate="many_to_one",  # Enforces M:1 relational validation
    )

    print(f"  • Merged DataFrame Shape: {df_merged.shape}")
    print(f"  • Total Columns ({len(df_merged.columns)}): {list(df_merged.columns)}")

    # 4. Post-Merge Defensive Invariant Assertions
    print("\n" + "-" * 80)
    print("[STEP 5] Post-Merge Invariant Assertions")
    print("-" * 80)

    # Invariant 1: Row count must match left feature dataframe exactly
    assert len(df_merged) == len(df_features), "Row count unexpectedly changed during join!"
    print("  [OK] Invariant 1: Row count preserved perfectly!")

    # Invariant 2: Zero missing values in newly merged demographic columns
    missing_demo_vals = df_merged[["age", "gender", "bmi"]].isnull().sum().sum()
    assert missing_demo_vals == 0, "Unmatched demographic entries detected!"
    print("  [OK] Invariant 2: Demographics 100% matched (0 null values)!")

    print("\nFirst 10 rows of Enriched Master CSV:")
    print(df_merged.head(10).to_string(index=False))

    # 5. Save Enriched Mastersheet CSV
    output_path = os.path.join(script_dir, "data", "eeg_mastersheet_enriched.csv")
    df_merged.to_csv(output_path, index=False)
    print(f"\n[OK] Enriched Mastersheet exported successfully to:\n  {output_path}")

    print("\n" + "=" * 80)
    print(" MERGE & DEFENSIVE VALIDATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
