"""
EEGOdyssey — Day 3: Generate Sociodemographics Dataset

This script generates a sociodemographics dataset (`sociodemographics.csv`)
containing Age, Gender, and BMI for all participants across groups and timepoints.

Note on Subject Identifier Uniqueness:
  `subject_id` (e.g. `Sub_01`) is NOT globally unique on its own because each group
  (Control, Patient, Treatment) has its own `Sub_01` to `Sub_30`.
  To uniquely identify a subject, you MUST use the composite key:
    ['group', 'timepoint', 'subject_id']  or  ['group', 'subject_id']
"""

import os
import numpy as np
import pandas as pd


def generate_sociodemographics():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "sociodemographics.csv")

    rng = np.random.default_rng(2026)
    groups = ["Control", "Patient", "Treatment"]
    timepoints = ["Baseline", "Task", "Rest"]
    subjects = [f"Sub_{s+1:02d}" for s in range(30)]

    rows = []
    for g in groups:
        for s in subjects:
            # Baseline demographic traits per participant
            age_base = int(rng.integers(22, 65))
            gender = rng.choice(["M", "F"])
            bmi_base = round(float(rng.normal(24.5, 3.5)), 1)
            bmi_base = max(18.5, min(35.0, bmi_base))

            for t_idx, t in enumerate(timepoints):
                rows.append(
                    {
                        "group": g,
                        "timepoint": t,
                        "subject_id": s,
                        "age": age_base,
                        "gender": gender,
                        "bmi": round(bmi_base + t_idx * 0.1, 1),
                    }
                )

    df_demo = pd.DataFrame(rows)
    df_demo.to_csv(output_csv, index=False)

    print(f"Generated sociodemographics CSV with {len(df_demo)} rows.")
    print(f"Saved to: {output_csv}")
    print("\nFirst 10 rows:")
    print(df_demo.head(10).to_string(index=False))
    return output_csv


if __name__ == "__main__":
    generate_sociodemographics()
