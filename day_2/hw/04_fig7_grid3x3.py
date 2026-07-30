"""
Day 2 Homework Solution — Figure 7: 3x3 Timepoint x Group Correlation Grid
EEG Analysis Odyssey — Day 2
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dynamic file path loading
script_dir = os.path.dirname(os.path.abspath(__file__))
day2_dir = os.path.dirname(script_dir)
workspace_dir = os.path.dirname(day2_dir)

data_candidates = [
    os.path.join(workspace_dir, "day_1", "data", "neural_data.npy"),
    os.path.join(day2_dir, "..", "day_1", "data", "neural_data.npy"),
    "day_1/data/neural_data.npy",
]
meta_candidates = [
    os.path.join(workspace_dir, "day_1", "data", "metadata.pkl"),
    os.path.join(day2_dir, "..", "day_1", "data", "metadata.pkl"),
    "day_1/data/metadata.pkl",
]

data_path = next((p for p in data_candidates if os.path.exists(p)), "day_1/data/neural_data.npy")
meta_path = next((p for p in meta_candidates if os.path.exists(p)), "day_1/data/metadata.pkl")

data = np.load(data_path)
with open(meta_path, "rb") as fh:
    meta = pickle.load(fh)

G, T, S, C, F = data.shape
gnames = meta["group_names"]
fnames = meta["feature_names"]
t_names = ["Baseline", "Task", "Rest"]

# Ensure hw/plots directory exists inside the hw folder
output_dir = os.path.join(script_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# 3x3 Grid: Rows = Timepoints, Columns = Groups
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

for row_t, t_label in enumerate(t_names):
    for col_g, g_label in enumerate(gnames):
        ax = axes[row_t, col_g]
        
        # Reshape data for specific Group & Timepoint across subjects & channels -> (S*C, F)
        mat = data[col_g, row_t].reshape(-1, F)
        corr = np.corrcoef(mat.T)  # (F, F)
        
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        
        # Title and Labels
        ax.set_title(f"{g_label} — {t_label}", fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks(range(F))
        ax.set_yticks(range(F))
        
        if row_t == 2:
            ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([])
            
        if col_g == 0:
            ax.set_yticklabels(fnames, fontsize=8, fontweight="medium")
        else:
            ax.set_yticklabels([])
            
        # Annotate correlation values in each cell
        for i in range(F):
            for j in range(F):
                r_val = corr[i, j]
                text_col = "white" if abs(r_val) > 0.60 else "black"
                ax.text(j, i, f"{r_val:.2f}", ha="center", va="center", fontsize=7, color=text_col)
                
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

fig.suptitle("Dynamic Feature Correlation Grid (3×3: Timepoint × Cohort)", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout()

save_path = os.path.join(output_dir, "hw_fig7_3x3_correlations.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[HW 4 Solution] Figure saved successfully to: {save_path}")
