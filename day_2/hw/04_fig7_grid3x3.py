"""
Day 2 Homework Solution — Figure 7: 3x3 Timepoint x Group Correlation Grid
EEG Analysis Odyssey — Day 2
"""

import os
import pickle
import numpy as np
import matplotlib
# Interactive backend enabled by default so pop-up figure windows display for students
import matplotlib.pyplot as plt

# Load neural dataset and metadata directly using relative path
data = np.load("day_1/data/neural_data.npy")
with open("day_1/data/metadata.pkl", "rb") as fh:
    meta = pickle.load(fh)

G, T, S, C, F = data.shape
gnames = meta["group_names"]
fnames = meta["feature_names"]
t_names = ["Baseline", "Task", "Rest"]

# Ensure output directory exists
os.makedirs("day_2/hw/plots", exist_ok=True)

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

save_path = "day_2/hw/plots/hw_fig7_3x3_correlations.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"[HW 4 Solution] Figure saved successfully to: {save_path}")
