"""
Day 2 Homework Solution — Figure 4: Shared Y-Axis & Overlayed Cohorts (1x3 Band Grid)
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
t_names = ["Baseline", "Task", "Rest"]
COLORS = ["#2196F3", "#F44336", "#4CAF50"]   # Blue, Red
t_axis = np.arange(T)

# Ensure output directory exists
os.makedirs("day_2/hw/plots", exist_ok=True)

def _mean_sem(g, fi):
    sig = data[g, :, :, :, fi]            # (T, S, C)
    mu = sig.mean(axis=(1, 2))            # (T,)
    per_subj = sig.mean(axis=2)          # (T, S)
    sem = per_subj.std(axis=1) / np.sqrt(S)  # (T,)
    return mu, sem

# 1x3 Subplot grid with sharey=True
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
feat_cols = [("Alpha Band", 2), ("Beta Band", 3), ("Gamma Band", 4)]

for col, (fname, fi) in enumerate(feat_cols):
    ax = axes[col]
    for g in [0, 1]:  # Control vs Patient overlayed
        mu, sem = _mean_sem(g, fi)
        ax.plot(t_axis, mu, color=COLORS[g], lw=2.2, marker="o", markersize=6, label=gnames[g])
        ax.fill_between(t_axis, mu - sem, mu + sem, color=COLORS[g], alpha=0.20)
    
    ax.set_title(fname, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(t_axis)
    ax.set_xticklabels(t_names, fontsize=10, fontweight="medium")
    ax.set_xlabel("Timepoint", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if col == 0:
        ax.set_ylabel("Mean Power (μV²/Hz)", fontsize=11, fontweight="bold")

# Consolidated legend outside right plot frame
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=11)

fig.suptitle("Frequency Band Comparison: Control vs Patient (Shared Y-Axis Scale)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()

save_path = "day_2/hw/plots/hw_fig4_shared_y.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"[HW 2 Solution] Figure saved successfully to: {save_path}")
