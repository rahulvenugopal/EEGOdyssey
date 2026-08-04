"""
Day 2 Homework Solution — Figure 3: Error Bars Instead of Ribbon (Mean ± SEM)
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
COLORS = ["#2196F3", "#F44336", "#4CAF50"]   # Blue, Red, Green
t_axis = np.arange(T)

# Ensure output directory exists
os.makedirs("day_2/hw/plots", exist_ok=True)

# Helper: mean and SEM calculation
def _mean_sem(g, fi=2):
    sig = data[g, :, :, :, fi]            # (T, S, C)
    mu = sig.mean(axis=(1, 2))            # (T,) mean over subjects & channels
    per_subj = sig.mean(axis=2)          # (T, S) mean over channels
    sem = per_subj.std(axis=1) / np.sqrt(S)  # (T,) SEM across subjects
    return mu, sem

# Plotting with error bars
fig, ax = plt.subplots(figsize=(10, 4.5))

for g in range(G):
    mu, sem = _mean_sem(g, fi=2)   # Alpha band
    ax.errorbar(
        t_axis, mu, yerr=sem,
        fmt="-o", color=COLORS[g], lw=2.2, markersize=7,
        capsize=6, capthick=1.8, elinewidth=1.8,
        label=gnames[g]
    )

ax.set_xticks(t_axis)
ax.set_xticklabels(t_names, fontsize=11, fontweight="medium")
ax.set_xlabel("Timepoint", fontsize=11, fontweight="bold")
ax.set_ylabel("Alpha Power (μV²/Hz) ± SEM", fontsize=11, fontweight="bold")
ax.set_title("Alpha Power Over Time — Discrete Error Bars (Mean ± SEM)", fontsize=13, fontweight="bold", pad=12)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
save_path = "day_2/hw/plots/hw_fig3_errorbars.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"[HW 1 Solution] Figure saved successfully to: {save_path}")
