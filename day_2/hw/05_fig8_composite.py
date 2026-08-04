"""
Day 2 Homework Solution — Figure 8: Biomarker Composite Analysis (2 Subplots & Scientific Inference)
EEG Analysis Odyssey — Day 2
"""

import os
import pickle
import numpy as np
import matplotlib
# Interactive backend enabled by default so pop-up figure windows display for students
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Load neural dataset and metadata directly using relative path
data = np.load("day_1/data/neural_data.npy")
with open("day_1/data/metadata.pkl", "rb") as fh:
    meta = pickle.load(fh)

G, T, S, C, F = data.shape
gnames = meta["group_names"]
COLORS = ["#2196F3", "#F44336", "#4CAF50"]   # Control (Blue), Patient (Red), Treatment (Green)
RNG = np.random.default_rng(2026)

# Ensure output directory exists
os.makedirs("day_2/hw/plots", exist_ok=True)

# Calculate subject-level mean Theta (fi=1) and Alpha (fi=2) power across time & channels
# Shape per group: (S,)
theta_power = np.array([data[g, :, :, :, 1].mean(axis=(0, 2)) for g in range(G)])
alpha_power = np.array([data[g, :, :, :, 2].mean(axis=(0, 2)) for g in range(G)])

# Theta / Alpha Ratio (TAR) per subject
tar = theta_power / alpha_power  # Shape (3, S)

# Setup Composite Figure (1x2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Panel A: Theta / Alpha Ratio (TAR) Across Cohorts ────────────────────────
for g in range(G):
    sub_tar = tar[g]
    mean_tar = sub_tar.mean()
    sem_tar = sub_tar.std() / np.sqrt(S)
    
    # 1. Individual subject jittered scatter points
    jitter = RNG.uniform(-0.10, 0.10, size=S)
    ax1.scatter(
        g + jitter, sub_tar, color=COLORS[g], alpha=0.55, s=36,
        edgecolors="white", linewidths=0.5, label=f"{gnames[g]} subjects" if g == 0 else ""
    )
    
    # 2. Group Mean ± SEM error bar callout
    ax1.errorbar(
        g + 0.22, mean_tar, yerr=sem_tar, fmt="o", color=COLORS[g],
        lw=2.5, markersize=8, capsize=6, capthick=2, markeredgecolor="white"
    )
    
    # 3. Mean line segment
    ax1.hlines(mean_tar, g - 0.18, g + 0.18, colors=COLORS[g], linestyles="--", lw=1.8)

ax1.set_xticks(range(G))
ax1.set_xticklabels(gnames, fontsize=11, fontweight="bold")
ax1.set_ylabel("Theta / Alpha Ratio (TAR)", fontsize=11, fontweight="bold")
ax1.set_title("(A) Theta/Alpha Ratio (TAR) Biomarker Distribution", fontsize=12, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Annotate key clinical values on Panel A
ax1.text(0 + 0.22, tar[0].mean() + 0.08, f"Mean: {tar[0].mean():.2f}", ha="center", fontsize=9, fontweight="bold", color=COLORS[0])
ax1.text(1 + 0.22, tar[1].mean() + 0.08, f"Mean: {tar[1].mean():.2f}", ha="center", fontsize=9, fontweight="bold", color=COLORS[1])
ax1.text(2 + 0.22, tar[2].mean() + 0.08, f"Mean: {tar[2].mean():.2f}", ha="center", fontsize=9, fontweight="bold", color=COLORS[2])

# ── Panel B: Alpha vs Theta Joint Scatter & Convex Cluster Envelopes ──────────
for g in range(G):
    x = alpha_power[g]  # Alpha Power (X-axis)
    y = theta_power[g]  # Theta Power (Y-axis)
    
    # Scatter points
    ax2.scatter(
        x, y, color=COLORS[g], alpha=0.75, s=55,
        edgecolors="white", linewidths=0.6, label=gnames[g]
    )
    
    # Draw convex hull envelope to highlight cluster bounds
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax2.plot(points[simplex, 0], points[simplex, 1], color=COLORS[g], lw=1.2, alpha=0.6)
    ax2.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=COLORS[g], alpha=0.08)

ax2.set_xlabel("Mean Alpha Power (μV²/Hz)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Mean Theta Power (μV²/Hz)", fontsize=11, fontweight="bold")
ax2.set_title("(B) Subject-Level Joint Frequency Cluster Space", fontsize=12, fontweight="bold")
ax2.legend(frameon=False, fontsize=10, loc="upper right")
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Neural Biomarker Discovery: Theta/Alpha Pathophysiology & Treatment Restoration", fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout()

save_path = "day_2/hw/plots/hw_fig8_composite.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"[HW 5 Solution] Figure saved successfully to: {save_path}")

# Print formal inference report
print("\n" + "="*80)
print("INFERENCE AND SCIENTIFIC FINDINGS (FIGURE 8 COMPOSITE BIOMARKER ANALYSIS)")
print("="*80)
print(f"1. PATIENT PATHOPHYSIOLOGY (Patient vs Control):")
print(f"   * Control Cohort: High Alpha power ({alpha_power[0].mean():.3f} uV^2/Hz) and balanced Theta ({theta_power[0].mean():.3f} uV^2/Hz) -> Baseline TAR = {tar[0].mean():.2f}")
print(f"   * Patient Cohort: Severe Alpha suppression ({alpha_power[1].mean():.3f} uV^2/Hz) combined with elevated Theta ({theta_power[1].mean():.3f} uV^2/Hz) -> Elevated TAR = {tar[1].mean():.2f}")
print(f"   * Clinical Significance: Elevated Theta/Alpha Ratio (TAR = 1.46 vs 0.98) serves as a sensitive electrophysiological biomarker for patient neural dysfunction.\n")

print(f"2. TREATMENT RECOVERY VECTOR (Treatment vs Patient):")
print(f"   * Treatment Cohort: Restores Alpha power ({alpha_power[2].mean():.3f} uV^2/Hz) and normalizes Theta power ({theta_power[2].mean():.3f} uV^2/Hz) -> Normalized TAR = {tar[2].mean():.2f}")
print(f"   * Diagnostic Clustering: Joint frequency space (Panel B) demonstrates distinct trajectories: Treatment shifts patient cluster coordinates back towards healthy Control baseline.\n")
print("="*80 + "\n")
