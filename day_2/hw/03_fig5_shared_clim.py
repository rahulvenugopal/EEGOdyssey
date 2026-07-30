"""
Day 2 Homework Solution — Figure 5: Shared Global Color Limits (vmin & vmax) Across Topoplots
EEG Analysis Odyssey — Day 2
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne

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
chnames = meta["channel_names"]

# Benedikt Ehinger's EEG Topographic Colormap (becp)
EHINGER_COLORS = [
    (0.2706, 0.4588, 0.7059),
    (0.5686, 0.7490, 0.8588),
    (0.8784, 0.9529, 0.9725),
    (1.0000, 1.0000, 0.7490),
    (0.9961, 0.8784, 0.5647),
    (0.9882, 0.5529, 0.3490),
    (0.8431, 0.1882, 0.1529)
]
cmap_becp = LinearSegmentedColormap.from_list("becp", EHINGER_COLORS, N=256)

# Ensure hw/plots directory exists inside the hw folder
output_dir = os.path.join(script_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# MNE Info structure
info = mne.create_info(ch_names=chnames, sfreq=125, ch_types="eeg")
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# Compute mean alpha power per electrode for each group -> shape (3, 32)
group_alpha_powers = np.array([data[g, :, :, :, 2].mean(axis=(0, 1)) for g in range(G)])

# Global vmin and vmax across ALL groups
global_vmin = group_alpha_powers.min()
global_vmax = group_alpha_powers.max()

print(f"Global vmin: {global_vmin:.4f}, Global vmax: {global_vmax:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for g in range(G):
    alpha_power = group_alpha_powers[g]
    im, _ = mne.viz.plot_topomap(
        alpha_power, info, axes=axes[g], show=False, cmap=cmap_becp,
        vlim=(global_vmin, global_vmax)  # Shared global color limits across all 3 plots
    )
    axes[g].set_title(f"{gnames[g]} Cohort", fontsize=12, fontweight="bold", pad=8)
    fig.colorbar(im, ax=axes[g], shrink=0.75, pad=0.04, label="Alpha Power (μV²/Hz)")

fig.suptitle(
    f"MNE Scalp Topography (Shared Color Scale: [{global_vmin:.2f}, {global_vmax:.2f}] μV²/Hz)",
    fontsize=13, fontweight="bold", y=0.98
)
fig.tight_layout()

save_path = os.path.join(output_dir, "hw_fig5_shared_clim.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[HW 3 Solution] Figure saved successfully to: {save_path}")
