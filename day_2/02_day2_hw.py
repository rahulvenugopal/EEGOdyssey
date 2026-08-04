import os
import pickle
import numpy as np
import matplotlib
# Interactive backend enabled by default so pop-up figure windows display for students
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne

os.makedirs("plots", exist_ok=True)

# Benedikt Ehinger's EEG Topographic Colormap (becp)
EHINGER_COLORS = [
    (0.2706, 0.4588, 0.7059),  # Deep Blue
    (0.5686, 0.7490, 0.8588),  # Soft Blue
    (0.8784, 0.9529, 0.9725),  # Ice Blue
    (1.0000, 1.0000, 0.7490),  # Neutral Yellow
    (0.9961, 0.8784, 0.5647),  # Warm Sand
    (0.9882, 0.5529, 0.3490),  # Coral Orange
    (0.8431, 0.1882, 0.1529)   # Deep Crimson
]
cmap_becp = LinearSegmentedColormap.from_list("becp", EHINGER_COLORS, N=256)

# Load neural dataset and metadata directly using relative path
data = np.load("day_1/data/neural_data.npy")
with open("day_1/data/metadata.pkl", "rb") as fh:
    meta = pickle.load(fh)

chnames = meta["channel_names"]
info = mne.create_info(ch_names=chnames, sfreq=125, ch_types="eeg")
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# ── HOMEWORK PROBLEM: HIGH ALPHA ELECTRODE HIGHLIGHT TOPOPLOT ──────────────────
print("Homework topoplot   ->  plots/s09_homework_topoplot.png")

# Control group (g=0), Baseline timepoint (t=0), Alpha feature (fi=2)
# Mean alpha power per electrode channel across subjects -> shape (32,)
alpha_base = data[0, 0, :, :, 2].mean(axis=0)

# Explicitly set vmin and vmax to data minimum and maximum
vmin, vmax = alpha_base.min(), alpha_base.max()

# Threshold: channels exceeding 75th percentile power
threshold = np.percentile(alpha_base, 75)
mask = alpha_base > threshold

fig, ax = plt.subplots(figsize=(6, 5))
im, _ = mne.viz.plot_topomap(
    alpha_base, info, axes=ax, show=False, cmap=cmap_becp, vlim=(vmin, vmax),
    mask=mask, mask_params=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=9)
)
ax.set_title("Homework Challenge: High Alpha Electrode Topography", fontsize=11, fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04, label="Alpha Power (μV²/Hz)")
fig.tight_layout()
fig.savefig("plots/s09_homework_topoplot.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)