"""
Python Basics: Plotting for Insights
Turning arrays of raw numbers into a visual story


  s01_single_timeseries.png       Line plot basics
  s02_multigroup_timeseries.png   Multiple lines on one axes
  s03_confidence_bands.png        fill_between for ±SEM bands
  s04_subplots_2x3.png            subplots() grid
  s05_mne_topoplot.png            MNE Topographic scalp map (plot_topomap)
  s06_raincloud_plots.png         Raincloud plots (KDE cloud + boxplot + rain scatter)
  s07_correlation_matrix.png      Annotated correlation matrix
  s08_publication_figure.png      Publication figure with MNE Delta Topomap

"""
# Load libraries
import numpy as np
import pickle, os
import matplotlib
matplotlib.use("Agg")               # non-interactive: saves files, no pop-up windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
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

# Load data directly from day_1 shared data directory
data = np.load("neural_data.npy")
with open("metadata.pkl", "rb") as fh:
    meta = pickle.load(fh)

G, T, S, C, F = data.shape
gnames  = meta["group_names"]
fnames  = meta["feature_names"]
chnames = meta["channel_names"]

COLORS  = ["#2196F3", "#F44336", "#4CAF50"]   # blue · red · green
t_axis  = np.arange(T) 
RNG     = np.random.default_rng(2026)

# Style your plot theme
def setup_publication_style():
    """
    Applies clean, publication-quality defaults across Matplotlib figures.
    - Removes top/right clutter spines
    - Establishes crisp typography hierarchy
    - Ensures high-DPI vector rendering defaults
    """
    plt.rcParams.update({
        "font.family"          : "sans-serif",            # Primary font family category
        "font.sans-serif"     : ["Inter", "DejaVu Sans", "Helvetica", "Arial"],  # Preferred font fallback hierarchy
        "axes.spines.top"      : False,                  # Hide top spine boundary to reduce clutter
        "axes.spines.right"    : False,                  # Hide right spine boundary to reduce clutter
        "axes.linewidth"       : 1.2,                    # Line thickness for remaining left/bottom spines
        "axes.edgecolor"       : "#2E3654",              # Dark slate color for spine borders
        "axes.labelcolor"      : "#1E293B",              # High-contrast color for X/Y axis labels
        "axes.titlesize"       : 12,                     # Subplot title font size in points
        "axes.titleweight"     : "bold",                 # Bold font weight for subplot titles
        "axes.labelsize"       : 11,                     # Font size for X/Y axis labels
        "axes.labelweight"     : "medium",               # Medium font weight for axis labels
        "xtick.major.width"    : 1.2,                    # Line thickness of X-axis major tick marks
        "ytick.major.width"    : 1.2,                    # Line thickness of Y-axis major tick marks
        "xtick.labelsize"      : 9,                      # Font size for X-axis numeric tick labels
        "ytick.labelsize"      : 9,                      # Font size for Y-axis numeric tick labels
        "figure.titlesize"     : 14,                     # Main figure super-title font size
        "figure.titleweight"   : "bold",                 # Bold font weight for super-title
        "figure.dpi"           : 150,                    # Interactive display resolution (dots per inch)
        "savefig.dpi"          : 300,                    # High-resolution export DPI for publication
        "savefig.bbox"         : "tight",                # Automatically strip white padding around figure
    })

# Helper: mean ± SEM for one group's feature over time
def _mean_sem(g, fi):
    sig       = data[g, :, :, :, fi]          # (T, S, C)
    mu        = sig.mean(axis=(1, 2))          # (T,)  mean over S & C
    per_subj  = sig.mean(axis=2)              # (T, S) mean over C
    sem       = per_subj.std(axis=1) / np.sqrt(S)  # (T,)
    return mu, sem


# Single time series, weird x axis
print("Single time-series  →  plots/s01_single_timeseries.png")

fig, ax = plt.subplots(figsize=(10, 3.5))

ts = data[0, :, 0, 9, 2]          # Control, subject 0, Cz, alpha
ax.plot(ts, color=COLORS[0], lw=1.5, label="Control · sub0 · Cz · alpha")

ax.set_xlabel("Time ", fontsize=11)
ax.set_ylabel("Power (μV²/Hz)", fontsize=11)
ax.set_title("Alpha Power — Control Group, Subject 0, Channel Cz", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/s01_single_timeseries.png", dpi=120)
plt.close(fig)

# ── MULTI-GROUP TIME-SERIES OVERLAY ───────────────────────────────────────────
print("Multi-group time-series  →  plots/s02_multigroup_timeseries.png")

fig, ax = plt.subplots(figsize=(11, 4))
for g in range(G):
    ts_g = data[g, :, :, :, 2].mean(axis=(1, 2))   # (T,)
    ax.plot(ts_g, color=COLORS[g], lw=2, label=gnames[g])

ax.set_xlabel("Time (s)"); ax.set_ylabel("Mean Alpha Power")
ax.set_title("Alpha Power Over Time — All Groups")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/s02_multigroup_timeseries.png", dpi=120)
plt.close(fig)

# ── CONFIDENCE BANDS (fill_between ±SEM) ─────────────────────────────────────
print("Confidence bands  →  plots/s03_confidence_bands.png")

fig, ax = plt.subplots(figsize=(11, 4))
for g in range(G):
    mu, sem = _mean_sem(g, fi=2)   # alpha feature
    ax.plot( mu, color=COLORS[g], lw=2, label=gnames[g])
    ax.fill_between(t_axis, mu - sem, mu + sem, color=COLORS[g], alpha=0.20)

ax.set_xlabel("Time (s)"); ax.set_ylabel("Alpha Power ± SEM")
ax.set_title("Mean Alpha Power with Standard Error Bands")

# Position legend at the bottom below the plot area so it never obscures confidence bands
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/s03_confidence_bands.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ── SUBPLOTS GRID ─────────────────────────────────────────────────────────────
print("Subplots grid  →  plots/s04_subplots_2x3.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
feat_cols = [("Alpha", 2), ("Beta", 3), ("Gamma", 4)]

for row, g in enumerate([0, 1]):
    for col, (fname, fi) in enumerate(feat_cols):
        ax  = axes[row, col]
        mu, sem = _mean_sem(g, fi)
        ax.plot(t_axis, mu, color=COLORS[g], lw=1.8, label=gnames[g])
        ax.fill_between(t_axis, mu - sem, mu + sem, color=COLORS[g], alpha=0.20)
        ax.set_title(f"{fname} Band", fontsize=10, fontweight="bold", pad=8)
        if col == 0:
            ax.set_ylabel(f"{gnames[g]}\nMean Power (μV²/Hz)", fontsize=9, fontweight="bold")
        if row == 1:
            ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Place label legend outside the plot frame so it never obscures signal traces
        if col == 2:
            ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False, fontsize=9)

fig.suptitle("Feature Comparison: Control vs Patient", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/s04_subplots_2x3.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ── MNE SCALP TOPOGRAPHY ─────────────────────────────────────────────────────
print("MNE Topoplot  →  plots/s05_mne_topoplot.png")

# Construct MNE Info object with standard 10-20 montage
info = mne.create_info(ch_names=chnames, sfreq=125, ch_types="eeg")
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
vmax = max(data[g, :, :, :, 2].mean(axis=(0, 1)).max() for g in range(G))
vmin = min(data[g, :, :, :, 2].mean(axis=(0, 1)).min() for g in range(G))

for g in range(G):
    # Mean alpha power per channel across time and subjects
    alpha_power = data[g, :, :, :, 2].mean(axis=(0, 1))   # shape (32,)
    im, _ = mne.viz.plot_topomap(
        alpha_power, info, axes=axes[g], show=False, cmap=cmap_becp, vlim=(vmin, vmax)
    )
    axes[g].set_title(f"{gnames[g]} — Alpha Topography", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[g], shrink=0.75, pad=0.04)

fig.suptitle("MNE Scalp Topography: Alpha Power Across Cohorts", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/s05_mne_topoplot.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ── RAINCLOUD PLOTS ───────────────────────────────────────────────────────────
print("Raincloud plots  →  plots/s06_raincloud_plots.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
target_feats = [("Alpha", 2), ("Beta", 3), ("Gamma", 4)]

for col_idx, (fname, fi) in enumerate(target_feats):
    ax = axes[col_idx]
    for g in range(G):
        sub_vals = data[g, :, :, :, fi].mean(axis=(0, 2))  # per-subject mean
        color = COLORS[g]
        
        # 1. Cloud: Half-KDE density plot
        kde = stats.gaussian_kde(sub_vals)
        y_grid = np.linspace(sub_vals.min() - 0.04, sub_vals.max() + 0.04, 100)
        density = (kde(y_grid) / kde(y_grid).max()) * 0.28
        ax.fill_betweenx(y_grid, g + 0.06, g + 0.06 + density, color=color, alpha=0.45)
        ax.plot(g + 0.06 + density, y_grid, color=color, lw=1.5)
        
        # 2. Umbrella: Narrow box plot
        ax.boxplot(sub_vals, positions=[g], widths=0.08, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.8, edgecolor="white"),
                   medianprops=dict(color="white", lw=1.8),
                   whiskerprops=dict(color=color, lw=1.2),
                   capprops=dict(color=color, lw=1.2),
                   flierprops=dict(marker=""))
        
        # 3. Rain: Jittered raw subject data points
        jitter = RNG.uniform(-0.06, 0.06, size=len(sub_vals))
        ax.scatter(g - 0.18 + jitter, sub_vals, color=color,
                   alpha=0.65, s=28, edgecolors="white", linewidths=0.5)

    ax.set_xticks(range(G))
    ax.set_xticklabels(gnames, fontweight="bold")
    ax.set_title(f"{fname} Power Distribution", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    if col_idx == 0:
        ax.set_ylabel("Mean Power (μV²/Hz)", fontsize=10)

fig.suptitle("Raincloud Plots: Subject Distributions Across Frequency Bands",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/s06_raincloud_plots.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ── ANNOTATED CORRELATION MATRIX ─────────────────────────────────────────────
print("Correlation matrix  →  plots/s07_correlation_matrix.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for g in range(G):
    mat  = data[g].reshape(-1, F)          # (T*S*C, F)
    corr = np.corrcoef(mat.T)              # (F, F)
    ax   = axes[g]
    im   = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(F)); ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(F)); ax.set_yticklabels(fnames, fontsize=8)
    ax.set_title(gnames[g], fontweight="bold")
    for i in range(F):
        for j in range(F):
            r   = corr[i, j]
            col = "w" if abs(r) > 0.60 else "k"
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=6, color=col)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

fig.suptitle("Feature Correlation Matrices — All Groups",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/s07_correlation_matrix.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ── JOURNAL-READY FIGURE WITH MNE DELTA TOPOMAP ───────────────────────────────
print("Publication figure  →  plots/s08_publication_figure.png")

setup_publication_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Alpha vs Beta scatter (per subject, Control vs Patient)
for g in [0, 1]:
    sub_alpha = data[g, :, :, :, 2].mean(axis=(0, 2))   # (S,)
    sub_beta  = data[g, :, :, :, 3].mean(axis=(0, 2))   # (S,)
    ax1.scatter(sub_alpha, sub_beta, c=COLORS[g], alpha=0.72,
                s=65, label=gnames[g], edgecolors="white", linewidths=0.6)

ax1.set_xlabel("Mean Alpha Power")
ax1.set_ylabel("Mean Beta Power")
ax1.set_title("(A) Alpha vs Beta Scatter (per subject)")
ax1.legend(frameon=False)

# Panel B: Patient − Control Delta Topomap (MNE topoplot)
diff_alpha = (data[1, :, :, :, 2].mean(axis=(0, 1))        # Patient mean alpha per channel
            - data[0, :, :, :, 2].mean(axis=(0, 1)))       # Control mean alpha per channel
vlim = np.abs(diff_alpha).max()

im, _ = mne.viz.plot_topomap(
    diff_alpha, info, axes=ax2, show=False, cmap=cmap_becp, vlim=(-vlim, vlim)
)
ax2.set_title("(B) Patient − Control Δ Alpha Topography")
fig.colorbar(im, ax=ax2, shrink=0.75, pad=0.04, label="Δ Alpha Power")

fig.suptitle("Neural Biomarkers: Patient vs Control Analysis", y=0.98)
fig.tight_layout()
fig.savefig("plots/s08_publication_figure.png", dpi=150, bbox_inches="tight")
plt.close(fig)
plt.rcParams.update(plt.rcParamsDefault)