"""
================================================================================
DAY 2 STUDY GUIDE  |  Python Basics: Plotting for Insights
"Turning arrays of raw numbers into a visual story."
================================================================================

Plots produced (saved to  plots/)
----------------------------------
  s01_single_timeseries.png       §1   Line plot basics
  s02_multigroup_timeseries.png   §2   Multiple lines on one axes
  s03_confidence_bands.png        §3   fill_between for ±SEM bands
  s04_subplots_2x3.png            §4   subplots() grid
  s05_channel_heatmap.png         §5   imshow heatmap
  s06_barchart_with_errors.png    §6   Bar chart + error bars
  s07_correlation_matrix.png      §7   Annotated correlation heatmap
  s08_gridspec_dashboard.png      §8   GridSpec multi-panel layout
  s09_histograms.png              §9   Overlapping histograms
  s10_publication_figure.png      §10  Publication-quality styling

Prerequisite: run  00_generate_dataset.py  first.
Run         : python 02_day2_study_guide.py
================================================================================
"""

import numpy as np
import pickle, os
import matplotlib
matplotlib.use("Agg")               # non-interactive: saves files, no pop-up windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("plots", exist_ok=True)

data = np.load("data/neural_data.npy")
with open("data/metadata.pkl", "rb") as fh:
    meta = pickle.load(fh)

G, T, S, C, F = data.shape
gnames  = meta["group_names"]
fnames  = meta["feature_names"]
chnames = meta["channel_names"]
hz      = meta["sampling_hz"]

t_axis  = np.arange(T) / hz        # time in seconds
COLORS  = ["#2196F3", "#F44336", "#4CAF50"]   # blue · red · green
DIV     = "─" * 60

# Helper: mean ± SEM for one group's feature over time
def _mean_sem(g, fi):
    sig       = data[g, :, :, :, fi]          # (T, S, C)
    mu        = sig.mean(axis=(1, 2))          # (T,)  mean over S & C
    per_subj  = sig.mean(axis=2)              # (T, S) mean over C
    sem       = per_subj.std(axis=1) / np.sqrt(S)  # (T,)
    return mu, sem


# ── §1  SINGLE TIME-SERIES ────────────────────────────────────────────────────
print("§1  Single time-series  →  plots/s01_single_timeseries.png")

fig, ax = plt.subplots(figsize=(10, 3.5))

ts = data[0, :, 0, 9, 2]          # Control, subject 0, Cz, alpha
ax.plot(t_axis, ts, color=COLORS[0], lw=1.5, label="Control · sub0 · Cz · alpha")

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Power (μV²/Hz)", fontsize=11)
ax.set_title("Alpha Power — Control Group, Subject 0, Channel Cz", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/s01_single_timeseries.png", dpi=120)
plt.close(fig)
print("  Done.")


# ── §2  MULTIPLE LINES ────────────────────────────────────────────────────────
print("§2  Multi-group time-series  →  plots/s02_multigroup_timeseries.png")

fig, ax = plt.subplots(figsize=(11, 4))
for g in range(G):
    ts_g = data[g, :, :, :, 2].mean(axis=(1, 2))   # (T,)
    ax.plot(t_axis, ts_g, color=COLORS[g], lw=2, label=gnames[g])

ax.set_xlabel("Time (s)"); ax.set_ylabel("Mean Alpha Power")
ax.set_title("Alpha Power Over Time — All Groups")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/s02_multigroup_timeseries.png", dpi=120)
plt.close(fig)
print("  Done.")


# ── §3  CONFIDENCE BANDS ─────────────────────────────────────────────────────
print("§3  Confidence bands  →  plots/s03_confidence_bands.png")

fig, ax = plt.subplots(figsize=(11, 4))
for g in range(G):
    mu, sem = _mean_sem(g, fi=2)   # alpha feature
    ax.plot(t_axis, mu, color=COLORS[g], lw=2, label=gnames[g])
    ax.fill_between(t_axis, mu - sem, mu + sem, color=COLORS[g], alpha=0.20)

# Vertical reference line
ax.axvline(x=T / (2*hz), ls="--", color="k", alpha=0.45, label="mid-session")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Alpha Power ± SEM")
ax.set_title("Mean Alpha Power with Standard Error Bands")
ax.legend(ncol=2); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/s03_confidence_bands.png", dpi=120)
plt.close(fig)
print("  Done.")


# ── §4  SUBPLOTS GRID ─────────────────────────────────────────────────────────
print("§4  Subplots grid  →  plots/s04_subplots_2x3.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
feat_cols = [("alpha", 2), ("beta", 3), ("gamma", 4)]

for row, g in enumerate([0, 1]):
    for col, (fname, fi) in enumerate(feat_cols):
        ax  = axes[row, col]
        mu, sem = _mean_sem(g, fi)
        ax.plot(t_axis, mu, color=COLORS[g], lw=1.5)
        ax.fill_between(t_axis, mu - sem, mu + sem, color=COLORS[g], alpha=0.20)
        ax.set_title(f"{gnames[g]} — {fname}", fontsize=9)
        if col == 0: ax.set_ylabel("Mean Power", fontsize=8)
        if row == 1: ax.set_xlabel("Time (s)", fontsize=8)
        ax.grid(True, alpha=0.3)

fig.suptitle("Feature Comparison: Control vs Patient", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/s04_subplots_2x3.png", dpi=120)
plt.close(fig)
print("  Done.")


# ── §5  CHANNEL HEATMAP ───────────────────────────────────────────────────────
print("§5  Channel heatmap  →  plots/s05_channel_heatmap.png")

grp_ch_feat = data.mean(axis=(1, 2))    # (G, C, F)  mean over T & S
vmin = grp_ch_feat.min(); vmax = grp_ch_feat.max()

fig, axes = plt.subplots(1, 3, figsize=(16, 8))
for g in range(G):
    ax = axes[g]
    im = ax.imshow(grp_ch_feat[g], aspect="auto",
                   cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title(gnames[g], fontsize=12, fontweight="bold")
    ax.set_xticks(range(F))
    ax.set_xticklabels(fnames, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(C))
    ax.set_yticklabels(chnames, fontsize=5)
    ax.set_xlabel("Feature")
    if g == 0: ax.set_ylabel("Channel")

fig.colorbar(im, ax=axes.tolist(), shrink=0.55, label="Mean Power")
fig.suptitle("Channel × Feature Power Heatmaps — All Groups",
             fontsize=12, fontweight="bold")
fig.savefig("plots/s05_channel_heatmap.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  Done.")


# ── §6  BAR CHART WITH ERROR BARS ────────────────────────────────────────────
print("§6  Bar chart  →  plots/s06_barchart_with_errors.png")

# Per-group, per-feature mean and SEM (SEM across subjects)
grp_mean = data.mean(axis=(1, 2, 3))    # (G, F)   mean over T, S, C
grp_sem  = np.zeros((G, F))
for g in range(G):
    # data[g] shape (T, S, C, F); mean over T=0 and C=2 → (S, F)
    per_subj       = data[g].mean(axis=(0, 2))
    grp_sem[g]     = per_subj.std(axis=0) / np.sqrt(S)

x, w = np.arange(F), 0.25
fig, ax = plt.subplots(figsize=(12, 5))
for g in range(G):
    ax.bar(x + g*w, grp_mean[g], w,
           yerr=grp_sem[g], capsize=4,
           color=COLORS[g], alpha=0.82, label=gnames[g])

ax.set_xticks(x + w); ax.set_xticklabels(fnames)
ax.set_ylabel("Mean Power"); ax.set_xlabel("Feature Band")
ax.set_title("Mean Feature Power by Group  (±SEM across subjects)")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("plots/s06_barchart_with_errors.png", dpi=120)
plt.close(fig)
print("  Done.")


# ── §7  CORRELATION MATRIX ────────────────────────────────────────────────────
print("§7  Correlation matrix  →  plots/s07_correlation_matrix.png")

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
print("  Done.")


# ── §8  GridSpec DASHBOARD ────────────────────────────────────────────────────
print("§8  GridSpec dashboard  →  plots/s08_gridspec_dashboard.png")

fig = plt.figure(figsize=(17, 10))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.50, wspace=0.35,
                        left=0.07, right=0.97, top=0.90, bottom=0.07)
PK  = dict(fontsize=12, fontweight="bold", va="top", ha="left")

# Row 0 — alpha time-courses (full width)
ax_A = fig.add_subplot(gs[0, :])
for g in range(G):
    mu, sem = _mean_sem(g, fi=2)
    ax_A.plot(t_axis, mu, color=COLORS[g], lw=2, label=gnames[g])
    ax_A.fill_between(t_axis, mu-sem, mu+sem, color=COLORS[g], alpha=0.15)
ax_A.axvline(1.0, ls="--", color="k", alpha=0.4)
ax_A.set_title("(A) Alpha Power Time-Course ± SEM", fontweight="bold")
ax_A.legend(ncol=3, fontsize=9, frameon=False); ax_A.grid(alpha=0.3)
ax_A.set_xlabel("Time (s)"); ax_A.set_ylabel("Alpha Power")
ax_A.text(0.01, 0.97, "A", transform=ax_A.transAxes, **PK)

# Row 1 — per-group channel heatmaps
for g in range(G):
    ax = fig.add_subplot(gs[1, g])
    im = ax.imshow(grp_ch_feat[g], aspect="auto", cmap="inferno")
    ax.set_title(f"(B) {gnames[g]}", fontsize=9, fontweight="bold")
    ax.set_xticks(range(F)); ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=6)
    ax.set_yticks([]); ax.set_ylabel("Channels" if g == 0 else "")
    fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
    if g == 0: ax.text(0.01, 0.99, "B", transform=ax.transAxes, **PK)

# Row 2 — bar chart (full width)
ax_C = fig.add_subplot(gs[2, :])
for g in range(G):
    ax_C.bar(x + g*w, grp_mean[g], w,
             yerr=grp_sem[g], capsize=3,
             color=COLORS[g], alpha=0.83, label=gnames[g])
ax_C.set_xticks(x + w); ax_C.set_xticklabels(fnames)
ax_C.set_title("(C) Feature Power Comparison (±SEM)", fontweight="bold")
ax_C.legend(ncol=3, fontsize=9, frameon=False); ax_C.grid(axis="y", alpha=0.3)
ax_C.text(0.01, 0.97, "C", transform=ax_C.transAxes, **PK)

fig.suptitle("Neural Signal Analysis Dashboard",
             fontsize=15, fontweight="bold", y=0.95)
fig.savefig("plots/s08_gridspec_dashboard.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  Done.")


# ── §9  HISTOGRAMS ────────────────────────────────────────────────────────────
print("§9  Histograms  →  plots/s09_histograms.png")

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for fi, fname in enumerate(fnames):
    ax = axes[fi // 3, fi % 3]
    for g in range(G):
        vals = data[g, :, :, :, fi].ravel()
        ax.hist(vals, bins=60, density=True, alpha=0.50,
                color=COLORS[g], label=gnames[g] if fi == 0 else "")
    ax.set_title(f"{fname} power distribution", fontsize=9)
    ax.set_xlabel("Power"); ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

axes[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("Feature Power Distributions — All Groups",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/s09_histograms.png", dpi=120)
plt.close(fig)
print("  Done.")


# ── §10  PUBLICATION-QUALITY STYLING ─────────────────────────────────────────
print("§10  Publication figure  →  plots/s10_publication_figure.png")

plt.rcParams.update({
    "font.family"          : "sans-serif",
    "axes.spines.top"      : False,
    "axes.spines.right"    : False,
    "axes.linewidth"       : 1.2,
    "xtick.major.width"    : 1.2,
    "ytick.major.width"    : 1.2,
    "axes.labelsize"       : 11,
    "xtick.labelsize"      : 9,
    "ytick.labelsize"      : 9,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Alpha vs Beta scatter (per subject, Control vs Patient)
for g in [0, 1]:
    sub_alpha = data[g, :, :, :, 2].mean(axis=(0, 2))   # (S,)
    sub_beta  = data[g, :, :, :, 3].mean(axis=(0, 2))   # (S,)
    ax1.scatter(sub_alpha, sub_beta, c=COLORS[g], alpha=0.72,
                s=65, label=gnames[g], edgecolors="white", linewidths=0.6)

ax1.set_xlabel("Mean Alpha Power"); ax1.set_ylabel("Mean Beta Power")
ax1.set_title("Alpha vs Beta (per subject)", fontweight="bold")
ax1.legend(frameon=False)

# Panel B: Patient − Control difference heatmap
diff = (data[1].mean(axis=(0, 1))        # (C, F)   Patient mean
      - data[0].mean(axis=(0, 1)))       # (C, F)   Control mean
vm   = np.abs(diff).max()
im   = ax2.imshow(diff, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
ax2.set_xticks(range(F)); ax2.set_xticklabels(fnames, rotation=40, ha="right")
ax2.set_yticks(range(0, C, 4))
ax2.set_yticklabels([chnames[i] for i in range(0, C, 4)])
ax2.set_title("Patient − Control  (channel × feature)", fontweight="bold")
fig.colorbar(im, ax=ax2, label="Δ Power", shrink=0.85)

fig.suptitle("Neural Biomarkers: Patient vs Control",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/s10_publication_figure.png", dpi=150, bbox_inches="tight")
plt.close(fig)
plt.rcParams.update(plt.rcParamsDefault)
print("  Done.")

print(f"\n{DIV}")
print("  DAY 2 STUDY GUIDE  COMPLETE  ✓")
print(f"  10 plots saved to  plots/")
print(f"{DIV}")
print("  Next → run  03_homework_10_problems.py")
