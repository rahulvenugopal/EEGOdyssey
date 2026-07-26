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
plt.close(fig)