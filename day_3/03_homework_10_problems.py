"""
================================================================================
GRADUATE HOMEWORK  |  10 Problems — Indexing, Slicing & Visualization
Days 1 & 2  |  5-D Neural Signal Dataset
================================================================================

Dataset : data/neural_data.npy   shape=(G=3, T=200, S=30, C=32, F=6)
          dims: groups · timepoints · subjects · channels · features

Groups    : 0=Control   1=Patient   2=Treatment
Features  : 0=delta  1=theta  2=alpha  3=beta  4=gamma  5=broadband
Channels  : 0–1=prefrontal  2–6=frontal  7–11=central
           12–16=parietal  17–19=occipital  20–31=extra

Plots saved to hw_plots/

Grading rubric (10 pts per problem = 100 total)
------------------------------------------------
  4 pts  Correct NumPy indexing / broadcasting  (no Python loops unless noted)
  3 pts  Numerically correct shapes & values
  3 pts  Code clarity + 3–5 sentence interpretation comment (marked # INTERP)

Prerequisite: run  00_generate_dataset.py  first.
Run         : python 03_homework_10_problems.py
================================================================================
"""

import numpy as np
import pickle, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

os.makedirs("hw_plots", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
data = np.load("data/neural_data.npy")
with open("data/metadata.pkl", "rb") as fh:
    meta = pickle.load(fh)

G, T, S, C, F = data.shape
gnames  = meta["group_names"]
fnames  = meta["feature_names"]
chnames = meta["channel_names"]
regions = meta["regions"]
hz      = meta["sampling_hz"]

t_axis = np.arange(T) / hz
COLORS = ["#2196F3", "#F44336", "#4CAF50"]   # blue · red · green
SEP    = "═" * 68


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 1  —  Epoch Extraction & Cohen's d Effect Size   [Day 1]
# ══════════════════════════════════════════════════════════════════════════════
"""
Background
──────────
Cohen's d quantifies effect size between two groups:
    d = (μ₁ − μ₂) / s_pooled         s_pooled = √((s₁² + s₂²) / 2)
Values > 0.8 are considered "large" in behavioural/clinical neuroscience.

Task
────
1. Extract the resting-state epoch: timepoints [0:50] for all groups.
2. Compute per-subject averages over those 50 timepoints → shape (G, S, C, F).
3. Compute Cohen's d between Control (g=0) and Patient (g=1) for every
   (channel, feature) pair using ONLY NumPy broadcasting — no scipy, no loops.
4. Return a (C, F) matrix of d values.
5. Print the (channel, feature) with the largest absolute effect and
   write a 3-5 sentence biological interpretation.

Expected output shape: (32, 6)
Hint: axis=0 in .mean() / .std() collapses the subjects dimension.
"""

print(f"\n{SEP}")
print("PROBLEM 1  —  Epoch Extraction & Cohen's d Effect Size")
print(SEP)


def cohens_d_epoch(data: np.ndarray) -> np.ndarray:
    """Return (C, F) Cohen's d matrix comparing Control vs Patient on t=[0:50]."""
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    epoch    = data[:, :50, :, :, :]           # (G, 50, S, C, F)
    subj_avg = epoch.mean(axis=1)              # (G, S, C, F)  collapse time

    ctrl, pat = subj_avg[0], subj_avg[1]       # each (S, C, F)

    mu1, mu2  = ctrl.mean(axis=0), pat.mean(axis=0)              # (C, F)
    s1,  s2   = ctrl.std(axis=0, ddof=1), pat.std(axis=0, ddof=1)
    s_pooled  = np.sqrt((s1**2 + s2**2) / 2 + 1e-12)            # (C, F) avoid /0

    return (mu1 - mu2) / s_pooled
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


result_p1 = cohens_d_epoch(data)
max_idx   = np.unravel_index(np.abs(result_p1).argmax(), result_p1.shape)
print(f"  Output shape  : {result_p1.shape}")
print(f"  Max |Cohen's d|: {result_p1[max_idx]:.4f}")
print(f"  Location      : channel={chnames[max_idx[0]]},  feature={fnames[max_idx[1]]}")
print(f"\n  # INTERP: ___ (fill in 3–5 sentences about which brain region and"
      f"\n  #              frequency band shows the strongest group difference)")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 2  —  Artifact Rejection Pipeline               [Day 1]
# ══════════════════════════════════════════════════════════════════════════════
"""
Background
──────────
Artifacts are transient high-amplitude events that corrupt neural data.
A common criterion: any cell exceeding μ + k·σ of broadband power is an artifact.

Task
────
1. For each group g compute a boolean mask of shape (T, S, C):
       mask[t,s,c] = True  iff  data[g,t,s,c, feat=5]  >  μ_g + k·σ_g
   where μ_g and σ_g are computed over the entire (T,S,C) volume for that group.
2. Zero ALL features at flagged cells.
3. Return:
   a. clean_data  (G, T, S, C, F)  — artifacts zeroed
   b. art_rate    (G,)             — fraction of (T,S,C) cells flagged per group
4. Report and interpret per-group artifact rates. No Python loops.
"""

print(f"\n{SEP}")
print("PROBLEM 2  —  Artifact Rejection Pipeline")
print(SEP)


def artifact_rejection(data: np.ndarray, k: float = 2.5):
    """Return (clean_data, artifact_rate) with broadband-based thresholding."""
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    bb   = data[:, :, :, :, 5]                              # (G, T, S, C)
    mu   = bb.mean(axis=(1, 2, 3), keepdims=True)           # (G, 1, 1, 1)
    sig  = bb.std(axis=(1, 2, 3),  keepdims=True)
    mask = bb > (mu + k * sig)                              # (G, T, S, C) bool

    rate  = mask.mean(axis=(1, 2, 3))                       # (G,)
    clean = data.copy()
    clean[mask] = 0.0                                       # zero ALL features
    return clean, rate
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


clean_data, art_rate = artifact_rejection(data)
print(f"  Artifact rates per group:")
for g, nm in enumerate(gnames):
    print(f"    {nm:12s}: {100*art_rate[g]:.3f} %")
print(f"\n  # INTERP: ___ (why might one group have more artifacts? relate to"
      f"\n  #              the data-generation model for that group)")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 3  —  Baseline Correction & Overlapping Epochs  [Day 1]
# ══════════════════════════════════════════════════════════════════════════════
"""
Background
──────────
Baseline correction removes pre-stimulus mean; overlapping epochs with step < W
increase statistical power without inflating sample count independently.

Task
────
Part A — Baseline Correction (no loops):
  1. Compute baseline mean over t=[0:20] for every (g, s, c, f) cell
     → shape (G, 1, S, C, F)  via keepdims=True.
  2. Subtract from the full 200-timepoint array via broadcasting.
  3. Assert: corrected[:, :20, :, :, :].mean()  < 1e-5.

Part B — Overlapping Epochs:
  4. Window W=40, step=20 (50 % overlap).  n_epochs = (T−W)//step + 1.
  5. Build epoch array shape (n_epochs, G, W, S, C, F) with array indexing.
  6. Per-epoch mean power for each group → (n_epochs, G, F).
  7. Report the epoch index and group with the highest mean broadband power.
"""

print(f"\n{SEP}")
print("PROBLEM 3  —  Baseline Correction & Overlapping Epochs")
print(SEP)


def baseline_correct(data: np.ndarray, baseline_end: int = 20) -> np.ndarray:
    """Subtract mean of [0:baseline_end] from every (g,s,c,f) cell."""
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    bl = data[:, :baseline_end, :, :, :].mean(axis=1, keepdims=True)  # (G,1,S,C,F)
    return data - bl
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


def make_epochs(data: np.ndarray, W: int = 40, step: int = 20) -> np.ndarray:
    """Return epoch array shape (n_epochs, G, W, S, C, F)."""
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    starts = np.arange(0, T - W + 1, step)
    epochs = np.stack([data[:, s:s+W, :, :, :] for s in starts], axis=0)
    return epochs
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


corrected  = baseline_correct(data)
bl_check   = corrected[:, :20, :, :, :].mean()
print(f"  Baseline window mean after correction : {bl_check:.2e}  (target < 1e-5)")

epochs     = make_epochs(corrected)
print(f"  Epoch array shape                     : {epochs.shape}")

# (n_epochs, G, W, S, C, F) → mean over W, S, C
ep_power   = epochs.mean(axis=(2, 3, 4))          # (n_epochs, G, F)
max_ep, max_g = np.unravel_index(ep_power[:, :, 5].argmax(), ep_power[:, :, 5].shape)
print(f"  Peak broadband epoch={max_ep}, group={gnames[max_g]}, "
      f"power={ep_power[max_ep, max_g, 5]:.5f}")
print(f"\n  # INTERP: ___ (what does baseline correction achieve statistically?"
      f"\n  #              how does overlap affect epoch independence?)")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 4  —  Leave-One-Subject-Out Cross-Validation    [Day 1]
# ══════════════════════════════════════════════════════════════════════════════
"""
Background
──────────
LOSO-CV evaluates how well a group average generalises to an unseen individual.
Large distance ⟹ that subject is an outlier in feature space.

Task
────
For every (g, k) pair  (k = 0 … S−1):
  1. Mean feature vector across all OTHER subjects (mean over T and C): (F,)
  2. Mean feature vector for the test subject: (F,)
  3. Euclidean distance  d_k = ‖ μ_train − μ_test ‖₂
  → Return a (G, S) distance matrix.

No explicit loops over k.  Use the cumulative-sum trick:
    train_mean[k] = (sum_all − x_k) / (S−1)
where sum_all = per_subj.sum(axis=subject_axis, keepdims=True).

Identify the most atypical subject per group (largest distance).
"""

print(f"\n{SEP}")
print("PROBLEM 4  —  Leave-One-Subject-Out Cross-Validation")
print(SEP)


def loso_distances(data: np.ndarray) -> np.ndarray:
    """Return (G, S) matrix of leave-one-subject-out Euclidean distances."""
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    # Mean over T (axis 1) and C (axis 3) → (G, S, F)
    per_subj   = data.mean(axis=(1, 3))                             # (G, S, F)
    grand_sum  = per_subj.sum(axis=1, keepdims=True)               # (G, 1, F)
    train_mean = (grand_sum - per_subj) / (S - 1)                  # (G, S, F)

    diff = train_mean - per_subj                                    # (G, S, F)
    return np.linalg.norm(diff, axis=2)                            # (G, S)
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


loso_dist = loso_distances(data)
print(f"  Distance matrix shape : {loso_dist.shape}")
for g in range(G):
    atypical = loso_dist[g].argmax()
    print(f"  {gnames[g]:12s}: most atypical subject={atypical:2d}  "
          f"dist={loso_dist[g, atypical]:.5f}  "
          f"(group mean dist={loso_dist[g].mean():.5f})")
print(f"\n  # INTERP: ___ (what could make a subject atypical? how does"
      f"\n  #              LOSO differ from random train/test split?)")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 5  —  PCA-Ready Design Matrix & Variance Analysis [Day 1]
# ══════════════════════════════════════════════════════════════════════════════
"""
Background
──────────
PCA requires a 2-D, mean-centred matrix.  Here each row is one subject-group
observation; columns are all temporal × channel × feature combinations.

Task
────
1. Transpose to (G, S, T, C, F)  then reshape → (G*S, T*C*F).
2. Mean-centre every column (subtract column means; broadcasting, no loops).
3. Assert: |column means|_max  < 1e-6.
4. Compute the total variance explained (trace of covariance):
       trace(Σ) = ‖X‖_F² / (n−1)   when X is already mean-centred.
5. Report the condition number of the first 100 columns (use np.linalg.cond).
6. Compute and report the rank of X via np.linalg.matrix_rank.
"""

print(f"\n{SEP}")
print("PROBLEM 5  —  PCA-Ready Design Matrix & Variance Analysis")
print(SEP)


def build_design_matrix(data: np.ndarray):
    """Return (X_centred, cov_trace, cond_num, rank)."""
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    d_swap = data.transpose(0, 2, 1, 3, 4)          # (G, S, T, C, F)
    X      = d_swap.reshape(G * S, T * C * F).astype("float64")
    X     -= X.mean(axis=0, keepdims=True)           # column mean-centering

    assert np.abs(X.mean(axis=0)).max() < 1e-6, "Mean-centering failed"

    cov_trace = (X ** 2).sum() / (G * S - 1)        # Frobenius shortcut
    cond_num  = np.linalg.cond(X[:, :100])          # first 100 cols
    rank      = np.linalg.matrix_rank(X[:, :100])
    return X, cov_trace, cond_num, rank
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


X_design, trace, cond, rank = build_design_matrix(data)
print(f"  Design matrix shape       : {X_design.shape}")
print(f"  Max |column mean|         : {np.abs(X_design.mean(axis=0)).max():.2e}  (<1e-6 ✓)")
print(f"  Covariance trace (total Δ): {trace:.5f}")
print(f"  Condition number (cols:100): {cond:.2f}")
print(f"  Rank (cols:100)            : {rank}")
print(f"\n  # INTERP: ___ (what does the condition number imply about numerical"
      f"\n  #              stability? why might we need dimensionality reduction?)")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 6  —  Alpha Time-Course with Bootstrapped CI   [Day 2]
# ══════════════════════════════════════════════════════════════════════════════
"""
Task
────
1. Compute mean alpha time-course per group: mean over S and C → (G, T).
2. Bootstrap 95 % CI (500 iterations, resample S subjects with replacement).
3. Plot: mean line + shaded CI, all 3 groups on one axes.
4. Add vertical dashed line at t = 1.0 s (mid-session).
5. Mark timepoints where |z(Patient − Control)| > 2 with red × at the
   bottom margin of the axes.
6. Save to  hw_plots/hw6_alpha_bootstrap.png  at dpi=150.

No sklearn.  Use only NumPy RNG (np.random.default_rng).
"""

print(f"\n{SEP}")
print("PROBLEM 6  —  Alpha Time-Course with Bootstrapped CI")
print(SEP)


def bootstrap_ci(data: np.ndarray, n_boot: int = 500, seed: int = 0):
    """Return (means, ci_lo, ci_hi) each shape (G, T)."""
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    rng_b = np.random.default_rng(seed)
    alpha = data[:, :, :, :, 2]                    # (G, T, S, C)
    means = alpha.mean(axis=(2, 3))                # (G, T)

    boots = np.zeros((G, n_boot, T))
    for b in range(n_boot):
        idx = rng_b.integers(0, S, size=S)         # resample subjects
        boots[:, b, :] = alpha[:, :, idx, :].mean(axis=(2, 3))

    ci_lo = np.percentile(boots, 2.5,  axis=1)    # (G, T)
    ci_hi = np.percentile(boots, 97.5, axis=1)
    return means, ci_lo, ci_hi
    # ── END YOUR CODE ─────────────────────────────────────────────────────────


means_p6, ci_lo, ci_hi = bootstrap_ci(data, n_boot=300)   # 300 for speed

diff_gc = means_p6[1] - means_p6[0]
z_score = (diff_gc - diff_gc.mean()) / (diff_gc.std() + 1e-9)
sig_t   = np.where(np.abs(z_score) > 2)[0]

fig, ax = plt.subplots(figsize=(11, 4.5))
for g in range(G):
    ax.plot(t_axis, means_p6[g], color=COLORS[g], lw=2, label=gnames[g])
    ax.fill_between(t_axis, ci_lo[g], ci_hi[g], color=COLORS[g], alpha=0.18)

y_bot = ci_lo.min() - 0.015
ax.scatter(t_axis[sig_t], np.full(len(sig_t), y_bot),
           color="red", marker="x", s=35, zorder=5, label="|z|>2 (Pat vs Ctrl)")
ax.axvline(1.0, ls="--", color="k", alpha=0.45, label="mid-session")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Alpha Power")
ax.set_title("HW6: Alpha Time-Course ± 95% Bootstrap CI", fontweight="bold")
ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("hw_plots/hw6_alpha_bootstrap.png", dpi=150)
plt.close(fig)
print(f"  Saved hw_plots/hw6_alpha_bootstrap.png")
print(f"  {len(sig_t)} timepoints with |z|>2  (Patient vs Control)")
print(f"\n  # INTERP: ___ (what time windows show reliable group differences?"
      f"\n  #              what is the advantage of bootstrap over parametric CI?)")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 7  —  3×6 Feature Grid with Artifact Overlay   [Day 2]
# ══════════════════════════════════════════════════════════════════════════════
"""
Task
────
1. Create a 3×6 subplot grid (rows = groups, cols = features), sharex=True.
2. Each cell:
   a. Plot mean ± std time-series (mean & std over S and C).
   b. Shade time regions where ANY (s,c) cell in that group is flagged as
      an artifact (broadband > μ_g + 2.5σ_g) in translucent red.
3. Consistent y-axis scale within each feature column (sharey along column).
4. Row labels = group names; column labels = feature names.
5. Save to  hw_plots/hw7_feature_grid.png  dpi=150.
"""

print(f"\n{SEP}")
print("PROBLEM 7  —  3×6 Feature Grid with Artifact Overlay")
print(SEP)


# ── YOUR CODE HERE ────────────────────────────────────────────────────────────
# Precompute artifact time-mask per group: (G, T)
bb       = data[:, :, :, :, 5]                      # (G, T, S, C)
bb_mu    = bb.mean(axis=(1, 2, 3), keepdims=True)   # (G, 1, 1, 1)
bb_sig   = bb.std(axis=(1, 2, 3),  keepdims=True)
art_time = (bb > (bb_mu + 2.5 * bb_sig)).any(axis=(2, 3))   # (G, T) bool

fig, axes = plt.subplots(3, 6, figsize=(22, 10), sharex=True)
for g in range(G):
    for fi in range(F):
        ax  = axes[g, fi]
        sig = data[g, :, :, :, fi]                  # (T, S, C)
        mu  = sig.mean(axis=(1, 2))                 # (T,)
        std = sig.mean(axis=2).std(axis=1)          # (T,)  std across subjects

        ax.plot(t_axis, mu, color=COLORS[g], lw=1.2)
        ax.fill_between(t_axis, mu - std, mu + std, color=COLORS[g], alpha=0.20)

        # Shade artifact timepoints in red
        y0, y1 = (mu - std).min() * 0.9, (mu + std).max() * 1.1
        ax.fill_between(t_axis, y0, y1,
                        where=art_time[g], color="red", alpha=0.18, zorder=0)
        ax.set_ylim(y0, y1)

        if g == 0: ax.set_title(fnames[fi], fontsize=8, fontweight="bold")
        if fi == 0: ax.set_ylabel(gnames[g], fontsize=8, fontweight="bold")
        ax.grid(alpha=0.25); ax.tick_params(labelsize=6)

fig.suptitle("HW7: Feature × Group Grid  (mean±std, red=artifact region)",
             fontsize=11, fontweight="bold")
fig.text(0.5, 0.01, "Time (s)", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.025, 1, 0.97])
fig.savefig("hw_plots/hw7_feature_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved hw_plots/hw7_feature_grid.png")
print(f"\n  # INTERP: ___ (which features show largest std? which group has"
      f"\n  #              the most artifact timepoints and why?)")
# ── END YOUR CODE ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 8  —  Channel-Space Heatmap (Patient − Control)  [Day 2]
# ══════════════════════════════════════════════════════════════════════════════
"""
Task
────
1. Compute per-group channel×feature mean: data[g].mean(axis=(0,1)) → (C,F).
2. Compute difference: diff = grp[Patient] − grp[Control].
3. Plot 1×3 figure:
   • Panel 1: Control heatmap (cmap='hot')
   • Panel 2: Patient heatmap (cmap='hot', same scale as Panel 1)
   • Panel 3: Patient − Control heatmap (cmap='RdBu_r', centred at 0)
4. All 32 channel names on y-axis (fontsize ≤ 6).
5. Feature names on x-axis, rotated 40°.
6. Circle the (channel, feature) with max |diff| using a lime Rectangle patch.
7. Save to  hw_plots/hw8_channel_heatmaps.png  dpi=150.
"""

print(f"\n{SEP}")
print("PROBLEM 8  —  Channel-Space Heatmaps")
print(SEP)


# ── YOUR CODE HERE ────────────────────────────────────────────────────────────
grp_cf = data.mean(axis=(1, 2))                   # (G, C, F)  mean over T & S
diff8  = grp_cf[1] - grp_cf[0]                   # (C, F)  Patient − Control
vm8    = np.abs(diff8).max()
v01    = grp_cf[:2].max()

fig, axes = plt.subplots(1, 3, figsize=(17, 9))
panels = [
    (grp_cf[0], "Control",           "hot",    0,    v01),
    (grp_cf[1], "Patient",           "hot",    0,    v01),
    (diff8,     "Patient − Control", "RdBu_r", -vm8, vm8),
]
for col, (mat, title, cmap, vmin, vmax) in enumerate(panels):
    ax = axes[col]
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(F)); ax.set_xticklabels(fnames, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(C)); ax.set_yticklabels(chnames, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.60, pad=0.02)

# Outline max-|diff| cell in lime
mr, mc = np.unravel_index(np.abs(diff8).argmax(), diff8.shape)
axes[2].add_patch(Rectangle((mc-0.5, mr-0.5), 1, 1,
                              edgecolor="lime", facecolor="none", lw=2.5))

fig.suptitle("HW8: Channel × Feature Power Heatmaps", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("hw_plots/hw8_channel_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved hw_plots/hw8_channel_heatmaps.png")
print(f"  Max |diff|: channel={chnames[mr]}, feature={fnames[mc]}, Δ={diff8[mr,mc]:.5f}")
print(f"\n  # INTERP: ___ (which brain region and frequency band differs most?"
      f"\n  #              is this consistent with the data-generation model?)")
# ── END YOUR CODE ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 9  —  Feature Correlation Matrices & Matrix Distance  [Day 2]
# ══════════════════════════════════════════════════════════════════════════════
"""
Background
──────────
The Frobenius norm ‖A − B‖_F measures distance between two matrices.
It tells us how much the correlation structure of one group differs from another.

Task
────
1. For each group: reshape data[g] → (T*S*C, F) and compute np.corrcoef
   → (F, F).  Stack into corr_mats of shape (G, F, F).
2. Plot 1×3 annotated heatmaps (values to 2 dp).
3. Bold text where |r| > 0.50.  Red Rectangle border where |r| > 0.70 and
   i ≠ j (off-diagonal only).
4. Compute ‖Patient − Control‖_F  and  ‖Treatment − Control‖_F.
5. State which group's correlation structure is closer to Control.
6. Save to  hw_plots/hw9_correlations.png  dpi=150.
"""

print(f"\n{SEP}")
print("PROBLEM 9  —  Feature Correlation Matrices & Frobenius Distance")
print(SEP)


# ── YOUR CODE HERE ────────────────────────────────────────────────────────────
corr_mats = np.zeros((G, F, F))
for g in range(G):
    corr_mats[g] = np.corrcoef(data[g].reshape(-1, F).T)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for g in range(G):
    ax   = axes[g]
    corr = corr_mats[g]
    im   = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(F)); ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(F)); ax.set_yticklabels(fnames, fontsize=8)
    ax.set_title(gnames[g], fontweight="bold")

    for i in range(F):
        for j in range(F):
            r   = corr[i, j]
            fw  = "bold"   if abs(r) > 0.50 else "normal"
            col = "white"  if abs(r) > 0.60 else "black"
            ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight=fw, color=col)
            if abs(r) > 0.70 and i != j:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1,
                                       edgecolor="red", facecolor="none", lw=1.8))
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

fig.suptitle("HW9: Feature Correlation Matrices", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("hw_plots/hw9_correlations.png", dpi=150, bbox_inches="tight")
plt.close(fig)

frob_pat = np.linalg.norm(corr_mats[1] - corr_mats[0], "fro")
frob_trt = np.linalg.norm(corr_mats[2] - corr_mats[0], "fro")
closer   = "Treatment" if frob_trt < frob_pat else "Patient"
print(f"  ‖Patient − Control‖_F    = {frob_pat:.5f}")
print(f"  ‖Treatment − Control‖_F  = {frob_trt:.5f}")
print(f"  Closer to Control        : {closer}")
print(f"\n  # INTERP: ___ (what do the Frobenius distances tell us about"
      f"\n  #              whether treatment normalises the correlation structure?)")
# ── END YOUR CODE ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 10  —  Publication-Quality Multi-Panel Dashboard  [Day 2]
# ══════════════════════════════════════════════════════════════════════════════
"""
Task
────
Combine results from HW 6–9 into a single publication-quality figure
using GridSpec:

    Row 0  (full width)   Alpha time-courses ± bootstrap CI  (from HW6)
    Row 1  (3 panels)     [B] Patient−Control heatmap (HW8)
                          [C] Feature power bar chart (from study §6)
                          [D] LOSO atypicality boxplot (HW4)
    Row 2  (full width)   Feature correlation matrices for all 3 groups (HW9)

Requirements:
  • Figure: 18 × 12 inches, dpi=300
  • Panel labels A–E in upper-left corner of each panel (bold, 12 pt)
  • Consistent colour scheme (COLORS) throughout
  • Overall title + subtitle (data provenance line)
  • Save to  hw_plots/hw10_dashboard.png  dpi=300

You are graded on design choices (no clutter, readable fonts, informative
titles, no overlapping labels) as much as on technical correctness.
"""

print(f"\n{SEP}")
print("PROBLEM 10  —  Publication-Quality Multi-Panel Dashboard")
print(SEP)


# Pre-compute bar chart stats (mean ± SEM per group per feature)
grp_mean = data.mean(axis=(1, 2, 3))        # (G, F)
grp_sem  = np.zeros((G, F))
for g in range(G):
    per_subj     = data[g].mean(axis=(0, 2))   # (S, F)  mean over T & C
    grp_sem[g]   = per_subj.std(axis=0) / np.sqrt(S)

# ── YOUR CODE HERE ────────────────────────────────────────────────────────────
PK = dict(fontsize=12, fontweight="bold", va="top", ha="left")
x_b, w = np.arange(F), 0.25

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.52, wspace=0.36,
                        left=0.06, right=0.97, top=0.91, bottom=0.06)

# ── Panel A: alpha time-courses (full width) ──────────────────────────────
ax_A = fig.add_subplot(gs[0, :])
for g in range(G):
    ax_A.plot(t_axis, means_p6[g], color=COLORS[g], lw=2, label=gnames[g])
    ax_A.fill_between(t_axis, ci_lo[g], ci_hi[g], color=COLORS[g], alpha=0.15)
ax_A.axvline(1.0, ls="--", color="k", alpha=0.40)
ax_A.scatter(t_axis[sig_t], np.full(len(sig_t), ci_lo.min() - 0.01),
             color="red", marker="x", s=28, zorder=5)
ax_A.set_xlabel("Time (s)"); ax_A.set_ylabel("Alpha Power")
ax_A.set_title("Alpha Power Time-Course ± 95% Bootstrap CI", fontsize=10)
ax_A.legend(ncol=4, fontsize=8, frameon=False); ax_A.grid(alpha=0.28)
ax_A.text(0.005, 0.96, "A", transform=ax_A.transAxes, **PK)

# ── Panel B: Patient − Control heatmap ───────────────────────────────────
ax_B = fig.add_subplot(gs[1, 0])
im_B = ax_B.imshow(diff8, cmap="RdBu_r", vmin=-vm8, vmax=vm8, aspect="auto")
ax_B.set_title("Patient − Control\n(channel × feature)", fontsize=9)
ax_B.set_xticks(range(F)); ax_B.set_xticklabels(fnames, rotation=40, ha="right", fontsize=6)
ax_B.set_yticks(range(0, C, 4))
ax_B.set_yticklabels([chnames[i] for i in range(0, C, 4)], fontsize=6)
fig.colorbar(im_B, ax=ax_B, shrink=0.72, pad=0.03, label="ΔPower")
ax_B.text(0.02, 0.98, "B", transform=ax_B.transAxes, **PK)

# ── Panel C: feature bar chart ────────────────────────────────────────────
ax_C = fig.add_subplot(gs[1, 1])
for g in range(G):
    ax_C.bar(x_b + g*w, grp_mean[g], w,
             yerr=grp_sem[g], capsize=2.5,
             color=COLORS[g], alpha=0.85, label=gnames[g])
ax_C.set_xticks(x_b + w); ax_C.set_xticklabels(fnames, rotation=38, ha="right", fontsize=7)
ax_C.set_ylabel("Mean Power", fontsize=8); ax_C.set_title("Feature Power ±SEM", fontsize=9)
ax_C.legend(fontsize=7, frameon=False, ncol=1); ax_C.grid(axis="y", alpha=0.28)
ax_C.text(0.02, 0.98, "C", transform=ax_C.transAxes, **PK)

# ── Panel D: LOSO atypicality boxplot ─────────────────────────────────────
ax_D = fig.add_subplot(gs[1, 2])
bp   = ax_D.boxplot([loso_dist[g] for g in range(G)],
                    patch_artist=True, widths=0.50, notch=False)
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col); patch.set_alpha(0.72)
for element in ("whiskers","caps","fliers","medians"):
    plt.setp(bp[element], color="k")
ax_D.set_xticklabels(gnames, fontsize=8)
ax_D.set_ylabel("LOSO Distance", fontsize=8); ax_D.set_title("Subject Atypicality", fontsize=9)
ax_D.grid(axis="y", alpha=0.28)
ax_D.text(0.02, 0.98, "D", transform=ax_D.transAxes, **PK)

# ── Panel E: correlation matrices (full width) ────────────────────────────
ax_E = [fig.add_subplot(gs[2, col]) for col in range(3)]
for g in range(G):
    im_e = ax_E[g].imshow(corr_mats[g], cmap="RdBu_r", vmin=-1, vmax=1)
    ax_E[g].set_title(f"{gnames[g]} Correlations", fontsize=9)
    ax_E[g].set_xticks(range(F)); ax_E[g].set_xticklabels(fnames, rotation=40, ha="right", fontsize=7)
    ax_E[g].set_yticks(range(F)); ax_E[g].set_yticklabels(fnames, fontsize=7)
    for i in range(F):
        for j in range(F):
            r   = corr_mats[g, i, j]
            col = "w" if abs(r) > 0.6 else "k"
            ax_E[g].text(j, i, f"{r:.2f}", ha="center", va="center",
                         fontsize=5.5, color=col)
ax_E[0].text(0.02, 0.98, "E", transform=ax_E[0].transAxes, **PK)
fig.colorbar(im_e, ax=ax_E, shrink=0.55, label="Pearson r", pad=0.02)

# Title + subtitle
fig.suptitle("Neural Signal Analysis Dashboard",
             fontsize=15, fontweight="bold", y=0.95)
fig.text(0.5, 0.921,
         (f"N={S} subjects/group · {T} timepoints @ {hz} Hz · "
          f"{C} channels · {F} spectral features · 3 groups"),
         ha="center", fontsize=8, color="#555555")

fig.savefig("hw_plots/hw10_dashboard.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved hw_plots/hw10_dashboard.png  (300 DPI, publication ready)")
print(f"\n  # INTERP: ___ (summarise the key findings visible across all panels;"
      f"\n  #              what story do Panels A–E tell collectively?)")
# ── END YOUR CODE ─────────────────────────────────────────────────────────────


# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ALL 10 PROBLEMS COMPLETE  ✓")
print(f"  Plots saved to  hw_plots/")
print(f"{SEP}")
print("""
  PROBLEM SUMMARY
  ═══════════════════════════════════════════════════════════════════
  Day 1 — Indexing & Slicing
    P1   Cohen's d via broadcasting across a 5-D epoch array
    P2   Artifact rejection with boolean masks (no loops)
    P3   Baseline correction + overlapping-epoch construction
    P4   Leave-one-subject-out CV via cumulative-sum indexing trick
    P5   PCA design matrix: transpose → reshape → mean-centre

  Day 2 — Visualization
    P6   Bootstrap CI on time-courses, z-score significance markers
    P7   3×6 subplot grid with shaded artifact regions
    P8   Channel×feature heatmaps with diverging colormap
    P9   Correlation matrices, Frobenius distance between groups
    P10  GridSpec publication dashboard integrating P6–P9
  ═══════════════════════════════════════════════════════════════════
""")
