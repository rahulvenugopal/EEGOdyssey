"""
Python Basics: Indexing, Slicing & Dicing

Dataset loaded from  neural_data.npy

Sections
--------
  1   Integer indexing        — single element access
  2   Range slicing           — start:stop along one axis
  3   Multi-axis slicing      — combining slices across all dims
  4   Step / stride slicing   — downsampling, reversal
  5   Boolean (mask) indexing — filter by value condition
  6   np.where & np.take      — conditional / gather utilities
  7   Reshape & ravel         — changing array shape
  8   Transpose               — reordering axes
  9  Views vs copies         — the critical memory gotcha

"""

import numpy as np, pickle

# Load 
data = np.load("data/neural_data.npy")
with open("data/metadata.pkl", "rb") as fh:
    meta = pickle.load(fh)

G, T, S, C, F = data.shape
gnames  = meta["group_names"]     # ["Control","Patient","Treatment"]
fnames  = meta["feature_names"]   # ["delta","theta","alpha","beta","gamma","broadband"]
chnames = meta["channel_names"]   # 32 channels
regions = meta["regions"]

#%% 1. Integer indexing
v = data[0, 0, 0, 0, 0]

v2 = data[1, 2, 14, 9, 2]

# Negative indices count from the end of each axis
data[-1, -1, -1, -1, -1]

#%% 2. Slicing | Start and Stop

# ':' means ALL elements along that axis
ts = data[1, :, :, 9, 2]  # (3,) Patient, all time, sub0, Cz, alpha

first_half = data[1, 0, 0, :C//2, 2]   # first half channels
secnd_half = data[1, 0, 0, C//2:, 2]   # last  half channels

epoch = data[0, 0:2, :, :, 2]    # (30, 30, 32) Controls, Baseline, all S, all C, alpha

#%% 3. Multi axis slicing

fron = regions["frontal"]

# Patient · baseline · all subjects · frontal channels only · alpha
fron_alpha_incorrect = data[1, 0, :, fron, 2]
fron_alpha_correct = data[1, 0, :, 2:7, 2]

# All groups · patients+treatment · first 10 subjects · all channels · theta+alpha
combo = data[:, 1:3, :10, :, 1:3]

# Compare occipital vs frontal alpha across all groups
occ_alpha = data[:, :, :, 17:20, 2].mean()
fro_alpha = data[:, :, :,  2:7,  2].mean()

#%% 4. Step wise slicing
# Take every alternate subject
ds4 = data[:, :, 0::2, :, :]

ds4_odd = data[:, :, 1::2, :, :]

# Reverse time axis
rev = data[0, ::-1, 0, 0, :]

#%% 5. Boolean mask based indexing

# What fraction of values could be artifacts (> 1.0)?
art_mask = data > 1.0
art_mask.sum()

# Per-subject mean alpha for Control group
ctrl_alpha  = data[0, :, :, :, 2]
subj_mean   = ctrl_alpha.mean(axis=(0,2)) # (S=30,)  — collapse time & channels
threshold   = subj_mean.mean() + subj_mean.std()
high_alpha  = np.where(subj_mean > threshold)[0]

# Find timepoints where Patient broadband spikes above mean + 2σ
pat_bb  = data[1, :, :, :, 5].mean(axis=(1,2))
thr2    = pat_bb.mean() + 2 * pat_bb.std()
spike_t = np.where(pat_bb > thr2)[0]
print(f"  Broadband spikes (Patient)  : {len(spike_t)} timepoints, first few={spike_t[:5]}")

# Compound condition: high alpha AND high beta simultaneously
hi_a = data[1, :, :, :, 2] > data[1,:,:,:,2].mean() + data[1,:,:,:,2].std()
hi_b = data[1, :, :, :, 3] > data[1,:,:,:,3].mean() + data[1,:,:,:,3].std()
print(f"  High-alpha & high-beta (Patient): {(hi_a & hi_b).sum():,} cells  "
      f"({100*(hi_a & hi_b).mean():.2f}%)")

#%% 6.  where, clip, find and take

# np.where(condition, value_if_true, value_if_false)
clean = np.where(data > 1.0, 0.0, data)      # zero out artifacts

# np.clip — related utility for clamping (not indexing but often paired)
clamped = np.clip(data, 0, 1.0)

# np.where with one arg returns indices (like find())
idxs = np.where(subj_mean > threshold)       # returns tuple of arrays
print(f"  np.where(condition) → indices: {idxs[0]}")

# np.take — gather along one axis with an index array
tba  = np.take(data, indices=[1, 2, 3], axis=4)

#%% 7. RESHAPE & RAVEL

# Some analysis pipelines want the data in specific order
# Bring subjects axis next to groups → (G, S, T, C, F)
d_swap  = data.transpose(0, 2, 1, 3, 4)

# Move features axis first: (G,T,S,C,F) → (F,G,T,S,C)
feat_first = data.transpose(4, 0, 1, 2, 3)

# Step 2: flatten groups × subjects into rows, rest into columns
X = d_swap.reshape(G * S, T * C * F)
print(f"  Design matrix (G*S, T*C*F)  : {X.shape}  — ready for PCA / sklearn")

# Flatten completely to 1-D
flat = data.ravel()

# Merge subjects & channels into one axis (common for connectivity analyses)
merged = data.reshape(G, T, S * C, F)
print(f"  Merge S×C                    : {merged.shape}")

#%% 8.  VIEWS vs COPIES

print(  "     Basic slicing    → VIEW   (shares memory  ⟹  mutations propagate)")
print(  "     Fancy indexing   → COPY  (independent  ⟹  safe to mutate)")
print(  "     Boolean indexing → COPY")

view        = data[0, :2, :, :, :]
fancy_copy  = data[[0, 1], :, :, :, :]
bool_copy   = data[data > 0.5]
transpose_v = data.transpose(4, 0, 1, 2, 3)

print(f"\n  Slice is a view? {np.shares_memory(view,       data)}")   # True
print(f"  Fancy index is a copy?   {not np.shares_memory(fancy_copy,  data)}")  # True
print(f"  Boolean index is a copy?   {not np.shares_memory(bool_copy,   data)}")  # True
print(f"  Transpose is a view?   {np.shares_memory(transpose_v, data)}")  # True

# Proof: mutation through a slice changes the original
backup = data[0, 0, 0, 0, 0]
view[0, 0, 0, 0] = 999.0          # writing into the view
print(f"\n  After view[0,0,0,0]=999 the original data[0,0,0,0,0]={data[0,0,0,0,0]:.1f}  ← changed!")
data[0, 0, 0, 0, 0] = backup       # restore

# Safe pattern: .copy() to protect the original
safe  = data[0].copy()
safe *= 99