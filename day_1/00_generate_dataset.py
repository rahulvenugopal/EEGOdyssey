"""
Neural data generator

Shape : (G=3, T=3, S=30, C=32, F=6)

  axis 0  groups      3   Control | Patient | Treatment
  axis 1  timepoints  3   baseline | 90 days | 180 days
  axis 2  subjects    30  30 participants per group
  axis 3  channels    32  EEG electrode positions (standard 10–20)
  axis 4  features    6   delta | theta | alpha | beta | gamma | broadband
                             1–4Hz  4–8Hz  8–13Hz 13–30Hz 30–60Hz  1–60Hz


    data/neural_data.npy     float32 array
    data/metadata.pkl        dict: labels, channel regions, sampling rate

"""

import numpy as np
import pickle, os

# ── Seed & Dimensions ─────────────────────────────────────────────────────────
SEED  = 2026
RNG   = np.random.default_rng(SEED)
G, T, S, C, F = 3, 3, 30, 32, 6 # groups · time · subjects · chans · features

# ── Semantic Labels ───────────────────────────────────────────────────────────
GROUP_NAMES   = ["Control", "Patient", "Treatment"]
FEATURE_NAMES = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]

CHANNEL_NAMES = [
    # prefrontal (0–1)
    "Fp1", "Fp2",
    # frontal    (2–6)
    "F7",  "F3",  "Fz",  "F4",  "F8",
    # central    (7–11)
    "T3",  "C3",  "Cz",  "C4",  "T4",
    # parietal   (12–16)
    "T5",  "P3",  "Pz",  "P4",  "T6",
    # occipital  (17–19)
    "O1",  "Oz",  "O2",
    # extra      (20–31)
    "AF3", "AF4", "FC5", "FC1", "FC2", "FC6",
    "CP5", "CP1", "CP2", "CP6", "PO3", "PO4",
]

REGIONS = {
    "prefrontal" : [0, 1],
    "frontal"    : [2, 3, 4, 5, 6],
    "central"    : [7, 8, 9, 10, 11],
    "parietal"   : [12, 13, 14, 15, 16],
    "occipital"  : [17, 18, 19],
    "extra"      : list(range(20, 32)),
}


# ── Signal construction ───────────────────────────────────────────────────────
def _baseline(rng):
    """Gaussian prior shared by all groups."""
    return rng.normal(0.50, 0.15, (G, T, S, C, F)).astype("float32").clip(0)


def _group_effects(d, rng):
    """
    Patient (index 1):
        ↑ theta & beta in frontal channels  (hyperactivity marker)
        ↓ alpha globally                    (cognitive impairment proxy)
    Treatment (index 2):
        partial alpha recovery above Patient
        ↓ gamma relative to Patient

    Note: we index d[g] first (returning a view) so that the remaining
    fancy/scalar indices are CONTIGUOUS — avoiding the NumPy rule that
    moves non-contiguous advanced-index dims to the array front.
    """
    fron = REGIONS["frontal"]
    nf   = len(fron)

    # d[1] → view (T, S, C, F); then [:, :, fron, 1] has contiguous advanced dims
    d[1][:, :, fron, 1] += rng.normal(0.25, 0.05, (T, S, nf)).astype("float32")
    d[1][:, :, fron, 3] += rng.normal(0.20, 0.04, (T, S, nf)).astype("float32")
    d[1][:, :, :,    2] -= rng.normal(0.15, 0.03, (T, S, C )).astype("float32")

    d[2][:, :, :, 2]  = (d[1][:, :, :, 2]
                         + rng.normal(0.08, 0.02, (T, S, C)).astype("float32"))
    d[2][:, :, :, 4] -= rng.normal(0.10, 0.02, (T, S, C)).astype("float32")
    return d.clip(0)


def _temporal_trends(d, rng):
    """Slow alpha oscillation + broadband fatigue across time."""
    t    = np.linspace(0, 2 * np.pi, T, dtype="float32")
    fade = np.linspace(0, -0.05,     T, dtype="float32")

    d[:, :, :, :, 2] += 0.06 * np.sin(t)[None, :, None, None]   # alpha wave
    d[:, :, :, :, 5] += fade[None, :, None, None]                # broadband fade
    return d.clip(0)


def _subject_variability(d, rng):
    """Per-subject global offset — individual differences."""
    offsets = rng.normal(0, 0.06, (1, 1, S, 1, 1)).astype("float32")
    return (d + offsets).clip(0)


def _channel_topology(d, rng):
    """
    Spatial priors:
        occipital channels  → higher alpha (visual cortex)
        frontal channels    → higher delta (slow-wave activity)
    """
    occ = REGIONS["occipital"]
    fro = REGIONS["frontal"]
    d[:, :, :, occ, 2] += rng.normal(0.12, 0.02, (G, T, S, len(occ))).astype("float32")
    d[:, :, :, fro, 0] += rng.normal(0.08, 0.02, (G, T, S, len(fro))).astype("float32")
    return d.clip(0)


def _noise_and_artifacts(d, rng):
    """Gaussian measurement noise + rare high-amplitude artifacts (~1.5 %)."""
    d        += rng.normal(0, 0.02, d.shape).astype("float32")
    art_mask  = rng.random(d.shape) < 0.015
    d[art_mask] += rng.exponential(0.6, d.shape)[art_mask].astype("float32")
    return d.clip(0)


def generate_dataset():
    print(f"Generating 5-D neural dataset   shape=({G},{T},{S},{C},{F})")
    d = _baseline(RNG)
    d = _group_effects(d,        RNG)
    d = _temporal_trends(d,      RNG)
    d = _subject_variability(d,  RNG)
    d = _channel_topology(d,     RNG)
    d = _noise_and_artifacts(d,  RNG)

    metadata = {
        "shape"         : (G, T, S, C, F),
        "dim_names"     : ["groups","timepoints","subjects","channels","features"],
        "group_names"   : GROUP_NAMES,
        "feature_names" : FEATURE_NAMES,
        "channel_names" : CHANNEL_NAMES,
        "regions"       : REGIONS,
        "timepoints"   : 3,
        "seed"          : SEED,
    }
    return d, metadata


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    data, meta = generate_dataset()
    np.save("data/neural_data.npy", data)
    with open("data/metadata.pkl", "wb") as fh:
        pickle.dump(meta, fh)