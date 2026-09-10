"""Global configuration: dataset switch, shared preprocessing/feature/experiment params."""
from pathlib import Path

# ---------- dataset switch ----------
# "ds003478" -> original experiment; "modma" -> external validation
DATASET = "ds003478"

# ---------- shared paths ----------
RESULTS_DIR = Path(__file__).parent / "results" / DATASET  # per-dataset results
FEATURES_DIR = RESULTS_DIR / "features_cache"  # ICA fitting is expensive; never recompute
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------- preprocessing (shared; dataset-specific bits live in datasets.py) ----------
RESAMPLE_SFREQ = 250        # all analyses target < 40 Hz, so 250 Hz is ample
HIGH_PASS = 1.0
LOW_PASS = 40.0
MIN_DURATION = 20.0         # reject recordings shorter than this (seconds)
ICA_CORR_THRESHOLD = 0.5    # |r| between IC time course and frontal proxy
ICA_MAX_COMPONENTS = 3      # cap on rejected ICs per subject

# ---------- features ----------
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}
CHANNELS_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}  # old -> new 10-20 names

# ---------- grouping thresholds (ds003478 only; MODMA uses clinical labels) ----------
BDI_DEPRESSION_MIN = 13
BDI_CONTROL_MAX = 7

# ---------- experiment ----------
METHODS = ["none", "infomax", "fastica", "picard"]
