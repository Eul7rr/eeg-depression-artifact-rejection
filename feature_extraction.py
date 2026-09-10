"""Feature extraction: 19 channels x 4 bands = 76-dim log-band-power vector."""
import numpy as np
from scipy.signal import welch

import config

try:
    from numpy import trapezoid as _trapz      # NumPy >= 2.0
except ImportError:
    from numpy import trapz as _trapz          # NumPy < 2.0


def _resolve_picks(raw):
    """Map config.CHANNELS_1020 to channel indices in raw.

    Case-insensitive: this dataset names channels in all caps (FP1/FZ/CZ).
    Alias-aware: old 10-20 names (T3/T4/T5/T6) map to new ones via config.
    Unresolved channels become None; the caller decides how to fail.
    """
    lower_to_name = {ch.lower(): ch for ch in raw.ch_names}
    resolved = []
    for target in config.CHANNELS_1020:
        aliases = config.CHANNEL_ALIASES.get(target, [])
        if isinstance(aliases, str):
            aliases = [aliases]
        found = None
        for cand in [target] + list(aliases):
            real = lower_to_name.get(cand.lower())
            if real is not None:
                found = raw.ch_names.index(real)
                break
        resolved.append(found)
    return resolved


def extract_features(raw):
    """Preprocessed raw -> 76-dim feature vector (log10 band power)."""
    resolved = _resolve_picks(raw)

    # Fail loudly: a silently misaligned feature vector is worse than no vector
    missing = [ch for ch, idx in zip(config.CHANNELS_1020, resolved) if idx is None]
    if missing:
        raise RuntimeError(
            f"channels not found: {missing}; available: {raw.ch_names}"
        )

    sfreq = raw.info["sfreq"]
    n_per_seg = min(int(sfreq * 4), raw.n_times)  # 4-s Welch segments
    n_overlap = n_per_seg // 2

    feats = []
    for ch_idx in resolved:
        data = raw.get_data(picks=[ch_idx])[0]
        freqs, psd = welch(data, fs=sfreq, window="hamming",
                           nperseg=n_per_seg, noverlap=n_overlap)
        for band, (fmin, fmax) in config.BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            power = _trapz(psd[mask], freqs[mask])
            feats.append(np.log10(power))
    return np.array(feats)
