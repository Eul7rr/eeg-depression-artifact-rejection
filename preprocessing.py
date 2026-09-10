"""Preprocessing: load -> pick EEG -> resample -> bandpass -> notch -> [ICA] -> avg ref.

Dataset-specific loading lives in datasets.py; this module is the shared engine.
"""
import numpy as np
import mne
import config
from datasets import ADAPTER

mne.set_log_level("WARNING")


def apply_ica(raw: mne.io.BaseRaw, method: str) -> mne.io.BaseRaw:
    """Fit ICA and reject components strongly correlated with the frontal proxy."""
    ica = mne.preprocessing.ICA(
        n_components=0.99,
        method=method,
        fit_params={"extended": True} if method == "infomax" else None,
        max_iter=1000,       # Picard can fail to converge at the default max_iter
        random_state=42,     # fixed seed: bit-for-bit reproducibility
    )
    ica.fit(raw)

    # Fp1/Fp2 serve as the ocular proxy; |r| because IC polarity is arbitrary.
    # (ds003478: HEOG/VEOG labels unreliable; MODMA: no dedicated EOG channels.)
    lower_to_name = {ch.lower(): ch for ch in raw.ch_names}
    frontal = [lower_to_name[c] for c in ("fp1", "fp2") if c in lower_to_name]
    if not frontal:
        raise RuntimeError(f"Fp1/Fp2 not found; channels: {raw.ch_names}")
    sources = ica.get_sources(raw).get_data()
    ref = raw.get_data(picks=frontal).mean(axis=0)
    corrs = np.array([abs(np.corrcoef(s, ref)[0, 1]) for s in sources])

    bads = []
    for idx in np.argsort(corrs)[::-1]:
        if corrs[idx] >= config.ICA_CORR_THRESHOLD and len(bads) < config.ICA_MAX_COMPONENTS:
            bads.append(int(idx))
    ica.exclude = bads
    print(f"      ICA({method}) rejected {len(bads)} component(s)")
    return ica.apply(raw.copy())


def preprocess_subject(subject_id: str, method: str = "none") -> mne.io.BaseRaw:
    """Full preprocessing. method in {"none", "infomax", "fastica", "picard"}."""
    raw = ADAPTER.load_raw(subject_id)         # dataset-specific loading
    raw.pick_types(eeg=True)
    raw.resample(config.RESAMPLE_SFREQ)        # no-op for MODMA (already 250 Hz)
    raw.filter(config.HIGH_PASS, config.LOW_PASS)
    raw.notch_filter(ADAPTER.NOTCH_FREQ)       # 60 Hz US / 50 Hz China
    if method != "none":
        # ICA before average re-referencing: the average reference drops one
        # rank, and ICA expects full-rank data.
        raw = apply_ica(raw, method)
    raw.set_eeg_reference("average")
    return raw
