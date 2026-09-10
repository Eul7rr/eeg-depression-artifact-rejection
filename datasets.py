"""Dataset adapters: everything dataset-specific lives behind this interface.

The engine (preprocessing / feature_extraction / evaluation / stats) is
dataset-agnostic. Switch datasets by flipping config.DATASET.

Each adapter provides:
    DATA_DIR    : path to the dataset root
    NOTCH_FREQ  : mains frequency at the recording site (Hz)
    load_groups()        -> {subject_id: 0/1/None}
    load_raw(subject_id) -> mne.io.BaseRaw
"""
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat

import config


class DS003478:
    """OpenNeuro ds003478: BIDS layout, .bdf/.set files, recorded in the US."""

    DATA_DIR = Path.home() / "Downloads" / "ds003478"
    NOTCH_FREQ = 60.0  # US mains

    @staticmethod
    def load_groups() -> dict:
        """Group by BDI thresholds; intermediate scores (7-13) excluded."""
        df = pd.read_csv(DS003478.DATA_DIR / "participants.tsv", sep="\t")
        bdi_cols = [c for c in df.columns if "bdi" in c.lower()]
        if not bdi_cols:
            raise KeyError(f"no BDI column in participants.tsv: {list(df.columns)}")
        bdi_col = bdi_cols[0]

        groups = {}
        for _, row in df.iterrows():
            sub = row["participant_id"]
            try:
                bdi = float(row[bdi_col])
            except (TypeError, ValueError):
                groups[sub] = None
                continue
            if bdi >= config.BDI_DEPRESSION_MIN:
                groups[sub] = 1
            elif bdi < config.BDI_CONTROL_MAX:
                groups[sub] = 0
            else:
                groups[sub] = None
        return groups

    @staticmethod
    def load_raw(subject_id: str) -> mne.io.BaseRaw:
        """Load one subject's EEG; if several files exist, keep the longest."""
        eeg_dir = DS003478.DATA_DIR / subject_id / "eeg"
        files = sorted(eeg_dir.glob("*.bdf")) + sorted(eeg_dir.glob("*.set"))
        if not files:
            raise FileNotFoundError(f"{subject_id}: no .bdf/.set under {eeg_dir}")

        # Probe headers only (preload=False); full loads would be slow
        best = None
        for f in files:
            try:
                if f.suffix.lower() == ".bdf":
                    probe = mne.io.read_raw_bdf(f, preload=False, verbose=False)
                else:
                    probe = mne.io.read_raw_eeglab(f, preload=False, verbose=False)
                duration = probe.times[-1] if probe.n_times > 0 else 0.0
                if best is None or duration > best[0]:
                    best = (duration, f)
            except Exception:
                continue  # unreadable file: skip it, don't abort the subject

        if best is None or best[0] < config.MIN_DURATION:
            raise ValueError(
                f"{subject_id}: no recording longer than {config.MIN_DURATION:.0f} s")

        _, f = best
        if f.suffix.lower() == ".bdf":
            return mne.io.read_raw_bdf(f, preload=True, verbose=False)
        return mne.io.read_raw_eeglab(f, preload=True, verbose=False)


class MODMA:
    """MODMA 128-ch resting state (Lanzhou): one .mat per subject + xlsx metadata.

    Expected layout after unzipping EEG_128channels_resting_lanzhou_2015.zip:
        <DATA_DIR>/subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx
        <DATA_DIR>/*.mat          (one per subject)
    The i-th sorted .mat file corresponds to row i of the xlsx (dataset docs).
    subject_id = .mat filename stem (e.g. "0201xx").
    """

    DATA_DIR = Path.home() / "Data" / "MODMA" / "EEG_128channels_resting_lanzhou_2015"
    NOTCH_FREQ = 50.0          # recorded in China (50 Hz mains)
    SFREQ = 250.0              # native sampling rate; no resampling needed
    INFO_XLSX = "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"

    @staticmethod
    def _subject_table() -> pd.DataFrame:
        df = pd.read_excel(MODMA.DATA_DIR / MODMA.INFO_XLSX)
        mats = sorted(MODMA.DATA_DIR.glob("*.mat"))
        if len(df) != len(mats):
            raise RuntimeError(
                f"xlsx has {len(df)} rows but found {len(mats)} .mat files; "
                "check the folder before trusting any label")
        df = df.copy()
        df["subject_id"] = [p.stem for p in mats]  # row i <-> i-th sorted .mat
        return df

    @staticmethod
    def load_groups() -> dict:
        """Labels from the metadata sheet: MDD -> 1, HC -> 0 (clinical diagnosis)."""
        df = MODMA._subject_table()
        type_col = [c for c in df.columns if str(c).lower() == "type"]
        if not type_col:
            raise KeyError(f"no 'type' column in info xlsx: {list(df.columns)}")
        groups = {}
        for _, row in df.iterrows():
            t = str(row[type_col[0]]).strip().upper()
            groups[row["subject_id"]] = 1 if "MDD" in t else 0
        return groups

    @staticmethod
    def load_raw(subject_id: str) -> mne.io.BaseRaw:
        path = MODMA.DATA_DIR / f"{subject_id}.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)

        # .mat layout is not standardized: pick the 2-D array with ~128 rows
        candidates = [v for k, v in mat.items() if not k.startswith("__")
                      and isinstance(v, np.ndarray) and v.ndim == 2]
        if not candidates:
            raise RuntimeError(f"{subject_id}: no 2-D array in {path.name}; "
                               f"keys={list(mat.keys())}")
        data = max(candidates, key=lambda a: a.size)
        if data.shape[0] not in (128, 129) and data.shape[1] in (128, 129):
            data = data.T  # channels must be rows
        if data.shape[0] not in (128, 129):
            raise RuntimeError(f"{subject_id}: unexpected data shape {data.shape}")

        # Units are not documented. EEG in microvolts reads ~1e1-1e2; in nV ~1e4-1e5.
        # Heuristic below; VERIFY once with check_modma.py and then hard-code.
        peak = np.nanmax(np.abs(data))
        scale = 1e-6 if peak < 1e3 else 1e-9
        print(f"      [units] {subject_id}: peak={peak:.3g}, assuming scale={scale:g} "
              "— report this printout before the full run")
        data = data * scale  # -> volts for MNE

        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        ch_names = list(montage.ch_names)            # E1..E128 + Cz
        if data.shape[0] == 128:
            ch_names.remove("Cz")                    # Cz was the online reference
        info = mne.create_info(ch_names, sfreq=MODMA.SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(montage, on_missing="ignore")
        return MODMA._rename_to_1020(raw)

    @staticmethod
    def _rename_to_1020(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Rename the EGI electrode nearest to each 10-20 target to the target name.

        Keeps all 128 channels (ICA needs full coverage); only renames the 19
        electrodes that downstream code looks up by 10-20 name (Fp1/Fp2 proxy,
        feature channels). Nearest-neighbour matching on standard positions.
        """
        egi_pos = raw.get_montage().get_positions()["ch_pos"]
        std_pos = mne.channels.make_standard_montage("standard_1005") \
            .get_positions()["ch_pos"]
        mapping, used = {}, set()
        for target in config.CHANNELS_1020:
            tp = std_pos[target]
            free = {ch: p for ch, p in egi_pos.items() if ch not in used}
            best = min(free, key=lambda ch: np.linalg.norm(free[ch] - tp))
            mapping[best] = target
            used.add(best)
        raw.rename_channels(mapping)
        print(f"      [montage] mapped {len(mapping)} EGI electrodes to 10-20 names")
        return raw


ADAPTER = {"ds003478": DS003478, "modma": MODMA}[config.DATASET]
