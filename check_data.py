"""Recon script: inspect dataset structure. Not part of the analysis pipeline."""
from pathlib import Path
import mne

mne.set_log_level("ERROR")
root = Path.home() / "Downloads" / "ds003478"

for sub in ["sub-001", "sub-012", "sub-013", "sub-014"]:
    d = root / sub / "eeg"
    print(f"\n{sub}:")
    for f in sorted(d.iterdir()):
        print(f"   {f.name}   {f.stat().st_size / 1e6:.2f} MB")

f = sorted((root / "sub-012" / "eeg").glob("*.set"))[0]
raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
print(f"\n{f.name}: {len(raw.ch_names)} channels, "
      f"{raw.info['sfreq']:.0f} Hz, duration {raw.times[-1]:.1f} s")
