"""check_data.py —— 侦察数据集结构，不参与正式流程"""
from pathlib import Path
import mne

mne.set_log_level("ERROR")
root = Path.home() / "Downloads" / "ds003478"

# 1) 看几个被试的 eeg 文件夹里到底有什么文件、各多大
for sub in ["sub-001", "sub-012", "sub-013", "sub-014"]:
    d = root / sub / "eeg"
    print(f"\n{sub}:")
    for f in sorted(d.iterdir()):
        print(f"   {f.name}   {f.stat().st_size / 1e6:.2f} MB")

# 2) 打开 sub-012 的第一个 .set，看真实时长和通道数
f = sorted((root / "sub-012" / "eeg").glob("*.set"))[0]
raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
print(f"\n{f.name}: {len(raw.ch_names)} 通道, "
      f"{raw.info['sfreq']:.0f} Hz, 时长 {raw.times[-1]:.1f} 秒")