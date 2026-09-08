"""
preprocessing.py
流程：读取 → 选 EEG 通道 → 降采样 → 带通 → 陷波 → (可选) ICA → 平均重参考
融合的两个教训：
1) 一个被试可能有多个文件，自动选"时长最长"的那份；
2) ICA 放在平均重参考之前（重参考会让数据缺一秩，ICA 在满秩数据上更稳）。
"""
import numpy as np
import mne
import config

mne.set_log_level("WARNING")


def load_raw(subject_id: str) -> mne.io.BaseRaw:
    """读取一个被试的 EEG。若文件夹里有多个文件，选时长最长的那份。"""
    eeg_dir = config.DATA_DIR / subject_id / "eeg"
    files = sorted(eeg_dir.glob("*.bdf")) + sorted(eeg_dir.glob("*.set"))
    if not files:
        raise FileNotFoundError(f"{subject_id}: {eeg_dir} 下没有 .bdf / .set 文件")

    # 先只读"文件头"摸清每个文件的时长（不加载数据，速度很快）
    best = None                       # (时长秒, 文件路径)
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
            continue                  # 打不开的文件直接略过

    if best is None or best[0] < config.MIN_DURATION:
        raise ValueError(f"{subject_id}: 没有超过 {config.MIN_DURATION:.0f} 秒的有效记录")

    _, f = best
    if f.suffix.lower() == ".bdf":
        return mne.io.read_raw_bdf(f, preload=True, verbose=False)
    return mne.io.read_raw_eeglab(f, preload=True, verbose=False)


def apply_ica(raw: mne.io.BaseRaw, method: str) -> mne.io.BaseRaw:
    """拟合 ICA，自动剔除与额区（眼电代理）高度相关的成分。"""
    ica = mne.preprocessing.ICA(
        n_components=0.99,
        method=method,                # infomax / fastica / picard
        fit_params={"extended": True} if method == "infomax" else None,
        max_iter=1000,                # 多给迭代次数，减少 Picard 不收敛
        random_state=42,              # 固定随机种子 → 可复现
    )
    ica.fit(raw)

    lower_to_name = {ch.lower(): ch for ch in raw.ch_names}
    frontal = [lower_to_name[c] for c in ("fp1", "fp2") if c in lower_to_name]
    if not frontal:
        raise RuntimeError(f"找不到 Fp1/Fp2,实际通道:{raw.ch_names}")
    sources = ica.get_sources(raw).get_data()
    ref = raw.get_data(picks=frontal).mean(axis=0)
    corrs = np.array([abs(np.corrcoef(s, ref)[0, 1]) for s in sources])

    bads = []
    for idx in np.argsort(corrs)[::-1]:
        if corrs[idx] >= config.ICA_CORR_THRESHOLD and len(bads) < config.ICA_MAX_COMPONENTS:
            bads.append(int(idx))
    ica.exclude = bads
    print(f"      ICA({method}) 剔除 {len(bads)} 个成分")
    return ica.apply(raw.copy())


def preprocess_subject(subject_id: str, method: str = "none") -> mne.io.BaseRaw:
    """完整预处理。method ∈ {"none", "infomax", "fastica", "picard"}"""
    raw = load_raw(subject_id)                # 已内置"太短就拒收"
    raw.pick_types(eeg=True)                  # 只保留 EEG 通道
    raw.resample(config.RESAMPLE_SFREQ)       # → 250 Hz
    raw.filter(config.HIGH_PASS, config.LOW_PASS)   # 1–40 Hz 带通
    raw.notch_filter(config.NOTCH_FREQ)       # 60 Hz 陷波
    if method != "none":
        raw = apply_ica(raw, method)          # ICA：在满秩数据上做
    raw.set_eeg_reference("average")          # 平均重参考放最后
    return raw