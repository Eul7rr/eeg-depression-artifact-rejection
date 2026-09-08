"""特征提取：19 通道 × 4 频带 = 76 维 log 功率"""
import numpy as np
from scipy.signal import welch

import config

try:
    from numpy import trapezoid as _trapz      # numpy >= 2.0
except ImportError:
    from numpy import trapz as _trapz          # numpy < 2.0


def _resolve_picks(raw):
    """按 config.CHANNELS_1020 的顺序，找出每个通道在 raw 里的下标。

    - 大小写不敏感：这个数据集的通道名是全大写（FP1/FZ/CZ/PZ），
      而我们的名单是标准写法（Fp1/Fz/...），必须忽略大小写匹配。
    - 兼容新旧命名别名（如 T3=T7），见 config.CHANNEL_ALIASES。
    - 找不到就返回 None（由 extract_features 统一报错）。
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
    """raw（已预处理）→ 76 维特征向量（log10 功率）"""
    resolved = _resolve_picks(raw)

    # 宁可大声报错，也不静默产出 NaN：缺一个通道都直接失败
    missing = [ch for ch, idx in zip(config.CHANNELS_1020, resolved) if idx is None]
    if missing:
        raise RuntimeError(
            f"找不到通道 {missing}；这份数据实际有：{raw.ch_names}"
        )

    sfreq = raw.info["sfreq"]
    n_per_seg = min(int(sfreq * 4), raw.n_times)   # 4 秒窗
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