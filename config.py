"""
config.py —— 全局配置：路径、预处理参数、特征定义、分组阈值、方法列表
"""
from pathlib import Path

# ---------- 路径 ----------
DATA_DIR = Path.home() / "Downloads" / "ds003478"
RESULTS_DIR = Path(__file__).parent / "results"
FEATURES_DIR = RESULTS_DIR / "features_cache"   # 特征缓存（ICA 很慢，必须缓存）
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------- 预处理 ----------
RESAMPLE_SFREQ = 250        # 降采样目标 (Hz)
HIGH_PASS = 1.0             # 高通 (Hz)
LOW_PASS = 40.0             # 低通 (Hz)
NOTCH_FREQ = 60.0           # 陷波 (Hz)：数据采集于美国，工频 60Hz
MIN_DURATION = 20.0         # 记录短于 20 秒直接拒收
ICA_CORR_THRESHOLD = 0.5    # ICA 成分与额区信号的相关系数阈值
ICA_MAX_COMPONENTS = 3      # 每个被试最多剔除的 ICA 成分数

# ---------- 特征 ----------
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
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# ---------- 分组（BDI 阈值） ----------
BDI_DEPRESSION_MIN = 13     # BDI ≥ 13 → 抑郁组 (1)
BDI_CONTROL_MAX = 7         # BDI < 7  → 对照组 (0)；7–13 剔除

# ---------- 实验 ----------
METHODS = ["none", "infomax", "fastica", "picard"]