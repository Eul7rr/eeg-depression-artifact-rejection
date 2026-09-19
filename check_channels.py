"""Recon: dead (zero-variance) channels or NaNs inside MODMA .mat files?"""
import numpy as np
from scipy.io import loadmat
import config
from datasets import MODMA

assert config.DATASET == "modma"

for p in sorted(MODMA.DATA_DIR.glob("*.mat"))[:6]:
    mat = loadmat(p, struct_as_record=False, squeeze_me=True)
    arrs = [v for k, v in mat.items() if not k.startswith("__")
            and isinstance(v, np.ndarray) and v.ndim == 2]
    a = max(arrs, key=lambda x: x.size)
    std = np.nanstd(a, axis=1)
    live = std[std > 0]
    print(f"{p.name[:22]:24s} shape={a.shape}  dead_rows={(std == 0).sum():3d}  "
          f"NaNs={np.isnan(a).sum()}  live-std {live.min():.3g}..{live.max():.3g}")