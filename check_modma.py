"""Recon script for MODMA. Run once after unzipping, BEFORE the full pipeline.

Verifies the three assumptions the MODMA adapter makes:
  1. layout: one .mat per subject + the info xlsx, counts match
  2. .mat payload: which key holds the (channels, samples) array
  3. units: peak amplitude decides whether data is uV (scale 1e-6) or nV (1e-9)

Usage: python check_modma.py   (then send the full printout for review)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

import config
from datasets import MODMA

assert config.DATASET == "modma",
root = MODMA.DATA_DIR

print("== 1) folder layout ==")
for p in sorted(root.iterdir()):
    print(f"   {p.name}")
mats = sorted(root.glob("*.mat"))
print(f"\n.mat files: {len(mats)} (expected 53)")

print("\n== 2) metadata sheet ==")
xlsx = root / MODMA.INFO_XLSX
df = pd.read_excel(xlsx)
print("columns:", list(df.columns))
print("rows:", len(df))
type_col = [c for c in df.columns if str(c).lower() == "type"]
if type_col:
    print("label counts:", df[type_col[0]].value_counts().to_dict())
print(df.head(3).to_string())

print("\n== 3) one .mat payload ==")
mat = loadmat(mats[0], struct_as_record=False, squeeze_me=True)
for k, v in mat.items():
    if k.startswith("__"):
        continue
    print(f"   key={k!r:30s} type={type(v).__name__} "
          f"shape={getattr(v, 'shape', None)}")
arr = max((v for k, v in mat.items() if not k.startswith("__")
           and isinstance(v, np.ndarray) and v.ndim == 2),
          key=lambda a: a.size)
print(f"\nchosen array: shape={arr.shape}  peak|value|={np.nanmax(np.abs(arr)):.3g}")
print("peak ~1e1-1e2 -> microvolts (scale 1e-6); peak ~1e4-1e5 -> nanovolts (1e-9)")

print("\n== 4) groups via adapter ==")
groups = MODMA.load_groups()
n1 = sum(1 for v in groups.values() if v == 1)
n0 = sum(1 for v in groups.values() if v == 0)
print(f"MDD={n1} (expected 24), HC={n0} (expected 29)")
