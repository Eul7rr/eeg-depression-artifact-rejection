"""
demographics.py —— 为论文 2.1.1 节生成两组被试的人口统计学数据
用法（终端，确认 (eeg) 环境）： python demographics.py
"""
import pandas as pd
from scipy import stats

import config
import main as m

df = pd.read_csv(config.DATA_DIR / "participants.tsv", sep="\t")
print("participants.tsv 列名:", list(df.columns))

groups = m.load_groups()
bdi_col = [c for c in df.columns if "bdi" in c.lower()][0]
age_col = next((c for c in df.columns if "age" in c.lower()), None)
sex_col = next((c for c in df.columns if c.lower() in ("sex", "gender")), None)
print(f"识别到: BDI列={bdi_col}  年龄列={age_col}  性别列={sex_col}\n")

# 只保留进入实验的两组被试（和主实验完全同一套分组逻辑）
included = df[df["participant_id"].isin(
    [s for s, v in groups.items() if v in (0, 1)])].copy()
included["group"] = included["participant_id"].map(groups)
if age_col:
    included[age_col] = pd.to_numeric(included[age_col], errors="coerce")
included[bdi_col] = pd.to_numeric(included[bdi_col], errors="coerce")

# 检查 sub-544 的下落（我们说好要剔除它，验证一下）
for sub in getattr(config, "EXCLUDE_SUBJECTS", []):
    r = df[df["participant_id"] == sub]
    if not r.empty:
        inside = sub in set(included["participant_id"])
        print(f"[检查] {sub}: BDI={r[bdi_col].values[0]}, "
              f"是否进入了实验={'是' if inside else '否'}\n")

dep = included[included["group"] == 1]
ctl = included[included["group"] == 0]

print(f"抑郁组 n={len(dep)}, 对照组 n={len(ctl)}")
print(f"BDI: 抑郁 {dep[bdi_col].mean():.1f}±{dep[bdi_col].std():.1f} | "
      f"对照 {ctl[bdi_col].mean():.1f}±{ctl[bdi_col].std():.1f}")

if age_col:
    print(f"年龄: 抑郁 {dep[age_col].mean():.1f}±{dep[age_col].std():.1f} | "
          f"对照 {ctl[age_col].mean():.1f}±{ctl[age_col].std():.1f}")
    t, p = stats.ttest_ind(dep[age_col].dropna(), ctl[age_col].dropna())
    print(f"  组间比较: t={t:.2f}, p={p:.3f}")

if sex_col:
    print(f"性别: 抑郁 {dep[sex_col].value_counts().to_dict()} | "
          f"对照 {ctl[sex_col].value_counts().to_dict()}")
    tab = pd.crosstab(included[sex_col], included["group"])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    print(f"  组间比较: 卡方={chi2:.2f}, p={p:.3f}")