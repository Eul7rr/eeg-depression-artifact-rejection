"""Demographic and clinical summary for the two groups (paper Section 2.1.1)."""
import pandas as pd
from scipy import stats

import config
import main as m

df = pd.read_csv(config.DATA_DIR / "participants.tsv", sep="\t")
print("participants.tsv columns:", list(df.columns))

groups = m.load_groups()

# Column names vary across dataset versions — detect, don't hard-code
bdi_col = [c for c in df.columns if "bdi" in c.lower()][0]
age_col = next((c for c in df.columns if "age" in c.lower()), None)
sex_col = next((c for c in df.columns if c.lower() in ("sex", "gender")), None)
print(f"Detected: BDI={bdi_col}  age={age_col}  sex={sex_col}\n")

# Same grouping logic as the main experiment — never reimplement it here
included = df[df["participant_id"].isin(
    [s for s, v in groups.items() if v in (0, 1)])].copy()
included["group"] = included["participant_id"].map(groups)
if age_col:
    included[age_col] = pd.to_numeric(included[age_col], errors="coerce")
included[bdi_col] = pd.to_numeric(included[bdi_col], errors="coerce")

# Sanity check: subjects on the exclusion list must not appear in the analysis
for sub in getattr(config, "EXCLUDE_SUBJECTS", []):
    r = df[df["participant_id"] == sub]
    if not r.empty:
        inside = sub in set(included["participant_id"])
        print(f"[check] {sub}: BDI={r[bdi_col].values[0]}, included={inside}\n")

dep = included[included["group"] == 1]
ctl = included[included["group"] == 0]

print(f"depression n={len(dep)}, control n={len(ctl)}")
print(f"BDI: dep {dep[bdi_col].mean():.1f}±{dep[bdi_col].std():.1f} | "
      f"ctl {ctl[bdi_col].mean():.1f}±{ctl[bdi_col].std():.1f}")

if age_col:
    print(f"Age: dep {dep[age_col].mean():.1f}±{dep[age_col].std():.1f} | "
          f"ctl {ctl[age_col].mean():.1f}±{ctl[age_col].std():.1f}")
    t, p = stats.ttest_ind(dep[age_col].dropna(), ctl[age_col].dropna())
    print(f"  between-group: t={t:.2f}, p={p:.3f}")

if sex_col:
    print(f"Sex: dep {dep[sex_col].value_counts().to_dict()} | "
          f"ctl {ctl[sex_col].value_counts().to_dict()}")
    tab = pd.crosstab(included[sex_col], included["group"])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    print(f"  between-group: chi2={chi2:.2f}, p={p:.3f}")
