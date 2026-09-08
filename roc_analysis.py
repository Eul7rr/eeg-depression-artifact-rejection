"""roc_analysis.py —— 3.2.2 的 F1 + 3.3.2 的 ROC/AUC
前置：evaluation.py 已加入 y_score（两行改动）
用法：python roc_analysis.py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import config
import main as m
from evaluation import run_losocv

METHODS = ["None", "Infomax", "FastICA", "Picard"]

def load_cached(method):
    """与 main 同序读缓存特征：先抑郁组，后对照组"""
    groups = m.load_groups()
    selected = ([s for s, v in groups.items() if v == 1]
                + [s for s, v in groups.items() if v == 0])
    X = np.vstack([np.load(config.FEATURES_DIR / f"{s}_{method}.npy")
                   for s in selected])
    y = np.array([groups[s] for s in selected])
    return X, y

# ---- 任务 A：F1 ----
print("== F1（阳性类=抑郁） ==")
f1 = {}
for method in METHODS:
    df = pd.read_csv(config.RESULTS_DIR / f"predictions_{method}.csv")
    f1[method] = f1_score(df["y_true"], df["y_pred"], pos_label=1)
    print(f"  {method:8s} F1 = {f1[method]:.3f}")
pd.DataFrame([{"method": k, "f1": v} for k, v in f1.items()]).to_csv(
    config.RESULTS_DIR / "f1_summary.csv", index=False)

# ---- 任务 B：ROC ----
fig, ax = plt.subplots(figsize=(6, 6))
rows = []
for method in METHODS:
    X, y = load_cached(method)
    res = run_losocv(X, y)
    if "y_score" not in res:
        raise KeyError("run_losocv 没返回 y_score——回 evaluation.py 检查那两行")
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_score"])
    a = auc(fpr, tpr)
    rows.append({"method": method, "auc": a})
    ax.plot(fpr, tpr, label=f"{method} (AUC={a:.3f})")
    print(f"  {method:8s} AUC = {a:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
ax.set_xlabel("False Positive Rate (1 − Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.legend()
fig.tight_layout()
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=300)   # 投稿图 300 dpi 起步
pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "roc_auc.csv", index=False)
print("图已存:", config.FIGURES_DIR / "roc_curves.png")