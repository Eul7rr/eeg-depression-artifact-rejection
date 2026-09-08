"""
stats_compare.py —— 统计比较：None vs 三种 ICA（读缓存特征，一两分钟跑完）

做什么：
  1. 直接读缓存好的特征，重跑 LOSOCV，拿到"每个被试"的预测结果
  2. 逐被试预测存成 results/predictions_<方法>.csv（论文附录材料）
  3. McNemar 检验：None 和每种 ICA 的差距是真效应还是运气
  4. 汇总表存 results/stats_summary.csv

用法（终端，先确认是 (eeg) 环境）：
    python stats_compare.py
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

import config
import main as m
from evaluation import run_losocv


def load_cached_dataset(groups, method):
    """和 main.build_dataset 全量模式完全同序，但从缓存读特征，秒回。"""
    dep = [s for s, v in sorted(groups.items()) if v == 1]
    ctl = [s for s, v in sorted(groups.items()) if v == 0]
    subs, X, y = [], [], []
    for sub in dep + ctl:
        try:
            feats = m.get_features(sub, method)   # 缓存命中，不会重算
        except Exception as e:
            print(f"  [跳过] {sub}: {e}")
            continue
        subs.append(sub)
        X.append(feats)
        y.append(groups[sub])
    return subs, np.array(X), np.array(y)


def summarize(yt, yp):
    """从逐被试预测算四个指标。"""
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(yt)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    bal = (sens + spec) / 2
    return acc, sens, spec, bal


def main():
    groups = m.load_groups()

    # 第一步：四种方法各跑一遍 LOSOCV，收逐被试预测
    store = {}   # method -> (subs, y_true, y_pred)
    for method in config.METHODS:
        print(f"\n===== 方法: {method} =====")
        subs, X, y = load_cached_dataset(groups, method)
        res = run_losocv(X, y)
        if "y_true" not in res or "y_pred" not in res:
            raise KeyError(
                f"run_losocv 的返回里没有 y_true / y_pred，"
                f"实际键为: {list(res.keys())} —— 把这行原样发给我"
            )
        yt, yp = np.asarray(res["y_true"]), np.asarray(res["y_pred"])
        store[method] = (subs, yt, yp)
        acc, sens, spec, bal = summarize(yt, yp)
        print(f"  n={len(y)}  Acc={acc:.3f}  Sens={sens:.3f}  "
              f"Spec={spec:.3f}  BalancedAcc={bal:.3f}")
        pd.DataFrame({"subject": subs, "y_true": yt, "y_pred": yp}).to_csv(
            config.RESULTS_DIR / f"predictions_{method}.csv", index=False)

    # 第二步：McNemar 检验——None 当基准，和每种 ICA 两两比较
    base_subs, base_yt, base_yp = store["none"]
    base_correct = dict(zip(base_subs, base_yp == base_yt))

    print("\n===== McNemar 检验：None vs ICA =====")
    print("(b = None 猜对而 ICA 猜错的人数；c = ICA 猜对而 None 猜错的人数)")
    rows = []
    for method in config.METHODS:
        if method == "none":
            continue
        subs, yt, yp = store[method]
        b = c = 0
        for s, t, p in zip(subs, yt, yp):
            if s not in base_correct:
                continue
            none_ok = base_correct[s]
            ica_ok = bool(p == t)
            if none_ok and not ica_ok:
                b += 1
            elif ica_ok and not none_ok:
                c += 1
        if b + c == 0:
            pval, note = float("nan"), "两法完全一致"
        else:
            pval = binomtest(b, b + c, 0.5).pvalue
            note = "差异显著" if pval < 0.05 else "差异不显著"
        print(f"  none vs {method:<8s}  b={b:2d}  c={c:2d}  "
              f"p={pval:.4f}  {note}")
        rows.append({"comparison": f"none vs {method}", "b": b, "c": c,
                     "p_value": pval, "significant_0.05": pval < 0.05})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "stats_summary.csv",
                              index=False)
    print(f"\n已保存到 {config.RESULTS_DIR}: predictions_*.csv 和 stats_summary.csv")


if __name__ == "__main__":
    main()