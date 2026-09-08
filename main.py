"""
main.py —— 主程序入口
用法（终端里，先 conda activate eeg）：
    python check_data.py               # 第 4 步：侦察数据
    python main.py --max_subjects 10   # 第 5 步：冒烟测试（5 抑郁 + 5 对照）
    python main.py                     # 第 6 步：全量正式实验
"""
import argparse
import numpy as np
import pandas as pd
import config
from preprocessing import preprocess_subject
from feature_extraction import extract_features
from evaluation import run_losocv
from visualization import plot_results


def load_groups() -> dict:
    """读 participants.tsv，按 BDI 阈值分组。返回 {subject_id: 0 / 1 / None}"""
    df = pd.read_csv(config.DATA_DIR / "participants.tsv", sep="\t")
    bdi_cols = [c for c in df.columns if "bdi" in c.lower()]
    if not bdi_cols:
        raise KeyError(f"participants.tsv 里找不到 BDI 列，实际列名为: {list(df.columns)}")
    bdi_col = bdi_cols[0]

    groups = {}
    for _, row in df.iterrows():
        sub = row["participant_id"]
        try:
            bdi = float(row[bdi_col])
        except (TypeError, ValueError):
            groups[sub] = None
            continue
        if bdi >= config.BDI_DEPRESSION_MIN:
            groups[sub] = 1
        elif bdi < config.BDI_CONTROL_MAX:
            groups[sub] = 0
        else:
            groups[sub] = None          # 7–13 边缘值剔除
    return groups


def get_features(subject_id: str, method: str) -> np.ndarray:
    """带磁盘缓存的特征提取——算过一次就直接读缓存，重跑不重复劳动。"""
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.FEATURES_DIR / f"{subject_id}_{method}.npy"
    if cache.exists():
        return np.load(cache)
    raw = preprocess_subject(subject_id, method)
    feats = extract_features(raw)
    np.save(cache, feats)
    return feats


def build_dataset(groups: dict, method: str, max_subjects=None):
    # 两类人分开排队再各取一半——保证快速测试时两组都有代表
    dep = [s for s, v in sorted(groups.items()) if v == 1]
    ctl = [s for s, v in sorted(groups.items()) if v == 0]
    if max_subjects is not None:
        half = max_subjects // 2
        selected = dep[:half] + ctl[:max_subjects - half]
    else:
        selected = dep + ctl

    X, y = [], []
    for sub in selected:
        try:
            feats = get_features(sub, method)
        except Exception as e:
            print(f"  [跳过] {sub}: {e}")
            continue
        X.append(feats)
        y.append(groups[sub])
        print(f"  [{len(X)}] {sub}  (label={groups[sub]}) 完成")
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_subjects", type=int, default=None)
    args = parser.parse_args()

    groups = load_groups()
    n_dep = sum(1 for v in groups.values() if v == 1)
    n_ctl = sum(1 for v in groups.values() if v == 0)
    print(f"分组结果：抑郁组 {n_dep} 人，对照组 {n_ctl} 人（边缘值已剔除）")

    results = {}
    for method in config.METHODS:
        print(f"\n===== 方法: {method} =====")
        try:
            X, y = build_dataset(groups, method, args.max_subjects)
        except Exception as e:
            print(f"方法 {method} 整体失败（常见原因：picard 未安装）: {e}")
            continue
        if len(set(y)) < 2:
            print("  当前子集只有一个类别，无法分类——请把 --max_subjects 调大")
            continue
        res = run_losocv(X, y)
        results[method] = res
        print(f"  n={len(y)}  Acc={res['accuracy']:.3f}  "
              f"Sens={res['sensitivity']:.3f}  Spec={res['specificity']:.3f}")
        print(f"  混淆矩阵 [[TN,FP],[FN,TP]]:\n{res['confusion']}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"method": m, "accuracy": r["accuracy"],
             "sensitivity": r["sensitivity"],
             "specificity": r["specificity"]} for m, r in results.items()]
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    if results:
        out = config.FIGURES_DIR / "methods_comparison.png"
        plot_results(results, out)
        print(f"\n对比图已保存: {out}")


if __name__ == "__main__":
    main()