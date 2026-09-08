"""
visualization.py
绘制四种方法 Accuracy / Sensitivity / Specificity 分组柱状图，保存 300 dpi PNG。
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_results(results: dict, save_path):
    methods = list(results.keys())
    metrics = ["accuracy", "sensitivity", "specificity"]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, metric in enumerate(metrics):
        vals = [results[m][metric] for m in methods]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=metric.capitalize())
        ax.bar_label(bars, fmt="%.2f", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["None" if m == "none" else m.upper() for m in methods])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Effect of Artifact Rejection Method on Depression Detection (LOSOCV)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)