"""Main entry point: grouping -> per-method feature building -> LOSOCV -> report."""
import argparse
import numpy as np
import pandas as pd
import config
from datasets import ADAPTER
from preprocessing import preprocess_subject
from feature_extraction import extract_features
from evaluation import run_losocv
from visualization import plot_results


def load_groups() -> dict:
    """Delegate to the active dataset adapter (BDI thresholds or clinical labels)."""
    return ADAPTER.load_groups()


def get_features(subject_id: str, method: str) -> np.ndarray:
    """Disk-cached feature extraction: preprocess+extract once, reuse forever."""
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.FEATURES_DIR / f"{subject_id}_{method}.npy"
    if cache.exists():
        return np.load(cache)
    raw = preprocess_subject(subject_id, method)
    feats = extract_features(raw)
    np.save(cache, feats)
    return feats


def build_dataset(groups: dict, method: str, max_subjects=None):
    # Sample both classes separately so small smoke-test subsets stay balanced
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
            print(f"  [skip] {sub}: {e}")
            continue
        X.append(feats)
        y.append(groups[sub])
        print(f"  [{len(X)}] {sub}  (label={groups[sub]}) done")
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_subjects", type=int, default=None)
    args = parser.parse_args()

    groups = load_groups()
    n_dep = sum(1 for v in groups.values() if v == 1)
    n_ctl = sum(1 for v in groups.values() if v == 0)
    print(f"dataset={config.DATASET}  groups: {n_dep} depression, {n_ctl} control")

    results = {}
    for method in config.METHODS:
        print(f"\n===== method: {method} =====")
        try:
            X, y = build_dataset(groups, method, args.max_subjects)
        except Exception as e:
            # e.g. python-picard not installed: skip this method, keep the rest
            print(f"method {method} failed as a whole: {e}")
            continue
        if len(set(y)) < 2:
            print("  subset has a single class; increase --max_subjects")
            continue
        res = run_losocv(X, y)
        results[method] = res
        print(f"  n={len(y)}  Acc={res['accuracy']:.3f}  "
              f"Sens={res['sensitivity']:.3f}  Spec={res['specificity']:.3f}")
        print(f"  confusion [[TN,FP],[FN,TP]]:\n{res['confusion']}")

    config.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    config.FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    rows = [{"method": m, "accuracy": r["accuracy"],
             "sensitivity": r["sensitivity"],
             "specificity": r["specificity"]} for m, r in results.items()]
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    if results:
        out = config.FIGURES_DIR / "methods_comparison.png"
        plot_results(results, out)
        print(f"\nfigure saved: {out}")


if __name__ == "__main__":
    main()
