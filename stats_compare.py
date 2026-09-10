"""Statistical comparison: None vs each ICA variant (McNemar exact + Holm)."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import multipletests

import config
import main as m
from evaluation import run_losocv


def load_cached_dataset(groups, method):
    """Same subject order as main.build_dataset (full mode), but from cache."""
    dep = [s for s, v in sorted(groups.items()) if v == 1]
    ctl = [s for s, v in sorted(groups.items()) if v == 0]
    subs, X, y = [], [], []
    for sub in dep + ctl:
        try:
            feats = m.get_features(sub, method)
        except Exception as e:
            print(f"  [skip] {sub}: {e}")
            continue
        subs.append(sub)
        X.append(feats)
        y.append(groups[sub])
    return subs, np.array(X), np.array(y)


def summarize(yt, yp):
    """Accuracy, sensitivity, specificity, balanced accuracy from predictions."""
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(yt)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    bal = (sens + spec) / 2
    return acc, sens, spec, bal


def main():
    groups = m.load_groups()

    store = {}  # method -> (subs, y_true, y_pred)
    for method in config.METHODS:
        print(f"\n===== method: {method} =====")
        subs, X, y = load_cached_dataset(groups, method)
        res = run_losocv(X, y)
        if "y_true" not in res or "y_pred" not in res:
            raise KeyError(
                f"run_losocv returned no y_true/y_pred; keys: {list(res.keys())}"
            )
        yt, yp = np.asarray(res["y_true"]), np.asarray(res["y_pred"])
        store[method] = (subs, yt, yp)
        acc, sens, spec, bal = summarize(yt, yp)
        print(f"  n={len(y)}  Acc={acc:.3f}  Sens={sens:.3f}  "
              f"Spec={spec:.3f}  BalancedAcc={bal:.3f}")
        pd.DataFrame({"subject": subs, "y_true": yt, "y_pred": yp}).to_csv(
            config.RESULTS_DIR / f"predictions_{method}.csv", index=False)

    # McNemar compares discordant pairs only:
    # b = correct under None but wrong under ICA; c = the reverse.
    base_subs, base_yt, base_yp = store["none"]
    base_correct = dict(zip(base_subs, base_yp == base_yt))

    print("\n===== McNemar: None vs ICA =====")
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
            pval, note = float("nan"), "identical predictions"
        else:
            pval = binomtest(b, b + c, 0.5).pvalue
            note = "significant" if pval < 0.05 else "n.s."
        print(f"  none vs {method:<8s}  b={b:2d}  c={c:2d}  "
              f"p={pval:.4f}  {note}")
        rows.append({"comparison": f"none vs {method}", "b": b, "c": c,
                     "p_value": pval, "significant_0.05": pval < 0.05})

    # Holm-adjust across the three comparisons; keep raw and adjusted side by side
    df = pd.DataFrame(rows)
    _, p_holm, _, _ = multipletests(df["p_value"], method="holm")
    df["p_holm"] = p_holm
    df.to_csv(config.RESULTS_DIR / "stats_summary.csv", index=False)
    for cmp_, ph in zip(df["comparison"], df["p_holm"]):
        print(f"  {cmp_:<20s}  Holm-adjusted p = {ph:.4f}")


if __name__ == "__main__":
    main()
