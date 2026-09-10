"""Linear SVM (C=1.0) under strict leave-one-subject-out cross-validation."""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def run_losocv(X: np.ndarray, y: np.ndarray) -> dict:
    logo = LeaveOneGroupOut()
    groups = np.arange(len(y))  # one group per subject -> leave-one-subject-out

    # Imputation and scaling live inside the Pipeline so each fold fits them
    # on training data only — fitting them upstream would leak test information.
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0)),
    ])

    y_true, y_pred = [], []
    y_score = []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        y_true.append(y[test_idx][0])
        y_pred.append(pipe.predict(X[test_idx])[0])
        y_score.append(pipe.decision_function(X[test_idx])[0])
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "confusion": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": np.array(y_score),
    }
