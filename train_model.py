"""Train, compare and evaluate URL classifiers with domain-separated splits."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from url_features import extract_many, registrable_domain

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "processed" / "phishing_small.csv"
MODEL_PATH = ROOT / "model.joblib"
REPORT_PATH = ROOT / "results.json"
SEED = 42


def split_by_domain(df):
    groups = df["url"].map(registrable_domain)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_valid_idx, test_idx = next(outer.split(df, df["label"], groups))
    train_valid = df.iloc[train_valid_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    inner_groups = train_valid["url"].map(registrable_domain)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, valid_idx = next(inner.split(train_valid, train_valid["label"], inner_groups))
    return train_valid.iloc[train_idx].reset_index(drop=True), train_valid.iloc[valid_idx].reset_index(drop=True), test


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run prepare_data.py first. Missing: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["url", "label"]).drop_duplicates("url")
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("The processed CSV must contain both labels: 0=legitimate, 1=phishing.")

    train, valid, test = split_by_domain(df)
    X_train, X_valid, X_test = (extract_many(part["url"]) for part in (train, valid, test))
    feature_names = X_train.columns[X_train.nunique(dropna=False) > 1].tolist()
    X_train, X_valid, X_test = (features[feature_names] for features in (X_train, X_valid, X_test))
    y_train, y_valid, y_test = (part["label"].to_numpy() for part in (train, valid, test))
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced", random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
    }
    comparisons = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold(y_valid, scores)
        predictions = (scores >= threshold).astype(int)
        comparisons[name] = {"validation_threshold": threshold,
                             "validation_precision": float(precision_score(y_valid, predictions, zero_division=0)),
                             "validation_recall": float(recall_score(y_valid, predictions, zero_division=0)),
                             "validation_f1": float(f1_score(y_valid, predictions, zero_division=0))}
        fitted[name] = (model, threshold)

    best_name = max(comparisons, key=lambda name: comparisons[name]["validation_f1"])
    best_model, threshold = fitted[best_name]
    test_scores = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    report = {
        "dataset_rows": int(len(df)),
        "label_counts": {"legitimate": int((df.label == 0).sum()), "phishing": int((df.label == 1).sum())},
        "split_rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "unique_domains": int(df["url"].map(registrable_domain).nunique()),
        "models_validation": comparisons,
        "selected_model": best_name,
        "selected_threshold": threshold,
        "held_out_test": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "confusion_matrix_labels_0_legitimate_1_phishing": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
        },
        "feature_names": feature_names,
    }
    joblib.dump({"model": best_model, "model_name": best_name, "threshold": threshold,
                 "feature_names": feature_names, "background": X_train.sample(min(100, len(X_train)), random_state=SEED)}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Selected by validation F1: {best_name} (threshold {threshold:.3f})")
    for name, metrics in comparisons.items():
        print(f"{name}: validation precision={metrics['validation_precision']:.3f}, recall={metrics['validation_recall']:.3f}, F1={metrics['validation_f1']:.3f}")
    print("Held-out domain test:", json.dumps(report["held_out_test"], indent=2))
    print(f"Saved model to {MODEL_PATH} and metrics to {REPORT_PATH}")


if __name__ == "__main__":
    main()
