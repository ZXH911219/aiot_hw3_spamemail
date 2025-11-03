"""Train a logistic regression baseline for spam classification.

Saves vectorizer and model into `models/` directory using joblib.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


def load_data(path: str, label_col: str = None, text_col: str = "text_clean") -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if label_col is None:
        label_col = df.columns[0]
    X = df[text_col].astype(str)
    y = df[label_col]
    return X, y


def train_and_eval(X: pd.Series, y: pd.Series, out_dir: str, random_state: int = 42) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    reports_dir = os.path.join(os.path.dirname(out_dir), "reports", "visualizations")
    os.makedirs(reports_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    probs = None
    try:
        probs = clf.predict_proba(Xte)
    except Exception:
        # some classifiers do not implement predict_proba
        probs = None
    report = classification_report(y_test, preds, output_dict=True)
    # Save a textual evaluation report
    report_path = os.path.join(os.path.dirname(out_dir), "reports", "phase1-eval.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.Series(report).to_json(report_path)
    # Save confusion matrix figure
    labels = np.unique(y)
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig_cm.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    # ROC & PR curves (binary only)
    if probs is not None and len(labels) == 2:
        # map positive class to index 1
        pos_label = labels[1]
        # ensure binary mapping of y_test
        y_true_bin = (y_test == pos_label).astype(int)
        y_score = probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        fig_roc, axr = plt.subplots(figsize=(6, 4))
        axr.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        axr.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axr.set_xlabel("False Positive Rate")
        axr.set_ylabel("True Positive Rate")
        axr.set_title("ROC Curve")
        axr.legend(loc="lower right")
        roc_path = os.path.join(reports_dir, "roc_curve.png")
        fig_roc.tight_layout()
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)

        precision, recall, thresholds = precision_recall_curve(y_true_bin, y_score)
        fig_pr, axp = plt.subplots(figsize=(6, 4))
        axp.plot(recall, precision)
        axp.set_xlabel("Recall")
        axp.set_ylabel("Precision")
        axp.set_title("Precision-Recall Curve")
        pr_path = os.path.join(reports_dir, "pr_curve.png")
        fig_pr.tight_layout()
        fig_pr.savefig(pr_path)
        plt.close(fig_pr)

        # Threshold sweep CSV
        threshs = np.linspace(0.0, 1.0, 101)
        rows = []
        for t in threshs:
            preds_t = (y_score >= t).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true_bin, preds_t, average="binary", zero_division=0)
            rows.append({"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)})
        thresh_df = pd.DataFrame(rows)
        thresh_csv = os.path.join(reports_dir, "threshold_sweep.csv")
        thresh_df.to_csv(thresh_csv, index=False)
    # save artifacts
    joblib.dump(vec, os.path.join(out_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(out_dir, "model.joblib"))
    joblib.dump(np.unique(y), os.path.join(out_dir, "labels.joblib"))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--label-col", default=None)
    p.add_argument("--text-col", default="text_clean")
    p.add_argument("--out", default="models")
    args = p.parse_args()
    X, y = load_data(args.input, args.label_col, args.text_col)
    report = train_and_eval(X, y, args.out)
    print("Training complete. Evaluation summary:")
    # print a compact summary
    for k, v in report.items():
        if k in ("accuracy",):
            print(f"{k}: {v}")
        elif isinstance(v, dict):
            print(f"{k}: precision={v.get('precision'):.3f} recall={v.get('recall'):.3f} f1={v.get('f1-score'):.3f}")


if __name__ == "__main__":
    main()
