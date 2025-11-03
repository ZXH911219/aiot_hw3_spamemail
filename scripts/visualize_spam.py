"""Visualize spam classifier results and dataset statistics.

Saves images/CSVs to reports/visualizations/ by default.

Usage:
  python scripts/visualize_spam.py --input data/processed/sms_spam_clean.csv --models models --out reports/visualizations
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_curve)


def ensure_out(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, label_col: str, out_dir: str) -> str:
    counts = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_xlabel(label_col)
    ax.set_ylabel("count")
    ax.set_title("Class distribution")
    path = os.path.join(out_dir, "class_distribution.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def top_tokens(df: pd.DataFrame, text_col: str, label_col: str, topn: int, out_dir: str) -> list[str]:
    results = []
    for cls in sorted(df[label_col].unique()):
        texts = df.loc[df[label_col] == cls, text_col].astype(str)
        cv = CountVectorizer(ngram_range=(1, 1), stop_words="english", min_df=1)
        X = cv.fit_transform(texts)
        tokens = cv.get_feature_names_out()
        freqs = np.asarray(X.sum(axis=0)).ravel()
        series = pd.Series(freqs, index=tokens).sort_values(ascending=False).head(topn)
        csv_path = os.path.join(out_dir, f"top_tokens_{cls}.csv")
        series.to_csv(csv_path, header=["count"]) if len(series) > 0 else pd.Series(dtype=int).to_csv(csv_path)
        # plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=series.values, y=series.index, ax=ax)
        ax.set_xlabel("count")
        ax.set_ylabel("token")
        ax.set_title(f"Top {topn} tokens for {cls}")
        fig.tight_layout()
        img_path = os.path.join(out_dir, f"top_tokens_{cls}.png")
        fig.savefig(img_path)
        plt.close(fig)
        results.append(csv_path)
    return results


def plot_confusion(y_true: Iterable, y_pred: Iterable, labels: list, out_dir: str) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc_pr(y_true_bin: np.ndarray, y_score: np.ndarray, out_dir: str) -> tuple[str, str]:
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    roc_path = os.path.join(out_dir, "roc_curve.png")
    fig1.tight_layout()
    fig1.savefig(roc_path)
    plt.close(fig1)

    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(recall, precision)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    pr_path = os.path.join(out_dir, "pr_curve.png")
    fig2.tight_layout()
    fig2.savefig(pr_path)
    plt.close(fig2)
    return roc_path, pr_path


def threshold_sweep(y_true_bin: np.ndarray, y_score: np.ndarray, out_dir: str) -> str:
    threshs = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in threshs:
        preds_t = (y_score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true_bin, preds_t, average="binary", zero_division=0)
        rows.append({"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)})
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "threshold_sweep.csv")
    df.to_csv(out_csv, index=False)
    # plot f1 vs threshold
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["threshold"], df["f1"], label="f1")
    ax.set_xlabel("threshold")
    ax.set_ylabel("f1")
    ax.set_title("Threshold sweep (f1)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_sweep.png"))
    plt.close(fig)
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--models", default="models")
    p.add_argument("--text-col", default="text_clean")
    p.add_argument("--label-col", default=None)
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--out", default="reports/visualizations")
    args = p.parse_args()

    ensure_out(args.out)
    df = pd.read_csv(args.input)
    label_col = args.label_col or df.columns[0]
    text_col = args.text_col

    print("Plotting class distribution...")
    plot_class_distribution(df, label_col, args.out)

    print("Computing top tokens per class...")
    top_tokens(df, text_col, label_col, args.topn, args.out)

    # if models exist, run evaluation plots
    vec_path = os.path.join(args.models, "vectorizer.joblib")
    model_path = os.path.join(args.models, "model.joblib")
    labels_path = os.path.join(args.models, "labels.joblib")
    if os.path.exists(vec_path) and os.path.exists(model_path) and os.path.exists(labels_path):
        vec = joblib.load(vec_path)
        clf = joblib.load(model_path)
        labels = list(joblib.load(labels_path))
        print("Generating evaluation plots using model artifacts...")
        Xv = vec.transform(df[text_col].astype(str))
        preds = clf.predict(Xv)
        try:
            probs = clf.predict_proba(Xv)[:, 1]
        except Exception:
            probs = None

        plot_confusion(df[label_col].astype(str), preds, labels, args.out)

        if probs is not None and len(labels) == 2:
            y_true_bin = (df[label_col].astype(str) == labels[1]).astype(int).values
            plot_roc_pr(y_true_bin, probs, args.out)
            threshold_sweep(y_true_bin, probs, args.out)
    else:
        print("Model artifacts not found; skipping model evaluation plots.")

    print(f"Visualizations written to {args.out}")


if __name__ == "__main__":
    main()
