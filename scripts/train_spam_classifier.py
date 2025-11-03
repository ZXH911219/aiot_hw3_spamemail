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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    report = classification_report(y_test, preds, output_dict=True)
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
