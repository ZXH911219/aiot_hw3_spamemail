"""Predict spam for single texts or a batch CSV using saved model artifacts.

Usage examples:
  python scripts/predict_spam.py --text "Free money" --models models
  python scripts/predict_spam.py --input datasets/processed/sms_spam_clean.csv --output preds.csv --models models
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
import pandas as pd


def predict_text(text: str, models_dir: str) -> dict:
    vec = joblib.load(os.path.join(models_dir, "vectorizer.joblib"))
    clf = joblib.load(os.path.join(models_dir, "model.joblib"))
    labels = joblib.load(os.path.join(models_dir, "labels.joblib"))
    x = vec.transform([text])
    prob = clf.predict_proba(x)[0]
    pred = clf.predict(x)[0]
    return {"text": text, "pred": str(pred), "probs": prob.tolist(), "labels": labels.tolist()}


def predict_batch(input_csv: str, output_csv: str, text_col: str, models_dir: str) -> None:
    df = pd.read_csv(input_csv)
    vec = joblib.load(os.path.join(models_dir, "vectorizer.joblib"))
    clf = joblib.load(os.path.join(models_dir, "model.joblib"))
    labels = joblib.load(os.path.join(models_dir, "labels.joblib"))
    X = df[text_col].astype(str)
    Xv = vec.transform(X)
    preds = clf.predict(Xv)
    probs = clf.predict_proba(Xv)
    out = df.copy()
    out["pred"] = preds
    # store probability for predicted class
    out["pred_prob"] = [p.max() for p in probs]
    out.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", help="Single text to classify")
    p.add_argument("--input", help="CSV input for batch predictions")
    p.add_argument("--output", help="CSV output path for batch predictions")
    p.add_argument("--text-col", default="text_clean")
    p.add_argument("--models", default="models")
    args = p.parse_args()
    if args.text:
        res = predict_text(args.text, args.models)
        print(res)
    elif args.input and args.output:
        predict_batch(args.input, args.output, args.text_col, args.models)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
