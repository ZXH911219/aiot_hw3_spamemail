"""Preprocess SMS/email dataset.

Simple pipeline:
- Read CSV (optionally no header, label col index and text col index configurable)
- Produce a cleaned text column `text_clean` with lowercase and minimal punctuation removal
- Save processed CSV

Functions are exposed for unit testing.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Iterable

import pandas as pd


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    # remove simple punctuation
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].fillna("")
    df["text_clean"] = df[text_col].astype(str).map(clean_text)
    return df


def process_file(input_path: str, output_path: str, no_header: bool = False, label_col_index: int = 0, text_col_index: int = 1) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if no_header:
        df = pd.read_csv(input_path, header=None)
        label_col = df.columns[label_col_index]
        text_col = df.columns[text_col_index]
    else:
        df = pd.read_csv(input_path)
        # default assume first column is label and second is text if names unknown
        cols = list(df.columns)
        label_col = cols[label_col_index]
        text_col = cols[text_col_index]

    out = preprocess_df(df, text_col)
    out.to_csv(output_path, index=False)
    print(f"Wrote processed file to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-header", action="store_true", default=False)
    p.add_argument("--label-col-index", type=int, default=0)
    p.add_argument("--text-col-index", type=int, default=1)
    args = p.parse_args()
    process_file(args.input, args.output, args.no_header, args.label_col_index, args.text_col_index)


if __name__ == "__main__":
    main()
