"""Download and vendor the SMS spam CSV into data/raw/.

Usage:
    python scripts/download_data.py --url <CSV_URL> --out data/raw/sms_spam_no_header.csv
"""
from __future__ import annotations

import argparse
import os
from urllib.request import urlopen


def download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    with urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())
    print("Download complete")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    download(args.url, args.out)


if __name__ == "__main__":
    main()
