"""Smoke inspection for the leukipp Kaggle archive.

Prints: per-subreddit submission counts and date ranges.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = REPO_ROOT / "data" / "raw" / "kaggle"


def inspect_leukipp() -> None:
    base = KAGGLE_DIR / "finance_multi"
    print(f"=== leukipp/reddit-finance-data ({base}) ===")
    for sub in ["wallstreetbets", "stocks", "investing", "options"]:
        csv = base / sub / "submissions_reddit.csv"
        if not csv.exists():
            print(f"  {sub:16s} MISSING")
            continue
        df = pd.read_csv(csv, usecols=["created", "id"])
        dt = pd.to_datetime(df["created"], errors="coerce")
        print(
            f"  r/{sub:14s}  rows={len(df):>8,}  "
            f"date={dt.min()}  ->  {dt.max()}"
        )


if __name__ == "__main__":
    inspect_leukipp()
