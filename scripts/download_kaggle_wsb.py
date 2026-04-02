"""Download the leukipp Reddit finance archive into data/raw/kaggle/.

Dataset:
  - leukipp/reddit-finance-data  (2021 full year, 14 subreddit subfolders;
                                  we only USE {wallstreetbets, stocks,
                                  investing, options} downstream, but the
                                  archive is unzipped as-is — the other
                                  subfolders are left on disk)

Auth: reads KAGGLE_API_TOKEN from .env (new "KGAT_" token format).
Tested with `kaggle` 2.0.2 / kagglesdk 0.1.20 (April 2026).

Run from repo root:
    python scripts/download_kaggle_wsb.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from alt_sentiment import _ssl_fix  # noqa: F401  (SSL cert path repair, must be first)
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "kaggle"

DATASET = "leukipp/reddit-finance-data"


def _authenticate() -> None:
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        sys.exit("KAGGLE_API_TOKEN missing from .env — see .env.example")
    # kagglesdk (used by the kaggle CLI wrapper) reads KAGGLE_API_TOKEN directly
    # from the environment, so no kaggle.json file is needed.
    os.environ["KAGGLE_API_TOKEN"] = token


def _download(ref: str, dest: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {ref} -> {dest}")
    api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=False)


def main() -> None:
    _authenticate()

    dest = RAW_DIR / "finance_multi"
    if not dest.exists() or not any(dest.iterdir()):
        _download(DATASET, dest)
    else:
        print(f"[skip] {dest} already populated")

    print("\n[done] archive in:", RAW_DIR)
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {size_mb:8.1f} MB  {p.relative_to(RAW_DIR)}")


if __name__ == "__main__":
    main()
