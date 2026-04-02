"""Loader for Dataset B: leukipp/reddit-finance-data.

Multi-subreddit submissions-only archive; 2021 full year. We only consume
four subfolders downstream:

    wallstreetbets  775,326 submissions
    stocks          75,857
    investing       41,912
    options         28,782

Raw schema (per-subreddit submissions_reddit.csv):
    id            str  submission id
    author        str
    created       str  "YYYY-MM-DD HH:MM:SS" (already formatted, NOT epoch)
    title         str
    selftext      str  body text; NaN for link posts / pure-media
    score         float
    num_comments  int
    ...           (many other columns)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

SUBREDDITS = ("wallstreetbets", "stocks", "investing", "options")
_CHUNK = 100_000


def _combine_text(row_title: object, row_body: object) -> str:
    parts = []
    if isinstance(row_title, str) and row_title.strip():
        parts.append(row_title.strip())
    if isinstance(row_body, str) and row_body.strip():
        parts.append(row_body.strip())
    return " ".join(parts)


def iter_docs(
    kaggle_root: str | Path,
    subreddits: tuple[str, ...] = SUBREDDITS,
    chunk_size: int = _CHUNK,
) -> Iterator[pd.DataFrame]:
    """Stream the leukipp archive as chunks in the common schema.

    Iterates each listed subreddit's submissions_reddit.csv in turn.
    Yields DataFrames with columns:
        doc_id, subreddit, created_utc, text, score, is_comment
    (is_comment is always False here — this archive is submissions-only.)
    """
    base = Path(kaggle_root) / "finance_multi"
    usecols = ["id", "created", "title", "selftext", "score"]
    for sub in subreddits:
        csv = base / sub / "submissions_reddit.csv"
        if not csv.exists():
            raise FileNotFoundError(csv)
        for ch in pd.read_csv(csv, usecols=usecols, chunksize=chunk_size):
            out = pd.DataFrame(
                {
                    "doc_id": f"leukipp/{sub}/" + ch["id"].astype(str),
                    "subreddit": sub,
                    "created_utc": pd.to_datetime(ch["created"], errors="coerce"),
                    "text": [
                        _combine_text(t, b)
                        for t, b in zip(ch["title"], ch["selftext"])
                    ],
                    "score": pd.to_numeric(ch["score"], errors="coerce"),
                    "is_comment": False,
                }
            )
            out = out[out["text"].str.len() > 0]
            out = out[out["created_utc"].notna()]
            yield out.reset_index(drop=True)
