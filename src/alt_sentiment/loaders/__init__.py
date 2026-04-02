"""Per-dataset loaders.

Each loader returns a DataFrame with the common schema:

    doc_id        str   unique document id (prefixed with dataset/subreddit)
    subreddit     str   source subreddit (e.g. "wallstreetbets")
    created_utc   Timestamp (naive UTC; callers add tz-aware conversion)
    text          str   combined title + body, stripped
    score         float upvote score (may be NaN for comments in some archives)
    is_comment    bool  True for comments, False for submissions/posts

The two datasets have very different schemas; the loaders normalise both
to this shape so the downstream entity-linking / FinBERT pipeline is
dataset-agnostic.
"""
