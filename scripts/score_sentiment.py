"""FinBERT sentiment scoring pipeline.

Per Codex pre-M3 review:
  - Scoring is per *document* (O(docs)), not per (doc, ticker) (O(docs·tickers)).
  - After scoring, the same (pos, neg, neu) is fanned out to every S&P 500
    ticker linked to that document.
  - Entity linking runs BEFORE FinBERT; documents with zero linked tickers
    are skipped (they contribute nothing to any per-ticker factor).

Run modes:
    --benchmark N           Score first N linked docs only; print throughput and exit.
    (default, full run)     Stream the archive in chunks, write parquet parts to
                            data/processed/scored_leukipp/part_XXXXX.parquet.
                            Resumable — on restart, skips chunks already written.

Output schema (one row per (doc, ticker)):
    doc_id, subreddit, created_utc, ticker, is_comment, doc_score,
    score_pos, score_neg, score_neu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Windows Traditional-Chinese cp950 stdout will choke on math symbols in
# status prints. Force UTF-8 before anything prints.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
except Exception:
    pass

import pandas as pd

from alt_sentiment import _ssl_fix  # noqa: F401  (SSL cert path repair, must be first)
from alt_sentiment.entity_linking import extract_tickers_batch, load_sp500_whitelist
from alt_sentiment.loaders import leukipp
from alt_sentiment.sentiment import load_finbert, score_batch

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_ROOT = REPO_ROOT / "data" / "raw" / "kaggle"
TICKERS_CSV = REPO_ROOT / "data" / "raw" / "sp500_tickers.csv"
PROCESSED = REPO_ROOT / "data" / "processed"
DATASET = "leukipp"


def _score_chunk(
    chunk: pd.DataFrame,
    whitelist,
    tokenizer,
    model,
    batch_size: int,
) -> pd.DataFrame:
    """Entity-link, FinBERT-score, fan out to (doc, ticker) rows.

    Returns a DataFrame in the output schema, possibly empty.
    """
    tickers_per_doc = extract_tickers_batch(chunk["text"].tolist(), whitelist)
    chunk = chunk.copy()
    chunk["_tickers"] = tickers_per_doc

    linked = chunk[chunk["_tickers"].str.len() > 0].reset_index(drop=True)
    if linked.empty:
        return pd.DataFrame(
            columns=[
                "doc_id", "subreddit", "created_utc", "ticker",
                "is_comment", "doc_score", "score_pos", "score_neg", "score_neu",
            ]
        )

    scores = score_batch(
        linked["text"].tolist(), tokenizer, model, batch_size=batch_size
    )
    # Attach scores to linked (one row per doc)
    linked["score_pos"] = scores[:, 0]
    linked["score_neg"] = scores[:, 1]
    linked["score_neu"] = scores[:, 2]

    # Fan out: one row per (doc, ticker)
    fanned = linked.explode("_tickers", ignore_index=True).rename(
        columns={"_tickers": "ticker"}
    )
    return fanned[
        [
            "doc_id", "subreddit", "created_utc", "ticker",
            "is_comment", "score", "score_pos", "score_neg", "score_neu",
        ]
    ].rename(columns={"score": "doc_score"})


def run_benchmark(n: int, batch_size: int) -> None:
    whitelist = load_sp500_whitelist(TICKERS_CSV)
    print(f"[bench] loading FinBERT ...")
    t0 = time.time()
    tokenizer, model = load_finbert()
    print(f"[bench] FinBERT loaded in {time.time()-t0:.1f}s")

    # Collect first n LINKED docs (docs with ≥1 S&P 500 ticker)
    print(f"[bench] streaming leukipp until {n} linked docs reached ...")
    collected_rows = 0
    scanned_rows = 0
    accumulated = []
    for chunk in leukipp.iter_docs(KAGGLE_ROOT):
        scanned_rows += len(chunk)
        tickers = extract_tickers_batch(chunk["text"].tolist(), whitelist)
        chunk = chunk.copy()
        chunk["_tickers"] = tickers
        linked = chunk[chunk["_tickers"].str.len() > 0]
        take = min(len(linked), n - collected_rows)
        accumulated.append(linked.head(take))
        collected_rows += take
        if collected_rows >= n:
            break
    bench_df = pd.concat(accumulated, ignore_index=True)
    print(f"[bench] scanned {scanned_rows:,} rows; linked docs for bench = {len(bench_df):,}")
    print(f"[bench] link rate this slice = {len(bench_df)/max(scanned_rows,1):.3%}")

    # Warm-up FinBERT (first batch is always slower due to graph compile)
    _ = score_batch(bench_df["text"].head(batch_size).tolist(), tokenizer, model,
                    batch_size=batch_size)

    # Timed pass
    t0 = time.time()
    scores = score_batch(bench_df["text"].tolist(), tokenizer, model,
                         batch_size=batch_size)
    dt = time.time() - t0
    rate = len(bench_df) / dt if dt > 0 else float("inf")
    print(f"[bench] scored {len(bench_df):,} docs in {dt:.1f}s -> {rate:.1f} docs/s")


def run_full(
    batch_size: int,
    chunk_size: int | None,
    max_chunks: int | None,
) -> None:
    whitelist = load_sp500_whitelist(TICKERS_CSV)
    out_dir = PROCESSED / f"scored_{DATASET}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: count existing parts
    existing = sorted(out_dir.glob("part_*.parquet"))
    start_idx = len(existing)
    if start_idx:
        print(f"[full] resuming: {start_idx} parts already written in {out_dir}")

    print("[full] loading FinBERT ...")
    tokenizer, model = load_finbert()

    t_run = time.time()
    total_docs_written = 0
    chunks_done_this_run = 0
    for chunk_i, chunk in enumerate(leukipp.iter_docs(KAGGLE_ROOT)):
        if chunk_i < start_idx:
            continue
        if max_chunks is not None and chunks_done_this_run >= max_chunks:
            print(
                f"[full] hit --max-chunks {max_chunks}; exiting so caller can "
                f"restart python (RAM hygiene)."
            )
            break
        t0 = time.time()
        out = _score_chunk(chunk, whitelist, tokenizer, model, batch_size)
        if not out.empty:
            out.to_parquet(out_dir / f"part_{chunk_i:05d}.parquet", index=False)
            total_docs_written += out["doc_id"].nunique()
        dt = time.time() - t0
        print(
            f"[full] chunk {chunk_i:04d}  in={len(chunk):>6}  "
            f"linked_docs~{out['doc_id'].nunique() if not out.empty else 0:>5}  "
            f"rows_out={len(out):>6}  {dt:.1f}s",
            flush=True,
        )
        chunks_done_this_run += 1

    print(
        f"[done-this-run] {total_docs_written:,} unique linked docs "
        f"in {chunks_done_this_run} chunks; wallclock {(time.time()-t_run)/60:.1f} min",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=int, default=0,
                   help="If >0: score first N LINKED docs and print throughput; exit.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Process at most N chunks this invocation (RAM hygiene). "
        "Resume state is picked up from existing part files.",
    )
    args = p.parse_args()

    if args.benchmark:
        run_benchmark(args.benchmark, args.batch_size)
    else:
        run_full(args.batch_size, args.chunk_size, args.max_chunks)


if __name__ == "__main__":
    main()
