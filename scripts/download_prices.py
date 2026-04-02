"""Download S&P 500 OHLCV via qtools.data.loaders.us.

Coverage: 2020-01-01 to 2025-07-31, wide enough for forward returns
over the two archive regimes (2021 meme era + 2023-06 onward).

Outputs:
  data/raw/prices.parquet            wide close-price panel (date x symbol)
  data/raw/sp500_tickers.csv         cached Wikipedia table (symbol, name, sector)
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from qtools.data.loaders.us import get_us_prices

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

START = "2020-01-01"
END = "2025-07-31"


def _load_tickers() -> pd.DataFrame:
    cache = RAW_DIR / "sp500_tickers.csv"
    if cache.exists():
        return pd.read_csv(cache)
    headers = {"User-Agent": "Mozilla/5.0 (alt-data-sentiment)"}
    html = requests.get(WIKI_URL, headers=headers, timeout=30).text
    tables = pd.read_html(StringIO(html))
    df = tables[0][["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"}
    )
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def main() -> None:
    tickers_df = _load_tickers()
    symbols = tickers_df["symbol"].tolist()
    print(f"[tickers] {len(symbols)} symbols cached at {RAW_DIR/'sp500_tickers.csv'}")

    print(f"[prices] fetching {START} to {END} via qtools.data.loaders.us ...")
    prices = get_us_prices(symbols, start=START, end=END)
    print(f"[prices] shape={prices.shape}  cols(top5)={list(prices.columns[:5])}")

    out = RAW_DIR / "prices.parquet"
    prices.to_parquet(out)
    print(f"[done] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
