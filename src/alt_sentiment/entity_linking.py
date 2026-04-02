"""Ticker extraction from free-form Reddit text.

Strategy: regex candidates -> S&P 500 whitelist -> common-word blacklist.
Precision is prioritised over recall; false positives contaminate
sentiment aggregates far worse than a missed mention.

S&P 500 tickers deliberately excluded from the blacklist (so they are
recognised in text): A, ALL, HAS, LOW, NOW, ON, SO, WELL — these are
rarely written in ALL CAPS as English words, so false positives are
acceptable in exchange for real ticker coverage.

S&P 500 tickers kept in the blacklist on purpose (accepting that these
companies will be invisible to this pipeline):

    COO  — overwhelmingly used as "Chief Operating Officer" in Reddit prose
    DD   — overwhelmingly used as "due diligence" on r/wallstreetbets
    IT   — overwhelmingly used for "information technology"
    MAR  — overwhelmingly used as the month abbreviation "Mar"

These four trade real coverage for cleaner aggregates; documented in the
README caveats section.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Tokens that match \$?[A-Z]{2,5}\b in Reddit text but are almost never
# tickers. Conservative; easier to expand than to debug false positives.
# Intentionally KEEPS the four ambiguous S&P 500 tickers documented above
# (COO / DD / IT / MAR); other S&P 500 constituents have been removed.
BLACKLIST: frozenset[str] = frozenset(
    {
        # short pronouns / articles / common English (most won't match regex
        # anyway because they are rarely written in ALL CAPS, but keeping
        # them is harmless and documents intent)
        "AN", "OR", "AT", "TO", "IS", "OF", "IN", "IF",
        "BE", "AM", "AS", "DO", "GO", "UP", "US", "WE", "HE", "NO", "OK",
        "ANY", "FOR", "NOT", "BUT", "CAN", "ONE", "TWO", "WAY", "NEW",
        "OUR", "OUT", "HOW", "WHY", "WHO", "HIS", "HER", "HAD",
        "GET", "GOT", "OWN", "SEE", "USE", "PUT", "DAY", "END", "LOT", "MAY",
        "TOP", "BIG", "OFF", "MUCH", "MORE", "MOST", "SOME", "ONLY",
        "JUST", "EVEN", "ALSO", "GOOD", "BEST", "MAKE", "SAID",
        "BACK", "OVER", "LIKE", "THAT", "THIS", "WITH", "WHAT", "WHEN",
        "YEAR", "TIME", "NEXT", "LAST", "WEEK", "STILL", "GONNA", "WANNA",
        "DONT", "CANT", "WONT", "ISNT", "ITS", "HES", "SHES",
        # WSB / finance slang that reliably hits all-caps
        "DD",          # "due diligence" — intentionally drops DuPont ticker
        "IT",          # "information technology" — drops Gartner ticker
        "COO",         # "chief operating officer" — drops Cooper Companies
        "WSB", "YOLO", "FOMO", "EV", "OP", "PT", "ATH", "ATL", "ITM",
        "OTM", "IV", "HV", "SL", "TP", "CE", "PE", "MC", "CC", "HOLD",
        # generic finance / macro acronyms
        "USD", "EUR", "GBP", "JPY", "CEO", "CFO", "CTO", "IPO", "ETF",
        "EPS", "PB", "ROE", "ROI", "ROIC", "FCF", "GDP", "CPI", "PPI",
        "PMI", "FOMC", "FED", "SEC", "IRS", "IRA", "SSN", "LLC", "INC",
        "NYSE", "NASDAQ", "SPY", "QQQ", "DIA", "IWM", "VIX",
        "FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2", "YOY", "YTD", "MOM", "QOQ",
        # calendars — MAR kept because month usage dominates Marriott ticker
        "JAN", "FEB", "MAR", "APR", "JUN", "JUL", "AUG", "SEP",
        "OCT", "DEC", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
        # loud interjections
        "LOL", "LMAO", "WTF", "OMG", "IMO", "IMHO", "TLDR", "AFAIK", "IIRC",
        "EDIT", "UPDATE", "NEWS", "POST", "LINK", "HERE", "REAL",
    }
)

CANDIDATE_RE = re.compile(r"\$?([A-Z]{2,5})\b")


def load_sp500_whitelist(tickers_csv: str | Path) -> frozenset[str]:
    """Load the S&P 500 symbol whitelist from the cached Wikipedia table."""
    df = pd.read_csv(tickers_csv)
    return frozenset(df["symbol"].str.upper().str.replace(".", "-", regex=False))


def extract_tickers(text: str, whitelist: frozenset[str]) -> list[str]:
    """Extract S&P 500 tickers from text.

    A match must:
      1. Match regex (2-5 uppercase letters, optional leading $)
      2. Be in the S&P 500 whitelist
      3. Not be in the blacklist
    Duplicates within one text are de-duplicated; order of first occurrence
    is preserved.
    """
    if not isinstance(text, str) or not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in CANDIDATE_RE.finditer(text):
        sym = m.group(1)
        if sym in BLACKLIST:
            continue
        if sym not in whitelist:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def extract_tickers_batch(
    texts: Iterable[str], whitelist: frozenset[str]
) -> list[list[str]]:
    """Batch variant — returns one ticker list per input text."""
    return [extract_tickers(t, whitelist) for t in texts]
