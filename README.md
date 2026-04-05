# alt-data-sentiment

> **Languages**: **English** · [繁體中文](docs/README.zh-TW.md)

I wanted to know whether Reddit sentiment is actually a new cross-sectional factor or just a restatement of momentum in noisier form. The 2021 meme-stock episode made it obvious that retail forums can move prices short-term; the harder question is whether that effect survives after you control for what momentum and reversal already capture.

## Headline result

Sample: leukipp 2021 four-subreddit corpus, ~55k linked documents, 76k `(doc, ticker)` rows, 470 S&P 500 names.

Pooled rank-IC vs forward 21-day return is weak (HAC |t| < 2 across all four factor variants). The interesting structure is **per-subreddit**:

| Variant            | r/wallstreetbets | r/stocks | r/investing  | r/options |
|--------------------|-----------------:|---------:|-------------:|----------:|
| `sum_sent_x_attn`  | **+2.00**        | +0.96    | **−2.41**    | +0.15     |
| `shrinkage_sent`   | +1.94            | +0.97    | **−3.09**    | +0.17     |
| `mean_sent`        | +1.79            | +1.08    | **−2.68**    | −0.48     |

(HAC t-stats on daily rank-IC vs 21d fwd return)

r/investing is a statistically-significant contrarian signal in 2021; WSB is borderline-positive. But once you control for momentum and reversal, the per-subreddit signal collapses entirely (|t_sent| < 1.3 across all four). **Reddit sentiment is a price-momentum echo, not a new cross-sectional factor** — though the event study finds a real short-horizon meme-momentum effect on bullish mention spikes (CAR(+10) ≈ +1.52% in WSB).

## Research question

Retail-forum chatter is noisy, biased, and self-promotional. But 2021 showed it can move prices. Two things I wanted to know: is there a tradable cross-sectional sentiment factor after accounting for costs and ticker-recognition errors, and does any of its alpha survive after controlling for momentum and reversal?

The cross-subreddit design is the key check. WSB (meme crowd), r/stocks (general retail), r/investing (longer-horizon bias), and r/options (leverage-biased) should behave differently if the signal is real. If it varies by sub-population in a predictable way, that's evidence the signal tracks something genuine. If it collapses once you control for factor exposures, it's just momentum in disguise.

## Method

- **Data**: one Kaggle archive — `leukipp/reddit-finance-data`,
  2021 full year, submissions only across four subreddits (WSB, stocks,
  investing, options). S&P 500 OHLCV via `qtools.data.loaders.us`.
- **Entity linking**: regex `\$?[A-Z]{2,5}\b` → S&P 500 whitelist →
  common-word blacklist (CEO / YOLO / FY / …). Four S&P 500 tickers —
  `COO` (Cooper), `DD` (DuPont), `IT` (Gartner), `MAR` (Marriott) — are
  deliberately blacklisted because their all-caps form is dominated by
  slang / calendar usage; these four companies are invisible to this
  pipeline by design, documented in `src/alt_sentiment/entity_linking.py`.
  Precision + recall spot-checked on 100 hand-labelled posts in
  notebook 01.
- **Sentiment model**: `ProsusAI/finbert` (HuggingFace, CPU batch).
  Truncated to 512 tokens. **Scored once per document**, then fanned
  out to every ticker linked to that document (cost is O(documents),
  not O(documents × tickers)).
- **Daily factor**: four variants reported side by side to avoid
  cherry-picking a friendly formula —
  1. `sum((pos − neg) · log1p(mentions))` — combined sentiment × attention
  2. `mean(pos − neg)` — pure sentiment
  3. `log1p(doc_count)` — pure attention
  4. `mean(pos − neg) · n/(n+10)` — shrinkage toward zero for thin names
- **Evaluation**: Rank IC with **Newey-West / HAC t-stat** (daily
  signal has autocorrelation; plain t-stat overstates significance),
  p-value, quintile long-short **and long-only** (monthly rebalance,
  US_EQUITY cost), incremental OLS vs momentum + reversal, event
  study ±10 trading days on mention-spike pump events (GameStop
  2021-01 as a dedicated case study).
- **Cross-subreddit analysis**: the same factor is computed per
  subreddit in addition to the pooled version, so the WSB-vs-rest
  contrast is explicit.

## Results

**Pooled rank-IC.** Across all 55k linked documents in 2021, none of the four factor variants clears HAC |t| ≥ 2:

| Variant            | Mean rank-IC vs fwd 21d | HAC t-stat (lag 5) |
|--------------------|------------------------:|-------------------:|
| `sum_sent_x_attn`  | +0.015                  | +1.53              |
| `mean_sent`        | +0.012                  | +1.51              |
| `attn_log`         | −0.020                  | −1.07              |
| `shrinkage_sent`   | +0.014                  | +1.66              |

Direction is right (positive sentiment → positive forward returns), magnitude isn't.

**Per-subreddit decomposition** (see Headline). The contrarian r/investing sign is consistent with "long-horizon retail leans bearish on names that subsequently outperform"; r/options and r/stocks are basically noise.

**After classic-factor controls.** Per-subreddit pooled HAC OLS of `fwd21 ~ sent + mom + rev` (notebook 03):

| Subreddit         | β_sent  | t_sent | n_obs  |
|-------------------|--------:|-------:|-------:|
| r/wallstreetbets  | −0.000  | −0.22  | 11,202 |
| r/stocks          | +0.002  | +1.25  |  9,531 |
| r/investing       | +0.002  | +0.30  |  2,462 |
| r/options         | +0.001  | +0.18  |  4,438 |

The contrarian r/investing signal disappears once mom/rev are in the regression. Reddit sentiment hypes names that have already been moving — momentum already captures the same information.

**Backtests** (monthly rebalance, 5 bps per side, signal window 2021):

| Variant            | L/S Sharpe | L/S ann ret | LO Sharpe | LO ann ret |
|--------------------|-----------:|------------:|----------:|-----------:|
| `sum_sent_x_attn`  | −1.07      | −12.6%      | +1.57     | +30.3%     |
| `mean_sent`        |  …         |  …          | **+2.30** | **+45.3%** |
| `attn_log`         |  …         |  …          | +1.65     | +42.1%     |
| `shrinkage_sent`   |  …         |  …          | +1.69     | +33.2%     |

Long-only top-quintile basket beats 2021 SPX (≈ +27%) with smaller MDD, but is heavily SPX-correlated. The L/S leg loses money because the signal is so concentrated on a few hot meme names that shorting the bottom quintile creates a large negative-beta drag in a strong bull year. Any tradable edge here is on the long side — follow-through, not contrarian alpha.

**Event study** (notebook 04). 139 bullish mention spikes (mentions ≥ ticker p99 + polarity > 0) deliver mean abnormal returns of +1.17% (t+1) → +1.52% (t+10), concentrated in WSB (CAR(+5) = +1.43%) vs r/investing (+0.27%). That's a real short-horizon meme-momentum effect, not a durable cross-sectional factor.

## Layout

```
alt-data-sentiment/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── download_kaggle_wsb.py      # Kaggle archive pull
│   ├── download_prices.py          # S&P 500 OHLCV via qtools
│   └── score_sentiment.py          # FinBERT batch scorer
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_sentiment_factor.ipynb
│   ├── 03_vs_classic_factors.ipynb
│   └── 04_event_study.ipynb
├── src/alt_sentiment/
│   ├── entity_linking.py           # ticker extraction + whitelist/blacklist
│   ├── sentiment.py                # FinBERT wrapper
│   ├── factor.py                   # daily aggregation + IC + quintile helpers
│   └── loaders/
│       └── leukipp.py
├── reports/figures/
└── data/                           # gitignored
```

## Limitations

This is a single-year sample — 2021 was a meme-stock year, and the cross-subreddit contrasts are robust within 2021 but I'm not claiming they hold in 2022+. All four subreddits are bullish and US-centric; WSB especially skews toward high-beta meme names. The ticker-recognition blacklist is hand-maintained and deliberately excludes four real S&P 500 names (`COO`, `DD`, `IT`, `MAR`) whose all-caps forms are dominated by slang or calendar usage. The universe is a snapshot — stocks added or removed after the first download are handled consistently but not historically accurately. FinBERT was trained on financial news, not social slang; "to the moon 🚀" and similar WSB idioms are routinely mis-scored. Notebook 01 has a spot check on 20 typical WSB lines.

## Notebook tour

- [`01_data_quality.ipynb`](notebooks/01_data_quality.ipynb) — raw
  archive coverage by subreddit, link-rate after entity linking, top
  ticker frequencies, naive bot heuristic, FinBERT class-probability
  distribution, 10-row text spot check.
- [`02_sentiment_factor.ipynb`](notebooks/02_sentiment_factor.ipynb) —
  four factor variants × Newey-West HAC IC × monthly L/S × long-only
  × per-subreddit IC matrix. The headline subreddit-contrast result
  lives here.
- [`03_vs_classic_factors.ipynb`](notebooks/03_vs_classic_factors.ipynb) —
  pairwise correlation, monthly OLS, pooled HAC regression of fwd21 on
  sentiment + momentum + reversal. Per-subreddit incremental check.
- [`04_event_study.ipynb`](notebooks/04_event_study.ipynb) — ±10
  trading-day CAR around bullish mention spikes, per subreddit, plus a
  single-name case-study panel.

## Reproducing

```bash
conda create -n alt-data-sentiment python=3.13 -y
conda activate alt-data-sentiment
pip install -e .

# Archive (requires KAGGLE_API_TOKEN in .env, see .env.example)
python scripts/download_kaggle_wsb.py     # leukipp
python scripts/download_prices.py         # 503 S&P 500 symbols, 2020-2025

# Sentiment scoring (local CPU, ~30-60 min for leukipp)
python scripts/score_sentiment.py --batch-size 16

# Execute the four notebooks (each takes 1-3 min after scoring)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
