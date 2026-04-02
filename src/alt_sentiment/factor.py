"""Sentiment-factor aggregation, IC, and quintile backtest helpers.

Design decisions:
- A post's **trade_date** is the NYSE session the bet would land on:
  * convert created_utc -> America/New_York
  * if wall-clock <= 16:00 ET and the day is a trading day: same day
  * otherwise: next trading day (CustomBusinessDay with USFederalHolidayCalendar)
  This is a conservative approximation — it treats regional / bank holidays
  (e.g. Good Friday) correctly but does not model early closes.
- **Four factor variants** are reported side by side to avoid cherry-picking
  a single friendly formula.
- **Newey-West / HAC** t-stat on the daily rank-IC series to deflate the
  serial correlation inherent in a daily signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def assign_trade_date(
    created_utc: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Map each post timestamp to the NYSE session it rolls onto.

    Rule:
      - created in ET, if hour < 16 and that date is a trading day -> same day
      - else -> next trading day

    `trading_dates` must be the calendar of *actual* NYSE sessions (e.g. the
    sorted unique index of the wide close-price panel). This avoids the
    well-known mismatch between `USFederalHolidayCalendar` and NYSE: NYSE
    closes on Good Friday and is open on Columbus Day / Veterans Day,
    neither of which a federal calendar gets right.

    Returns a tz-naive Series of date-only Timestamps. Rows whose roll-on
    date falls past the last available session are filled with NaT.
    """
    s = pd.to_datetime(created_utc, errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    et = s.dt.tz_convert("America/New_York")
    same_day_candidate = et.dt.normalize().dt.tz_localize(None)
    before_close = (et.dt.hour < 16).values

    cal = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_dates).normalize())))
    cal_arr = cal.values
    base_arr = same_day_candidate.values

    # searchsorted("left") gives the index of the smallest session >= base.
    # If that session equals base, base is itself a trading day.
    pos = np.searchsorted(cal_arr, base_arr, side="left")
    # Clip to keep indices valid; we filter to NaT below.
    pos_clipped = np.clip(pos, 0, len(cal_arr) - 1)
    is_session_same_day = (pos < len(cal_arr)) & (cal_arr[pos_clipped] == base_arr)

    # Same-session rule: same day if before close AND is a session;
    # otherwise roll forward to the next session.
    same_mask = before_close & is_session_same_day
    next_pos = np.where(
        is_session_same_day,
        pos + 1,           # base is itself a session, so "next" is +1
        pos,               # base isn't a session, "next" is already pos
    )
    chosen_pos = np.where(same_mask, pos, next_pos)
    valid = chosen_pos < len(cal_arr)

    out_vals = np.full(len(base_arr), np.datetime64("NaT"), dtype="datetime64[ns]")
    out_vals[valid] = cal_arr[chosen_pos[valid]]
    return pd.Series(out_vals, index=created_utc.index)


# ---------------------------------------------------------------------------
# Four factor variants
# ---------------------------------------------------------------------------


def factor_variants(
    scored_long: pd.DataFrame,
    shrinkage_k: float = 10.0,
) -> dict[str, pd.DataFrame]:
    """Aggregate (doc, ticker) rows into daily per-ticker signal wide tables.

    Input columns (required):
        trade_date   pd.Timestamp (tz-naive date)
        ticker       str
        score_pos    float
        score_neg    float
    Optional:
        score_neu    float (ignored)

    Returns a dict of wide DataFrames keyed by variant name:
        "sum_sent_x_attn"     :  sum((pos-neg) * log1p(mentions)) per day
        "mean_sent"           :  mean(pos - neg) per day
        "attn_log"            :  log1p(doc_count) per day
        "shrinkage_sent"      :  mean(pos-neg) * n/(n+k) per day

    Each wide DataFrame is indexed by trade_date, columns are tickers.
    """
    df = scored_long.copy()
    df["polarity"] = df["score_pos"] - df["score_neg"]
    grp = df.groupby(["trade_date", "ticker"])
    per_day = grp.agg(
        polarity_sum=("polarity", "sum"),
        polarity_mean=("polarity", "mean"),
        n=("polarity", "size"),
    ).reset_index()
    per_day["mentions_log1p"] = np.log1p(per_day["n"])
    per_day["sum_sent_x_attn"] = per_day["polarity_sum"] * per_day["mentions_log1p"]
    per_day["mean_sent"] = per_day["polarity_mean"]
    per_day["attn_log"] = per_day["mentions_log1p"]
    per_day["shrinkage_sent"] = (
        per_day["polarity_mean"] * per_day["n"] / (per_day["n"] + shrinkage_k)
    )

    out: dict[str, pd.DataFrame] = {}
    for name in ("sum_sent_x_attn", "mean_sent", "attn_log", "shrinkage_sent"):
        wide = per_day.pivot(index="trade_date", columns="ticker", values=name)
        out[name] = wide.sort_index()
    return out


# ---------------------------------------------------------------------------
# Forward returns + cross-sectional rank IC
# ---------------------------------------------------------------------------


def forward_returns(prices_long: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Return a wide DataFrame of forward simple returns.

    prices_long columns: ['date', 'symbol', 'close'] (others ignored).
    Missing dates are assumed absent (no reindex to a business-day grid).
    """
    wide = prices_long.pivot(index="date", columns="symbol", values="close").sort_index()
    fwd = wide.shift(-horizon_days) / wide - 1.0
    return fwd


def rank_ic_per_day(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman rank correlation of signal vs fwd
    return. Critically, on each date BOTH series are masked to the paired
    subset before ranking — otherwise sparse signals (typical for daily
    sentiment) get a biased correlation because mean / variance of the
    fwd-return ranks are computed across unmentioned tickers."""
    common_cols = signal.columns.intersection(fwd.columns)
    s, f = signal[common_cols], fwd[common_cols]
    common_idx = s.index.intersection(f.index)
    s, f = s.loc[common_idx], f.loc[common_idx]

    # Apply a paired mask so unsignal columns drop out of BOTH series before
    # ranking. After this, NaN cells are aligned, and rank/mean/var are
    # computed over the paired subset on each row.
    paired_mask = s.notna() & f.notna()
    s_paired = s.where(paired_mask)
    f_paired = f.where(paired_mask)

    s_ranked = s_paired.rank(axis=1)
    f_ranked = f_paired.rank(axis=1)
    mean_s = s_ranked.mean(axis=1)
    mean_f = f_ranked.mean(axis=1)
    num = ((s_ranked.sub(mean_s, axis=0)) * (f_ranked.sub(mean_f, axis=0))).sum(axis=1)
    denom = np.sqrt(
        ((s_ranked.sub(mean_s, axis=0)) ** 2).sum(axis=1)
        * ((f_ranked.sub(mean_f, axis=0)) ** 2).sum(axis=1)
    )
    ic = num / denom.replace(0, np.nan)
    valid = paired_mask.sum(axis=1)
    return ic.where(valid >= 20)


# ---------------------------------------------------------------------------
# Newey-West / HAC t-stat
# ---------------------------------------------------------------------------


def newey_west_tstat(x: pd.Series, lags: int = 5) -> tuple[float, float]:
    """Return (mean, HAC t-stat) for a stationary series.

    Bartlett weights. Used on daily IC to correct for autocorrelation.
    """
    x = x.dropna().values
    n = len(x)
    if n < 10:
        return float("nan"), float("nan")
    mu = x.mean()
    resid = x - mu
    gamma0 = (resid @ resid) / n
    var = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1)
        cov = (resid[lag:] @ resid[:-lag]) / n
        var += 2 * w * cov
    se = np.sqrt(max(var, 0) / n)
    tstat = mu / se if se > 0 else float("nan")
    return float(mu), float(tstat)


# ---------------------------------------------------------------------------
# Quintile long-short and long-only, with per-trade cost
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    daily_pnl: pd.Series
    cum_pnl: pd.Series
    turnover: pd.Series
    annualised_return: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float


def _quintile_weights(signal_row: pd.Series, n_quintiles: int = 5) -> pd.Series:
    """Return per-stock long/short weights: +1/top quintile, -1/bottom, 0 else.
    Weights sum to 0 (market-neutral)."""
    ranks = signal_row.rank(pct=True)
    top = ranks >= (1 - 1 / n_quintiles)
    bot = ranks <= (1 / n_quintiles)
    n_top = top.sum(); n_bot = bot.sum()
    w = pd.Series(0.0, index=signal_row.index)
    if n_top > 0: w[top] = 1.0 / n_top
    if n_bot > 0: w[bot] = -1.0 / n_bot
    return w


def _long_only_weights(signal_row: pd.Series, n_quintiles: int = 5) -> pd.Series:
    ranks = signal_row.rank(pct=True)
    top = ranks >= (1 - 1 / n_quintiles)
    n_top = top.sum()
    w = pd.Series(0.0, index=signal_row.index)
    if n_top > 0: w[top] = 1.0 / n_top
    return w


def _run_backtest(
    weight_wide: pd.DataFrame,
    fwd_1d: pd.DataFrame,
    cost_bps: float,
) -> BacktestResult:
    """Given per-period position weights and 1-day forward returns on the
    same grid, compute daily pnl net of one-way transaction cost at the
    specified bps."""
    common_cols = weight_wide.columns.intersection(fwd_1d.columns)
    w = weight_wide[common_cols].fillna(0.0)
    r = fwd_1d[common_cols].reindex(w.index).fillna(0.0)
    gross = (w * r).sum(axis=1)
    # One-way turnover = 0.5 * sum(|w_t - w_{t-1}|); cost per unit turnover =
    # cost_bps * 1e-4, applied as round-trip on turnover already (turnover is
    # absolute change, so cost is proportional)
    dw = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())
    cost = dw * cost_bps * 1e-4
    net = gross - cost
    cum = (1 + net).cumprod() - 1
    # Annualised
    ann_ret = (1 + net.mean()) ** 252 - 1 if len(net) else 0.0
    ann_std = net.std() * np.sqrt(252) if len(net) else np.nan
    sharpe = (net.mean() * 252) / ann_std if ann_std and ann_std > 0 else np.nan
    # MDD on cumulative pnl curve
    running_max = cum.cummax()
    dd = (cum - running_max)
    mdd = float(dd.min()) if len(dd) else np.nan
    return BacktestResult(
        daily_pnl=net,
        cum_pnl=cum,
        turnover=dw,
        annualised_return=float(ann_ret),
        sharpe=float(sharpe),
        max_drawdown=mdd,
        avg_turnover=float(dw.mean()),
    )


def _resolve_rebalance_dates(
    signal: pd.DataFrame, rebalance: str
) -> list[pd.Timestamp]:
    """Pick the **actual last available signal date** in each resample bin.

    `signal.resample("ME").last()` labels rows with the calendar month-end
    (which is often a weekend / holiday) even though the row's data comes
    from the last in-bin trading day. Filtering by `d in signal.index`
    therefore drops months whose calendar end is non-trading (5 of 12 in
    2021). Instead, group by period and pick the last index date with any
    non-null signal in that group.
    """
    if signal.empty:
        return []
    # Map each row to its resample bin label, then find the max in-bin date
    # that has at least one non-null value.
    have_signal = signal.notna().any(axis=1)
    grouper = pd.Grouper(level=0, freq=rebalance)
    last_dates = (
        signal.index.to_series()
        .where(have_signal)
        .groupby(grouper)
        .max()
        .dropna()
    )
    return [pd.Timestamp(d) for d in last_dates.values]


def _build_position_wide(
    signal: pd.DataFrame,
    fwd_1d: pd.DataFrame,
    rebalance: str,
    weight_fn,
    n_quintiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the per-day position panel, clipped to the signal sample window.

    Returns (positions_wide, fwd_window). Both share the same index, which
    runs from the first rebalance date to the last fwd_1d date that is
    still within the signal-period window. This keeps the L/S backtest
    bounded to the period the signal actually covers (Codex review P1)."""
    rebal_dates = _resolve_rebalance_dates(signal, rebalance)
    pos_at_rebal = {
        d: weight_fn(signal.loc[d].dropna(), n_quintiles) for d in rebal_dates
    }
    if not pos_at_rebal:
        empty = fwd_1d.iloc[:0]
        return empty, empty
    pos_df = pd.DataFrame(pos_at_rebal).T.sort_index()

    first_rebal = pos_df.index.min()
    last_signal = signal.dropna(how="all").index.max()

    # Clip fwd_1d to [first_rebal, last_signal] so we only earn returns
    # over the period the signal actually covers, not 2020-2025.
    fwd_window = fwd_1d.loc[(fwd_1d.index >= first_rebal) & (fwd_1d.index <= last_signal)]
    pos_wide = pos_df.reindex(fwd_window.index, method="ffill").fillna(0.0)
    return pos_wide, fwd_window


def quintile_long_short(
    signal: pd.DataFrame,
    fwd_1d: pd.DataFrame,
    cost_bps: float = 5.0,
    rebalance: str = "ME",
    n_quintiles: int = 5,
) -> BacktestResult:
    """Monthly-rebalanced L/S top-quintile minus bottom-quintile backtest.

    signal wide: rows = trade_date, cols = ticker.
    fwd_1d  wide: rows = trade_date, cols = ticker; each cell is the NEXT
                  day's simple return. The function clips the daily return
                  grid to the signal window so the reported PnL is over
                  the actual signal period, not the full priced span.
    """
    pos_wide, fwd_window = _build_position_wide(
        signal, fwd_1d, rebalance, _quintile_weights, n_quintiles
    )
    return _run_backtest(pos_wide, fwd_window, cost_bps)


def long_only_top_quintile(
    signal: pd.DataFrame,
    fwd_1d: pd.DataFrame,
    cost_bps: float = 5.0,
    rebalance: str = "ME",
    n_quintiles: int = 5,
) -> BacktestResult:
    """Long-only variant: top quintile only, clipped to the signal window."""
    pos_wide, fwd_window = _build_position_wide(
        signal, fwd_1d, rebalance, _long_only_weights, n_quintiles
    )
    return _run_backtest(pos_wide, fwd_window, cost_bps)
