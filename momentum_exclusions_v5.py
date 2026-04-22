"""
momentum_exclusions.py
======================
Computes and caches a daily list of stocks to exclude from portfolio
construction based on a joint (momentum + OU mean reversion) condition.

A stock is excluded on date t if BOTH conditions hold:
  1. Rescaled short-term return > percentile threshold
     (extreme recent momentum — crowding / reversal risk)
  2. OU score < ou_threshold
     (residual price above equilibrium — not a genuine buying opportunity)

The two-stage filter was calibrated via joint (pct, k, ou_threshold) grid
search. With k* settling at p50=40 (near-hard exclusion), the practical
implementation is a binary exclude/include rather than a soft downweight.

Exclusions are cached in DB table `momentum_exclusions` (date, ticker).
A separate `momentum_exclusions_processed` table tracks dates that have
been computed (including dates with zero exclusions, to avoid recomputation).

Self-contained — no dependencies on other pipeline scripts beyond DB access
and Pxs_df / ou_scores_df inputs from the kernel.

Pipeline position
-----------------
Run after:  factor_model_v2.py  (OU scores in v2_ou_reversion_df)
Run before: run_mvo_backtest()  (exclusions consumed in candidate pre-filter)

Usage
-----
    from momentum_exclusions import (
        run_pct_analysis,
        run_exclusions,
        build_ou_scores_df,
        load_exclusions,
        get_exclusion_stats,
    )

    # Step 1 — build percentile distributions (run once or after universe change)
    pct_df = run_pct_analysis(Pxs_df, list(sectors_s.index))

    # Step 2 — load OU scores from DB
    ou_scores_df = build_ou_scores_df()

    # Step 3 — build / update exclusion cache
    run_exclusions(Pxs_df, pct_df, ou_scores_df)             # incremental (default)
    run_exclusions(Pxs_df, pct_df, ou_scores_df,
                   force_rebuild=True)                        # full rebuild

    # Step 4 — consume in run_mvo_backtest (one line before candidate selection)
    excluded = load_exclusions(dt)
    cands    = cands[~cands.index.isin(excluded)]
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================

DB_URL = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'
ENGINE = create_engine(DB_URL)

# Date range — mirrors MB_START_DATE in mvo_backtest.py
START_DATE = pd.Timestamp('2019-01-01')

# DB tables
EXCLUSIONS_TBL      = 'momentum_exclusions'
EXCLUSIONS_PROC_TBL = 'momentum_exclusions_processed'

# ── Distribution analysis parameters ─────────────────────────────────────────
LOOKBACK_1Y = 252    # trading days for short-term distribution
LOOKBACK_5Y = 1260   # trading days for long-run rescaling
WINDOWS_TD  = {'1M': 21, '2M': 42, '3M': 63}
PERCENTILES = [95, 96, 97, 98, 99]
MIN_OBS     = 30     # minimum pooled observations required

# ── Optimal calibration parameters ───────────────────────────────────────────
# From calibrate_k() with ou_scores_df and k_grid=[0,1,2,3,5,7,10,15,20,30,50,100]:
#   k* p50=40, p75=40  → near-hard exclusion
#   pct* dominant: p95 (26%), p97 (12%), p99 (14%)
#   ou_thr* dominant: OU < 0.0 (40%), OU < -0.5 (14%), OU < -1.0 (14%)
# Update these constants after each recalibration run.
OPT_PCT_THRESHOLD = 95     # percentile threshold for rescaled return
OPT_OU_THRESHOLD  = 0.0    # OU score threshold (exclude if ou_score < this)
USE_ROBUST        = False   # use median/IQR rescaling (vs mean/std)


# ==============================================================================
# DISTRIBUTION ANALYSIS HELPERS
# ==============================================================================

def _non_overlapping_returns(px_series, window_td, end_idx, n_windows):
    """Non-overlapping W-period returns ending at end_idx."""
    rets = []
    for k in range(n_windows):
        i_end   = end_idx - k * window_td
        i_start = i_end - window_td
        if i_start < 0:
            break
        p_end   = px_series.iloc[i_end]
        p_start = px_series.iloc[i_start]
        if p_start > 0 and not np.isnan(p_start) and not np.isnan(p_end):
            rets.append(p_end / p_start - 1)
    return np.array(rets)


def _pool_cross_sectional(Pxs_df, universe, date_idx, window_td, lookback):
    """Pool non-overlapping W-period returns across universe over lookback days."""
    n_windows = lookback // window_td
    all_rets  = []
    for tkr in universe:
        if tkr not in Pxs_df.columns:
            continue
        px = Pxs_df[tkr].iloc[max(0, date_idx - lookback): date_idx + 1].dropna()
        if len(px) < window_td + 1:
            continue
        px_arr   = px.values
        n_end    = len(px_arr) - 1
        tkr_rets = _non_overlapping_returns(
            pd.Series(px_arr), window_td, n_end,
            min(n_windows, n_end // window_td)
        )
        all_rets.extend(tkr_rets.tolist())
    return np.array(all_rets)


def _moments(arr):
    """Compute mean, std, median, IQR, skewness, kurtosis."""
    if len(arr) < 4:
        return dict(mean=np.nan, std=np.nan, median=np.nan,
                    iqr=np.nan, skew=np.nan, kurt=np.nan)
    return dict(
        mean   = float(np.mean(arr)),
        std    = float(np.std(arr, ddof=1)),
        median = float(np.median(arr)),
        iqr    = float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        skew   = float(scipy_stats.skew(arr)),
        kurt   = float(scipy_stats.kurtosis(arr)),
    )


def _rescale(z_scores, moments_5y, use_robust=False):
    """Un-z-score using 5Y moments."""
    if use_robust:
        loc   = moments_5y.get('median', 0.0) or 0.0
        scale = moments_5y.get('iqr',    1.0) or 1.0
    else:
        loc   = moments_5y.get('mean', 0.0) or 0.0
        scale = moments_5y.get('std',  1.0) or 1.0
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    return z_scores * scale + loc


def _compute_rescaled_return(px_series, dt_idx, window_td,
                              moments_1y, moments_5y, use_robust=False):
    """Compute a single stock's rescaled return over window_td ending at dt_idx."""
    if dt_idx < window_td:
        return np.nan
    p_end   = px_series.iloc[dt_idx]
    p_start = px_series.iloc[dt_idx - window_td]
    if p_start <= 0 or np.isnan(p_start) or np.isnan(p_end):
        return np.nan
    raw_ret = p_end / p_start - 1
    std_1y  = moments_1y.get('std',  1.0) or 1.0
    mean_1y = moments_1y.get('mean', 0.0) or 0.0
    if np.isnan(std_1y) or std_1y == 0:
        std_1y = 1.0
    z = (raw_ret - mean_1y) / std_1y
    if use_robust:
        scale = moments_5y.get('iqr',    1.0) or 1.0
        loc   = moments_5y.get('median', 0.0) or 0.0
    else:
        scale = moments_5y.get('std',  1.0) or 1.0
        loc   = moments_5y.get('mean', 0.0) or 0.0
    if np.isnan(scale) or scale == 0:
        scale = 1.0
    return float(z * scale + loc)


# ==============================================================================
# PERCENTILE DISTRIBUTION ANALYSIS
# ==============================================================================

def run_pct_analysis(Pxs_df, universe,
                     lookback_1y=LOOKBACK_1Y,
                     lookback_5y=LOOKBACK_5Y,
                     windows_td=None,
                     percentiles=None,
                     min_obs=MIN_OBS,
                     use_robust=USE_ROBUST):
    """
    Compute cross-sectional return distribution percentiles for all trading
    dates from START_DATE onwards. Used to build pct_df input for run_exclusions().

    Parameters
    ----------
    Pxs_df      : Price DataFrame (dates x tickers).
    universe    : List of tickers.
    lookback_1y : Trading days for 1Y distribution (default 252).
    lookback_5y : Trading days for 5Y rescaling (default 1260).
    windows_td  : Dict {'label': n_days}. Default: {'1M':21, '2M':42, '3M':63}.
    percentiles : List of percentile levels. Default: [95,96,97,98,99].
    min_obs     : Minimum pooled observations to compute percentiles.
    use_robust  : Use median/IQR rescaling instead of mean/std.

    Returns
    -------
    pd.DataFrame indexed by date with columns:
        p{pct}_{window}     — rescaled return percentile thresholds
        mean_1Y_{window}    — 1Y pooled mean
        std_1Y_{window}     — 1Y pooled std
        mean_5Y_{window}    — 5Y pooled mean
        std_5Y_{window}     — 5Y pooled std
        median_5Y_{window}  — 5Y pooled median
        iqr_5Y_{window}     — 5Y pooled IQR
        n_obs_1Y_{window}   — n pooled observations (1Y)
        n_obs_5Y_{window}   — n pooled observations (5Y)
    """
    if windows_td  is None: windows_td  = WINDOWS_TD
    if percentiles is None: percentiles = PERCENTILES

    universe  = [t for t in universe if t in Pxs_df.columns]
    px_dates  = Pxs_df.index
    all_dates = pd.DatetimeIndex(sorted(px_dates[px_dates >= START_DATE]))

    print(f"\n{'='*72}")
    print(f"  MOMENTUM PERCENTILE ANALYSIS")
    print(f"{'='*72}")
    print(f"  Universe : {len(universe)} stocks")
    print(f"  Dates    : {len(all_dates)} "
          f"({all_dates[0].date()} -> {all_dates[-1].date()})")
    print(f"  Windows  : {list(windows_td.keys())}")
    print(f"  Lookback : 1Y={lookback_1y}d  5Y={lookback_5y}d")
    print(f"  Rescaling: {'robust (median/IQR)' if use_robust else 'mean/std'}\n")

    records = []
    n       = len(all_dates)
    t_start = pd.Timestamp.now()

    for i, dt in enumerate(all_dates):
        if dt not in px_dates:
            prior = px_dates[px_dates <= dt]
            if prior.empty:
                continue
            dt_px = prior[-1]
        else:
            dt_px = dt

        dt_idx = px_dates.get_loc(dt_px)
        row    = {'date': dt}

        for wlbl, wtd in windows_td.items():
            pool_1y = _pool_cross_sectional(
                Pxs_df, universe, dt_idx, wtd, lookback_1y)
            m1y = _moments(pool_1y)
            row[f'n_obs_1Y_{wlbl}'] = len(pool_1y)
            row[f'mean_1Y_{wlbl}']  = m1y['mean']
            row[f'std_1Y_{wlbl}']   = m1y['std']

            pool_5y = _pool_cross_sectional(
                Pxs_df, universe, dt_idx, wtd, lookback_5y)
            m5y = _moments(pool_5y)
            row[f'n_obs_5Y_{wlbl}']  = len(pool_5y)
            row[f'mean_5Y_{wlbl}']   = m5y['mean']
            row[f'std_5Y_{wlbl}']    = m5y['std']
            row[f'median_5Y_{wlbl}'] = m5y['median']
            row[f'iqr_5Y_{wlbl}']    = m5y['iqr']

            if len(pool_1y) < min_obs:
                for p in percentiles:
                    row[f'p{p}_{wlbl}'] = np.nan
                continue

            std_1y  = m1y['std']  or 1.0
            mean_1y = m1y['mean'] or 0.0
            if np.isnan(std_1y) or std_1y == 0:
                std_1y = 1.0
            z_scores = (pool_1y - mean_1y) / std_1y
            rescaled = (_rescale(z_scores, m5y, use_robust=use_robust)
                        if len(pool_5y) >= min_obs else pool_1y)

            for p in percentiles:
                row[f'p{p}_{wlbl}'] = float(np.percentile(rescaled, p))

        records.append(row)

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = (pd.Timestamp.now() - t_start).total_seconds()
            rate    = elapsed / (i + 1)
            eta     = rate * (n - i - 1)
            p95_1m  = row.get('p95_1M', np.nan)
            print(f"  [{i+1:>4}/{n}] {dt.date()}  "
                  f"p95_1M={p95_1m:.3f}  "
                  f"elapsed={elapsed:>5.0f}s  "
                  f"eta={eta:>5.0f}s", end='\r')

    print(f"\n  Done: {len(records)} dates computed")
    df = pd.DataFrame(records).set_index('date')

    print(f"\n  Percentile thresholds (median across dates):")
    for wlbl in windows_td:
        vals = {p: df[f'p{p}_{wlbl}'].median()
                for p in percentiles if f'p{p}_{wlbl}' in df.columns}
        print(f"  {wlbl}: " +
              "  ".join(f"p{p}={v:.3f}" for p, v in vals.items()))

    return df


# ==============================================================================
# OU SCORES LOADER
# ==============================================================================

def build_ou_scores_df(model_version='v2'):
    """Load OU scores from DB, return wide DataFrame (dates x tickers)."""
    tbl = 'v2_ou_reversion_df' if model_version == 'v2' else 'ou_reversion_df'
    print(f"  Loading OU scores from {tbl}...")
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(
            f"SELECT date, ticker, ou_score FROM {tbl} ORDER BY date, ticker"
        ), conn)
    if df.empty:
        warnings.warn(f"  No OU scores found in {tbl}")
        return pd.DataFrame()
    df['date']     = pd.to_datetime(df['date'])
    df['ticker']   = df['ticker'].apply(
        lambda t: str(t).split(' ')[0].strip().upper())
    df['ou_score'] = df['ou_score'].astype(float)
    pivot = df.pivot_table(index='date', columns='ticker',
                           values='ou_score', aggfunc='last')
    print(f"  OU scores: {pivot.shape[0]} dates x {pivot.shape[1]} tickers")
    return pivot


# ==============================================================================
# DB HELPERS
# ==============================================================================

def _ensure_tables():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {EXCLUSIONS_TBL} (
                date    DATE           NOT NULL,
                ticker  VARCHAR(20)    NOT NULL,
                score   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                PRIMARY KEY (date, ticker)
            )
        """))
        # Add score column if table exists without it (migration)
        try:
            conn.execute(text(f"""
                ALTER TABLE {EXCLUSIONS_TBL}
                ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION NOT NULL DEFAULT 0.0
            """))
        except Exception:
            pass
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {EXCLUSIONS_PROC_TBL} (
                date DATE NOT NULL PRIMARY KEY
            )
        """))


def _get_processed_dates():
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                f"SELECT date FROM {EXCLUSIONS_PROC_TBL}"
            )).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_exclusions(exclusions_by_date):
    """
    exclusions_by_date: {date: dict {ticker: score}}
    Only saves rows with score > 0.
    """
    if not exclusions_by_date:
        return
    date_vals = [d.date() for d in exclusions_by_date.keys()]
    rows = []
    for dt, ticker_scores in exclusions_by_date.items():
        for tkr, score in ticker_scores.items():
            if float(score) > 0:
                rows.append({'date': dt.date(), 'ticker': str(tkr),
                             'score': float(score)})
    with ENGINE.begin() as conn:
        conn.execute(text(
            f"DELETE FROM {EXCLUSIONS_TBL} WHERE date = ANY(:dates)"
        ), {"dates": date_vals})
        if rows:
            conn.execute(text(f"""
                INSERT INTO {EXCLUSIONS_TBL} (date, ticker, score)
                VALUES (:date, :ticker, :score)
            """), rows)
    n_excl = len(rows)
    print(f"  Saved: {len(date_vals)} dates, {n_excl} total exclusions "
          f"({n_excl / max(len(date_vals), 1):.1f} avg/day)")


def _mark_processed(dates):
    if not dates:
        return
    with ENGINE.begin() as conn:
        for d in dates:
            conn.execute(text(f"""
                INSERT INTO {EXCLUSIONS_PROC_TBL} (date)
                VALUES (:d) ON CONFLICT (date) DO NOTHING
            """), {'d': d.date()})


# ==============================================================================
# CORE EXCLUSION LOGIC — SINGLE DATE
# ==============================================================================

def _compute_exclusions_for_date(dt, px_dates, Pxs_df, universe,
                                  pct_df, ou_scores_df,
                                  pct_threshold, ou_threshold,
                                  windows_td, use_robust):
    """
    Returns dict {ticker: joint_score} for all stocks meeting the joint condition,
    ranked by score descending. No cap applied here — capping is done at
    consumption time in run_mvo_backtest via MR_CAP.

    Joint condition:
      rescaled_return > p{pct_threshold}  AND  ou_score < ou_threshold

    Joint score = overshot_magnitude * ou_excess
      where overshot_magnitude = max(rescaled_return - threshold) across windows
            ou_excess           = -ou_score (higher = further above equilibrium)
    """
    avail_pct = pct_df[pct_df.index <= dt]
    if avail_pct.empty:
        return {}
    pct_row = avail_pct.iloc[-1]

    if dt in px_dates:
        dt_idx = px_dates.get_loc(dt)
    else:
        prior = px_dates[px_dates <= dt]
        if prior.empty:
            return {}
        dt_idx = px_dates.get_loc(prior[-1])

    if not ou_scores_df.empty:
        ou_avail = ou_scores_df[ou_scores_df.index <= dt]
        ou_row   = ou_avail.iloc[-1] if not ou_avail.empty else pd.Series(dtype=float)
    else:
        ou_row = pd.Series(dtype=float)

    # Build candidate list with joint scores
    candidates = {}   # tkr -> joint_score
    for tkr in universe:
        if tkr not in Pxs_df.columns:
            continue

        # Gate 1: OU check — skip if stock is below equilibrium
        ou_val = float(ou_row.get(tkr, np.nan))
        if np.isnan(ou_val) or ou_val >= ou_threshold:
            continue
        ou_excess = -ou_val   # positive: how far above equilibrium

        # Gate 2: find max overshot magnitude across windows
        max_overshot = np.nan
        for wlbl, wtd in windows_td.items():
            thr_val = pct_row.get(f'p{pct_threshold}_{wlbl}', np.nan)
            if np.isnan(thr_val):
                continue
            m1y = {'mean': pct_row.get(f'mean_1Y_{wlbl}', np.nan),
                   'std':  pct_row.get(f'std_1Y_{wlbl}',  np.nan)}
            m5y = {'mean':   pct_row.get(f'mean_5Y_{wlbl}',   np.nan),
                   'std':    pct_row.get(f'std_5Y_{wlbl}',    np.nan),
                   'median': pct_row.get(f'median_5Y_{wlbl}', np.nan),
                   'iqr':    pct_row.get(f'iqr_5Y_{wlbl}',    np.nan)}
            rr = _compute_rescaled_return(
                Pxs_df[tkr], dt_idx, wtd, m1y, m5y, use_robust)
            if not np.isnan(rr) and rr > thr_val:
                overshot = rr - thr_val
                if np.isnan(max_overshot) or overshot > max_overshot:
                    max_overshot = overshot

        if np.isnan(max_overshot):
            continue   # didn't exceed threshold on any window

        # Joint score: product of both magnitudes
        candidates[tkr] = max_overshot * ou_excess

    if not candidates:
        return {}

    # Return all candidates ranked by joint score descending
    # Capping is done at consumption time in run_mvo_backtest via MR_CAP
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_exclusions(Pxs_df, pct_df, ou_scores_df,
                   universe=None,
                   incremental=True,
                   force_rebuild=False,
                   pct_threshold=OPT_PCT_THRESHOLD,
                   ou_threshold=OPT_OU_THRESHOLD,
                   windows_td=None,
                   use_robust=USE_ROBUST):
    """
    Compute and cache momentum exclusion lists for all trading dates.

    Parameters
    ----------
    Pxs_df        : Price panel (dates x tickers).
    pct_df        : Percentile distributions from run_pct_analysis().
    ou_scores_df  : OU z-scores wide DataFrame from build_ou_scores_df().
    universe      : Tickers to consider. Default: all Pxs_df equity columns.
    incremental   : Only compute dates not already processed (default True).
    force_rebuild : Wipe cache and recompute all (default False).
    pct_threshold : Percentile threshold for rescaled return (default 95).
    ou_threshold  : OU score threshold — exclude if ou_score < this (default 0.0).
    windows_td    : Window dict. Default: {'1M':21, '2M':42, '3M':63}.
    use_robust    : Use median/IQR rescaling (default False).

    Returns
    -------
    dict {date: dict {ticker: score}} for the computed dates.
    """
    _ensure_tables()
    if windows_td is None:
        windows_td = WINDOWS_TD

    non_equity = {'USGG10YR', 'SPX', 'SPY', 'QQQ', 'VIX'}
    if universe is None:
        universe = [c for c in Pxs_df.columns if c not in non_equity]

    px_dates  = Pxs_df.index
    all_dates = pd.DatetimeIndex(sorted(px_dates[px_dates >= START_DATE]))

    if force_rebuild:
        with ENGINE.begin() as conn:
            conn.execute(text(f"DELETE FROM {EXCLUSIONS_TBL}"))
            conn.execute(text(f"DELETE FROM {EXCLUSIONS_PROC_TBL}"))
        print(f"  Cache cleared (force_rebuild=True)")
        dates_to_compute = all_dates
    elif incremental:
        processed        = _get_processed_dates()
        dates_to_compute = pd.DatetimeIndex(
            [d for d in all_dates if d not in processed])
        print(f"  Exclusions cache: {len(processed)} dates processed, "
              f"{len(dates_to_compute)} new dates to compute")
    else:
        dates_to_compute = all_dates
        print(f"  Computing all {len(dates_to_compute)} dates")

    if len(dates_to_compute) == 0:
        print("  OK — cache fully up to date")
        return {}

    print(f"  Parameters : p{pct_threshold}  OU < {ou_threshold}  "
          f"windows={list(windows_td.keys())}  universe={len(universe)}")
    print(f"  Date range : {dates_to_compute[0].date()} -> "
          f"{dates_to_compute[-1].date()}\n")

    exclusions_by_date = {}
    n          = len(dates_to_compute)
    n_excl_run = 0
    t_start    = pd.Timestamp.now()

    for i, dt in enumerate(dates_to_compute):
        excl = _compute_exclusions_for_date(
            dt=dt, px_dates=px_dates, Pxs_df=Pxs_df,
            universe=universe, pct_df=pct_df,
            ou_scores_df=ou_scores_df,
            pct_threshold=pct_threshold,
            ou_threshold=ou_threshold,
            windows_td=windows_td,
            use_robust=use_robust,
        )
        exclusions_by_date[dt] = excl
        n_excl_run += len(excl)

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = (pd.Timestamp.now() - t_start).total_seconds()
            rate    = elapsed / (i + 1)
            eta     = rate * (n - i - 1)
            print(f"  [{i+1:>4}/{n}] {dt.date()}  "
                  f"excl_today={len(excl):>2}  "
                  f"avg_excl={n_excl_run/(i+1):>4.1f}  "
                  f"elapsed={elapsed:>5.0f}s  "
                  f"eta={eta:>5.0f}s", end='\r')

    print(f"\n  Computed {len(exclusions_by_date)} dates")
    _save_exclusions(exclusions_by_date)
    _mark_processed(list(dates_to_compute))

    counts  = [len(v) for v in exclusions_by_date.values()]
    nonzero = [c for c in counts if c > 0]
    print(f"  Dates with 1+ exclusion : {len(nonzero)}/{len(counts)} "
          f"({len(nonzero)/max(len(counts),1)*100:.1f}%)")
    if nonzero:
        print(f"  Avg excluded (when >0)  : {np.mean(nonzero):.1f}  "
              f"max={max(nonzero)}")

    return exclusions_by_date


# ==============================================================================
# LOADERS
# ==============================================================================

def load_exclusions(dt, top_n=None):
    """
    Load excluded tickers for a single date, ordered by joint score descending.

    Parameters
    ----------
    dt    : date to load
    top_n : if provided, return only the top_n highest-scoring tickers.
            Useful for applying a tighter cap at rebalance time (MR_CAP in mvo_backtest).

    Returns
    -------
    dict {ticker: score} ordered by score descending.
    Empty dict if date not cached or no exclusions.
    """
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT ticker, score FROM {EXCLUSIONS_TBL}
                WHERE date = :d
                AND score > 0
                ORDER BY score DESC
            """), {'d': pd.Timestamp(dt).date()}).fetchall()
        result = {r[0]: float(r[1]) for r in rows}
        if top_n is not None and len(result) > top_n:
            result = dict(list(result.items())[:top_n])
        return result
    except Exception:
        return {}


def load_exclusions_range(start, end, top_n_per_date=None):
    """
    Load all exclusions between start and end dates (inclusive),
    ordered by score descending within each date.

    Parameters
    ----------
    top_n_per_date : if provided, cap each date at top_n highest-scoring tickers.

    Returns
    -------
    dict {date: dict {ticker: score}}
    """
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT date, ticker, score FROM {EXCLUSIONS_TBL}
                WHERE date BETWEEN :s AND :e
                ORDER BY date, score DESC
            """), conn, params={
                's': pd.Timestamp(start).date(),
                'e': pd.Timestamp(end).date()
            })
        if df.empty:
            return {}
        df['date'] = pd.to_datetime(df['date'])
        result = {}
        for dt, grp in df.groupby('date'):
            grp_sorted = grp.sort_values('score', ascending=False)
            if top_n_per_date is not None:
                grp_sorted = grp_sorted.head(top_n_per_date)
            result[pd.Timestamp(dt)] = dict(
                zip(grp_sorted['ticker'], grp_sorted['score'].astype(float)))
        return result
    except Exception:
        return {}


def get_exclusion_stats(start=None, end=None):
    """Per-date exclusion counts for diagnostics."""
    where, params = "", {}
    if start:
        where += " AND date >= :s"
        params['s'] = pd.Timestamp(start).date()
    if end:
        where += " AND date <= :e"
        params['e'] = pd.Timestamp(end).date()
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT date, COUNT(*) as n_excluded
            FROM {EXCLUSIONS_TBL} WHERE 1=1 {where}
            GROUP BY date ORDER BY date
        """), conn, params=params)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


PCT_CACHE_TBL = 'momentum_pct_cache'


def _save_pct_df(pct_df):
    """Save pct_df to DB cache, replacing existing rows for those dates."""
    if pct_df is None or pct_df.empty:
        return
    df = pct_df.reset_index().copy()
    df.columns = [c.lower().replace('%', 'pct') for c in df.columns]
    date_vals = [d.date() for d in pct_df.index]
    try:
        # Use to_sql for initial creation / append — it handles schema automatically
        if not _pct_cache_exists():
            df.to_sql(PCT_CACHE_TBL, ENGINE, if_exists='replace', index=False)
        else:
            with ENGINE.begin() as conn:
                conn.execute(text(
                    f"DELETE FROM {PCT_CACHE_TBL} WHERE date = ANY(:dates)"
                ), {"dates": date_vals})
            df.to_sql(PCT_CACHE_TBL, ENGINE, if_exists='append', index=False)
        print(f"  pct_df cache: saved {len(date_vals)} dates to {PCT_CACHE_TBL}")
    except Exception as e:
        print(f"  WARNING: _save_pct_df failed: {e}")


def _pct_cache_exists():
    try:
        with ENGINE.connect() as conn:
            return conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{PCT_CACHE_TBL}'
                )
            """)).scalar()
    except Exception:
        return False


def load_pct_df():
    """Load pct_df from DB cache. Returns empty DataFrame if not cached."""
    if not _pct_cache_exists():
        return pd.DataFrame()
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(
                f"SELECT * FROM {PCT_CACHE_TBL} ORDER BY date"
            ), conn)
        if df.empty:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        # Restore original column casing: e.g. p95_1m → p95_1M, mean_1y_1m → mean_1Y_1M
        def _restore_case(col):
            # window suffixes: 1m→1M, 2m→2M, 3m→3M
            for w in ('1m', '2m', '3m'):
                if col.endswith(f'_{w}'):
                    col = col[:-len(w)] + w.upper()
            # year indicators: 1y→1Y, 5y→5Y
            col = col.replace('_1y_', '_1Y_').replace('_5y_', '_5Y_')
            return col
        df.columns = [_restore_case(c) for c in df.columns]
        return df
    except Exception as e:
        print(f"  WARNING: load_pct_df failed: {e}")
        return pd.DataFrame()


def run_pct_analysis_incremental(Pxs_df, universe, **kwargs):
    """
    Incremental wrapper for run_pct_analysis.
    Loads existing pct_df from DB cache, computes only new dates, saves back.
    Returns the full pct_df (cached + new).
    """
    existing = load_pct_df()
    all_px_dates = Pxs_df.index
    all_dates    = pd.DatetimeIndex(sorted(
        all_px_dates[all_px_dates >= START_DATE]))

    if not existing.empty:
        last_cached = existing.index[-1]
        new_dates   = all_dates[all_dates > last_cached]
        print(f"  pct_df cache: {len(existing)} dates cached, "
              f"{len(new_dates)} new dates to compute")
        if len(new_dates) == 0:
            print("  pct_df OK — fully up to date")
            return existing
        # Restrict Pxs_df to compute only new dates by temporarily
        # overriding START_DATE via kwargs
        import copy
        orig_start = START_DATE
        # Monkey-patch module-level START_DATE for the duration of this call
        import momentum_exclusions as _me_mod
        _me_mod.START_DATE = last_cached + pd.Timedelta(days=1)
        try:
            new_pct = run_pct_analysis(Pxs_df, universe, **kwargs)
        finally:
            _me_mod.START_DATE = orig_start
        if not new_pct.empty:
            _save_pct_df(new_pct)
        return pd.concat([existing, new_pct]).sort_index()
    else:
        print("  pct_df cache empty — running full analysis...")
        pct_df = run_pct_analysis(Pxs_df, universe, **kwargs)
        _save_pct_df(pct_df)
        return pct_df


# ==============================================================================
# DAILY UPDATE — SELF-CONTAINED
# Loads all required inputs from DB, computes today's exclusions only.
# No kernel state required beyond Pxs_df and ou_scores_df.
# ==============================================================================

def run_daily_exclusions_update(Pxs_df, ou_scores_df=None, force_rebuild=False,
                                focus_list=None):
    """
    Self-contained daily update. Loads pct_df from DB cache, computes
    exclusions for any unprocessed dates only. Typically just today.

    Parameters
    ----------
    Pxs_df         : price panel (needs today's prices)
    ou_scores_df   : optional — loaded from DB if not provided
    force_rebuild  : if True, wipes all cached exclusions and recomputes
                     all dates from scratch using cached pct_df and OU scores
    focus_list     : optional list of tickers — prints last 50 dates of MR
                     scores for each, useful for tracking patterns
    """
    # Load pct_df from cache — no recomputation
    pct_df = load_pct_df()
    if pct_df.empty:
        print("  ERROR: pct_df cache empty. Run run_pct_analysis() first.")
        return

    # Load OU scores if not provided
    if ou_scores_df is None:
        ou_scores_df = build_ou_scores_df(model_version='v2')

    # Warn if OU coverage is low vs Pxs_df universe
    non_equity = {'USGG10YR', 'SPX', 'SPY', 'QQQ', 'VIX'}
    _universe_size = len([c for c in Pxs_df.columns if c not in non_equity])
    _ou_coverage   = len(ou_scores_df.columns) if not ou_scores_df.empty else 0
    if _ou_coverage < _universe_size * 0.9:
        print(f"  WARNING: OU scores only cover {_ou_coverage} tickers vs "
              f"{_universe_size} in universe ({_ou_coverage/_universe_size*100:.0f}%). "
              f"Stocks without OU scores are never flagged. "
              f"Consider passing a fuller ou_scores_df explicitly.")

    # Run exclusions
    run_exclusions(Pxs_df, pct_df, ou_scores_df,
                   incremental=not force_rebuild,
                   force_rebuild=force_rebuild)

    # Show top-50 MR scores for the last date
    last_dt = Pxs_df.index[-1]
    top50   = load_exclusions(last_dt, top_n=50)
    if top50:
        print(f"\n  Top MR scores — {last_dt.date()}  ({len(top50)} stocks with score > 0 shown, capped at 50)")
        print(f"  {'Ticker':<10}  {'Score':>8}")
        print(f"  {'─'*22}")
        for tkr, score in top50.items():
            print(f"  {tkr:<10}  {score:>8.4f}")
    else:
        print(f"\n  No exclusions scored > 0 for {last_dt.date()}")

    # Show score history for focus_list tickers
    if focus_list:
        print(f"\n  {'='*68}")
        print(f"  FOCUS LIST — MR score history (last 50 dates with any score > 0)")
        print(f"  {'='*68}")
        try:
            with ENGINE.connect() as conn:
                tickers_upper = [t.upper() for t in focus_list]
                df_focus = pd.read_sql(text(f"""
                    SELECT date, ticker, score
                    FROM {EXCLUSIONS_TBL}
                    WHERE ticker = ANY(:tickers)
                    AND score > 0
                    ORDER BY date DESC
                """), conn, params={'tickers': tickers_upper})
            if df_focus.empty:
                print("  No MR scores recorded for any ticker in focus list")
            else:
                df_focus['date'] = pd.to_datetime(df_focus['date'])
                # Pivot: dates on index, tickers on columns
                pivot = (df_focus
                         .pivot_table(index='date', columns='ticker',
                                      values='score', aggfunc='last')
                         .reindex(columns=tickers_upper)  # preserve input order
                         .sort_index(ascending=False)
                         .head(50))
                pivot = pivot.fillna(0.0)
                # Format for display
                pivot.index = pivot.index.strftime('%Y-%m-%d')
                pivot.index.name = 'Date'
                with pd.option_context('display.float_format', '{:.4f}'.format,
                                       'display.max_columns', None,
                                       'display.width', 120):
                    print(pivot.to_string())
        except Exception as e:
            print(f"  WARNING: focus_list query failed: {e}")


# Execute daily update
run_daily_exclusions_update(Pxs_df)
