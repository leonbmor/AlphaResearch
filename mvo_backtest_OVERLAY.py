#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
mvo_backtest.py
===============
Runs three portfolios in parallel and compares performance:
  1. Baseline      -- quality factor only (from primary_factor_backtest.py)
  2. Pure Alpha    -- composite alpha signal, equal-weight + concentration
  3. MVO           -- composite alpha signal, mean-variance optimized weights

MVO uses an ensemble of four covariance matrices (Empirical EWMA,
Ledoit-Wolf, Factor-driven XFX', PCA) with Grinold-Kahn alpha scaling,
eligibility filtering (min matrix count), and floor/cap weight constraints.

User prompts mirror composite_backtest.py exactly.
MVO-specific parameters are function arguments (not prompts).

Assumes all factor_model_step1, quality_factor, primary_factor_backtest,
factor_ic_study, portfolio_risk_decomp, mvo_diagnostics functions are
loaded in the Jupyter kernel.

Entry point
-----------
    results = run_mvo_backtest(
        Pxs_df, sectors_s, weights_by_year, regime_s,
        volumeTrd_df=None,
        # MVO parameters:
        ic=0.04,
        max_weight=0.10,
        min_weight=0.025,
        zscore_cap=2.5,
        min_matrix_count=2,
        pca_var_threshold=0.65,
        universe_mult=5,
        risk_aversion=1.0,
        force_rebuild_cache=True,
    )

Returns
-------
    dict with nav_baseline, nav_alpha, nav_mvo,
         port_baseline, port_alpha, port_mvo,
         and intermediate data
"""

import warnings
import sys
import os
import json
import hashlib
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import spearmanr
import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, text

# Wrapper for momentum exclusions (populated by momentum_exclusions.py pipeline step)
def _load_momentum_exclusions(dt, top_n=None):
    try:
        return load_exclusions(dt, top_n=top_n)
    except Exception:
        return {}


def _load_mr_scores_by_date(start_dt=None, end_dt=None):
    """
    Load all MR scores from momentum_exclusions table grouped by date.
    Returns {date: pd.Series({ticker: score})} for dates with score > 0.
    Used to build the per-date penalty map for MR_K soft penalty mode.
    """
    try:
        where = "WHERE score > 0"
        params = {}
        if start_dt is not None:
            where += " AND date >= :sd"
            params['sd'] = pd.Timestamp(start_dt).date()
        if end_dt is not None:
            where += " AND date <= :ed"
            params['ed'] = pd.Timestamp(end_dt).date()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT date, ticker, score FROM momentum_exclusions
                {where} ORDER BY date, score DESC
            """), params).fetchall()
        result = {}
        for r in rows:
            dt_ = pd.Timestamp(r[0])
            result.setdefault(dt_, {})[r[1]] = float(r[2])
        return {dt_: pd.Series(d) for dt_, d in result.items()}
    except Exception:
        return {}

warnings.filterwarnings('ignore')

# ── Database connection ────────────────────────────────────────────────────────
_DB_URL = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'
try:
    ENGINE = create_engine(_DB_URL, pool_pre_ping=True)
    with ENGINE.connect() as _test_conn:
        pass
except Exception as _e:
    print(f"  WARNING: DB connection failed: {_e}")
    ENGINE = None

# ===============================================================================
# PARAMETERS  (all tuneable constants in one place)
# ===============================================================================

# ── Backtest universe / rebalancing ───────────────────────────────────────────
MB_START_DATE  = pd.Timestamp('2019-01-01')
MB_TOP_N       = 20       # default number of stocks
MB_REBAL_FREQ  = 30       # default rebalance frequency (calendar days)
MB_MODEL_VER   = 'v2'

# ── Portfolio / position limits ───────────────────────────────────────────────
AUM              = 5_000_000   # default AUM ($5M)
TRADING_COST_BPS = 10          # one-way cost (bps)
VOLUME_WINDOW    = 10          # rolling window for volume de-trending
ADV_WINDOW       = 20          # days for median ADVP calculation
VOL_LOOKBACK     = 63          # rolling window for vol filter

# ── MVO covariance estimation ─────────────────────────────────────────────────
MVO_LOOKBACK         = 252    # return history for cov estimation (days)
MVO_EWMA_HL          = 126    # EWMA half-life for covariance
MVO_PCA_VAR_THRESH   = 0.65   # PCA variance explained threshold
MVO_DEFAULT_IC       = 0.04   # default IC for Grinold-Kahn alpha scaling
MVO_OVERLAP_TARGET   = 0.65   # target portfolio overlap across cov matrices
MVO_MAX_WEIGHT       = 0.10   # hard weight cap (single name)
MVO_ZSCORE_CAP       = 2.50   # composite z-score winsorisation cap
MVO_MIN_MATRIX_COUNT = 2      # stock must appear in >= N cov matrices
MVO_MIN_WEIGHT       = 0.025  # floor on non-zero single-name weight

# ── Smart hybrid drawdown thresholds ─────────────────────────────────────────
SH_DD_ALPHA       = 0.075  # enter hybrid below this drawdown
SH_DD_HYBRID      = 0.175  # enter MVO below this drawdown
SH_DD_EXIT_ALPHA  = 0.050  # recovery needed to exit hybrid → alpha
SH_DD_EXIT_HYBRID = 0.150  # recovery needed to exit MVO → hybrid
SH_PERSIST_DAYS   = 3      # days signal must persist before regime switch

# ── Dynamic rebalancing triggers ──────────────────────────────────────────────
DYN_TO_THRESHOLD_ALPHA  = 0.25   # turnover trigger in alpha regime
DYN_TO_THRESHOLD_HYBRID = 0.30   # turnover trigger in hybrid regime
DYN_TO_THRESHOLD_MVO    = 0.35   # turnover trigger in MVO regime
DYN_VOLDIFF_CAP         = 0.175  # max vol increase alongside TO trigger
DYN_VOLDIFF_DERISK      = -0.750 # vol de-risk trigger (effectively disabled)
DYN_MIN_HOLD_DAYS       = 10     # minimum days between rebalances

# ── Drawdown policy (strategy 8: Dyn + Hedge + DD) ───────────────────────────
# Each tuple: (dd_threshold, fraction_of_remaining_to_cut)
DD_LEVELS = [
    (0.175, 2/5),  # -17.5%: cut 40% of remaining → 60% exposed
    (0.300, 3/7),  # -30.0%: cut 43% of remaining → 34% exposed
    (0.350, 1/2),  # -35.0%: cut 50% of remaining → 17% exposed
    (0.400, 2/3),  # -40.0%: cut 67% of remaining →  6% exposed
    (0.450, 1/1),  # -45.0%: cut 100%              →  0% exposed
]
# Regime forced by each DD level (index matches DD_LEVELS)
# All DD levels fire at or below SH_DD_HYBRID=-17.5% which is already MVO territory
DD_LEVEL_REGIME = ['mvo', 'mvo', 'mvo', 'mvo', 'mvo']
DD_REENTRY_PCT      = 0.075
DD_REENTRY_CONFIRM  = 5
DD_ANNUAL_RESET_PCT = 0.30

# ── Strategy 9 exclusion filter ───────────────────────────────────────────────
# MR_CAP=0.0  → no exclusions → strategy 9 = strategy 8 (convergence check)
# MR_CAP=0.025 → top 2.5% overextended names excluded (default)
MR_CAP = 0.0

# ── MR momentum penalty (soft penalty mode) ───────────────────────────────────
# Divides Idio_Mom and Mom_12M1 z-scores by exp(MR_K * MR_score) per stock.
# MR_K=0.0  → no penalty (standard composite)
# MR_K=1.0  → full exp(score) penalty
# MR_K=0.5  → gentler curve (default starting point)
MR_K = 0.5   # set > 0 to activate soft penalty mode

# ── Quality weight floor ───────────────────────────────────────────────────────
# Sets a minimum share for Quality within the {Idio_Mom + Mom_12M1 + Quality}
# sub-sum, leaving Value untouched. When floor binds, momentum factors are
# reduced proportionally (preserving their ratio to each other).
# QUALITY_FLOOR=0.0 → inactive, weights unchanged
# QUALITY_FLOOR=0.5 → Quality gets at least 50% of the mom+quality sub-sum
QUALITY_FLOOR = 0.0   # set to e.g. 0.5 to activate
MOM_FILTER    = 3     # alpha/hybrid candidate pool = MOM_FILTER x top_n by composite momentum
TIER2_T       = 0     # total stocks incl overlay (0 = disabled, user sets at runtime)
TIERONE_ALLOC = 0.85  # fraction allocated to tier 1 core portfolio
CORR_FILTER   = 0.50  # fraction of non-core universe kept after correlation filter
MOM_6M1_WIN   = 126   # 6M1 momentum lookback (trading days)
MOM_6M1_SKIP  = 21    # 6M1 momentum skip

# ── Hedge engine ──────────────────────────────────────────────────────────────
BETA_WINDOW    = 63     # rolling window for beta calculation
CORR_WINDOW    = 63     # rolling window for correlation ranking
EFF_MAV_WINDOW = 20     # MAV window for smoothing effectiveness
EFF_FLOOR      = 0.75   # minimum effectiveness score to qualify
CORR_FLOOR     = 0.50   # minimum correlation to portfolio to qualify
HEDGE_RATIO    = 0.25   # hedge size per instrument (fraction of NAV)
MAX_HEDGE      = 0.50   # maximum total hedge (fraction of NAV)
TRIGGER_ASSETS = ['QQQ', 'SPY']  # assets that trigger hedge on/off

# ── Cache / DB table names ────────────────────────────────────────────────────
MB_COV_CACHE_TBL  = 'mvo_cov_cache'
MB_X_CACHE_TBL    = 'mvo_x_snapshots'
DAILY_PORT_TBL    = 'mvo_daily_portfolios'
DAILY_TRIGGER_TBL = 'mvo_daily_triggers'
MB_DAILY_PORT_TBL = 'mvo_daily_portfolios'   # alias kept for compatibility
MB_MIN_COV_MATRICES = 2

# ── ICS / composite score constants ──────────────────────────────────────────
ICS_MIN_STOCKS = 50
ICS_WEIGHT_MIN = 0.10
ICS_WEIGHT_MAX = 0.50
ICS_MOM_LONG   = 252   # 12M1 momentum lookback (trading days)
ICS_MOM_SKIP   = 21    # 12M1 momentum skip period
ICS_MOM_RESID  = {'v1': 'factor_residuals_vol', 'v2': 'v2_factor_residuals_quality'}
ICS_OU_TBL     = {'v1': 'ou_reversion_df',      'v2': 'v2_ou_reversion_df'}
ICS_VALUE_TBL  = {'v1': 'value_scores_df', 'v2': 'value_scores_df'}

# ── X-matrix / factor model internals ────────────────────────────────────────
_MB_MOM_RESID   = {'v1': 'factor_residuals_vol',     'v2': 'v2_factor_residuals_quality'}
_MB_OU_CACHE    = {'v1': 'ou_reversion_df',           'v2': 'v2_ou_reversion_df'}
_MB_SCALAR_TBLS = {
    'v1': {'Beta':'factor_lambdas_mkt','Size':'factor_lambdas_size',
           'Quality':'factor_lambdas_quality','SI':'factor_lambdas_si',
           'GK_Vol':'factor_lambdas_vol','Idio_Mom':'factor_lambdas_mom',
           'Value':'factor_lambdas_joint','OU':'factor_lambdas_ou'},
    'v2': {'Beta':'v2_factor_lambdas_mkt','Size':'v2_factor_lambdas_size',
           'Quality':'v2_factor_lambdas_quality','SI':'v2_factor_lambdas_si',
           'GK_Vol':'v2_factor_lambdas_vol','Idio_Mom':'v2_factor_lambdas_mom',
           'Value':'v2_factor_lambdas_value','OU':'v2_factor_lambdas_ou'},
}
_MB_SEC_TBL     = {'v1': 'factor_lambdas_sec',  'v2': 'v2_factor_lambdas_sec'}
_MB_LAMBDA_META = {'intercept', 'r2', 'ridge_lambda', 'date'}
_MB_F_LOOKBACK  = 60   # months of lambda history for F matrix (5 years)
_MB_F_EWMA_HL   = 42


# ── Inline dependency: select_with_sector_cap ─────────────────────────────────
def select_with_sector_cap(ranked_df, sector_cap, top_n):
    """Select top_n stocks with max sector_cap per sector (relaxes cap if needed)."""
    cap = sector_cap
    while cap <= top_n:
        selected      = []
        sector_counts = {}
        for ticker, row in ranked_df.iterrows():
            sector = row['Sector']
            count  = sector_counts.get(sector, 0)
            if count < cap:
                selected.append(ticker)
                sector_counts[sector] = count + 1
            if len(selected) == top_n:
                break
        if len(selected) == top_n:
            if cap > sector_cap:
                print(f"    Sector cap relaxed to {cap} to fill {top_n} slots")
            return ranked_df.loc[selected]
        cap += 1
    return ranked_df.head(top_n)


# ── Inline dependency: apply_vol_filter ───────────────────────────────────────
def apply_vol_filter(tickers, dt, Pxs_df, lookback=63, vol_cap_mult=3.0):
    """
    Filter out stocks with annualised volatility exceeding vol_cap_mult × median
    cross-sectional vol over the last `lookback` trading days.
    Returns filtered list of tickers.
    """
    past = [d for d in Pxs_df.index if d <= dt][-lookback:]
    if len(past) < 10:
        return list(tickers)
    rets = Pxs_df.loc[past, [t for t in tickers if t in Pxs_df.columns]].pct_change()
    vols = rets.std() * np.sqrt(252)
    vols = vols.dropna()
    if vols.empty:
        return list(tickers)
    threshold = vols.median() * vol_cap_mult
    surviving = vols[vols <= threshold].index.tolist()
    no_data   = [t for t in tickers if t not in vols.index]
    return surviving + no_data


def get_universe(Pxs_df, sectors_s, extended_st_dt):
    """
    Build stock universe: sector-mapped stocks with sufficient price history.
    Uses sectors_s as the canonical universe — NOT filtered by Pxs_df.columns —
    so the universe is invariant to the end date of Pxs_df (no lookahead bias).
    Stocks not yet in Pxs_df at a given calc_date will produce NaN scores and
    be naturally excluded from that date's cross-section.
    """
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT ticker FROM income_data"
            )).fetchall()
        db_tickers = {r[0].upper() for r in rows}
    except Exception as e:
        print(f"  WARNING: DB universe query failed ({e}) — using sectors_s")
        db_tickers = set(sectors_s.index)

    etf_tickers = set(sectors_s.values)
    pre_dates   = Pxs_df.index[Pxs_df.index < extended_st_dt]

    universe = []
    for col in sectors_s.index:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        # Pre-start history check — skip if stock not yet in Pxs_df or insufficient data
        # Per-calc-date filtering is handled by _get_active_universe
        if col in Pxs_df.columns and len(pre_dates) >= BETA_WINDOW:
            col_data = Pxs_df.loc[pre_dates[-BETA_WINDOW:], col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            if int(col_data.notna().sum()) < BETA_WINDOW // 2:
                continue
        universe.append(col)

    print(f"  Universe: {len(universe)} stocks "
          f"(sector mapped + DB + sufficient pre-start history)")
    return universe


def generate_calc_dates(Pxs_df, step_days=30):
    """Generate rebalancing dates from MB_START_DATE at step_days intervals."""
    end_date = Pxs_df.index.max()
    dates    = []
    current  = MB_START_DATE
    while current <= end_date:
        available = Pxs_df.index[Pxs_df.index >= current]
        if available.empty:
            break
        dates.append(available[0])
        current += pd.Timedelta(days=step_days)
    return sorted(set(dates))




def _ics_zscore(s):
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd

def _ics_bounded_normalize(raw_series, w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX):
    s = raw_series.clip(lower=0)
    total = s.sum()
    if total == 0:
        return s * 0 + 1.0 / len(s)
    s = s / total
    # Iterative clipping
    for _ in range(50):
        over  = s > w_max
        under = s < w_min
        if not over.any() and not under.any():
            break
        s[over]  = w_max
        s[under] = w_min
        remainder = 1.0 - s[over].sum() - s[under].sum()
        mid = ~over & ~under
        if mid.any() and s[mid].sum() > 0:
            s[mid] = s[mid] / s[mid].sum() * remainder
    return s / s.sum() if s.sum() > 0 else s

def _ics_pctrank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank: [0,1] range, sector-independent."""
    s = s.dropna()
    if len(s) < 2:
        return s
    r = s.rank(method='average')
    return (r - 1) / (len(r) - 1)


def _ics_compute_mom_12m1(universe, calc_dates, Pxs_df):
    """Compute 12M-1M momentum: percentile-ranked return from t-252 to t-21."""
    print("  Computing 12M1 momentum scores...")
    all_px_dates   = Pxs_df.index
    valid_universe = [t for t in universe if t in Pxs_df.columns]
    results = {}
    for dt in calc_dates:
        past = all_px_dates[all_px_dates < dt]
        if len(past) < ICS_MOM_LONG + 1:
            continue
        date_start = past[-(ICS_MOM_LONG + 1)]
        date_end   = past[-(ICS_MOM_SKIP + 1)]
        px_start = Pxs_df.loc[date_start, valid_universe]
        px_end   = Pxs_df.loc[date_end,   valid_universe]
        mom = ((px_end - px_start) / px_start.replace(0, np.nan)).dropna()
        if len(mom) < ICS_MIN_STOCKS:
            continue
        results[dt] = _ics_pctrank(mom)
    out = pd.DataFrame(results).T
    out.index.name = 'date'
    print(f"  12M1 momentum scores: {len(out)} dates")
    return out.reindex(columns=valid_universe)

def _mb_load_ou(universe, model_version):
    """Load O-U scores directly from DB — fallback if kernel version unavailable."""
    tbl = ICS_OU_TBL.get(model_version, 'v2_ou_reversion_df')
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(f"SELECT date, ticker, ou_score FROM {tbl} ORDER BY date", conn)
        df['date'] = pd.to_datetime(df['date'])
        ou = df.pivot_table(index='date', columns='ticker', values='ou_score', aggfunc='last')
        return ou.reindex(columns=universe).apply(_ics_zscore, axis=1)
    except Exception as e:
        print(f"  WARNING: O-U load failed ({e}) — skipping")
        return pd.DataFrame()


_MIN_HISTORY_DAYS  = 50    # minimum trading days of real price history for inclusion
_STALE_RUN_LIMIT   = 3     # consecutive unchanged closes = stale/backfilled price

def _get_active_universe(universe, Pxs_df, dt,
                          min_days=_MIN_HISTORY_DAYS,
                          stale_run=_STALE_RUN_LIMIT):
    """
    Return subset of universe with genuine price history up to dt.
    Filters out:
      1. Stocks not yet in Pxs_df at dt
      2. Stocks with < min_days of non-stale trading days up to dt
    Stale detection: any run of >stale_run consecutive identical closes
    is treated as backfilled/pre-IPO data and replaced with NaN.
    """
    px_to_dt = Pxs_df.loc[:dt, [t for t in universe if t in Pxs_df.columns]]
    if px_to_dt.empty:
        return []

    active = []
    for tkr in px_to_dt.columns:
        s = px_to_dt[tkr].dropna()
        if s.empty:
            continue
        # Replace stale runs with NaN
        # A run of >stale_run identical values = backfilled
        runs = (s != s.shift()).cumsum()
        run_lengths = runs.map(runs.value_counts())
        s_clean = s.where(run_lengths <= stale_run)
        if s_clean.notna().sum() >= min_days:
            active.append(tkr)
    return active


def _cb_build_composite_scores(universe, calc_dates, Pxs_df, sectors_s,
                                weights_by_year, regime_s, volumeTrd_df,
                                model_version, exclude_factors=None,
                                weights_by_date=None,
                                mr_scores_by_date=None):
    _load_quality   = _ics_load_quality
    _load_idio_mom  = _ics_load_idio_mom
    _calc_idio_mom  = _ics_compute_idio_mom_scores
    _load_value     = _ics_load_value
    _load_ou        = _mb_load_ou
    _load_mom12m1   = _ics_compute_mom_12m1
    exclude_factors = exclude_factors or ['OU']
    first_w = next(iter(weights_by_year.values()))
    active  = [f for f in first_w.columns if f not in exclude_factors]

    # Pre-sort weights_by_date keys for fast lookup
    _wbd_dates = sorted(weights_by_date.keys()) if weights_by_date else []

    print(f"  Active factors: {active}")
    if weights_by_date:
        print(f"  Using point-in-time weights_by_date ({len(_wbd_dates)} dates)")
    print(f"  Point-in-time weights (sample -- year 2022)"
          + (f"  [QUALITY_FLOOR={QUALITY_FLOOR}]:" if QUALITY_FLOOR > 0 else ":"))
    if 2022 in weights_by_year:
        w_sample = weights_by_year[2022][active].copy()
        for r in w_sample.index:
            w_sample.loc[r] = _ics_bounded_normalize(
                w_sample.loc[r], w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX)
        # Apply quality floor to display
        if QUALITY_FLOOR > 0:
            _mom_factors = [f for f in ('Idio_Mom', 'Mom_12M1') if f in w_sample.columns]
            if 'Quality' in w_sample.columns and _mom_factors:
                for r in w_sample.index:
                    _wm  = sum(w_sample.loc[r, f] for f in _mom_factors)
                    _wq  = w_sample.loc[r, 'Quality']
                    _mqs = _wm + _wq
                    if _mqs > 0 and (_wq / _mqs) < QUALITY_FLOOR:
                        w_sample.loc[r, 'Quality'] = _mqs * QUALITY_FLOOR
                        _rem = _mqs * (1.0 - QUALITY_FLOOR)
                        for f in _mom_factors:
                            w_sample.loc[r, f] = (_rem * (w_sample.loc[r, f] / _wm)
                                                   if _wm > 0 else _rem / len(_mom_factors))
        print(w_sample.round(4))
    print()

    score_dfs = {}
    if 'Quality' in active:
        print("  Loading quality scores...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            score_dfs['Quality'] = _load_quality(universe, calc_dates, Pxs_df, sectors_s)
        _n = score_dfs['Quality'].notna().any(axis=1).sum() if not score_dfs['Quality'].empty else 0
        print(f"{_n} dates" if _n else "WARNING: no data returned")
    if 'Idio_Mom' in active:
        print("  Loading idio momentum...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            resid_df = _load_idio_mom(universe, model_version)
            score_dfs['Idio_Mom'] = _calc_idio_mom(resid_df, calc_dates)
        _n = score_dfs['Idio_Mom'].notna().any(axis=1).sum() if not score_dfs['Idio_Mom'].empty else 0
        print(f"{_n} dates" if _n else "WARNING: no data returned")
    if 'Value' in active:
        print("  Loading value scores...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            score_dfs['Value'] = _load_value(universe, calc_dates, sectors_s, model_version)
        _n = score_dfs['Value'].notna().any(axis=1).sum() if not score_dfs['Value'].empty else 0
        print(f"{_n} dates" if _n else "WARNING: no data returned")
    if 'Mom_12M1' in active:
        print("  Computing 12M1 momentum...", end=' ')
        score_dfs['Mom_12M1'] = _load_mom12m1(universe, calc_dates, Pxs_df)
        _n = len(score_dfs['Mom_12M1'])
        print(f"{_n} dates")
    if 'OU' in active:
        print("  Loading O-U scores...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            ou = _load_ou(universe, model_version)
        if not ou.empty:
            all_d = calc_dates.union(ou.index).sort_values()
            score_dfs['OU'] = ou.reindex(all_d).ffill().reindex(calc_dates)
            print(f"{ou.shape[0]} dates")
        else:
            print("WARNING: no data returned")

    composite_by_date = {}
    n = len(calc_dates)
    _diag_dt = calc_dates[0] if len(calc_dates) > 0 else None
    for i, dt in enumerate(calc_dates):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Building composite [{i+1}/{n}] {dt.date()}", end='\r')

        # ── Per-date universe: only stocks with genuine price history up to dt ─
        active_u = _get_active_universe(universe, Pxs_df, dt)

        if dt == _diag_dt:
            print(f"\n  [DIAG {dt.date()}] active_u={len(active_u)}, today_ts={Pxs_df.index[-1].date()}")
            for fname in active:
                if fname in score_dfs and score_dfs[fname] is not None:
                    sdf = score_dfs[fname]
                    if dt in sdf.index:
                        vals = sdf.loc[dt].reindex(active_u).dropna()
                        print(f"    {fname}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}", flush=True)

        # ── Weight lookup: weights_by_date (preferred) or weights_by_year ────
        if _wbd_dates:
            # Find latest weights_by_date entry strictly before or on dt
            past = [d for d in _wbd_dates if d <= dt]
            if past:
                w_t = weights_by_date[past[-1]]
                # w_t is already a Series(factor -> weight) for current regime
                w_t = w_t.reindex(active).fillna(1.0 / len(active))
                w_t = w_t / w_t.sum()
                if dt == _diag_dt:
                    print(f"    w_t from weights_by_date[{past[-1].date()}]: {w_t.round(4).to_dict()}", flush=True)
            else:
                # Before any weights_by_date entry — fall back to equal weight
                w_t = pd.Series(1.0 / len(active), index=active)
                if dt == _diag_dt:
                    print(f"    w_t: equal weight fallback", flush=True)
        else:
            # Fall back to annual snapshot
            yr = dt.year
            if yr in weights_by_year:
                w_df = weights_by_year[yr]
            else:
                avail   = sorted(weights_by_year.keys())
                nearest = min(avail, key=lambda y: abs(y - yr))
                w_df    = weights_by_year[nearest]

            w_df_active = w_df[active].copy()
            rs = w_df_active.sum(axis=1)
            for ri in w_df_active.index:
                w_df_active.loc[ri] = (w_df_active.loc[ri] / rs[ri]
                                       if rs[ri] > 0 else 1.0 / len(active))

            reg_cands = regime_s[regime_s.index <= dt]
            if reg_cands.empty:
                continue
            r = float(reg_cands.iloc[-1])
            if r not in w_df_active.index:
                r = min(w_df_active.index, key=lambda x: abs(x - r))
            w_t = w_df_active.loc[r]

        parts = []
        # ── Quality floor: redistribute within {Idio_Mom, Mom_12M1, Quality} ──
        if QUALITY_FLOOR > 0:
            _mom_factors = [f for f in ('Idio_Mom', 'Mom_12M1') if f in w_t.index]
            _qual_factor = 'Quality'
            if _qual_factor in w_t.index and _mom_factors:
                _w_mom_sum  = sum(w_t[f] for f in _mom_factors)
                _w_qual     = w_t[_qual_factor]
                _w_mqsum    = _w_mom_sum + _w_qual   # Value untouched
                _qual_share = _w_qual / _w_mqsum if _w_mqsum > 0 else 0.0
                if _qual_share < QUALITY_FLOOR:
                    # Floor binds — redistribute
                    w_t = w_t.copy()
                    _qual_new  = _w_mqsum * QUALITY_FLOOR
                    _mom_rem   = _w_mqsum * (1.0 - QUALITY_FLOOR)
                    w_t[_qual_factor] = _qual_new
                    # Split remainder in original Idio_Mom : Mom_12M1 ratio
                    if _w_mom_sum > 0:
                        for f in _mom_factors:
                            w_t[f] = _mom_rem * (w_t[f] / _w_mom_sum)
                    else:
                        for f in _mom_factors:
                            w_t[f] = _mom_rem / len(_mom_factors)
        # MR penalty: load scores for this date if MR_K > 0
        _mr_pen = None
        if MR_K > 0 and mr_scores_by_date is not None:
            _mr_dt = max((d for d in mr_scores_by_date if d <= dt),
                         default=None)
            if _mr_dt is not None:
                _mr_pen = mr_scores_by_date[_mr_dt]  # Series {ticker: score}

        MOMENTUM_FACTORS    = {'Idio_Mom', 'Mom_12M1'}
        FUNDAMENTAL_FACTORS = {'Quality', 'Value'}
        fundamental_parts   = []
        momentum_parts      = []
        coverage            = pd.Series(0, index=active_u)

        for fname in active:
            if fname not in score_dfs or score_dfs[fname] is None:
                continue
            sdf = score_dfs[fname]
            if dt not in sdf.index:
                continue
            col = sdf.loc[dt].reindex(active_u)
            if col.isna().all():
                continue
            col_valid = col.replace(0, np.nan).dropna()
            if fname in MOMENTUM_FACTORS:
                # Percentile rank [0,1] — same scale as quality/value
                r = col_valid.rank(method='average')
                col_valid = (r - 1) / (len(r) - 1) if len(r) > 1 else r * 0 + 0.5
            else:
                col_valid = _ics_zscore(col_valid)
            if col_valid.empty:
                continue
            coverage[col_valid.index] += 1
            col = col_valid.reindex(active_u).fillna(0)
            if MR_K > 0 and _mr_pen is not None and fname in MOMENTUM_FACTORS:
                pen = _mr_pen.reindex(col.index).fillna(0.0)
                col = col / np.exp(MR_K * pen)
            if fname in FUNDAMENTAL_FACTORS:
                fundamental_parts.append(col * w_t[fname])
            else:
                momentum_parts.append(col * w_t[fname])

        parts = fundamental_parts + momentum_parts

        if not parts:
            continue

        composite = pd.concat(parts, axis=1).sum(axis=1)

        # Penalize stocks with insufficient fundamental factor coverage
        # Regime-dependent gate:
        #   easing (q=0.0)        → require quality only
        #   neutral/tight (q>0.0) → require both quality AND value
        rate_q = float(regime_s.reindex([dt], method='ffill').iloc[0]) \
                 if dt in regime_s.index or len(regime_s) > 0 else 0.5
        if rate_q == 0.0:
            required_fundamentals = {'Quality'}
        else:
            required_fundamentals = {'Quality', 'Value'}

        fund_coverage = pd.Series(0, index=active_u)
        for fname in required_fundamentals:
            if fname in score_dfs and score_dfs[fname] is not None \
                    and dt in score_dfs[fname].index:
                valid = score_dfs[fname].loc[dt].reindex(active_u).notna()
                fund_coverage += valid.astype(int)

        # Stocks missing all required fundamentals get composite score = 0
        no_fundamentals = fund_coverage[fund_coverage == 0].index
        composite[no_fundamentals] = 0.0
        composite = _ics_zscore(composite.replace(0, np.nan).dropna())
        if len(composite) >= ICS_MIN_STOCKS:
            composite_by_date[dt] = composite
            if i < 3:
                import hashlib
                for fname, sdf in score_dfs.items():
                    if sdf is not None and dt in sdf.index:
                        vals = sdf.loc[dt].reindex(active_u).dropna()
                        _fhash = hashlib.md5(vals.round(6).to_json().encode()).hexdigest()[:8]
                        print(f"  [HASH {dt.date()}] {fname}: n={len(vals)} hash={_fhash}", flush=True)
                _chash = hashlib.md5(composite.round(6).to_json().encode()).hexdigest()[:8]
                if dt == _diag_dt:
                    print(f"\n  [COMPOSITE DIAG {dt.date()}] top-5, today_ts={Pxs_df.index[-1].date()}")
                    print(composite.nlargest(5).round(4).to_string(), flush=True)
                print(f"  [HASH {dt.date()}] composite={_chash} n={len(composite)}", flush=True)

    print(f"\n  Composite scores built: {len(composite_by_date)} dates")
    return composite_by_date, score_dfs

# Hedge engine functions embedded below (see HEDGE ENGINE section)

class _SuppressOutput:
    """
    Context manager to suppress stdout in both scripts and Jupyter notebooks.
    Tries fd-level redirect first; falls back to sys.stdout swap if fileno()
    is unavailable (Jupyter kernel streams don't expose a real fd).
    """
    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        try:
            self._stdout_fd = sys.stdout.fileno()
            self._old_fd    = os.dup(self._stdout_fd)
            os.dup2(self._devnull.fileno(), self._stdout_fd)
            self._use_fd = True
        except Exception:
            # Jupyter: no real fd -- fall back to replacing sys.stdout
            self._orig_stdout = sys.stdout
            sys.stdout = self._devnull
            self._use_fd = False
        return self

    def __exit__(self, *args):
        if self._use_fd:
            os.dup2(self._old_fd, self._stdout_fd)
            os.close(self._old_fd)
        else:
            sys.stdout = self._orig_stdout
        self._devnull.close()



# ── Covariance helpers (inlined from mvo_diagnostics.py) ─────────────────────

def _mvo_ewma_cov(ret_df, hl):
    """EWMA covariance matrix (T x N returns → N x N). Returns np.ndarray."""
    T     = len(ret_df)
    decay = np.log(2) / hl
    w     = np.exp(-decay * np.arange(T - 1, -1, -1))
    w    /= w.sum()
    v     = ret_df.values
    mu    = (w[:, None] * v).sum(0)
    d     = v - mu
    return (d * w[:, None]).T @ d


def _mvo_ewma_vol(ret_df, hl):
    """EWMA volatility per stock. Returns pd.Series."""
    T     = len(ret_df)
    decay = np.log(2) / hl
    w     = np.exp(-decay * np.arange(T - 1, -1, -1))
    w    /= w.sum()
    v     = ret_df.values
    mu    = (w[:, None] * v).sum(0)
    var   = (w[:, None] * (v - mu) ** 2).sum(0)
    return pd.Series(np.sqrt(var), index=ret_df.columns)


def _mvo_ledoit_wolf(ret_matrix):
    """Ledoit-Wolf shrinkage. Returns (Sigma_lw, rho_bar, shrinkage_coef)."""
    from sklearn.covariance import LedoitWolf
    T, N        = ret_matrix.shape
    lw          = LedoitWolf(assume_centered=False)
    lw.fit(ret_matrix)
    Sigma_lw    = lw.covariance_
    shrink_coef = float(lw.shrinkage_)
    std         = np.sqrt(np.diag(Sigma_lw))
    std_mat     = np.outer(std, std)
    with np.errstate(invalid='ignore', divide='ignore'):
        Corr = np.where(std_mat > 0, Sigma_lw / std_mat, 0.0)
    n_off   = N * (N - 1)
    rho_bar = (Corr.sum() - np.trace(Corr)) / n_off if n_off > 0 else 0.0
    return Sigma_lw, rho_bar, shrink_coef


def _mvo_pca_cov(ret_matrix, var_threshold=MVO_PCA_VAR_THRESH):
    """PCA covariance. Returns (Sigma_pca, n_components, var_explained, eigvals)."""
    Sigma            = _mvo_ewma_cov(pd.DataFrame(ret_matrix), MVO_EWMA_HL)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx              = np.argsort(eigvals)[::-1]
    eigvals          = np.maximum(eigvals[idx], 0.0)
    eigvecs          = eigvecs[:, idx]
    total_var        = eigvals.sum()
    cum_var          = np.cumsum(eigvals) / total_var if total_var > 0 else np.zeros_like(eigvals)
    n_comp           = max(1, int(np.searchsorted(cum_var, var_threshold) + 1))
    var_expl         = float(cum_var[n_comp - 1])
    Vk               = eigvecs[:, :n_comp]
    Lk               = np.diag(eigvals[:n_comp])
    Sigma_pca        = Vk @ Lk @ Vk.T
    resid_var        = np.diag(Sigma) - np.diag(Sigma_pca)
    Sigma_pca       += np.diag(np.maximum(resid_var, 0.0))
    return Sigma_pca, n_comp, var_expl, eigvals


# ===============================================================================
# SELF-CONTAINED FLOOR/CAP (does not depend on mvo_diagnostics.py version)
# ===============================================================================

def _mb_floor_then_cap(w, min_weight, max_weight, max_iter=20):
    """
    Post-solve weight adjustment:

    Step 1 -- Floor:
        Raise non-zero weights to min_weight, renormalize.
    Step 2 -- Cap (iterative):
        Reduce weights above max_weight to max_weight,
        redistribute excess proportionally to remaining stocks.
    """
    w = w.copy().clip(lower=0)
    if w.sum() == 0:
        return w

    # Step 1: raise sub-floor weights to min_weight then renorm
    w = w.clip(lower=min_weight)
    if w.sum() > 0:
        w = w / w.sum()

    # Step 2: iterative cap
    for _ in range(max_iter):
        over = w > max_weight + 1e-9
        if not over.any():
            break
        excess   = (w[over] - max_weight).sum()
        w[over]  = max_weight
        under    = ~over
        if w[under].sum() > 1e-12:
            w[under] += excess * (w[under] / w[under].sum())
        else:
            w += excess / len(w)

    if w.sum() > 0:
        w = w / w.sum()
    return w



# ===============================================================================
# COVARIANCE CACHE
# ===============================================================================

def _mb_get_cached_dates(force_rebuild):
    """Return set of dates already cached, or empty set if force_rebuild."""
    if force_rebuild:
        return set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": MB_COV_CACHE_TBL}).scalar()
        if not exists:
            return set()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                f"SELECT DISTINCT date FROM {MB_COV_CACHE_TBL}"
            )).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _mb_build_cov_matrices(dt, candidates, Pxs_df, sectors_s,
                             volumeTrd_df, model_version,
                             pca_var_threshold, X_df_cached=None,
                             lambda_dfs=None):
    """
    Build all four covariance matrices for the candidate universe on date dt.
    Returns (Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens).
    """
    # Return matrix — strictly up to dt (no lookahead)
    ret_df = (Pxs_df.loc[:dt, candidates].pct_change()
              .dropna(how='all')
              .iloc[-MVO_LOOKBACK:]
              .dropna(axis=1, how='any'))
    valid = ret_df.columns.tolist()

    # Empirical
    Sigma_emp = _mvo_ewma_cov(ret_df, MVO_EWMA_HL)

    # Ledoit-Wolf
    Sigma_lw, _, _ = _mvo_ledoit_wolf(ret_df.values)

    # Factor-driven (self-contained -- no _mvo_factor_cov dependency)
    try:
        F_mat, factor_names_f, sec_cols_f = _mb_build_F(model_version, dt=dt,
                                                         lambda_dfs=lambda_dfs)
        X_df = pd.DataFrame(0.0, index=valid, columns=factor_names_f)
        # Use most recent X snapshot if available, else build on the fly
        if X_df_cached is not None:
            X_df = X_df_cached.reindex(index=valid, columns=factor_names_f).fillna(0.0)
        else:
            pxs_slice = Pxs_df.loc[:dt]
            X_built   = _mb_build_X(dt, valid, factor_names_f,
                                     sec_cols_f, pxs_slice, sectors_s,
                                     volumeTrd_df, model_version)
            if X_built is not None:
                X_df = X_built.reindex(index=valid, columns=factor_names_f).fillna(0.0)
        X          = X_df.values
        Sigma_factor_raw = X @ F_mat @ X.T
        emp_mean_var     = np.diag(Sigma_emp).mean()
        factor_mean_var  = np.diag(Sigma_factor_raw).mean()
        if factor_mean_var > 1e-12:
            Sigma_factor = Sigma_factor_raw * (emp_mean_var / factor_mean_var)
        else:
            Sigma_factor = Sigma_emp.copy()
    except Exception as e:
        warnings.warn(f"  Factor cov failed ({e}) -- using empirical")
        Sigma_factor = Sigma_emp.copy()

    # PCA
    Sigma_pca, _, _, _ = _mvo_pca_cov(ret_df.values,
                                        var_threshold=pca_var_threshold)

    # Ensemble
    Sigma_ens = (Sigma_emp + Sigma_lw + Sigma_factor + Sigma_pca) / 4.0

    return valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens


# ===============================================================================
# SELF-CONTAINED X-MATRIX BUILDERS (no portfolio_risk_decomp dependency)
# ===============================================================================




def _mb_load_quality_scores(universe, calc_dates):
    """Load quality scores directly from DB cache. Self-contained."""
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(
                "SELECT date, ticker, score FROM quality_scores_df ORDER BY date"
            ), conn)
        if df.empty:
            return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)
        df['date']   = pd.to_datetime(df['date'])
        df['ticker'] = df['ticker'].str.replace(' US', '', regex=False)
        df['score']  = df['score'].astype(float)
        pivot = df.pivot_table(index='date', columns='ticker',
                               values='score', aggfunc='last')
        pivot = pivot.reindex(columns=universe)
        all_dates = calc_dates.union(pivot.index).sort_values()
        return pivot.reindex(all_dates).ffill().reindex(calc_dates).astype(float)
    except Exception as e:
        warnings.warn(f"  quality_scores_df load failed ({e})")
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)


def _mb_load_value_scores(universe, calc_dates):
    """Load value scores directly from DB cache. Self-contained."""
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(
                "SELECT date, ticker, score FROM value_scores_df ORDER BY date"
            ), conn)
        if df.empty:
            return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)
        df['date']   = pd.to_datetime(df['date'])
        df['ticker'] = df['ticker'].str.replace(' US', '', regex=False)
        df['score']  = df['score'].astype(float)
        pivot = df.pivot_table(index='date', columns='ticker',
                               values='score', aggfunc='last')
        pivot = pivot.reindex(columns=universe)
        all_dates = calc_dates.union(pivot.index).sort_values()
        return pivot.reindex(all_dates).ffill().reindex(calc_dates).astype(float)
    except Exception as e:
        warnings.warn(f"  value_scores_df load failed ({e})")
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)


def _mb_zscore(s):
    mu, sd = s.mean(), s.std()
    return s * 0.0 if (sd == 0 or np.isnan(sd)) else (s - mu) / sd


def _mb_get_factor_names(model_version, lambda_dfs=None):
    """Return (factor_names, sec_cols). Uses pre-loaded lambda_dfs if provided."""
    scalar_names = list(_MB_SCALAR_TBLS[model_version].keys())
    if lambda_dfs is not None and not lambda_dfs['sector'].empty:
        sec_cols = sorted(lambda_dfs['sector'].columns.tolist())
    else:
        try:
            with ENGINE.connect() as conn:
                sdf = pd.read_sql(
                    f"SELECT * FROM {_MB_SEC_TBL[model_version]} "
                    f"ORDER BY date DESC LIMIT 1", conn
                )
            sec_cols = sorted([c for c in sdf.columns
                               if c not in _MB_LAMBDA_META
                               and pd.api.types.is_float_dtype(sdf[c])])
        except Exception as e:
            warnings.warn(f"Could not load sector columns: {e}")
            sec_cols = []
    head = ['Beta', 'Size']
    tail = ['Quality', 'SI', 'GK_Vol', 'Idio_Mom', 'Value', 'OU']
    factor_names = (
        [f for f in head if f in scalar_names]
        + list(MACRO_COLS) + sec_cols
        + [f for f in tail if f in scalar_names]
    )
    return factor_names, sec_cols


def _mb_ewma_cov_f(df, hl):
    """EWMA covariance for building F matrix."""
    decay = np.log(2) / hl
    T     = len(df)
    w     = np.exp(-decay * np.arange(T - 1, -1, -1))
    w    /= w.sum()
    v     = df.values
    mu    = (w[:, None] * v).sum(0)
    d     = v - mu
    return (d * w[:, None]).T @ d


def _mb_load_lambda_dfs(model_version):
    """Load all lambda tables into DataFrames once at startup.
    Returns dict with keys: 'scalar' {fname: Series}, 'macro' DataFrame, 'sector' DataFrame.
    All indexed by date (monthly cadence).
    """
    tbls   = _MB_SCALAR_TBLS[model_version]
    scalar = {}
    for fname, tbl in tbls.items():
        try:
            with ENGINE.connect() as conn:
                df = pd.read_sql(f"SELECT * FROM {tbl} ORDER BY date", conn)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            num = df.drop(columns=[c for c in _MB_LAMBDA_META if c in df.columns],
                          errors='ignore').select_dtypes(include=np.number)
            scalar[fname] = num.iloc[:, 0].rename(fname)
        except Exception as e:
            warnings.warn(f"Lambda table {tbl} failed: {e}")

    macro_tbl = {'v1': 'factor_lambdas_macro', 'v2': 'v2_factor_lambdas_macro'}[model_version]
    try:
        with ENGINE.connect() as conn:
            mdf = pd.read_sql(f"SELECT * FROM {macro_tbl} ORDER BY date", conn)
        mdf['date'] = pd.to_datetime(mdf['date'])
        mdf = mdf.set_index('date').sort_index()
        macro_df = mdf.drop(columns=[c for c in _MB_LAMBDA_META if c in mdf.columns],
                            errors='ignore').select_dtypes(include=np.number)
    except Exception as e:
        warnings.warn(f"Macro lambda failed: {e}")
        macro_df = pd.DataFrame()

    try:
        with ENGINE.connect() as conn:
            sdf = pd.read_sql(f"SELECT * FROM {_MB_SEC_TBL[model_version]} ORDER BY date", conn)
        sdf['date'] = pd.to_datetime(sdf['date'])
        sdf = sdf.set_index('date').sort_index()
        sec_df = sdf.drop(columns=[c for c in _MB_LAMBDA_META if c in sdf.columns],
                          errors='ignore').select_dtypes(include=np.number)
    except Exception as e:
        warnings.warn(f"Sector lambda failed: {e}")
        sec_df = pd.DataFrame()

    return {'scalar': scalar, 'macro': macro_df, 'sector': sec_df}


def _mb_build_F(model_version, dt=None, lambda_dfs=None):
    """Build F matrix — strictly point-in-time up to dt.
    Uses pre-loaded lambda_dfs if provided, otherwise loads from DB.
    """
    if lambda_dfs is None:
        lambda_dfs = _mb_load_lambda_dfs(model_version)

    cutoff = pd.Timestamp(dt) if dt is not None else pd.Timestamp('2099-12-31')
    n      = _MB_F_LOOKBACK
    frames = []

    # Scalar factor lambdas
    for fname, s in lambda_dfs['scalar'].items():
        pit = s[s.index <= cutoff].tail(n)
        if not pit.empty:
            frames.append(pit)

    # Macro lambdas
    macro_df = lambda_dfs['macro']
    if not macro_df.empty:
        pit_m = macro_df[macro_df.index <= cutoff].tail(n)
        for mc in MACRO_COLS:
            if mc in pit_m.columns:
                frames.append(pit_m[mc].rename(mc))

    # Sector lambdas
    sec_df = lambda_dfs['sector']
    if not sec_df.empty:
        pit_s = sec_df[sec_df.index <= cutoff].tail(n)
        for sc in pit_s.columns:
            frames.append(pit_s[sc].rename(sc))

    combined = pd.concat(frames, axis=1).dropna()
    factor_names, sec_cols = _mb_get_factor_names(model_version, lambda_dfs=lambda_dfs)
    ordered  = [c for c in factor_names if c in combined.columns]
    combined = combined[ordered]
    F        = _mb_ewma_cov_f(combined, _MB_F_EWMA_HL)
    return F, ordered, sec_cols


def _mb_sector_dummies(universe, sectors_s, sec_cols):
    sect = sectors_s.reindex(universe).fillna('Unknown')
    data = pd.DataFrame(0.0, index=universe, columns=sec_cols)
    for ticker in universe:
        s = sect[ticker]
        if s in sec_cols:
            data.loc[ticker, s] = 1.0
    return data


def _mb_build_X(dt, universe, factor_names, sec_cols,
                 Pxs_df, sectors_s, volumeTrd_df, model_version):
    """Self-contained X matrix builder -- uses kernel factor model functions."""
    pxs_to_dt  = Pxs_df.loc[:dt]
    if len(pxs_to_dt) < BETA_WINDOW // 2:
        return None
    calc_dates = pd.DatetimeIndex([dt])

    # Resolve kernel functions
    # These functions come from factor_model_step1 (run before mvo_backtest)
    try:
        beta_df = calc_rolling_betas(pxs_to_dt, universe, calc_dates)
        beta_s  = _mb_zscore(beta_df.iloc[-1].reindex(universe)).rename('Beta')

        size_df = load_dynamic_size(universe, pxs_to_dt, calc_dates)
        size_s  = _mb_zscore(np.log(size_df.iloc[-1].reindex(universe).clip(lower=1e-6))).rename('Size')

        macro_betas  = calc_macro_betas(pxs_to_dt, universe, calc_dates)
        macro_series = {}
        for mc in MACRO_COLS:
            if mc in macro_betas and not macro_betas[mc].empty:
                macro_series[mc] = _mb_zscore(macro_betas[mc].iloc[-1].reindex(universe)).rename(mc)
            else:
                macro_series[mc] = pd.Series(0.0, index=universe, name=mc)

        dummies = _mb_sector_dummies(universe, sectors_s, sec_cols)

        quality_df = _mb_load_quality_scores(universe, calc_dates)
        quality_s  = _mb_zscore(quality_df.reindex(calc_dates).iloc[-1].reindex(universe)).rename('Quality')

        recent_dates = pxs_to_dt.index[-60:]
        si_full = load_si_composite(universe, recent_dates)
        si_s    = _mb_zscore(si_full.reindex(recent_dates).ffill().iloc[-1].reindex(universe)).rename('SI')

        open_df, high_df, low_df = load_ohlc_tables(universe)
        vol_df = calc_vol_factor(pxs_to_dt, universe, calc_dates,
                                 open_df=open_df, high_df=high_df, low_df=low_df)
        vol_s  = _mb_zscore(vol_df.iloc[-1].reindex(universe)).rename('GK_Vol')

        try:
            with ENGINE.connect() as conn:
                res_mom = pd.read_sql(
                    f"SELECT * FROM {_MB_MOM_RESID[model_version]} ORDER BY date", conn)
            res_mom['date'] = pd.to_datetime(res_mom['date'])
            if 'ticker' in res_mom.columns and 'resid' in res_mom.columns:
                res_mom = res_mom.pivot_table(
                    index='date', columns='ticker', values='resid', aggfunc='last'
                ).reindex(columns=universe)
            else:
                res_mom = res_mom.set_index('date').reindex(columns=universe)
            res_mom = res_mom[res_mom.index <= dt]
            if volumeTrd_df is not None:
                mom_df = calc_idio_momentum_volscaled(res_mom, volumeTrd_df, calc_dates)
            else:
                mom_df = calc_idio_momentum(res_mom, calc_dates)
            mom_s = _mb_zscore(mom_df.iloc[-1].reindex(universe)).rename('Idio_Mom')
        except Exception as e:
            warnings.warn(f"  Idio momentum failed ({e}) -- filling 0")
            mom_s = pd.Series(0.0, index=universe, name='Idio_Mom')

        value_df = _mb_load_value_scores(universe, calc_dates)
        value_s  = _mb_zscore(value_df.reindex(calc_dates).iloc[-1].reindex(universe)).rename('Value')

        try:
            with ENGINE.connect() as conn:
                ou_raw = pd.read_sql(
                    f"SELECT date, ticker, ou_score FROM {_MB_OU_CACHE[model_version]} "
                    f"WHERE date <= '{dt.date()}' ORDER BY date DESC "
                    f"LIMIT {len(universe) * 5}", conn
                )
            ou_raw['date'] = pd.to_datetime(ou_raw['date'])
            ou_pivot = ou_raw.pivot_table(
                index='date', columns='ticker',
                values='ou_score', aggfunc='last'
            ).reindex(columns=universe)
            ou_s = _mb_zscore(ou_pivot.iloc[-1].reindex(universe)).rename('OU')
        except Exception as e:
            warnings.warn(f"  O-U load failed ({e}) -- filling 0")
            ou_s = pd.Series(0.0, index=universe, name='OU')

        scalar_map = {
            'Beta': beta_s, 'Size': size_s, 'Quality': quality_s,
            'SI': si_s, 'GK_Vol': vol_s, 'Idio_Mom': mom_s,
            'Value': value_s, 'OU': ou_s,
        }
        cols = []
        for fname in factor_names:
            if fname in scalar_map:
                cols.append(scalar_map[fname])
            elif fname in MACRO_COLS:
                cols.append(macro_series.get(
                    fname, pd.Series(0.0, index=universe, name=fname)
                ))
            elif fname in sec_cols:
                cols.append(dummies[fname].rename(fname))
            else:
                cols.append(pd.Series(0.0, index=universe, name=fname))

        return pd.concat(cols, axis=1).reindex(universe).fillna(0.0)

    except Exception as e:
        warnings.warn(f"  X build failed for {dt.date()}: {e}")
        return None


def _mb_month_end_dates(date_index, latest_date):
    idx   = pd.DatetimeIndex(date_index)
    s     = pd.Series(idx, index=idx)
    mends = s.groupby([s.index.year, s.index.month]).last().values
    return pd.DatetimeIndex(mends).union([latest_date]).sort_values()


# ===============================================================================
# X SNAPSHOT CACHE (PostgreSQL-backed)
# ===============================================================================



def _mb_save_x_snapshot(dt, model_version, xdf):
    """Persist a single X snapshot to the DB cache table (upsert)."""
    rec = {
        'date'         : dt.strftime('%Y-%m-%d'),
        'model_version': model_version,
        'tickers'      : json.dumps(xdf.index.tolist()),
        'factors'      : json.dumps(xdf.columns.tolist()),
        'values'       : json.dumps(xdf.values.tolist()),
    }
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MB_X_CACHE_TBL} (
                date          DATE         NOT NULL,
                model_version VARCHAR(10)  NOT NULL,
                tickers       TEXT         NOT NULL,
                factors       TEXT         NOT NULL,
                values        TEXT         NOT NULL,
                PRIMARY KEY (date, model_version)
            )
        """))
        conn.execute(text(f"""
            INSERT INTO {MB_X_CACHE_TBL} (date, model_version, tickers, factors, values)
            VALUES (:date, :model_version, :tickers, :factors, :values)
            ON CONFLICT (date, model_version) DO UPDATE
                SET tickers = EXCLUDED.tickers,
                    factors = EXCLUDED.factors,
                    values  = EXCLUDED.values
        """), rec)


def _mb_load_x_cache(model_version):
    """Load all cached X snapshots for a given model version. Returns dict {date: df}."""
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": MB_X_CACHE_TBL}).scalar()
        if not exists:
            return {}
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT date, tickers, factors, values
                FROM {MB_X_CACHE_TBL}
                WHERE model_version = :mv
                ORDER BY date
            """), {"mv": model_version}).fetchall()
        cached = {}
        for row in rows:
            dt      = pd.Timestamp(row[0])
            tickers = json.loads(row[1])
            factors = json.loads(row[2])
            vals    = json.loads(row[3])
            cached[dt] = pd.DataFrame(vals, index=tickers, columns=factors)
        return cached
    except Exception as e:
        warnings.warn(f"  X cache load failed: {e}")
        return {}


def _mb_clear_x_cache(model_version):
    """Delete all cached X snapshots for a given model version."""
    try:
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                DELETE FROM {MB_X_CACHE_TBL}
                WHERE model_version = :mv
            """), {"mv": model_version})
        print(f"  X cache cleared for model_version={model_version}")
    except Exception:
        pass


# ===============================================================================
# X SNAPSHOT BUILDER
# ===============================================================================

def _mb_build_x_snapshots(rebal_dates, Pxs_df, sectors_s,
                            volumeTrd_df, model_version,
                            force_rebuild, lambda_dfs=None):
    """
    Build X exposure matrix snapshots with DB caching.

    Behavior:
    - force_rebuild=False (default): load cache, compute only missing dates,
      save new ones. On typical daily runs this means only today is computed.
    - force_rebuild=True: clear cache, rebuild all month-end dates from scratch.

    Returns dict {date: DataFrame(universe x factors)}.
    """
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    factor_names, sec_cols = _mb_get_factor_names(model_version, lambda_dfs=lambda_dfs)

    # All dates we need: month-ends from backtest start + latest date
    # Bound to MB_START_DATE -- no point building X before backtest begins
    pxs_idx_bounded = Pxs_df.index[Pxs_df.index >= MB_START_DATE - pd.Timedelta(days=90)]
    x_snap_dates = _mb_month_end_dates(pxs_idx_bounded, rebal_dates[-1])

    if force_rebuild:
        _mb_clear_x_cache(model_version)
        cached = {}
        print(f"  force_rebuild=True: rebuilding all {len(x_snap_dates)} snapshots...")
    else:
        cached = _mb_load_x_cache(model_version)
        print(f"  X cache loaded: {len(cached)} dates already cached")

    # Only compute dates not already in cache
    to_build = [d for d in x_snap_dates if d not in cached]
    print(f"  Dates to build: {len(to_build)}"
          f"  ({to_build[0].date() if to_build else 'none'}"
          f" -> {to_build[-1].date() if to_build else 'none'})")

    X_snapshots = dict(cached)
    n_built = 0
    for i, xdt in enumerate(to_build):
        print(f"  Building snapshot {i+1}/{len(to_build)}: {xdt.date()}...", end='\r')
        with _SuppressOutput():
            xdf = _mb_build_X(xdt, universe, factor_names, sec_cols,
                              Pxs_df, sectors_s, volumeTrd_df, model_version)
        if xdf is not None:
            xdf = xdf.fillna(0.0)
            X_snapshots[xdt] = xdf
            try:
                _mb_save_x_snapshot(xdt, model_version, xdf)
            except Exception as e:
                warnings.warn(f"  Failed to save X snapshot {xdt.date()}: {e}")
            n_built += 1

    if to_build:
        print(f"\n  Built and cached: {n_built}/{len(to_build)} new snapshots")
    print(f"  Total X snapshots available: {len(X_snapshots)}")
    return X_snapshots, universe, factor_names, sec_cols


# ===============================================================================
# DAILY PORTFOLIO CACHE (PostgreSQL-backed)
# ===============================================================================

def _make_quality_fingerprint():
    """Compact fingerprint of quality score coverage — changes when new historical data added."""
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(text(f"""
                SELECT COUNT(*), MIN(date), MAX(date)
                FROM quality_scores_df
            """)).fetchone()
        return f"{row[0]}_{row[1]}_{row[2]}"
    except Exception:
        return "no_quality"


def _make_make_params_hash(ic, max_weight, min_weight, zscore_cap, pca_var_threshold,
                      universe_mult, risk_aversion, top_n, conc_factor,
                      prefilt_pct, min_cov_matrices, model_version, mom_filter=3):
    """Stable 12-char hash of all portfolio construction parameters + quality coverage."""
    params = dict(
        ic=ic, maxw=max_weight, minw=min_weight, zc=zscore_cap,
        pca=pca_var_threshold, um=universe_mult, ra=risk_aversion,
        n=top_n, conc=conc_factor, pf=round(prefilt_pct, 4),
        mcm=min_cov_matrices, mv=model_version, mf=mom_filter,
        qfp=_make_quality_fingerprint(),
    )
    return hashlib.md5(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:12]


def _ensure_daily_cache_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MB_DAILY_PORT_TBL} (
                date          DATE        NOT NULL,
                model_version VARCHAR(10) NOT NULL,
                params_hash   VARCHAR(16) NOT NULL,
                strategy      VARCHAR(16) NOT NULL,
                weights_json  TEXT        NOT NULL,
                PRIMARY KEY (date, model_version, params_hash, strategy)
            )
        """))


def _get_cached_portfolio_dates(params_hash, model_version):
    try:
        _ensure_daily_cache_table()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT date FROM {MB_DAILY_PORT_TBL}
                WHERE params_hash = :ph AND model_version = :mv
                  AND strategy = 'alpha'
            """), {'ph': params_hash, 'mv': model_version}).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_daily_portfolios(dt, model_version, params_hash,
                            w_alpha, w_mvo, w_hybrid, w_smart=None):
    for strategy, w in [('alpha', w_alpha), ('mvo', w_mvo),
                        ('hybrid', w_hybrid), ('smart', w_smart)]:
        if w is None or w.empty:
            continue
        wj = json.dumps({t: float(v) for t, v in w.items()})
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {MB_DAILY_PORT_TBL}
                    (date, model_version, params_hash, strategy, weights_json)
                VALUES (:dt, :mv, :ph, :st, :wj)
                ON CONFLICT (date, model_version, params_hash, strategy)
                DO UPDATE SET weights_json = EXCLUDED.weights_json
            """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': model_version,
                   'ph': params_hash, 'st': strategy, 'wj': wj})


def load_daily_portfolios(params_hash, model_version='v2'):
    """
    Load all cached daily portfolios for given params_hash.
    Returns: {date: {'alpha': pd.Series, 'mvo': pd.Series, 'hybrid': pd.Series}}
    """
    _ensure_daily_cache_table()
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, strategy, weights_json
            FROM {MB_DAILY_PORT_TBL}
            WHERE params_hash = :ph AND model_version = :mv
            ORDER BY date
        """), {'ph': params_hash, 'mv': model_version}).fetchall()
    result = {}
    for row in rows:
        dt = pd.Timestamp(row[0])
        st = row[1]
        w  = pd.Series(json.loads(row[2]))
        result.setdefault(dt, {})[st] = w
    return result


# ===============================================================================
# MVO WEIGHT SOLVER -- exact replica of mvo_diagnostics logic
# ===============================================================================

def _mb_max_alpha_portfolio(Sigma, alpha, tickers, risk_aversion,
                              max_weight, min_weight):
    """
    Unconstrained MVO (w>=0, sum=1), then post-solve floor+cap.
    Identical to mvo_diagnostics._mvo_max_alpha_portfolio.
    """
    import cvxpy as cp
    try:
        N = len(tickers)
        w = cp.Variable(N)
        a = alpha.reindex(tickers).fillna(0.0).values
        prob = cp.Problem(
            cp.Maximize(a @ w - (risk_aversion / 2) * cp.quad_form(w, Sigma)),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            prob.solve(solver=cp.SCS, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            w_sol = pd.Series(np.maximum(w.value, 0.0), index=tickers)
            if w_sol.sum() > 0:
                w_sol = w_sol / w_sol.sum()
            return _mb_floor_then_cap(w_sol, min_weight, max_weight)
    except Exception as e:
        warnings.warn(f"  MVO solver failed: {e}")
    # Fallback: equal weight
    w_eq = pd.Series(1.0 / len(tickers), index=tickers)
    return _mb_floor_then_cap(w_eq, min_weight, max_weight)


def _advp_filter_and_replace(weights: pd.Series,
                              candidates: pd.Series,
                              dt: pd.Timestamp,
                              Pxs_df: pd.DataFrame,
                              volumeRaw_df: pd.DataFrame,
                              current_aum: float,
                              advp_cap: float,
                              min_weight: float,
                              max_weight: float,
                              top_n: int,
                              conc_factor: float = 1.0,
                              target_n: int = None,
                              full_universe: pd.Series = None) -> pd.Series:
    """
    Apply ADVP liquidity filter to a portfolio, replacing illiquid stocks
    with the next-best liquid candidates from the ranked candidates Series.

    target_n: target number of stocks after filtering. Defaults to top_n.
              For hybrid portfolios, pass len(weights) to preserve the
              larger union without truncating to top_n.
    """
    if target_n is None:
        target_n = top_n
    """
    Apply ADVP liquidity filter to a portfolio, replacing illiquid stocks
    with the next-best liquid candidates from the ranked candidates Series.

    Parameters
    ----------
    weights    : current portfolio weights (top_n stocks)
    candidates : composite scores for full candidate universe (ranked desc)
    current_aum: AUM at this date (AUM × running_nav)
    advp_cap   : fraction of ADV allowed per stock
    min_weight : minimum portfolio weight — stocks below this capacity are excluded
    top_n      : target number of stocks
    conc_factor: used to rebuild weights after replacement

    Returns
    -------
    Filtered and replaced weights Series, always attempting top_n stocks.
    """
    if volumeRaw_df is None or current_aum <= 0 or weights.empty:
        return weights, set()

    past_dates = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
    pxs_cols   = set(Pxs_df.columns)

    def _is_liquid(tkr):
        if tkr not in pxs_cols or tkr not in volumeRaw_df.columns:
            return False  # no volume data — treat as illiquid (conservative)
        px_s  = Pxs_df.loc[past_dates, tkr].dropna()
        vol_s = volumeRaw_df.loc[past_dates, tkr].dropna()
        com   = px_s.index.intersection(vol_s.index)
        if len(com) < 3: return False
        dv = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
        return (dv * advp_cap) / current_aum >= min_weight

    # Split current portfolio into liquid / illiquid
    liquid   = [t for t in weights.index if _is_liquid(t)]
    illiquid = [t for t in weights.index if not _is_liquid(t)]

    if not illiquid:
        return weights, set()   # nothing to do — (weights, affected_tickers)

    # Replace illiquid with next-best liquid candidates
    # First search pre-filtered candidates, then full_universe if still short
    replacements = []
    _already = set(liquid) | set(illiquid)

    def _fill_from(pool):
        for tkr in pool.sort_values(ascending=False).index:
            if len(liquid) + len(replacements) >= target_n:
                break
            if tkr in _already or tkr not in pxs_cols:
                continue
            if _is_liquid(tkr):
                replacements.append(tkr)
                _already.add(tkr)

    _fill_from(candidates)

    # If still short, search full universe beyond pre-filtered candidates
    if len(liquid) + len(replacements) < target_n and full_universe is not None:
        _extra = full_universe.drop(
            index=[t for t in full_universe.index if t in _already],
            errors='ignore')
        _fill_from(_extra)

    final = liquid + replacements
    n     = len(final)
    if n == 0:
        return weights, set(illiquid)   # fallback — nothing survived

    # Track affected tickers: excluded (illiquid, not replaced) + capped
    _excluded = set(illiquid) - set(replacements)

    # Compute ADVP caps for all stocks
    _advp_caps = {}
    for tkr in final:
        if tkr not in pxs_cols or tkr not in volumeRaw_df.columns:
            continue
        px_s  = Pxs_df.loc[past_dates, tkr].dropna()
        vol_s = volumeRaw_df.loc[past_dates, tkr].dropna()
        com   = px_s.index.intersection(vol_s.index)
        if len(com) < 3: continue
        dv    = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
        cap_w = (dv * advp_cap) / current_aum
        if cap_w < max_weight:
            _advp_caps[tkr] = max(cap_w, min_weight)

    # Assign stocks to tiers (top half gets concentrated allocation)
    n_top  = int(np.ceil(n / 2))
    n_bot  = n - n_top
    top_tks = final[:n_top]
    bot_tks = final[n_top:]

    # Build weights tier by tier, respecting ADVP caps and min/max bounds
    # Each tier gets a budget: top gets top_a, bot gets bot_a
    if conc_factor == 1.0 or n_bot == 0:
        top_a = 1.0
        bot_a = 0.0
    else:
        top_a = conc_factor / (conc_factor + 1.0)
        bot_a = 1.0 / (conc_factor + 1.0)

    def _alloc_tier(tickers, budget):
        """Allocate budget across tickers respecting ADVP caps and max_weight."""
        if not tickers or budget <= 0:
            return pd.Series(dtype=float), 0.0
        w = pd.Series(budget / len(tickers), index=tickers)
        # Iteratively cap and redistribute within tier
        for _ in range(20):
            changed = False
            for tkr in list(w.index):
                cap = min(_advp_caps.get(tkr, max_weight), max_weight)
                if w[tkr] > cap + 1e-9:
                    w[tkr] = cap
                    changed = True
            # Renorm to budget
            if w.sum() > 1e-12:
                w = w / w.sum() * budget
            if not changed:
                break
        return w, budget - w.sum()

    w_top, excess_top = _alloc_tier(top_tks, top_a)
    w_bot, excess_bot = _alloc_tier(bot_tks, bot_a)

    # If top tier has excess (e.g. all stocks capped), try to absorb in bottom
    # and vice versa
    if excess_top > 1e-9 and not w_bot.empty:
        w_bot2, _ = _alloc_tier(list(w_bot.index), bot_a + excess_top)
        w_bot = w_bot2
    if excess_bot > 1e-9 and not w_top.empty:
        w_top2, _ = _alloc_tier(list(w_top.index), top_a + excess_bot)
        w_top = w_top2

    w_new = pd.concat([w_top, w_bot])
    if w_new.empty:
        return weights, _excluded

    # Final renorm to sum to 1.0
    if w_new.sum() > 0:
        w_new = w_new / w_new.sum()

    _capped = {t for t in final if t in _advp_caps and
               w_new.get(t, 0) <= _advp_caps[t] + 1e-9 and
               t not in _excluded}
    # Only flag as capped if the ADVP cap was actually binding
    _capped = {t for t in _advp_caps if t in w_new and
               _advp_caps[t] < (top_a / n_top if t in top_tks else
                                 bot_a / n_bot if n_bot > 0 else top_a / n_top)}

    _affected = _excluded | _capped
    w_final = w_new / w_new.sum() if w_new.sum() > 0 else w_new
    return w_final, _affected


def _apply_advp_cap(weights: pd.Series,
                    dt:      pd.Timestamp,
                    Pxs_df:  pd.DataFrame,
                    volumeRaw_df: pd.DataFrame,
                    current_aum: float,
                    advp_cap:    float,
                    min_weight:  float,
                    max_weight:  float) -> tuple:
    """
    Apply ADVP cap to portfolio weights.

    For each stock, compute median dollar ADV over last ADV_WINDOW days,
    then cap the dollar allocation at advp_cap * ADV.

    If the cap falls below min_weight, the stock is excluded entirely
    (including it at min_weight would violate the cap).

    Excess weight is redistributed iteratively to remaining uncapped stocks.

    Returns (capped_weights, capped_set) where capped_set is the set of
    tickers where the cap was binding.
    """
    if volumeRaw_df is None or current_aum <= 0:
        return weights, set()

    tickers = weights.index.tolist()

    # Compute median dollar ADV and derived max weight for each stock
    past_dates = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
    adv_cap_w  = {}
    excluded   = set()
    for tkr in tickers:
        if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
            adv_cap_w[tkr] = max_weight
            continue
        px_s   = Pxs_df.loc[past_dates, tkr].dropna()
        vol_s  = volumeRaw_df.loc[past_dates, tkr].dropna()
        common = px_s.index.intersection(vol_s.index)
        if len(common) < 3:
            adv_cap_w[tkr] = max_weight
            continue
        dollar_vol = (px_s.reindex(common) * vol_s.reindex(common).replace({0: np.nan})).median()
        max_dollar = dollar_vol * advp_cap
        cap_w      = max_dollar / current_aum if current_aum > 0 else max_weight
        if cap_w < min_weight:
            # Cap is below floor — exclude stock entirely
            excluded.add(tkr)
            adv_cap_w[tkr] = 0.0
        else:
            adv_cap_w[tkr] = min(cap_w, max_weight)

    # Remove excluded stocks and redistribute their weight
    w = weights.copy()
    if excluded:
        w = w.drop(index=excluded, errors='ignore')
        if w.sum() > 0:
            w = w / w.sum()   # renormalise before capping

    # Iterative water-filling for remaining stocks
    # Set of tickers where ADVP cap (not max_weight) is the binding constraint
    advp_binding = {tkr for tkr, cap in adv_cap_w.items()
                    if cap < max_weight - 1e-6 and tkr not in excluded}

    # Track raw ADVP caps (before min with max_weight) to distinguish binding constraint
    advp_only_cap = {}   # cap derived purely from ADVP (may be > max_weight)
    for tkr in tickers:
        if tkr in excluded:
            advp_only_cap[tkr] = 0.0
        elif tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
            advp_only_cap[tkr] = max_weight * 10   # effectively unconstrained by ADVP
        else:
            advp_only_cap[tkr] = adv_cap_w.get(tkr, max_weight * 10)
            # If adv_cap_w was set as min(cap_w, max_weight), recover raw cap_w
            # by checking if it equals max_weight (means ADVP wasn't binding)

    water_capped = set()   # only stocks where ADVP cap (not max_weight) was binding
    for _ in range(20):
        excess = 0.0
        newly_capped = set()
        for tkr in w.index:
            cap = adv_cap_w.get(tkr, max_weight)
            if w.get(tkr, 0) > cap + 1e-8:
                excess += w[tkr] - cap
                w[tkr]  = cap
                # Only flag if ADVP cap is strictly below max_weight
                if adv_cap_w.get(tkr, max_weight) < max_weight - 1e-6:
                    newly_capped.add(tkr)
        water_capped |= newly_capped
        if excess < 1e-6:
            break
        uncapped = [t for t in w.index if t not in water_capped]
        if not uncapped:
            break
        total_uncapped = w.reindex(uncapped).sum()
        if total_uncapped <= 0:
            break
        for tkr in uncapped:
            w[tkr] += excess * (w[tkr] / total_uncapped)

    # Re-normalise and apply max_weight cap only
    # (floor is NOT applied here — caller's _mb_floor_then_cap handles that
    #  for alpha/hybrid; for MVO cached weights we just cap and renorm)
    w = w[w > 1e-8]
    if w.sum() > 0:
        w = w / w.sum()
    # Apply max_weight cap iteratively
    for _ in range(20):
        over = w > max_weight + 1e-9
        if not over.any():
            break
        excess  = (w[over] - max_weight).sum()
        w[over] = max_weight
        under   = ~over
        if w[under].sum() > 1e-12:
            w[under] += excess * (w[under] / w[under].sum())
        else:
            w += excess / len(w)
    # Return only water-filled stocks as capped (*** flag) — excluded handled upstream
    return w, water_capped


def _mb_solve_mvo(dt, candidates, composite_scores, Pxs_df, sectors_s,
                   volumeTrd_df, model_version, pca_var_threshold,
                   ic, max_weight, min_weight, zscore_cap,
                   risk_aversion,
                   X_snapshots=None, snapshot_dates=None,
                   top_n=20, min_cov_matrices=2, lambda_dfs=None):
    """
    Exact replica of mvo_diagnostics.run_mvo_diagnostics portfolio logic:
      1. Build 4 cov matrices + ensemble
      2. Run _mb_max_alpha_portfolio on each of the 5 matrices
      3. Count matrix appearances per stock (excluding ensemble)
      4. Eligible = stocks appearing in >= min_cov_matrices of the 4 matrices
         (if min_cov_matrices=0, use all candidates)
      5. Final solve: _mb_max_alpha_portfolio on ensemble, eligible universe
    Returns (weights pd.Series, diagnostics dict).
    """
    # -- Get cached X snapshot -------------------------------------------------
    X_df_cached = None
    if X_snapshots is not None and snapshot_dates is not None:
        valid_snaps = [d for d in snapshot_dates if d <= dt]
        if valid_snaps:
            X_df_cached = X_snapshots[valid_snaps[-1]]

    # -- Return history — strictly up to dt (no lookahead) --------------------
    ret_df = (Pxs_df.loc[:dt, candidates].pct_change()
              .dropna(how='all')
              .iloc[-MVO_LOOKBACK:]
              .dropna(axis=1, how='any'))
    valid = [t for t in candidates if t in ret_df.columns]
    if len(valid) < top_n:
        warnings.warn(f"  {dt.date()} SKIP: only {len(valid)} valid candidates")
        return pd.Series(dtype=float), {}

    # -- Build covariance matrices ---------------------------------------------
    try:
        valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens = \
            _mb_build_cov_matrices(dt, valid, Pxs_df, sectors_s,
                                    volumeTrd_df, model_version, pca_var_threshold,
                                    X_df_cached=X_df_cached, lambda_dfs=lambda_dfs)
    except Exception as e:
        warnings.warn(f"  {dt.date()} COV MATRIX FAILED: {e}")
        return pd.Series(dtype=float), {}

    # -- Alpha signal ----------------------------------------------------------
    vol_daily = _mvo_ewma_vol(ret_df[valid], MVO_EWMA_HL)
    vol_ann   = vol_daily * np.sqrt(252)
    z_s       = composite_scores.reindex(valid).fillna(0.0)
    z_capped  = z_s.clip(-zscore_cap, zscore_cap)
    alpha     = (ic * vol_daily * z_capped).fillna(0.0)

    # -- Per-matrix MVO (unconstrained, post-solve floor+cap) ------------------
    matrices = {
        'E': Sigma_emp,
        'L': Sigma_lw,
        'F': Sigma_factor,
        'P': Sigma_pca,
        'Ens': Sigma_ens,
    }
    top_n_by = {}
    for mname, Sigma in matrices.items():
        w_m = _mb_max_alpha_portfolio(Sigma, alpha, valid,
                                       risk_aversion, max_weight, min_weight)
        top_n_by[mname] = w_m.nlargest(top_n).index.tolist()

    # -- Eligibility: count appearances in 4 individual matrices (not ensemble) -
    count_matrices = ['E', 'L', 'F', 'P']
    selection_count = {t: sum(1 for m in count_matrices
                              if t in top_n_by[m])
                       for t in valid}

    if min_cov_matrices > 0:
        eligible = [t for t in valid
                    if selection_count.get(t, 0) >= min_cov_matrices]
        # Graceful fallback: lower threshold until we have enough stocks
        thresh = min_cov_matrices
        while len(eligible) < top_n and thresh > 0:
            thresh -= 1
            eligible = [t for t in valid
                        if selection_count.get(t, 0) >= thresh]
    else:
        eligible = valid  # alpha-only mode

    # -- Final MVO on ensemble cov, eligible universe --------------------------
    elig_idx  = [valid.index(t) for t in eligible]
    S_ens_el  = Sigma_ens[np.ix_(elig_idx, elig_idx)]
    alpha_el  = alpha.reindex(eligible).fillna(0.0)

    w_out = _mb_max_alpha_portfolio(S_ens_el, alpha_el, eligible,
                                     risk_aversion, max_weight, min_weight)
    # Trim to exactly top_n by keeping highest-weight stocks, renorm
    if len(w_out) > top_n:
        w_out = w_out.nlargest(top_n)
        w_out = w_out / w_out.sum()


    # -- Diagnostics dict for display ------------------------------------------
    diag = {
        'n_valid':    len(valid),
        'n_eligible': len(eligible),
        'vol_ann'        : vol_ann,
        'z_s'            : z_s,
        'alpha_ann'      : alpha * 252,
        'top_n_by'       : {k: v for k, v in top_n_by.items() if k != 'Ens'},
        'selection_count': selection_count,
        'eligible'       : eligible,
        'ic_used'        : ic,
    }
    return w_out, diag



# ===============================================================================
# BACKTEST ENGINE
# ===============================================================================

def _ensure_tables():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {DAILY_PORT_TBL} (
                date          DATE        NOT NULL,
                model_version VARCHAR(10) NOT NULL,
                params_hash   VARCHAR(16) NOT NULL,
                strategy      VARCHAR(16) NOT NULL,
                weights_json  TEXT        NOT NULL,
                PRIMARY KEY (date, model_version, params_hash, strategy)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {DAILY_TRIGGER_TBL} (
                date               DATE        NOT NULL,
                model_version      VARCHAR(10) NOT NULL,
                params_hash        VARCHAR(16) NOT NULL,
                implied_turnover   NUMERIC(8,6),
                vol_diff           NUMERIC(8,6),
                PRIMARY KEY (date, model_version, params_hash)
            )
        """))


def _get_cached_dates(ph, mv):
    try:
        _ensure_tables()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT date FROM {DAILY_PORT_TBL}
                WHERE params_hash=:ph AND model_version=:mv AND strategy='alpha'
            """), {'ph': ph, 'mv': mv}).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_portfolios(dt, mv, ph, portfolios):
    for strategy, w in portfolios.items():
        if w is None or w.empty:
            continue
        wj = json.dumps({t: float(v) for t, v in w.items()})
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {DAILY_PORT_TBL}
                    (date, model_version, params_hash, strategy, weights_json)
                VALUES (:dt, :mv, :ph, :st, :wj)
                ON CONFLICT (date, model_version, params_hash, strategy)
                DO UPDATE SET weights_json = EXCLUDED.weights_json
            """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': mv,
                   'ph': ph, 'st': strategy, 'wj': wj})


def _save_triggers(dt, mv, ph, implied_to, vol_diff):
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {DAILY_TRIGGER_TBL}
                (date, model_version, params_hash,
                 implied_turnover, vol_diff)
            VALUES (:dt, :mv, :ph, :ito, :vd)
            ON CONFLICT (date, model_version, params_hash)
            DO UPDATE SET
                implied_turnover = EXCLUDED.implied_turnover,
                vol_diff         = EXCLUDED.vol_diff
        """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': mv, 'ph': ph,
               'ito': float(implied_to), 'vd': float(vol_diff)})


# -- Vol helper ----------------------------------------------------------------
def _portfolio_vol(w, Pxs_df, dt, lookback=VOL_LOOKBACK):
    """Annualized realized vol of portfolio w over last `lookback` trading days."""
    if w is None or w.empty:
        return 0.0
    tickers = [t for t in w.index if t in Pxs_df.columns]
    if not tickers:
        return 0.0
    px = Pxs_df.loc[:dt, tickers].iloc[-lookback:]
    if len(px) < 10:
        return 0.0
    rets   = px.pct_change().dropna()
    w_s    = w.reindex(tickers).fillna(0)
    if w_s.sum() > 0:
        w_s = w_s / w_s.sum()
    port_r = (rets * w_s).sum(axis=1)
    return float(port_r.std() * np.sqrt(252))


# -- Main builder --------------------------------------------------------------

# ================================================================================
# HEDGE ENGINE (embedded from hedge_engine.py)
# ================================================================================




# ================================================================================
# HELPERS
# ================================================================================

def _portfolio_returns(weights_by_date: dict,
                       Pxs_df:          pd.DataFrame,
                       all_dates:        pd.DatetimeIndex) -> pd.Series:
    """
    Build daily portfolio return series using fixed rebalance weights.
    Between rebalances, apply the last rebalance weights to daily stock returns.
    """
    rebal_dates = sorted(weights_by_date.keys())
    port_rets   = {}

    for i, dt in enumerate(all_dates):
        # Find last rebalance date <= dt
        past = [d for d in rebal_dates if d <= dt]
        if not past:
            continue
        last_rebal = past[-1]
        w = weights_by_date[last_rebal]
        tickers = [t for t in w.index if t in Pxs_df.columns]
        if not tickers:
            continue
        prev_dt_idx = all_dates.get_loc(dt)
        if prev_dt_idx == 0:
            continue
        prev_dt = all_dates[prev_dt_idx - 1]
        if prev_dt not in Pxs_df.index or dt not in Pxs_df.index:
            continue
        px_prev = Pxs_df.loc[prev_dt, tickers]
        px_cur  = Pxs_df.loc[dt,      tickers]
        ret     = (px_cur / px_prev - 1).fillna(0)
        port_rets[dt] = (w.reindex(tickers).fillna(0) * ret).sum()

    return pd.Series(port_rets, name='portfolio_ret')


def _compute_beta(port_ret_s: pd.Series,
                  inst_ret_s: pd.Series,
                  window:     int) -> float:
    """OLS beta of portfolio returns to instrument returns over last `window` days."""
    common = port_ret_s.dropna().index.intersection(inst_ret_s.dropna().index)
    common = common[-window:] if len(common) >= window else common
    if len(common) < window // 2:
        return np.nan
    p = port_ret_s.reindex(common)
    h = inst_ret_s.reindex(common)
    var_h = h.var()
    return p.cov(h) / var_h if var_h > 0 else np.nan


def _compute_corr(port_ret_s: pd.Series,
                  inst_ret_s: pd.Series,
                  window:     int) -> float:
    """Rolling correlation of portfolio to instrument over last `window` days."""
    common = port_ret_s.dropna().index.intersection(inst_ret_s.dropna().index)
    common = common[-window:] if len(common) >= window else common
    if len(common) < window // 2:
        return np.nan
    return port_ret_s.reindex(common).corr(inst_ret_s.reindex(common))


def _get_effectiveness(signal_df: pd.DataFrame,
                       dt:        pd.Timestamp,
                       mav_win:   int) -> float:
    """Smoothed effectiveness at date dt (MAV of last mav_win values).
    Returns 1.0 (neutral) if no effectiveness history available yet."""
    if 'effectiveness' not in signal_df.columns:
        return 1.0
    past = signal_df['effectiveness'].loc[:dt].dropna()
    if past.empty:
        return 1.0
    return past.iloc[-mav_win:].mean()


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def run_backtest(
    Pxs_df,
    sectors_s,
    weights_by_year,
    regime_s,
    volumeTrd_df        = None,
    weights_by_date     = None,
    mode                = 'incremental',   # 'incremental' | 'rebuild'
    rebuild_cov         = False,           # True only when new stocks added to DB
    # Portfolio construction
    top_n               = 25,
    universe_mult       = 5,
    conc_factor         = 2.0,
    prefilt_pct         = 0.5,
    min_weight          = MVO_MIN_WEIGHT,
    max_weight          = MVO_MAX_WEIGHT,
    ic                  = MVO_DEFAULT_IC,
    zscore_cap          = MVO_ZSCORE_CAP,
    pca_var_threshold   = MVO_PCA_VAR_THRESH,
    risk_aversion       = 10,
    min_cov_matrices    = MB_MIN_COV_MATRICES,
    model_version       = MB_MODEL_VER,
    # ADVP
    advp_cap            = 0.04,
    # Dynamic rebalancing
    min_hold_days       = DYN_MIN_HOLD_DAYS,
    to_thresh_alpha     = DYN_TO_THRESHOLD_ALPHA,
    to_thresh_hybrid    = DYN_TO_THRESHOLD_HYBRID,
    to_thresh_mvo       = DYN_TO_THRESHOLD_MVO,
    voldiff_cap         = DYN_VOLDIFF_CAP,
    # Smart hybrid / DD regime thresholds
    sh_dd_alpha         = SH_DD_ALPHA,
    sh_dd_hybrid        = SH_DD_HYBRID,
    sh_dd_exit_alpha    = SH_DD_EXIT_ALPHA,
    sh_dd_exit_hybrid   = SH_DD_EXIT_HYBRID,
    sh_persist_days     = SH_PERSIST_DAYS,
    # Drawdown policy
    dd_levels           = DD_LEVELS,
    dd_level_regime     = DD_LEVEL_REGIME,
    dd_reentry_pct      = DD_REENTRY_PCT,
    dd_reentry_confirm  = DD_REENTRY_CONFIRM,
    # Hedge — omit hedge_multi / hedges_l to disable hedge layer (6 strategies)
    hedge_multi         = None,            # from run_macro_hedge_cached
    hedges_l            = None,            # list of hedge instrument tickers
    eff_floor           = EFF_FLOOR,
    corr_floor          = CORR_FLOOR,
    hedge_ratio         = HEDGE_RATIO,
    max_hedge           = MAX_HEDGE,
    hedge_trigger_assets= TRIGGER_ASSETS,
    # MR exclusion
    mr_k                = MR_K,
    mr_cap              = MR_CAP,
    # Misc
    aum                 = AUM,
    trading_cost_bps    = TRADING_COST_BPS,
    rebal_freq          = 15,
    quality_floor       = QUALITY_FLOOR,
    mom_filter          = MOM_FILTER,
    tier2_t             = TIER2_T,       # total stocks incl overlay (0=disabled)
    tierone_alloc       = TIERONE_ALLOC, # fraction to tier 1 core
    corr_filter         = CORR_FILTER,   # fraction kept after correlation filter
):
    """
    Unified single-pass backtest for all 9 strategies.
    Each strategy tracks its own NAV, drawdown, and rebalancing schedule.
    ADVP filter uses each strategy's exact current AUM at every rebalance.
    """
    import json, warnings
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    print(f"\n{'='*72}")
    print(f"  UNIFIED BACKTEST  |  mode={mode}  |  top_n={top_n}  |  AUM=${aum/1e6:.1f}M")
    print(f"{'='*72}")

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    # Derive raw volume from volumeTrd_df (same structure, used for ADVP)
    volumeRaw_df = volumeTrd_df.copy() if volumeTrd_df is not None else None

    # Hedge layer only active when hedge_multi and hedges_l are provided
    hedge_enabled = hedge_multi is not None and hedges_l is not None

    # ── User prompts (same as run_mvo_backtest) ───────────────────────────────
    print("  PORTFOLIO CONSTRUCTION OPTIONS:")
    topn_input    = input(f"  Number of stocks [default={top_n}]: ").strip()
    rebal_input   = input(f"  Rebalancing frequency in days [default={rebal_freq}]: ").strip()
    advp_input    = input(f"  ADVP cap — max % of median daily $ volume per stock [default={advp_cap:.0%}]: ").strip()
    prefilt_input = input("  Pre-filter fraction by quality score 0<x<=1 (or Enter for none): ").strip()
    conc_input    = input("  Concentration factor for Pure Alpha >=1.0 (or Enter for equal weight): ").strip()
    min_cov_inp   = input(f"  Min cov matrices for stock selection "
                          f"(0=alpha-only, default={min_cov_matrices}): ").strip()

    top_n            = int(topn_input)           if topn_input    else top_n
    rebal_freq       = int(rebal_input)          if rebal_input   else rebal_freq
    advp_cap         = float(advp_input) / 100   if advp_input    else advp_cap
    prefilt_pct      = float(prefilt_input)      if prefilt_input else prefilt_pct
    if prefilt_pct <= 0 or prefilt_pct > 1: prefilt_pct = 1.0
    conc_factor      = float(conc_input)         if conc_input    else conc_factor
    if conc_factor < 1.0: conc_factor = 1.0
    min_cov_matrices = int(min_cov_inp)          if min_cov_inp   else min_cov_matrices
    min_cov_matrices = max(0, min(min_cov_matrices, 4))

    # ── Tier 2 overlay prompt ─────────────────────────────────────────────────
    t2_input  = input(f"  Total stocks incl overlay T (>={top_n}, Enter=disabled): ").strip()
    if t2_input:
        tier2_t = int(t2_input)
        t_cap   = 2 * top_n
        if tier2_t < top_n:
            print(f"  WARNING: T={tier2_t} < N={top_n} — overlay disabled, TIERONE_ALLOC=1.0")
            tier2_t = 0
        elif tier2_t > t_cap:
            print(f"  WARNING: T={tier2_t} > 2×N={t_cap} cap — setting T={t_cap}")
            tier2_t = t_cap
    else:
        tier2_t = 0
    tier2_active = tier2_t > top_n

    n_cands = top_n * universe_mult
    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"advp_cap={advp_cap:.1%}, prefilt={prefilt_pct:.0%}, "
          f"conc={conc_factor:.1f}x, min_cov={min_cov_matrices}, "
          f"mom_filter={mom_filter}x"
          + (f", T={tier2_t} (tier1={tierone_alloc:.0%})" if tier2_active else ""))
    print(f"  Alpha/Hybrid pool: {mom_filter*top_n} stocks ({mom_filter}x{top_n})")
    print(f"  MVO candidate pool: {n_cands} stocks ({universe_mult}x{top_n})\n")
    ph           = _make_make_params_hash(ic, max_weight, min_weight, zscore_cap,
                                          pca_var_threshold, universe_mult, risk_aversion,
                                          top_n, conc_factor, prefilt_pct,
                                          min_cov_matrices, model_version, mom_filter)
    print(f"  params_hash        : {ph}")
    print(f"  quality_fingerprint: {_make_quality_fingerprint()}")
    start_date   = MB_START_DATE
    # Guardrail: ensure at least 12 months of PIT weight history before backtest starts
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(text(
                "SELECT MIN(cutoff_date) FROM quality_weights_pit"
            )).fetchone()
        _first_pit = pd.Timestamp(row[0]) if row and row[0] else None
        if _first_pit:
            _min_start = _first_pit + pd.DateOffset(months=12)
            _min_start = Pxs_df.index[Pxs_df.index >= _min_start][0] \
                         if any(Pxs_df.index >= _min_start) else _min_start
            if _min_start > start_date:
                print(f"  Guardrail: MB_START_DATE adjusted from {start_date.date()} "
                      f"to {_min_start.date()} "
                      f"(first PIT weights: {_first_pit.date()} + 12m)")
                start_date = _min_start
    except Exception as _e:
        print(f"  WARNING: PIT guardrail check failed ({_e}) — using MB_START_DATE as-is")
    trading_days = Pxs_df.index[Pxs_df.index >= start_date]
    pxs_cols     = set(Pxs_df.columns)
    n_cands      = top_n * universe_mult
    ext_st       = MB_START_DATE - pd.Timedelta(days=365)
    today_ts     = Pxs_df.index[-1]

    # ── 1. Build composite scores (all dates) ─────────────────────────────────
    print("  Building composite scores...")
    universe    = get_universe(Pxs_df, sectors_s, ext_st)
    calc_dates  = pd.DatetimeIndex(generate_calc_dates(Pxs_df, step_days=rebal_freq))
    calc_dates  = calc_dates[calc_dates >= start_date]
    # Ensure last trading day is always included for hypothetical trade summary
    if today_ts not in calc_dates:
        calc_dates_full = calc_dates.append(pd.DatetimeIndex([today_ts]))
    else:
        calc_dates_full = calc_dates

    composite_by_date, score_dfs = _cb_build_composite_scores(
        universe        = universe,
        calc_dates      = calc_dates_full,
        Pxs_df          = Pxs_df,
        sectors_s       = sectors_s,
        weights_by_year = weights_by_year,
        regime_s        = regime_s,
        volumeTrd_df    = volumeTrd_df,
        model_version   = model_version,
        exclude_factors = ['OU'],
        weights_by_date = weights_by_date,
    )
    print(f"  Composite scores: {len(composite_by_date)} dates")
    # Diagnostic: print top-5 composite scores at first calc_date to detect lookahead
    _diag_dt = sorted(composite_by_date.keys())[0]
    _diag_s  = composite_by_date[_diag_dt].dropna().sort_values(ascending=False)
    print(f"\n  [DIAG] Composite at {_diag_dt.date()} top-5: "
          + "  ".join(f"{t}={v:.4f}" for t, v in _diag_s.head(5).items()))
    if rebuild_cov:
        _mb_clear_x_cache(model_version)
        print("  X snapshot cache cleared (rebuild_cov=True)")
    X_snapshots = dict(_mb_load_x_cache(model_version))
    snapshot_dates = sorted(X_snapshots.keys())
    print(f"  X snapshots: {len(X_snapshots)} already cached  "
          f"(new ones will be built on-demand during main loop)")
    lambda_dfs = _mb_load_lambda_dfs(model_version)
    print(f"  Lambda tables loaded: {len(lambda_dfs['scalar'])} scalar, "
          f"{len(lambda_dfs['macro'].columns)} macro, "
          f"{len(lambda_dfs['sector'].columns)} sector factors")

    # Helper to get/build X snapshot for a given date on demand
    _x_snap_universe     = get_universe(Pxs_df, sectors_s, ext_st)
    _x_factor_names, _x_sec_cols = _mb_get_factor_names(model_version, lambda_dfs=lambda_dfs)
    _x_snap_dates_needed = set(_mb_month_end_dates(
        Pxs_df.index[Pxs_df.index >= start_date - pd.Timedelta(days=90)],
        Pxs_df.index[-1]))

    def _ensure_x_snapshot(dt):
        """Build X snapshot for the month-end prior to dt if not yet cached."""
        # Find the most recent month-end date <= dt that we need
        needed = [d for d in _x_snap_dates_needed if d <= dt and d not in X_snapshots]
        for x_dt in sorted(needed):
            with _SuppressOutput():
                xdf = _mb_build_X(x_dt, _x_snap_universe, _x_factor_names,
                                  _x_sec_cols, Pxs_df, sectors_s,
                                  volumeTrd_df, model_version)
            if xdf is not None:
                xdf = xdf.fillna(0.0)
                X_snapshots[x_dt] = xdf
                if x_dt not in snapshot_dates:
                    snapshot_dates.append(x_dt)
                    snapshot_dates.sort()
                try:
                    _mb_save_x_snapshot(x_dt, model_version, xdf)
                except Exception:
                    pass

    # ── 3. Quality scores for baseline ───────────────────────────────────────
    print("  Loading quality scores...")
    all_tickers  = list(sectors_s.index)
    quality_wide = _mb_load_quality_scores(all_tickers, calc_dates_full)

    print("  Loading quality scores...")
    all_tickers  = list(sectors_s.index)
    quality_wide = _mb_load_quality_scores(all_tickers, calc_dates_full)

    # ── Factor exposure tracking ──────────────────────────────────────────────
    # Load all 5 factors for weighted-average exposure display and caching
    EXPOSURE_FACTORS = ['Quality', 'Value', 'Mom_12M1', 'Idio_Mom', 'OU', 'SI']
    EXPOSURE_TBL     = 'portfolio_factor_exposures'

    def _ensure_exposure_table():
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {EXPOSURE_TBL} (
                    date         DATE         NOT NULL,
                    strategy     VARCHAR(20)  NOT NULL,
                    params_hash  VARCHAR(20)  NOT NULL,
                    quality      FLOAT,
                    value        FLOAT,
                    mom_12m1     FLOAT,
                    idio_mom     FLOAT,
                    ou           FLOAT,
                    si           FLOAT,
                    PRIMARY KEY (date, strategy, params_hash)
                )
            """))

    def _load_factor_scores_wide(calc_dates):
        """Load Quality, Value, Idio_Mom, OU, SI as wide DataFrames."""
        scores = {}

        def _pctrank_row(s):
            """Cross-sectional percentile rank [0,1] — matches composite builder."""
            s = s.dropna()
            if len(s) < 2: return s
            r = s.rank(method='average')
            return (r - 1) / (len(r) - 1)

        # Quality — already loaded, just z-score
        scores['Quality'] = quality_wide.copy()

        # Value
        try:
            scores['Value'] = _mb_load_value_scores(all_tickers, calc_dates)
        except Exception:
            scores['Value'] = pd.DataFrame(index=calc_dates,
                                           columns=all_tickers, dtype=float)

        # Idio Momentum — percentile rank, consistent with composite builder
        try:
            with ENGINE.connect() as conn:
                res = pd.read_sql(text(
                    "SELECT date, ticker, resid FROM v2_factor_residuals_quality "
                    "ORDER BY date"
                ), conn)
            res['date'] = pd.to_datetime(res['date'])
            res_piv = res.pivot_table(index='date', columns='ticker',
                                      values='resid', aggfunc='last')
            mom_raw = res_piv.rolling(MOM_LONG, min_periods=MOM_LONG//2).sum()
            all_d   = calc_dates.union(mom_raw.index).sort_values()
            mom_ff  = mom_raw.reindex(all_d).ffill().reindex(calc_dates)
            # Percentile rank [0,1] — matches composite builder
            scores['Idio_Mom'] = mom_ff.apply(_pctrank_row, axis=1)
        except Exception:
            scores['Idio_Mom'] = pd.DataFrame(index=calc_dates,
                                              columns=all_tickers, dtype=float)

        # Mom_12M1 — percentile rank, consistent with composite builder
        try:
            with ENGINE.connect() as conn:
                res = pd.read_sql(text(
                    "SELECT date, ticker, resid FROM v2_factor_residuals_mkt "
                    "ORDER BY date"
                ), conn)
            res['date'] = pd.to_datetime(res['date'])
            res_piv  = res.pivot_table(index='date', columns='ticker',
                                       values='resid', aggfunc='last')
            m12_raw  = res_piv.rolling(MOM_LONG, min_periods=MOM_LONG//2).sum()
            if MOM_SKIP > 0:
                m12_raw = m12_raw.shift(MOM_SKIP)
            all_d    = calc_dates.union(m12_raw.index).sort_values()
            m12_ff   = m12_raw.reindex(all_d).ffill().reindex(calc_dates)
            # Percentile rank [0,1] — matches composite builder
            scores['Mom_12M1'] = m12_ff.apply(_pctrank_row, axis=1)
        except Exception:
            scores['Mom_12M1'] = pd.DataFrame(index=calc_dates,
                                              columns=all_tickers, dtype=float)
        try:
            with ENGINE.connect() as conn:
                ou = pd.read_sql(text(
                    "SELECT date, ticker, ou_score FROM v2_ou_reversion_df "
                    "ORDER BY date"
                ), conn)
            ou['date'] = pd.to_datetime(ou['date'])
            ou_piv  = ou.pivot_table(index='date', columns='ticker',
                                     values='ou_score', aggfunc='last')
            all_d   = calc_dates.union(ou_piv.index).sort_values()
            ou_ff   = ou_piv.reindex(all_d).ffill().reindex(calc_dates)
            scores['OU'] = ou_ff.apply(_mb_zscore, axis=1)
        except Exception:
            scores['OU'] = pd.DataFrame(index=calc_dates,
                                        columns=all_tickers, dtype=float)

        # SI — from short_interest_data, introspect column names first
        try:
            with ENGINE.connect() as conn:
                cols = conn.execute(text("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'short_interest_data'
                    ORDER BY ordinal_position
                """)).fetchall()
            col_names = [c[0] for c in cols]
            # Find SI float column — could be si_float, si_pct, short_interest etc.
            si_col = next((c for c in col_names
                          if 'float' in c.lower() or 'pct' in c.lower()
                          or 'short' in c.lower() and 'interest' in c.lower()), None)
            if si_col is None:
                raise ValueError(f"No SI float column found. Columns: {col_names}")
            with ENGINE.connect() as conn:
                si = pd.read_sql(text(f"""
                    SELECT date, ticker, "{si_col}" as si_val
                    FROM short_interest_data
                    WHERE "{si_col}" IS NOT NULL
                    ORDER BY date
                """), conn)
            if si.empty:
                raise ValueError("short_interest_data is empty")
            si['date']   = pd.to_datetime(si['date'])
            si['ticker'] = si['ticker'].str.replace(' US', '', regex=False).str.strip()
            si_piv = si.pivot_table(index='date', columns='ticker',
                                    values='si_val', aggfunc='last')
            all_d  = calc_dates.union(si_piv.index).sort_values()
            si_ff  = si_piv.reindex(all_d).ffill().reindex(calc_dates)
            si_ff  = si_ff.reindex(columns=all_tickers)
            scores['SI'] = si_ff.apply(_mb_zscore, axis=1)
        except Exception as e:
            print(f"  SI scores unavailable ({e}) — will show n/a in display")
            scores['SI'] = pd.DataFrame(index=calc_dates,
                                        columns=all_tickers, dtype=float)

        return scores

    def _get_factor_exposures(w, dt, factor_scores):
        """Compute weighted-average factor exposures for a portfolio."""
        result = {}
        ref_dt = dt
        for fname, fdf in factor_scores.items():
            if fdf.empty:
                result[fname] = np.nan
                continue
            # Get most recent available date
            avail = [d for d in fdf.index if d <= ref_dt]
            if not avail:
                result[fname] = np.nan
                continue
            row = fdf.loc[avail[-1]].reindex(w.index).dropna()
            common = w.index.intersection(row.index)
            if common.empty:
                result[fname] = np.nan
                continue
            w_norm = w.reindex(common)
            w_norm = w_norm / w_norm.sum() if w_norm.sum() > 0 else w_norm
            result[fname] = float((w_norm * row.reindex(common)).sum())
        return result

    def _save_exposure_row(dt, strategy_name, exposures):
        """Save one row of factor exposures to DB."""
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {EXPOSURE_TBL}
                    (date, strategy, params_hash, quality, value,
                     mom_12m1, idio_mom, ou, si)
                VALUES (:date, :strategy, :ph, :quality, :value,
                        :mom_12m1, :idio_mom, :ou, :si)
                ON CONFLICT (date, strategy, params_hash) DO UPDATE SET
                    quality=EXCLUDED.quality, value=EXCLUDED.value,
                    mom_12m1=EXCLUDED.mom_12m1, idio_mom=EXCLUDED.idio_mom,
                    ou=EXCLUDED.ou, si=EXCLUDED.si
            """), {
                'date':     dt.date(), 'strategy': strategy_name, 'ph': ph,
                'quality':  exposures.get('Quality'),
                'value':    exposures.get('Value'),
                'mom_12m1': exposures.get('Mom_12M1'),
                'idio_mom': exposures.get('Idio_Mom'),
                'ou':       exposures.get('OU'),
                'si':       exposures.get('SI'),
            })

    _ensure_exposure_table()
    print("  Loading factor scores for exposure tracking...")
    _factor_scores = _load_factor_scores_wide(calc_dates_full)
    print("  Factor scores loaded.")
    if mode == 'rebuild':
        with ENGINE.connect() as conn:
            conn.execute(text(f"""
                DELETE FROM {DAILY_PORT_TBL}
                WHERE params_hash=:ph AND model_version=:mv
            """), {'ph': ph, 'mv': model_version})
            conn.commit()
        print("  Cache wiped (rebuild mode)")

    cached_dates = set(_get_cached_dates(ph, model_version))

    # ── 5. Per-strategy state ─────────────────────────────────────────────────
    # Strategies: 0=Baseline, 1=Alpha, 2=MVO, 3=Hybrid, 4=Smart,
    #             5=Dynamic, 6=Dyn+Hedge, 7=DD Policy, 8=Excl
    N_STRAT = 9
    S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART, S_DYN, S_HEDGE, S_DD, S_EXCL = range(9)

    STRATEGY_LABELS = {
        S_BASE: 'Baseline', S_ALPHA: 'Alpha', S_MVO: 'MVO',
        S_HYB: 'Hybrid', S_SMART: 'Smart', S_DYN: 'Dynamic',
        S_HEDGE: 'Dyn+Hedge', S_DD: 'DD Policy', S_EXCL: 'Excl',
    }

    # Static strategies rebalance on calc_dates; dynamic on triggers
    STATIC  = {S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART}
    # Dynamic strategies — hedge/DD/excl only active when hedge layer enabled
    DYNAMIC = {S_DYN, S_HEDGE, S_DD, S_EXCL} if hedge_enabled else {S_DYN}
    ACTIVE  = set(range(N_STRAT)) if hedge_enabled else \
              {S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART, S_DYN}
    print(f"  Strategies active: {len(ACTIVE)}  "
          f"({'hedge+DD+Excl enabled' if hedge_enabled else 'no hedge/DD/Excl'})")

    state = [{
        'nav':          1.0,
        'hwm':          1.0,
        'dd':           0.0,
        'gross':        1.0,          # de-grossing level (DD policy)
        'trough':       1.0,          # nav trough for re-entry measurement
        'theo_nav':     1.0,          # theoretical nav (no de-gross costs)
        'weights':      pd.Series(dtype=float),
        'last_rebal':   None,
        'days_held':    0,
        'regime':       'alpha',      # current signal regime
        'deployed_regime': None,      # regime of last actual rebalance
        'dd_level':     -1,           # current DD level index (-1 = none)
        'dd_active':    False,        # True when de-grossed
        'dd_regime_forced': False,    # True when DD forced a regime change
        'reentry_count':0,
        'hedge_acct':   0.0,
        'active_hedges':{},
        'hedge_log':    [],           # list of completed hedge episodes
        'costs':        0.0,          # cumulative trading costs (as NAV fraction)
        'costs_by_year':{},           # {year: cumulative costs in that year}
        'nav_series':   {},           # {date: nav}
        'weights_by_date': {},        # {date: weights}
        'rebal_log':    [],           # list of rebal events
        'dd_log':       [],           # list of DD events
    } for _ in range(N_STRAT)]

    # NAV series storage
    nav_records   = {s: {} for s in range(N_STRAT)}   # {date: nav_level}
    gross_records = {s: {} for s in range(N_STRAT)}   # {date: gross_exposure}

    pxs_cols_idx = Pxs_df.index  # for efficient date lookup in helpers

    # ── Tier 2 overlay helpers ────────────────────────────────────────────────
    def _compute_port_returns(w, dt, lookback):
        """Compute portfolio daily return series over lookback days before dt."""
        past = [d for d in pxs_cols_idx if d < dt]
        past = past[-lookback:] if len(past) >= lookback else past
        if len(past) < 10:
            return pd.Series(dtype=float)
        tks = [t for t in w.index if t in pxs_cols]
        if not tks:
            return pd.Series(dtype=float)
        px       = Pxs_df.loc[past, tks]
        rets     = px.pct_change().fillna(0)
        w_curr   = w.reindex(tks).fillna(0)
        if w_curr.sum() > 0: w_curr /= w_curr.sum()
        port_rets = []
        for d in past[1:]:
            r_d = rets.loc[d].reindex(tks).fillna(0)
            port_rets.append(float((w_curr * r_d).sum()))
            w_curr = w_curr * (1 + r_d)
            if w_curr.sum() > 0: w_curr /= w_curr.sum()
        return pd.Series(port_rets, index=past[1:])

    def _compute_corr_blend(candidates, w_tier1, dt):
        """Compute 126d correlation of each candidate to tier1 portfolio."""
        r126 = _compute_port_returns(w_tier1, dt, 126)
        if len(r126) < 10:
            return pd.Series(0.0, index=candidates)
        corrs = {}
        for tkr in candidates:
            if tkr not in pxs_cols:
                continue
            t126 = Pxs_df[tkr].pct_change().reindex(r126.index).fillna(0)
            c    = float(t126.corr(r126))
            corrs[tkr] = 0.0 if np.isnan(c) else c
        return pd.Series(corrs)

    def _compute_6m1_scores(candidates, dt):
        """6M1 momentum percentile rank for candidates."""
        past = [d for d in pxs_cols_idx if d < dt]
        if len(past) < MOM_6M1_WIN + 1:
            return pd.Series(dtype=float)
        d_start = past[-(MOM_6M1_WIN + 1)]
        d_end   = past[-(MOM_6M1_SKIP + 1)]
        valid   = [t for t in candidates if t in pxs_cols]
        mom     = ((Pxs_df.loc[d_end, valid] - Pxs_df.loc[d_start, valid]) /
                   Pxs_df.loc[d_start, valid].replace(0, np.nan)).dropna()
        if len(mom) < 5: return pd.Series(dtype=float)
        r = mom.rank(method='average')
        return ((r - 1) / (len(r) - 1)).reindex(candidates)

    def _zs_overlay(s):
        s = s.dropna()
        if s.std() < 1e-8: return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / s.std()

    def _build_overlay(w_tier1, dt, non_core_cands, comp_scores):
        """Build tier 2 overlay. Returns Series {ticker: equal_weight_fraction}."""
        if not tier2_active or w_tier1.empty: return pd.Series(dtype=float)
        n_tier1  = len(w_tier1)
        n_ov     = tier2_t - n_tier1
        if n_ov <= 0: return pd.Series(dtype=float)

        # Adjusted tier 1 alloc (same logic as _apply_tier2)
        if n_tier1 > top_n:
            adj = tierone_alloc + (1 - tierone_alloc) * min(
                1.0, (n_tier1 - top_n) / max(1, tier2_t - top_n))
        else:
            adj = tierone_alloc

        # Known overlay weight per stock
        ov_weight_per_stock = (1 - adj) / n_ov

        # Pre-filter: ADVP capacity check
        # A stock is eligible if median_daily_vol × advp_cap / (aum × nav) >= ov_weight
        cands = [t for t in non_core_cands
                 if t not in w_tier1.index and t in _universe_set and t in pxs_cols]
        if volumeRaw_df is not None and not volumeRaw_df.empty:
            past_vol = [d for d in volumeRaw_df.index if d <= dt]
            if len(past_vol) >= 20:
                vol_window = past_vol[-20:]
                # Use nav of the strategy being built — approximate with S_ALPHA nav
                cur_aum = aum * state[S_ALPHA]['nav'] * state[S_ALPHA].get('gross', 1.0)
                if cur_aum > 0:
                    advp_eligible = []
                    for t in cands:
                        if t not in volumeRaw_df.columns:
                            advp_eligible.append(t)  # no volume data — include
                            continue
                        med_vol = float(volumeRaw_df.loc[vol_window, t].median())
                        if pd.isna(med_vol) or med_vol <= 0:
                            advp_eligible.append(t)
                            continue
                        capacity = (med_vol * advp_cap) / cur_aum
                        if capacity >= ov_weight_per_stock:
                            advp_eligible.append(t)
                    cands = advp_eligible

        if len(cands) < n_ov: return pd.Series(dtype=float)

        # Step 1: correlation filter
        corr_s = _compute_corr_blend(cands, w_tier1, dt)
        n_keep  = max(n_ov, int(np.ceil(len(cands) * corr_filter)))
        survivors = corr_s.nsmallest(min(n_keep, len(corr_s))).index.tolist() \
                    if len(corr_s) >= n_ov else corr_s.index.tolist()
        if len(survivors) < n_ov: return pd.Series(dtype=float)

        # Step 2: overlay score components
        c1 = (1 - corr_s.reindex(survivors).fillna(0))
        c2 = _compute_6m1_scores(survivors, dt).fillna(0)
        avail = [d for d in composite_by_date.keys() if d <= dt]
        c3 = composite_by_date[avail[-1]].reindex(survivors).fillna(0) \
             if avail else pd.Series(0.0, index=survivors)

        # Z-score and combine equally
        ov_score = (_zs_overlay(c1).reindex(survivors).fillna(0) +
                    _zs_overlay(c2).reindex(survivors).fillna(0) +
                    _zs_overlay(c3).reindex(survivors).fillna(0)) / 3.0
        top_ov = ov_score.nlargest(n_ov).index.tolist()
        return pd.Series(1.0 / n_ov, index=top_ov)  # equal weights, sum=1 (pre-scaling)

    def _apply_tier2(w_tier1, dt, non_core_cands, comp_scores,
                     force_flat_alloc=False):
        """Combine tier 1 and overlay into final portfolio summing to 1.0.
        Returns (combined_weights, overlay_tickers_set).
        Overlay weight per stock is always fixed at (1-TIERONE_ALLOC)/(T-N),
        regardless of how many overlay slots are available.
        """
        if not tier2_active or w_tier1.empty:
            return w_tier1, set()
        n_tier1 = len(w_tier1)
        n_ov    = tier2_t - n_tier1
        if n_ov <= 0:
            return w_tier1, set()

        # Fixed overlay weight per stock — always (1-TIERONE_ALLOC)/(T-N)
        # This ensures overlay stocks never get outsized allocation
        # when fewer than (T-N) overlay slots are available
        fixed_ov_weight  = (1.0 - tierone_alloc) / max(1, tier2_t - top_n)
        total_ov_alloc   = n_ov * fixed_ov_weight
        tier1_alloc      = 1.0 - total_ov_alloc

        ov = _build_overlay(w_tier1, dt, non_core_cands, comp_scores)
        if ov.empty:
            return w_tier1, set()

        # Scale tier1 to tier1_alloc, each overlay stock gets fixed_ov_weight
        w1_scaled = w_tier1 * tier1_alloc
        ov_scaled = pd.Series(fixed_ov_weight, index=ov.index)
        combined  = pd.concat([w1_scaled, ov_scaled])
        combined  = combined.groupby(level=0).sum()
        if combined.sum() > 0: combined /= combined.sum()
        return combined, set(ov.index)

    # ── Helper: build alpha portfolio from composite scores ───────────────────
    _universe_set = set(universe)   # stable canonical set, independent of Pxs_df end date

    def _build_alpha(dt, cands, excl_set=None):
        """Build concentrated alpha portfolio from ranked candidates."""
        c = cands.dropna()
        c = c.loc[[t for t in c.index if t in _universe_set and t in pxs_cols]]
        if excl_set:
            c = c[~c.index.isin(excl_set)]
        # Do NOT re-expand beyond the passed-in candidate pool
        ap = [t for t in c.sort_values(ascending=False).head(top_n).index
              if t in _universe_set and t in pxs_cols]
        n  = len(ap)
        if n == 0: return pd.Series(dtype=float)
        if conc_factor == 1.0 or n < 2:
            return pd.Series(1.0 / n, index=ap)
        nt = int(np.ceil(n / 2)); nb = n - nt
        ta = conc_factor / (conc_factor + 1.0)
        ba = 1.0 / (conc_factor + 1.0)
        return pd.Series({t: (ta/nt if j < nt else ba/nb)
                          for j, t in enumerate(ap)})

    # ── Helper: build MVO portfolio ───────────────────────────────────────────
    def _build_mvo(dt, cands, comp_scores):
        valid_snaps = [d for d in snapshot_dates if d <= dt]
        if not valid_snaps: return pd.Series(dtype=float)
        with _SuppressOutput():
            w, _ = _mb_solve_mvo(
                dt=dt, candidates=cands.head(n_cands).index.tolist(),
                composite_scores=comp_scores, Pxs_df=Pxs_df,
                sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                model_version=model_version,
                pca_var_threshold=pca_var_threshold,
                ic=ic, max_weight=max_weight, min_weight=min_weight,
                zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                top_n=top_n, min_cov_matrices=min_cov_matrices,
                lambda_dfs=lambda_dfs,
            )
        return w[w > 1e-6] if not w.empty else w

    # ── Helper: apply ADVP filter with correct strategy AUM ──────────────────
    def _apply_advp(w, cands, dt, strat_idx, target_n=None, overlay_stocks=None):
        if volumeRaw_df is None or w.empty: return w, set()
        cur_aum = aum * state[strat_idx]['nav'] * state[strat_idx]['gross']
        if cur_aum <= 0: return w, set()
        _full_u = composite_by_date.get(last_calc_dt)
        if _full_u is not None:
            _full_u = _full_u.dropna().loc[
                [t for t in _full_u.index if t in pxs_cols]]
        result = _advp_filter_and_replace(
            w, cands, dt, Pxs_df, volumeRaw_df,
            cur_aum, advp_cap, min_weight, max_weight, top_n, conc_factor,
            target_n=target_n if target_n is not None else top_n,
            full_universe=_full_u)
        if isinstance(result, tuple):
            return result
        return result, set()

    # ── Helper: compute penalised composite ───────────────────────────────────
    def _penalise(comp, dt):
        if mr_k <= 0: return comp
        cp = comp.copy()
        mr = _load_momentum_exclusions(dt)
        for tkr, sc in mr.items():
            if tkr in cp and sc > 0:
                cp[tkr] = cp[tkr] / np.exp(mr_k * sc)
        return cp

    # ── Helper: compute portfolio turnover ───────────────────────────────────
    def _turnover(w_new, w_old):
        if w_old.empty: return 1.0
        # Deduplicate in case of accidental duplicate labels
        w_new = w_new.groupby(level=0).sum() if w_new.index.duplicated().any() else w_new
        w_old = w_old.groupby(level=0).sum() if w_old.index.duplicated().any() else w_old
        all_t = list(set(w_new.index) | set(w_old.index))
        return (w_new.reindex(all_t).fillna(0) -
                w_old.reindex(all_t).fillna(0)).abs().sum() / 2

    # ── Helper: drift weights ─────────────────────────────────────────────────
    def _drift(w, d_from, d_to):
        if w.empty or d_from not in Pxs_df.index or d_to not in Pxs_df.index:
            return w
        tks = [t for t in w.index if t in pxs_cols]
        if not tks: return w
        ratio = (Pxs_df.loc[d_to, tks] / Pxs_df.loc[d_from, tks]).fillna(1)
        wd    = w.reindex(tks) * ratio
        return wd / wd.sum() if wd.sum() > 0 else w.reindex(tks)

    # ── Helper: record rebalance cost ─────────────────────────────────────────
    def _record_cost(s_idx, w_new, w_old, dt=None):
        to  = _turnover(w_new, w_old)
        c   = to * trading_cost_bps / 10000
        state[s_idx]['costs'] += c
        if dt is not None:
            yr = dt.year
            state[s_idx]['costs_by_year'][yr] = \
                state[s_idx]['costs_by_year'].get(yr, 0.0) + c
        return c

    # ── Helper: determine portfolio regime from DD only ──────────────────────
    def _get_regime(dd, exog_regime=None):
        """
        Portfolio regime is determined solely by drawdown level.
        Rates regime (exog_regime) affects factor weights only, not portfolio type.
        """
        if dd <= -sh_dd_hybrid:
            return 'mvo'
        elif dd <= -sh_dd_alpha:
            return 'hybrid'
        else:
            return 'alpha'

    # ── Helper: get exogenous rates regime at date ───────────────────────────
    def _exog_regime(dt):
        idx = regime_s.index[regime_s.index <= dt]
        if idx.empty: return 'alpha'
        rv = regime_s.loc[idx[-1]]
        return 'mvo' if rv >= 1.0 else 'hybrid' if rv >= 0.5 else 'alpha'

    # ── Helper: select portfolio by regime ───────────────────────────────────
    def _regime_weights(dt, regime, cached):
        """Return the appropriate cached portfolio for given regime."""
        if regime == 'mvo':
            return cached.get('mvo', cached.get('alpha', pd.Series(dtype=float)))
        elif regime == 'hybrid':
            return cached.get('hybrid', cached.get('alpha', pd.Series(dtype=float)))
        else:
            return cached.get('alpha', pd.Series(dtype=float))

    # ── Helper: build baseline portfolio ─────────────────────────────────────
    def _build_baseline(dt):
        if dt not in quality_wide.index: return pd.Series(dtype=float)
        scores = quality_wide.loc[dt].dropna()
        scores = scores.loc[[t for t in scores.index if t in pxs_cols]]
        if scores.empty: return pd.Series(dtype=float)
        top    = scores.nlargest(top_n)
        return pd.Series(1.0 / len(top), index=top.index)

    # ── Helper: check dynamic trigger ────────────────────────────────────────
    def _check_trigger(s_idx, dt, w_hyp, regime):
        """
        Return True if rebalance should be triggered.
        Matches old run_mvo_backtest trigger logic exactly.
        """
        st = state[s_idx]
        if st['last_rebal'] is None: return True
        if w_hyp.empty: return False

        days_held     = (dt - st['last_rebal']).days
        deployed_reg  = st.get('deployed_regime', 'alpha')

        # DD-forced regime override: immediate rebalance, no min hold
        dd_regime_override = st.get('dd_regime_forced', False)
        if dd_regime_override:
            st['dd_regime_forced'] = False  # consume the flag
            return True

        # Drift current weights to today
        w_drift = _drift(st['weights'], st['last_rebal'], dt)

        # Compute turnover against drifted portfolio
        if w_drift.empty:
            to_val = 1.0
        else:
            all_t = list(set(w_hyp.index) | set(w_drift.index))
            to_val = (w_hyp.reindex(all_t).fillna(0) -
                      w_drift.reindex(all_t).fillna(0)).abs().sum() / 2

        # Vol diff: new portfolio vol minus deployed portfolio vol
        vol_new  = _portfolio_vol(w_hyp,   Pxs_df, dt)
        vol_live = _portfolio_vol(w_drift, Pxs_df, dt)
        vd_val   = vol_new - vol_live

        # TO trigger: TO > threshold AND vol not spiking
        to_thr     = (to_thresh_mvo    if deployed_reg == 'mvo' else
                      to_thresh_hybrid if deployed_reg == 'hybrid' else
                      to_thresh_alpha)
        to_trigger = (to_val > to_thr and
                      vd_val < voldiff_cap and
                      days_held >= min_hold_days)

        # Regime switch trigger
        regime_switch = (deployed_reg is not None and
                         regime != deployed_reg and
                         days_held >= min_hold_days)

        # De-risk trigger (vol spike independent of TO)
        derisk = (vd_val < DYN_VOLDIFF_DERISK and
                  days_held >= min_hold_days)

        return to_trigger or regime_switch or derisk

    # ── Helper: apply DD de-grossing ─────────────────────────────────────────
    def _apply_dd(s_idx, dt):
        """Update DD state — runs daily, matches old run_mvo_backtest DD logic exactly."""
        st       = state[s_idx]
        nav      = st['nav']
        hwm      = st['hwm']

        # Update theo_nav daily — tracks full-exposure portfolio (no de-gross costs)
        # Same as old code: theo_nav *= (1 + full_return) each day when dd_active
        if st['dd_active'] and prev_dt is not None:
            w   = st['weights']
            tks = [t for t in w.index if t in pxs_cols]
            if tks and prev_dt in Pxs_df.index and dt in Pxs_df.index:
                full_ret = (w.reindex(tks).fillna(0) *
                            (Pxs_df.loc[dt, tks] /
                             Pxs_df.loc[prev_dt, tks] - 1).fillna(0)).sum()
                st['theo_nav'] *= (1 + full_ret)

        # HWM only updates when not in a de-gross event
        if not st['dd_active']:
            st['hwm'] = max(hwm, nav)
        dd_current = nav / st['hwm'] - 1

        # Check next DD level (sequential — only next level can trigger)
        next_lvl = st['dd_level'] + 1
        if next_lvl < len(dd_levels):
            dd_thresh, cut_frac = dd_levels[next_lvl]
            if dd_current <= -dd_thresh:
                old_gross        = st['gross']
                st['gross']     *= (1 - cut_frac)
                st['dd_active']  = True
                st['trough']     = nav
                st['theo_nav']   = nav
                st['reentry_count'] = 0
                st['dd_level']   = next_lvl
                st['regime']         = dd_level_regime[next_lvl]
                st['dd_regime_forced'] = True   # override min hold on next check
                st['dd_log'].append({
                    'dt': dt, 'event': f'DE-GROSS lv{next_lvl+1}',
                    'dd': dd_current,
                    'exp_from': old_gross, 'exp_to': st['gross'],
                })

        # Re-entry logic (runs every day when de-grossed)
        if st['dd_active'] and st['gross'] < 1.0:
            st['trough']    = min(st['trough'], nav)
            theo_recovery   = st['theo_nav'] / st['trough'] - 1

            # Annual reset condition
            yr_navs = [v for d, v in nav_records[s_idx].items()
                       if d.year == dt.year]
            ytd_dd  = (nav / yr_navs[0] - 1) if yr_navs else 0
            new_year_ok = (dt.month == 1 and dt.day <= 10 and
                           ytd_dd >= -DD_ANNUAL_RESET_PCT)

            recovery_ok = theo_recovery >= dd_reentry_pct
            if recovery_ok or new_year_ok:
                st['reentry_count'] += 1
            else:
                st['reentry_count'] = 0

            if st['reentry_count'] >= dd_reentry_confirm:
                old_gross        = st['gross']
                st['gross']      = 1.0
                st['dd_active']  = False
                st['dd_level']   = -1
                st['reentry_count'] = 0
                st['hwm']        = nav   # reset HWM
                st['trough']     = nav
                trigger = "recovery" if recovery_ok else "annual reset"
                st['dd_log'].append({
                    'dt': dt, 'event': 'RE-ENTRY 100%',
                    'dd': theo_recovery,
                    'exp_from': old_gross, 'exp_to': 1.0,
                })

    # ── Helper: hedge selection and P&L ──────────────────────────────────────
    def _update_hedge(s_idx, dt, port_w, port_nav):
        """Check hedge triggers using macro signal from hedge_multi."""
        if not hedge_enabled: return
        st = state[s_idx]
        if dt not in Pxs_df.index: return

        prev_dates = [d for d in trading_days if d < dt]
        if not prev_dates: return

        # Macro signal trigger: 1-day lag from hedge_multi signal_df
        # hedge_multi structure: {'results': {ticker: {'signal_df': ...}}, ...}
        _hedge_results = hedge_multi.get('results', hedge_multi) if hedge_multi else {}
        trigger_on = False
        for ta in hedge_trigger_assets:
            if ta in _hedge_results:
                sig_df = _hedge_results[ta].get('signal_df', pd.DataFrame())
                if not sig_df.empty and dt in sig_df.index:
                    prev_sig_dates = sig_df.index[sig_df.index < dt]
                    if len(prev_sig_dates) > 0:
                        prev_sig = sig_df.loc[prev_sig_dates[-1], 'signal']
                        if prev_sig == 1:
                            trigger_on = True
                            break

        # Close hedges when trigger no longer on
        if st['active_hedges'] and not trigger_on:
            total_pnl = 0.0
            details   = {}
            for inst, h in st['active_hedges'].items():
                if inst in Pxs_df.columns:
                    pnl = -(Pxs_df.loc[dt, inst] / h['entry_px'] - 1) * h['weight']
                    st['hedge_acct'] += pnl
                    total_pnl += pnl
                    details[inst] = {
                        'weight':   h['weight'],
                        'entry_px': h['entry_px'],
                        'entry_dt': h.get('entry_dt', dt),
                        'beta':     h.get('beta', 0),
                        'eff':      h.get('eff', 0),
                    }
            days_held = (dt - min(h.get('entry_dt', dt)
                                  for h in st['active_hedges'].values())).days
            st['hedge_log'].append({
                'event':       'CLOSE',
                'date':        dt,
                'instruments': list(st['active_hedges'].keys()),
                'total_pnl':   total_pnl,
                'days_held':   days_held,
                'details':     details,
            })
            if s_idx == S_HEDGE:
                print(f"  [HEDGE {dt.date()}] Signal OFF → P&L={total_pnl:+.2%}  "
                      f"held={days_held}d", flush=True)
            st['active_hedges'] = {}

        # Open new hedges when trigger fires
        if trigger_on and not st['active_hedges'] and hedges_l:
            _wbd_dyn   = state[S_DYN]['weights_by_date']
            _ret_dates = pd.DatetimeIndex(prev_dates[-BETA_WINDOW:])
            port_ret_s = (_portfolio_returns(_wbd_dyn, Pxs_df, _ret_dates)
                          if _wbd_dyn else pd.Series(dtype=float))
            if not port_ret_s.empty:
                inst_ret = {inst: Pxs_df[inst].pct_change().dropna()
                            for inst in hedges_l if inst in Pxs_df.columns}
                selected = _select_hedge_instruments(
                    dt             = dt,
                    hedges_l       = hedges_l,
                    results        = _hedge_results,
                    port_ret_s     = port_ret_s,
                    inst_ret       = inst_ret,
                    beta_window    = BETA_WINDOW,
                    corr_window    = CORR_WINDOW,
                    eff_mav_win    = EFF_MAV_WINDOW,
                    eff_floor      = eff_floor,
                    corr_floor     = corr_floor,
                    trigger_assets = hedge_trigger_assets,
                )
                if not selected:
                    if s_idx == S_HEDGE:
                        print(f"  [HEDGE {dt.date()}] Signal ON — no qualifying instruments", flush=True)
                if selected:
                    n_inst      = len(selected)
                    total_hedge = min(n_inst * hedge_ratio, max_hedge)
                    raw_scores  = {i: d['beta'] * d['effectiveness']
                                   for i, d in selected.items()
                                   if not np.isnan(d.get('beta', np.nan)) and
                                      not np.isnan(d.get('effectiveness', np.nan))}
                    total_score = sum(raw_scores.values())
                    for inst, h in selected.items():
                        if inst in Pxs_df.columns:
                            if total_score > 0 and inst in raw_scores:
                                w_inst = total_hedge * raw_scores[inst] / total_score
                            else:
                                w_inst = total_hedge / n_inst
                            st['active_hedges'][inst] = {
                                'entry_px': Pxs_df.loc[dt, inst],
                                'entry_dt': dt,
                                'weight':   w_inst,
                                'beta':     h.get('beta', 0),
                                'eff':      h.get('effectiveness', 0),
                            }
                    st['hedge_log'].append({
                        'event':       'OPEN',
                        'date':        dt,
                        'instruments': list(st['active_hedges'].keys()),
                    })
                    if s_idx == S_HEDGE:
                        _inst_str = '  '.join(
                            f"{inst}(w={st['active_hedges'][inst]['weight']:.1%},"
                            f"eff={st['active_hedges'][inst]['eff']:.2f})"
                            for inst in st['active_hedges'])
                        print(f"  [HEDGE {dt.date()}] Signal ON  → {_inst_str}", flush=True)

    # ── Helper: compute daily NAV update ─────────────────────────────────────
    def _update_nav(s_idx, d_from, d_to, gross=None):
        """
        Update strategy NAV using price returns from d_from to d_to.
        gross: override gross exposure (for DD strategies)
        """
        st  = state[s_idx]
        w   = st['weights']
        if w.empty or d_from not in Pxs_df.index or d_to not in Pxs_df.index:
            nav_records[s_idx][d_to] = st['nav']
            return 0.0

        tks = [t for t in w.index if t in pxs_cols and t not in
               state[s_idx].get('active_hedges', {})]
        hedge_ret = 0.0

        # Hedge P&L
        for inst, h in st.get('active_hedges', {}).items():
            if inst in Pxs_df.columns:
                r = -(Pxs_df.loc[d_to, inst] / Pxs_df.loc[d_from, inst] - 1)
                hedge_ret += r * h['weight']

        port_ret = 0.0
        if tks:
            port_ret = (w.reindex(tks).fillna(0) *
                        (Pxs_df.loc[d_to, tks] /
                         Pxs_df.loc[d_from, tks] - 1).fillna(0)).sum()

        g         = gross if gross is not None else st.get('gross', 1.0)
        total_ret = (port_ret + hedge_ret) * g
        st['nav'] *= (1 + total_ret)
        if st['nav'] > st['hwm']:
            st['hwm'] = st['nav']
        st['dd'] = st['nav'] / st['hwm'] - 1
        nav_records[s_idx][d_to] = st['nav']
        return total_ret

    # ── 6. Main loop over all trading days ───────────────────────────────────
    print(f"  Running backtest: {len(trading_days)} days, {N_STRAT} strategies...")

    # Track last calc_date for static strategies
    last_calc_dt = None
    calc_date_set = set(calc_dates)   # only original calc_dates drive rebalancing

    for day_idx, dt in enumerate(trading_days):
        is_calc_date = dt in calc_date_set
        prev_dt = trading_days[day_idx - 1] if day_idx > 0 else None

        # -- Update days_held for dynamic strategies --------------------------
        for s_idx in DYNAMIC:
            if state[s_idx]['last_rebal'] is not None:
                state[s_idx]['days_held'] += 1

        # -- Static strategies: rebalance on calc_dates -----------------------
        if is_calc_date:
            # Build X snapshot on-demand if not yet cached
            n_before = len(X_snapshots)
            _ensure_x_snapshot(dt)
            if len(X_snapshots) > n_before:
                print(f"  [{day_idx+1}/{len(trading_days)}] {dt.date()}  "
                      f"X snapshot built  ({len(X_snapshots)} total)", flush=True)
            comp = composite_by_date.get(dt)
            if comp is not None:
                comp = comp.dropna()
                comp = comp.loc[[t for t in comp.index if t in pxs_cols]]

                # Quality pre-filter → quality-gated candidate pool
                n_pf   = max(n_cands, int(np.ceil(len(comp) * prefilt_pct)))
                cands  = comp.nlargest(min(n_pf, len(comp)))

                # Momentum filter for alpha/hybrid
                # Rank by sum of Idio_Mom + Mom_12M1 (both in percentile rank space)
                # then apply composite score within that momentum-filtered pool
                n_mom = mom_filter * top_n
                if n_mom < len(cands):
                    idio_row = pd.Series(dtype=float)
                    m12_row  = pd.Series(dtype=float)
                    if 'Idio_Mom' in score_dfs and score_dfs['Idio_Mom'] is not None:
                        avail = [d for d in score_dfs['Idio_Mom'].index if d <= dt]
                        if avail:
                            idio_row = score_dfs['Idio_Mom'].loc[avail[-1]]\
                                           .reindex(cands.index).fillna(0)
                    if 'Mom_12M1' in score_dfs and score_dfs['Mom_12M1'] is not None:
                        avail = [d for d in score_dfs['Mom_12M1'].index if d <= dt]
                        if avail:
                            m12_row = score_dfs['Mom_12M1'].loc[avail[-1]]\
                                          .reindex(cands.index).fillna(0)
                    mom_combined = idio_row.add(m12_row, fill_value=0)
                    top_mom      = mom_combined.nlargest(min(n_mom, len(cands))).index
                    cands_alpha  = cands.reindex(top_mom).dropna()
                else:
                    cands_alpha = cands
                # MVO uses full quality-gated pool (universe_mult×N)
                cands_mvo = cands.nlargest(min(n_cands, len(cands)))

                # Check cache
                if dt not in cached_dates or mode == 'rebuild':
                    # Alpha uses momentum-filtered pool
                    w_alpha = _build_alpha(dt, cands_alpha)
                    # MVO uses full quality-gated pool
                    w_mvo   = _build_mvo(dt, cands_mvo, comp)
                    if w_mvo.empty: w_mvo = w_alpha.copy()
                    # Hybrid = blend of alpha and MVO
                    all_t   = list(set(w_alpha.index) | set(w_mvo.index))
                    w_hyb   = (w_alpha.reindex(all_t).fillna(0) +
                               w_mvo.reindex(all_t).fillna(0)) / 2
                    w_hyb   = w_hyb[w_hyb > 0]
                    if w_hyb.sum() > 0: w_hyb /= w_hyb.sum()
                    # Penalised (Excl) — alpha uses momentum-filtered pool
                    comp_pen    = _penalise(comp, dt)
                    cands_pen_a = comp_pen.reindex(cands_alpha.index).dropna().nlargest(
                                      min(n_mom, len(cands_alpha)))
                    cands_pen_m = comp_pen.reindex(cands_mvo.index).dropna().nlargest(
                                      min(n_cands, len(cands_mvo)))
                    w_alpha_p = _build_alpha(dt, cands_pen_a)
                    w_mvo_p   = _build_mvo(dt, cands_pen_m, comp_pen)
                    if w_mvo_p.empty: w_mvo_p = w_alpha_p.copy()
                    all_tp    = list(set(w_alpha_p.index) | set(w_mvo_p.index))
                    w_hyb_p   = (w_alpha_p.reindex(all_tp).fillna(0) +
                                 w_mvo_p.reindex(all_tp).fillna(0)) / 2
                    w_hyb_p   = w_hyb_p[w_hyb_p > 0]
                    if w_hyb_p.sum() > 0: w_hyb_p /= w_hyb_p.sum()
                    # Save PRE-tier2 portfolios to cache
                    _save_portfolios(dt, model_version, ph, {
                        'alpha':        w_alpha,
                        'mvo':          w_mvo,
                        'hybrid':       w_hyb,
                        'alpha_excl':   w_alpha_p,
                        'mvo_excl':     w_mvo_p,
                        'hybrid_excl':  w_hyb_p,
                        'candidates':   cands,
                    })
                    cached_dates.add(dt)

                else:
                    # Load pre-tier2 portfolios from cache
                    with ENGINE.connect() as conn:
                        rows = conn.execute(text(f"""
                            SELECT strategy, weights_json FROM {DAILY_PORT_TBL}
                            WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                        """), {'dt': dt.strftime('%Y-%m-%d'),
                               'mv': model_version, 'ph': ph}).fetchall()
                    pc = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
                    w_alpha   = pc.get('alpha',   pd.Series(dtype=float))
                    w_mvo     = pc.get('mvo',     pd.Series(dtype=float))
                    w_hyb     = pc.get('hybrid',  pd.Series(dtype=float))
                    w_alpha_p = pc.get('alpha_excl',  w_alpha.copy())
                    w_mvo_p   = pc.get('mvo_excl',    w_mvo.copy())
                    w_hyb_p   = pc.get('hybrid_excl', w_hyb.copy())
                    cands     = pc.get('candidates',  pd.Series(dtype=float))
                    comp_pen  = _penalise(comp, dt)

                # Baseline (S0) ──────────────────────────────────────────
                w_bl = _build_baseline(dt)
                _record_cost(S_BASE, w_bl, state[S_BASE]['weights'], dt)
                state[S_BASE]['rebal_log'].append({'dt': dt, 'w': w_bl.copy()})
                state[S_BASE]['weights'] = w_bl
                state[S_BASE]['weights_by_date'][dt] = w_bl.copy()
                state[S_BASE]['last_rebal'] = dt
                _save_exposure_row(dt, 'Baseline',
                    _get_factor_exposures(w_bl, dt, _factor_scores))

                # Helper: run ADVP on tier1, then append overlay
                def _advp_then_tier2(w_t1, s_idx, non_core_cands, comp_s):
                    t_n = len(w_t1) if len(w_t1) > top_n else top_n
                    w_out, aff = _apply_advp(w_t1, cands, dt, s_idx,
                                             target_n=t_n)
                    if tier2_active:
                        nc = [t for t in non_core_cands
                              if t not in w_out.index and t in pxs_cols]
                        w_out, ov_set = _apply_tier2(w_out, dt, nc, comp_s)
                    else:
                        ov_set = set()
                    return w_out, aff, ov_set

                non_core     = [t for t in cands.index if t in pxs_cols]
                non_core_pen = [t for t in cands.index if t in pxs_cols]

                # -- Pure Alpha (S1) ------------------------------------------
                w_a1, _, _ov_a1 = _advp_then_tier2(w_alpha, S_ALPHA,
                                                     non_core, comp)
                if day_idx == len(trading_days)-1 or day_idx > len(trading_days)-10:
                    print(f"  [AUM DIAG] {dt.date()}  "
                          f"Alpha=${aum*state[S_ALPHA]['nav']/1e6:.1f}M  "
                          f"MVO=${aum*state[S_MVO]['nav']/1e6:.1f}M  "
                          f"Hybrid=${aum*state[S_HYB]['nav']/1e6:.1f}M  "
                          f"Smart=${aum*state[S_SMART]['nav']/1e6:.1f}M  "
                          f"Dyn=${aum*state[S_DYN]['nav']/1e6:.1f}M  "
                          f"DynHdg=${aum*state[S_HEDGE]['nav']/1e6:.1f}M  "
                          f"DD=${aum*state[S_DD]['nav']/1e6:.1f}M", flush=True)
                _record_cost(S_ALPHA, w_a1, state[S_ALPHA]['weights'], dt)
                state[S_ALPHA]['rebal_log'].append({'dt': dt, 'w': w_a1.copy()})
                state[S_ALPHA]['weights'] = w_a1
                state[S_ALPHA]['weights_by_date'][dt] = w_a1.copy()
                state[S_ALPHA]['last_rebal'] = dt
                _save_exposure_row(dt, 'Alpha',
                    _get_factor_exposures(w_a1, dt, _factor_scores))

                # -- MVO (S2) -------------------------------------------------
                w_m2, _, _ov_m2 = _advp_then_tier2(w_mvo, S_MVO, non_core, comp)
                _record_cost(S_MVO, w_m2, state[S_MVO]['weights'], dt)
                state[S_MVO]['rebal_log'].append({'dt': dt, 'w': w_m2.copy()})
                state[S_MVO]['weights'] = w_m2
                state[S_MVO]['weights_by_date'][dt] = w_m2.copy()
                state[S_MVO]['last_rebal'] = dt
                _save_exposure_row(dt, 'MVO',
                    _get_factor_exposures(w_m2, dt, _factor_scores))

                # -- Hybrid (S3) ----------------------------------------------
                w_h3, _, _ov_h3 = _advp_then_tier2(w_hyb, S_HYB, non_core,
                                                     comp)
                _record_cost(S_HYB, w_h3, state[S_HYB]['weights'], dt)
                state[S_HYB]['rebal_log'].append({'dt': dt, 'w': w_h3.copy()})
                state[S_HYB]['weights'] = w_h3
                state[S_HYB]['weights_by_date'][dt] = w_h3.copy()
                state[S_HYB]['last_rebal'] = dt
                _save_exposure_row(dt, 'Hybrid',
                    _get_factor_exposures(w_h3, dt, _factor_scores))

                # -- Smart Hybrid (S4) ----------------------------------------
                reg4 = state[S_SMART]['regime']
                w_s4_base = _regime_weights(dt, reg4, {
                    'alpha': w_alpha, 'mvo': w_mvo, 'hybrid': w_hyb})
                w_s4, _, _ov_s4 = _advp_then_tier2(w_s4_base, S_SMART,
                                                     non_core, comp)
                _record_cost(S_SMART, w_s4, state[S_SMART]['weights'], dt)
                state[S_SMART]['rebal_log'].append({'dt': dt, 'w': w_s4.copy()})
                state[S_SMART]['weights'] = w_s4
                state[S_SMART]['weights_by_date'][dt] = w_s4.copy()
                state[S_SMART]['last_rebal'] = dt
                _save_exposure_row(dt, 'Smart',
                    _get_factor_exposures(w_s4, dt, _factor_scores))

            last_calc_dt = dt

        # -- Dynamic strategies (S5-S8): check triggers daily ----------------
        if last_calc_dt is not None and last_calc_dt in composite_by_date:
            comp_dyn  = composite_by_date[last_calc_dt].dropna()
            comp_dyn  = comp_dyn.loc[[t for t in comp_dyn.index if t in pxs_cols]]
            n_pf_dyn  = max(n_cands, int(np.ceil(len(comp_dyn) * prefilt_pct)))
            cands_dyn = comp_dyn.nlargest(min(n_pf_dyn, len(comp_dyn)))

            # Use in-memory portfolios if this is a calc_date, else load from DB
            if is_calc_date and 'w_alpha' in dir():
                _cached_dyn = {
                    'alpha':        w_alpha,
                    'mvo':          w_mvo,
                    'hybrid':       w_hyb,
                    'alpha_excl':   w_alpha_p,
                    'mvo_excl':     w_mvo_p,
                    'hybrid_excl':  w_hyb_p,
                    'candidates':   cands,
                }
            else:
                _cached_dyn = {}
                if last_calc_dt in cached_dates:
                    try:
                        with ENGINE.connect() as conn:
                            rows = conn.execute(text(f"""
                                SELECT strategy, weights_json FROM {DAILY_PORT_TBL}
                                WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                            """), {'dt': last_calc_dt.strftime('%Y-%m-%d'),
                                   'mv': model_version, 'ph': ph}).fetchall()
                        _cached_dyn = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
                    except Exception:
                        pass

            for s_idx in [S_DYN, S_HEDGE, S_DD, S_EXCL]:
                st   = state[s_idx]
                reg  = st['regime']

                # Select hypothetical new portfolio for trigger check
                if s_idx == S_EXCL:
                    comp_s  = _penalise(comp_dyn, last_calc_dt)
                    cands_s = comp_s.nlargest(min(n_pf_dyn, len(comp_s)))
                    w_hyp = _regime_weights(dt, reg, {
                        'alpha':  _cached_dyn.get('alpha_excl',
                                  _cached_dyn.get('alpha', pd.Series(dtype=float))),
                        'mvo':    _cached_dyn.get('mvo_excl',
                                  _cached_dyn.get('mvo',   pd.Series(dtype=float))),
                        'hybrid': _cached_dyn.get('hybrid_excl',
                                  _cached_dyn.get('hybrid',pd.Series(dtype=float))),
                    })
                else:
                    cands_s = cands_dyn
                    w_hyp = _regime_weights(dt, reg, {
                        'alpha':  _cached_dyn.get('alpha',  pd.Series(dtype=float)),
                        'mvo':    _cached_dyn.get('mvo',    pd.Series(dtype=float)),
                        'hybrid': _cached_dyn.get('hybrid', pd.Series(dtype=float)),
                    })

                if w_hyp.empty: continue

                # ADVP on tier1, then append overlay
                w_hyp, _advp_affected, _ov_dyn = _advp_then_tier2(
                    w_hyp, s_idx,
                    [t for t in cands_dyn.index if t in pxs_cols],
                    comp_s if s_idx == S_EXCL else comp_dyn
                )

                # Check rebalance trigger
                if _check_trigger(s_idx, dt, w_hyp, reg):
                    to   = _turnover(w_hyp, _drift(st['weights'],
                                                    st['last_rebal'] or dt, dt))
                    cost = _record_cost(s_idx, w_hyp, st['weights'], dt)
                    st['weights']             = w_hyp
                    st['weights_by_date'][dt] = w_hyp.copy()
                    st['last_rebal']          = dt
                    st['days_held']           = 0
                    st['deployed_regime']     = reg
                    st['rebal_log'].append({
                        'dt': dt, 'regime': reg, 'w': w_hyp.copy(),
                        'to': to, 'nav': st['nav'],
                    })
                    _save_exposure_row(dt, STRATEGY_LABELS[s_idx],
                        _get_factor_exposures(w_hyp, dt, _factor_scores))

                    # -- Per-rebalance print for DD Policy (S_DD) ---------------
                    if s_idx == S_DD:
                        _cur_aum  = aum * st['nav'] * st.get('gross', 1.0)
                        _rebal_n  = len(st['rebal_log'])
                        _eff_n    = 1.0 / (w_hyp**2).sum() if not w_hyp.empty else 0
                        _pv       = _portfolio_vol(w_hyp, Pxs_df, dt)
                        _dd_str   = f"dd={st['dd']*100:+.1f}%"
                        _gr_str   = f"gross={st['gross']*100:.0f}%"
                        print(f"\n  -- {dt.date()}  [{_rebal_n}]  "
                              f"regime={reg}  {_dd_str}  {_gr_str}  "
                              f"n={len(w_hyp)}  eff_N={_eff_n:.1f}  "
                              f"TO={to*100:.0f}%  "
                              f"AUM=${_cur_aum/1e6:.1f}M  "
                              f"port_vol={_pv*100:.1f}%", flush=True)
                        # MR top scores
                        _mr_dt = _load_momentum_exclusions(last_calc_dt)
                        if _mr_dt and mr_k > 0:
                            _mr_top = sorted(_mr_dt.items(), key=lambda x: -x[1])[:10]
                            _mr_str = '  '.join(f"{t}({s:.2f})"
                                                for t, s in _mr_top if s > 0)
                            if _mr_str:
                                print(f"  MR top-10 (k={mr_k}): {_mr_str}", flush=True)
                        # Portfolio weights + factor exposures
                        exp = _get_factor_exposures(w_hyp, dt, _factor_scores)
                        _save_exposure_row(dt, STRATEGY_LABELS[s_idx], exp)

                        hdr_exp = '  '.join(f"{f:>9}" for f in EXPOSURE_FACTORS)
                        print(f"  {'Ticker':<8} {'Weight%':>8}  {'Sector':<28}  {hdr_exp}",
                              flush=True)
                        print("  " + "-"*80, flush=True)
                        if _advp_affected:
                            print(f"  *** ADVP cap active "
                                  f"(AUM=${_cur_aum:,.0f}, cap={advp_cap:.1%} ADV): "
                                  f"{', '.join(sorted(_advp_affected))}", flush=True)
                        # Identify overlay stocks from _advp_then_tier2 result
                        overlay_set = _ov_dyn

                        # Tier 1 first (sorted by weight desc), overlay at bottom
                        t1_sorted = [t for t in w_hyp.sort_values(ascending=False).index
                                     if t not in overlay_set]
                        ov_sorted = [t for t in w_hyp.sort_values(ascending=False).index
                                     if t in overlay_set]

                        for _tkr in t1_sorted + ov_sorted:
                            _flag = '***' if _tkr in _advp_affected else '   '
                            _ov   = ' OV' if _tkr in overlay_set else '   '
                            _sec  = sectors_s.get(_tkr, '')
                            tkr_exp = []
                            for fname, fdf in _factor_scores.items():
                                if not fdf.empty:
                                    avail = [d for d in fdf.index if d <= dt]
                                    val = fdf.loc[avail[-1], _tkr] \
                                          if avail and _tkr in fdf.columns else np.nan
                                else:
                                    val = np.nan
                                tkr_exp.append(f"{val:>+9.2f}" if pd.notna(val)
                                               else f"{'n/a':>9}")
                            exp_str = '  '.join(tkr_exp)
                            # Separator before overlay section
                            if ov_sorted and _tkr == ov_sorted[0]:
                                print(f"  {'--- OVERLAY ---':<80}", flush=True)
                            print(f"  {_tkr:<8} {w_hyp[_tkr]*100:>7.2f}%  "
                                  f"{_sec:<28}  {exp_str}{_ov}{_flag}", flush=True)
                        # Portfolio totals
                        tot_exp = '  '.join(
                            f"{exp.get(f, np.nan):>+9.2f}"
                            if pd.notna(exp.get(f)) else f"{'n/a':>9}"
                            for f in EXPOSURE_FACTORS
                        )
                        print("  " + "-"*80, flush=True)
                        print(f"  {'PORT AVG':<8} {'':>8}  {'weighted':28}  {tot_exp}",
                              flush=True)

        # -- Update all NAVs first (using current gross before any DD change) --
        if prev_dt is not None:
            for s_idx in range(N_STRAT):
                gross = state[s_idx].get('gross', 1.0)
                gross_records[s_idx][dt] = gross
                _update_nav(s_idx, prev_dt, dt, gross=gross)
        else:
            for s_idx in range(N_STRAT):
                nav_records[s_idx][dt]   = state[s_idx]['nav']
                gross_records[s_idx][dt] = state[s_idx].get('gross', 1.0)

        # -- Update regime for all strategies (uses updated dd) ---------------
        for s_idx in range(N_STRAT):
            state[s_idx]['regime'] = _get_regime(state[s_idx]['dd'])

        # -- Update hedge for S6/S7/S8 ----------------------------------------
        for s_idx in [S_HEDGE, S_DD, S_EXCL]:
            if prev_dt is not None and not state[s_idx]['weights'].empty:
                _update_hedge(s_idx, dt, state[s_idx]['weights'],
                              state[s_idx]['nav'])

        # -- Apply DD de-grossing (after NAV update, takes effect next day) ---
        for s_idx in [S_DD, S_EXCL]:
            _apply_dd(s_idx, dt)

        # -- Progress ----------------------------------------------------------
        if (day_idx + 1) % 50 == 0 or day_idx == len(trading_days) - 1:
            print(f"  [{day_idx+1}/{len(trading_days)}] {dt.date()}  "
                  f"nav_dyn={state[S_DYN]['nav']:.3f}  "
                  f"nav_dd={state[S_DD]['nav']:.3f}  "
                  f"AUM_dyn=${aum*state[S_DYN]['nav']/1e6:.0f}M",
                  end='\r', flush=True)

    print()  # newline after progress

    # ── 7. Build NAV series ───────────────────────────────────────────────────
    def _to_nav_series(s_idx):
        d = nav_records[s_idx]
        if not d: return pd.Series(dtype=float)
        return pd.Series(d).sort_index()

    nav_series = [_to_nav_series(s) for s in range(N_STRAT)]
    labels     = ['Baseline', 'Pure Alpha', 'MVO', 'Hybrid', 'Smart Hybrid',
                  'Dynamic', 'Dyn+Hedge', 'DD Policy', 'Excl']

    # ── 8. Performance summary ────────────────────────────────────────────────
    def _perf(nav_s):
        if nav_s is None or len(nav_s) < 2:
            return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan,
                        mdd=np.nan, cagr_dd=np.nan)
        rets  = nav_s.pct_change().dropna()
        n_yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr  = nav_s.iloc[-1] ** (1 / n_yrs) - 1 if n_yrs > 0 else np.nan
        vol   = rets.std() * np.sqrt(252)
        sharpe= cagr / vol if vol > 0 else np.nan
        roll_max = nav_s.cummax()
        mdd   = ((nav_s - roll_max) / roll_max).min()
        cagr_dd = cagr / abs(mdd) if mdd != 0 else np.nan
        return dict(cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd, cagr_dd=cagr_dd)

    print(f"\n  {'='*72}")
    print(f"  COMPARISON")
    print(f"  {'='*72}")
    print(f"  All returns net of {trading_cost_bps}bps trading costs")
    print(f"  {'Strategy':<42} {'CAGR':>7} {'Vol':>7} {'Sharpe':>8} "
          f"{'MDD':>8} {'CAGR/DD':>9}")
    print(f"  {'-'*70}")
    for s_idx, (nav_s, lbl) in enumerate(zip(nav_series, labels)):
        p = _perf(nav_s)
        if np.isnan(p['cagr']): continue
        print(f"  {lbl:<42} {p['cagr']*100:>6.1f}%  {p['vol']*100:>6.1f}%  "
              f"{p['sharpe']:>7.2f}  {p['mdd']*100:>7.1f}%  "
              f"{p['cagr_dd']:>7.2f}x")

    # Yearly returns — all columns use consistent 10-char width
    _aum_nav = nav_series[S_HEDGE]
    _COL  = 10   # fixed column width for all strategy columns
    _COLS = [
        (S_BASE,  'Baseline'),
        (S_ALPHA, 'Alpha'),
        (S_MVO,   'MVO'),
        (S_HYB,   'Hybrid'),
        (S_SMART, 'Smart'),
        (S_DYN,   'Dynamic'),
        (S_HEDGE, 'Dyn+Hdg'),
        (S_DD,    'DD Pol'),
        (S_EXCL,  'Excl'),
    ]
    _sep  = '  ' + '-' * (6 + (len(_COLS) + 1) * (_COL + 2))

    print(f"\n  Yearly returns  (starting AUM: ${aum/1e6:.1f}M)")
    # Header
    hdr  = f"  {'Year':<6}"
    for _, lbl in _COLS:
        hdr += f"  {lbl:>{_COL}}"
    hdr += f"  {'AUM($M)':>{_COL}}"
    print(hdr)
    # DD row
    dd_row = f"  {'DD now':<6}"
    for s_idx, _ in _COLS:
        if s_idx == S_BASE:
            dd_row += f"  {'':>{_COL}}"
        else:
            dd = state[s_idx]['dd'] * 100
            dd_row += f"  {dd:>+{_COL-1}.1f}%"
    dd_row += f"  {'':>{_COL}}"
    print(dd_row)
    print(_sep)

    def _fmt_ret(r):
        return f"{r:>+{_COL-1}.2f}%" if not np.isnan(r) else f"{'n/a':>{_COL}}"

    all_years = sorted(set(nav_series[0].index.year))
    for yr in all_years:
        def yr_ret(nav_s):
            yn = nav_s[nav_s.index.year == yr] if nav_s is not None else pd.Series()
            if len(yn) < 2: return np.nan
            return (yn.iloc[-1] / yn.iloc[0] - 1) * 100
        nav_to_yr = _aum_nav[_aum_nav.index.year <= yr]
        closing   = aum * float(nav_to_yr.iloc[-1]) if not nav_to_yr.empty else aum
        row = f"  {yr:<6}"
        for s_idx, _ in _COLS:
            row += f"  {_fmt_ret(yr_ret(nav_series[s_idx]))}"
        row += f"  ${closing/1e6:>{_COL-2}.1f}M"
        print(row)

    # Trading costs per strategy per year
    print(f"\n  Trading costs per year (as % of NAV at time of trade)")
    print(hdr.replace(f"{'AUM($M)':>{_COL}}", f"{'Total':>{_COL}}"))
    print(_sep)
    for yr in all_years:
        row = f"  {yr:<6}"
        for s_idx, _ in _COLS:
            c = state[s_idx]['costs_by_year'].get(yr, 0.0)
            row += f"  {c*100:>+{_COL-1}.2f}%" if c > 0 else f"  {'':>{_COL}}"
        total_c = sum(state[s]['costs_by_year'].get(yr, 0.0) for s in range(N_STRAT))
        row += f"  {total_c*100:>+{_COL-1}.2f}%" if total_c > 0 else f"  {'':>{_COL}}"
        print(row)
    # Totals row
    print(_sep)
    tot_row = f"  {'Total':<6}"
    for s_idx, _ in _COLS:
        c = state[s_idx]['costs']
        tot_row += f"  {c*100:>+{_COL-1}.2f}%" if c > 0 else f"  {'':>{_COL}}"
    tot_row += f"  {'':>{_COL}}"
    print(tot_row)

    # DD policy summary
    print(f"\n  DD Policy summary: {len(state[S_DD]['dd_log'])} events")
    if state[S_DD]['dd_log']:
        print(f"  {'Date':<12} {'Event':<24} {'DD':>8}  {'Exp From':>10}  {'Exp To':>8}")
        print("  " + "-" * 60)
        for ev in state[S_DD]['dd_log']:
            print(f"  {str(ev['dt'].date()):<12} {ev['event']:<24} "
                  f"{ev['dd']*100:>+7.1f}%  {ev['exp_from']*100:>9.1f}%  "
                  f"{ev['exp_to']*100:>7.1f}%")

    # ── 9. Live portfolio display ─────────────────────────────────────────────
    today_ts = trading_days[-1]

    # Unified current portfolio table
    _strat_display = [
        ('Baseline', S_BASE),
        ('Alpha',    S_ALPHA),
        ('MVO',      S_MVO),
        ('Hybrid',   S_HYB),
        ('Smart',    S_SMART),
        ('Dynamic',  S_DYN),
        ('Dyn+Hdg',  S_HEDGE),
        ('DD Pol',   S_DD),
        ('Excl',     S_EXCL),
    ]

    # Collect drifted weights per strategy
    def _curr_weights(s_idx):
        st  = state[s_idx]
        ldt = st['last_rebal']
        w   = st['weights']
        if w.empty or ldt is None: return pd.Series(dtype=float), ldt
        if ldt < today_ts:
            wd = _drift(w, ldt, today_ts)
        else:
            wd = w.copy()
        # Scale by gross exposure
        wd = wd * st.get('gross', 1.0)
        # Add hedge positions as negative weights
        for inst, h in st.get('active_hedges', {}).items():
            wd[inst] = wd.get(inst, 0.0) - h['weight']
        return wd, ldt

    _all_tks = set()
    _curr_w  = {}
    _curr_ldt= {}
    for lbl, s_idx in _strat_display:
        wd, ldt = _curr_weights(s_idx)
        _curr_w[lbl]   = wd
        _curr_ldt[lbl] = ldt
        _all_tks |= set(wd.index)

    # Last-day returns
    prev_td = trading_days[-2] if len(trading_days) >= 2 else today_ts
    _day_ret = {}
    for tkr in _all_tks:
        if tkr in Pxs_df.columns:
            p1 = Pxs_df.loc[prev_td, tkr]; p2 = Pxs_df.loc[today_ts, tkr]
            _day_ret[tkr] = (p2/p1 - 1)*100 if p1 > 0 else 0.0
        else:
            _day_ret[tkr] = 0.0

    # Sort by average weight
    _avg_w = {t: np.mean([_curr_w[l].get(t, 0.0) for l, _ in _strat_display])
              for t in _all_tks}
    _sorted_tks = sorted(_all_tks, key=lambda t: -_avg_w[t])
    _col_lbls = [l for l, _ in _strat_display]

    print(f"\n  {'='*72}")
    print("  CURRENT LIVE PORTFOLIOS — DRIFTED WEIGHTS (% of AUM)")
    print(f"  {'='*72}")
    _hdr = f"  {'Ticker':<8}  {'Day%':>6}" + "".join(f"  {l:>8}" for l in _col_lbls)
    print(_hdr)
    print(f"  {'Last reb:':8}  {'':6}" +
          "".join(f"  {(_curr_ldt[l].strftime('%m/%d') if _curr_ldt[l] else 'n/a'):>8}"
                  for l in _col_lbls))
    print("  " + "-"*8 + "  " + "-"*6 + ("  " + "-"*8)*len(_col_lbls))
    for tkr in _sorted_tks:
        dr  = _day_ret.get(tkr, 0.0)
        row = f"  {tkr:<8}  {dr:>+5.1f}%"
        any_nz = False
        for lbl in _col_lbls:
            wt = _curr_w[lbl].get(tkr, 0.0) * 100
            if abs(wt) >= 0.01:
                any_nz = True
                row += f"  {wt:>7.2f}%"
            else:
                row += f"  {'':>8}"
        if any_nz:
            print(row)

    # 10-day P&L table
    print(f"\n  {'='*72}")
    print("  DAILY P&L — LAST 10 TRADING DAYS (all strategies)")
    print(f"  {'='*72}")
    _last10 = list(trading_days[-10:])
    print(f"\n  {'Date':<12}" + "".join(f"  {l:>9}" for l in _col_lbls))
    print("  " + "-"*12 + ("  " + "-"*9)*len(_col_lbls))
    _cum = {l: 1.0 for l in _col_lbls}
    for dt_p in _last10:
        row = f"  {dt_p.strftime('%Y-%m-%d'):<12}"
        for lbl, s_idx in _strat_display:
            nav_s = nav_series[s_idx]
            prev_dates = [d for d in nav_s.index if d < dt_p]
            nav_at = nav_s[nav_s.index <= dt_p]
            if nav_at.empty or not prev_dates:
                row += f"  {'n/a':>9}"; continue
            r = (float(nav_at.iloc[-1]) / float(nav_s.loc[prev_dates[-1]]) - 1)*100
            _cum[lbl] *= (1 + r/100)
            row += f"  {r:>+8.2f}%"
        print(row)
    print("  " + "-"*12 + ("  " + "-"*9)*len(_col_lbls))
    cum_row = f"  {'Cumul':<12}"
    for lbl in _col_lbls:
        cum_row += f"  {(_cum[lbl]-1)*100:>+8.2f}%"
    print(cum_row)

    # Trade summary — all 9 strategies
    # Delta = fresh portfolio today - drifted weights from last rebalance
    # Fresh portfolios are computed using last calc_date scores but today's
    # ADVP filter (today's prices/volumes and each strategy's current AUM)
    print(f"\n  {'='*72}")
    print(f"  TRADE SUMMARY  —  {today_ts.date()}")
    print(f"  {'='*72}")

    _all_strats_summary = [
        ('Baseline', S_BASE),
        ('Alpha',    S_ALPHA),
        ('MVO',      S_MVO),
        ('Hybrid',   S_HYB),
        ('Smart',    S_SMART),
        ('Dynamic',  S_DYN),
        ('Dyn+Hdg',  S_HEDGE),
        ('DD Pol',   S_DD),
        ('Excl',     S_EXCL),
    ]

    # Find last calc_date with cached portfolios
    _last_cd = sorted([d for d in cached_dates if d <= today_ts])
    _last_cd = _last_cd[-1] if _last_cd else None
    _fresh_w = {}

    if _last_cd:
        # Load cached portfolios for last calc_date
        try:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT strategy, weights_json FROM {DAILY_PORT_TBL}
                    WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                """), {'dt': _last_cd.strftime('%Y-%m-%d'),
                       'mv': model_version, 'ph': ph}).fetchall()
            _cache_td = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
        except Exception:
            _cache_td = {}

        _cands_td = _cache_td.get('candidates', pd.Series(dtype=float))
        def _fresh(s_idx, w_base):
            """Apply today's ADVP filter using this strategy's current AUM."""
            if w_base.empty: return w_base
            w, _ = _apply_advp(w_base, _cands_td, today_ts, s_idx,
                               target_n=len(w_base))
            return w

        for lbl, s_idx in _all_strats_summary:
            st  = state[s_idx]
            reg = _get_regime(st['dd'])
            if s_idx == S_BASE:
                _qual_dates   = [d for d in quality_wide.index if d <= today_ts]
                _qual_dt      = _qual_dates[-1] if _qual_dates else None
                _fresh_w[lbl] = _build_baseline(_qual_dt) if _qual_dt else pd.Series(dtype=float)
            elif s_idx == S_ALPHA:
                _fresh_w[lbl] = _fresh(s_idx, _cache_td.get('alpha',
                                        pd.Series(dtype=float)))
            elif s_idx == S_MVO:
                _fresh_w[lbl] = _fresh(s_idx, _cache_td.get('mvo',
                                        pd.Series(dtype=float)))
            elif s_idx == S_HYB:
                _fresh_w[lbl] = _fresh(s_idx, _cache_td.get('hybrid',
                                        pd.Series(dtype=float)))
            elif s_idx == S_SMART:
                _fresh_w[lbl] = _fresh(s_idx, _regime_weights(today_ts, reg, {
                    'alpha':  _cache_td.get('alpha',  pd.Series(dtype=float)),
                    'mvo':    _cache_td.get('mvo',    pd.Series(dtype=float)),
                    'hybrid': _cache_td.get('hybrid', pd.Series(dtype=float)),
                }))
            elif s_idx in (S_DYN, S_HEDGE, S_DD):
                _fresh_w[lbl] = _fresh(s_idx, _regime_weights(today_ts, reg, {
                    'alpha':  _cache_td.get('alpha',  pd.Series(dtype=float)),
                    'mvo':    _cache_td.get('mvo',    pd.Series(dtype=float)),
                    'hybrid': _cache_td.get('hybrid', pd.Series(dtype=float)),
                }))
            elif s_idx == S_EXCL:
                _fresh_w[lbl] = _fresh(s_idx, _regime_weights(today_ts, reg, {
                    'alpha':  _cache_td.get('alpha_excl',
                              _cache_td.get('alpha',  pd.Series(dtype=float))),
                    'mvo':    _cache_td.get('mvo_excl',
                              _cache_td.get('mvo',    pd.Series(dtype=float))),
                    'hybrid': _cache_td.get('hybrid_excl',
                              _cache_td.get('hybrid', pd.Series(dtype=float))),
                }))

    # Drifted current weights (B side of delta):
    # - if today IS a rebalancing date for this strategy: use log[-2] drifted to today
    # - if today is NOT a rebalancing date: use log[-1] drifted to today
    _drifted_w = {}
    for lbl, s_idx in _all_strats_summary:
        st  = state[s_idx]
        log = st['rebal_log']
        if not log:
            _drifted_w[lbl] = pd.Series(dtype=float)
            continue
        today_was_rebal = log[-1]['dt'] == today_ts
        if today_was_rebal:
            ref = log[-2] if len(log) >= 2 else None
        else:
            ref = log[-1]
        if ref is None:
            _drifted_w[lbl] = pd.Series(dtype=float)
        else:
            _drifted_w[lbl] = _drift(ref['w'], ref['dt'], today_ts)

    # Compute deltas
    all_trade_tks = set()
    for lbl, _ in _all_strats_summary:
        if lbl in _fresh_w:   all_trade_tks |= set(_fresh_w[lbl].index)
        if lbl in _drifted_w: all_trade_tks |= set(_drifted_w[lbl].index)

    deltas = {}
    for lbl, _ in _all_strats_summary:
        deltas[lbl] = {}
        if lbl not in _fresh_w: continue
        for tkr in all_trade_tks:
            n = _fresh_w[lbl].get(tkr, 0.0) * 100
            o = _drifted_w.get(lbl, pd.Series()).get(tkr, 0.0) * 100
            d = n - o
            if abs(d) >= 0.01:
                deltas[lbl][tkr] = d

    all_delta_tks = sorted(
        {t for d in deltas.values() for t in d},
        key=lambda t: -max(abs(deltas[l].get(t, 0)) for l in deltas))

    if all_delta_tks:
        col_w = 9
        print(f"\n  {'Ticker':<8}" +
              "".join(f"  {l:>{col_w}}" for l, _ in _all_strats_summary))
        print("  " + "-"*8 + ("  " + "-"*col_w)*len(_all_strats_summary))
        for tkr in all_delta_tks:
            row = f"  {tkr:<8}"
            for lbl, _ in _all_strats_summary:
                d = deltas[lbl].get(tkr, 0.0)
                row += f"  {d:>+8.2f}%" if abs(d) >= 0.01 else f"  {'':>9}"
            print(row)

    # ── 10. Hedge summary ────────────────────────────────────────────────────
    if hedge_enabled:
        hedge_log = state[S_HEDGE]['hedge_log']
        close_ev  = [e for e in hedge_log if e['event'] == 'CLOSE']
        print(f"\n  {'='*72}")
        print(f"  HEDGE SUMMARY")
        print(f"  {'='*72}")
        if not close_ev:
            print("\n  No completed hedge episodes.")
        else:
            all_pnl = [e['total_pnl'] for e in close_ev]
            n_pos   = sum(1 for p in all_pnl if p > 0)
            print(f"\n  Total episodes  : {len(close_ev)}")
            print(f"  Hit rate        : {n_pos/len(close_ev)*100:.1f}%  "
                  f"({n_pos} positive / {len(close_ev)-n_pos} negative)")
            print(f"  Total P&L       : {sum(all_pnl):+.4%}")
            print(f"  Avg P&L/episode : {np.mean(all_pnl):+.4%}")
            print(f"  Avg days held   : "
                  f"{np.mean([e.get('days_held',0) for e in close_ev]):.1f}")

            # Per-year
            print(f"\n  {'─'*68}")
            print(f"  PER-YEAR BREAKDOWN")
            print(f"  {'─'*68}")
            print(f"  {'Year':<6}  {'Episodes':>9}  {'Hit Rate':>9}  "
                  f"{'Total P&L':>10}  {'Avg P&L':>10}  {'Avg Days':>9}")
            print(f"  {'-'*60}")
            for yr in sorted(set(e['date'].year for e in close_ev)):
                ye  = [e for e in close_ev if e['date'].year == yr]
                yp  = [e['total_pnl'] for e in ye]
                ypos= sum(1 for p in yp if p > 0)
                yd  = np.mean([e.get('days_held', 0) for e in ye])
                print(f"  {yr:<6}  {len(ye):>9}  "
                      f"{ypos/len(ye)*100:>8.1f}%  "
                      f"{sum(yp):>+9.4%}  {np.mean(yp):>+9.4%}  {yd:>9.1f}")

            # Per-instrument
            inst_ev = {}
            for e in close_ev:
                for inst in e.get('instruments', []):
                    if inst not in inst_ev: inst_ev[inst] = []
                    d    = e['details'].get(inst, {})
                    w    = d.get('weight', 0)
                    epx  = d.get('entry_px', np.nan)
                    edt  = d.get('entry_dt', e['date'])
                    cdt  = e['date']
                    if (epx and not np.isnan(epx) and
                            inst in Pxs_df.columns and cdt in Pxs_df.index):
                        inst_pnl = -(Pxs_df.loc[cdt, inst] / epx - 1) * w
                    else:
                        inst_pnl = np.nan
                    inst_ev[inst].append({
                        'pnl': inst_pnl, 'weight': w,
                        'eff': d.get('eff', np.nan), 'beta': d.get('beta', np.nan),
                    })
            if inst_ev:
                print(f"\n  {'─'*68}")
                print(f"  PER-INSTRUMENT BREAKDOWN")
                print(f"  {'─'*68}")
                print(f"  {'Instrument':<12}  {'Uses':>5}  {'Hit Rate':>9}  "
                      f"{'Total P&L':>10}  {'Avg Weight':>11}  "
                      f"{'Avg Beta':>9}  {'Avg Eff':>8}")
                print(f"  {'-'*68}")
                for inst in sorted(inst_ev.keys()):
                    evs   = inst_ev[inst]
                    pnls  = [e['pnl'] for e in evs if not np.isnan(e['pnl'])]
                    pos   = sum(1 for p in pnls if p > 0)
                    avg_w = np.mean([e['weight'] for e in evs])
                    avg_b = np.nanmean([e['beta'] for e in evs])
                    avg_e = np.nanmean([e['eff']  for e in evs])
                    print(f"  {inst:<12}  {len(evs):>5}  "
                          f"{pos/len(pnls)*100:.1f}%  " if pnls else
                          f"  {inst:<12}  {len(evs):>5}  {'n/a':>9}  ", end='')
                    if pnls:
                        print(f"{sum(pnls):>+9.4%}  {avg_w:>10.2%}  "
                              f"{avg_b:>9.3f}  {avg_e:>8.3f}")

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        GRAY  = '#888780'; TEAL = '#1D9E75'; BLUE = '#378ADD'; CORAL = '#D85A30'
        COLORS = {
            'Baseline':    GRAY,
            'Pure Alpha':  TEAL,
            'MVO':         BLUE,
            'Hybrid':      '#7F77DD',
            'Smart Hybrid':'#E8A838',
            'Dynamic':     '#9B59B6',
            'Dyn+Hedge':   '#2ECC71',
            'DD Policy':   '#E74C3C',
            'Excl':        '#F39C12',
        }

        fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                                 gridspec_kw={'height_ratios': [3, 2, 2, 1]})
        fig.patch.set_facecolor('#FAFAF9')
        for ax in axes:
            ax.set_facecolor('#FAFAF9')

        # Common date index
        common = nav_series[0].index
        for ns in nav_series[1:]:
            if ns is not None and not ns.empty:
                common = common.intersection(ns.index)

        nav_r = [ns.reindex(common) if ns is not None and not ns.empty
                 else None for ns in nav_series]

        # Regime shading
        reg = regime_s.reindex(common, method='ffill').fillna(0)
        REGIME_BG = {0.0: '#E1F5EE', 0.5: '#F1EFE8', 1.0: '#FAECE7'}
        for ax in axes[:3]:
            prev_r, prev_d = float(reg.iloc[0]), common[0]
            for d, r in reg.items():
                r = float(r)
                if r != prev_r or d == common[-1]:
                    ax.axvspan(prev_d, d,
                               color=REGIME_BG.get(prev_r, '#F1EFE8'),
                               alpha=0.3, linewidth=0)
                    prev_r, prev_d = r, d

        # Panel 1: NAV
        ax = axes[0]
        for i, (ns, lbl) in enumerate(zip(nav_r, labels)):
            if ns is None: continue
            lw = 2.0 if lbl in ('Dyn+Hedge', 'DD Policy') else 1.2
            ax.plot(ns.index.to_numpy(), ns.values,
                    label=lbl, color=COLORS.get(lbl, GRAY), linewidth=lw)
        ax.set_ylabel("NAV", fontsize=10, color='#5F5E5A')
        ax.set_title("Strategy NAV Comparison",
                     fontsize=12, fontweight='500', color='#2C2C2A')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
        ax.grid(color='#D3D1C7', linewidth=0.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # Panel 2: Relative to baseline
        ax2 = axes[1]
        nb = nav_r[S_BASE]
        for i, (ns, lbl) in enumerate(zip(nav_r, labels)):
            if ns is None or i == S_BASE or nb is None: continue
            rel = (ns / nb - 1) * 100
            ax2.plot(rel.index.to_numpy(), rel.values,
                     label=f'{lbl} vs baseline',
                     color=COLORS.get(lbl, GRAY), linewidth=1.2)
            ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                             where=(rel.values >= 0),
                             color=COLORS.get(lbl, GRAY), alpha=0.06)
        ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
        ax2.set_ylabel("Relative to baseline (%)", fontsize=10, color='#5F5E5A')
        ax2.legend(fontsize=8, loc='upper left', framealpha=0.85)
        ax2.grid(color='#D3D1C7', linewidth=0.5)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        # Panel 3: Drawdown
        ax3 = axes[2]
        for ns, lbl in zip(nav_r, labels):
            if ns is None: continue
            dd = (ns / ns.cummax() - 1) * 100
            ax3.plot(dd.index.to_numpy(), dd.values,
                     label=lbl, color=COLORS.get(lbl, GRAY), linewidth=1.0)
        ax3.axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
        ax3.set_ylabel("Drawdown (%)", fontsize=10, color='#5F5E5A')
        ax3.legend(fontsize=8, loc='lower left', framealpha=0.85)
        ax3.grid(color='#D3D1C7', linewidth=0.5)
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

        # Panel 4: Regime
        ax4 = axes[3]
        ax4.fill_between(reg.index.to_numpy(), reg.values, 0,
                         color=BLUE, alpha=0.25)
        ax4.plot(reg.index.to_numpy(), reg.values, color=BLUE, linewidth=1.0)
        ax4.set_yticks([0.0, 0.5, 1.0])
        ax4.set_yticklabels(['Easy', 'Neutral', 'Tight'], fontsize=8)
        ax4.set_ylabel("Regime", fontsize=10, color='#5F5E5A')
        ax4.grid(color='#D3D1C7', linewidth=0.5)
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

        # DD exposure plot (if hedge enabled)
        if hedge_enabled:
            exposure_s = pd.Series(gross_records[S_DD]).sort_index()
            fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
            fig2.patch.set_facecolor('#FAFAF9')
            for ax in axes2:
                ax.set_facecolor('#FAFAF9')

            axes2[0].fill_between(exposure_s.index.to_numpy(),
                                  exposure_s.values, 0,
                                  where=(exposure_s.values < 1.0),
                                  color='#E74C3C', alpha=0.3, label='De-grossed')
            axes2[0].fill_between(exposure_s.index.to_numpy(),
                                  exposure_s.values, 0,
                                  where=(exposure_s.values >= 1.0),
                                  color='#2ECC71', alpha=0.2, label='Fully invested')
            axes2[0].set_ylim(0, 1.2)
            axes2[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axes2[0].set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            axes2[0].set_title('Gross Exposure (DD Policy)',
                               fontsize=11, fontweight='500', color='#2C2C2A')
            axes2[0].legend(fontsize=8)
            axes2[0].grid(color='#D3D1C7', linewidth=0.5)
            axes2[0].spines['top'].set_visible(False)
            axes2[0].spines['right'].set_visible(False)

            ndh = nav_series[S_HEDGE].reindex(common)
            ndd = nav_series[S_DD].reindex(common)
            axes2[1].plot(ndh.index.to_numpy(), ndh.values,
                          label='Dyn+Hedge', color='#2ECC71', linewidth=1.8)
            axes2[1].plot(ndd.index.to_numpy(), ndd.values,
                          label='DD Policy', color='#E74C3C', linewidth=1.8)
            axes2[1].set_title('NAV: Dyn+Hedge vs DD Policy',
                               fontsize=11, fontweight='500', color='#2C2C2A')
            axes2[1].legend(fontsize=8)
            axes2[1].grid(color='#D3D1C7', linewidth=0.5)
            axes2[1].spines['top'].set_visible(False)
            axes2[1].spines['right'].set_visible(False)

            plt.tight_layout()
            plt.show()

    except Exception as _plot_err:
        print(f"\n  (Plot skipped: {_plot_err})")

    # ── 12. Return results ────────────────────────────────────────────────────
    return {
        'nav_baseline':               nav_series[S_BASE],
        'nav_alpha':                  nav_series[S_ALPHA],
        'nav_mvo':                    nav_series[S_MVO],
        'nav_hybrid':                 nav_series[S_HYB],
        'nav_smart':                  nav_series[S_SMART],
        'nav_dynamic':                nav_series[S_DYN],
        'nav_dyn_hedged':             nav_series[S_HEDGE],
        'nav_dd':                     nav_series[S_DD],
        'nav_dd_excl':                nav_series[S_EXCL],
        'state':                      state,
        'nav_series':                 nav_series,
        'labels':                     labels,
        # ── Portfolio weights by strategy ──────────────────────────────────────
        'alpha_weights_by_date':      state[S_ALPHA]['weights_by_date'],
        'mvo_weights_by_date':        state[S_MVO]['weights_by_date'],
        'hybrid_weights_by_date':     state[S_HYB]['weights_by_date'],
        'smart_weights_by_date':      state[S_SMART]['weights_by_date'],
        'dyn_weights_by_date':        state[S_DYN]['weights_by_date'],
        'hedge_weights_by_date':      state[S_HEDGE]['weights_by_date'],
        'dd_weights_by_date':         state[S_DD]['weights_by_date'],
        'excl_weights_by_date':       state[S_EXCL]['weights_by_date'],
        'baseline_weights_by_date':   state[S_BASE]['weights_by_date'],
        'exposure_tbl':               EXPOSURE_TBL,
        'params_hash':                ph,
    }

