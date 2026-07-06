#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    results = run_backtest(
        Pxs_df, sectors_s, weights_by_year, regime_s,
        volumeTrd_df    = volumeTrd_df,
        weights_by_date = weights_by_date,
        # MVO parameters:
        ic=0.05,
        max_weight=0.075,
        min_weight=0.025,
        zscore_cap=2.5,
        pca_var_threshold=0.65,
        universe_mult=7,
        risk_aversion=5,            # default MVO_RISK_AVERSION; override here per run
        min_cov_matrices=2,
        mode='rebuild',             # 'incremental' (fast last-day) | 'rebuild' (flag-controlled)
        # On mode='rebuild' these flags are honoured as passed (False=reuse cache,
        # True=recompute). On mode='incremental' they are ignored. Set True after:
        universe_changed     = False,   # added/removed stocks
        factor_model_changed = False,   # changed HISTORICAL factor data (not new dates)
        weights_changed      = False,   # regenerated alpha IC weights_by_date
        regime_changed       = False,   # changed regime series
        mr_changed           = False,   # changed MR params / scores
        # Hedge layer (omit hedge_multi / hedges_l to run without hedging):
        hedge_multi          = multi,
        hedges_l             = hedges_l,
        hedge_trigger_assets = ['QQQ'],
    )

Returns
-------
    dict with nav_baseline, nav_alpha, nav_mvo, nav_hybrid, nav_smart,
         nav_dynamic, nav_dyn_hedged, nav_dd, nav_dd_excl, nav_mvo_hedge,
         per-strategy weights_by_date, and intermediate data
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
MB_TOP_N       = 25       # default number of stocks
MB_REBAL_FREQ  = 15       # default rebalance frequency (trading days)
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
MVO_RISK_AVERSION    = 5      # default risk-aversion (variance penalty) for the MVO solve;
                              # overridable per run via the run_backtest(risk_aversion=...) arg

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
DD_REENTRY_PCT      = 0.150
DD_REENTRY_CONFIRM  = 5
DD_HWM_WINDOW       = 12    # rolling HWM lookback in months (set to 0 for lifetime HWM)

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
QUALITY_FLOOR = 0.2   # set to e.g. 0.5 to activate
MOM_FILTER    = 3     # alpha/hybrid candidate pool = MOM_FILTER x top_n by momentum
TIER2_T       = 0     # total stocks incl overlay (0 = disabled, user sets at runtime)
TIERONE_ALLOC = 0.90  # fraction allocated to tier 1 core portfolio
CORR_FILTER   = 0.70  # fraction of non-core universe kept after correlation filter
MOM_6M1_WIN   = 126   # 6M1 momentum lookback (trading days)
MOM_6M1_SKIP  = 21    # 6M1 momentum skip

# ── Hedge engine ──────────────────────────────────────────────────────────────
BETA_WINDOW    = 63     # rolling window for beta calculation
CORR_WINDOW    = 63     # rolling window for correlation ranking
EFF_MAV_WINDOW = 20     # MAV window for smoothing effectiveness
EFF_FLOOR      = 1.00   # minimum effectiveness score to qualify
CORR_FLOOR     = 0.50   # minimum correlation to portfolio to qualify
HEDGE_RATIO    = 0.25   # hedge size per instrument (fraction of NAV)
MAX_HEDGE      = 0.50   # maximum total hedge (fraction of NAV)
TRIGGER_ASSETS = ['QQQ', 'SPY']  # assets that trigger hedge on/off

# ── Cache / DB table names ────────────────────────────────────────────────────
MB_COV_CACHE_TBL  = 'mvo_cov_cache'
MB_X_CACHE_TBL    = 'mvo_x_snapshots'
MB_COMPOSITE_CACHE_TBL = 'mvo_composite_cache'   # daily composite + factor-score panel
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
_MB_F_LOOKBACK    = 504  # lookback window for F matrix (trading days; 1260 ≈ 5 years)
_MB_F_EWMA_HL     = 252   # EWMA half-life for F matrix (trading days; also used as min_periods)


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
    Build the broad candidate universe: sector-mapped, DB-present stocks
    with any valid price in Pxs_df. No pre-start history gate — recent IPOs
    and re-listings are included as soon as they have any price data.

    Per-date eligibility (MIN_HIST days of valid, non-stale price history)
    is enforced at each calc_date by _get_active_universe, which naturally
    handles both entry (new listings) and exit (delisted/stale stocks).
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

    universe = []
    for col in sectors_s.index:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        if col not in Pxs_df.columns:
            continue
        if Pxs_df[col].notna().sum() < 1:
            continue
        universe.append(col)

    print(f"  Universe: {len(universe)} stocks "
          f"(sector mapped + DB + any valid price in Pxs_df)")
    return universe


def generate_calc_dates(Pxs_df, step_days=30):
    """Generate rebalancing dates from MB_START_DATE at step_days CALENDAR-day intervals.

    Each stepped (calendar-lagged) date is adjusted to the first available trading
    day in Pxs_df.index that is equal to or later than it. Weekends and market
    holidays need no special handling — they simply have no row in Pxs_df.index,
    so the snap lands on the next genuine trading day automatically.
    """
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


_MIN_HISTORY_DAYS  = 126   # minimum trading days of valid price history (aligned with factor model MIN_HIST)
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
        rate_q = float(regime_s.reindex([dt], method='ffill').iloc[0])                  if dt in regime_s.index or len(regime_s) > 0 else 0.5
        if rate_q == 0.0:
            required_fundamentals = {'Quality'}
        else:
            required_fundamentals = {'Quality', 'Value'}

        fund_coverage = pd.Series(0, index=active_u)
        for fname in required_fundamentals:
            if fname in score_dfs and score_dfs[fname] is not None                     and dt in score_dfs[fname].index:
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

def _mb_build_cov_matrices(dt, candidates, Pxs_df, sectors_s,
                             volumeTrd_df, model_version,
                             pca_var_threshold, X_df_cached=None,
                             lambda_dfs=None):
    """
    Build all four covariance matrices for the candidate universe on date dt.
    Returns (Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens).
    """
    # Return matrix — strictly up to dt (no lookahead), PIT
    # _get_active_universe already guarantees >= _MIN_HISTORY_DAYS valid days
    # per stock, so dropna(axis=1, how='any') within the MVO_LOOKBACK window
    # is safe and correct — no fillna needed
    ret_raw = (Pxs_df.loc[:dt, candidates].pct_change()
               .dropna(how='all')
               .iloc[-MVO_LOOKBACK:])
    min_obs  = min(_MIN_HISTORY_DAYS, len(ret_raw))
    ret_df   = ret_raw.loc[:, ret_raw.notna().sum() >= min_obs]
    ret_df   = ret_df.dropna(axis=1, how='any')
    valid    = ret_df.columns.tolist()

    # Empirical
    Sigma_emp = _mvo_ewma_cov(ret_df, MVO_EWMA_HL)

    # Ledoit-Wolf
    Sigma_lw, _, _ = _mvo_ledoit_wolf(ret_df.values)

    # Factor-driven (self-contained -- no _mvo_factor_cov dependency)
    try:
        F_mat, factor_names_f, sec_cols_f, _f_shape = _mb_build_F(
            model_version, dt=dt, lambda_dfs=lambda_dfs)
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
        _f_active = factor_mean_var > 1e-12
        if not _f_active:
            _x_nonzero = np.count_nonzero(X)
            _f_nonzero = np.count_nonzero(F_mat)
            _x_cols_in_cache = (len([c for c in factor_names_f
                                     if X_df_cached is not None
                                     and c in X_df_cached.columns])
                                if X_df_cached is not None else 0)
            warnings.warn(
                f"  F-ZERO diag: X.shape={X.shape} X_nonzero={_x_nonzero} "
                f"F.shape={F_mat.shape} F_nonzero={_f_nonzero} "
                f"F_max={np.abs(F_mat).max():.2e} "
                f"X_cache_cols_matched={_x_cols_in_cache}/{len(factor_names_f)}"
            )
        if _f_active:
            Sigma_factor = Sigma_factor_raw * (emp_mean_var / factor_mean_var)
        else:
            Sigma_factor = Sigma_emp.copy()
    except Exception as e:
        warnings.warn(f"  Factor cov failed ({e}) -- using empirical")
        Sigma_factor  = Sigma_emp.copy()
        factor_mean_var = 0.0
        _f_shape      = (0, 0)
        _f_active     = False

    # PCA
    Sigma_pca, _, _, _ = _mvo_pca_cov(ret_df.values,
                                        var_threshold=pca_var_threshold)

    # Ensemble
    Sigma_ens = (Sigma_emp + Sigma_lw + Sigma_factor + Sigma_pca) / 4.0

    return (valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens,
            {'f_shape': _f_shape, 'f_mean_var': factor_mean_var,
             'f_active': _f_active})


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
    raw    = {}

    # Scalar factor lambdas
    for fname, s in lambda_dfs['scalar'].items():
        pit = s[s.index <= cutoff]
        if not pit.empty:
            raw[fname] = pit

    # Macro lambdas
    macro_df = lambda_dfs['macro']
    if not macro_df.empty:
        pit_m = macro_df[macro_df.index <= cutoff]
        for mc in pit_m.columns:
            raw[mc] = pit_m[mc]

    # Sector lambdas
    sec_df = lambda_dfs['sector']
    if not sec_df.empty:
        pit_s = sec_df[sec_df.index <= cutoff]
        for sc in pit_s.columns:
            raw[sc] = pit_s[sc]

    if not raw:
        factor_names, sec_cols = _mb_get_factor_names(model_version, lambda_dfs=lambda_dfs)
        return np.zeros((len(factor_names), len(factor_names))), [], sec_cols, (0, 0)

    # Build unified daily date range, reindex all series to it
    all_dates = pd.date_range(
        start = min(s.index.min() for s in raw.values()),
        end   = cutoff, freq='B'
    )
    combined = pd.DataFrame(
        {name: s.reindex(all_dates) for name, s in raw.items()},
        index=all_dates
    )

    # Expand all series to unified daily cadence — ffill only (fills macro gaps
    # with last known value; no bfill to avoid extrapolating backwards)
    combined = combined.ffill()

    # Drop factors with no data at all in the window (entirely NaN)
    combined = combined.dropna(axis=1, how='all')

    # Keep only rows where ALL remaining factors have data — these are dates
    # where every factor had at least one actual observation up to that point
    combined = combined.dropna(how='any')

    # Take the last _MB_F_LOOKBACK days of actual joint coverage.
    # If fewer than _MB_F_EWMA_HL are available (e.g. Value still new),
    # use whatever is available — a shorter window is better than no F matrix.
    combined = combined.iloc[-_MB_F_LOOKBACK:]

    factor_names, sec_cols = _mb_get_factor_names(model_version, lambda_dfs=lambda_dfs)
    ordered  = [c for c in factor_names if c in combined.columns]
    combined = combined[ordered]

    if combined.empty:
        warnings.warn(f"  _mb_build_F: no joint coverage found "
                      f"(cutoff={cutoff.date()}) — F matrix will be zero")
        F = np.zeros((len(ordered), len(ordered)))
    else:
        F = _mb_ewma_cov_f(combined, _MB_F_EWMA_HL)

    return F, ordered, sec_cols, combined.shape


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
    if len(pxs_to_dt) < _MIN_HISTORY_DAYS:
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


def _make_composite_cache_key(universe, model_version):
    """
    Cache key for the composite/factor panel. Invalidation is controlled EXPLICITLY
    by the run_backtest event flags (factor_model_changed / weights_changed /
    regime_changed / mr_changed) -- NOT by content fingerprints -- so the key needs
    only to separate panels that must never be mixed under one key:
      - model_version  (v1 vs v2 are different factor models entirely)
      - universe        (structural name-set hash): a backstop so that a changed
        stock universe can never silently reuse composites built for a different
        ticker set, even if `universe_changed=True` is forgotten. This is a name-set
        hash (structural), not a value comparison.
    Everything else (prices, weights_by_date, MVO params, regime, MR, factor values)
    is intentionally OUT of the key: prices/weights only affect the always-rebuilt
    last day; regime/MR/factor-data invalidation is the user's explicit flag call.
    """
    _ufp = hashlib.md5(
        json.dumps(sorted(universe), sort_keys=True).encode()).hexdigest()[:12]
    params = dict(mv=model_version, uni=_ufp)
    return hashlib.md5(
        json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _ensure_composite_cache_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MB_COMPOSITE_CACHE_TBL} (
                cache_key     VARCHAR(16) NOT NULL,
                date          DATE        NOT NULL,
                composite_json TEXT       NOT NULL,
                scores_json    TEXT       NOT NULL,
                PRIMARY KEY (cache_key, date)
            )
        """))


def _load_composite_cache(cache_key, before_dt):
    """
    Load cached composite + factor scores for cache_key, for all dates STRICTLY
    BEFORE before_dt (the never-frozen last day is never trusted from cache).
    Returns (composite_by_date dict, score_dfs dict, set_of_cached_dates).
    """
    try:
        _ensure_composite_cache_table()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT date, composite_json, scores_json
                FROM {MB_COMPOSITE_CACHE_TBL}
                WHERE cache_key = :k AND date < :bd
                ORDER BY date
            """), {'k': cache_key, 'bd': pd.Timestamp(before_dt).strftime('%Y-%m-%d')}
            ).fetchall()
    except Exception:
        return {}, {}, set()

    composite_by_date = {}
    _factor_rows = {}   # factor -> {date -> {ticker: val}}
    for r in rows:
        dt = pd.Timestamp(r[0])
        composite_by_date[dt] = pd.Series(json.loads(r[1]))
        sc = json.loads(r[2])   # {factor: {ticker: val}}
        for fname, tv in sc.items():
            _factor_rows.setdefault(fname, {})[dt] = tv

    score_dfs = {}
    for fname, by_dt in _factor_rows.items():
        score_dfs[fname] = pd.DataFrame.from_dict(by_dt, orient='index').sort_index()

    return composite_by_date, score_dfs, set(composite_by_date.keys())


def _save_composite_cache(cache_key, dates, composite_by_date, score_dfs):
    """Persist composite + per-date factor-score slices for the given dates.
    Only dates STRICTLY BEFORE the current last day should be passed here
    (the last day is never frozen)."""
    if not dates:
        return
    _ensure_composite_cache_table()
    factors = list(score_dfs.keys())
    with ENGINE.begin() as conn:
        for dt in dates:
            comp = composite_by_date.get(dt)
            if comp is None:
                continue
            comp_j = json.dumps({k: float(v) for k, v in comp.dropna().items()})
            sc = {}
            for fname in factors:
                fdf = score_dfs.get(fname)
                if fdf is not None and dt in fdf.index:
                    row = fdf.loc[dt].dropna()
                    sc[fname] = {k: float(v) for k, v in row.items()}
            conn.execute(text(f"""
                INSERT INTO {MB_COMPOSITE_CACHE_TBL}
                    (cache_key, date, composite_json, scores_json)
                VALUES (:k, :dt, :cj, :sj)
                ON CONFLICT (cache_key, date) DO UPDATE
                    SET composite_json = EXCLUDED.composite_json,
                        scores_json    = EXCLUDED.scores_json
            """), {'k': cache_key, 'dt': pd.Timestamp(dt).strftime('%Y-%m-%d'),
                   'cj': comp_j, 'sj': json.dumps(sc)})


def _clear_composite_cache(cache_key=None):
    """Clear composite cache for a key (or all keys if None)."""
    try:
        _ensure_composite_cache_table()
        with ENGINE.begin() as conn:
            if cache_key is None:
                conn.execute(text(f"DELETE FROM {MB_COMPOSITE_CACHE_TBL}"))
            else:
                conn.execute(text(
                    f"DELETE FROM {MB_COMPOSITE_CACHE_TBL} WHERE cache_key = :k"),
                    {'k': cache_key})
    except Exception:
        pass


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

    Parameters
    ----------
    weights    : current portfolio weights (top_n stocks)
    candidates : composite scores for full candidate universe (ranked desc)
    current_aum: AUM at this date (AUM × running_nav)
    advp_cap   : fraction of ADV allowed per stock
    min_weight : minimum portfolio weight — stocks below this capacity are excluded
    top_n      : target number of stocks
    conc_factor: used to rebuild weights after replacement
    target_n   : target number of stocks after filtering. Defaults to top_n. For
                 hybrid portfolios, pass len(weights) to preserve the larger union
                 without truncating to top_n.

    Returns
    -------
    (filtered_and_replaced_weights Series, affected_tickers set), always
    attempting target_n stocks.
    """
    if target_n is None:
        target_n = top_n
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

    # Flag a stock as ADVP-capped only if its cap was actually BINDING — i.e. the
    # cap sits below the equal-weight allocation the stock would otherwise receive
    # in its tier (so the cap forced its weight down). This drives the '***' display
    # flag only; it does not affect the weights themselves.
    _capped = {t for t in _advp_caps if t in w_new and
               _advp_caps[t] < (top_a / n_top if t in top_tks else
                                 bot_a / n_bot if n_bot > 0 else top_a / n_top)}

    _affected = _excluded | _capped
    w_final = w_new / w_new.sum() if w_new.sum() > 0 else w_new
    return w_final, _affected


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
    ret_raw = (Pxs_df.loc[:dt, candidates].pct_change()
               .dropna(how='all')
               .iloc[-MVO_LOOKBACK:])
    min_obs  = min(_MIN_HISTORY_DAYS, len(ret_raw))
    ret_df   = ret_raw.loc[:, ret_raw.notna().sum() >= min_obs]
    ret_df   = ret_df.dropna(axis=1, how='any')
    valid = [t for t in candidates if t in ret_df.columns]
    if len(valid) < top_n:
        warnings.warn(f"  {dt.date()} SKIP: only {len(valid)} valid candidates")
        return pd.Series(dtype=float), {}

    # -- Build covariance matrices ---------------------------------------------
    try:
        valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens, _f_diag =             _mb_build_cov_matrices(dt, valid, Pxs_df, sectors_s,
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
        'cov_sums'       : {
            'Emp'   : float(Sigma_emp.sum()),
            'LW'    : float(Sigma_lw.sum()),
            'Factor': float(Sigma_factor.sum()),
            'PCA'   : float(Sigma_pca.sum()),
        },
        'f_diag'         : _f_diag,
    }
    return w_out, diag



def get_last_rebal_diagnostics(results, Pxs_df,
                                top_n=MB_TOP_N,
                                prefilt_pct=0.4,
                                mom_filter=MOM_FILTER,
                                universe_mult=5,
                                min_cov_matrices=2):
    """
    Return stock-level diagnostics for the last rebalancing date.

    Parameters
    ----------
    results        : dict returned by run_backtest
    Pxs_df         : price panel (same one passed to run_backtest)
    top_n / prefilt_pct / mom_filter / universe_mult / min_cov_matrices :
                    same values used in the backtest run

    Returns
    -------
    scores_df      : DataFrame (index=ticker) with factor scores, composite,
                     composite rank, and filter_stage column showing where
                     each stock dropped out
    ic_weights     : Series of PIT IC weights used on that date
    prefilt_stocks : list — passed quality prefilt
    alpha_pool     : list — top N × mom_filter by composite (after prefilt)
    mvo_pool       : list — top N × universe_mult by composite (full universe)
    """
    composite_by_date = results['composite_scores']
    score_dfs         = results['score_dfs']
    weights_by_date   = results['ic_weights_by_date']
    universe          = results['universe']

    # ── Last rebalancing date ─────────────────────────────────────────────────
    dyn_wbd = results.get('dyn_weights_by_date', {})
    if dyn_wbd:
        last_dt = max(dyn_wbd.keys())
    else:
        last_dt = max(composite_by_date.keys())

    print(f"  Diagnostics for last rebalancing date: {last_dt.date()}")

    # ── Composite scores at last_dt ───────────────────────────────────────────
    avail_dates = [d for d in sorted(composite_by_date.keys()) if d <= last_dt]
    comp_dt     = avail_dates[-1] if avail_dates else None
    if comp_dt is None:
        print("  No composite scores available.")
        return None, None, [], [], []

    comp_s = composite_by_date[comp_dt].dropna().sort_values(ascending=False)

    # ── PIT IC weights ────────────────────────────────────────────────────────
    wbd_dates  = sorted(weights_by_date.keys()) if weights_by_date else []
    past_wbd   = [d for d in wbd_dates if d <= last_dt]
    ic_weights = pd.Series(weights_by_date[past_wbd[-1]], name='IC weight')                  if past_wbd else pd.Series(dtype=float)

    # ── Individual factor scores ──────────────────────────────────────────────
    factor_cols = {}
    for fname, sdf in score_dfs.items():
        if sdf is None or sdf.empty:
            continue
        avail = [d for d in sdf.index if d <= last_dt]
        if avail:
            factor_cols[fname] = sdf.loc[avail[-1]]

    # ── Build scores_df ───────────────────────────────────────────────────────
    scores_df = pd.DataFrame(index=universe)
    scores_df.index.name = 'ticker'
    for fname, s in factor_cols.items():
        scores_df[fname] = s.reindex(universe)
    scores_df['Composite']      = comp_s.reindex(universe)
    scores_df['Composite_rank'] = scores_df['Composite'].rank(ascending=False,
                                                               na_option='bottom')
    scores_df = scores_df.sort_values('Composite', ascending=False)

    # ── Filter stage replication ──────────────────────────────────────────────
    active_u = _get_active_universe(universe, Pxs_df, last_dt)

    # Stage 1: active universe (price/history filter)
    scores_df['filter_stage'] = 'inactive (price/history)'
    scores_df.loc[scores_df.index.isin(active_u), 'filter_stage'] =         f'active ({len(active_u)} stocks)'

    # Stage 2: quality prefilt
    n_prefilt   = max(1, int(len(active_u) * prefilt_pct))
    qual_s      = factor_cols.get('Quality', pd.Series(dtype=float))
    qual_ranked = qual_s.reindex(active_u).sort_values(ascending=False)
    prefilt_stocks = qual_ranked.iloc[:n_prefilt].index.tolist()
    scores_df.loc[scores_df.index.isin(prefilt_stocks), 'filter_stage'] =         f'prefilt ({n_prefilt} stocks)'

    # Stage 3: alpha/hybrid pool (top N × mom_filter by composite, after prefilt)
    n_alpha    = top_n * mom_filter
    comp_prefilt = comp_s.reindex(prefilt_stocks).dropna().sort_values(ascending=False)
    alpha_pool = comp_prefilt.iloc[:n_alpha].index.tolist()
    scores_df.loc[scores_df.index.isin(alpha_pool), 'filter_stage'] =         f'alpha pool (top {n_alpha})'

    # Stage 4: MVO pool (top N × universe_mult by composite, full active_u)
    n_mvo    = top_n * universe_mult
    comp_active = comp_s.reindex(active_u).dropna().sort_values(ascending=False)
    mvo_pool    = comp_active.iloc[:n_mvo].index.tolist()
    scores_df.loc[scores_df.index.isin(mvo_pool), 'filter_stage'] =         f'MVO pool (top {n_mvo})'

    # Stocks in both pools get the more specific label
    in_both = set(alpha_pool) & set(mvo_pool)
    scores_df.loc[scores_df.index.isin(in_both), 'filter_stage'] =         f'alpha+MVO pool'

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n  Universe        : {len(universe)} stocks")
    print(f"  Active (price)  : {len(active_u)} stocks")
    print(f"  Prefilt (qual)  : {len(prefilt_stocks)} stocks  "
          f"(top {prefilt_pct*100:.0f}% by Quality)")
    print(f"  Alpha pool      : {len(alpha_pool)} stocks  "
          f"(top {n_alpha} by Composite after prefilt)")
    print(f"  MVO pool        : {len(mvo_pool)} stocks  "
          f"(top {n_mvo} by Composite from active_u)")
    print(f"\n  PIT IC weights ({comp_dt.date()}):")
    # Replicate the composite builder's QUALITY_FLOOR redistribution EXACTLY (it
    # redistributes within {Idio_Mom, Mom_12M1, Quality}, Value untouched), so the
    # printed weights match what actually drove the composite — not the raw
    # pre-floor weights_by_date.
    _disp = ic_weights.copy()
    _floor_applied = False
    if QUALITY_FLOOR > 0 and not _disp.empty:
        _mom_f = [f for f in ('Idio_Mom', 'Mom_12M1') if f in _disp.index]
        if 'Quality' in _disp.index and _mom_f:
            _wmom = sum(_disp[f] for f in _mom_f)
            _wq   = _disp['Quality']
            _wmq  = _wmom + _wq
            _share = (_wq / _wmq) if _wmq > 0 else 0.0
            if _share < QUALITY_FLOOR:
                _disp = _disp.copy()
                _disp['Quality'] = _wmq * QUALITY_FLOOR
                _rem = _wmq * (1.0 - QUALITY_FLOOR)
                if _wmom > 0:
                    for f in _mom_f:
                        _disp[f] = _rem * (ic_weights[f] / _wmom)
                _floor_applied = True
    for f, w in _disp.items():
        print(f"    {f:<15} {w:.4f}")
    if _floor_applied:
        print(f"    (post-QUALITY_FLOOR={QUALITY_FLOOR}: Quality lifted from "
              f"{ic_weights['Quality']:.4f}; momentum rescaled, Value untouched)")
    elif QUALITY_FLOOR > 0:
        print(f"    (QUALITY_FLOOR={QUALITY_FLOOR} did not bind this date)")

    return scores_df, ic_weights, prefilt_stocks, alpha_pool, mvo_pool




MVO_NAV_CACHE_TBL    = 'mvo_nav_cache'
MVO_REBAL_CACHE_TBL  = 'mvo_rebal_cache'
MVO_STATE_CACHE_TBL  = 'mvo_state_cache'


def _ensure_cache_tables():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MVO_NAV_CACHE_TBL} (
                params_hash  VARCHAR(16) NOT NULL,
                strategy     VARCHAR(20) NOT NULL,
                date         DATE        NOT NULL,
                nav          DOUBLE PRECISION,
                PRIMARY KEY (params_hash, strategy, date)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MVO_REBAL_CACHE_TBL} (
                params_hash  VARCHAR(16) NOT NULL,
                strategy     VARCHAR(20) NOT NULL,
                date         DATE        NOT NULL,
                ticker       VARCHAR(20) NOT NULL,
                weight       DOUBLE PRECISION,
                gross_factor DOUBLE PRECISION DEFAULT 1.0,
                PRIMARY KEY (params_hash, strategy, date, ticker)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MVO_STATE_CACHE_TBL} (
                params_hash  VARCHAR(16) NOT NULL,
                strategy     VARCHAR(20) NOT NULL,
                state_json   TEXT,
                last_date    DATE,
                PRIMARY KEY (params_hash, strategy)
            )
        """))


def _compute_params_hash(params: dict) -> str:
    import hashlib, json
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _state_to_json(st: dict) -> str:
    import json
    fields = ['nav','hwm','hwm_rolling','dd','gross','trough','theo_nav',
              'theo_trough','days_held','regime','deployed_regime','dd_level',
              'dd_active','dd_regime_forced','reentry_count','hedge_acct',
              'costs','costs_by_year','active_hedges']
    snap = {}
    for f in fields:
        v = st.get(f)
        if isinstance(v, pd.Timestamp):
            snap[f] = v.isoformat()
        else:
            snap[f] = v
    # weights and last_rebal stored separately
    snap['last_rebal'] = st['last_rebal'].isoformat()                          if isinstance(st.get('last_rebal'), pd.Timestamp) else None
    snap['weights'] = st['weights'].to_dict() if not st['weights'].empty else {}
    # Persist the DD de-gross/re-entry schedule so incremental runs carry the
    # full history (previously wiped on restore -> incremental reported 0 events
    # and the schedule only existed transiently during a rebuild run). Timestamps
    # inside each event dict are serialized to ISO strings.
    snap['dd_log'] = [
        {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
         for k, v in ev.items()}
        for ev in st.get('dd_log', [])
    ]
    return json.dumps(snap)


def _state_from_json(js: str) -> dict:
    import json
    snap = json.loads(js)
    snap['last_rebal'] = pd.Timestamp(snap['last_rebal'])                          if snap.get('last_rebal') else None
    snap['weights'] = pd.Series(snap['weights'], dtype=float)
    # fields not in JSON: init with correct types
    snap['nav_series']      = {}
    snap['weights_by_date'] = {}
    # dd_log IS persisted (full de-gross/re-entry schedule survives incremental
    # runs); deserialize its 'dt' timestamps. hedge_log remains reset -- the hedge
    # schedule is produced independently by the hedging script.
    snap['dd_log'] = [
        {k: (pd.Timestamp(v) if k == 'dt' and v is not None else v)
         for k, v in ev.items()}
        for ev in snap.get('dd_log', []) or []
    ]
    snap['hedge_log'] = []
    # Reconstruct minimal rebal_log so trade summary can compute _drifted_w
    if snap.get('last_rebal') and not snap['weights'].empty:
        snap['rebal_log'] = [{'dt':     snap['last_rebal'],
                              'w':      snap['weights'].copy(),
                              'regime': snap.get('regime', 'alpha'),
                              'nav':    snap.get('nav', 1.0)}]
    else:
        snap['rebal_log'] = []
    return snap


def _cache_load(params_hash: str, strategy_labels: dict):
    """
    Load cached NAV, last weights, and state from DB.
    Returns (nav_by_sidx, last_date, state_by_sidx) or (None, None, None).
    """
    import json
    _ensure_cache_tables()

    # Check if ANY data exists for this hash
    with ENGINE.connect() as conn:
        cnt = conn.execute(text(f"""
            SELECT COUNT(*) FROM {MVO_NAV_CACHE_TBL}
            WHERE params_hash = :ph
        """), {'ph': params_hash}).scalar()
    if cnt == 0:
        return None, None, None

    # Load NAV
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT strategy, date, nav FROM {MVO_NAV_CACHE_TBL}
            WHERE params_hash = :ph ORDER BY strategy, date
        """), {'ph': params_hash}).fetchall()
    nav_by_label = {}
    for lbl, dt, nav in rows:
        nav_by_label.setdefault(lbl, {})[pd.Timestamp(dt)] = float(nav)

    # Last cached date (minimum across strategies = safe restart point)
    all_last = [max(d.keys()) for d in nav_by_label.values() if d]
    last_date = min(all_last) if all_last else None

    # Load state JSON
    with ENGINE.connect() as conn:
        srows = conn.execute(text(f"""
            SELECT strategy, state_json FROM {MVO_STATE_CACHE_TBL}
            WHERE params_hash = :ph
        """), {'ph': params_hash}).fetchall()
    state_by_label = {lbl: _state_from_json(js) for lbl, js in srows}

    # Map back to s_idx
    label_to_sidx = {v: k for k, v in strategy_labels.items()}
    nav_by_sidx   = {label_to_sidx[lbl]: d
                     for lbl, d in nav_by_label.items()
                     if lbl in label_to_sidx}
    state_by_sidx = {label_to_sidx[lbl]: s
                     for lbl, s in state_by_label.items()
                     if lbl in label_to_sidx}

    return nav_by_sidx, last_date, state_by_sidx


def _cache_save_nav(params_hash: str, strategy: str, new_nav: dict):
    """Bulk-upsert {date: nav} dict."""
    if not new_nav:
        return
    rows = [{'ph': params_hash, 'st': strategy,
             'dt': dt.date() if isinstance(dt, pd.Timestamp) else dt,
             'nav': float(v)}
            for dt, v in new_nav.items()]
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {MVO_NAV_CACHE_TBL} (params_hash, strategy, date, nav)
            VALUES (:ph, :st, :dt, :nav)
            ON CONFLICT (params_hash, strategy, date) DO UPDATE SET nav = EXCLUDED.nav
        """), rows)


def _cache_save_rebal(params_hash: str, strategy: str,
                      date: pd.Timestamp, weights: pd.Series,
                      gross: float = 1.0):
    """Save a rebalancing event."""
    if weights.empty:
        return
    rows = [{'ph': params_hash, 'st': strategy,
             'dt': date.date(), 'tk': tk,
             'w': float(w), 'gf': float(gross)}
            for tk, w in weights.items()]
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {MVO_REBAL_CACHE_TBL}
                (params_hash, strategy, date, ticker, weight, gross_factor)
            VALUES (:ph, :st, :dt, :tk, :w, :gf)
            ON CONFLICT (params_hash, strategy, date, ticker)
            DO UPDATE SET weight = EXCLUDED.weight,
                          gross_factor = EXCLUDED.gross_factor
        """), rows)


def _cache_save_state(params_hash: str, strategy: str,
                      state: dict, last_date: pd.Timestamp):
    js = _state_to_json(state)
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {MVO_STATE_CACHE_TBL}
                (params_hash, strategy, state_json, last_date)
            VALUES (:ph, :st, :js, :ld)
            ON CONFLICT (params_hash, strategy)
            DO UPDATE SET state_json = EXCLUDED.state_json,
                          last_date  = EXCLUDED.last_date
        """), {'ph': params_hash, 'st': strategy,
               'js': js, 'ld': last_date.date()})


def _cache_clear(params_hash: str):
    with ENGINE.begin() as conn:
        for tbl in [MVO_NAV_CACHE_TBL, MVO_REBAL_CACHE_TBL, MVO_STATE_CACHE_TBL]:
            conn.execute(text(f"""
                DELETE FROM {tbl} WHERE params_hash = :ph
            """), {'ph': params_hash})
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


def _select_hedge_instruments(dt:             pd.Timestamp,
                               hedges_l:      list,
                               results:       dict,
                               port_ret_s:    pd.Series,
                               inst_ret:      dict,
                               beta_window:   int,
                               corr_window:   int,
                               eff_mav_win:   int,
                               eff_floor:     float,
                               corr_floor:    float,
                               trigger_assets: list) -> dict:
    """
    Select qualifying hedge instruments at activation date dt.

    1. Always check trigger assets (QQQ, SPY) first — include if corr >= corr_floor
       AND effectiveness >= eff_floor.
    2. Add top 3 from correlation ranking (excluding already selected) subject to
       the same corr/eff filters.

    Returns dict {inst: {'beta', 'corr', 'eff', 'effectiveness'}}
    """
    scores = {}
    for inst in hedges_l:
        if inst not in inst_ret or inst not in results:
            continue
        corr = _compute_corr(port_ret_s, inst_ret[inst], corr_window)
        eff  = _get_effectiveness(results[inst]['signal_df'], dt, eff_mav_win)
        beta = _compute_beta(port_ret_s, inst_ret[inst], beta_window)
        scores[inst] = {'corr': corr, 'eff': eff, 'beta': beta,
                        'effectiveness': eff}

    # Rank by correlation descending
    ranked = sorted(scores.items(),
                    key=lambda x: x[1]['corr'] if not np.isnan(x[1]['corr']) else -999,
                    reverse=True)

    selected = {}

    # Always check trigger assets first (QQQ, SPY)
    for ta in trigger_assets:
        if ta in scores:
            d    = scores[ta]
            corr = d['corr'] if not np.isnan(d.get('corr', np.nan)) else 0
            eff  = d['eff']  if not np.isnan(d.get('eff',  np.nan)) else 0
            if corr >= corr_floor and eff >= eff_floor:
                selected[ta] = d

    # Add top 3 from correlation ranking (excluding already selected)
    n_from_ranking = 0
    for inst, d in ranked:
        if inst in selected:
            continue
        if n_from_ranking >= 3:
            break
        corr = d['corr'] if not np.isnan(d.get('corr', np.nan)) else 0
        eff  = d['eff']  if not np.isnan(d.get('eff',  np.nan)) else 0
        if corr < corr_floor:
            break   # ranked by corr descending — all remaining will also fail
        if eff >= eff_floor:
            selected[inst] = d
            n_from_ranking += 1

    return selected


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
    mode                = 'incremental',   # 'incremental' (fast) | 'rebuild' (flag-controlled)
    # ── Cache-invalidation flags (event-named) ────────────────────────────────
    # TWO modes only:
    #   mode='incremental' -> these flags are INERT (ignored); trust all frozen
    #                         caches and recompute only the live last day.
    #   mode='rebuild'     -> these flags are HONOURED AS PASSED. Leave False to
    #                         reuse that cache; set True to recompute it. The normal
    #                         calibration sweep is mode='rebuild' with all flags
    #                         False (vary MVO params, reuse all upstream data). A
    #                         full burn is mode='rebuild' with all five set True.
    # Each flag is named after the change-event you would recognise making.
    universe_changed     = False,  # added/removed stocks -> composite+factors+X rebuilt
    factor_model_changed = False,  # HISTORICAL factor data changed (not new dates)
                                   #   -> composite+factors+X rebuilt
    weights_changed      = False,  # regenerated alpha IC weights_by_date -> composite
    regime_changed       = False,  # regime series changed              -> composite
    mr_changed           = False,  # MR params / MR scores changed       -> composite
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
    risk_aversion       = MVO_RISK_AVERSION,
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
    dd_hwm_window       = DD_HWM_WINDOW,   # rolling HWM window in months (0=lifetime)
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
            _min_start = Pxs_df.index[Pxs_df.index >= _min_start][0]                          if any(Pxs_df.index >= _min_start) else _min_start
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

    # ── 1. Build composite scores ─────────────────────────────────────────────
    # APPROACH A: composites are built for EVERY trading day (not just calc_dates),
    # because the dynamic turnover trigger redoes the full portfolio build daily to
    # assess turnover (gated to eligible days at run time). The path-dependent set
    # of eligible days isn't known in advance, so we build the daily panel up front.
    # The factor-score loaders run once over the daily index; in incremental mode
    # trading_days is a single day, so this carries no extra cost there. The static
    # strategies still rebalance only on calc_dates (calc_date_set below).
    print("  Building composite scores (daily panel for turnover assessment)...")
    universe    = get_universe(Pxs_df, sectors_s, ext_st)
    calc_dates  = pd.DatetimeIndex(generate_calc_dates(Pxs_df, step_days=rebal_freq))
    calc_dates  = calc_dates[calc_dates >= start_date]
    # Ensure last trading day is always included (hypothetical trade summary).
    if today_ts not in calc_dates:
        calc_dates_full = calc_dates.append(pd.DatetimeIndex([today_ts]))
    else:
        calc_dates_full = calc_dates

    # Daily composite calendar: every trading day in-sample (today_ts already in it).
    comp_calendar = pd.DatetimeIndex(trading_days)

    # ── Composite/factor-panel cache ──────────────────────────────────────────
    # Reuse frozen historical composites across runs (esp. calibration sweeps that
    # vary MVO params, which don't affect the composite). Cache key depends ONLY on
    # composite-relevant inputs (universe, model, factor-table coverage, MR, regime,
    # exclude_factors) -- NOT prices, NOT weights_by_date, NOT any MVO param. The
    # last trading day (today_ts) is NEVER frozen: it is always recomputed (intraday
    # prices + marginal weight still move), and every date strictly before it is
    # frozen-cacheable once it is no longer the tail.
    # ── Resolve cache-invalidation flags by mode (TWO modes only) ─────────────
    # mode='incremental' -> booleans are INERT (forced False): trust all frozen
    #                       history, recompute only the live last day (fast path).
    # mode='rebuild'     -> booleans are HONOURED AS PASSED: each flag left False
    #                       reuses that cache; each set True recomputes it. A full
    #                       burn = set all five True yourself (rare).
    if mode == 'incremental':
        _f_universe = _f_factor = _f_weights = _f_regime = _f_mr = False
    else:   # 'rebuild' (the boolean-controlled mode)
        _f_universe = universe_changed
        _f_factor   = factor_model_changed
        _f_weights  = weights_changed
        _f_regime   = regime_changed
        _f_mr       = mr_changed

    # Composite/factor panel is invalidated by ANY composite-relevant change-event.
    _force_comp_rebuild = (_f_universe or _f_factor or _f_weights
                           or _f_regime or _f_mr)
    # X / risk-model layer is invalidated only by universe or (historical) factor-model
    # changes; weights/regime/MR do not affect exposures.
    _force_x_rebuild    = (_f_universe or _f_factor)

    _comp_key = _make_composite_cache_key(universe, model_version)

    if _force_comp_rebuild:
        _clear_composite_cache(_comp_key)
        composite_by_date, score_dfs, _cached_set = {}, {}, set()
        _reasons = [n for n, f in [('universe', _f_universe), ('factor_model', _f_factor),
                    ('weights', _f_weights), ('regime', _f_regime), ('mr', _f_mr)] if f]
        print(f"  Composite cache cleared (key={_comp_key}; "
              f"changed: {', '.join(_reasons)})")
    else:
        composite_by_date, score_dfs, _cached_set = _load_composite_cache(
            _comp_key, before_dt=today_ts)
        if _cached_set:
            print(f"  Composite cache HIT (key={_comp_key}): "
                  f"{len(_cached_set)} frozen dates reused")
        else:
            print(f"  Composite cache MISS (key={_comp_key}): full rebuild "
                  f"(no frozen history for this key yet)")

    # Dates still needed = calendar minus frozen-cached, ALWAYS including today_ts.
    _missing = [d for d in comp_calendar if d not in _cached_set]
    if today_ts not in _missing:
        _missing.append(today_ts)        # never trust cache for the live last day
    _missing = pd.DatetimeIndex(sorted(set(_missing)))

    if len(_missing) > 0:
        print(f"  Building composite for {len(_missing)} date(s) "
              f"({_missing[0].date()} .. {_missing[-1].date()})"
              f"{' [daily full panel]' if not _cached_set else ' [gap + live day]'}...")
        _new_comp, _new_scores = _cb_build_composite_scores(
            universe        = universe,
            calc_dates      = _missing,
            Pxs_df          = Pxs_df,
            sectors_s       = sectors_s,
            weights_by_year = weights_by_year,
            regime_s        = regime_s,
            volumeTrd_df    = volumeTrd_df,
            model_version   = model_version,
            exclude_factors = ['OU'],
            weights_by_date = weights_by_date,
        )
        # Merge newly-built into the (possibly cached) panel.
        composite_by_date.update(_new_comp)
        for _fn, _fdf in _new_scores.items():
            if _fn in score_dfs and not score_dfs[_fn].empty:
                score_dfs[_fn] = pd.concat(
                    [score_dfs[_fn], _fdf]).sort_index()
                score_dfs[_fn] = score_dfs[_fn][~score_dfs[_fn].index.duplicated(keep='last')]
            else:
                score_dfs[_fn] = _fdf

        # Freeze: save only newly-built dates STRICTLY BEFORE today_ts.
        _to_freeze = [d for d in _missing if d < today_ts]
        if _to_freeze and not _force_comp_rebuild:
            _save_composite_cache(_comp_key, _to_freeze, composite_by_date, score_dfs)
            print(f"  Composite cache: froze {len(_to_freeze)} new date(s)")
        elif _to_freeze and _force_comp_rebuild:
            # On a forced rebuild we cleared the key; re-freeze the full history now.
            _save_composite_cache(_comp_key, _to_freeze, composite_by_date, score_dfs)
            print(f"  Composite cache: rebuilt + froze {len(_to_freeze)} date(s)")

    print(f"  Composite scores: {len(composite_by_date)} dates (daily)")
    # Diagnostic: print top-5 composite scores at first calc_date to detect lookahead
    _diag_dt = sorted(composite_by_date.keys())[0]
    _diag_s  = composite_by_date[_diag_dt].dropna().sort_values(ascending=False)
    print(f"\n  [DIAG] Composite at {_diag_dt.date()} top-5: "
          + "  ".join(f"{t}={v:.4f}" for t, v in _diag_s.head(5).items()))
    if _force_x_rebuild:
        _mb_clear_x_cache(model_version)
        _xr = 'universe_changed' if _f_universe else 'factor_model_changed'
        print(f"  X snapshot cache cleared ({_xr})")
    X_snapshots = dict(_mb_load_x_cache(model_version))
    snapshot_dates = sorted(X_snapshots.keys())
    print(f"  X snapshots: {len(X_snapshots)} already cached  "
          f"(new ones will be built on-demand during main loop)")
    lambda_dfs = _mb_load_lambda_dfs(model_version)
    print(f"  Lambda tables loaded: {len(lambda_dfs['scalar'])} scalar, "
          f"{len(lambda_dfs['macro'].columns)} macro, "
          f"{len(lambda_dfs['sector'].columns)} sector factors")
    if X_snapshots:
        _sample_snap = X_snapshots[snapshot_dates[-1]]
        _factor_names_check, _ = _mb_get_factor_names(model_version, lambda_dfs=lambda_dfs)
        _overlap = len([c for c in _sample_snap.columns if c in _factor_names_check])
        _pct     = _overlap / len(_factor_names_check) * 100 if _factor_names_check else 0
        print(f"  X snapshot validation: latest={snapshot_dates[-1].date()}  "
              f"stocks={len(_sample_snap)}  "
              f"factor overlap={_overlap}/{len(_factor_names_check)} ({_pct:.0f}%)"
              + ("  ✓" if _pct == 100 else "  ← STALE — run with factor_model_changed=True"))

    # ── Diagnostics: detect drift across runs ────────────────────────────────
    import hashlib as _hl

    # weights_by_date hash
    _wbd_repr = str([(str(k), sorted((f, round(v,6)) for f,v in d.items()))
                     for k, d in sorted(weights_by_date.items())])
    _wbd_hash = _hl.md5(_wbd_repr.encode()).hexdigest()[:12]
    print(f"  weights_by_date hash: {_wbd_hash}  ({len(weights_by_date)} dates)")

    # Residuals fingerprint — spot-check 3 stocks on 3 historical dates
    try:
        with ENGINE.connect() as conn:
            _resid_rows = pd.read_sql(text("""
                SELECT date, ticker, resid
                FROM v2_factor_residuals_quality
                WHERE date IN ('2019-06-04','2020-03-16','2022-06-15')
                  AND ticker IN ('ENPH','ZS','MDB')
                ORDER BY date, ticker
            """), conn)
        print(f"  Residuals fingerprint (v2_factor_residuals_quality):")
        for _, r in _resid_rows.iterrows():
            print(f"    {str(r['date'])[:10]}  {r['ticker']:<6}  {float(r['resid']):+.8f}")
    except Exception as e:
        print(f"  Residuals fingerprint: could not load ({e})")

    # Helper to get/build X snapshot for a given date on demand
    _x_snap_universe     = get_universe(Pxs_df, sectors_s, ext_st)
    _x_factor_names, _x_sec_cols = _mb_get_factor_names(model_version, lambda_dfs=lambda_dfs)
    _x_snap_dates_needed = set(_mb_month_end_dates(
        Pxs_df.index[Pxs_df.index >= start_date - pd.Timedelta(days=90)],
        Pxs_df.index[-1]))

    def _ensure_x_snapshot(dt):
        """Build X snapshot for the month-end prior to dt if not yet cached."""
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
            else:
                warnings.warn(f"  X snapshot FAILED for {x_dt.date()} "
                              f"— MVO will use empirical cov only")

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
    N_STRAT = 10
    S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART, S_DYN, S_HEDGE, S_DD, S_EXCL,         S_MVO_HEDGE = range(10)

    STRATEGY_LABELS = {
        S_BASE: 'Baseline', S_ALPHA: 'Alpha', S_MVO: 'MVO',
        S_HYB: 'Hybrid', S_SMART: 'Smart', S_DYN: 'Dynamic',
        S_HEDGE: 'Dyn+Hedge', S_DD: 'DD Policy', S_EXCL: 'Excl',
        S_MVO_HEDGE: 'MVO+Hedge',
    }

    # Static strategies rebalance on calc_dates; dynamic on triggers
    STATIC  = {S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART}
    # Dynamic strategies — hedge/DD/excl only active when hedge layer enabled
    DYNAMIC = {S_DYN, S_HEDGE, S_DD, S_EXCL, S_MVO_HEDGE} if hedge_enabled else {S_DYN}
    ACTIVE  = set(range(N_STRAT)) if hedge_enabled else               {S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART, S_DYN}
    print(f"  Strategies active: {len(ACTIVE)}  "
          f"({'hedge+DD+Excl enabled' if hedge_enabled else 'no hedge/DD/Excl'})")

    # ── Params hash & cache ───────────────────────────────────────────────────
    _ph_params = {
        'top_n': top_n, 'rebal_freq': rebal_freq, 'risk_aversion': risk_aversion,
        'min_cov_matrices': min_cov_matrices, 'prefilt_pct': prefilt_pct,
        'mom_filter': mom_filter, 'universe_mult': universe_mult,
        'advp_cap': advp_cap, 'max_weight': max_weight, 'min_weight': min_weight,
        'trading_cost_bps': trading_cost_bps, 'aum': aum,
        'dd_levels': str(dd_levels), 'dd_hwm_window': dd_hwm_window,
        'hedge_ratio': hedge_ratio, 'max_hedge': max_hedge,
        'mr_k': mr_k, 'mr_cap': mr_cap, 'model_version': model_version,
        'start_date': str(start_date.date()),
    }
    ph = _compute_params_hash(_ph_params)
    print(f"  Params hash: {ph}")

    _ensure_cache_tables()

    if mode == 'rebuild':
        _cache_clear(ph)
        print("  Cache cleared for rebuild.")

    # Try to load cache for incremental mode
    _cached_nav, _cache_last_dt, _cached_states =         _cache_load(ph, STRATEGY_LABELS) if mode == 'incremental'         else (None, None, None)

    _incremental = (mode == 'incremental' and _cached_nav is not None)

    if _incremental:
        # Always recompute the last date in Pxs_df (prices may be incomplete
        # earlier in the day — wipe and recalculate on every incremental run)
        _last_pxs_dt = Pxs_df.index[-1]
        if any(_last_pxs_dt in d for d in _cached_nav.values()):
            with ENGINE.begin() as _c:
                for _tbl in [MVO_NAV_CACHE_TBL, MVO_REBAL_CACHE_TBL]:
                    _c.execute(text(f"""
                        DELETE FROM {_tbl}
                        WHERE params_hash = :ph AND date = :dt
                    """), {'ph': ph, 'dt': _last_pxs_dt.date()})
            for s_idx in _cached_nav:
                _cached_nav[s_idx].pop(_last_pxs_dt, None)
            # Adjust last cached date after removal
            all_last = [max(d.keys()) for d in _cached_nav.values() if d]
            _cache_last_dt = min(all_last) if all_last else None

        # ── Roll back state to _cache_last_dt ────────────────────────────────
        # The state JSON was saved at end of last run (which may have included
        # _last_pxs_dt). Reset nav to the cached value at _cache_last_dt and
        # weights to the last rebalancing at or before _cache_last_dt so the
        # NAV computation for the rerun date starts from the correct base.
        if _cache_last_dt is not None:
            for s_idx in range(N_STRAT):
                lbl = STRATEGY_LABELS[s_idx]
                # Reset nav
                if (s_idx in _cached_nav
                        and _cache_last_dt in _cached_nav[s_idx]):
                    _cached_states[s_idx]['nav'] =                         _cached_nav[s_idx][_cache_last_dt]
                # Reset weights to last rebalancing at or before _cache_last_dt
                try:
                    with ENGINE.connect() as _c:
                        _wrows = _c.execute(text(f"""
                            SELECT ticker, weight, date FROM {MVO_REBAL_CACHE_TBL}
                            WHERE params_hash = :ph AND strategy = :st
                              AND date = (
                                  SELECT MAX(date) FROM {MVO_REBAL_CACHE_TBL}
                                  WHERE params_hash = :ph AND strategy = :st
                                    AND date <= :dt
                              )
                        """), {'ph': ph, 'st': lbl,
                               'dt': _cache_last_dt.date()}).fetchall()
                    if _wrows:
                        _cached_states[s_idx]['weights'] =                             pd.Series({r[0]: float(r[1]) for r in _wrows})
                        # Roll back last_rebal to match the weights date
                        _cached_states[s_idx]['last_rebal'] =                             pd.Timestamp(_wrows[0][2])
                except Exception:
                    pass

    if _incremental:
        _new_start = Pxs_df.index[Pxs_df.index > _cache_last_dt][0]                      if any(Pxs_df.index > _cache_last_dt) else None
        if _new_start is None:
            print(f"\n  Cache is up to date ({_cache_last_dt.date()}) — nothing to compute.")
            # Rebuild nav_series from cache and return
            _nav_series_out = {}
            for s_idx in range(N_STRAT):
                lbl = STRATEGY_LABELS[s_idx]
                if s_idx in _cached_nav:
                    _nav_series_out[s_idx] = pd.Series(_cached_nav[s_idx]).sort_index()
                else:
                    _nav_series_out[s_idx] = pd.Series(dtype=float)
            return _build_results(_nav_series_out, {s: _cached_states.get(s, {})
                                                    for s in range(N_STRAT)},
                                   STRATEGY_LABELS, ph)
        print(f"\n  Cache status  (params_hash={ph})")
        _nav_cnt  = sum(len(v) for v in _cached_nav.values())
        with ENGINE.connect() as _c:
            _reb_cnt = _c.execute(text(f"""
                SELECT COUNT(*) FROM {MVO_REBAL_CACHE_TBL}
                WHERE params_hash = :ph
            """), {'ph': ph}).scalar()
        print(f"  mvo_nav_cache    : {max(len(v) for v in _cached_nav.values())} dates"
              f"  (up to {_cache_last_dt.date()})")
        print(f"  mvo_rebal_cache  : {_reb_cnt} events across {N_STRAT} strategies")
        print(f"  Running incremental from {_new_start.date()}...")
    else:
        _new_start = None

    state = [{
        'nav':          1.0,
        'hwm':          1.0,
        'hwm_rolling':  1.0,          # rolling-window HWM — de-gross trigger only
        'dd':           0.0,
        'gross':        1.0,          # de-grossing level (DD policy)
        'trough':       1.0,          # real nav trough (de-grossed)
        'theo_nav':     1.0,          # theoretical nav at full exposure (no de-gross)
        'theo_trough':  1.0,          # trough of theo_nav — basis for re-entry trigger
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
    gross_records = {s: {} for s in range(N_STRAT)}

    # ── Restore from cache in incremental mode ────────────────────────────────
    if _incremental:
        for s_idx in range(N_STRAT):
            if s_idx in _cached_nav:
                nav_records[s_idx].update(_cached_nav[s_idx])
            if s_idx in _cached_states:
                cs = _cached_states[s_idx]
                for k, v in cs.items():
                    state[s_idx][k] = v
        trading_days = Pxs_df.index[Pxs_df.index >= _new_start]
        print(f"  New trading days to compute: {len(trading_days)}")
    # accumulate new nav/rebal for later cache save
    _new_nav    = {s: {} for s in range(N_STRAT)}
    _new_rebal  = []   # list of (s_idx, dt, weights, gross)   # {date: gross_exposure}

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

    def _build_overlay(w_tier1, dt, non_core_cands, comp_scores, strat_idx):
        """Build tier 2 overlay. Returns Series {ticker: equal_weight_fraction}.

        Overlay ADVP capacity is sized against the AUM of the STRATEGY being
        built (strat_idx), identical to _apply_advp — each strategy has its own
        capital, and de-grossed strategies (DD) have materially less, so the
        liquidity gate must use that strategy's nav*gross, not a shared proxy.
        """
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
        # Valid price filter: exclude stocks with NaN price at current date
        valid_px_dt = set(Pxs_df.loc[dt].dropna().index) if dt in Pxs_df.index else set(pxs_cols)

        cands = [t for t in non_core_cands
                 if t not in w_tier1.index and t in _universe_set
                 and t in pxs_cols and t in valid_px_dt]
        if volumeRaw_df is not None and not volumeRaw_df.empty:
            past_vol = [d for d in volumeRaw_df.index if d <= dt]
            if len(past_vol) >= 20:
                vol_window = past_vol[-20:]
                # AUM of the strategy being built (same basis as _apply_advp):
                # aum * this strategy's nav * its current gross (de-gross aware).
                cur_aum = aum * state[strat_idx]['nav'] * state[strat_idx].get('gross', 1.0)
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
        survivors = corr_s.nsmallest(min(n_keep, len(corr_s))).index.tolist()                     if len(corr_s) >= n_ov else corr_s.index.tolist()
        if len(survivors) < n_ov: return pd.Series(dtype=float)

        # Step 2: overlay score components
        c1 = (1 - corr_s.reindex(survivors).fillna(0))
        c2 = _compute_6m1_scores(survivors, dt).fillna(0)
        avail = [d for d in composite_by_date.keys() if d <= dt]
        c3 = composite_by_date[avail[-1]].reindex(survivors).fillna(0)              if avail else pd.Series(0.0, index=survivors)

        # Z-score and combine equally
        ov_score = (_zs_overlay(c1).reindex(survivors).fillna(0) +
                    _zs_overlay(c2).reindex(survivors).fillna(0) +
                    _zs_overlay(c3).reindex(survivors).fillna(0)) / 3.0
        top_ov = ov_score.nlargest(n_ov).index.tolist()
        return pd.Series(1.0 / n_ov, index=top_ov)  # equal weights, sum=1 (pre-scaling)

    def _apply_tier2(w_tier1, dt, non_core_cands, comp_scores, strat_idx,
                     force_flat_alloc=False):
        """Combine tier 1 and overlay into final portfolio summing to 1.0.
        Returns (combined_weights, overlay_tickers_set).
        Overlay weight per stock is always fixed at (1-TIERONE_ALLOC)/(T-N),
        regardless of how many overlay slots are available.
        strat_idx identifies the strategy whose AUM governs overlay ADVP capacity.
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

        ov = _build_overlay(w_tier1, dt, non_core_cands, comp_scores, strat_idx)
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
        with _SuppressOutput():
            w, diag = _mb_solve_mvo(
                dt=dt, candidates=cands.head(n_cands).index.tolist(),
                composite_scores=comp_scores, Pxs_df=Pxs_df,
                sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                model_version=model_version,
                pca_var_threshold=pca_var_threshold,
                ic=ic, max_weight=max_weight, min_weight=min_weight,
                zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                X_snapshots=X_snapshots if snapshot_dates else None,
                snapshot_dates=snapshot_dates if snapshot_dates else None,
                top_n=top_n, min_cov_matrices=min_cov_matrices,
                lambda_dfs=lambda_dfs,
            )
        if not w.empty:
            return w[w > 1e-6], diag
        return w, {}

    # ── Helper: apply ADVP filter with correct strategy AUM ──────────────────
    def _apply_advp(w, cands, dt, strat_idx, target_n=None, overlay_stocks=None):
        if volumeRaw_df is None or w.empty: return w, set()
        cur_aum = aum * state[strat_idx]['nav'] * state[strat_idx]['gross']
        if cur_aum <= 0: return w, set()
        # Replacement universe = composite at the date being processed (today),
        # falling back to most recent available <= dt. Uses the dt PARAMETER so it
        # is correct for both static (dt==calc_date) and dynamic fresh-daily builds.
        _full_u = composite_by_date.get(dt)
        if _full_u is None:
            _av = [d for d in composite_by_date.keys() if d <= dt]
            _full_u = composite_by_date.get(max(_av)) if _av else None
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
        gross = state[s_idx].get('gross', 1.0)
        to    = _turnover(w_new, w_old)
        c     = to * trading_cost_bps / 10000 * gross   # scale by actual exposure
        state[s_idx]['nav']  *= (1 - c)                 # deduct from NAV
        state[s_idx]['costs'] += c
        if dt is not None:
            yr = dt.year
            state[s_idx]['costs_by_year'][yr] =                 state[s_idx]['costs_by_year'].get(yr, 0.0) + c
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
        """Update DD state — runs daily after NAV update."""
        st  = state[s_idx]
        nav = st['nav']

        # Update theo_nav daily when de-grossed — tracks what full-exposure
        # portfolio would have returned (used only for re-entry assessment)
        if st['dd_active'] and prev_dt is not None:
            w   = st['weights']
            tks = [t for t in w.index if t in pxs_cols]
            if tks and prev_dt in Pxs_df.index and dt in Pxs_df.index:
                full_ret = (w.reindex(tks).fillna(0) *
                            (Pxs_df.loc[dt, tks] /
                             Pxs_df.loc[prev_dt, tks] - 1).fillna(0)).sum()
                st['theo_nav']    *= (1 + full_ret)
                st['theo_trough']  = min(st['theo_trough'], st['theo_nav'])

        # ── Rolling HWM (de-gross trigger only) ──────────────────────────────
        # Updated regardless of dd_active — always tracks recent peak
        if dd_hwm_window > 0:
            cutoff   = dt - pd.DateOffset(months=dd_hwm_window)
            past_nav = {d: v for d, v in nav_records[s_idx].items()
                        if d >= cutoff}
            st['hwm_rolling'] = max(past_nav.values()) if past_nav else nav
        else:
            st['hwm_rolling'] = st['hwm']   # fallback: use lifetime HWM

        # dd_rolling drives de-gross decisions; st['dd'] (lifetime) drives regime
        dd_rolling = nav / st['hwm_rolling'] - 1

        # ── De-gross: check next sequential level ────────────────────────────
        next_lvl = st['dd_level'] + 1
        if next_lvl < len(dd_levels):
            dd_thresh, cut_frac = dd_levels[next_lvl]
            if dd_rolling <= -dd_thresh:
                old_gross           = st['gross']
                st['gross']        *= (1 - cut_frac)
                st['dd_active']     = True
                st['trough']        = nav
                st['theo_nav']      = nav
                st['theo_trough']   = nav
                st['reentry_count'] = 0
                st['dd_level']      = next_lvl
                st['regime']        = dd_level_regime[next_lvl]
                st['dd_regime_forced'] = True
                st['dd_log'].append({
                    'dt': dt, 'event': f'DE-GROSS lv{next_lvl+1}',
                    'dd': dd_rolling,
                    'exp_from': old_gross, 'exp_to': st['gross'],
                })

        # ── Re-entry: theo recovery >= threshold for N consecutive days ──────
        if st['dd_active'] and st['gross'] < 1.0:
            st['trough']  = min(st['trough'], nav)
            theo_recovery = st['theo_nav'] / st['theo_trough'] - 1
            if theo_recovery >= dd_reentry_pct:
                st['reentry_count'] += 1
            else:
                st['reentry_count']  = 0
            if st['reentry_count'] >= dd_reentry_confirm:
                old_gross           = st['gross']
                st['gross']         = 1.0
                st['dd_active']     = False
                st['dd_level']      = -1
                st['reentry_count'] = 0
                st['hwm']           = nav   # reset HWM to current NAV on re-entry
                st['trough']        = nav
                st['theo_trough']   = nav
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

        # Macro signal trigger: signal is observed at dt's close (real-time, ~30min
        # before close) and acted on SAME DAY (d0=dt, entry_px=Px(dt)); the first
        # hedge P&L is naturally Px(dt+1)/Px(dt)-1, applied by _update_nav on dt+1.
        # Reading dt's OWN signal (not the prior day's) is correct and not lookahead:
        # the signal uses only information available at dt's close. The previous code
        # read prev_sig_dates[-1] (yesterday's signal), which added an unwanted extra
        # day -- a signal firing at t only opened the hedge at t+1, first P&L t+2 --
        # and that one-day miss is enough to flip the hedge's contribution.
        # hedge_multi structure: {'results': {ticker: {'signal_df': ...}}, ...}
        _hedge_results = hedge_multi.get('results', hedge_multi) if hedge_multi else {}
        trigger_on = False
        for ta in hedge_trigger_assets:
            if ta in _hedge_results:
                sig_df = _hedge_results[ta].get('signal_df', pd.DataFrame())
                if not sig_df.empty and dt in sig_df.index:
                    if sig_df.loc[dt, 'signal'] == 1:
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

    calc_date_set = set(calc_dates)   # only original calc_dates drive static rebalancing

    # Hoisted ADVP+tier2 helper at run_backtest scope (explicit dt/cands), so the
    # dynamic block can call it on NON-calc-dates where the static branch's local
    # closure of the same name was never defined. Returns (weights, advp_affected,
    # overlay_set) to match the static-branch usage.
    def _advp_then_tier2_loop(w_t1, s_idx, dt_, cands_, non_core_cands, comp_s):
        t_n = len(w_t1) if len(w_t1) > top_n else top_n
        w_out, aff = _apply_advp(w_t1, cands_, dt_, s_idx, target_n=t_n)
        if tier2_active:
            nc = [t for t in non_core_cands
                  if t not in w_out.index and t in pxs_cols]
            w_out, ov_set = _apply_tier2(w_out, dt_, nc, comp_s, s_idx)
        else:
            ov_set = set()
        return w_out, aff, ov_set

    for day_idx, dt in enumerate(trading_days):
        is_calc_date = dt in calc_date_set
        # In incremental mode on day_idx=0, prev_dt must come from full
        # Pxs_df history — not trading_days (which only has new dates)
        if day_idx > 0:
            prev_dt = trading_days[day_idx - 1]
        else:
            _all_px_before = Pxs_df.index[Pxs_df.index < dt]
            prev_dt = _all_px_before[-1] if len(_all_px_before) > 0 else None

        # -- Update days_held for dynamic strategies --------------------------
        for s_idx in DYNAMIC:
            if state[s_idx]['last_rebal'] is not None:
                state[s_idx]['days_held'] += 1

        # ══════════════════════════════════════════════════════════════════════
        # MARK-TO-MARKET + REGIME/DD ASSESSMENT  (must run BEFORE any rebalance)
        # ══════════════════════════════════════════════════════════════════════
        # The rebalance need is assessed using the last available prices (today's
        # close). So today's NAV/drawdown/regime, and any DD de-gross breach, are
        # established HERE, before the static/dynamic rebalance sections consume
        # them. The gross cut from a de-gross still takes effect next day (today's
        # NAV is already marked just below); the rebalance a breach triggers deploys
        # at today's close and earns its first P&L on t+1.
        #   - regime change (construction-mode switch) -> respects min-hold
        #   - DD de-gross (risk reduction) -> sets dd_regime_forced, overrides min-hold
        if prev_dt is not None:
            for s_idx in range(N_STRAT):
                gross = state[s_idx].get('gross', 1.0)
                gross_records[s_idx][dt] = gross
                _update_nav(s_idx, prev_dt, dt, gross=gross)
        else:
            for s_idx in range(N_STRAT):
                nav_records[s_idx][dt]   = state[s_idx]['nav']
                gross_records[s_idx][dt] = state[s_idx].get('gross', 1.0)

        # Record new NAV for cache (every day, every strategy)
        for s_idx in range(N_STRAT):
            _new_nav[s_idx][dt] = state[s_idx]['nav']

        # Update regime for all strategies from TODAY's drawdown
        for s_idx in range(N_STRAT):
            state[s_idx]['regime'] = _get_regime(state[s_idx]['dd'])

        # Apply DD de-gross DETECTION from today's dd (gross cut affects t+1 NAV;
        # dd_regime_forced flag is now visible to today's rebalance trigger).
        for s_idx in [S_DD, S_EXCL]:
            _apply_dd(s_idx, dt)

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
                            idio_row = score_dfs['Idio_Mom'].loc[avail[-1]]                                           .reindex(cands.index).fillna(0)
                    if 'Mom_12M1' in score_dfs and score_dfs['Mom_12M1'] is not None:
                        avail = [d for d in score_dfs['Mom_12M1'].index if d <= dt]
                        if avail:
                            m12_row = score_dfs['Mom_12M1'].loc[avail[-1]]                                          .reindex(cands.index).fillna(0)
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
                    w_mvo, _mvo_diag = _build_mvo(dt, cands_mvo, comp)
                    if w_mvo.empty:
                        w_mvo = w_alpha.copy()
                        print(f"  [WARN {dt.date()}] MVO returned empty — falling back to Pure Alpha")
                    else:
                        _cs = _mvo_diag.get('cov_sums', {})
                        _fd = _mvo_diag.get('f_diag', {})
                        if _cs:
                            _f_status = (f"F-rows={_fd.get('f_shape',(0,0))[0]}  "
                                         f"f_mean_var={_fd.get('f_mean_var',0):.2e}  "
                                         f"{'✓ active' if _fd.get('f_active') else '✗ fallback→Emp'}")
                            print(f"  [COV {dt.date()}] "
                                  f"Emp={_cs['Emp']:.4f}  "
                                  f"LW={_cs['LW']:.4f}  "
                                  f"Factor={_cs['Factor']:.4f}  "
                                  f"PCA={_cs['PCA']:.4f}  |  "
                                  f"{_f_status}")
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
                    w_mvo_p, _ = _build_mvo(dt, cands_pen_m, comp_pen)
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
                _new_rebal.append((S_BASE, dt, w_bl.copy(), 1.0))

                # Helper: run ADVP on tier1, then append overlay
                def _advp_then_tier2(w_t1, s_idx, non_core_cands, comp_s):
                    t_n = len(w_t1) if len(w_t1) > top_n else top_n
                    w_out, aff = _apply_advp(w_t1, cands, dt, s_idx,
                                             target_n=t_n)
                    if tier2_active:
                        nc = [t for t in non_core_cands
                              if t not in w_out.index and t in pxs_cols]
                        w_out, ov_set = _apply_tier2(w_out, dt, nc, comp_s, s_idx)
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
                _new_rebal.append((S_ALPHA, dt, w_a1.copy(), 1.0))

                # -- MVO (S2) -------------------------------------------------
                w_m2, _, _ov_m2 = _advp_then_tier2(w_mvo, S_MVO, non_core, comp)
                _record_cost(S_MVO, w_m2, state[S_MVO]['weights'], dt)
                state[S_MVO]['rebal_log'].append({'dt': dt, 'w': w_m2.copy()})
                state[S_MVO]['weights'] = w_m2
                state[S_MVO]['weights_by_date'][dt] = w_m2.copy()
                state[S_MVO]['last_rebal'] = dt
                _save_exposure_row(dt, 'MVO',
                    _get_factor_exposures(w_m2, dt, _factor_scores))
                _new_rebal.append((S_MVO, dt, w_m2.copy(), 1.0))
                w_h3, _, _ov_h3 = _advp_then_tier2(w_hyb, S_HYB, non_core,
                                                     comp)
                _record_cost(S_HYB, w_h3, state[S_HYB]['weights'], dt)
                state[S_HYB]['rebal_log'].append({'dt': dt, 'w': w_h3.copy()})
                state[S_HYB]['weights'] = w_h3
                state[S_HYB]['weights_by_date'][dt] = w_h3.copy()
                state[S_HYB]['last_rebal'] = dt
                _save_exposure_row(dt, 'Hybrid',
                    _get_factor_exposures(w_h3, dt, _factor_scores))
                _new_rebal.append((S_HYB, dt, w_h3.copy(), 1.0))

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
                _new_rebal.append((S_SMART, dt, w_s4.copy(), 1.0))

        # -- Dynamic strategies (S5-S8): redo full build daily for TO assessment --
        # APPROACH A: each eligible day, rebuild the candidate portfolio from TODAY's
        # composite scores + TODAY's X snapshot (not last_calc_dt's), then assess
        # turnover. A rebalance can only fire outside the min-hold window (or on a
        # DD-forced regime override, which depends solely on the live book). Inside
        # min-hold, turnover is irrelevant -- it cannot trigger anything -- so we skip
        # the (expensive) rebuild entirely. This is result-neutral vs. "rebuild every
        # day" because the skipped days' turnover would have been discarded anyway,
        # while making the rebuild use fresh data on the days it actually matters.
        _dyn_today = composite_by_date.get(dt)
        if _dyn_today is not None and not _dyn_today.empty:
            # Which dynamic strategies are ELIGIBLE to rebalance today?
            #   eligible = never rebalanced yet, OR min-hold elapsed, OR DD-forced.
            _dyn_list = [S_DYN, S_HEDGE, S_DD, S_EXCL, S_MVO_HEDGE]
            def _eligible(s_idx):
                st = state[s_idx]
                if st['last_rebal'] is None:
                    return True
                if st.get('dd_regime_forced', False):
                    return True
                return (dt - st['last_rebal']).days >= min_hold_days
            _any_eligible = any(_eligible(s) for s in _dyn_list)

            if _any_eligible:
                # Build shared blocks ONCE from today's scores + today's X.
                _ensure_x_snapshot(dt)   # today's exposures (only on eligible days)
                comp_dyn  = _dyn_today.dropna()
                comp_dyn  = comp_dyn.loc[[t for t in comp_dyn.index if t in pxs_cols]]
                n_pf_dyn  = max(n_cands, int(np.ceil(len(comp_dyn) * prefilt_pct)))
                cands_dyn = comp_dyn.nlargest(min(n_pf_dyn, len(comp_dyn)))

                # Momentum-filtered alpha candidate pool (mirror static branch)
                _n_mom = mom_filter * top_n
                if _n_mom < len(cands_dyn):
                    _idio = pd.Series(dtype=float); _m12 = pd.Series(dtype=float)
                    if score_dfs.get('Idio_Mom') is not None:
                        _av = [d for d in score_dfs['Idio_Mom'].index if d <= dt]
                        if _av:
                            _idio = score_dfs['Idio_Mom'].loc[_av[-1]].reindex(cands_dyn.index).fillna(0)
                    if score_dfs.get('Mom_12M1') is not None:
                        _av = [d for d in score_dfs['Mom_12M1'].index if d <= dt]
                        if _av:
                            _m12 = score_dfs['Mom_12M1'].loc[_av[-1]].reindex(cands_dyn.index).fillna(0)
                    _topmom    = _idio.add(_m12, fill_value=0).nlargest(min(_n_mom, len(cands_dyn))).index
                    cands_a_dyn = cands_dyn.reindex(_topmom).dropna()
                else:
                    cands_a_dyn = cands_dyn
                cands_m_dyn = cands_dyn.nlargest(min(n_cands, len(cands_dyn)))

                # Unpenalised blocks
                _wa = _build_alpha(dt, cands_a_dyn)
                _wm, _ = _build_mvo(dt, cands_m_dyn, comp_dyn)
                if _wm.empty: _wm = _wa.copy()
                _at = list(set(_wa.index) | set(_wm.index))
                _wh = (_wa.reindex(_at).fillna(0) + _wm.reindex(_at).fillna(0)) / 2
                _wh = _wh[_wh > 0]
                if _wh.sum() > 0: _wh /= _wh.sum()

                # Penalised blocks (Excl strategy) -- only if S_EXCL eligible
                _comp_pen = _wa_p = _wm_p = _wh_p = None
                if _eligible(S_EXCL):
                    _comp_pen = _penalise(comp_dyn, dt)
                    _ca_p = _comp_pen.reindex(cands_a_dyn.index).dropna().nlargest(
                                min(_n_mom, len(cands_a_dyn)))
                    _cm_p = _comp_pen.reindex(cands_m_dyn.index).dropna().nlargest(
                                min(n_cands, len(cands_m_dyn)))
                    _wa_p = _build_alpha(dt, _ca_p)
                    _wm_p, _ = _build_mvo(dt, _cm_p, _comp_pen)
                    if _wm_p.empty: _wm_p = _wa_p.copy()
                    _atp = list(set(_wa_p.index) | set(_wm_p.index))
                    _wh_p = (_wa_p.reindex(_atp).fillna(0) + _wm_p.reindex(_atp).fillna(0)) / 2
                    _wh_p = _wh_p[_wh_p > 0]
                    if _wh_p.sum() > 0: _wh_p /= _wh_p.sum()

                for s_idx in _dyn_list:
                    if not _eligible(s_idx):
                        continue   # inside min-hold, no DD force -> cannot rebalance
                    st   = state[s_idx]
                    reg  = st['regime']

                    # Select regime portfolio from today's freshly-built blocks
                    if s_idx == S_EXCL:
                        comp_s  = _comp_pen
                        w_hyp = _regime_weights(dt, reg, {
                            'alpha': _wa_p, 'mvo': _wm_p, 'hybrid': _wh_p})
                    elif s_idx == S_MVO_HEDGE:
                        comp_s = comp_dyn
                        w_hyp  = _wm.copy()
                    else:
                        comp_s = comp_dyn
                        w_hyp = _regime_weights(dt, reg, {
                            'alpha': _wa, 'mvo': _wm, 'hybrid': _wh})

                    if w_hyp.empty: continue

                    # ADVP on tier1, then append overlay (today's prices/AUM).
                    # Uses the hoisted helper (explicit dt/cands) so it works on
                    # non-calc-dates where the static-branch closure is undefined.
                    #   cands_        = full candidate pool for ADVP replacement
                    #   non_core_cands= overlay-eligible names (tradable subset)
                    w_hyp, _advp_affected, _ov_dyn = _advp_then_tier2_loop(
                        w_hyp, s_idx, dt,
                        cands_dyn,
                        [t for t in cands_dyn.index if t in pxs_cols],
                        comp_s)

                    # Check rebalance trigger
                    _did_rebal = False
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
                        _new_rebal.append((s_idx, dt, w_hyp.copy(),
                                           st.get('gross', 1.0)))
                        _did_rebal = True

                    # -- Per-rebalance print for the TRACKED strategy (DD+Excl) ----
                    # Fires ONLY on a true rebalance commit (not on hold days), so the
                    # runtime log is an honest record of actual trades. DD+Excl is the
                    # MOST feature-complete strategy: dynamic rebalancing calendar +
                    # portfolio hedges + drawdown de-grossing + momentum-exclusion penalty.
                    if s_idx == S_EXCL and _did_rebal:
                        _cur_aum  = aum * st['nav'] * st.get('gross', 1.0)
                        _rebal_n  = len(st['rebal_log'])
                        _eff_n    = 1.0 / (w_hyp**2).sum() if not w_hyp.empty else 0
                        _pv       = _portfolio_vol(w_hyp, Pxs_df, dt)
                        _dd_str   = f"dd={st['dd']*100:+.1f}%"
                        _gr_str   = f"gross={st['gross']*100:.0f}%"
                        print(f"\n  -- {dt.date()}  [DD+Excl #{_rebal_n}]  "
                              f"active_u={len(_get_active_universe(universe, Pxs_df, dt))}  "
                              f"regime={reg}  {_dd_str}  {_gr_str}  "
                              f"n={len(w_hyp)}  eff_N={_eff_n:.1f}  "
                              f"TO={to*100:.0f}%  "
                              f"AUM=${_cur_aum/1e6:.1f}M  "
                              f"port_vol={_pv*100:.1f}%", flush=True)
                        # MR top scores
                        _mr_dt = _load_momentum_exclusions(dt)
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
                                    val = fdf.loc[avail[-1], _tkr]                                           if avail and _tkr in fdf.columns else np.nan
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

        # -- Update hedge for S6/S7/S8 (uses today's deployed weights) --------
        # Runs AFTER the rebalance sections so it hedges the portfolio actually
        # held today; hedge P&L is applied on t+1 like the rest of the book.
        for s_idx in [S_HEDGE, S_DD, S_EXCL, S_MVO_HEDGE]:
            if prev_dt is not None and not state[s_idx]['weights'].empty:
                _update_hedge(s_idx, dt, state[s_idx]['weights'],
                              state[s_idx]['nav'])

        # -- Progress ----------------------------------------------------------
        if (day_idx + 1) % 50 == 0 or day_idx == len(trading_days) - 1:
            print(f"  [{day_idx+1}/{len(trading_days)}] {dt.date()}  "
                  f"nav_dyn={state[S_DYN]['nav']:.3f}  "
                  f"nav_dd={state[S_DD]['nav']:.3f}  "
                  f"AUM_dyn=${aum*state[S_DYN]['nav']/1e6:.0f}M",
                  end='\r', flush=True)

    print()  # newline after progress

    # ── Save cache ────────────────────────────────────────────────────────────
    print("  Saving NAV and rebalancing cache...")
    for s_idx in range(N_STRAT):
        lbl = STRATEGY_LABELS[s_idx]
        _cache_save_nav(ph, lbl, _new_nav[s_idx])
        _cache_save_state(ph, lbl, state[s_idx],
                          trading_days[-1] if len(trading_days) else _cache_last_dt)
    for s_idx, dt_r, w_r, gf in _new_rebal:
        lbl = STRATEGY_LABELS[s_idx]
        _cache_save_rebal(ph, lbl, dt_r, w_r, gf)
    print(f"  Cache updated: {len(_new_nav[S_ALPHA])} new NAV dates  "
          f"| {len(_new_rebal)} new rebal events")

    if _incremental:
        print(f"\n  Incremental run complete — new dates: {len(trading_days)}")
        print(f"  New rebalancing events: {len(_new_rebal)}")
        for s_idx, dt_r, w_r, gf in _new_rebal:
            st   = state[s_idx]
            reg  = st.get('regime', 'alpha')
            mode_str = f"  mode={reg}" if s_idx in DYNAMIC else ""
            dd_str   = (f"  gross={gf:.0%}" if gf < 1.0 else "")
            print(f"    [{STRATEGY_LABELS[s_idx]}] {dt_r.date()}  "
                  f"n={len(w_r)}{mode_str}{dd_str}")
        # Show current regime for all dynamic strategies
        print(f"\n  Current regime (dynamic strategies):")
        for s_idx in sorted(DYNAMIC):
            st   = state[s_idx]
            reg  = st.get('regime', 'alpha')
            gross = st.get('gross', 1.0)
            dd   = st.get('dd', 0.0)
            print(f"    {STRATEGY_LABELS[s_idx]:<14} mode={reg:<8}  "
                  f"gross={gross:.0%}  dd={dd*100:+.1f}%")
    def _build_results(ns, st, labels, params_h):
        """Build the standard results dict from nav_series and state."""
        return {
            'nav_baseline':               ns[S_BASE],
            'nav_alpha':                  ns[S_ALPHA],
            'nav_mvo':                    ns[S_MVO],
            'nav_hybrid':                 ns[S_HYB],
            'nav_smart':                  ns[S_SMART],
            'nav_dynamic':                ns[S_DYN],
            'nav_dyn_hedged':             ns[S_HEDGE],
            'nav_dd':                     ns[S_DD],
            'nav_dd_excl':                ns[S_EXCL],
            'nav_mvo_hedge':              ns[S_MVO_HEDGE],
            'nav_series':                 ns,
            'labels':                     labels,
            'state':                      st,
            'alpha_weights_by_date':      st[S_ALPHA].get('weights_by_date', {}),
            'mvo_weights_by_date':        st[S_MVO].get('weights_by_date', {}),
            'hybrid_weights_by_date':     st[S_HYB].get('weights_by_date', {}),
            'smart_weights_by_date':      st[S_SMART].get('weights_by_date', {}),
            'dyn_weights_by_date':        st[S_DYN].get('weights_by_date', {}),
            'hedge_weights_by_date':      st[S_HEDGE].get('weights_by_date', {}),
            'dd_weights_by_date':         st[S_DD].get('weights_by_date', {}),
            'excl_weights_by_date':       st[S_EXCL].get('weights_by_date', {}),
            'baseline_weights_by_date':   st[S_BASE].get('weights_by_date', {}),
            'mvo_hedge_weights_by_date':  st[S_MVO_HEDGE].get('weights_by_date', {}),
            'dd_log':                     st[S_DD].get('dd_log', []),
            'dd_log_excl':                st[S_EXCL].get('dd_log', []),
            'params_hash':                params_h,
            'composite_scores':           composite_by_date,
            'score_dfs':                  score_dfs,
            'ic_weights_by_date':         weights_by_date,
            'universe':                   universe,
        }

    def _to_nav_series(s_idx):
        d = nav_records[s_idx]
        if not d: return pd.Series(dtype=float)
        return pd.Series(d).sort_index()

    nav_series = [_to_nav_series(s) for s in range(N_STRAT)]
    labels     = ['Baseline', 'Pure Alpha', 'MVO', 'Hybrid', 'Smart Hybrid',
                  'Dynamic', 'Dyn+Hedge', 'DD Policy', 'Excl', 'MVO+Hedge']

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
        (S_MVO_HEDGE, 'MVO+Hedge'),
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

    # ── Average holding period per year (rows) × dynamic strategy (cols) ───────
    # Holding period = trading days between consecutive rebalances. Sourced from
    # each strategy's actual rebalance dates (weights_by_date keys). Note: after an
    # INCREMENTAL run the rebal history is minimal (only a full 'rebuild' carries
    # the complete sequence), so this table is complete on rebuild runs.
    _dyn_cols = [(S_DYN, 'Dynamic'), (S_HEDGE, 'Dyn+Hdg'), (S_DD, 'DD Pol'),
                 (S_EXCL, 'Excl'), (S_MVO_HEDGE, 'MVO+Hdg')]
    # trading-day index position lookup for converting date gaps -> trading days
    _td_index = {d: i for i, d in enumerate(trading_days)}

    def _holding_periods_by_year(s_idx):
        """{year: mean trading-day holding period} from the strategy's rebalances."""
        rdates = sorted(state[s_idx].get('weights_by_date', {}).keys())
        by_year = {}
        for a, b in zip(rdates[:-1], rdates[1:]):
            ia, ib = _td_index.get(a), _td_index.get(b)
            if ia is None or ib is None:
                continue
            # attribute the holding span to the year it STARTED in
            by_year.setdefault(a.year, []).append(ib - ia)
        return {y: (sum(v) / len(v)) for y, v in by_year.items() if v}

    _hp = {s_idx: _holding_periods_by_year(s_idx) for s_idx, _ in _dyn_cols}
    _hp_years = sorted(set().union(*[set(d.keys()) for d in _hp.values()])) if _hp else []

    if _hp_years:
        print(f"\n  Average holding period — trading days "
              f"(year × dynamic strategy)")
        _hpw = 9
        _hdr = f"  {'Year':<6}" + "".join(f"  {lbl:>{_hpw}}" for _, lbl in _dyn_cols)
        print(_hdr)
        print("  " + "-" * (6 + len(_dyn_cols) * (_hpw + 2)))
        for yr in _hp_years:
            row = f"  {yr:<6}"
            for s_idx, _ in _dyn_cols:
                v = _hp[s_idx].get(yr)
                row += f"  {v:>{_hpw}.1f}" if v is not None else f"  {'—':>{_hpw}}"
            print(row)
        # Overall (all years pooled) row
        print("  " + "-" * (6 + len(_dyn_cols) * (_hpw + 2)))
        row = f"  {'All':<6}"
        for s_idx, _ in _dyn_cols:
            rdates = sorted(state[s_idx].get('weights_by_date', {}).keys())
            gaps = [(_td_index[b] - _td_index[a])
                    for a, b in zip(rdates[:-1], rdates[1:])
                    if a in _td_index and b in _td_index]
            row += f"  {(sum(gaps)/len(gaps)):>{_hpw}.1f}" if gaps else f"  {'—':>{_hpw}}"
        print(row)

    # ── Regime map for the TRACKED strategy (S_EXCL = DD+Excl) ─────────────────
    # One char per rebalance, in chronological order, grouped by year. Shows which
    # portfolio regime governed each actual rebalance:  A=alpha  H=hybrid  M=MVO.
    _REGIME_CHAR = {'alpha': 'A', 'hybrid': 'H', 'mvo': 'M'}
    _track_idx, _track_lbl = S_EXCL, 'DD+Excl'
    _rlog = state[_track_idx].get('rebal_log', [])
    if _rlog:
        _by_year = {}
        for ev in _rlog:
            y = pd.Timestamp(ev['dt']).year
            _by_year.setdefault(y, []).append(
                _REGIME_CHAR.get(ev.get('regime', 'alpha'), '?'))
        print(f"\n  Rebalance regime map — {_track_lbl} (tracked strategy)")
        print(f"    A=alpha  H=hybrid  M=MVO   (one char per rebalance, chronological)")
        for yr in sorted(_by_year.keys()):
            print(f"  {yr:<6} {''.join(_by_year[yr])}")

    # ── 9. Live portfolio display ─────────────────────────────────────────────
    today_ts = trading_days[-1]

    # ── 9a. THEORETICAL fresh portfolios for today_ts ────────────────────────
    # Builds the would-be portfolio for every strategy AS IF today were a
    # rebalance date, using today's composite scores and each strategy's CURRENT
    # (drawdown-implied) regime / gross / hedge state. Display-only: writes to a
    # local dict, never mutates state / NAV / cache / rebal log. Reuses the exact
    # construction helpers the main loop uses, so the result is bit-identical to
    # what a real rebalance on today_ts would produce. The trade summary uses these
    # as the "fresh" leg for strategies that did NOT rebalance today (THEORETICAL);
    # strategies that DID rebalance today show their actually-executed book (REAL).
    _theo_fresh = {}   # {label: would-be portfolio Series, hedge-inclusive basis}

    def _advp_then_tier2_at(w_t1, s_idx, dt_, cands_, non_core_cands, comp_s):
        """Replica of the loop's _advp_then_tier2, with explicit dt/cands."""
        t_n = len(w_t1) if len(w_t1) > top_n else top_n
        w_out, _aff = _apply_advp(w_t1, cands_, dt_, s_idx, target_n=t_n)
        if tier2_active:
            nc = [t for t in non_core_cands
                  if t not in w_out.index and t in pxs_cols]
            w_out, _ov = _apply_tier2(w_out, dt_, nc, comp_s, s_idx)
        return w_out

    def _apply_live_basis(w, s_idx):
        """Match _curr_weights basis: scale by current gross, subtract hedges."""
        if w is None or w.empty:
            return pd.Series(dtype=float)
        wd = w.copy() * state[s_idx].get('gross', 1.0)
        for inst, h in state[s_idx].get('active_hedges', {}).items():
            wd[inst] = wd.get(inst, 0.0) - h['weight']
        return wd

    _comp_today = composite_by_date.get(today_ts)
    if _comp_today is not None and not _comp_today.empty:
        _comp_today = _comp_today.dropna()
        _comp_today = _comp_today.loc[[t for t in _comp_today.index if t in pxs_cols]]

        # Quality pre-filter + candidate pools (mirror loop)
        _n_pf  = max(n_cands, int(np.ceil(len(_comp_today) * prefilt_pct)))
        _cands = _comp_today.nlargest(min(_n_pf, len(_comp_today)))
        _n_mom = mom_filter * top_n
        if _n_mom < len(_cands):
            _idio = pd.Series(dtype=float); _m12 = pd.Series(dtype=float)
            if score_dfs.get('Idio_Mom') is not None:
                _av = [d for d in score_dfs['Idio_Mom'].index if d <= today_ts]
                if _av:
                    _idio = score_dfs['Idio_Mom'].loc[_av[-1]].reindex(_cands.index).fillna(0)
            if score_dfs.get('Mom_12M1') is not None:
                _av = [d for d in score_dfs['Mom_12M1'].index if d <= today_ts]
                if _av:
                    _m12 = score_dfs['Mom_12M1'].loc[_av[-1]].reindex(_cands.index).fillna(0)
            _topmom      = _idio.add(_m12, fill_value=0).nlargest(min(_n_mom, len(_cands))).index
            _cands_alpha = _cands.reindex(_topmom).dropna()
        else:
            _cands_alpha = _cands
        _cands_mvo = _cands.nlargest(min(n_cands, len(_cands)))
        _non_core  = [t for t in _cands.index if t in pxs_cols]

        # Building blocks (unpenalised)
        _w_alpha = _build_alpha(today_ts, _cands_alpha)
        _w_mvo, _ = _build_mvo(today_ts, _cands_mvo, _comp_today)
        if _w_mvo.empty:
            _w_mvo = _w_alpha.copy()
        _allt   = list(set(_w_alpha.index) | set(_w_mvo.index))
        _w_hyb  = (_w_alpha.reindex(_allt).fillna(0) + _w_mvo.reindex(_allt).fillna(0)) / 2
        _w_hyb  = _w_hyb[_w_hyb > 0]
        if _w_hyb.sum() > 0: _w_hyb /= _w_hyb.sum()

        # Penalised blocks (Excl strategy)
        _comp_pen   = _penalise(_comp_today, today_ts)
        _cands_pen_a = _comp_pen.reindex(_cands_alpha.index).dropna().nlargest(
                           min(_n_mom, len(_cands_alpha)))
        _cands_pen_m = _comp_pen.reindex(_cands_mvo.index).dropna().nlargest(
                           min(n_cands, len(_cands_mvo)))
        _w_alpha_p = _build_alpha(today_ts, _cands_pen_a)
        _w_mvo_p, _ = _build_mvo(today_ts, _cands_pen_m, _comp_pen)
        if _w_mvo_p.empty: _w_mvo_p = _w_alpha_p.copy()
        _alltp  = list(set(_w_alpha_p.index) | set(_w_mvo_p.index))
        _w_hyb_p = (_w_alpha_p.reindex(_alltp).fillna(0) + _w_mvo_p.reindex(_alltp).fillna(0)) / 2
        _w_hyb_p = _w_hyb_p[_w_hyb_p > 0]
        if _w_hyb_p.sum() > 0: _w_hyb_p /= _w_hyb_p.sum()

        # ---- Static strategies (always full rebalance basis) ----
        _theo_fresh['Baseline'] = _apply_live_basis(_build_baseline(today_ts), S_BASE)
        _theo_fresh['Alpha']    = _apply_live_basis(
            _advp_then_tier2_at(_w_alpha, S_ALPHA, today_ts, _cands, _non_core, _comp_today), S_ALPHA)
        _theo_fresh['MVO']      = _apply_live_basis(
            _advp_then_tier2_at(_w_mvo, S_MVO, today_ts, _cands, _non_core, _comp_today), S_MVO)
        _theo_fresh['Hybrid']   = _apply_live_basis(
            _advp_then_tier2_at(_w_hyb, S_HYB, today_ts, _cands, _non_core, _comp_today), S_HYB)
        _w_s4_base = _regime_weights(today_ts, state[S_SMART]['regime'],
                                     {'alpha': _w_alpha, 'mvo': _w_mvo, 'hybrid': _w_hyb})
        _theo_fresh['Smart']    = _apply_live_basis(
            _advp_then_tier2_at(_w_s4_base, S_SMART, today_ts, _cands, _non_core, _comp_today), S_SMART)

        # ---- Dynamic strategies (regime by current drawdown state) ----
        _dyn_blocks = {
            S_DYN:       {'alpha': _w_alpha,   'mvo': _w_mvo,   'hybrid': _w_hyb},
            S_HEDGE:     {'alpha': _w_alpha,   'mvo': _w_mvo,   'hybrid': _w_hyb},
            S_DD:        {'alpha': _w_alpha,   'mvo': _w_mvo,   'hybrid': _w_hyb},
            S_EXCL:      {'alpha': _w_alpha_p, 'mvo': _w_mvo_p, 'hybrid': _w_hyb_p},
        }
        _dyn_labels = {S_DYN: 'Dynamic', S_HEDGE: 'Dyn+Hdg',
                       S_DD: 'DD Pol',  S_EXCL: 'Excl'}
        for _sx, _blocks in _dyn_blocks.items():
            _comp_s = _comp_pen if _sx == S_EXCL else _comp_today
            _w_hyp  = _regime_weights(today_ts, state[_sx]['regime'], _blocks)
            if _w_hyp.empty:
                _theo_fresh[_dyn_labels[_sx]] = pd.Series(dtype=float)
                continue
            _w_hyp = _advp_then_tier2_at(_w_hyp, _sx, today_ts, _cands, _non_core, _comp_s)
            _theo_fresh[_dyn_labels[_sx]] = _apply_live_basis(_w_hyp, _sx)
        # MVO+Hedge: pure MVO, no regime switch
        _theo_fresh['MVO+Hedge'] = _apply_live_basis(
            _advp_then_tier2_at(_w_mvo, S_MVO_HEDGE, today_ts, _cands_mvo, _non_core, _comp_today),
            S_MVO_HEDGE)

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
        ('MVO+Hedge', S_MVO_HEDGE),
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

    # Last-day returns — use full Pxs_df index for previous day
    _px_dates_before = Pxs_df.index[Pxs_df.index < today_ts]
    prev_td  = _px_dates_before[-1] if len(_px_dates_before) > 0 else today_ts
    _day_ret = {}
    for tkr in _all_tks:
        if tkr in Pxs_df.columns and prev_td in Pxs_df.index                 and today_ts in Pxs_df.index:
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
    # 10-day P&L — use full nav_series (includes cached history)
    _full_dates = sorted(nav_series[S_ALPHA].index)
    _last10     = _full_dates[-10:]
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

    # Trade summary — all strategies, with PER-STRATEGY real/theoretical labels.
    # A strategy's displayed trades are REAL if it actually rebalanced on the last
    # day (state['last_rebal'] == today_ts) -- dynamic strategies can fire on any
    # eligible day, not just calc_dates -- and THEORETICAL otherwise (the trades you
    # WOULD place if you forced a rebalance today). Delta legs differ by case:
    #   REAL: fresh = today's executed book (_curr_weights); held = PREVIOUS rebalance.
    #   THEO: fresh = would-be book (_theo_fresh); held = currently-held book (_curr_weights).
    print(f"\n  {'='*72}")
    print(f"  TRADE SUMMARY  —  {today_ts.date()}   (per-strategy REAL / THEORETICAL)")
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
        ('MVO+Hedge', S_MVO_HEDGE),
    ]

    def _is_real(s_idx):
        """True iff this strategy actually rebalanced on the last day."""
        lr = state[s_idx].get('last_rebal')
        return lr is not None and pd.Timestamp(lr) == today_ts

    # Per-strategy label for display.
    _strat_kind = {lbl: ('REAL' if _is_real(s_idx) else 'THEORETICAL')
                   for lbl, s_idx in _all_strats_summary}

    # Helper: previous (pre-today) rebalance for a strategy, on the live basis.
    # Used as the HELD leg for REAL strategies (today's commit overwrote state).
    def _prev_book(s_idx):
        try:
            with ENGINE.connect() as _conn:
                _db_lbl = STRATEGY_LABELS[s_idx]
                _prev_rows = _conn.execute(text(f"""
                    SELECT r.date, r.ticker, r.weight
                    FROM {MVO_REBAL_CACHE_TBL} r
                    WHERE r.params_hash = :ph AND r.strategy = :st
                      AND r.date = (
                          SELECT MAX(date) FROM {MVO_REBAL_CACHE_TBL}
                          WHERE params_hash = :ph AND strategy = :st
                            AND date < :today
                      )
                """), {'ph': ph, 'st': _db_lbl, 'today': today_ts.date()}).fetchall()
            if _prev_rows:
                _pdt = pd.Timestamp(_prev_rows[0][0])
                _pw  = pd.Series({r[1]: float(r[2]) for r in _prev_rows})
                return _apply_live_basis(_drift(_pw, _pdt, today_ts), s_idx)
        except Exception:
            pass
        return pd.Series(dtype=float)

    # Build fresh (post-rebalance target) and drifted (held) legs PER STRATEGY.
    _fresh_w = {}
    _drifted_w = {}
    for lbl, s_idx in _all_strats_summary:
        if _is_real(s_idx):
            # Executed today: fresh = today's book; held = previous rebalance.
            _fresh_w[lbl], _   = _curr_weights(s_idx)
            _drifted_w[lbl]    = _prev_book(s_idx)
        else:
            # Did not trade today: fresh = would-be book; held = currently held.
            if lbl in _theo_fresh and not _theo_fresh[lbl].empty:
                _fresh_w[lbl] = _theo_fresh[lbl]
            else:
                _fresh_w[lbl], _ = _curr_weights(s_idx)
            _drifted_w[lbl], _ = _curr_weights(s_idx)

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
        # Per-strategy REAL/THEO tag row
        print(f"  {'':<8}" +
              "".join(f"  {('REAL' if _strat_kind[l]=='REAL' else 'THEO'):>{col_w}}"
                      for l, _ in _all_strats_summary))
        print("  " + "-"*8 + ("  " + "-"*col_w)*len(_all_strats_summary))
        for tkr in all_delta_tks:
            row = f"  {tkr:<8}"
            for lbl, _ in _all_strats_summary:
                d = deltas[lbl].get(tkr, 0.0)
                row += f"  {d:>+8.2f}%" if abs(d) >= 0.01 else f"  {'':>9}"
            print(row)
        # Legend
        _n_real = sum(1 for l, _ in _all_strats_summary if _strat_kind[l] == 'REAL')
        print(f"\n  REAL = rebalanced today (trades to execute now) | "
              f"THEORETICAL = would-be trades if rebalanced today.  "
              f"[{_n_real} REAL, {len(_all_strats_summary)-_n_real} THEORETICAL]")

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
        'nav_mvo_hedge':              nav_series[S_MVO_HEDGE],
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
        'mvo_hedge_weights_by_date':  state[S_MVO_HEDGE]['weights_by_date'],
        'baseline_weights_by_date':   state[S_BASE]['weights_by_date'],
        # ── DD de-gross/re-entry schedule (persisted across incremental runs) ──
        'dd_log':                     state[S_DD]['dd_log'],
        'dd_log_excl':                state[S_EXCL]['dd_log'],
        'exposure_tbl':               EXPOSURE_TBL,
        'params_hash':                ph,
        # ── Diagnostic data for get_last_rebal_diagnostics() ──────────────────
        'composite_scores':           composite_by_date,
        'score_dfs':                  score_dfs,
        'ic_weights_by_date':         weights_by_date,
        'universe':                   universe,
    }

