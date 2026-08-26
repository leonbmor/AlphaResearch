#!/usr/bin/env python
# coding: utf-8
"""
etf_regime_signals.py
=====================

SELF-CONTAINED per-ETF volatility-regime signal generation + caching.

Runs, for every ETF price column supplied, two independent regime detectors --
a 3-state Gaussian HMM (Low/Med/High vol) and a two-sided vol-CUSUM change-point
-- across a configurable set of trailing windows (default [1, 3] years), point-in-
time, and caches the RAW per-ETF outputs to dedicated tables. No composition, no
penalty logic, no MOVE/QBD/quintile machinery: each ETF's signal is built purely
from its own price series.

This module is deliberately STANDALONE. The HMM forward-filter / ordered-fit and
the CUSUM recursion are copied verbatim from the validated hmm_vol_regimes.py and
cusum.py so this file has ZERO import dependency on them -- but the detector maths
is identical, so ETF signals are consistent with the rest of the regime work.

Public API
----------
run_etf_signals(etf_px_df, start_date, windows=[1, 3], mode='incremental', engine=...)
    Compute + cache HMM and CUSUM outputs for every column of etf_px_df.

load_etf_hmm(assets=None, windows=None, engine=...)   -> tidy DataFrame
load_etf_cusum(assets=None, windows=None, engine=...)  -> tidy DataFrame

Cache tables (separate from the QQQ/quintile caches):
    etf_hmm_regimes    : asset, date, window_years, n_obs, p_low, p_med, p_high,
                         converged, vol_low, vol_med, vol_high
    etf_cusum_regimes  : asset, date, window_years, sigma0, stat_up, stat_dn,
                         alarm_up, alarm_dn, inc_up, inc_dn
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# hmmlearn imported lazily inside the HMM fit so the CUSUM detector, the cache
# loaders, and the CUSUM half of run_etf_signals remain importable/usable in
# environments without hmmlearn installed.


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONN    = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'

# separate cache tables (walled off from the QQQ / quintile regime caches)
ETF_HMM_TBL     = 'etf_hmm_regimes'
ETF_CUSUM_TBL   = 'etf_cusum_regimes'
MODEL_VERSION   = 'etf_v1'

# default windows for the ETF signals (override per call)
DEFAULT_WINDOWS = [1, 3]

TRADING_DAYS_PER_YEAR = 252

# ── HMM config (identical to hmm_vol_regimes.py) ──────────────────────────────
N_STATES        = 3                       # Low / Med / High vol
STATE_LABELS    = ('Low', 'Med', 'High')  # index order == ascending variance
MIN_OBS_TO_FIT  = TRADING_DAYS_PER_YEAR   # hard floor: >= 1y of returns
VAR_FLOOR       = 1e-12
MIN_COVAR       = 1e-8                     # daily-return scale (default 1e-3 too large)
VOL_ANN_CEILING = 5.0                     # 500% annualized vol -> degenerate
EM_RESTARTS     = 12
EM_MAX_ITER     = 200
EM_TOL          = 1e-4
BASE_SEED       = 20260101

# ── CUSUM config (identical to cusum.py) ──────────────────────────────────────
CUSUM_SHIFT     = 1.5                      # target vol shift (multiplicative)
H_UP            = 3.374                    # MC-calibrated, ARL0 ~ 500d (shift=1.5)
H_DN            = 3.888
MIN_REF_OBS     = 60                       # min obs before sigma0 is trusted

RHO_UP = CUSUM_SHIFT ** 2
RHO_DN = 1.0 / CUSUM_SHIFT ** 2


# ==============================================================================
# HMM DETECTOR CORE  (copied verbatim from hmm_vol_regimes.py)
# ==============================================================================

@dataclass
class RegimeRow:
    asset: str
    date: pd.Timestamp
    window_years: int
    n_obs: int
    p_low: float
    p_med: float
    p_high: float
    converged: bool
    vol_low: float
    vol_med: float
    vol_high: float


def _gaussian_logpdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Log N(x; mean, var) for a 1-D array x. var already variance-floored."""
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mean) ** 2 / var)


def forward_filter_last(returns: np.ndarray,
                        startprob: np.ndarray,
                        transmat: np.ndarray,
                        means: np.ndarray,
                        variances: np.ndarray) -> Optional[np.ndarray]:
    """
    Explicit, scaled forward pass. Returns the FILTERED state distribution at the
    LAST observation: P(state_T | returns_{1:T}). PIT quantity -- conditions only
    on observations up to and including T. Returns None if degenerate.
    """
    T = returns.shape[0]
    K = startprob.shape[0]
    if T == 0:
        return None

    var = np.maximum(variances, VAR_FLOOR)
    alpha = np.zeros(K)

    log_b0 = np.array([_gaussian_logpdf(returns[0:1], means[i], var[i])[0]
                       for i in range(K)])
    with np.errstate(under='ignore'):
        log_a0 = np.log(np.maximum(startprob, 1e-300)) + log_b0
    a0 = np.exp(log_a0 - log_a0.max())
    s0 = a0.sum()
    if not np.isfinite(s0) or s0 <= 0:
        return None
    alpha = a0 / s0

    for t in range(1, T):
        log_bt = np.array([_gaussian_logpdf(returns[t:t + 1], means[i], var[i])[0]
                           for i in range(K)])
        pred = alpha @ transmat
        pred = np.maximum(pred, 1e-300)
        log_a = np.log(pred) + log_bt
        a = np.exp(log_a - log_a.max())
        s = a.sum()
        if not np.isfinite(s) or s <= 0:
            return None
        alpha = a / s

    return alpha


def _fit_hmm_ordered(returns: np.ndarray, seed: int,
                     warm: Optional[Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]] = None
                     ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, bool]]:
    """
    Fit a K-state Gaussian HMM with multi-restart EM, keep best log-likelihood,
    REORDER states by ascending variance (0=Low,1=Med,2=High). Rejects degenerate
    fits (runaway variance). Returns ordered params + converged flag, or None.
    """
    n = returns.shape[0]
    if n < MIN_OBS_TO_FIT:
        return None

    from hmmlearn.hmm import GaussianHMM

    X = returns.reshape(-1, 1).astype(float)
    best = None
    best_ll = -np.inf

    def _is_valid(model) -> bool:
        try:
            mu = model.means_.ravel()
            va = np.maximum(model.covars_.ravel(), VAR_FLOOR)
            sp = model.startprob_; tm = model.transmat_
        except Exception:
            return False
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(va))
                and np.all(np.isfinite(sp)) and np.all(np.isfinite(tm))):
            return False
        ann = np.sqrt(va) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if np.any(ann > VOL_ANN_CEILING):
            return False
        return True

    def _consider(model):
        nonlocal best, best_ll
        if not _is_valid(model):
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ll = model.score(X)
        except Exception:
            return
        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best = model

    if warm is not None:
        try:
            wsp, wtm, wmu, wvar = warm
            wm = GaussianHMM(n_components=N_STATES, covariance_type='diag',
                             n_iter=EM_MAX_ITER, tol=EM_TOL, random_state=seed,
                             init_params='', min_covar=MIN_COVAR)
            wm.startprob_ = np.clip(wsp, 1e-6, None); wm.startprob_ /= wm.startprob_.sum()
            wm.transmat_  = wtm.copy()
            wm.means_     = wmu.reshape(-1, 1).copy()
            wm.covars_    = np.maximum(wvar, VAR_FLOOR).reshape(-1, 1).copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wm.fit(X)
            _consider(wm)
        except Exception:
            pass

    for r in range(EM_RESTARTS):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type='diag',
            n_iter=EM_MAX_ITER,
            tol=EM_TOL,
            random_state=seed + r,
            init_params='stmc',
            min_covar=MIN_COVAR,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X)
        except Exception:
            continue
        _consider(model)

    if best is None:
        return None

    means = best.means_.ravel().astype(float)
    variances = best.covars_.ravel().astype(float)
    variances = np.maximum(variances, VAR_FLOOR)
    startprob = best.startprob_.astype(float)
    transmat = best.transmat_.astype(float)

    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(variances))
            and np.all(np.isfinite(startprob)) and np.all(np.isfinite(transmat))):
        return None

    ann_vol = np.sqrt(variances) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if np.any(ann_vol > VOL_ANN_CEILING):
        return None

    order = np.argsort(variances)
    means_o = means[order]
    var_o = variances[order]
    startprob_o = startprob[order]
    transmat_o = transmat[np.ix_(order, order)]
    row_sums = transmat_o.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    transmat_o = transmat_o / row_sums

    converged = bool(getattr(best.monitor_, 'converged', False))
    return startprob_o, transmat_o, means_o, var_o, converged


def _calc_dates(index: pd.DatetimeIndex, start_date: pd.Timestamp,
                frequency: str = 'monthly') -> pd.DatetimeIndex:
    """Calc dates drawn from the price index (holidays handled automatically)."""
    idx = index[index >= start_date]
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    if frequency == 'daily':
        return pd.DatetimeIndex(idx)
    df = pd.DataFrame(index=idx)
    first_of_month = (df.groupby([idx.year, idx.month])
                        .apply(lambda g: g.index.min()))
    dates = pd.DatetimeIndex(sorted(set(first_of_month.values)))
    last = index[-1]
    if last not in dates:
        dates = dates.append(pd.DatetimeIndex([last])).sort_values()
    return dates


def _compute_hmm(prices: pd.DataFrame,
                 start_date,
                 windows_years,
                 warm_start: bool = False,
                 frequency: str = 'monthly',
                 verbose: bool = True) -> pd.DataFrame:
    """PIT filtered vol-regime probabilities for every ETF column. Tidy frame."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    start_date = pd.Timestamp(start_date)

    calc_dates = _calc_dates(prices.index, start_date, frequency=frequency)
    if len(calc_dates) == 0:
        if verbose:
            print(f"  [HMM] no trading dates on/after {start_date.date()}")
        return _empty_hmm_frame()

    log_px = np.log(prices.where(prices > 0))
    rets = log_px.diff()

    win_obs = {w: int(round(w * TRADING_DAYS_PER_YEAR)) for w in windows_years}
    rows: List[RegimeRow] = []
    assets = list(prices.columns)

    for ai, asset in enumerate(assets):
        r_full = rets[asset].dropna()
        if r_full.empty:
            if verbose:
                print(f"  [HMM {ai+1}/{len(assets)}] {asset}: no returns, skipped")
            continue

        a_seed = BASE_SEED + (abs(hash(asset)) % 100000)
        warm_by_window: Dict[int, Tuple] = {}
        n_emit = 0

        for dt in calc_dates:
            r_upto = r_full.loc[:dt]
            avail = r_upto.shape[0]
            if avail < MIN_OBS_TO_FIT:
                continue

            for w in windows_years:
                w_obs = win_obs[w]
                use = r_upto.iloc[-min(w_obs, avail):].values
                if use.shape[0] < MIN_OBS_TO_FIT:
                    continue

                fit = _fit_hmm_ordered(use, seed=a_seed,
                                       warm=warm_by_window.get(w) if warm_start else None)
                if fit is None:
                    rows.append(RegimeRow(asset, dt, w, use.shape[0],
                                          np.nan, np.nan, np.nan, False,
                                          np.nan, np.nan, np.nan))
                    continue

                startprob, transmat, means, variances, converged = fit
                warm_by_window[w] = (startprob, transmat, means, variances)
                filt = forward_filter_last(use, startprob, transmat, means, variances)
                if filt is None:
                    rows.append(RegimeRow(asset, dt, w, use.shape[0],
                                          np.nan, np.nan, np.nan, False,
                                          np.nan, np.nan, np.nan))
                    continue

                ann = np.sqrt(variances) * np.sqrt(TRADING_DAYS_PER_YEAR)
                rows.append(RegimeRow(
                    asset, dt, w, use.shape[0],
                    float(filt[0]), float(filt[1]), float(filt[2]),
                    converged,
                    float(ann[0]), float(ann[1]), float(ann[2]),
                ))
                n_emit += 1

        if verbose:
            print(f"  [HMM {ai+1}/{len(assets)}] {asset}: {n_emit} window-rows "
                  f"across {len(calc_dates)} calc dates")

    return _hmm_rows_to_frame(rows)


def _empty_hmm_frame() -> pd.DataFrame:
    cols = ['asset', 'date', 'window_years', 'n_obs', 'p_low', 'p_med', 'p_high',
            'converged', 'vol_low', 'vol_med', 'vol_high']
    return pd.DataFrame(columns=cols)


def _hmm_rows_to_frame(rows: List[RegimeRow]) -> pd.DataFrame:
    if not rows:
        return _empty_hmm_frame()
    df = pd.DataFrame([r.__dict__ for r in rows])
    return df.sort_values(['asset', 'date', 'window_years']).reset_index(drop=True)


# ==============================================================================
# CUSUM DETECTOR CORE  (copied verbatim from cusum.py)
# ==============================================================================

def _llr_inc(z2: float, rho: float) -> float:
    return 0.5 * ((1.0 - 1.0 / rho) * z2 - np.log(rho))


def cusum_series(returns: pd.Series,
                 window_years: int,
                 start_date=None,
                 init_up: float = 0.0,
                 init_dn: float = 0.0) -> pd.DataFrame:
    """
    Two-sided vol-CUSUM over `returns` (daily log returns, one asset). Emits rows
    from start_date onward. init_up/init_dn resume the statistic (incremental).
    Returns date-indexed frame: sigma0, stat_up, stat_dn, alarm_up, alarm_dn,
    inc_up, inc_dn.
    """
    r = returns.dropna()
    if len(r) < MIN_REF_OBS + 1:
        return pd.DataFrame(columns=['sigma0', 'stat_up', 'stat_dn',
                                     'alarm_up', 'alarm_dn', 'inc_up', 'inc_dn'])
    W = int(window_years * TRADING_DAYS_PER_YEAR)

    sigma0 = r.rolling(W, min_periods=MIN_REF_OBS).std(ddof=1).shift(1)
    start = pd.Timestamp(start_date) if start_date is not None else r.index[0]

    su, sd = float(init_up), float(init_dn)
    rows = []
    vals = r.values
    sig  = sigma0.values
    idx  = r.index
    for i in range(len(r)):
        s0 = sig[i]
        if not np.isfinite(s0) or s0 <= 0:
            continue
        z2 = (vals[i] / s0) ** 2
        i_up = _llr_inc(z2, RHO_UP)
        i_dn = _llr_inc(z2, RHO_DN)
        su = max(0.0, su + i_up)
        sd = max(0.0, sd + i_dn)
        a_up = su > H_UP
        a_dn = sd > H_DN
        if idx[i] >= start:
            rows.append((idx[i], s0, su, sd, a_up, a_dn, i_up, i_dn))
        if a_up: su = 0.0
        if a_dn: sd = 0.0
    out = pd.DataFrame(rows, columns=['date', 'sigma0', 'stat_up', 'stat_dn',
                                      'alarm_up', 'alarm_dn', 'inc_up', 'inc_dn'])
    return out.set_index('date')


def _empty_cusum_frame() -> pd.DataFrame:
    cols = ['asset', 'date', 'window_years', 'sigma0', 'stat_up', 'stat_dn',
            'alarm_up', 'alarm_dn', 'inc_up', 'inc_dn']
    return pd.DataFrame(columns=cols)


# ==============================================================================
# CACHE TABLES  (separate ETF-specific schema)
# ==============================================================================

def _ensure_tables(engine):
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {ETF_HMM_TBL} (
                asset          VARCHAR(24) NOT NULL,
                date           DATE        NOT NULL,
                window_years   INTEGER     NOT NULL,
                model_version  VARCHAR(16) NOT NULL,
                n_obs          INTEGER,
                p_low          DOUBLE PRECISION,
                p_med          DOUBLE PRECISION,
                p_high         DOUBLE PRECISION,
                converged      BOOLEAN,
                vol_low        DOUBLE PRECISION,
                vol_med        DOUBLE PRECISION,
                vol_high       DOUBLE PRECISION,
                PRIMARY KEY (asset, date, window_years, model_version)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {ETF_CUSUM_TBL} (
                asset          VARCHAR(24) NOT NULL,
                date           DATE        NOT NULL,
                window_years   INTEGER     NOT NULL,
                model_version  VARCHAR(16) NOT NULL,
                sigma0         DOUBLE PRECISION,
                stat_up        DOUBLE PRECISION,
                stat_dn        DOUBLE PRECISION,
                alarm_up       BOOLEAN,
                alarm_dn       BOOLEAN,
                inc_up         DOUBLE PRECISION,
                inc_dn         DOUBLE PRECISION,
                PRIMARY KEY (asset, date, window_years, model_version)
            )
        """))


def _last_cached_date(engine, table, asset, window, ) -> Optional[pd.Timestamp]:
    """Most recent cached date for (asset, window) under MODEL_VERSION, or None."""
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text(
            f"SELECT MAX(date) FROM {table} WHERE asset=:a AND window_years=:w "
            f"AND model_version=:mv"),
            {'a': asset, 'w': int(window), 'mv': MODEL_VERSION}).fetchone()
    if row and row[0] is not None:
        return pd.Timestamp(row[0])
    return None


def _cusum_resume_state(engine, asset, window) -> Tuple[float, float, Optional[pd.Timestamp]]:
    """Resume (stat_up, stat_dn) and the last cached date for incremental CUSUM."""
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text(
            f"SELECT date, stat_up, stat_dn FROM {ETF_CUSUM_TBL} "
            f"WHERE asset=:a AND window_years=:w AND model_version=:mv "
            f"ORDER BY date DESC LIMIT 1"),
            {'a': asset, 'w': int(window), 'mv': MODEL_VERSION}).fetchone()
    if row and row[0] is not None:
        return float(row[1] or 0.0), float(row[2] or 0.0), pd.Timestamp(row[0])
    return 0.0, 0.0, None


def _wipe_asset(engine, table, asset):
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(
            f"DELETE FROM {table} WHERE asset=:a AND model_version=:mv"),
            {'a': asset, 'mv': MODEL_VERSION})


def _save_hmm(engine, df: pd.DataFrame):
    if df.empty:
        return
    from sqlalchemy import text
    rows = []
    for _, r in df.iterrows():
        def _f(v):
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        rows.append({
            'asset': str(r['asset']), 'date': pd.Timestamp(r['date']).strftime('%Y-%m-%d'),
            'window_years': int(r['window_years']), 'model_version': MODEL_VERSION,
            'n_obs': None if pd.isna(r['n_obs']) else int(r['n_obs']),
            'p_low': _f(r['p_low']), 'p_med': _f(r['p_med']), 'p_high': _f(r['p_high']),
            'converged': bool(r['converged']) if pd.notna(r['converged']) else None,
            'vol_low': _f(r['vol_low']), 'vol_med': _f(r['vol_med']), 'vol_high': _f(r['vol_high']),
        })
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {ETF_HMM_TBL}
                (asset, date, window_years, model_version, n_obs, p_low, p_med,
                 p_high, converged, vol_low, vol_med, vol_high)
            VALUES (:asset, :date, :window_years, :model_version, :n_obs, :p_low,
                    :p_med, :p_high, :converged, :vol_low, :vol_med, :vol_high)
            ON CONFLICT (asset, date, window_years, model_version) DO UPDATE SET
                n_obs=EXCLUDED.n_obs, p_low=EXCLUDED.p_low, p_med=EXCLUDED.p_med,
                p_high=EXCLUDED.p_high, converged=EXCLUDED.converged,
                vol_low=EXCLUDED.vol_low, vol_med=EXCLUDED.vol_med, vol_high=EXCLUDED.vol_high
        """), rows)


def _save_cusum(engine, asset: str, window: int, df: pd.DataFrame):
    if df.empty:
        return
    from sqlalchemy import text
    rows = []
    for dt, r in df.iterrows():
        def _f(v):
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        rows.append({
            'asset': asset, 'date': pd.Timestamp(dt).strftime('%Y-%m-%d'),
            'window_years': int(window), 'model_version': MODEL_VERSION,
            'sigma0': _f(r['sigma0']), 'stat_up': _f(r['stat_up']),
            'stat_dn': _f(r['stat_dn']),
            'alarm_up': bool(r['alarm_up']), 'alarm_dn': bool(r['alarm_dn']),
            'inc_up': _f(r['inc_up']), 'inc_dn': _f(r['inc_dn']),
        })
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {ETF_CUSUM_TBL}
                (asset, date, window_years, model_version, sigma0, stat_up,
                 stat_dn, alarm_up, alarm_dn, inc_up, inc_dn)
            VALUES (:asset, :date, :window_years, :model_version, :sigma0,
                    :stat_up, :stat_dn, :alarm_up, :alarm_dn, :inc_up, :inc_dn)
            ON CONFLICT (asset, date, window_years, model_version) DO UPDATE SET
                sigma0=EXCLUDED.sigma0, stat_up=EXCLUDED.stat_up,
                stat_dn=EXCLUDED.stat_dn, alarm_up=EXCLUDED.alarm_up,
                alarm_dn=EXCLUDED.alarm_dn, inc_up=EXCLUDED.inc_up, inc_dn=EXCLUDED.inc_dn
        """), rows)


# ==============================================================================
# PUBLIC API
# ==============================================================================

def run_etf_signals(etf_px_df: pd.DataFrame,
                    start_date,
                    windows=None,
                    mode: str = 'incremental',
                    frequency: str = 'monthly',
                    conn_str: str = DEFAULT_CONN,
                    engine=None,
                    verbose: bool = True) -> dict:
    """
    Compute + cache HMM and CUSUM regime signals for every column of etf_px_df.

    Parameters
    ----------
    etf_px_df : DataFrame
        Daily prices, DatetimeIndex x ETF tickers (columns). Each column uses only
        its own non-NaN history.
    start_date : str | Timestamp
        First date for which HMM outputs are produced (CUSUM emits from its first
        trustworthy reference date, or resumes from cache in incremental mode).
    windows : list[int]
        Trailing windows in years. Default [1, 3].
    mode : 'incremental' | 'rebuild'
        'incremental' : resume each (asset, window) from the last cached date;
                        only new dates are computed/stored.
        'rebuild'     : delete each asset's rows (this MODEL_VERSION) and recompute
                        from start_date -- overrides everything.
    frequency : 'monthly' | 'daily'
        HMM calc-date frequency (CUSUM is always daily).

    Returns
    -------
    dict with 'hmm' and 'cusum' tidy DataFrames (the newly computed rows), plus
    'assets' and 'windows'.
    """
    if windows is None:
        windows = list(DEFAULT_WINDOWS)
    windows = [int(w) for w in windows]

    if engine is None:
        from sqlalchemy import create_engine
        engine = create_engine(conn_str)

    if not isinstance(etf_px_df.index, pd.DatetimeIndex):
        etf_px_df = etf_px_df.copy()
        etf_px_df.index = pd.to_datetime(etf_px_df.index)
    etf_px_df = etf_px_df.sort_index()
    start_date = pd.Timestamp(start_date)

    _ensure_tables(engine)

    assets = list(etf_px_df.columns)
    if verbose:
        print('=' * 72)
        print('  ETF REGIME SIGNALS')
        print('=' * 72)
        print(f"  assets={len(assets)}  windows={windows}  mode={mode}  "
              f"model_version={MODEL_VERSION}")
        print(f"  HMM table={ETF_HMM_TBL}  CUSUM table={ETF_CUSUM_TBL}")

    # rebuild: wipe every asset up-front so a partial prior run can't linger
    if mode == 'rebuild':
        for asset in assets:
            _wipe_asset(engine, ETF_HMM_TBL, asset)
            _wipe_asset(engine, ETF_CUSUM_TBL, asset)
        if verbose:
            print(f"  rebuild: wiped {len(assets)} assets from both tables")

    # ---- HMM ----------------------------------------------------------------
    if verbose:
        print("\n  --- HMM ---")
    hmm_all = []
    for asset in assets:
        px1 = etf_px_df[[asset]].dropna()
        if px1.empty:
            continue
        # incremental start: day after the last cached date (min across windows)
        hmm_start = start_date
        if mode == 'incremental':
            last_dates = [_last_cached_date(engine, ETF_HMM_TBL, asset, w) for w in windows]
            last_dates = [d for d in last_dates if d is not None]
            if last_dates:
                # resume from the day after the EARLIEST per-window last date, so
                # every window gets its missing tail (per-window dedupe on upsert)
                resume = min(last_dates)
                nxt = px1.index[px1.index > resume]
                if len(nxt) == 0:
                    if verbose:
                        print(f"  [HMM] {asset}: up to date, nothing new")
                    continue
                hmm_start = nxt[0]
        hmm_df = _compute_hmm(px1, hmm_start, windows,
                              warm_start=False, frequency=frequency, verbose=verbose)
        if not hmm_df.empty:
            _save_hmm(engine, hmm_df)
            hmm_all.append(hmm_df)

    # ---- CUSUM --------------------------------------------------------------
    if verbose:
        print("\n  --- CUSUM ---")
    cusum_all = []
    for asset in assets:
        r_full = np.log(etf_px_df[asset].where(etf_px_df[asset] > 0)).diff().dropna()
        if len(r_full) < MIN_REF_OBS + 1:
            continue
        for w in windows:
            if mode == 'incremental':
                init_up, init_dn, last_dt = _cusum_resume_state(engine, asset, w)
                cus_start = start_date
                if last_dt is not None:
                    nxt = r_full.index[r_full.index > last_dt]
                    if len(nxt) == 0:
                        continue
                    cus_start = nxt[0]
                cdf = cusum_series(r_full, w, start_date=cus_start,
                                   init_up=init_up, init_dn=init_dn)
            else:
                cdf = cusum_series(r_full, w, start_date=start_date)
            if not cdf.empty:
                _save_cusum(engine, asset, w, cdf)
                cdf2 = cdf.reset_index()
                cdf2.insert(0, 'window_years', w)
                cdf2.insert(0, 'asset', asset)
                cusum_all.append(cdf2)
        if verbose:
            print(f"  [CUSUM] {asset}: done")

    hmm_out   = pd.concat(hmm_all, ignore_index=True) if hmm_all else _empty_hmm_frame()
    cusum_out = pd.concat(cusum_all, ignore_index=True) if cusum_all else _empty_cusum_frame()

    if verbose:
        print(f"\n  DONE: HMM {len(hmm_out)} rows, CUSUM {len(cusum_out)} rows cached.")

    return {'hmm': hmm_out, 'cusum': cusum_out,
            'assets': assets, 'windows': windows}


def load_etf_hmm(assets=None, windows=None, conn_str: str = DEFAULT_CONN,
                 engine=None) -> pd.DataFrame:
    """Load cached ETF HMM rows (tidy). Optionally filter by assets / windows."""
    from sqlalchemy import create_engine, text
    if engine is None:
        engine = create_engine(conn_str)
    q = (f"SELECT asset, date, window_years, n_obs, p_low, p_med, p_high, "
         f"converged, vol_low, vol_med, vol_high FROM {ETF_HMM_TBL} "
         f"WHERE model_version = :mv")
    params = {'mv': MODEL_VERSION}
    if assets is not None:
        q += " AND asset = ANY(:assets)"; params['assets'] = list(assets)
    if windows is not None:
        q += " AND window_years = ANY(:wins)"; params['wins'] = [int(w) for w in windows]
    q += " ORDER BY asset, date, window_years"
    return pd.read_sql(text(q), engine, params=params, parse_dates=['date'])


def load_etf_cusum(assets=None, windows=None, conn_str: str = DEFAULT_CONN,
                   engine=None) -> pd.DataFrame:
    """Load cached ETF CUSUM rows (tidy). Optionally filter by assets / windows."""
    from sqlalchemy import create_engine, text
    if engine is None:
        engine = create_engine(conn_str)
    q = (f"SELECT asset, date, window_years, sigma0, stat_up, stat_dn, "
         f"alarm_up, alarm_dn, inc_up, inc_dn FROM {ETF_CUSUM_TBL} "
         f"WHERE model_version = :mv")
    params = {'mv': MODEL_VERSION}
    if assets is not None:
        q += " AND asset = ANY(:assets)"; params['assets'] = list(assets)
    if windows is not None:
        q += " AND window_years = ANY(:wins)"; params['wins'] = [int(w) for w in windows]
    q += " ORDER BY asset, date, window_years"
    return pd.read_sql(text(q), engine, params=params, parse_dates=['date'])
