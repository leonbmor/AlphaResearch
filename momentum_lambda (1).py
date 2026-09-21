#!/usr/bin/env python
# coding: utf-8
"""
momentum_lambda.py
==================
Single-factor (12M1 price momentum) cross-sectional model.

Each trading day t, regress the cross-section of stock returns on the
standardized 12M1 momentum exposure known BEFORE t:

    r_i,t = a_t + lambda_t * z_i,t + e_i,t

    r_i,t   : simple return of stock i from t-1 to t
    z_i,t   : 12M1 momentum of stock i, computed from prices strictly before t
              (start = 253rd prior trading day, end = 22nd prior trading day),
              cross-sectionally z-scored and winsorized at +/- Z_CLIP
    a_t     : intercept -- absorbs the market move
    lambda_t: pure momentum factor return (market-neutral long-short spread)
    e_i,t   : residual

The momentum definition is identical to _ics_compute_mom_12m1 in the IC study
(pit_alpha_research.py): same lookback (252), skip (21), strict "< t" dating
and plain cross-sectional z-score. Winsorization of the z-score is the only
addition, so a few extreme names can't dominate the daily slope.

With an intercept, the OLS slope equals the return of a portfolio weighted by
each stock's demeaned exposure -- i.e. the classic long-short momentum factor
return. Its crashes are the textbook momentum crashes, uncontaminated by market
beta, which is what the regime detectors should read.

OUTPUT
    lambda_df : date x [lambda, alpha, n_obs, r2]
    residuals : date x ticker panel of e_i,t (in memory only, not cached)
    level     : 100 * cumprod(1 + lambda) -- the level index fed to the
                HMM/CUSUM machinery as a "price" column named MOM_12M1

USAGE
    from momentum_lambda import build_momentum_lambda, load_momentum_lambda
    res = build_momentum_lambda(Pxs_df, engine=engine, mode='rebuild')   # 1st run
    res = build_momentum_lambda(Pxs_df, engine=engine)                   # then incremental
    mom_px = res['level'].to_frame()             # column 'MOM_12M1'

    run_etf_signals(mom_px, start_date='2019-01-01', windows=[1, 3],
                    frequency='daily', warm_start=True, mode='rebuild',
                    model_version='mom_v1', engine=engine)
    mom_hmm   = load_etf_hmm(windows=[1, 3],   model_version='mom_v1', engine=engine)
    mom_cusum = load_etf_cusum(windows=[1, 3], model_version='mom_v1', engine=engine)
"""

import numpy as np
import pandas as pd

# ---- parameters (match the IC study's Mom_12M1) -----------------------------
MOM_LONG    = 252     # 12M1 lookback (trading days)      == ICS_MOM_LONG
MOM_SKIP    = 21      # 12M1 skip period (trading days)   == ICS_MOM_SKIP
MIN_STOCKS  = 50      # min names for a valid momentum cross-section == ICS_MIN_STOCKS
MIN_REG_OBS = 100     # min names with exposure AND return for a valid lambda
Z_CLIP      = 3.0     # winsorize the standardized exposure at +/- Z_CLIP
LEVEL_NAME  = 'MOM_12M1'
LEVEL_BASE  = 100.0

# ---- cache ------------------------------------------------------------------
LAMBDA_TBL        = 'momentum_lambda_cache'
LAMBDA_MODEL_VER  = 'mom_v1'   # bump if the momentum definition / params change,
                               # so lambdas from a different spec are never mixed


def _compute_lambdas(px: pd.DataFrame, compute_from=None,
                     mom_long: int = MOM_LONG, mom_skip: int = MOM_SKIP,
                     min_stocks: int = MIN_STOCKS, min_reg_obs: int = MIN_REG_OBS,
                     z_clip: float = Z_CLIP, verbose: bool = True):
    """
    Pure compute core. Returns (lambda_df, residuals) for every date >=
    compute_from (all dates if None). Each lambda_t depends only on prices up to
    and including t, so any date range can be computed independently -- which is
    what makes incremental caching exact (no carried state).
    """
    P = px.values
    T, N = P.shape
    dates, tickers = px.index, px.columns

    R = np.full((T, N), np.nan)
    R[1:] = P[1:] / P[:-1] - 1.0

    k0 = mom_long + 1
    t_start = k0
    if compute_from is not None:
        t_start = max(k0, int(dates.searchsorted(pd.Timestamp(compute_from))))

    lam   = np.full(T, np.nan)
    alpha = np.full(T, np.nan)
    nobs  = np.zeros(T, dtype=int)
    r2    = np.full(T, np.nan)
    E     = np.full((T, N), np.nan)

    n_todo = max(T - t_start, 0)
    if verbose:
        print(f"  computing {n_todo} date(s)"
              + (f" from {dates[t_start].date()}" if n_todo else ""), flush=True)

    for t in range(t_start, T):
        # 12M1 from prices STRICTLY before t (identical to the IC study)
        p_start = P[t - (mom_long + 1)]
        p_end   = P[t - (mom_skip + 1)]
        mom = p_end / p_start - 1.0
        mv = np.isfinite(mom)
        if mv.sum() < min_stocks:
            continue
        m = mom[mv]
        sd = m.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = np.full(N, np.nan)
        z[mv] = np.clip((m - m.mean()) / sd, -z_clip, z_clip)

        r = R[t]
        ok = np.isfinite(z) & np.isfinite(r)
        n = int(ok.sum())
        if n < min_reg_obs:
            continue
        x, y = z[ok], r[ok]
        xm, ym = x.mean(), y.mean()
        sxx = ((x - xm) ** 2).sum()
        if sxx <= 0:
            continue
        b = ((x - xm) * (y - ym)).sum() / sxx          # OLS slope (with intercept)
        a = ym - b * xm
        e = y - a - b * x
        sst = ((y - ym) ** 2).sum()

        lam[t], alpha[t], nobs[t] = b, a, n
        r2[t] = 1.0 - (e ** 2).sum() / sst if sst > 0 else np.nan
        E[t, ok] = e

        if verbose and (t - t_start) % 250 == 0:
            print(f"    {dates[t].date()}  [{t-t_start+1}/{n_todo}]  "
                  f"lambda={b*100:+.3f}%  n={n}", flush=True)

    lambda_df = pd.DataFrame({'lambda': lam, 'alpha': alpha,
                              'n_obs': nobs, 'r2': r2}, index=dates)
    lambda_df.index.name = 'date'
    keep = lambda_df['lambda'].notna()
    lambda_df = lambda_df[keep]
    residuals = pd.DataFrame(E, index=dates, columns=tickers).loc[lambda_df.index]
    return lambda_df, residuals


# =============================================================================
# Cache (Postgres)
# =============================================================================
def _ensure_table(engine):
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {LAMBDA_TBL} (
                model_version VARCHAR(24) NOT NULL,
                date          DATE        NOT NULL,
                lambda        DOUBLE PRECISION,
                alpha         DOUBLE PRECISION,
                n_obs         INTEGER,
                r2            DOUBLE PRECISION,
                PRIMARY KEY (model_version, date)
            )
        """))


def load_momentum_lambda(engine, model_version: str = LAMBDA_MODEL_VER) -> pd.DataFrame:
    """Full cached lambda history (date-indexed: lambda, alpha, n_obs, r2)."""
    from sqlalchemy import text
    df = pd.read_sql(text(f"SELECT date, lambda, alpha, n_obs, r2 FROM {LAMBDA_TBL} "
                          f"WHERE model_version = :mv ORDER BY date"),
                     engine, params={'mv': model_version}, parse_dates=['date'])
    return df.set_index('date')


def _last_cached_date(engine, model_version):
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text(f"SELECT MAX(date) FROM {LAMBDA_TBL} "
                                f"WHERE model_version = :mv"),
                           {'mv': model_version}).fetchone()
    return pd.Timestamp(row[0]) if row and row[0] is not None else None


def _delete_from(engine, model_version, dt):
    """Delete cached rows on/after dt (used to wipe the provisional last date)."""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {LAMBDA_TBL} "
                          f"WHERE model_version = :mv AND date >= :d"),
                     {'mv': model_version, 'd': pd.Timestamp(dt).strftime('%Y-%m-%d')})


def _wipe_all(engine, model_version):
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {LAMBDA_TBL} WHERE model_version = :mv"),
                     {'mv': model_version})


def _save(engine, model_version, lambda_df):
    if lambda_df.empty:
        return
    from sqlalchemy import text
    rows = [{'mv': model_version, 'd': d.strftime('%Y-%m-%d'),
             'l': float(r['lambda']), 'a': float(r['alpha']),
             'n': int(r['n_obs']), 'r2': (None if pd.isna(r['r2']) else float(r['r2']))}
            for d, r in lambda_df.iterrows()]
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {LAMBDA_TBL} (model_version, date, lambda, alpha, n_obs, r2)
            VALUES (:mv, :d, :l, :a, :n, :r2)
            ON CONFLICT (model_version, date) DO UPDATE SET
                lambda=EXCLUDED.lambda, alpha=EXCLUDED.alpha,
                n_obs=EXCLUDED.n_obs, r2=EXCLUDED.r2
        """), rows)


# =============================================================================
# Entry point
# =============================================================================
def build_momentum_lambda(Pxs_df: pd.DataFrame,
                          engine=None,
                          mode: str = 'incremental',
                          model_version: str = LAMBDA_MODEL_VER,
                          mom_long: int = MOM_LONG,
                          mom_skip: int = MOM_SKIP,
                          min_stocks: int = MIN_STOCKS,
                          min_reg_obs: int = MIN_REG_OBS,
                          z_clip: float = Z_CLIP,
                          verbose: bool = True) -> dict:
    """
    Parameters
    ----------
    Pxs_df : DataFrame of CLEAN stock prices (dates x tickers), universe only,
             NaN where unavailable. Non-stock series must not be included.
    engine : SQLAlchemy engine. If None, nothing is cached (full in-memory compute).
    mode   : 'rebuild'     -> wipe this model_version and recompute all history.
             'incremental' -> keep settled cached lambdas; ALWAYS wipe and
                              recompute the most recent (provisional) Pxs_df date,
                              plus any dates after the last cached one.

    Returns
    -------
    dict:
      'lambda_df' : FULL lambda history (from cache when engine given)
      'residuals' : residual panel for the dates COMPUTED IN THIS RUN only
                    (not cached; in incremental mode that's just the new tail)
      'level'     : 100 * cumprod(1 + lambda) over the full history, named MOM_12M1
    """
    if mode not in ('incremental', 'rebuild'):
        raise ValueError("mode must be 'incremental' or 'rebuild'")

    px = Pxs_df.copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)
    px = px.sort_index().astype(float)
    px = px.where(px > 0)                       # non-positive prices -> NaN
    last_px_dt = px.index[-1]

    if verbose:
        print(f"build_momentum_lambda: {px.shape[1]} stocks x {px.shape[0]} dates  "
              f"mode={mode}  cache={'on' if engine is not None else 'off'}  "
              f"model_version={model_version}", flush=True)

    kw = dict(mom_long=mom_long, mom_skip=mom_skip, min_stocks=min_stocks,
              min_reg_obs=min_reg_obs, z_clip=z_clip, verbose=verbose)

    # -- no cache: plain full compute -----------------------------------------
    if engine is None:
        lambda_df, residuals = _compute_lambdas(px, None, **kw)
        full = lambda_df
    else:
        _ensure_table(engine)
        compute_from = None
        if mode == 'rebuild':
            _wipe_all(engine, model_version)
            if verbose:
                print("  rebuild: wiped cached lambdas", flush=True)
        else:
            # Provisional-date discipline. Every run caches up to ITS last price
            # date, so the LAST CACHED date was provisional when written (possibly
            # an intraday price that has since moved or been finalised). Wipe and
            # recompute from that date -- not merely from the current last price
            # date -- so a stale intraday lambda can never survive as "settled".
            # Anything cached beyond the price frame is dropped too.
            last_cached = _last_cached_date(engine, model_version)
            wipe_from = (min(last_cached, last_px_dt) if last_cached is not None
                         else last_px_dt)
            _delete_from(engine, model_version, wipe_from)
            if last_cached is not None:
                compute_from = wipe_from
            if verbose:
                print(f"  incremental: recomputing from "
                      f"{pd.Timestamp(wipe_from).date()} (last cached date was "
                      f"provisional when written)", flush=True)

        lambda_df, residuals = _compute_lambdas(px, compute_from, **kw)
        _save(engine, model_version, lambda_df)
        full = load_momentum_lambda(engine, model_version)

    level = LEVEL_BASE * (1.0 + full['lambda']).cumprod()
    level.name = LEVEL_NAME

    if verbose and len(full):
        ann_ret = full['lambda'].mean() * 252
        ann_vol = full['lambda'].std() * np.sqrt(252)
        print(f"  lambda: {len(full)} days "
              f"{full.index[0].date()}..{full.index[-1].date()}  "
              f"(computed this run: {len(lambda_df)})  "
              f"ann.ret={ann_ret*100:+.2f}%  ann.vol={ann_vol*100:.2f}%  "
              f"mean n={full['n_obs'].mean():.0f}  "
              f"mean R2={full['r2'].mean():.3f}", flush=True)

    return {'lambda_df': full, 'residuals': residuals, 'level': level}


if __name__ == "__main__":
    print(__doc__)
