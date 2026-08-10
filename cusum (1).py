#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CUSUM

"""
cusum_vol_regimes.py
====================
Point-in-time two-sided volatility CUSUM on level series (designed for the
CUSTOM_VOLQ1 / CUSTOM_VOLQ5 quintile indices already cached by
hmm_vol_regimes.py, but works on any price/level DataFrame).

Detector (per asset, per reference window W in {1,3,5} years)
-------------------------------------------------------------
Daily log returns r_t. Reference (in-control) vol sigma0_t = rolling std of
r over the trailing W-year window ENDING AT t-1 -- today's return is the
monitored observation and is never part of its own reference (PIT).

Standardize z_t = r_t / sigma0_t. Log-likelihood-ratio CUSUM for a variance
shift rho = (sigma1/sigma0)^2:

    l_t(rho) = 0.5 * [ (1 - 1/rho) * z_t^2 - ln(rho) ]
    S_t      = max(0, S_{t-1} + l_t)        alarm when S_t > h

Two one-sided detectors:
    UP   rho = CUSUM_SHIFT^2      (vol rising to SHIFT x sigma0)  -> risk alarm
    DOWN rho = 1 / CUSUM_SHIFT^2  (vol falling to sigma0 / SHIFT) -> all-clear

After an alarm the statistic resets to 0 (re-arm). The recursion is
sequential, so the statistic is PIT by construction; the only estimated
quantity is sigma0, which uses trailing data only.

Thresholds h are Monte-Carlo calibrated so the in-control average run length
(ARL0) is ~500 trading days (~2y between false alarms per detector), for
CUSUM_SHIFT = 1.5:   h_up = 3.374,  h_dn = 3.888.
`calibrate_h()` reproduces/re-derives these for other shifts or ARL budgets.

Cache
-----
Table cusum_regime_cache, PK (model_version, asset, date, window_years):
    sigma0, stat_up, stat_dn, alarm_up, alarm_dn
Incremental mode resumes each (asset, window) recursion from the last cached
row (stat_up/stat_dn persisted per day); rebuild deletes the asset's rows for
MODEL_VERSION and recomputes from start_date.

Usage
-----
    from cusum_vol_regimes import run_cusum_cache, load_cusum_results
    # qidx = run_regime_analysis(...) return value, or any levels frame
    run_cusum_cache(qidx, start_date='2017-01-01', mode='rebuild')
    res = load_cusum_results(assets=['CUSTOM_VOLQ5', 'CUSTOM_VOLQ1'])
"""

import numpy as np
import pandas as pd
# sqlalchemy imported lazily inside the DB functions so the core detector is
# importable (and unit-testable) in environments without DB drivers.

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_VERSION   = 'cusum_v1'
CUSUM_TBL       = 'cusum_regime_cache'
WINDOWS_YEARS   = (1, 3, 5)              # same reference windows as the HMM
TRADING_DAYS_YR = 252
CUSUM_SHIFT     = 1.5                    # target vol shift (multiplicative)
H_UP            = 3.374                  # MC-calibrated, ARL0 ~ 500d (shift=1.5)
H_DN            = 3.888                  # MC-calibrated, ARL0 ~ 500d
MIN_REF_OBS     = 60                     # min obs before sigma0 is trusted
DEFAULT_CONN    = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'

RHO_UP = CUSUM_SHIFT ** 2
RHO_DN = 1.0 / CUSUM_SHIFT ** 2


# ── Core recursion ────────────────────────────────────────────────────────────
def _llr_inc(z2: float, rho: float) -> float:
    return 0.5 * ((1.0 - 1.0 / rho) * z2 - np.log(rho))


def cusum_series(returns: pd.Series,
                 window_years: int,
                 start_date=None,
                 init_up: float = 0.0,
                 init_dn: float = 0.0) -> pd.DataFrame:
    """
    Run the two-sided vol-CUSUM over `returns` (daily log returns, NaN-free
    index of one asset). Emits rows from `start_date` onward (or from the
    first date with a trustworthy reference).

    init_up/init_dn: statistic values to resume from (incremental mode) --
    they are the cached S values of the day BEFORE the first emitted date.

    Returns DataFrame indexed by date:
        sigma0, stat_up, stat_dn, alarm_up, alarm_dn
    """
    r = returns.dropna()
    if len(r) < MIN_REF_OBS + 1:
        return pd.DataFrame(columns=['sigma0', 'stat_up', 'stat_dn',
                                     'alarm_up', 'alarm_dn'])
    W = int(window_years * TRADING_DAYS_YR)

    # PIT reference: rolling std over the window ENDING AT t-1 (shift(1)).
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
            continue                      # reference not ready -> no monitoring
        z2 = (vals[i] / s0) ** 2
        su = max(0.0, su + _llr_inc(z2, RHO_UP))
        sd = max(0.0, sd + _llr_inc(z2, RHO_DN))
        a_up = su > H_UP
        a_dn = sd > H_DN
        if idx[i] >= start:
            rows.append((idx[i], s0, su, sd, a_up, a_dn))
        if a_up: su = 0.0                 # reset-to-zero after alarm (re-arm)
        if a_dn: sd = 0.0
    out = pd.DataFrame(rows, columns=['date', 'sigma0', 'stat_up', 'stat_dn',
                                      'alarm_up', 'alarm_dn'])
    return out.set_index('date')


# ── Threshold calibration (reproducibility / other shifts) ────────────────────
def calibrate_h(arl0_target: float = 500.0, rho: float = RHO_UP,
                n_paths: int = 3000, max_len: int = 6000,
                seed: int = 20260710) -> float:
    """Bisect h so the in-control ARL under N(0,1) z's hits arl0_target."""
    rng = np.random.default_rng(seed)
    c1, c2 = 0.5 * (1 - 1 / rho), 0.5 * np.log(rho)

    def arl(h):
        tot = 0.0
        for _ in range(n_paths):
            z2 = rng.standard_normal(max_len) ** 2
            s, t = 0.0, max_len
            for j in range(max_len):
                s = max(0.0, s + c1 * z2[j] - c2)
                if s > h:
                    t = j + 1
                    break
            tot += t
        return tot / n_paths

    lo, hi = 0.5, 15.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if arl(mid) < arl0_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── Postgres cache ────────────────────────────────────────────────────────────
def _ensure_table(engine):
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {CUSUM_TBL} (
                model_version VARCHAR(20)      NOT NULL,
                asset         VARCHAR(30)      NOT NULL,
                date          DATE             NOT NULL,
                window_years  INT              NOT NULL,
                sigma0        DOUBLE PRECISION,
                stat_up       DOUBLE PRECISION,
                stat_dn       DOUBLE PRECISION,
                alarm_up      BOOLEAN,
                alarm_dn      BOOLEAN,
                PRIMARY KEY (model_version, asset, date, window_years)
            )"""))


def _last_cached(engine, asset, window_years):
    from sqlalchemy import text
    q = f"""SELECT date, stat_up, stat_dn FROM {CUSUM_TBL}
            WHERE model_version=:mv AND asset=:a AND window_years=:w
            ORDER BY date DESC LIMIT 1"""
    with engine.connect() as conn:
        row = conn.execute(text(q), {'mv': MODEL_VERSION, 'a': asset,
                                     'w': window_years}).fetchone()
    return (pd.Timestamp(row[0]), float(row[1]), float(row[2])) if row else None


def run_cusum_cache(levels: pd.DataFrame,
                    start_date,
                    mode: str = 'incremental',
                    conn_str: str = DEFAULT_CONN,
                    engine=None,
                    verbose: bool = True):
    """
    Compute + cache the two-sided vol-CUSUM for every column of `levels`
    (price/level series), per reference window in WINDOWS_YEARS.

    mode='rebuild'     : delete the asset's rows (MODEL_VERSION) and recompute
                         from start_date
    mode='incremental' : resume each (asset, window) recursion from the last
                         cached statistic; only new dates are computed/stored
    """
    from sqlalchemy import create_engine, text
    if engine is None:
        engine = create_engine(conn_str)
    _ensure_table(engine)
    start_date = pd.Timestamp(start_date)
    if not isinstance(levels.index, pd.DatetimeIndex):
        levels = levels.copy()
        levels.index = pd.to_datetime(levels.index)
    levels = levels.sort_index()

    if verbose:
        print(f"  [CUSUM] shift={CUSUM_SHIFT}x  h_up={H_UP}  h_dn={H_DN}  "
              f"windows={WINDOWS_YEARS}  mode={mode}")

    for asset in levels.columns:
        r = np.log(levels[asset]).diff().dropna()
        if mode == 'rebuild':
            with engine.begin() as conn:
                conn.execute(text(
                    f"DELETE FROM {CUSUM_TBL} WHERE model_version=:mv "
                    f"AND asset=:a"), {'mv': MODEL_VERSION, 'a': asset})

        n_rows = 0
        for w in WINDOWS_YEARS:
            emit_from = start_date
            last = _last_cached(engine, asset, w) if mode == 'incremental' else None
            if last is not None:
                last_dt, init_up, init_dn = last
                emit_from = last_dt + pd.Timedelta(days=1)
                if r.index.max() <= last_dt:
                    continue              # nothing new for this window
                df = _resumed_series(r, w, last_dt, init_up, init_dn)
            else:
                df = cusum_series(r, w, start_date=start_date)
            if df.empty:
                continue
            df = df[df.index >= emit_from]
            if df.empty:
                continue
            recs = [{'mv': MODEL_VERSION, 'a': asset, 'd': d.date(), 'w': w,
                     's0': float(row.sigma0), 'su': float(row.stat_up),
                     'sd': float(row.stat_dn), 'au': bool(row.alarm_up),
                     'ad': bool(row.alarm_dn)}
                    for d, row in df.iterrows()]
            with engine.begin() as conn:
                conn.execute(text(f"""
                    INSERT INTO {CUSUM_TBL}
                    (model_version, asset, date, window_years,
                     sigma0, stat_up, stat_dn, alarm_up, alarm_dn)
                    VALUES (:mv, :a, :d, :w, :s0, :su, :sd, :au, :ad)
                    ON CONFLICT (model_version, asset, date, window_years)
                    DO UPDATE SET sigma0=:s0, stat_up=:su, stat_dn=:sd,
                                  alarm_up=:au, alarm_dn=:ad"""), recs)
            n_rows += len(recs)
        if verbose:
            n_al = 0
            with engine.connect() as conn:
                n_al = conn.execute(text(
                    f"""SELECT COUNT(*) FROM {CUSUM_TBL}
                        WHERE model_version=:mv AND asset=:a
                          AND (alarm_up OR alarm_dn)"""),
                    {'mv': MODEL_VERSION, 'a': asset}).scalar()
            print(f"  [CUSUM] {asset}: {n_rows} rows written "
                  f"({n_al} alarm days total in cache)")


def _resumed_series(r: pd.Series, window_years: int, last_dt: pd.Timestamp,
                    init_up: float, init_dn: float) -> pd.DataFrame:
    """Incremental resume: reference sigma0 uses full trailing history, but the
    recursion is fed ONLY returns after last_dt, starting from the cached
    statistics (exactly reproducing an uninterrupted run)."""
    W = int(window_years * TRADING_DAYS_YR)
    sigma0 = r.rolling(W, min_periods=MIN_REF_OBS).std(ddof=1).shift(1)
    su, sd = float(init_up), float(init_dn)
    rows = []
    mask = r.index > last_dt
    for dt, ret in r[mask].items():
        s0 = sigma0.loc[dt]
        if not np.isfinite(s0) or s0 <= 0:
            continue
        z2 = (ret / s0) ** 2
        su = max(0.0, su + _llr_inc(z2, RHO_UP))
        sd = max(0.0, sd + _llr_inc(z2, RHO_DN))
        a_up, a_dn = su > H_UP, sd > H_DN
        rows.append((dt, s0, su, sd, a_up, a_dn))
        if a_up: su = 0.0
        if a_dn: sd = 0.0
    return pd.DataFrame(rows, columns=['date', 'sigma0', 'stat_up', 'stat_dn',
                                       'alarm_up', 'alarm_dn']).set_index('date')


def _mc_prob_alarm(S0: float, h: float, rho: float, sigma_z: float,
                   horizons, n_paths: int = 20000,
                   seed: int = 20260811) -> dict:
    """Monte-Carlo P(alarm within each horizon) for a ONE-SIDED CUSUM starting
    at statistic S0, threshold h, LLR variance-shift parameter rho, when the
    forward standardized returns z are drawn N(0, sigma_z^2).

    sigma_z encodes the forward vol NULL:
        in-control : sigma_z = 1
        shifted    : UP  side -> sigma_z = CUSUM_SHIFT      (vol elevated)
                     DOWN side -> sigma_z = 1/CUSUM_SHIFT   (vol compressed)
    Returns {horizon: probability}. Alarm = S crosses h at least once within
    the horizon; after a crossing that path is counted (we don't reset -- we
    want first-passage probability, not steady-state).
    """
    horizons = sorted(int(x) for x in horizons)
    Hmax = horizons[-1]
    rng = np.random.default_rng(seed)
    # simulate all paths at once, step by step, recording first-passage day
    S = np.full(n_paths, float(S0))
    hit_day = np.full(n_paths, -1, dtype=int)   # -1 = not yet hit
    for j in range(Hmax):
        z2 = (rng.standard_normal(n_paths) * sigma_z) ** 2
        S = np.maximum(0.0, S + _llr_inc(z2, rho))
        newly = (hit_day < 0) & (S > h)
        hit_day[newly] = j + 1
    out = {}
    for H in horizons:
        out[H] = float(np.mean((hit_day >= 1) & (hit_day <= H)))
    return out


def _recent_increment(r: pd.Series, window_years: int, side: str,
                      n_recent: int = 10) -> float:
    """Mean per-step CUSUM increment over the last `n_recent` monitored days,
    for the given side ('up'/'dn'). Recomputed exactly from returns via the LLR
    (independent of alarm resets, which only affect the running S, not the
    per-step increment). Returns the mean increment (can be negative)."""
    W = int(window_years * TRADING_DAYS_YR)
    sigma0 = r.rolling(W, min_periods=MIN_REF_OBS).std(ddof=1).shift(1)
    rho = RHO_UP if side == 'up' else RHO_DN
    incs = []
    for dt, ret in r.items():
        s0 = sigma0.loc[dt]
        if not np.isfinite(s0) or s0 <= 0:
            continue
        z2 = (ret / s0) ** 2
        incs.append(_llr_inc(z2, rho))
    if len(incs) < 1:
        return np.nan
    tail = incs[-int(n_recent):] if len(incs) >= n_recent else incs
    return float(np.mean(tail))


def cusum_proximity(levels: pd.DataFrame,
                    assets=None,
                    conn_str: str = DEFAULT_CONN,
                    engine=None,
                    n_recent: int = 10,
                    horizons=(5, 10, 20),
                    n_paths: int = 20000,
                    verbose: bool = True) -> pd.DataFrame:
    """How close is each asset to a CUSUM trigger, RIGHT NOW (latest cached date
    per asset/window). Three proximity views per side (UP=risk/vol-rising,
    DOWN=all-clear/vol-falling):

      (1) prox     = stat / h           (0..1+, >=1 means already alarmed)
      (2) eta      = (h - stat) / mean(last n_recent increments)
                     -> est. trading days to alarm at the recent pace;
                        NaN/inf when the statistic is flat or moving away
      (3) p_alarm  = MC P(alarm within {horizons} days) under TWO forward nulls:
                       in-control (vol unchanged) and shifted (the regime the
                       detector targets)

    `levels` supplies the price series (needed for the pace estimate and to
    align with the cached statistics). `assets` defaults to all columns.
    Returns ONE ROW PER (asset, window_years) for the latest cached date.
    """
    from sqlalchemy import create_engine, text
    if engine is None:
        engine = create_engine(conn_str)
    if assets is None:
        assets = list(levels.columns)
    if not isinstance(levels.index, pd.DatetimeIndex):
        levels = levels.copy(); levels.index = pd.to_datetime(levels.index)
    levels = levels.sort_index()

    rows = []
    for asset in assets:
        if asset not in levels.columns:
            if verbose:
                print(f"  [prox] {asset}: not in levels frame -- skipped")
            continue
        r = np.log(levels[asset]).diff().dropna()
        for w in WINDOWS_YEARS:
            last = _last_cached(engine, asset, w)
            if last is None:
                if verbose:
                    print(f"  [prox] {asset} w={w}: no cached row -- skipped")
                continue
            last_dt, s_up, s_dn = last

            # (1) proximity ratios
            prox_up = s_up / H_UP
            prox_dn = s_dn / H_DN

            # (2) ETA at recent pace (returns up to the cached date)
            r_asof = r[r.index <= last_dt]
            inc_up = _recent_increment(r_asof, w, 'up', n_recent)
            inc_dn = _recent_increment(r_asof, w, 'dn', n_recent)
            eta_up = ((H_UP - s_up) / inc_up) if (np.isfinite(inc_up) and inc_up > 0) else np.nan
            eta_dn = ((H_DN - s_dn) / inc_dn) if (np.isfinite(inc_dn) and inc_dn > 0) else np.nan

            # (3) MC probabilities, both nulls, all horizons
            p_up_ic = _mc_prob_alarm(s_up, H_UP, RHO_UP, 1.0, horizons, n_paths)
            p_up_sh = _mc_prob_alarm(s_up, H_UP, RHO_UP, CUSUM_SHIFT, horizons, n_paths)
            p_dn_ic = _mc_prob_alarm(s_dn, H_DN, RHO_DN, 1.0, horizons, n_paths)
            p_dn_sh = _mc_prob_alarm(s_dn, H_DN, RHO_DN, 1.0 / CUSUM_SHIFT, horizons, n_paths)

            rec = {
                'asset': asset, 'date': last_dt, 'window_years': w,
                'stat_up': s_up, 'stat_dn': s_dn,
                'prox_up': prox_up, 'prox_dn': prox_dn,
                'eta_up': eta_up, 'eta_dn': eta_dn,
                'inc_up': inc_up, 'inc_dn': inc_dn,
            }
            for H in horizons:
                rec[f'p_up_ic_{H}'] = p_up_ic[H]
                rec[f'p_up_sh_{H}'] = p_up_sh[H]
                rec[f'p_dn_ic_{H}'] = p_dn_ic[H]
                rec[f'p_dn_sh_{H}'] = p_dn_sh[H]
            rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['asset', 'window_years']).reset_index(drop=True)
    if verbose and not out.empty:
        print(f"  [prox] computed for {out['asset'].nunique()} asset(s), "
              f"{len(out)} (asset,window) row(s), latest date per asset/window.")
    return out


def load_cusum_results(assets,
                       conn_str: str = DEFAULT_CONN,
                       engine=None) -> pd.DataFrame:
    """Long frame of cached CUSUM rows for the given assets."""
    from sqlalchemy import create_engine, text
    if engine is None:
        engine = create_engine(conn_str)
    q = f"""SELECT asset, date, window_years, sigma0, stat_up, stat_dn,
                   alarm_up, alarm_dn
            FROM {CUSUM_TBL}
            WHERE model_version = :mv AND asset = ANY(:assets)
            ORDER BY asset, date, window_years"""
    return pd.read_sql(text(q), engine,
                       params={'mv': MODEL_VERSION, 'assets': list(assets)},
                       parse_dates=['date'])



# qidx = the quintile levels frame (run_regime_analysis return, or however you hold it)
run_cusum_cache(qidx[['CUSTOM_VOLQ5', 'CUSTOM_VOLQ1']],
                start_date='2017-01-01',   # match the HMM cache start
                mode='incremental')            # first run; 'incremental' daily after

res = load_cusum_results(['CUSTOM_VOLQ5', 'CUSTOM_VOLQ1'])
res[res.alarm_up].groupby('window_years').date.count()   # quick footprint check

cusumHVQup1 = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 1]['alarm_up'].replace({True:1,False:0})
cusumHVQup3 = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 3]['alarm_up'].replace({True:1,False:0})
cusumHVQup5 = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 5]['alarm_up'].replace({True:1,False:0})
cusumHVQup1.index = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 1]['date']
cusumHVQup3.index = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 3]['date']
cusumHVQup5.index = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 5]['date']

cusumSup = 1 - (cusumHVQup1 + cusumHVQup3 + cusumHVQup5).clip(0, 1)

cusumSup.plot(figsize = (12, 8), c = 'r')
plt.show()

cusumHVQdown1 = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 1]['alarm_dn'].replace({True:1,False:0})
cusumHVQdown3 = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 3]['alarm_dn'].replace({True:1,False:0})
cusumHVQdown5 = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 5]['alarm_dn'].replace({True:1,False:0})
cusumHVQdown1.index = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 1]['date']
cusumHVQdown3.index = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 3]['date']
cusumHVQdown5.index = res[res['asset']=='CUSTOM_VOLQ5'][res[res['asset']=='CUSTOM_VOLQ5']['window_years'] == 5]['date']

cusumSdown = 1 - (cusumHVQdown1 + cusumHVQdown3 + cusumHVQdown5).clip(0, 1)

cusumSdown.plot(figsize = (12, 8), c = 'b')
plt.show()

