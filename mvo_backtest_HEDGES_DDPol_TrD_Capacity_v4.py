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
warnings.filterwarnings('ignore')

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

# ===============================================================================
# PARAMETERS
# ===============================================================================

MB_START_DATE  = pd.Timestamp('2019-01-01')
MB_TOP_N       = 20
MB_REBAL_FREQ  = 30
MB_MODEL_VER   = 'v2'

# Cache table for precomputed covariance matrices and X snapshots
MB_COV_CACHE_TBL    = 'mvo_cov_cache'
MB_DAILY_PORT_TBL   = 'mvo_daily_portfolios'  # daily portfolio cache
MB_MIN_COV_MATRICES  = 2      # default: stock must appear in >=2 matrices (0=alpha-only)

# Smart hybrid drawdown thresholds (entry into defensive regimes)
SH_DD_ALPHA    = 0.075   # dd below this: shift from alpha to hybrid
SH_DD_HYBRID   = 0.175   # dd below this: shift from hybrid to MVO
# Asymmetric exit thresholds (recovery needed before shifting back)
SH_DD_EXIT_ALPHA  = 0.050  # dd must recover above this to exit hybrid -> alpha
SH_DD_EXIT_HYBRID = 0.150  # dd must recover above this to exit MVO -> hybrid
# Regime persistence (days signal must persist before switching)
SH_PERSIST_DAYS   = 3      # days signal must persist before regime switch

# Dynamic rebalancing thresholds
DYN_TO_THRESHOLD_ALPHA  = 0.20   # TO trigger in alpha regime
DYN_TO_THRESHOLD_HYBRID = 0.25   # TO trigger in hybrid regime
DYN_TO_THRESHOLD_MVO    = 0.30   # TO trigger in MVO regime (higher bar)
DYN_VOLDIFF_CAP    = 0.175   # max vol increase allowed alongside TO trigger
DYN_VOLDIFF_DERISK = -0.750  # de-risk trigger (effectively disabled at -75%)
DYN_MIN_HOLD_DAYS  = 7       # minimum calendar days between rebalances

# Drawdown policy (8th strategy: Dynamic + Hedge + DD)
# Drawdown policy — flexible multi-level de-grossing
# Each tuple: (dd_threshold, fraction_of_remaining_to_cut)
# Cuts compound: e.g. 100% → 80% → 60% → 42% → 21% → 10.5%
DD_LEVELS = [
    (0.175, 2/5),  # at -17.5% DD: cut 40% of remaining → 60% exposed
    (0.300, 3/7),  # at -30.0% DD: cut 43% of remaining → 34% exposed
    (0.350, 1/2),  # at -35.0% DD: cut 50% of remaining → 17% exposed
    (0.400, 2/3),  # at -40.0% DD: cut 67% of remaining →  6% exposed
    (0.450, 1/1),  # at -45.0% DD: cut 100% of remaining → 0% exposed
]
DD_REENTRY_PCT          = 0.075  # theoretical recovery from trough → full re-entry
DD_REENTRY_CONFIRM      = 5      # days recovery must persist before re-entry
DD_ANNUAL_RESET_PCT     = 0.30   # max YTD DD allowed for calendar-year re-entry

# Daily cache builder
VOL_LOOKBACK      = 63                   # trading days for rolling vol
DAILY_PORT_TBL    = 'mvo_daily_portfolios'
DAILY_TRIGGER_TBL = 'mvo_daily_triggers'

# Trading costs
TRADING_COST_BPS = 10    # one-way cost in basis points (10bps = 0.10%)
VOLUME_WINDOW    = 10        # rolling mean window for volume de-trending
ADV_WINDOW       = 20        # days for median ADV calculation (dollar volume)
AUM              = 1_000_000 # starting AUM in dollars (2019-01-01)


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
                             pca_var_threshold, X_df_cached=None):
    """
    Build all four covariance matrices for the candidate universe on date dt.
    Returns (Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens).
    """
    # Return matrix
    ret_df = (Pxs_df[candidates].pct_change()
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
        F_mat, factor_names_f, sec_cols_f = _mb_build_F(model_version)
        factor_names_f2, sec_cols_f2 = _mb_get_factor_names(model_version)
        X_df = pd.DataFrame(0.0, index=valid, columns=factor_names_f)
        # Use most recent X snapshot if available, else build on the fly
        if X_df_cached is not None:
            X_df = X_df_cached.reindex(index=valid, columns=factor_names_f).fillna(0.0)
        else:
            pxs_slice = Pxs_df.loc[:Pxs_df.index[-1]]
            X_built   = _mb_build_X(Pxs_df.index[-1], valid, factor_names_f,
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
_MB_F_LOOKBACK  = 252
_MB_F_EWMA_HL   = 42


def _mb_zscore(s):
    mu, sd = s.mean(), s.std()
    return s * 0.0 if (sd == 0 or np.isnan(sd)) else (s - mu) / sd


def _mb_get_factor_names(model_version):
    scalar_names = list(_MB_SCALAR_TBLS[model_version].keys())
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


def _mb_build_F(model_version):
    """Self-contained F matrix builder."""
    tbls   = _MB_SCALAR_TBLS[model_version]
    frames = []
    for fname, tbl in tbls.items():
        try:
            with ENGINE.connect() as conn:
                df = pd.read_sql(
                    f"SELECT * FROM {tbl} ORDER BY date DESC LIMIT {_MB_F_LOOKBACK}",
                    conn
                )
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            num = df.drop(columns=[c for c in _MB_LAMBDA_META if c in df.columns],
                          errors='ignore').select_dtypes(include=np.number)
            col = num.iloc[:, 0].rename(fname)
            frames.append(col)
        except Exception as e:
            warnings.warn(f"Lambda table {tbl} failed: {e}")

    macro_tbl_map = {'v1': 'factor_lambdas_macro', 'v2': 'v2_factor_lambdas_macro'}
    try:
        macro_tbl = macro_tbl_map[model_version]
        with ENGINE.connect() as conn:
            mdf = pd.read_sql(
                f"SELECT * FROM {macro_tbl} ORDER BY date DESC LIMIT {_MB_F_LOOKBACK}",
                conn
            )
        mdf['date'] = pd.to_datetime(mdf['date'])
        mdf = mdf.set_index('date').sort_index()
        mnum = mdf.drop(columns=[c for c in _MB_LAMBDA_META if c in mdf.columns],
                        errors='ignore').select_dtypes(include=np.number)
        for mc in MACRO_COLS:
            if mc in mnum.columns:
                frames.append(mnum[mc].rename(mc))
    except Exception as e:
        warnings.warn(f"Macro lambda failed: {e}")

    try:
        with ENGINE.connect() as conn:
            sdf = pd.read_sql(
                f"SELECT * FROM {_MB_SEC_TBL[model_version]} "
                f"ORDER BY date DESC LIMIT {_MB_F_LOOKBACK}", conn
            )
        sdf['date'] = pd.to_datetime(sdf['date'])
        sdf = sdf.set_index('date').sort_index()
        sec_cols = sorted([c for c in sdf.columns
                           if c not in _MB_LAMBDA_META
                           and pd.api.types.is_float_dtype(sdf[c])])
        for sc in sec_cols:
            frames.append(sdf[sc].rename(sc))
    except Exception as e:
        warnings.warn(f"Sector lambda failed: {e}")
        sec_cols = []

    combined = pd.concat(frames, axis=1).dropna()
    factor_names, sec_cols = _mb_get_factor_names(model_version)
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
    """Self-contained X matrix builder -- no _rd_build_X dependency."""
    pxs_to_dt = Pxs_df.loc[:dt]
    if len(pxs_to_dt) < BETA_WINDOW // 2:
        return None
    calc_dates = pd.DatetimeIndex([dt])
    try:
        beta_df = calc_rolling_betas(pxs_to_dt, universe, calc_dates)
        beta_s  = _mb_zscore(beta_df.iloc[-1].reindex(universe)).rename('Beta')

        size_df = load_dynamic_size(universe, pxs_to_dt, calc_dates)
        size_s  = _mb_zscore(
            np.log(size_df.iloc[-1].reindex(universe).clip(lower=1e-6))
        ).rename('Size')

        macro_betas  = calc_macro_betas(pxs_to_dt, universe, calc_dates)
        macro_series = {}
        for mc in MACRO_COLS:
            if mc in macro_betas and not macro_betas[mc].empty:
                macro_series[mc] = _mb_zscore(
                    macro_betas[mc].iloc[-1].reindex(universe)
                ).rename(mc)
            else:
                macro_series[mc] = pd.Series(0.0, index=universe, name=mc)

        dummies = _mb_sector_dummies(universe, sectors_s, sec_cols)

        quality_df = load_quality_scores(universe, calc_dates, pxs_to_dt, sectors_s)
        quality_s  = _mb_zscore(
            quality_df.reindex(calc_dates).iloc[-1].reindex(universe)
        ).rename('Quality')

        recent_dates = pxs_to_dt.index[-60:]
        si_full = load_si_composite(universe, recent_dates)
        si_s    = _mb_zscore(
            si_full.reindex(recent_dates).ffill().iloc[-1].reindex(universe)
        ).rename('SI')

        open_df, high_df, low_df = load_ohlc_tables(universe)
        vol_df = calc_vol_factor(pxs_to_dt, universe, calc_dates,
                                 open_df=open_df, high_df=high_df, low_df=low_df)
        vol_s  = _mb_zscore(vol_df.iloc[-1].reindex(universe)).rename('GK_Vol')

        try:
            with ENGINE.connect() as conn:
                res_mom = pd.read_sql(
                    f"SELECT * FROM {_MB_MOM_RESID[model_version]} ORDER BY date",
                    conn
                )
            res_mom['date'] = pd.to_datetime(res_mom['date'])
            if 'ticker' in res_mom.columns and 'resid' in res_mom.columns:
                res_mom = res_mom.pivot_table(
                    index='date', columns='ticker',
                    values='resid', aggfunc='last'
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

        value_df = load_value_scores(universe, calc_dates, sectors_s)
        value_s  = _mb_zscore(
            value_df.reindex(calc_dates).iloc[-1].reindex(universe)
        ).rename('Value')

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

MB_X_CACHE_TBL = 'mvo_x_snapshots'


def _mb_save_x_snapshot(dt, model_version, xdf):
    """Persist a single X snapshot to the DB cache table (upsert)."""
    import json
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
    import json
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
                            force_rebuild):
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
    factor_names, sec_cols = _mb_get_factor_names(model_version)

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

def _make_make_params_hash(ic, max_weight, min_weight, zscore_cap, pca_var_threshold,
                      universe_mult, risk_aversion, top_n, conc_factor,
                      prefilt_pct, min_cov_matrices, model_version):
    """Stable 12-char hash of all portfolio construction parameters."""
    import hashlib, json
    params = dict(
        ic=ic, maxw=max_weight, minw=min_weight, zc=zscore_cap,
        pca=pca_var_threshold, um=universe_mult, ra=risk_aversion,
        n=top_n, conc=conc_factor, pf=round(prefilt_pct, 4),
        mcm=min_cov_matrices, mv=model_version,
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
                            w_alpha, w_mvo, w_hybrid):
    import json
    for strategy, w in [('alpha', w_alpha), ('mvo', w_mvo), ('hybrid', w_hybrid)]:
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
    import json
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
                   top_n=20, min_cov_matrices=2):
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

    # -- Return history --------------------------------------------------------
    ret_df = (Pxs_df[candidates].pct_change()
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
                                    X_df_cached=X_df_cached)
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

def _mb_run_nav(weights_by_date, calc_dates, Pxs_df, cost_by_date=None):
    """
    Compute NAV series from a dict of {rebal_date: pd.Series(weights)}.
    Mirrors run_backtest NAV logic from primary_factor_backtest.py.
    cost_by_date: optional dict {date: cost_fraction} applied at rebalance.
    """
    nav        = 1.0
    nav_series = {}
    portfolio  = []
    weights    = {}

    for i, rebal_date in enumerate(calc_dates):
        next_date = (calc_dates[i+1] if i+1 < len(calc_dates)
                     else Pxs_df.index.max())

        if rebal_date in weights_by_date:
            w = weights_by_date[rebal_date]
            w = w[w > 1e-6]
            if len(w) > 0 and w.sum() > 0:
                # Apply trading cost at rebalance date
                if cost_by_date and rebal_date in cost_by_date:
                    nav *= (1 - cost_by_date[rebal_date])
                portfolio = w.index.tolist()
                weights   = (w / w.sum()).to_dict()

        period_dates = Pxs_df.index[
            (Pxs_df.index >= rebal_date) & (Pxs_df.index <= next_date)
        ]

        if not portfolio or len(period_dates) < 2:
            for d in period_dates:
                nav_series[d] = nav
            continue

        px_start   = Pxs_df.loc[period_dates[0],  portfolio]
        px_end     = Pxs_df.loc[period_dates[-1], portfolio]
        stk_rets   = (px_end / px_start - 1).fillna(0)
        w_s        = pd.Series(weights).reindex(portfolio).fillna(0)
        period_ret = (stk_rets * w_s).sum()
        nav       *= (1 + period_ret)

        px_period  = Pxs_df.loc[period_dates, portfolio]
        stk_daily  = px_period.div(px_start, axis=1) - 1
        port_cum   = stk_daily.mul(w_s, axis=1).sum(axis=1)
        period_nav = nav / (1 + period_ret) * (1 + port_cum)
        for d, v in period_nav.items():
            nav_series[d] = v

    nav_s = pd.Series(nav_series).sort_index()
    if MB_START_DATE not in nav_s.index:
        nav_s[MB_START_DATE] = 1.0
        nav_s = nav_s.sort_index()
    return nav_s


# ===============================================================================
# COMPARISON PLOT
# ===============================================================================

def _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s, nav_hybrid=None,
             nav_smart=None, nav_dynamic=None,
             nav_dyn_hedged=None, nav_dd=None):
    """Four panels: NAV, relative vs baseline, drawdown, regime."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={'height_ratios': [3, 2, 2, 1]})
    fig.patch.set_facecolor('#FAFAF9')
    for ax in axes:
        ax.set_facecolor('#FAFAF9')

    GRAY  = '#888780'; TEAL = '#1D9E75'; BLUE = '#378ADD'; CORAL = '#D85A30'

    common = nav_baseline.index\
               .intersection(nav_alpha.index)\
               .intersection(nav_mvo.index)
    if nav_hybrid is not None:
        common = common.intersection(nav_hybrid.index)
    if nav_smart is not None:
        common = common.intersection(nav_smart.index)
    if nav_dynamic is not None and not nav_dynamic.empty:
        common = common.intersection(nav_dynamic.index)
    if nav_dyn_hedged is not None and not nav_dyn_hedged.empty:
        common = common.intersection(nav_dyn_hedged.index)
    if nav_dd is not None and not nav_dd.empty:
        common = common.intersection(nav_dd.index)

    nb  = nav_baseline.reindex(common)
    na  = nav_alpha.reindex(common)
    nm  = nav_mvo.reindex(common)
    nh  = nav_hybrid.reindex(common)     if nav_hybrid     is not None else None
    ns  = nav_smart.reindex(common)      if nav_smart      is not None else None
    nd  = nav_dynamic.reindex(common)    if (nav_dynamic    is not None and not nav_dynamic.empty)    else None
    ndh = nav_dyn_hedged.reindex(common) if (nav_dyn_hedged is not None and not nav_dyn_hedged.empty) else None
    ndd = nav_dd.reindex(common)         if (nav_dd         is not None and not nav_dd.empty)         else None

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
    ps = [(nb,'Baseline',GRAY,1.2),(na,'Pure alpha',TEAL,1.5),(nm,'MVO',BLUE,1.8)]
    if nh  is not None: ps.append((nh, 'Hybrid',         '#7F77DD', 1.5))
    if ns  is not None: ps.append((ns, 'Smart Hybrid',   '#E8A838', 1.8))
    if nd  is not None: ps.append((nd, 'Dynamic',        '#9B59B6', 1.8))
    if ndh is not None: ps.append((ndh,'Dyn+Hedge',      '#2ECC71', 2.0))
    if ndd is not None: ps.append((ndd,'Dyn+Hedge+DD',   '#E74C3C', 2.0))
    for nav_s, lbl, color, lw in ps:
        ax.plot(nav_s.index.to_numpy(), nav_s.values,
                label=lbl, color=color, linewidth=lw)
    ax.set_ylabel("NAV", fontsize=10, color='#5F5E5A')
    ax.set_title("Strategy NAV Comparison",
                 fontsize=12, fontweight='500', color='#2C2C2A')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Relative to baseline
    ax2 = axes[1]
    rs = [(na,'Pure alpha vs baseline',TEAL),(nm,'MVO vs baseline',BLUE)]
    if nh  is not None: rs.append((nh, 'Hybrid vs baseline',       '#7F77DD'))
    if ns  is not None: rs.append((ns, 'Smart Hybrid vs baseline', '#E8A838'))
    if nd  is not None: rs.append((nd, 'Dynamic vs baseline',      '#9B59B6'))
    if ndh is not None: rs.append((ndh,'Dyn+Hedge vs baseline',    '#2ECC71'))
    if ndd is not None: rs.append((ndd,'Dyn+Hedge+DD vs baseline', '#E74C3C'))
    for nav_s, lbl, color in rs:
        rel = (nav_s / nb - 1) * 100
        ax2.plot(rel.index.to_numpy(), rel.values,
                 label=lbl, color=color, linewidth=1.2)
        ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                         where=(rel.values >= 0), color=color, alpha=0.08)
        ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                         where=(rel.values < 0), color=CORAL, alpha=0.08)
    ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
    ax2.set_ylabel("Relative to baseline (%)", fontsize=10, color='#5F5E5A')
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.85)
    ax2.grid(color='#D3D1C7', linewidth=0.5)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # Panel 3: Drawdown
    ax3 = axes[2]
    ds = [(nb,'Baseline',GRAY),(na,'Pure alpha',TEAL),(nm,'MVO',BLUE)]
    if nh  is not None: ds.append((nh, 'Hybrid',       '#7F77DD'))
    if ns  is not None: ds.append((ns, 'Smart Hybrid', '#E8A838'))
    if nd  is not None: ds.append((nd, 'Dynamic',      '#9B59B6'))
    if ndh is not None: ds.append((ndh,'Dyn+Hedge',    '#2ECC71'))
    if ndd is not None: ds.append((ndd,'Dyn+Hedge+DD', '#E74C3C'))
    for nav_s, lbl, color in ds:
        dd = (nav_s / nav_s.cummax() - 1) * 100
        ax3.plot(dd.index.to_numpy(), dd.values,
                 label=lbl, color=color, linewidth=1.0)
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
    return fig




# ===============================================================================
# DAILY PORTFOLIO CACHE BUILDER
# ===============================================================================

# -- Suppress output -----------------------------------------------------------
# _Suppress = _SuppressOutput (already defined above)


# -- Parameter hash ------------------------------------------------------------

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
                drawdown           NUMERIC(8,6),
                live_strategy      VARCHAR(16),
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


def _save_triggers(dt, mv, ph, implied_to, vol_diff, drawdown, live_strategy):
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {DAILY_TRIGGER_TBL}
                (date, model_version, params_hash,
                 implied_turnover, vol_diff, drawdown, live_strategy)
            VALUES (:dt, :mv, :ph, :ito, :vd, :dd, :ls)
            ON CONFLICT (date, model_version, params_hash)
            DO UPDATE SET
                implied_turnover = EXCLUDED.implied_turnover,
                vol_diff         = EXCLUDED.vol_diff,
                drawdown         = EXCLUDED.drawdown,
                live_strategy    = EXCLUDED.live_strategy
        """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': mv, 'ph': ph,
               'ito': float(implied_to), 'vd': float(vol_diff),
               'dd': float(drawdown), 'ls': live_strategy})


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

# ── Default parameters ────────────────────────────────────────────────────────
BETA_WINDOW    = 63     # rolling window for beta calculation
CORR_WINDOW    = 63     # rolling window for correlation ranking
EFF_MAV_WINDOW = 20     # MAV window for smoothing effectiveness
EFF_FLOOR      = 1.0    # minimum effectiveness score to qualify
CORR_FLOOR     = 0.50   # minimum correlation to portfolio to qualify
HEDGE_RATIO    = 0.25   # hedge size per instrument (fraction of portfolio NAV)
MAX_HEDGE      = 0.50   # maximum total hedge (fraction of portfolio NAV)
TRIGGER_ASSETS = ['QQQ', 'SPY']  # assets whose signal triggers hedge on/off


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
    """Smoothed effectiveness at date dt (MAV of last mav_win values)."""
    if 'effectiveness' not in signal_df.columns:
        return np.nan
    past = signal_df['effectiveness'].loc[:dt].dropna()
    if past.empty:
        return np.nan
    return past.iloc[-mav_win:].mean()


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def run_hedge_backtest(Pxs_df:            pd.DataFrame,
                       multi:             dict,
                       portfolio_weights: dict,
                       rebal_dates:       list,
                       hedges_l:          list,
                       beta_window:       int   = BETA_WINDOW,
                       corr_window:       int   = CORR_WINDOW,
                       eff_mav_window:    int   = EFF_MAV_WINDOW,
                       eff_floor:         float = EFF_FLOOR,
                       corr_floor:        float = CORR_FLOOR,
                       hedge_ratio:       float = HEDGE_RATIO,
                       max_hedge:         float = MAX_HEDGE,
                       trigger_assets:    list  = TRIGGER_ASSETS) -> dict:
    """
    Run hedge backtest in parallel with main portfolio.

    Parameters
    ----------
    Pxs_df            : price DataFrame
    multi             : output of run_macro_hedge_cached
    portfolio_weights : {date: pd.Series} rebalance weights
    rebal_dates       : sorted list of portfolio rebalancing dates
    hedges_l          : list of hedge instrument tickers
    (all other params: see module defaults)

    Returns
    -------
    dict with keys:
      'hedge_account_by_date' : {date: float}  hedge account balance each day
      'hedge_sweep_by_date'   : {date: float}  amount swept to portfolio at rebal
      'hedge_log'             : list of event dicts
      'summary_by_year'       : pd.DataFrame
      'summary_by_instrument' : pd.DataFrame
      'port_ret_s'            : pd.Series  daily portfolio returns
    """
    results      = multi['results']
    all_dates    = Pxs_df.index
    rebal_set    = set(rebal_dates)

    # Validate trigger assets
    for ta in trigger_assets:
        if ta not in results:
            print(f"  WARNING: trigger asset {ta} not in multi results — "
                  f"will be treated as always OFF")

    # Validate hedges
    hedges_l = [h for h in hedges_l if h in results and h in Pxs_df.columns]
    missing  = [h for h in hedges_l if h not in results or h not in Pxs_df.columns]
    if missing:
        print(f"  WARNING: skipping {missing} — not in multi results or Pxs_df")

    print('=' * 72)
    print('  HEDGE BACKTEST ENGINE')
    print('=' * 72)
    print(f"\n  Trigger assets : {trigger_assets}")
    print(f"  Hedge universe : {hedges_l}")
    print(f"  Parameters     : beta_win={beta_window}d  corr_win={corr_window}d  "
          f"eff_mav={eff_mav_window}d")
    print(f"                   eff_floor={eff_floor}  corr_floor={corr_floor:.0%}  "
          f"hedge_ratio={hedge_ratio:.0%}  max_hedge={max_hedge:.0%}")

    # Build portfolio return series
    print("\n  Building portfolio return series...")
    port_ret_s = _portfolio_returns(portfolio_weights, Pxs_df, all_dates)

    # Pre-compute instrument return series
    inst_ret = {inst: Pxs_df[inst].pct_change().fillna(0)
                for inst in hedges_l if inst in Pxs_df.columns}

    # ── Main loop ─────────────────────────────────────────────────────────────
    hedge_account      = 0.0    # running hedge account balance (as % of NAV)
    hedge_sweep        = {}     # {rebal_date: amount swept to portfolio}
    hedge_account_hist = {}     # {date: balance}
    hedge_log          = []     # all hedge events

    active_hedges      = {}     # {inst: {'entry_px': float, 'weight': float}}
    hedge_on           = False  # current hedge state
    prev_trigger       = 0      # previous day's trigger signal

    print("\n  Running hedge simulation...\n")

    for dt in all_dates:
        if dt not in port_ret_s.index:
            hedge_account_hist[dt] = hedge_account
            continue

        # ── Sweep hedge account at portfolio rebalancing dates ────────────────
        # Only sweep when no active hedges — deferred if hedges still open
        if dt in rebal_set and not hedge_on:
            if hedge_account != 0.0:
                sweep_amt        = hedge_account
                hedge_sweep[dt]  = hedge_sweep.get(dt, 0) + sweep_amt
                print(f"  {'='*68}")
                print(f"  PORTFOLIO REBALANCE {dt.date()}  |  "
                      f"Hedge account sweep: {sweep_amt:+.4%} of NAV")
                print(f"  {'='*68}")
                hedge_account = 0.0
        elif dt in rebal_set and hedge_on:
            print(f"  PORTFOLIO REBALANCE {dt.date()}  |  "
                  f"Hedges active — sweep deferred")

        # ── Compute trigger signal (1-day lag) ────────────────────────────────
        trigger_on = False
        for ta in trigger_assets:
            if ta in results:
                sig_df = results[ta]['signal_df']
                if dt in sig_df.index:
                    # 1-day lag: use previous day's signal
                    prev_dates = sig_df.index[sig_df.index < dt]
                    if len(prev_dates) > 0:
                        prev_sig = sig_df.loc[prev_dates[-1], 'signal']
                        if prev_sig == 1:
                            trigger_on = True
                            break

        # ── Hedge activation ──────────────────────────────────────────────────
        if trigger_on and not hedge_on:
            # New hedge event — select instruments
            selected = _select_hedge_instruments(
                dt, hedges_l, results, port_ret_s, inst_ret,
                beta_window, corr_window, eff_mav_window,
                eff_floor, corr_floor, trigger_assets
            )

            if selected:
                hedge_on      = True
                active_hedges = {}
                n_inst        = len(selected)
                total_hedge   = min(n_inst * hedge_ratio, max_hedge)

                # Allocate by beta * effectiveness, normalized to total_hedge
                raw_scores = {inst: d['beta'] * d['effectiveness']
                              for inst, d in selected.items()
                              if not np.isnan(d['beta']) and
                                 not np.isnan(d['effectiveness'])}
                total_score = sum(raw_scores.values())

                print(f"\n  {'─'*68}")
                print(f"  HEDGE OPEN  {dt.date()}  |  "
                      f"{n_inst} instruments  total_hedge={total_hedge:.1%}")
                print(f"  {'─'*68}")
                print(f"  {'Instrument':<10}  {'Corr':>7}  {'Beta':>7}  "
                      f"{'Eff (MAV)':>10}  {'Score':>8}  {'Weight':>8}")
                print(f"  {'-'*60}")

                for inst, d in selected.items():
                    raw_sc = raw_scores.get(inst, 0)
                    w      = (raw_sc / total_score * total_hedge
                              if total_score > 0 else total_hedge / n_inst)
                    active_hedges[inst] = {
                        'weight':   w,
                        'entry_px': Pxs_df.loc[dt, inst]
                                    if dt in Pxs_df.index else np.nan,
                        'entry_dt': dt,
                        'beta':     d['beta'],
                        'corr':     d['corr'],
                        'eff':      d['effectiveness'],
                        'score':    raw_sc,
                    }
                    print(f"  {inst:<10}  {d['corr']:>7.3f}  {d['beta']:>7.3f}  "
                          f"{d['effectiveness']:>10.3f}  {raw_sc:>8.3f}  {w:>7.2%}")

                print(f"  {'─'*60}")
                print(f"  Total hedge: {total_hedge:.1%}  "
                      f"(allocated {sum(v['weight'] for v in active_hedges.values()):.1%})")

                hedge_log.append({
                    'date': dt, 'event': 'OPEN',
                    'instruments': list(active_hedges.keys()),
                    'total_hedge': total_hedge,
                    'details': {inst: dict(v) for inst, v in active_hedges.items()},
                })
            else:
                print(f"\n  {dt.date()}  Trigger ON but no qualifying instruments — "
                      f"no hedge activated")

        # ── Hedge unwind ──────────────────────────────────────────────────────
        elif not trigger_on and hedge_on:
            hedge_on = False
            print(f"\n  {'─'*68}")
            print(f"  HEDGE CLOSE {dt.date()}")
            print(f"  {'─'*68}")
            print(f"  {'Instrument':<10}  {'Entry':>12}  {'Exit':>12}  "
                  f"{'Weight':>8}  {'P&L':>10}")
            print(f"  {'-'*57}")

            total_pnl = 0.0
            for inst, h in active_hedges.items():
                if dt in Pxs_df.index and h['entry_px'] and not np.isnan(h['entry_px']):
                    exit_px      = Pxs_df.loc[dt, inst]
                    inst_ret_pct = (exit_px / h['entry_px'] - 1)
                    # Short position: profit when instrument falls
                    # Deduct 10bps entry + 10bps exit = 20bps round-trip per instrument
                    trade_cost = 2 * TRADING_COST_BPS / 10000 * h['weight']
                    pnl        = -inst_ret_pct * h['weight'] - trade_cost
                    total_pnl += pnl
                    print(f"  {inst:<10}  {h['entry_px']:>12.4f}  {exit_px:>12.4f}  "
                          f"{h['weight']:>8.2%}  {pnl:>+9.4%}")
                else:
                    pnl = 0.0
                    print(f"  {inst:<10}  {'n/a':>12}  {'n/a':>12}  "
                          f"{h['weight']:>8.2%}  {'n/a':>10}")

            hedge_account += total_pnl
            total_costs = sum(2 * TRADING_COST_BPS / 10000 * h['weight']
                              for h in active_hedges.values())
            print(f"  {'─'*57}")
            print(f"  Episode P&L : {total_pnl:+.4%}  (incl. {total_costs*100:.2f}bps round-trip costs)")
            print(f"  Hedge acct  : {hedge_account:+.4%} (pending sweep at next rebal)")

            days_held = (dt - active_hedges[list(active_hedges.keys())[0]]['entry_dt']).days
            hedge_log.append({
                'date': dt, 'event': 'CLOSE',
                'instruments': list(active_hedges.keys()),
                'total_pnl': total_pnl,
                'days_held': days_held,
                'hedge_account': hedge_account,
                'details': {inst: dict(v) for inst, v in active_hedges.items()},
            })
            active_hedges = {}

        # ── Mark-to-market active hedges (display only, not booked to account) ─
        elif hedge_on and active_hedges:
            daily_pnl = 0.0
            prev_dates_all = all_dates[all_dates < dt]
            if len(prev_dates_all) > 0:
                prev_dt = prev_dates_all[-1]
                for inst, h in active_hedges.items():
                    if (inst in Pxs_df.columns and
                            prev_dt in Pxs_df.index and dt in Pxs_df.index):
                        r = Pxs_df.loc[dt, inst] / Pxs_df.loc[prev_dt, inst] - 1
                        daily_pnl += -r * h['weight']
            # MTM tracked separately — not added to hedge_account
            # hedge_account only updates at close

        hedge_account_hist[dt] = hedge_account

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_hedge_summary(hedge_log, Pxs_df, results)

    return {
        'hedge_account_by_date': hedge_account_hist,
        'hedge_sweep_by_date':   hedge_sweep,
        'hedge_log':             hedge_log,
        'port_ret_s':            port_ret_s,
        'active_hedges':         active_hedges,
    }


# ================================================================================
# INSTRUMENT SELECTION
# ================================================================================

def _select_hedge_instruments(dt:            pd.Timestamp,
                               hedges_l:     list,
                               results:      dict,
                               port_ret_s:   pd.Series,
                               inst_ret:     dict,
                               beta_window:  int,
                               corr_window:  int,
                               eff_mav_win:  int,
                               eff_floor:    float,
                               corr_floor:   float,
                               trigger_assets: list) -> dict:
    """
    Select qualifying hedge instruments at activation date dt.
    Returns dict {inst: {'beta', 'corr', 'effectiveness'}} for selected instruments.
    """
    # Compute corr and effectiveness for all instruments
    scores = {}
    for inst in hedges_l:
        if inst not in inst_ret or inst not in results:
            continue
        corr = _compute_corr(port_ret_s, inst_ret[inst], corr_window)
        eff  = _get_effectiveness(results[inst]['signal_df'], dt, eff_mav_win)
        beta = _compute_beta(port_ret_s, inst_ret[inst], beta_window)
        scores[inst] = {'corr': corr, 'eff': eff, 'beta': beta,
                        'effectiveness': eff}

    # Sort by correlation descending
    ranked = sorted(scores.items(),
                    key=lambda x: x[1]['corr'] if not np.isnan(x[1]['corr']) else -999,
                    reverse=True)

    selected = {}

    # Always check trigger assets first (QQQ, SPY)
    for ta in trigger_assets:
        if ta in scores:
            d = scores[ta]
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
            break   # ranked by corr, so all remaining will also fail
        if eff >= eff_floor:
            selected[inst] = d
            n_from_ranking += 1

    return selected


# ================================================================================
# SUMMARY
# ================================================================================

def _print_hedge_summary(hedge_log:  list,
                         Pxs_df:    pd.DataFrame,
                         results:   dict) -> None:
    """Print per-year and per-instrument hedge summary."""
    if not hedge_log:
        print("\n  No hedge events recorded.")
        return

    close_events = [e for e in hedge_log if e['event'] == 'CLOSE']
    if not close_events:
        print("\n  No completed hedge episodes.")
        return

    print(f"\n{'='*72}")
    print(f"  HEDGE SUMMARY")
    print(f"{'='*72}")

    # Overall stats
    all_pnl    = [e['total_pnl'] for e in close_events]
    n_pos      = sum(1 for p in all_pnl if p > 0)
    print(f"\n  Total episodes  : {len(close_events)}")
    print(f"  Hit rate        : {n_pos/len(close_events)*100:.1f}%  "
          f"({n_pos} positive / {len(close_events)-n_pos} negative)")
    print(f"  Total P&L       : {sum(all_pnl):+.4%}")
    print(f"  Avg P&L/episode : {np.mean(all_pnl):+.4%}")
    print(f"  Avg days held   : {np.mean([e.get('days_held',0) for e in close_events]):.1f}")

    # Per-year breakdown
    print(f"\n  {'─'*68}")
    print(f"  PER-YEAR BREAKDOWN")
    print(f"  {'─'*68}")
    print(f"  {'Year':<6}  {'Episodes':>9}  {'Hit Rate':>9}  "
          f"{'Total P&L':>10}  {'Avg P&L':>10}  {'Avg Days':>9}")
    print(f"  {'-'*60}")

    years = sorted(set(e['date'].year for e in close_events))
    for yr in years:
        yr_ev  = [e for e in close_events if e['date'].year == yr]
        yr_pnl = [e['total_pnl'] for e in yr_ev]
        yr_pos = sum(1 for p in yr_pnl if p > 0)
        yr_days= np.mean([e.get('days_held', 0) for e in yr_ev])
        print(f"  {yr:<6}  {len(yr_ev):>9}  "
              f"{yr_pos/len(yr_ev)*100:>8.1f}%  "
              f"{sum(yr_pnl):>+9.4%}  "
              f"{np.mean(yr_pnl):>+9.4%}  "
              f"{yr_days:>9.1f}")

    # Per-instrument breakdown (only instruments that were used)
    inst_events = {}
    for e in close_events:
        for inst in e.get('instruments', []):
            if inst not in inst_events:
                inst_events[inst] = []
            w   = e['details'].get(inst, {}).get('weight', 0)
            px_entry = e['details'].get(inst, {}).get('entry_px', np.nan)
            entry_dt = e['details'].get(inst, {}).get('entry_dt', e['date'])
            # Recompute instrument-level P&L from entry to close
            close_dt = e['date']
            if (px_entry and not np.isnan(px_entry) and
                    inst in Pxs_df.columns and close_dt in Pxs_df.index):
                exit_px  = Pxs_df.loc[close_dt, inst]
                inst_pnl = -(exit_px / px_entry - 1) * w
            else:
                inst_pnl = np.nan
            inst_events[inst].append({
                'pnl': inst_pnl, 'weight': w,
                'eff': e['details'].get(inst, {}).get('eff', np.nan),
                'beta': e['details'].get(inst, {}).get('beta', np.nan),
            })

    if inst_events:
        print(f"\n  {'─'*68}")
        print(f"  PER-INSTRUMENT BREAKDOWN")
        print(f"  {'─'*68}")
        print(f"  {'Instrument':<12}  {'Uses':>5}  {'Hit Rate':>9}  "
              f"{'Total P&L':>10}  {'Avg Weight':>11}  {'Avg Beta':>9}  {'Avg Eff':>8}")
        print(f"  {'-'*68}")
        for inst in sorted(inst_events.keys()):
            evs     = inst_events[inst]
            pnls    = [e['pnl'] for e in evs if not np.isnan(e['pnl'])]
            pos     = sum(1 for p in pnls if p > 0)
            avg_w   = np.mean([e['weight'] for e in evs])
            avg_b   = np.nanmean([e['beta'] for e in evs])
            avg_e   = np.nanmean([e['eff']  for e in evs])
            hit_str = f"{pos/len(pnls)*100:.1f}%" if pnls else "n/a"
            pnl_str = f"{sum(pnls):+.4%}" if pnls else "n/a"
            print(f"  {inst:<12}  {len(evs):>5}  {hit_str:>9}  "
                  f"{pnl_str:>10}  {avg_w:>10.2%}  {avg_b:>9.3f}  {avg_e:>8.3f}")


def run_daily_cache_build(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df=None,
    ic=0.03,
    max_weight=0.075,
    min_weight=0.025,
    zscore_cap=2.5,
    pca_var_threshold=0.65,
    universe_mult=5,
    risk_aversion=10,
    top_n=25,
    conc_factor=2.0,
    prefilt_pct=0.5,
    min_cov_matrices=2,
    model_version='v2',
    force_rebuild=False,
):
    """
    Build daily portfolio cache for all trading days since MB_START_DATE.

    Computes alpha / MVO / hybrid / smart weights and 3 trigger variables:
      implied_turnover : how much we'd trade vs current live portfolio
      vol_diff         : vol(new portfolio) - vol(live portfolio), annualized
      drawdown         : running drawdown of smart hybrid NAV from HWM

    force_rebuild=False : only compute missing dates (safe to re-run daily)
    force_rebuild=True  : clear cache and recompute all dates
    """
    ph = _make_params_hash(ic, max_weight, min_weight, zscore_cap, pca_var_threshold,
                      universe_mult, risk_aversion, top_n, conc_factor,
                      prefilt_pct, min_cov_matrices, model_version)

    print("=" * 72)
    print("  DAILY PORTFOLIO CACHE BUILDER")
    print("=" * 72)
    print(f"\n  params_hash={ph}  model={model_version}")
    print(f"  IC={ic}  ra={risk_aversion}  N={top_n}  "
          f"max_w={max_weight}  min_w={min_weight}")
    print(f"  universe_mult={universe_mult}  conc={conc_factor}  "
          f"prefilt={prefilt_pct}  min_cov={min_cov_matrices}")
    print(f"  SH thresholds: alpha<{SH_DD_ALPHA:.0%}  "
          f"hybrid<{SH_DD_HYBRID:.0%}  else MVO")

    _ensure_tables()

    if force_rebuild:
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                DELETE FROM {DAILY_PORT_TBL}
                WHERE params_hash=:ph AND model_version=:mv
            """), {'ph': ph, 'mv': model_version})
            conn.execute(text(f"""
                DELETE FROM {DAILY_TRIGGER_TBL}
                WHERE params_hash=:ph AND model_version=:mv
            """), {'ph': ph, 'mv': model_version})
        print("  Cache cleared (force_rebuild=True)")
        cached = set()
    else:
        cached = _get_cached_dates(ph, model_version)
        print(f"  Already cached: {len(cached)} dates")

    # All trading days since MB_START_DATE
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    pxs_cols  = set(Pxs_df.columns)

    all_dates   = Pxs_df.index
    st_loc      = all_dates.searchsorted(MB_START_DATE)
    ext_loc     = max(0, st_loc - MOM_LONG_BUFFER)
    ext_st      = all_dates[ext_loc]
    daily_dates = all_dates[all_dates >= MB_START_DATE]
    to_compute  = sorted([d for d in daily_dates if d not in cached])

    print(f"\n  Total trading days : {len(daily_dates)}")
    print(f"  Dates to compute   : {len(to_compute)}")

    if not to_compute:
        print("\n  OK Cache fully up to date")
        return ph

    # -- [1/3] Composite scores ------------------------------------------------
    print("\n  [1/3] Building composite scores...")
    dates_idx = pd.DatetimeIndex(to_compute)
    with _SuppressOutput():
        composite_by_date, _ = _cb_build_composite_scores(
            universe        = get_universe(Pxs_df, sectors_s, ext_st),
            calc_dates      = dates_idx,
            Pxs_df          = Pxs_df,
            sectors_s       = sectors_s,
            weights_by_year = weights_by_year,
            regime_s        = regime_s,
            volumeTrd_df    = volumeTrd_df,
            model_version   = model_version,
            exclude_factors = ['OU'],
        )
    print(f"  Composite scores: {len(composite_by_date)} dates")

    # -- [2/3] X snapshots ----------------------------------------------------
    print("  [2/3] Loading X snapshots from cache...")
    with _SuppressOutput():
        X_snapshots, _, _, _ = _mb_build_x_snapshots(
            to_compute, Pxs_df, sectors_s,
            volumeTrd_df, model_version, False
        )
    snapshot_dates = sorted(X_snapshots.keys())
    print(f"  X snapshots: {len(snapshot_dates)}")

    # -- [3/3] Per-day computation ---------------------------------------------
    print(f"\n  [3/3] Computing daily portfolios...")
    print(f"  {'Date':<14} {'TO':>6} {'DVol':>7} {'DD':>7}  "
          f"{'Regime':<8}  {'Elapsed':>7}  {'ETA':>7}")
    print(f"  {'-'*62}")

    n_cands    = top_n * universe_mult
    n_saved    = 0
    n_errors   = 0
    t_start    = time.time()

    # State trackers
    w_live     = pd.Series(dtype=float)   # current live portfolio
    w_hyb_prev = pd.Series(dtype=float)   # previous hybrid for NAV tracking
    sh_nav     = 1.0
    sh_hwm     = 1.0
    sh_regime  = 'alpha'                  # current smart hybrid regime (asymmetric)

    for idx, dt in enumerate(to_compute):
        if dt not in composite_by_date:
            continue

        comp_scores = composite_by_date[dt]
        cands       = comp_scores.dropna()
        cands       = cands.loc[[t for t in cands.index if t in pxs_cols]]
        n_pf        = max(n_cands, int(np.ceil(len(cands) * prefilt_pct)))
        cands       = cands.nlargest(min(n_pf, len(cands)))

        if len(cands) < top_n:
            continue

        valid_snap = [d for d in snapshot_dates if d <= dt]
        if not valid_snap:
            continue

        # -- Alpha -------------------------------------------------------------
        ranked     = cands.sort_values(ascending=False)
        alpha_port = [t for t in ranked.head(top_n).index if t in pxs_cols]
        n_p        = len(alpha_port)
        n_top_h    = int(np.ceil(n_p / 2))
        n_bot_h    = n_p - n_top_h
        if conc_factor == 1.0 or n_bot_h == 0:
            w_alpha = pd.Series(1.0 / n_p, index=alpha_port)
        else:
            top_a = conc_factor / (conc_factor + 1.0)
            bot_a = 1.0 / (conc_factor + 1.0)
            w_alpha = pd.Series({
                t: (top_a / n_top_h if j < n_top_h else bot_a / n_bot_h)
                for j, t in enumerate(alpha_port)
            })

        # -- MVO ---------------------------------------------------------------
        try:
            with _SuppressOutput():
                w_mvo, _ = _mb_solve_mvo(
                    dt=dt, candidates=cands.index.tolist()[:n_cands],
                    composite_scores=comp_scores, Pxs_df=Pxs_df,
                    sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                    model_version=model_version,
                    pca_var_threshold=pca_var_threshold,
                    ic=ic, max_weight=max_weight, min_weight=min_weight,
                    zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                    X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                    top_n=top_n, min_cov_matrices=min_cov_matrices,
                )
        except Exception:
            n_errors += 1
            continue

        if w_mvo.empty or w_mvo.sum() == 0:
            continue
        w_mvo_nz = w_mvo[w_mvo > 1e-6]

        # -- Hybrid ------------------------------------------------------------
        all_t    = list(set(w_alpha.index) | set(w_mvo_nz.index))
        w_hybrid = (w_alpha.reindex(all_t).fillna(0.0) +
                    w_mvo_nz.reindex(all_t).fillna(0.0)) / 2.0
        w_hybrid = w_hybrid[w_hybrid > 0]
        if w_hybrid.sum() > 0:
            w_hybrid = w_hybrid / w_hybrid.sum()

        # -- Smart hybrid regime -----------------------------------------------
        if not w_hyb_prev.empty and idx > 0:
            prev_dt = to_compute[idx - 1]
            tks_sh  = [t for t in w_hyb_prev.index if t in pxs_cols]
            if tks_sh and prev_dt in Pxs_df.index and dt in Pxs_df.index:
                px_s    = Pxs_df.loc[prev_dt, tks_sh]
                px_e    = Pxs_df.loc[dt,       tks_sh]
                per_ret = (w_hyb_prev.reindex(tks_sh).fillna(0) *
                           (px_e / px_s - 1).fillna(0)).sum()
                sh_nav  = sh_nav * (1 + per_ret)
        sh_hwm = max(sh_hwm, sh_nav)
        dd     = sh_nav / sh_hwm - 1

        if dd >= -SH_DD_ALPHA:
            w_smart    = w_alpha
            live_strat = 'alpha'
        elif dd >= -SH_DD_HYBRID:
            w_smart    = w_hybrid
            live_strat = 'hybrid'
        else:
            w_smart    = w_mvo_nz
            live_strat = 'mvo'

        w_hyb_prev = w_hybrid.copy()

        # -- Trigger variables -------------------------------------------------
        # Drift w_live by today's price moves before comparing
        if not w_live.empty and idx > 0:
            prev_dt_live = to_compute[idx - 1]
            tks_live = [t for t in w_live.index if t in pxs_cols]
            if tks_live and prev_dt_live in Pxs_df.index and dt in Pxs_df.index:
                px_prev_live = Pxs_df.loc[prev_dt_live, tks_live]
                px_cur_live  = Pxs_df.loc[dt,           tks_live]
                w_drifted = w_live.reindex(tks_live) * (px_cur_live / px_prev_live).fillna(1)
                if w_drifted.sum() > 0:
                    w_live = w_drifted / w_drifted.sum()

        # Implied turnover vs drifted live portfolio
        if w_live.empty:
            implied_to = 1.0
        else:
            all_t_to   = list(set(w_smart.index) | set(w_live.index))
            implied_to = (w_smart.reindex(all_t_to).fillna(0) -
                          w_live.reindex(all_t_to).fillna(0)).abs().sum() / 2

        # Vol difference: new vs live
        vol_new  = _portfolio_vol(w_smart, Pxs_df, dt)
        vol_live = _portfolio_vol(w_live,  Pxs_df, dt) if not w_live.empty else vol_new
        vol_diff = vol_new - vol_live

        # Reset w_live to today's smart hybrid for next day's drift baseline
        w_live = w_smart.copy()

        # -- Save --------------------------------------------------------------
        try:
            _save_portfolios(dt, model_version, ph, {
                'alpha' : w_alpha,
                'mvo'   : w_mvo_nz,
                'hybrid': w_hybrid,
                'smart' : w_smart,
            })
            _save_triggers(dt, model_version, ph,
                           implied_to, vol_diff, dd, live_strat)
            n_saved += 1
        except Exception as e:
            warnings.warn(f"  Save failed {dt.date()}: {e}")
            n_errors += 1
            continue

        # Progress every 50 dates
        if idx % 50 == 0 or idx == len(to_compute) - 1:
            elapsed   = time.time() - t_start
            rate      = elapsed / max(idx + 1, 1)
            remaining = rate * (len(to_compute) - idx - 1)
            print(f"  {str(dt.date()):<14} "
                  f"{implied_to*100:>5.1f}%  {vol_diff*100:>+6.1f}%  "
                  f"{dd*100:>+6.1f}%  {live_strat:<8}  "
                  f"{elapsed:>5.0f}s  {remaining:>5.0f}s")

    elapsed_total = time.time() - t_start
    print(f"\n  OK Done: {n_saved} dates saved, {n_errors} errors  "
          f"({elapsed_total/60:.1f} min)")
    print(f"  params_hash = '{ph}'")
    print(f"\n  Load trigger data with:")
    print(f"    triggers = pd.read_sql(")
    print(f"        \"SELECT * FROM {DAILY_TRIGGER_TBL} WHERE params_hash='{ph}'\",")
    print(f"        ENGINE, index_col='date', parse_dates=['date'])")
    return ph

# ===============================================================================
# ENTRY POINT
# ===============================================================================

def run_mvo_backtest(Pxs_df, sectors_s, weights_by_year, regime_s,
                     volumeTrd_df=None,
                     # MVO parameters (not user prompts)
                     ic=0.04,
                     max_weight=0.10,
                     min_weight=0.025,
                     zscore_cap=2.5,
                     pca_var_threshold=0.65,
                     universe_mult=5,
                     risk_aversion=1.0,
                     force_rebuild_cache=False,
                     portfolio_cache_override=False,
                     # Hedge layer (optional)
                     hedge_multi=None,      # output of run_macro_hedge_cached
                     hedges_l=None,         # list of hedge instrument tickers
                     hedge_trigger_assets=None):  # default ['QQQ','SPX']
    """
    Run three portfolios in parallel: Baseline, Pure Alpha, MVO.

    Parameters
    ----------
    Pxs_df, sectors_s, weights_by_year, regime_s, volumeTrd_df
        -- same as composite_backtest.py

    MVO parameters (function inputs, not user prompts):
    ic                 : float -- Grinold-Kahn IC scaling (default 0.04)
    max_weight         : float -- single-name cap (default 0.10)
    min_weight         : float -- single-name floor (default 0.025)
    zscore_cap         : float -- alpha z-score winsorization (default 2.5)
    pca_var_threshold  : float -- PCA variance explained threshold (default 0.65)
    universe_mult      : int   -- candidate pool = port_n x universe_mult (default 5)
    risk_aversion      : float -- MVO risk aversion lambda (default 1.0)
    force_rebuild_cache: bool  -- False (default): load X cache from DB, only
                                 compute missing dates (typically just today).
                                 True: clear cache and rebuild all dates from scratch.
    """
    print("=" * 72)
    print("  MVO BACKTEST -- Baseline vs Pure Alpha vs MVO")
    print("=" * 72)
    print(f"\n  MVO params: IC={ic}, max_w={max_weight}, min_w={min_weight}, "
          f"zscore_cap={zscore_cap}")
    print(f"  PCA thresh={pca_var_threshold}, universe_mult={universe_mult}, "
          f"risk_aversion={risk_aversion}")
    print(f"  force_rebuild_cache={force_rebuild_cache} (False=incremental/cache, True=full rebuild)\n")

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    # De-trend raw daily volume: divide by rolling mean to get relative volume
    # volumeTrd_df input is expected to be RAW daily traded volume per stock
    if volumeTrd_df is not None:
        volumeRaw_df = volumeTrd_df.copy()   # preserve raw volume for ADV cap
        vol_roll     = (volumeTrd_df
                        .replace({0: np.nan})
                        .fillna(method='ffill')
                        .fillna(0)
                        .rolling(VOLUME_WINDOW).mean())
        volumeTrd_df = volumeTrd_df / vol_roll.replace({0: np.nan})
    else:
        volumeRaw_df = None

    # -- User prompts (identical to composite_backtest.py) ---------------------
    print("  PORTFOLIO CONSTRUCTION OPTIONS:")
    topn_input    = input(f"  Number of stocks [default={MB_TOP_N}]: ").strip()
    rebal_input   = input(f"  Rebalancing frequency in days [default={MB_REBAL_FREQ}]: ").strip()
    sector_input  = input("  Max stocks per sector (or Enter to skip): ").strip()
    mktcap_input  = input("  Min market cap floor ($M, or Enter to skip): ").strip()
    advp_input    = input(f"  ADVP cap — max % of median daily $ volume per stock [default=4%]: ").strip()
    vol_input     = input("  Apply vol filter? (y/n) [default=n]: ").strip().lower()
    prefilt_input = input("  Pre-filter fraction by composite score 0<x<=1 (or Enter for none): ").strip()
    conc_input    = input("  Concentration factor for Pure Alpha >=1.0 (or Enter for equal weight): ").strip()

    top_n        = int(topn_input)           if topn_input    else MB_TOP_N
    rebal_freq   = int(rebal_input)          if rebal_input   else MB_REBAL_FREQ
    sector_cap   = int(sector_input)         if sector_input  else None
    mktcap_floor = float(mktcap_input) * 1e6 if mktcap_input  else None
    advp_cap     = float(advp_input) / 100   if advp_input    else 0.04
    use_vol      = vol_input == 'y'
    prefilt_pct  = float(prefilt_input)      if prefilt_input else 1.0
    if prefilt_pct <= 0 or prefilt_pct > 1:
        prefilt_pct = 1.0
    conc_factor  = float(conc_input) if conc_input else 1.0
    if conc_factor < 1.0:
        conc_factor = 1.0

    min_cov_inp      = input(f"  Min cov matrices for stock selection "
                             f"(0=alpha-only, default={MB_MIN_COV_MATRICES}): ").strip()
    min_cov_matrices = int(min_cov_inp) if min_cov_inp else MB_MIN_COV_MATRICES
    min_cov_matrices = max(0, min(min_cov_matrices, 4))

    enforce_hybrid_floor = False

    n_cands = top_n * universe_mult
    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"sector_cap={sector_cap}, mktcap_floor={mktcap_input or 'none'}, "
          f"advp_cap={advp_cap:.1%} (AUM=${AUM:,.0f}), "
          f"vol_filter={use_vol}, prefilt={prefilt_pct:.0%}, "
          f"conc={conc_factor:.1f}x (pure alpha only)")
    print(f"  MVO candidate pool: {n_cands} stocks ({universe_mult}x{top_n})  |  "
          f"min_cov_matrices={min_cov_matrices}\n")

    # -- Daily portfolio cache setup -------------------------------------------
    params_hash = _make_make_params_hash(
        ic, max_weight, min_weight, zscore_cap, pca_var_threshold,
        universe_mult, risk_aversion, top_n, conc_factor,
        prefilt_pct, min_cov_matrices, MB_MODEL_VER,
    )
    print(f"  Portfolio cache: params_hash={params_hash}  "
          f"override={portfolio_cache_override}")
    _ensure_daily_cache_table()
    if portfolio_cache_override:
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                DELETE FROM {MB_DAILY_PORT_TBL}
                WHERE params_hash = :ph AND model_version = :mv
            """), {'ph': params_hash, 'mv': MB_MODEL_VER})
        print("  Portfolio cache cleared (override=True)")
        cached_port_dates = set()
    else:
        cached_port_dates = _get_cached_portfolio_dates(params_hash, MB_MODEL_VER)
        print(f"  Portfolio cache: {len(cached_port_dates)} dates already cached\n")

    # -- Universe and calc dates -----------------------------------------------
    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(MB_START_DATE)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    calc_dates     = generate_calc_dates(Pxs_df, step_days=rebal_freq)
    calc_dates_idx = pd.DatetimeIndex(calc_dates)
    pxs_cols       = set(Pxs_df.columns)

    print(f"  Universe: {len(universe)} stocks  |  "
          f"Rebalance dates: {len(calc_dates)}")

    # -- [1/4] Composite alpha scores -----------------------------------------
    print("\n[1/4] Building composite alpha scores...")
    composite_by_date, _ = _cb_build_composite_scores(
        universe        = universe,
        calc_dates      = calc_dates_idx,
        Pxs_df          = Pxs_df,
        sectors_s       = sectors_s,
        weights_by_year = weights_by_year,
        regime_s        = regime_s,
        volumeTrd_df    = volumeTrd_df,
        model_version   = MB_MODEL_VER,
        exclude_factors = ['OU'],
    )

    # -- [2/4] X snapshots (cached) --------------------------------------------
    print("\n[2/4] Building X snapshots (monthly, cached)...")
    X_snapshots, _, _, _ = _mb_build_x_snapshots(
        calc_dates, Pxs_df, sectors_s,
        volumeTrd_df, MB_MODEL_VER, force_rebuild_cache
    )
    snapshot_dates = sorted(X_snapshots.keys())
    if snapshot_dates:
        print(f"  X snapshots available: {len(snapshot_dates)}  "
              f"({snapshot_dates[0].date()} -> {snapshot_dates[-1].date()})")

    # -- [3/4] Compute portfolio weights per rebalance date --------------------
    print("\n[3/4] Computing portfolio weights...")

    alpha_weights_by_date  = {}   # pure alpha (equal/concentration)
    mvo_weights_by_date    = {}   # MVO
    hybrid_weights_by_date = {}   # hybrid (built incrementally for turnover display)
    quality_factor_by_date = {}   # baseline
    all_stock_returns      = []   # pool of individual stock returns across all periods
    _last_mvo_diag         = {}   # last known MVO diagnostics for dynamic display
    _diag_by_date          = {}   # {date: mvo_diag} for each rebal date
    # Per-strategy turnover tracking for cost calculation
    _prev_w = {"baseline": pd.Series(dtype=float),
               "alpha":    pd.Series(dtype=float),
               "mvo":      pd.Series(dtype=float),
               "hybrid":   pd.Series(dtype=float),
               "smart":    pd.Series(dtype=float)}
    _cost_by_date = {"baseline": {}, "alpha": {}, "mvo": {},
                     "hybrid": {}, "smart": {}}
    all_turnover_ratios    = []   # turnover % at each rebalance
    all_portfolio_returns        = []   # hybrid portfolio return per holding period
    smart_hybrid_weights_by_date = {}   # drawdown-regime-aware hybrid
    _sh_nav_running = 1.0              # running NAV for smart hybrid regime detection
    _sh_hwm         = 1.0              # high-water mark for regime detection
    _sh_regime_counts = {"alpha": 0, "hybrid": 0, "mvo": 0}  # regime prevalence
    _sh_regime_by_date = {}   # {date: regime_str} for yearly breakdown

    # Running NAV for ADVP cap (tracks alpha strategy, used to scale AUM)
    _running_nav   = 1.0
    _prev_rebal_dt = None
    _advp_capped   = {}   # {dt: set of tickers where ADVP cap was binding}

    # Load quality scores for baseline
    print("  Loading quality scores for baseline...")
    all_tickers  = list(sectors_s.index)
    with _SuppressOutput():
        quality_wide = get_quality_scores(
            calc_dates         = calc_dates_idx,
            universe           = all_tickers,
            Pxs_df             = Pxs_df,
            sectors_s          = sectors_s,
            use_cached_weights = True,
            force_recompute    = False,
        )
    print(f"  Quality scores loaded: {len(quality_wide)} dates")

    n = len(calc_dates)
    for i, dt in enumerate(calc_dates):
        print(f"  Processing [{i+1}/{n}] {dt.date()}...", end='\r')

        # -- Update running NAV for ADVP cap scaling ---------------------------
        if _prev_rebal_dt is not None:
            prev_w_alpha = alpha_weights_by_date.get(_prev_rebal_dt, pd.Series(dtype=float))
            tks_nav = [t for t in prev_w_alpha.index if t in Pxs_df.columns]
            if tks_nav and _prev_rebal_dt in Pxs_df.index and dt in Pxs_df.index:
                px_prev = Pxs_df.loc[_prev_rebal_dt, tks_nav]
                px_curr = Pxs_df.loc[dt,             tks_nav]
                period_ret = (prev_w_alpha.reindex(tks_nav).fillna(0) *
                              (px_curr / px_prev - 1).fillna(0)).sum()
                _running_nav *= (1 + period_ret)
        _prev_rebal_dt = dt
        current_aum = AUM * _running_nav

        # -- Baseline quality factor -------------------------------------------
        if dt in quality_wide.index:
            scores = quality_wide.loc[dt].dropna()
            if not scores.empty:
                fdf = scores.rename('factor').to_frame()
                fdf['Sector'] = fdf.index.map(sectors_s)
                fdf = fdf.dropna(subset=['Sector'])
                fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
                if not fdf.empty:
                    quality_factor_by_date[dt] = fdf

        if dt not in composite_by_date:
            continue

        comp_scores = composite_by_date[dt]

        # -- Candidate universe ------------------------------------------------
        cands = comp_scores.dropna()
        cands = cands.loc[[t for t in cands.index if t in pxs_cols]]

        # Vol filter
        if use_vol and len(cands) > top_n:
            surviving = apply_vol_filter(cands.index, dt, Pxs_df)
            cands = cands.reindex(surviving).dropna()

        # Pre-filter
        n_prefilt = max(n_cands, int(np.ceil(len(cands) * prefilt_pct)))
        cands = cands.nlargest(min(n_prefilt, len(cands)))

        # Sector cap pre-filter (for pure alpha path)
        if sector_cap is not None:
            sec_counts = {}
            cands_filtered = []
            for t in cands.index:
                sec = sectors_s.get(t, 'Unknown')
                if sec_counts.get(sec, 0) < sector_cap * universe_mult:
                    cands_filtered.append(t)
                    sec_counts[sec] = sec_counts.get(sec, 0) + 1
            cands = cands.reindex(cands_filtered).dropna()

        if len(cands) < top_n:
            continue

        candidates = cands.index.tolist()

        # -- ADVP liquidity filter: build liquid universe before alpha ranking --
        # A stock is liquid enough if its ADVP cap >= min_weight
        # If not enough liquid stocks, go deeper into ranking until top_n reached
        current_aum_filt = AUM * _running_nav
        if volumeRaw_df is not None and current_aum_filt > 0:
            past_dates_filt = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
            liquid_set = set()
            for tkr in cands.index:
                if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                    liquid_set.add(tkr)   # no volume data — assume liquid
                    continue
                px_s   = Pxs_df.loc[past_dates_filt, tkr].dropna()
                vol_s  = volumeRaw_df.loc[past_dates_filt, tkr].dropna()
                common = px_s.index.intersection(vol_s.index)
                if len(common) < 3:
                    liquid_set.add(tkr)   # insufficient history — assume liquid
                    continue
                dollar_vol = (px_s.reindex(common) * vol_s.reindex(common).replace({0: np.nan})).median()
                cap_w      = (dollar_vol * advp_cap) / current_aum_filt
                if cap_w >= min_weight:
                    liquid_set.add(tkr)
            # Filter ranked to liquid stocks only, keep original alpha ranking
            cands_liquid = cands.reindex([t for t in cands.index if t in liquid_set])
        else:
            cands_liquid = cands

        # -- Pure alpha weights (concentration) --------------------------------
        ranked  = cands_liquid.sort_values(ascending=False)
        if sector_cap is not None:
            top_sel = select_with_sector_cap(
                ranked.rename('factor').to_frame().assign(
                    Sector=ranked.index.map(sectors_s)
                ),
                sector_cap, top_n
            )
            alpha_port = top_sel.index.tolist()
        else:
            alpha_port = ranked.head(top_n).index.tolist()
        alpha_port = [t for t in alpha_port if t in pxs_cols]

        n_port   = len(alpha_port)
        n_top_h  = int(np.ceil(n_port / 2))
        n_bot_h  = n_port - n_top_h
        if conc_factor == 1.0 or n_bot_h == 0:
            alpha_w = {t: 1.0 / n_port for t in alpha_port}
        else:
            top_alloc = conc_factor / (conc_factor + 1.0)
            bot_alloc = 1.0 / (conc_factor + 1.0)
            alpha_w   = {}
            for j, t in enumerate(alpha_port):
                alpha_w[t] = (top_alloc / n_top_h if j < n_top_h
                              else bot_alloc / n_bot_h)
        alpha_weights_by_date[dt] = pd.Series(alpha_w)
        # Apply ADVP cap to alpha weights
        if volumeRaw_df is not None and advp_cap > 0:
            w_alpha_capped, capped_a = _apply_advp_cap(
                pd.Series(alpha_w), dt, Pxs_df, volumeRaw_df,
                current_aum, advp_cap, min_weight, max_weight)
            if capped_a:
                alpha_weights_by_date[dt] = w_alpha_capped
                _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_a

        # Update running NAV from last rebal to now using previous alpha weights
        if _prev_rebal_dt is not None and _prev_rebal_dt in alpha_weights_by_date:
            w_prev_alpha = alpha_weights_by_date[_prev_rebal_dt]
            prev_px_dates = [d for d in Pxs_df.index
                             if _prev_rebal_dt <= d <= dt]
            if len(prev_px_dates) >= 2:
                tks_nav = [t for t in w_prev_alpha.index if t in Pxs_df.columns]
                if tks_nav:
                    px_start = Pxs_df.loc[prev_px_dates[0],  tks_nav]
                    px_end   = Pxs_df.loc[prev_px_dates[-1], tks_nav]
                    period_ret = (w_prev_alpha.reindex(tks_nav).fillna(0) *
                                  (px_end / px_start - 1).fillna(0)).sum()
                    _running_nav *= (1 + period_ret)
        _prev_rebal_dt = dt

        # Apply ADVP cap to alpha weights
        current_aum = AUM * _running_nav
        w_alpha_s   = pd.Series(alpha_w)
        w_alpha_s, capped_alpha = _apply_advp_cap(
            w_alpha_s, dt, Pxs_df, volumeRaw_df,
            current_aum, advp_cap, min_weight, max_weight)
        alpha_weights_by_date[dt] = w_alpha_s
        if capped_alpha:
            _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_alpha
        # Trading cost for alpha
        w_new_a = pd.Series(alpha_w)
        w_old_a = _prev_w["alpha"]
        all_t_a = list(set(w_new_a.index) | set(w_old_a.index))
        to_a    = (w_new_a.reindex(all_t_a).fillna(0) -
                   w_old_a.reindex(all_t_a).fillna(0)).abs().sum() / 2
        _cost_by_date["alpha"][dt] = to_a * TRADING_COST_BPS / 10000
        _prev_w["alpha"] = w_new_a

        # -- MVO weights -------------------------------------------------------
        # Use top n_cands from liquid universe for MVO
        candidates_liquid = cands_liquid.index.tolist()
        mvo_cands = candidates_liquid[:n_cands]
        valid_snap = [d for d in snapshot_dates if d <= dt]
        if not valid_snap:
            continue

        # Load from cache if available (skip MVO solve for weights, keep for diagnostics)
        if dt in cached_port_dates:
            import json
            try:
                with ENGINE.connect() as conn:
                    rows = conn.execute(text(f"""
                        SELECT strategy, weights_json FROM {MB_DAILY_PORT_TBL}
                        WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                    """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': MB_MODEL_VER,
                           'ph': params_hash}).fetchall()
                port_cache = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
                if 'alpha' in port_cache and 'mvo' in port_cache:
                    # alpha_weights_by_date[dt] already set above from fresh computation
                    # alpha costs already computed correctly above — don't overwrite
                    mvo_weights_by_date[dt]   = port_cache['mvo']
                    w_hyb_cached = port_cache.get('hybrid', pd.Series(dtype=float))
                    hybrid_weights_by_date[dt] = w_hyb_cached
                    # Compute MVO/hybrid trading costs for cached dates
                    w_mvo_c  = port_cache['mvo']
                    w_old_mc = _prev_w["mvo"]
                    all_t_mc = list(set(w_mvo_c.index) | set(w_old_mc.index))
                    to_mc    = (w_mvo_c.reindex(all_t_mc).fillna(0) -
                                w_old_mc.reindex(all_t_mc).fillna(0)).abs().sum() / 2
                    _cost_by_date["mvo"][dt] = to_mc * TRADING_COST_BPS / 10000
                    _prev_w["mvo"] = w_mvo_c
                    if not w_hyb_cached.empty:
                        w_old_hc = _prev_w["hybrid"]
                        all_t_hc = list(set(w_hyb_cached.index) | set(w_old_hc.index))
                        to_hc    = (w_hyb_cached.reindex(all_t_hc).fillna(0) -
                                    w_old_hc.reindex(all_t_hc).fillna(0)).abs().sum() / 2
                        _cost_by_date["hybrid"][dt] = to_hc * TRADING_COST_BPS / 10000
                        _prev_w["hybrid"] = w_hyb_cached
                    # Update hybrid weights for turnover tracking
                    hybrid_weights_by_date[dt] = w_hyb_cached
                    # Update smart hybrid regime tracking using daily return
                    all_hyb_dates_c = sorted([d for d in hybrid_weights_by_date if d <= dt])
                    if all_hyb_dates_c:
                        last_hyb_dt_c = all_hyb_dates_c[-1]
                        w_prev_c      = hybrid_weights_by_date[last_hyb_dt_c]
                        tks_c         = [t for t in w_prev_c.index if t in pxs_cols]
                        prev_days_c   = [d for d in Pxs_df.index if d < dt]
                        if tks_c and prev_days_c and dt in Pxs_df.index:
                            prev_day_c = prev_days_c[-1]
                            if prev_day_c in Pxs_df.index:
                                px_sc = Pxs_df.loc[prev_day_c, tks_c]
                                px_ec = Pxs_df.loc[dt,          tks_c]
                                pr_c  = (w_prev_c.reindex(tks_c).fillna(0) *
                                         (px_ec / px_sc - 1).fillna(0)).sum()
                                _sh_nav_running = _sh_nav_running * (1 + pr_c)
                    _sh_hwm = max(_sh_hwm, _sh_nav_running)
                    # Run MVO solve for diagnostics, then fall through to display
                    try:
                        _, _diag_cached = _mb_solve_mvo(
                            dt=dt, candidates=mvo_cands,
                            composite_scores=comp_scores, Pxs_df=Pxs_df,
                            sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                            model_version=MB_MODEL_VER,
                            pca_var_threshold=pca_var_threshold,
                            ic=ic, max_weight=max_weight, min_weight=min_weight,
                            zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                            X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                            top_n=top_n, min_cov_matrices=min_cov_matrices,
                        )
                        _diag_by_date[dt] = _diag_cached
                        _last_mvo_diag    = _diag_cached
                        mvo_diag          = _diag_cached
                    except Exception:
                        mvo_diag = {}
                    # Use cached weights for display
                    w_mvo    = port_cache['mvo']
                    w_mvo_nz = w_mvo[w_mvo > 1e-6]
            except Exception:
                pass  # fall through to recompute

        if dt not in mvo_weights_by_date:
            try:
                w_mvo, mvo_diag = _mb_solve_mvo(
                        dt            = dt,
                        candidates    = mvo_cands,
                        composite_scores = comp_scores,
                        Pxs_df        = Pxs_df,
                        sectors_s     = sectors_s,
                        volumeTrd_df  = volumeTrd_df,
                        model_version = MB_MODEL_VER,
                        pca_var_threshold = pca_var_threshold,
                        ic            = ic,
                        max_weight    = max_weight,
                        min_weight    = min_weight,
                        zscore_cap    = zscore_cap,
                        risk_aversion = risk_aversion,
                        X_snapshots   = X_snapshots,
                        snapshot_dates = snapshot_dates,
                        top_n             = top_n,
                        min_cov_matrices  = min_cov_matrices,
                    )
            except Exception as e:
                print(f"  {dt.date()} MVO ERROR: {e}")
                w_mvo, mvo_diag = pd.Series(dtype=float), {}
        else:
            # Already loaded from cache — use cached weights and diag
            w_mvo    = mvo_weights_by_date[dt]
            mvo_diag = _diag_by_date.get(dt, _last_mvo_diag)

        if not w_mvo.empty and w_mvo.sum() > 0:
            w_mvo_nz = w_mvo[w_mvo > 1e-6]

            # Apply ADVP cap to MVO weights
            w_mvo_nz, capped_mvo = _apply_advp_cap(
                w_mvo_nz, dt, Pxs_df, volumeRaw_df,
                AUM * _running_nav, advp_cap, min_weight, max_weight)
            if capped_mvo:
                _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_mvo

            mvo_weights_by_date[dt] = w_mvo_nz
            _last_mvo_diag    = mvo_diag  # keep for dynamic display
            _diag_by_date[dt] = mvo_diag  # store per date for dynamic display
            # Pre-compute hybrid for caching (same logic as display block)
            _alpha_w_cache = alpha_weights_by_date.get(dt, pd.Series(dtype=float))
            if not _alpha_w_cache.empty:
                _all_t_c = list(set(_alpha_w_cache.index) | set(w_mvo_nz.index))
                _w_hyb_c = (_alpha_w_cache.reindex(_all_t_c).fillna(0.0) +
                             w_mvo_nz.reindex(_all_t_c).fillna(0.0)) / 2.0
                _w_hyb_c = _w_hyb_c[_w_hyb_c > 0]
                if _w_hyb_c.sum() > 0:
                    _w_hyb_c = _w_hyb_c / _w_hyb_c.sum()
            else:
                _w_hyb_c = w_mvo_nz.copy()
            try:
                _save_daily_portfolios(dt, MB_MODEL_VER, params_hash,
                                       alpha_weights_by_date.get(dt),
                                       w_mvo_nz, _w_hyb_c)
            except Exception as e:
                warnings.warn(f"  Cache save failed for {dt.date()}: {e}")
            vol_ann   = mvo_diag.get('vol_ann',   pd.Series(dtype=float))
            z_s       = mvo_diag.get('z_s',       pd.Series(dtype=float))
            alpha_ann = mvo_diag.get('alpha_ann', pd.Series(dtype=float))
            top_n_by  = mvo_diag.get('top_n_by', {})  # keys: E, L, F, P

            # Fill diagnostics for alpha-only stocks missing from MVO universe
            comp_dt = composite_by_date.get(dt, pd.Series(dtype=float))
            for tkr in alpha_weights_by_date.get(dt, pd.Series()).index:
                if tkr not in vol_ann.index and tkr in Pxs_df.columns:
                    # Annualised vol from last 63 trading days
                    px_hist = Pxs_df[tkr].dropna()
                    px_hist = px_hist[px_hist.index <= dt].tail(64)
                    if len(px_hist) >= 10:
                        daily_vol = px_hist.pct_change().dropna().std()
                        vol_ann[tkr]   = daily_vol * np.sqrt(252)
                        alpha_ann[tkr] = comp_dt.get(tkr, 0.0)  # raw composite score as proxy
                        # z-score: normalise composite score vs cross-section
                        if not comp_dt.empty and comp_dt.std() > 0:
                            z_s[tkr] = (comp_dt.get(tkr, 0.0) - comp_dt.mean()) / comp_dt.std()
                        else:
                            z_s[tkr] = 0.0

            # -- Build hybrid weights for display ------------------------------
            alpha_w_dt = alpha_weights_by_date.get(dt, pd.Series(dtype=float))
            if not alpha_w_dt.empty:
                all_t  = list(set(alpha_w_dt.index) | set(w_mvo_nz.index))
                w_hyb  = (alpha_w_dt.reindex(all_t).fillna(0.0) +
                          w_mvo_nz.reindex(all_t).fillna(0.0)) / 2.0
                w_hyb  = w_hyb[w_hyb > 0]
                w_hyb  = w_hyb / w_hyb.sum()
                if enforce_hybrid_floor:
                    for _ in range(50):
                        below = w_hyb < min_weight - 1e-9
                        if not below.any():
                            break
                        shortfall = (min_weight - w_hyb[below]).sum()
                        w_hyb[below] = min_weight
                        above = ~below
                        if w_hyb[above].sum() > shortfall + 1e-9:
                            w_hyb[above] -= shortfall * (w_hyb[above] / w_hyb[above].sum())
                        w_hyb = w_hyb.clip(lower=0)
                        w_hyb = w_hyb / w_hyb.sum()
            else:
                w_hyb = w_mvo_nz.copy()
            mvo_only_tickers = set(w_mvo_nz.index) - set(alpha_w_dt.index)
            alpha_only_tickers = set(alpha_w_dt.index) - set(w_mvo_nz.index)
            n_common = len(set(alpha_w_dt.index) & set(w_mvo_nz.index))

            # Trading cost for MVO (skip if already computed in cache block)
            if dt not in _cost_by_date["mvo"]:
                w_old_m  = _prev_w["mvo"]
                all_t_m  = list(set(w_mvo_nz.index) | set(w_old_m.index))
                to_m     = (w_mvo_nz.reindex(all_t_m).fillna(0) -
                            w_old_m.reindex(all_t_m).fillna(0)).abs().sum() / 2
                _cost_by_date["mvo"][dt] = to_m * TRADING_COST_BPS / 10000
                _prev_w["mvo"] = w_mvo_nz

            # Store hybrid weights immediately so turnover can reference prior dates
            # Apply ADVP cap to hybrid weights
            w_hyb, capped_hyb = _apply_advp_cap(
                w_hyb, dt, Pxs_df, volumeRaw_df,
                AUM * _running_nav, advp_cap, min_weight, max_weight)
            if capped_hyb:
                _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_hyb
            hybrid_weights_by_date[dt] = w_hyb
            # Trading cost for hybrid (skip if already computed in cache block)
            if dt not in _cost_by_date["hybrid"]:
                w_old_h  = _prev_w["hybrid"]
                all_t_h  = list(set(w_hyb.index) | set(w_old_h.index))
                to_h     = (w_hyb.reindex(all_t_h).fillna(0) -
                            w_old_h.reindex(all_t_h).fillna(0)).abs().sum() / 2
                _cost_by_date["hybrid"][dt] = to_h * TRADING_COST_BPS / 10000
                _prev_w["hybrid"] = w_hyb

            # -- Determine smart hybrid regime for this date -------------------
            # Update running NAV using daily return of hybrid portfolio
            # Use previous day's hybrid weights applied to today's price change
            all_hyb_dates = sorted([d for d in hybrid_weights_by_date if d <= dt])
            if all_hyb_dates:
                # Get most recent hybrid weights
                last_hyb_dt = all_hyb_dates[-1]
                w_prev_sh   = hybrid_weights_by_date[last_hyb_dt]
                tks_sh      = [t for t in w_prev_sh.index if t in Pxs_df.columns]
                # Find previous trading day for daily return
                prev_days = [d for d in Pxs_df.index if d < dt]
                if tks_sh and prev_days and dt in Pxs_df.index:
                    prev_day = prev_days[-1]
                    if prev_day in Pxs_df.index:
                        px_s    = Pxs_df.loc[prev_day, tks_sh]
                        px_e    = Pxs_df.loc[dt,       tks_sh]
                        day_ret = (w_prev_sh.reindex(tks_sh).fillna(0) *
                                   (px_e / px_s - 1).fillna(0)).sum()
                        _sh_nav_running = _sh_nav_running * (1 + day_ret)
            _sh_hwm = max(_sh_hwm, _sh_nav_running)
            dd_disp = _sh_nav_running / _sh_hwm - 1

            if dd_disp >= -SH_DD_ALPHA:
                w_disp   = alpha_weights_by_date.get(dt, w_hyb)
                regime_lbl = f"alpha (dd={dd_disp*100:+.1f}%)"
                _sh_regime_counts["alpha"] += 1
                _sh_regime_by_date[dt] = "alpha"
            elif dd_disp >= -SH_DD_HYBRID:
                w_disp   = w_hyb
                regime_lbl = f"hybrid (dd={dd_disp*100:+.1f}%)"
                _sh_regime_counts["hybrid"] += 1
                _sh_regime_by_date[dt] = "hybrid"
            else:
                w_disp   = w_mvo_nz
                regime_lbl = f"MVO (dd={dd_disp*100:+.1f}%)"
                _sh_regime_counts["mvo"] += 1
                _sh_regime_by_date[dt] = "mvo"

            if w_disp.empty or w_disp.sum() == 0:
                w_disp = w_hyb
            w_disp = w_disp[w_disp > 1e-6]
            w_disp = w_disp / w_disp.sum()

            # -- Per-date display: smart hybrid portfolio ----------------------
            eff_n  = 1.0 / (w_disp**2).sum() if len(w_disp) > 0 else 0
            n_disp = len(w_disp)

            # -- Turnover vs previous smart hybrid portfolio -------------------
            prev_hyb_dates = sorted([d for d in hybrid_weights_by_date if d < dt])
            if prev_hyb_dates:
                w_prev   = hybrid_weights_by_date[prev_hyb_dates[-1]]
                all_t_to = list(set(w_disp.index) | set(w_prev.index))
                turnover = (w_disp.reindex(all_t_to).fillna(0.0) -
                            w_prev.reindex(all_t_to).fillna(0.0)).abs().sum() * 100
                turn_str = f"  TO={turnover:.0f}%"
                all_turnover_ratios.append(turnover)
            else:
                turn_str = ""

            # -- Individual stock contributions for this rebal period ----------
            next_rebal_dates = [d for d in calc_dates if d > dt]
            period_end = next_rebal_dates[0] if next_rebal_dates else Pxs_df.index[-1]
            contrib_s  = pd.Series(dtype=float)
            if period_end in Pxs_df.index and dt in Pxs_df.index:
                px_start  = Pxs_df.loc[dt,         [t for t in w_disp.index if t in Pxs_df.columns]]
                px_end_p  = Pxs_df.loc[period_end, [t for t in w_disp.index if t in Pxs_df.columns]]
                stk_ret   = (px_end_p / px_start - 1).fillna(0.0)
                port_ret  = (w_disp.reindex(stk_ret.index).fillna(0.0) * stk_ret).sum()
                contrib_s = stk_ret
                c_avg    = contrib_s.mean() * 100
                c_med    = contrib_s.median() * 100
                c_best   = contrib_s.max() * 100
                c_worst  = contrib_s.min() * 100
                c_std    = contrib_s.std() * 100
                contrib_str = (f"  port={port_ret*100:+.1f}%  "
                               f"avg={c_avg:+.1f}%  med={c_med:+.1f}%  "
                               f"best={c_best:+.1f}%  worst={c_worst:+.1f}%  "
                               f"std={c_std:.1f}%")
                all_stock_returns.extend(contrib_s.tolist())
                all_portfolio_returns.append(port_ret * 100)
                period_str = f"  [{dt.date()} -> {period_end.date()}]"
            else:
                contrib_str = ""
                period_str  = ""

            # Source tag: use top_n_by from MVO diag; alpha-only = not in any matrix
            mvo_tickers_set   = set(w_mvo_nz.index)
            alpha_tickers_set = set(alpha_w_dt.index) if not alpha_w_dt.empty else set()

            n_illiquid = len(cands) - len(cands_liquid)
            liq_str    = f"  liq_filt={n_illiquid}" if n_illiquid > 0 else ""
            print(f"\n  -- {dt.date()}  [{i+1}/{n}]  regime={regime_lbl}  "
                  f"n={n_disp}  eff_N={eff_n:.1f}  "
                  f"min={w_disp.min():.1%}  max={w_disp.max():.1%}"
                  f"{turn_str}{liq_str} --")
            if contrib_str:
                print(f"  period{period_str}{contrib_str}")
            print(f"  {'Ticker':<8}  {'Weight%':>7}  {'AnnAlpha%':>10}  "
                  f"{'AnnVol%':>8}  {'Z-score':>8}  {'Contrib%':>9}  {'Sector':<28}  Source")
            print(f"  {'-'*105}")
            if _advp_capped.get(dt):
                print(f"  *** ADVP cap active (AUM=${AUM*_running_nav:,.0f}, "
                      f"cap={advp_cap:.1%} ADV): "
                      f"{', '.join(sorted(_advp_capped[dt]))}")
            for tkr, wt in w_disp.sort_values(ascending=False).items():
                sec   = sectors_s.get(tkr, '')
                a     = alpha_ann.get(tkr, 0.0) * 100
                v     = vol_ann.get(tkr,   0.0) * 100
                z     = z_s.get(tkr,       0.0)
                c     = contrib_s.get(tkr, np.nan) * 100 if not contrib_s.empty else np.nan
                c_str = f"{c:>+8.2f}%" if not np.isnan(c) else f"{'n/a':>9}"
                # Source: show matrix tags if in MVO universe, else 'alpha'
                if tkr in mvo_tickers_set:
                    in_m = '/'.join(m for m in ['E','L','F','P']
                                    if tkr in top_n_by.get(m, []))
                    src  = (in_m + '/E') if in_m else 'mvo'
                else:
                    src  = 'alpha'
                advp_flag = '***' if tkr in _advp_capped.get(dt, set()) else '   '
                print(f"  {tkr:<8}  {wt*100:>6.2f}%  {a:>+9.1f}%  "
                      f"{v:>7.1f}%  {z:>+8.3f}  {c_str}  {sec:<28}  {src}{advp_flag}")

    # hybrid_weights_by_date is already fully populated from the main loop above

    print(f"\n  Weights computed: "
          f"alpha={len(alpha_weights_by_date)}, "
          f"mvo={len(mvo_weights_by_date)}, "
          f"hybrid={len(hybrid_weights_by_date)}, "
          f"baseline={len(quality_factor_by_date)}")
    if _advp_capped:
        n_capped_dates  = len(_advp_capped)
        all_capped_tkrs = set().union(*_advp_capped.values())
        print(f"  ADVP cap triggered: {n_capped_dates} rebal dates, "
              f"{len(all_capped_tkrs)} unique tickers: {sorted(all_capped_tkrs)}")

    # -- [4/4] NAV series ------------------------------------------------------
    print("\n[4/4] Computing NAV series...")

    nav_alpha    = _mb_run_nav(alpha_weights_by_date,  calc_dates, Pxs_df,
                              cost_by_date=_cost_by_date["alpha"])
    nav_mvo      = _mb_run_nav(mvo_weights_by_date,    calc_dates, Pxs_df,
                              cost_by_date=_cost_by_date["mvo"])
    nav_hybrid   = _mb_run_nav(hybrid_weights_by_date, calc_dates, Pxs_df,
                              cost_by_date=_cost_by_date["hybrid"])

    # -- Smart hybrid: switch regime based on running drawdown of hybrid NAV --
    smart_hybrid_weights_by_date = {}
    nav_h_hwm = 1.0
    for rdt in sorted(hybrid_weights_by_date.keys()):
        if rdt in nav_hybrid.index:
            nav_h_hwm = max(nav_h_hwm, nav_hybrid.loc[rdt])
            dd = nav_hybrid.loc[rdt] / nav_h_hwm - 1
        else:
            dd = 0.0
        if dd >= -SH_DD_ALPHA:
            w_smart = alpha_weights_by_date.get(rdt, pd.Series(dtype=float))
        elif dd >= -SH_DD_HYBRID:
            w_smart = hybrid_weights_by_date.get(rdt, pd.Series(dtype=float))
        else:
            w_smart = mvo_weights_by_date.get(rdt, pd.Series(dtype=float))
        if not w_smart.empty and w_smart.sum() > 0:
            smart_hybrid_weights_by_date[rdt] = w_smart / w_smart.sum()
    # Trading costs for smart hybrid
    _prev_w_sm = pd.Series(dtype=float)
    for rdt in sorted(smart_hybrid_weights_by_date.keys()):
        w_sm     = smart_hybrid_weights_by_date[rdt]
        all_t_sm = list(set(w_sm.index) | set(_prev_w_sm.index))
        to_sm    = (w_sm.reindex(all_t_sm).fillna(0) -
                    _prev_w_sm.reindex(all_t_sm).fillna(0)).abs().sum() / 2
        _cost_by_date["smart"][rdt] = to_sm * TRADING_COST_BPS / 10000
        _prev_w_sm = w_sm

    nav_smart = _mb_run_nav(smart_hybrid_weights_by_date, calc_dates, Pxs_df,
                             cost_by_date=_cost_by_date["smart"])

    # ── Dynamic rebalancing strategy ─────────────────────────────────────────
    # Uses daily cached trigger variables to decide when to rebalance.
    # Rebalance if ANY of:
    #   1. Regime switch (always, ignores min hold)
    #   2. TO > DYN_TO_THRESHOLD AND vol_diff < DYN_VOLDIFF_CAP
    #   3. vol_diff < DYN_VOLDIFF_DERISK (de-risk, ignores min hold)
    # All subject to DYN_MIN_HOLD_DAYS except regime switch and de-risk.
    dyn_weights_by_date = {}
    _cost_by_date["dynamic"] = {}
    _dyn_rebal_log = []   # collect rebal info for forward-looking display
    try:
        triggers_dyn = load_daily_portfolios(params_hash, MB_MODEL_VER)
        with ENGINE.connect() as _conn:
            _trig_rows = _conn.execute(text(f"""
                SELECT date, implied_turnover, vol_diff, drawdown, live_strategy
                FROM {DAILY_TRIGGER_TBL}
                WHERE params_hash=:ph AND model_version=:mv
                ORDER BY date
            """), {'ph': params_hash, 'mv': MB_MODEL_VER}).fetchall()
        trig_df = pd.DataFrame(
            _trig_rows,
            columns=['date','implied_turnover','vol_diff','drawdown','live_strategy']
        )
        trig_df['date'] = pd.to_datetime(trig_df['date'])
        trig_df = trig_df.set_index('date')

        if not trig_df.empty and triggers_dyn:
            w_dyn_live      = pd.Series(dtype=float)
            w_dyn_live_prev_dt = None   # date when w_dyn_live was last set
            last_rebal_dt   = None
            prev_regime     = None    # signal regime (previous day)
            deployed_regime = None    # regime actually in the portfolio
            _prev_w_dyn     = pd.Series(dtype=float)

            for dt in sorted(trig_df.index):
                if dt not in triggers_dyn:
                    continue
                row      = trig_df.loc[dt]
                vd_val   = float(row['vol_diff'])
                regime   = str(row['live_strategy'])
                w_new    = triggers_dyn[dt].get('smart', pd.Series(dtype=float))
                if w_new.empty:
                    continue

                days_held = (dt - last_rebal_dt).days if last_rebal_dt else 999

                # Drift deployed portfolio to today for accurate TO computation
                if not w_dyn_live.empty and w_dyn_live_prev_dt is not None:
                    tks_drift = [t for t in w_dyn_live.index if t in Pxs_df.columns]
                    if tks_drift and w_dyn_live_prev_dt in Pxs_df.index and dt in Pxs_df.index:
                        px_prev_d = Pxs_df.loc[w_dyn_live_prev_dt, tks_drift]
                        px_cur_d  = Pxs_df.loc[dt,                 tks_drift]
                        w_drift_d = w_dyn_live.reindex(tks_drift) * (px_cur_d / px_prev_d).fillna(1)
                        if w_drift_d.sum() > 0:
                            w_dyn_live = w_drift_d / w_drift_d.sum()
                w_dyn_live_prev_dt = dt

                # Recompute TO against actual drifted deployed portfolio
                if w_dyn_live.empty:
                    to_val = 1.0
                else:
                    all_t_to = list(set(w_new.index) | set(w_dyn_live.index))
                    to_val   = (w_new.reindex(all_t_to).fillna(0) -
                                w_dyn_live.reindex(all_t_to).fillna(0)).abs().sum() / 2

                # Determine if we should rebalance — all triggers respect min hold
                # Regime switch: only if signal regime differs from DEPLOYED regime
                regime_switch = (deployed_regime is not None and
                                 regime != deployed_regime and
                                 days_held >= DYN_MIN_HOLD_DAYS)
                # Regime-specific TO threshold
                _to_thresh = (DYN_TO_THRESHOLD_MVO    if deployed_regime == 'mvo' else
                              DYN_TO_THRESHOLD_HYBRID if deployed_regime == 'hybrid' else
                              DYN_TO_THRESHOLD_ALPHA)
                to_trigger    = (to_val > _to_thresh and
                                 vd_val < DYN_VOLDIFF_CAP and
                                 days_held >= DYN_MIN_HOLD_DAYS)
                derisk        = (vd_val < DYN_VOLDIFF_DERISK and
                                 days_held >= DYN_MIN_HOLD_DAYS)

                should_rebal  = regime_switch or to_trigger or derisk

                if should_rebal or w_dyn_live.empty:
                    # Trading cost
                    if not _prev_w_dyn.empty:
                        all_t_dyn = list(set(w_new.index) | set(_prev_w_dyn.index))
                        to_dyn    = (w_new.reindex(all_t_dyn).fillna(0) -
                                     _prev_w_dyn.reindex(all_t_dyn).fillna(0)).abs().sum() / 2
                        _cost_by_date["dynamic"][dt] = to_dyn * TRADING_COST_BPS / 10000
                    else:
                        to_dyn = 1.0  # first rebalance

                    # Trigger label and type
                    if w_dyn_live.empty:
                        trigger_lbl  = 'init'
                        trigger_type = 'init'
                    elif regime_switch:
                        trigger_lbl  = f'regime->{regime}'
                        trigger_type = 'regime'
                    elif derisk:
                        trigger_lbl  = f'derisk(vd={vd_val*100:+.1f}%)'
                        trigger_type = 'derisk'
                    else:
                        trigger_lbl  = f'TO={to_val*100:.1f}%,vd={vd_val*100:+.1f}% (thr={_to_thresh*100:.0f}%)'
                        trigger_type = 'turnover'

                    # Store rebalance info for forward-looking display (second pass)
                    _dyn_rebal_log.append({
                        'dt': dt, 'trigger': trigger_lbl, 'trigger_type': trigger_type,
                        'regime': regime, 'w': w_new.copy(), 'to_dyn': to_dyn,
                        'days_held': days_held, 'eff_n': 1.0/(w_new**2).sum() if len(w_new)>0 else 0,
                        'diag_dt': (_past_d[-1] if (_past_d := [d for d in sorted(mvo_weights_by_date.keys()) if d <= dt]) else
                                    ([d for d in sorted(mvo_weights_by_date.keys()) if d > dt] or [None])[0]),
                    })

                    w_dyn_live      = w_new.copy()
                    w_dyn_live_prev_dt = dt   # reset drift baseline to rebalance date
                    last_rebal_dt   = dt
                    _prev_w_dyn     = w_new.copy()
                    deployed_regime = regime   # update deployed regime on actual rebalance

                dyn_weights_by_date[dt] = w_dyn_live
                prev_regime = regime   # always track signal regime

            n_rebal_total = len(_cost_by_date['dynamic'])
            print(f"\n  Dynamic strategy: {len(dyn_weights_by_date)} daily weights, "
                  f"{n_rebal_total} rebalances  "
                  f"(avg {len(dyn_weights_by_date)/max(n_rebal_total,1):.1f} days between rebalances)")

            # -- Second pass: display each rebalance with forward-looking period stats --
            print(f"\n  {'='*95}")
            print(f"  DYNAMIC REBALANCING LOG")
            print(f"  {'='*95}")
            rebal_dates_list = [r['dt'] for r in _dyn_rebal_log]
            for ri, rec in enumerate(_dyn_rebal_log):
                dt_r    = rec['dt']
                w_r     = rec['w']
                # Forward-looking period: this rebal -> next rebal (or last trading day)
                next_dt = rebal_dates_list[ri+1] if ri+1 < len(rebal_dates_list) else Pxs_df.index[-1]
                contrib_str = ''
                if dt_r in Pxs_df.index and next_dt in Pxs_df.index:
                    tks_r    = [t for t in w_r.index if t in Pxs_df.columns]
                    px_s_r   = Pxs_df.loc[dt_r,   tks_r]
                    px_e_r   = Pxs_df.loc[next_dt, tks_r]
                    stk_r_r  = (px_e_r / px_s_r - 1).fillna(0)
                    port_r_r = (w_r.reindex(tks_r).fillna(0) * stk_r_r).sum()
                    contrib_str = (f"  port={port_r_r*100:+.1f}%  "
                                   f"avg={stk_r_r.mean()*100:+.1f}%  "
                                   f"med={stk_r_r.median()*100:+.1f}%  "
                                   f"best={stk_r_r.max()*100:+.1f}%  "
                                   f"worst={stk_r_r.min()*100:+.1f}%  "
                                   f"std={stk_r_r.std()*100:.1f}%")
                _ttype_map = {'init': 'INIT', 'regime': 'REGIME SWITCH',
                              'derisk': 'DE-RISK', 'turnover': 'TURNOVER'}
                _ttype_lbl = _ttype_map.get(rec.get('trigger_type', 'turnover'), 'TURNOVER')
                print(f"\n  -- {dt_r.date()}  [rebal #{ri+1}]  [{_ttype_lbl}]  "
                      f"trigger={rec['trigger']}  regime={rec['regime']}  "
                      f"n={len(w_r)}  eff_N={rec['eff_n']:.1f}  "
                      f"held={rec['days_held']}d  TO={rec['to_dyn']*100:.1f}% --")
                if contrib_str:
                    print(f"  period [{dt_r.date()} -> {next_dt.date()}]{contrib_str}")
                print(f"  {'Ticker':<8}  {'Weight%':>7}  {'AnnAlpha%':>10}  "
                      f"{'AnnVol%':>8}  {'Z-score':>8}  {'Sector':<28}  Source")
                print(f"  {'-'*95}")
                # Diagnostics
                _diag_dt_r = rec['diag_dt']
                _diag_r    = (_diag_by_date.get(_diag_dt_r) or _last_mvo_diag
                              if _diag_dt_r else _last_mvo_diag)
                _alpha_r   = _diag_r.get('alpha_ann', pd.Series(dtype=float))
                _vol_r     = _diag_r.get('vol_ann',   pd.Series(dtype=float))
                _z_r       = _diag_r.get('z_s',       pd.Series(dtype=float))
                _top_r     = _diag_r.get('top_n_by',  {})
                _mvo_r     = set(mvo_weights_by_date.get(
                                  _diag_dt_r or dt_r, pd.Series(dtype=float)).index)
                for tkr, wt in w_r.sort_values(ascending=False).items():
                    sec  = sectors_s.get(tkr, '')
                    a    = _alpha_r.get(tkr, 0.0) * 100
                    v    = _vol_r.get(tkr,   0.0) * 100
                    z    = _z_r.get(tkr,     0.0)
                    if tkr in _mvo_r:
                        in_m = '/'.join(m for m in ['E','L','F','P']
                                        if tkr in _top_r.get(m, []))
                        src  = (in_m + '/E') if in_m else 'mvo'
                    else:
                        src = 'alpha'
                    advp_flag = '***' if tkr in _advp_capped.get(rdt, set()) else '   '
                    print(f"  {tkr:<8}  {wt*100:>6.2f}%  {a:>+9.1f}%  "
                          f"{v:>7.1f}%  {z:>+8.3f}  {sec:<28}  {src}{advp_flag}")
        else:
            print("  Dynamic strategy: no trigger data -- run run_daily_cache_build() first")
    except Exception as e:
        import traceback
        print(f"  Dynamic strategy failed: {e}")
        traceback.print_exc()

    nav_dynamic = (_mb_run_nav(dyn_weights_by_date, sorted(dyn_weights_by_date.keys()),
                               Pxs_df, cost_by_date=_cost_by_date["dynamic"])
                   if dyn_weights_by_date else pd.Series(dtype=float))

    # Baseline trading costs -- equal weight assumed, turnover from port changes
    _prev_bl = pd.Series(dtype=float)
    for rdt in sorted(quality_factor_by_date.keys()):
        fdf    = quality_factor_by_date[rdt]
        ranked = fdf.sort_values('factor', ascending=False).head(top_n)
        w_bl   = pd.Series(1.0 / len(ranked), index=ranked.index)
        all_t_bl = list(set(w_bl.index) | set(_prev_bl.index))
        to_bl    = (w_bl.reindex(all_t_bl).fillna(0) -
                    _prev_bl.reindex(all_t_bl).fillna(0)).abs().sum() / 2
        _cost_by_date["baseline"][rdt] = to_bl * TRADING_COST_BPS / 10000
        _prev_bl = w_bl

    nav_baseline_raw, port_baseline = run_backtest(
        factor_by_date = quality_factor_by_date,
        calc_dates     = calc_dates,
        Pxs_df         = Pxs_df,
        use_vol_filter = use_vol,
        mktcap_floor   = mktcap_floor,
        sector_cap     = sector_cap,
        top_n          = top_n,
        prefilt_pct    = prefilt_pct,
        conc_factor    = 1.0,
    )
    # Apply baseline costs post-hoc
    nav_baseline = nav_baseline_raw.copy()
    for rdt in sorted(_cost_by_date["baseline"].keys()):
        if rdt in nav_baseline.index:
            cost_factor = 1 - _cost_by_date["baseline"][rdt]
            nav_baseline.loc[rdt:] *= cost_factor

    # Port DataFrames
    port_alpha = pd.DataFrame.from_dict(
        {dt: (list(w.index) + [None] * max(0, top_n - len(w)))[:top_n]
         for dt, w in alpha_weights_by_date.items()},
        orient='index',
        columns=[f'Stock{i+1}' for i in range(top_n)]
    )
    port_mvo = pd.DataFrame.from_dict(
        {dt: (list(w[w > 1e-6].index) +
              [None] * max(0, top_n - (w > 1e-6).sum()))[:top_n]
         for dt, w in mvo_weights_by_date.items()},
        orient='index',
        columns=[f'Stock{i+1}' for i in range(top_n)]
    )
    max_hybrid_n = max((len(w) for w in hybrid_weights_by_date.values()), default=top_n)
    port_hybrid = pd.DataFrame.from_dict(
        {dt: (list(w[w > 1e-6].index) +
              [None] * max(0, max_hybrid_n - (w > 1e-6).sum()))[:max_hybrid_n]
         for dt, w in hybrid_weights_by_date.items()},
        orient='index',
        columns=[f'Stock{i+1}' for i in range(max_hybrid_n)]
    ) if hybrid_weights_by_date else pd.DataFrame()

    # -- Hedge layer (optional) ------------------------------------------------
    nav_dyn_hedged = None
    hedge_results  = None
    if hedge_multi is not None and hedges_l is not None:
        _trigger_assets = hedge_trigger_assets or ['QQQ']
        print(f"\n  {'='*72}")
        print(f"  RUNNING HEDGE LAYER")
        print(f"  {'='*72}")
        hedge_results = run_hedge_backtest(
            Pxs_df            = Pxs_df,
            multi             = hedge_multi,
            portfolio_weights = dyn_weights_by_date,
            rebal_dates       = [r['dt'] for r in _dyn_rebal_log],
            hedges_l          = hedges_l,
            trigger_assets    = _trigger_assets,
        )
        # Build hedged NAV: apply sweep amounts to dynamic NAV at each rebal date
        if nav_dynamic is not None and not nav_dynamic.empty:
            nav_dyn_hedged = nav_dynamic.copy()
            sweep_by_date  = hedge_results['hedge_sweep_by_date']
            cum_hedge      = 0.0
            for dt in nav_dyn_hedged.index:
                if dt in sweep_by_date:
                    cum_hedge += sweep_by_date[dt]
                # Hedged NAV = portfolio NAV * (1 + cumulative hedge contribution)
                nav_dyn_hedged[dt] = nav_dynamic[dt] * (1 + cum_hedge)
            nav_dyn_hedged.name = 'nav_dyn_hedged'

    # -- DD policy (8th strategy: Dynamic + Hedge + DD) -----------------------
    nav_dd = None
    _dd_log = []
    if nav_dyn_hedged is not None and not nav_dyn_hedged.empty:
        _lvl_str = '  '.join([f"-{d:.0%}→cut {c:.0%}" for d, c in DD_LEVELS])
        print(f"\n  {'='*72}")
        print(f"  DRAWDOWN POLICY  ({_lvl_str})")
        print(f"  Re-entry: +{DD_REENTRY_PCT:.0%} from trough (confirm {DD_REENTRY_CONFIRM}d, MVO mode)  "
              f"|  Annual reset if YTD dd < {DD_ANNUAL_RESET_PCT:.0%}")
        print(f"  HWM resets at each full re-entry")
        print(f"  {'='*72}")

        base_ret   = nav_dyn_hedged.pct_change().fillna(0)
        mvo_ret_s  = nav_mvo.pct_change().fillna(0) if nav_mvo is not None else base_ret

        dd_nav      = pd.Series(index=nav_dyn_hedged.index, dtype=float)
        dd_nav.iloc[0] = nav_dyn_hedged.iloc[0]
        exposure_s  = pd.Series(index=nav_dyn_hedged.index, dtype=float)
        exposure_s.iloc[0] = 1.0

        hwm          = nav_dyn_hedged.iloc[0]   # resets on full re-entry
        exposure     = 1.0
        trough       = nav_dyn_hedged.iloc[0]
        theo_nav     = nav_dyn_hedged.iloc[0]
        dd_active    = False
        confirm_cnt  = 0
        # Track which DD levels have fired (by index into DD_LEVELS)
        next_level_idx = 0   # next level to watch for

        for i, dt in enumerate(nav_dyn_hedged.index[1:], 1):
            prev_nav = dd_nav.iloc[i-1]

            # Update theoretical MVO NAV only when DD active
            if dd_active:
                theo_nav = theo_nav * (1 + mvo_ret_s.get(dt, 0))

            # Actual return scaled by exposure
            raw_r    = base_ret.get(dt, 0)
            curr_nav = prev_nav * (1 + raw_r * exposure)
            dd_nav.iloc[i]   = curr_nav
            exposure_s.iloc[i] = exposure

            # Update HWM
            if not dd_active:
                hwm = max(hwm, curr_nav)
            dd_current = curr_nav / hwm - 1

            # ── Check next DD level ────────────────────────────────────────
            if dd_active and next_level_idx < len(DD_LEVELS):
                dd_thresh, cut_frac = DD_LEVELS[next_level_idx]
                if dd_current <= -dd_thresh:
                    old_exp  = exposure
                    exposure = exposure * (1 - cut_frac)
                    trough    = curr_nav
                    theo_nav  = curr_nav
                    confirm_cnt = 0
                    next_level_idx += 1   # arm the next level
                    port_cost = (old_exp - exposure) * TRADING_COST_BPS / 10000
                    curr_nav  = curr_nav * (1 - port_cost)
                    dd_nav.iloc[i]    = curr_nav
                    exposure_s.iloc[i] = exposure
                    _dd_log.append({
                        'date': dt, 'event': f'DE-GROSS lv{next_level_idx}',
                        'dd': dd_current, 'exposure_from': old_exp,
                        'exposure_to': exposure,
                    })
                    print(f"\n  {dt.date()}  [DD POLICY: DE-GROSS lv{next_level_idx}]  "
                          f"dd={dd_current:.1%}  cut {cut_frac:.0%} of remaining  "
                          f"exposure {old_exp:.1%}→{exposure:.1%}  "
                          f"costs≈{port_cost*100:.2f}bps")

            # ── Check initial DD trigger (not yet active) ───────────────────
            elif not dd_active and next_level_idx < len(DD_LEVELS):
                dd_thresh, cut_frac = DD_LEVELS[next_level_idx]
                if dd_current <= -dd_thresh:
                    old_exp  = exposure
                    exposure = exposure * (1 - cut_frac)
                    dd_active = True
                    trough    = curr_nav
                    theo_nav  = curr_nav
                    confirm_cnt = 0
                    next_level_idx += 1
                    port_cost = (old_exp - exposure) * TRADING_COST_BPS / 10000
                    curr_nav  = curr_nav * (1 - port_cost)
                    dd_nav.iloc[i]    = curr_nav
                    exposure_s.iloc[i] = exposure
                    _dd_log.append({
                        'date': dt, 'event': f'DE-GROSS lv{next_level_idx}',
                        'dd': dd_current, 'exposure_from': old_exp,
                        'exposure_to': exposure,
                    })
                    print(f"\n  {dt.date()}  [DD POLICY: DE-GROSS lv{next_level_idx}]  "
                          f"dd={dd_current:.1%}  cut {cut_frac:.0%} of remaining  "
                          f"exposure {old_exp:.1%}→{exposure:.1%}  "
                          f"costs≈{port_cost*100:.2f}bps")

            # ── Re-entry logic (runs independently of DD level checks) ──────
            if dd_active and exposure < 1.0:
                trough = min(trough, curr_nav)
                theo_recovery = theo_nav / trough - 1

                # Condition 1: theoretical recovery from trough
                recovery_ok = theo_recovery >= DD_REENTRY_PCT
                # Condition 2: new calendar year AND YTD DD within limit
                yr_start_nav = dd_nav[dd_nav.index.year == dt.year]
                ytd_dd = (curr_nav / yr_start_nav.iloc[0] - 1
                          if len(yr_start_nav) > 1 else 0)
                new_year_ok = (dt.month == 1 and dt.day <= 10 and
                               ytd_dd >= -DD_ANNUAL_RESET_PCT)

                if recovery_ok or new_year_ok:
                    confirm_cnt += 1
                else:
                    confirm_cnt = 0

                if confirm_cnt >= DD_REENTRY_CONFIRM:
                    old_exp  = exposure
                    exposure = 1.0
                    dd_active = False
                    next_level_idx = 0   # reset all levels
                    hwm = curr_nav       # reset HWM from here
                    port_cost = (1.0 - old_exp) * TRADING_COST_BPS / 10000
                    curr_nav  = curr_nav * (1 - port_cost)
                    dd_nav.iloc[i]    = curr_nav
                    exposure_s.iloc[i] = exposure
                    confirm_cnt = 0
                    trigger = "recovery" if recovery_ok else "annual reset"
                    _dd_log.append({
                        'date': dt, 'event': 'RE-ENTRY 100%',
                        'theo_recovery': theo_recovery,
                        'exposure_from': old_exp, 'exposure_to': 1.0,
                        'trigger': trigger,
                    })
                    print(f"\n  {dt.date()}  [DD POLICY: FULL RE-ENTRY ({trigger})]  "
                          f"theoretical recovery={theo_recovery:.1%}  "
                          f"exposure {old_exp:.1%}→100%  MVO mode  "
                          f"costs≈{port_cost*100:.2f}bps  [HWM RESET]")

        nav_dd = dd_nav
        nav_dd.name = 'nav_dd'

        # -- Gross exposure plot -----------------------------------------------
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
            fig.patch.set_facecolor('#FAFAF9')
            for ax in axes:
                ax.set_facecolor('#FAFAF9')
                ax.grid(color='#D3D1C7', linewidth=0.4)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            axes[0].fill_between(exposure_s.index.to_numpy(),
                                 exposure_s.to_numpy(), 0,
                                 where=exposure_s.to_numpy() < 1.0,
                                 color='#D85A30', alpha=0.4, step='post',
                                 label='reduced exposure')
            axes[0].fill_between(exposure_s.index.to_numpy(),
                                 exposure_s.to_numpy(), 0,
                                 where=exposure_s.to_numpy() >= 1.0,
                                 color='#1D9E75', alpha=0.3, step='post',
                                 label='full exposure')
            axes[0].set_ylim(0, 1.2)
            axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axes[0].set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            axes[0].set_title('Gross Exposure (DD Policy)',
                              fontsize=10, fontweight='500', color='#2C2C2A')
            axes[0].legend(fontsize=8)

            axes[1].plot(nav_dyn_hedged.index.to_numpy(),
                         nav_dyn_hedged.to_numpy(),
                         color='#378ADD', lw=1.0, label='Dynamic+Hedge')
            axes[1].plot(nav_dd.index.to_numpy(),
                         nav_dd.to_numpy(),
                         color='#1D9E75', lw=1.0, label='Dyn+Hedge+DD')
            axes[1].set_title('NAV: Dynamic+Hedge vs DD Policy',
                              fontsize=10, fontweight='500', color='#2C2C2A')
            axes[1].legend(fontsize=8)
            plt.tight_layout()
            plt.show()
        except Exception as _e:
            print(f"  (exposure plot failed: {_e})")

        # Summary
        degross_events = [e for e in _dd_log if 'DE-GROSS' in e['event']]
        reentry_events = [e for e in _dd_log if 'RE-ENTRY' in e['event']]
        print(f"\n  DD Policy summary: {len(degross_events)} de-gross events, "
              f"{len(reentry_events)} re-entries")
        if degross_events:
            print(f"  {'Date':<12}  {'Event':<18}  {'DD':>8}  {'Exp From':>9}  {'Exp To':>8}")
            print(f"  {'-'*60}")
            for e in _dd_log:
                print(f"  {str(e['date'].date()):<12}  {e['event']:<18}  "
                      f"{e.get('dd', e.get('theo_recovery', 0)):>+7.1%}  "
                      f"{e.get('exposure_from', 0):>8.1%}  "
                      f"{e.get('exposure_to', 0):>8.1%}")
        if not _dd_log:
            print(f"  DD Policy: no events triggered over backtest period")

    # -- Summary ---------------------------------------------------------------
    print(f"\n  {'='*72}")
    print(f"  COMPARISON")
    print(f"  {'='*72}")
    print(f"  All returns net of {TRADING_COST_BPS}bps trading costs")
    print(f"  {'Strategy':<30} {'CAGR':>8} {'Vol':>8} "
          f"{'Sharpe':>8} {'MDD':>8} {'CAGR/DD':>9}")
    print(f"  {'-'*69}")
    for nav_s, lbl in [
        (nav_baseline, 'Baseline (quality)'),
        (nav_alpha,    f'Pure Alpha (conc={conc_factor:.1f}x)'),
        (nav_mvo,      f'MVO (IC={ic}, max={max_weight:.0%})'),
        (nav_hybrid,   'Hybrid (Alpha+MVO avg)'),
        (nav_smart,    f'Smart Hybrid (<{SH_DD_ALPHA:.0%}=alpha,<{SH_DD_HYBRID:.0%}=hyb,else MVO)'),
        (nav_dynamic,      f'Dynamic (TO>a{DYN_TO_THRESHOLD_ALPHA:.0%}/h{DYN_TO_THRESHOLD_HYBRID:.0%}/m{DYN_TO_THRESHOLD_MVO:.0%},hold={DYN_MIN_HOLD_DAYS}d)'),
        (nav_dyn_hedged,   'Dynamic + Hedge'),
        (nav_dd,           f'Dyn+Hedge+DD ({len(DD_LEVELS)} levels)'),
    ]:
        if nav_s is None or nav_s.empty:
            print(f"  {lbl:<30} {'(no data -- run run_daily_cache_build first)':>40}")
            continue
        n_yrs  = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr   = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1/n_yrs) - 1
        vol    = nav_s.pct_change().dropna().std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        mdd    = ((nav_s / nav_s.cummax()) - 1).min()
        cagr_dd = cagr / abs(mdd) if mdd != 0 else np.nan
        print(f"  {lbl:<30} {cagr*100:>7.1f}% {vol*100:>7.1f}% "
              f"{sharpe:>8.2f} {mdd*100:>7.1f}% {cagr_dd:>8.2f}x")

    print(f"\n  Yearly returns:")
    print(f"  {'Year':<6} {'Baseline':>10} {'Pure Alpha':>12} {'MVO':>10} {'Hybrid':>10} {'Smart':>10} {'Dynamic':>10} {'Dyn+Hedge':>11} {'DD Policy':>11}")
    print(f"  {'-'*104}")
    for yr in sorted(set(nav_baseline.index.year)):
        def yr_ret(nav_s):
            yr_nav = nav_s[nav_s.index.year == yr]
            if len(yr_nav) < 2:
                return np.nan
            return (yr_nav.iloc[-1] / yr_nav.iloc[0] - 1) * 100
        dyn_str = f"{yr_ret(nav_dynamic):>+9.2f}%" if (nav_dynamic is not None and not nav_dynamic.empty) else f"{'n/a':>10}"
        hdg_str = f"{yr_ret(nav_dyn_hedged):>+10.2f}%" if (nav_dyn_hedged is not None and not nav_dyn_hedged.empty) else f"{'n/a':>11}"
        dd_str  = f"{yr_ret(nav_dd):>+10.2f}%" if (nav_dd is not None and not nav_dd.empty) else f"{'n/a':>11}"
        print(f"  {yr:<6} {yr_ret(nav_baseline):>+9.2f}%  "
              f"{yr_ret(nav_alpha):>+10.2f}%  "
              f"{yr_ret(nav_mvo):>+9.2f}%  "
              f"{yr_ret(nav_hybrid):>+9.2f}%  "
              f"{yr_ret(nav_smart):>+9.2f}%  "
              f"{dyn_str}  {hdg_str}  {dd_str}")

    # -- Today snapshot --------------------------------------------------------
    today = Pxs_df.index[-1]
    print(f"\n  Computing today's snapshot ({today.date()})...")

    def _append_today(port_df, scores_dict, label):
        """Append today's top-N to port_df if not already present."""
        if today in port_df.index:
            return port_df
        if today not in scores_dict:
            print(f"  WARNING: no {label} scores for today")
            return port_df
        scores = scores_dict[today]
        if hasattr(scores, 'rename'):
            fdf = scores.rename('factor').to_frame()
        else:
            fdf = pd.Series(scores).rename('factor').to_frame()
        fdf['Sector'] = fdf.index.map(sectors_s)
        fdf = fdf.dropna(subset=['Sector'])
        fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
        if prefilt_pct < 1.0 and len(fdf) > top_n:
            n_keep = max(top_n, int(np.ceil(len(fdf) * prefilt_pct)))
            fdf    = fdf.nlargest(n_keep, 'factor')
        if len(fdf) < top_n:
            print(f"  WARNING: only {len(fdf)} {label} stocks for today")
            return port_df
        ranked = fdf.sort_values('factor', ascending=False)
        if sector_cap is not None:
            from composite_backtest import select_with_sector_cap
            top = select_with_sector_cap(ranked, sector_cap, top_n)
        else:
            top = ranked.head(top_n)
        row = {f'Stock{i+1}': t for i, t in enumerate(top.index)}
        for j in range(len(top), top_n):
            row[f'Stock{j+1}'] = None
        today_row = pd.DataFrame([row], index=[today])
        print(f"  {label} today: {top.index.tolist()}")
        return pd.concat([port_df, today_row])

    # Composite scores for today
    if today not in composite_by_date:
        today_comp, _ = _cb_build_composite_scores(
            universe=universe, calc_dates=pd.DatetimeIndex([today]),
            Pxs_df=Pxs_df, sectors_s=sectors_s,
            weights_by_year=weights_by_year, regime_s=regime_s,
            volumeTrd_df=volumeTrd_df, model_version=MB_MODEL_VER,
            exclude_factors=['OU'],
        )
    else:
        today_comp = {today: composite_by_date[today]}

    # Quality scores for today
    today_qual = {}
    if today in quality_wide.index:
        s = quality_wide.loc[today].dropna()
        if not s.empty:
            today_qual[today] = s

    port_alpha    = _append_today(port_alpha,    today_comp, 'Pure Alpha')
    port_mvo      = _append_today(port_mvo,      today_comp, 'MVO')
    port_baseline = _append_today(port_baseline, today_qual, 'Baseline')
    if not port_hybrid.empty:
        port_hybrid = _append_today(port_hybrid, today_comp, 'Smart Hybrid')

    print(f"\n  port_baseline : {len(port_baseline)} dates x {top_n} stocks (last = today)")
    print(f"  port_alpha    : {len(port_alpha)} dates x {top_n} stocks (last = today)")
    print(f"  port_mvo      : {len(port_mvo)} dates x {top_n} stocks (last = today)")
    print(f"  port_hybrid   : {len(port_hybrid)} dates x {len(port_hybrid.columns) if not port_hybrid.empty else 0} stocks (last = today)")


    # -- Live hybrid portfolio P&L since last rebalance -----------------------
    print("\n  " + "=" * 72)
    print("  LIVE SMART HYBRID PORTFOLIO -- P&L SINCE LAST REBALANCE")
    print("  " + "=" * 72)

    def _live_pnl_hybrid(weights_by_date, Pxs_df):
        if not weights_by_date:
            print("  No hybrid weights available.")
            return
        today_ts   = Pxs_df.index[-1]
        past_dates = sorted([d for d in weights_by_date if d < today_ts])
        if not past_dates:
            print("  No past rebalance dates found.")
            return
        rebal_dt = past_dates[-1]
        w0       = weights_by_date[rebal_dt]
        w0       = w0[w0 > 1e-6]
        tickers  = [t for t in w0.index if t in Pxs_df.columns]
        if not tickers:
            print("  No valid tickers in portfolio.")
            return
        w0 = w0.reindex(tickers) / w0.reindex(tickers).sum()

        # Price series starting from the day AFTER rebalance to today
        all_px    = Pxs_df.loc[rebal_dt:, tickers].copy()
        if len(all_px) < 2:
            print("  Insufficient price data since rebalance.")
            return
        # Use rebalance date close as cost basis, show P&L from next trading day
        px_base   = all_px.iloc[0]
        px        = all_px.iloc[1:]   # exclude rebalance date itself
        px_norm   = px / px_base
        port_val  = (px_norm * w0.values).sum(axis=1) * 100
        daily_ret = port_val.pct_change()
        daily_ret.iloc[0] = port_val.iloc[0] / 100 - 1  # first day vs rebal close
        days_held = (today_ts - rebal_dt).days

        print(f"\n  Rebalance date : {rebal_dt.date()}  ({days_held} calendar days ago)")
        print(f"  Stocks         : {len(tickers)}")
        print(f"  Cum P&L        : {port_val.iloc[-1] - 100:+.2f}%")
        print(f"\n  {'Date':<12} {'Port':>8} {'Day P&L':>10} {'Cum P&L':>10}  Top contributors")
        print("  " + "-" * 80)

        for dt in daily_ret.index:
            day_ret  = daily_ret.loc[dt]
            cum_ret  = port_val.loc[dt] / 100 - 1
            loc      = px.index.get_loc(dt)
            px_prev  = px_base if loc == 0 else px.iloc[loc - 1]
            w_drift  = w0 * (px_prev / px_base)
            if w_drift.sum() > 0:
                w_drift = w_drift / w_drift.sum()
            stk_ret  = (px.loc[dt] / px_prev - 1).fillna(0)
            contrib  = (stk_ret * w_drift).sort_values()
            top_neg  = contrib.nsmallest(3)
            top_pos  = contrib.nlargest(3)
            parts    = ([f"{t}:{v*100:+.1f}%" for t, v in top_neg.items()] +
                        [f"{t}:{v*100:+.1f}%" for t, v in top_pos.items()])
            print(f"  {str(dt.date()):<12} {port_val.loc[dt]:>7.2f}  "
                  f"{day_ret*100:>+8.2f}%  {cum_ret*100:>+8.2f}%  "
                  + "  ".join(parts))

    _live_pnl_hybrid(smart_hybrid_weights_by_date, Pxs_df)

    # -- Current live portfolio (most recent rebalance, held today) ------------
    print("\n  " + "=" * 72)
    print("  CURRENT LIVE SMART HYBRID PORTFOLIO (as of last rebalance)")
    print("  " + "=" * 72)
    today_ts    = Pxs_df.index[-1]
    past_rebals = sorted([d for d in smart_hybrid_weights_by_date if d <= today_ts])
    if past_rebals:
        live_dt = past_rebals[-1]
        w_live  = smart_hybrid_weights_by_date[live_dt]
        w_live  = w_live[w_live > 1e-6].sort_values(ascending=False)
        # Drifted weights: adjust for price moves since rebalance
        tickers_live = [t for t in w_live.index if t in Pxs_df.columns]
        if tickers_live and live_dt < today_ts:
            px_rebal = Pxs_df.loc[live_dt, tickers_live]
            px_today = Pxs_df.loc[today_ts, tickers_live]
            w_drift  = w_live.reindex(tickers_live) * (px_today / px_rebal)
            if w_drift.sum() > 0:
                w_drift = w_drift / w_drift.sum()
        else:
            w_drift = w_live.reindex(tickers_live) if tickers_live else w_live
        print(f"\n  Rebalance: {live_dt.date()}  |  As of: {today_ts.date()}  "
              f"|  {len(w_live)} positions")
        print(f"  {'Ticker':<8}  {'Rebal Wt%':>10}  {'Curr Wt%':>10}  {'Drift':>8}  Sector")
        print("  " + "-" * 60)
        for tkr in w_live.index:
            rw  = w_live.get(tkr, 0.0) * 100
            cw  = w_drift.get(tkr, 0.0) * 100 if tkr in tickers_live else rw
            sec = sectors_s.get(tkr, '')
            print(f"  {tkr:<8}  {rw:>9.2f}%  {cw:>9.2f}%  "
                  f"{cw-rw:>+7.2f}%  {sec}")


    # -- Dynamic strategy: live P&L since last rebalance ----------------------
    if dyn_weights_by_date:
        print("\n  " + "=" * 72)
        print("  LIVE DYNAMIC PORTFOLIO -- P&L SINCE LAST REBALANCE")
        print("  " + "=" * 72)
        dyn_rebal_only = {dt: w for dt, w in dyn_weights_by_date.items()
                          if dt in _cost_by_date.get('dynamic', {})}
        if dyn_weights_by_date:
            first_dt = sorted(dyn_weights_by_date.keys())[0]
            dyn_rebal_only[first_dt] = dyn_weights_by_date[first_dt]
        _live_pnl_hybrid(dyn_rebal_only, Pxs_df)

        # -- Current live dynamic portfolio ------------------------------------
        print("\n  " + "=" * 72)
        print("  CURRENT LIVE DYNAMIC PORTFOLIO (as of last rebalance)")
        print("  " + "=" * 72)
        past_rebals_dyn = sorted([d for d in dyn_rebal_only if d <= today_ts])
        if past_rebals_dyn:
            live_dt_dyn = past_rebals_dyn[-1]
            w_live_dyn  = dyn_weights_by_date[live_dt_dyn]
            w_live_dyn  = w_live_dyn[w_live_dyn > 1e-6].sort_values(ascending=False)
            tickers_dyn = [t for t in w_live_dyn.index if t in Pxs_df.columns]
            if tickers_dyn and live_dt_dyn < today_ts:
                px_rebal_dyn = Pxs_df.loc[live_dt_dyn, tickers_dyn]
                px_today_dyn = Pxs_df.loc[today_ts, tickers_dyn]
                w_drift_dyn  = w_live_dyn.reindex(tickers_dyn) * (px_today_dyn / px_rebal_dyn)
                if w_drift_dyn.sum() > 0:
                    w_drift_dyn = w_drift_dyn / w_drift_dyn.sum()
            else:
                w_drift_dyn = w_live_dyn.reindex(tickers_dyn) if tickers_dyn else w_live_dyn
            last_rebal_rec = next((r for r in reversed(_dyn_rebal_log)
                                   if r['dt'] == live_dt_dyn), None)
            if last_rebal_rec:
                ttype = last_rebal_rec.get('trigger_type', '').upper()
                trigger_info = f"  trigger={last_rebal_rec['trigger']}  [{ttype}]"
            else:
                trigger_info = ""
            print(f"\n  Rebalance: {live_dt_dyn.date()}  |  As of: {today_ts.date()}"
                  f"  |  {len(w_live_dyn)} positions{trigger_info}")
            print(f"  {'Ticker':<8}  {'Rebal Wt%':>10}  {'Curr Wt%':>10}  {'Drift':>8}  Sector")
            print("  " + "-" * 60)
            for tkr in w_live_dyn.index:
                rw  = w_live_dyn.get(tkr, 0.0) * 100
                cw  = w_drift_dyn.get(tkr, 0.0) * 100 if tkr in tickers_dyn else rw
                sec = sectors_s.get(tkr, '')
                print(f"  {tkr:<8}  {rw:>9.2f}%  {cw:>9.2f}%  {cw-rw:>+7.2f}%  {sec}")

    # -- Live hedge status -----------------------------------------------------
    if hedge_results is not None:
        print("\n  " + "=" * 72)
        print("  LIVE HEDGE STATUS")
        print("  " + "=" * 72)
        active = hedge_results.get('active_hedges', {})
        acct   = hedge_results['hedge_account_by_date']
        acct_s = pd.Series(acct)
        curr_bal = acct_s.iloc[-1] if not acct_s.empty else 0.0

        if active:
            # Find open entry from hedge_log
            open_events = [e for e in hedge_results['hedge_log'] if e['event'] == 'OPEN']
            last_open   = open_events[-1] if open_events else None
            entry_dt    = last_open['date'] if last_open else None
            days_open   = (today_ts - entry_dt).days if entry_dt else 0

            # Unrealised P&L
            unreal_pnl = 0.0
            for inst, h in active.items():
                if inst in Pxs_df.columns and today_ts in Pxs_df.index:
                    if h['entry_px'] and not np.isnan(h['entry_px']):
                        curr_px   = Pxs_df.loc[today_ts, inst]
                        unreal_pnl += -(curr_px / h['entry_px'] - 1) * h['weight']

            print(f"\n  Status         : HEDGED  (open since {entry_dt.date() if entry_dt else 'n/a'}, {days_open}d)")
            print(f"  Unrealised P&L : {unreal_pnl:+.4%}")
            print(f"  Hedge acct bal : {curr_bal:+.4%} (closed episodes)")
            print(f"\n  {'Instrument':<10}  {'Weight':>8}  {'Entry Px':>10}  {'Curr Px':>10}  {'Unreal P&L':>12}  {'Beta':>7}  {'Eff':>7}")
            print("  " + "-" * 70)
            for inst, h in active.items():
                if inst in Pxs_df.columns and today_ts in Pxs_df.index:
                    curr_px  = Pxs_df.loc[today_ts, inst]
                    inst_pnl = -(curr_px / h['entry_px'] - 1) * h['weight'] \
                               if h['entry_px'] and not np.isnan(h['entry_px']) else np.nan
                    print(f"  {inst:<10}  {h['weight']:>8.2%}  "
                          f"{h['entry_px']:>10.4f}  {curr_px:>10.4f}  "
                          f"{inst_pnl:>+11.4%}  {h['beta']:>7.3f}  {h['eff']:>7.3f}")
        else:
            print(f"\n  Status         : FLAT (no active hedges)")
            print(f"  Hedge acct bal : {curr_bal:+.4%} (pending sweep at next rebal)"
                  if curr_bal != 0 else
                  f"  Hedge acct bal : 0.00% (clean)")

    # -- Hypothetical smart hybrid rebalance as of today ---------------------
    print("\n  " + "=" * 72)
    print(f"  HYPOTHETICAL SMART HYBRID REBALANCE AS OF {today_ts.date()}")
    print("  " + "=" * 72)
    try:
        if today_ts in composite_by_date:
            comp_today = composite_by_date[today_ts]
        elif today_comp:
            comp_today = today_comp[today_ts]
        else:
            comp_today = None

        if comp_today is not None:
            cands_t      = comp_today.dropna()
            cands_t      = cands_t.loc[[t for t in cands_t.index if t in pxs_cols]]
            n_pf         = max(n_cands, int(np.ceil(len(cands_t) * prefilt_pct)))
            cands_t      = cands_t.nlargest(min(n_pf, len(cands_t)))
            mvo_cands_t  = cands_t.index.tolist()[:n_cands]
            valid_snap_t = [d for d in snapshot_dates if d <= today_ts]

            if valid_snap_t:
                w_mvo_t, mvo_diag_t = _mb_solve_mvo(
                    dt=today_ts, candidates=mvo_cands_t,
                    composite_scores=comp_today, Pxs_df=Pxs_df,
                    sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                    model_version=MB_MODEL_VER,
                    pca_var_threshold=pca_var_threshold,
                    ic=ic, max_weight=max_weight, min_weight=min_weight,
                    zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                    X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                    top_n=top_n, min_cov_matrices=min_cov_matrices,
                )

                # Alpha weights for today
                ranked_t     = cands_t.sort_values(ascending=False)
                alpha_port_t = [t for t in ranked_t.head(top_n).index if t in pxs_cols]
                n_pt         = len(alpha_port_t)
                n_top_t      = int(np.ceil(n_pt / 2))
                n_bot_t      = n_pt - n_top_t
                if conc_factor == 1.0 or n_bot_t == 0:
                    w_alpha_t = pd.Series(1.0 / n_pt, index=alpha_port_t)
                else:
                    top_a = conc_factor / (conc_factor + 1.0)
                    bot_a = 1.0 / (conc_factor + 1.0)
                    w_alpha_t = pd.Series({
                        t: (top_a / n_top_t if j < n_top_t else bot_a / n_bot_t)
                        for j, t in enumerate(alpha_port_t)
                    })

                if not w_mvo_t.empty and w_mvo_t.sum() > 0:
                    w_mvo_t_nz = w_mvo_t[w_mvo_t > 1e-6]

                    # Blind hybrid for today
                    all_t_hyp = list(set(w_alpha_t.index) | set(w_mvo_t_nz.index))
                    w_hyp_t   = (w_alpha_t.reindex(all_t_hyp).fillna(0.0) +
                                 w_mvo_t_nz.reindex(all_t_hyp).fillna(0.0)) / 2.0
                    w_hyp_t   = w_hyp_t[w_hyp_t > 0]
                    w_hyp_t   = w_hyp_t / w_hyp_t.sum()

                    # Current drawdown from smart hybrid running NAV
                    dd_today = _sh_nav_running / _sh_hwm - 1
                    if dd_today >= -SH_DD_ALPHA:
                        w_smart_t  = w_alpha_t
                        regime_t   = f"alpha (dd={dd_today*100:+.1f}%)"
                    elif dd_today >= -SH_DD_HYBRID:
                        w_smart_t  = w_hyp_t
                        regime_t   = f"hybrid (dd={dd_today*100:+.1f}%)"
                    else:
                        w_smart_t  = w_mvo_t_nz
                        regime_t   = f"MVO (dd={dd_today*100:+.1f}%)"
                    w_smart_t = w_smart_t[w_smart_t > 1e-6]
                    w_smart_t = w_smart_t / w_smart_t.sum()

                    mvo_set_t   = set(w_mvo_t_nz.index)
                    alpha_set_t = set(w_alpha_t.index)
                    top_n_by_t  = mvo_diag_t.get('top_n_by', {})
                    eff_n_t     = 1.0 / (w_smart_t**2).sum()

                    print(f"\n  regime={regime_t}  n={len(w_smart_t)}  "
                          f"eff_N={eff_n_t:.1f}  "
                          f"min={w_smart_t.min():.1%}  max={w_smart_t.max():.1%}")
                    vol_ann_t   = mvo_diag_t.get('vol_ann',   pd.Series(dtype=float))
                    z_s_t       = mvo_diag_t.get('z_s',       pd.Series(dtype=float))
                    alpha_ann_t = mvo_diag_t.get('alpha_ann', pd.Series(dtype=float))
                    print(f"  {'Ticker':<8}  {'Weight%':>7}  {'AnnAlpha%':>10}  "
                          f"{'AnnVol%':>8}  {'Z-score':>8}  {'Sector':<28}  Source")
                    print("  " + "-" * 95)
                    for tkr, wt in w_smart_t.sort_values(ascending=False).items():
                        sec  = sectors_s.get(tkr, '')
                        a    = alpha_ann_t.get(tkr, 0.0) * 100
                        v    = vol_ann_t.get(tkr,   0.0) * 100
                        z    = z_s_t.get(tkr,       0.0)
                        if tkr in mvo_set_t:
                            in_m = '/'.join(m for m in ['E','L','F','P']
                                            if tkr in top_n_by_t.get(m, []))
                            src  = (in_m + '/E') if in_m else 'mvo'
                        else:
                            src = 'alpha'
                        # Apply ADVP cap check for today's hypothetical
                        curr_aum_t = AUM * _running_nav
                        advp_flag_t = ''
                        if volumeRaw_df is not None and curr_aum_t > 0:
                            past_d = [d for d in Pxs_df.index if d <= today_ts][-ADV_WINDOW:]
                            if tkr in Pxs_df.columns and tkr in volumeRaw_df.columns:
                                px_s  = Pxs_df.loc[past_d, tkr].dropna()
                                vol_s = volumeRaw_df.loc[past_d, tkr].dropna()
                                com   = px_s.index.intersection(vol_s.index)
                                if len(com) >= 3:
                                    adv_d = (px_s.reindex(com) * vol_s.reindex(com)).median()
                                    cap_w = adv_d * advp_cap / curr_aum_t
                                    if wt > cap_w + 1e-4:
                                        advp_flag_t = '***'
                        print(f"  {tkr:<8}  {wt*100:>6.2f}%  {a:>+9.1f}%  "
                              f"{v:>7.1f}%  {z:>+8.3f}  {sec:<28}  {src}{advp_flag_t}")
        else:
            print("  No composite scores available for today.")
    except Exception as e:
        print(f"  Could not compute hypothetical smart hybrid rebalance: {e}")

    # -- Unified trade summary table -------------------------------------------
    print("\n  " + "=" * 72)
    print(f"  TRADE SUMMARY  —  {today_ts.date()}")
    print("  " + "=" * 72)
    try:
        # Determine if today has any ACTUAL events for each strategy
        dyn_rebal_today   = today_ts in set([r['dt'] for r in _dyn_rebal_log])
        hedge_event_today = hedge_results is not None and any(
            e['date'].date() == today_ts.date()
            for e in hedge_results.get('hedge_log', []))
        dd_event_today    = any(
            e['date'].date() == today_ts.date() for e in _dd_log)
        has_actual = dyn_rebal_today or hedge_event_today or dd_event_today

        # Current exposure for DD policy
        curr_exposure = exposure_s.iloc[-1] if (nav_dd is not None and
                        not nav_dd.empty and len(exposure_s) > 0) else 1.0

        # Get current deployed weights for each strategy
        # Dynamic: last rebalance weights
        past_dyn = sorted([d for d in dyn_weights_by_date if d <= today_ts])
        w_dyn_deployed = dyn_weights_by_date[past_dyn[-1]] if past_dyn else pd.Series(dtype=float)

        # Hypothetical new portfolio (already computed above as w_smart_t / w_mvo_t)
        # Use w_smart_t as the Dynamic hypothetical (same regime logic)
        # Fall back to w_dyn_deployed if not computed
        try:
            w_new_dyn = w_smart_t if 'w_smart_t' in dir() and w_smart_t is not None else w_dyn_deployed
        except Exception:
            w_new_dyn = w_dyn_deployed

        # For Dyn+Hedge: same stock weights, plus active hedge short positions
        active_hedges_now = hedge_results.get('active_hedges', {}) if hedge_results else {}

        # Compute allocation changes per strategy
        all_tickers = set(w_dyn_deployed.index) | set(w_new_dyn.index)
        # Add hedge instruments if involved
        if active_hedges_now:
            all_tickers |= set(active_hedges_now.keys())

        rows = []
        for tkr in sorted(all_tickers):
            # Dynamic column
            old_dyn = w_dyn_deployed.get(tkr, 0.0)
            new_dyn = w_new_dyn.get(tkr, 0.0)
            delta_dyn = (new_dyn - old_dyn) * 100

            # Dyn+Hedge column: same stock changes + hedge positions
            if tkr in active_hedges_now:
                # Hedge instrument — show current short weight (no change if already open)
                h = active_hedges_now[tkr]
                if hedge_event_today:
                    # Hedge just opened or closed today
                    close_events_today = [e for e in hedge_results.get('hedge_log',[])
                                         if e['date'].date() == today_ts.date()
                                         and e['event'] == 'CLOSE'
                                         and tkr in e.get('instruments',[])]
                    open_events_today  = [e for e in hedge_results.get('hedge_log',[])
                                         if e['date'].date() == today_ts.date()
                                         and e['event'] == 'OPEN'
                                         and tkr in e.get('instruments',[])]
                    if open_events_today:
                        delta_hdg = -h['weight'] * 100   # new short
                    elif close_events_today:
                        delta_hdg = +h['weight'] * 100   # covering short
                    else:
                        delta_hdg = 0.0
                else:
                    delta_hdg = 0.0  # active but no change today
            else:
                delta_hdg = delta_dyn  # same stock changes as Dynamic

            # DD Policy column: stock changes scaled by exposure
            delta_dd = delta_dyn * curr_exposure
            if dd_event_today:
                # Find today's DD event to get exposure change
                dd_ev_today = [e for e in _dd_log
                               if e['date'].date() == today_ts.date()]
                if dd_ev_today and tkr not in active_hedges_now:
                    ev = dd_ev_today[-1]
                    old_exp = ev.get('exposure_from', curr_exposure)
                    new_exp = ev.get('exposure_to', curr_exposure)
                    # Scale existing position by new exposure
                    delta_dd = old_dyn * (new_exp - old_exp) * 100

            if abs(delta_dyn) > 0.01 or abs(delta_hdg) > 0.01 or abs(delta_dd) > 0.01:
                rows.append((tkr, delta_dyn, delta_hdg, delta_dd))

        label = "[ACTUAL TRADES]" if has_actual else "[THEORETICAL — no changes required today]"
        print(f"\n  {label}")
        if rows:
            print(f"\n  {'Ticker':<10}  {'Δ Dynamic':>12}  {'Δ Dyn+Hedge':>13}  {'Δ DD Policy':>13}")
            print(f"  {'-'*52}")
            for tkr, dd, dh, dp in sorted(rows, key=lambda x: x[3], reverse=True):
                hedge_tag = '  [SHORT]' if tkr in active_hedges_now else ''
                print(f"  {tkr:<10}  {dd:>+11.2f}%  {dh:>+12.2f}%  {dp:>+12.2f}%{hedge_tag}")
        else:
            print(f"\n  No material allocation changes  (threshold: 0.01%)")
    except Exception as e:
        print(f"  Could not compute trade summary: {e}")

    # -- Trading cost summary -------------------------------------------------
    print("\n  " + "=" * 72)
    print(f"  TRADING COSTS SUMMARY  (@ {TRADING_COST_BPS}bps per side)")
    print("  " + "=" * 72)
    strat_labels = {
        "baseline": "Baseline",
        "alpha":    f"Pure Alpha",
        "mvo":      "MVO",
        "hybrid":   "Hybrid",
        "smart":    "Smart Hybrid",
        "dynamic":  "Dynamic",
    }

    # Build hedge costs by year from hedge_log
    hedge_costs_by_year = {}
    if hedge_results is not None:
        for e in hedge_results.get('hedge_log', []):
            if e['event'] == 'CLOSE':
                yr = e['date'].year
                ep_cost = sum(2 * TRADING_COST_BPS / 10000 * h['weight']
                              for h in e.get('details', {}).values())
                hedge_costs_by_year[yr] = hedge_costs_by_year.get(yr, 0) + ep_cost
    show_hedge = bool(hedge_costs_by_year)

    # Build DD policy costs by year from _dd_log
    dd_costs_by_year = {}
    if _dd_log:
        for e in _dd_log:
            yr = e['date'].year
            traded_pct = abs(e.get('exposure_to', 0) - e.get('exposure_from', 0))
            ep_cost    = traded_pct * TRADING_COST_BPS / 10000
            dd_costs_by_year[yr] = dd_costs_by_year.get(yr, 0) + ep_cost
    show_dd = bool(dd_costs_by_year)

    # Header
    print(f"\n  {'Year':<6}", end="")
    for s in strat_labels:
        print(f"  {strat_labels[s]:>13}", end="")
    if show_hedge:
        print(f"  {'Hedge Costs':>13}", end="")
    if show_dd:
        print(f"  {'DD Costs':>13}", end="")
    print()
    sep_width = 78 + (16 if show_hedge else 0) + (16 if show_dd else 0)
    print(f"  {'-'*sep_width}")

    all_years = sorted(set(d.year for costs in _cost_by_date.values()
                           for d in costs.keys()))
    for yr in all_years:
        print(f"  {yr:<6}", end="")
        for s in strat_labels:
            yr_cost = sum(v for d, v in _cost_by_date[s].items() if d.year == yr)
            print(f"  {yr_cost*100:>12.2f}%", end="")
        if show_hedge:
            hc = hedge_costs_by_year.get(yr, 0)
            print(f"  {hc*100:>12.2f}%", end="")
        if show_dd:
            dc = dd_costs_by_year.get(yr, 0)
            print(f"  {dc*100:>12.2f}%", end="")
        print()
    # Total row
    print(f"  {'Total':<6}", end="")
    for s in strat_labels:
        tot = sum(_cost_by_date[s].values())
        print(f"  {tot*100:>12.2f}%", end="")
    if show_hedge:
        print(f"  {sum(hedge_costs_by_year.values())*100:>12.2f}%", end="")
    if show_dd:
        print(f"  {sum(dd_costs_by_year.values())*100:>12.2f}%", end="")
    print()
    # Annualized row
    n_yrs_cost = (max(all_years) - min(all_years) + 1) if all_years else 1
    print(f"  {'Ann.':<6}", end="")
    for s in strat_labels:
        tot = sum(_cost_by_date[s].values())
        print(f"  {tot/n_yrs_cost*100:>12.2f}%", end="")
    if show_hedge:
        print(f"  {sum(hedge_costs_by_year.values())/n_yrs_cost*100:>12.2f}%", end="")
    if show_dd:
        print(f"  {sum(dd_costs_by_year.values())/n_yrs_cost*100:>12.2f}%", end="")
    print()

    # -- Dynamic trigger type summary --
    if _dyn_rebal_log:
        from collections import Counter
        type_counts = Counter(r.get('trigger_type', 'turnover') for r in _dyn_rebal_log)
        total_r     = len(_dyn_rebal_log)
        print(f"\n  Dynamic rebalancing trigger summary ({total_r} total):")
        print(f"  {'Trigger':<20}  {'Count':>6}  {'%':>7}  Distribution")
        print(f"  {'-'*55}")
        for ttype, tlbl in [('init',     'Initialisation'),
                             ('regime',   'Regime switch'),
                             ('turnover', 'Turnover > thres'),
                             ('derisk',   'De-risk (vol)')]:
            cnt = type_counts.get(ttype, 0)
            bar = '#' * int(cnt / total_r * 30)
            print(f"  {tlbl:<20}  {cnt:>6}  {cnt/total_r*100:>6.1f}%  {bar}")

    # -- Dynamic rebalancing frequency stats --
    if _cost_by_date.get('dynamic'):
        rebal_dates_dyn = sorted(_cost_by_date['dynamic'].keys())
        if len(rebal_dates_dyn) > 1:
            gaps = [(rebal_dates_dyn[i+1] - rebal_dates_dyn[i]).days
                    for i in range(len(rebal_dates_dyn)-1)]
            print(f"\n  Dynamic rebalancing frequency ({len(rebal_dates_dyn)} rebalances):")
            print(f"    Avg holding period : {np.mean(gaps):.1f} days")
            print(f"    Median             : {np.median(gaps):.1f} days")
            print(f"    Min                : {min(gaps)} days")
            print(f"    Max                : {max(gaps)} days")
            print(f"    First rebalance    : {rebal_dates_dyn[0].date()}")
            print(f"    Last rebalance     : {rebal_dates_dyn[-1].date()}")
            # Yearly breakdown
            print(f"\n    {'Year':<6}  {'Rebalances':>12}  {'Avg hold (days)':>16}")
            print(f"    {'-'*38}")
            all_years_dyn = sorted(set(d.year for d in rebal_dates_dyn))
            for yr in all_years_dyn:
                yr_rebals = [d for d in rebal_dates_dyn if d.year == yr]
                yr_gaps   = [(rebal_dates_dyn[rebal_dates_dyn.index(d)+1] - d).days
                             for d in yr_rebals
                             if rebal_dates_dyn.index(d)+1 < len(rebal_dates_dyn)
                             and rebal_dates_dyn[rebal_dates_dyn.index(d)+1].year == yr]
                avg_hold  = f"{np.mean(yr_gaps):.1f}" if yr_gaps else "n/a"
                print(f"    {yr:<6}  {len(yr_rebals):>12}  {avg_hold:>16}")

    # -- Consolidated portfolio return statistics (per holding period) --------
    import matplotlib.pyplot as plt
    if all_portfolio_returns:
        port_arr = np.array(all_portfolio_returns)
        q_labels = ['Q1 (0-20%)', 'Q2 (20-40%)', 'Q3 (40-60%)', 'Q4 (60-80%)', 'Q5 (80-100%)']
        port_q   = np.percentile(port_arr, [0, 20, 40, 60, 80, 100])

        print("\n  " + "=" * 72)
        print("  PORTFOLIO RETURN STATISTICS (per holding period)")
        print("  " + "=" * 72)
        print(f"\n  Total periods      : {len(port_arr)}")
        print(f"  Mean               : {port_arr.mean():>+.2f}%")
        print(f"  Median             : {np.median(port_arr):>+.2f}%")
        print(f"  Std Dev            : {port_arr.std():>.2f}%")
        print(f"  Min                : {port_arr.min():>+.2f}%")
        print(f"  Max                : {port_arr.max():>+.2f}%")
        pct_pos = (port_arr > 0).mean() * 100
        print(f"  % positive periods : {pct_pos:.1f}%")
        print(f"\n  Quintile boundaries:")
        for i, lbl in enumerate(q_labels):
            print(f"    {lbl:<18}  {port_q[i]:>+7.2f}%  ->  {port_q[i+1]:>+7.2f}%")
        if all_turnover_ratios:
            print(f"\n  Avg turnover per rebalance : {np.mean(all_turnover_ratios):.1f}%")
            print(f"  Median turnover            : {np.median(all_turnover_ratios):.1f}%")

        total_periods = sum(_sh_regime_counts.values())
        if total_periods > 0:
            print(f"\n  Smart hybrid regime prevalence ({total_periods} rebalances):")
            regime_display = {'alpha': 'alpha', 'hybrid': 'hybrid', 'mvo': 'mvo'}
            regime_bar     = {'alpha': 'a', 'hybrid': 'h', 'mvo': 'M'}
            for regime, count in _sh_regime_counts.items():
                bar = regime_bar[regime] * int(count / total_periods * 40)
                print(f"    {regime_display[regime]:<8}  {count:>4}  ({count/total_periods*100:>5.1f}%)  {bar}")

            # Yearly breakdown
            if _sh_regime_by_date:
                years = sorted(set(d.year for d in _sh_regime_by_date))
                print(f"\n  Regime breakdown by year:")
                print(f"  {'Year':<6}  {'Total':>6}  {'Alpha':>6}  {'Hybrid':>7}  {'MVO':>5}  Distribution")
                print(f"  {'-'*65}")
                for yr in years:
                    yr_dates   = {d: r for d, r in _sh_regime_by_date.items() if d.year == yr}
                    yr_total   = len(yr_dates)
                    yr_alpha   = sum(1 for r in yr_dates.values() if r == "alpha")
                    yr_hybrid  = sum(1 for r in yr_dates.values() if r == "hybrid")
                    yr_mvo     = sum(1 for r in yr_dates.values() if r == "mvo")
                    # Mini bar: alpha=green block, h=yellow, m=red
                    bar = ('a' * yr_alpha + 'h' * yr_hybrid + 'M' * yr_mvo)
                    print(f"  {yr:<6}  {yr_total:>6}  "
                          f"{yr_alpha:>4} ({yr_alpha/yr_total*100:>4.0f}%)  "
                          f"{yr_hybrid:>4} ({yr_hybrid/yr_total*100:>4.0f}%)  "
                          f"{yr_mvo:>3} ({yr_mvo/yr_total*100:>4.0f}%)  {bar}")

        # Portfolio returns histogram
        fig_port, ax_port = plt.subplots(figsize=(12, 4))
        fig_port.patch.set_facecolor('#FAFAF9')
        ax_port.set_facecolor('#FAFAF9')
        n_bins_p = min(40, max(10, len(port_arr) // 5))
        ax_port.hist(port_arr, bins=n_bins_p, color='#1D9E75', alpha=0.75,
                     edgecolor='white', linewidth=0.4)
        ax_port.axvline(port_arr.mean(),     color='#378ADD', linewidth=1.5,
                        linestyle='--', label=f"Mean {port_arr.mean():+.1f}%")
        ax_port.axvline(np.median(port_arr), color='#D85A30', linewidth=1.5,
                        linestyle='--', label=f"Median {np.median(port_arr):+.1f}%")
        ax_port.axvline(0, color='#888780', linewidth=0.8, linestyle=':')
        ax_port.set_xlabel("Hybrid portfolio return per holding period (%)",
                           fontsize=10, color='#5F5E5A')
        ax_port.set_ylabel("Count", fontsize=10, color='#5F5E5A')
        ax_port.set_title("Distribution of Portfolio Returns per Holding Period",
                          fontsize=12, fontweight='500', color='#2C2C2A')
        ax_port.legend(fontsize=9, framealpha=0.85)
        ax_port.grid(color='#D3D1C7', linewidth=0.5)
        ax_port.spines['top'].set_visible(False)
        ax_port.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    # -- Consolidated individual stock return statistics ----------------------
    if all_stock_returns:
        ret_arr = np.array(all_stock_returns) * 100  # in percent
        q_labels = ['Q1 (0-20%)', 'Q2 (20-40%)', 'Q3 (40-60%)', 'Q4 (60-80%)', 'Q5 (80-100%)']
        quintiles = np.percentile(ret_arr, [0, 20, 40, 60, 80, 100])

        print("\n  " + "=" * 72)
        print("  CONSOLIDATED INDIVIDUAL STOCK RETURN STATISTICS")
        print("  " + "=" * 72)
        print(f"\n  Total observations : {len(ret_arr)}")
        print(f"  Mean               : {ret_arr.mean():>+.2f}%")
        print(f"  Median             : {np.median(ret_arr):>+.2f}%")
        print(f"  Std Dev            : {ret_arr.std():>.2f}%")
        print(f"  Min                : {ret_arr.min():>+.2f}%")
        print(f"  Max                : {ret_arr.max():>+.2f}%")
        print(f"\n  Quintile boundaries:")
        for i, lbl in enumerate(q_labels):
            print(f"    {lbl:<18}  {quintiles[i]:>+7.2f}%  ->  {quintiles[i+1]:>+7.2f}%")
        # Histogram
        fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
        fig_hist.patch.set_facecolor('#FAFAF9')
        ax_hist.set_facecolor('#FAFAF9')
        pct5, pct95 = np.percentile(ret_arr, 5), np.percentile(ret_arr, 95)
        clip_arr = ret_arr[(ret_arr >= pct5) & (ret_arr <= pct95)]
        n_bins   = min(60, max(20, len(ret_arr) // 20))
        ax_hist.hist(ret_arr, bins=n_bins, color='#378ADD', alpha=0.75,
                     edgecolor='white', linewidth=0.4)
        ax_hist.axvline(ret_arr.mean(),    color='#1D9E75', linewidth=1.5,
                        linestyle='--', label=f"Mean {ret_arr.mean():+.1f}%")
        ax_hist.axvline(np.median(ret_arr), color='#D85A30', linewidth=1.5,
                        linestyle='--', label=f"Median {np.median(ret_arr):+.1f}%")
        ax_hist.axvline(0, color='#888780', linewidth=0.8, linestyle=':')
        ax_hist.set_xlabel("Individual stock return per holding period (%)",
                           fontsize=10, color='#5F5E5A')
        ax_hist.set_ylabel("Count", fontsize=10, color='#5F5E5A')
        ax_hist.set_title("Distribution of Individual Stock Returns (all periods)",
                          fontsize=12, fontweight='500', color='#2C2C2A')
        ax_hist.legend(fontsize=9, framealpha=0.85)
        ax_hist.grid(color='#D3D1C7', linewidth=0.5)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s, nav_hybrid, nav_smart,
             nav_dynamic, nav_dyn_hedged, nav_dd)

    return {
        'nav_baseline'           : nav_baseline,
        'nav_alpha'              : nav_alpha,
        'nav_mvo'                : nav_mvo,
        'nav_hybrid'                  : nav_hybrid,
        'nav_smart'                   : nav_smart,
        'smart_hybrid_weights_by_date': smart_hybrid_weights_by_date,
        'nav_dynamic'                 : nav_dynamic,
        'nav_dyn_hedged'              : nav_dyn_hedged,
        'nav_dd'                      : nav_dd,
        'hedge_results'               : hedge_results,
        'dyn_weights_by_date'         : dyn_weights_by_date,
        'dyn_rebal_dates'             : [r['dt'] for r in _dyn_rebal_log],
        'port_baseline'          : port_baseline,
        'port_alpha'             : port_alpha,
        'port_mvo'               : port_mvo,
        'port_hybrid'            : port_hybrid,
        'alpha_weights_by_date'  : alpha_weights_by_date,
        'mvo_weights_by_date'    : mvo_weights_by_date,
        'hybrid_weights_by_date' : hybrid_weights_by_date,
        'composite_by_date'      : composite_by_date,
        'regime_s'               : regime_s,
        'weights_by_year'        : weights_by_year,
    }

    # -- Load daily trigger variables from cache if available ------------------
    try:
        with ENGINE.connect() as _conn2:
            _trig_rows2 = _conn2.execute(text(f"""
                SELECT date, implied_turnover, vol_diff, drawdown, live_strategy
                FROM {DAILY_TRIGGER_TBL}
                WHERE params_hash=:ph AND model_version=:mv
                ORDER BY date
            """), {'ph': params_hash, 'mv': MB_MODEL_VER}).fetchall()
        triggers_df = pd.DataFrame(
            _trig_rows2,
            columns=['date','implied_turnover','vol_diff','drawdown','live_strategy']
        )
        triggers_df['date'] = pd.to_datetime(triggers_df['date'])
        triggers_df = triggers_df.set_index('date')
        if not triggers_df.empty:
            print(f"\n  Daily trigger variables loaded: {len(triggers_df)} dates")
            print(f"  (implied_turnover, vol_diff, drawdown, live_strategy)")
            results['daily_triggers'] = triggers_df
        else:
            print("\n  No daily trigger data found -- run run_daily_cache_build() first")
            results['daily_triggers'] = pd.DataFrame()
    except Exception as e:
        warnings.warn(f"  Could not load trigger data: {e}")
        results['daily_triggers'] = pd.DataFrame()

    return results
