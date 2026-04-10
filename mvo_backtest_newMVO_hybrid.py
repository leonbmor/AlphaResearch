#!/usr/bin/env python
# coding: utf-8

"""
mvo_backtest.py
===============
Runs three portfolios in parallel and compares performance:
  1. Baseline      — quality factor only (from primary_factor_backtest.py)
  2. Pure Alpha    — composite alpha signal, equal-weight + concentration
  3. MVO           — composite alpha signal, mean-variance optimized weights

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
            # Jupyter: no real fd — fall back to replacing sys.stdout
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
MB_MIN_COV_MATRICES  = 2      # default: stock must appear in >=2 matrices (0=alpha-only)


# ===============================================================================
# SELF-CONTAINED FLOOR/CAP (does not depend on mvo_diagnostics.py version)
# ===============================================================================

def _mb_floor_then_cap(w, min_weight, max_weight, max_iter=20):
    """
    Post-solve weight adjustment:

    Step 1 — Floor:
        Raise non-zero weights to min_weight, renormalize.
    Step 2 — Cap (iterative):
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

    # Factor-driven (self-contained — no _mvo_factor_cov dependency)
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
    """Self-contained X matrix builder — no _rd_build_X dependency."""
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
    # Bound to MB_START_DATE — no point building X before backtest begins
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
          f" → {to_build[-1].date() if to_build else 'none'})")

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
# MVO WEIGHT SOLVER — exact replica of mvo_diagnostics logic
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
    # ── Get cached X snapshot ─────────────────────────────────────────────────
    X_df_cached = None
    if X_snapshots is not None and snapshot_dates is not None:
        valid_snaps = [d for d in snapshot_dates if d <= dt]
        if valid_snaps:
            X_df_cached = X_snapshots[valid_snaps[-1]]

    # ── Return history ────────────────────────────────────────────────────────
    ret_df = (Pxs_df[candidates].pct_change()
              .dropna(how='all')
              .iloc[-MVO_LOOKBACK:]
              .dropna(axis=1, how='any'))
    valid = [t for t in candidates if t in ret_df.columns]
    if len(valid) < top_n:
        warnings.warn(f"  {dt.date()} SKIP: only {len(valid)} valid candidates")
        return pd.Series(dtype=float), {}

    # ── Build covariance matrices ─────────────────────────────────────────────
    try:
        valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens = \
            _mb_build_cov_matrices(dt, valid, Pxs_df, sectors_s,
                                    volumeTrd_df, model_version, pca_var_threshold,
                                    X_df_cached=X_df_cached)
    except Exception as e:
        warnings.warn(f"  {dt.date()} COV MATRIX FAILED: {e}")
        return pd.Series(dtype=float), {}

    # ── Alpha signal ──────────────────────────────────────────────────────────
    vol_daily = _mvo_ewma_vol(ret_df[valid], MVO_EWMA_HL)
    vol_ann   = vol_daily * np.sqrt(252)
    z_s       = composite_scores.reindex(valid).fillna(0.0)
    z_capped  = z_s.clip(-zscore_cap, zscore_cap)
    alpha     = (ic * vol_daily * z_capped).fillna(0.0)

    # ── Per-matrix MVO (unconstrained, post-solve floor+cap) ──────────────────
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

    # ── Eligibility: count appearances in 4 individual matrices (not ensemble) ─
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

    # ── Final MVO on ensemble cov, eligible universe ──────────────────────────
    elig_idx  = [valid.index(t) for t in eligible]
    S_ens_el  = Sigma_ens[np.ix_(elig_idx, elig_idx)]
    alpha_el  = alpha.reindex(eligible).fillna(0.0)

    w_out = _mb_max_alpha_portfolio(S_ens_el, alpha_el, eligible,
                                     risk_aversion, max_weight, min_weight)
    # Trim to exactly top_n by keeping highest-weight stocks, renorm
    if len(w_out) > top_n:
        w_out = w_out.nlargest(top_n)
        w_out = w_out / w_out.sum()


    # ── Diagnostics dict for display ──────────────────────────────────────────
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

def _mb_run_nav(weights_by_date, calc_dates, Pxs_df):
    """
    Compute NAV series from a dict of {rebal_date: pd.Series(weights)}.
    Mirrors run_backtest NAV logic from primary_factor_backtest.py.
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

def _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s, nav_hybrid=None):
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
    nb = nav_baseline.reindex(common)
    na = nav_alpha.reindex(common)
    nm = nav_mvo.reindex(common)
    nh = nav_hybrid.reindex(common) if nav_hybrid is not None else None

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
    if nh is not None: ps.append((nh,'Hybrid','#7F77DD',1.5))
    for nav_s, lbl, color, lw in ps:
        ax.plot(nav_s.index.to_numpy(), nav_s.values,
                label=lbl, color=color, linewidth=lw)
    ax.set_ylabel("NAV", fontsize=10, color='#5F5E5A')
    ax.set_title("Three-way NAV Comparison",
                 fontsize=12, fontweight='500', color='#2C2C2A')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Relative to baseline
    ax2 = axes[1]
    rs = [(na,'Pure alpha vs baseline',TEAL),(nm,'MVO vs baseline',BLUE)]
    if nh is not None: rs.append((nh,'Hybrid vs baseline','#7F77DD'))
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
    if nh is not None: ds.append((nh,'Hybrid','#7F77DD'))
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
                     force_rebuild_cache=False):
    """
    Run three portfolios in parallel: Baseline, Pure Alpha, MVO.

    Parameters
    ----------
    Pxs_df, sectors_s, weights_by_year, regime_s, volumeTrd_df
        — same as composite_backtest.py

    MVO parameters (function inputs, not user prompts):
    ic                 : float — Grinold-Kahn IC scaling (default 0.04)
    max_weight         : float — single-name cap (default 0.10)
    min_weight         : float — single-name floor (default 0.025)
    zscore_cap         : float — alpha z-score winsorization (default 2.5)
    pca_var_threshold  : float — PCA variance explained threshold (default 0.65)
    universe_mult      : int   — candidate pool = port_n × universe_mult (default 5)
    risk_aversion      : float — MVO risk aversion λ (default 1.0)
    force_rebuild_cache: bool  — False (default): load X cache from DB, only
                                 compute missing dates (typically just today).
                                 True: clear cache and rebuild all dates from scratch.
    """
    print("=" * 72)
    print("  MVO BACKTEST — Baseline vs Pure Alpha vs MVO")
    print("=" * 72)
    print(f"\n  MVO params: IC={ic}, max_w={max_weight}, min_w={min_weight}, "
          f"zscore_cap={zscore_cap}")
    print(f"  PCA thresh={pca_var_threshold}, universe_mult={universe_mult}, "
          f"risk_aversion={risk_aversion}")
    print(f"  force_rebuild_cache={force_rebuild_cache} (False=incremental/cache, True=full rebuild)\n")

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    # ── User prompts (identical to composite_backtest.py) ─────────────────────
    print("  PORTFOLIO CONSTRUCTION OPTIONS:")
    topn_input    = input(f"  Number of stocks [default={MB_TOP_N}]: ").strip()
    rebal_input   = input(f"  Rebalancing frequency in days [default={MB_REBAL_FREQ}]: ").strip()
    sector_input  = input("  Max stocks per sector (or Enter to skip): ").strip()
    mktcap_input  = input("  Min market cap floor ($M, or Enter to skip): ").strip()
    vol_input     = input("  Apply vol filter? (y/n) [default=n]: ").strip().lower()
    prefilt_input = input("  Pre-filter fraction by composite score 0<x<=1 (or Enter for none): ").strip()
    conc_input    = input("  Concentration factor for Pure Alpha >=1.0 (or Enter for equal weight): ").strip()

    top_n        = int(topn_input)           if topn_input    else MB_TOP_N
    rebal_freq   = int(rebal_input)          if rebal_input   else MB_REBAL_FREQ
    sector_cap   = int(sector_input)         if sector_input  else None
    mktcap_floor = float(mktcap_input) * 1e6 if mktcap_input  else None
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

    floor_inp = input("  Enforce floor/cap on hybrid portfolio? (y/n) [default=y]: ").strip().lower()
    enforce_hybrid_floor = (floor_inp != 'n')

    n_cands = top_n * universe_mult
    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"sector_cap={sector_cap}, mktcap_floor={mktcap_input or 'none'}, "
          f"vol_filter={use_vol}, prefilt={prefilt_pct:.0%}, "
          f"conc={conc_factor:.1f}x (pure alpha only)")
    print(f"  MVO candidate pool: {n_cands} stocks ({universe_mult}×{top_n})  |  "
          f"min_cov_matrices={min_cov_matrices}  |  "
          f"enforce_hybrid_floor={enforce_hybrid_floor}\n")

    # ── Universe and calc dates ───────────────────────────────────────────────
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

    # ── [1/4] Composite alpha scores ─────────────────────────────────────────
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

    # ── [2/4] X snapshots (cached) ────────────────────────────────────────────
    print("\n[2/4] Building X snapshots (monthly, cached)...")
    X_snapshots, _, _, _ = _mb_build_x_snapshots(
        calc_dates, Pxs_df, sectors_s,
        volumeTrd_df, MB_MODEL_VER, force_rebuild_cache
    )
    snapshot_dates = sorted(X_snapshots.keys())
    if snapshot_dates:
        print(f"  X snapshots available: {len(snapshot_dates)}  "
              f"({snapshot_dates[0].date()} → {snapshot_dates[-1].date()})")

    # ── [3/4] Compute portfolio weights per rebalance date ────────────────────
    print("\n[3/4] Computing portfolio weights...")

    alpha_weights_by_date = {}   # pure alpha (equal/concentration)
    mvo_weights_by_date   = {}   # MVO
    quality_factor_by_date = {}  # baseline

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

        # ── Baseline quality factor ───────────────────────────────────────────
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

        # ── Candidate universe ────────────────────────────────────────────────
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

        # ── Pure alpha weights (concentration) ────────────────────────────────
        ranked  = cands.sort_values(ascending=False)
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

        # ── MVO weights ───────────────────────────────────────────────────────
        # Use top n_cands candidates for MVO universe
        mvo_cands = candidates[:n_cands]
        valid_snap = [d for d in snapshot_dates if d <= dt]
        if not valid_snap:
            continue

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

        if not w_mvo.empty and w_mvo.sum() > 0:
            w_mvo_nz = w_mvo[w_mvo > 1e-6]
            mvo_weights_by_date[dt] = w_mvo_nz
            vol_ann   = mvo_diag.get('vol_ann',   pd.Series(dtype=float))
            z_s       = mvo_diag.get('z_s',       pd.Series(dtype=float))
            alpha_ann = mvo_diag.get('alpha_ann', pd.Series(dtype=float))
            top_n_by  = mvo_diag.get('top_n_by', {})  # keys: E, L, F, P

            # ── Build hybrid weights for display ──────────────────────────────
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

            # ── Per-date display: hybrid portfolio ────────────────────────────
            eff_n = 1.0 / (w_hyb**2).sum() if len(w_hyb) > 0 else 0
            n_hyb = len(w_hyb)
            print(f"\n  ── {dt.date()}  [{i+1}/{n}]  "
                  f"n={len(w_mvo_nz)} mvo / {n_hyb} hybrid  eff_N={eff_n:.1f}  "
                  f"min={w_hyb.min():.1%}  max={w_hyb.max():.1%}  "
                  f"({'▲' if n_hyb > top_n else '='}{n_hyb - top_n:+d} vs N={top_n}) ──")
            print(f"  {n_common} common  |  "
                  f"{len(alpha_only_tickers)} alpha-only  |  "
                  f"{len(mvo_only_tickers)} mvo-only")
            print(f"  {'Ticker':<8}  {'Weight%':>7}  {'AnnAlpha%':>10}  "
                  f"{'AnnVol%':>8}  {'Z-score':>8}  {'Sector':<28}  Source")
            print(f"  {'-'*95}")
            for tkr, wt in w_hyb.sort_values(ascending=False).items():
                sec  = sectors_s.get(tkr, '')
                a    = alpha_ann.get(tkr, 0.0) * 100
                v    = vol_ann.get(tkr,   0.0) * 100
                z    = z_s.get(tkr,       0.0)
                if tkr in alpha_only_tickers:
                    src = 'alpha'
                else:
                    in_m = '/'.join(m for m in ['E','L','F','P']
                                    if tkr in top_n_by.get(m, []))
                    src  = (in_m + '/E') if in_m else 'mvo'
                print(f"  {tkr:<8}  {wt*100:>6.2f}%  {a:>+9.1f}%  "
                      f"{v:>7.1f}%  {z:>+8.3f}  {sec:<28}  {src}")

    # ── Hybrid: simple average of alpha and MVO weights ──────────────────────
    hybrid_weights_by_date = {}
    for dt in set(alpha_weights_by_date) & set(mvo_weights_by_date):
        w_a = alpha_weights_by_date[dt]
        w_m = mvo_weights_by_date[dt]
        all_t = list(set(w_a.index) | set(w_m.index))
        w_avg = (w_a.reindex(all_t).fillna(0.0) + w_m.reindex(all_t).fillna(0.0)) / 2.0
        w_avg = w_avg[w_avg > 0]
        if w_avg.sum() > 0:
            w_avg = w_avg / w_avg.sum()
            if enforce_hybrid_floor:
                # Raise below-floor weights to min_weight,
                # funding the shortfall proportionally from above-floor stocks
                for _ in range(50):
                    below = w_avg < min_weight - 1e-9
                    if not below.any():
                        break
                    shortfall = (min_weight - w_avg[below]).sum()
                    w_avg[below] = min_weight
                    above = ~below
                    if w_avg[above].sum() > shortfall + 1e-9:
                        w_avg[above] -= shortfall * (w_avg[above] / w_avg[above].sum())
                    w_avg = w_avg.clip(lower=0)
                    w_avg = w_avg / w_avg.sum()
            hybrid_weights_by_date[dt] = w_avg

    print(f"\n  Weights computed: "
          f"alpha={len(alpha_weights_by_date)}, "
          f"mvo={len(mvo_weights_by_date)}, "
          f"hybrid={len(hybrid_weights_by_date)}, "
          f"baseline={len(quality_factor_by_date)}")

    # ── [4/4] NAV series ──────────────────────────────────────────────────────
    print("\n[4/4] Computing NAV series...")

    nav_alpha    = _mb_run_nav(alpha_weights_by_date,  calc_dates, Pxs_df)
    nav_mvo      = _mb_run_nav(mvo_weights_by_date,    calc_dates, Pxs_df)
    nav_hybrid   = _mb_run_nav(hybrid_weights_by_date, calc_dates, Pxs_df)

    nav_baseline, port_baseline = run_backtest(
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'='*72}")
    print(f"  COMPARISON")
    print(f"  {'='*72}")
    print(f"  {'Strategy':<30} {'CAGR':>8} {'Vol':>8} "
          f"{'Sharpe':>8} {'MDD':>8}")
    print(f"  {'-'*58}")
    for nav_s, lbl in [
        (nav_baseline, 'Baseline (quality)'),
        (nav_alpha,    f'Pure Alpha (conc={conc_factor:.1f}x)'),
        (nav_mvo,      f'MVO (IC={ic}, max={max_weight:.0%})'),
        (nav_hybrid,   'Hybrid (Alpha+MVO avg)'),
    ]:
        n_yrs  = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr   = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1/n_yrs) - 1
        vol    = nav_s.pct_change().dropna().std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        mdd    = ((nav_s / nav_s.cummax()) - 1).min()
        print(f"  {lbl:<30} {cagr*100:>7.1f}% {vol*100:>7.1f}% "
              f"{sharpe:>8.2f} {mdd*100:>7.1f}%")

    # Yearly returns
    print(f"\n  Yearly returns:")
    print(f"  {'Year':<6} {'Baseline':>10} {'Pure Alpha':>12} {'MVO':>10} {'Hybrid':>10}")
    print(f"  {'-'*54}")
    for yr in sorted(set(nav_baseline.index.year)):
        def yr_ret(nav_s):
            yr_nav = nav_s[nav_s.index.year == yr]
            if len(yr_nav) < 2:
                return np.nan
            return (yr_nav.iloc[-1] / yr_nav.iloc[0] - 1) * 100
        print(f"  {yr:<6} {yr_ret(nav_baseline):>+9.2f}%  "
              f"{yr_ret(nav_alpha):>+10.2f}%  "
              f"{yr_ret(nav_mvo):>+9.2f}%  "
              f"{yr_ret(nav_hybrid):>+9.2f}%")

    # ── Today snapshot ────────────────────────────────────────────────────────
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
        port_hybrid = _append_today(port_hybrid, today_comp, 'Hybrid')

    print(f"\n  port_baseline : {len(port_baseline)} dates x {top_n} stocks (last = today)")
    print(f"  port_alpha    : {len(port_alpha)} dates x {top_n} stocks (last = today)")
    print(f"  port_mvo      : {len(port_mvo)} dates x {top_n} stocks (last = today)")
    print(f"  port_hybrid   : {len(port_hybrid)} dates x {len(port_hybrid.columns) if not port_hybrid.empty else 0} stocks (last = today)")

    _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s, nav_hybrid)

    return {
        'nav_baseline'           : nav_baseline,
        'nav_alpha'              : nav_alpha,
        'nav_mvo'                : nav_mvo,
        'nav_hybrid'             : nav_hybrid,
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
