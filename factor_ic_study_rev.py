#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
factor_ic_study.py
==================
Computes yearly Information Coefficients (IC) for the five desirable
alpha factors vs forward returns. Used to derive regime-conditioned
alpha weights for portfolio optimization.

Factors studied (raw z-scored cross-sectional characteristics):
  - Quality    : rate-conditioned quality composite (GQF/CQF blend)
  - Idio_Mom   : idiosyncratic momentum from v2_factor_residuals_quality
  - Value      : IC-weighted value composite
  - OU         : O-U mean reversion score
  - Mom_12M1   : 12-month price momentum (skip 1 month), not a model factor

Forward return horizons: 21d and 63d.

IC per date = Spearman rank correlation between factor scores and
              forward returns across the universe.

Yearly summary: mean IC, IC volatility, ICIR (mean/std) per factor.

Assumes all factor_model_step1 functions/constants live in kernel.
No imports needed.

Entry point
-----------
    ic_df, ic_annual = run_ic_study_alpha(Pxs_df, sectors_s,
                                           model_version='v2',
                                           rebal_freq=21)

Parameters
----------
    Pxs_df        : pd.DataFrame — price/macro panel
    sectors_s     : pd.Series   — ticker -> sector label
    model_version : 'v1' or 'v2' (determines which cached tables to load)
    rebal_freq    : int — rebalance frequency in trading days (default 21)

Returns
-------
    ic_df     : pd.DataFrame — dates x factors x horizons, daily IC values
                MultiIndex columns: (factor, horizon)
    ic_annual : pd.DataFrame — year x (factor, horizon, metric)
                metrics: IC_mean, IC_std, ICIR
"""

# ===============================================================================
# PARAMETERS
# ===============================================================================

ICS_HORIZONS     = [21, 63]      # forward return horizons in trading days
ICS_MOM_LONG     = 252           # 12M1 momentum lookback
ICS_MOM_SKIP     = 21            # 12M1 momentum skip period
ICS_MIN_STOCKS   = 50            # minimum stocks for a valid IC observation
ICS_REBAL_FREQ   = 21            # default rebalance frequency (trading days)

# Which residuals table feeds idio momentum per model version
ICS_MOM_RESID = {
    'v1': 'factor_residuals_vol',       # v1: momentum on vol residuals
    'v2': 'v2_factor_residuals_quality' # v2: momentum on quality residuals
}

# Which OU cache table per model version
ICS_OU_TBL = {
    'v1': 'ou_reversion_df',
    'v2': 'v2_ou_reversion_df',
}

# Which value scores table per model version
ICS_VALUE_TBL = {
    'v1': 'value_scores_df',
    'v2': 'v2_value_scores_df',   # falls back to v1 if not populated
}

# Weight bounds — applied after IC-based normalization to prevent
# extreme allocations from small-sample IC estimates.
# Same bounds for all factors: regime dynamic is Value vs others,
# symmetric caps prevent any single factor dominating.
ICS_WEIGHT_MIN = 0.10   # floor: no factor fully excluded
ICS_WEIGHT_MAX = 0.50   # cap: no factor fully dominates

# IC results cache table
ICS_IC_CACHE_TBL = 'factor_ic_cache'   # cached IC values (date x factor x horizon)

def get_universe(Pxs_df: pd.DataFrame, sectors_s: pd.Series,
                 extended_st_dt: pd.Timestamp) -> list:
    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ticker FROM income_data
        """)).fetchall()
    db_tickers  = {r[0].upper() for r in rows}
    etf_tickers = set(sectors_s.values)
    pre_dates   = Pxs_df.index[Pxs_df.index < extended_st_dt]

    universe = []
    for col in Pxs_df.columns:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        if col not in sectors_s.index:
            continue
        if len(pre_dates) >= BETA_WINDOW:
            col_data = Pxs_df.loc[pre_dates[-BETA_WINDOW:], col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            if int(col_data.notna().sum()) < BETA_WINDOW // 2:
                continue
        universe.append(col)

    print(f"  Universe: {len(universe)} stocks "
          f"(in DB + sector mapped + sufficient history)")
    return universe

def _ics_ensure_ic_cache_table():
    """Create factor_ic_cache table if it doesn't exist."""
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {ICS_IC_CACHE_TBL} (
                date           DATE        NOT NULL,
                factor         VARCHAR(20) NOT NULL,
                horizon        VARCHAR(10) NOT NULL,
                model_version  VARCHAR(5)  NOT NULL,
                ic_value       FLOAT,
                PRIMARY KEY (date, factor, horizon, model_version)
            )
        """))


def _ics_load_cached_ic(model_version):
    """Load all cached IC values for a model version. Returns dict {factor: DataFrame}."""
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT date, factor, horizon, ic_value
                FROM {ICS_IC_CACHE_TBL}
                WHERE model_version = :mv
                ORDER BY date
            """), conn, params={'mv': model_version})
        if df.empty:
            return {}, set()
        df['date'] = pd.to_datetime(df['date'])
        cached_dates = set(df['date'].unique())
        ic_results = {}
        for factor, grp in df.groupby('factor'):
            pivot = grp.pivot_table(index='date', columns='horizon',
                                    values='ic_value', aggfunc='last')
            ic_results[factor] = pivot
        return ic_results, cached_dates
    except Exception:
        return {}, set()


def _ics_save_ic_cache(ic_results_new, model_version):
    """Save new IC values to cache table. Never overwrites existing entries."""
    if not ic_results_new:
        return
    # Load already-cached dates to avoid overwriting
    _, cached_dates = _ics_load_cached_ic(model_version)
    rows = []
    for factor, ic_df in ic_results_new.items():
        for dt in ic_df.index:
            if pd.Timestamp(dt) in cached_dates:
                continue   # never overwrite existing IC values
            for hz in ic_df.columns:
                val = ic_df.loc[dt, hz]
                if not np.isnan(val):
                    rows.append({
                        'date': dt.date(), 'factor': factor,
                        'horizon': hz, 'model_version': model_version,
                        'ic_value': float(val)
                    })
    if not rows:
        print("  IC cache: no new dates to save (all already cached)")
        return
    df    = pd.DataFrame(rows)
    dates = list({r['date'] for r in rows})
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {ICS_IC_CACHE_TBL}
                (date, factor, horizon, model_version, ic_value)
            VALUES (:date, :factor, :horizon, :model_version, :ic_value)
            ON CONFLICT (date, factor, horizon, model_version) DO NOTHING
        """), rows)
    print(f"  Saved IC cache: {len(df)} rows ({len(dates)} dates)")


def compute_rolling_regime_weights(ic_results, ic_annual, Pxs_df,
                                    horizons=None, w1=63, w2=42, threshold=20,
                                    rebal_freq=21):
    """
    Derive point-in-time regime weights at each rebalance date using only
    IC observations strictly before that date — no lookahead bias.

    Weights are computed per rebalance date (not per year), so within-year
    IC changes feed into weights immediately at the next rebalance.

    For each rebalance date t with current regime r_t:
        raw_IC(factor) = Σ_{obs < t} similarity(r_t, r_obs) × IC_obs(factor)
                       / Σ_{obs < t} similarity(r_t, r_obs)

    similarity = triangular kernel: max(0, 1 - |r_t - r_obs|)

    Returns
    -------
    weights_by_date : dict {rebalance_date: pd.Series(factor -> weight)}
                      Point-in-time weights at each rebalance date.
    weights_by_year : dict {year: weights_df}
                      regime x factor DataFrame — for backward compatibility
                      with backtest scripts. Uses Jan 1 snapshot per year.
    regime_s        : pd.Series — daily regime score
    """
    # ── Build regime indicator ────────────────────────────────────────────────
    regime_s = build_rates_regime(Pxs_df, w1=w1, w2=w2, threshold=threshold)

    # ── Determine horizons ────────────────────────────────────────────────────
    all_horizons = set()
    for ic_df in ic_results.values():
        all_horizons.update(ic_df.columns.tolist())
    if horizons is None:
        horizons = sorted(all_horizons)
    else:
        horizons = [h for h in horizons if h in all_horizons]

    factors   = list(ic_results.keys())
    regimes   = [0.0, 0.5, 1.0]
    n_factors = len(factors)
    all_years = sorted(ic_annual.index.get_level_values('Year').unique())

    # ── Equal weight prior ────────────────────────────────────────────────────
    eq_w    = max(ICS_WEIGHT_MIN, min(ICS_WEIGHT_MAX, 1.0 / n_factors))
    equal_w = pd.Series({f: eq_w for f in factors})
    equal_w = equal_w / equal_w.sum()

    # ── Build flat IC observation list ────────────────────────────────────────
    # Each entry: (date, factor, ic_value, regime_at_date)
    def _get_regime(dt):
        past = regime_s[regime_s.index <= dt]
        return float(past.iloc[-1]) if not past.empty else 0.5

    ic_obs = []
    for fname in factors:
        ic_df = ic_results[fname]
        hz_cols = [h for h in horizons if h in ic_df.columns]
        if not hz_cols:
            continue
        ic_mean = ic_df[hz_cols].mean(axis=1).dropna()
        for dt, ic_val in ic_mean.items():
            ic_obs.append((pd.Timestamp(dt), fname, float(ic_val),
                           _get_regime(pd.Timestamp(dt))))

    ic_obs.sort(key=lambda x: x[0])

    # ── Helper: compute weights at a given date for a given target regime ─────
    def _weights_at(cutoff_dt, r_target):
        """Similarity-weighted IC from all observations before cutoff_dt."""
        prior = [(fn, ic, r_obs) for dt, fn, ic, r_obs in ic_obs
                 if dt < cutoff_dt]
        if not prior:
            return equal_w.copy()

        factor_scores = {}
        for fname in factors:
            num = den = 0.0
            for fn, ic_val, r_obs in prior:
                if fn != fname:
                    continue
                sim = max(0.0, 1.0 - abs(r_target - r_obs))
                if sim == 0.0:
                    continue
                num += sim * ic_val
                den += sim
            factor_scores[fname] = (num / den) if den > 0 else 0.0

        return _ics_bounded_normalize(
            pd.Series(factor_scores),
            w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX
        )

    # ── Compute per-rebalance-date weights ────────────────────────────────────
    # Rebalance dates: same cadence as IC study (rebal_freq trading days)
    all_rets   = Pxs_df.iloc[:, :5].pct_change().dropna(how='all')
    start_dt   = pd.Timestamp('2019-01-01')
    end_dt     = Pxs_df.index[-1]
    valid_idx  = all_rets.index[all_rets.index >= start_dt]
    rebal_dates = pd.DatetimeIndex(valid_idx[::rebal_freq])

    WEIGHTS_CACHE_TBL = 'factor_weights_by_date'

    def _ensure_weights_table():
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {WEIGHTS_CACHE_TBL} (
                    date           DATE        NOT NULL,
                    factor         VARCHAR(20) NOT NULL,
                    weight         FLOAT       NOT NULL,
                    model_version  VARCHAR(5)  NOT NULL,
                    PRIMARY KEY (date, factor, model_version)
                )
            """))

    def _load_weights_cache(mv):
        try:
            with ENGINE.connect() as conn:
                df = pd.read_sql(text(f"""
                    SELECT date, factor, weight FROM {WEIGHTS_CACHE_TBL}
                    WHERE model_version = :mv ORDER BY date
                """), conn, params={'mv': mv})
            if df.empty:
                return {}
            df['date'] = pd.to_datetime(df['date'])
            result = {}
            for dt, grp in df.groupby('date'):
                result[pd.Timestamp(dt)] = pd.Series(
                    grp.set_index('factor')['weight'])
            return result
        except Exception:
            return {}

    def _save_weights_cache(wbd, mv):
        rows = []
        for dt, w_s in wbd.items():
            for factor, weight in w_s.items():
                rows.append({'date': dt.date(), 'factor': factor,
                             'weight': float(weight), 'model_version': mv})
        if not rows:
            return
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {WEIGHTS_CACHE_TBL}
                    (date, factor, weight, model_version)
                VALUES (:date, :factor, :weight, :model_version)
                ON CONFLICT (date, factor, model_version) DO NOTHING
            """), rows)
        print(f"  weights_by_date: saved {len(wbd)} new dates to cache")

    _ensure_weights_table()
    mv = 'v2'   # model_version not directly accessible here — infer from ic_results
    cached_wbd = _load_weights_cache(mv)

    weights_by_date = {}
    new_wbd = {}
    for dt in rebal_dates:
        if dt in cached_wbd:
            weights_by_date[dt] = cached_wbd[dt]   # use cached — never recompute
        else:
            r_t = _get_regime(dt)
            w   = _weights_at(dt, r_t)
            weights_by_date[dt] = w
            new_wbd[dt] = w

    if new_wbd:
        _save_weights_cache(new_wbd, mv)
        print(f"  weights_by_date: {len(cached_wbd)} loaded, "
              f"{len(new_wbd)} newly computed")

    # ── Also build weights_by_year for backward compatibility ─────────────────
    # Uses Jan 1 snapshot for each year — same regime logic
    weights_by_year = {}
    min_yr = min(all_years)
    max_yr = max(all_years) + 1

    print(f"\n  Rolling point-in-time weights (per-date, regime-similarity kernel)")
    print(f"  Horizons: {horizons}  |  Rebal dates: {len(rebal_dates)}\n")
    print(f"  {'Year':<6}  {'N_obs':<8}  "
          + "  ".join(f"{f:>10}" for f in factors)
          + "  (Jan 1 snapshot, modal regime)")
    print("  " + "-" * (6 + 10 + 13 * n_factors + 25))

    for yr in range(min_yr, max_yr + 1):
        cutoff   = pd.Timestamp(f"{yr}-01-01")
        prior    = [x for x in ic_obs if x[0] < cutoff]
        n_obs    = len(prior)

        # Modal regime for this year
        yr_vals  = regime_s[regime_s.index.year == yr]
        modal_r  = float(yr_vals.mode().iloc[0]) if not yr_vals.empty else 0.5

        # Build full regime x factor DataFrame for this year
        rows = []
        for r_target in regimes:
            w_row = _weights_at(cutoff, r_target)
            row   = w_row.to_dict()
            row['regime'] = r_target
            rows.append(row)
        w_df = pd.DataFrame(rows).set_index('regime')
        weights_by_year[yr] = w_df

        # Print at modal regime
        w_show = w_df.loc[modal_r] if modal_r in w_df.index else w_df.iloc[1]
        print(f"  {yr:<6}  {n_obs:<8}  "
              + "  ".join(f"{w_show[f]:>10.4f}" for f in factors)
              + f"  [regime={modal_r}]")

    print(f"\n  weights_by_year built for years: {sorted(weights_by_year.keys())}")
    print(f"  weights_by_date built for dates: "
          f"{rebal_dates[0].date()} → {rebal_dates[-1].date()}")
    return weights_by_year, regime_s, weights_by_date


# ===============================================================================
# HELPERS
# ===============================================================================

def _ics_zscore(s):
    """Cross-sectional z-score, robust to NaNs."""
    mu, sd = s.mean(), s.std()
    return s * 0.0 if (sd == 0 or np.isnan(sd)) else (s - mu) / sd


def _ics_bounded_normalize(raw_series, w_min=ICS_WEIGHT_MIN,
                            w_max=ICS_WEIGHT_MAX, max_iter=10):
    """
    Normalize a pd.Series of non-negative weights to sum to 1,
    subject to per-element bounds [w_min, w_max].

    Algorithm: clip → renormalize, repeated until convergence.
    Guaranteed to converge for feasible bounds (w_min * n <= 1 <= w_max * n).
    """
    w = raw_series.clip(lower=0).copy()
    # If all zero, return equal weights
    if w.sum() == 0:
        return pd.Series(1.0 / len(w), index=w.index)
    for _ in range(max_iter):
        w = w / w.sum()                        # normalize
        w_clipped = w.clip(lower=w_min, upper=w_max)
        if (w_clipped - w).abs().max() < 1e-9:
            break
        w = w_clipped
    # Final normalize to ensure exact sum = 1
    w = w / w.sum()
    return w.round(4)


def _ics_spearman_ic(scores, fwd_rets):
    """
    Spearman rank correlation between scores and forward returns.
    Only uses stocks present in both series with non-NaN values.
    Returns float IC or np.nan if insufficient data.
    """
    common = scores.index.intersection(fwd_rets.index)
    s = scores.reindex(common).dropna()
    r = fwd_rets.reindex(s.index).dropna()
    s = s.reindex(r.index)
    if len(s) < ICS_MIN_STOCKS:
        return np.nan
    # Spearman = Pearson on ranks
    rs = s.rank()
    rr = r.rank()
    rs -= rs.mean(); rr -= rr.mean()
    denom = (rs.std() * rr.std())
    if denom < 1e-12:
        return np.nan
    return float((rs * rr).mean() / (rs.std() * rr.std()))


def _ics_load_table(table, conn=None):
    """Load a full DB table as DataFrame with date index."""
    try:
        if conn is not None:
            df = pd.read_sql(f"SELECT * FROM {table} ORDER BY date", conn)
        else:
            with ENGINE.connect() as c:
                df = pd.read_sql(f"SELECT * FROM {table} ORDER BY date", c)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()
    except Exception as e:
        warnings.warn(f"Could not load '{table}': {e}")
        return pd.DataFrame()


# ===============================================================================
# FACTOR SCORE LOADERS
# ===============================================================================

def _ics_load_quality(universe, calc_dates, Pxs_df, sectors_s):
    """Raw z-scored quality composite — delegates to load_quality_scores."""
    print("  Loading quality scores...")
    df = load_quality_scores(universe, calc_dates, Pxs_df, sectors_s)
    return df.apply(_ics_zscore, axis=1)


def _ics_load_idio_mom(universe, model_version):
    """
    Idiosyncratic momentum from cached residuals table.
    Cumulative residuals over [t-MOM_LONG, t-MOM_SKIP], z-scored.
    Returns wide DataFrame (dates x tickers).
    """
    print(f"  Loading idio momentum residuals ({ICS_MOM_RESID[model_version]})...")
    with ENGINE.connect() as conn:
        df = pd.read_sql(
            f"SELECT * FROM {ICS_MOM_RESID[model_version]} ORDER BY date",
            conn
        )
    df['date'] = pd.to_datetime(df['date'])
    if 'ticker' in df.columns and 'resid' in df.columns:
        resid = df.pivot_table(
            index='date', columns='ticker', values='resid', aggfunc='last'
        ).reindex(columns=universe)
    else:
        resid = df.set_index('date').reindex(columns=universe)

    print(f"  Residuals loaded: {resid.shape}")
    return resid


def _ics_compute_idio_mom_scores(resid_df, calc_dates):
    """
    Compute idio momentum scores on each calc_date from residual history.
    Returns wide DataFrame (calc_dates x tickers).
    """
    print("  Computing idio momentum scores...")
    results = {}
    all_resid_dates = resid_df.index

    for dt in calc_dates:
        past = all_resid_dates[all_resid_dates < dt]
        if len(past) < ICS_MOM_LONG + 1:
            continue
        window    = past[-ICS_MOM_LONG:-ICS_MOM_SKIP]
        if len(window) < ICS_MOM_LONG - ICS_MOM_SKIP - 10:
            continue
        cum_resid = resid_df.loc[window].sum(axis=0)
        valid     = cum_resid.dropna()
        if len(valid) < ICS_MIN_STOCKS:
            continue
        results[dt] = _ics_zscore(valid)

    out = pd.DataFrame(results).T
    out.index.name = 'date'
    print(f"  Idio momentum scores: {len(out)} dates")
    return out.reindex(columns=resid_df.columns)


def _ics_load_value(universe, calc_dates, sectors_s, model_version):
    """Raw z-scored value composite from cache, with v1 fallback."""
    print("  Loading value scores...")
    # Try v2 table first if model_version='v2'
    tbl = ICS_VALUE_TBL[model_version]
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        if df.empty:
            raise ValueError("empty")
        df['date']   = pd.to_datetime(df['date'])
        df['ticker'] = df['ticker'].apply(clean_ticker)
        df['score']  = df['score'].astype(float)
        val = df.pivot_table(index='date', columns='ticker',
                             values='score', aggfunc='last')
        val = val.reindex(columns=universe)
    except Exception:
        if model_version == 'v2':
            warnings.warn(f"  {tbl} not found — falling back to v1 value cache")
        val = load_value_scores(universe, calc_dates, sectors_s)

    # Forward-fill to calc_dates then z-score
    all_dates = calc_dates.union(val.index).sort_values()
    val_ff    = val.reindex(all_dates).ffill().reindex(calc_dates)
    print(f"  Value scores: {val_ff.notna().any(axis=1).sum()} dates with data")
    return val_ff.apply(_ics_zscore, axis=1)


def _ics_load_ou(universe, model_version):
    """Raw z-scored O-U scores from cache."""
    print(f"  Loading O-U scores ({ICS_OU_TBL[model_version]})...")
    tbl = ICS_OU_TBL[model_version]
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(
                f"SELECT date, ticker, ou_score FROM {tbl} ORDER BY date",
                conn
            )
        df['date']     = pd.to_datetime(df['date'])
        df['ou_score'] = df['ou_score'].astype(float)
        ou = df.pivot_table(index='date', columns='ticker',
                            values='ou_score', aggfunc='last')
        ou = ou.reindex(columns=universe)
        print(f"  O-U scores: {ou.shape}")
        return ou.apply(_ics_zscore, axis=1)
    except Exception as e:
        warnings.warn(f"  O-U load failed ({e}) — skipping")
        return pd.DataFrame()


def _ics_compute_mom_12m1(universe, calc_dates, Pxs_df):
    """
    12M1 price momentum: z-scored cross-sectional return over
    [t-252, t-21] trading days. Computed on each calc_date.
    Returns wide DataFrame (calc_dates x tickers).
    """
    print("  Computing 12M1 momentum scores...")
    all_px_dates = Pxs_df.index
    results      = {}

    valid_universe = [t for t in universe if t in Pxs_df.columns]

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

        results[dt] = _ics_zscore(mom)

    out = pd.DataFrame(results).T
    out.index.name = 'date'
    print(f"  12M1 momentum scores: {len(out)} dates")
    return out.reindex(columns=valid_universe)


# ===============================================================================
# FORWARD RETURN COMPUTATION
# ===============================================================================

def _ics_forward_returns(universe, calc_dates, Pxs_df, horizon):
    """
    Compute h-day forward returns for each stock on each calc_date.
    r_i_{t+h} = Pxs[t+h] / Pxs[t] - 1
    Returns wide DataFrame (calc_dates x tickers).
    """
    all_px_dates = Pxs_df.index
    valid        = [t for t in universe if t in Pxs_df.columns]
    results      = {}

    for dt in calc_dates:
        future = all_px_dates[all_px_dates > dt]
        if len(future) < horizon:
            continue
        dt_end   = future[horizon - 1]
        px_start = Pxs_df.loc[dt,     valid]
        px_end   = Pxs_df.loc[dt_end, valid]
        fwd      = (px_end / px_start.replace(0, np.nan) - 1).dropna()
        if len(fwd) < ICS_MIN_STOCKS:
            continue
        results[dt] = fwd

    out = pd.DataFrame(results).T
    out.index.name = 'date'
    return out.reindex(columns=valid)


# ===============================================================================
# IC COMPUTATION
# ===============================================================================

def _ics_compute_ic_series(factor_scores, fwd_rets_dict):
    """
    For each date in factor_scores.index, compute IC vs each horizon.
    Returns DataFrame (dates x horizons).
    """
    results = {}
    for dt in factor_scores.index:
        if dt not in factor_scores.index:
            continue
        scores = factor_scores.loc[dt].dropna()
        if len(scores) < ICS_MIN_STOCKS:
            continue
        row = {}
        for h, fwd_df in fwd_rets_dict.items():
            if dt not in fwd_df.index:
                continue
            fwd = fwd_df.loc[dt].dropna()
            row[h] = _ics_spearman_ic(scores, fwd)
        if row:
            results[dt] = row

    ic_df = pd.DataFrame(results).T
    ic_df.index.name = 'date'
    return ic_df


# ===============================================================================
# YEARLY SUMMARY
# ===============================================================================

def _ics_annual_summary(ic_results):
    """
    ic_results : dict {factor_name: DataFrame(dates x horizons)}
    Returns MultiIndex DataFrame: year x (factor, horizon, metric)
    metrics: IC_mean, IC_std, ICIR
    """
    rows = []
    for fname, ic_df in ic_results.items():
        if ic_df.empty:
            continue
        for h in ic_df.columns:
            s = ic_df[h].dropna()
            if s.empty:
                continue
            for yr, grp in s.groupby(s.index.year):
                ic_mean = grp.mean()
                ic_std  = grp.std()
                icir    = ic_mean / ic_std if ic_std > 0 else np.nan
                rows.append({
                    'Year'    : yr,
                    'Factor'  : fname,
                    'Horizon' : h,
                    'IC_mean' : round(ic_mean, 4),
                    'IC_std'  : round(ic_std,  4),
                    'ICIR'    : round(icir,    3),
                    'N_dates' : len(grp),
                })

    return pd.DataFrame(rows).set_index(['Year', 'Factor', 'Horizon'])


def _ics_print_summary(ic_annual):
    """Print yearly IC summary grouped by factor."""
    factors = ic_annual.index.get_level_values('Factor').unique()

    print("\n" + "=" * 80)
    print("  ALPHA FACTOR IC STUDY — Yearly Summary")
    print("=" * 80)

    for fname in factors:
        print(f"\n  ── {fname} ──")
        print(f"  {'Year':<6}  {'Horizon':<8}  {'IC_mean':>8}  "
              f"{'IC_std':>8}  {'ICIR':>8}  {'N':>5}")
        print("  " + "-" * 52)

        sub = ic_annual.loc[ic_annual.index.get_level_values('Factor') == fname]
        for (yr, fn, hz), row in sub.iterrows():
            ic_str   = f"{row['IC_mean']:>+8.4f}"
            std_str  = f"{row['IC_std']:>8.4f}"
            icir_str = f"{row['ICIR']:>+8.3f}"
            n_str    = f"{row['N_dates']:>5}"
            flag     = "  *" if abs(row['ICIR']) > 0.5 else ""
            print(f"  {yr:<6}  {hz:<8}  {ic_str}  {std_str}  {icir_str}  {n_str}{flag}")

    print("\n  * ICIR > 0.5 flagged as potentially significant")
    print("=" * 80 + "\n")


def _ics_plot(ic_results, ic_annual):
    """
    Two panels per factor:
      Left:  cumulative IC over time (both horizons)
      Right: yearly ICIR bar chart (both horizons)
    """
    factors = list(ic_results.keys())
    n       = len(factors)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    fig.suptitle("Alpha Factor IC Study", fontsize=13, fontweight='bold', y=1.01)
    fig.patch.set_facecolor('#FAFAF9')

    COLORS = {'21d': '#1D9E75', '63d': '#378ADD'}

    for i, fname in enumerate(factors):
        ic_df = ic_results[fname]
        ax_cum  = axes[i, 0]
        ax_bar  = axes[i, 1]
        ax_cum.set_facecolor('#FAFAF9')
        ax_bar.set_facecolor('#FAFAF9')

        # Left: cumulative IC
        for h in ic_df.columns:
            s   = ic_df[h].dropna()
            cum = s.cumsum()
            ax_cum.plot(cum.index.to_numpy(), cum.values,
                        label=f'{h}d', color=COLORS.get(f'{h}d', '#888780'),
                        linewidth=1.2)
        ax_cum.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
        ax_cum.set_title(f"{fname} — Cumulative IC", fontsize=10, fontweight='bold')
        ax_cum.set_ylabel("Cumulative IC", fontsize=9)
        ax_cum.legend(fontsize=8)
        ax_cum.grid(color='#D3D1C7', linewidth=0.5)
        ax_cum.spines['top'].set_visible(False)
        ax_cum.spines['right'].set_visible(False)

        # Right: yearly ICIR bars
        sub = ic_annual.loc[
            ic_annual.index.get_level_values('Factor') == fname
        ].reset_index()
        if sub.empty:
            continue

        years    = sorted(sub['Year'].unique())
        horizons = sorted(sub['Horizon'].unique())
        x        = np.arange(len(years))
        width    = 0.35

        for j, hz in enumerate(horizons):
            hz_sub  = sub[sub['Horizon'] == hz].set_index('Year')
            icirs   = [hz_sub.loc[y, 'ICIR'] if y in hz_sub.index else 0.0
                       for y in years]
            colors  = ['#1D9E75' if v >= 0 else '#D85A30' for v in icirs]
            offset  = (j - 0.5) * width
            bars    = ax_bar.bar(x + offset, icirs, width,
                                 color=colors, alpha=0.8, label=hz)

        ax_bar.axhline(0, color='#888780', linewidth=0.8)
        ax_bar.axhline( 0.5, color='#1D9E75', linewidth=0.6,
                        linestyle=':', alpha=0.6)
        ax_bar.axhline(-0.5, color='#D85A30', linewidth=0.6,
                        linestyle=':', alpha=0.6)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(years, fontsize=8, rotation=45)
        ax_bar.set_title(f"{fname} — Yearly ICIR", fontsize=10, fontweight='bold')
        ax_bar.set_ylabel("ICIR", fontsize=9)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(axis='y', color='#D3D1C7', linewidth=0.5)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


# ===============================================================================
# ENTRY POINT
# ===============================================================================

def run_ic_study_alpha(Pxs_df, sectors_s,
                       model_version='v2',
                       rebal_freq=ICS_REBAL_FREQ,
                       force_recompute=False):
    """
    Run IC study for all five alpha factors — incremental by default.

    Daily fast path: loads cached IC values from DB, only computes
    new dates not yet cached. On first run computes all dates (slow).
    Subsequent runs are fast — typically 0-1 new dates.

    Parameters
    ----------
    Pxs_df        : pd.DataFrame — price/macro panel
    sectors_s     : pd.Series   — ticker -> sector label
    model_version : 'v1' or 'v2'
    rebal_freq    : int — rebalance frequency in trading days (default 21)
    force_recompute : bool — recompute all dates from scratch

    Returns
    -------
    ic_results : dict {factor_name: DataFrame(dates x horizons)}
    ic_annual  : pd.DataFrame — yearly IC summary (MultiIndex)
    """
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    print(f"\nAlpha Factor IC Study [{model_version}] — starting...\n")

    # ── Universe and calc dates ───────────────────────────────────────────────
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    print(f"  Universe: {len(universe)} stocks")

    all_rets  = Pxs_df[universe].pct_change().dropna(how='all')
    start_dt  = pd.Timestamp('2019-01-01')
    end_dt    = Pxs_df.index[-max(ICS_HORIZONS) - 5]
    all_valid = all_rets.index[
        (all_rets.index >= start_dt) &
        (all_rets.index <= end_dt) &
        (all_rets.notna().sum(axis=1) >= ICS_MIN_STOCKS)
    ]
    calc_dates = pd.DatetimeIndex(all_valid[::rebal_freq])
    print(f"  Calc dates: {len(calc_dates)} "
          f"({calc_dates[0].date()} → {calc_dates[-1].date()})")

    # ── Load IC cache ─────────────────────────────────────────────────────────
    _ics_ensure_ic_cache_table()
    if force_recompute:
        cached_ic, cached_dates = {}, set()
        print(f"  Force recompute: computing all {len(calc_dates)} dates")
    else:
        cached_ic, cached_dates = _ics_load_cached_ic(model_version)
        missing_dates = pd.DatetimeIndex(
            [d for d in calc_dates if d not in cached_dates]
        )
        print(f"  IC cache: {len(cached_dates)} dates cached, "
              f"{len(missing_dates)} new dates to compute")

    missing_dates = pd.DatetimeIndex(
        [d for d in calc_dates if d not in cached_dates]
    ) if not force_recompute else calc_dates

    # ── Compute IC for missing dates only ─────────────────────────────────────
    if len(missing_dates) > 0:
        print(f"\n[1/6] Computing forward returns for {len(missing_dates)} new dates...")
        fwd_rets = {}
        for h in ICS_HORIZONS:
            fwd_rets[h] = _ics_forward_returns(universe, missing_dates, Pxs_df, h)
            print(f"  {h}d forward returns: {fwd_rets[h].notna().any(axis=1).sum()} dates")

        print("\n[2/6] Loading quality scores...")
        quality_scores = _ics_load_quality(universe, missing_dates, Pxs_df, sectors_s)

        print("\n[3/6] Loading idio momentum...")
        resid_df    = _ics_load_idio_mom(universe, model_version)
        idio_scores = _ics_compute_idio_mom_scores(resid_df, missing_dates)

        print("\n[4/6] Loading value scores...")
        value_scores = _ics_load_value(universe, missing_dates, sectors_s, model_version)

        print("\n[5/6] Loading O-U scores...")
        ou_scores = _ics_load_ou(universe, model_version)
        if not ou_scores.empty:
            all_dates = missing_dates.union(ou_scores.index).sort_values()
            ou_scores = ou_scores.reindex(all_dates).ffill().reindex(missing_dates)

        print("\n[6/6] Computing 12M1 momentum...")
        mom_scores = _ics_compute_mom_12m1(universe, missing_dates, Pxs_df)

        factor_map = {
            'Quality'  : quality_scores,
            'Idio_Mom' : idio_scores,
            'Value'    : value_scores,
            'OU'       : ou_scores,
            'Mom_12M1' : mom_scores,
        }

        print("\nComputing IC series for new dates...")
        new_ic_results = {}
        for fname, scores_df in factor_map.items():
            if scores_df is None or scores_df.empty:
                warnings.warn(f"  {fname}: no scores available — skipping")
                continue
            print(f"  Computing IC: {fname}...")
            ic_df = _ics_compute_ic_series(scores_df, fwd_rets)
            ic_df.columns = [f'{h}d' for h in ic_df.columns]
            new_ic_results[fname] = ic_df
            for col in ic_df.columns:
                s = ic_df[col].dropna()
                if not s.empty:
                    print(f"    {col}: mean IC={s.mean():+.4f}  "
                          f"ICIR={s.mean()/s.std():+.3f}  N={len(s)}")

        # Save new IC values to cache
        _ics_save_ic_cache(new_ic_results, model_version)

        # Merge with cached results
        all_ic_results = {}
        all_factors = set(cached_ic.keys()) | set(new_ic_results.keys())
        for fname in all_factors:
            parts = []
            if fname in cached_ic and not cached_ic[fname].empty:
                parts.append(cached_ic[fname])
            if fname in new_ic_results and not new_ic_results[fname].empty:
                parts.append(new_ic_results[fname])
            if parts:
                merged = pd.concat(parts).sort_index()
                merged = merged[~merged.index.duplicated(keep='last')]
                all_ic_results[fname] = merged
    else:
        # All dates cached — just use cached results
        print("  All dates cached — loading from DB only")
        all_ic_results = cached_ic

    # Filter to calc_dates
    ic_results = {}
    for fname, ic_df in all_ic_results.items():
        filtered = ic_df.reindex(calc_dates).dropna(how='all')
        if not filtered.empty:
            ic_results[fname] = filtered

    # ── Yearly summary ────────────────────────────────────────────────────────
    print("\nBuilding yearly summary...")
    ic_annual = _ics_annual_summary(ic_results)
    _ics_print_summary(ic_annual)
    _ics_plot(ic_results, ic_annual)

    return ic_results, ic_annual


# ===============================================================================
# REGIME INDICATOR + WEIGHT DERIVATION
# ===============================================================================

def build_rates_regime(Pxs_df, w1=63, w2=42, threshold=30):
    """
    Build the rates regime indicator from the 10Y yield.

    Construction:
        s  = round(100 * (USGG10YR - USGG10YR.rolling(w1).mean()), 2)  [bps vs trend]
        s  = s[s.index >= 2018-01-01]
        s  = (((s.rolling(w2).mean().dropna() // 30) + 1) / 2).clip(0, 1)

    Output values:
        0.0  — easy/easing regime     (smoothed deviation < -30 bps)
        0.5  — neutral/balanced       (-30 to +30 bps)
        1.0  — tight/hiking regime    (smoothed deviation > +30 bps)

    Parameters
    ----------
    Pxs_df : pd.DataFrame — must contain 'USGG10YR' column
    w1     : int — rolling window for trend (default 63 — ~3 months)
    w2        : int — smoothing window (default 42 — ~2 months)
    threshold : int — bps threshold for regime discretization (default 30)

    Returns
    -------
    regime_s : pd.Series  — daily regime score (0.0 / 0.5 / 1.0), index = dates
    """
    if 'USGG10YR' not in Pxs_df.columns:
        raise ValueError("'USGG10YR' column not found in Pxs_df.")

    raw    = Pxs_df['USGG10YR'].dropna()
    s      = (100 * (raw - raw.rolling(w1).mean())).round(2).dropna()
    s      = s[s.index >= pd.Timestamp('2018-01-01')]
    regime = (((s.rolling(w2).mean().dropna() // threshold) + 1) / 2).clip(0, 1)

    print(f"  Rates regime built: {len(regime)} dates  "
          f"({regime.index[0].date()} → {regime.index[-1].date()})")
    print(f"  Distribution:\n{regime.value_counts().sort_index().to_string()}")
    return regime


def compute_regime_weights(ic_results, ic_annual, Pxs_df,
                            horizons=None, w1=63, w2=42, threshold=30):
    """
    Derive IC-weighted alpha factor weights for each regime.

    For each regime r in {0.0, 0.5, 1.0}:
        1. Identify years where the regime score at year-start equals r
        2. Average the IC_mean across those years for each factor
        3. Zero-out negative ICs (a negative IC factor is not a signal to use)
        4. Normalise to sum to 1 — these are the alpha weights for regime r

    Weights are derived separately per horizon, then averaged across horizons
    (or you can pass horizons=['21d'] or ['63d'] to use one only).

    Parameters
    ----------
    ic_results   : dict {factor: DataFrame(dates x horizons)} from run_ic_study_alpha
    ic_annual    : pd.DataFrame — yearly IC summary from run_ic_study_alpha
    Pxs_df       : pd.DataFrame — price panel (needs 'USGG10YR' column)
    horizons     : list of horizon strings to average over, e.g. ['21d','63d']
                   None = use all available
    w1, w2       : int — regime indicator construction parameters (default 252/252)

    Returns
    -------
    weights_df   : pd.DataFrame — regime (0.0/0.5/1.0) x factor, summing to 1
    regime_s     : pd.Series   — daily regime score for live use
    regime_by_yr : pd.Series   — year -> regime score (for diagnostics)
    """
    # ── Build regime indicator ────────────────────────────────────────────────
    regime_s = build_rates_regime(Pxs_df, w1=w1, w2=w2, threshold=threshold)

    # ── Map each year to its regime (use modal value across the year) ──────────
    # Modal (most frequent) regime value over the full year correctly places
    # transition years like 2022 — which crossed into tight mid-year and stayed
    # there — rather than using the Jan 1 snapshot which may pre-date the move.
    regime_by_yr = {}
    for yr in range(ic_annual.index.get_level_values('Year').min(),
                    ic_annual.index.get_level_values('Year').max() + 1):
        yr_mask = (regime_s.index.year == yr)
        yr_vals = regime_s[yr_mask]
        if yr_vals.empty:
            continue
        # Modal value: most frequently occurring regime during the year
        regime_by_yr[yr] = float(yr_vals.mode().iloc[0])

    regime_by_yr = pd.Series(regime_by_yr)
    print(f"\n  Year-to-regime mapping:")
    print(f"  {'Year':<6}  {'Regime':>7}  {'Label'}")
    print("  " + "-" * 36)
    labels = {0.0: 'easy/easing', 0.5: 'neutral', 1.0: 'tight/hiking'}
    for yr, r in regime_by_yr.items():
        ic_yr = ic_annual.xs(yr, level='Year') if yr in                 ic_annual.index.get_level_values('Year') else None
        print(f"  {yr:<6}  {r:>7.1f}  {labels.get(r, '?')}")

    # ── Determine horizons to use ─────────────────────────────────────────────
    all_horizons = set()
    for ic_df in ic_results.values():
        all_horizons.update(ic_df.columns.tolist())
    if horizons is None:
        horizons = sorted(all_horizons)
    else:
        horizons = [h for h in horizons if h in all_horizons]
    print(f"\n  Averaging weights across horizons: {horizons}")

    # ── Compute per-regime IC means ───────────────────────────────────────────
    factors  = list(ic_results.keys())
    regimes  = [0.0, 0.5, 1.0]
    rows     = []

    for r in regimes:
        years_in_regime = [yr for yr, rv in regime_by_yr.items() if rv == r]
        if not years_in_regime:
            empty = {f: 0.0 for f in factors}
            empty['regime'] = r
            rows.append(empty)
            continue

        # For each factor, average IC_mean across years in this regime
        # and across requested horizons
        factor_scores = {}
        for fname in factors:
            ic_vals = []
            for yr in years_in_regime:
                for hz in horizons:
                    try:
                        val = ic_annual.loc[(yr, fname, hz), 'IC_mean']
                        ic_vals.append(float(val))
                    except KeyError:
                        pass
            factor_scores[fname] = np.mean(ic_vals) if ic_vals else 0.0

        factor_scores['regime'] = r
        rows.append(factor_scores)

    raw_df = pd.DataFrame(rows).set_index('regime')

    # ── Zero negatives, apply bounds, normalise ──────────────────────────────
    weights_df = raw_df.copy()
    for r in regimes:
        weights_df.loc[r] = _ics_bounded_normalize(
            raw_df.loc[r], w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX
        )

    # ── Print results ─────────────────────────────────────────────────────────
    _ics_print_weights(weights_df, raw_df, regime_by_yr)
    _ics_plot_weights(weights_df, raw_df)

    return weights_df, regime_s, regime_by_yr


def _ics_print_weights(weights_df, raw_df, regime_by_yr):
    """Print regime weight table with raw IC averages for transparency."""
    labels  = {0.0: 'easy/easing (0.0)', 0.5: 'neutral (0.5)',
               1.0: 'tight/hiking (1.0)'}
    factors = weights_df.columns.tolist()

    print("\n" + "=" * 80)
    print("  REGIME-CONDITIONED ALPHA WEIGHTS")
    print("=" * 80)

    print(f"\n  Raw IC averages per regime (before zeroing negatives):")
    print(f"  {'Regime':<22}" + "".join(f"  {f:>12}" for f in factors))
    print("  " + "-" * (22 + 14 * len(factors)))
    for r in weights_df.index:
        row_str = f"  {labels[r]:<22}"
        for f in factors:
            v = raw_df.loc[r, f]
            row_str += f"  {v:>+12.4f}"
        print(row_str)

    print(f"\n  Normalised weights (negative IC → 0, sum to 1 per regime):")
    print(f"  {'Regime':<22}" + "".join(f"  {f:>12}" for f in factors))
    print("  " + "-" * (22 + 14 * len(factors)))
    for r in weights_df.index:
        row_str = f"  {labels[r]:<22}"
        for f in factors:
            v = weights_df.loc[r, f]
            row_str += f"  {v:>12.4f}"
        print(row_str)

    # Years per regime
    print(f"\n  Years per regime:")
    for r in weights_df.index:
        yrs = sorted([yr for yr, rv in regime_by_yr.items() if rv == r])
        print(f"  {labels[r]:<22}  {yrs}")

    print("=" * 80 + "\n")


def _ics_plot_weights(weights_df, raw_df):
    """
    Stacked bar chart of normalised weights per regime,
    plus raw IC averages as a reference line chart.
    """
    factors  = weights_df.columns.tolist()
    regimes  = weights_df.index.tolist()
    labels   = {0.0: 'Easy\n(0.0)', 0.5: 'Neutral\n(0.5)',
                1.0: 'Tight\n(1.0)'}
    colors   = {
        'Quality'  : '#7F77DD',
        'Idio_Mom' : '#1D9E75',
        'Value'    : '#D85A30',
        'OU'       : '#888780',
        'Mom_12M1' : '#378ADD',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#FAFAF9')

    # ── Left: stacked weight bars ─────────────────────────────────────────────
    ax  = axes[0]
    ax.set_facecolor('#FAFAF9')
    x   = np.arange(len(regimes))
    bot = np.zeros(len(regimes))

    for f in factors:
        vals = weights_df[f].values
        c    = colors.get(f, '#B4B2A9')
        ax.bar(x, vals, bottom=bot, label=f, color=c, width=0.5)
        # Label inside bar if large enough
        for i, (v, b) in enumerate(zip(vals, bot)):
            if v > 0.05:
                ax.text(x[i], b + v/2, f'{v:.2f}',
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='500')
        bot += vals

    ax.set_xticks(x)
    ax.set_xticklabels([labels[r] for r in regimes], fontsize=10)
    ax.set_ylabel("Weight", fontsize=10, color='#5F5E5A')
    ax.set_title("Normalised alpha weights per regime",
                 fontsize=11, fontweight='500', color='#2C2C2A')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85,
              edgecolor='#D3D1C7')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D3D1C7')
    ax.spines['bottom'].set_color('#D3D1C7')
    ax.grid(axis='y', color='#D3D1C7', linewidth=0.5)

    # ── Right: raw IC averages per regime ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#FAFAF9')
    w   = 0.15
    for i, f in enumerate(factors):
        vals = raw_df[f].values
        off  = (i - len(factors)/2 + 0.5) * w
        bc   = [colors.get(f, '#B4B2A9') if v >= 0 else '#F0997B'
                for v in vals]
        ax2.bar(x + off, vals, width=w, color=bc, label=f, alpha=0.85)

    ax2.axhline(0, color='#888780', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[r] for r in regimes], fontsize=10)
    ax2.set_ylabel("Mean IC", fontsize=10, color='#5F5E5A')
    ax2.set_title("Raw IC averages per regime\n(before zeroing negatives)",
                  fontsize=11, fontweight='500', color='#2C2C2A')
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.85,
               edgecolor='#D3D1C7')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#D3D1C7')
    ax2.spines['bottom'].set_color('#D3D1C7')
    ax2.grid(axis='y', color='#D3D1C7', linewidth=0.5)

    plt.tight_layout()
    plt.show()
    return fig


# ===============================================================================
# COMPOSITE ALPHA IC STUDY
# ===============================================================================

def run_composite_ic_study(ic_results, weights_df, regime_s, Pxs_df, sectors_s,
                            exclude_factors=None, model_version='v2',
                            rebal_freq=ICS_REBAL_FREQ,
                            w1=63, w2=42, threshold=30):
    """
    Compute IC of the regime-conditioned composite alpha signal vs forward returns.

    For each rebalance date t:
        regime_t    = regime_s[t]                           (0.0 / 0.5 / 1.0)
        w_t         = weights_df.loc[regime_t]              (factor weights)
        alpha_i_t   = sum_k  w_t[k] * factor_score_i_t[k]  (composite score)
        IC_t        = Spearman(alpha_i_t, r_i_{t+h})        per horizon h

    Parameters
    ----------
    ic_results      : dict {factor: DataFrame(dates x horizons)} from run_ic_study_alpha
    weights_df      : pd.DataFrame — regime x factor weights from compute_regime_weights
    regime_s        : pd.Series   — daily regime score from build_rates_regime
    Pxs_df          : pd.DataFrame — price panel
    sectors_s       : pd.Series   — ticker -> sector label
    exclude_factors : list or None — factors to exclude (e.g. ['OU'])
    model_version   : 'v1' or 'v2'
    rebal_freq      : int — rebalance frequency in trading days

    Returns
    -------
    composite_ic_df : pd.DataFrame — dates x horizons, composite IC per date
    composite_annual: pd.DataFrame — yearly IC summary for composite
    """
    exclude_factors = exclude_factors or []

    # Rebuild regime indicator with specified parameters
    regime_s = build_rates_regime(Pxs_df, w1=w1, w2=w2, threshold=threshold)

    # Active factors (those in weights_df excluding requested)
    active_factors = [f for f in weights_df.columns if f not in exclude_factors]
    print(f"\nComposite IC Study — active factors: {active_factors}")
    print(f"  Regime params: w1={w1}, w2={w2}, threshold={threshold}\n")

    # Re-normalise weights excluding dropped factors, respecting bounds
    w_active = weights_df[active_factors].copy()
    for r in w_active.index:
        w_active.loc[r] = _ics_bounded_normalize(
            w_active.loc[r], w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX
        )

    print("  Re-normalised weights (OU excluded):")
    labels = {0.0: 'Easy  (0.0)', 0.5: 'Neutral (0.5)', 1.0: 'Tight (1.0)'}
    print(f"  {'Regime':<16}" + "".join(f"  {f:>12}" for f in active_factors))
    print("  " + "-" * (16 + 14 * len(active_factors)))
    for r in w_active.index:
        row_str = f"  {labels[r]:<16}"
        for f in active_factors:
            row_str += f"  {w_active.loc[r,f]:>12.4f}"
        print(row_str)
    print()

    # ── Universe and calc dates ───────────────────────────────────────────────
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)

    all_rets  = Pxs_df[universe].pct_change().dropna(how='all')
    start_dt  = pd.Timestamp('2019-01-01')
    end_dt    = Pxs_df.index[-max(ICS_HORIZONS) - 5]
    all_valid = all_rets.index[
        (all_rets.index >= start_dt) &
        (all_rets.index <= end_dt) &
        (all_rets.notna().sum(axis=1) >= ICS_MIN_STOCKS)
    ]
    calc_dates = pd.DatetimeIndex(all_valid[::rebal_freq])
    print(f"  Calc dates: {len(calc_dates)} "
          f"({calc_dates[0].date()} → {calc_dates[-1].date()})")

    # ── Forward returns ───────────────────────────────────────────────────────
    fwd_rets = {}
    for h in ICS_HORIZONS:
        fwd_rets[h] = _ics_forward_returns(universe, calc_dates, Pxs_df, h)

    # ── Align all factor score DataFrames to calc_dates ───────────────────────
    factor_scores_aligned = {}
    for fname in active_factors:
        if fname not in ic_results:
            warnings.warn(f"  {fname} not in ic_results — skipping")
            continue
        raw_ic_df = ic_results[fname]
        # ic_results stores IC series, not factor scores — we need the raw scores
        # So we rebuild them here from the cached sources
        factor_scores_aligned[fname] = None  # placeholder — filled below

    # Rebuild raw factor scores (same logic as run_ic_study_alpha)
    print("\n  Rebuilding factor scores for composite...")

    score_dfs = {}

    if 'Quality' in active_factors:
        print("  Quality...")
        score_dfs['Quality'] = _ics_load_quality(
            universe, calc_dates, Pxs_df, sectors_s
        )

    if 'Idio_Mom' in active_factors:
        print("  Idio_Mom...")
        resid_df = _ics_load_idio_mom(universe, model_version)
        score_dfs['Idio_Mom'] = _ics_compute_idio_mom_scores(resid_df, calc_dates)

    if 'Value' in active_factors:
        print("  Value...")
        score_dfs['Value'] = _ics_load_value(
            universe, calc_dates, sectors_s, model_version
        )

    if 'Mom_12M1' in active_factors:
        print("  Mom_12M1...")
        score_dfs['Mom_12M1'] = _ics_compute_mom_12m1(universe, calc_dates, Pxs_df)

    if 'OU' in active_factors:
        print("  OU...")
        ou = _ics_load_ou(universe, model_version)
        if not ou.empty:
            all_d = calc_dates.union(ou.index).sort_values()
            score_dfs['OU'] = ou.reindex(all_d).ffill().reindex(calc_dates)

    # ── Build composite score per date ────────────────────────────────────────
    print("\n  Building composite scores...")
    composite_scores = {}

    for dt in calc_dates:
        # Get regime on this date (forward-fill if date not in regime_s)
        reg_candidates = regime_s[regime_s.index <= dt]
        if reg_candidates.empty:
            continue
        r = float(reg_candidates.iloc[-1])

        # Get weights for this regime
        if r not in w_active.index:
            # Snap to nearest regime
            r = min(w_active.index, key=lambda x: abs(x - r))
        w_t = w_active.loc[r]

        # Build weighted composite score for each stock
        parts = []
        for fname in active_factors:
            if fname not in score_dfs or score_dfs[fname] is None:
                continue
            sdf = score_dfs[fname]
            if dt not in sdf.index:
                continue
            col = sdf.loc[dt].reindex(universe)
            if col.isna().all():
                continue
            parts.append(col.fillna(0) * w_t[fname])

        if not parts:
            continue

        composite = pd.concat(parts, axis=1).sum(axis=1)
        composite = _ics_zscore(composite.replace(0, np.nan).dropna())
        if len(composite) >= ICS_MIN_STOCKS:
            composite_scores[dt] = composite

    print(f"  Composite scores built: {len(composite_scores)} dates")

    # ── Compute composite IC series ───────────────────────────────────────────
    print("  Computing composite IC...")
    composite_score_df = pd.DataFrame(composite_scores).T
    composite_score_df.index.name = 'date'

    composite_ic_rows = {}
    for dt in composite_score_df.index:
        scores = composite_score_df.loc[dt].dropna()
        if len(scores) < ICS_MIN_STOCKS:
            continue
        row = {}
        for h, fwd_df in fwd_rets.items():
            if dt not in fwd_df.index:
                continue
            fwd = fwd_df.loc[dt].dropna()
            row[f'{h}d'] = _ics_spearman_ic(scores, fwd)
        if row:
            composite_ic_rows[dt] = row

    composite_ic_df            = pd.DataFrame(composite_ic_rows).T
    composite_ic_df.index.name = 'date'

    # ── Yearly summary ────────────────────────────────────────────────────────
    print("\n  Composite IC yearly summary:")
    print(f"  {'Year':<6}  {'Horizon':<8}  {'IC_mean':>8}  "
          f"{'IC_std':>8}  {'ICIR':>8}  {'N':>5}")
    print("  " + "-" * 52)

    rows = []
    for h in composite_ic_df.columns:
        s = composite_ic_df[h].dropna()
        for yr, grp in s.groupby(s.index.year):
            ic_mean = grp.mean()
            ic_std  = grp.std()
            icir    = ic_mean / ic_std if ic_std > 0 else np.nan
            flag    = "  *" if abs(icir) > 0.5 else ""
            print(f"  {yr:<6}  {h:<8}  {ic_mean:>+8.4f}  "
                  f"{ic_std:>8.4f}  {icir:>+8.3f}  {len(grp):>5}{flag}")
            rows.append({'Year': yr, 'Horizon': h,
                         'IC_mean': round(ic_mean, 4),
                         'IC_std' : round(ic_std,  4),
                         'ICIR'   : round(icir,    3),
                         'N_dates': len(grp)})

    composite_annual = pd.DataFrame(rows).set_index(['Year', 'Horizon'])

    # ── Plot: composite IC vs best individual factor per year ─────────────────
    _ics_plot_composite(composite_ic_df, ic_results, active_factors,
                        regime_s, w_active)

    return composite_ic_df, composite_annual


def _ics_plot_composite(composite_ic_df, ic_results, active_factors,
                        regime_s, w_active):
    """
    Three panels:
      1. Cumulative composite IC (both horizons) vs best individual factor
      2. Yearly ICIR: composite vs each active factor (grouped bars)
      3. Regime over time (background shading)
    """
    COLORS = {
        'Quality'  : '#7F77DD',
        'Idio_Mom' : '#1D9E75',
        'Value'    : '#D85A30',
        'Mom_12M1' : '#378ADD',
        'Composite': '#2C2C2A',
    }
    REGIME_COLORS = {0.0: '#E1F5EE', 0.5: '#F1EFE8', 1.0: '#FAECE7'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                             gridspec_kw={'height_ratios': [2.5, 2, 1]})
    fig.suptitle("Composite Alpha IC Study",
                 fontsize=13, fontweight='500', color='#2C2C2A', y=0.99)
    fig.patch.set_facecolor('#FAFAF9')
    for ax in axes:
        ax.set_facecolor('#FAFAF9')

    # ── Panel 1: cumulative IC ────────────────────────────────────────────────
    ax = axes[0]
    for h in composite_ic_df.columns:
        s   = composite_ic_df[h].dropna()
        cum = s.cumsum()
        ax.plot(cum.index.to_numpy(), cum.values,
                label=f'Composite {h}', color=COLORS['Composite'],
                linewidth=2.0,
                linestyle='-' if h == '63d' else '--')

    for fname in active_factors:
        if fname not in ic_results:
            continue
        ic_df = ic_results[fname]
        col   = '63d' if '63d' in ic_df.columns else ic_df.columns[0]
        s     = ic_df[col].dropna()
        cum   = s.cumsum()
        ax.plot(cum.index.to_numpy(), cum.values,
                label=f'{fname} ({col})',
                color=COLORS.get(fname, '#888780'),
                linewidth=1.0, alpha=0.6)

    ax.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax.set_ylabel("Cumulative IC", fontsize=10, color='#5F5E5A')
    ax.set_title("Cumulative IC — composite vs individual factors (63d)",
                 fontsize=11, fontweight='500')
    ax.legend(fontsize=8, ncol=3, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 2: yearly ICIR bars ─────────────────────────────────────────────
    ax2    = axes[1]
    hz     = '63d'
    years  = sorted(composite_ic_df.index.year.unique())
    all_series = {'Composite': composite_ic_df[hz] if hz in composite_ic_df.columns
                  else composite_ic_df.iloc[:, 0]}
    for fname in active_factors:
        if fname in ic_results and hz in ic_results[fname].columns:
            all_series[fname] = ic_results[fname][hz]

    n_series = len(all_series)
    x        = np.arange(len(years))
    width    = 0.8 / n_series

    for i, (name, s) in enumerate(all_series.items()):
        icirs = []
        for yr in years:
            grp = s[s.index.year == yr].dropna()
            if len(grp) < 2:
                icirs.append(0.0)
            else:
                icirs.append(float(grp.mean() / grp.std())
                             if grp.std() > 0 else 0.0)
        offset = (i - n_series / 2 + 0.5) * width
        lw     = 1.5 if name == 'Composite' else 0
        bc     = [COLORS.get(name, '#888780') if v >= 0 else '#F0997B'
                  for v in icirs]
        ax2.bar(x + offset, icirs, width, color=bc, alpha=0.85,
                label=name, linewidth=lw,
                edgecolor=COLORS.get(name, '#888780') if name == 'Composite' else 'none')

    ax2.axhline(0, color='#888780', linewidth=0.8)
    ax2.axhline( 0.5, color='#1D9E75', linewidth=0.6, linestyle=':', alpha=0.5)
    ax2.axhline(-0.5, color='#D85A30', linewidth=0.6, linestyle=':', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, fontsize=9)
    ax2.set_ylabel("ICIR (63d)", fontsize=10, color='#5F5E5A')
    ax2.set_title("Yearly ICIR — composite vs individual factors",
                  fontsize=11, fontweight='500')
    ax2.legend(fontsize=8, ncol=3, loc='upper right', framealpha=0.85)
    ax2.grid(axis='y', color='#D3D1C7', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Panel 3: regime over time ─────────────────────────────────────────────
    ax3 = axes[2]
    reg = regime_s.reindex(composite_ic_df.index, method='ffill').dropna()
    ax3.fill_between(reg.index.to_numpy(), reg.values, 0,
                     color='#378ADD', alpha=0.3)
    ax3.plot(reg.index.to_numpy(), reg.values,
             color='#378ADD', linewidth=1.0)
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.set_yticklabels(['Easy', 'Neutral', 'Tight'], fontsize=8)
    ax3.set_ylabel("Regime", fontsize=10, color='#5F5E5A')
    ax3.set_title("Rates regime over time", fontsize=11, fontweight='500')
    ax3.grid(color='#D3D1C7', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
    
    

# ===============================================================================
# ROLLING (POINT-IN-TIME) REGIME WEIGHTS
# ===============================================================================


# Step 1 — generate ic_results and ic_annual
# force rebuild:
# ic_results, ic_annual = run_ic_study_alpha(Pxs_df, sectors_s, force_recompute=True)

# daily use:
ic_results, ic_annual = run_ic_study_alpha(Pxs_df, sectors_s, model_version='v2')

# Step 2 — derive rolling point-in-time weights
weights_by_year, regime_s, weights_by_date = compute_rolling_regime_weights(
    ic_results, ic_annual, Pxs_df,
    horizons=['21d', '63d'], w1=63, w2=63, threshold=40
)

