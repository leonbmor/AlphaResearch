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

# ===============================================================================
# HELPERS
# ===============================================================================

def _ics_zscore(s):
    """Cross-sectional z-score, robust to NaNs."""
    mu, sd = s.mean(), s.std()
    return s * 0.0 if (sd == 0 or np.isnan(sd)) else (s - mu) / sd


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
                       rebal_freq=ICS_REBAL_FREQ):
    """
    Run IC study for all five alpha factors.

    Parameters
    ----------
    Pxs_df        : pd.DataFrame — price/macro panel
    sectors_s     : pd.Series   — ticker -> sector label
    model_version : 'v1' or 'v2'
    rebal_freq    : int — rebalance frequency in trading days (default 21)

    Returns
    -------
    ic_results : dict {factor_name: DataFrame(dates x horizons)}
    ic_annual  : pd.DataFrame — yearly IC summary (MultiIndex)
    """
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    print(f"\nAlpha Factor IC Study [{model_version}] — starting...\n")

    # ── Universe ──────────────────────────────────────────────────────────────
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    print(f"  Universe: {len(universe)} stocks")

    # ── Rebalance dates ───────────────────────────────────────────────────────
    all_rets   = Pxs_df[universe].pct_change().dropna(how='all')
    # Start from 2019 (factor model common sample start)
    start_dt   = pd.Timestamp('2019-01-01')
    # Need max(horizon) of future price data — exclude last 63 trading days
    end_dt     = Pxs_df.index[-max(ICS_HORIZONS) - 5]

    all_valid  = all_rets.index[
        (all_rets.index >= start_dt) &
        (all_rets.index <= end_dt) &
        (all_rets.notna().sum(axis=1) >= ICS_MIN_STOCKS)
    ]
    # Sample at rebal_freq cadence
    calc_dates = pd.DatetimeIndex(all_valid[::rebal_freq])
    print(f"  Calc dates: {len(calc_dates)} "
          f"({calc_dates[0].date()} → {calc_dates[-1].date()})\n")

    # ── Forward returns ───────────────────────────────────────────────────────
    print("[1/6] Computing forward returns...")
    fwd_rets = {}
    for h in ICS_HORIZONS:
        fwd_rets[h] = _ics_forward_returns(universe, calc_dates, Pxs_df, h)
        print(f"  {h}d forward returns: {fwd_rets[h].notna().any(axis=1).sum()} dates")

    # ── Factor scores ─────────────────────────────────────────────────────────
    print("\n[2/6] Loading quality scores...")
    quality_scores = _ics_load_quality(universe, calc_dates, Pxs_df, sectors_s)

    print("\n[3/6] Loading idio momentum...")
    resid_df    = _ics_load_idio_mom(universe, model_version)
    idio_scores = _ics_compute_idio_mom_scores(resid_df, calc_dates)

    print("\n[4/6] Loading value scores...")
    value_scores = _ics_load_value(universe, calc_dates, sectors_s, model_version)

    print("\n[5/6] Loading O-U scores...")
    ou_scores = _ics_load_ou(universe, model_version)
    if not ou_scores.empty:
        # Forward-fill to calc_dates
        all_dates = calc_dates.union(ou_scores.index).sort_values()
        ou_scores = ou_scores.reindex(all_dates).ffill().reindex(calc_dates)

    print("\n[6/6] Computing 12M1 momentum...")
    mom_scores = _ics_compute_mom_12m1(universe, calc_dates, Pxs_df)

    # ── Compute IC series per factor ──────────────────────────────────────────
    print("\nComputing IC series...")
    factor_map = {
        'Quality'  : quality_scores,
        'Idio_Mom' : idio_scores,
        'Value'    : value_scores,
        'OU'       : ou_scores,
        'Mom_12M1' : mom_scores,
    }

    ic_results = {}
    for fname, scores_df in factor_map.items():
        if scores_df is None or scores_df.empty:
            warnings.warn(f"  {fname}: no scores available — skipping")
            continue
        print(f"  Computing IC: {fname}...")
        ic_df = _ics_compute_ic_series(scores_df, fwd_rets)
        ic_df.columns = [f'{h}d' for h in ic_df.columns]
        ic_results[fname] = ic_df
        # Quick preview
        for col in ic_df.columns:
            s = ic_df[col].dropna()
            if not s.empty:
                print(f"    {col}: mean IC={s.mean():+.4f}  "
                      f"ICIR={s.mean()/s.std():+.3f}  N={len(s)}")

    # ── Yearly summary ────────────────────────────────────────────────────────
    print("\nBuilding yearly summary...")
    ic_annual = _ics_annual_summary(ic_results)
    _ics_print_summary(ic_annual)
    _ics_plot(ic_results, ic_annual)

    return ic_results, ic_annual


# ===============================================================================
# REGIME INDICATOR + WEIGHT DERIVATION
# ===============================================================================

def build_rates_regime(Pxs_df, w1=63, w2=42):
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
    w2     : int — smoothing window (default 42 — ~2 months)

    Returns
    -------
    regime_s : pd.Series  — daily regime score (0.0 / 0.5 / 1.0), index = dates
    """
    if 'USGG10YR' not in Pxs_df.columns:
        raise ValueError("'USGG10YR' column not found in Pxs_df.")

    raw    = Pxs_df['USGG10YR'].dropna()
    s      = (100 * (raw - raw.rolling(w1).mean())).round(2).dropna()
    s      = s[s.index >= pd.Timestamp('2018-01-01')]
    regime = (((s.rolling(w2).mean().dropna() // 30) + 1) / 2).clip(0, 1)

    print(f"  Rates regime built: {len(regime)} dates  "
          f"({regime.index[0].date()} → {regime.index[-1].date()})")
    print(f"  Distribution:\n{regime.value_counts().sort_index().to_string()}")
    return regime


def compute_regime_weights(ic_results, ic_annual, Pxs_df,
                            horizons=None, w1=63, w2=42):
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
    regime_s = build_rates_regime(Pxs_df, w1=w1, w2=w2)

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
        ic_yr = ic_annual.xs(yr, level='Year') if yr in \
                ic_annual.index.get_level_values('Year') else None
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

    # ── Zero-out negative ICs and normalise ───────────────────────────────────
    weights_df = raw_df.clip(lower=0)

    # If a whole row sums to zero (no positive ICs in this regime),
    # fall back to equal weights as a safety net
    row_sums = weights_df.sum(axis=1)
    for r in regimes:
        if row_sums[r] == 0:
            weights_df.loc[r] = 1.0 / len(factors)
            warnings.warn(f"  Regime {r}: no positive IC — using equal weights")
        else:
            weights_df.loc[r] = weights_df.loc[r] / row_sums[r]

    weights_df = weights_df.round(4)

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
