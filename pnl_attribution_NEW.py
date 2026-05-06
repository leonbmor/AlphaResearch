#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
pnl_attribution.py
==================
Daily portfolio PnL attribution across factor model factors.

For each day t where portfolio weights are provided:
    PnL_total_t      = sum_i  w_i_t * r_i_t
    PnL_factor_k_t   = (X_t' w_t)[k] * lambda_k_t      for each factor k
    PnL_idio_t       = PnL_total_t - sum_k PnL_factor_k_t

X_t uses the most recent available monthly snapshot from the
risk_residuals cache (same basis as portfolio_risk_decomp.py).
Lambda comes from the versioned factor lambda tables.

Assumes all factor_model_step1 functions/constants are live in kernel.
No imports needed.

Entry point
-----------
    attrib_df = run_pnl_attribution(port_df, Pxs_df, sectors_s,
                                    volumeTrd_df=None, model_version='v1')

Parameters
----------
    port_df       : pd.DataFrame — dates x tickers, % weight allocations
                    (values sum to ~1 per row; zero for stocks not held)
    Pxs_df        : pd.DataFrame — price/macro panel
    sectors_s     : pd.Series   — ticker -> sector label
    volumeTrd_df  : pd.DataFrame or None — vol scalars for idio momentum
    model_version : 'v1' (default) or 'v2'

Returns
-------
    attrib_df : pd.DataFrame — dates x (factors + Idiosyncratic + Total)
                daily PnL contribution as % of portfolio (decimal, e.g. 0.0123)
"""

# ===============================================================================
# PARAMETERS
# ===============================================================================

PA_PLOT_FIGSIZE   = (14, 10)
PA_ROLLING_WINDOW = 63       # rolling window for R² and Sharpe displays (days)

# ===============================================================================
# HELPERS
# ===============================================================================

def _pa_load_lambdas(model_version, lookback=None):
    """
    Load all factor lambda series from versioned tables.
    Returns wide DataFrame (dates x factor_names) aligned to canonical order.
    Also returns factor_names list and sec_cols list.
    """
    tbls     = RD_SCALAR_TABLES[model_version]
    frames   = []

    # Scalar factors
    for fname, tbl in tbls.items():
        lim = f"LIMIT {lookback}" if lookback else ""
        with ENGINE.connect() as conn:
            df = pd.read_sql(
                f"SELECT * FROM {tbl} ORDER BY date {lim}", conn
            )
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        col = df.drop(columns=[c for c in RD_LAMBDA_META if c in df.columns],
                      errors='ignore').select_dtypes(include=np.number)
        if col.shape[1] >= 1:
            frames.append(col.iloc[:, 0].rename(fname))

    # Macro
    with ENGINE.connect() as conn:
        mdf = pd.read_sql(
            f"SELECT * FROM {RD_MACRO_TBL[model_version]} ORDER BY date", conn
        )
    mdf['date'] = pd.to_datetime(mdf['date'])
    mdf = mdf.set_index('date').sort_index()
    mnum = mdf.drop(columns=[c for c in RD_LAMBDA_META if c in mdf.columns],
                    errors='ignore').select_dtypes(include=np.number)
    for mc in MACRO_COLS:
        if mc in mnum.columns:
            frames.append(mnum[mc].rename(mc))

    # Sectors
    with ENGINE.connect() as conn:
        sdf = pd.read_sql(
            f"SELECT * FROM {RD_SEC_TBL[model_version]} ORDER BY date", conn
        )
    sdf['date'] = pd.to_datetime(sdf['date'])
    sdf = sdf.set_index('date').sort_index()
    sec_cols = sorted([
        c for c in sdf.columns
        if c not in RD_LAMBDA_META
        and pd.api.types.is_float_dtype(sdf[c])
    ])
    for sc in sec_cols:
        frames.append(sdf[sc].rename(sc))

    # Combine in canonical order
    combined = pd.concat(frames, axis=1)
    head     = ['Beta', 'Size']
    tail     = ['Quality', 'SI', 'GK_Vol', 'Idio_Mom', 'Value', 'OU']
    ordered  = (
        [c for c in head         if c in combined.columns]
        + [c for c in MACRO_COLS if c in combined.columns]
        + [c for c in sec_cols   if c in combined.columns]
        + [c for c in tail       if c in combined.columns]
    )
    lam_df       = combined[ordered]
    factor_names = ordered
    return lam_df, factor_names, sec_cols


def _pa_load_x_snapshots(model_version):
    """
    Load X snapshots from risk_residuals cache dates.
    We re-derive the snapshot dates (month-ends) from what was cached,
    then rebuild X on those dates using _rd_build_X_on_date.
    Returns dict {date: DataFrame(universe x K)}.

    Note: this reuses the same monthly snapshot logic as _rd_update_risk_residuals.
    Snapshots are built lazily — only dates not already in the local dict.
    """
    # We just return None here and build on-the-fly in the main loop
    # using the forward-fill approach — the X snapshots are rebuilt
    # from Pxs_df which is available in the calling scope.
    return {}


def _pa_get_universe(Pxs_df, sectors_s):
    extended_st_dt = Pxs_df.index[0]
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT ticker FROM income_data"
            )).fetchall()
        db_tickers = {r[0].upper() for r in rows}
    except Exception:
        db_tickers = set(sectors_s.index)
    etf_tickers = set(sectors_s.values)
    pre_dates   = Pxs_df.index[Pxs_df.index < extended_st_dt]
    universe = []
    for col in sectors_s.index:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        if col not in Pxs_df.columns:
            continue
        if len(pre_dates) >= 252:
            cd = Pxs_df.loc[pre_dates[-252:], col]
            if isinstance(cd, pd.DataFrame): cd = cd.iloc[:, 0]
            if int(cd.notna().sum()) < 126:
                continue
        universe.append(col)
    return universe


# ===============================================================================
# MAIN ATTRIBUTION ENGINE
# ===============================================================================

def run_pnl_attribution(port_df, Pxs_df, sectors_s,
                        volumeTrd_df=None, model_version='v2'):
    """
    Compute daily PnL attribution across all factor model factors.

    Parameters
    ----------
    port_df       : pd.DataFrame — dates x tickers, % weight allocations
                    (rows sum to ~1; use 0 for stocks not held on a date)
    Pxs_df        : pd.DataFrame — price/macro panel
    sectors_s     : pd.Series   — ticker -> sector label
    volumeTrd_df  : pd.DataFrame or None — vol scalars for idio momentum
    model_version : 'v1' or 'v2'

    Returns
    -------
    attrib_df : pd.DataFrame — dates x factors (decimal daily PnL contribution)
                Columns: factor names + 'Idiosyncratic' + 'Total'
    """
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    port_df   = port_df.loc[:, ~port_df.columns.duplicated(keep='first')]

    print(f"\nPnL Attribution [{model_version}] — starting...\n")

    # ── Universe & factor structure ───────────────────────────────────────────
    print("[1/4] Loading universe and factor structure...")
    universe = _pa_get_universe(Pxs_df, sectors_s)

    # Build F to get factor_names and sec_cols (reuse _rd_build_F)
    F, factor_names, sec_cols = _rd_build_F(model_version=model_version)
    print(f"  Universe: {len(universe)} stocks  |  Factors: {len(factor_names)}")

    # ── Lambda time series ────────────────────────────────────────────────────
    print("[2/4] Loading factor lambda series...")
    lam_df, _, _ = _pa_load_lambdas(model_version)
    lam_df = lam_df.reindex(columns=factor_names)
    print(f"  Lambdas: {len(lam_df)} dates  "
          f"({lam_df.index[0].date()} → {lam_df.index[-1].date()})")

    # ── Build X snapshots (monthly, forward-filled) ───────────────────────────
    print("[3/4] Building X snapshots (monthly refresh)...")

    # Determine dates we need to attribute
    port_dates = port_df.index.intersection(lam_df.index)
    port_dates = port_dates.intersection(Pxs_df.index)
    port_dates = port_dates.sort_values()

    if len(port_dates) == 0:
        raise ValueError("No overlapping dates between port_df, lambda tables "
                         "and Pxs_df.")

    # X snapshot dates: month-ends within attribution window + latest date
    x_snap_dates = _rd_month_end_dates(port_dates, port_dates[-1])

    X_snapshots = {}
    n_snaps = len(x_snap_dates)
    for i, xdt in enumerate(x_snap_dates):
        print(f"  Building X snapshot [{i+1}/{n_snaps}] {xdt.date()}...",
              end='\r')
        xdf = _rd_build_X_on_date(
            xdt, universe, factor_names, sec_cols,
            Pxs_df, sectors_s, volumeTrd_df, model_version
        )
        if xdf is not None:
            X_snapshots[xdt] = xdf.fillna(0.0)
    print(f"\n  X snapshots built: {len(X_snapshots)}")

    if not X_snapshots:
        raise ValueError("Could not build any X snapshots — check Pxs_df coverage.")

    snapshot_dates = sorted(X_snapshots.keys())

    # ── Stock returns ─────────────────────────────────────────────────────────
    all_rets = Pxs_df.pct_change()

    # ── Daily attribution loop ────────────────────────────────────────────────
    print("[4/4] Computing daily PnL attribution...")

    results           = {}
    stock_idio_results = {}
    n_dates   = len(port_dates)
    missing_X = 0

    for i, dt in enumerate(port_dates):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n_dates}] {dt.date()}", end='\r')

        # Portfolio weights on this date
        w_raw = port_df.loc[dt].reindex(universe).fillna(0.0)
        if w_raw.sum() == 0:
            continue
        w = w_raw / w_raw.sum()   # normalise to sum to 1

        # Stock returns on this date
        if dt not in all_rets.index:
            continue
        r_t = all_rets.loc[dt].reindex(universe).fillna(0.0)

        # Actual portfolio return
        pnl_total = float(w @ r_t)

        # Most recent X snapshot on or before dt
        valid_snaps = [d for d in snapshot_dates if d <= dt]
        if not valid_snaps:
            missing_X += 1
            continue
        X_t = X_snapshots[valid_snaps[-1]]  # DataFrame (universe x K)

        # Factor returns on this date
        if dt not in lam_df.index:
            continue
        lam_t = lam_df.loc[dt].reindex(factor_names).fillna(0.0).values  # (K,)

        # Portfolio factor exposures: (K,)
        Xw = X_t.values.T @ w.values   # (K,)

        # Factor PnL contributions: (K,)
        factor_pnl = Xw * lam_t        # element-wise: exposure × factor return

        # Per-stock predicted return: X_t @ lam_t  (N,)
        pred_t   = X_t.values @ lam_t                    # (N,)
        epsilon_t = r_t.values - pred_t                   # (N,) stock idio returns

        # Per-stock idio PnL contribution: w_i * epsilon_i
        stock_idio_t = w.values * epsilon_t               # (N,)

        # Aggregate idiosyncratic PnL
        pnl_systematic = float(factor_pnl.sum())
        pnl_idio       = float(stock_idio_t.sum())

        row = dict(zip(factor_names, factor_pnl))
        row['Idiosyncratic'] = pnl_idio
        row['Total']         = pnl_total
        results[dt]          = row
        stock_idio_results[dt] = pd.Series(stock_idio_t, index=universe)

    print(f"\n  Done: {len(results)} dates attributed")
    if missing_X > 0:
        warnings.warn(f"  {missing_X} dates skipped — no X snapshot available")

    attrib_df            = pd.DataFrame(results).T
    attrib_df.index.name = 'date'
    attrib_df            = attrib_df[factor_names + ['Idiosyncratic', 'Total']]

    # Per-stock idio DataFrame (dates x universe)
    stock_idio_df            = pd.DataFrame(stock_idio_results).T
    stock_idio_df.index.name = 'date'
    stock_idio_df            = stock_idio_df.reindex(columns=universe)

    # ── Summary & plots ───────────────────────────────────────────────────────
    _pa_print_summary(attrib_df, stock_idio_df, factor_names, sec_cols, model_version)
    _pa_plot(attrib_df, factor_names, sec_cols, model_version)

    return attrib_df, stock_idio_df


# ===============================================================================
# SUMMARY
# ===============================================================================

def _pa_print_summary(attrib_df, stock_idio_df, factor_names, sec_cols, model_version):
    """Print attribution summary: annualized contribution, vol, t-stat."""
    total = attrib_df['Total']
    idio  = attrib_df['Idiosyncratic']
    sys   = total - idio

    # Realized R²
    r2 = 1.0 - idio.var() / total.var() if total.var() > 0 else np.nan

    print("\n" + "=" * 74)
    print(f"  PnL ATTRIBUTION SUMMARY  [{model_version}]")
    print("=" * 74)
    print(f"  Dates: {len(attrib_df)}  |  "
          f"Period: {attrib_df.index[0].date()} → {attrib_df.index[-1].date()}")
    print(f"  Realized R² (systematic share): {r2*100:.2f}%")
    print(f"  Ann. portfolio return : {total.mean()*252*100:+.2f}%")
    print(f"  Ann. portfolio vol    : {total.std()*np.sqrt(252)*100:.2f}%")

    def _block(label, factors):
        sub = attrib_df[[f for f in factors if f in attrib_df.columns]]
        if sub.empty:
            return
        grp_total = sub.sum(axis=1)
        ann_ret   = grp_total.mean() * 252 * 100
        ann_vol   = grp_total.std()  * np.sqrt(252) * 100
        t         = (grp_total.mean() / (grp_total.std() / np.sqrt(len(grp_total)))
                     if grp_total.std() > 0 else np.nan)
        pct_var   = grp_total.var() / total.var() * 100 if total.var() > 0 else np.nan
        print(f"\n  -- {label} --")
        print(f"  {'Factor':<28}  {'AnnRet%':>8}  {'AnnVol%':>8}  "
              f"{'t-stat':>7}  {'%VarShare':>10}")
        print("  " + "-" * 68)
        for col in sub.columns:
            s      = sub[col]
            ar     = s.mean() * 252 * 100
            av     = s.std()  * np.sqrt(252) * 100
            tv     = (s.mean() / (s.std() / np.sqrt(len(s)))
                      if s.std() > 0 else np.nan)
            vs     = s.var() / total.var() * 100 if total.var() > 0 else np.nan
            print(f"  {col:<28}  {ar:>+8.3f}  {av:>8.3f}  "
                  f"{tv:>+7.2f}  {vs:>10.2f}%")
        print(f"  {'  subtotal':<28}  {ann_ret:>+8.3f}  {ann_vol:>8.3f}  "
              f"{t:>+7.2f}  {pct_var:>10.2f}%")

    _block("Structural",    ['Beta', 'Size'])
    _block("Macro",         MACRO_COLS)
    _block("Sectors",       sec_cols)
    _block("Alpha",         ['Quality','SI','GK_Vol','Idio_Mom','Value','OU'])
    _block("Idiosyncratic", ['Idiosyncratic'])

    # ── Per-stock idiosyncratic breakdown ────────────────────────────────────
    if stock_idio_df is not None and not stock_idio_df.empty:
        # Cumulative idio contribution per stock over the period
        cum_idio   = stock_idio_df.sum(axis=0)             # total over all dates
        ann_idio   = stock_idio_df.mean(axis=0) * 252      # annualised mean
        vol_idio   = stock_idio_df.std(axis=0) * np.sqrt(252)
        n          = stock_idio_df.notna().sum(axis=0)
        t_idio     = ann_idio / (vol_idio / np.sqrt(n))                      .replace(0, np.nan)

        # Only show stocks with non-zero cumulative contribution
        active     = cum_idio[cum_idio.abs() > 1e-6].sort_values()

        print(f"\n  -- Per-stock idiosyncratic contribution --")
        print(f"  {'Ticker':<12}  {'CumPnL%':>9}  {'AnnRet%':>9}  "
              f"{'AnnVol%':>9}  {'t-stat':>8}")
        print("  " + "-" * 54)
        for ticker in active.index:
            cp = cum_idio[ticker] * 100
            ar = ann_idio[ticker] * 100
            av = vol_idio[ticker] * 100
            tv = t_idio[ticker]
            print(f"  {ticker:<12}  {cp:>+9.3f}  {ar:>+9.3f}  "
                  f"{av:>9.3f}  {tv:>+8.2f}")
        print(f"  {'  total':<12}  {active.sum()*100:>+9.3f}")

    print("\n" + "=" * 74 + "\n")


# ===============================================================================
# PLOTS
# ===============================================================================

def _pa_plot(attrib_df, factor_names, sec_cols, model_version):
    """
    Three panels:
      1. Cumulative PnL by group (structural / macro / sectors / alpha / idio)
      2. Rolling realized R² (systematic share of daily PnL variance)
      3. Daily factor PnL heatmap — recent 60 days, factors on y-axis
    """
    COLORS = {
        'Structural'    : '#378ADD',
        'Macro'         : '#EF9F27',
        'Sectors'       : '#5DCAA5',
        'Alpha'         : '#7F77DD',
        'Idiosyncratic' : '#D85A30',
        'Total'         : '#2C2C2A',
    }

    alpha_cols = [c for c in ['Quality','SI','GK_Vol','Idio_Mom','Value','OU']
                  if c in attrib_df.columns]
    struct_cols = [c for c in ['Beta','Size'] if c in attrib_df.columns]
    macro_cols  = [c for c in MACRO_COLS      if c in attrib_df.columns]
    sector_cols = [c for c in sec_cols        if c in attrib_df.columns]
    total       = attrib_df['Total']

    fig, axes = plt.subplots(3, 1, figsize=PA_PLOT_FIGSIZE,
                             gridspec_kw={'height_ratios': [3, 1.5, 2]})
    fig.suptitle(f"PnL Attribution [{model_version}]",
                 fontsize=13, fontweight='bold', y=0.99)
    fig.patch.set_facecolor('#FAFAF9')
    for ax in axes:
        ax.set_facecolor('#FAFAF9')

    # ── Panel 1: Cumulative PnL by group ──────────────────────────────────────
    ax = axes[0]
    groups = [
        ('Total',          [attrib_df['Total']]),
        ('Structural',     [attrib_df[c] for c in struct_cols]),
        ('Macro',          [attrib_df[c] for c in macro_cols]),
        ('Sectors',        [attrib_df[c] for c in sector_cols]),
        ('Alpha',          [attrib_df[c] for c in alpha_cols]),
        ('Idiosyncratic',  [attrib_df['Idiosyncratic']]),
    ]
    for gname, series_list in groups:
        grp = pd.concat(series_list, axis=1).sum(axis=1)
        cum = grp.cumsum() * 100
        lw  = 2.0 if gname == 'Total' else 1.2
        ls  = '-'  if gname == 'Total' else '--' if gname == 'Idiosyncratic' else '-'
        ax.plot(cum.index.to_numpy(), cum.values, label=gname,
                color=COLORS.get(gname, '#888780'),
                linewidth=lw, linestyle=ls)

    ax.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax.set_ylabel("Cumulative PnL (%)", fontsize=10, color='#5F5E5A')
    ax.set_title("Cumulative PnL by Factor Group", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 2: Rolling realized R² ──────────────────────────────────────────
    ax = axes[1]
    idio  = attrib_df['Idiosyncratic']
    roll_r2 = 1.0 - (
        idio.rolling(PA_ROLLING_WINDOW).var() /
        total.rolling(PA_ROLLING_WINDOW).var()
    )
    ax.plot(roll_r2.index.to_numpy(), roll_r2.values * 100,
            color='#1D9E75', linewidth=1.2, label=f'Rolling {PA_ROLLING_WINDOW}d R²')
    ax.axhline(roll_r2.mean() * 100, color='#1D9E75', linewidth=0.8,
               linestyle='--', alpha=0.6,
               label=f'Mean: {roll_r2.mean()*100:.1f}%')
    ax.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax.set_ylabel("Realized R² (%)", fontsize=10, color='#5F5E5A')
    ax.set_title(f"Rolling {PA_ROLLING_WINDOW}d Realized R² "
                 f"(systematic share of PnL variance)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-20, 110)

    # ── Panel 3: Recent factor PnL heatmap ────────────────────────────────────
    ax = axes[2]
    recent      = attrib_df[factor_names + ['Idiosyncratic']].iloc[-60:]
    hmap_data   = recent.T * 100   # factors x dates, in %

    # Cap colorscale at ±0.5% for readability
    vmax = 0.5
    im   = ax.imshow(hmap_data.values, aspect='auto', cmap='RdYlGn',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')

    ax.set_yticks(range(len(hmap_data.index)))
    ax.set_yticklabels(hmap_data.index, fontsize=7)
    # Show only every ~10th date on x-axis
    step = max(1, len(recent) // 10)
    ax.set_xticks(range(0, len(recent), step))
    ax.set_xticklabels(
        [d.strftime('%m/%d') for d in recent.index[::step]],
        fontsize=7, rotation=45, ha='right'
    )
    ax.set_title("Daily Factor PnL — last 60 days (%)",
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='vertical',
                 label='Daily PnL (%)', shrink=0.8)

    plt.tight_layout()
    plt.show()
    return fig

