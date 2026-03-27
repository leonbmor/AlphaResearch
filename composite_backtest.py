#!/usr/bin/env python
# coding: utf-8

"""
composite_backtest.py
=====================
Backtests a long-only portfolio constructed using the regime-conditioned
composite alpha signal (Quality + Idio_Mom + Value + Mom_12M1) derived
from factor_ic_study.py.

Compares against the baseline quality-only portfolio from
primary_factor_backtest.py using identical universe, rebalancing frequency,
and portfolio construction rules.

Assumes all factor_model_step1, quality_factor, primary_factor_backtest,
and factor_ic_study functions are already loaded in the Jupyter kernel.

Entry point
-----------
    results = run_composite_backtest(Pxs_df, sectors_s,
                                     weights_df, regime_s,
                                     volumeTrd_df=None)

Parameters
----------
    Pxs_df       : pd.DataFrame — price/macro panel
    sectors_s    : pd.Series   — ticker -> sector label
    weights_df   : pd.DataFrame — regime x factor weights (from compute_regime_weights)
    regime_s     : pd.Series   — daily regime score (from build_rates_regime)
    volumeTrd_df : pd.DataFrame or None — vol scalars for idio momentum

Returns
-------
    dict with keys:
        nav_composite  : pd.Series   — daily NAV, composite portfolio
        nav_baseline   : pd.Series   — daily NAV, quality baseline
        port_composite : pd.DataFrame — rebalance dates x stock slots
        port_baseline  : pd.DataFrame — rebalance dates x stock slots
        factor_by_date : dict         — composite scores per rebalance date
"""

# ===============================================================================
# PARAMETERS (defaults — all overridable via user prompts)
# ===============================================================================

CB_START_DATE  = pd.Timestamp('2019-01-01')
CB_TOP_N       = 20
CB_REBAL_FREQ  = 30     # trading days between rebalances
CB_MODEL_VER   = 'v2'  # which residuals/scores to use

# Regime indicator defaults (should match what was used in compute_regime_weights)
CB_W1         = 63
CB_W2         = 42
CB_THRESHOLD  = 20

# ===============================================================================
# COMPOSITE SCORE BUILDER
# ===============================================================================

def _cb_build_composite_scores(universe, calc_dates, Pxs_df, sectors_s,
                                weights_df, regime_s, volumeTrd_df,
                                model_version, exclude_factors=None):
    """
    Build composite alpha scores for all rebalance dates.

    For each date t:
        r_t     = regime_s[t]                            (0.0 / 0.5 / 1.0)
        w_t     = weights_df.loc[r_t]                    (factor weights)
        alpha_i = sum_k  w_t[k] * factor_score_i_t[k]   (composite)

    Returns dict {date: pd.Series(ticker -> composite_score)}
    """
    exclude_factors = exclude_factors or ['OU']

    # Active factors and re-normalised weights
    active = [f for f in weights_df.columns if f not in exclude_factors]
    w_act  = weights_df[active].copy()
    rs     = w_act.sum(axis=1)
    for r in w_act.index:
        w_act.loc[r] = w_act.loc[r] / rs[r] if rs[r] > 0 else 1.0 / len(active)

    print(f"  Active factors: {active}")
    print(f"  Re-normalised weights:\n{w_act.round(3)}\n")

    # ── Load factor scores across all calc_dates ──────────────────────────────
    score_dfs = {}

    if 'Quality' in active:
        print("  Loading quality scores...")
        score_dfs['Quality'] = _ics_load_quality(
            universe, calc_dates, Pxs_df, sectors_s
        )

    if 'Idio_Mom' in active:
        print("  Loading idio momentum...")
        resid_df = _ics_load_idio_mom(universe, model_version)
        score_dfs['Idio_Mom'] = _ics_compute_idio_mom_scores(resid_df, calc_dates)

    if 'Value' in active:
        print("  Loading value scores...")
        score_dfs['Value'] = _ics_load_value(
            universe, calc_dates, sectors_s, model_version
        )

    if 'Mom_12M1' in active:
        print("  Computing 12M1 momentum...")
        score_dfs['Mom_12M1'] = _ics_compute_mom_12m1(universe, calc_dates, Pxs_df)

    if 'OU' in active:
        print("  Loading O-U scores...")
        ou = _ics_load_ou(universe, model_version)
        if not ou.empty:
            all_d = calc_dates.union(ou.index).sort_values()
            score_dfs['OU'] = ou.reindex(all_d).ffill().reindex(calc_dates)

    # ── Build composite per date ───────────────────────────────────────────────
    composite_by_date = {}
    n = len(calc_dates)

    for i, dt in enumerate(calc_dates):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Building composite [{i+1}/{n}] {dt.date()}", end='\r')

        # Get regime — forward-fill if date not in regime_s
        reg_cands = regime_s[regime_s.index <= dt]
        if reg_cands.empty:
            continue
        r = float(reg_cands.iloc[-1])
        # Snap to nearest available regime
        if r not in w_act.index:
            r = min(w_act.index, key=lambda x: abs(x - r))
        w_t = w_act.loc[r]

        # Weighted sum of factor scores
        parts = []
        for fname in active:
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
            composite_by_date[dt] = composite

    print(f"\n  Composite scores built: {len(composite_by_date)} dates")
    return composite_by_date, score_dfs


# ===============================================================================
# ADAPTED run_backtest WRAPPER
# ===============================================================================

def _cb_run_backtest(composite_by_date, calc_dates, Pxs_df, sectors_s,
                     use_vol_filter, mktcap_floor, sector_cap,
                     top_n, prefilt_pct, conc_factor):
    """
    Runs the backtest using composite scores as the ranking signal.
    Reuses run_backtest from primary_factor_backtest.py with the composite
    scores injected as the 'factor' column.
    """
    # Build factor_by_date in the format expected by run_backtest:
    # {date: DataFrame(ticker -> ['factor', 'Sector', 'mkt_cap'])}
    factor_by_date = {}
    pxs_cols = set(Pxs_df.columns)

    for dt, scores in composite_by_date.items():
        fdf = scores.rename('factor').to_frame()
        fdf['Sector']  = fdf.index.map(sectors_s)
        fdf = fdf.dropna(subset=['Sector'])
        fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
        if not fdf.empty:
            factor_by_date[dt] = fdf

    nav_s, port_df = run_backtest(
        factor_by_date = factor_by_date,
        calc_dates     = calc_dates,
        Pxs_df         = Pxs_df,
        use_vol_filter = use_vol_filter,
        use_mom_12m1   = False,   # momentum already in composite
        use_mom_idio   = False,
        resid_pivot    = None,
        mktcap_floor   = mktcap_floor,
        sector_cap     = sector_cap,
        top_n          = top_n,
        prefilt_pct    = prefilt_pct,
        conc_factor    = conc_factor,
    )
    return nav_s, port_df, factor_by_date


# ===============================================================================
# COMPARISON PLOT
# ===============================================================================

def _cb_plot(nav_composite, nav_baseline, regime_s):
    """NAV comparison + regime background + drawdown panel."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={'height_ratios': [3, 1.5, 1]})
    fig.patch.set_facecolor('#FAFAF9')
    for ax in axes:
        ax.set_facecolor('#FAFAF9')

    TEAL  = '#1D9E75'
    GRAY  = '#888780'
    BLUE  = '#378ADD'

    # Align both series to common dates
    common = nav_composite.index.intersection(nav_baseline.index)
    nc = nav_composite.reindex(common)
    nb = nav_baseline.reindex(common)

    # ── Regime background shading ─────────────────────────────────────────────
    reg = regime_s.reindex(common, method='ffill').fillna(0)
    REGIME_BG = {0.0: '#E1F5EE', 0.5: '#F1EFE8', 1.0: '#FAECE7'}
    prev_r, prev_d = float(reg.iloc[0]), common[0]
    for ax in axes[:2]:
        for d, r in reg.items():
            r = float(r)
            if r != prev_r or d == common[-1]:
                ax.axvspan(prev_d, d,
                           color=REGIME_BG.get(prev_r, '#F1EFE8'),
                           alpha=0.35, linewidth=0)
                prev_r, prev_d = r, d

    # ── Panel 1: NAV ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(nc.index.to_numpy(), nc.values,
            color=TEAL, linewidth=1.8, label='Composite alpha')
    ax.plot(nb.index.to_numpy(), nb.values,
            color=GRAY, linewidth=1.2, label='Baseline (quality)',
            linestyle='--', alpha=0.8)
    ax.set_ylabel("NAV", fontsize=10, color='#5F5E5A')
    ax.set_title("Composite Alpha vs Quality Baseline",
                 fontsize=12, fontweight='500', color='#2C2C2A')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 2: relative performance ─────────────────────────────────────────
    ax2 = axes[1]
    rel  = (nc / nb - 1) * 100
    ax2.plot(rel.index.to_numpy(), rel.values,
             color=BLUE, linewidth=1.2, label='Composite vs baseline (%)')
    ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
    ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                     where=(rel.values >= 0), color=TEAL, alpha=0.15)
    ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                     where=(rel.values < 0), color='#D85A30', alpha=0.15)
    ax2.set_ylabel("Relative (%)", fontsize=10, color='#5F5E5A')
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.85)
    ax2.grid(color='#D3D1C7', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Panel 3: regime ───────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(reg.index.to_numpy(), reg.values, 0,
                     color=BLUE, alpha=0.25)
    ax3.plot(reg.index.to_numpy(), reg.values,
             color=BLUE, linewidth=1.0)
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.set_yticklabels(['Easy', 'Neutral', 'Tight'], fontsize=8)
    ax3.set_ylabel("Regime", fontsize=10, color='#5F5E5A')
    ax3.grid(color='#D3D1C7', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


# ===============================================================================
# ENTRY POINT
# ===============================================================================

def run_composite_backtest(Pxs_df, sectors_s, weights_df, regime_s,
                            volumeTrd_df=None):
    """
    Run composite alpha backtest vs quality baseline.

    Parameters
    ----------
    Pxs_df       : pd.DataFrame — price/macro panel
    sectors_s    : pd.Series   — ticker -> sector label
    weights_df   : pd.DataFrame — regime x factor weights
    regime_s     : pd.Series   — daily regime score
    volumeTrd_df : optional vol scalars

    Returns
    -------
    dict with nav_composite, nav_baseline, port_composite, port_baseline,
         factor_by_date, quality_factor_by_date
    """
    print("=" * 70)
    print("  COMPOSITE ALPHA BACKTEST")
    print("=" * 70)

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    # ── User inputs (mirrors primary_factor_backtest.py) ──────────────────────
    print("\n  PORTFOLIO CONSTRUCTION OPTIONS:")
    topn_input   = input(f"  Number of stocks [default={CB_TOP_N}]: ").strip()
    rebal_input  = input(f"  Rebalancing frequency in days [default={CB_REBAL_FREQ}]: ").strip()
    sector_input = input("  Max stocks per sector (or Enter to skip): ").strip()
    mktcap_input = input("  Min market cap floor ($M, or Enter to skip): ").strip()
    vol_input    = input("  Apply vol filter? (y/n) [default=n]: ").strip().lower()
    prefilt_input = input("  Pre-filter fraction by composite score 0<x<=1 (or Enter for none): ").strip()
    conc_input   = input("  Concentration factor >=1.0 (or Enter for equal weight): ").strip()

    top_n        = int(topn_input)        if topn_input    else CB_TOP_N
    rebal_freq   = int(rebal_input)       if rebal_input   else CB_REBAL_FREQ
    sector_cap   = int(sector_input)      if sector_input  else None
    mktcap_floor = float(mktcap_input) * 1e6 if mktcap_input else None
    use_vol      = vol_input == 'y'
    prefilt_pct  = float(prefilt_input)   if prefilt_input else 1.0
    if prefilt_pct <= 0 or prefilt_pct > 1:
        prefilt_pct = 1.0
    conc_factor  = float(conc_input)      if conc_input    else 1.0
    if conc_factor < 1.0:
        conc_factor = 1.0

    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"sector_cap={sector_cap}, mktcap_floor={mktcap_input or 'none'}, "
          f"vol_filter={use_vol}, prefilt={prefilt_pct:.0%}, "
          f"conc={conc_factor:.1f}x")

    # ── Universe and calc dates ───────────────────────────────────────────────
    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(CB_START_DATE)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)

    # Rebalance dates from CB_START_DATE
    calc_dates = generate_calc_dates(Pxs_df, step_days=rebal_freq)
    calc_dates_idx = pd.DatetimeIndex(calc_dates)
    print(f"  Universe: {len(universe)} stocks  |  "
          f"Rebalance dates: {len(calc_dates)}")

    # ── Build composite scores ────────────────────────────────────────────────
    print("\n[1/3] Building composite alpha scores...")
    composite_by_date, score_dfs = _cb_build_composite_scores(
        universe      = universe,
        calc_dates    = calc_dates_idx,
        Pxs_df        = Pxs_df,
        sectors_s     = sectors_s,
        weights_df    = weights_df,
        regime_s      = regime_s,
        volumeTrd_df  = volumeTrd_df,
        model_version = CB_MODEL_VER,
        exclude_factors = ['OU'],
    )

    # ── Composite backtest ────────────────────────────────────────────────────
    print("\n[2/3] Running composite backtest...")
    nav_composite, port_composite, factor_by_date = _cb_run_backtest(
        composite_by_date = composite_by_date,
        calc_dates        = calc_dates,
        Pxs_df            = Pxs_df,
        sectors_s         = sectors_s,
        use_vol_filter    = use_vol,
        mktcap_floor      = mktcap_floor,
        sector_cap        = sector_cap,
        top_n             = top_n,
        prefilt_pct       = prefilt_pct,
        conc_factor       = conc_factor,
    )
    print_performance(nav_composite, "COMPOSITE ALPHA")

    # ── Baseline backtest (quality only) ─────────────────────────────────────
    print("\n[3/3] Running quality baseline...")
    all_tickers  = list(sectors_s.index)
    quality_wide = get_quality_scores(
        calc_dates         = calc_dates_idx,
        universe           = all_tickers,
        Pxs_df             = Pxs_df,
        sectors_s          = sectors_s,
        use_cached_weights = True,
        force_recompute    = False,
    )
    quality_factor_by_date = {}
    pxs_cols = set(Pxs_df.columns)
    for dt in calc_dates:
        if dt not in quality_wide.index:
            continue
        scores = quality_wide.loc[dt].dropna()
        if scores.empty:
            continue
        fdf = scores.rename('factor').to_frame()
        fdf['Sector'] = fdf.index.map(sectors_s)
        fdf = fdf.dropna(subset=['Sector'])
        fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
        if not fdf.empty:
            quality_factor_by_date[dt] = fdf

    nav_baseline, port_baseline = run_backtest(
        factor_by_date = quality_factor_by_date,
        calc_dates     = calc_dates,
        Pxs_df         = Pxs_df,
        use_vol_filter = use_vol,
        mktcap_floor   = mktcap_floor,
        sector_cap     = sector_cap,
        top_n          = top_n,
        prefilt_pct    = prefilt_pct,
        conc_factor    = conc_factor,
    )
    print_performance(nav_baseline, "QUALITY BASELINE")

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print(f"\n  {'='*70}")
    print(f"  COMPARISON")
    print(f"  {'='*70}")
    print(f"  {'Strategy':<35} {'CAGR':>8} {'Vol':>8} "
          f"{'Sharpe':>8} {'MDD':>8}")
    print(f"  {'-'*71}")
    for nav_s, lbl in [(nav_composite, 'Composite Alpha'),
                        (nav_baseline,  'Quality Baseline')]:
        n_yrs  = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr   = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / n_yrs) - 1
        vol    = nav_s.pct_change().dropna().std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        mdd    = ((nav_s / nav_s.cummax()) - 1).min()
        print(f"  {lbl:<35} {cagr*100:>7.1f}% {vol*100:>7.1f}% "
              f"{sharpe:>8.2f} {mdd*100:>7.1f}%")

    print(f"\n  port_composite : {len(port_composite)} rebalance dates "
          f"x {len(port_composite.columns)} stocks")
    print(f"  port_baseline  : {len(port_baseline)} rebalance dates "
          f"x {len(port_baseline.columns)} stocks")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _cb_plot(nav_composite, nav_baseline, regime_s)

    return {
        'nav_composite'        : nav_composite,
        'nav_baseline'         : nav_baseline,
        'port_composite'       : port_composite,
        'port_baseline'        : port_baseline,
        'factor_by_date'       : factor_by_date,
        'quality_factor_by_date': quality_factor_by_date,
        'composite_by_date'    : composite_by_date,
        'score_dfs'            : score_dfs,
        'regime_s'             : regime_s,
        'weights_df'           : weights_df,
    }
