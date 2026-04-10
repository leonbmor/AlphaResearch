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

Entry point
-----------
    results = run_composite_backtest(Pxs_df, sectors_s,
                                     weights_by_year, regime_s,
                                     volumeTrd_df=None)
"""

# ===============================================================================
# PARAMETERS
# ===============================================================================

CB_START_DATE  = pd.Timestamp('2019-01-01')
CB_TOP_N       = 20
CB_REBAL_FREQ  = 30
CB_MODEL_VER   = 'v2'
CB_W1          = 63
CB_W2          = 42
CB_THRESHOLD   = 20
START_DATE     = pd.Timestamp('2019-01-01')
STEP_DAYS      = 60
TOP_N          = 20
N_QUANTILES    = 5
MOM_LONG       = 252
MOM_SKIP       = 21

# ===============================================================================
# SECTOR CAP SELECTION
# ===============================================================================

def select_with_sector_cap(ranked_df, sector_cap, top_n):
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

# ===============================================================================
# PERFORMANCE SUMMARY
# ===============================================================================

def print_performance(nav_s, label=''):
    total_ret  = nav_s.iloc[-1] / nav_s.iloc[0] - 1
    n_years    = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    cagr       = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / n_years) - 1
    daily_rets = nav_s.pct_change().dropna()
    vol        = daily_rets.std() * np.sqrt(252)
    sharpe     = cagr / vol if vol > 0 else np.nan
    max_dd     = ((nav_s / nav_s.cummax()) - 1).min()
    print(f"\n  {'='*70}")
    print(f"  PERFORMANCE: {label}")
    print(f"  {'='*70}")
    print(f"  Period      : {nav_s.index[0].date()} -> {nav_s.index[-1].date()}")
    print(f"  Total return: {total_ret*100:+.1f}%")
    print(f"  CAGR        : {cagr*100:+.1f}%")
    print(f"  Annual vol  : {vol*100:.1f}%")
    print(f"  Sharpe      : {sharpe:.2f}")
    print(f"  Max drawdown: {max_dd*100:.1f}%")
    print(f"  Final NAV   : {nav_s.iloc[-1]:.4f}")

# ===============================================================================
# BACKTEST ENGINE
# ===============================================================================

def run_backtest(factor_by_date, calc_dates, Pxs_df,
                 use_vol_filter=False, use_mom_12m1=False, use_mom_idio=False,
                 resid_pivot=None, vol_pivot=None, mktcap_floor=None,
                 sector_cap=None, top_n=TOP_N, mom_weight=1.0,
                 prefilt_pct=1.0, conc_factor=1.0):
    nav          = 1.0
    nav_series   = {}
    portfolio    = []
    port_records = {}
    pxs_columns  = set(Pxs_df.columns)

    for i, rebal_date in enumerate(calc_dates):
        next_date = calc_dates[i + 1] if i + 1 < len(calc_dates) else Pxs_df.index.max()

        if rebal_date in factor_by_date:
            fdf = factor_by_date[rebal_date].copy()
            if mktcap_floor is not None and 'mkt_cap' in fdf.columns:
                fdf = fdf[fdf['mkt_cap'].fillna(0) >= mktcap_floor]
            fdf = fdf.loc[[t for t in fdf.index if t in pxs_columns]]
            if use_vol_filter and len(fdf) > top_n:
                surviving = apply_vol_filter(fdf.index, rebal_date, Pxs_df)
                fdf       = fdf.loc[fdf.index.intersection(surviving)]
            if prefilt_pct < 1.0 and len(fdf) > top_n:
                n_keep = max(top_n, int(np.ceil(len(fdf) * prefilt_pct)))
                fdf    = fdf.nlargest(n_keep, 'factor')
            if len(fdf) < top_n:
                print(f"  Skipping {rebal_date.date()}: only {len(fdf)} stocks")
                portfolio = []
            else:
                if use_mom_12m1 or use_mom_idio:
                    fdf['factor_z'] = zscore(fdf['factor'])
                    if use_mom_12m1:
                        mom_z = calc_momentum_12m1(rebal_date, fdf.index, Pxs_df, vol_pivot=vol_pivot)
                    else:
                        mom_z = calc_idio_momentum_score(rebal_date, fdf.index, resid_pivot, vol_pivot=vol_pivot)
                    fdf['mom_z']    = mom_z.reindex(fdf.index)
                    fdf['combined'] = (fdf['factor_z'] + mom_weight * fdf['mom_z']) / (1.0 + mom_weight)
                    fdf             = fdf.dropna(subset=['combined'])
                    rank_col        = 'combined'
                else:
                    rank_col = 'factor'

                if len(fdf) < top_n:
                    portfolio = []
                else:
                    ranked    = fdf.sort_values(rank_col, ascending=False)
                    if sector_cap is not None:
                        top = select_with_sector_cap(ranked, sector_cap, top_n)
                    else:
                        top = ranked.head(top_n)
                    portfolio = [t for t in top.index if t in pxs_columns]
                    n_port    = len(portfolio)
                    n_top_h   = int(np.ceil(n_port / 2))
                    n_bot_h   = n_port - n_top_h
                    if conc_factor == 1.0 or n_bot_h == 0:
                        weights = {t: 1.0 / n_port for t in portfolio}
                    else:
                        top_alloc = conc_factor / (conc_factor + 1.0)
                        bot_alloc = 1.0 / (conc_factor + 1.0)
                        weights   = {}
                        for j, t in enumerate(portfolio):
                            if j < n_top_h:
                                weights[t] = top_alloc / n_top_h
                            else:
                                weights[t] = bot_alloc / n_bot_h
                    port_records[rebal_date] = (
                        list(top.index) + [None] * (top_n - len(top.index))
                    )[:top_n]

        if not portfolio:
            period_dates = Pxs_df.index[(Pxs_df.index >= rebal_date) & (Pxs_df.index <= next_date)]
            for d in period_dates:
                nav_series[d] = nav
            continue

        period_dates = Pxs_df.index[(Pxs_df.index >= rebal_date) & (Pxs_df.index <= next_date)]
        if len(period_dates) < 2:
            nav_series[rebal_date] = nav
            continue

        px_start   = Pxs_df.loc[period_dates[0],  portfolio]
        px_end     = Pxs_df.loc[period_dates[-1], portfolio]
        stk_rets   = (px_end / px_start - 1).fillna(0)
        w_series   = pd.Series(weights).reindex(portfolio).fillna(0)
        period_ret = (stk_rets * w_series).sum()
        nav       *= (1 + period_ret)

        px_period  = Pxs_df.loc[period_dates, portfolio]
        stk_daily  = px_period.div(px_start, axis=1) - 1
        port_cum   = stk_daily.mul(w_series, axis=1).sum(axis=1)
        period_nav = nav / (1 + period_ret) * (1 + port_cum)
        for d, v in period_nav.items():
            nav_series[d] = v

    nav_s = pd.Series(nav_series).sort_index()
    if START_DATE not in nav_s.index:
        nav_s[START_DATE] = 1.0
        nav_s = nav_s.sort_index()

    port_df = pd.DataFrame.from_dict(
        port_records, orient='index',
        columns=[f'Stock{i+1}' for i in range(top_n)]
    )
    return nav_s, port_df

# ===============================================================================
# CALC DATES
# ===============================================================================

def generate_calc_dates(Pxs_df, step_days=STEP_DAYS):
    end_date = Pxs_df.index.max()
    dates    = []
    current  = START_DATE
    while current <= end_date:
        available = Pxs_df.index[Pxs_df.index >= current]
        if available.empty:
            break
        dates.append(available[0])
        current += pd.Timedelta(days=step_days)
    return sorted(set(dates))

# ===============================================================================
# COMPOSITE SCORE BUILDER
# ===============================================================================

def _cb_build_composite_scores(universe, calc_dates, Pxs_df, sectors_s,
                                weights_by_year, regime_s, volumeTrd_df,
                                model_version, exclude_factors=None):
    exclude_factors = exclude_factors or ['OU']
    first_w = next(iter(weights_by_year.values()))
    active  = [f for f in first_w.columns if f not in exclude_factors]

    print(f"  Active factors: {active}")
    print(f"  Point-in-time weights (sample -- year 2022):")
    if 2022 in weights_by_year:
        w_sample = weights_by_year[2022][active].copy()
        for r in w_sample.index:
            w_sample.loc[r] = _ics_bounded_normalize(
                w_sample.loc[r], w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX
            )
        print(w_sample.round(3))
    print()

    score_dfs = {}
    if 'Quality' in active:
        print("  Loading quality scores...")
        score_dfs['Quality'] = _ics_load_quality(universe, calc_dates, Pxs_df, sectors_s)
    if 'Idio_Mom' in active:
        print("  Loading idio momentum...")
        resid_df = _ics_load_idio_mom(universe, model_version)
        score_dfs['Idio_Mom'] = _ics_compute_idio_mom_scores(resid_df, calc_dates)
    if 'Value' in active:
        print("  Loading value scores...")
        score_dfs['Value'] = _ics_load_value(universe, calc_dates, sectors_s, model_version)
    if 'Mom_12M1' in active:
        print("  Computing 12M1 momentum...")
        score_dfs['Mom_12M1'] = _ics_compute_mom_12m1(universe, calc_dates, Pxs_df)
    if 'OU' in active:
        print("  Loading O-U scores...")
        ou = _ics_load_ou(universe, model_version)
        if not ou.empty:
            all_d = calc_dates.union(ou.index).sort_values()
            score_dfs['OU'] = ou.reindex(all_d).ffill().reindex(calc_dates)

    composite_by_date = {}
    n = len(calc_dates)
    for i, dt in enumerate(calc_dates):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Building composite [{i+1}/{n}] {dt.date()}", end='\r')

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
# BACKTEST WRAPPER
# ===============================================================================

def _cb_run_backtest(composite_by_date, calc_dates, Pxs_df, sectors_s,
                     use_vol_filter, mktcap_floor, sector_cap,
                     top_n, prefilt_pct, conc_factor):
    factor_by_date = {}
    pxs_cols = set(Pxs_df.columns)
    for dt, scores in composite_by_date.items():
        fdf = scores.rename('factor').to_frame()
        fdf['Sector'] = fdf.index.map(sectors_s)
        fdf = fdf.dropna(subset=['Sector'])
        fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
        if not fdf.empty:
            factor_by_date[dt] = fdf

    nav_s, port_df = run_backtest(
        factor_by_date=factor_by_date, calc_dates=calc_dates, Pxs_df=Pxs_df,
        use_vol_filter=use_vol_filter, use_mom_12m1=False, use_mom_idio=False,
        resid_pivot=None, mktcap_floor=mktcap_floor, sector_cap=sector_cap,
        top_n=top_n, prefilt_pct=prefilt_pct, conc_factor=conc_factor,
    )
    return nav_s, port_df, factor_by_date

# ===============================================================================
# PLOT
# ===============================================================================

def _cb_plot(nav_composite, nav_baseline, regime_s):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={'height_ratios': [3, 1.5, 1]})
    fig.patch.set_facecolor('#FAFAF9')
    for ax in axes:
        ax.set_facecolor('#FAFAF9')
    TEAL = '#1D9E75'; GRAY = '#888780'; BLUE = '#378ADD'
    common = nav_composite.index.intersection(nav_baseline.index)
    nc = nav_composite.reindex(common)
    nb = nav_baseline.reindex(common)
    reg = regime_s.reindex(common, method='ffill').fillna(0)
    REGIME_BG = {0.0: '#E1F5EE', 0.5: '#F1EFE8', 1.0: '#FAECE7'}
    prev_r, prev_d = float(reg.iloc[0]), common[0]
    for ax in axes[:2]:
        for d, r in reg.items():
            r = float(r)
            if r != prev_r or d == common[-1]:
                ax.axvspan(prev_d, d, color=REGIME_BG.get(prev_r, '#F1EFE8'), alpha=0.35, linewidth=0)
                prev_r, prev_d = r, d
    ax = axes[0]
    ax.plot(nc.index.to_numpy(), nc.values, color=TEAL, linewidth=1.8, label='Composite alpha')
    ax.plot(nb.index.to_numpy(), nb.values, color=GRAY, linewidth=1.2, label='Baseline (quality)', linestyle='--', alpha=0.8)
    ax.set_ylabel("NAV", fontsize=10, color='#5F5E5A')
    ax.set_title("Composite Alpha vs Quality Baseline", fontsize=12, fontweight='500', color='#2C2C2A')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax2 = axes[1]
    rel  = (nc / nb - 1) * 100
    ax2.plot(rel.index.to_numpy(), rel.values, color=BLUE, linewidth=1.2, label='Composite vs baseline (%)')
    ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
    ax2.fill_between(rel.index.to_numpy(), rel.values, 0, where=(rel.values >= 0), color=TEAL, alpha=0.15)
    ax2.fill_between(rel.index.to_numpy(), rel.values, 0, where=(rel.values < 0), color='#D85A30', alpha=0.15)
    ax2.set_ylabel("Relative (%)", fontsize=10, color='#5F5E5A')
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.85)
    ax2.grid(color='#D3D1C7', linewidth=0.5)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax3 = axes[2]
    ax3.fill_between(reg.index.to_numpy(), reg.values, 0, color=BLUE, alpha=0.25)
    ax3.plot(reg.index.to_numpy(), reg.values, color=BLUE, linewidth=1.0)
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.set_yticklabels(['Easy', 'Neutral', 'Tight'], fontsize=8)
    ax3.set_ylabel("Regime", fontsize=10, color='#5F5E5A')
    ax3.grid(color='#D3D1C7', linewidth=0.5)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig

# ===============================================================================
# ENTRY POINT
# ===============================================================================

def run_composite_backtest(Pxs_df, sectors_s, weights_by_year, regime_s,
                            volumeTrd_df=None):
    print("=" * 70)
    print("  COMPOSITE ALPHA BACKTEST")
    print("=" * 70)

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    print("\n  PORTFOLIO CONSTRUCTION OPTIONS:")
    topn_input    = input(f"  Number of stocks [default={CB_TOP_N}]: ").strip()
    rebal_input   = input(f"  Rebalancing frequency in days [default={CB_REBAL_FREQ}]: ").strip()
    sector_input  = input("  Max stocks per sector (or Enter to skip): ").strip()
    mktcap_input  = input("  Min market cap floor ($M, or Enter to skip): ").strip()
    vol_input     = input("  Apply vol filter? (y/n) [default=n]: ").strip().lower()
    prefilt_input = input("  Pre-filter fraction by composite score 0<x<=1 (or Enter for none): ").strip()
    conc_input    = input("  Concentration factor >=1.0 (or Enter for equal weight): ").strip()

    top_n        = int(topn_input)            if topn_input    else CB_TOP_N
    rebal_freq   = int(rebal_input)           if rebal_input   else CB_REBAL_FREQ
    sector_cap   = int(sector_input)          if sector_input  else None
    mktcap_floor = float(mktcap_input) * 1e6 if mktcap_input  else None
    use_vol      = vol_input == 'y'
    prefilt_pct  = float(prefilt_input)       if prefilt_input else 1.0
    if prefilt_pct <= 0 or prefilt_pct > 1:
        prefilt_pct = 1.0
    conc_factor  = float(conc_input)          if conc_input    else 1.0
    if conc_factor < 1.0:
        conc_factor = 1.0

    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"sector_cap={sector_cap}, mktcap_floor={mktcap_input or 'none'}, "
          f"vol_filter={use_vol}, prefilt={prefilt_pct:.0%}, conc={conc_factor:.1f}x")

    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(CB_START_DATE)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    pxs_cols       = set(Pxs_df.columns)

    calc_dates     = generate_calc_dates(Pxs_df, step_days=rebal_freq)
    calc_dates_idx = pd.DatetimeIndex(calc_dates)
    print(f"  Universe: {len(universe)} stocks  |  Rebalance dates: {len(calc_dates)}")

    # [1/3] Composite scores
    print("\n[1/3] Building composite alpha scores...")
    composite_by_date, score_dfs = _cb_build_composite_scores(
        universe=universe, calc_dates=calc_dates_idx, Pxs_df=Pxs_df,
        sectors_s=sectors_s, weights_by_year=weights_by_year, regime_s=regime_s,
        volumeTrd_df=volumeTrd_df, model_version=CB_MODEL_VER, exclude_factors=['OU'],
    )

    # [2/3] Composite backtest
    print("\n[2/3] Running composite backtest...")
    nav_composite, port_composite, factor_by_date = _cb_run_backtest(
        composite_by_date=composite_by_date, calc_dates=calc_dates, Pxs_df=Pxs_df,
        sectors_s=sectors_s, use_vol_filter=use_vol, mktcap_floor=mktcap_floor,
        sector_cap=sector_cap, top_n=top_n, prefilt_pct=prefilt_pct, conc_factor=conc_factor,
    )
    print_performance(nav_composite, "COMPOSITE ALPHA")

    # [3/3] Quality baseline
    print("\n[3/3] Running quality baseline...")
    all_tickers  = list(sectors_s.index)
    quality_wide = get_quality_scores(
        calc_dates=calc_dates_idx, universe=all_tickers, Pxs_df=Pxs_df,
        sectors_s=sectors_s, use_cached_weights=True, force_recompute=False,
    )
    quality_factor_by_date = {}
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
        factor_by_date=quality_factor_by_date, calc_dates=calc_dates, Pxs_df=Pxs_df,
        use_vol_filter=use_vol, mktcap_floor=mktcap_floor, sector_cap=sector_cap,
        top_n=top_n, prefilt_pct=prefilt_pct, conc_factor=conc_factor,
    )
    print_performance(nav_baseline, "QUALITY BASELINE")

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  COMPARISON")
    print(f"  {'='*70}")
    print(f"  {'Strategy':<35} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MDD':>8}")
    print(f"  {'-'*71}")
    for nav_s, lbl in [(nav_composite, 'Composite Alpha'), (nav_baseline, 'Quality Baseline')]:
        n_yrs  = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr   = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / n_yrs) - 1
        vol    = nav_s.pct_change().dropna().std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        mdd    = ((nav_s / nav_s.cummax()) - 1).min()
        print(f"  {lbl:<35} {cagr*100:>7.1f}% {vol*100:>7.1f}% {sharpe:>8.2f} {mdd*100:>7.1f}%")

    # ── Today snapshot ────────────────────────────────────────────────────────
    today = Pxs_df.index[-1]
    print(f"\n  Computing today's snapshot ({today.date()})...")

    def _build_today_port(scores_dict, label):
        if today not in scores_dict:
            print(f"  WARNING: no {label} scores for {today.date()}")
            return None
        scores = scores_dict[today]
        fdf    = scores.rename('factor').to_frame()
        fdf['Sector'] = fdf.index.map(sectors_s)
        fdf = fdf.dropna(subset=['Sector'])
        fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
        if prefilt_pct < 1.0 and len(fdf) > top_n:
            n_keep = max(top_n, int(np.ceil(len(fdf) * prefilt_pct)))
            fdf    = fdf.nlargest(n_keep, 'factor')
        if len(fdf) < top_n:
            print(f"  WARNING: only {len(fdf)} {label} stocks for today")
            return None
        ranked = fdf.sort_values('factor', ascending=False)
        if sector_cap is not None:
            top = select_with_sector_cap(ranked, sector_cap, top_n)
        else:
            top = ranked.head(top_n)
        row = {f'Stock{i+1}': t for i, t in enumerate(top.index)}
        for j in range(len(top), top_n):
            row[f'Stock{j+1}'] = None
        return pd.DataFrame([row], index=[today])

    # Composite today
    if today not in composite_by_date:
        today_comp, _ = _cb_build_composite_scores(
            universe=universe, calc_dates=pd.DatetimeIndex([today]),
            Pxs_df=Pxs_df, sectors_s=sectors_s, weights_by_year=weights_by_year,
            regime_s=regime_s, volumeTrd_df=volumeTrd_df,
            model_version=CB_MODEL_VER, exclude_factors=['OU'],
        )
    else:
        today_comp = {today: composite_by_date[today]}

    today_port_composite = _build_today_port(today_comp, 'composite')
    if today_port_composite is not None and today not in port_composite.index:
        port_composite = pd.concat([port_composite, today_port_composite])

    # Baseline today
    today_qual = {}
    if today in quality_wide.index:
        s = quality_wide.loc[today].dropna()
        if not s.empty:
            today_qual[today] = s
    today_port_baseline = _build_today_port(today_qual, 'baseline')
    if today_port_baseline is not None and today not in port_baseline.index:
        port_baseline = pd.concat([port_baseline, today_port_baseline])

    if today_port_composite is not None:
        print(f"  Composite today : {today_port_composite.iloc[0].dropna().tolist()}")
    if today_port_baseline is not None:
        print(f"  Baseline  today : {today_port_baseline.iloc[0].dropna().tolist()}")

    print(f"\n  port_composite : {len(port_composite)} dates x {len(port_composite.columns)} stocks")
    print(f"  port_baseline  : {len(port_baseline)} dates x {len(port_baseline.columns)} stocks")

    _cb_plot(nav_composite, nav_baseline, regime_s)

    return {
        'nav_composite'         : nav_composite,
        'nav_baseline'          : nav_baseline,
        'port_composite'        : port_composite,
        'port_baseline'         : port_baseline,
        'factor_by_date'        : factor_by_date,
        'quality_factor_by_date': quality_factor_by_date,
        'composite_by_date'     : composite_by_date,
        'score_dfs'             : score_dfs,
        'regime_s'              : regime_s,
        'weights_by_year'       : weights_by_year,
    }

results = run_composite_backtest(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df=volumeS_df
)
