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
warnings.filterwarnings('ignore')

# ===============================================================================
# PARAMETERS
# ===============================================================================

MB_START_DATE  = pd.Timestamp('2019-01-01')
MB_TOP_N       = 20
MB_REBAL_FREQ  = 30
MB_MODEL_VER   = 'v2'

# Cache table for precomputed covariance matrices and X snapshots
MB_COV_CACHE_TBL = 'mvo_cov_cache'


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

    # Factor-driven
    Sigma_factor_raw, _, _ = _mvo_factor_cov(
        valid, Pxs_df, sectors_s, volumeTrd_df, model_version
    )
    emp_mean_var    = np.diag(Sigma_emp).mean()
    factor_mean_var = np.diag(Sigma_factor_raw).mean()
    if factor_mean_var > 1e-12:
        Sigma_factor = Sigma_factor_raw * (emp_mean_var / factor_mean_var)
    else:
        Sigma_factor = Sigma_emp.copy()

    # PCA
    Sigma_pca, _, _, _ = _mvo_pca_cov(ret_df.values,
                                        var_threshold=pca_var_threshold)

    # Ensemble
    Sigma_ens = (Sigma_emp + Sigma_lw + Sigma_factor + Sigma_pca) / 4.0

    return valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens


# ===============================================================================
# X SNAPSHOT CACHE (monthly, same as portfolio_risk_decomp)
# ===============================================================================

def _mb_build_x_snapshots(rebal_dates, Pxs_df, sectors_s,
                            volumeTrd_df, model_version,
                            force_rebuild):
    """
    Build X exposure matrix on month-end dates + latest date.
    Returns dict {date: DataFrame(universe x factors)}.
    Rebuilds on every call — no caching needed since X build is
    the correct behavior and notebook-level caching is fragile.
    """
    F_mat, factor_names, sec_cols = _rd_build_F(model_version=model_version)
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    F_mat, factor_names, sec_cols = _rd_build_F(model_version=model_version)
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)

    x_snap_dates = _rd_month_end_dates(
        pd.DatetimeIndex(rebal_dates), rebal_dates[-1]
    )

    print(f"  Building X snapshots on {len(x_snap_dates)} month-end dates...")
    X_snapshots = {}
    for i, xdt in enumerate(x_snap_dates):
        pxs_to_dt = Pxs_df.loc[:xdt]
        if len(pxs_to_dt) < BETA_WINDOW // 2:
            continue
        try:
            xdf = _rd_build_X(
                universe, factor_names, sec_cols,
                pxs_to_dt, sectors_s, volumeTrd_df,
                model_version=model_version
            ).fillna(0.0)
            X_snapshots[xdt] = xdf
            if (i+1) % 4 == 0:
                print(f"    {i+1}/{len(x_snap_dates)} done", end='\r')
        except Exception as e:
            warnings.warn(f"  X build failed for {xdt.date()}: {e}")

    print(f"\n  X snapshots built: {len(X_snapshots)}")
    return X_snapshots, universe, factor_names, sec_cols


# ===============================================================================
# MVO WEIGHT SOLVER (per rebalance date)
# ===============================================================================

def _mb_solve_mvo(dt, candidates, composite_scores, Pxs_df, sectors_s,
                   volumeTrd_df, model_version, pca_var_threshold,
                   ic, max_weight, min_weight, zscore_cap,
                   min_matrix_count, risk_aversion,
                   X_snapshots=None, snapshot_dates=None,
                   top_n=20):
    """
    Solve MVO for a single rebalance date.
    Returns pd.Series of weights indexed by tickers (zeros for non-selected).
    """
    # Get cached X snapshot (forward-fill from nearest month-end)
    X_df_cached = None
    if X_snapshots is not None and snapshot_dates is not None:
        valid_snaps = [d for d in snapshot_dates if d <= dt]
        if valid_snaps:
            X_df_cached = X_snapshots[valid_snaps[-1]]

    # Filter candidates to those with sufficient return history
    ret_df = (Pxs_df[candidates].pct_change()
              .dropna(how='all')
              .iloc[-MVO_LOOKBACK:]
              .dropna(axis=1, how='any'))
    valid = [t for t in candidates if t in ret_df.columns]
    if len(valid) < 5:
        print(f"  {dt.date()} SKIP: only {len(valid)} valid candidates")
        return pd.Series(dtype=float)

    # Build covariance matrices
    try:
        valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens = \
            _mb_build_cov_matrices(
                dt, valid, Pxs_df, sectors_s,
                volumeTrd_df, model_version, pca_var_threshold,
                X_df_cached=X_df_cached
            )
    except Exception as e:
        import traceback
        print(f"  {dt.date()} COV MATRIX FAILED: {e}")
        traceback.print_exc()
        return pd.Series(dtype=float)

    matrices = {
        'Empirical'    : Sigma_emp,
        'Ledoit-Wolf'  : Sigma_lw,
        'Factor-driven': Sigma_factor,
        'PCA'          : Sigma_pca,
    }

    # Alpha signal (Grinold-Kahn, daily units)
    vol_daily = _mvo_ewma_vol(ret_df[valid], MVO_EWMA_HL)
    z_s       = composite_scores.reindex(valid).fillna(0.0)
    z_capped  = z_s.clip(-zscore_cap, zscore_cap)
    alpha     = (ic * vol_daily * z_capped).fillna(0.0)

    # Eligibility filter: use min_matrix_count only if enough stocks pass
    # If fewer than 2*top_n stocks qualify, fall back to full valid universe
    import cvxpy as cp
    top_n_by = {}
    for mname, Sigma in matrices.items():
        try:
            N_e = len(valid)
            w_e = cp.Variable(N_e)
            a_e = alpha.reindex(valid).fillna(0.0).values
            p_e = cp.Problem(
                cp.Maximize(a_e @ w_e -
                            (risk_aversion / 2) * cp.quad_form(w_e, Sigma)),
                [cp.sum(w_e) == 1, w_e >= 0]
            )
            p_e.solve(solver=cp.OSQP, verbose=False)
            if (p_e.status in ['optimal', 'optimal_inaccurate'] and
                    w_e.value is not None):
                w_s = pd.Series(np.maximum(w_e.value, 0.0), index=valid)
                top_n_by[mname] = set(w_s[w_s > 1e-4].index.tolist())
            else:
                top_n_by[mname] = set(
                    alpha.reindex(valid).nlargest(top_n).index.tolist()
                )
        except Exception:
            top_n_by[mname] = set(
                alpha.reindex(valid).nlargest(top_n).index.tolist()
            )

    eligible_filtered = [t for t in valid
                         if sum(1 for m in top_n_by
                                if t in top_n_by[m]) >= min_matrix_count]

    # Use filtered universe only if it has at least 2*top_n stocks
    if len(eligible_filtered) >= 2 * top_n:
        eligible = eligible_filtered
        print(f"  {dt.date()} eligible={len(eligible)} (filtered) valid={len(valid)}")
    else:
        eligible = valid
        print(f"  {dt.date()} eligible={len(eligible)} (full, filter={len(eligible_filtered)}<{2*top_n}) valid={len(valid)}")

    # ── MVO with adaptive cap to enforce exactly top_n stocks ────────────────
    # Key insight: solver with BOTH floor >= min_weight AND cap <= C guarantees
    # at least ceil(1/C) stocks. Setting C = 1/top_n guarantees exactly top_n.
    # We start at max_weight and reduce until we get top_n non-zero stocks.
    # At cap = 1/top_n the problem is always feasible with exactly top_n stocks.
    alpha_el  = alpha.reindex(eligible).fillna(0.0)
    elig_idx  = [valid.index(t) for t in eligible]
    S_ens_el  = Sigma_ens[np.ix_(elig_idx, elig_idx)]
    a_el      = alpha_el.values
    N         = len(eligible)

    import cvxpy as cp

    # Cap schedule: try max_weight first, then step down to 1/top_n
    guaranteed_cap = 1.0 / max(top_n, 1)
    caps_to_try    = []
    c = max_weight
    while c > guaranteed_cap + 1e-6:
        caps_to_try.append(round(c, 6))
        c -= (max_weight - guaranteed_cap) / 8.0
    caps_to_try.append(guaranteed_cap)  # always end with guaranteed cap

    w_out    = None
    used_cap = guaranteed_cap

    for cap in caps_to_try:
        try:
            w_var = cp.Variable(N)
            prob  = cp.Problem(
                cp.Maximize(a_el @ w_var -
                            (risk_aversion / 2) * cp.quad_form(w_var, S_ens_el)),
                [cp.sum(w_var) == 1,
                 w_var >= 0,
                 w_var <= cap]          # cap only — no floor here
            )
            prob.solve(solver=cp.OSQP, verbose=False,
                       eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)
            if prob.status not in ['optimal', 'optimal_inaccurate']:
                prob.solve(solver=cp.SCS, verbose=False)
            if (prob.status in ['optimal', 'optimal_inaccurate']
                    and w_var.value is not None):
                w_sol     = pd.Series(np.maximum(w_var.value, 0.0), index=eligible)
                n_nonzero = (w_sol > 1e-4).sum()
                used_cap  = cap
                w_out     = w_sol
                print(f"  cap={cap:.4f} n_nonzero={n_nonzero} target={top_n}")
                if n_nonzero >= top_n:
                    break   # achieved target — use this solution
            else:
                print(f"  cap={cap:.4f} SOLVER STATUS: {prob.status}")
        except Exception as e:
            print(f"  cap={cap:.4f} SOLVER EXCEPTION: {e}")
            continue

    if w_out is None:
        w_out = pd.Series(1.0 / top_n,
                          index=alpha_el.nlargest(top_n).index)

    # ── Apply floor then cap post-solve ───────────────────────────────────────
    # Keep only stocks with meaningful weight, apply floor, renorm, then cap
    w_nz  = w_out[w_out > 1e-4]
    w_out = _mb_floor_then_cap(w_nz, min_weight=min_weight,
                                max_weight=used_cap)
    if w_out.sum() > 0:
        w_out = w_out / w_out.sum()
    return w_out


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

def _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s):
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
    nb = nav_baseline.reindex(common)
    na = nav_alpha.reindex(common)
    nm = nav_mvo.reindex(common)

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
    for nav_s, lbl, color, lw in [
        (nb, 'Baseline (quality)', GRAY, 1.2),
        (na, 'Pure alpha',         TEAL, 1.5),
        (nm, 'MVO',                BLUE, 1.8),
    ]:
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
    for nav_s, lbl, color in [
        (na, 'Pure alpha vs baseline', TEAL),
        (nm, 'MVO vs baseline',        BLUE),
    ]:
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
    for nav_s, lbl, color in [
        (nb, 'Baseline', GRAY),
        (na, 'Pure alpha', TEAL),
        (nm, 'MVO', BLUE),
    ]:
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
                     min_matrix_count=2,
                     pca_var_threshold=0.65,
                     universe_mult=5,
                     risk_aversion=1.0,
                     force_rebuild_cache=True):
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
    min_matrix_count   : int   — eligibility filter (default 2)
    pca_var_threshold  : float — PCA variance explained threshold (default 0.65)
    universe_mult      : int   — candidate pool = port_n × universe_mult (default 5)
    risk_aversion      : float — MVO risk aversion λ (default 1.0)
    """
    print("=" * 72)
    print("  MVO BACKTEST — Baseline vs Pure Alpha vs MVO")
    print("=" * 72)
    print(f"\n  MVO params: IC={ic}, max_w={max_weight}, min_w={min_weight}, "
          f"zscore_cap={zscore_cap}")
    print(f"  PCA thresh={pca_var_threshold}, min_matrix={min_matrix_count}, "
          f"universe_mult={universe_mult}, risk_aversion={risk_aversion}")
    print(f"  force_rebuild_cache={force_rebuild_cache}\n")

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

    n_cands = top_n * universe_mult
    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"sector_cap={sector_cap}, mktcap_floor={mktcap_input or 'none'}, "
          f"vol_filter={use_vol}, prefilt={prefilt_pct:.0%}, "
          f"conc={conc_factor:.1f}x (pure alpha only)")
    print(f"  MVO candidate pool: {n_cands} stocks ({universe_mult}×{top_n})\n")

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
    print(f"  snapshot_dates: {len(snapshot_dates)} dates, first={snapshot_dates[0].date() if snapshot_dates else None}, last={snapshot_dates[-1].date() if snapshot_dates else None}")

    # ── [3/4] Compute portfolio weights per rebalance date ────────────────────
    print("\n[3/4] Computing portfolio weights...")

    alpha_weights_by_date = {}   # pure alpha (equal/concentration)
    mvo_weights_by_date   = {}   # MVO
    quality_factor_by_date = {}  # baseline

    # Load quality scores for baseline
    all_tickers  = list(sectors_s.index)
    quality_wide = get_quality_scores(
        calc_dates         = calc_dates_idx,
        universe           = all_tickers,
        Pxs_df             = Pxs_df,
        sectors_s          = sectors_s,
        use_cached_weights = True,
        force_recompute    = False,
    )

    n = len(calc_dates)
    for i, dt in enumerate(calc_dates):
        if (i+1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n}] {dt.date()}", end='\r')

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
            w_mvo = _mb_solve_mvo(
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
                    min_matrix_count = min_matrix_count,
                    risk_aversion = risk_aversion,
                    X_snapshots   = X_snapshots,
                    snapshot_dates = snapshot_dates,
                    top_n         = top_n,
                )
        except Exception as e:
            print(f"  {dt.date()} MVO ERROR: {e}")
            w_mvo = pd.Series(dtype=float)

        if not w_mvo.empty and w_mvo.sum() > 0:
            w_mvo_nz = w_mvo[w_mvo > 1e-6]
            mvo_weights_by_date[dt] = w_mvo_nz
            n_mvo  = len(w_mvo_nz)
            top_w  = w_mvo_nz.max()
            min_w  = w_mvo_nz.min()
            eff_n  = 1.0 / (w_mvo_nz**2).sum() if n_mvo > 0 else 0
            print(f"  {dt.date()}  MVO: n={n_mvo:3d}  "
                  f"min={min_w:.1%}  max={top_w:.1%}  eff_N={eff_n:.1f}")

    print(f"\n  Weights computed: "
          f"alpha={len(alpha_weights_by_date)}, "
          f"mvo={len(mvo_weights_by_date)}, "
          f"baseline={len(quality_factor_by_date)}")

    # ── [4/4] NAV series ──────────────────────────────────────────────────────
    print("\n[4/4] Computing NAV series...")

    nav_alpha    = _mb_run_nav(alpha_weights_by_date, calc_dates, Pxs_df)
    nav_mvo      = _mb_run_nav(mvo_weights_by_date,   calc_dates, Pxs_df)
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'='*72}")
    print(f"  COMPARISON")
    print(f"  {'='*72}")
    print(f"  {'Strategy':<30} {'CAGR':>8} {'Vol':>8} "
          f"{'Sharpe':>8} {'MDD':>8}")
    print(f"  {'-'*58}")
    for nav_s, lbl in [(nav_baseline, 'Baseline (quality)'),
                        (nav_alpha,    f'Pure Alpha (conc={conc_factor:.1f}x)'),
                        (nav_mvo,      f'MVO (IC={ic}, max={max_weight:.0%})')]:
        n_yrs  = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr   = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1/n_yrs) - 1
        vol    = nav_s.pct_change().dropna().std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        mdd    = ((nav_s / nav_s.cummax()) - 1).min()
        print(f"  {lbl:<30} {cagr*100:>7.1f}% {vol*100:>7.1f}% "
              f"{sharpe:>8.2f} {mdd*100:>7.1f}%")

    # Yearly returns
    print(f"\n  Yearly returns:")
    print(f"  {'Year':<6} {'Baseline':>10} {'Pure Alpha':>12} {'MVO':>10}")
    print(f"  {'-'*42}")
    for yr in sorted(set(nav_baseline.index.year)):
        def yr_ret(nav_s):
            yr_nav = nav_s[nav_s.index.year == yr]
            if len(yr_nav) < 2:
                return np.nan
            return (yr_nav.iloc[-1] / yr_nav.iloc[0] - 1) * 100
        print(f"  {yr:<6} {yr_ret(nav_baseline):>+9.2f}%  "
              f"{yr_ret(nav_alpha):>+10.2f}%  "
              f"{yr_ret(nav_mvo):>+9.2f}%")

    print(f"\n  port_baseline : {len(port_baseline)} dates x {top_n} stocks")
    print(f"  port_alpha    : {len(port_alpha)} dates x {top_n} stocks")
    print(f"  port_mvo      : {len(port_mvo)} dates x {top_n} stocks")

    _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s)

    return {
        'nav_baseline'         : nav_baseline,
        'nav_alpha'            : nav_alpha,
        'nav_mvo'              : nav_mvo,
        'port_baseline'        : port_baseline,
        'port_alpha'           : port_alpha,
        'port_mvo'             : port_mvo,
        'alpha_weights_by_date': alpha_weights_by_date,
        'mvo_weights_by_date'  : mvo_weights_by_date,
        'composite_by_date'    : composite_by_date,
        'regime_s'             : regime_s,
        'weights_by_year'      : weights_by_year,
    }
