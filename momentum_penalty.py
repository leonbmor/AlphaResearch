"""
momentum_penalty.py
===================
Analyses cross-sectional return distributions across non-overlapping windows
(1M, 2M, 3M, 6M) to support design of an exponential alpha-signal penalty
for high-flying stocks.

METHODOLOGY
-----------
For each rebalancing date and each window size W:

1. Build 1Y pooled distribution:
   - Collect all non-overlapping W-period returns for every stock
     over the 1 year prior to the rebalancing date
   - Pool cross-sectionally → one distribution per (date, window)
   - Compute mean, std, skewness, kurtosis

2. Z-score each stock's current W-period return using 1Y mean/std

3. Rescale using 5Y moments:
   - Un-z-score with 5Y mean/std  → mean/std rescaling
   - Un-z-score with 5Y median/IQR → robust rescaling (alternative)

4. Output percentile levels (75, 80, 85, 90, 95, 97.5) of the
   rescaled cross-sectional distribution at each rebalancing date.

USAGE
-----
    from momentum_penalty import run_momentum_penalty_analysis

    pct_df = run_momentum_penalty_analysis(
        Pxs_df       = Pxs_df,
        rebal_dates  = rebal_dates,   # list of pd.Timestamp rebalancing dates
        universe     = universe,      # list of ticker strings
    )

    # pct_df has one row per rebalancing date, columns:
    #   p75_1M, p80_1M, ..., p97.5_1M,
    #   p75_2M, p80_2M, ..., p97.5_2M,
    #   p75_3M, p80_3M, ..., p97.5_3M,
    #   p75_6M, p80_6M, ..., p97.5_6M
    # plus moment columns for diagnostics:
    #   mean_1Y_1M, std_1Y_1M, skew_1Y_1M, kurt_1Y_1M  (per window)

PARAMETERS
----------
    LOOKBACK_1Y    = 252    trading days for short-term distribution
    LOOKBACK_5Y    = 1260   trading days for long-run rescaling
    WINDOWS_TD     = {      approximate trading-day lengths per window label
        '1M': 21, '2M': 42, '3M': 63, '6M': 126
    }
    PERCENTILES    = [75, 80, 85, 90, 95, 97.5]
    MIN_OBS        = 30     minimum pooled observations to compute percentiles
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── Parameters ────────────────────────────────────────────────────────────────
LOOKBACK_1Y  = 252     # trading days
LOOKBACK_5Y  = 1260    # trading days
WINDOWS_TD   = {'1M': 21, '2M': 42, '3M': 63}
PERCENTILES  = [95, 96, 97, 98, 99]
MIN_OBS      = 30      # minimum pooled observations required


# ================================================================================
# HELPERS
# ================================================================================

def _non_overlapping_returns(px_series: pd.Series,
                              window_td: int,
                              end_idx:   int,
                              n_windows: int) -> np.ndarray:
    """
    Compute n_windows non-overlapping returns of length window_td
    ending at end_idx (inclusive) in px_series.

    Returns array of length <= n_windows (fewer if history is insufficient).
    """
    rets = []
    for k in range(n_windows):
        i_end   = end_idx - k * window_td
        i_start = i_end - window_td
        if i_start < 0:
            break
        p_end   = px_series.iloc[i_end]
        p_start = px_series.iloc[i_start]
        if p_start > 0 and not np.isnan(p_start) and not np.isnan(p_end):
            rets.append(p_end / p_start - 1)
    return np.array(rets)


def _pool_cross_sectional(Pxs_df:    pd.DataFrame,
                           universe:  list,
                           date_idx:  int,
                           window_td: int,
                           lookback:  int) -> np.ndarray:
    """
    For a given rebalancing date (by integer index into Pxs_df.index),
    pool non-overlapping W-period returns across all stocks over `lookback`
    trading days prior.

    Returns 1D array of all pooled returns.
    """
    n_windows = lookback // window_td
    all_rets  = []
    for tkr in universe:
        if tkr not in Pxs_df.columns:
            continue
        px = Pxs_df[tkr].iloc[max(0, date_idx - lookback): date_idx + 1].dropna()
        if len(px) < window_td + 1:
            continue
        # Re-index within the slice
        px_arr = px.values
        n_end  = len(px_arr) - 1
        tkr_rets = _non_overlapping_returns(
            pd.Series(px_arr), window_td, n_end,
            min(n_windows, n_end // window_td)
        )
        all_rets.extend(tkr_rets.tolist())
    return np.array(all_rets)


def _moments(arr: np.ndarray) -> dict:
    """Compute mean, std, median, IQR, skewness, kurtosis."""
    if len(arr) < 4:
        return dict(mean=np.nan, std=np.nan, median=np.nan,
                    iqr=np.nan, skew=np.nan, kurt=np.nan)
    return dict(
        mean   = float(np.mean(arr)),
        std    = float(np.std(arr, ddof=1)),
        median = float(np.median(arr)),
        iqr    = float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        skew   = float(stats.skew(arr)),
        kurt   = float(stats.kurtosis(arr)),   # excess kurtosis
    )


def _rescale(z_scores:    np.ndarray,
             moments_5y:  dict,
             use_robust:  bool = False) -> np.ndarray:
    """
    Un-z-score using 5Y moments.
    use_robust=False: use mean/std
    use_robust=True:  use median/IQR (more robust to fat tails)
    """
    if use_robust:
        loc   = moments_5y.get('median', 0.0) or 0.0
        scale = moments_5y.get('iqr',    1.0) or 1.0
    else:
        loc   = moments_5y.get('mean', 0.0) or 0.0
        scale = moments_5y.get('std',  1.0) or 1.0
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    return z_scores * scale + loc


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def run_momentum_penalty_analysis(Pxs_df:      pd.DataFrame,
                                   rebal_dates: list,
                                   universe:    list,
                                   lookback_1y: int  = LOOKBACK_1Y,
                                   lookback_5y: int  = LOOKBACK_5Y,
                                   windows_td:  dict = None,
                                   percentiles: list = None,
                                   min_obs:     int  = MIN_OBS,
                                   use_robust:  bool = False) -> pd.DataFrame:
    """
    Compute cross-sectional return distribution percentiles at each rebalancing
    date for non-overlapping windows of 1M, 2M, 3M, 6M.

    Parameters
    ----------
    Pxs_df      : price DataFrame (dates × tickers)
    rebal_dates : list of pd.Timestamp rebalancing dates
    universe    : list of tickers to include
    lookback_1y : trading days for short-term distribution (default 252)
    lookback_5y : trading days for long-run rescaling (default 1260)
    windows_td  : dict of {'label': n_trading_days} (default WINDOWS_TD)
    percentiles : list of percentile levels (default PERCENTILES)
    min_obs     : minimum observations to compute stats (default 30)
    use_robust  : use median/IQR for 5Y rescaling instead of mean/std

    Returns
    -------
    pd.DataFrame with index = rebal_dates and columns:
        p{pct}_{window}      — rescaled return percentiles
        mean_1Y_{window}     — 1Y pooled mean (diagnostic)
        std_1Y_{window}      — 1Y pooled std
        skew_1Y_{window}     — 1Y pooled skewness
        kurt_1Y_{window}     — 1Y pooled excess kurtosis
        mean_5Y_{window}     — 5Y pooled mean
        std_5Y_{window}      — 5Y pooled std
        median_5Y_{window}   — 5Y pooled median
        iqr_5Y_{window}      — 5Y pooled IQR
        n_obs_1Y_{window}    — number of pooled observations (1Y)
        n_obs_5Y_{window}    — number of pooled observations (5Y)
    """
    if windows_td is None:
        windows_td = WINDOWS_TD
    if percentiles is None:
        percentiles = PERCENTILES

    universe    = [t for t in universe if t in Pxs_df.columns]
    rebal_dates = sorted([pd.Timestamp(d) for d in rebal_dates])
    px_dates    = Pxs_df.index

    # Build column list
    pct_cols  = [f"p{p}_{w}"      for w in windows_td for p in percentiles]
    diag_cols = [f"{m}_{yr}_{w}"
                 for w in windows_td
                 for yr in ['1Y', '5Y']
                 for m in (['mean','std','skew','kurt'] if yr=='1Y'
                           else ['mean','std','median','iqr'])]
    obs_cols  = [f"n_obs_{yr}_{w}" for w in windows_td for yr in ['1Y','5Y']]

    records = []

    print(f"  Running momentum penalty analysis...")
    print(f"  Universe: {len(universe)} stocks  |  "
          f"Rebal dates: {len(rebal_dates)}")
    print(f"  Windows: {list(windows_td.keys())}  |  "
          f"Percentiles: {percentiles}")
    print(f"  Lookback: 1Y={lookback_1y}d  5Y={lookback_5y}d  |  "
          f"Rescaling: {'robust (median/IQR)' if use_robust else 'mean/std'}\n")

    for i, dt in enumerate(rebal_dates):
        if dt not in px_dates:
            # Find nearest prior date
            prior = px_dates[px_dates <= dt]
            if prior.empty:
                continue
            dt_px = prior[-1]
        else:
            dt_px = dt

        dt_idx = px_dates.get_loc(dt_px)
        row    = {'date': dt}

        for wlbl, wtd in windows_td.items():

            # ── Pool 1Y distribution ────────────────────────────────────────
            pool_1y = _pool_cross_sectional(
                Pxs_df, universe, dt_idx, wtd, lookback_1y)
            m1y     = _moments(pool_1y)
            row[f'n_obs_1Y_{wlbl}'] = len(pool_1y)
            for k, v in m1y.items():
                if k in ('mean', 'std', 'skew', 'kurt'):
                    row[f'{k}_1Y_{wlbl}'] = v

            # ── Pool 5Y distribution ────────────────────────────────────────
            pool_5y = _pool_cross_sectional(
                Pxs_df, universe, dt_idx, wtd, lookback_5y)
            m5y     = _moments(pool_5y)
            row[f'n_obs_5Y_{wlbl}'] = len(pool_5y)
            for k, v in m5y.items():
                if k in ('mean', 'std', 'median', 'iqr'):
                    row[f'{k}_5Y_{wlbl}'] = v

            if len(pool_1y) < min_obs:
                for p in percentiles:
                    row[f'p{p}_{wlbl}'] = np.nan
                continue

            # ── Z-score 1Y returns using 1Y moments ─────────────────────────
            std_1y = m1y['std']
            if std_1y is None or std_1y == 0 or np.isnan(std_1y):
                std_1y = 1.0
            mean_1y = m1y['mean'] or 0.0
            z_scores = (pool_1y - mean_1y) / std_1y

            # ── Rescale using 5Y moments ─────────────────────────────────────
            if len(pool_5y) >= min_obs:
                rescaled = _rescale(z_scores, m5y, use_robust=use_robust)
            else:
                rescaled = pool_1y   # fallback: use raw if 5Y insufficient

            # ── Compute percentiles of rescaled distribution ─────────────────
            for p in percentiles:
                row[f'p{p}_{wlbl}'] = float(np.percentile(rescaled, p))

        records.append(row)

        if (i + 1) % 20 == 0 or i == len(rebal_dates) - 1:
            print(f"  [{i+1}/{len(rebal_dates)}] {dt.date()}  "
                  f"  1M: n_obs_1Y={row.get('n_obs_1Y_1M','?')}  "
                  f"p95={row.get('p95_1M', np.nan):.3f}  "
                  f"p95_3M={row.get('p95_3M', np.nan):.3f}", end='\r')

    print()  # newline after progress

    df = pd.DataFrame(records).set_index('date')

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  MOMENTUM PENALTY DISTRIBUTION SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Cross-sectional percentiles of rescaled returns "
          f"({'robust' if use_robust else 'mean/std'} rescaling)\n")

    for wlbl in windows_td:
        pct_cols_w = [f'p{p}_{wlbl}' for p in percentiles
                      if f'p{p}_{wlbl}' in df.columns]
        if not pct_cols_w:
            continue
        sub = df[pct_cols_w].dropna()
        if sub.empty:
            continue
        print(f"  Window: {wlbl}  (n_obs_1Y avg: "
              f"{df[f'n_obs_1Y_{wlbl}'].mean():.0f}  "
              f"n_obs_5Y avg: {df[f'n_obs_5Y_{wlbl}'].mean():.0f})")
        print(f"  {'Percentile':<12}  {'Min':>8}  {'Median':>8}  "
              f"{'Mean':>8}  {'Max':>8}")
        print(f"  {'-'*52}")
        for col in pct_cols_w:
            p_lbl = col.split('_')[0]
            print(f"  {p_lbl:<12}  {sub[col].min():>8.3f}  "
                  f"{sub[col].median():>8.3f}  "
                  f"{sub[col].mean():>8.3f}  "
                  f"{sub[col].max():>8.3f}")
        print()

        # Also show avg skew and kurtosis for reference
        skew_col = f'skew_1Y_{wlbl}'
        kurt_col = f'kurt_1Y_{wlbl}'
        if skew_col in df.columns:
            print(f"  1Y distribution moments (avg across dates):")
            print(f"    skewness={df[skew_col].mean():.3f}  "
                  f"excess_kurtosis={df[kurt_col].mean():.3f}")
        print()

    return df


# ================================================================================
# K CALIBRATION
# ================================================================================

def _compute_rescaled_return(px_series:  pd.Series,
                              dt_idx:     int,
                              window_td:  int,
                              moments_1y: dict,
                              moments_5y: dict,
                              use_robust: bool = False) -> float:
    """
    Compute a single stock's rescaled return over window_td ending at dt_idx.
    Returns nan if insufficient history.
    """
    if dt_idx < window_td:
        return np.nan
    p_end   = px_series.iloc[dt_idx]
    p_start = px_series.iloc[dt_idx - window_td]
    if p_start <= 0 or np.isnan(p_start) or np.isnan(p_end):
        return np.nan
    raw_ret = p_end / p_start - 1

    # Z-score with 1Y moments
    std_1y  = moments_1y.get('std',  1.0) or 1.0
    mean_1y = moments_1y.get('mean', 0.0) or 0.0
    if np.isnan(std_1y) or std_1y == 0:
        std_1y = 1.0
    z = (raw_ret - mean_1y) / std_1y

    # Rescale with 5Y moments
    if use_robust:
        scale = moments_5y.get('iqr',    1.0) or 1.0
        loc   = moments_5y.get('median', 0.0) or 0.0
    else:
        scale = moments_5y.get('std',  1.0) or 1.0
        loc   = moments_5y.get('mean', 0.0) or 0.0
    if np.isnan(scale) or scale == 0:
        scale = 1.0
    return float(z * scale + loc)


def _rank_ic(scores: np.ndarray, ground_truth: np.ndarray) -> float:
    """Spearman rank IC between scores and ground truth returns."""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(scores) | np.isnan(ground_truth))
    if mask.sum() < 10:
        return np.nan
    r, _ = spearmanr(scores[mask], ground_truth[mask])
    return float(r)


def _apply_penalty(alpha_z:      np.ndarray,
                   rescaled_rets: np.ndarray,
                   p99_threshold: float,
                   k:             float) -> np.ndarray:
    """
    Apply exponential penalty to alpha z-scores.
    penalty_i = exp(-k * max(0, rescaled_ret_i - p99_threshold)) - 1, floored at 0
    adjusted_z_i = alpha_z_i * exp(-k * max(0, rescaled_ret_i - p99_threshold))
    """
    excess  = np.maximum(0.0, rescaled_rets - p99_threshold)
    penalty = np.exp(-k * excess)
    return alpha_z * penalty



def calibrate_k(Pxs_df,
                composite_by_date,
                rebal_dates,
                universe,
                pct_df,
                validation_days=21,
                k_grid=None,
                k_max=10,
                threshold_pcts=None,
                windows_td=None,
                use_robust=False,
                min_stocks=20,
                min_ic_improvement=0.005):
    """
    Calibrate (threshold_percentile, k) jointly for momentum penalty.
    For each rebalancing date independently, finds (pct*, k*) that
    maximises rank IC improvement over the baseline alpha signal by
    >= min_ic_improvement. If no improvement found, k*=0.
    """
    from scipy.stats import spearmanr, ttest_ind

    if k_grid is None:
        # Build grid up to k_max
        if k_max <= 1.0:
            k_grid = list(np.linspace(0.0, k_max, 21))
        elif k_max <= 10.0:
            k_grid = list(np.concatenate([
                np.linspace(0.0, 1.0, 21),
                np.linspace(1.0, k_max, 19)[1:],
            ]))
        else:
            k_grid = list(np.concatenate([
                np.linspace(0.0, 1.0, 21),
                np.linspace(1.0, 10.0, 19)[1:],
                np.linspace(10.0, k_max, 9)[1:],
            ]))
    # Always enforce k_max cap
    k_grid = [k for k in k_grid if k <= k_max + 1e-9]
    if threshold_pcts is None:
        threshold_pcts = [97, 99]
    if windows_td is None:
        windows_td = WINDOWS_TD

    universe    = [t for t in universe if t in Pxs_df.columns]
    rebal_dates = sorted([pd.Timestamp(d) for d in rebal_dates])
    px_dates    = Pxs_df.index

    print("\n" + "="*72)
    print("  MOMENTUM PENALTY CALIBRATION  (joint threshold x k)")
    print("="*72)
    print(f"\n  Universe: {len(universe)} stocks  |  Rebal dates: {len(rebal_dates)}")
    print(f"  Validation window: {validation_days}d  |  "
          f"k grid: {len(k_grid)} values [{min(k_grid):.2f} -> {max(k_grid):.2f}]")
    print(f"  Threshold candidates: p{threshold_pcts}  |  "
          f"Windows: {list(windows_td.keys())}  |  "
          f"k_max: {k_max}  |  "
          f"min_IC_improvement: {min_ic_improvement:.4f}\n")

    records = []

    for i, rebal_dt in enumerate(rebal_dates):

        prior_px = px_dates[px_dates < rebal_dt]
        if len(prior_px) < validation_days + 1:
            continue
        signal_dt  = prior_px[-validation_days]
        signal_idx = px_dates.get_loc(signal_dt)
        rebal_idx  = px_dates.get_loc(prior_px[-1])

        avail_comp = {d: v for d, v in composite_by_date.items() if d <= signal_dt}
        if not avail_comp:
            continue
        alpha_s = avail_comp[max(avail_comp.keys())]
        stocks  = [t for t in universe
                   if t in alpha_s.index and not np.isnan(alpha_s[t])
                   and t in Pxs_df.columns]
        if len(stocks) < min_stocks:
            continue

        fwd_rets = {}
        for tkr in stocks:
            p_s = Pxs_df[tkr].iloc[signal_idx]
            p_e = Pxs_df[tkr].iloc[rebal_idx]
            if p_s > 0 and not np.isnan(p_s) and not np.isnan(p_e):
                fwd_rets[tkr] = p_e / p_s - 1
        stocks = [t for t in stocks if t in fwd_rets]
        if len(stocks) < min_stocks:
            continue

        alpha_arr = np.array([float(alpha_s[t]) for t in stocks])
        fwd_arr   = np.array([fwd_rets[t] for t in stocks])

        mask_base = ~(np.isnan(alpha_arr) | np.isnan(fwd_arr))
        if mask_base.sum() < 10:
            continue
        ic_base = float(spearmanr(alpha_arr[mask_base], fwd_arr[mask_base])[0])
        if np.isnan(ic_base):
            continue

        avail_pct = pct_df[pct_df.index <= signal_dt]
        if avail_pct.empty:
            continue
        pct_row = avail_pct.iloc[-1]

        # Rescaled returns per window at signal_dt
        rescaled_by_window = {}
        for wlbl, wtd in windows_td.items():
            m1y = {'mean': pct_row.get(f'mean_1Y_{wlbl}', np.nan),
                   'std':  pct_row.get(f'std_1Y_{wlbl}',  np.nan)}
            m5y = {'mean':   pct_row.get(f'mean_5Y_{wlbl}',   np.nan),
                   'std':    pct_row.get(f'std_5Y_{wlbl}',    np.nan),
                   'median': pct_row.get(f'median_5Y_{wlbl}', np.nan),
                   'iqr':    pct_row.get(f'iqr_5Y_{wlbl}',    np.nan)}
            rescaled_by_window[wlbl] = np.array([
                _compute_rescaled_return(
                    Pxs_df[tkr], signal_idx, wtd, m1y, m5y, use_robust)
                for tkr in stocks
            ])

        # Joint grid search
        best_k     = 0.0
        best_pct   = threshold_pcts[-1]
        best_ic    = ic_base
        best_n_pen = 0
        dom_window = 'none'

        for pct in threshold_pcts:
            pct_vals = {wlbl: pct_row.get(f'p{pct}_{wlbl}', np.nan)
                        for wlbl in windows_td}
            for k in k_grid:
                if k == 0:
                    ic_k = ic_base
                else:
                    penalties = np.zeros(len(stocks))
                    for wlbl in windows_td:
                        thr = pct_vals.get(wlbl, np.nan)
                        if np.isnan(thr):
                            continue
                        rr   = rescaled_by_window[wlbl]
                        mask = ~np.isnan(rr)
                        w_pen = np.zeros(len(stocks))
                        w_pen[mask] = np.maximum(0.0, rr[mask] - thr)
                        penalties = np.maximum(penalties, w_pen)
                    adj_alpha = alpha_arr * np.exp(-k * penalties)
                    m = ~(np.isnan(adj_alpha) | np.isnan(fwd_arr))
                    if m.sum() < 10:
                        continue
                    ic_k = float(spearmanr(adj_alpha[m], fwd_arr[m])[0])

                if not np.isnan(ic_k) and (ic_k - ic_base) >= min_ic_improvement and ic_k > best_ic:
                    best_ic  = ic_k
                    best_k   = k
                    best_pct = pct
                    pen_mask   = np.zeros(len(stocks), dtype=bool)
                    dom_counts = {}
                    for wlbl in windows_td:
                        thr = pct_vals.get(wlbl, np.nan)
                        if np.isnan(thr):
                            continue
                        rr   = rescaled_by_window[wlbl]
                        mask = ~np.isnan(rr) & (rr > thr)
                        dom_counts[wlbl] = mask.sum()
                        pen_mask |= mask
                    best_n_pen = pen_mask.sum()
                    dom_window = max(dom_counts, key=dom_counts.get) if dom_counts else 'none'

        records.append({
            'date':            rebal_dt,
            'k_star':          best_k,
            'pct_star':        best_pct,
            'ic_baseline':     ic_base,
            'ic_penalized':    best_ic,
            'ic_improvement':  best_ic - ic_base,
            'n_stocks':        len(stocks),
            'n_penalized':     best_n_pen,
            'dominant_window': dom_window,
        })

        if (i + 1) % 10 == 0 or i == len(rebal_dates) - 1:
            print(f"  [{i+1}/{len(rebal_dates)}] {rebal_dt.date()}  "
                  f"pct*=p{best_pct}  k*={best_k:.2f}  "
                  f"IC_base={ic_base:+.3f}  IC_pen={best_ic:+.3f}  "
                  f"d={best_ic-ic_base:+.4f}  n_pen={best_n_pen}")

    df = pd.DataFrame(records).set_index('date')
    if df.empty:
        return df

    # ── Summary ───────────────────────────────────────────────────────────────
    n_adopted = (df['k_star'] > 0).sum()
    n_total   = len(df)
    print("\n" + "="*72)
    print("  CALIBRATION SUMMARY")
    print("="*72)
    print(f"\n  Dates penalty adopted (IC_improvement >= {min_ic_improvement:.4f}): "
          f"{n_adopted}/{n_total} ({n_adopted/n_total*100:.1f}%)")

    pct_counts = df.loc[df['k_star']>0, 'pct_star'].value_counts().sort_index()
    print("\n  Optimal threshold (pct*) breakdown:")
    for pct_v, cnt in pct_counts.items():
        print(f"    p{int(pct_v)}: {cnt} dates ({cnt/n_total*100:.1f}%)")

    k_pos = df.loc[df['k_star'] > 0, 'k_star']
    if not k_pos.empty:
        print(f"\n  k* distribution (when > 0, n={len(k_pos)}):")
        for p in [25, 50, 75, 90, 95]:
            print(f"    p{p:>3}: {np.percentile(k_pos, p):.2f}")

    print(f"\n  IC baseline    : median={df['ic_baseline'].median():+.3f}  "
          f"mean={df['ic_baseline'].mean():+.3f}")
    print(f"  IC penalized   : median={df['ic_penalized'].median():+.3f}  "
          f"mean={df['ic_penalized'].mean():+.3f}")
    print(f"  IC improvement : median={df['ic_improvement'].median():+.4f}  "
          f"mean={df['ic_improvement'].mean():+.4f}")
    print(f"  Avg n_penalized: {df.loc[df['k_star']>0,'n_penalized'].mean():.1f}")
    print(f"\n  Dominant window breakdown:")
    print(f"  {df['dominant_window'].value_counts().to_string()}")

    # ── Penalized stock diagnostic ────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  PENALIZED STOCK DIAGNOSTIC  (forward {validation_days}d returns)")
    print(f"  Dates where penalty was adopted (k* > 0)")
    print("="*72)

    SEP = "-" * 128
    print(f"\n  {'Date':<12}  {'k*':>6}  {'pct*':>5}  {'n_pen':>5}  "
          f"{'Penalized':^43}  {'Rest (universe)':^43}")
    print(f"  {'':12}  {'':6}  {'':5}  {'':5}  "
          f"{'min':>7}  {'p25':>7}  {'med':>7}  {'p75':>7}  {'max':>7}  {'mean':>7}  "
          f"{'min':>7}  {'p25':>7}  {'med':>7}  {'p75':>7}  {'max':>7}  {'mean':>7}")
    print("  " + SEP)

    pen_fwd_all  = []
    rest_fwd_all = []

    def _stats6(arr):
        if len(arr) == 0:
            return (np.nan,) * 6
        return (np.min(arr), np.percentile(arr, 25), np.median(arr),
                np.percentile(arr, 75), np.max(arr), np.mean(arr))

    for rebal_dt, row in df[df['k_star'] > 0].iterrows():
        best_k   = row['k_star']
        best_pct = int(row['pct_star'])

        prior_px = px_dates[px_dates < rebal_dt]
        if len(prior_px) < validation_days + 1:
            continue
        signal_dt  = prior_px[-validation_days]
        signal_idx = px_dates.get_loc(signal_dt)
        rebal_idx  = px_dates.get_loc(prior_px[-1])

        avail_comp = {d: v for d, v in composite_by_date.items() if d <= signal_dt}
        if not avail_comp:
            continue
        alpha_s = avail_comp[max(avail_comp.keys())]
        stocks  = [t for t in universe
                   if t in alpha_s.index and not np.isnan(alpha_s[t])
                   and t in Pxs_df.columns]
        fwd_rets = {}
        for tkr in stocks:
            p_s = Pxs_df[tkr].iloc[signal_idx]
            p_e = Pxs_df[tkr].iloc[rebal_idx]
            if p_s > 0 and not np.isnan(p_s) and not np.isnan(p_e):
                fwd_rets[tkr] = p_e / p_s - 1
        stocks = [t for t in stocks if t in fwd_rets]
        if not stocks:
            continue

        avail_pct = pct_df[pct_df.index <= signal_dt]
        if avail_pct.empty:
            continue
        pct_row = avail_pct.iloc[-1]

        pen_mask = np.zeros(len(stocks), dtype=bool)
        for wlbl, wtd in windows_td.items():
            m1y = {'mean': pct_row.get(f'mean_1Y_{wlbl}', np.nan),
                   'std':  pct_row.get(f'std_1Y_{wlbl}',  np.nan)}
            m5y = {'mean':   pct_row.get(f'mean_5Y_{wlbl}',   np.nan),
                   'std':    pct_row.get(f'std_5Y_{wlbl}',    np.nan),
                   'median': pct_row.get(f'median_5Y_{wlbl}', np.nan),
                   'iqr':    pct_row.get(f'iqr_5Y_{wlbl}',    np.nan)}
            thr = pct_row.get(f'p{best_pct}_{wlbl}', np.nan)
            if np.isnan(thr):
                continue
            rr = np.array([
                _compute_rescaled_return(
                    Pxs_df[tkr], signal_idx, wtd, m1y, m5y, use_robust)
                for tkr in stocks
            ])
            pen_mask |= (~np.isnan(rr)) & (rr > thr)

        fwd_arr     = np.array([fwd_rets[t] for t in stocks])
        pen_rets    = fwd_arr[pen_mask]
        rest_rets   = fwd_arr[~pen_mask]
        pen_tickers = [t for t, m in zip(stocks, pen_mask) if m]

        pen_fwd_all.extend(pen_rets.tolist())
        rest_fwd_all.extend(rest_rets.tolist())

        ps = _stats6(pen_rets)
        rs = _stats6(rest_rets)

        print(f"  {str(rebal_dt.date()):<12}  {best_k:>6.1f}  "
              f"p{best_pct:>2}  {len(pen_tickers):>5}  "
              f"{ps[0]:>+7.1%}  {ps[1]:>+7.1%}  {ps[2]:>+7.1%}  "
              f"{ps[3]:>+7.1%}  {ps[4]:>+7.1%}  {ps[5]:>+7.1%}  "
              f"{rs[0]:>+7.1%}  {rs[1]:>+7.1%}  {rs[2]:>+7.1%}  "
              f"{rs[3]:>+7.1%}  {rs[4]:>+7.1%}  {rs[5]:>+7.1%}")
        ticker_str = ', '.join(pen_tickers[:25]) + ('...' if len(pen_tickers) > 25 else '')
        print(f"  {'':12}  tickers: {ticker_str}")

    # ── Pooled stats ─────────────────────────────────────────────────────────
    if pen_fwd_all and rest_fwd_all:
        pa = np.array(pen_fwd_all)
        ra = np.array(rest_fwd_all)
        print("\n  " + "="*70)
        print("  POOLED ACROSS ALL PENALIZED DATES")
        print("  " + "="*70)
        print(f"  {'Group':<20}  {'n':>5}  "
              f"{'min':>7}  {'p25':>7}  {'med':>7}  {'p75':>7}  {'max':>7}  {'mean':>7}")
        print(f"  {'Penalized':<20}  {len(pa):>5}  "
              f"{np.min(pa):>+7.1%}  {np.percentile(pa,25):>+7.1%}  "
              f"{np.median(pa):>+7.1%}  {np.percentile(pa,75):>+7.1%}  "
              f"{np.max(pa):>+7.1%}  {np.mean(pa):>+7.1%}")
        print(f"  {'Rest (universe)':<20}  {len(ra):>5}  "
              f"{np.min(ra):>+7.1%}  {np.percentile(ra,25):>+7.1%}  "
              f"{np.median(ra):>+7.1%}  {np.percentile(ra,75):>+7.1%}  "
              f"{np.max(ra):>+7.1%}  {np.mean(ra):>+7.1%}")
        t_stat, p_val = ttest_ind(pa, ra, equal_var=False)
        direction = 'underperformed' if np.mean(pa) < np.mean(ra) else 'outperformed'
        sig = '  *** significant' if p_val < 0.05 else ''
        print(f"\n  Penalized group {direction} rest by "
              f"{abs(np.mean(pa) - np.mean(ra)):>+.4f}  "
              f"t={t_stat:>+.2f}  p={p_val:.4f}{sig}")

    return df


# ================================================================================
# DATE INSPECTOR
# ================================================================================

def inspect_date(Pxs_df,
                 composite_by_date,
                 universe,
                 pct_df,
                 k_df,
                 rebal_dt=None,
                 windows_td=None,
                 validation_days=21,
                 use_robust=False):
    """
    For a single rebalancing date (default: last penalized date in k_df),
    print per-stock penalty details: alpha z-score, adjusted z-score,
    penalty factor, % reduction, dominant window, rescaled return vs
    threshold for each window.

    Parameters
    ----------
    k_df       : output of calibrate_k
    rebal_dt   : rebalancing date to inspect (default: last date where k*>0)
    """
    if windows_td is None:
        windows_td = WINDOWS_TD

    universe = [t for t in universe if t in Pxs_df.columns]
    px_dates = Pxs_df.index

    # Select date
    penalized_dates = k_df[k_df['k_star'] > 0].index
    if len(penalized_dates) == 0:
        print("  No penalized dates found in k_df.")
        return None

    if rebal_dt is None:
        rebal_dt = penalized_dates[-1]
    else:
        rebal_dt = pd.Timestamp(rebal_dt)
        if rebal_dt not in k_df.index:
            print(f"  {rebal_dt.date()} not found in k_df.")
            return None

    best_k   = k_df.loc[rebal_dt, 'k_star']
    best_pct = int(k_df.loc[rebal_dt, 'pct_star'])
    ic_base  = k_df.loc[rebal_dt, 'ic_baseline']
    ic_pen   = k_df.loc[rebal_dt, 'ic_penalized']

    # Signal date
    prior_px   = px_dates[px_dates < rebal_dt]
    if len(prior_px) < validation_days + 1:
        print(f"  Insufficient history before {rebal_dt.date()}")
        return None
    signal_dt  = prior_px[-validation_days]
    signal_idx = px_dates.get_loc(signal_dt)
    rebal_idx  = px_dates.get_loc(prior_px[-1])

    print("\n" + "="*72)
    print(f"  PENALTY INSPECTOR  @  {rebal_dt.date()}")
    print("="*72)
    print(f"  signal_dt={signal_dt.date()}  k*={best_k}  pct*=p{best_pct}")
    print(f"  IC_baseline={ic_base:+.4f}  IC_penalized={ic_pen:+.4f}  "
          f"delta={ic_pen-ic_base:+.4f}\n")

    # Alpha scores at signal_dt
    avail_comp = {d: v for d, v in composite_by_date.items() if d <= signal_dt}
    if not avail_comp:
        print("  No composite scores available.")
        return None
    alpha_s = avail_comp[max(avail_comp.keys())]

    # pct_df row at signal_dt
    avail_pct = pct_df[pct_df.index <= signal_dt]
    if avail_pct.empty:
        print("  No pct_df row available.")
        return None
    pct_row = avail_pct.iloc[-1]

    # Thresholds for this pct
    thresholds = {wlbl: pct_row.get(f'p{best_pct}_{wlbl}', np.nan)
                  for wlbl in windows_td}

    # Forward returns for context
    fwd_rets = {}
    for tkr in universe:
        if tkr not in Pxs_df.columns:
            continue
        p_s = Pxs_df[tkr].iloc[signal_idx]
        p_e = Pxs_df[tkr].iloc[rebal_idx]
        if p_s > 0 and not np.isnan(p_s) and not np.isnan(p_e):
            fwd_rets[tkr] = p_e / p_s - 1

    # Build per-stock rows
    rows = []
    for tkr in universe:
        if tkr not in alpha_s.index:
            continue
        alpha_z = float(alpha_s[tkr])
        if np.isnan(alpha_z):
            continue

        max_excess = 0.0
        dom_window = None
        per_window = {}

        for wlbl, wtd in windows_td.items():
            m1y = {'mean': pct_row.get(f'mean_1Y_{wlbl}', np.nan),
                   'std':  pct_row.get(f'std_1Y_{wlbl}',  np.nan)}
            m5y = {'mean':   pct_row.get(f'mean_5Y_{wlbl}',   np.nan),
                   'std':    pct_row.get(f'std_5Y_{wlbl}',    np.nan),
                   'median': pct_row.get(f'median_5Y_{wlbl}', np.nan),
                   'iqr':    pct_row.get(f'iqr_5Y_{wlbl}',    np.nan)}
            thr = thresholds[wlbl]
            rr  = _compute_rescaled_return(
                Pxs_df[tkr], signal_idx, wtd, m1y, m5y, use_robust)
            excess = max(0.0, rr - thr) if not (np.isnan(rr) or np.isnan(thr)) else 0.0
            per_window[wlbl] = (rr, thr, excess)
            if excess > max_excess:
                max_excess = excess
                dom_window = wlbl

        if max_excess == 0:
            continue

        penalty_factor = np.exp(-best_k * max_excess)
        rows.append({
            'ticker':         tkr,
            'alpha_z':        alpha_z,
            'adj_alpha_z':    alpha_z * penalty_factor,
            'penalty_factor': penalty_factor,
            'pct_reduction':  (1 - penalty_factor) * 100,
            'dom_window':     dom_window or 'none',
            'max_excess':     max_excess,
            'fwd_ret_21d':    fwd_rets.get(tkr, np.nan),
            **{f'rr_{w}':     per_window[w][0] for w in windows_td},
            **{f'thr_{w}':    per_window[w][1] for w in windows_td},
            **{f'excess_{w}': per_window[w][2] for w in windows_td},
        })

    if not rows:
        print("  No stocks penalized at this date.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('penalty_factor').reset_index(drop=True)

    # Print header
    w_labels = list(windows_td.keys())
    rr_hdrs  = ''.join(f"  {'rr_'+w:>8}  {'thr_'+w:>8}" for w in w_labels)
    print(f"  {'Ticker':<8}  {'alpha_z':>8}  {'adj_z':>8}  "
          f"{'factor':>8}  {'reduc%':>7}  {'dom':>4}  "
          f"{'fwd_21d':>8}" + rr_hdrs)
    print("  " + "-" * (70 + len(w_labels) * 20))

    for _, r in df.iterrows():
        rr_vals = ''.join(
            f"  {r[f'rr_{w}']:>8.3f}  {r[f'thr_{w}']:>8.3f}"
            for w in w_labels
        )
        fwd_str = f"{r['fwd_ret_21d']:>+8.1%}" if not np.isnan(r['fwd_ret_21d']) else f"{'n/a':>8}"
        print(f"  {r['ticker']:<8}  {r['alpha_z']:>+8.3f}  {r['adj_alpha_z']:>+8.3f}  "
              f"{r['penalty_factor']:>8.4f}  {r['pct_reduction']:>6.1f}%  "
              f"{r['dom_window']:>4}  {fwd_str}" + rr_vals)

    # Summary stats for penalized group
    pen_fwd = df['fwd_ret_21d'].dropna()
    if not pen_fwd.empty:
        print(f"\n  Penalized group fwd 21d  (n={len(pen_fwd)}):")
        print(f"  min={pen_fwd.min():>+.1%}  p25={np.percentile(pen_fwd,25):>+.1%}  "
              f"med={pen_fwd.median():>+.1%}  p75={np.percentile(pen_fwd,75):>+.1%}  "
              f"max={pen_fwd.max():>+.1%}  mean={pen_fwd.mean():>+.1%}")

    print(f"\n  p{best_pct} thresholds used:")
    for wlbl in windows_td:
        print(f"    {wlbl}: {thresholds[wlbl]:.4f}")

    return df
