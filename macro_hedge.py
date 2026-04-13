"""
macro_indicators.py
===================
Self-contained module: builds MOVE proxy and QBD indicators, fits optimal
thresholds monthly (point-in-time), and generates a binary hedge signal.

INDICATORS
----------
  MOVE proxy  : average ewm(halflife=MOVE_HL).std() across rate tenors in rates_l
                Units: same as input (bps), no annualisation
  QBD         : Quintile Breadth Difference with false-signal correction
                  = (breadth_highQ - breadth_lowQ) - max(BREADTH_T - breadth_lowQ, 0)
                where breadth = % of stocks in quintile trading above MAV50
                and quintiles assigned daily by ewm(halflife=VOL_HL).std()

SIGNAL FITTING
--------------
  Monthly point-in-time grid search over (MOVE_T, QBD_T) pairs.
  Objective: maximise weighted capture of ref_asset excess drawdown below REF_DD_T.
  Weights: 5-year rolling window with annual step weights [5,4,3,2,1].
  Signal = 1 when MOVE >= MOVE_T AND QBD <= QBD_T, else 0.

HYPERPARAMETERS
---------------
  HARD_START   = '2010-01-01'    earliest date for all computations
  MOVE_HL      = 10              ewm halflife for MOVE (trading days)
  VOL_HL       = 42              ewm halflife for vol quintile assignment
  BREADTH_T    = 30              breadth threshold for QBD correction (%)
  MAV_WIN      = 50              moving average window for breadth (fixed)
  REF_ASSET    = 'QQQ'           reference asset for DD objective
  REF_DD_T     = -0.075          DD threshold for objective (-7.5%)
  WEIGHT_YEARS = [5,4,3,2,1]    annual weights, most recent first

USAGE
-----
    from macro_indicators import run_macro_hedge, plot_macro_hedge

    result = run_macro_hedge(
        Pxs_df    = Pxs_df,
        stocks_l  = stocks_l,    # stocks for QBD quintile analysis
        rates_l   = rates_l,     # rate tenor columns for MOVE
        move_grid = move_grid,   # list of MOVE_T candidates (user-provided)
        qbd_grid  = qbd_grid,    # list of QBD_T candidates (user-provided)
    )

    # result keys:
    #   'signal_df'  -- daily DataFrame with columns:
    #                   signal, move_val, qbd_val, move_t, qbd_t,
    #                   score, move_signal, qbd_signal
    #   'move_s'     -- raw MOVE proxy series
    #   'qbd_s'      -- raw QBD series
    #   'fitted'     -- dict {fit_date: (move_t, qbd_t, score)}

    plot_macro_hedge(result, Pxs_df, start='2015-01-01')
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Hyperparameters ───────────────────────────────────────────────────────────
HARD_START   = '2010-01-01'
MOVE_HL      = 10             # ewm halflife for MOVE vol calculation (trading days)
MOVE_MAV     = 10             # rolling mean window for MOVE differentiation
CORR_WIN     = 63             # rolling window for MOVE-QQQ correlation adjustment
VOL_HL       = 42
BREADTH_T    = 30
MAV_WIN      = 50
REF_ASSET    = 'QQQ'
REF_DD_T     = -0.075
WEIGHT_YEARS = [5, 4, 3, 2, 1]
HEDGE_LAMBDA = 0.5   # penalty weight for missed upside (kept for API compat)
HEDGE_MIN_HOLD  = 3     # minimum days to hold hedge state before switching (2-5)
RETURN_EXP      = 2/3   # concave return exponent for objective (2/3 = risk-averse utility)


# ================================================================================
# STEP 1: MOVE PROXY
# ================================================================================

def _build_move(Pxs_df:     pd.DataFrame,
                rates_l:    list,
                move_hl:    int = MOVE_HL,
                move_mav:   int = MOVE_MAV,
                corr_win:   int = CORR_WIN,
                hard_start: str = HARD_START) -> pd.Series:
    """Average rolling(move_hl).std() across rate tenors, differentiated by
    subtracting rolling(move_mav).mean(), then scaled by max(-corr(MOVE,QQQ),0)
    to filter out rate vol spikes that are equity-neutral or equity-positive."""
    available = [r for r in rates_l if r in Pxs_df.columns]
    if not available:
        raise ValueError(f"No rates_l columns found in Pxs_df: {rates_l}")
    missing = set(rates_l) - set(available)
    if missing:
        print(f"  MOVE: ignoring {len(missing)} missing tenors: {sorted(missing)}")

    rates_df   = Pxs_df[available].loc[pd.Timestamp(hard_start):]
    tenor_vols = []
    for col in available:
        s = rates_df[col].dropna()
        if len(s) < move_hl * 2:
            continue
        tenor_vols.append(
            s.ewm(halflife=move_hl, min_periods=move_hl // 2).std() * 100
        )

    if not tenor_vols:
        raise ValueError("Insufficient data to compute MOVE proxy")

    raw_move  = pd.concat(tenor_vols, axis=1).mean(axis=1)
    move_diff = raw_move - raw_move.rolling(move_mav, min_periods=1).mean()

    # Correlation adjustment: scale by max(-corr(MOVE_changes, QQQ_returns), 0)
    # Zeroes out MOVE when rate vol is not causing equity selloffs
    if 'QQQ' in Pxs_df.columns:
        qqq_ret      = Pxs_df['QQQ'].loc[pd.Timestamp(hard_start):].pct_change()
        move_chg     = raw_move.diff()
        common_idx   = move_chg.dropna().index.intersection(qqq_ret.dropna().index)
        rolling_corr = (move_chg.reindex(common_idx)
                        .rolling(corr_win, min_periods=corr_win//2)
                        .corr(qqq_ret.reindex(common_idx)))
        adj_factor   = np.ceil((-rolling_corr).clip(lower=0)).reindex(move_diff.index).fillna(0)
        move_s       = move_diff * adj_factor
        pct_zero     = (adj_factor == 0).mean() * 100
        pct_active   = 100 - pct_zero
        print(f"  MOVE: corr-filter applied (corr_win={corr_win}d)  "
              f"active={pct_active:.1f}%  zeroed={pct_zero:.1f}%")
    else:
        print(f"  MOVE: QQQ not in Pxs_df — skipping correlation adjustment")
        move_s = move_diff

    move_s.name = 'MOVE_proxy'
    print(f"  MOVE: {len(available)} tenors  "
          f"{move_s.notna().sum()} dates  "
          f"range [{move_s.min():.3f}, {move_s.max():.3f}]")
    return move_s


# ================================================================================
# STEP 2: QBD (Quintile Breadth Difference)
# ================================================================================

def _build_qbd(Pxs_df:     pd.DataFrame,
               stocks_l:   list,
               vol_hl:     int   = VOL_HL,
               breadth_t:  float = BREADTH_T,
               mav_win:    int   = MAV_WIN,
               hard_start: str   = HARD_START) -> pd.Series:
    """
    QBD = (breadth_highQ - breadth_lowQ) - max(BREADTH_T - breadth_lowQ, 0)
    Quintiles assigned daily by ewm vol. Breadth = % stocks above MAV_WIN-day MAV.
    """
    available = [s for s in stocks_l if s in Pxs_df.columns]
    if not available:
        raise ValueError("No stocks_l columns found in Pxs_df")
    missing = set(stocks_l) - set(available)
    if missing:
        print(f"  QBD: ignoring {len(missing)} missing stocks")

    px = Pxs_df[available].loc[pd.Timestamp(hard_start):]

    mav   = px.rolling(mav_win, min_periods=mav_win // 2).mean()
    above = (px > mav).astype(float)
    above[px.isna() | mav.isna()] = np.nan

    ewm_vol  = px.pct_change().ewm(halflife=vol_hl, min_periods=vol_hl).std() * 100
    min_date = px.index[max(mav_win, vol_hl * 3)]
    qbd_vals = {}
    n_dates  = len(px.index)

    for i, dt in enumerate(px.index):
        if dt < min_date:
            continue

        vol_row   = ewm_vol.loc[dt].dropna()
        above_row = above.loc[dt].reindex(vol_row.index).dropna()
        common    = vol_row.index.intersection(above_row.index)
        if len(common) < 20:
            continue

        v = vol_row.loc[common]
        a = above_row.loc[common]

        breadth_low  = a[v <= v.quantile(0.20)].mean() * 100
        breadth_high = a[v >= v.quantile(0.80)].mean() * 100

        raw_qbd      = breadth_high - breadth_low
        correction   = max(breadth_t - breadth_low, 0)
        qbd_vals[dt] = raw_qbd - correction

        if i % 500 == 0:
            print(f"  QBD: [{i}/{n_dates}] {dt.date()}...", end='\r')

    qbd_s      = pd.Series(qbd_vals, name='QBD')
    print(f"  QBD: {len(available)} stocks  "
          f"{len(qbd_s)} dates  "
          f"range [{qbd_s.min():.1f}, {qbd_s.max():.1f}]  "
          f"mean {qbd_s.mean():.1f}")
    return qbd_s


# ================================================================================
# STEP 3: THRESHOLD FITTING (point-in-time, monthly)
# ================================================================================

def _fit_thresholds(move_s:       pd.Series,
                    qbd_s:        pd.Series,
                    ref_s:        pd.Series,
                    fit_date:     pd.Timestamp,
                    move_grid:    list,
                    qbd_grid:     list,
                    ref_dd_t:     float = REF_DD_T,
                    weight_years: list  = WEIGHT_YEARS,
                    hedge_lambda: float = HEDGE_LAMBDA,
                    return_exp:   float = RETURN_EXP) -> tuple:
    """
    Find (MOVE_T, QBD_T) maximising: weighted_mean(sign(r)*|r|^f) / |log(vol)|
    where f=return_exp (default 2/3) — concave utility, rewards higher hedge
    frequency vs linear return while preserving sign of losses.
    Returns (move_t, qbd_t, score) or (None, None, nan) if insufficient data.
    """
    n_years  = len(weight_years)
    start_dt = fit_date - pd.DateOffset(years=n_years)

    common = (move_s.loc[start_dt:fit_date].dropna().index
              .intersection(qbd_s.loc[start_dt:fit_date].dropna().index)
              .intersection(ref_s.loc[start_dt:fit_date].dropna().index))

    if len(common) < 63:
        return None, None, np.nan

    m         = move_s.reindex(common)
    q         = qbd_s.reindex(common)
    daily_ret = ref_s.reindex(common).pct_change().fillna(0)

    # Annual step weights
    weights = pd.Series(1.0, index=common)
    for yr_idx, w in enumerate(weight_years):
        yr_end   = fit_date - pd.DateOffset(years=yr_idx)
        yr_start = fit_date - pd.DateOffset(years=yr_idx + 1)
        weights.loc[(common >= yr_start) & (common < yr_end)] = w
    weights /= weights.sum()

    best_score  = -np.inf
    best_move_t = None
    best_qbd_t  = None

    LOG_VOL_FLOOR = np.log(0.002)   # floor at ~0.2% daily vol to avoid log blowup

    for mt in move_grid:
        for qt in qbd_grid:
            signal        = ((m >= mt) & (q <= qt)).astype(float)
            signal_lagged = signal.shift(1).fillna(0)   # 1-day lag
            hedged_ret    = daily_ret * (1 - signal_lagged)
            # Concave return transformation: sign(r) * |r|^f preserves sign
            trans_ret = np.sign(hedged_ret) * np.abs(hedged_ret) ** return_exp
            w_mean    = (trans_ret * weights).sum()
            w_var     = (weights * (hedged_ret - hedged_ret.mean()) ** 2).sum()
            w_std     = np.sqrt(w_var) if w_var > 0 else np.nan
            log_vol   = max(np.log(w_std), LOG_VOL_FLOOR) if (w_std and w_std > 0) else LOG_VOL_FLOOR
            score     = (w_mean * np.sqrt(252)) / abs(log_vol) if log_vol != 0 else -np.inf
            if score > best_score:
                best_score  = score
                best_move_t = mt
                best_qbd_t  = qt

    return best_move_t, best_qbd_t, best_score


# ================================================================================
# STEP 4: DAILY SIGNAL GENERATION
# ================================================================================

def _build_signal(move_s:       pd.Series,
                  qbd_s:        pd.Series,
                  ref_s:        pd.Series,
                  move_grid:    list,
                  qbd_grid:     list,
                  ref_dd_t:     float = REF_DD_T,
                  weight_years: list  = WEIGHT_YEARS,
                  hedge_lambda: float = HEDGE_LAMBDA,
                  return_exp:   float = RETURN_EXP,
                  min_hold:     int   = HEDGE_MIN_HOLD,
                  hard_start:   str   = HARD_START,
                  refit_freq:   str   = '10B') -> tuple:
    """Monthly fitting + daily signal. Returns (signal_df, fitted_dict)."""
    start_dt  = pd.Timestamp(hard_start)
    n_years   = len(weight_years)
    fit_start = start_dt + pd.DateOffset(years=n_years)

    common_dates = (move_s.loc[start_dt:].dropna().index
                    .intersection(qbd_s.loc[start_dt:].dropna().index)
                    .intersection(ref_s.loc[start_dt:].dropna().index))

    if len(common_dates) == 0:
        raise ValueError("No common dates across indicators and ref asset")

    first_fit = common_dates[common_dates >= fit_start]
    if len(first_fit) == 0:
        raise ValueError("No dates available after fit_start")

    refit_dates = pd.date_range(
        start = first_fit[0],
        end   = common_dates[-1],
        freq  = refit_freq,
    )

    print(f"  Refit schedule: {len(refit_dates)} refit dates  "
          f"grid {len(move_grid)}x{len(qbd_grid)}="
          f"{len(move_grid)*len(qbd_grid)} pairs")

    fitted = {}
    for i, fd in enumerate(refit_dates):
        mt, qt, sc = _fit_thresholds(
            move_s, qbd_s, ref_s, fd,
            move_grid, qbd_grid, ref_dd_t, weight_years, hedge_lambda, return_exp
        )
        fitted[fd] = (mt, qt, sc)
        if (i+1) % 50 == 0 or i == len(refit_dates)-1:
            print(f"  [{i+1}/{len(refit_dates)}] {fd.date()}...", end='\r')

    print()  # newline after progress

    # -- Parameter distribution histograms ------------------------------------
    import matplotlib.pyplot as plt
    valid      = [(mt, qt, sc) for mt, qt, sc in fitted.values() if mt is not None]
    move_vals  = [v[0] for v in valid]
    qbd_vals_h = [v[1] for v in valid]
    score_vals = [v[2] for v in valid]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    fig.patch.set_facecolor('#FAFAF9')
    fig.suptitle(f'Fitted parameter distributions ({len(valid)} refit dates)',
                 fontsize=10, fontweight='500', color='#2C2C2A')

    for ax in axes:
        ax.set_facecolor('#FAFAF9')
        ax.grid(color='#D3D1C7', linewidth=0.4, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].hist(move_vals,  bins=len(set(move_vals)),  color='#D85A30',
                 alpha=0.75, edgecolor='white', linewidth=0.5)
    axes[0].set_title('MOVE_T', fontsize=10)
    axes[0].set_xlabel('threshold value')

    axes[1].hist(qbd_vals_h, bins=len(set(qbd_vals_h)), color='#378ADD',
                 alpha=0.75, edgecolor='white', linewidth=0.5)
    axes[1].set_title('QBD_T', fontsize=10)
    axes[1].set_xlabel('threshold value')

    axes[2].hist(score_vals, bins=20, color='#1D9E75',
                 alpha=0.75, edgecolor='white', linewidth=0.5)
    axes[2].set_title('Objective score', fontsize=10)
    axes[2].set_xlabel('score value')

    plt.tight_layout()
    plt.show()

    print(f"  MOVE_T  mode={max(set(move_vals),  key=move_vals.count):.2f}  "
          f"values={sorted(set(move_vals))}")
    print(f"  QBD_T   mode={max(set(qbd_vals_h), key=qbd_vals_h.count):.1f}  "
          f"values={sorted(set(qbd_vals_h))}")

    # Daily signal using last fitted thresholds
    records     = []
    past_refits = sorted(fitted.keys())

    cur_move_t = None
    cur_qbd_t  = None
    cur_score  = np.nan

    for dt in common_dates:
        candidates = [rd for rd in past_refits if rd <= dt]
        if candidates:
            lr = candidates[-1]
            cur_move_t, cur_qbd_t, cur_score = fitted[lr]

        if cur_move_t is None:
            continue

        mv = move_s.get(dt, np.nan)
        qv = qbd_s.get(dt, np.nan)
        if np.isnan(mv) or np.isnan(qv):
            continue

        mf = int(mv >= cur_move_t)
        qf = int(qv <= cur_qbd_t)

        records.append({
            'date':        dt,
            'signal_raw':  int(mf and qf),
            'move_val':    mv,
            'qbd_val':     qv,
            'move_t':      cur_move_t,
            'qbd_t':       cur_qbd_t,
            'score':       cur_score,
            'move_signal': mf,
            'qbd_signal':  qf,
        })

    signal_df = pd.DataFrame(records).set_index('date')

    # Apply minimum holding period — only on ON state (hedge stays on min_hold days)
    raw       = signal_df['signal_raw'].values
    smoothed  = raw.copy()
    days_on   = 0
    for i in range(len(raw)):
        if smoothed[i-1] == 1 if i > 0 else False:
            days_on += 1
        else:
            days_on = 0
        # If signal turns OFF but min hold not reached, keep ON
        if raw[i] == 0 and days_on > 0 and days_on < min_hold:
            smoothed[i] = 1
        else:
            smoothed[i] = raw[i]
            if raw[i] == 1:
                days_on = max(days_on, 1)
    signal_df['signal'] = smoothed
    n_on      = signal_df['signal'].sum()
    n_tot     = len(signal_df)
    print(f"\n  Signal: {n_tot} days  "
          f"ON={n_on} ({n_on/n_tot*100:.1f}%)  "
          f"OFF={n_tot-n_on} ({(n_tot-n_on)/n_tot*100:.1f}%)")

    # Yearly breakdown
    print(f"\n  {'Year':<6}  {'Days':>6}  {'Hedged':>8}  {'%':>7}  Distribution")
    print(f"  {'-'*52}")
    for yr in sorted(signal_df.index.year.unique()):
        yr_s   = signal_df[signal_df.index.year == yr]['signal']
        yr_on  = int(yr_s.sum())
        yr_tot = len(yr_s)
        pct    = yr_on / yr_tot * 100 if yr_tot > 0 else 0
        bar    = '#' * int(pct / 3)
        print(f"  {yr:<6}  {yr_tot:>6}  {yr_on:>8}  {pct:>6.1f}%  {bar}")

    return signal_df, fitted


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def run_macro_hedge(Pxs_df:       pd.DataFrame,
                    stocks_l:     list,
                    rates_l:      list,
                    move_grid:    list,
                    qbd_grid:     list,
                    move_hl:      int   = MOVE_HL,
                    move_mav:     int   = MOVE_MAV,
                    corr_win:     int   = CORR_WIN,
                    vol_hl:       int   = VOL_HL,
                    breadth_t:    float = BREADTH_T,
                    ref_asset:    str   = REF_ASSET,
                    ref_dd_t:     float = REF_DD_T,
                    weight_years: list  = WEIGHT_YEARS,
                    hedge_lambda: float = HEDGE_LAMBDA,
                    return_exp:   float = RETURN_EXP,
                    min_hold:     int   = HEDGE_MIN_HOLD,
                    hard_start:   str   = HARD_START,
                    refit_freq:   str   = '10B') -> dict:
    """
    Full pipeline: build MOVE + QBD -> fit thresholds monthly -> generate signal.

    Parameters
    ----------
    Pxs_df       : price DataFrame containing stocks, rates and ref_asset
    stocks_l     : stock tickers for QBD quintile analysis
    rates_l      : rate tenor column names for MOVE proxy
    move_grid    : list of MOVE_T candidates (user-provided)
    qbd_grid     : list of QBD_T candidates (user-provided)
    move_hl      : MOVE ewm halflife (default 10)
    vol_hl       : vol quintile ewm halflife (default 42)
    breadth_t    : QBD correction threshold % (default 30)
    ref_asset    : reference asset for DD objective (default 'QQQ')
    ref_dd_t     : DD threshold for objective (default -7.5%)
    weight_years : annual weights most-recent-first (default [5,4,3,2,1])
    hard_start   : earliest computation date (default '2010-01-01')
    refit_freq   : refit frequency (default '10B' = every 10 business days)

    Returns
    -------
    dict:
      'signal_df' : pd.DataFrame  columns: signal, move_val, qbd_val,
                                           move_t, qbd_t, score,
                                           move_signal, qbd_signal
      'move_s'    : pd.Series     raw MOVE proxy
      'qbd_s'     : pd.Series     raw QBD
      'fitted'    : dict          {fit_date: (move_t, qbd_t, score)}
    """
    if ref_asset not in Pxs_df.columns:
        raise ValueError(f"ref_asset '{ref_asset}' not found in Pxs_df")

    print('=' * 72)
    print('  MACRO HEDGE SIGNAL BUILDER')
    print('=' * 72)
    print(f"\n  ref={ref_asset}  ref_dd_t={ref_dd_t:.1%}  "
          f"lambda={hedge_lambda}  hard_start={hard_start}")
    print(f"  MOVE_HL={move_hl}  MOVE_MAV={move_mav}  CORR_WIN={corr_win}  VOL_HL={vol_hl}  "
          f"BREADTH_T={breadth_t}%  MAV={MAV_WIN}d  min_hold={min_hold}d  return_exp={return_exp}")
    print(f"  grid: {len(move_grid)} MOVE_T x {len(qbd_grid)} QBD_T = "
          f"{len(move_grid)*len(qbd_grid)} pairs  "
          f"weights={weight_years}")

    print('\n[1/3] Building MOVE proxy...')
    move_s = _build_move(Pxs_df, rates_l,
                         move_hl=move_hl, move_mav=move_mav,
                         corr_win=corr_win, hard_start=hard_start)

    print('\n[2/3] Building QBD...')
    qbd_s = _build_qbd(Pxs_df, stocks_l,
                       vol_hl=vol_hl, breadth_t=breadth_t,
                       mav_win=MAV_WIN, hard_start=hard_start)

    ref_s = Pxs_df[ref_asset].dropna()
    print('\n[3/3] Fitting thresholds and building signal...')
    signal_df, fitted = _build_signal(
        move_s, qbd_s, ref_s,
        move_grid, qbd_grid,
        ref_dd_t=ref_dd_t,
        weight_years=weight_years,
        hedge_lambda=hedge_lambda,
        return_exp=return_exp,
        min_hold=min_hold,
        hard_start=hard_start,
        refit_freq=refit_freq,
    )

    return {
        'signal_df': signal_df,
        'move_s':    move_s,
        'qbd_s':     qbd_s,
        'fitted':    fitted,
    }


# ================================================================================
# DIAGNOSTIC PLOT
# ================================================================================

def plot_macro_hedge(result:    dict,
                     Pxs_df:   pd.DataFrame,
                     ref_asset: str = REF_ASSET,
                     start:     str = None) -> None:
    """
    4-panel plot: MOVE proxy, QBD, hedge signal, reference asset with signal shading.
    """
    import matplotlib.pyplot as plt

    signal_df = result['signal_df']
    move_s    = result['move_s']
    qbd_s     = result['qbd_s']

    st       = pd.Timestamp(start) if start else signal_df.index[0]
    df       = signal_df.loc[st:]
    ref      = Pxs_df[ref_asset].reindex(df.index)
    ref_norm = ref / ref.iloc[0] * 100

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.patch.set_facecolor('#FAFAF9')

    for ax in axes:
        ax.set_facecolor('#FAFAF9')
        ax.grid(color='#D3D1C7', linewidth=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Panel 1: MOVE
    axes[0].plot(df.index.to_numpy(), move_s.reindex(df.index).to_numpy(),
                 color='#D85A30', lw=0.8)
    axes[0].plot(df.index.to_numpy(), df['move_t'].to_numpy(),
                 color='#D85A30', lw=0.8, linestyle='--', alpha=0.6, label='threshold')
    axes[0].set_title('MOVE proxy (rate vol, bps)', fontsize=10, fontweight='500')
    axes[0].legend(fontsize=8)

    # Panel 2: QBD
    axes[1].plot(df.index.to_numpy(), qbd_s.reindex(df.index).to_numpy(),
                 color='#378ADD', lw=0.8)
    axes[1].plot(df.index.to_numpy(), df['qbd_t'].to_numpy(),
                 color='#378ADD', lw=0.8, linestyle='--', alpha=0.6, label='threshold')
    axes[1].axhline(0, color='#888780', lw=0.6, linestyle=':')
    axes[1].set_title('QBD (quintile breadth difference, %)',
                      fontsize=10, fontweight='500')
    axes[1].legend(fontsize=8)

    # Panel 3: Signal
    axes[2].fill_between(df.index.to_numpy(), df['signal'].to_numpy(), 0,
                         color='#D85A30', alpha=0.4, step='post')
    axes[2].set_ylim(-0.1, 1.3)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['flat', 'hedge'])
    axes[2].set_title('Hedge signal  (MOVE >= thr AND QBD <= thr)',
                      fontsize=10, fontweight='500')

    # Panel 4: QQQ vs Hedged QQQ
    ref_ret       = ref.pct_change().fillna(0)
    # 1-day lag: signal on day T applies from day T+1
    signal_lagged = df['signal'].shift(1).fillna(0)
    hedged_ret    = ref_ret * (1 - signal_lagged.reindex(ref_ret.index).fillna(0))
    qqq_nav       = (1 + ref_ret).cumprod() * 100
    hedged_nav    = (1 + hedged_ret).cumprod() * 100

    axes[3].plot(df.index.to_numpy(), qqq_nav.reindex(df.index).to_numpy(),
                 color='#888780', lw=1.0, label=ref_asset)
    axes[3].plot(df.index.to_numpy(), hedged_nav.reindex(df.index).to_numpy(),
                 color='#1D9E75', lw=1.0, label=f'Hedged {ref_asset}')
    axes[3].fill_between(df.index.to_numpy(),
                         qqq_nav.reindex(df.index).to_numpy().min(),
                         qqq_nav.reindex(df.index).to_numpy(),
                         where=signal_lagged.reindex(df.index).fillna(0).to_numpy() == 1,
                         color='#D85A30', alpha=0.15, step='post', label='hedge on')

    # Final NAV ratio and Sharpe comparison annotation
    final_qqq    = qqq_nav.reindex(df.index).dropna().iloc[-1]
    final_hedged = hedged_nav.reindex(df.index).dropna().iloc[-1]
    nav_ratio    = final_hedged / final_qqq * 100

    qqq_ret      = ref_ret.reindex(df.index).fillna(0)
    hedged_ret_s = hedged_ret.reindex(df.index).fillna(0)

    def _sharpe(r):
        return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    sharpe_qqq    = _sharpe(qqq_ret)
    sharpe_hedged = _sharpe(hedged_ret_s)
    sharpe_ratio  = sharpe_hedged / sharpe_qqq if sharpe_qqq and sharpe_qqq > 0 else np.nan

    annotation = (f'Hedged NAV = {nav_ratio:.1f}% of {ref_asset}\n'
                  f'Sharpe: {sharpe_hedged:.2f} vs {sharpe_qqq:.2f} '
                  f'({sharpe_ratio:.2f}x)')
    axes[3].annotate(annotation,
                     xy=(0.02, 0.05), xycoords='axes fraction',
                     fontsize=9, color='#1D9E75',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', edgecolor='#1D9E75', alpha=0.8))
    axes[3].set_title(f'{ref_asset} vs Hedged {ref_asset} (shaded = hedge active)',
                      fontsize=10, fontweight='500')
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # -- Yearly performance breakdown -----------------------------------------
    print(f"\n  {'Year':<6}  {'QQQ Ret':>9}  {'Hedged Ret':>11}  "
          f"{'Diff':>7}  {'QQQ Sharpe':>11}  {'Hedged Sharpe':>14}  {'Sharpe Ratio':>13}")
    print(f"  {'-'*82}")

    all_years = sorted(qqq_ret.index.year.unique())
    for yr in all_years:
        mask     = qqq_ret.index.year == yr
        r_qqq    = qqq_ret[mask]
        r_hedged = hedged_ret_s[mask]

        ret_qqq    = (1 + r_qqq).prod() - 1
        ret_hedged = (1 + r_hedged).prod() - 1
        diff       = ret_hedged - ret_qqq
        sh_qqq     = _sharpe(r_qqq)
        sh_hedged  = _sharpe(r_hedged)
        sh_ratio   = sh_hedged / sh_qqq if sh_qqq and sh_qqq > 0 else np.nan

        diff_str  = f"{diff*100:>+6.1f}%"
        ratio_str = f"{sh_ratio:.2f}x" if not np.isnan(sh_ratio) else "  n/a"
        print(f"  {yr:<6}  {ret_qqq*100:>8.1f}%  {ret_hedged*100:>10.1f}%  "
              f"{diff_str:>7}  {sh_qqq:>11.2f}  {sh_hedged:>14.2f}  {ratio_str:>13}")

    print(f"  {'-'*82}")
    print(f"  {'Full'::<6}  {(final_qqq/100-1)*100:>8.1f}%  {(final_hedged/100-1)*100:>10.1f}%  "
          f"  {'':>7}  {sharpe_qqq:>11.2f}  {sharpe_hedged:>14.2f}  {sharpe_ratio:>12.2f}x")
