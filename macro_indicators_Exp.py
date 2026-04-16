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
MOVE_MAV     = 5              # rolling mean window for MOVE differentiation
CORR_WIN     = 63             # rolling window for MOVE-QQQ correlation adjustment
VOL_HL       = 42
BREADTH_T    = 30
MAV_WIN      = 50
REF_ASSET    = 'QQQ'
REF_DD_T     = -0.075
WEIGHT_YEARS = [5, 4, 3, 2, 1]
HEDGE_LAMBDA = 0.5   # penalty weight for missed upside (kept for API compat)
HEDGE_MIN_HOLD  = 10    # minimum days to hold hedge state before switching
SIGNAL_LOGIC    = 'AND' # how to combine MOVE and QBD signals: 'AND' or 'OR'
RETURN_EXP      = 1/3   # concave return exponent for objective (1/3 = more risk-averse)


# ================================================================================
# STEP 1: MOVE PROXY
# ================================================================================

def _build_move(Pxs_df:     pd.DataFrame,
                rates_l:    list,
                move_hl:    int = MOVE_HL,
                move_mav:   int = MOVE_MAV,
                corr_win:   int = CORR_WIN,
                ref_asset:  str = REF_ASSET,
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
    if ref_asset in Pxs_df.columns:
        qqq_ret      = Pxs_df[ref_asset].loc[pd.Timestamp(hard_start):].pct_change()
        move_chg     = raw_move.diff()
        common_idx   = move_chg.dropna().index.intersection(qqq_ret.dropna().index)
        rolling_corr = (move_chg.reindex(common_idx)
                        .rolling(corr_win, min_periods=corr_win//2)
                        .corr(qqq_ret.reindex(common_idx)))
        adj_factor   = np.ceil((-rolling_corr).clip(lower=0)).reindex(move_diff.index).fillna(0)
        move_s       = move_diff * adj_factor
        pct_zero     = (adj_factor == 0).mean() * 100
        pct_active   = 100 - pct_zero
        print(f"  MOVE: corr-filter applied vs {ref_asset} (corr_win={corr_win}d)  "
              f"active={pct_active:.1f}%  zeroed={pct_zero:.1f}%")
    else:
        print(f"  MOVE: {ref_asset} not in Pxs_df — skipping correlation adjustment")
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
                  signal_logic: str   = SIGNAL_LOGIC,
                  hard_start:   str   = HARD_START,
                  refit_freq:   str   = '10B',
                  engine=None,
                  instrument:   str   = '',
                  params_hash:  str   = '') -> tuple:
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

    # ── Load cached fitted params, only compute missing dates ─────────────────
    fitted = {}
    if engine is not None and instrument and params_hash:
        fitted = _load_cached_fitted(engine, instrument, params_hash)

    missing = [fd for fd in refit_dates if fd not in fitted]
    n_cached = len(refit_dates) - len(missing)

    print(f"  Refit schedule: {len(refit_dates)} refit dates  "
          f"grid {len(move_grid)}x{len(qbd_grid)}="
          f"{len(move_grid)*len(qbd_grid)} pairs  "
          f"({n_cached} cached, {len(missing)} to compute)")

    new_fitted = {}
    for i, fd in enumerate(missing):
        mt, qt, sc = _fit_thresholds(
            move_s, qbd_s, ref_s, fd,
            move_grid, qbd_grid, ref_dd_t, weight_years, hedge_lambda, return_exp
        )
        new_fitted[fd] = (mt, qt, sc)
        if (i+1) % 50 == 0 or i == len(missing)-1:
            print(f"  [{i+1}/{len(missing)}] {fd.date()}...", end='\r')

    if missing:
        print()  # newline after progress
        fitted.update(new_fitted)
        if engine is not None and instrument and params_hash:
            _save_fitted(engine, instrument, params_hash, new_fitted)

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

        mf  = int(mv >= cur_move_t)
        qf  = int(qv <= cur_qbd_t)
        raw = int(mf and qf) if signal_logic.upper() == 'AND' else int(mf or qf)

        records.append({
            'date':        dt,
            'signal_raw':  raw,
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
    raw      = signal_df['signal_raw'].values.copy()
    smoothed = np.zeros(len(raw), dtype=int)
    i = 0
    while i < len(raw):
        if raw[i] == 1:
            # Enter ON state — hold for at least min_hold days
            hold_end = min(i + min_hold, len(raw))
            smoothed[i:hold_end] = 1
            # Continue holding as long as raw signal stays ON beyond min_hold
            j = hold_end
            while j < len(raw) and raw[j] == 1:
                smoothed[j] = 1
                j += 1
            i = j
        else:
            smoothed[i] = 0
            i += 1
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
                    signal_logic: str   = SIGNAL_LOGIC,
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
                         corr_win=corr_win, ref_asset=ref_asset,
                         hard_start=hard_start)

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
        signal_logic=signal_logic,
        hard_start=hard_start,
        refit_freq=refit_freq,
    )

    return {
        'signal_df':    signal_df,
        'move_s':       move_s,
        'qbd_s':        qbd_s,
        'fitted':       fitted,
        'signal_logic': signal_logic,
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
    _logic_str = result.get('signal_logic', 'AND')
    _op = '>= MOVE_T AND QBD <=' if _logic_str == 'AND' else '>= MOVE_T OR QBD <='
    axes[2].set_title(f'Hedge signal  (MOVE {_op} QBD_T)  [{_logic_str}]',
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


# ================================================================================
# MULTI-INSTRUMENT HEDGE ANALYSIS
# ================================================================================

def _weighted_effectiveness(ret_raw:    pd.Series,
                             ret_hedged: pd.Series,
                             calc_date:  pd.Timestamp,
                             weight_years: list = WEIGHT_YEARS) -> tuple:
    """
    Compute weighted NAV ratio, Sharpe improvement, and geometric effectiveness
    up to calc_date using annual step weights [5,4,3,2,1].

    Returns (nav_ratio, sharpe_raw, sharpe_hedged, sharpe_improvement, effectiveness)
    all as floats, or (nan, nan, nan, nan, nan) if insufficient data.
    """
    n_years  = len(weight_years)
    start_dt = calc_date - pd.DateOffset(years=n_years)

    common = ret_raw.loc[start_dt:calc_date].dropna().index.intersection(
             ret_hedged.loc[start_dt:calc_date].dropna().index)

    if len(common) < 63:
        return (np.nan,) * 5

    r   = ret_raw.reindex(common)
    h   = ret_hedged.reindex(common)

    # Annual step weights
    w = pd.Series(1.0, index=common)
    for yr_idx, wt in enumerate(weight_years):
        yr_end   = calc_date - pd.DateOffset(years=yr_idx)
        yr_start = calc_date - pd.DateOffset(years=yr_idx + 1)
        w.loc[(common >= yr_start) & (common < yr_end)] = wt
    w /= w.sum()

    # Weighted NAV ratio: cumulative product of hedged vs raw
    nav_raw    = (1 + r).prod()
    nav_hedged = (1 + h).prod()
    nav_ratio  = nav_hedged / nav_raw if nav_raw > 0 else np.nan

    # Weighted Sharpe
    def _w_sharpe(rets, weights):
        mu    = (rets * weights).sum()
        var   = (weights * (rets - mu) ** 2).sum()
        std   = np.sqrt(var) if var > 0 else np.nan
        return mu / std * np.sqrt(252) if std else np.nan

    sh_raw    = _w_sharpe(r, w)
    sh_hedged = _w_sharpe(h, w)

    # Sharpe improvement (difference) — robust to negative Sharpe regimes
    # Normalised by abs(sh_raw) so it's scale-independent
    if sh_raw is not None and sh_raw != 0 and not np.isnan(sh_raw):
        sh_improvement = (sh_hedged - sh_raw) / abs(sh_raw)
    else:
        sh_improvement = np.nan

    # Effectiveness: nav_ratio * (1 + sharpe_improvement), clipped to [0, 5]
    # nav_ratio > 1 means hedged NAV outperformed raw
    # sh_improvement > 0 means hedged Sharpe improved
    if (nav_ratio is not None and not np.isnan(nav_ratio) and
            sh_improvement is not None and not np.isnan(sh_improvement)):
        effectiveness = float(np.clip(nav_ratio * (1 + sh_improvement), 0, 5))
    else:
        effectiveness = np.nan

    # sh_ratio kept for reference but no longer used in effectiveness
    sh_ratio = sh_hedged / sh_raw if (sh_raw and sh_raw != 0) else np.nan

    return nav_ratio, sh_raw, sh_hedged, sh_improvement, effectiveness


def run_macro_hedge_multi(Pxs_df:       pd.DataFrame,
                          stocks_l:     list,
                          rates_l:      list,
                          hedges_l:     list,
                          move_grid:    list,
                          qbd_grid:     list,
                          move_hl:      int   = MOVE_HL,
                          move_mav:     int   = MOVE_MAV,
                          corr_win:     int   = CORR_WIN,
                          vol_hl:       int   = VOL_HL,
                          breadth_t:    float = BREADTH_T,
                          ref_dd_t:     float = REF_DD_T,
                          weight_years: list  = WEIGHT_YEARS,
                          hedge_lambda: float = HEDGE_LAMBDA,
                          return_exp:   float = RETURN_EXP,
                          min_hold:     int   = HEDGE_MIN_HOLD,
                          signal_logic: str   = SIGNAL_LOGIC,
                          hard_start:   str   = HARD_START,
                          refit_freq:   str   = '10B') -> dict:
    """
    Run macro hedge analysis for all instruments in hedges_l.

    Builds QBD once (shared), then per instrument:
      - Builds correlation-adjusted MOVE
      - Fits signal thresholds (point-in-time, every 10B days)
      - Computes daily signal
      - Computes rolling weighted effectiveness (nav_ratio, sharpe_improvement,
        effectiveness) using same [5,4,3,2,1] annual weights

    Parameters
    ----------
    Pxs_df       : price DataFrame (stocks, rates, ETFs)
    stocks_l     : stock tickers for QBD quintile analysis
    rates_l      : rate tenor columns for MOVE proxy
    hedges_l     : list of hedge instrument tickers (e.g. ['QQQ','SPY','XLK',...])
    move_grid    : list of MOVE_T candidates
    qbd_grid     : list of QBD_T candidates
    (all other params: same as run_macro_hedge)

    Returns
    -------
    dict with keys:
      'qbd_s'      : pd.Series          shared QBD series
      'raw_move_s' : pd.Series          raw (unadjusted) MOVE series
      'results'    : dict of dicts, keyed by instrument ticker:
          {
            'signal_df'   : pd.DataFrame  daily signal + effectiveness columns:
                              signal, signal_raw, move_val, qbd_val,
                              move_t, qbd_t, score, move_signal, qbd_signal,
                              nav_ratio, sharpe_raw, sharpe_hedged,
                              sharpe_improvement, effectiveness
            'move_s'      : pd.Series  corr-adjusted MOVE for this instrument
            'fitted'      : dict       {fit_date: (move_t, qbd_t, score)}
            'signal_logic': str
          }
    """
    missing_hedges = [h for h in hedges_l if h not in Pxs_df.columns]
    if missing_hedges:
        print(f"  WARNING: {len(missing_hedges)} hedges not in Pxs_df: {missing_hedges}")
    hedges_l = [h for h in hedges_l if h in Pxs_df.columns]

    print('=' * 72)
    print('  MACRO HEDGE MULTI-INSTRUMENT ANALYSIS')
    print('=' * 72)
    print(f"\n  Instruments: {hedges_l}")
    print(f"  signal_logic={signal_logic}  min_hold={min_hold}d  "
          f"return_exp={return_exp}  refit_freq={refit_freq}")

    # ── Step 1: Build shared QBD (once) ──────────────────────────────────────
    print('\n[1/3] Building shared QBD...')
    qbd_s = _build_qbd(Pxs_df, stocks_l, vol_hl=vol_hl,
                       breadth_t=breadth_t, mav_win=MAV_WIN,
                       hard_start=hard_start)

    # ── Step 2: Build shared raw MOVE (once) ─────────────────────────────────
    print('\n[2/3] Building raw MOVE proxy...')
    # Build without corr filter to get raw series for reuse
    available  = [r for r in rates_l if r in Pxs_df.columns]
    rates_df   = Pxs_df[available].loc[pd.Timestamp(hard_start):]
    tenor_vols = []
    for col in available:
        s = rates_df[col].dropna()
        if len(s) < move_hl * 2:
            continue
        tenor_vols.append(s.ewm(halflife=move_hl, min_periods=move_hl//2).std() * 100)
    raw_move_base = pd.concat(tenor_vols, axis=1).mean(axis=1)
    raw_move_diff = raw_move_base - raw_move_base.rolling(move_mav, min_periods=1).mean()
    print(f"  Raw MOVE: {len(available)} tenors  {raw_move_diff.notna().sum()} dates")

    # ── Step 3: Per-instrument loop ───────────────────────────────────────────
    print(f'\n[3/3] Processing {len(hedges_l)} instruments...')
    results = {}

    for inst_idx, inst in enumerate(hedges_l):
        print(f"\n  [{inst_idx+1}/{len(hedges_l)}] {inst} {'─'*40}")

        ref_s = Pxs_df[inst].dropna()

        # Correlation-adjusted MOVE for this instrument
        qqq_ret      = Pxs_df[inst].loc[pd.Timestamp(hard_start):].pct_change()
        move_chg     = raw_move_base.diff()
        common_idx   = move_chg.dropna().index.intersection(qqq_ret.dropna().index)
        rolling_corr = (move_chg.reindex(common_idx)
                        .rolling(corr_win, min_periods=corr_win//2)
                        .corr(qqq_ret.reindex(common_idx)))
        adj_factor   = np.ceil((-rolling_corr).clip(lower=0)).reindex(
                           raw_move_diff.index).fillna(0)
        move_s       = raw_move_diff * adj_factor
        move_s.name  = f'MOVE_{inst}'
        pct_active   = (adj_factor > 0).mean() * 100
        print(f"  MOVE corr-filter: active={pct_active:.1f}% of days")

        # Fit signal
        signal_df, fitted = _build_signal(
            move_s, qbd_s, ref_s,
            move_grid, qbd_grid,
            ref_dd_t=ref_dd_t,
            weight_years=weight_years,
            hedge_lambda=hedge_lambda,
            return_exp=return_exp,
            min_hold=min_hold,
            signal_logic=signal_logic,
            hard_start=hard_start,
            refit_freq=refit_freq,
        )

        # Rolling weighted effectiveness — compute daily
        ref_ret    = ref_s.pct_change().fillna(0)
        sig_lag    = signal_df['signal'].shift(1).fillna(0)
        hedged_ret = ref_ret * (1 - sig_lag.reindex(ref_ret.index).fillna(0))

        # Align to signal_df index
        ref_ret_a    = ref_ret.reindex(signal_df.index).fillna(0)
        hedged_ret_a = hedged_ret.reindex(signal_df.index).fillna(0)

        nav_ratio_s    = pd.Series(np.nan, index=signal_df.index)
        sharpe_raw_s   = pd.Series(np.nan, index=signal_df.index)
        sharpe_hedged_s= pd.Series(np.nan, index=signal_df.index)
        sharpe_improv_s = pd.Series(np.nan, index=signal_df.index)
        effectiveness_s= pd.Series(np.nan, index=signal_df.index)

        # Compute monthly (same cadence as refits) to avoid daily O(n²)
        refit_dates_eff = pd.date_range(
            start = signal_df.index[0],
            end   = signal_df.index[-1],
            freq  = refit_freq,
        )
        last_vals = (np.nan,) * 5
        eff_by_date = {}
        for rd in refit_dates_eff:
            nav_r, sh_r, sh_h, sh_ratio, eff = _weighted_effectiveness(
                ref_ret_a, hedged_ret_a, rd, weight_years
            )
            eff_by_date[rd] = (nav_r, sh_r, sh_h, sh_ratio, eff)

        # Forward-fill to daily
        sorted_eff_dates = sorted(eff_by_date.keys())
        for dt in signal_df.index:
            past = [d for d in sorted_eff_dates if d <= dt]
            if past:
                vals = eff_by_date[past[-1]]
                nav_ratio_s[dt]     = vals[0]
                sharpe_raw_s[dt]    = vals[1]
                sharpe_hedged_s[dt] = vals[2]
                sharpe_improv_s[dt] = vals[3]
                effectiveness_s[dt] = vals[4]

        signal_df['nav_ratio']     = nav_ratio_s
        signal_df['sharpe_raw']    = sharpe_raw_s
        signal_df['sharpe_hedged'] = sharpe_hedged_s
        signal_df['sharpe_improvement'] = sharpe_improv_s
        signal_df['effectiveness'] = effectiveness_s

        # Summary
        n_on  = signal_df['signal'].sum()
        n_tot = len(signal_df)
        eff_last = effectiveness_s.dropna().iloc[-1] if effectiveness_s.notna().any() else np.nan
        sh_r_last= sharpe_raw_s.dropna().iloc[-1]    if sharpe_raw_s.notna().any()    else np.nan
        sh_h_last= sharpe_hedged_s.dropna().iloc[-1] if sharpe_hedged_s.notna().any() else np.nan
        nav_last = nav_ratio_s.dropna().iloc[-1]     if nav_ratio_s.notna().any()     else np.nan
        print(f"  Signal ON={n_on} ({n_on/n_tot*100:.1f}%)  "
              f"nav_ratio={nav_last:.3f}  "
              f"sharpe={sh_r_last:.2f}->{sh_h_last:.2f}  "
              f"effectiveness={eff_last:.3f}")

        results[inst] = {
            'signal_df':    signal_df,
            'move_s':       move_s,
            'fitted':       fitted,
            'signal_logic': signal_logic,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  EFFECTIVENESS SUMMARY (as of {signal_df.index[-1].date()})")
    print(f"{'='*72}")
    print(f"  {'Instrument':<10}  {'Signal ON%':>10}  {'NAV Ratio':>10}  "
          f"{'Sharpe Raw':>11}  {'Sharpe Hdg':>11}  {'Sharpe Imp':>11}  {'Effectiveness':>14}")
    print(f"  {'-'*72}")
    for inst, res in results.items():
        df      = res['signal_df']
        n_on    = df['signal'].sum()
        n_tot   = len(df)
        nav     = df['nav_ratio'].dropna().iloc[-1]          if df['nav_ratio'].notna().any()          else np.nan
        sh_r    = df['sharpe_raw'].dropna().iloc[-1]         if df['sharpe_raw'].notna().any()         else np.nan
        sh_h    = df['sharpe_hedged'].dropna().iloc[-1]      if df['sharpe_hedged'].notna().any()      else np.nan
        sh_i    = df['sharpe_improvement'].dropna().iloc[-1] if df['sharpe_improvement'].notna().any() else np.nan
        eff     = df['effectiveness'].dropna().iloc[-1]      if df['effectiveness'].notna().any()      else np.nan
        print(f"  {inst:<10}  {n_on/n_tot*100:>9.1f}%  {nav:>10.3f}  "
              f"{sh_r:>11.2f}  {sh_h:>11.2f}  {sh_i:>11.3f}  {eff:>14.3f}")

    return {
        'qbd_s':      qbd_s,
        'raw_move_s': raw_move_diff,
        'results':    results,
    }


# ================================================================================
# CACHING LAYER
# ================================================================================

MACRO_SIGNAL_TBL    = 'macro_signal_daily'     # per-instrument daily signal + effectiveness
MACRO_INDICATOR_TBL = 'macro_indicators_daily'  # shared QBD + raw MOVE
MACRO_FITTED_TBL    = 'macro_fitted_params'     # cached refit parameters (move_t, qbd_t, score)
REBUILD_QBD         = False                     # hardcoded: only rebuild QBD if True


def _macro_params_hash(move_hl, move_mav, corr_win, vol_hl, breadth_t,
                       ref_dd_t, weight_years, return_exp, min_hold,
                       signal_logic, refit_freq, move_grid, qbd_grid) -> str:
    """12-char MD5 hash of all construction parameters."""
    import hashlib, json
    params = {
        'move_hl': move_hl, 'move_mav': move_mav, 'corr_win': corr_win,
        'vol_hl': vol_hl, 'breadth_t': breadth_t, 'ref_dd_t': ref_dd_t,
        'weight_years': list(weight_years), 'return_exp': round(return_exp, 10),
        'min_hold': min_hold, 'signal_logic': signal_logic,
        'refit_freq': refit_freq, 'move_grid': sorted(move_grid),
        'qbd_grid': sorted(qbd_grid),
    }
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def _ensure_macro_tables(engine):
    """Create cache tables if they don't exist."""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MACRO_INDICATOR_TBL} (
                date           DATE        NOT NULL,
                params_hash    VARCHAR(12) NOT NULL,
                qbd_val        FLOAT,
                move_val_raw   FLOAT,
                PRIMARY KEY (date, params_hash)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MACRO_SIGNAL_TBL} (
                date           DATE        NOT NULL,
                instrument     VARCHAR(20) NOT NULL,
                params_hash    VARCHAR(12) NOT NULL,
                signal         SMALLINT,
                signal_raw     SMALLINT,
                move_val       FLOAT,
                qbd_val        FLOAT,
                move_t         FLOAT,
                qbd_t          FLOAT,
                score          FLOAT,
                move_signal    SMALLINT,
                qbd_signal     SMALLINT,
                nav_ratio      FLOAT,
                sharpe_raw     FLOAT,
                sharpe_hedged  FLOAT,
                sharpe_improvement FLOAT,
                effectiveness  FLOAT,
                PRIMARY KEY (date, instrument, params_hash)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MACRO_FITTED_TBL} (
                refit_date     DATE        NOT NULL,
                instrument     VARCHAR(20) NOT NULL,
                params_hash    VARCHAR(12) NOT NULL,
                move_t         FLOAT,
                qbd_t          FLOAT,
                score          FLOAT,
                PRIMARY KEY (refit_date, instrument, params_hash)
            )
        """))


def _load_cached_fitted(engine, instrument: str, params_hash: str) -> dict:
    """Load cached fitted params for one instrument. Returns {date: (move_t, qbd_t, score)}."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT refit_date, move_t, qbd_t, score FROM {MACRO_FITTED_TBL} "
                     f"WHERE instrument=:inst AND params_hash=:ph ORDER BY refit_date"),
                conn, params={'inst': instrument, 'ph': params_hash},
                parse_dates=['refit_date']
            )
        return {row.refit_date: (row.move_t, row.qbd_t, row.score)
                for row in df.itertuples()}
    except Exception:
        return {}


def _save_fitted(engine, instrument: str, params_hash: str, fitted: dict):
    """Save newly computed fitted params to DB."""
    if not fitted:
        return
    rows = [{'refit_date': k, 'instrument': instrument,
              'params_hash': params_hash,
              'move_t': v[0], 'qbd_t': v[1], 'score': v[2]}
            for k, v in fitted.items() if v[0] is not None]
    if not rows:
        return
    df = pd.DataFrame(rows)
    try:
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {MACRO_FITTED_TBL} (
                    refit_date DATE, instrument VARCHAR(20),
                    params_hash VARCHAR(12), move_t FLOAT, qbd_t FLOAT, score FLOAT,
                    PRIMARY KEY (refit_date, instrument, params_hash)
                )
            """))
            # upsert
            for row in rows:
                conn.execute(text(f"""
                    INSERT INTO {MACRO_FITTED_TBL}
                        (refit_date, instrument, params_hash, move_t, qbd_t, score)
                    VALUES (:refit_date, :instrument, :params_hash, :move_t, :qbd_t, :score)
                    ON CONFLICT (refit_date, instrument, params_hash) DO NOTHING
                """), row)
    except Exception as e:
        import warnings
        warnings.warn(f"  Could not save fitted params: {e}")


def _load_cached_indicators(engine, params_hash: str) -> pd.DataFrame:
    """Load cached QBD + raw MOVE from DB. Returns empty df if none."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT * FROM {MACRO_INDICATOR_TBL} "
                     f"WHERE params_hash=:ph ORDER BY date"),
                conn, params={'ph': params_hash}, parse_dates=['date']
            )
        if not df.empty:
            df = df.set_index('date')
        return df
    except Exception:
        return pd.DataFrame()


def _load_cached_signals(engine, params_hash: str,
                         instrument: str) -> pd.DataFrame:
    """Load cached signal_df for one instrument. Returns empty df if none."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT * FROM {MACRO_SIGNAL_TBL} "
                     f"WHERE params_hash=:ph AND instrument=:inst ORDER BY date"),
                conn, params={'ph': params_hash, 'inst': instrument},
                parse_dates=['date']
            )
        if not df.empty:
            df = df.set_index('date').drop(
                columns=['instrument', 'params_hash'], errors='ignore')
        return df
    except Exception:
        return pd.DataFrame()


def _save_indicators(engine, qbd_s: pd.Series,
                     raw_move_s: pd.Series, params_hash: str):
    """Upsert QBD + raw MOVE to cache table."""
    from sqlalchemy import text
    common = qbd_s.dropna().index.intersection(raw_move_s.dropna().index)
    rows   = [{'date': d.strftime('%Y-%m-%d'), 'params_hash': params_hash,
               'qbd_val': float(qbd_s[d]), 'move_val_raw': float(raw_move_s[d])}
              for d in common]
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {MACRO_INDICATOR_TBL}
                (date, params_hash, qbd_val, move_val_raw)
            VALUES (:date, :params_hash, :qbd_val, :move_val_raw)
            ON CONFLICT (date, params_hash) DO UPDATE SET
                qbd_val=EXCLUDED.qbd_val,
                move_val_raw=EXCLUDED.move_val_raw
        """), rows)


def _save_signals(engine, signal_df: pd.DataFrame,
                  instrument: str, params_hash: str):
    """Upsert signal_df rows for one instrument to cache table."""
    from sqlalchemy import text
    cols = ['signal','signal_raw','move_val','qbd_val','move_t','qbd_t',
            'score','move_signal','qbd_signal',
            'nav_ratio','sharpe_raw','sharpe_hedged','sharpe_improvement','effectiveness']
    rows = []
    for dt, row in signal_df.iterrows():
        r = {'date': dt.strftime('%Y-%m-%d'), 'instrument': instrument,
             'params_hash': params_hash}
        for c in cols:
            v = row.get(c, None)
            r[c] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        rows.append(r)
    if not rows:
        return
    col_list   = ', '.join(['date','instrument','params_hash'] + cols)
    val_list   = ', '.join([f':{c}' for c in ['date','instrument','params_hash'] + cols])
    update_set = ', '.join([f'{c}=EXCLUDED.{c}' for c in cols])
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {MACRO_SIGNAL_TBL} ({col_list})
            VALUES ({val_list})
            ON CONFLICT (date, instrument, params_hash) DO UPDATE SET {update_set}
        """), rows)


def run_macro_hedge_cached(Pxs_df:        pd.DataFrame,
                           stocks_l:      list,
                           rates_l:       list,
                           hedges_l:      list,
                           move_grid:     list,
                           qbd_grid:      list,
                           engine,
                           move_hl:       int   = MOVE_HL,
                           move_mav:      int   = MOVE_MAV,
                           corr_win:      int   = CORR_WIN,
                           vol_hl:        int   = VOL_HL,
                           breadth_t:     float = BREADTH_T,
                           ref_dd_t:      float = REF_DD_T,
                           weight_years:  list  = WEIGHT_YEARS,
                           return_exp:    float = RETURN_EXP,
                           min_hold:      int   = HEDGE_MIN_HOLD,
                           signal_logic:  str   = SIGNAL_LOGIC,
                           hard_start:    str   = HARD_START,
                           refit_freq:    str   = '10B',
                           force_rebuild: bool  = False,
                           incremental_only: bool = True) -> dict:
    """
    Cached version of run_macro_hedge_multi.
    Builds QBD + MOVE once, fits signals per instrument, caches all results
    to PostgreSQL. On subsequent runs, only computes missing dates.

    Parameters
    ----------
    engine           : SQLAlchemy engine (e.g. from create_engine(...))
    force_rebuild    : if True, wipe cache and recompute everything
    incremental_only : if True (default), only compute the last date of Pxs_df
                       — fast daily update. Set False to compute all missing dates.
    REBUILD_QBD      : module-level bool (default False) — set True to force
                       QBD recomputation even if cached

    Returns same structure as run_macro_hedge_multi plus 'params_hash'.
    """
    import time
    from sqlalchemy import text

    params_hash = _macro_params_hash(
        move_hl, move_mav, corr_win, vol_hl, breadth_t,
        ref_dd_t, weight_years, return_exp, min_hold,
        signal_logic, refit_freq, move_grid, qbd_grid
    )

    print('=' * 72)
    print('  MACRO HEDGE CACHED BUILDER')
    print('=' * 72)
    print(f"\n  params_hash={params_hash}  force_rebuild={force_rebuild}  "
          f"incremental_only={incremental_only}")
    print(f"  Instruments: {hedges_l}")

    _ensure_macro_tables(engine)

    # -- Wipe cache if force_rebuild ------------------------------------------
    if force_rebuild:
        with engine.begin() as conn:
            conn.execute(text(
                f"DELETE FROM {MACRO_INDICATOR_TBL} WHERE params_hash=:ph"),
                {'ph': params_hash})
            conn.execute(text(
                f"DELETE FROM {MACRO_SIGNAL_TBL} WHERE params_hash=:ph"),
                {'ph': params_hash})
            conn.execute(text(
                f"DELETE FROM {MACRO_FITTED_TBL} WHERE params_hash=:ph"),
                {'ph': params_hash})
        print("  Cache wiped.")

    # -- Load or build shared indicators --------------------------------------
    ind_cache = _load_cached_indicators(engine, params_hash)
    all_dates = Pxs_df.loc[pd.Timestamp(hard_start):].index

    if not ind_cache.empty and not REBUILD_QBD:
        cached_ind_dates = set(ind_cache.index)
        missing_ind_dates = [d for d in all_dates if d not in cached_ind_dates]
        if incremental_only and missing_ind_dates:
            # Only compute the last date of Pxs_df
            last_date = all_dates[-1]
            missing_ind_dates = [last_date] if last_date not in cached_ind_dates else []
        print(f"\n  Indicators: {len(ind_cache)} cached, "
              f"{len(missing_ind_dates)} new dates"
              f"{' (incremental)' if incremental_only else ''}")
    else:
        missing_ind_dates = list(all_dates)
        print(f"\n  Indicators: building from scratch ({len(missing_ind_dates)} dates)")

    if missing_ind_dates or REBUILD_QBD:
        print('\n  Building QBD...')
        qbd_s = _build_qbd(Pxs_df, stocks_l, vol_hl=vol_hl,
                           breadth_t=breadth_t, mav_win=MAV_WIN,
                           hard_start=hard_start)

        # Raw MOVE (no corr filter)
        available  = [r for r in rates_l if r in Pxs_df.columns]
        rates_df   = Pxs_df[available].loc[pd.Timestamp(hard_start):]
        tenor_vols = []
        for col in available:
            s = rates_df[col].dropna()
            if len(s) < move_hl * 2:
                continue
            tenor_vols.append(
                s.ewm(halflife=move_hl, min_periods=move_hl//2).std() * 100)
        raw_move_base = pd.concat(tenor_vols, axis=1).mean(axis=1)
        raw_move_diff = (raw_move_base -
                         raw_move_base.rolling(move_mav, min_periods=1).mean())

        print('  Saving indicators to cache...')
        _save_indicators(engine, qbd_s, raw_move_diff, params_hash)

        # Merge with existing cache
        if not ind_cache.empty and not REBUILD_QBD:
            new_ind = pd.DataFrame({
                'qbd_val': qbd_s, 'move_val_raw': raw_move_diff
            }).loc[[d for d in missing_ind_dates if d in qbd_s.index]]
            ind_cache = pd.concat([ind_cache, new_ind]).sort_index()
        else:
            ind_cache = pd.DataFrame({
                'qbd_val': qbd_s, 'move_val_raw': raw_move_diff
            })
    else:
        qbd_s         = ind_cache['qbd_val'].rename('QBD')
        raw_move_diff = ind_cache['move_val_raw']
        # Reconstruct raw_move_base for corr-adjusted MOVE computation
        available  = [r for r in rates_l if r in Pxs_df.columns]
        rates_df   = Pxs_df[available].loc[pd.Timestamp(hard_start):]
        tenor_vols = []
        for col in available:
            s = rates_df[col].dropna()
            if len(s) < move_hl * 2:
                continue
            tenor_vols.append(
                s.ewm(halflife=move_hl, min_periods=move_hl//2).std() * 100)
        raw_move_base = pd.concat(tenor_vols, axis=1).mean(axis=1)

    # -- Per-instrument signals -----------------------------------------------
    results = {}
    missing_hedges = [h for h in hedges_l if h not in Pxs_df.columns]
    if missing_hedges:
        print(f"  WARNING: skipping {missing_hedges} — not in Pxs_df")
    hedges_l = [h for h in hedges_l if h in Pxs_df.columns]

    for inst_idx, inst in enumerate(hedges_l):
        print(f"\n  [{inst_idx+1}/{len(hedges_l)}] {inst} {'─'*40}")

        sig_cache = _load_cached_signals(engine, params_hash, inst)
        all_sig_dates = set(all_dates)

        if not sig_cache.empty:
            missing_sig = sorted(all_sig_dates - set(sig_cache.index))
            if incremental_only and missing_sig:
                last_date = all_dates[-1]
                missing_sig = [last_date] if last_date not in set(sig_cache.index) else []
            print(f"  Signal cache: {len(sig_cache)} dates cached, "
                  f"{len(missing_sig)} new"
                  f"{' (incremental)' if incremental_only else ''}")
        else:
            missing_sig = sorted(all_sig_dates)
            print(f"  Signal cache: empty — building from scratch")

        if missing_sig:
            # Corr-adjusted MOVE for this instrument
            ref_ret_inst = Pxs_df[inst].loc[pd.Timestamp(hard_start):].pct_change()
            move_chg     = raw_move_diff.diff() if hasattr(raw_move_diff, 'diff') \
                           else pd.Series(raw_move_diff).diff()
            # Use raw_move_base diff for correlation
            raw_base_chg = raw_move_base.diff()
            common_idx   = raw_base_chg.dropna().index.intersection(
                           ref_ret_inst.dropna().index)
            rolling_corr = (raw_base_chg.reindex(common_idx)
                            .rolling(corr_win, min_periods=corr_win//2)
                            .corr(ref_ret_inst.reindex(common_idx)))
            adj_factor   = np.ceil((-rolling_corr).clip(lower=0)).reindex(
                           raw_move_diff.index).fillna(0)
            move_s       = raw_move_diff * adj_factor
            move_s.name  = f'MOVE_{inst}'
            pct_active   = (adj_factor > 0).mean() * 100
            print(f"  MOVE filter active={pct_active:.1f}%")

            ref_s = Pxs_df[inst].dropna()
            signal_df_new, fitted = _build_signal(
                move_s, qbd_s, ref_s,
                move_grid, qbd_grid,
                ref_dd_t=ref_dd_t,
                weight_years=weight_years,
                return_exp=return_exp,
                min_hold=min_hold,
                signal_logic=signal_logic,
                hard_start=hard_start,
                refit_freq=refit_freq,
                engine=engine,
                instrument=inst,
                params_hash=params_hash,
            )

            # Rolling effectiveness
            ref_ret_a    = ref_s.pct_change().fillna(0).reindex(
                           signal_df_new.index).fillna(0)
            sig_lag      = signal_df_new['signal'].shift(1).fillna(0)
            hedged_ret_a = ref_ret_a * (1 - sig_lag)

            eff_dates = pd.date_range(start=signal_df_new.index[0],
                                      end=signal_df_new.index[-1],
                                      freq=refit_freq)
            eff_by_date = {}
            for rd in eff_dates:
                vals = _weighted_effectiveness(
                    ref_ret_a, hedged_ret_a, rd, weight_years)
                eff_by_date[rd] = vals

            sorted_eff = sorted(eff_by_date.keys())
            nav_r_s = pd.Series(np.nan, index=signal_df_new.index)
            sh_r_s  = pd.Series(np.nan, index=signal_df_new.index)
            sh_h_s  = pd.Series(np.nan, index=signal_df_new.index)
            sh_rt_s = pd.Series(np.nan, index=signal_df_new.index)
            eff_s   = pd.Series(np.nan, index=signal_df_new.index)
            for dt in signal_df_new.index:
                past = [d for d in sorted_eff if d <= dt]
                if past:
                    v = eff_by_date[past[-1]]
                    nav_r_s[dt] = v[0]; sh_r_s[dt]  = v[1]
                    sh_h_s[dt]  = v[2]; sh_rt_s[dt]  = v[3]; eff_s[dt] = v[4]

            signal_df_new['nav_ratio']     = nav_r_s
            signal_df_new['sharpe_raw']    = sh_r_s
            signal_df_new['sharpe_hedged'] = sh_h_s
            signal_df_new['sharpe_improvement'] = sh_rt_s
            signal_df_new['effectiveness'] = eff_s

            # Only save new dates
            to_save = signal_df_new.loc[
                [d for d in signal_df_new.index if d in set(missing_sig)]]
            print(f"  Saving {len(to_save)} new rows to cache...")
            _save_signals(engine, to_save, inst, params_hash)

            # Merge with existing cache
            if not sig_cache.empty:
                signal_df_full = pd.concat(
                    [sig_cache, signal_df_new.loc[
                        [d for d in signal_df_new.index
                         if d not in set(sig_cache.index)]]]
                ).sort_index()
            else:
                signal_df_full = signal_df_new

            move_s_out = move_s
        else:
            # Fully cached — rebuild corr-adj MOVE for reference
            ref_ret_inst = Pxs_df[inst].loc[pd.Timestamp(hard_start):].pct_change()
            raw_base_chg = raw_move_base.diff()
            common_idx   = raw_base_chg.dropna().index.intersection(
                           ref_ret_inst.dropna().index)
            rolling_corr = (raw_base_chg.reindex(common_idx)
                            .rolling(corr_win, min_periods=corr_win//2)
                            .corr(ref_ret_inst.reindex(common_idx)))
            adj_factor   = np.ceil((-rolling_corr).clip(lower=0)).reindex(
                           raw_move_diff.index).fillna(0)
            move_s_out   = (raw_move_diff * adj_factor)
            move_s_out.name = f'MOVE_{inst}'
            signal_df_full  = sig_cache
            fitted          = {}   # not reconstructed from cache
            print(f"  Fully cached — loaded {len(signal_df_full)} dates")

        # Summary line
        n_on  = signal_df_full['signal'].sum()
        n_tot = len(signal_df_full)
        eff_l = signal_df_full['effectiveness'].dropna().iloc[-1] \
                if signal_df_full['effectiveness'].notna().any() else np.nan
        sh_r  = signal_df_full['sharpe_raw'].dropna().iloc[-1] \
                if signal_df_full['sharpe_raw'].notna().any() else np.nan
        sh_h  = signal_df_full['sharpe_hedged'].dropna().iloc[-1] \
                if signal_df_full['sharpe_hedged'].notna().any() else np.nan
        nav_l = signal_df_full['nav_ratio'].dropna().iloc[-1] \
                if signal_df_full['nav_ratio'].notna().any() else np.nan
        print(f"  ON={n_on} ({n_on/n_tot*100:.1f}%)  nav={nav_l:.3f}  "
              f"sharpe={sh_r:.2f}->{sh_h:.2f}  eff={eff_l:.3f}")

        results[inst] = {
            'signal_df':    signal_df_full,
            'move_s':       move_s_out,
            'fitted':       fitted,
            'signal_logic': signal_logic,
        }

    # -- Summary table --------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  EFFECTIVENESS SUMMARY (as of {Pxs_df.index[-1].date()})")
    print(f"{'='*72}")
    print(f"  {'Instrument':<10}  {'Signal ON%':>10}  {'NAV Ratio':>10}  "
          f"{'Sharpe Raw':>11}  {'Sharpe Hdg':>11}  {'Sharpe Imp':>11}  {'Effectiveness':>14}")
    print(f"  {'-'*72}")
    for inst, res in results.items():
        df  = res['signal_df']
        n   = df['signal'].sum(); tot = len(df)
        nav = df['nav_ratio'].dropna().iloc[-1]          if df['nav_ratio'].notna().any()          else np.nan
        shr = df['sharpe_raw'].dropna().iloc[-1]         if df['sharpe_raw'].notna().any()         else np.nan
        shh = df['sharpe_hedged'].dropna().iloc[-1]      if df['sharpe_hedged'].notna().any()      else np.nan
        shi = df['sharpe_improvement'].dropna().iloc[-1] if df['sharpe_improvement'].notna().any() else np.nan
        eff = df['effectiveness'].dropna().iloc[-1]      if df['effectiveness'].notna().any()      else np.nan
        print(f"  {inst:<10}  {n/tot*100:>9.1f}%  {nav:>10.3f}  "
              f"{shr:>11.2f}  {shh:>11.2f}  {shi:>11.3f}  {eff:>14.3f}")

    return {
        'qbd_s':       qbd_s,
        'raw_move_s':  raw_move_diff,
        'results':     results,
        'params_hash': params_hash,
    }


# ================================================================================
# EFFECTIVENESS PLOT
# ================================================================================

def plot_effectiveness(multi_result: dict,
                       start:        str = None) -> None:
    """
    Plot rolling effectiveness for all hedging instruments over time.
    One line per instrument, coloured by final effectiveness rank.
    Also plots a horizontal reference line at 1.0 (no improvement).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    results = multi_result['results']
    if not results:
        print("No results to plot.")
        return

    st = pd.Timestamp(start) if start else None

    # Collect effectiveness series per instrument
    eff_dict = {}
    for inst, res in results.items():
        df = res['signal_df']
        if 'effectiveness' in df.columns and df['effectiveness'].notna().any():
            s = df['effectiveness']
            if st:
                s = s.loc[st:]
            eff_dict[inst] = s.dropna()

    if not eff_dict:
        print("No effectiveness data to plot.")
        return

    # Sort instruments by final effectiveness value for legend ordering
    final_eff = {inst: s.iloc[-1] for inst, s in eff_dict.items() if len(s) > 0}
    sorted_insts = sorted(final_eff, key=lambda x: final_eff[x], reverse=True)

    # Colour map — green for strong, red for weak
    n = len(sorted_insts)
    cmap = cm.get_cmap('RdYlGn', n)
    colors = {inst: cmap(i / max(n-1, 1)) for i, inst in enumerate(sorted_insts)}

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#FAFAF9')
    ax.set_facecolor('#FAFAF9')
    ax.grid(color='#D3D1C7', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reference line at 1.0
    ax.axhline(1.0, color='#888780', lw=1.0, linestyle='--', alpha=0.7,
               label='baseline (=1.0)')

    for inst in sorted_insts:
        s = eff_dict[inst]
        ax.plot(s.index.to_numpy(), s.to_numpy(),
                color=colors[inst], lw=1.0, alpha=0.85,
                label=f"{inst} ({final_eff[inst]:.3f})")

    ax.set_title('Rolling weighted effectiveness by hedging instrument',
                 fontsize=11, fontweight='500', color='#2C2C2A')
    ax.set_ylabel('Effectiveness  (nav_ratio × (1 + sharpe_improvement))',
                  fontsize=9, color='#5F5E5A')
    ax.legend(fontsize=8, loc='upper left', ncol=2,
              framealpha=0.85, edgecolor='#D3D1C7')

    # Set x-axis to actual data range
    all_dates = pd.concat(eff_dict.values()).index
    ax.set_xlim(all_dates.min(), all_dates.max())

    plt.tight_layout()
    plt.show()

    # Quick summary table
    print(f"\n  {'Instrument':<10}  {'Final Eff':>10}  {'Avg Eff':>10}  {'Min Eff':>10}  {'Max Eff':>10}")
    print(f"  {'-'*52}")
    for inst in sorted_insts:
        s = eff_dict[inst]
        print(f"  {inst:<10}  {s.iloc[-1]:>10.3f}  {s.mean():>10.3f}  "
              f"{s.min():>10.3f}  {s.max():>10.3f}")
