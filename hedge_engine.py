"""
hedge_engine.py
===============
Standalone macro hedge backtest engine.

Runs in parallel with the main portfolio strategy. Hedge positions are
opened/closed independently of portfolio rebalances, with net P&L swept
into the portfolio NAV at each rebalancing date.

HEDGE TRIGGER
-------------
  Signal ON  : QQQ signal = 1 OR SPY signal = 1  (1-day lag applied)
  Signal OFF : QQQ signal = 0 AND SPY signal = 0

INSTRUMENT SELECTION (at each activation event)
------------------------------------------------
  1. Rank all instruments in hedges_l by rolling correlation to live portfolio
  2. Select top 3 with correlation >= corr_floor AND effectiveness >= eff_floor
  3. Always add QQQ and SPY if they pass both tests and not already selected
  4. Final set: 0-5 instruments
  5. Total hedge = min(n * hedge_ratio, max_hedge), allocated by beta*effectiveness

HEDGING ACCOUNT
---------------
  - Tracks short P&L independently from main portfolio
  - At each portfolio rebalancing date: balance swept to portfolio NAV, resets to 0
  - Hedge positions can span multiple portfolio rebalancing dates

PARAMETERS (all overridable)
-----------------------------
  BETA_WINDOW    = 63    rolling window for portfolio beta to hedge instruments
  CORR_WINDOW    = 63    rolling window for portfolio-instrument correlation
  EFF_MAV_WINDOW = 20    MAV window for smoothing effectiveness
  EFF_FLOOR      = 1.0   minimum effectiveness to qualify as hedge instrument
  CORR_FLOOR     = 0.50  minimum correlation to qualify
  HEDGE_RATIO    = 0.25  hedge size per instrument (as % of portfolio NAV)
  MAX_HEDGE      = 0.50  maximum total hedge size

USAGE
-----
    from hedge_engine import run_hedge_backtest

    hedge_results = run_hedge_backtest(
        Pxs_df               = Pxs_df,
        multi                = multi,           # from run_macro_hedge_cached
        portfolio_weights    = dyn_weights_by_date,  # {date: pd.Series}
        rebal_dates          = sorted(dyn_weights_by_date.keys()),
        hedges_l             = hedges_l,
    )

    # Keys:
    #   'hedge_account_by_date' : dict {date: float}  cumulative hedge P&L
    #   'hedge_log'             : list of event dicts
    #   'nav_hedged'            : pd.Series  portfolio NAV adjusted for hedge account
    #   'nav_portfolio'         : pd.Series  base portfolio NAV (pre-hedge)
    #   'summary'               : pd.DataFrame  per-year per-instrument breakdown
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Default parameters ────────────────────────────────────────────────────────
BETA_WINDOW    = 63     # rolling window for beta calculation
CORR_WINDOW    = 63     # rolling window for correlation ranking
EFF_MAV_WINDOW = 20     # MAV window for smoothing effectiveness
EFF_FLOOR      = 1.0    # minimum effectiveness score to qualify
CORR_FLOOR     = 0.50   # minimum correlation to portfolio to qualify
HEDGE_RATIO    = 0.25   # hedge size per instrument (fraction of portfolio NAV)
MAX_HEDGE      = 0.50   # maximum total hedge (fraction of portfolio NAV)
TRIGGER_ASSETS = ['QQQ', 'SPY']  # assets whose signal triggers hedge on/off


# ================================================================================
# HELPERS
# ================================================================================

def _portfolio_returns(weights_by_date: dict,
                       Pxs_df:          pd.DataFrame,
                       all_dates:        pd.DatetimeIndex) -> pd.Series:
    """
    Build daily portfolio return series using fixed rebalance weights.
    Between rebalances, apply the last rebalance weights to daily stock returns.
    """
    rebal_dates = sorted(weights_by_date.keys())
    port_rets   = {}

    for i, dt in enumerate(all_dates):
        # Find last rebalance date <= dt
        past = [d for d in rebal_dates if d <= dt]
        if not past:
            continue
        last_rebal = past[-1]
        w = weights_by_date[last_rebal]
        tickers = [t for t in w.index if t in Pxs_df.columns]
        if not tickers:
            continue
        prev_dt_idx = all_dates.get_loc(dt)
        if prev_dt_idx == 0:
            continue
        prev_dt = all_dates[prev_dt_idx - 1]
        if prev_dt not in Pxs_df.index or dt not in Pxs_df.index:
            continue
        px_prev = Pxs_df.loc[prev_dt, tickers]
        px_cur  = Pxs_df.loc[dt,      tickers]
        ret     = (px_cur / px_prev - 1).fillna(0)
        port_rets[dt] = (w.reindex(tickers).fillna(0) * ret).sum()

    return pd.Series(port_rets, name='portfolio_ret')


def _compute_beta(port_ret_s: pd.Series,
                  inst_ret_s: pd.Series,
                  window:     int) -> float:
    """OLS beta of portfolio returns to instrument returns over last `window` days."""
    common = port_ret_s.dropna().index.intersection(inst_ret_s.dropna().index)
    common = common[-window:] if len(common) >= window else common
    if len(common) < window // 2:
        return np.nan
    p = port_ret_s.reindex(common)
    h = inst_ret_s.reindex(common)
    var_h = h.var()
    return p.cov(h) / var_h if var_h > 0 else np.nan


def _compute_corr(port_ret_s: pd.Series,
                  inst_ret_s: pd.Series,
                  window:     int) -> float:
    """Rolling correlation of portfolio to instrument over last `window` days."""
    common = port_ret_s.dropna().index.intersection(inst_ret_s.dropna().index)
    common = common[-window:] if len(common) >= window else common
    if len(common) < window // 2:
        return np.nan
    return port_ret_s.reindex(common).corr(inst_ret_s.reindex(common))


def _get_effectiveness(signal_df: pd.DataFrame,
                       dt:        pd.Timestamp,
                       mav_win:   int) -> float:
    """Smoothed effectiveness at date dt (MAV of last mav_win values)."""
    if 'effectiveness' not in signal_df.columns:
        return np.nan
    past = signal_df['effectiveness'].loc[:dt].dropna()
    if past.empty:
        return np.nan
    return past.iloc[-mav_win:].mean()


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def run_hedge_backtest(Pxs_df:            pd.DataFrame,
                       multi:             dict,
                       portfolio_weights: dict,
                       rebal_dates:       list,
                       hedges_l:          list,
                       beta_window:       int   = BETA_WINDOW,
                       corr_window:       int   = CORR_WINDOW,
                       eff_mav_window:    int   = EFF_MAV_WINDOW,
                       eff_floor:         float = EFF_FLOOR,
                       corr_floor:        float = CORR_FLOOR,
                       hedge_ratio:       float = HEDGE_RATIO,
                       max_hedge:         float = MAX_HEDGE,
                       trigger_assets:    list  = TRIGGER_ASSETS) -> dict:
    """
    Run hedge backtest in parallel with main portfolio.

    Parameters
    ----------
    Pxs_df            : price DataFrame
    multi             : output of run_macro_hedge_cached
    portfolio_weights : {date: pd.Series} rebalance weights
    rebal_dates       : sorted list of portfolio rebalancing dates
    hedges_l          : list of hedge instrument tickers
    (all other params: see module defaults)

    Returns
    -------
    dict with keys:
      'hedge_account_by_date' : {date: float}  hedge account balance each day
      'hedge_sweep_by_date'   : {date: float}  amount swept to portfolio at rebal
      'hedge_log'             : list of event dicts
      'summary_by_year'       : pd.DataFrame
      'summary_by_instrument' : pd.DataFrame
      'port_ret_s'            : pd.Series  daily portfolio returns
    """
    results      = multi['results']
    all_dates    = Pxs_df.index
    rebal_set    = set(rebal_dates)

    # Validate trigger assets
    for ta in trigger_assets:
        if ta not in results:
            print(f"  WARNING: trigger asset {ta} not in multi results — "
                  f"will be treated as always OFF")

    # Validate hedges
    hedges_l = [h for h in hedges_l if h in results and h in Pxs_df.columns]
    missing  = [h for h in hedges_l if h not in results or h not in Pxs_df.columns]
    if missing:
        print(f"  WARNING: skipping {missing} — not in multi results or Pxs_df")

    print('=' * 72)
    print('  HEDGE BACKTEST ENGINE')
    print('=' * 72)
    print(f"\n  Trigger assets : {trigger_assets}")
    print(f"  Hedge universe : {hedges_l}")
    print(f"  Parameters     : beta_win={beta_window}d  corr_win={corr_window}d  "
          f"eff_mav={eff_mav_window}d")
    print(f"                   eff_floor={eff_floor}  corr_floor={corr_floor:.0%}  "
          f"hedge_ratio={hedge_ratio:.0%}  max_hedge={max_hedge:.0%}")

    # Build portfolio return series
    print("\n  Building portfolio return series...")
    port_ret_s = _portfolio_returns(portfolio_weights, Pxs_df, all_dates)

    # Pre-compute instrument return series
    inst_ret = {inst: Pxs_df[inst].pct_change().fillna(0)
                for inst in hedges_l if inst in Pxs_df.columns}

    # ── Main loop ─────────────────────────────────────────────────────────────
    hedge_account      = 0.0    # running hedge account balance (as % of NAV)
    hedge_sweep        = {}     # {rebal_date: amount swept to portfolio}
    hedge_account_hist = {}     # {date: balance}
    hedge_log          = []     # all hedge events

    active_hedges      = {}     # {inst: {'entry_px': float, 'weight': float}}
    hedge_on           = False  # current hedge state
    prev_trigger       = 0      # previous day's trigger signal

    print("\n  Running hedge simulation...\n")

    for dt in all_dates:
        if dt not in port_ret_s.index:
            hedge_account_hist[dt] = hedge_account
            continue

        # ── Sweep hedge account at portfolio rebalancing dates ────────────────
        # Only sweep when no active hedges — deferred if hedges still open
        if dt in rebal_set and not hedge_on:
            if hedge_account != 0.0:
                sweep_amt        = hedge_account
                hedge_sweep[dt]  = hedge_sweep.get(dt, 0) + sweep_amt
                print(f"  {'='*68}")
                print(f"  PORTFOLIO REBALANCE {dt.date()}  |  "
                      f"Hedge account sweep: {sweep_amt:+.4%} of NAV")
                print(f"  {'='*68}")
                hedge_account = 0.0
        elif dt in rebal_set and hedge_on:
            print(f"  PORTFOLIO REBALANCE {dt.date()}  |  "
                  f"Hedges active — sweep deferred")

        # ── Compute trigger signal (1-day lag) ────────────────────────────────
        trigger_on = False
        for ta in trigger_assets:
            if ta in results:
                sig_df = results[ta]['signal_df']
                if dt in sig_df.index:
                    # 1-day lag: use previous day's signal
                    prev_dates = sig_df.index[sig_df.index < dt]
                    if len(prev_dates) > 0:
                        prev_sig = sig_df.loc[prev_dates[-1], 'signal']
                        if prev_sig == 1:
                            trigger_on = True
                            break

        # ── Hedge activation ──────────────────────────────────────────────────
        if trigger_on and not hedge_on:
            # New hedge event — select instruments
            selected = _select_hedge_instruments(
                dt, hedges_l, results, port_ret_s, inst_ret,
                beta_window, corr_window, eff_mav_window,
                eff_floor, corr_floor, trigger_assets
            )

            if selected:
                hedge_on      = True
                active_hedges = {}
                n_inst        = len(selected)
                total_hedge   = min(n_inst * hedge_ratio, max_hedge)

                # Allocate by beta * effectiveness, normalized to total_hedge
                raw_scores = {inst: d['beta'] * d['effectiveness']
                              for inst, d in selected.items()
                              if not np.isnan(d['beta']) and
                                 not np.isnan(d['effectiveness'])}
                total_score = sum(raw_scores.values())

                print(f"\n  {'─'*68}")
                print(f"  HEDGE OPEN  {dt.date()}  |  "
                      f"{n_inst} instruments  total_hedge={total_hedge:.1%}")
                print(f"  {'─'*68}")
                print(f"  {'Instrument':<10}  {'Corr':>7}  {'Beta':>7}  "
                      f"{'Eff (MAV)':>10}  {'Score':>8}  {'Weight':>8}")
                print(f"  {'-'*60}")

                for inst, d in selected.items():
                    raw_sc = raw_scores.get(inst, 0)
                    w      = (raw_sc / total_score * total_hedge
                              if total_score > 0 else total_hedge / n_inst)
                    active_hedges[inst] = {
                        'weight':   w,
                        'entry_px': Pxs_df.loc[dt, inst]
                                    if dt in Pxs_df.index else np.nan,
                        'entry_dt': dt,
                        'beta':     d['beta'],
                        'corr':     d['corr'],
                        'eff':      d['effectiveness'],
                        'score':    raw_sc,
                    }
                    print(f"  {inst:<10}  {d['corr']:>7.3f}  {d['beta']:>7.3f}  "
                          f"{d['effectiveness']:>10.3f}  {raw_sc:>8.3f}  {w:>7.2%}")

                print(f"  {'─'*60}")
                print(f"  Total hedge: {total_hedge:.1%}  "
                      f"(allocated {sum(v['weight'] for v in active_hedges.values()):.1%})")

                hedge_log.append({
                    'date': dt, 'event': 'OPEN',
                    'instruments': list(active_hedges.keys()),
                    'total_hedge': total_hedge,
                    'details': {inst: dict(v) for inst, v in active_hedges.items()},
                })
            else:
                print(f"\n  {dt.date()}  Trigger ON but no qualifying instruments — "
                      f"no hedge activated")

        # ── Hedge unwind ──────────────────────────────────────────────────────
        elif not trigger_on and hedge_on:
            hedge_on = False
            print(f"\n  {'─'*68}")
            print(f"  HEDGE CLOSE {dt.date()}")
            print(f"  {'─'*68}")
            print(f"  {'Instrument':<10}  {'Entry':>12}  {'Exit':>12}  "
                  f"{'Weight':>8}  {'P&L':>10}")
            print(f"  {'-'*57}")

            total_pnl = 0.0
            for inst, h in active_hedges.items():
                if dt in Pxs_df.index and h['entry_px'] and not np.isnan(h['entry_px']):
                    exit_px  = Pxs_df.loc[dt, inst]
                    inst_ret_pct = (exit_px / h['entry_px'] - 1)
                    # Short position: profit when instrument falls
                    pnl      = -inst_ret_pct * h['weight']
                    total_pnl += pnl
                    print(f"  {inst:<10}  {h['entry_px']:>12.4f}  {exit_px:>12.4f}  "
                          f"{h['weight']:>8.2%}  {pnl:>+9.4%}")
                else:
                    pnl = 0.0
                    print(f"  {inst:<10}  {'n/a':>12}  {'n/a':>12}  "
                          f"{h['weight']:>8.2%}  {'n/a':>10}")

            hedge_account += total_pnl
            print(f"  {'─'*57}")
            print(f"  Episode P&L : {total_pnl:+.4%}")
            print(f"  Hedge acct  : {hedge_account:+.4%} (pending sweep at next rebal)")

            days_held = (dt - active_hedges[list(active_hedges.keys())[0]]['entry_dt']).days
            hedge_log.append({
                'date': dt, 'event': 'CLOSE',
                'instruments': list(active_hedges.keys()),
                'total_pnl': total_pnl,
                'days_held': days_held,
                'hedge_account': hedge_account,
                'details': {inst: dict(v) for inst, v in active_hedges.items()},
            })
            active_hedges = {}

        # ── Mark-to-market active hedges (display only, not booked to account) ─
        elif hedge_on and active_hedges:
            daily_pnl = 0.0
            prev_dates_all = all_dates[all_dates < dt]
            if len(prev_dates_all) > 0:
                prev_dt = prev_dates_all[-1]
                for inst, h in active_hedges.items():
                    if (inst in Pxs_df.columns and
                            prev_dt in Pxs_df.index and dt in Pxs_df.index):
                        r = Pxs_df.loc[dt, inst] / Pxs_df.loc[prev_dt, inst] - 1
                        daily_pnl += -r * h['weight']
            # MTM tracked separately — not added to hedge_account
            # hedge_account only updates at close

        hedge_account_hist[dt] = hedge_account

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_hedge_summary(hedge_log, Pxs_df, results)

    return {
        'hedge_account_by_date': hedge_account_hist,
        'hedge_sweep_by_date':   hedge_sweep,
        'hedge_log':             hedge_log,
        'port_ret_s':            port_ret_s,
        'active_hedges':         active_hedges,
    }


# ================================================================================
# INSTRUMENT SELECTION
# ================================================================================

def _select_hedge_instruments(dt:            pd.Timestamp,
                               hedges_l:     list,
                               results:      dict,
                               port_ret_s:   pd.Series,
                               inst_ret:     dict,
                               beta_window:  int,
                               corr_window:  int,
                               eff_mav_win:  int,
                               eff_floor:    float,
                               corr_floor:   float,
                               trigger_assets: list) -> dict:
    """
    Select qualifying hedge instruments at activation date dt.
    Returns dict {inst: {'beta', 'corr', 'effectiveness'}} for selected instruments.
    """
    # Compute corr and effectiveness for all instruments
    scores = {}
    for inst in hedges_l:
        if inst not in inst_ret or inst not in results:
            continue
        corr = _compute_corr(port_ret_s, inst_ret[inst], corr_window)
        eff  = _get_effectiveness(results[inst]['signal_df'], dt, eff_mav_win)
        beta = _compute_beta(port_ret_s, inst_ret[inst], beta_window)
        scores[inst] = {'corr': corr, 'eff': eff, 'beta': beta,
                        'effectiveness': eff}

    # Sort by correlation descending
    ranked = sorted(scores.items(),
                    key=lambda x: x[1]['corr'] if not np.isnan(x[1]['corr']) else -999,
                    reverse=True)

    selected = {}

    # Always check trigger assets first (QQQ, SPY)
    for ta in trigger_assets:
        if ta in scores:
            d = scores[ta]
            corr = d['corr'] if not np.isnan(d.get('corr', np.nan)) else 0
            eff  = d['eff']  if not np.isnan(d.get('eff',  np.nan)) else 0
            if corr >= corr_floor and eff >= eff_floor:
                selected[ta] = d

    # Add top 3 from correlation ranking (excluding already selected)
    n_from_ranking = 0
    for inst, d in ranked:
        if inst in selected:
            continue
        if n_from_ranking >= 3:
            break
        corr = d['corr'] if not np.isnan(d.get('corr', np.nan)) else 0
        eff  = d['eff']  if not np.isnan(d.get('eff',  np.nan)) else 0
        if corr < corr_floor:
            break   # ranked by corr, so all remaining will also fail
        if eff >= eff_floor:
            selected[inst] = d
            n_from_ranking += 1

    return selected


# ================================================================================
# SUMMARY
# ================================================================================

def _print_hedge_summary(hedge_log:  list,
                         Pxs_df:    pd.DataFrame,
                         results:   dict) -> None:
    """Print per-year and per-instrument hedge summary."""
    if not hedge_log:
        print("\n  No hedge events recorded.")
        return

    close_events = [e for e in hedge_log if e['event'] == 'CLOSE']
    if not close_events:
        print("\n  No completed hedge episodes.")
        return

    print(f"\n{'='*72}")
    print(f"  HEDGE SUMMARY")
    print(f"{'='*72}")

    # Overall stats
    all_pnl    = [e['total_pnl'] for e in close_events]
    n_pos      = sum(1 for p in all_pnl if p > 0)
    print(f"\n  Total episodes  : {len(close_events)}")
    print(f"  Hit rate        : {n_pos/len(close_events)*100:.1f}%  "
          f"({n_pos} positive / {len(close_events)-n_pos} negative)")
    print(f"  Total P&L       : {sum(all_pnl):+.4%}")
    print(f"  Avg P&L/episode : {np.mean(all_pnl):+.4%}")
    print(f"  Avg days held   : {np.mean([e.get('days_held',0) for e in close_events]):.1f}")

    # Per-year breakdown
    print(f"\n  {'─'*68}")
    print(f"  PER-YEAR BREAKDOWN")
    print(f"  {'─'*68}")
    print(f"  {'Year':<6}  {'Episodes':>9}  {'Hit Rate':>9}  "
          f"{'Total P&L':>10}  {'Avg P&L':>10}  {'Avg Days':>9}")
    print(f"  {'-'*60}")

    years = sorted(set(e['date'].year for e in close_events))
    for yr in years:
        yr_ev  = [e for e in close_events if e['date'].year == yr]
        yr_pnl = [e['total_pnl'] for e in yr_ev]
        yr_pos = sum(1 for p in yr_pnl if p > 0)
        yr_days= np.mean([e.get('days_held', 0) for e in yr_ev])
        print(f"  {yr:<6}  {len(yr_ev):>9}  "
              f"{yr_pos/len(yr_ev)*100:>8.1f}%  "
              f"{sum(yr_pnl):>+9.4%}  "
              f"{np.mean(yr_pnl):>+9.4%}  "
              f"{yr_days:>9.1f}")

    # Per-instrument breakdown (only instruments that were used)
    inst_events = {}
    for e in close_events:
        for inst in e.get('instruments', []):
            if inst not in inst_events:
                inst_events[inst] = []
            w   = e['details'].get(inst, {}).get('weight', 0)
            px_entry = e['details'].get(inst, {}).get('entry_px', np.nan)
            entry_dt = e['details'].get(inst, {}).get('entry_dt', e['date'])
            # Recompute instrument-level P&L from entry to close
            close_dt = e['date']
            if (px_entry and not np.isnan(px_entry) and
                    inst in Pxs_df.columns and close_dt in Pxs_df.index):
                exit_px  = Pxs_df.loc[close_dt, inst]
                inst_pnl = -(exit_px / px_entry - 1) * w
            else:
                inst_pnl = np.nan
            inst_events[inst].append({
                'pnl': inst_pnl, 'weight': w,
                'eff': e['details'].get(inst, {}).get('eff', np.nan),
                'beta': e['details'].get(inst, {}).get('beta', np.nan),
            })

    if inst_events:
        print(f"\n  {'─'*68}")
        print(f"  PER-INSTRUMENT BREAKDOWN")
        print(f"  {'─'*68}")
        print(f"  {'Instrument':<12}  {'Uses':>5}  {'Hit Rate':>9}  "
              f"{'Total P&L':>10}  {'Avg Weight':>11}  {'Avg Beta':>9}  {'Avg Eff':>8}")
        print(f"  {'-'*68}")
        for inst in sorted(inst_events.keys()):
            evs     = inst_events[inst]
            pnls    = [e['pnl'] for e in evs if not np.isnan(e['pnl'])]
            pos     = sum(1 for p in pnls if p > 0)
            avg_w   = np.mean([e['weight'] for e in evs])
            avg_b   = np.nanmean([e['beta'] for e in evs])
            avg_e   = np.nanmean([e['eff']  for e in evs])
            hit_str = f"{pos/len(pnls)*100:.1f}%" if pnls else "n/a"
            pnl_str = f"{sum(pnls):+.4%}" if pnls else "n/a"
            print(f"  {inst:<12}  {len(evs):>5}  {hit_str:>9}  "
                  f"{pnl_str:>10}  {avg_w:>10.2%}  {avg_b:>9.3f}  {avg_e:>8.3f}")
