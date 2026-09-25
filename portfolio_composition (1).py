#!/usr/bin/env python
# coding: utf-8
"""
portfolio_composition.py
========================
Management view of a backtest's portfolios: how the book's SECTOR and SUB-SECTOR
composition evolved, and what it looks like today.

Pick one of the 10 strategies built by mvo_backtest, then the script draws:

  1. stacked-area evolution of composition by SECTOR
  2. stacked-area evolution of composition by SUB-SECTOR
  3. donut of the CURRENT composition by SECTOR
  4. donut + sorted bar of the CURRENT composition by SUB-SECTOR

READABILITY (sub-sectors are ~30 buckets, which is too many for a pie)
  * top-N buckets by average weight are kept, the rest collapse into "Other"
  * every bucket keeps the SAME COLOUR across all charts
  * slices below LABEL_MIN get no inline label (they stay in the legend)
  * for sub-sectors a sorted HORIZONTAL BAR chart is drawn as well -- it shows
    all buckets legibly, which a 30-slice pie cannot

WEIGHTS
  Portfolios are target weights recorded at each rebalance. Between rebalances
  the book is held, so weights are forward-filled onto a daily grid. Pass
  Pxs_df to let positions DRIFT with prices between rebalances instead, which
  is what was actually held.

USAGE
    from portfolio_composition import plot_composition
    plot_composition(results, sectors_s, subsec_map_s)              # prompts
    plot_composition(results, sectors_s, subsec_map_s, strategy='Excl')
    plot_composition(results, sectors_s, subsec_map_s, strategy='Excl',
                     Pxs_df=Pxs_df)                                  # with drift

    # charts only (default). For the underlying panels:
    panels = plot_composition(..., return_data=True)   # {'Sector': df, 'Sub-sector': df}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- display settings -------------------------------------------------------
TOP_N_SECTOR    = 12      # sectors kept individually in the charts
TOP_N_SUBSECTOR = 12      # sub-sectors kept individually ("Other" collects the rest)
LABEL_MIN       = 0.03    # donut slices below this get no inline label
OTHER_LABEL     = 'Other'
RESIDUAL_LABEL  = 'SPX (cap residual)'
FIGSIZE_TS      = (15, 6)
FIGSIZE_PIE     = (14, 7)

# strategy label -> results key holding {date: weights Series}
STRATEGY_KEYS = {
    'Baseline':  'baseline_weights_by_date',
    'Alpha':     'alpha_weights_by_date',
    'MVO':       'mvo_weights_by_date',
    'Hybrid':    'hybrid_weights_by_date',
    'Smart':     'smart_weights_by_date',
    'Dynamic':   'dyn_weights_by_date',
    'Dyn+Hedge': 'hedge_weights_by_date',
    'DD Policy': 'dd_weights_by_date',
    'Excl':      'excl_weights_by_date',
    'MVO+Hedge': 'mvo_hedge_weights_by_date',
}


# =============================================================================
# Data preparation
# =============================================================================
def _pick_strategy(results, strategy=None):
    """Resolve the strategy label, prompting if not supplied."""
    avail = [k for k, v in STRATEGY_KEYS.items()
             if results.get(v)]                      # non-empty history only
    if not avail:
        raise ValueError("No portfolio weights found in `results`. Note that in "
                         "incremental mode weights_by_date only covers the dates "
                         "computed in that run -- use mode='rebuild' for history.")
    if strategy is not None:
        if strategy not in STRATEGY_KEYS:
            raise ValueError(f"Unknown strategy '{strategy}'. "
                             f"Choose from: {list(STRATEGY_KEYS)}")
        if strategy not in avail:
            raise ValueError(f"'{strategy}' has no recorded weights in this run.")
        return strategy

    print("\n" + "=" * 54)
    print("  SELECT STRATEGY")
    print("=" * 54)
    for i, nm in enumerate(avail, 1):
        print(f"  {i:>3}. {nm}   ({len(results[STRATEGY_KEYS[nm]])} rebalance dates)")
    print("=" * 54)
    while True:
        raw = input("  strategy number: ").strip()
        if raw.lstrip('+-').isdigit() and 1 <= int(raw) <= len(avail):
            return avail[int(raw) - 1]
        print("  Invalid selection")


def _weights_panel(wbd, Pxs_df=None):
    """
    {date: Series} -> daily DataFrame (dates x tickers) of weights actually held.

    Without Pxs_df the target weights are forward-filled (the book is held
    unchanged between rebalances). With Pxs_df the positions DRIFT with prices
    between rebalances and are renormalised, which is what was really held.
    """
    dates = sorted(wbd)
    panel = pd.DataFrame({d: wbd[d] for d in dates}).T.sort_index()
    panel = panel.fillna(0.0)
    if Pxs_df is None:
        grid = pd.date_range(panel.index[0], panel.index[-1], freq='B')
        return panel.reindex(panel.index.union(grid)).ffill().reindex(grid).ffill()

    px = Pxs_df.reindex(columns=panel.columns).astype(float)
    grid = px.index[(px.index >= panel.index[0]) & (px.index <= panel.index[-1])]
    out, cur, cur_dt = {}, None, None
    for dt in grid:
        if dt in panel.index:
            cur, cur_dt = panel.loc[dt], dt
        if cur is None:
            continue
        rel = (px.loc[dt] / px.loc[cur_dt]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        v = (cur * rel)
        out[dt] = v / v.sum() if v.sum() > 0 else v
    return pd.DataFrame(out).T.fillna(0.0)


def _group_panel(panel, mapping):
    """Aggregate a ticker-level weight panel into buckets (no collapsing)."""
    m = pd.Series(mapping)
    grp = pd.Series({t: m.get(t, RESIDUAL_LABEL if t == 'SPX' else 'Unmapped')
                     for t in panel.columns})
    agg = panel.T.groupby(grp).sum().T
    return agg[agg.mean().sort_values(ascending=False).index]


def _collapse(agg, top_n):
    """Keep the top_n buckets by AVERAGE weight, collapse the rest into `Other`.
    Used for the time series, where too many bands are unreadable."""
    keep = list(agg.mean().sort_values(ascending=False).index[:top_n])
    if agg.shape[1] <= len(keep):
        return agg[keep]
    other = agg[[c for c in agg.columns if c not in keep]].sum(axis=1)
    out = agg[keep].copy()
    out[OTHER_LABEL] = other
    return out


def _collapse_snapshot(cur, min_slice=0.02):
    """For the CURRENT mix, collapse only genuinely tiny holdings, chosen by
    TODAY's weight (not the historical average), so the snapshot reflects what
    is actually held now."""
    cur = cur[cur > 0].sort_values(ascending=False)
    small = cur[cur < min_slice]
    if len(small) <= 1:
        return cur
    out = cur[cur >= min_slice].copy()
    out[OTHER_LABEL] = small.sum()
    return out


def _colour_map(buckets):
    """Stable bucket -> colour, so a bucket looks the same in every chart."""
    base = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)
    cols = {}
    for i, b in enumerate(buckets):
        cols[b] = (0.72, 0.72, 0.72) if b == OTHER_LABEL else base[i % len(base)]
    return cols


# =============================================================================
# Charts
# =============================================================================
def _plot_area(agg, title, cols, ax):
    ax.stackplot(agg.index.to_numpy(), [agg[c].to_numpy() for c in agg.columns],
                 labels=list(agg.columns),
                 colors=[cols[c] for c in agg.columns], linewidth=0)
    ax.set_title(title, fontsize=11, fontweight='500')
    ax.set_ylabel('weight')
    ax.set_ylim(0, max(1.0, float(agg.sum(axis=1).max())))
    ax.margins(x=0)
    ax.grid(color='#D3D1C7', linewidth=0.4, axis='y')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.005, 0.5),
              fontsize=8, frameon=False, ncol=1)


def _plot_donut(series, title, cols, ax):
    s = series[series > 0].sort_values(ascending=False)
    lab = [n if v >= LABEL_MIN else '' for n, v in s.items()]
    ax.pie(s.values, labels=lab, colors=[cols.get(n, '#BBBBBB') for n in s.index],
           startangle=90, counterclock=False,
           autopct=lambda p: f'{p:.0f}%' if p >= LABEL_MIN * 100 else '',
           pctdistance=0.78, textprops={'fontsize': 8},
           wedgeprops={'width': 0.42, 'edgecolor': 'white', 'linewidth': 1})
    ax.set_title(title, fontsize=11, fontweight='500')
    ax.axis('equal')


def _plot_bars(series, title, cols, ax):
    s = series[series > 0].sort_values()
    ax.barh(range(len(s)), s.values * 100,
            color=[cols.get(n, '#BBBBBB') for n in s.index])
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel('weight (%)')
    ax.set_title(title, fontsize=11, fontweight='500')
    ax.grid(color='#D3D1C7', linewidth=0.4, axis='x')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    for i, v in enumerate(s.values * 100):
        ax.text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=7, color='#5F5E5A')


# =============================================================================
# Entry point
# =============================================================================
def plot_composition(results, sectors_s, subsec_map_s=None, strategy=None,
                     Pxs_df=None, top_n_sector=TOP_N_SECTOR,
                     top_n_subsector=TOP_N_SUBSECTOR, show_table=True,
                     return_data=False):
    """Draw the four composition views for one strategy. See module docstring.

    Returns None by default, so calling this in a notebook cell does not echo
    the whole aggregated panel. Pass return_data=True to get
    {'Sector': DataFrame, 'Sub-sector': DataFrame} back for further analysis.
    """
    name = _pick_strategy(results, strategy)
    wbd = results[STRATEGY_KEYS[name]]
    panel = _weights_panel(wbd, Pxs_df=Pxs_df)
    drift = 'drifted' if Pxs_df is not None else 'target (held)'
    print(f"\n  {name}: {len(wbd)} rebalances, {panel.index[0].date()} -> "
          f"{panel.index[-1].date()}, weights = {drift}")

    levels = [('Sector', sectors_s, top_n_sector)]
    if subsec_map_s is not None:
        levels.append(('Sub-sector', subsec_map_s, top_n_subsector))

    out = {}
    for label, mapping, top_n in levels:
        full = _group_panel(panel, mapping)          # every bucket
        agg  = _collapse(full, top_n)                # for the time series
        cur  = _collapse_snapshot(full.iloc[-1])     # for the donut (today's mix)
        cols = _colour_map(list(full.columns) + [OTHER_LABEL])
        out[label] = full

        # ── evolution ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=FIGSIZE_TS)
        fig.patch.set_facecolor('#FAFAF9'); ax.set_facecolor('#FAFAF9')
        _plot_area(agg, f'{name} — composition by {label.lower()} over time '
                        f'(top {top_n} + {OTHER_LABEL.lower()})', cols, ax)
        plt.tight_layout(); plt.show()

        # ── current snapshot ─────────────────────────────────────────────────
        is_sub = label.startswith('Sub')
        fig, axes = plt.subplots(1, 2 if is_sub else 1,
                                 figsize=FIGSIZE_PIE if is_sub else (7.5, 7))
        fig.patch.set_facecolor('#FAFAF9')
        axes = np.atleast_1d(axes)
        for a in axes:
            a.set_facecolor('#FAFAF9')
        _plot_donut(cur, f'{name} — current {label.lower()} mix '
                         f'({panel.index[-1].date()})', cols, axes[0])
        if is_sub:
            # a 30-slice pie is unreadable; the sorted bar shows every bucket
            _plot_bars(full.iloc[-1],
                       f'{name} — current {label.lower()} mix (all buckets)',
                       cols, axes[1])
        plt.tight_layout(); plt.show()

        if show_table:
            t = full.iloc[-1]
            t = t[t > 0].sort_values(ascending=False)
            print(f"\n  CURRENT {label.upper()} MIX ({panel.index[-1].date()})")
            print(f"  {'-'*46}")
            for b, v in t.items():
                print(f"  {b[:34]:<36}{v*100:>7.2f}%")
            print(f"  {'-'*46}\n  {'TOTAL':<36}{t.sum()*100:>7.2f}%\n")

    return out if return_data else None


if __name__ == "__main__":
    print(__doc__)
