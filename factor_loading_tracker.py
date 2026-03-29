"""
factor_loading_tracker.py
=========================
Tracks portfolio-level factor loadings on each rebalancing date.

For each rebalance date t and factor k:
    loading_k_t = (X_t' w_t)[k]  =  sum_i  w_i_t * X_ik_t

where X_ik_t is the raw z-scored factor exposure of stock i on date t,
and w_i_t is the portfolio weight.

This gives the empirical distribution of factor loadings that the
pure alpha-driven portfolio naturally produces — the baseline before
any optimization constraints are imposed.

Assumes all factor_model_step1, portfolio_risk_decomp functions
are already loaded in the Jupyter kernel.

Entry point
-----------
    loading_df = run_factor_loading_tracker(
        port_df, Pxs_df, sectors_s,
        volumeTrd_df=None, model_version='v2'
    )

Parameters
----------
    port_df       : pd.DataFrame — dates x tickers, % weight allocations
                    (same format as pnl_attribution.py input)
    Pxs_df        : pd.DataFrame — price/macro panel
    sectors_s     : pd.Series   — ticker -> sector label
    volumeTrd_df  : pd.DataFrame or None — vol scalars for idio momentum
    model_version : 'v1' or 'v2'

Returns
-------
    loading_df : pd.DataFrame — dates x factors, portfolio-level loadings
                 (X'w)[k] for each factor k on each rebalance date
"""

# ===============================================================================
# PARAMETERS
# ===============================================================================

FLT_REBAL_SAMPLE = 5     # sample every N rebalance dates for speed
                          # (set to 1 for every date, higher for faster runs)

# ===============================================================================
# HELPERS
# ===============================================================================

def _flt_build_X_snapshot(dt, universe, factor_names, sec_cols,
                           Pxs_df, sectors_s, volumeTrd_df, model_version):
    """Build raw factor exposure matrix X on a single date. Reuses _rd_build_X."""
    pxs_to_dt = Pxs_df.loc[:dt]
    if len(pxs_to_dt) < BETA_WINDOW // 2:
        return None
    try:
        return _rd_build_X(
            universe, factor_names, sec_cols,
            pxs_to_dt, sectors_s, volumeTrd_df,
            model_version=model_version
        ).fillna(0.0)
    except Exception as e:
        warnings.warn(f"  X build failed for {dt.date()}: {e}")
        return None


def _flt_get_weights(port_df, dt, universe):
    """Extract normalised weight vector for date dt."""
    if dt not in port_df.index:
        return None
    w_raw = port_df.loc[dt].reindex(universe).fillna(0.0)
    if w_raw.sum() == 0:
        return None
    return w_raw / w_raw.sum()


# ===============================================================================
# MAIN TRACKER
# ===============================================================================

def run_factor_loading_tracker(port_df, Pxs_df, sectors_s,
                                volumeTrd_df=None, model_version='v2'):
    """
    Compute portfolio factor loadings on each rebalancing date.

    Parameters
    ----------
    port_df       : pd.DataFrame — dates x tickers, % weight allocations
    Pxs_df        : pd.DataFrame — price/macro panel
    sectors_s     : pd.Series   — ticker -> sector label
    volumeTrd_df  : optional vol scalars
    model_version : 'v1' or 'v2'

    Returns
    -------
    loading_df : pd.DataFrame — dates x factors
    """
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    port_df   = port_df.loc[:, ~port_df.columns.duplicated(keep='first')]

    print(f"\nFactor Loading Tracker [{model_version}] — starting...\n")

    # ── Universe & factor structure ───────────────────────────────────────────
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    F, factor_names, sec_cols = _rd_build_F(model_version=model_version)

    print(f"  Universe: {len(universe)} stocks  |  Factors: {len(factor_names)}")

    # ── X snapshot dates (month-ends + latest, reuse _rd_month_end_dates) ────
    rebal_dates = port_df.index.sort_values()
    x_snap_dates = _rd_month_end_dates(rebal_dates, rebal_dates[-1])

    print(f"  Building X snapshots on {len(x_snap_dates)} month-end dates...")
    X_snapshots = {}
    for i, xdt in enumerate(x_snap_dates):
        xdf = _flt_build_X_snapshot(
            xdt, universe, factor_names, sec_cols,
            Pxs_df, sectors_s, volumeTrd_df, model_version
        )
        if xdf is not None:
            X_snapshots[xdt] = xdf
        if (i+1) % 6 == 0:
            print(f"    {i+1}/{len(x_snap_dates)} snapshots built", end='\r')
    print(f"\n  X snapshots built: {len(X_snapshots)}")

    snapshot_dates = sorted(X_snapshots.keys())

    # ── Compute loadings on each rebalance date ───────────────────────────────
    print("  Computing factor loadings...")
    results  = {}
    n        = len(rebal_dates)

    for i, dt in enumerate(rebal_dates):
        if (i+1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n}] {dt.date()}", end='\r')

        w = _flt_get_weights(port_df, dt, universe)
        if w is None:
            continue

        # Most recent X snapshot on or before dt
        valid = [d for d in snapshot_dates if d <= dt]
        if not valid:
            continue
        X_t = X_snapshots[valid[-1]]

        # Portfolio factor loadings: (X'w) — one number per factor
        Xw = X_t.values.T @ w.values   # (K,)
        results[dt] = dict(zip(factor_names, Xw))

    loading_df            = pd.DataFrame(results).T
    loading_df.index.name = 'date'
    loading_df            = loading_df[factor_names]

    print(f"\n  Loadings computed: {len(loading_df)} dates\n")

    # ── Summary statistics ────────────────────────────────────────────────────
    _flt_print_summary(loading_df, factor_names, sec_cols)
    _flt_plot(loading_df, factor_names, sec_cols)

    return loading_df


# ===============================================================================
# SUMMARY
# ===============================================================================

def _flt_print_summary(loading_df, factor_names, sec_cols, pct=[5, 25, 50, 75, 95]):
    """Print loading distribution per factor."""
    print("\n" + "=" * 82)
    print("  FACTOR LOADING DISTRIBUTION  (X'w)[k] across rebalance dates")
    print("=" * 82)
    print(f"  {'Factor':<28}  {'Mean':>8}  {'Std':>8}  "
          f"{'p5':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}")
    print("  " + "-" * 80)

    # Group: structural, macro, sectors, alpha
    groups = [
        ('Structural',  ['Beta', 'Size']),
        ('Macro',       MACRO_COLS),
        ('Sectors',     sec_cols),
        ('Alpha',       ['Quality', 'SI', 'GK_Vol', 'Idio_Mom', 'Value', 'OU']),
    ]

    for gname, factors in groups:
        sub = [f for f in factors if f in loading_df.columns]
        if not sub:
            continue
        print(f"\n  -- {gname} --")
        for f in sub:
            s   = loading_df[f].dropna()
            ps  = np.percentile(s, pct)
            print(f"  {f:<28}  {s.mean():>+8.3f}  {s.std():>8.3f}  "
                  + "  ".join(f"{p:>+8.3f}" for p in ps))

    print("\n" + "=" * 82 + "\n")


# ===============================================================================
# PLOTS
# ===============================================================================

def _flt_plot(loading_df, factor_names, sec_cols):
    """
    Four panels:
      1. Structural & alpha factor loadings over time
      2. Macro factor loadings over time
      3. Sector loadings over time
      4. Loading distribution boxplots — key factors only
    """
    COLORS = {
        'Beta'     : '#378ADD',
        'Size'     : '#7F77DD',
        'Quality'  : '#7F77DD',
        'SI'       : '#D85A30',
        'GK_Vol'   : '#EF9F27',
        'Idio_Mom' : '#1D9E75',
        'Value'    : '#D85A30',
        'OU'       : '#888780',
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={'height_ratios': [2.5, 1.5, 2, 2]})
    fig.suptitle(f"Portfolio Factor Loading Tracker",
                 fontsize=13, fontweight='500', color='#2C2C2A', y=0.99)
    fig.patch.set_facecolor('#FAFAF9')
    for ax in axes:
        ax.set_facecolor('#FAFAF9')

    dates = loading_df.index.to_numpy()

    # ── Panel 1: Structural + Alpha ───────────────────────────────────────────
    ax = axes[0]
    key_factors = ['Beta', 'Size', 'Quality', 'Idio_Mom', 'GK_Vol', 'Value', 'OU']
    for f in key_factors:
        if f not in loading_df.columns:
            continue
        lw = 1.8 if f == 'Beta' else 1.1
        ax.plot(dates, loading_df[f].values,
                label=f, color=COLORS.get(f, '#888780'),
                linewidth=lw, alpha=0.85)
    ax.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax.set_ylabel("Loading (X'w)[k]", fontsize=10, color='#5F5E5A')
    ax.set_title("Structural & Alpha factor loadings over time",
                 fontsize=11, fontweight='500')
    ax.legend(fontsize=8, ncol=4, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 2: Macro ────────────────────────────────────────────────────────
    ax2   = axes[1]
    macro = [c for c in MACRO_COLS if c in loading_df.columns]
    cmap  = plt.cm.tab10.colors
    for i, f in enumerate(macro):
        ax2.plot(dates, loading_df[f].values,
                 label=f, color=cmap[i % len(cmap)],
                 linewidth=0.9, alpha=0.8)
    ax2.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax2.set_ylabel("Loading", fontsize=10, color='#5F5E5A')
    ax2.set_title("Macro factor loadings over time",
                  fontsize=11, fontweight='500')
    ax2.legend(fontsize=7, ncol=4, loc='upper left', framealpha=0.85)
    ax2.grid(color='#D3D1C7', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Panel 3: Sectors ──────────────────────────────────────────────────────
    ax3  = axes[2]
    secs = [c for c in sec_cols if c in loading_df.columns]
    cmap2 = plt.cm.tab20.colors
    for i, f in enumerate(secs):
        ax3.plot(dates, loading_df[f].values,
                 label=f, color=cmap2[i % len(cmap2)],
                 linewidth=0.8, alpha=0.7)
    ax3.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax3.set_ylabel("Loading", fontsize=10, color='#5F5E5A')
    ax3.set_title("Sector loadings over time",
                  fontsize=11, fontweight='500')
    ax3.legend(fontsize=6, ncol=5, loc='upper left', framealpha=0.85)
    ax3.grid(color='#D3D1C7', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ── Panel 4: Boxplots — key factors ──────────────────────────────────────
    ax4  = axes[3]
    box_factors = ['Beta', 'Size', 'Quality', 'SI',
                   'GK_Vol', 'Idio_Mom', 'Value', 'OU']
    box_data    = [loading_df[f].dropna().values
                   for f in box_factors if f in loading_df.columns]
    box_labels  = [f for f in box_factors if f in loading_df.columns]
    box_colors  = [COLORS.get(f, '#888780') for f in box_labels]

    bp = ax4.boxplot(box_data, patch_artist=True, notch=False,
                     medianprops={'color': 'white', 'linewidth': 2},
                     whiskerprops={'color': '#888780'},
                     capprops={'color': '#888780'},
                     flierprops={'marker': 'o', 'markersize': 3,
                                 'alpha': 0.4, 'color': '#888780'})
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax4.axhline(0, color='#888780', linewidth=0.8, linestyle='--')
    ax4.set_xticks(range(1, len(box_labels) + 1))
    ax4.set_xticklabels(box_labels, fontsize=9, rotation=30, ha='right')
    ax4.set_ylabel("Loading (X'w)[k]", fontsize=10, color='#5F5E5A')
    ax4.set_title("Factor loading distribution (boxplots across rebalance dates)",
                  fontsize=11, fontweight='500')
    ax4.grid(axis='y', color='#D3D1C7', linewidth=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
