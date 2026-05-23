"""
================================================================================
SECTOR / GROUP INDEX BUILDER
================================================================================
Builds aggregate price indexes for groups of stocks defined by a classification
column in Sectors_df, weighted equally or by market cap.

USAGE
-----
    from index_builder import main
    main(Pxs_df, Sectors_df)

INPUTS
------
    Pxs_df      : pd.DataFrame — DatetimeIndex rows, ticker columns (no " US")
    Sectors_df  : pd.DataFrame — ticker index, one column per classification
                  criterion (same ticker format as Pxs_df)
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from sqlalchemy import create_engine, text

# ── DB connection ─────────────────────────────────────────────────────────────
CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
ENGINE = create_engine(CONNECTION_STRING)

MAX_INDEXES = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(ticker: str) -> str:
    return ticker.split(' ')[0].upper()


def _add_quarters(period: str, n: int) -> str:
    year, q = int(period[:4]), int(period[5])
    q += n
    while q > 4: q -= 4; year += 1
    while q < 1: q += 4; year -= 1
    return f"{year}Q{q}"


def _last_completed_cal_qtr(today: pd.Timestamp) -> pd.Timestamp:
    """Return the end-date of the most recently completed calendar quarter."""
    q = (today.month - 1) // 3   # 0=Q1, 1=Q2, 2=Q3, 3=Q4
    if q == 0:
        return pd.Timestamp(today.year - 1, 12, 31)
    ends = {1: pd.Timestamp(today.year, 3, 31),
            2: pd.Timestamp(today.year, 6, 30),
            3: pd.Timestamp(today.year, 9, 30)}
    return ends[q]


def _cal_qtr_ends_sequence(base_date: pd.Timestamp, n: int) -> list:
    """
    Return a list of n calendar quarter-end dates going backwards from base_date.
    base_date is assumed to already be a quarter-end (e.g. March 31).
    """
    dates = []
    dt = base_date
    for _ in range(n):
        dates.append(dt)
        # Step back one calendar quarter
        if dt.month == 3:
            dt = pd.Timestamp(dt.year - 1, 12, 31)
        elif dt.month == 6:
            dt = pd.Timestamp(dt.year, 3, 31)
        elif dt.month == 9:
            dt = pd.Timestamp(dt.year, 6, 30)
        else:  # December
            dt = pd.Timestamp(dt.year, 9, 30)
    return dates


# ── Share count loading ───────────────────────────────────────────────────────

def _load_shares(tickers: list, n_quarters: int = 60) -> pd.DataFrame:
    """
    For each ticker, build a sparse daily share-count series by mapping
    fiscal quarters (going back from FEQ-1) to calendar quarter-end dates.
    Returns DataFrame indexed by date, columns = tickers.
    """
    today    = pd.Timestamp.today().normalize()
    base_cal = _last_completed_cal_qtr(today)
    cal_ends = _cal_qtr_ends_sequence(base_cal, n_quarters)

    # Load FEPs
    clean_tks = [_clean(t) for t in tickers]
    with ENGINE.connect() as conn:
        fep_df = pd.read_sql(text("""
            SELECT ticker, first_estimated_period
            FROM estimation_status
            WHERE category = 'income' AND ticker = ANY(:tks)
        """), conn, params={'tks': clean_tks})

    fep_map = dict(zip(fep_df['ticker'], fep_df['first_estimated_period']))

    # Load dilutedAverageShares (latest snapshot for each ticker)
    with ENGINE.connect() as conn:
        shares_df = pd.read_sql(text("""
            SELECT ticker, period, value,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, period
                       ORDER BY download_date DESC
                   ) AS rn
            FROM income_data
            WHERE metric_name = 'dilutedAverageShares'
              AND ticker = ANY(:tks)
        """), conn, params={'tks': clean_tks})

    shares_df = shares_df[shares_df['rn'] == 1].drop(columns='rn')
    shares_df['value'] = pd.to_numeric(shares_df['value'], errors='coerce')
    shares_by_ticker = {}

    for raw_t in tickers:
        t   = _clean(raw_t)
        fep = fep_map.get(t)
        if fep is None:
            shares_by_ticker[raw_t] = pd.Series(dtype=float)
            continue

        tk_shares = shares_df[shares_df['ticker'] == t].set_index('period')['value']
        sparse = {}
        for i, cal_dt in enumerate(cal_ends):
            fiscal_qtr = _add_quarters(fep, -(i + 1))  # FEP-1, FEP-2, ...
            val = tk_shares.get(fiscal_qtr, np.nan)
            if not pd.isna(val):
                sparse[cal_dt] = float(val)

        if not sparse:
            shares_by_ticker[raw_t] = pd.Series(dtype=float)
        else:
            shares_by_ticker[raw_t] = pd.Series(sparse).sort_index()

    result = pd.DataFrame(shares_by_ticker)
    return result


def _build_shares_daily(tickers: list, date_index: pd.DatetimeIndex,
                         n_quarters: int = 60) -> pd.DataFrame:
    """
    Build a daily share-count panel aligned to date_index via ffill then bfill.
    """
    sparse = _load_shares(tickers, n_quarters=n_quarters)
    if sparse.empty:
        return pd.DataFrame(index=date_index, columns=tickers, dtype=float)

    all_dates = date_index.union(sparse.index).sort_values()
    daily = sparse.reindex(all_dates).ffill().bfill().reindex(date_index)
    return daily


# ── Index construction ────────────────────────────────────────────────────────

def _build_index(tickers: list, Pxs_df: pd.DataFrame,
                  weighting: str, cutoff: pd.Timestamp,
                  n_quarters: int = 60) -> pd.Series:
    """
    Build a price index (base 100) for a group of tickers.

    weighting : 'equal' or 'mktcap'
    """
    # Filter to tickers present in Pxs_df
    valid_tks = [t for t in tickers if t in Pxs_df.columns]
    if not valid_tks:
        return pd.Series(dtype=float)

    px = Pxs_df[valid_tks].loc[Pxs_df.index >= cutoff].copy()
    if px.empty:
        return pd.Series(dtype=float)

    rets = px.pct_change()

    if weighting == 'equal':
        # Equal weight: mean return across stocks with valid price each day
        idx_rets = rets.mean(axis=1, skipna=True)

    else:  # mktcap
        print(f"    Loading share counts for {len(valid_tks)} stocks...",
              end=' ', flush=True)
        shares = _build_shares_daily(valid_tks, px.index, n_quarters=n_quarters)
        print("done")

        # Market cap = price × shares (lagged one day for weighting)
        mktcap = px * shares
        mktcap_lag = mktcap.shift(1)
        total_lag  = mktcap_lag.sum(axis=1).replace(0, np.nan)
        weights    = mktcap_lag.div(total_lag, axis=0)
        idx_rets   = (rets * weights).sum(axis=1, skipna=True)

    # Rebase to 100
    idx_rets.iloc[0] = 0.0
    index_level = 100 * (1 + idx_rets).cumprod()
    return index_level


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_indexes(index_dict: dict, criterion: str,
                   weighting: str, n_years: float):
    """Plot all selected indexes on one chart, rebased to 100."""
    fig, ax = plt.subplots(figsize=(15, 8))
    colors  = plt.cm.tab10.colors

    for i, (label, idx_s) in enumerate(index_dict.items()):
        s     = idx_s.dropna()
        if s.empty:
            continue
        color = colors[i % len(colors)]
        ax.plot(s.index.to_numpy(), s.values,
                color=color, linewidth=1.8, label=label)

        # ATH distance label at end of line
        ath      = s.max()
        current  = s.iloc[-1]
        pct_diff = (current / ath - 1) * 100
        label_str = f"{pct_diff:.1f}%"
        ax.annotate(
            label_str,
            xy=(s.index[-1], current),
            xytext=(6, 0), textcoords='offset points',
            fontsize=8.5, color=color, va='center',
        )

    weight_str = "Equal Weight" if weighting == 'equal' else "Market Cap Weight"
    ax.set_title(f"{criterion} — {weight_str} Indexes  "
                 f"({n_years:.0f}Y, base=100)",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Index Level (base = 100)", fontsize=11)
    ax.axhline(100, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.85)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.15)
    fig.text(0.13, 0.01,
             "End-of-line label = % distance from all-time high (0.0% = at ATH)",
             fontsize=8, color='gray', style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()


# ── Prompts ───────────────────────────────────────────────────────────────────

def _prompt_criterion(Sectors_df: pd.DataFrame) -> str:
    cols = list(Sectors_df.columns)
    print("\nSELECT CLASSIFICATION CRITERION")
    print("=" * 45)
    for i, c in enumerate(cols, 1):
        n_groups = Sectors_df[c].nunique()
        print(f"  {i:<3} {c}  ({n_groups} groups)")
    while True:
        raw = input("\nEnter number: ").strip()
        try:
            n = int(raw)
            if 1 <= n <= len(cols):
                return cols[n - 1]
            print(f"  Choose between 1 and {len(cols)}")
        except ValueError:
            print("  Enter a number")


def _prompt_labels(Sectors_df: pd.DataFrame, criterion: str,
                    Pxs_df: pd.DataFrame) -> list:
    """List all labels, show stock count, return selected label names."""
    mapping = Sectors_df[criterion].dropna()
    labels  = sorted(mapping.unique())

    print(f"\nGROUPS UNDER '{criterion}'")
    print("=" * 55)
    for i, lbl in enumerate(labels, 1):
        tks_in_pxs = [t for t in mapping[mapping == lbl].index
                      if t in Pxs_df.columns]
        print(f"  {i:<4} {str(lbl):<35} ({len(tks_in_pxs)} stocks in Pxs_df)")

    print(f"\nSELECT INDEXES  (up to {MAX_INDEXES}, blank to stop)")
    print("-" * 40)
    selected = []
    while len(selected) < MAX_INDEXES:
        raw = input(f"  Index {len(selected) + 1}: ").strip()
        if raw == '':
            if not selected:
                print("  Select at least one.")
                continue
            break
        try:
            n = int(raw)
            if not (1 <= n <= len(labels)):
                print(f"  Choose between 1 and {len(labels)}")
                continue
            lbl = labels[n - 1]
            if lbl in selected:
                print(f"  '{lbl}' already selected")
                continue
            selected.append(lbl)
            print(f"  ✓ '{lbl}' added ({len(selected)}/{MAX_INDEXES})")
        except ValueError:
            print("  Enter a number")

    return selected


def _prompt_weighting() -> str:
    print("\nWEIGHTING METHOD")
    print("  1 - Equal weight")
    print("  2 - Market cap weight")
    while True:
        raw = input("Choice [1/2]: ").strip()
        if raw == '1': return 'equal'
        if raw == '2': return 'mktcap'
        print("  Enter 1 or 2")


def _prompt_years() -> float:
    print("\nHISTORICAL WINDOW")
    while True:
        raw = input("  Years of history [default=5]: ").strip()
        if raw == '':
            return 5.0
        try:
            n = float(raw)
            if n > 0:
                return n
            print("  Must be positive")
        except ValueError:
            print("  Enter a number")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(Pxs_df: pd.DataFrame, Sectors_df: pd.DataFrame):
    """
    Build and plot group indexes.

    Parameters
    ----------
    Pxs_df     : price panel — DatetimeIndex, ticker columns (no ' US' suffix)
    Sectors_df : classification panel — ticker index, one column per criterion
    """
    print("\n" + "=" * 60)
    print("  SECTOR / GROUP INDEX BUILDER")
    print("=" * 60)

    criterion = _prompt_criterion(Sectors_df)
    selected  = _prompt_labels(Sectors_df, criterion, Pxs_df)
    weighting = _prompt_weighting()
    n_years   = _prompt_years()

    today  = pd.Timestamp.today().normalize()
    cutoff = today - pd.DateOffset(years=n_years)
    # Quarters to load (history + small buffer)
    n_qtrs = int(n_years * 4) + 8

    print(f"\nBuilding {len(selected)} index(es)...\n")

    mapping     = Sectors_df[criterion].dropna()
    index_dict  = {}

    for lbl in selected:
        tickers = [t for t in mapping[mapping == lbl].index
                   if t in Pxs_df.columns]
        print(f"  [{lbl}]  {len(tickers)} stocks", end='')
        if not tickers:
            print(" — no price data, skipping")
            continue
        print()
        idx_s = _build_index(tickers, Pxs_df, weighting, cutoff,
                              n_quarters=n_qtrs)
        if idx_s.empty or idx_s.notna().sum() < 2:
            print(f"    → insufficient data, skipping")
            continue
        index_dict[lbl] = idx_s
        ath_pct = (idx_s.iloc[-1] / idx_s.max() - 1) * 100
        print(f"    → {idx_s.notna().sum()} days  "
              f"current={idx_s.iloc[-1]:.1f}  "
              f"ATH={idx_s.max():.1f}  "
              f"dist={ath_pct:.1f}%")

    if not index_dict:
        print("No valid indexes to plot.")
        return

    _plot_indexes(index_dict, criterion, weighting, n_years)


if __name__ == "__main__":
    print("Import and call main(Pxs_df, Sectors_df) from your notebook.")
