#!/usr/bin/env python
# coding: utf-8

"""
Sector consolidation of valuation metrics (market-cap weighted)
===============================================================
Consolidates ONE selected metric from `valuation_metrics_v2` into sector-level
market-cap-weighted averages, across all calculation dates.

Usage
-----
    import pandas as pd
    from sector_consolidate import consolidate_metric

    # sector_series: index = ticker (same format as v2), value = sector label
    sector_series = pd.Series({...})           # e.g. {'NVDA US': 'Tech', ...}

    result = consolidate_metric(sector_series)  # prompts for the metric
    # or skip the prompt:
    result = consolidate_metric(sector_series, metric='Trailing P/E')

Returns a DataFrame: index = calculation dates, columns = sectors,
values = market-cap-weighted average of the chosen metric.

Method (per date, per sector)
-----------------------------
  weighted_avg = sum(size_i * metric_i) / sum(size_i)
  over stocks i that are (a) in BOTH sector_series and v2 on that date, and
  (b) have a NON-NaN metric AND a non-NaN, positive size. Weights are implicitly
  renormalised to that subset (a stock missing the metric simply does not
  participate for that sector/date). Negative metric values ARE included
  (e.g. negative P/E is averaged in, per spec -- not filtered).
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
ENGINE = create_engine(CONNECTION_STRING)

TABLE       = "valuation_metrics_v2"
SIZE_COL    = "size"        # market-cap column used as the weight
KEY_COLS    = ("date", "ticker")


def _table_columns(table):
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :t
        ORDER BY ordinal_position
    """
    with ENGINE.connect() as c:
        return [r[0] for r in c.execute(text(q), {"t": table}).fetchall()]


def _metric_columns(table):
    """All numeric metric columns eligible for consolidation: everything except
    the keys and the weight column itself."""
    cols = _table_columns(table)
    exclude = set(KEY_COLS) | {SIZE_COL}
    return [c for c in cols if c not in exclude]


def _prompt_metrics(metrics):
    """Prompt repeatedly; each valid number adds a metric. Empty input (Enter)
    finishes. Returns the selected metric names in selection order (no dupes)."""
    print("\n" + "=" * 60)
    print("  SELECT METRIC(S) TO CONSOLIDATE")
    print("=" * 60)
    for i, m in enumerate(metrics, 1):
        print(f"  {i:>3}. {m}")
    print("=" * 60)
    print("  Enter a number to add a metric; press Enter alone when done.")
    chosen = []
    while True:
        raw = input(f"  Add metric (1-{len(metrics)}, Enter=done): ").strip()
        if raw == "":
            if chosen:
                return chosen
            print("  select at least one metric.")
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(metrics):
            m = metrics[int(raw) - 1]
            if m in chosen:
                print(f"  '{m}' already selected.")
            else:
                chosen.append(m)
                print(f"  + {m}   (selected: {len(chosen)})")
        else:
            print("  invalid selection.")


def _weighted_by_sector(df, metric_col):
    """Given a frame with columns [date, ticker, sector, <metric_col>, wt],
    return a [dates x sectors] DataFrame of market-cap-weighted averages of
    metric_col. Rows with NaN metric or non-positive weight are dropped; weights
    renormalise over the survivors. Negative metric values are kept."""
    sub = df[["date", "sector", metric_col, "wt"]].copy()
    before = len(sub)
    sub = sub[sub[metric_col].notna() & sub["wt"].notna() & (sub["wt"] > 0)]
    dropped = before - len(sub)
    if sub.empty:
        return pd.DataFrame(), 0, dropped
    sub["wx"] = sub["wt"] * sub[metric_col]
    grp = sub.groupby(["date", "sector"], sort=True).agg(
        sum_wx=("wx", "sum"),
        sum_w =("wt", "sum"),
    ).reset_index()
    grp["wavg"] = grp["sum_wx"] / grp["sum_w"]
    out = grp.pivot(index="date", columns="sector", values="wavg")
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns.name = None
    out.index.name = "date"
    return out, int(sub["date"].groupby(sub["date"]).ngroups), dropped


def consolidate_metric(sector_series, metric=None):
    """Market-cap-weighted sector consolidation of one OR MORE v2 metrics.

    sector_series : pd.Series, index=ticker (v2 format), value=sector label.
    metric        : None  -> prompt (select one or many; Enter finishes);
                    str   -> a single metric (no prompt);
                    list  -> several metrics (no prompt).
    Returns       : a single DataFrame [dates x sectors] if ONE metric is
                    selected; a LIST of such DataFrames (selection order) if
                    MORE THAN ONE. Each returned DataFrame carries its metric
                    name in df.attrs['metric'].
    """
    if not isinstance(sector_series, pd.Series):
        raise TypeError("sector_series must be a pandas Series (index=ticker, "
                        "value=sector).")

    all_metrics = _metric_columns(TABLE)
    if SIZE_COL not in _table_columns(TABLE):
        raise RuntimeError(f"weight column '{SIZE_COL}' not found in {TABLE}.")

    # ---- resolve the requested metric(s) ----
    if metric is None:
        selected = _prompt_metrics(all_metrics)
    elif isinstance(metric, str):
        selected = [metric]
    elif isinstance(metric, (list, tuple)):
        selected = list(metric)
    else:
        raise TypeError("metric must be None, a str, or a list/tuple of str.")
    bad = [m for m in selected if m not in all_metrics]
    if bad:
        raise ValueError(f"metric(s) not in {TABLE}: {bad}. Available: {all_metrics}")
    if not selected:
        print("  No metric selected."); return pd.DataFrame()

    print(f"\n  Consolidating {len(selected)} metric(s) weighted by "
          f"'{SIZE_COL}': {selected}")

    # ---- ONE DB read: date, ticker, size, and every selected metric ----
    metric_cols_sql = ", ".join(f'"{m}"' for m in selected)
    q = f'SELECT "date", "ticker", "{SIZE_COL}" AS wt, {metric_cols_sql} FROM {TABLE}'
    df = pd.read_sql_query(text(q), ENGINE)
    if df.empty:
        print("  v2 returned no rows."); return pd.DataFrame()

    # ---- intersection of tickers (series ∩ v2) + sector map (done ONCE) ----
    sec = sector_series.copy()
    sec.index = sec.index.astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df = df[df["ticker"].isin(sec.index)]
    if df.empty:
        print("  No overlap between sector_series tickers and v2 tickers.")
        return pd.DataFrame()
    df["sector"] = df["ticker"].map(sec)
    print(f"  Tickers used (series ∩ v2): {df['ticker'].nunique()}")

    # ---- compute each metric off the shared frame ----
    results = []
    for m in selected:
        out, _, dropped = _weighted_by_sector(df, m)
        if out.empty:
            print(f"  [{m}] no usable rows (metric/positive-weight) -- empty result.")
        else:
            out.attrs["metric"] = m
            print(f"  [{m}] {out.shape[0]} date(s) x {out.shape[1]} sector(s)"
                  + (f"; {dropped} row(s) dropped" if dropped else ""))
        results.append(out)

    # ---- single metric -> DataFrame; multiple -> list of DataFrames ----
    return results[0] if len(results) == 1 else results

if __name__ == "__main__":
    # Minimal self-test of the weighting math on synthetic data (no DB).
    demo = pd.DataFrame({
        "date":   ["2026-01-01"] * 4 + ["2026-01-11"] * 4,
        "ticker": ["A", "B", "C", "D"] * 2,
        "metric_val": [10.0, 20.0, np.nan, -5.0,   12.0, 22.0, 30.0, -4.0],
        "wt":         [100.0, 300.0, 50.0, 200.0,  100.0, 300.0, 0.0, 200.0],
    })
    sec = pd.Series({"A": "Tech", "B": "Tech", "C": "Fin", "D": "Fin"})
    demo["ticker"] = demo["ticker"].astype(str)
    demo = demo[demo["metric_val"].notna() & demo["wt"].notna() & (demo["wt"] > 0)]
    demo["sector"] = demo["ticker"].map(sec)
    demo["wx"] = demo["wt"] * demo["metric_val"]
    g = demo.groupby(["date", "sector"]).agg(sum_wx=("wx","sum"), sum_w=("wt","sum")).reset_index()
    g["wavg"] = g["sum_wx"] / g["sum_w"]
    # Tech 2026-01-01: (100*10+300*20)/(400)=17.5 ; Fin: only D (C is NaN) -> -5.0
    print(g.pivot(index="date", columns="sector", values="wavg"))
