#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ADDRESSES STOCK SPLITS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import requests
from datetime import datetime
from datetime import timedelta
from datetime import date
from sqlalchemy import create_engine, text

def Set_DF(dframe):
    
    dframe.index = dframe[dframe.columns.values.tolist()[0]]
    dframe.index.name = dframe.columns.values.tolist()[0]
    New_df = dframe.drop(dframe.columns.values.tolist()[0], axis=1)
    
    return New_df

def DD_Index(dframe):
    
    dframe['dummy'] = dframe.index
    dframe.drop_duplicates(['dummy'], inplace=True)
    dframe.drop('dummy', axis=1, inplace=True)
    New_df = dframe
    
    return New_df 

def open_df(*args):
    open_str = args[0]
    
    query_open = 'SELECT * FROM ' + open_str
    opened_df = pd.read_sql_query(query_open, engine)
    opened_df = Set_DF(opened_df)
    opened_df = DD_Index(opened_df)
    opened_df = opened_df.sort_index()
    
    return opened_df

def openF_df(*args):
    open_str = args[0]
    
    query_open = 'SELECT * FROM ' + open_str
    opened_df = pd.read_sql_query(query_open, Fengine)
    opened_df = Set_DF(opened_df)
    opened_df = DD_Index(opened_df)
    opened_df = opened_df.sort_index()
    
    return opened_df

dbase = "visiblealpha_laptop"
Fdbase = "factormodel_db"
cnxn_string = ("postgresql+psycopg2://{username}:{pswd}""@{host}:{port}/{database}")
engine = create_engine(cnxn_string.format(username = "postgres", pswd = "akf7a7j5", host = "localhost", 
                                          port = 5432, database = dbase))
Fengine = create_engine(cnxn_string.format(username = "postgres", pswd = "akf7a7j5", host = "localhost", 
                                           port = 5432, database = Fdbase))

VOLUME_WINDOW = 10
VIX_WINDOW = 21

s_l = ['Custom Sector 1', 'Custom Sector 2', 'Custom Sector 3', 'Custom Sector 4', 'Custom Sector 5']
sector_key = 'Custom Sector 5'
subsector_key = 'Custom Sector GPT'

Pxs_df = openF_df('prices_relation')
Pxs_df = Pxs_df[Pxs_df.index >= datetime(2004, 1, 1).date()]

volumeTrd_df = openF_df('trading_volume')
volumeTrd_df.columns = volumeTrd_df.columns.map(lambda x: x.split(' ')[0])
Pxs_df.columns = Pxs_df.columns.map(lambda x: x.split(' ')[0])
Pxs_df.index = Pxs_df.index.map(lambda x: pd.Timestamp(x))
volumeTrd_df.index = volumeTrd_df.index.map(lambda x: pd.Timestamp(x))

sectors = pd.read_csv(r"C:\Users\Utilizador\OneDrive\Documentos\Malta\Systematic\Custom_Segmentation.csv")
Sectors_df = pd.DataFrame(sectors)
Sectors_df = Set_DF(Sectors_df)
Sectors_df = DD_Index(Sectors_df)

sectors_df = Sectors_df[[sector_key, subsector_key]]    
sectors_df = sectors_df.rename(columns = {sector_key: 'sector', subsector_key: 'sub_sector'})
sectors_df = sectors_df[sectors_df.index.map(lambda x: x.split(' ')[1]) == 'US']    
sectors_df.index = sectors_df.index.map(lambda x: x.split(' ')[0])

Pxs_df = Pxs_df.T[~Pxs_df.T.index.duplicated(keep='first')].T
sectors_df = sectors_df[~sectors_df.index.duplicated(keep='first')]






#!/usr/bin/env python
# coding: utf-8
"""
Stock-split ADJUSTMENT  (interactive, reversible — writes to income_data)
=========================================================================
Step 2 of split handling. Step 1 (detect_stock_splits.py) found suspects;
this script walks the genuine candidates one at a time, asks YOU for the
true split ratio (Googled), and applies the back-adjustment.

Convention (confirmed):
  You type the SHARE MULTIPLIER.
    forward split 25:1  -> type 25     (shares ×25, EPS ÷25)
    reverse split 1:10  -> type 0.1    (shares ×0.1, EPS ÷0.1 == ×10)
  Mechanical, direction-proof, identical both ways:
    shares × ratio ,  EPS ÷ ratio  — for every period BEFORE the split quarter.

  This lifts pre-split history onto the post-split basis, consistent with the
  (already split-adjusted) price data. Empty input -> skip to next name.

Scope (SCOPE='download-date', default): basis follows the DOWNLOAD VINTAGE.
Ortex restates server-side at the split date, so rows downloaded before it
are old-basis for EVERY period they cover (incl. forward estimates), and
rows downloaded on/after it are already new-basis (incl. pre-split actuals
refetched via restatement maintenance / force-fetch -- which a period scope
would double-adjust).

Adjusted fields:  dilutedAverageShares (xratio), dilutedEps (/ratio), eps (/ratio)
Affected rows  :  ticker = T  AND  download_date < split_date  AND
                  metric_name IN (those three)     [all periods]
Legacy scope   :  SCOPE='period' keeps the old period < split_qtr behavior
                  for row-count comparison.

Safety:
  - dry-run preview per name (row counts + before/after sample) then confirm
  - timestamped backup of every touched row to SPLIT_BACKUP_TBL before UPDATE
  - idempotency log (SPLIT_LOG_TBL): re-running won't double-adjust an already
    applied (ticker, split_qtr) — it warns and skips unless you force.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────
CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
ENGINE = create_engine(CONNECTION_STRING)

TABLE            = 'income_data'
CATEGORY         = 'income'
SHARE_METRIC     = 'dilutedAverageShares'
EPS_METRICS      = ['dilutedEps', 'eps']           # divided by ratio
SHARE_METRICS    = [SHARE_METRIC]                  # multiplied by ratio
ALL_ADJ_METRICS  = SHARE_METRICS + EPS_METRICS

SPLIT_HI = 2.0
SPLIT_LO = 0.5
MIN_SHARES = 1.0
MIN_SPLIT_QTR = '2020Q1'                            # only act on jumps in/after this quarter

SPLIT_BACKUP_TBL = 'split_adjust_backup'
SPLIT_LOG_TBL    = 'split_adjust_log'

# Adjustment scope:
#   'download-date' (default): adjust per-share rows with
#        download_date < split_date, ACROSS ALL PERIODS.
#        Basis is a property of the DOWNLOAD VINTAGE, not the period: Ortex
#        restates server-side at the split date, so rows downloaded before it
#        are old-basis for every period they cover (incl. forward estimates),
#        and rows downloaded after it are already new-basis (incl. refetched
#        pre-split actuals via restatement maintenance / force-fetch).
#        Period-scoping errs both ways: it leaves pre-split-downloaded
#        forward estimates old-scale (PIT forward reads before the split are
#        then off by the ratio vs back-adjusted prices) and DOUBLE-adjusts
#        post-split-refetched history.
#   'period' (legacy): adjust all vintages with period < split_qtr. Kept for
#        row-count comparison only.
SCOPE = 'download-date'

# Detection search space (per stock): [floor quarter, FEP + 1] inclusive.
# Splits are backward-looking events, so deep-future estimate quarters can
# never legitimately host one (2028+ candidate suggestions are noise by
# construction) -- but the basis seam of a RECENT split sits exactly at the
# frontier (last old-vintage row vs first refetched estimate: FEP or FEP+1;
# KLAC's was the (FEP, FEP+1) pair 2026Q3->2026Q4). Capping one quarter past
# the FEP keeps that seam visible while excluding everything deeper.


# ── Period helpers ────────────────────────────────────────────────────────
def period_to_int(p):
    return int(p[:4]) * 4 + (int(p[5]) - 1)


# ── Detection (reused from step 1, directional-EPS filter) ────────────────
def get_all_tickers():
    with ENGINE.connect() as conn:
        rows = conn.execute(text(
            f"SELECT DISTINCT ticker FROM {TABLE} ORDER BY ticker"
        )).fetchall()
    return [r[0] for r in rows]


def get_fep(ticker):
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period FROM estimation_status
            WHERE ticker = :t AND category = :c
        """), {"t": ticker, "c": CATEGORY}).fetchone()
    return row[0] if row and row[0] else None


def get_latest_vintage_series(ticker, metric):
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            WITH latest AS (
                SELECT period, value,
                       ROW_NUMBER() OVER (PARTITION BY period
                                          ORDER BY download_date DESC) rn
                FROM {TABLE}
                WHERE ticker = :t AND metric_name = :m
            )
            SELECT period, value FROM latest WHERE rn = 1
        """), {"t": ticker, "m": metric}).fetchall()
    out = {}
    for period, value in rows:
        if value is None:
            continue
        try:
            out[period] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def actual_series(raw, fep, min_floor=None, cap_extra_qtrs=1):
    """Latest-vintage series, ascending, capped at FEP + cap_extra_qtrs
    (inclusive). The +1 default keeps the vintage seam of a recent split
    visible (last old-vintage row vs first refetched estimate lands at FEP or
    FEP+1 -- KLAC: the (FEP, FEP+1) pair) while excluding deep-future
    estimate quarters, which can never legitimately host a split.
    cap_extra_qtrs=None disables the cap."""
    fep_idx = period_to_int(fep) if fep else None
    cap_idx = (fep_idx + cap_extra_qtrs + 1
               if (fep_idx is not None and cap_extra_qtrs is not None) else None)
    items = []
    for p, v in raw.items():
        try:
            pidx = period_to_int(p)
        except Exception:
            continue
        if cap_idx is not None and pidx >= cap_idx:
            continue
        if v is None or pd.isna(v):
            continue
        if min_floor is not None and v < min_floor:
            continue
        items.append((pidx, p, v))
    if not items:
        return pd.Series(dtype=float)
    items.sort(key=lambda x: x[0])
    return pd.Series([v for _, _, v in items], index=[p for _, p, v in items])


def detect_candidates(min_split_qtr=None):
    """Return directional-EPS-corroborated candidates at/after the floor.
    Scans the FULL latest-vintage series (actuals + estimates); each candidate
    carries a region tag: 'actual', 'act->est boundary', or 'estimate'."""
    tickers = get_all_tickers()
    cands, dropped = [], []
    cut = period_to_int(min_split_qtr or MIN_SPLIT_QTR)
    for i, t in enumerate(tickers, 1):
        if i % 100 == 0:
            print(f"    scanning ...{i}/{len(tickers)}")
        try:
            fep = get_fep(t)
            fep_idx = period_to_int(fep) if fep else None
            raw_sh = get_latest_vintage_series(t, SHARE_METRIC)
            if not raw_sh:
                continue
            sh = actual_series(raw_sh, fep, min_floor=MIN_SHARES)
            if len(sh) < 2:
                continue
            raw_eps = get_latest_vintage_series(t, 'dilutedEps')
            eps = (actual_series(raw_eps, fep, min_floor=None)
                   if raw_eps else pd.Series(dtype=float))
            periods = list(sh.index)
            for j in range(1, len(periods)):
                p_prev, p_cur = periods[j - 1], periods[j]
                if period_to_int(p_cur) < cut:
                    continue
                v_prev, v_cur = sh.iloc[j - 1], sh.iloc[j]
                if v_prev < MIN_SHARES:
                    continue
                r = v_cur / v_prev
                if not (r >= SPLIT_HI or r <= SPLIT_LO):
                    continue
                direction = 'split' if r >= SPLIT_HI else 'reverse'
                # directional EPS test (same-sign; sign-flip = ambiguous, still shown)
                eps_ratio, eps_dir_ok, eps_note = np.nan, False, 'no-eps'
                if p_prev in eps.index and p_cur in eps.index:
                    e_prev, e_cur = eps[p_prev], eps[p_cur]
                    if abs(e_prev) > 1e-9 and not pd.isna(e_prev):
                        eps_ratio = e_cur / e_prev
                        if eps_ratio < 0:
                            eps_note = 'eps-signflip'
                        elif direction == 'split' and eps_ratio < 1.0:
                            eps_dir_ok, eps_note = True, 'eps-down-OK'
                        elif direction == 'reverse' and eps_ratio > 1.0:
                            eps_dir_ok, eps_note = True, 'eps-up-OK'
                        else:
                            eps_note = 'eps-wrong-dir'
                # keep directional-OK and ambiguous sign-flips (user decides via Google);
                # drop clear wrong-direction (almost always coverage artifacts)
                # -- but NEVER silently: mid-restatement data can put shares
                # and dilutedEps on different bases and fake a wrong-direction
                # reading on a genuine split.
                if eps_note in ('eps-wrong-dir',):
                    dropped.append((t, p_cur, r, eps_ratio))
                    continue
                if fep_idx is None:
                    region = 'unknown'
                elif period_to_int(p_cur) < fep_idx:
                    region = 'actual'
                elif period_to_int(p_prev) < fep_idx:
                    region = 'act->est boundary'
                else:
                    region = 'estimate'
                cands.append({
                    'ticker': t, 'split_qtr': p_cur, 'prev_qtr': p_prev,
                    'shares_prev': v_prev, 'shares_cur': v_cur,
                    'share_ratio': r, 'direction': direction,
                    'eps_ratio': eps_ratio, 'eps_note': eps_note,
                    'region': region,
                })
        except Exception as e:
            print(f"    WARNING: {t} scan failed: {type(e).__name__}: {e}")
    cands.sort(key=lambda c: (period_to_int(c['split_qtr']), c['ticker']))
    if dropped:
        print(f"\n    {len(dropped)} share-jump(s) DROPPED on eps-wrong-direction "
              f"(coverage artifact OR mixed-basis data -- verify manually):")
        for t, q, r, er in dropped:
            print(f"      {t:<10} {q}  share_ratio={r:.3f}  eps_ratio={er:.3f}")
    return cands


# ── Infra: backup + log tables, idempotency ───────────────────────────────
def _ensure_tables():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SPLIT_BACKUP_TBL} (
                backup_stamp   TEXT NOT NULL,
                ticker         TEXT NOT NULL,
                download_date  DATE NOT NULL,
                period         TEXT NOT NULL,
                metric_name    TEXT NOT NULL,
                old_value      NUMERIC,
                new_value      NUMERIC,
                split_qtr      TEXT NOT NULL,
                ratio          NUMERIC NOT NULL,
                est_flag       BOOLEAN
            )
        """))
        conn.execute(text(f"""
            ALTER TABLE {SPLIT_BACKUP_TBL} ADD COLUMN IF NOT EXISTS est_flag BOOLEAN
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SPLIT_LOG_TBL} (
                ticker     TEXT NOT NULL,
                split_qtr  TEXT NOT NULL,
                ratio      NUMERIC NOT NULL,
                n_rows     INTEGER NOT NULL,
                applied_at TIMESTAMP NOT NULL,
                backup_stamp TEXT NOT NULL,
                PRIMARY KEY (ticker, split_qtr)
            )
        """))


def already_applied(ticker, split_qtr):
    with ENGINE.connect() as conn:
        row = conn.execute(text(f"""
            SELECT ratio, applied_at FROM {SPLIT_LOG_TBL}
            WHERE ticker = :t AND split_qtr = :q
        """), {"t": ticker, "q": split_qtr}).fetchone()
    return row  # (ratio, applied_at) or None


# ── Preview + apply ───────────────────────────────────────────────────────
def price_continuity_check(Pxs_df, ticker, ratio, split_date):
    """
    Cross-check the PRICE series at the SPLIT DATE (first post-split trading day).

    Prices are expected to be already split-adjusted -> smooth across that date.
    If the raw split jump is still present on that one day, the price was NOT
    adjusted and needs action.

    split_date = first trading day POST-split. On an unadjusted series the
    one-day move from the prior trading day shows the mechanical split jump:
      forward split r>1 -> price ratio ~1/r  (a drop; e.g. r=25 -> ~0.04)
      reverse split r<1 -> price ratio ~1/r  (a jump; e.g. r=0.1 -> ~10x)

    Threshold = halfway between no-move (1.0) and implied (1/r): (1 + 1/r)/2.
    Checking ONE specific date (not full history) removes the false positives
    that full-history scanning produced at small (2:1) ratios.

    Prints:
      'split action pending for XXXX prices'   (raw jump on that day -> not adjusted)
      'prices already adjusted for XXXX'        (smooth on that day -> already adjusted)
    """
    if Pxs_df is None or ticker not in Pxs_df.columns:
        print(f"      [price check] {ticker} not in Pxs_df — skipped")
        return None

    px = Pxs_df[ticker].dropna()
    if len(px) < 2:
        print(f"      [price check] {ticker} insufficient price history — skipped")
        return None

    # Snap to first available trading date >= split_date
    idx = px.index
    on_or_after = idx[idx >= split_date]
    if len(on_or_after) == 0:
        print(f"      [price check] split_date {split_date.date()} is beyond "
              f"{ticker} price history — skipped")
        return None
    d_post = on_or_after[0]
    pos = idx.get_loc(d_post)
    if pos == 0:
        print(f"      [price check] no prior trading day before {d_post.date()} "
              f"for {ticker} — cannot check continuity")
        return None
    d_prev = idx[pos - 1]
    if d_post != split_date:
        print(f"      [price check] {split_date.date()} not a trading day; "
              f"using first post-split session {d_post.date()}")

    p_post = float(px.loc[d_post])
    p_prev = float(px.loc[d_prev])
    if p_prev == 0:
        print(f"      [price check] zero prior price for {ticker} — skipped")
        return None

    day_ratio = p_post / p_prev
    implied   = 1.0 / ratio
    threshold = (1.0 + implied) / 2.0

    if ratio > 1.0:
        jump_present = day_ratio <= threshold      # forward split -> expect drop
    else:
        jump_present = day_ratio >= threshold      # reverse split -> expect jump

    if jump_present:
        print(f"      split action pending for {ticker} prices  "
              f"({d_prev.date()}->{d_post.date()} ratio {day_ratio:.3f}, "
              f"implied {implied:.3f})")
    else:
        print(f"      prices already adjusted for {ticker}  "
              f"({d_prev.date()}->{d_post.date()} ratio {day_ratio:.3f}, smooth)")
    return jump_present


def preview_affected(ticker, split_qtr, split_date=None,
                     actuals_unrestated=False):
    """Return DataFrame of all rows that WOULD be adjusted.

    SCOPE='download-date': rows with download_date < split_date, ALL periods
        (basis follows the download vintage -- see config note).
    actuals_unrestated=True widens the set with ACTUAL rows (est=false/NULL)
        of ANY vintage: in the transitional window right after a split the
        provider still serves as-reported (old-basis) actuals even on fresh
        downloads -- restatement arrives only with the next report cycle.
        Post-split-downloaded ESTIMATES stay excluded (already new-basis).
    SCOPE='period' (legacy): all vintages with period < split_qtr."""
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT ticker, download_date, period, metric_name, value,
                   estimated_values
            FROM {TABLE}
            WHERE ticker = :t
              AND metric_name = ANY(:metrics)
              AND value IS NOT NULL
        """), {"t": ticker, "metrics": ALL_ADJ_METRICS}).fetchall()
    df = pd.DataFrame(rows, columns=['ticker', 'download_date', 'period',
                                     'metric_name', 'value', 'estimated_values'])
    if df.empty:
        return df
    if SCOPE == 'download-date':
        assert split_date is not None, "download-date scope requires split_date"
        dd   = pd.to_datetime(df['download_date'])
        mask = dd < pd.Timestamp(split_date)
        if actuals_unrestated:
            is_actual = (df['estimated_values'].isna()
                         | ~df['estimated_values'].fillna(False).astype(bool))
            mask = mask | is_actual
        df = df[mask].reset_index(drop=True)
        return df.drop(columns=['estimated_values'])
    else:
        cut = period_to_int(split_qtr)
        df['_pidx'] = df['period'].map(lambda p: period_to_int(p)
                                       if _valid_period(p) else 10**9)
        df = df[df['_pidx'] < cut].drop(columns='_pidx').reset_index(drop=True)
    return df.drop(columns=['estimated_values'], errors='ignore')


def _valid_period(p):
    try:
        period_to_int(p)
        return True
    except Exception:
        return False


def _adjusted_value(metric, value, ratio):
    if metric in SHARE_METRICS:
        return value * ratio
    else:                       # EPS metrics
        return value / ratio


def apply_adjustment(ticker, split_qtr, ratio, affected_df, split_date=None,
                     actuals_unrestated=False):
    """Backup affected rows, then UPDATE them. Returns (n_rows, backup_stamp)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Build backup rows (old + new) and execute updates in one transaction
    backup_rows = []
    for _, r in affected_df.iterrows():
        old_v = float(r['value'])
        new_v = _adjusted_value(r['metric_name'], old_v, ratio)
        backup_rows.append({
            'backup_stamp': stamp, 'ticker': ticker,
            'download_date': r['download_date'], 'period': r['period'],
            'metric_name': r['metric_name'], 'old_value': old_v,
            'new_value': new_v, 'split_qtr': split_qtr, 'ratio': ratio,
        })

    with ENGINE.begin() as conn:
        # 1. write backup
        conn.execute(text(f"""
            INSERT INTO {SPLIT_BACKUP_TBL}
                (backup_stamp, ticker, download_date, period, metric_name,
                 old_value, new_value, split_qtr, ratio)
            VALUES (:backup_stamp, :ticker, :download_date, :period, :metric_name,
                    :old_value, :new_value, :split_qtr, :ratio)
        """), backup_rows)

        # 2. apply updates. Share metrics × ratio, EPS metrics ÷ ratio.
        #    The WHERE must reproduce preview_affected's scope EXACTLY so the
        #    backup and the update cover the same row set.
        if SCOPE == 'download-date':
            assert split_date is not None
            if actuals_unrestated:
                scope_sql = ("AND (download_date < :sd OR "
                             "estimated_values = false OR "
                             "estimated_values IS NULL)")
            else:
                scope_sql = "AND download_date < :sd"
            scope_params = {"sd": pd.Timestamp(split_date).date()}
        else:
            aff_periods  = sorted(set(affected_df['period'].tolist()))
            scope_sql    = "AND period = ANY(:periods)"
            scope_params = {"periods": aff_periods}
        conn.execute(text(f"""
            UPDATE {TABLE} SET value = value * :ratio
            WHERE ticker = :t
              AND metric_name = ANY(:share_metrics)
              {scope_sql}
              AND value IS NOT NULL
        """), {"ratio": ratio, "t": ticker,
               "share_metrics": SHARE_METRICS, **scope_params})
        conn.execute(text(f"""
            UPDATE {TABLE} SET value = value / :ratio
            WHERE ticker = :t
              AND metric_name = ANY(:eps_metrics)
              {scope_sql}
              AND value IS NOT NULL
        """), {"ratio": ratio, "t": ticker,
               "eps_metrics": EPS_METRICS, **scope_params})

        # 3. log
        conn.execute(text(f"""
            INSERT INTO {SPLIT_LOG_TBL}
                (ticker, split_qtr, ratio, n_rows, applied_at, backup_stamp)
            VALUES (:t, :q, :r, :n, :ts, :stamp)
            ON CONFLICT (ticker, split_qtr) DO UPDATE
                SET ratio = :r, n_rows = :n, applied_at = :ts, backup_stamp = :stamp
        """), {"t": ticker, "q": split_qtr, "r": ratio,
               "n": len(backup_rows), "ts": datetime.now(), "stamp": stamp})

    return len(backup_rows), stamp


# ── Interactive driver ────────────────────────────────────────────────────
def run(Pxs_df=None):
    print("=" * 78)
    print("  STOCK-SPLIT ADJUSTMENT  (interactive, reversible)")
    print(f"  fields: {SHARE_METRIC} ×ratio  |  {', '.join(EPS_METRICS)} ÷ratio")
    print(f"  scope : {'download_date < split date, ALL periods (basis follows vintage)' if SCOPE == 'download-date' else 'periods BEFORE the split quarter, all vintages (legacy)'}")
    print("  Type the SHARE MULTIPLIER (25 for 25:1, 0.1 for 1:10). Enter = skip.")
    if Pxs_df is not None:
        print("  Price-continuity check ENABLED (runs on each adjusted name).")
    print("=" * 78)

    _ensure_tables()

    # ── Mode select ──────────────────────────────────────────────────────
    print("\n  MODE:")
    print("    1. DETECT & adjust split candidates (scan universe, review each)")
    print("    2. FORCED split-ratio adjust a known ticker (bypass detection):")
    print("       dilutedAverageShares xratio, dilutedEps /ratio, eps /ratio")
    print("       (ratio=1 => pure eps-repair from dilutedEps). All vintages.")
    mode = input("  Select (1/2, Enter=1): ").strip()

    if mode == '2':
        tk = input("  Ticker: ").strip().upper()
        if not tk:
            print("  No ticker -- aborted.")
            return
        rraw = input("  Ratio (share MULTIPLIER; 4 for 1:4 fix, 1 for eps-only "
                     "repair): ").strip()
        try:
            rr = float(rraw)
            if rr <= 0:
                print("  ratio must be positive -- aborted."); return
        except ValueError:
            print(f"  '{rraw}' not a number -- aborted."); return
        # dry-run preview first, then confirm to apply
        mirror_and_split_adjust(tk, rr, dry_run=True)
        if _confirm(f"\n  Apply the above to {tk}? (y/n): "):
            mirror_and_split_adjust(tk, rr, dry_run=False)
        else:
            print("  Not applied.")
        return

    # ── Mode 1: detection (default) ──────────────────────────────────────
    # Search-space floor: quarters BEFORE this are excluded from detection --
    # keeps ancient non-split jumps (old coverage artifacts) from dragging
    # down every session. Enter accepts the default.
    import re as _re
    min_qtr = MIN_SPLIT_QTR
    while True:
        raw = input(f"\n  Detection floor quarter (YYYYQn, Enter = "
                    f"{MIN_SPLIT_QTR}): ").strip().upper()
        if raw == "":
            break
        if _re.match(r'^\d{4}Q[1-4]$', raw):
            min_qtr = raw
            break
        print("    Invalid format -- must be YYYYQn (e.g. 2025Q1).")

    print(f"\n  Detecting candidates ({min_qtr}+, cap = FEP+1 per stock, "
          f"dilutedEps corroboration)...")
    cands = detect_candidates(min_qtr)
    print(f"\n  {len(cands)} candidate split events to review.\n")

    applied, skipped = 0, 0
    for k, c in enumerate(cands, 1):
        t, q = c['ticker'], c['split_qtr']
        prior = already_applied(t, q)
        print("\n" + "-" * 78)
        print(f"  [{k}/{len(cands)}]  {t}   split quarter: {q}   "
              f"({c['direction']}, region: {c.get('region', '?')})")
        print(f"      detected share ratio : {c['share_ratio']:.3f}   "
              f"({c['shares_prev']:,.0f} -> {c['shares_cur']:,.0f})")
        print(f"      EPS corroboration    : {c['eps_note']}"
              + (f"  (eps ratio {c['eps_ratio']:.3f})" if not pd.isna(c['eps_ratio']) else ""))
        print(f"      Google: \"{t} stock split\"")
        if c.get('region') in ('estimate', 'act->est boundary'):
            print(f"      NOTE: boundary sits in the ESTIMATE region (recent "
                  f"split, actuals not yet reported post-split).")
            print(f"            Remedies: force-fetch (authoritative, costs "
                  f"credits) OR this adjuster")
            print(f"            (download-date scope rescales all old-vintage "
                  f"rows, credit-free).")
        if prior:
            print(f"      ALREADY ADJUSTED on record: ratio={float(prior[0])}, "
                  f"at {prior[1]} — type a value to RE-apply (will re-backup), Enter to skip.")

        raw = input(f"      true share multiplier for {t} (Enter=skip): ").strip()
        if not raw:
            print("      skipped.")
            skipped += 1
            continue
        try:
            ratio = float(raw)
            if ratio <= 0:
                raise ValueError
        except ValueError:
            print(f"      '{raw}' is not a valid positive number — skipping.")
            skipped += 1
            continue

        # Split date: MANDATORY under download-date scope (it defines the
        # adjustment boundary -- basis follows the download vintage); also
        # drives the price-continuity check. Under legacy period scope it is
        # optional and used for the price check only.
        sd = None
        draw = input(f"      split date for {t} — FIRST trading day "
                     f"POST-split (YY-MM-DD"
                     + (", REQUIRED" if SCOPE == 'download-date'
                        else ", Enter=skip price check") + "): ").strip()
        if draw:
            sd = _parse_split_date(draw)
            if sd is None:
                print(f"      '{draw}' not a valid YY-MM-DD date.")
        if SCOPE == 'download-date' and sd is None:
            print("      download-date scope requires a valid split date — "
                  "skipping this name.")
            skipped += 1
            continue
        if Pxs_df is not None and sd is not None:
            price_continuity_check(Pxs_df, t, ratio, sd)

        if prior and not _confirm(f"      {t} {q} already adjusted — re-apply at {ratio}? (y/n): "):
            print("      skipped.")
            skipped += 1
            continue

        # Preview
        aff = preview_affected(t, q, split_date=sd)
        if aff.empty:
            print("      no pre-split rows found to adjust — skipping.")
            skipped += 1
            continue
        n_rows = len(aff)
        vint   = aff['download_date'].nunique()
        per    = aff['period'].nunique()
        print(f"\n      WOULD ADJUST {n_rows} rows  "
              f"({per} periods × {vint} vintages, metrics: "
              f"{sorted(aff['metric_name'].unique())})")
        # before/after sample (3 rows)
        print(f"      sample:")
        for _, r in aff.head(3).iterrows():
            old_v = float(r['value'])
            new_v = _adjusted_value(r['metric_name'], old_v, ratio)
            print(f"        {r['period']} {r['metric_name']:<20} "
                  f"{old_v:>18,.4f}  ->  {new_v:>18,.4f}")

        if not _confirm(f"      apply ×{ratio} (shares) / ÷{ratio} (eps) to {n_rows} rows? (y/n): "):
            print("      skipped.")
            skipped += 1
            continue

        n, stamp = apply_adjustment(t, q, ratio, aff, split_date=sd)
        print(f"      ✓ adjusted {n} rows. backup_stamp={stamp} "
              f"(reversible via {SPLIT_BACKUP_TBL}).")
        applied += 1

    print("\n" + "=" * 78)
    print(f"  DONE.  applied: {applied}   skipped: {skipped}   "
          f"of {len(cands)} candidates.")
    print(f"  Backups in '{SPLIT_BACKUP_TBL}', audit log in '{SPLIT_LOG_TBL}'.")
    print("=" * 78)


def _confirm(prompt):
    return input(prompt).strip().lower() in ('y', 'yes')


def _parse_split_date(s):
    """Parse a YY-MM-DD (or YYYY-MM-DD) date. 2-digit year -> 20YY. None on failure."""
    s = s.strip()
    for fmt in ('%y-%m-%d', '%Y-%m-%d'):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    # last resort: let pandas try, but guard against silent misparse
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


# ── Manual apply (detection-independent) ──────────────────────────────────
def apply_known_split(ticker, ratio, split_date, split_qtr=None, Pxs_df=None,
                      actuals_unrestated=False):
    """Apply a KNOWN split without the detector's agreement. Needed because
    detection reads the latest vintage only: after a force-fetch the series
    is uniform new-basis and the detector is correctly silent -- yet the
    old-vintage rows still carry the old basis and need scaling for PIT
    consistency with back-adjusted prices. Same preview/confirm/backup/log
    path as the interactive driver; split_qtr defaults to the split date's
    calendar quarter (label only -- the download-date scope does the work)."""
    t  = ticker.strip().upper()
    sd = _parse_split_date(str(split_date))
    if sd is None:
        print(f"  '{split_date}' is not a valid date."); return
    ratio = float(ratio)
    if ratio <= 0:
        print("  ratio must be positive."); return
    q = split_qtr or f"{sd.year}Q{(sd.month - 1) // 3 + 1}"
    _ensure_tables()
    prior = already_applied(t, q)
    if prior:
        print(f"  ALREADY ADJUSTED on record: ratio={float(prior[0])} at "
              f"{prior[1]} -- proceeding will re-backup and re-apply.")
    if Pxs_df is not None:
        price_continuity_check(Pxs_df, t, ratio, sd)
    aff = preview_affected(t, q, split_date=sd,
                           actuals_unrestated=actuals_unrestated)
    if aff.empty:
        print(f"  No rows with download_date < {sd.date()} for {t} -- "
              f"nothing to adjust (vintages already consistent).")
        return
    print(f"  WOULD ADJUST {len(aff)} rows "
          f"({aff['period'].nunique()} periods x "
          f"{aff['download_date'].nunique()} vintages)")
    for _, r in aff.head(3).iterrows():
        old_v = float(r['value'])
        print(f"    {r['period']} {r['metric_name']:<20} "
              f"{old_v:>18,.4f} -> {_adjusted_value(r['metric_name'], old_v, ratio):>18,.4f}")
    if not _confirm(f"  apply x{ratio} (shares) / /{ratio} (eps) to "
                    f"{len(aff)} rows of {t}? (y/n): "):
        print("  aborted."); return
    n, stamp = apply_adjustment(t, q, ratio, aff, split_date=sd,
                                actuals_unrestated=actuals_unrestated)
    print(f"  ✓ adjusted {n} rows. backup_stamp={stamp} (revert(t, '{q}') to undo).")


# ── Reversal utility ──────────────────────────────────────────────────────
def mirror_and_split_adjust(ticker, ratio, dry_run=True):
    """FORCED, UNCONDITIONAL correction for stocks whose ENTIRE stored series
    is unadjusted (e.g. CRWD: Ortex never applied a real split, and the 'eps'
    field is unreliable while 'dilutedEps' is trustworthy).

    Split-ratio operations only, ALL download_dates, ALL periods (no date/scope
    prompt). EPS-series CONTENT adjustment (mirroring, non-GAAP reconciliation,
    etc.) is intentionally NOT done here -- that belongs to the dedicated EPS
    adjustment tooling. This mode purely applies the mechanical split ratio:
      1. dilutedAverageShares  x ratio   (more shares post-split)
      2. dilutedEps            / ratio   (per-share, so divided)
      3. eps                   / ratio   (per-share, so divided)

    No eps<-dilutedEps copy, no orphan-eps deletion. ratio=1.0 makes all three
    no-ops. Detection is bypassed entirely -- this is a deliberate, known-ratio
    split fix.
    Backup + log under split_qtr label 'MIRROR_ADJ' so revert(ticker,
    'MIRROR_ADJ') undoes it. dry_run=True previews without writing.
    """
    t = ticker.strip().upper()
    ratio = float(ratio)
    if ratio <= 0:
        print("  ratio must be positive."); return
    _ensure_tables()

    prior = already_applied(t, 'MIRROR_ADJ')
    if prior:
        print(f"  ALREADY applied MIRROR_ADJ to {t} (ratio={float(prior[0])}, "
              f"{prior[1]}). revert(t,'MIRROR_ADJ') first to re-run cleanly.")
        if not dry_run:
            if not _confirm("  Proceed anyway (stacks on prior)? (y/n): "):
                return

    # Pull every row for the three metrics, all vintages/periods.
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT download_date, period, metric_name, value, estimated_values
            FROM {TABLE}
            WHERE ticker = :t
              AND metric_name IN ('dilutedAverageShares','dilutedEps','eps')
        """), {"t": t}).fetchall()
    if not rows:
        print(f"  No rows for {t}."); return

    df = pd.DataFrame(rows, columns=['download_date','period','metric_name','value','est'])

    # Build the change set: (dd, period, metric, old, new). Pure split ratio:
    # shares x ratio; dilutedEps and eps each / ratio (both are per-share).
    changes = []
    for _, r in df.iterrows():
        dd, per, m, v = r['download_date'], r['period'], r['metric_name'], r['value']
        if v is None:
            continue
        v = float(v)
        if m == 'dilutedAverageShares':
            new = v * ratio
        elif m in ('dilutedEps', 'eps'):
            new = v / ratio
        else:
            continue
        if abs(new - v) > 1e-12:
            changes.append((dd, per, m, v, new))

    if not changes:
        print(f"  {t}: nothing to change."); return

    # Preview
    print(f"\n  {t}: SPLIT_ADJ ratio={ratio}  ({len(changes)} row updates)")
    print(f"    dilutedAverageShares x{ratio} | dilutedEps /{ratio} | eps /{ratio}")
    for m in ('dilutedAverageShares','dilutedEps','eps'):
        sample = [c for c in changes if c[2] == m][:3]
        for dd, per, _, old, new in sample:
            print(f"      {m:<22} {per} [{dd}]  {old:>14,.4f} -> {new:>14,.4f}")
        n = sum(1 for c in changes if c[2] == m)
        print(f"      ({n} {m} rows)")

    if dry_run:
        print("  DRY RUN -- nothing written. Call with dry_run=False to apply.")
        return
    if not _confirm(f"  Apply MIRROR_ADJ to {t}? (y/n): "):
        print("  aborted."); return

    stamp = f"MIRROR_{t}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    with ENGINE.begin() as conn:
        # backup
        for dd, per, m, old, new in changes:
            conn.execute(text(f"""
                INSERT INTO {SPLIT_BACKUP_TBL}
                  (backup_stamp,ticker,download_date,period,metric_name,
                   old_value,new_value,split_qtr,ratio)
                VALUES (:s,:t,:dd,:p,:m,:ov,:nv,'MIRROR_ADJ',:r)
            """), {"s":stamp,"t":t,"dd":dd,"p":per,"m":m,"ov":old,"nv":new,"r":ratio})
        # apply
        for dd, per, m, old, new in changes:
            conn.execute(text(f"""
                UPDATE {TABLE} SET value = :v
                WHERE ticker=:t AND download_date=:dd AND period=:p AND metric_name=:m
            """), {"v":new,"t":t,"dd":dd,"p":per,"m":m})
        conn.execute(text(f"""
            INSERT INTO {SPLIT_LOG_TBL}
              (ticker,split_qtr,ratio,n_rows,applied_at,backup_stamp)
            VALUES (:t,'MIRROR_ADJ',:r,:n,:ts,:s)
            ON CONFLICT (ticker,split_qtr) DO UPDATE
              SET ratio=:r,n_rows=:n,applied_at=:ts,backup_stamp=:s
        """), {"t":t,"r":ratio,"n":len(changes),
               "ts":pd.Timestamp.now(),"s":stamp})
    print(f"  Applied SPLIT_ADJ to {t}: {len(changes)} rows updated. "
          f"revert('{t}','MIRROR_ADJ') to undo.")


def revert(ticker, split_qtr):
    """Undo a previously applied adjustment by restoring old_value from backup."""
    with ENGINE.connect() as conn:
        log = conn.execute(text(f"""
            SELECT backup_stamp, ratio, n_rows FROM {SPLIT_LOG_TBL}
            WHERE ticker = :t AND split_qtr = :q
        """), {"t": ticker, "q": split_qtr}).fetchone()
    if not log:
        print(f"  No adjustment on record for {ticker} {split_qtr}.")
        return
    stamp = log[0]
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT download_date, period, metric_name, old_value, est_flag
            FROM {SPLIT_BACKUP_TBL}
            WHERE backup_stamp = :s AND ticker = :t AND split_qtr = :q
        """), {"s": stamp, "t": ticker, "q": split_qtr}).fetchall()
    if not rows:
        print(f"  No backup rows for stamp {stamp}.")
        return
    # Restore each row's old_value. UPDATE if the row still exists; if the
    # backup came from a DELETION (the row was removed), the UPDATE affects 0
    # rows -> re-INSERT it. We detect this by trying UPDATE and checking
    # rowcount, then INSERTing on miss. (Kept for backward-compat: OLD MIRROR_ADJ
    # runs deleted orphan eps rows; those must be re-created on revert. Current
    # split-ratio mode deletes nothing, so this path simply UPDATEs in place.)
    payload = [{"v": float(r[3]), "t": ticker, "dd": r[0], "p": r[1], "m": r[2],
                "e": r[4]} for r in rows]
    restored = reinserted = 0
    with ENGINE.begin() as conn:
        for p in payload:
            res = conn.execute(text(f"""
                UPDATE {TABLE} SET value = :v
                WHERE ticker = :t AND download_date = :dd
                  AND period = :p AND metric_name = :m
            """), p)
            if res.rowcount and res.rowcount > 0:
                restored += 1
            else:
                conn.execute(text(f"""
                    INSERT INTO {TABLE} (ticker, download_date, period,
                                         metric_name, value, estimated_values)
                    VALUES (:t, :dd, :p, :m, :v, :e)
                """), p)
                reinserted += 1
        conn.execute(text(f"""
            DELETE FROM {SPLIT_LOG_TBL} WHERE ticker = :t AND split_qtr = :q
        """), {"t": ticker, "q": split_qtr})
    print(f"  Reverted {ticker} {split_qtr}: {restored} restored, "
          f"{reinserted} re-inserted (from backup {stamp}).")


if __name__ == '__main__':
    # Pxs_df is expected as a kernel global (same df used across the pipeline).
    # If present, the price-continuity check runs on each adjusted name.
    run(Pxs_df=Pxs_df)

