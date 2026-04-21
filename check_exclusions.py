"""
check_exclusions.py
===================
Quick diagnostic for momentum exclusion lists and MR scores.

Usage (in kernel):
    exec(open('check_exclusions.py').read())
"""

from sqlalchemy import text
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
EXCLUSIONS_TBL = 'momentum_exclusions'
TOP_N_DATES    = 10     # dates to show in recent history
TOP_N_STOCKS   = 15     # top stocks to show per section

# ── 1. DB overview ────────────────────────────────────────────────────────────
print("=" * 68)
print("  MOMENTUM EXCLUSIONS — DB OVERVIEW")
print("=" * 68)

with ENGINE.connect() as conn:
    r = conn.execute(text(f"""
        SELECT
            COUNT(*)                                    AS total_rows,
            COUNT(DISTINCT date)                        AS total_dates,
            SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END) AS positive_scores,
            MIN(date)                                   AS first_date,
            MAX(date)                                   AS last_date,
            MIN(score)                                  AS min_score,
            MAX(score)                                  AS max_score,
            ROUND(AVG(score)::numeric, 4)               AS avg_score
        FROM {EXCLUSIONS_TBL}
    """)).fetchone()

print(f"  Total rows       : {r[0]:,}")
print(f"  Distinct dates   : {r[1]:,}")
print(f"  Score > 0        : {r[2]:,}  ({r[2]/max(r[0],1)*100:.1f}%)")
print(f"  Date range       : {r[3]}  →  {r[4]}")
print(f"  Score range      : min={r[5]:.4f}  max={r[6]:.4f}  avg={r[7]:.4f}")

# ── 2. Exclusions per date (recent history) ───────────────────────────────────
print(f"\n{'─'*68}")
print(f"  RECENT DATES — exclusion counts (last {TOP_N_DATES} dates with data)")
print(f"{'─'*68}")

with ENGINE.connect() as conn:
    df_counts = pd.read_sql(text(f"""
        SELECT date, COUNT(*) AS n_excl, MAX(score) AS max_score,
               ROUND(AVG(score)::numeric, 3) AS avg_score
        FROM {EXCLUSIONS_TBL}
        WHERE score > 0
        GROUP BY date
        ORDER BY date DESC
        LIMIT {TOP_N_DATES}
    """), conn)

df_counts['date'] = pd.to_datetime(df_counts['date'])
print(f"  {'Date':<14}  {'N excl':>8}  {'Max score':>10}  {'Avg score':>10}")
print(f"  {'─'*50}")
for _, row in df_counts.iterrows():
    print(f"  {str(row['date'].date()):<14}  {int(row['n_excl']):>8}  "
          f"{row['max_score']:>10.3f}  {row['avg_score']:>10.3f}")

# ── 3. Most recently excluded stocks (latest date) ───────────────────────────
print(f"\n{'─'*68}")
print(f"  LATEST DATE — full exclusion list")
print(f"{'─'*68}")

latest_date = df_counts['date'].max() if not df_counts.empty else None
if latest_date:
    with ENGINE.connect() as conn:
        df_latest = pd.read_sql(text(f"""
            SELECT ticker, score
            FROM {EXCLUSIONS_TBL}
            WHERE date = :d AND score > 0
            ORDER BY score DESC
        """), conn, params={'d': latest_date.date()})

    print(f"  Date: {latest_date.date()}  ({len(df_latest)} stocks)")
    print(f"  {'Ticker':<10}  {'Score':>8}")
    print(f"  {'─'*22}")
    for _, row in df_latest.iterrows():
        print(f"  {row['ticker']:<10}  {row['score']:>8.4f}")

# ── 4. All-time top stocks by max MR score ────────────────────────────────────
print(f"\n{'─'*68}")
print(f"  ALL-TIME TOP {TOP_N_STOCKS} STOCKS BY MAX MR SCORE")
print(f"{'─'*68}")

with ENGINE.connect() as conn:
    df_top = pd.read_sql(text(f"""
        SELECT ticker,
               COUNT(*)                        AS n_dates,
               ROUND(MAX(score)::numeric, 4)   AS max_score,
               ROUND(AVG(score)::numeric, 4)   AS avg_score,
               MIN(date)                       AS first_seen,
               MAX(date)                       AS last_seen
        FROM {EXCLUSIONS_TBL}
        WHERE score > 0
        GROUP BY ticker
        ORDER BY max_score DESC
        LIMIT {TOP_N_STOCKS}
    """), conn)

print(f"  {'Ticker':<10}  {'N dates':>8}  {'Max score':>10}  "
      f"{'Avg score':>10}  {'First seen':<12}  {'Last seen':<12}")
print(f"  {'─'*68}")
for _, row in df_top.iterrows():
    print(f"  {row['ticker']:<10}  {int(row['n_dates']):>8}  "
          f"{float(row['max_score']):>10.4f}  {float(row['avg_score']):>10.4f}  "
          f"{str(row['first_seen']):<12}  {str(row['last_seen']):<12}")

# ── 5. Score distribution (percentiles) ──────────────────────────────────────
print(f"\n{'─'*68}")
print(f"  MR SCORE DISTRIBUTION (score > 0 only)")
print(f"{'─'*68}")

with ENGINE.connect() as conn:
    scores = pd.read_sql(text(f"""
        SELECT score FROM {EXCLUSIONS_TBL} WHERE score > 0
    """), conn)['score'].values

pcts = [50, 75, 90, 95, 99, 99.9]
print(f"  N={len(scores):,}  mean={scores.mean():.4f}  std={scores.std():.4f}")
print(f"  {'Percentile':<14}  {'Score':>8}")
print(f"  {'─'*26}")
for p in pcts:
    print(f"  p{p:<13.1f}  {np.percentile(scores, p):>8.4f}")

# ── 6. Score distribution by year ────────────────────────────────────────────
print(f"\n{'─'*68}")
print(f"  MR SCORE BY YEAR")
print(f"{'─'*68}")

with ENGINE.connect() as conn:
    df_yr = pd.read_sql(text(f"""
        SELECT EXTRACT(YEAR FROM date)::int AS year,
               COUNT(*)                         AS n_rows,
               COUNT(DISTINCT date)             AS n_dates,
               ROUND(AVG(score)::numeric, 4)    AS avg_score,
               ROUND(MAX(score)::numeric, 4)    AS max_score,
               ROUND(AVG(COUNT(*)) OVER
                   (PARTITION BY EXTRACT(YEAR FROM date))::numeric, 1)
                                                AS avg_per_date
        FROM {EXCLUSIONS_TBL}
        WHERE score > 0
        GROUP BY year
        ORDER BY year
    """), conn)

print(f"  {'Year':>6}  {'N rows':>8}  {'N dates':>8}  "
      f"{'Avg/date':>10}  {'Avg score':>10}  {'Max score':>10}")
print(f"  {'─'*60}")
for _, row in df_yr.iterrows():
    avg_per_date = row['n_rows'] / max(row['n_dates'], 1)
    print(f"  {int(row['year']):>6}  {int(row['n_rows']):>8}  "
          f"{int(row['n_dates']):>8}  {avg_per_date:>10.1f}  "
          f"{float(row['avg_score']):>10.4f}  {float(row['max_score']):>10.4f}")

# ── 7. Lookup: scores for a specific ticker ───────────────────────────────────
def show_ticker(ticker, last_n=20):
    """Show MR score history for a specific ticker."""
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT date, score
            FROM {EXCLUSIONS_TBL}
            WHERE ticker = :t AND score > 0
            ORDER BY date DESC
            LIMIT {last_n}
        """), conn, params={'t': ticker.upper()})
    if df.empty:
        print(f"  {ticker}: no exclusion records found")
        return
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n  {ticker} — last {len(df)} exclusion dates")
    print(f"  {'Date':<14}  {'Score':>8}")
    print(f"  {'─'*26}")
    for _, row in df.iterrows():
        print(f"  {str(row['date'].date()):<14}  {row['score']:>8.4f}")

# ── 8. Lookup: full exclusion list for a specific date ───────────────────────
def show_date(dt):
    """Show full exclusion list for a specific date."""
    dt_ts = pd.Timestamp(dt)
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT ticker, score
            FROM {EXCLUSIONS_TBL}
            WHERE date = :d AND score > 0
            ORDER BY score DESC
        """), conn, params={'d': dt_ts.date()})
    if df.empty:
        print(f"  {dt_ts.date()}: no exclusions")
        return
    print(f"\n  {dt_ts.date()} — {len(df)} excluded stocks")
    print(f"  {'Ticker':<10}  {'Score':>8}")
    print(f"  {'─'*22}")
    for _, row in df.iterrows():
        print(f"  {row['ticker']:<10}  {row['score']:>8.4f}")

print(f"\n{'=' * 68}")
print(f"  Helper functions available:")
print(f"    show_ticker('NVDA')          — score history for a stock")
print(f"    show_date('2024-11-18')      — full exclusion list for a date")
print(f"{'=' * 68}")
