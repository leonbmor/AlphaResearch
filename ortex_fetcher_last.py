#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ortex Data Fetcher

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Ortex Fundamentals Data Fetcher - V2
=====================================
Changes from V1:
  1. Fetches ESTIMATES ONLY -- actuals are left as NaN (forward-filled downstream)
  2. Skips stock entirely if first-estimate totalRevenues has not changed
  3. On earnings release (first_estimated_period became actual):
       a. Checks AlphaVantage FIRST (saves Ortex credits)
       b. Validates fiscalDateEnding matches expected calendar quarter
       c. If AV not ready / data incomplete: skips stock (moves to next)
       d. If AV ready: applies non-GAAP adjustments to today's rows only
  4. Copies normalizedNetIncome -> netIncome for ALL estimate rows (SQL bulk copy)
  5. Always updates estimation_status for every stock
  6. Robust ticker handling: always splits on ' ' and takes first token
  7. Robust date handling: always converts to pd.Timestamp
"""

import os
import requests
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, date
import time
import logging
import sys
from typing import List, Dict, Optional, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================

START_PERIOD = '2018Q1'
END_PERIOD   = '2030Q4'

ORTEX_API_BASE_URL = "https://api.ortex.com/api/v1/stock/US/"
# V3: credentials from environment. Set in the kernel BEFORE running this cell:
#   os.environ['ORTEX_API_KEY']     = '...'
#   os.environ['AV_API_KEY']        = '...'
#   os.environ['FACTORMODEL_DB_URL'] = 'postgresql+psycopg2://user:pwd@localhost:5432/factormodel_db'
#ORTEX_API_KEY      = os.environ.get("ORTEX_API_KEY", "")
ORTEX_HEADERS = {
    "accept": "application/json",
    "Ortex-Api-Key": ORTEX_API_KEY
}

#AV_API_KEY        = os.environ.get("AV_API_KEY", "")
AV_URL_ROOT       = 'https://www.alphavantage.co/query?function='
AV_RETRY_WAIT_SEC = 20
AV_MAX_RETRIES    = 6

CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
ENGINE = create_engine(CONNECTION_STRING)

CREDITS_PER_ROUND     = 1000
MIN_CREDITS_REMAINING = 500

ORTEX_CUTOFF_DATE = pd.Timestamp('2026-02-06')

# --- V3 constants ---------------------------------------------------------
AV_EXPECTED_TOLERANCE_DAYS = 15    # pattern-derived fiscalDateEnding tolerance
                                   # (absorbs 52/53-wk calendars + AV month-end
                                   #  normalisation inconsistencies)
AV_EPS_SANITY_REL_DIFF     = 0.50  # AV estimatedEPS vs Ortex consensus: prompt
                                   # if relative diff exceeds this
REV_MATERIALITY_PCT        = 0.0   # skip-gate threshold on totalRevenues.
                                   # 0.0 == any-change (current behaviour).
                                   # Raise (e.g. 0.002) if credit burn from
                                   # immaterial jitter is ever confirmed.
MAINT_TRAILING_N_DEFAULT   = 4     # restatement maintenance: trailing actual
                                   # quarters re-checked per stock

ACCEPT_GAAP = object()  # sentinel: user chose "accept GAAP as-is, mark done"


def _check_credentials():
    missing = [name for name, val in
               [("ORTEX_API_KEY", ORTEX_API_KEY), ("AV_API_KEY", AV_API_KEY)]
               if not val]
    if missing:
        raise RuntimeError(
            f"Missing credentials: {', '.join(missing)}. Set them in the kernel "
            f"BEFORE this cell runs, e.g. os.environ['ORTEX_API_KEY'] = '...'")
    if "CHANGE_ME" in CONNECTION_STRING:
        raise RuntimeError(
            "FACTORMODEL_DB_URL not set -- export it or edit CONNECTION_STRING.")

# ==============================================================================
# LOGGING
# ==============================================================================

_file_handler   = logging.FileHandler('ortex_fetcher_v2.log', encoding='utf-8')
_stream_handler = logging.StreamHandler(sys.stdout)
if hasattr(_stream_handler.stream, 'reconfigure'):
    _stream_handler.stream.reconfigure(encoding='utf-8', errors='replace')
_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
_file_handler.setFormatter(_formatter)
_stream_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.INFO)
_stream_handler.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


# ==============================================================================
# UTILITY HELPERS
# ==============================================================================

def clean_ticker(ticker: str) -> str:
    return ticker.split(' ')[0]


def to_ts(d) -> pd.Timestamp:
    return pd.Timestamp(d)


def to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def epsilon_equal(a, b, eps=0.01) -> bool:
    fa, fb = to_float(a), to_float(b)
    if fa is None or fb is None:
        return fa == fb
    return abs(fa - fb) < eps


def materially_different(a, b, rel_eps: float = REV_MATERIALITY_PCT) -> bool:
    """
    V3: relative materiality comparison for the skip gate.
    At rel_eps=0.0 this reproduces any-change behaviour exactly.
    """
    fa, fb = to_float(a), to_float(b)
    if fa is None or fb is None:
        return fa != fb
    base = max(abs(fa), abs(fb), 1.0)
    return abs(fa - fb) / base > rel_eps


def mmdd_to_date(mmdd: str, today: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    V3: 'MMDD' -> most recent PAST occurrence of that month/day.
    Year-boundary safe: a December earnings date parsed in January maps to
    December of LAST year, not a future date.
    """
    if today is None:
        today = pd.Timestamp(date.today())
    d = pd.Timestamp(datetime(today.year, int(mmdd[:2]), int(mmdd[-2:])))
    if d > today:
        d = pd.Timestamp(datetime(today.year - 1, int(mmdd[:2]), int(mmdd[-2:])))
    return d


def generate_periods(start_period: str, end_period: str) -> List[str]:
    start_year, start_q = int(start_period[:4]), int(start_period[5])
    end_year,   end_q   = int(end_period[:4]),   int(end_period[5])
    periods = []
    for year in range(start_year, end_year + 1):
        sq = start_q if year == start_year else 1
        eq = end_q   if year == end_year   else 4
        for q in range(sq, eq + 1):
            periods.append(f"{year}Q{q}")
    return periods


def get_next_period(period: str) -> str:
    year, q = int(period[:4]), int(period[5])
    return f"{year + 1}Q1" if q == 4 else f"{year}Q{q + 1}"


def add_quarters(quarter: str, n: int) -> str:
    year, q = int(quarter[:4]), int(quarter[5])
    q += n
    while q > 4:
        q -= 4
        year += 1
    while q < 1:
        q += 4
        year -= 1
    return f"{year}Q{q}"


AV_OVERRIDE_IF_NA: bool = False


def get_expected_fiscal_date() -> str:
    """
    V3 NOTE: no longer used for AV validation (replaced by the pattern-derived
    check inside fetch_av_earnings). Kept only as metadata for manual-override
    av_data dicts.
    """
    today = datetime.today()
    Y = today.year
    P = Y - 1
    if today <= datetime(Y, 1, 10):
        return f"{P}-09-30"
    elif today <= datetime(Y, 4, 10):
        return f"{P}-12-31"
    elif today <= datetime(Y, 7, 10):
        return f"{Y}-03-31"
    elif today <= datetime(Y, 10, 10):
        return f"{Y}-06-30"
    else:
        return f"{Y}-09-30"


# ==============================================================================
# DATABASE HELPERS
# ==============================================================================

def initialize_database():
    with ENGINE.connect() as conn:
        for table in ['income_data', 'cash_data', 'summary_data']:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    ticker TEXT NOT NULL,
                    download_date DATE NOT NULL,
                    period TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value NUMERIC,
                    estimated_values BOOLEAN DEFAULT FALSE,
                    PRIMARY KEY (ticker, download_date, period, metric_name)
                )
            """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS estimation_status (
                ticker TEXT NOT NULL,
                category TEXT NOT NULL,
                first_estimated_period TEXT,
                last_checked DATE,
                PRIMARY KEY (ticker, category)
            )
        """))
        conn.commit()
    logger.info("Database initialised")


def get_already_fetched_periods_today(ticker: str, category: str) -> set:
    t = clean_ticker(ticker)
    today = pd.Timestamp(date.today())
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT period FROM {category}_data
            WHERE ticker = :t AND download_date = :d
        """), {"t": t, "d": today}).fetchall()
    return {r[0] for r in rows}


def get_all_fetched_periods_ever(ticker: str, category: str) -> set:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT period FROM {category}_data WHERE ticker = :t
        """), {"t": t}).fetchall()
    return {r[0] for r in rows}


def get_estimation_status(ticker: str, category: str) -> Optional[str]:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period FROM estimation_status
            WHERE ticker = :t AND category = :c
        """), {"t": t, "c": category}).fetchone()
    return row[0] if row else None


def update_estimation_status(ticker: str, category: str, fep: str):
    t     = clean_ticker(ticker)
    today = pd.Timestamp(date.today())
    with ENGINE.connect() as conn:
        conn.execute(text("""
            INSERT INTO estimation_status (ticker, category, first_estimated_period, last_checked)
            VALUES (:t, :c, :p, :d)
            ON CONFLICT (ticker, category) DO UPDATE
                SET first_estimated_period = :p, last_checked = :d
        """), {"t": t, "c": category, "p": fep, "d": today})
        conn.commit()


def detect_first_estimated_from_db(ticker: str, category: str) -> Optional[str]:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            WITH latest AS (
                SELECT period, estimated_values,
                       ROW_NUMBER() OVER (PARTITION BY period ORDER BY download_date DESC) rn
                FROM {category}_data WHERE ticker = :t
            )
            SELECT period, estimated_values FROM latest WHERE rn = 1 ORDER BY period
        """), {"t": t}).fetchall()
    for period, est in rows:
        if est:
            return period
    return None


def get_metric_value_ffill(ticker: str, metric_name: str, period: str,
                            download_date, table_name: str) -> Optional[float]:
    t  = clean_ticker(ticker)
    dt = to_ts(download_date)
    query = text(f"""
        SELECT download_date, value
        FROM {table_name}
        WHERE ticker = :ticker
          AND metric_name = :metric
          AND period = :period
          AND download_date <= :download_date
        ORDER BY download_date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn, params={
            'ticker': t, 'metric': metric_name,
            'period': period, 'download_date': dt
        })
    if df.empty:
        return None
    df['value'] = pd.to_numeric(df['value'], errors='coerce').ffill()
    return to_float(df['value'].iloc[-1])


def get_metric_value_strict(ticker: str, metric_name: str, period: str,
                             download_date, table_name: str) -> Optional[float]:
    t         = clean_ticker(ticker)
    date_only = to_ts(download_date).date()
    with ENGINE.connect() as conn:
        row = conn.execute(text(f"""
            SELECT value FROM {table_name}
            WHERE ticker = :ticker
              AND metric_name = :metric
              AND period = :period
              AND download_date::date = :date_only
              AND value IS NOT NULL
            ORDER BY download_date DESC
            LIMIT 1
        """), {
            'ticker': t, 'metric': metric_name,
            'period': period, 'date_only': date_only
        }).fetchone()
    if row is None:
        return None
    return to_float(row[0])


def get_last_value_db(ticker: str, category: str, period: str, metric: str) -> Optional[float]:
    t = clean_ticker(ticker)
    with ENGINE.connect() as conn:
        row = conn.execute(text(f"""
            SELECT value FROM {category}_data
            WHERE ticker = :t AND period = :p AND metric_name = :m
            ORDER BY download_date DESC LIMIT 1
        """), {"t": t, "p": period, "m": metric}).fetchone()
    return to_float(row[0]) if row else None


def save_fundamentals_data(ticker: str, period: str, category: str,
                            data: Dict, estimated: bool, download_date):
    t     = clean_ticker(ticker)
    dt    = to_ts(download_date)
    table = f"{category}_data"

    if category == 'summary':
        allowed  = {'cashAndCashEquivalents', 'debt', 'netDebt', 'ebitda'}
        filtered = {k: v for k, v in data.items() if k in allowed and k != 'estimatedValues'}
    else:
        filtered = {k: v for k, v in data.items() if k != 'estimatedValues'}

    if not filtered:
        logger.warning(f"No data to save: {t} | {category} | {period}")
        return

    records = [
        {'ticker': t, 'download_date': dt, 'period': period,
         'metric_name': k, 'value': v, 'estimated_values': estimated}
        for k, v in filtered.items()
    ]

    # V3: delete + insert in ONE transaction -- previously a crash between the
    # two left the (ticker, period, date) vintage deleted with no replacement.
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            DELETE FROM {table}
            WHERE ticker = :ticker AND period = :period AND download_date = :download_date
        """), {'ticker': t, 'period': period, 'download_date': dt})
        pd.DataFrame(records).to_sql(table, conn, if_exists='append', index=False)
    logger.info(f"Saved {t} | {category} | {period} | est={estimated} | {len(records)} metrics")


def upsert_metric(ticker: str, table: str, period: str,
                  download_date, metric: str, value: float, estimated: bool = False):
    t  = clean_ticker(ticker)
    dt = to_ts(download_date)
    with ENGINE.begin() as conn:
        result = conn.execute(text(f"""
            UPDATE {table} SET value = :v
            WHERE ticker = :t AND period = :p AND download_date = :d AND metric_name = :m
        """), {"v": value, "t": t, "p": period, "d": dt, "m": metric})
        if result.rowcount == 0:
            conn.execute(text(f"""
                INSERT INTO {table}
                    (ticker, download_date, period, metric_name, value, estimated_values)
                VALUES (:t, :d, :p, :m, :v, :est)
            """), {"t": t, "d": dt, "p": period, "m": metric, "v": value, "est": estimated})


# ==============================================================================
# V3: MANUAL-INPUT SAFETY NET (Ortex side)
# ==============================================================================

def get_or_prompt(ticker: str, metric: str, period: str, download_date,
                  table: str, label: str = None):
    """
    Ffill read; if the metric is missing, prompt:
      <number> -> persist to today's vintage (available downstream) and return it
      'g'      -> ACCEPT_GAAP sentinel (caller treats stock as done, no adj)
      Enter    -> None (caller skips stock; status must NOT advance -> retried)
    Uses 'is None' semantics throughout: a legitimate 0.0 passes untouched.
    """
    val = get_metric_value_ffill(ticker, metric, period, download_date, table)
    if val is not None:
        return val
    print(f"\n  (!) {ticker}: Ortex missing '{label or metric}' for {period}")
    raw = input(f"  Enter value for {metric}, 'g' = accept GAAP as-is "
                f"(no adj, mark done), Enter = skip stock: ").strip()
    if raw == '':
        return None
    if raw.lower() == 'g':
        print(f"  NOTE: '{metric}' will remain NULL in the DB for {period} -- "
              f"downstream features relying on it will ffill stale data or stay NaN.")
        if input(f"  Confirm GAAP-as-is for {ticker} {period}? (y/n): ").strip().lower() == 'y':
            logger.warning(f"  {ticker}: USER ACCEPTED GAAP AS-IS for {period} "
                           f"(missing: {metric}) -- no adjustments, status will advance")
            return ACCEPT_GAAP
        return None
    try:
        fval = float(raw)
    except ValueError:
        print(f"  Invalid input '{raw}', skipping stock")
        return None
    upsert_metric(ticker, table, period, download_date, metric, fval, estimated=False)
    logger.info(f"  {ticker}: manual override {metric}={fval} for {period} (persisted)")
    return fval


# ==============================================================================
# ORTEX API
# ==============================================================================

def fetch_fundamentals(ticker: str, period: str, category: str) -> Optional[Dict]:
    t   = clean_ticker(ticker)
    url = f"{ORTEX_API_BASE_URL}{t}/fundamentals/{category}?period={period}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=ORTEX_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < 2:
                delay = 2 * (2 ** attempt)
                logger.warning(f"{t} | {category} | {period} | 429 retry {attempt+1} after {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"{t} | {category} | {period} | HTTP error: {e}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"{t} | {category} | {period} | Request error: {e}")
            return None
    return None


def _is_empty_response(data: Dict) -> bool:
    """
    Returns True if all metric fields are None — i.e. Ortex has no data
    for this period. estimatedValues is excluded from the check.
    """
    return all(
        v is None
        for k, v in data.items()
        if k != 'estimatedValues'
    )

def fetch_av_earnings(ticker: str, ortex_period: Optional[str] = None):
    """
    V3 validation (replaces the calendar-grid expected-date check):
      1. PATTERN: expected fiscalDateEnding = quarterly[3].fiscalDateEnding
         + 1 year (the company's own history IS its fiscal calendar).
         Mismatch beyond ±AV_EXPECTED_TOLERANCE_DAYS -> data not ready / glitch
         -> return None (skip & retry). <4 quarters of history -> check
         unavailable, fall through to the EPS sanity prompt.
      2. SANITY: AV estimatedEPS vs Ortex pre-release consensus EPS for
         ortex_period; gross mismatch -> human confirm prompt.
    Returns av_data dict, ACCEPT_GAAP sentinel, or None.
    """
    t                  = clean_ticker(ticker)
    url                = f"{AV_URL_ROOT}EARNINGS&symbol={t}&apikey={AV_API_KEY}"
    attempts           = 0

    while attempts < AV_MAX_RETRIES:
        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            if 'Error Message' in data:
                logger.info(f"    (!) AV Error Message for {t}, retrying in {AV_RETRY_WAIT_SEC}s "
                            f"(attempt {attempts+1}/{AV_MAX_RETRIES})")
                time.sleep(AV_RETRY_WAIT_SEC)
                attempts += 1
                continue

            quarterly = data.get('quarterlyEarnings', [])
            if not quarterly:
                logger.info(f"    AV: no quarterlyEarnings for {t}")
                return None

            latest       = quarterly[0]
            fiscal_end   = latest.get('fiscalDateEnding', '')
            reported_eps = latest.get('reportedEPS')

            # --- (1) Pattern-derived expected date ---------------------------
            try:
                latest_end = pd.Timestamp(fiscal_end)
            except (ValueError, TypeError):
                logger.warning(f"    AV: {t} unparseable fiscalDateEnding="
                               f"{fiscal_end!r} -- skipping stock")
                return None

            if len(quarterly) >= 4:
                try:
                    year_ago = pd.Timestamp(quarterly[3].get('fiscalDateEnding'))
                    expected = year_ago + pd.Timedelta(days=365)
                    if abs((latest_end - expected).days) > AV_EXPECTED_TOLERANCE_DAYS:
                        logger.info(f"    AV: {t} latest={latest_end.date()} vs "
                                    f"expected~{expected.date()} -- data not "
                                    f"ready (or AV gap/glitch), skipping stock")
                        return None
                except (ValueError, TypeError):
                    logger.warning(f"    AV: {t} unparseable prior-year "
                                   f"fiscalDateEnding -- pattern check unavailable")
            else:
                # Recent IPOs etc.: cannot derive the pattern; the EPS sanity
                # check below becomes the only gate.
                logger.warning(f"    AV: {t} has <4 quarters of AV history -- "
                               f"pattern check unavailable")

            # --- (2) estimatedEPS vs Ortex pre-release consensus -------------
            if ortex_period is not None:
                av_est    = to_float(latest.get('estimatedEPS'))
                ortex_est = get_metric_value_ffill(
                    t, 'eps', ortex_period, pd.Timestamp(date.today()), 'income_data')
                if av_est is not None and ortex_est is not None:
                    base = max(abs(av_est), abs(ortex_est))
                    if base > 0.05 and abs(av_est - ortex_est) / base > AV_EPS_SANITY_REL_DIFF:
                        print(f"\n  (!) {t}: AV estimatedEPS={av_est} vs Ortex "
                              f"consensus EPS={ortex_est} for {ortex_period} "
                              f"differ by >{AV_EPS_SANITY_REL_DIFF:.0%}")
                        print(f"      Possible wrong-quarter match, split, or scale issue.")
                        if input(f"  Proceed with AV data for {t}? (y/n): ").strip().lower() != 'y':
                            logger.warning(f"    {t}: user rejected AV data on EPS sanity check")
                            return None

            # Check reportedEPS is available
            if reported_eps in (None, 'None', ''):
                logger.info(f"    AV: {t} reportedEPS not yet available for {fiscal_end}")
                if AV_OVERRIDE_IF_NA:
                    print(f"\n  AV reportedEPS not available for {t} | {fiscal_end}.")
                    raw = input(f"  Enter non-GAAP EPS for {t}, 'g' = accept GAAP "
                                f"as-is (mark done), or Enter to skip: ").strip()
                    if raw == '':
                        return None
                    if raw.lower() == 'g':
                        if input(f"  Confirm GAAP-as-is for {t} | {fiscal_end}? "
                                 f"(y/n): ").strip().lower() == 'y':
                            logger.warning(f"    {t}: USER ACCEPTED GAAP AS-IS "
                                           f"(AV reportedEPS unavailable)")
                            return ACCEPT_GAAP
                        return None
                    try:
                        reported_eps = float(raw)
                        logger.info(f"    {t}: user provided EPS override: {reported_eps}")
                    except ValueError:
                        print(f"  Invalid input '{raw}', skipping stock")
                        return None
                else:
                    return None

            if float(reported_eps) == 0.0:
                print(f"\n  (!) AV reportedEPS = 0 for {t} | {fiscal_end}")
                print(f"      estimatedEPS={latest.get('estimatedEPS')} | "
                      f"reportedDate={latest.get('reportedDate')}")
                ans = input(f"  Proceed with zero EPS for {t}? (y / n / enter number): ").strip()
                if ans.lower() == 'y':
                    pass
                elif ans.lower() == 'n' or ans == '':
                    logger.info(f"    {t}: user rejected zero reportedEPS -- skipping stock")
                    return None
                else:
                    try:
                        reported_eps = float(ans)
                        logger.info(f"    {t}: user overrode zero EPS with {reported_eps}")
                    except ValueError:
                        print(f"  Invalid input '{ans}', skipping stock")
                        return None

            return {
                'reportedEPS':      float(reported_eps),
                'fiscalDateEnding': fiscal_end
            }

        except Exception as e:
            logger.info(f"    AV exception for {t}: {e}, retrying in {AV_RETRY_WAIT_SEC}s "
                        f"(attempt {attempts+1}/{AV_MAX_RETRIES})")
            time.sleep(AV_RETRY_WAIT_SEC)
            attempts += 1

    raise RuntimeError(f"AlphaVantage API issue - {t} failed after {AV_MAX_RETRIES} attempts")


# ==============================================================================
# NON-GAAP ADJUSTMENT LOGIC
# ==============================================================================

# Default: 0.75 (assumes ~25% effective tax rate). Edit here if needed.
GROSS_UP_FACTOR = 0.75

def apply_nongaap_adjustments(ticker: str, last_actual: str,
                               av_data: Dict, download_date) -> str:
    """
    V3: returns 'applied' | 'gaap' (user accepted GAAP as-is, mark done) |
    'retry' (skip for now -- caller must NOT advance estimation_status, so
    the stock re-enters this path on the next run).
    Missing Ortex inputs trigger a manual-entry prompt; entered values are
    persisted to today's vintage so downstream consumers get them too.
    """
    t  = clean_ticker(ticker)
    dt = to_ts(download_date)
    logger.info(f"  Applying non-GAAP adjustments for {t} | {last_actual}...")

    av_non_gaap_eps = av_data['reportedEPS']

    needed = [('eps',                  'income_data'),
              ('dilutedAverageShares', 'income_data'),
              ('netIncome',            'income_data'),
              ('operatingIncome',      'income_data'),
              ('ebitda',               'summary_data'),
              ('dilutedEps',           'income_data')]
    vals = {}
    for metric, table in needed:
        v = get_or_prompt(t, metric, last_actual, dt, table)
        if v is ACCEPT_GAAP:
            return 'gaap'
        if v is None:
            logger.info(f"  {t}: '{metric}' unresolved -- deferring adjustments (retry)")
            return 'retry'
        vals[metric] = v

    ortex_eps      = vals['eps']
    diluted_shares = vals['dilutedAverageShares']
    ortex_NI       = vals['netIncome']
    ortex_OI       = vals['operatingIncome']
    ortex_EBITDA   = vals['ebitda']
    ortex_dEPS     = vals['dilutedEps']

    adjustment_per_share = av_non_gaap_eps - ortex_eps
    adjustment_NI        = adjustment_per_share * diluted_shares

    first_estimate = add_quarters(last_actual, 1)
    prev_actual    = add_quarters(last_actual, -1)

    # OI reference: ref_OI and ref_NNI must come from the same quarter.
    # If FED has both -> use FED. If either missing -> use prev_actual for both.
    ref_OI_fed      = get_metric_value_strict(t, 'operatingIncome',     first_estimate, dt, 'income_data')
    ref_NNI_fed_OI  = get_metric_value_strict(t, 'normalizedNetIncome', first_estimate, dt, 'income_data')
    if ref_OI_fed is not None and ref_NNI_fed_OI is not None:
        ref_OI     = ref_OI_fed
        ref_NNI_OI = ref_NNI_fed_OI
        logger.info(f"  {t}: OI ref from FED ({first_estimate}): OI={ref_OI} NNI={ref_NNI_OI}")
    else:
        ref_OI     = get_metric_value_ffill(t, 'operatingIncome',     prev_actual, dt, 'income_data')
        ref_NNI_OI = get_metric_value_ffill(t, 'normalizedNetIncome', prev_actual, dt, 'income_data')
        if ref_NNI_OI is None:
            ref_NNI_OI = get_metric_value_ffill(t, 'netIncome', prev_actual, dt, 'income_data')
        logger.info(f"  {t}: OI ref from prev_actual ({prev_actual}): OI={ref_OI} NNI={ref_NNI_OI}")

    # EBITDA reference: ref_EBITDA and ref_NNI must come from the same quarter.
    # If FED has both -> use FED. If either missing -> use prev_actual for both.
    ref_EBITDA_fed      = get_metric_value_strict(t, 'ebitda',              first_estimate, dt, 'summary_data')
    ref_NNI_fed_EBITDA  = get_metric_value_strict(t, 'normalizedNetIncome', first_estimate, dt, 'income_data')
    if ref_EBITDA_fed is not None and ref_NNI_fed_EBITDA is not None:
        ref_EBITDA     = ref_EBITDA_fed
        ref_NNI_EBITDA = ref_NNI_fed_EBITDA
        logger.info(f"  {t}: EBITDA ref from FED ({first_estimate}): EBITDA={ref_EBITDA} NNI={ref_NNI_EBITDA}")
    else:
        ref_EBITDA     = get_metric_value_ffill(t, 'ebitda',              prev_actual, dt, 'summary_data')
        ref_NNI_EBITDA = get_metric_value_ffill(t, 'normalizedNetIncome', prev_actual, dt, 'income_data')
        if ref_NNI_EBITDA is None:
            ref_NNI_EBITDA = get_metric_value_ffill(t, 'netIncome', prev_actual, dt, 'income_data')
        logger.info(f"  {t}: EBITDA ref from prev_actual ({prev_actual}): EBITDA={ref_EBITDA} NNI={ref_NNI_EBITDA}")

    # V3: final guard -- previously None refs flowed into exp_interp_pretax
    # and crashed the whole run with a TypeError.
    if any(v is None for v in [ref_OI, ref_NNI_OI, ref_EBITDA, ref_NNI_EBITDA]):
        logger.error(f"  {t}: no usable reference values for OI/EBITDA blend")
        print(f"\n  (!) {t}: no reference values available for the OI/EBITDA blend.")
        if input(f"  'g' = accept GAAP as-is and mark done, "
                 f"Enter = skip/retry: ").strip().lower() == 'g':
            logger.warning(f"  {t}: USER ACCEPTED GAAP AS-IS for {last_actual} (no blend refs)")
            return 'gaap'
        return 'retry'

    adjusted_NI   = ortex_NI + adjustment_NI
    adjusted_eps  = av_non_gaap_eps
    adjusted_dEPS = ortex_dEPS + adjustment_per_share

    def exp_interp_pretax(ortex_val: float, ref_val: float, ref_nni: float) -> float:
        """
        Three-candidate exponentially-weighted blend matching spreadsheet formula.
        ref_nni must come from the same quarter as ref_val.
        """
        metric_adj_I  = ortex_val + adjustment_NI
        metric_adj_II = ortex_val + adjustment_NI / GROSS_UP_FACTOR

        adjusted_NNI = ortex_NI + adjustment_NI
        nni_diff = abs(adjusted_NNI - ref_nni)
        if nni_diff < 1000:
            nni_diff = 1000

        w_gaap  = (abs(metric_adj_I  - ref_val) + abs(metric_adj_II - ref_val)) / nni_diff
        w_ng_I  = (abs(ortex_val     - ref_val) + abs(metric_adj_II - ref_val)) / nni_diff
        w_ng_II = (abs(metric_adj_I  - ref_val) + abs(ortex_val     - ref_val)) / nni_diff

        weights = [w_gaap, w_ng_I, w_ng_II]
        if max(weights) > 200:
            min_w   = min(weights)
            w_gaap  = min(w_gaap  - min_w, 100)
            w_ng_I  = min(w_ng_I  - min_w, 100)
            w_ng_II = min(w_ng_II - min_w, 100)

        e_gaap  = np.exp(w_gaap)
        e_ng_I  = np.exp(w_ng_I)
        e_ng_II = np.exp(w_ng_II)

        # Spreadsheet: adj_I * exp(w_ng_I) + adj_II * exp(w_ng_II) + GAAP * exp(w_gaap)
        return (metric_adj_I * e_ng_I + metric_adj_II * e_ng_II + ortex_val * e_gaap) /                (e_gaap + e_ng_I + e_ng_II)

    adjusted_OI     = exp_interp_pretax(ortex_OI,     ref_OI,     ref_NNI_OI)
    adjusted_EBITDA = exp_interp_pretax(ortex_EBITDA, ref_EBITDA, ref_NNI_EBITDA)

    income_updates = {
        'netIncome':           adjusted_NI,
        'normalizedNetIncome': adjusted_NI,
        'eps':                 adjusted_eps,
        'dilutedEps':          adjusted_dEPS,
        'operatingIncome':     adjusted_OI,
    }
    for metric, val in income_updates.items():
        upsert_metric(t, 'income_data', last_actual, dt, metric, val, estimated=False)

    upsert_metric(t, 'summary_data', last_actual, dt, 'ebitda', adjusted_EBITDA, estimated=False)

    logger.info(f"  {t}: adjustments applied | adj_per_share={adjustment_per_share:.4f} "
                f"| adj_NI={adjustment_NI:,.0f}")

    print(f"  Non-GAAP adjustments for {t} | {last_actual}:")
    print(f"    EPS adj        : {adjustment_per_share:+.4f}  ({ortex_eps:.4f} -> {adjusted_eps:.4f})")
    print(f"    dilutedEps adj : {adjustment_per_share:+.4f}  ({ortex_dEPS:.4f} -> {adjusted_dEPS:.4f})")
    print(f"    NI adj         : {adjustment_NI:+,.0f}  ({ortex_NI:,.0f} -> {adjusted_NI:,.0f})")
    print(f"    OI adj         : {adjusted_OI - ortex_OI:+,.0f}  ({ortex_OI:,.0f} -> {adjusted_OI:,.0f})")
    print(f"    EBITDA adj     : {adjusted_EBITDA - ortex_EBITDA:+,.0f}  ({ortex_EBITDA:,.0f} -> {adjusted_EBITDA:,.0f})")
    return 'applied'


# ==============================================================================
# netIncome / normalizedNetIncome UNIFICATION
# ==============================================================================

def unify_net_income(ticker: str, download_date):
    t  = clean_ticker(ticker)
    dt = to_ts(download_date)

    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT period, estimated_values
            FROM income_data
            WHERE ticker = :t AND download_date = :d
        """), {"t": t, "d": dt}).fetchall()

    for period, is_estimated in rows:
        if is_estimated:
            nni = get_metric_value_ffill(t, 'normalizedNetIncome', period, dt, 'income_data')
            ni  = get_metric_value_ffill(t, 'netIncome',           period, dt, 'income_data')
            if nni is not None and not epsilon_equal(nni, ni):
                upsert_metric(t, 'income_data', period, dt, 'netIncome', nni, estimated=True)
        else:
            ni  = get_metric_value_ffill(t, 'netIncome',           period, dt, 'income_data')
            nni = get_metric_value_ffill(t, 'normalizedNetIncome', period, dt, 'income_data')
            if ni is not None and not epsilon_equal(ni, nni):
                upsert_metric(t, 'income_data', period, dt, 'normalizedNetIncome', ni, estimated=False)

    logger.info(f"{t}: netIncome/normalizedNetIncome unification done")


def copy_normalized_to_netincome_estimates(ticker: str, first_estimate: str, download_date):
    t  = clean_ticker(ticker)
    dt = to_ts(download_date)
    with ENGINE.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO income_data
                (ticker, download_date, period, metric_name, value, estimated_values)
            SELECT ticker, download_date, period, 'netIncome', value, TRUE
            FROM income_data
            WHERE ticker = :ticker
              AND download_date = :date
              AND metric_name = 'normalizedNetIncome'
              AND period >= :first_estimate
            ON CONFLICT (ticker, download_date, period, metric_name)
            DO UPDATE SET value = EXCLUDED.value
            WHERE income_data.value IS DISTINCT FROM EXCLUDED.value
        """), {'ticker': t, 'date': dt, 'first_estimate': first_estimate})
        rows_copied = result.rowcount
    if rows_copied > 0:
        logger.info(f"  {t}: copied {rows_copied} normalizedNetIncome -> netIncome for estimates")


# ==============================================================================
# CREDIT MANAGEMENT
# ==============================================================================

def update_credits(tracker: Dict, used: float, remaining: Optional[float]):
    tracker['used_this_round'] += used
    tracker['total_used']      += used
    if remaining is not None:
        tracker['remaining'] = remaining


def check_credits_and_confirm(tracker: Dict) -> bool:
    remaining = tracker['remaining']
    if remaining is None:
        return True
    if tracker['used_this_round'] >= CREDITS_PER_ROUND or remaining < MIN_CREDITS_REMAINING:
        print("\n" + "="*60)
        print(f"Credits used this round : {tracker['used_this_round']:.2f}")
        print(f"Total credits used      : {tracker['total_used']:.2f}")
        print(f"Credits remaining       : {remaining:.2f}")
        print("="*60)
        if input("Continue? (y/n): ").lower().strip() != 'y':
            logger.info("User chose to stop")
            return False
        tracker['used_this_round'] = 0
    return True


# ==============================================================================
# CORE: should_skip_stock_v2
# ==============================================================================

def should_skip_stock_v2(ticker: str, credits_tracker: Dict) -> Tuple[bool, Optional[str], Optional[Dict], Optional[str]]:
    t               = clean_ticker(ticker)
    first_estimated = get_estimation_status(t, 'income')

    if first_estimated is None:
        return False, None, None, None

    response = fetch_fundamentals(t, first_estimated, 'income')
    if response is None:
        return False, None, None, None

    update_credits(credits_tracker,
                   response.get('creditsUsed', 0),
                   response.get('creditsLeft'))

    if not check_credits_and_confirm(credits_tracker):
        return True, "USER_ABORT", None, None

    data      = response['data']
    estimated = data.get('estimatedValues', False)

    if not estimated:
        logger.info(f"{t} -- {first_estimated} is now actual, checking AV...")
        av_data = fetch_av_earnings(t, ortex_period=first_estimated)

        if av_data is ACCEPT_GAAP:
            logger.warning(f"{t} -- GAAP accepted as-is at AV-fetch stage for {first_estimated}")
            return False, None, ACCEPT_GAAP, first_estimated

        if av_data is None:
            if AV_OVERRIDE_IF_NA:
                print(f"\n  AV data not available for {t}.")
                raw = input(f"  Enter non-GAAP EPS for {t}, 'g' = accept GAAP "
                            f"as-is (mark done), Enter = skip: ").strip()
                if raw.lower() == 'g':
                    if input(f"  Confirm GAAP-as-is for {t} | {first_estimated}? "
                             f"(y/n): ").strip().lower() == 'y':
                        logger.warning(f"{t}: USER ACCEPTED GAAP AS-IS for "
                                       f"{first_estimated} (AV unavailable)")
                        return False, None, ACCEPT_GAAP, first_estimated
                    logger.info(f"{t} -- GAAP-as-is not confirmed, skipping stock")
                    return True, "av_data_not_ready", None, None
                if raw == '':
                    logger.info(f"{t} -- no manual EPS provided, skipping stock")
                    return True, "av_data_not_ready", None, None
                try:
                    manual_eps = float(raw)
                    av_data = {
                        'reportedEPS':      manual_eps,
                        'fiscalDateEnding': get_expected_fiscal_date(),
                        'manual_override':  True
                    }
                    logger.info(f"{t} -- using manual EPS override: {manual_eps}")
                except ValueError:
                    print(f"  Invalid EPS value '{raw}', skipping stock")
                    return True, "av_data_not_ready", None, None
            else:
                logger.info(f"{t} -- AV data not ready, skipping stock")
                return True, "av_data_not_ready", None, None

        return False, None, av_data, first_estimated

    total_rev = data.get('totalRevenues')
    last_rev  = get_last_value_db(t, 'income', first_estimated, 'totalRevenues')

    if last_rev is None:
        return False, None, None, None

    if not materially_different(total_rev, last_rev):
        return True, f"totalRevenues unchanged (<= {REV_MATERIALITY_PCT:.2%}) in {first_estimated}", None, None

    logger.info(f"{t} -- totalRevenues changed: {last_rev} -> {total_rev}")
    return False, None, None, None


# ==============================================================================
# CORE: process_ticker_v2
# ==============================================================================

def process_ticker_v2(ticker: str, category: str, all_periods: List[str],
                       credits_tracker: Dict,
                       max_period: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Returns (success, first_empty_period).
    first_empty_period: the first period that returned all-None data,
                        so caller can skip it for remaining categories.
    max_period: if set, skip all periods beyond this (inclusive) — used to
                propagate the income group's empty cutoff to cash/summary.
    """
    t     = clean_ticker(ticker)
    today = pd.Timestamp(date.today())

    fetched_today   = get_already_fetched_periods_today(t, category)
    fetched_ever    = get_all_fetched_periods_ever(t, category)
    first_estimated = get_estimation_status(t, category)

    if first_estimated is None:
        periods_to_fetch = [p for p in all_periods if p not in fetched_today]
    else:
        try:
            est_idx = all_periods.index(first_estimated)
        except ValueError:
            est_idx = len(all_periods)

        estimate_periods = [p for p in all_periods[est_idx:] if p not in fetched_today]
        new_hist         = [p for p in all_periods[:est_idx]
                            if p not in fetched_ever and p not in fetched_today]
        periods_to_fetch = new_hist + estimate_periods

    # Apply max_period cutoff from income group if provided
    if max_period is not None:
        periods_to_fetch = [p for p in periods_to_fetch if p < max_period]

    if not periods_to_fetch:
        logger.info(f"{t} | {category} -- nothing to fetch")
        return True, None

    logger.info(f"{t} | {category} -- fetching {len(periods_to_fetch)} periods")

    first_estimated_found = None
    first_empty_period    = None

    for period in sorted(periods_to_fetch):
        response = fetch_fundamentals(t, period, category)
        if response is None:
            continue

        update_credits(credits_tracker,
                       response.get('creditsUsed', 0),
                       response.get('creditsLeft'))

        if not check_credits_and_confirm(credits_tracker):
            return False, first_empty_period

        data      = response['data']
        estimated = data.get('estimatedValues', False)

        # Stop fetching further periods if this one has no data
        if _is_empty_response(data):
            logger.info(f"{t} | {category} | {period} -- empty response, stopping")
            first_empty_period = period
            break

        save_fundamentals_data(t, period, category, data, estimated, today)

        if estimated and first_estimated_found is None:
            first_estimated_found = period

        time.sleep(0.1)

    if category == 'income':
        if first_estimated_found:
            update_estimation_status(t, category, first_estimated_found)

        actual_first = detect_first_estimated_from_db(t, category)
        if actual_first:
            current = get_estimation_status(t, category)
            if actual_first != current:
                logger.info(f"{t} | {category} -- correcting status: {current} -> {actual_first}")
                update_estimation_status(t, category, actual_first)

    if category == 'income':
        effective_first = actual_first or first_estimated_found or first_estimated
        if effective_first:
            copy_normalized_to_netincome_estimates(t, effective_first, today)

    return True, first_empty_period


# ==============================================================================
# V3: RESTATEMENT MAINTENANCE (mode 5)
# ==============================================================================

def run_restatement_maintenance(stock_list: List[str], trailing_n: int,
                                 credits_tracker: Optional[Dict] = None):
    """
    Re-checks the trailing N ACTUAL quarters per stock for restatements,
    using totalRevenues as the detector (adjustments never touch revenue,
    so it is the one metric where fetched-vs-stored diff == restatement).

    Per stock, per quarter:
      1. Fetch income only (1 credit); compare totalRevenues vs latest vintage.
      2. Unchanged -> write NOTHING (the non-GAAP-adjusted vintage stays the
         latest, so downstream ffill reads are untouched).
      3. Changed   -> genuine restatement: full re-fetch (income/cash/summary)
         saved under today's vintage (old vintages preserved -> PIT intact),
         then prompt: EPS = re-apply adjustments / 'g' = accept restated GAAP
         / Enter = skip.

    NEVER touches estimation_status.
    Known blind spot (accepted by design): restatements that leave revenue
    untouched are invisible to the gate.
    """
    today = pd.Timestamp(date.today())
    if credits_tracker is None:
        credits_tracker = {'used_this_round': 0, 'total_used': 0, 'remaining': None}

    est_credits = len(stock_list) * trailing_n
    print(f"\nRestatement maintenance: {len(stock_list)} stocks x "
          f"{trailing_n} trailing actual quarters")
    print(f"Estimated credit cost (no restatements found): ~{est_credits} credits")
    print(f"(3x per restated quarter found, plus the prompt)")
    if input("Proceed? (y/n): ").strip().lower() != 'y':
        print("Cancelled.")
        return

    restatements_found = []
    for idx, ticker in enumerate(stock_list, 1):
        t   = clean_ticker(ticker)
        fep = get_estimation_status(t, 'income') or detect_first_estimated_from_db(t, 'income')
        if fep is None:
            logger.info(f"[maint] {t}: no estimation status -- skipping")
            continue
        quarters = [add_quarters(fep, -k) for k in range(trailing_n, 0, -1)]  # oldest first
        print(f"\n[{idx}/{len(stock_list)}] {t}: checking {', '.join(quarters)}")

        for period in quarters:
            response = fetch_fundamentals(t, period, 'income')
            if response is None:
                continue
            update_credits(credits_tracker,
                           response.get('creditsUsed', 0),
                           response.get('creditsLeft'))
            if not check_credits_and_confirm(credits_tracker):
                logger.info("[maint] aborted by user")
                return
            data = response['data']
            if _is_empty_response(data):
                continue

            new_rev    = to_float(data.get('totalRevenues'))
            stored_rev = get_metric_value_ffill(t, 'totalRevenues', period,
                                                 today, 'income_data')
            if new_rev is None or stored_rev is None:
                logger.info(f"[maint] {t} {period}: revenue unavailable -- skipped")
                continue
            if not materially_different(new_rev, stored_rev, rel_eps=0.0):
                continue   # no restatement: write nothing, adjusted vintage stays live

            # --- Restatement detected ---------------------------------------
            print(f"  (!) RESTATEMENT: {t} {period} revenue "
                  f"{stored_rev:,.0f} -> {new_rev:,.0f}")
            logger.warning(f"[maint] RESTATEMENT {t} {period}: "
                           f"rev {stored_rev} -> {new_rev}")
            restatements_found.append((t, period))

            estimated = data.get('estimatedValues', False)
            save_fundamentals_data(t, period, 'income', data, estimated, today)
            for category in ['cash', 'summary']:
                resp2 = fetch_fundamentals(t, period, category)
                if resp2 is not None:
                    update_credits(credits_tracker,
                                   resp2.get('creditsUsed', 0),
                                   resp2.get('creditsLeft'))
                    d2 = resp2['data']
                    save_fundamentals_data(t, period, category, d2,
                                           d2.get('estimatedValues', False), today)
                    time.sleep(0.1)

            raw = input(f"  Enter non-GAAP EPS to re-apply adjustments for {t} "
                        f"{period}, 'g' = accept restated GAAP, "
                        f"Enter = skip: ").strip()
            if raw.lower() == 'g':
                logger.warning(f"[maint] {t} {period}: USER ACCEPTED RESTATED GAAP")
                continue
            if raw == '':
                logger.warning(f"[maint] {t} {period}: restated GAAP saved, "
                               f"re-adjustment SKIPPED (no auto-retry: maintenance is manual)")
                continue
            try:
                manual_eps = float(raw)
            except ValueError:
                print(f"  Invalid '{raw}', skipping re-adjustment")
                continue
            av_data = {'reportedEPS': manual_eps,
                       'fiscalDateEnding': 'maintenance_override'}
            res = apply_nongaap_adjustments(t, period, av_data, today)
            logger.warning(f"[maint] {t} {period}: re-adjustment result = {res}")
            time.sleep(0.1)

    print(f"\nMaintenance done. Restatements found: {len(restatements_found)}")
    for t, p in restatements_found:
        print(f"  - {t} {p}")
    logger.info(f"[maint] complete | restatements: {restatements_found} | "
                f"credits used: {credits_tracker['total_used']:.2f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main(stock_list: List[str], start_period: str, end_period: str,
         lastEarnings_l: Optional[pd.Series] = None):
    _check_credentials()
    logger.info("Ortex Fetcher V3 starting")
    logger.info(f"Tickers: {len(stock_list)} | Range: {start_period} -> {end_period}")

    print("\n" + "="*70)
    print("SELECT FETCH MODE")
    print("="*70)
    print("1. Fetch ALL stocks")
    print("2. Fetch only stocks with earnings date < today")
    print("3. Fetch only stocks with earnings in last 10 days")
    print("4. Fetch only stocks with earnings in last 5 days (ULTRA FAST)")
    print("5. RESTATEMENT MAINTENANCE -- re-check trailing actual quarters")
    print("="*70)

    while True:
        mode = input("\nSelect mode (1/2/3/4/5): ").strip()
        if mode in ['1', '2', '3', '4', '5']:
            break

    mode  = int(mode)
    today = pd.Timestamp(date.today())

    if mode == 5:
        raw_n = input(f"Trailing actual quarters to re-check "
                      f"(default {MAINT_TRAILING_N_DEFAULT}): ").strip()
        trailing_n = int(raw_n) if raw_n.isdigit() and int(raw_n) > 0                      else MAINT_TRAILING_N_DEFAULT
        initialize_database()
        run_restatement_maintenance(stock_list, trailing_n)
        return

    if mode == 1:
        filtered_stocks = stock_list
        print(f"\nMode 1: processing all {len(stock_list)} stocks")
    else:
        if lastEarnings_l is None:
            print("\nlastEarnings_l required for mode 2/3/4. Exiting.")
            return
        threshold = {2: None, 3: 10, 4: 5}[mode]
        filtered_stocks = []
        for t in stock_list:
            if t not in lastEarnings_l.index:
                print(f"  (!) {t} not in earnings list -- skipping")
                continue
            ed       = to_ts(lastEarnings_l[t])
            days_ago = (today - ed).days
            if mode == 2 and ed < today:
                filtered_stocks.append(t)
            elif mode in (3, 4) and 0 <= days_ago <= threshold:
                filtered_stocks.append(t)
        print(f"\nMode {mode}: {len(filtered_stocks)} stocks after filtering")

    if not filtered_stocks:
        print("\nNo stocks to process.")
        return

    global AV_OVERRIDE_IF_NA
    print("\n" + "="*70)
    print("AV OVERRIDE OPTION")
    print("="*70)
    print("If AlphaVantage has no EPS data yet for a stock that just reported,")
    print("override allows you to enter the non-GAAP EPS manually.")
    av_ans = input("\nOverride AV if N/A? (y/n): ").strip().lower()
    AV_OVERRIDE_IF_NA = (av_ans == 'y')
    if AV_OVERRIDE_IF_NA:
        print("  Override ON -- you will be prompted for manual EPS when AV returns N/A")
    else:
        print("  Override OFF -- stocks with no AV data will be skipped as usual")

    if input(f"\nProceed with {len(filtered_stocks)} stocks? (y/n): ").lower().strip() != 'y':
        print("Cancelled.")
        return

    initialize_database()
    all_periods = generate_periods(start_period, end_period)
    logger.info(f"Periods in range: {len(all_periods)}")

    credits_tracker = {'used_this_round': 0, 'total_used': 0, 'remaining': None}
    categories      = ['income', 'cash', 'summary']
    total           = len(filtered_stocks)

    for idx, ticker in enumerate(filtered_stocks, 1):
        t = clean_ticker(ticker)
        print("\n" + "#"*70)
        print(f"  STOCK {idx}/{total}: {t}  ({(idx-1)/total*100:.1f}% done)")
        print("#"*70)
        logger.info(f"\n[{idx}/{total}] {t}")

        try:
            should_skip, skip_reason, av_data, last_actual = should_skip_stock_v2(t, credits_tracker)
        except RuntimeError as e:
            print(f"\n  {e}")
            logger.error(str(e))
            break

        if should_skip:
            if skip_reason == "USER_ABORT":
                logger.info("Aborted by user")
                break
            logger.info(f"Skipping {t} -- {skip_reason}")
            print(f"  Skipped: {skip_reason}")
            continue

        adj_result = None   # V3: 'applied' | 'gaap' | 'retry' | None (no release)
        if av_data is not None and last_actual is not None:
            first_estimate = add_quarters(last_actual, 1)

            print(f"  Earnings released! Fetching fresh actual for {last_actual}...")
            for category in categories:
                response = fetch_fundamentals(t, last_actual, category)
                if response is not None:
                    update_credits(credits_tracker,
                                   response.get('creditsUsed', 0),
                                   response.get('creditsLeft'))
                    data      = response['data']
                    estimated = data.get('estimatedValues', False)
                    save_fundamentals_data(t, last_actual, category, data, estimated, today)
                    time.sleep(0.1)

            if av_data is ACCEPT_GAAP:
                adj_result = 'gaap'
                print(f"  GAAP accepted as-is for {t} -- skipping non-GAAP adjustments.")
                logger.warning(f"{t}: GAAP AS-IS for {last_actual} -- "
                               f"fresh actuals saved, no adjustments applied")
            else:
                print(f"  Fetching fresh FED estimates for {first_estimate} (adjustment reference)...")
                for category in ['income', 'summary']:
                    response = fetch_fundamentals(t, first_estimate, category)
                    if response is not None:
                        update_credits(credits_tracker,
                                       response.get('creditsUsed', 0),
                                       response.get('creditsLeft'))
                        data      = response['data']
                        estimated = data.get('estimatedValues', True)
                        save_fundamentals_data(t, first_estimate, category, data, estimated, today)
                        time.sleep(0.1)

                print(f"  Applying AV non-GAAP adjustments for {last_actual}...")
                adj_result = apply_nongaap_adjustments(t, last_actual, av_data, today)
                if adj_result == 'retry':
                    print(f"  Adjustments deferred for {t} -- status NOT advanced, will retry next run.")

        income_empty_cutoff = None   # propagated from income to cash/summary
        for cat_idx, category in enumerate(categories, 1):
            print(f"  |- {cat_idx}/3: {category}")
            # Pass income's empty-period cutoff to cash and summary
            max_p = income_empty_cutoff if category != 'income' else None
            ok, empty_cutoff = process_ticker_v2(t, category, all_periods,
                                                  credits_tracker, max_period=max_p)
            if category == 'income':
                income_empty_cutoff = empty_cutoff  # propagate to cash + summary
            if not ok:
                logger.info("Stopped by user")
                break
        else:
            income_fep = get_estimation_status(t, 'income')
            if income_fep:
                for cat in ['cash', 'summary']:
                    if get_estimation_status(t, cat) != income_fep:
                        update_estimation_status(t, cat, income_fep)
                        logger.info(f"{t} | {cat} -- synced to income: {income_fep}")

            # V3: status advances ONLY on success or explicit GAAP acceptance.
            # 'retry' leaves status untouched -> stock re-enters the
            # earnings-release path on the next run.
            if last_actual is not None and adj_result in ('applied', 'gaap'):
                new_fep = get_next_period(last_actual)
                for cat in ['income', 'cash', 'summary']:
                    update_estimation_status(t, cat, new_fep)
                logger.info(f"{t} -- estimation_status advanced to {new_fep} ({adj_result})")

            unify_net_income(t, today)
            continue
        break

    print("\n" + "#"*70)
    print(f"  Done -- {total} stocks processed")
    print("#"*70)
    logger.info(f"Session complete | credits used: {credits_tracker['total_used']:.2f} | "
                f"remaining: {credits_tracker.get('remaining', 'unknown')}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

SOFTWARE_L = ['DDOG', 'SNOW', 'MDB', 'ZS', 'PLTR', 'RBRK', 'APP', 'CRWD', 'NET', 'TWLO', 'SHOP', 'AXON', 
              'ADSK', 'SNPS', 'CDNS', 'GTLB']
HARDWARE_L = ['NVDA', 'AVGO', 'MRVL', 'AMD', 'AMAT', 'LRCX', 'MU', 'SNDK', 'LITE', 'AAOI', 'TER', 'VIAV', 
              'FORM', 'ALAB', 'CRDO', 'VRT', 'SITM', 'CLS']

# NOTE (V3): the block below runs inside a Jupyter kernel where Set_DF,
# DD_Index, openF_df and `engine` are pre-loaded from other cells.
# It is NOT standalone-runnable and is intentionally left as-is.
if __name__ == "__main__":
            
    joker_dikt = {'ZI US': 'GTM US', 'BIGC US': 'CMRC US'}
    lastEarnings_s = None
    
    query_earnings = "SELECT * FROM ed_relation"
    ed_df = pd.read_sql_query(query_earnings, engine)
    ed_df = Set_DF(ed_df)
    ed_df = DD_Index(ed_df)      
    ed_df = ed_df.rename(columns = joker_dikt)
    Pxs_df = openF_df('prices_relation')
    Pxs_df = Pxs_df.rename(columns = joker_dikt)
    vayf_df = openF_df('va_yf')    
    stock_l = [d for d in ed_df.columns if d in vayf_df.index and d in Pxs_df.columns]
    stock_l = pd.Series(stock_l)[pd.Series(stock_l).map(lambda x: x.split(' ')[-1]) == 'US']
    stock_l = Pxs_df[stock_l][Pxs_df[stock_l][-10:].pct_change().rolling(5).sum() != 0].iloc[-1].dropna().index.tolist()
    stock_l = pd.Series(stock_l).map(lambda x: x.split(' ')[0]).tolist()
    lastEarnings_s = ed_df.iloc[-1].map(mmdd_to_date)   # V3: year-boundary safe
    lastEarnings_s.index = lastEarnings_s.index.map(lambda x: vayf_df.loc[x, 'YF Ticker'])
    
    # Get earnings dates (assume you have a Series with ticker as index, earnings_date as value)
    # Example: lastEarnings_s = pd.Series({'AAPL': '2026-01-30', 'MSFT': '2026-01-25', ...})
    # For now, set to None to use mode 1 (fetch all)
        
    # er_lag lag, run everyday on mode (1)
    er_lag_l = [2, 3, 4]
    er_lag_l = [1]
    # stock_l = []
    # for er_lag in er_lag_l:
    #     stock_l += ed_df.iloc[-1][ed_df.iloc[-1] == Pxs_df.index[-er_lag].strftime('%m%d')].index.map(lambda x: x.split(' ')[0]).tolist()         
    
    # stock_l = ['MRVL']  # Replace with your ~700 ticker list
    # stock_l = HARDWARE_L
    START_PERIOD = '2018Q1'
    END_PERIOD = '2030Q4'    

    # ------------------------------------------------------------------
    # CONFIGURE YOUR RUN HERE
    # ------------------------------------------------------------------

    # Tickers to process -- bare or with ' US' suffix, both work            

    # Earnings dates -- only needed for modes 2/3/4
    # Index = ticker (bare or with suffix), value = any date format
    # Set to None to use mode 1 (all stocks)
    
    # Example:
    # lastEarnings = pd.Series({
    #     'AAPL': '2026-01-30',
    #     'MSFT': '2026-01-25',
    # })

    main(
        stock_list     = stock_l,
        start_period   = START_PERIOD,
        end_period     = END_PERIOD,
        lastEarnings_l = lastEarnings_s
    )

