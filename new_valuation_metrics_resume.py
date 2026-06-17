#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Calculate All Valuation Metrics -- v2 (June 2026)
=================================================

Ground-up correction of the feature-engineering layer. Behavioural changes vs v1,
all discussed and agreed:

  FEP determination    : computed ONCE per (stock, date) from totalRevenues (the
                         ground-truth series) and inherited by every metric incl.
                         FCF. A stock's fiscal convention is fixed, so FEP motion
                         between dates is governed only by elapsed time: predict
                         FEP = current_fep - round(dt_gap / 91.25) quarters, then
                         confirm against the observed actual/estimate frontier
                         (a quarter is ACTUAL iff its rounded revenue is vintage-
                         stable; ESTIMATEs still drift) within +/-1 quarter of the
                         prediction. No clean frontier -> calendar N, logged.
                         Rounding applied before all comparisons (vendor noise).
  strict sums          : LTM/NTM dollar denominators require >=3 of 4 quarters and
                         annualise (sum * 4/n). Pair-completeness for PIG/PSG and
                         GGP growths. Replaces silent series.get(q, 0) partial sums.
  forward growth (GE)  : arithmetic switch at ratio<=1 (continuous, monotone, no
                         complex numbers); arithmetic final aggregation. |fwd| base
                         retained (hyper-growth compression). Applies to GS/GE/GGP.
  FCF table            : materialised builder (see build_fcf_table). Ortex capex sign
                         corrected (CFO + capex, capex stored negative). Legacy wide
                         tables mapped purely by value-matching (labels are NOT PIT).
  universe             : per-calc-date eligibility (no today-anchored [-10:]/iloc[-1]).
  modes                : validate (subset -> side tables + diff/coverage report),
                         rebuild (burn + regenerate existing DISTINCT dates),
                         incremental (single new date = last Pxs_df date, skip if present).
  hygiene              : env-var credentials, safe() logging, dead imports removed.

CONVENTION: SQL is used ONLY to fetch rows into DataFrames. All data manipulation
is pandas. (No GROUP BY / ROW_NUMBER analytics in SQL.)
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------------- #
# Logging
# ----------------------------------------------------------------------------- #
logger = logging.getLogger("valuation_metrics_v2")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------- #
# Credentials / engines  (env-var driven; set in the kernel before running)
#   os.environ['FACTORMODEL_DB_URL'] = 'postgresql+psycopg2://postgres:<pwd>@localhost:5432/factormodel_db'
#   os.environ['VISIBLEALPHA_DB_URL'] = 'postgresql+psycopg2://postgres:<pwd>@localhost:5432/visiblealpha_laptop'
# ----------------------------------------------------------------------------- #
CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
VA_CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/visiblealpha_laptop"

ENGINE  = create_engine(CONNECTION_STRING)
VENGINE = create_engine(VA_CONNECTION_STRING)   # legacy VisibleAlpha DB (prices, ed_relation, va_yf, wide FCF tables)

# Production / validation table names
# This script manages valuation_metrics_anchors only -- it is the live table used
# downstream, and the sole write/rebuild target. valuation_consolidated is NOT
# touched by this script (never written, renamed, or wiped): it is left exactly as
# it is, serving as a frozen backup of the prior synced state. A rebuild additionally
# snapshots anchors to a timestamped *_backup_<stamp> before regenerating it.
TABLES          = ['valuation_metrics_anchors']
VALIDATE_TABLES = ['valuation_metrics_anchors_validate']
# Reads (canonical calc-date set + diff baseline) also come from anchors.
DATE_SOURCE_TABLE = 'valuation_metrics_anchors'
# Rebuild writes to a FRESH table (non-destructive): valuation_metrics_anchors and
# valuation_consolidated are left exactly as they are, so the live pipeline can keep
# running on the old data while the new table builds. Cut over downstream by pointing
# at REBUILD_TABLE when ready. Dates/diff baseline still come from anchors.
REBUILD_TABLE = 'valuation_metrics_v2'
# Minimum calendar-day spacing between consecutive calc dates. Inherited dates that
# fall within MINIMUM_LAG days of the last kept date are dropped (valuation metrics
# don't move fast enough to justify near-adjacent recomputation). Applied in both
# validate and rebuild so the dry-run mirrors the real run.
MINIMUM_LAG = 10
# Where validate-mode CSV snapshots are written. Set to your local path.
OUTPUT_DIR = r"C:\Users\Utilizador\OneDrive\Documentos\Malta\Systematic"
FCF_TABLE       = 'fcf_consolidated'                 # materialised (ticker, period, download_date, fcf)
FCF_TABLE_VALID = 'fcf_consolidated_validate'
RESTATEMENT_LOG = 'restatement_log'                  # (ticker, period, download_date) -- excluded from matching

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
QUARTER_DAYS          = 91.25     # mean reporting cycle (avoids drift vs 90 over ~9y)
FEP_SEARCH_WINDOW     = 1         # value-match search radius around predicted FEP (quarters)
FEP_FRONTIER_TOL      = 0.001     # rel tolerance for "value unchanged across vintages" (frontier)
REV_ROUND_UNIT        = 1000      # thousand-rounding for revenue value matching
USE_LEGACY_FCF        = False     # splice legacy {ticker}_fcf wide tables (disabled: see build_fcf_table)
FCF_MATCH_MIN_RUN     = 3         # exact run length for legacy FCF column matching
FCF_ALIGN_SEP_RATIO   = 5.0       # tolerant-align: best error must beat 2nd-best by this
FCF_ALIGN_MAX_ERR     = 0.02      # tolerant-align: best median rel-err must be below this
FCF_VALUE_SUSPECT_ERR = 0.02      # alignment accepted but values flagged suspect above this


def _check_credentials():
    # Connection strings are configured inline above.
    return True


# ----------------------------------------------------------------------------- #
# Legacy DB helpers (replicate v1 Set_DF / DD_Index / openF_df; fetch-only SQL)
# ----------------------------------------------------------------------------- #
def _set_df(dframe):
    first = dframe.columns.values.tolist()[0]
    dframe.index = dframe[first]
    dframe.index.name = first
    return dframe.drop(first, axis=1)

def _dd_index(dframe):
    dframe = dframe.copy()
    dframe['__dummy__'] = dframe.index
    dframe = dframe.drop_duplicates(['__dummy__']).drop('__dummy__', axis=1)
    return dframe

def open_factor_df(table_name):
    """Fetch a whole table from the factor DB and shape it (index = first col)."""
    df = pd.read_sql_query(f'SELECT * FROM {table_name}', ENGINE)
    return _dd_index(_set_df(df)).sort_index()

def open_va_df(table_name):
    """Fetch a whole table from the legacy VisibleAlpha DB and shape it."""
    df = pd.read_sql_query(f'SELECT * FROM {table_name}', VENGINE)
    return _dd_index(_set_df(df)).sort_index()


def normalize_ticker(ticker):
    """'AXON US' or 'AXON' -> 'AXON' for DB queries."""
    return ticker.strip().split(' ')[0].upper()


# ----------------------------------------------------------------------------- #
# Period arithmetic ('YYYYQn')
# ----------------------------------------------------------------------------- #
def period_to_int(period_str):
    """'2024Q3' -> 2024*4 + 2 (zero-based quarter). Monotone, for offset math."""
    y = int(period_str[:4]); q = int(period_str[5])
    return y * 4 + (q - 1)

def int_to_period(idx):
    y, q = divmod(idx, 4)
    return f"{y}Q{q + 1}"

def shift_period(period_str, n):
    """Shift a period by n quarters (n may be negative)."""
    return int_to_period(period_to_int(period_str) + n)

def get_quarters_before_period(period_str, n_quarters):
    """n quarters before period (ascending). '2024Q3',3 -> ['2023Q4','2024Q1','2024Q2']."""
    base = period_to_int(period_str)
    return [int_to_period(base - i) for i in range(n_quarters, 0, -1)]

def get_quarters_after_period(period_str, n_quarters):
    """n quarters after period (ascending)."""
    base = period_to_int(period_str)
    return [int_to_period(base + i) for i in range(1, n_quarters + 1)]

def next_period(period_str):
    return shift_period(period_str, 1)

def prev_period(period_str):
    return shift_period(period_str, -1)


# ----------------------------------------------------------------------------- #
# Strict sum: >=min_q of the requested quarters, annualised (sum * len/n)
# ----------------------------------------------------------------------------- #
def strict_sum(series, quarters, min_q=3, annualize_to=None):
    """
    Sum `series` over `quarters`, requiring at least `min_q` present & finite.
    Annualises to `annualize_to` quarters (default len(quarters)) via * target/n.
    Returns (value, n_used) or (None, n_used) if under threshold.
    """
    target = annualize_to if annualize_to is not None else len(quarters)
    vals = [series.get(q, np.nan) for q in quarters]
    present = [v for v in vals if pd.notna(v)]
    n = len(present)
    if n < min_q:
        return None, n
    return sum(present) * (target / n), n


# ----------------------------------------------------------------------------- #
# Restatement log (read once per run; used to exclude rewritten quarters)
# ----------------------------------------------------------------------------- #
def load_restatement_log():
    """Fetch the restatement log into a DataFrame; empty frame if table absent."""
    try:
        df = pd.read_sql_query(f'SELECT ticker, period, download_date FROM {RESTATEMENT_LOG}', ENGINE)
        df['download_date'] = pd.to_datetime(df['download_date'])
        return df
    except Exception:
        return pd.DataFrame(columns=['ticker', 'period', 'download_date'])


# ----------------------------------------------------------------------------- #
# Per-ticker vintage cache (fetch all income revenue vintages once per ticker)
# ----------------------------------------------------------------------------- #
_REV_VINTAGE_CACHE = {}     # db_ticker -> tidy DataFrame [period, download_date, value]
_FEP_CACHE         = {}     # (db_ticker, as_of_ts) -> first_estimated_period
_ESTSTATUS_CACHE   = {}     # db_ticker -> current first_estimated_period

def _reset_caches():
    _REV_VINTAGE_CACHE.clear()
    _FEP_CACHE.clear()
    _ESTSTATUS_CACHE.clear()


def fetch_revenue_vintages(db_ticker):
    """All totalRevenues rows (every vintage) for a ticker, as a tidy DataFrame."""
    if db_ticker in _REV_VINTAGE_CACHE:
        return _REV_VINTAGE_CACHE[db_ticker]
    q = """
        SELECT period, download_date, value
        FROM income_data
        WHERE ticker = %(ticker)s AND metric_name = 'totalRevenues'
    """
    df = pd.read_sql_query(q, ENGINE, params={'ticker': db_ticker})
    if not df.empty:
        df['download_date'] = pd.to_datetime(df['download_date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    _REV_VINTAGE_CACHE[db_ticker] = df
    return df


def get_current_fep(db_ticker):
    """Trusted current FEP from estimation_status (income)."""
    if db_ticker in _ESTSTATUS_CACHE:
        return _ESTSTATUS_CACHE[db_ticker]
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period FROM estimation_status
            WHERE ticker = :ticker AND category = 'income'
        """), {'ticker': db_ticker}).fetchone()
    fep = row[0] if row else None
    _ESTSTATUS_CACHE[db_ticker] = fep
    return fep


def _vintage_value_at(rev_df, period, as_of_ts):
    """Latest-non-null rounded totalRevenues for `period` as of a cutoff (or None)."""
    sub = rev_df[(rev_df['period'] == period) &
                 (rev_df['download_date'] <= as_of_ts) &
                 (rev_df['value'].notna())]
    if sub.empty:
        return None
    v = sub.sort_values('download_date')['value'].iloc[-1]
    return round(v / REV_ROUND_UNIT)


def get_first_estimated_period(ticker, as_of_date=None, restatement_df=None):
    """
    FEP determination -- revenue-anchored, independent-predict.

    The first-estimated-period for a (stock, date) is found ONCE here from
    totalRevenues (the ground-truth series) and inherited by every metric incl.
    FCF. Method:

      1. Anchor: current FEP from estimation_status, valid at MRUD (latest vintage).
      2. Predict: a stock's fiscal convention is fixed, so FEP motion between dates
         is governed only by elapsed time. Predicted FEP at the as-of date is
         current_fep shifted back by round(gap_days / QUARTER_DAYS) quarters.
      3. Confirm by value: within +/-FEP_SEARCH_WINDOW quarters of the prediction,
         find the observed actual/estimate frontier. An ACTUAL quarter's rounded
         totalRevenues is vintage-stable (unchanged from the as-of snapshot to the
         final snapshot); an ESTIMATE still drifts. FEP = first period at/after
         which the value is NOT vintage-stable. Restated periods are skipped.
      4. Fallback: if the window yields no clean frontier, use the calendar
         prediction N (logged).

    Rounding (REV_ROUND_UNIT) is applied before every comparison so vendor-side
    numerical noise never spuriously breaks a match.
    """
    db_ticker = normalize_ticker(ticker)
    current_fep = get_current_fep(db_ticker)

    if as_of_date is None:
        return current_fep

    as_of_ts = pd.Timestamp(as_of_date)
    cache_key = (db_ticker, as_of_ts)
    if cache_key in _FEP_CACHE:
        return _FEP_CACHE[cache_key]

    if current_fep is None:
        _FEP_CACHE[cache_key] = None
        return None

    rev_df = fetch_revenue_vintages(db_ticker)
    if rev_df.empty:
        _FEP_CACHE[cache_key] = None
        return None

    mrud = rev_df['download_date'].max()
    as_of_cutoff_rows = rev_df[rev_df['download_date'] <= as_of_ts]
    if as_of_cutoff_rows.empty:
        _FEP_CACHE[cache_key] = None
        return None
    as_of_cutoff = as_of_cutoff_rows['download_date'].max()

    # at/after the latest vintage -> estimation_status is exact
    if as_of_cutoff >= mrud:
        _FEP_CACHE[cache_key] = current_fep
        return current_fep

    # --- predict FEP from elapsed quarters (fiscal offset cancels out) ---
    gap_days = (mrud - as_of_cutoff).days
    N = int(round(gap_days / QUARTER_DAYS))
    cur_fep_idx = period_to_int(current_fep)
    pred_idx = cur_fep_idx - N
    n_default = int_to_period(pred_idx)

    # restated (period) set to skip when judging the frontier
    restated = set()
    if restatement_df is not None and not restatement_df.empty:
        rsub = restatement_df[restatement_df['ticker'] == db_ticker]
        restated = set(rsub['period'].tolist())

    # --- confirm by value: scan the narrow window for the actual/estimate frontier ---
    # A period is an ACTUAL iff its rounded value as-of equals the final rounded value
    # (vintage-stable). FEP = (highest stable-actual period in the window) + 1. Scan
    # the whole window and track the top stable actual rather than the first drift, so
    # a prediction that sits slightly high can't skip the true frontier.
    lo_idx = pred_idx - FEP_SEARCH_WINDOW - 1   # -1: see the actual just below prediction
    hi_idx = pred_idx + FEP_SEARCH_WINDOW
    last_actual_idx = None
    saw_any = False
    for p_idx in range(lo_idx, hi_idx + 1):
        p = int_to_period(p_idx)
        if p in restated:
            # restated periods are ambiguous for the frontier; treat as actual-side
            # (they are historical, reported quarters) so they don't break the run
            if saw_any and last_actual_idx is not None and p_idx == last_actual_idx + 1:
                last_actual_idx = p_idx
            continue
        v_asof = _vintage_value_at(rev_df, p, as_of_cutoff)
        v_final = _vintage_value_at(rev_df, p, mrud)
        if v_asof is None or v_final is None:
            continue
        saw_any = True
        denom = max(abs(v_final), 1)
        stable = abs(v_asof - v_final) / denom <= FEP_FRONTIER_TOL
        if stable:
            last_actual_idx = p_idx

    if last_actual_idx is not None:
        chosen = int_to_period(last_actual_idx + 1)
        # sanity: frontier should sit within +/-1 of the calendar prediction; if the
        # value scan wandered further, trust the calendar prediction instead.
        if abs(period_to_int(chosen) - pred_idx) > FEP_SEARCH_WINDOW:
            chosen = n_default
            logger.debug(f"[FEP fallback] {db_ticker} as_of={as_of_ts.date()} "
                         f"-> frontier far from pred, using N={n_default}")
    else:
        chosen = n_default
        logger.debug(f"[FEP fallback] {db_ticker} as_of={as_of_ts.date()} "
                     f"-> no stable actual in window, N={n_default}")

    _FEP_CACHE[cache_key] = chosen
    return chosen


# ----------------------------------------------------------------------------- #
# Metric -> table routing
# ----------------------------------------------------------------------------- #
_CASH_METRICS    = {'cashFromOperations', 'capitalExpenditure', 'cashFromInvesting',
                    'cashFromFinancing', 'netChangeInCash'}
_SUMMARY_METRICS = {'cashAndCashEquivalents', 'debt', 'netDebt', 'ebitda'}

def _get_table_for_metric(metric_name):
    if metric_name in _CASH_METRICS:
        return 'cash_data'
    if metric_name in _SUMMARY_METRICS:
        return 'summary_data'
    return 'income_data'


def get_metric_data(ticker, metric_name, as_of_date, estimated_flag=False,
                    restatement_df=None):
    """
    Metric series for a ticker as of a date, split actual/estimated by FEP.
    SQL fetches the ticker/metric rows; pandas does the latest-non-null vintage
    pick, ffill, and the actual/estimate split.
    """
    db_ticker = normalize_ticker(ticker)
    table     = _get_table_for_metric(metric_name)
    as_of_ts  = pd.Timestamp(as_of_date)

    q = f"""
        SELECT period, download_date, value
        FROM {table}
        WHERE ticker = %(ticker)s AND metric_name = %(metric)s
    """
    raw = pd.read_sql_query(q, ENGINE, params={'ticker': db_ticker, 'metric': metric_name})
    if raw.empty:
        return pd.DataFrame()

    raw['download_date'] = pd.to_datetime(raw['download_date'])
    raw['value'] = pd.to_numeric(raw['value'], errors='coerce')
    raw = raw[raw['download_date'] <= as_of_ts]
    if raw.empty:
        return pd.DataFrame()

    # latest non-null vintage per period
    raw['is_null'] = raw['value'].isna()
    raw = raw.sort_values(['period', 'is_null', 'download_date'],
                          ascending=[True, True, False])
    latest = raw.groupby('period', as_index=False).first()[['period', 'value', 'download_date']]
    latest = latest.sort_values('period').reset_index(drop=True)
    latest['value'] = latest['value'].ffill()

    first_estimated = get_first_estimated_period(ticker, as_of_date, restatement_df)
    if first_estimated is None:
        return pd.DataFrame() if estimated_flag else latest

    if estimated_flag:
        return latest[latest['period'] >= first_estimated].reset_index(drop=True)
    return latest[latest['period'] < first_estimated].reset_index(drop=True)


def get_normalized_ni(ticker, as_of_date, estimated_flag=False, restatement_df=None):
    """normalizedNetIncome, falling back to netIncome if entirely unavailable."""
    df = get_metric_data(ticker, 'normalizedNetIncome', as_of_date, estimated_flag, restatement_df)
    if not df.empty and df['value'].notna().any():
        return df
    return get_metric_data(ticker, 'netIncome', as_of_date, estimated_flag, restatement_df)


def get_last_actual_period(ticker, as_of_date, restatement_df=None):
    """Quarter immediately before FEP."""
    fep = get_first_estimated_period(ticker, as_of_date, restatement_df)
    if fep is None:
        return None
    return prev_period(fep)


# ============================================================================= #
# FORWARD GROWTH (GS / GE / GGP) -- arithmetic switch at ratio<=1, no complex
# ============================================================================= #
def fill_normalized_ni(ticker, as_of_date, quarters, restatement_df=None):
    """normalizedNetIncome estimates for requested quarters (netIncome fallback)."""
    norm_df = get_metric_data(ticker, 'normalizedNetIncome', as_of_date, True, restatement_df)
    norm_series = (pd.Series(norm_df['value'].values, index=norm_df['period'])
                   if not norm_df.empty else pd.Series(dtype=float))
    if not norm_series.empty:
        return {q: norm_series[q] for q in quarters if q in norm_series.index}
    ni_df = get_metric_data(ticker, 'netIncome', as_of_date, True, restatement_df)
    ni_series = (pd.Series(ni_df['value'].values, index=ni_df['period'])
                 if not ni_df.empty else pd.Series(dtype=float))
    return {q: ni_series[q] for q in quarters if q in ni_series.index}


def sym_growth(b, a):
    """
    Symmetric, blow-up-proof growth of a metric going from a -> b:

        (b - a) / ((|a| + |b|) / 2),  floored at -1.0 (-100%)

    Rationale (used uniformly for every metric-to-metric growth in this module):
      - Denominator uses |a|,|b| (not the signed midpoint), so it only vanishes when
        BOTH endpoints are ~0 -- i.e. genuinely no signal -> NaN. This removes the
        near-zero / sign-flip explosions (e.g. EPS or FCF crossing zero) that an
        (a+b)/2 or /b denominator produce.
      - mean-of-abs keeps ~2x the cross-sectional spread of max(|a|,|b|) in the common
        same-sign regime; it saturates at -2 for any positive->negative crossing, which
        is acceptable for a long-only book (extreme-bearish names only need to rank at
        the bottom, not be finely ordered among themselves).
      - Floor at -1.0: a >=50% drop / sign flip is "bad enough" for ranking; finer
        gradation below that is noise. No upper cap here -- winsorisation/scaling is
        applied per-factor downstream.

    Where a ratio/factor is needed (e.g. feeding _annualized_growth), use 1 + sym_growth,
    which floors the ratio at 0 (never negative) -- exactly what the annualiser expects.

    Returns np.nan when both endpoints are missing or zero.
    """
    if pd.isna(a) or pd.isna(b):
        return np.nan
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return np.nan
    return max((b - a) / denom, -1.0)


def _annualized_growth(ratio, n):
    """
    Annualise a forward ratio to a per-year growth.
      ratio  > 1 : geometric  ratio**(1/n) - 1   (identical to v1 in this regime)
      ratio <= 1 : arithmetic (ratio - 1) / n    (continuous & monotone at ratio=1,
                   no fractional power of a non-positive number -> no complex values)
    """
    if ratio > 1:
        return ratio ** (1.0 / n) - 1.0
    return (ratio - 1.0) / n


def calculate_forward_growth(ticker, as_of_date, metric_name, restatement_df=None):
    """
    Forward growth: arithmetic mean of per-year annualised growths vs the last
    4 actual quarters. |fwd| base retained (hyper-growth compression). Strict:
    each forward year requires all 4 estimate quarters; actual base requires >=3.
    """
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None

    actual_quarters = get_quarters_before_period(last_actual, 3) + [last_actual]
    est_q_y1 = get_quarters_after_period(last_actual, 4)
    est_q_y2 = get_quarters_after_period(est_q_y1[-1], 4)
    est_q_y3 = get_quarters_after_period(est_q_y2[-1], 4)

    if metric_name == 'normalizedNetIncome':
        actual_df = get_normalized_ni(ticker, as_of_date, False, restatement_df)
    else:
        actual_df = get_metric_data(ticker, metric_name, as_of_date, False, restatement_df)
    if actual_df.empty:
        return None
    actual_series = pd.Series(actual_df['value'].values, index=actual_df['period'])

    if metric_name == 'normalizedNetIncome':
        all_est = est_q_y1 + est_q_y2 + est_q_y3
        est_series = pd.Series(fill_normalized_ni(ticker, as_of_date, all_est, restatement_df))
    else:
        est_df = get_metric_data(ticker, metric_name, as_of_date, True, restatement_df)
        est_series = (pd.Series(est_df['value'].values, index=est_df['period'])
                      if not est_df.empty else pd.Series(dtype=float))

    # actual base: >=3 of 4, annualised to 4
    actual_base, _ = strict_sum(actual_series, actual_quarters, min_q=3, annualize_to=4)
    if actual_base is None or actual_base == 0:
        return None

    def fwd_sum(quarters):
        # forward years require the full 4 quarters (consensus completeness)
        val, n = strict_sum(est_series, quarters, min_q=4, annualize_to=4)
        return val

    fwd1, fwd2, fwd3 = fwd_sum(est_q_y1), fwd_sum(est_q_y2), fwd_sum(est_q_y3)
    if fwd1 is None:
        return None

    growths = []
    r1 = 1.0 + sym_growth(fwd1, actual_base)      # ratio for the annualiser (floored >=0)
    growths.append(_annualized_growth(r1, 1))

    for yr, fwd in [(2, fwd2), (3, fwd3)]:
        if fwd is None:
            continue
        r = 1.0 + sym_growth(fwd, actual_base)
        growths.append(_annualized_growth(r, yr))

    if not growths:
        return None
    return round(float(np.mean(growths)) * 100, 2)


def calculate_gs(ticker, as_of_date, restatement_df=None):
    return calculate_forward_growth(ticker, as_of_date, 'totalRevenues', restatement_df)

def calculate_ge(ticker, as_of_date, restatement_df=None):
    return calculate_forward_growth(ticker, as_of_date, 'normalizedNetIncome', restatement_df)


def calculate_ggp(ticker, as_of_date, restatement_df=None):
    """GGP = GS scaled by (GP actual growth / Sales actual growth). Strict sums."""
    gs = calculate_gs(ticker, as_of_date, restatement_df)
    if gs is None:
        return None
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return gs

    last4 = get_quarters_before_period(last_actual, 3) + [last_actual]
    prior4 = get_quarters_before_period(last4[0], 4)

    gp_df = get_metric_data(ticker, 'grossProfit', as_of_date, False, restatement_df)
    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)
    if gp_df.empty or rev_df.empty:
        return gs
    gp_series = pd.Series(gp_df['value'].values, index=gp_df['period'])
    rev_series = pd.Series(rev_df['value'].values, index=rev_df['period'])

    gp_last4, _  = strict_sum(gp_series, last4, min_q=3, annualize_to=4)
    gp_prior4, _ = strict_sum(gp_series, prior4, min_q=3, annualize_to=4)
    rev_last4, _  = strict_sum(rev_series, last4, min_q=3, annualize_to=4)
    rev_prior4, _ = strict_sum(rev_series, prior4, min_q=3, annualize_to=4)

    if any(v is None or v == 0 for v in [gp_last4, gp_prior4, rev_last4, rev_prior4]):
        return gs
    # growth FACTORS (1 + symmetric rate, floored at 0): robust to near-zero/negative bases
    gp_growth = 1.0 + sym_growth(gp_last4, gp_prior4)
    rev_growth = 1.0 + sym_growth(rev_last4, rev_prior4)
    if rev_growth == 0:
        return gs
    return round(gs * (gp_growth / rev_growth), 2)


# ============================================================================= #
# SIZE / HSG / SGD / r2 / VOLATILITY  (logic unchanged from v1; restatement_df threaded)
# ============================================================================= #
def _earliest_ever_shares(ticker, restatement_df=None):
    """
    Earliest dilutedAverageShares value across the FULL vintage history (any
    download_date), ignoring the as-of cutoff. Used only as a backfill when no
    point-in-time share count exists for an early calc date. Share count is a
    slow-moving scaling denominator with no predictive content, so anchoring a
    missing early value to the earliest-known count is a bounded, signal-free
    approximation -- "less bad" than dropping Size entirely.
    """
    db_ticker = normalize_ticker(ticker)
    table = _get_table_for_metric('dilutedAverageShares')
    try:
        raw = pd.read_sql_query(
            f"SELECT period, value FROM {table} "
            f"WHERE ticker = %(t)s AND metric_name = 'dilutedAverageShares'",
            ENGINE, params={'t': db_ticker})
    except Exception:
        return None
    if raw.empty:
        return None
    raw['value'] = pd.to_numeric(raw['value'], errors='coerce')
    raw = raw.dropna(subset=['value']).sort_values('period')
    if raw.empty:
        return None
    return raw.iloc[0]['value']


def calculate_size(ticker, as_of_date, Pxs_df, restatement_df=None):
    """
    Size = dilutedAverageShares * Price / 1e6

    Price is point-in-time from Pxs_df (guaranteed present). If the share count is
    unavailable as of an early calc date, fall back to the earliest-ever known share
    count (signal-free scaling term) rather than returning NaN -- see
    _earliest_ever_shares. The fallback never overrides a real point-in-time value.
    """
    if as_of_date not in Pxs_df.index or ticker not in Pxs_df.columns:
        return np.nan
    price = Pxs_df.loc[as_of_date, ticker]
    if pd.isna(price):
        return np.nan

    shares = np.nan
    shares_df = get_metric_data(ticker, 'dilutedAverageShares', as_of_date, False, restatement_df)
    if not shares_df.empty:
        shares = shares_df.iloc[-1]['value']

    if pd.isna(shares):
        shares = _earliest_ever_shares(ticker, restatement_df)
        if shares is not None and pd.notna(shares):
            logger.debug(f"[Size] {ticker} {pd.Timestamp(as_of_date).date()}: "
                         f"no point-in-time share count, using earliest-ever backfill")
        else:
            return np.nan

    return (shares * price) / 1_000_000


def calculate_hsg(ticker, as_of_date, restatement_df=None):
    """HSG = ewm_mean(YoY growth) / ewm_std(YoY growth). Drops missing YoY obs."""
    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)
    if len(rev_df) < 5:
        return None
    rev_series = pd.Series(rev_df['value'].values, index=rev_df['period'])
    periods = sorted(rev_series.index)
    yoy = []
    for i in range(4, len(periods)):
        cur, past = rev_series[periods[i]], rev_series[periods[i-4]]
        if pd.notna(cur) and pd.notna(past) and past != 0:
            yoy.append(cur / past - 1)
    if len(yoy) < 2:
        return None
    ys = pd.Series(yoy)
    mean = ys.ewm(halflife=4).mean().iloc[-1]
    std  = ys.ewm(halflife=4).std().iloc[-1]
    if pd.isna(std) or std == 0:
        return None
    return mean / std


def _yoy_series_with_gaps(ticker, as_of_date, restatement_df):
    """YoY growth list preserving gaps as NaN (for SGD). Returns (list, periods)."""
    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)
    if len(rev_df) < 8:
        return None, None
    rev_series = pd.Series(rev_df['value'].values, index=rev_df['period'])
    periods = sorted(rev_series.index)
    if len(periods) < 8:
        return None, None
    yoy = []
    for i in range(4, len(periods)):
        cur, past = rev_series[periods[i]], rev_series[periods[i-4]]
        if pd.notna(cur) and pd.notna(past) and past != 0:
            yoy.append(cur / past - 1)
        else:
            yoy.append(np.nan)
    return yoy, periods


def calculate_sgd_with_r2(ticker, as_of_date, restatement_df=None):
    """SGD = weighted linear regression on ewm-smoothed growth deltas."""
    yoy, _ = _yoy_series_with_gaps(ticker, as_of_date, restatement_df)
    if yoy is None or len(yoy) < 5:
        return None, None
    smoothed = pd.Series(yoy).ewm(halflife=4).mean()
    if smoothed.iloc[-5:].isna().any():
        return None, None
    g = smoothed.iloc[-5:].values
    deltas = [g[1]-g[0], g[2]-g[1], g[3]-g[2], g[4]-g[3]]
    weights = np.array([1, 2, 3, 4])
    X = np.array([0, 1, 2, 3]).reshape(-1, 1)
    y = np.array(deltas)
    model = LinearRegression().fit(X, y, sample_weight=weights)
    sgd = round(model.predict([[4]])[0] * 100, 5)
    r2 = model.score(X, y, sample_weight=weights)
    return sgd, r2


def calculate_sgd(ticker, as_of_date, restatement_df=None):
    return calculate_sgd_with_r2(ticker, as_of_date, restatement_df)[0]


def calculate_last_sgd(ticker, as_of_date, restatement_df=None):
    """LastSGD = last delta of ewm-smoothed YoY growth."""
    yoy, _ = _yoy_series_with_gaps(ticker, as_of_date, restatement_df)
    if yoy is None or len(yoy) < 2:
        return None
    smoothed = pd.Series(yoy).ewm(halflife=4).mean()
    if smoothed.iloc[-2:].isna().any():
        return None
    return round((smoothed.iloc[-1] - smoothed.iloc[-2]) * 100, 5)


def calculate_r2(ticker, as_of_date, metric_name, use_estimates=True, restatement_df=None):
    """R^2 of a linear fit to raw values (S/E: 8 actual + 8 est; GP: last 16 actual)."""
    actual_df = get_metric_data(ticker, metric_name, as_of_date, False, restatement_df)
    if use_estimates:
        last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
        if not last_actual:
            return None
        actual_series = (pd.Series(actual_df['value'].values, index=actual_df['period'])
                         if not actual_df.empty else pd.Series(dtype=float))
        last8 = sorted(actual_series.index)[-8:]
        next8 = get_quarters_after_period(last_actual, 8)
        if metric_name == 'normalizedNetIncome':
            est_filled = fill_normalized_ni(ticker, as_of_date, next8, restatement_df)
            est_points = [(q, est_filled[q]) for q in next8 if q in est_filled]
        else:
            est_df = get_metric_data(ticker, metric_name, as_of_date, True, restatement_df)
            est_series = (pd.Series(est_df['value'].values, index=est_df['period'])
                          if not est_df.empty else pd.Series(dtype=float))
            est_points = [(q, est_series[q]) for q in next8 if q in est_series.index]
        combined = [(q, actual_series[q]) for q in last8 if q in actual_series.index] + est_points
    else:
        if actual_df.empty:
            return None
        actual_series = pd.Series(actual_df['value'].values, index=actual_df['period'])
        last16 = sorted(actual_series.index)[-16:]
        combined = [(q, actual_series[q]) for q in last16]
    if len(combined) < 4:
        return None
    y = np.array([v for _, v in combined], dtype=float)
    X = np.arange(len(y)).reshape(-1, 1)
    mask = np.isfinite(y)
    y, X = y[mask], X[mask]
    if len(y) < 4:
        return None
    model = LinearRegression().fit(X, y)
    return round(model.score(X, y), 4)


def calculate_volatility_metrics(ticker, as_of_date, restatement_df=None):
    """S Vol, E Vol, GP Vol = ewm_std / |mean(last4)| * 100"""
    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)
    ni_df  = get_normalized_ni(ticker, as_of_date, False, restatement_df)
    gp_df  = get_metric_data(ticker, 'grossProfit', as_of_date, False, restatement_df)
    def calc_vol(df):
        if df.empty:
            return None
        series = pd.Series(df['value'].values, index=df['period'])
        ewm_std = series.ewm(halflife=4).std().iloc[-1]
        mean = abs(series.tail(4).mean())
        if mean == 0 or pd.isna(ewm_std):
            return None
        return round(ewm_std / mean * 100, 2)
    return calc_vol(rev_df), calc_vol(ni_df), calc_vol(gp_df)


# ============================================================================= #
# LTM / NTM RATIOS  (strict sums: >=3 of 4, annualised; GM zip re-keyed by quarter)
# ============================================================================= #
def calculate_ltm_ratios(ticker, as_of_date, size, restatement_df=None):
    """sP/S, sP/E, sP/GP over last 4 actual quarters (strict, annualised)."""
    if size is None:
        return None, None, None
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None, None, None
    quarters = get_quarters_before_period(last_actual, 3) + [last_actual]

    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)
    ni_df  = get_normalized_ni(ticker, as_of_date, False, restatement_df)
    gp_df  = get_metric_data(ticker, 'grossProfit', as_of_date, False, restatement_df)

    rev_s = pd.Series(rev_df['value'].values, index=rev_df['period']) if not rev_df.empty else pd.Series(dtype=float)
    ni_s  = pd.Series(ni_df['value'].values,  index=ni_df['period'])  if not ni_df.empty  else pd.Series(dtype=float)
    gp_s  = pd.Series(gp_df['value'].values,  index=gp_df['period'])  if not gp_df.empty  else pd.Series(dtype=float)

    rev_sum, _ = strict_sum(rev_s, quarters, min_q=3, annualize_to=4)
    ni_sum,  _ = strict_sum(ni_s,  quarters, min_q=3, annualize_to=4)
    gp_sum,  _ = strict_sum(gp_s,  quarters, min_q=3, annualize_to=4)

    def ratio(num, den):
        if den is None or den == 0:
            return None
        return round(num * 1_000_000 / den, 2)
    return ratio(size, rev_sum), ratio(size, ni_sum), ratio(size, gp_sum)


def calculate_ntm_ratios(ticker, as_of_date, size, restatement_df=None):
    """P/S, P/Ee, P/Eo, P/GP over next 4 estimated quarters (strict, annualised)."""
    if size is None:
        return None, None, None, None
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None, None, None, None

    first_est = get_quarters_after_period(last_actual, 1)[0]
    est_quarters = [first_est] + get_quarters_after_period(first_est, 3)

    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, True, restatement_df)
    rev_s  = pd.Series(rev_df['value'].values, index=rev_df['period']) if not rev_df.empty else pd.Series(dtype=float)
    rev_sum, _ = strict_sum(rev_s, est_quarters, min_q=4, annualize_to=4)  # forward: full set

    ni_filled = fill_normalized_ni(ticker, as_of_date, est_quarters, restatement_df)
    ni_s = pd.Series(ni_filled)
    ni_sum, _ = strict_sum(ni_s, est_quarters, min_q=4, annualize_to=4)

    actual_gp_df  = get_metric_data(ticker, 'grossProfit', as_of_date, False, restatement_df)
    actual_rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)

    # NTM gross profit via gross-margin regression; zip re-keyed by quarter
    gp_sum = None
    if not actual_gp_df.empty and not actual_rev_df.empty and rev_sum is not None:
        agp = pd.Series(actual_gp_df['value'].values, index=actual_gp_df['period'])
        arv = pd.Series(actual_rev_df['value'].values, index=actual_rev_df['period'])
        last8 = sorted(agp.index)[-8:]
        gm_values = [agp[q] / arv[q] for q in last8
                     if q in arv.index and arv[q] != 0 and pd.notna(agp[q]) and pd.notna(arv[q])]
        if len(gm_values) >= 4:
            X = np.arange(len(gm_values)).reshape(-1, 1)
            y = np.array(gm_values)
            gm_model = LinearRegression().fit(X, y)
            gm_r2 = gm_model.score(X, y)
            max_gm = max(gm_values)
            if gm_r2 >= 0.5:
                fc = gm_model.predict(np.arange(len(gm_values), len(gm_values) + 4).reshape(-1, 1))
                est_gms = [min(max_gm, g) for g in fc]
            else:
                est_gms = [float(np.mean(gm_values))] * 4
            # key the forecast margins to their quarters, then require completeness
            gm_by_q = dict(zip(est_quarters, est_gms))
            implied_gp = pd.Series({q: gm_by_q[q] * rev_s.get(q, np.nan)
                                    for q in est_quarters if q in gm_by_q})
            gp_sum, _ = strict_sum(implied_gp, est_quarters, min_q=4, annualize_to=4)

    # P/Eo: blended operating-income estimate (ratios of sums over same quarter set)
    actual_op_df  = get_metric_data(ticker, 'operatingIncome', as_of_date, False, restatement_df)
    est_op = None
    if not actual_op_df.empty and not actual_rev_df.empty and rev_sum is not None:
        actual_quarters = get_quarters_before_period(last_actual, 3) + [last_actual]
        op_s  = pd.Series(actual_op_df['value'].values, index=actual_op_df['period'])
        arv2  = pd.Series(actual_rev_df['value'].values, index=actual_rev_df['period'])
        # ratio of sums over the SAME surviving quarters -> annualisation cancels
        op_sum, n_op   = strict_sum(op_s,  actual_quarters, min_q=3, annualize_to=4)
        rev_a_sum, n_r = strict_sum(arv2,  actual_quarters, min_q=3, annualize_to=4)
        est_op_a = rev_sum * (op_sum / rev_a_sum) if (op_sum is not None and rev_a_sum not in (None, 0)) else None

        est_op_d = None
        if ni_sum not in (None, 0):
            actual_ni_df = get_normalized_ni(ticker, as_of_date, False, restatement_df)
            if not actual_ni_df.empty:
                ani = pd.Series(actual_ni_df['value'].values, index=actual_ni_df['period'])
                ani_sum, _ = strict_sum(ani, actual_quarters, min_q=3, annualize_to=4)
                if op_sum is not None and ani_sum not in (None, 0):
                    est_op_d = ni_sum * (op_sum / ani_sum)

        if est_op_a is not None and est_op_d is not None:
            est_op = (est_op_a + est_op_d) / 2
        else:
            est_op = est_op_a if est_op_a is not None else est_op_d

    def ratio(num, den):
        if not den or den == 0:
            return None
        return round(num * 1_000_000 / den, 2)
    return (ratio(size, rev_sum), ratio(size, ni_sum),
            ratio(size, est_op), ratio(size, gp_sum))


# ============================================================================= #
# ROI / ROE / OM  (ratio-averages: >=3 of 4 quarters, no annualisation)
# ============================================================================= #
def _ratio_avg_over_quarters(num_series, den_series, quarters, agg=np.mean, min_q=3):
    """Mean/median of per-quarter num/den over quarters with both present & den!=0."""
    vals = []
    for q in quarters:
        if q in num_series.index and q in den_series.index:
            n, d = num_series[q], den_series[q]
            if pd.notna(n) and pd.notna(d) and d != 0:
                vals.append(n / d)
    if len(vals) < min_q:
        return None
    return float(agg(vals))


def calculate_roi_metrics(ticker, as_of_date, restatement_df=None):
    """ROI-P / ROI / ROId. Denominator = costOfRevenues + totalOperatingExp
    (COGS and SG&A+S&M+R&D are disjoint in Ortex -> no double count)."""
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None, None, None
    cost_df = get_metric_data(ticker, 'costOfRevenues', as_of_date, False, restatement_df)
    opex_df = get_metric_data(ticker, 'totalOperatingExp', as_of_date, False, restatement_df)
    if cost_df.empty or opex_df.empty:
        return None, None, None
    income_df = get_metric_data(ticker, 'operatingIncome', as_of_date, False, restatement_df)
    if income_df.empty:
        income_df = get_normalized_ni(ticker, as_of_date, False, restatement_df)
    if income_df.empty:
        return None, None, None

    cost_s = pd.Series(cost_df['value'].values, index=cost_df['period'])
    opex_s = pd.Series(opex_df['value'].values, index=opex_df['period'])
    inc_s  = pd.Series(income_df['value'].values, index=income_df['period'])
    invested = cost_s.add(opex_s, fill_value=np.nan)  # both required per quarter

    roi_p_quarters = get_quarters_before_period(last_actual, 4)
    roi_p = _ratio_avg_over_quarters(inc_s, invested, roi_p_quarters, agg=np.mean, min_q=3)
    roi_p = round(roi_p * 100, 2) if roi_p is not None else None

    roi = None
    if (last_actual in inc_s.index and last_actual in cost_s.index and last_actual in opex_s.index):
        inv = cost_s[last_actual] + opex_s[last_actual]
        if pd.notna(inv) and inv != 0 and pd.notna(inc_s[last_actual]):
            roi = round(inc_s[last_actual] / inv * 100, 2)

    roid = round(roi - roi_p, 2) if roi is not None and roi_p is not None else None
    return roi_p, roi, roid


def calculate_roe_metrics(ticker, as_of_date, size, restatement_df=None):
    """ROE-P / ROE / ROEd (income / size)."""
    if size is None:
        return None, None, None
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None, None, None
    income_df = get_metric_data(ticker, 'operatingIncome', as_of_date, False, restatement_df)
    if income_df.empty:
        income_df = get_normalized_ni(ticker, as_of_date, False, restatement_df)
    if income_df.empty:
        return None, None, None
    inc_s = pd.Series(income_df['value'].values, index=income_df['period'])

    roe_p_quarters = get_quarters_before_period(last_actual, 4)
    vals = [inc_s[q] for q in roe_p_quarters if q in inc_s.index and pd.notna(inc_s[q])]
    roe_p = round(float(np.mean(vals)) / size / 1_000_000 * 100, 2) if len(vals) >= 3 else None

    roe = None
    if last_actual in inc_s.index and pd.notna(inc_s[last_actual]):
        roe = round(inc_s[last_actual] / size / 1_000_000 * 100, 2)

    roed = round(roe - roe_p, 2) if roe is not None and roe_p is not None else None
    return roe_p, roe, roed


def calculate_om_metrics(ticker, as_of_date, restatement_df=None):
    """OM-t0 / OM / OMd (operatingIncome / totalRevenues)."""
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None, None, None
    op_df  = get_metric_data(ticker, 'operatingIncome', as_of_date, False, restatement_df)
    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)
    if op_df.empty or rev_df.empty:
        return None, None, None
    op_s  = pd.Series(op_df['value'].values, index=op_df['period'])
    rev_s = pd.Series(rev_df['value'].values, index=rev_df['period'])

    om_t0_quarters = get_quarters_before_period(last_actual, 4)
    om_t0 = _ratio_avg_over_quarters(op_s, rev_s, om_t0_quarters, agg=np.median, min_q=3)
    om_t0 = round(om_t0 * 100, 2) if om_t0 is not None else None

    om = None
    if last_actual in op_s.index and last_actual in rev_s.index and rev_s[last_actual] != 0:
        om = round(op_s[last_actual] / rev_s[last_actual] * 100, 2)

    omd = round(om - om_t0, 2) if om is not None and om_t0 is not None else None
    return om_t0, om, omd


def calculate_rnd(ticker, as_of_date, restatement_df=None):
    """r&d = r&d / totalRevenues (last actual quarter)."""
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None
    rnd_df = get_metric_data(ticker, 'r&d', as_of_date, False, restatement_df)
    rev_df = get_metric_data(ticker, 'totalRevenues', as_of_date, False, restatement_df)
    if rnd_df.empty or rev_df.empty:
        return None
    rnd_s = pd.Series(rnd_df['value'].values, index=rnd_df['period'])
    rev_s = pd.Series(rev_df['value'].values, index=rev_df['period'])
    if last_actual not in rnd_s.index or last_actual not in rev_s.index:
        return None
    rv, rd = rev_s[last_actual], rnd_s[last_actual]
    if pd.isna(rd) or pd.isna(rv) or rv == 0:
        return None
    return round(rd / rv * 100, 2)


def calculate_pig_psg_isgd(ticker, as_of_date, restatement_df=None):
    """PIG (median income YoY), PSG (combined-spending YoY, pair-complete), ISGD=PIG-PSG."""
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return None, None, None
    last4 = [last_actual] + get_quarters_before_period(last_actual, 3)
    prior4 = [get_quarters_before_period(q, 4)[0] for q in last4]

    income_metrics = ['totalRevenues', 'normalizedNetIncome', 'operatingIncome', 'ebitda']
    income_ewms = []
    for metric in income_metrics:
        mdf = get_metric_data(ticker, metric, as_of_date, False, restatement_df)
        if mdf.empty:
            continue
        ms = pd.Series(mdf['value'].values, index=mdf['period'])
        growths = []
        for cq, pq in zip(last4, prior4):
            if cq in ms.index and pq in ms.index:
                growths.append(sym_growth(ms[cq], ms[pq]))   # b=current, a=prior
            else:
                growths.append(np.nan)
        if any(pd.notna(g) for g in growths):
            ewm_v = pd.Series(growths).ewm(halflife=2).mean().iloc[-1]
            if pd.notna(ewm_v):
                income_ewms.append(ewm_v)

    if len(income_ewms) >= 3:
        pig = np.median(income_ewms)
    elif len(income_ewms) == 2:
        pig = np.mean(income_ewms)
    elif len(income_ewms) == 1:
        pig = income_ewms[0]
    else:
        pig = np.nan

    # spending: combined cost+opex, pair-completeness (both components, both quarters)
    cost_df = get_metric_data(ticker, 'costOfRevenues', as_of_date, False, restatement_df)
    opex_df = get_metric_data(ticker, 'totalOperatingExp', as_of_date, False, restatement_df)
    psg = np.nan
    if not cost_df.empty and not opex_df.empty:
        cost_s = pd.Series(cost_df['value'].values, index=cost_df['period'])
        opex_s = pd.Series(opex_df['value'].values, index=opex_df['period'])

        def combined(q):
            if q in cost_s.index and q in opex_s.index:
                c, o = cost_s[q], opex_s[q]
                if pd.notna(c) and pd.notna(o):
                    return c + o
            return np.nan

        spending_growths = []
        for cq, pq in zip(last4, prior4):
            cur, prior = combined(cq), combined(pq)
            spending_growths.append(sym_growth(cur, prior))   # (current, prior)
        valid = [g for g in spending_growths if pd.notna(g)]
        if valid:
            psg = np.median(valid)

    isgd = (pig - psg) if (pd.notna(pig) and pd.notna(psg)) else np.nan
    pig = round(pig * 100, 2) if pd.notna(pig) else np.nan
    psg = round(psg * 100, 2) if pd.notna(psg) else np.nan
    isgd = round(isgd * 100, 2) if pd.notna(isgd) else np.nan
    return pig, psg, isgd


# ============================================================================= #
# FCF SUBSYSTEM
# ============================================================================= #
# Two sources, unified into a materialised table fcf_consolidated
#   (ticker, period, download_date, fcf):
#   - Ortex cash_data (post-cutoff): FCF = CFO + capex  (capex stored NEGATIVE)
#   - legacy wide tables {ticker}/{ticker}_fcf (pre-cutoff): column labels are
#     NOT point-in-time (Q0 was tied to fetch-time, re-centred every pull), so
#     mapping is by VALUE MATCHING walking back from the calendar-anchored latest
#     row. Exact run-match -> tolerant error-minimising alignment -> calendar last.
# ============================================================================= #

def fcf_from_ortex_vintage(cash_rows_df):
    """
    From tidy cash_data rows (period, download_date, value, metric_name) for ONE
    ticker, build per-(period, download_date) FCF = CFO + capex (capex negative).
    Missing capex -> 0 (logged as a warning when CFO is large). Returns tidy
    DataFrame [period, download_date, fcf].
    """
    if cash_rows_df.empty:
        return pd.DataFrame(columns=['period', 'download_date', 'fcf'])
    df = cash_rows_df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['download_date'] = pd.to_datetime(df['download_date'])
    wide = df.pivot_table(index=['period', 'download_date'], columns='metric_name',
                          values='value', aggfunc='first').reset_index()
    if 'cashFromOperations' not in wide.columns:
        return pd.DataFrame(columns=['period', 'download_date', 'fcf'])
    if 'capitalExpenditure' not in wide.columns:
        wide['capitalExpenditure'] = np.nan
    # capex stored negative -> ADD it; missing -> 0
    capex = wide['capitalExpenditure'].fillna(0.0)
    big_cfo_missing = wide['capitalExpenditure'].isna() & (wide['cashFromOperations'].abs() > 1e8)
    if big_cfo_missing.any():
        logger.warning(f"FCF: {int(big_cfo_missing.sum())} period-vintages with missing capex "
                       f"but large CFO (treated as 0)")
    wide['fcf'] = wide['cashFromOperations'] + capex
    return wide[['period', 'download_date', 'fcf']].dropna(subset=['fcf'])


def _q_columns(df):
    """{col_name: int_index} for Q-pattern columns ('Q0'->0,'Q-5'->-5,'Q15'->15)."""
    out = {}
    for col in df.columns:
        c = str(col)
        if c.startswith('Q'):
            try:
                out[col] = int(c[1:])
            except ValueError:
                pass
    return out


def _calendar_index_of_date(dt):
    """Calendar quarter index of a date (for anchoring the latest legacy row)."""
    dt = pd.Timestamp(dt)
    q = (dt.month - 1) // 3 + 1
    return dt.year * 4 + (q - 1)


def map_legacy_row(row, qmap, ref_by_calidx, restated_idx):
    """
    Determine the integer offset mapping this row's Q-columns to calendar indices,
    by value-matching against ref_by_calidx {cal_idx -> rounded_value}.

    Returns (offset, confidence, method, suspect): calendar_idx(col) = q_idx + offset.

    Offset selection is by LONGEST CONSECUTIVE RUN of matching columns, not vote
    count. This is what defeats the duplicate-frontier displacement: a legacy row
    often repeats its last actual at both Q-1 and Q0 (Q0 a placeholder), which casts
    one spurious vote for an offset one-quarter off. The correct offset still yields
    the long consecutive run of real actuals, so run-length picks it unambiguously.
    """
    vals = {}
    for col, qi in qmap.items():
        v = row.get(col)
        if pd.notna(v):
            vals[qi] = round(v / REV_ROUND_UNIT)
    if not vals or not ref_by_calidx:
        return None, 0.0, None, False

    # candidate offsets implied by any value match (offset = cal_idx - q_idx)
    val_to_cals = {}
    for ci, rv in ref_by_calidx.items():
        val_to_cals.setdefault(rv, []).append(ci)
    candidate_offsets = set()
    for qi, rv in vals.items():
        for ci in val_to_cals.get(rv, []):
            candidate_offsets.add(ci - qi)
    if not candidate_offsets:
        return None, 0.0, None, False

    qs_sorted = sorted(vals.keys())

    def run_and_count(offset):
        """(longest consecutive matching run, total matches) under an offset."""
        matchset = set()
        for qi in qs_sorted:
            cal = qi + offset
            if cal in restated_idx:
                continue
            ref_rv = ref_by_calidx.get(cal)
            if ref_rv is not None and ref_rv == vals[qi]:
                matchset.add(qi)
        best = cur = 0
        for qi in qs_sorted:
            if qi in matchset:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        return best, len(matchset)

    scored = [(off,) + run_and_count(off) for off in candidate_offsets]
    # pick by longest run, then by total matches; both break the duplicate tie
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    best_off, best_run, best_matches = scored[0]

    if best_run >= FCF_MATCH_MIN_RUN:
        conf = best_matches / max(len(vals), 1)
        return best_off, conf, 'exact_run', False

    return None, 0.0, None, False


def build_legacy_fcf_for_ticker(ticker, ortex_fcf_ref=None, restatement_df=None):
    """
    Build calendar-mapped FCF rows from the legacy wide tables for ONE ticker.

    Mapping is anchored on the ORTEX FCF series (ground truth): the legacy _fcf
    values are identical to Ortex's, only column-displaced, so we find the offset
    that aligns each legacy row's FCF values to the Ortex reference, then chain
    backward with a forward-growing reference of already-resolved legacy values.

    `ortex_fcf_ref`: Series period -> fcf (latest Ortex vintage per period).
    Returns tidy DataFrame [period, download_date, fcf, method, suspect].
    """
    suffix = ticker.replace(' ', '').lower()
    fcf_table = suffix + '_fcf'
    empty = pd.DataFrame(columns=['period', 'download_date', 'fcf', 'method', 'suspect'])

    # existence check (VA DB) -- only the _fcf table is needed now
    try:
        names = pd.read_sql_query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'",
            VENGINE)
        have = set(names['table_name'].tolist())
    except Exception:
        have = set()
    if fcf_table not in have:
        return empty

    fcf_df = pd.read_sql_query(f'SELECT * FROM "{fcf_table}"', VENGINE)
    if fcf_df.empty or 'Date' not in fcf_df.columns:
        return empty
    fcf_df['Date'] = pd.to_datetime(fcf_df['Date'])
    fcf_df = fcf_df.sort_values('Date').reset_index(drop=True)
    fqmap = _q_columns(fcf_df)
    if not fqmap:
        return empty

    db_ticker = normalize_ticker(ticker)
    restated_idx = set()
    if restatement_df is not None and not restatement_df.empty:
        rsub = restatement_df[restatement_df['ticker'] == db_ticker]
        restated_idx = set(period_to_int(p) for p in rsub['period'].tolist())

    # reference {cal_idx -> rounded_value}: seed from Ortex FCF (ground truth).
    # Keep a value-presence set so we never let a repeated value (e.g. the Q0/Q-1
    # frontier duplicate) overwrite or shadow an existing period.
    ref_by_calidx = {}
    seen_values = set()
    if ortex_fcf_ref is not None and not ortex_fcf_ref.empty:
        for period, v in ortex_fcf_ref.items():
            if pd.notna(v):
                try:
                    cal = period_to_int(period)
                except Exception:
                    continue
                rv = round(v / REV_ROUND_UNIT)
                ref_by_calidx[cal] = rv
                seen_values.add(rv)

    if not ref_by_calidx:
        logger.warning(f"[FCF legacy] {db_ticker}: no Ortex reference -- legacy FCF skipped")
        return empty

    out_rows = []
    # walk rows newest -> oldest; each re-anchors against the growing reference,
    # which extends backward as older rows reveal earlier quarters.
    for ridx in range(len(fcf_df) - 1, -1, -1):
        row = fcf_df.iloc[ridx]
        rdate = pd.Timestamp(row['Date'])
        offset, conf, method, suspect = map_legacy_row(row, fqmap, ref_by_calidx, restated_idx)
        if offset is None:
            logger.info(f"[FCF map fallback] {db_ticker} row={rdate.date()} -> unmapped, skipped")
            continue

        rdate_cal = _calendar_index_of_date(rdate)
        # collect this row's (cal -> value); drop a top duplicate (Q0==Q-1 frontier)
        col_by_cal = {}
        for col, qi in fqmap.items():
            v = row.get(col)
            if pd.notna(v):
                col_by_cal[qi + offset] = float(v)
        if not col_by_cal:
            continue
        # if the highest populated cal duplicates the value one quarter below it,
        # it's the placeholder frontier copy -> drop it (prevents phantom actual)
        hi = max(col_by_cal)
        if (hi - 1) in col_by_cal and            round(col_by_cal[hi] / REV_ROUND_UNIT) == round(col_by_cal[hi - 1] / REV_ROUND_UNIT):
            col_by_cal.pop(hi)

        for cal, v in col_by_cal.items():
            if cal in restated_idx or cal > rdate_cal + 1:   # no future actuals
                continue
            rv = round(v / REV_ROUND_UNIT)
            # extend reference backward only with genuinely new, non-colliding values
            if cal not in ref_by_calidx and rv not in seen_values:
                ref_by_calidx[cal] = rv
                seen_values.add(rv)
            out_rows.append({'period': int_to_period(cal), 'download_date': rdate,
                             'fcf': v, 'method': method, 'suspect': bool(suspect)})

    return pd.DataFrame(out_rows, columns=['period', 'download_date', 'fcf', 'method', 'suspect'])


def fetch_cash_rows_for_ticker(db_ticker):
    """All Ortex cash_data FCF-input rows for a ticker (tidy)."""
    q = """
        SELECT period, download_date, metric_name, value
        FROM cash_data
        WHERE ticker = %(ticker)s
        AND metric_name IN ('cashFromOperations', 'capitalExpenditure')
    """
    return pd.read_sql_query(q, ENGINE, params={'ticker': db_ticker})


def build_fcf_table(tickers, target_table, restatement_df=None):
    """
    Materialise FCF for `tickers` into `target_table` with schema
    (ticker, period, download_date, fcf). Replaces target_table wholesale.

    FCF is Ortex-derived only. The legacy {ticker}_fcf wide tables carry the same
    quarterly FCF values column-displaced with no point-in-time meaning; the
    value-keyed re-anchoring mapper could not be made reliable on live data
    (systematic multi-quarter displacement), so legacy splicing is disabled.
    Ortex FCF is clean back to ~2018, which covers all current and recent
    backtest dates; pre-2018 FCF_PG is NaN, which is harmless since nothing
    downstream consumes it. The legacy mapper (build_legacy_fcf_for_ticker /
    map_legacy_row) is retained dormant below; flip USE_LEGACY_FCF to re-enable.
    """
    all_rows = []
    splice_warnings = 0
    for ticker in tickers:
        db_ticker = normalize_ticker(ticker)

        cash_rows = fetch_cash_rows_for_ticker(db_ticker)
        ortex = fcf_from_ortex_vintage(cash_rows)
        ortex = ortex.assign(ticker=db_ticker) if not ortex.empty else ortex

        if USE_LEGACY_FCF:
            if not ortex.empty:
                ortex_ref = (ortex.sort_values('download_date')
                             .groupby('period')['fcf'].last())
            else:
                ortex_ref = pd.Series(dtype=float)

            legacy = build_legacy_fcf_for_ticker(ticker, ortex_ref, restatement_df)
            legacy = (legacy[['period', 'download_date', 'fcf']].assign(ticker=db_ticker)
                      if not legacy.empty else legacy)

            if not ortex.empty and not legacy.empty:
                o_latest = (ortex.sort_values('download_date')
                            .groupby('period')['fcf'].last())
                l_latest = (legacy.sort_values('download_date')
                            .groupby('period')['fcf'].last())
                common = o_latest.index.intersection(l_latest.index)
                for p in common:
                    a, b = o_latest[p], l_latest[p]
                    denom = max(abs(a), abs(b), 1.0)
                    if np.sign(a) != np.sign(b) or abs(a - b) / denom > 0.25:
                        splice_warnings += 1
                        logger.warning(f"[FCF splice] {db_ticker} {p}: ortex={a:,.0f} "
                                       f"legacy={b:,.0f} -- sign/scale mismatch")
            parts = (ortex, legacy)
        else:
            parts = (ortex,)

        for part in parts:
            if not part.empty:
                all_rows.append(part[['ticker', 'period', 'download_date', 'fcf']])

    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['ticker', 'period', 'download_date', 'fcf'])

    with ENGINE.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {target_table}"))
        result.to_sql(target_table, conn, if_exists='replace', index=False)
    logger.info(f"[FCF] built {target_table}: {len(result)} rows, "
                f"{result['ticker'].nunique() if not result.empty else 0} tickers, "
                f"Ortex-only" + (f", {splice_warnings} splice warnings" if USE_LEGACY_FCF else ""))
    return result


def get_fcf_series(ticker, as_of_date, fcf_table):
    """
    Read FCF as of a date from the materialised table (vintage-aware, pandas pick).
    Returns Series period -> fcf.
    """
    db_ticker = normalize_ticker(ticker)
    as_of_ts = pd.Timestamp(as_of_date)
    try:
        df = pd.read_sql_query(
            f"SELECT period, download_date, fcf FROM {fcf_table} WHERE ticker = %(t)s",
            ENGINE, params={'t': db_ticker})
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    df['download_date'] = pd.to_datetime(df['download_date'])
    df['fcf'] = pd.to_numeric(df['fcf'], errors='coerce')
    df = df[df['download_date'] <= as_of_ts]
    if df.empty:
        return pd.Series(dtype=float)
    df['is_null'] = df['fcf'].isna()
    df = df.sort_values(['period', 'is_null', 'download_date'], ascending=[True, True, False])
    latest = df.groupby('period', as_index=False).first()
    return pd.Series(latest['fcf'].values, index=latest['period'])


def calculate_fcf_pg(ticker, as_of_date, fcf_table=FCF_TABLE, restatement_df=None):
    """FCF past growth: 4 YoY midpoint growths, EWM(halflife=2), last value * 100."""
    last_actual = get_last_actual_period(ticker, as_of_date, restatement_df)
    if not last_actual:
        return np.nan
    fcf_series = get_fcf_series(ticker, as_of_date, fcf_table)
    if fcf_series.empty:
        return np.nan

    last4 = [last_actual] + get_quarters_before_period(last_actual, 3)
    prior4 = [get_quarters_before_period(q, 4)[0] for q in last4]
    growths = []
    for cq, pq in zip(last4, prior4):
        if cq in fcf_series.index and pq in fcf_series.index:
            growths.append(sym_growth(fcf_series[cq], fcf_series[pq]))   # b=current, a=prior
        else:
            growths.append(np.nan)
    if not any(pd.notna(g) for g in growths):
        return np.nan
    ewm_v = pd.Series(growths).ewm(halflife=2).mean().iloc[-1]
    if pd.isna(ewm_v):
        return np.nan
    return round(ewm_v * 100, 2)


# ============================================================================= #
# ORCHESTRATION
# ============================================================================= #
def calculate_all_metrics_for_stock(ticker, as_of_date, Pxs_df,
                                    fcf_table=FCF_TABLE, restatement_df=None):
    """All metrics for one stock. Failures are reported in one of two forms:
       FEP for stock XXX failed - date YYYY/MM/DD   (the per-stock revenue anchor)
       metric Y failed for stock XXX on date YYYY/MM/DD  (an individual metric)
    """
    results = {'ticker': ticker, 'date': as_of_date}
    dstr = pd.Timestamp(as_of_date).strftime('%Y/%m/%d')

    # FEP is the shared anchor for every metric (incl. FCF); resolve it once up front.
    # If it fails, every downstream metric would fail for the same reason -- so report
    # the FEP failure alone and return an all-null row (no metric-error cascade).
    try:
        get_first_estimated_period(ticker, as_of_date, restatement_df)
    except Exception as e:
        logger.warning(f"FEP for stock {ticker} failed - date {dstr}  "
                       f"({type(e).__name__}: {e})")
        return results          # ticker/date only; all metrics absent (null on save)

    def safe(key, fn, *args, **kwargs):
        try:
            results[key] = fn(*args, **kwargs)
        except Exception as e:
            results[key] = None
            logger.warning(f"metric {key} failed for stock {ticker} on date {dstr}  "
                           f"({type(e).__name__}: {e})")

    safe('Size', calculate_size, ticker, as_of_date, Pxs_df, restatement_df)
    size = results['Size']

    safe('HSG', calculate_hsg, ticker, as_of_date, restatement_df)
    safe('SGD', calculate_sgd, ticker, as_of_date, restatement_df)
    safe('LastSGD', calculate_last_sgd, ticker, as_of_date, restatement_df)
    safe('GS', calculate_gs, ticker, as_of_date, restatement_df)
    safe('GE', calculate_ge, ticker, as_of_date, restatement_df)
    safe('GGP', calculate_ggp, ticker, as_of_date, restatement_df)
    safe('r2 S', calculate_r2, ticker, as_of_date, 'totalRevenues', True, restatement_df)
    safe('r2 E', calculate_r2, ticker, as_of_date, 'normalizedNetIncome', True, restatement_df)
    safe('r2 GP', calculate_r2, ticker, as_of_date, 'grossProfit', False, restatement_df)

    def safe_tuple(keys, fn, *args):
        try:
            vals = fn(*args)
        except Exception as e:
            vals = (None,) * len(keys)
            logger.warning(f"metric {'/'.join(keys)} failed for stock {ticker} "
                           f"on date {dstr}  ({type(e).__name__}: {e})")
        for k, v in zip(keys, vals):
            results[k] = v

    safe_tuple(['sP/S', 'sP/E', 'sP/GP'], calculate_ltm_ratios, ticker, as_of_date, size, restatement_df)
    safe_tuple(['P/S', 'P/Ee', 'P/Eo', 'P/GP'], calculate_ntm_ratios, ticker, as_of_date, size, restatement_df)
    safe_tuple(['OM-t0', 'OM', 'OMd'], calculate_om_metrics, ticker, as_of_date, restatement_df)
    safe_tuple(['ROI-P', 'ROI', 'ROId'], calculate_roi_metrics, ticker, as_of_date, restatement_df)
    safe_tuple(['ROE-P', 'ROE', 'ROEd'], calculate_roe_metrics, ticker, as_of_date, size, restatement_df)
    safe_tuple(['S Vol', 'E Vol', 'GP Vol'], calculate_volatility_metrics, ticker, as_of_date, restatement_df)
    safe('r&d', calculate_rnd, ticker, as_of_date, restatement_df)
    safe_tuple(['PIG', 'PSG', 'ISGD'], calculate_pig_psg_isgd, ticker, as_of_date, restatement_df)
    safe('FCF_PG', calculate_fcf_pg, ticker, as_of_date, fcf_table, restatement_df)

    return results


# ============================================================================= #
# PER-DATE UNIVERSE (eligibility anchored to each calc date, not today)
# ============================================================================= #
def build_eligibility(Pxs_df, ed_df, vayf_df, exc_l, stale_window=5):
    """
    Boolean DataFrame (dates x tickers): True = in universe that day.
    Criteria: US listing, ed/vayf coverage, not excluded, price present that day,
    >=1 nonzero move over the trailing `stale_window` pct-changes (anti-stale;
    short-history names pass via NaN, matching live behaviour).
    """
    base = [d for d in ed_df.columns
            if d in vayf_df.index and d in Pxs_df.columns
            and str(d).split(' ')[-1] == 'US' and d not in exc_l]
    if not base:
        return pd.DataFrame(index=Pxs_df.index)
    px = Pxs_df[base]
    active = (px.pct_change().rolling(stale_window).sum() != 0)
    return active & px.notna()


def universe_for(eligibility, calc_date):
    """Tickers eligible as of calc_date (latest eligibility row <= calc_date)."""
    sub = eligibility.loc[:calc_date]
    if sub.empty:
        return []
    row = sub.iloc[-1]
    return row[row].index.tolist()


# ============================================================================= #
# SAVE (transactional per date; full-row replace -- single vintage per row)
# ============================================================================= #
def _sanitise(col):
    return col.replace(' ', '_').replace('/', '_').replace('&', '_').replace('-', '_')


def save_metrics(metrics_df, tables, replace_date=True):
    """
    Write one calc date's metrics transactionally. replace_date=True deletes the
    date's existing rows first so each row reflects exactly one calculation
    vintage (no partial-upsert blending). All-or-nothing per (table, date).
    """
    if metrics_df.empty:
        return
    calc_date = metrics_df['date'].iloc[0]
    metric_cols = [c for c in metrics_df.columns if c not in ['date', 'ticker']]

    for table in tables:
        with ENGINE.begin() as conn:
            if replace_date:
                conn.execute(text(f"DELETE FROM {table} WHERE date = :d"),
                             {'d': pd.Timestamp(calc_date)})
            for _, row in metrics_df.iterrows():
                present = [c for c in metric_cols if pd.notna(row[c])]
                all_cols = ['date', 'ticker'] + present
                cols_str = ', '.join(f'"{c}"' for c in all_cols)
                ph_str   = ', '.join(f':{_sanitise(c)}' for c in all_cols)
                upd_str  = ', '.join(f'"{c}" = :{_sanitise(c)}' for c in present)
                conn.execute(text(f"""
                    INSERT INTO {table} ({cols_str}) VALUES ({ph_str})
                    ON CONFLICT (date, ticker) DO UPDATE SET {upd_str}
                """) if present else text(f"""
                    INSERT INTO {table} (date, ticker) VALUES (:date, :ticker)
                    ON CONFLICT (date, ticker) DO NOTHING
                """), {_sanitise(c): row[c] for c in all_cols})
    logger.info(f"  saved {len(metrics_df)} rows for {pd.Timestamp(calc_date).date()} -> {', '.join(tables)}")


# ============================================================================= #
# VALIDATE-MODE REPORTING (diff vs old table + coverage)
# ============================================================================= #
def coverage_report(metrics_df):
    print("\nCoverage:")
    n = len(metrics_df)
    for col in metrics_df.columns:
        if col in ('ticker', 'date'):
            continue
        nn = metrics_df[col].notna().sum()
        print(f"  {col:8s}: {nn}/{n} ({nn/n*100:4.1f}%)")


def metrics_to_stock_dict(long_df):
    """
    Reshape a long metrics frame (rows = stock x date) into a stock-driven dict:
        { ticker: DataFrame indexed by calc date, columns = the full metric set }
    Sorted by date within each stock. This is the validate-mode return value --
    grab result['NVDA US'] to get that name's metric time series directly.
    """
    out = {}
    if long_df is None or long_df.empty:
        return out
    df = long_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    metric_cols = [c for c in df.columns if c not in ('ticker', 'date')]
    for ticker, g in df.groupby('ticker'):
        out[ticker] = g.set_index('date')[metric_cols].sort_index()
    return out


def dump_validation_metrics(metrics_df, path=None):
    """
    Print and save the FULL metrics table for every stock in a validate run, so
    values can be eyeballed for residual issues. Returns the saved path.
    """
    df = metrics_df.copy()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           'display.width', None, 'display.float_format',
                           lambda v: f"{v:,.4f}" if isinstance(v, float) else str(v)):
        print("\nFull validation metrics (all stocks):")
        print(df.to_string(index=False))
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, f"validation_metrics_{stamp}.csv")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\n[validate] full metrics written to {path}")
    except Exception as e:
        print(f"\n[validate] could not write csv: {e}")
    return path


def diff_vs_old(metrics_df, old_table=None):
    """Compare new metrics against the surviving prior table where they overlap."""
    if old_table is None:
        old_table = DATE_SOURCE_TABLE
    try:
        dates = tuple(pd.Timestamp(d) for d in metrics_df['date'].unique())
        tickers = tuple(metrics_df['ticker'].unique())
        old = pd.read_sql_query(
            f"SELECT * FROM {old_table} WHERE date IN %(d)s AND ticker IN %(t)s",
            ENGINE, params={'d': dates, 't': tickers})
    except Exception as e:
        print(f"\n(diff skipped: {e})")
        return
    if old.empty:
        print("\n(diff: no overlapping rows in old table)")
        return
    print("\nDiff vs old table (median |relative change| per metric, overlap only):")
    new = metrics_df.copy()
    new['date'] = pd.to_datetime(new['date'])
    old['date'] = pd.to_datetime(old['date'])
    merged = new.merge(old, on=['date', 'ticker'], suffixes=('_new', '_old'))
    for col in [c for c in metrics_df.columns if c not in ('ticker', 'date')]:
        cn, co = f"{col}_new", f"{col}_old"
        if cn in merged and co in merged:
            a = pd.to_numeric(merged[cn], errors='coerce')
            b = pd.to_numeric(merged[co], errors='coerce')
            mask = a.notna() & b.notna()
            if mask.sum() == 0:
                continue
            rel = (a[mask] - b[mask]).abs() / b[mask].abs().clip(lower=1e-9)
            newly_null = (a.isna() & b.notna()).sum()
            print(f"  {col:8s}: med|Δ|={rel.median()*100:6.1f}%  n={int(mask.sum()):4d}  "
                  f"newly_null={int(newly_null)}")


# ============================================================================= #
# MAIN  (validate / rebuild / incremental)
# ============================================================================= #
def thin_dates(dates, min_lag=MINIMUM_LAG):
    """
    Drop calc dates that fall within `min_lag` calendar days of the last kept date.
    Walks ascending, always keeps the first, and keeps a date only if it is at least
    min_lag days after the previously kept one. Returns the thinned, sorted list.
    """
    if not dates:
        return dates
    ds = sorted(pd.Timestamp(d) for d in dates)
    kept = [ds[0]]
    for d in ds[1:]:
        if (d - kept[-1]).days >= min_lag:
            kept.append(d)
    return kept


def _compute_dates(Pxs_df, eligibility, dates, tables, fcf_table,
                   restatement_df, save=True, replace_date=True):
    """Compute + save all stocks for each date in `dates`."""
    for calc_date in dates:
        calc_date = pd.Timestamp(calc_date)
        if calc_date not in Pxs_df.index:
            avail = Pxs_df.index[Pxs_df.index <= calc_date]
            if avail.empty:
                logger.warning(f"  no price data <= {calc_date.date()}, skipping")
                continue
            calc_date = avail[-1]
        tickers = universe_for(eligibility, calc_date)
        dstr = calc_date.strftime('%Y-%m-%d')
        rows = []
        for ticker in tickers:
            logger.info(f"{dstr} | {ticker}")
            try:
                rows.append(calculate_all_metrics_for_stock(
                    ticker, calc_date, Pxs_df, fcf_table, restatement_df))
            except Exception as e:
                # unexpected hard failure of the whole stock (not a single metric/FEP)
                logger.warning(f"stock {ticker} failed entirely on date "
                               f"{calc_date.strftime('%Y/%m/%d')}  ({type(e).__name__}: {e})")
        mdf = pd.DataFrame(rows)
        if save and not mdf.empty:
            save_metrics(mdf, tables, replace_date=replace_date)
        yield calc_date, mdf


def ensure_validate_tables():
    """Create the _validate side tables mirroring production schema, if absent."""
    with ENGINE.begin() as conn:
        for prod, val in zip(TABLES, VALIDATE_TABLES):
            conn.execute(text(
                f"CREATE TABLE IF NOT EXISTS {val} (LIKE {prod} INCLUDING ALL)"))


def run(Pxs_df, ed_df, vayf_df, exc_l, mode='validate', validate_tickers=None):
    """
    mode='validate'    -> DRY-RUN of rebuild: identical date set (inherited from the
                          prior valuation table) and identical per-date universe /
                          compute logic, but scoped to whatever stocks are in Pxs_df
                          (feed one name or a few) and NOTHING is written. Returns a
                          stock-driven dict { ticker: DataFrame(date-rows x metrics) }
                          -- a true snapshot of how the full rebuild output will look.
    mode='rebuild'     -> backup prod tables, rebuild FCF + regenerate inherited dates.
    mode='incremental' -> single new date = last Pxs_df date, skip if already present.

    Scope in validate/rebuild is controlled entirely by the Pxs_df passed in
    (per-date eligibility screen still applies). validate_tickers is ignored.
    """
    _check_credentials()
    _reset_caches()
    restatement_df = load_restatement_log()
    eligibility = build_eligibility(Pxs_df, ed_df, vayf_df, exc_l)

    if mode == 'validate':
        keep = [c for c in eligibility.columns]   # eligible names within the given Pxs_df

        # SAME date set as rebuild: the DISTINCT dates in the prior valuation table.
        # This makes validate a faithful dry-run -- the only differences from rebuild
        # are the (smaller) stock scope and that results are not persisted.
        try:
            existing = pd.read_sql_query(f"SELECT DISTINCT date FROM {DATE_SOURCE_TABLE}", ENGINE)
            dates = sorted(pd.to_datetime(existing['date']).tolist())
        except Exception:
            dates = []
        if not dates:
            logger.warning("[validate] no dates in prior table; falling back to last "
                           "Pxs_df date only (restore the baseline for a true dry-run)")
            dates = [Pxs_df.index[-1]]
        n_raw = len(dates)
        dates = thin_dates(dates)
        logger.info(f"[validate] DRY-RUN over {len(keep)} stock(s) x {len(dates)} dates "
                    f"({pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()}) "
                    f"-- {n_raw - len(dates)} dropped by MINIMUM_LAG={MINIMUM_LAG}d, no writes")

        # FCF: build to the transient validate FCF table (scratch, not prod) so the
        # metric loop can read vintage-aware FCF exactly as rebuild does.
        ensure_validate_tables()
        build_fcf_table(keep, FCF_TABLE_VALID, restatement_df)

        frames = []
        for calc_date, mdf in _compute_dates(
                Pxs_df, eligibility, dates, VALIDATE_TABLES, FCF_TABLE_VALID,
                restatement_df, save=False, replace_date=False):   # <- no writes
            if not mdf.empty:
                frames.append(mdf)

        long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not long_df.empty:
            coverage_report(long_df)
            dump_validation_metrics(long_df)            # full CSV snapshot (not a DB write)
            diff_vs_old(long_df, old_table=DATE_SOURCE_TABLE)

        by_stock = metrics_to_stock_dict(long_df)
        logger.info(f"[validate] returning {len(by_stock)} per-stock frame(s), "
                    f"{len(dates)} date-rows each")
        return by_stock

    if mode == 'rebuild':
        # NON-DESTRUCTIVE rebuild. Calc-date set = the DISTINCT dates already in the
        # prior table (anchors), thinned. We reproduce exactly those dates (NOT the
        # denser Pxs_df calendar) into a FRESH table REBUILD_TABLE. anchors and
        # consolidated are never touched, so the live pipeline keeps running on the
        # old data until you cut over downstream to REBUILD_TABLE.
        try:
            existing = pd.read_sql_query(f"SELECT DISTINCT date FROM {DATE_SOURCE_TABLE}", ENGINE)
            dates = sorted(pd.to_datetime(existing['date']).tolist())
        except Exception:
            dates = []
        if not dates:
            logger.warning("[rebuild] no existing dates found in prior table; "
                           "nothing to regenerate (restore the baseline first)")
            return {}
        n_raw = len(dates)
        dates = thin_dates(dates)
        logger.info(f"[rebuild] inheriting {len(dates)} calc dates from {DATE_SOURCE_TABLE} "
                    f"({pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()}) "
                    f"-- {n_raw - len(dates)} dropped by MINIMUM_LAG={MINIMUM_LAG}d")

        # Fresh target. It should NOT already exist; if it does (e.g. an aborted prior
        # run), drop and recreate for a clean slate -- safe, since it is not the live
        # table. Schema inherited from anchors.
        insp_exists = pd.read_sql_query(
            "SELECT to_regclass(%(t)s) AS t", ENGINE,
            params={'t': REBUILD_TABLE}).iloc[0]['t'] is not None
        with ENGINE.begin() as conn:
            if insp_exists:
                logger.warning(f"[rebuild] {REBUILD_TABLE} already exists "
                               f"(prior aborted run?) -- dropping and recreating")
                conn.execute(text(f"DROP TABLE {REBUILD_TABLE}"))
            conn.execute(text(
                f"CREATE TABLE {REBUILD_TABLE} (LIKE {DATE_SOURCE_TABLE} INCLUDING ALL)"))
        logger.info(f"[rebuild] writing to fresh {REBUILD_TABLE}; "
                    f"{DATE_SOURCE_TABLE} and valuation_consolidated left untouched; "
                    f"regenerating {len(dates)} dates")

        all_tickers = [c for c in eligibility.columns]
        build_fcf_table(all_tickers, FCF_TABLE, restatement_df)
        out = {}
        for calc_date, mdf in _compute_dates(
                Pxs_df, eligibility, dates, [REBUILD_TABLE], FCF_TABLE,
                restatement_df, save=True, replace_date=True):
            out[calc_date] = mdf
        logger.info(f"[rebuild] done -- {len(out)} dates written to {REBUILD_TABLE}. "
                    f"Diff against {DATE_SOURCE_TABLE} to verify, then cut over downstream.")
        return out

    if mode == 'incremental':
        new_date = pd.Timestamp(Pxs_df.index[-1])
        try:
            existing = pd.read_sql_query(f"SELECT DISTINCT date FROM {DATE_SOURCE_TABLE}", ENGINE)
            present = set(pd.to_datetime(existing['date']).tolist())
        except Exception:
            present = set()
        if new_date in present:
            logger.info(f"[incremental] {new_date.date()} already present -- skipping")
            return {}
        all_tickers = [c for c in eligibility.columns]
        build_fcf_table(all_tickers, FCF_TABLE, restatement_df)   # refresh FCF (idempotent)
        out = {}
        for calc_date, mdf in _compute_dates(
                Pxs_df, eligibility, [new_date], TABLES, FCF_TABLE,
                restatement_df, save=True, replace_date=True):
            out[calc_date] = mdf
        return out

    raise ValueError(f"unknown mode: {mode}")


# Default excluded universe (Chinese ADRs + financials/REITs/insurers -- COGS/opex
# undefined for these business models). STATIC, applied uniformly across history.
DEFAULT_EXC_L = [
    'MOMO US','BILI US','TME US','BABA US','JD US','BIDU US','PDD US','NIO US','XPEV US','ZTO US','ZLAB US',
    'TCEHY US','TSM US','OCFT US','VNET US','BGNE US','YY US','TAL US','EDU US','GOTU US','NTES US',
    'PAGS US','STNE US','WB US','MPNGF US','HZNP US','COUP US','ZEN US','LAW US','PLUG US','WKME US',
    'COLD US','BRK/A US','CBOE US','ALL US','ICE US','PGR US','MET US','KNSL US','SPLK US',
    'ESS US','PSA US','AJG US','AFL US','AIG US','GDS US','AVB US','COF US','CI US','CINF US',
    'DLR US','FRT US','FRC US','HIG US','IRM US','KIM US','LNC US','PFG US','O US','UNM US',
    'VTR US','WELL US','XP US','RDFN US','ESMT US']


if __name__ == "__main__":
    # Kernel-resident dependencies: ed_relation / prices_relation / va_yf live in
    # the legacy VA DB; loaded here via fetch-only SQL helpers.
    _check_credentials()
    ed_df = _dd_index(_set_df(pd.read_sql_query("SELECT * FROM ed_relation", VENGINE)))
    Pxs_df = open_va_df('prices_relation')
    Pxs_df.index = Pxs_df.index.map(lambda x: datetime(x.year, x.month, x.day))
    vayf_df = open_va_df('va_yf')

    print("=" * 80)
    print("VALUATION METRICS v2")
    print("Modes: validate (subset->side tables) | rebuild (burn+regen) | incremental")
    print("=" * 80)
    mode = input("mode [validate/rebuild/incremental]: ").strip().lower() or 'validate'

    if mode == 'validate':
        # Scope is controlled by Pxs_df (all its columns, eligibility-screened).
        results = run(Pxs_df, ed_df, vayf_df, DEFAULT_EXC_L, mode='validate')
    else:
        confirm = input(f"Run '{mode}' against PRODUCTION tables? (yes/n): ").strip().lower()
        if confirm == 'yes':
            results = run(Pxs_df, ed_df, vayf_df, DEFAULT_EXC_L, mode=mode)
        else:
            print("Cancelled.")

