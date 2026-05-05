#!/usr/bin/env python
# coding: utf-8

"""
Factor Model - v2 (Self-Contained)
====================================
Sequential Fama-MacBeth cross-sectional factor model.
Fully self-contained — no dependency on factor_model_v1 in the kernel.

Step sequence (v2):
  Step 1:  Market Beta       input=raw_rets
  Step 2:  Quality           input=resid_mkt       ⊥ {beta}
  Step 3:  Idio Momentum     input=resid_quality   ⊥ {beta, quality}
  Step 4:  Size              input=resid_mom       ⊥ {beta, quality, mom}
  Step 5:  Value             input=resid_size      ⊥ {beta, quality, mom, size}
  Step 6:  SI Composite      input=resid_value     ⊥ all prior
  Step 7:  GK Volatility     input=resid_si        ⊥ all prior
  Step 8:  Macro Factors     input=resid_vol       raw betas, joint Ridge CV
  Step 9:  Sector Dummies    input=resid_macro     sum-to-zero, Ridge CV
  Step 10: O-U Mean Rev      input=resid_sec       ⊥ all prior

PIT weight integration:
  - After Step 1: quality PIT weights updated if today is an anchor date
  - After Step 4: value PIT weights updated if today is an anchor date
  - Quality/value scores reloaded with new weights before proceeding

Usage:
    from factor_model_v2 import run
    run(Pxs_df, sectors_s)
    run(Pxs_df, sectors_s, volumeTrd_df=volumeTrd_df)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE           = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
DYNAMIC_SIZE_TBL = 'dynamic_size_df'
BETA_WINDOW      = 252
BETA_HL          = 126
VOL_WINDOW       = 84
VOL_HL           = 42
MACRO_COLS = [
    'USGG2YR', 'US10Y2Y_SPREAD_CHG', 'US10YREAL',
    'BE5Y5YFWD', 'MOVE', 'Crude', 'XAUUSD',
]
MOM_LONG          = 252
MOM_SKIP          = 21
MOM_LONG_BUFFER   = MOM_LONG
RIDGE_GRID_MACRO  = [0.15, 0.3, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
RIDGE_GRID_SEC    = [0.1, 0.2, 0.4, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
MIN_STOCKS        = 150
SI_COMPOSITE_TBL  = 'si_composite_df'
SI_HORIZON        = 21
QUALITY_ANCHOR_TBL = 'valuation_metrics_anchors'
OU_REVERSION_TBL  = 'ou_reversion_df'
OU_MEANREV_W      = 60
OU_MIN_OBS        = 30
OU_ST_REV_W       = 21
OU_VOLUME_W       = 10
OU_VOL_CLIP_LO    = 0.5
OU_VOL_CLIP_HI    = 3.0
OU_WEIGHT_REF     = 30.0
OU_WEIGHT_CAP     = 10.0
FM_START_DATE     = pd.Timestamp('2017-01-01')
_STALE_RUN_LIMIT  = 3
QUALITY_ANCHOR_TBL = 'valuation_metrics_anchors'

# V2 table prefix
V2 = 'v2'
def v2tbl(name): return f'{V2}_{name}'

V2_RESID_MKT     = v2tbl('factor_residuals_mkt')
V2_RESID_QUALITY = v2tbl('factor_residuals_quality')
V2_RESID_MOM     = v2tbl('factor_residuals_mom')
V2_RESID_SIZE    = v2tbl('factor_residuals_size')
V2_RESID_VALUE   = v2tbl('factor_residuals_value')
V2_RESID_SI      = v2tbl('factor_residuals_si')
V2_RESID_VOL     = v2tbl('factor_residuals_vol')
V2_RESID_MACRO   = v2tbl('factor_residuals_macro')
V2_RESID_SEC     = v2tbl('factor_residuals_sec')
V2_RESID_OU      = v2tbl('factor_residuals_ou')
V2_LAM_MKT       = v2tbl('lambda_mkt')
V2_LAM_QUALITY   = v2tbl('lambda_quality')
V2_LAM_MOM       = v2tbl('lambda_mom')
V2_LAM_SIZE      = v2tbl('lambda_size')
V2_LAM_VALUE     = v2tbl('lambda_value')
V2_LAM_SI        = v2tbl('lambda_si')
V2_LAM_VOL       = v2tbl('lambda_vol')
V2_LAM_MACRO     = v2tbl('lambda_macro')
V2_LAM_SEC       = v2tbl('lambda_sec')
V2_LAM_OU        = v2tbl('lambda_ou')
V2_OU_TBL        = v2tbl('ou_reversion_df')

# Quality/Value PIT tables
QUALITY_SCORES_TBL      = 'quality_scores_df'
QUALITY_WEIGHTS_PIT_TBL = 'quality_weights_pit'
VALUE_SCORES_TBL        = 'value_scores_df'
VALUE_WEIGHTS_PIT_TBL   = 'value_weights_pit'
VALUE_IC_CACHE_TBL      = 'value_ic_bank'
MAX_HORIZON_QUALITY     = 63
MAX_HORIZON_VALUE       = 63
QUALITY_METRICS_ALL     = [
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    'OM', 'ROI', 'FCF_PG', 'OMd', 'ROId', 'ISGD', 'r&d',
    'GS/S_Vol', 'HSG/S_Vol', 'PSG/S_Vol', 'GE/E_Vol', 'PIG/E_Vol', 'GGP/GP_Vol',
    'GS*r2_S', 'SGD*r2_S', 'OMd*r2_S', 'GE*r2_E', 'PIG*r2_E', 'GGP*r2_GP',
]
RAW_DB_COLS_QUALITY = [
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    'OM', 'ROI', 'FCF_PG', 'OMd', 'ROId', 'ISGD', 'r&d',
    'S Vol', 'E Vol', 'GP Vol', 'r2 S', 'r2 E', 'r2 GP',
]
QUALITY_EXCLUDE_METRICS = ['ROE', 'ROE-P', 'ROEd']
QUALITY_MAX_COMPONENTS  = 10
QUALITY_HORIZONS        = [21, 63]
QUALITY_TOP_PCTILE      = 0.10
QUALITY_WINSOR          = (0.01, 0.99)
QUALITY_VOL_MIN         = 1.0
QF_MAV_WINDOW           = 252
QF_THRESHOLD            = 15    # bps — '10Y RATE' column is in bps
QF_RATE_COL             = '10Y RATE'   # rate column in Pxs_df (in bps)
VALUE_METRICS           = ['P/S', 'P/Ee', 'P/Eo', 'sP/S', 'sP/E', 'sP/GP', 'P/GP']
VALUE_HORIZONS          = [21, 63]
RESIDUAL_SOURCE_QUALITY = 'v2_factor_residuals_mkt'
RESIDUAL_SOURCE_VALUE   = 'v2_factor_residuals_size'

def clean_ticker(t: str) -> str:
    return t.strip().split(' ')[0].upper()


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


# ==============================================================================
# UNIVERSE
# ==============================================================================

def get_universe(Pxs_df: pd.DataFrame, sectors_s: pd.Series,
                 extended_st_dt: pd.Timestamp) -> list:
    """
    Build stock universe: sector-mapped stocks with sufficient price history.
    Uses sectors_s as the canonical universe — NOT filtered by Pxs_df.columns —
    so the universe is invariant to the end date of Pxs_df (no lookahead bias).
    """
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT ticker FROM income_data"
            )).fetchall()
        db_tickers = {r[0].upper() for r in rows}
    except Exception as e:
        print(f"  WARNING: DB universe query failed ({e}) — using sectors_s")
        db_tickers = set(sectors_s.index)

    etf_tickers = set(sectors_s.values)
    pre_dates   = Pxs_df.index[Pxs_df.index < extended_st_dt]

    universe = []
    for col in sectors_s.index:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        if col not in Pxs_df.columns:
            continue
        if len(pre_dates) >= BETA_WINDOW:
            col_data = Pxs_df.loc[pre_dates[-BETA_WINDOW:], col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            if int(col_data.notna().sum()) < BETA_WINDOW // 2:
                continue
        universe.append(col)

    print(f"  Universe: {len(universe)} stocks "
          f"(sector mapped + DB + sufficient pre-start history)")
    return universe


# ==============================================================================
# DYNAMIC SIZE — DB CACHED
# ==============================================================================

def _compute_dynamic_size_for_dates(dates_to_calc: pd.DatetimeIndex,
                                     universe: list,
                                     Pxs_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date in dates_to_calc and each stock in universe:
      shares   = Size_db / Price_db  (last available Size in valuation_consolidated
                                      before calc_date, Price from Pxs_df on same date)
      dyn_size = shares * Price on calc_date
      Fallback: if Price_db is NaN -> use Size_db as-is
    """
    us_tickers = [t + ' US' for t in universe]

    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, ticker, "Size"
            FROM valuation_consolidated
            WHERE "Size" IS NOT NULL
              AND ticker = ANY(:tickers)
            ORDER BY ticker, date
        """), {"tickers": us_tickers}).fetchall()

    size_raw           = pd.DataFrame(rows, columns=['date', 'ticker', 'Size'])
    size_raw['date']   = pd.to_datetime(size_raw['date'])
    size_raw['ticker'] = size_raw['ticker'].str.replace(' US', '', regex=False)

    size_pivot = size_raw.pivot_table(
        index='date', columns='ticker', values='Size', aggfunc='last'
    )

    # Forward fill to all Pxs_df dates to get last known Size_db per date
    all_px_dates = Pxs_df.index
    size_ff      = size_pivot.reindex(all_px_dates).ffill().bfill()

    # Track which DB date each forward-filled value came from
    date_indicator = pd.DataFrame(
        index=size_pivot.index,
        columns=size_pivot.columns,
        data=np.tile(
            size_pivot.index.values.reshape(-1, 1),
            (1, len(size_pivot.columns))
        )
    )
    date_indicator = date_indicator.reindex(all_px_dates).ffill().bfill()

    # Build price lookup dict per ticker for fast access
    results = {}
    for dt in dates_to_calc:
        if dt not in Pxs_df.index:
            continue
        row = {}
        for ticker in universe:
            if ticker not in size_ff.columns:
                continue
            size_db = size_ff.loc[dt, ticker]
            if pd.isna(size_db):
                continue

            # Price on DB snapshot date
            db_date   = pd.Timestamp(date_indicator.loc[dt, ticker])
            price_db  = Pxs_df.loc[db_date, ticker]                         if db_date in Pxs_df.index else np.nan

            # Current price
            price_t   = Pxs_df.loc[dt, ticker]

            if pd.isna(price_db) or price_db == 0 or pd.isna(price_t):
                row[ticker] = size_db          # fallback
            else:
                shares      = size_db / price_db
                row[ticker] = shares * price_t

        results[dt] = row

    df             = pd.DataFrame(results).T
    df.index.name  = 'date'
    df             = df.reindex(columns=universe)
    return df


def load_dynamic_size(universe: list,
                       Pxs_df: pd.DataFrame,
                       all_calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load dynamic size from DB cache, computing only missing dates.
    Returns DataFrame: date x ticker (all dates in all_calc_dates).
    """
    # Check which dates already exist in DB
    already_done = set()
    try:
        with ENGINE.connect() as conn:
            # Check table exists
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": DYNAMIC_SIZE_TBL}).scalar()

        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT DISTINCT date FROM {DYNAMIC_SIZE_TBL}
                """)).fetchall()
            already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    dates_to_calc = [d for d in all_calc_dates if d not in already_done]

    if dates_to_calc:
        print(f"  Computing dynamic size for {len(dates_to_calc)} new dates "
              f"({len(already_done)} already in DB)...")
        new_df = _compute_dynamic_size_for_dates(
            pd.DatetimeIndex(dates_to_calc), universe, Pxs_df
        )
        # Save new dates to DB
        long           = new_df.stack(dropna=False).reset_index()
        long.columns   = ['date', 'ticker', 'size']
        long           = long.dropna(subset=['size'])
        long['date']   = pd.to_datetime(long['date'])

        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {DYNAMIC_SIZE_TBL} (
                    date   DATE,
                    ticker VARCHAR(20),
                    size   NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
        long.to_sql(DYNAMIC_SIZE_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} new rows to '{DYNAMIC_SIZE_TBL}'")
    else:
        print(f"  Dynamic size: all {len(all_calc_dates)} dates already in DB")

    # Load full table for requested dates
    date_list = [d.date() for d in all_calc_dates]
    print(f"  Loading dynamic size from DB...")
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, ticker, size FROM {DYNAMIC_SIZE_TBL}
            WHERE date = ANY(:dates)
        """), {"dates": date_list}).fetchall()

    df             = pd.DataFrame(rows, columns=['date', 'ticker', 'size'])
    df['date']     = pd.to_datetime(df['date'])
    df['size']     = df['size'].astype(float)
    pivot          = df.pivot_table(index='date', columns='ticker',
                                    values='size', aggfunc='last')
    pivot          = pivot.reindex(columns=universe)
    print(f"  Dynamic size loaded: {pivot.shape}")
    return pivot


def get_log_size(dynamic_size: pd.DataFrame,
                 calc_date: pd.Timestamp,
                 valid_idx: pd.Index) -> pd.Series:
    """
    Returns log(dynamic_size) for valid_idx on calc_date.
    Used as OLS weights (not z-scored — normalization inside WLS).
    Falls back to 1.0 where missing.
    """
    if calc_date not in dynamic_size.index:
        return pd.Series(1.0, index=valid_idx)
    s = dynamic_size.loc[calc_date, valid_idx].reindex(valid_idx)
    s = np.log(s.clip(lower=1).fillna(1))
    return s


# ==============================================================================
# SECTOR DUMMIES
# ==============================================================================

def build_sector_dummies(universe: list, sectors_s: pd.Series) -> pd.DataFrame:
    """
    Build sector dummy matrix using sum-to-zero (deviation) coding.

    Each stock gets +1 for its own sector and -1/(K-1) for all other sectors,
    where K = total number of sectors. This ensures sum(lambda_k) = 0 by
    construction, so the intercept captures the true equal-weighted market
    return rather than the return of an arbitrary reference sector.

    All K sectors are included — no reference sector dropped.
    """
    sectors_dedup = sectors_s[~sectors_s.index.duplicated(keep='first')]
    etfs = sorted(set(
        sectors_dedup.loc[sectors_dedup.index.isin(universe)].dropna().values
    ))
    K = len(etfs)

    # Deviation coding: +1 own sector, -1/(K-1) all others
    fill_val = -1.0 / (K - 1) if K > 1 else 0.0
    dummies  = pd.DataFrame(fill_val, index=universe, columns=etfs)
    for stk in universe:
        etf = sectors_dedup.get(stk)
        if etf is not None and etf in etfs:
            dummies.loc[stk, etf] = 1.0

    print(f"  Sector dummies: {K} sectors (sum-to-zero deviation coding, "
          f"no reference sector dropped)")
    return dummies


# ==============================================================================
# ROLLING CHARACTERISTICS
# ==============================================================================

def calc_rolling_betas(Pxs_df: pd.DataFrame, universe: list,
                        calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    print("  Calculating rolling EWMA betas...")
    spx_rets  = Pxs_df['SPX'].pct_change()
    stk_rets  = Pxs_df[universe].pct_change()
    all_dates = Pxs_df.index
    betas     = {}

    for dt in calc_dates:
        window   = all_dates[all_dates < dt][-BETA_WINDOW:]
        if len(window) < BETA_WINDOW // 2:
            continue

        spx_w    = spx_rets.loc[window].values
        stk_w_df = stk_rets.loc[window].fillna(0)
        cols     = stk_w_df.columns.tolist()
        stk_w    = stk_w_df.values

        n        = len(window)
        alpha    = 1 - np.exp(-np.log(2) / BETA_HL)
        weights  = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        spx_mean = np.dot(weights, spx_w)
        stk_mean = stk_w.T @ weights
        spx_dev  = spx_w - spx_mean
        stk_dev  = stk_w - stk_mean[np.newaxis, :]

        cov      = (stk_dev * spx_dev[:, np.newaxis] *
                    weights[:, np.newaxis]).sum(axis=0)
        var_spx  = np.dot(weights, spx_dev ** 2)

        beta_t    = cov / var_spx if var_spx > 0                     else np.full(len(cols), np.nan)
        betas[dt] = pd.Series(beta_t, index=cols)

    beta_df = pd.DataFrame(betas).T.reindex(columns=universe)
    beta_df.index.name = 'date'
    print(f"  Betas computed: {len(beta_df)} dates")
    return beta_df


def calc_macro_betas(Pxs_df: pd.DataFrame,
                     universe: list,
                     calc_dates: pd.DatetimeIndex) -> dict:
    """
    Compute EWMA rolling betas of each stock vs each macro factor change.
    Same EWMA structure as market beta: BETA_WINDOW=252, BETA_HL=126.

    Inputs (from Pxs_df columns, pre-computed daily changes):
      USGG2YR           2Y nominal rate changes
      USGG10YR_SPREAD   2Y/10Y spread changes (computed as USGG10YR - USGG2YR)
      US10YREAL         10Y real yield / inflation breakeven changes
      BE5Y5YFWD         5y5y forward breakeven inflation changes
      MOVE              Interest rate volatility index daily change
      Crude Oil USD/Bbl WTI crude changes
      XAUUSD            Gold changes
      M2MP Velocity     Money velocity changes (forward-filled weekly)
      VIX Mom           VIX momentum (already transformed, used as-is)

    For each date t, stock i, macro factor m:
        beta_im = Cov_EWMA(r_i, d_m) / Var_EWMA(d_m)

    Each macro beta series z-scored cross-sectionally per date.

    Returns dict: {macro_col: DataFrame(dates x tickers)} for run_factor_step.
    """
    print("  Calculating macro factor betas...")

    # Build macro change series from Pxs_df
    macro_raw = {}
    for col in MACRO_COLS:
        if col in Pxs_df.columns:
            macro_raw[col] = Pxs_df[col]
        else:
            print(f"  WARNING: '{col}' not found in Pxs_df — skipping")

    if not macro_raw:
        print("  WARNING: no macro factors found — skipping macro step")
        return {}

    avail_cols = list(macro_raw.keys())
    print(f"  Macro factors available: {avail_cols}")

    stk_rets  = Pxs_df[universe].pct_change()
    all_dates = Pxs_df.index
    alpha     = 1 - np.exp(-np.log(2) / BETA_HL)

    # Results: {macro_col: {date: Series(ticker -> beta)}}
    betas_by_macro = {col: {} for col in avail_cols}

    for dt in calc_dates:
        window = all_dates[all_dates < dt][-BETA_WINDOW:]
        if len(window) < BETA_WINDOW // 2:
            continue

        # EWMA weights
        n       = len(window)
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Stock returns over window
        stk_w = stk_rets.loc[window].fillna(0).values   # (n x tickers)
        stk_mean = stk_w.T @ weights                      # (tickers,)
        stk_dev  = stk_w - stk_mean[np.newaxis, :]        # (n x tickers)

        for col in avail_cols:
            macro_s = macro_raw[col].reindex(window).fillna(0).values  # (n,)

            macro_mean = np.dot(weights, macro_s)
            macro_dev  = macro_s - macro_mean
            var_macro  = np.dot(weights, macro_dev ** 2)

            if var_macro <= 0:
                continue

            cov    = (stk_dev * macro_dev[:, np.newaxis] * weights[:, np.newaxis]).sum(axis=0)
            beta_t = cov / var_macro

            betas_by_macro[col][dt] = pd.Series(beta_t, index=universe)

    # Build DataFrames and z-score cross-sectionally
    result = {}
    for col in avail_cols:
        if not betas_by_macro[col]:
            continue
        df = pd.DataFrame(betas_by_macro[col]).T.reindex(columns=universe)
        df.index.name = 'date'
        # Z-score cross-sectionally each date
        df = df.apply(zscore, axis=1)
        result[col] = df
        print(f"  {col}: {len(df)} dates computed")

    return result



def calc_idio_momentum(resid_sec_df: pd.DataFrame,
                        calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Idiosyncratic momentum: cumulative sum of sector residuals
    over [t-MOM_LONG, t-MOM_SKIP], z-scored cross-sectionally.
    """
    print("  Calculating idiosyncratic momentum from sec residuals...")
    all_resid_dates = resid_sec_df.index
    mom_dict        = {}

    for dt in calc_dates:
        past = all_resid_dates[all_resid_dates < dt]
        if len(past) < MOM_LONG + 1:
            continue

        window = past[-MOM_LONG:-MOM_SKIP]
        if len(window) < MOM_LONG - MOM_SKIP - 10:
            continue

        cum_resid = resid_sec_df.loc[window].sum(axis=0)
        valid     = cum_resid.dropna()

        if len(valid) < MIN_STOCKS:
            continue

        mom_dict[dt] = zscore(valid)

    mom_df = pd.DataFrame(mom_dict).T.reindex(columns=resid_sec_df.columns)
    mom_df.index.name = 'date'
    print(f"  Idiosyncratic momentum computed: {len(mom_df)} dates")
    return mom_df


def calc_idio_momentum_volscaled(resid_sec_df: pd.DataFrame,
                                  volumeTrd_df: pd.DataFrame,
                                  calc_dates: pd.DatetimeIndex,
                                  vol_lower: float = 0.5,
                                  vol_upper: float = 3.0) -> pd.DataFrame:
    """
    Volume-scaled idiosyncratic momentum.
    volumeTrd_df is assumed to contain precomputed volume scalars
    (e.g. volume(t) / mean(volume[t-10, t-1])), clipped to [vol_lower, vol_upper].
    Cumulative volume-weighted idio return over [t-MOM_LONG, t-MOM_SKIP],
    z-scored cross-sectionally per date.
    """
    print(f"  Calculating volume-scaled idio momentum "
          f"(clip=[{vol_lower}, {vol_upper}])...")

    # Clip scalars to bounds (in case not pre-clipped)
    vol_scalars = volumeTrd_df.clip(lower=vol_lower, upper=vol_upper)

    all_resid_dates = resid_sec_df.index
    mom_dict        = {}

    for dt in calc_dates:
        past = all_resid_dates[all_resid_dates < dt]
        if len(past) < MOM_LONG + 1:
            continue

        window = past[-MOM_LONG:-MOM_SKIP]
        if len(window) < MOM_LONG - MOM_SKIP - 10:
            continue

        # Align window to dates available in both resid and vol scalars
        common_days  = window.intersection(vol_scalars.index)
        if len(common_days) < MOM_LONG - MOM_SKIP - 10:
            continue

        weighted_sum = (resid_sec_df.loc[common_days] *
                        vol_scalars.loc[common_days]).sum(axis=0)
        valid        = weighted_sum.dropna()

        if len(valid) < MIN_STOCKS:
            continue

        mom_dict[dt] = zscore(valid)

    mom_df            = pd.DataFrame(mom_dict).T.reindex(columns=resid_sec_df.columns)
    mom_df.index.name = 'date'
    print(f"  Volume-scaled idio momentum computed: {len(mom_df)} dates")
    return mom_df


def calc_reversal_21d(Pxs_df: pd.DataFrame,
                       universe: list,
                       calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Short-term reversal: log(P[t-1] / P[t-22])
    Captures last 21 trading days of raw price performance.
    Expected sign: negative (recent winners mean-revert).
    Z-scored cross-sectionally per date.
    """
    print("  Calculating 21-day short-term reversal from prices...")
    all_px_dates = Pxs_df.index
    rev_dict     = {}

    for dt in calc_dates:
        past = all_px_dates[all_px_dates < dt]
        if len(past) < 22:
            continue

        p_recent = Pxs_df.loc[past[-1],  universe]   # yesterday
        p_old    = Pxs_df.loc[past[-22], universe]   # 21 trading days ago

        valid_mask = (p_recent > 0) & (p_old > 0)
        rev        = np.log(p_recent / p_old).where(valid_mask).dropna()

        if len(rev) < MIN_STOCKS:
            continue

        rev_dict[dt] = zscore(rev)

    rev_df            = pd.DataFrame(rev_dict).T.reindex(columns=universe)
    rev_df.index.name = 'date'
    print(f"  21d reversal computed: {len(rev_df)} dates")
    return rev_df


def load_ohlc_tables(universe: list) -> tuple:
    """
    Load open, high, low prices from DB for universe tickers.
    Tables: daily_open, daily_high, daily_low.
    Columns have ' US' extension in DB; stripped to bare tickers on return.
    Returns: (open_df, high_df, low_df) — each a DataFrame (dates x bare tickers).
    """
    def _load_table(tbl: str) -> pd.DataFrame:
        try:
            with ENGINE.connect() as conn:
                df = pd.read_sql(text(f"SELECT * FROM {tbl}"), conn)
        except Exception as e:
            print(f"  ERROR loading '{tbl}': {e}")
            return pd.DataFrame()
        # Prefer columns with 'date' in name, fall back to 'index'
        date_col = [c for c in df.columns if 'date' in c.lower()]
        if not date_col:
            date_col = [c for c in df.columns if c.lower() == 'index']
        if not date_col:
            print(f"  ERROR loading '{tbl}': no date column found "
                  f"(columns: {list(df.columns[:5])})")
            return pd.DataFrame()
        dc = date_col[0]
        df[dc] = pd.to_datetime(df[dc])
        df = df.set_index(dc).sort_index()
        df.columns = [clean_ticker(c) for c in df.columns]
        keep = [t for t in universe if t in df.columns]
        if not keep:
            print(f"  ERROR: no universe tickers found in '{tbl}' after normalization "
                  f"(sample cols: {list(df.columns[:5])})")
            return pd.DataFrame()
        return df[keep].astype(float)

    print("  Loading OHLC tables from DB...")
    open_df  = _load_table('daily_open')
    high_df  = _load_table('daily_high')
    low_df   = _load_table('daily_low')

    print(f"  daily_open : {open_df.shape if not open_df.empty else 'EMPTY'}")
    print(f"  daily_high : {high_df.shape if not high_df.empty else 'EMPTY'}")
    print(f"  daily_low  : {low_df.shape  if not low_df.empty  else 'EMPTY'}")

    # Check universe coverage
    for name, df in [('daily_open', open_df), ('daily_high', high_df), ('daily_low', low_df)]:
        if not df.empty:
            missing = [t for t in universe if t not in df.columns]
            if missing:
                print(f"  {name}: {len(missing)} universe tickers missing from columns "
                      f"(e.g. {missing[:5]})")
            else:
                print(f"  {name}: all {len(universe)} universe tickers present")

    if open_df.empty or high_df.empty or low_df.empty:
        print("  WARNING: OHLC tables incomplete — will fall back to close-to-close vol")
        return None, None, None

    print(f"  OHLC loaded: {open_df.shape[0]} dates x {len(open_df.columns)} tickers")
    return open_df, high_df, low_df


def calc_vol_factor(Pxs_df: pd.DataFrame,
                    universe: list,
                    calc_dates: pd.DatetimeIndex,
                    open_df: pd.DataFrame = None,
                    high_df: pd.DataFrame = None,
                    low_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Short-window EWMA realized volatility factor.
    Window: VOL_WINDOW (84d), half-life: VOL_HL (42d).
    Shorter than beta (252d/hl=126) to capture distinct variation.

    If OHLC DataFrames provided: uses Garman-Klass estimator —
      σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
    ~8x more efficient than close-to-close. Falls back gracefully if unavailable.

    Z-scored cross-sectionally per date.
    Expected sign: negative (high vol stocks underperform risk-adjusted).
    """
    use_gk = (open_df is not None and high_df is not None and low_df is not None)
    method = "Garman-Klass" if use_gk else "close-to-close"
    print(f"  Calculating vol factor ({method}, window={VOL_WINDOW}d, hl={VOL_HL}d)...")

    alpha        = 1 - np.exp(-np.log(2) / VOL_HL)
    all_px_dates = Pxs_df.index
    vol_dict     = {}

    for dt in calc_dates:
        past = all_px_dates[all_px_dates < dt]
        if len(past) < VOL_WINDOW // 2:
            continue

        window = past[-VOL_WINDOW:]

        if use_gk:
            # Garman-Klass: σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
            C = Pxs_df.loc[window, universe]
            O = open_df.reindex(index=window, columns=universe)
            H = high_df.reindex(index=window, columns=universe)
            L = low_df.reindex(index=window,  columns=universe)

            valid_mask = (C > 0) & (O > 0) & (H > 0) & (L > 0) & (H >= L)
            log_hl     = np.log(H / L).where(valid_mask)
            log_co     = np.log(C / O).where(valid_mask)
            gk_var     = (0.5 * log_hl ** 2
                          - (2 * np.log(2) - 1) * log_co ** 2).clip(lower=0)

            n        = len(window)
            weights  = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()

            ewma_var = gk_var.mul(weights, axis=0).sum(axis=0)
            ewma_vol = np.sqrt(ewma_var * 252)

        else:
            px_win = Pxs_df.loc[window, universe]
            rets   = px_win.pct_change().dropna(how='all')
            if len(rets) < VOL_WINDOW // 2:
                continue

            n        = len(rets)
            weights  = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()

            ewma_var = (rets ** 2).mul(weights, axis=0).sum(axis=0)
            ewma_vol = np.sqrt(ewma_var * 252)

        valid = ewma_vol.replace(0, np.nan).dropna()
        if len(valid) < MIN_STOCKS:
            continue

        vol_dict[dt] = zscore(valid)

    vol_df            = pd.DataFrame(vol_dict).T.reindex(columns=universe)
    vol_df.index.name = 'date'
    print(f"  Vol factor computed: {len(vol_df)} dates ({method})")
    return vol_df


# ==============================================================================
# CROSS-SECTIONAL WLS
# ==============================================================================

def wls_cross_section(y: pd.Series, X: pd.DataFrame,
                       w: pd.Series) -> tuple:
    idx  = y.index.intersection(X.index).intersection(w.index)
    if len(idx) < 10:
        return None, None, None

    y_   = y.loc[idx].values
    X_   = np.column_stack([np.ones(len(idx)), X.loc[idx].values])
    w_   = w.loc[idx].values
    w_   = np.where(np.isnan(w_) | (w_ <= 0), 1.0, w_)
    w_   = w_ / w_.sum()

    W    = np.diag(w_)
    try:
        XtW  = X_.T @ W
        lam  = np.linalg.solve(XtW @ X_, XtW @ y_)
    except np.linalg.LinAlgError:
        return None, None, None

    fitted  = X_ @ lam
    resid   = y_ - fitted
    ss_res  = np.dot(w_, resid ** 2)
    ss_tot  = np.dot(w_, (y_ - np.dot(w_, y_)) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return lam, pd.Series(resid, index=idx), r2


def wls_ridge_cross_section(y: pd.Series, X: pd.DataFrame,
                             w: pd.Series, ridge_lambda: float = 0.0) -> tuple:
    """
    Weighted least squares with optional L2 (ridge) regularization.
    Intercept is NOT penalized — ridge penalty applied to slope coefficients only.
    ridge_lambda=0.0 reduces to standard WLS (same as wls_cross_section).
    """
    idx  = y.index.intersection(X.index).intersection(w.index)
    if len(idx) < 10:
        return None, None, None

    y_   = y.loc[idx].values
    X_   = np.column_stack([np.ones(len(idx)), X.loc[idx].values])
    w_   = w.loc[idx].values
    w_   = np.where(np.isnan(w_) | (w_ <= 0), 1.0, w_)
    w_   = w_ / w_.sum()

    W    = np.diag(w_)
    XtW  = X_.T @ W
    XtWX = XtW @ X_

    # Ridge penalty — intercept (col 0) unpenalized, slopes penalized
    n_slopes = X_.shape[1] - 1
    pen      = np.zeros(X_.shape[1])
    pen[1:]  = ridge_lambda                         # skip intercept
    XtWX_reg = XtWX + np.diag(pen)

    try:
        lam = np.linalg.solve(XtWX_reg, XtW @ y_)
    except np.linalg.LinAlgError:
        return None, None, None

    fitted  = X_ @ lam
    resid   = y_ - fitted
    ss_res  = np.dot(w_, resid ** 2)
    ss_tot  = np.dot(w_, (y_ - np.dot(w_, y_)) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return lam, pd.Series(resid, index=idx), r2


# ==============================================================================
# STORAGE
# ==============================================================================

def save_lambdas(lambda_df: pd.DataFrame, table_name: str):
    with ENGINE.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    lambda_df.to_sql(table_name, ENGINE, if_exists='replace',
                     index=True, index_label='date')
    print(f"  Lambdas saved to '{table_name}' ({len(lambda_df)} rows)")


def save_lambdas_incremental(lambda_df: pd.DataFrame, table_name: str):
    """Upsert lambda rows — delete existing dates then reinsert.
    If the table schema doesn't match (e.g. new columns added), drops and recreates."""
    if lambda_df is None or len(lambda_df) == 0:
        return
    dates = [d.date() for d in pd.to_datetime(lambda_df.index)]
    try:
        with ENGINE.begin() as conn:
            conn.execute(text(f"DELETE FROM {table_name} WHERE date = ANY(:d)"), {"d": dates})
        lambda_df.to_sql(table_name, ENGINE, if_exists='append',
                         index=True, index_label='date')
    except Exception:
        # Schema mismatch or table doesn't exist — drop and recreate
        try:
            with ENGINE.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        except Exception:
            pass
        lambda_df.to_sql(table_name, ENGINE, if_exists='replace',
                         index=True, index_label='date')
    print(f"  Lambdas '{table_name}': saved {len(lambda_df)} date(s)")


def save_residuals_incremental(resid_df: pd.DataFrame, table_name: str):
    """Upsert residual rows — delete existing dates then reinsert."""
    if resid_df is None or resid_df.empty:
        return
    dates = [d.date() for d in pd.to_datetime(resid_df.index)]
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date   DATE,
                ticker VARCHAR(20),
                resid  NUMERIC,
                PRIMARY KEY (date, ticker)
            )
        """))
        try:
            conn.execute(text(f"DELETE FROM {table_name} WHERE date = ANY(:d)"), {"d": dates})
        except Exception:
            pass
    long           = resid_df.stack().reset_index()
    long.columns   = ['date', 'ticker', 'resid']
    long['date']   = pd.to_datetime(long['date'])
    long.to_sql(table_name, ENGINE, if_exists='append', index=False)
    print(f"  Residuals '{table_name}': saved {len(long):,} rows ({len(resid_df)} date(s))")


def save_residuals(resid_df: pd.DataFrame, table_name: str):
    print(f"  Saving residuals to '{table_name}'...")
    long           = resid_df.stack().reset_index()
    long.columns   = ['date', 'ticker', 'resid']
    long['date']   = pd.to_datetime(long['date'])

    with ENGINE.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                date   DATE,
                ticker VARCHAR(20),
                resid  NUMERIC,
                PRIMARY KEY (date, ticker)
            )
        """))
    long.to_sql(table_name, ENGINE, if_exists='append', index=False)
    print(f"  Saved {len(long):,} rows "
          f"({resid_df.shape[0]} dates x {resid_df.shape[1]} stocks)")


def get_anchor_date(table_name: str = 'factor_residuals_mkt'):
    """
    Returns the latest date already stored in the given table.
    Used as the shared anchor for incremental updates.
    Returns None if table doesn't exist or is empty.
    """
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(
                text(f"SELECT MAX(date) FROM {table_name}")
            ).fetchone()
        if row and row[0]:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None




# ==============================================================================
# VARIANCE / R2 / LAMBDA STATS
# ==============================================================================

def variance_stats(resid_df: pd.DataFrame, label: str,
                    reference_var: float = None) -> float:
    vals = resid_df.values.flatten()
    vals = vals[~np.isnan(vals)]
    var  = float(np.var(vals))
    std  = float(np.std(vals))

    print(f"\n  [{label}]")
    print(f"    Pooled variance : {var:.8f}")
    print(f"    Pooled std dev  : {std:.6f}")
    print(f"    N observations  : {len(vals):,}")
    if reference_var is not None:
        print(f"    % of reference  : {var / reference_var * 100:.2f}%")
    return var


def r2_stats(r2_series: pd.Series, label: str):
    r2 = r2_series.dropna()
    print(f"\n  [{label}] Daily cross-sectional R²:")
    print(f"    Mean   : {r2.mean():.4f}")
    print(f"    Median : {r2.median():.4f}")
    print(f"    10th   : {r2.quantile(0.10):.4f}")
    print(f"    90th   : {r2.quantile(0.90):.4f}")


def lambda_stats(series: pd.Series, label: str) -> float:
    s       = series.dropna()
    mean    = s.mean()
    std     = s.std()
    t_stat  = mean / (std / np.sqrt(len(s)))
    pct_pos = (s > 0).mean() * 100

    print(f"\n  {label}")
    print(f"    N          : {len(s):,}")
    print(f"    Mean       : {mean:+.6f}")
    print(f"    Std        : {std:.6f}")
    print(f"    t-stat     : {t_stat:+.2f}")
    print(f"    % positive : {pct_pos:.1f}%")
    print(f"    Min        : {s.min():+.6f}")
    print(f"    5th pct    : {s.quantile(0.05):+.6f}")
    print(f"    Median     : {s.median():+.6f}")
    print(f"    95th pct   : {s.quantile(0.95):+.6f}")
    print(f"    Max        : {s.max():+.6f}")
    return t_stat


def print_lambda_summary(lambda_df: pd.DataFrame,
                          factor_cols: list,
                          step_label: str,
                          common_dates: pd.DatetimeIndex,
                          annual_col: str = None):
    """Stats computed on common_dates only for clean comparability."""
    lm = lambda_df[lambda_df.index.isin(common_dates)].copy()

    print(f"\n{'='*70}")
    print(f"  LAMBDA DISTRIBUTIONS — {step_label} (common sample)")
    print(f"{'='*70}")

    for col in factor_cols:
        if col not in lm.columns:
            continue
        lambda_stats(lm[col], f"lambda_{col}")

        if col == annual_col:
            clean = lm[col].dropna()
            print(f"\n  Annual breakdown ({col}):")
            print(f"  {'Year':<6} {'Mean':>12} {'t-stat':>10} {'%pos':>8}")
            print(f"  {'-'*40}")
            for yr, grp in clean.groupby(clean.index.year):
                mean  = grp.mean()
                t     = mean / (grp.std() / np.sqrt(len(grp)))
                pct_p = (grp > 0).mean() * 100
                print(f"  {yr:<6} {mean:>+12.6f} {t:>+10.2f} {pct_p:>7.1f}%")
            cum = clean.cumsum()
            print(f"\n  Cumulative {col} lambda: {cum.iloc[-1]:+.4f}")

    print("\n--- Intercept ---")
    if 'intercept' in lm.columns:
        lambda_stats(lm['intercept'], "lambda_0 (intercept)")
    else:
        print("  (intercept not available in this lambda table)")

    if 'ridge_lambda' in lm.columns:
        rl = lm['ridge_lambda'].dropna()
        vc = rl.value_counts().sort_index()
        grid_vals = sorted(vc.index.tolist())
        print(f"\n  Ridge λ selected (optimal per-date, grid={grid_vals}):")
        print(f"  {'λ':>8} {'N days':>8} {'%':>7}")
        print(f"  {'-'*26}")
        for lv, cnt in vc.items():
            print(f"  {lv:>8.2f} {cnt:>8} {cnt/len(rl)*100:>6.1f}%")
        print(f"  Mean λ: {rl.mean():.3f}  |  Median λ: {rl.median():.3f}")


def print_sector_lambdas(lambda_df: pd.DataFrame,
                          sec_cols: list,
                          common_dates: pd.DatetimeIndex):
    lm = lambda_df[lambda_df.index.isin(common_dates)]
    print(f"\n  Sector lambdas ({len(sec_cols)} sectors):")
    print(f"  {'Sector':<10} {'Mean':>10} {'Std':>10} {'t-stat':>10} {'%pos':>8}")
    print(f"  {'-'*52}")
    for col in sorted(sec_cols):
        if col not in lm.columns:
            continue
        s       = lm[col].dropna()
        mean    = s.mean()
        std     = s.std()
        t       = mean / (std / np.sqrt(len(s)))
        pct_pos = (s > 0).mean() * 100
        print(f"  {col:<10} {mean:>+10.6f} {std:>10.6f} {t:>+10.2f} {pct_pos:>7.1f}%")


# ==============================================================================
# GENERIC FACTOR STEP RUNNER
# ==============================================================================    return summary


# ==============================================================================
# CHARACTERISTIC ORTHOGONALIZATION (Gram-Schmidt on characteristics)
# ==============================================================================

def orthogonalize_char(new_char: pd.Series,
                        prior_chars: dict,
                        dt: pd.Timestamp,
                        dynamic_size: pd.DataFrame = None) -> pd.Series:
    """
    Orthogonalize a new characteristic against all prior characteristics
    cross-sectionally on a single date via WLS (weighted by log market cap).

    Using the same cap-weights as run_factor_step ensures the orthogonalization
    is consistent with the WLS regressions — residuals are orthogonal in the
    same weighted inner-product space.

    Falls back to OLS if dynamic_size is not provided or has no data for dt.

    Runs: new_char = a + B * prior_chars + residual  (WLS)
    Returns the residual (new_char orthogonal to all prior factors).
    """
    y = new_char.dropna()
    if len(y) < MIN_STOCKS:
        return new_char

    # Build prior char matrix for this date
    X_parts = []
    for name, char_df in prior_chars.items():
        if char_df is None or char_df.empty:
            continue
        if dt not in char_df.index:
            continue
        col = char_df.loc[dt].reindex(y.index)
        if isinstance(col, pd.DataFrame):
            for c in col.columns:
                X_parts.append(col[c].rename(f"{name}_{c}"))
        else:
            X_parts.append(col.rename(name))

    if not X_parts:
        return new_char

    X = pd.concat(X_parts, axis=1).reindex(y.index).fillna(0)
    valid = y.index.intersection(X.index)
    if len(valid) < MIN_STOCKS:
        return new_char

    y_ = y.loc[valid].values
    X_ = np.column_stack([np.ones(len(valid)), X.loc[valid].values])

    # Build cap-weights — log(market cap), normalized to sum to 1
    w = None
    if dynamic_size is not None and not dynamic_size.empty:
        if dt in dynamic_size.index:
            raw_w = dynamic_size.loc[dt].reindex(valid).fillna(0)
            raw_w = np.log(raw_w.clip(lower=1).values)
            raw_w = np.where(raw_w > 0, raw_w, 0)
            if raw_w.sum() > 0:
                w = raw_w / raw_w.sum()

    try:
        if w is not None:
            # WLS: X^T W X coeff = X^T W y
            W    = np.diag(w)
            XtW  = X_.T @ W
            coeffs = np.linalg.lstsq(XtW @ X_, XtW @ y_, rcond=None)[0]
        else:
            coeffs, _, _, _ = np.linalg.lstsq(X_, y_, rcond=None)
        resid = y_ - X_ @ coeffs
        return pd.Series(resid, index=valid).reindex(new_char.index)
    except Exception:
        return new_char


def orthogonalize_char_df(char_df: pd.DataFrame,
                           prior_chars: dict,
                           calc_dates: pd.DatetimeIndex,
                           dynamic_size: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply orthogonalize_char across all dates in calc_dates using WLS.
    Returns a new DataFrame of the same shape with orthogonalized values.

    char_df      : DataFrame (dates x tickers)
    prior_chars  : dict of {name: char_df} covering the same dates
    dynamic_size : DataFrame (dates x tickers) of market caps for WLS weights.
                   If provided, orthogonalization uses cap-weighted regression
                   consistent with run_factor_step. Falls back to OLS if None.
    """
    result = {}
    for dt in calc_dates:
        if dt not in char_df.index:
            continue
        raw = char_df.loc[dt].dropna()
        if len(raw) < MIN_STOCKS:
            result[dt] = raw
            continue
        perp = orthogonalize_char(raw, prior_chars, dt, dynamic_size=dynamic_size)
        result[dt] = perp
    out = pd.DataFrame(result).T
    out.index.name = 'date'
    return out.reindex(columns=char_df.columns)


# ==============================================================================
# GENERIC FACTOR STEP RUNNER
# ==============================================================================

def run_factor_step(factor_cols: list,
                     char_by_date: dict,
                     all_rets: pd.DataFrame,
                     dynamic_size: pd.DataFrame,
                     calc_dates: pd.DatetimeIndex,
                     universe: list,
                     ridge_lambda: float = 0.0) -> tuple:
    resid_dict  = {}
    lambda_dict = {}
    r2_dict     = {}

    # Restrict universe to tickers present in all_rets (may be residuals with fewer tickers)
    valid_universe = [t for t in universe if t in all_rets.columns]

    for dt in calc_dates:
        if dt not in all_rets.index:
            continue
        y = all_rets.loc[dt, valid_universe].dropna()
        if len(y) < MIN_STOCKS:
            continue

        valid_idx = y.index
        X_parts   = []

        for col, char_df in char_by_date.items():
            if dt not in char_df.index:
                valid_idx = pd.Index([])
                break
            s         = char_df.loc[dt].reindex(valid_idx).dropna()
            valid_idx = s.index
            X_parts.append(s.rename(col))

        if len(valid_idx) < MIN_STOCKS or not X_parts:
            continue

        X  = pd.concat(X_parts, axis=1).loc[valid_idx]
        y_ = y.loc[valid_idx]
        w_ = get_log_size(dynamic_size, dt, valid_idx)

        lam, resid, r2 = wls_ridge_cross_section(y_, X, w_, ridge_lambda=ridge_lambda)
        if resid is None:
            continue

        resid_dict[dt]  = resid
        r2_dict[dt]     = r2
        cols            = ['intercept'] + factor_cols
        lambda_dict[dt] = {**dict(zip(cols, lam)), 'r2': r2}

    resid_df  = pd.DataFrame(resid_dict).T
    if not resid_df.empty:
        resid_df.index = pd.to_datetime(resid_df.index)
    lambda_df = pd.DataFrame(lambda_dict).T
    lambda_df.index.name = 'date'
    if not lambda_df.empty:
        lambda_df.index = pd.to_datetime(lambda_df.index)
    r2_s      = pd.Series(r2_dict)

    return resid_df, lambda_df, r2_s


# ==============================================================================
# OPTIMAL RIDGE — per-date λ selection minimizing cross-sectional residual variance
# ==============================================================================

def run_factor_step_optimal_ridge(factor_cols: list,
                                   char_by_date: dict,
                                   all_rets: pd.DataFrame,
                                   dynamic_size: pd.DataFrame,
                                   calc_dates: pd.DatetimeIndex,
                                   universe: list,
                                   lambda_grid: list = None,
                                   k_folds: int = 5,
                                   default_lambda: float = 0.2) -> tuple:
    """
    Same as run_factor_step but selects the ridge λ via k-fold cross-validation
    on the stock dimension — fit on k-1 folds, evaluate residual variance on
    the held-out fold, average across folds, pick the λ minimising OOS variance.

    Fallback to default_lambda if k-fold fails for a date.
    default_lambda: 0.5 for macro (6 correlated features), 0.2 for step 7 (3 features).

    Returns same (resid_df, lambda_df, r2_s) tuple as run_factor_step,
    with 'ridge_lambda' column appended to lambda_df.
    """
    if lambda_grid is None:
        lambda_grid = RIDGE_GRID

    resid_dict  = {}
    lambda_dict = {}
    r2_dict     = {}

    valid_universe = [t for t in universe if t in all_rets.columns]

    for dt in calc_dates:
        if dt not in all_rets.index:
            continue
        y = all_rets.loc[dt, valid_universe].dropna()
        if len(y) < MIN_STOCKS:
            continue

        valid_idx = y.index
        X_parts   = []

        for col, char_df in char_by_date.items():
            if dt not in char_df.index:
                valid_idx = pd.Index([])
                break
            s         = char_df.loc[dt].reindex(valid_idx).dropna()
            valid_idx = s.index
            X_parts.append(s.rename(col))

        if len(valid_idx) < MIN_STOCKS or not X_parts:
            continue

        X  = pd.concat(X_parts, axis=1).loc[valid_idx]
        y_ = y.loc[valid_idx]
        w_ = get_log_size(dynamic_size, dt, valid_idx)
        w_arr = w_.values
        w_arr = np.where(np.isnan(w_arr) | (w_arr <= 0), 1.0, w_arr)

        # --- k-fold CV on stock dimension ---
        n        = len(valid_idx)
        idx_arr  = np.arange(n)
        rng      = np.random.default_rng(seed=42)   # deterministic splits
        shuffled = rng.permutation(idx_arr)
        folds    = np.array_split(shuffled, k_folds)

        cv_var = {lv: [] for lv in lambda_grid}

        for fold_idx in folds:
            if len(fold_idx) < 5:
                continue
            train_idx = np.setdiff1d(idx_arr, fold_idx)
            if len(train_idx) < MIN_STOCKS // 2:
                continue

            # Train set
            y_tr  = y_.iloc[train_idx]
            X_tr  = X.iloc[train_idx]
            w_tr  = pd.Series(w_arr[train_idx], index=y_tr.index)

            # Held-out set
            y_ho  = y_.iloc[fold_idx]
            X_ho  = X.iloc[fold_idx]

            for lv in lambda_grid:
                lam_v, _, _ = wls_ridge_cross_section(y_tr, X_tr, w_tr, ridge_lambda=lv)
                if lam_v is None:
                    continue
                # Predict on held-out fold
                X_ho_mat  = np.column_stack([np.ones(len(fold_idx)), X_ho.values])
                resid_ho  = y_ho.values - X_ho_mat @ lam_v
                # Unweighted variance on held-out (no weights available for OOS)
                cv_var[lv].append(float(np.var(resid_ho)))

        # Pick λ with lowest mean OOS variance
        best_ridge_lv = default_lambda
        best_cv_var   = np.inf
        for lv in lambda_grid:
            if not cv_var[lv]:
                continue
            mean_var = float(np.mean(cv_var[lv]))
            if mean_var < best_cv_var:
                best_cv_var   = mean_var
                best_ridge_lv = lv

        # Final fit on full cross-section with chosen λ
        lam_v, resid_v, r2_v = wls_ridge_cross_section(y_, X, w_, ridge_lambda=best_ridge_lv)

        # Fallback to default if chosen λ fails
        if resid_v is None:
            lam_v, resid_v, r2_v = wls_ridge_cross_section(y_, X, w_, ridge_lambda=default_lambda)
            best_ridge_lv = default_lambda
        if resid_v is None:
            continue

        resid_dict[dt]  = resid_v
        r2_dict[dt]     = r2_v
        cols            = ['intercept'] + factor_cols
        lambda_dict[dt] = {**dict(zip(cols, lam_v)),
                           'r2': r2_v,
                           'ridge_lambda': best_ridge_lv}

    resid_df  = pd.DataFrame(resid_dict).T
    if not resid_df.empty:
        resid_df.index = pd.to_datetime(resid_df.index)
    lambda_df = pd.DataFrame(lambda_dict).T
    lambda_df.index.name = 'date'
    if not lambda_df.empty:
        lambda_df.index = pd.to_datetime(lambda_df.index)
    r2_s = pd.Series(r2_dict)

    return resid_df, lambda_df, r2_s


# ==============================================================================
# SHORT INTEREST COMPOSITE — DB CACHED
# ==============================================================================

def _compute_si_composite_for_dates(dates_to_calc: pd.DatetimeIndex,
                                     universe: list) -> pd.DataFrame:
    """
    For each date in dates_to_calc:
      1. Load SI % Free Float and Utilization from short_interest_data
      2. Forward-fill to calc dates (SI is lower frequency than daily)
      3. Cross-sectional z-score each metric
      4. Equal-weight composite = (z_si_float + z_utilization) / 2
    Returns DataFrame: date x ticker
    """
    us_tickers = [t + ' US' for t in universe]

    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, ticker,
                   short_interest_pct_free_float,
                   short_interest_shares_k,
                   short_availability_shares_k
            FROM short_interest_data
            WHERE ticker = ANY(:tickers)
            ORDER BY ticker, date
        """), {"tickers": us_tickers}).fetchall()

    if not rows:
        print("  WARNING: No SI data found for universe tickers")
        return pd.DataFrame(index=dates_to_calc, columns=universe, dtype=float)

    df           = pd.DataFrame(rows, columns=[
        'date', 'ticker', 'si_float', 'si_shares_k', 'avail_shares_k'
    ])
    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].str.replace(' US', '', regex=False).str.strip()
    for col in ['si_float', 'si_shares_k', 'avail_shares_k']:
        df[col]  = df[col].astype(float)

    df['utilization'] = (
        df['si_shares_k'] / df['avail_shares_k'].replace(0, np.nan)
    ).clip(0, 1)

    # Pivot each metric: date x ticker
    piv_float = df.pivot_table(
        index='date', columns='ticker', values='si_float', aggfunc='last'
    )
    piv_util  = df.pivot_table(
        index='date', columns='ticker', values='utilization', aggfunc='last'
    )

    # Reindex to calc dates and forward-fill (SI updates less than daily)
    piv_float = piv_float.reindex(dates_to_calc).ffill()
    piv_util  = piv_util.reindex(dates_to_calc).ffill()

    # Fill remaining NaNs with cross-sectional median per date
    # Stocks missing SI data get a neutral score (~0 after z-scoring)
    # rather than being dropped from the universe entirely
    piv_float = piv_float.apply(lambda row: row.fillna(row.median()), axis=1)
    piv_util  = piv_util.apply(lambda row: row.fillna(row.median()),  axis=1)

    # Cross-sectional z-score per date, equal-weight composite
    results = {}
    for dt in dates_to_calc:
        z_float = zscore(piv_float.loc[dt].dropna())                   if dt in piv_float.index else pd.Series(dtype=float)
        z_util  = zscore(piv_util.loc[dt].dropna())                   if dt in piv_util.index else pd.Series(dtype=float)

        common  = z_float.index.intersection(z_util.index)
        if len(common) < MIN_STOCKS:
            continue

        composite       = (z_float.loc[common] + z_util.loc[common]) / 2
        results[dt]     = composite

    df_out            = pd.DataFrame(results).T.reindex(columns=universe)
    df_out.index.name = 'date'
    return df_out


def load_si_composite(universe: list,
                       all_calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load SI composite from DB cache, computing only missing dates.
    Returns DataFrame: date x ticker.
    """
    already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": SI_COMPOSITE_TBL}).scalar()

        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT date FROM {SI_COMPOSITE_TBL}"
                )).fetchall()
            already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    dates_to_calc = [d for d in all_calc_dates if d not in already_done]

    if dates_to_calc:
        print(f"  Computing SI composite for {len(dates_to_calc)} new dates "
              f"({len(already_done)} already in DB)...")
        new_df = _compute_si_composite_for_dates(
            pd.DatetimeIndex(dates_to_calc), universe
        )
        long           = new_df.stack(dropna=False).reset_index()
        long.columns   = ['date', 'ticker', 'si_composite']
        long           = long.dropna(subset=['si_composite'])
        long['date']   = pd.to_datetime(long['date'])

        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {SI_COMPOSITE_TBL} (
                    date         DATE,
                    ticker       VARCHAR(20),
                    si_composite NUMERIC,
                    PRIMARY KEY  (date, ticker)
                )
            """))
        long.to_sql(SI_COMPOSITE_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} new rows to '{SI_COMPOSITE_TBL}'")
    else:
        print(f"  SI composite: all {len(all_calc_dates)} dates already in DB")

    # Load full table for requested dates
    date_list = [d.date() for d in all_calc_dates]
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, ticker, si_composite FROM {SI_COMPOSITE_TBL}
            WHERE date = ANY(:dates)
        """), {"dates": date_list}).fetchall()

    df             = pd.DataFrame(rows, columns=['date', 'ticker', 'si_composite'])
    df['date']     = pd.to_datetime(df['date'])
    df['si_composite'] = df['si_composite'].astype(float)
    pivot          = df.pivot_table(
        index='date', columns='ticker', values='si_composite', aggfunc='last'
    )
    pivot          = pivot.reindex(columns=universe)
    # Forward-fill missing dates (SI fetched less frequently than prices)
    pivot          = pivot.reindex(all_calc_dates).ffill()
    print(f"  SI composite loaded: {pivot.shape}")
    return pivot


# ==============================================================================
# O-U MEAN REVERSION — DB cached, computed on common sample dates
# ==============================================================================

def _fit_ou_single(resid_series: pd.Series,
                   px_series: pd.Series) -> tuple:
    """
    Fit AR(1)/O-U to compounded residual price index for one stock on one date.
    Returns (neg_dist_st, halflife) or (nan, nan) on failure.
    """
    from sklearn.linear_model import LinearRegression

    resid_clean = resid_series.replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
    if len(resid_clean) < OU_MIN_OBS:
        return np.nan, np.nan

    anchor_dates = px_series.index[px_series.index >= resid_clean.index[0]]
    if anchor_dates.empty:
        return np.nan, np.nan
    anchor_price = float(px_series.loc[anchor_dates[0]])
    if np.isnan(anchor_price) or anchor_price <= 0:
        return np.nan, np.nan

    px_idx = (1 + resid_clean).cumprod() * anchor_price
    sX1    = px_idx.iloc[:-1].values.reshape(-1, 1)
    sX2    = px_idx.iloc[1:].values

    try:
        mod = LinearRegression()
        mod.fit(sX1, sX2)
        a = float(mod.intercept_)
        b = float(mod.coef_[0])
    except Exception:
        return np.nan, np.nan

    if not (0 < b < 1):
        return np.nan, np.nan

    m = a / (1 - b)
    k = -np.log(b)

    residuals  = sX2 - mod.predict(sX1).flatten()
    resid_std  = float(np.std(residuals))
    if resid_std == 0 or k == 0:
        return np.nan, np.nan

    # Scale to actual price space
    last_px = float(px_series.dropna().iloc[-1])
    if np.isnan(last_px) or last_px <= 0:
        return np.nan, np.nan
    idx_last   = float(px_idx.iloc[-1])
    scale      = last_px / idx_last if idx_last != 0 else 1.0
    m_scaled   = m * scale
    std_scaled = resid_std * scale

    if m_scaled <= 0:
        return np.nan, np.nan

    dist_st  = (last_px - m_scaled) / (std_scaled / np.sqrt(2 * k))
    halflife = np.log(2) / k

    return -dist_st, halflife   # neg_dist_st, halflife


def _compute_ou_for_dates(calc_dates: pd.DatetimeIndex,
                           universe: list,
                           resid_pivot: pd.DataFrame,
                           Pxs_df: pd.DataFrame,
                           volumeTrd_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute O-U mean reversion z-scores for given dates.
    Returns DataFrame (dates x tickers), z-scored final_score.
    """
    # Volume-scale residuals if provided
    resid = resid_pivot.copy()
    if volumeTrd_df is not None:
        common = resid.columns.intersection(volumeTrd_df.columns)
        vol_norm  = (volumeTrd_df[common]
                     .rolling(OU_VOLUME_W).mean()
                     .reindex(resid.index).ffill())
        vol_ratio = (volumeTrd_df[common]
                     .reindex(resid.index).ffill()
                     / vol_norm).clip(OU_VOL_CLIP_LO, OU_VOL_CLIP_HI)
        resid[common] = resid[common] / vol_ratio

    result = {}
    n = len(calc_dates)

    # Pre-compute compounded residual price index for each stock
    # Used for both O-U fitting and ST reversal fallback
    # Anchor each stock to 1.0 at its first available residual date
    cum_resid = (1 + resid.fillna(0)).cumprod()

    for idx, dt in enumerate(calc_dates):
        if (idx + 1) % 50 == 0:
            print(f"  O-U: [{idx+1}/{n}] {dt.date()}", end='\r')

        # Get residual history up to this date
        past_resid = resid[resid.index < dt]
        past_cum   = cum_resid[cum_resid.index < dt]

        # ST reversal from cumulative residual index (fallback)
        # log(cum_resid[t-1] / cum_resid[t-22]) — idiosyncratic, net of all model factors
        st_rev = pd.Series(np.nan, index=universe)
        if len(past_cum) >= OU_ST_REV_W + 1:
            cum_recent = past_cum.iloc[-1]
            cum_old    = past_cum.iloc[-OU_ST_REV_W - 1]
            valid      = (cum_recent > 0) & (cum_old > 0)
            st_rev     = np.log(cum_recent / cum_old).where(valid).reindex(universe)

        neg_dists  = pd.Series(np.nan, index=universe)
        halflives  = pd.Series(np.nan, index=universe)

        for ticker in universe:
            if ticker not in past_resid.columns:
                continue
            stock_resid = past_resid[ticker].dropna().iloc[-OU_MEANREV_W:]
            if ticker not in Pxs_df.columns:
                continue
            neg_dist, hl = _fit_ou_single(stock_resid, Pxs_df[ticker].dropna())
            neg_dists[ticker] = neg_dist
            halflives[ticker] = hl

        # Cross-sectional ranks
        valid_neg = neg_dists.dropna()
        valid_rev = st_rev.dropna()

        ou_rank  = pd.Series(np.nan, index=universe)
        rev_rank = pd.Series(np.nan, index=universe)

        if len(valid_neg) > 1:
            r = valid_neg.rank(method='average', ascending=True)
            ou_rank[valid_neg.index] = (r - 1) / (len(r) - 1)

        if len(valid_rev) > 1:
            r = valid_rev.rank(method='average', ascending=False)
            rev_rank[valid_rev.index] = (r - 1) / (len(r) - 1)

        # ou_weight = min(OU_WEIGHT_REF / halflife, OU_WEIGHT_CAP), 0 if NaN
        ou_weight = (OU_WEIGHT_REF / halflives).clip(upper=OU_WEIGHT_CAP).fillna(0)

        # Blend O-U rank and ST reversal rank
        total_w = ou_weight + (1.0 - ou_weight.clip(upper=1.0))
        final   = (ou_weight * ou_rank + (1.0 - ou_weight.clip(upper=1.0)) * rev_rank)
        final   = final / total_w.where(total_w > 0)

        # Cross-sectional z-score
        valid = final.dropna()
        if len(valid) > 1:
            z = (valid - valid.mean()) / valid.std()
            final[valid.index] = z

        result[dt] = final

    out = pd.DataFrame(result).T
    out.index.name = 'date'
    return out.reindex(columns=universe)


def _compute_dynamic_size_for_dates(dates_to_calc: pd.DatetimeIndex,
                                     universe: list,
                                     Pxs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dynamic market cap for each date using price × shares."""
    us_tickers = [t + ' US' for t in universe]
    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, ticker, "Size" FROM valuation_consolidated
            WHERE "Size" IS NOT NULL AND ticker = ANY(:tickers)
            ORDER BY ticker, date
        """), {"tickers": us_tickers}).fetchall()
    size_raw           = pd.DataFrame(rows, columns=['date', 'ticker', 'Size'])
    size_raw['date']   = pd.to_datetime(size_raw['date'])
    size_raw['ticker'] = size_raw['ticker'].str.replace(' US', '', regex=False)
    size_pivot = size_raw.pivot_table(
        index='date', columns='ticker', values='Size', aggfunc='last'
    )
    all_px_dates   = Pxs_df.index
    size_ff        = size_pivot.reindex(all_px_dates).ffill().bfill()
    date_indicator = pd.DataFrame(
        index=size_pivot.index, columns=size_pivot.columns,
        data=np.tile(size_pivot.index.values.reshape(-1, 1),
                     (1, len(size_pivot.columns)))
    )
    date_indicator = date_indicator.reindex(all_px_dates).ffill().bfill()
    results = {}
    for dt in dates_to_calc:
        if dt not in Pxs_df.index:
            continue
        row = {}
        for ticker in universe:
            if ticker not in size_ff.columns:
                continue
            size_db = size_ff.loc[dt, ticker]
            if pd.isna(size_db):
                continue
            db_date  = pd.Timestamp(date_indicator.loc[dt, ticker])
            price_db = Pxs_df.loc[db_date, ticker] \
                       if db_date in Pxs_df.index else np.nan
            price_t  = Pxs_df.loc[dt, ticker]
            if pd.isna(price_db) or price_db == 0 or pd.isna(price_t):
                row[ticker] = size_db
            else:
                row[ticker] = (size_db / price_db) * price_t
        results[dt] = row
    df            = pd.DataFrame(results).T
    df.index.name = 'date'
    return df.reindex(columns=universe)


def _load_resid_from_db(table_name: str, universe: list,
                         last_n_dates: int = 300) -> pd.DataFrame:
    """Load the last N calendar days of a residual table from DB."""
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT date, ticker, resid FROM {table_name}
                WHERE date >= (SELECT MAX(date) - INTERVAL '{last_n_dates} days'
                               FROM {table_name})
                ORDER BY date
            """)).fetchall()
        df = pd.DataFrame(rows, columns=['date', 'ticker', 'resid'])
        df['date']  = pd.to_datetime(df['date'])
        df['resid'] = df['resid'].astype(float)
        return df.pivot_table(index='date', columns='ticker',
                               values='resid', aggfunc='last')
    except Exception as e:
        print(f"  WARNING: could not load {table_name} from DB: {e}")
        return pd.DataFrame()


def _load_char_from_db(table_name: str, universe: list,
                        last_n_dates: int = 300) -> pd.DataFrame:
    """Load last N calendar days of a wide characteristic table."""
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT date, ticker, score FROM {table_name}
                WHERE date >= (SELECT MAX(date) - INTERVAL '{last_n_dates} days'
                               FROM {table_name})
            """)).fetchall()
        df = pd.DataFrame(rows, columns=['date', 'ticker', 'score'])
        df['date'] = pd.to_datetime(df['date'])
        return df.pivot_table(index='date', columns='ticker',
                               values='score', aggfunc='last').reindex(
                               columns=universe)
    except Exception as e:
        print(f"  WARNING: could not load {table_name} from DB: {e}")
        return pd.DataFrame()


# ==============================================================================
# POINT-IN-TIME WEIGHT INTEGRATION
# These functions are called from within the factor model run sequence
# to update quality/value weights inline at the correct sequence points.
# ==============================================================================

def _get_anchor_dates_quality():
    with ENGINE.connect() as conn:
        rows = conn.execute(text(
            f"SELECT DISTINCT date FROM {QUALITY_ANCHOR_TBL} ORDER BY date"
        )).fetchall()
    return sorted([pd.Timestamp(r[0]) for r in rows])


def _get_anchor_dates_value():
    with ENGINE.connect() as conn:
        rows = conn.execute(text(
            f"SELECT DISTINCT date FROM valuation_consolidated ORDER BY date"
        )).fetchall()
    return sorted([pd.Timestamp(r[0]) for r in rows])


def _get_rate_signal_q(Pxs_df, dt):
    """Compute rate regime signal q at date dt.
    Uses '10Y RATE' column (in bps). Returns 0.0/0.5/1.0.
    """
    if QF_RATE_COL not in Pxs_df.columns:
        return 0.5
    rate = Pxs_df[QF_RATE_COL].dropna()
    rate = rate[rate.index <= dt]
    if len(rate) < QF_MAV_WINDOW // 2:
        return 0.5
    rate_mav = rate.iloc[-QF_MAV_WINDOW:].mean()
    rate_mom = float(rate.iloc[-1]) - float(rate_mav)
    if rate_mom >  QF_THRESHOLD:
        return 1.0
    elif rate_mom < -QF_THRESHOLD:
        return 0.0
    else:
        return 0.5


def _load_quality_pit_cache():
    """Load quality PIT weights. Returns dict {cutoff_date: {'gqf': {}, 'cqf': {}}}"""
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT cutoff_date, regime, metric, weight
                FROM {QUALITY_WEIGHTS_PIT_TBL} ORDER BY cutoff_date
            """)).fetchall()
        result = {}
        for cutoff, regime, metric, weight in rows:
            if regime == '_sentinel':
                continue
            dt = pd.Timestamp(cutoff)
            if dt not in result:
                result[dt] = {'gqf': {}, 'cqf': {}}
            key = 'gqf' if regime == 'growth' else 'cqf'
            result[dt][key][metric] = float(weight)
        return result
    except Exception:
        return {}


def _load_value_pit_cache():
    """Load value PIT weights. Returns dict {cutoff_date: {metric: weight}}"""
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT cutoff_date, metric, weight
                FROM {VALUE_WEIGHTS_PIT_TBL} ORDER BY cutoff_date
            """)).fetchall()
        result = {}
        for cutoff, metric, weight in rows:
            if metric == '_sentinel':
                continue
            dt = pd.Timestamp(cutoff)
            if dt not in result:
                result[dt] = {}
            result[dt][metric] = float(weight)
        return result
    except Exception:
        return {}


def _get_quality_weights_at(cutoff_date, pit_cache=None):
    """Return (gqf_w, cqf_w) from most recent prior cutoff <= cutoff_date."""
    if pit_cache is None:
        pit_cache = _load_quality_pit_cache()
    prior = [d for d in sorted(pit_cache.keys()) if d <= cutoff_date]
    if not prior:
        return {}, {}
    entry = pit_cache[prior[-1]]
    return entry.get('gqf', {}), entry.get('cqf', {})


def _get_value_weights_at(cutoff_date, pit_cache=None):
    """Return value weights from most recent prior cutoff <= cutoff_date."""
    if pit_cache is None:
        pit_cache = _load_value_pit_cache()
    prior = [d for d in sorted(pit_cache.keys()) if d <= cutoff_date]
    if not prior:
        return {m: 1.0/len(VALUE_METRICS) for m in VALUE_METRICS}
    return pit_cache[prior[-1]]


def _save_quality_pit_weights(cutoff_date, gqf_w, cqf_w):
    rows = []
    for m, w in gqf_w.items():
        rows.append({'cutoff_date': cutoff_date.date(), 'regime': 'growth',
                     'metric': m, 'weight': float(w)})
    for m, w in cqf_w.items():
        rows.append({'cutoff_date': cutoff_date.date(), 'regime': 'conservative',
                     'metric': m, 'weight': float(w)})
    if not rows:
        rows = [{'cutoff_date': cutoff_date.date(), 'regime': '_sentinel',
                 'metric': '_none', 'weight': 0.0}]
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {QUALITY_WEIGHTS_PIT_TBL}
                (cutoff_date, regime, metric, weight)
            VALUES (:cutoff_date, :regime, :metric, :weight)
            ON CONFLICT (cutoff_date, regime, metric) DO NOTHING
        """), rows)


def _save_value_pit_weights(cutoff_date, weights):
    rows = [{'cutoff_date': cutoff_date.date(), 'metric': m, 'weight': float(w)}
            for m, w in weights.items()]
    if not rows:
        rows = [{'cutoff_date': cutoff_date.date(), 'metric': '_sentinel',
                 'weight': 0.0}]
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {VALUE_WEIGHTS_PIT_TBL} (cutoff_date, metric, weight)
            VALUES (:cutoff_date, :metric, :weight)
            ON CONFLICT (cutoff_date, metric) DO NOTHING
        """), rows)


def _ensure_quality_pit_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {QUALITY_WEIGHTS_PIT_TBL} (
                cutoff_date DATE NOT NULL, regime VARCHAR(20) NOT NULL,
                metric VARCHAR(50) NOT NULL, weight FLOAT NOT NULL,
                PRIMARY KEY (cutoff_date, regime, metric)
            )
        """))


def _ensure_value_pit_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {VALUE_WEIGHTS_PIT_TBL} (
                cutoff_date DATE NOT NULL, metric VARCHAR(20) NOT NULL,
                weight FLOAT NOT NULL, PRIMARY KEY (cutoff_date, metric)
            )
        """))


def _ensure_value_ic_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {VALUE_IC_CACHE_TBL} (
                anchor_date DATE NOT NULL, metric VARCHAR(20) NOT NULL,
                horizon INTEGER NOT NULL, ic FLOAT NOT NULL,
                PRIMARY KEY (anchor_date, metric, horizon)
            )
        """))


def _load_quality_snapshot(anchor_date):
    """Load quality fundamental snapshot from valuation_metrics_anchors."""
    try:
        fetch_cols = list(dict.fromkeys(['Size'] + RAW_DB_COLS_QUALITY))
        cols_sql   = ', '.join([f'"{m}"' for m in fetch_cols])
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT ticker, {cols_sql}
                FROM {QUALITY_ANCHOR_TBL}
                WHERE date = :dt AND ticker IS NOT NULL
            """), conn, params={'dt': anchor_date.date()})
        if df.empty:
            return pd.DataFrame()
        df['ticker'] = df['ticker'].apply(_quality_normalize_ticker)
        snap = df.drop_duplicates('ticker').set_index('ticker')
        for col in fetch_cols:
            if col in snap.columns:
                snap[col] = pd.to_numeric(snap[col], errors='coerce')
        return snap
    except Exception as e:
        print(f"  WARNING: quality snapshot load failed ({e})")
        return pd.DataFrame()


def _compute_quality_ic(snap, resid_mkt, sectors_s, Pxs_df, anchor, horizon,
                         eligible_metrics, universe):
    """Compute IC for quality metrics at an anchor date.
    Matches derive_weights logic from quality_factor.py exactly:
    uses sector-ranked scores and top/bottom decile spread."""
    # Forward residuals from resid_mkt after anchor over horizon days
    fwd_dates = resid_mkt.index[resid_mkt.index > anchor]
    if len(fwd_dates) < horizon:
        return {}
    window_dates = fwd_dates[:horizon]
    fwd = (1 + resid_mkt.loc[window_dates]).prod(axis=0) - 1
    fwd = fwd.dropna()
    if len(fwd) < MIN_STOCKS:
        return {}

    # Sector-rank the snap
    snap_built = _build_derived_quality_metrics(snap)
    ranked     = _rank_within_sector_quality(snap_built, sectors_s, eligible_metrics)

    n_decile = max(1, int(np.floor(len(fwd) * QUALITY_TOP_PCTILE)))
    sorted_ret     = fwd.sort_values(ascending=False)
    top_tickers    = sorted_ret.iloc[:n_decile].index
    bottom_tickers = sorted_ret.iloc[-n_decile:].index

    ics = {}
    for m in eligible_metrics:
        if m not in ranked.columns:
            continue
        col = ranked[m].dropna()
        if col.empty:
            continue
        u_std = float(col.std())
        if u_std <= 0:
            continue
        top_med    = float(col.reindex(top_tickers).dropna().median())
        bottom_med = float(col.reindex(bottom_tickers).dropna().median())
        if np.isnan(top_med) or np.isnan(bottom_med):
            continue
        ics[m] = (top_med - bottom_med) / u_std
    return ics


def _derive_quality_weights_continuous(weighted_anchors, Pxs_df, resid_mkt,
                                        sectors_s, universe):
    """
    Derive quality metric weights using continuous regime blending.
    weighted_anchors: {anchor_date: (snap, regime_weight)}
    resid_mkt: DataFrame of market beta residuals
    """
    eligible = [m for m in QUALITY_METRICS_ALL if m not in QUALITY_EXCLUDE_METRICS]
    metric_stats_gqf = {m: [] for m in eligible}
    metric_stats_cqf = {m: [] for m in eligible}

    for anchor, (snap, reg_w) in weighted_anchors.items():
        if snap is None or snap.empty or reg_w <= 0:
            continue
        for horizon in QUALITY_HORIZONS:
            ics = _compute_quality_ic(snap, resid_mkt, sectors_s, Pxs_df,
                                       anchor, horizon, eligible, universe)
            for m, ic in ics.items():
                metric_stats_gqf[m].append(ic * (1 - reg_w))
                metric_stats_cqf[m].append(ic * reg_w)

    def _weights_from_stats(stats_dict):
        from scipy import stats as _st
        rows = []
        for m, vals in stats_dict.items():
            if len(vals) < 2:
                continue
            arr = np.array(vals)
            avg_sz = float(arr.mean())
            avg_t  = float(_st.ttest_1samp(arr, 0).statistic)
            rows.append({'metric': m, 'avg_sz': avg_sz, 'avg_t': avg_t})
        if not rows:
            return {}
        df = pd.DataFrame(rows).set_index('metric')
        med_sz = df['avg_sz'].median()
        med_t  = df['avg_t'].median()
        eligible_df = df[(df['avg_sz'] > 0) & (df['avg_sz'] > med_sz) &
                         (df['avg_t'] > med_t)].copy()
        if eligible_df.empty:
            return {}
        eligible_df = eligible_df.nlargest(QUALITY_MAX_COMPONENTS, 'avg_t')
        eligible_df['weight'] = eligible_df['avg_t'].clip(lower=0)
        total = eligible_df['weight'].sum()
        if total <= 0:
            return {}
        eligible_df['weight'] /= total
        return eligible_df['weight'].to_dict()

    gqf_w = _weights_from_stats(metric_stats_gqf)
    cqf_w = _weights_from_stats(metric_stats_cqf)
    if not cqf_w and gqf_w:
        cqf_w = gqf_w
    return gqf_w, cqf_w


def _update_quality_pit_weights(dt, resid_mkt, Pxs_df, sectors_s, universe,
                                  all_px_dates):
    """
    Called after Step 1 (mkt residuals). For each anchor date A where
    A + MAX_HORIZON < dt, derive quality PIT weights using resid_mkt.
    Only runs if dt is an anchor date or new cutoff dates need computing.
    """
    _ensure_quality_pit_table()
    anchor_dates = _get_anchor_dates_quality()
    if not anchor_dates:
        return

    # Find all cutoff dates not yet cached
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT cutoff_date FROM {QUALITY_WEIGHTS_PIT_TBL}
            """)).fetchall()
        cached_cutoffs = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        cached_cutoffs = set()

    # Build cutoff candidates
    new_cutoffs = []
    for anchor in anchor_dates:
        future = all_px_dates[all_px_dates > anchor]
        if len(future) < MAX_HORIZON_QUALITY:
            continue
        cutoff = future[MAX_HORIZON_QUALITY - 1]
        if cutoff not in cached_cutoffs and cutoff <= dt:
            new_cutoffs.append((anchor, cutoff))

    if not new_cutoffs:
        return

    print(f"\n  [Quality PIT] Computing weights for {len(new_cutoffs)} new cutoffs...")
    rate_signal = _compute_rate_signal_series(Pxs_df)

    for trigger_anchor, cutoff in new_cutoffs:
        # Eligible anchors: complete forward window before cutoff
        eligible_anchors = {}
        for a in anchor_dates:
            fut = all_px_dates[all_px_dates > a]
            if len(fut) < MAX_HORIZON_QUALITY:
                continue
            if fut[MAX_HORIZON_QUALITY - 1] < cutoff:
                snap = _load_quality_snapshot(a)
                if not snap.empty:
                    # Rate signal at anchor date
                    rate_dates = rate_signal.index[rate_signal.index <= a]
                    q = float(rate_signal.loc[rate_dates[-1]]) \
                        if not rate_dates.empty else 0.0
                    if (1 - q) > 0:
                        eligible_anchors[a] = (snap, q)
                    elif q > 0:
                        eligible_anchors[a] = (snap, q)

        if not eligible_anchors:
            _save_quality_pit_weights(cutoff, {}, {})
            continue

        gqf_w, cqf_w = _derive_quality_weights_continuous(
            eligible_anchors, Pxs_df, resid_mkt, sectors_s, universe
        )
        _save_quality_pit_weights(cutoff, gqf_w, cqf_w)
        print(f"    cutoff={cutoff.date()}: GQF={len(gqf_w)} CQF={len(cqf_w)}")


def _compute_rate_signal_series(Pxs_df):
    """Compute rate regime signal q for all dates.
    Uses '10Y RATE' column (in bps). q=0.0/0.5/1.0.
    """
    if QF_RATE_COL not in Pxs_df.columns:
        return pd.Series(0.5, index=Pxs_df.index)
    rate     = Pxs_df[QF_RATE_COL].dropna()
    rate_mav = rate.rolling(QF_MAV_WINDOW, min_periods=QF_MAV_WINDOW // 2).mean()
    rate_mom = rate - rate_mav
    q = pd.Series(0.5, index=rate_mom.index)
    q[rate_mom >  QF_THRESHOLD] = 1.0
    q[rate_mom < -QF_THRESHOLD] = 0.0
    return q


def _load_quality_scores_pit(universe, calc_dates, Pxs_df, sectors_s):
    """
    Load quality composite scores using PIT weights.
    For each calc_date, uses the most recent prior quality_weights_pit entry.
    """
    print("  Loading quality scores (PIT weights)...")
    _ensure_quality_pit_table()

    pit_cache   = _load_quality_pit_cache()
    anchor_dates = _get_anchor_dates_quality()
    if not anchor_dates:
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)

    rate_signal = _compute_rate_signal_series(Pxs_df)

    # Check which dates need computing
    try:
        with ENGINE.connect() as conn:
            cached = conn.execute(text(f"""
                SELECT DISTINCT date FROM {QUALITY_SCORES_TBL}
            """)).fetchall()
        cached_dates = {pd.Timestamp(r[0]) for r in cached}
    except Exception:
        cached_dates = set()

    dates_to_compute = [d for d in calc_dates if d not in cached_dates]
    print(f"  Quality scores: {len(cached_dates)} cached, "
          f"{len(dates_to_compute)} new dates to compute")

    if dates_to_compute:
        new_scores = {}
        for calc_date in dates_to_compute:
            candidates = [a for a in anchor_dates if a <= calc_date]
            if not candidates:
                continue
            anchor = candidates[-1]
            snap   = _load_quality_snapshot(anchor)
            if snap is None or snap.empty:
                continue
            snap   = _build_derived_quality_metrics(snap)

            # Get PIT weights for this date
            gqf_w, cqf_w = _get_quality_weights_at(calc_date, pit_cache)
            if not gqf_w:
                n = len([m for m in QUALITY_METRICS_ALL
                         if m not in QUALITY_EXCLUDE_METRICS])
                gqf_w = cqf_w = {m: 1.0/n for m in QUALITY_METRICS_ALL
                                  if m not in QUALITY_EXCLUDE_METRICS}
            elif not cqf_w:
                cqf_w = gqf_w

            # Rate signal at this date
            rate_dates = rate_signal.index[rate_signal.index <= calc_date]
            q = float(rate_signal.loc[rate_dates[-1]]) \
                if not rate_dates.empty else 0.0

            scores = _compute_quality_composite(snap, sectors_s, gqf_w, cqf_w, q,
                                                 universe)
            if not scores.empty:
                new_scores[calc_date] = scores

        if new_scores:
            _save_quality_scores(new_scores)

    # Load from DB
    result = _load_quality_scores_from_db(calc_dates, universe)
    print(f"  Quality scores: {result.notna().any(axis=1).sum()} dates "
          f"| {result.notna().any(axis=0).sum()} tickers")
    return result


def _quality_normalize_ticker(t):
    return str(t).split(' ')[0].strip().upper()


def _quality_winsorize(s):
    lo, hi = QUALITY_WINSOR
    return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))


def _build_derived_quality_metrics(snap):
    """Build derived metrics from raw snapshot. Mirrors quality_factor.build_derived_metrics."""
    s = snap.copy()

    def col(name):
        return s[name] if name in s.columns else pd.Series(np.nan, index=s.index)

    def safe_div(num, denom_name):
        return col(num) / col(denom_name).clip(lower=QUALITY_VOL_MIN)

    def safe_mul(base_name, r2_name):
        return col(base_name) * col(r2_name)

    s['GS/S_Vol']   = safe_div('GS',  'S Vol')
    s['HSG/S_Vol']  = safe_div('HSG', 'S Vol')
    s['PSG/S_Vol']  = safe_div('PSG', 'S Vol')
    s['GE/E_Vol']   = safe_div('GE',  'E Vol')
    s['PIG/E_Vol']  = safe_div('PIG', 'E Vol')
    s['GGP/GP_Vol'] = safe_div('GGP', 'GP Vol')
    s['GS*r2_S']    = safe_mul('GS',  'r2 S')
    s['SGD*r2_S']   = safe_mul('SGD', 'r2 S')
    s['OMd*r2_S']   = safe_mul('OMd', 'r2 S')
    s['GE*r2_E']    = safe_mul('GE',  'r2 E')
    s['PIG*r2_E']   = safe_mul('PIG', 'r2 E')
    s['GGP*r2_GP']  = safe_mul('GGP', 'r2 GP')
    return s


def _rank_within_sector_quality(snap, sectors_s, metrics):
    """Rank stocks within sector on each metric. Mirrors quality_factor.rank_within_sector."""
    ranked = pd.DataFrame(index=snap.index)
    # Normalize sectors_s index to bare tickers
    sec = sectors_s.copy()
    sec.index = [_quality_normalize_ticker(t) for t in sec.index]
    sec = sec.reindex(snap.index)

    for m in metrics:
        if m not in snap.columns:
            ranked[m] = np.nan
            continue
        col = snap[m].copy()
        out = pd.Series(np.nan, index=snap.index)
        for sector, grp_idx in sec.groupby(sec).groups.items():
            grp = col.reindex(grp_idx).dropna()
            if len(grp) < 3:
                continue
            grp_w = _quality_winsorize(grp)
            r     = grp_w.rank(method='average')
            out.loc[r.index] = (r - 1) / (len(r) - 1) if len(r) > 1 else 0.5
        ranked[m] = out
    return ranked


def _compute_quality_composite(snap, sectors_s, gqf_w, cqf_w, q, universe):
    """Compute composite quality score. Mirrors quality_factor.compute_composite_scores."""
    all_metrics = list(set(list(gqf_w.keys()) + list(cqf_w.keys())))
    ranked      = _rank_within_sector_quality(snap, sectors_s, all_metrics)

    def weighted_score(weights):
        if not weights:
            return pd.Series(np.nan, index=ranked.index)
        score = pd.Series(0.0, index=ranked.index)
        total = 0.0
        for m, w in weights.items():
            if m not in ranked.columns:
                continue
            score = score.add(ranked[m] * w, fill_value=0)
            total += w
        return score / total if total > 0 else pd.Series(np.nan, index=ranked.index)

    scores = ((1 - q) * weighted_score(gqf_w) +
               q      * weighted_score(cqf_w)).dropna()
    return scores.reindex(universe).dropna()


def _save_quality_scores(new_scores):
    """Save quality scores to quality_scores_df."""
    rows = []
    for dt, scores in new_scores.items():
        for ticker, score in scores.items():
            if pd.notna(score):
                rows.append({'date': dt.date(), 'ticker': ticker,
                             'score': float(score)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    dates = list({r['date'] for r in rows})
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            DELETE FROM {QUALITY_SCORES_TBL} WHERE date = ANY(:dates)
        """), {'dates': dates})
    df.to_sql(QUALITY_SCORES_TBL, ENGINE, if_exists='append', index=False)
    print(f"  Saved {len(df):,} quality score rows ({len(dates)} dates)")


def _load_quality_scores_from_db(calc_dates, universe):
    """Load quality scores from DB, forward-fill to calc_dates."""
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT date, ticker, score FROM {QUALITY_SCORES_TBL}
            """), conn)
        if df.empty:
            return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)
        df['date']   = pd.to_datetime(df['date'])
        df['ticker'] = df['ticker'].apply(clean_ticker)
        df['score']  = df['score'].astype(float)
        pivot = df.pivot_table(index='date', columns='ticker',
                                values='score', aggfunc='last')
        pivot = pivot.reindex(columns=universe)
        all_dates = calc_dates.union(pivot.index).sort_values()
        result = pivot.reindex(all_dates).ffill().reindex(calc_dates).astype(float)
        # Z-score cross-sectionally — matches v1 load_quality_scores_v2 exactly
        return result.apply(zscore, axis=1)
    except Exception as e:
        print(f"  WARNING: quality scores load failed ({e})")
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)


def _load_value_scores_pit(universe, calc_dates):
    """Load value scores from value_scores_df, forward-filled."""
    print("  Loading value scores (PIT weights)...")
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT date, ticker, score FROM {VALUE_SCORES_TBL}
            """), conn)
        if df.empty:
            return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)
        df['date']   = pd.to_datetime(df['date'])
        df['ticker'] = df['ticker'].apply(clean_ticker)
        df['score']  = df['score'].astype(float)
        pivot = df.pivot_table(index='date', columns='ticker',
                                values='score', aggfunc='last')
        pivot = pivot.reindex(columns=universe)
        all_dates = calc_dates.union(pivot.index).sort_values()
        val_ff = pivot.reindex(all_dates).ffill().reindex(calc_dates)
        # Z-score cross-sectionally — matches v1 load_value_scores_v2 exactly
        val_ff = val_ff.apply(zscore, axis=1)
        print(f"  Value scores: {val_ff.notna().any(axis=1).sum()} dates "
              f"| {val_ff.notna().any(axis=0).sum()} tickers")
        return val_ff.astype(float)
    except Exception as e:
        print(f"  WARNING: value scores load failed ({e})")
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)


def _update_value_pit_weights(dt, resid_size, Pxs_df, sectors_s, universe,
                               all_px_dates):
    """
    Called after Step 4 (size residuals). Updates value PIT weights for any
    new cutoff dates where resid_size now has complete forward windows.
    """
    _ensure_value_pit_table()
    _ensure_value_ic_table()

    anchor_dates = _get_anchor_dates_value()
    if not anchor_dates:
        return

    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT cutoff_date FROM {VALUE_WEIGHTS_PIT_TBL}
            """)).fetchall()
        cached_cutoffs = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        cached_cutoffs = set()

    new_cutoffs = []
    for anchor in anchor_dates:
        future = all_px_dates[all_px_dates > anchor]
        if len(future) < MAX_HORIZON_VALUE:
            continue
        cutoff = future[MAX_HORIZON_VALUE - 1]
        if cutoff not in cached_cutoffs and cutoff <= dt:
            new_cutoffs.append((anchor, cutoff))

    if not new_cutoffs:
        # Always refresh IC for last anchor
        _refresh_value_ic_last_anchor(resid_size, anchor_dates, all_px_dates)
        return

    print(f"\n  [Value PIT] Computing weights for {len(new_cutoffs)} new cutoffs...")

    # Load IC bank
    try:
        with ENGINE.connect() as conn:
            ic_rows = conn.execute(text(f"""
                SELECT anchor_date, metric, horizon, ic FROM {VALUE_IC_CACHE_TBL}
            """)).fetchall()
        ic_bank = {(pd.Timestamp(r[0]), r[1], r[2]): float(r[3]) for r in ic_rows}
        cached_ic = {pd.Timestamp(r[0]) for r in ic_rows}
    except Exception:
        ic_bank = {}
        cached_ic = set()

    # Compute missing ICs
    all_eligible = set()
    for _, cutoff in new_cutoffs:
        for anchor in anchor_dates:
            fut = all_px_dates[all_px_dates > anchor]
            if len(fut) >= MAX_HORIZON_VALUE and fut[MAX_HORIZON_VALUE-1] < cutoff:
                all_eligible.add(anchor)

    missing = sorted(all_eligible - cached_ic)
    new_ic_rows = []
    for j, anchor in enumerate(missing, 1):
        if j % 10 == 0:
            print(f"    [{j}/{len(missing)}] IC anchor={anchor.date()}", flush=True)
        snap = _load_value_snapshot(anchor)
        if snap.empty:
            continue
        for horizon in VALUE_HORIZONS:
            fwd = _compute_value_fwd_resid(resid_size, anchor, horizon,
                                            all_px_dates)
            if fwd.empty:
                continue
            for m in VALUE_METRICS:
                ic = _compute_value_ic_for_metric(snap, fwd, sectors_s, m)
                if not np.isnan(ic):
                    ic_bank[(anchor, m, horizon)] = ic
                    new_ic_rows.append({'anchor_date': anchor.date(),
                                        'metric': m, 'horizon': horizon,
                                        'ic': float(ic)})
    if new_ic_rows:
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {VALUE_IC_CACHE_TBL}
                    (anchor_date, metric, horizon, ic)
                VALUES (:anchor_date, :metric, :horizon, :ic)
                ON CONFLICT DO NOTHING
            """), new_ic_rows)

    # Derive weights per cutoff
    for trigger_anchor, cutoff in new_cutoffs:
        eligible = [a for a in anchor_dates
                    if len(all_px_dates[all_px_dates > a]) >= MAX_HORIZON_VALUE
                    and all_px_dates[all_px_dates > a][MAX_HORIZON_VALUE-1] < cutoff]
        metric_ics = {m: [] for m in VALUE_METRICS}
        for anchor in eligible:
            for m in VALUE_METRICS:
                for h in VALUE_HORIZONS:
                    ic = ic_bank.get((anchor, m, h))
                    if ic is not None:
                        metric_ics[m].append(ic)

        # ── Derive weights from current IC evidence ───────────────────────────
        current_weights = {}
        for m in VALUE_METRICS:
            ics = metric_ics[m]
            if len(ics) < 2:
                continue
            arr = np.array(ics)
            t_stat = float(scipy_stats.ttest_1samp(arr, 0).statistic)
            if t_stat > 0:
                current_weights[m] = t_stat

        # Normalize current weights
        if current_weights:
            total = sum(current_weights.values())
            current_weights = {m: w/total for m, w in current_weights.items()}

        # ── Warm-start blending if fewer than 2 metrics pass ─────────────────
        MIN_METRICS = 2
        if len(current_weights) < MIN_METRICS:
            # Find most recent prior cutoff with >= MIN_METRICS metrics
            prior_weights = None
            cached_pit = _load_value_pit_cache()
            for prior_cutoff in sorted(cached_pit.keys(), reverse=True):
                if prior_cutoff >= cutoff:
                    continue
                pw = cached_pit[prior_cutoff]
                # Exclude sentinel entries
                pw = {m: w for m, w in pw.items() if m != '_sentinel'}
                if len(pw) >= MIN_METRICS:
                    prior_weights = pw
                    break

            if prior_weights is None:
                # No prior stable period — use equal weights
                weights = {m: 1.0/len(VALUE_METRICS) for m in VALUE_METRICS}
                print(f"    cutoff={cutoff.date()}: {len(current_weights)} metrics — "
                      f"no prior stable period, using equal weights")
            elif len(current_weights) == 0:
                # Zero current metrics — forward-fill prior weights entirely
                weights = prior_weights.copy()
                print(f"    cutoff={cutoff.date()}: 0 metrics — "
                      f"using prior stable weights ({len(weights)} metrics)")
            else:
                # 1 current metric — 60% to current, 40% distributed from prior
                # excluding any metric already in current
                surviving = list(current_weights.keys())[0]
                imported  = {m: w for m, w in prior_weights.items()
                             if m != surviving}
                if not imported:
                    # Prior had same single metric — just use current
                    weights = current_weights.copy()
                else:
                    total_imported = sum(imported.values())
                    weights = {surviving: 0.60}
                    for m, w in imported.items():
                        weights[m] = 0.40 * (w / total_imported)
                print(f"    cutoff={cutoff.date()}: 1 current metric ({surviving}) "
                      f"+ {len(imported)} imported — warm-start blend applied")
        else:
            weights = current_weights
        _save_value_pit_weights(cutoff, weights)
        print(f"    cutoff={cutoff.date()}: {len(weights)} value metrics weighted")

    # Refresh last anchor IC
    _refresh_value_ic_last_anchor(resid_size, anchor_dates, all_px_dates)


def _recompute_value_scores(sectors_s):
    """
    Recompute and save value composite scores to value_scores_df
    using current PIT weights from value_weights_pit.
    Called after _update_value_pit_weights in full recalculation.
    """
    print("  Recomputing value scores with PIT weights...")
    pit_cache = _load_value_pit_cache()
    pit_dates = sorted(pit_cache.keys())

    # Get all valuation dates
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT date FROM valuation_consolidated ORDER BY date"
            )).fetchall()
        val_dates = [pd.Timestamp(r[0]) for r in rows]
    except Exception as e:
        print(f"  WARNING: could not get valuation dates: {e}")
        return

    if not val_dates:
        print("  WARNING: no valuation dates found")
        return

    # Load raw valuation data
    try:
        with ENGINE.connect() as conn:
            df_all = pd.read_sql(text(f"""
                SELECT date, ticker, {', '.join(f'"{m}"' for m in VALUE_METRICS)}
                FROM valuation_consolidated
                WHERE ticker IS NOT NULL
                ORDER BY date
            """), conn)
    except Exception as e:
        print(f"  WARNING: could not load valuation data: {e}")
        return

    df_all['date']   = pd.to_datetime(df_all['date'])
    df_all['ticker'] = df_all['ticker'].apply(
        lambda t: t.replace(' US', '') if isinstance(t, str) else t)

    # Normalize sectors_s index
    sec_s = sectors_s.copy()
    sec_s.index = sec_s.index.str.replace(' US', '')

    rows_to_save = []
    n_dates = 0

    for val_date, grp in df_all.groupby('date'):
        snap = grp.drop(columns='date').drop_duplicates('ticker').set_index('ticker')
        for m in VALUE_METRICS:
            if m in snap.columns:
                snap[m] = pd.to_numeric(snap[m], errors='coerce')
        snap['_sector'] = snap.index.map(sec_s)
        snap = snap[snap['_sector'].notna()]
        if snap.empty:
            continue

        # Get PIT weights for this date
        prior = [d for d in pit_dates if d <= val_date]
        if prior:
            w = pit_cache[prior[-1]]
            w = {m: v for m, v in w.items() if m != '_sentinel'}
        else:
            w = {m: 1.0/len(VALUE_METRICS) for m in VALUE_METRICS}

        if not w:
            w = {m: 1.0/len(VALUE_METRICS) for m in VALUE_METRICS}

        # Sector-rank each metric
        metric_scores = {}
        for m in VALUE_METRICS:
            if m not in snap.columns:
                continue
            scores = pd.Series(np.nan, index=snap.index)
            for sec, sg in snap.groupby('_sector'):
                vals = sg[m].dropna()
                if len(vals) < 3:
                    continue
                pos = vals > 0
                if pos.any():
                    adj = vals.copy()
                    adj[~pos] = vals[pos].max() + vals[~pos].abs()
                else:
                    adj = vals.abs().max() - vals
                ranks = adj.rank(method='average', ascending=True)
                normed = (1.0 - (ranks - 1) / (len(ranks) - 1)) \
                         if len(ranks) > 1 else pd.Series(0.5, index=ranks.index)
                scores.loc[normed.index] = normed
            metric_scores[m] = scores

        composite = pd.Series(0.0, index=snap.index)
        total_w = 0.0
        for m, wt in w.items():
            if m in metric_scores:
                s = metric_scores[m].reindex(snap.index)
                valid = s.notna()
                composite[valid] += wt * s[valid]
                total_w += wt
        if total_w > 0:
            composite /= total_w

        for ticker, score in composite.items():
            if pd.notna(score):
                rows_to_save.append({'date': pd.Timestamp(val_date).date(),
                                     'ticker': ticker, 'score': float(score)})
        n_dates += 1

    if not rows_to_save:
        print("  WARNING: no value scores computed")
        return

    df_save    = pd.DataFrame(rows_to_save)
    saved_dates = list({r['date'] for r in rows_to_save})
    with ENGINE.begin() as conn:
        conn.execute(text(
            f"DELETE FROM {VALUE_SCORES_TBL} WHERE date = ANY(:dates)"
        ), {'dates': saved_dates})
    df_save.to_sql(VALUE_SCORES_TBL, ENGINE, if_exists='append', index=False)
    print(f"  Saved {len(df_save):,} value score rows ({n_dates} dates)")


def _refresh_value_ic_last_anchor(resid_size, anchor_dates, all_px_dates):
    """Refresh IC for the last anchor date using latest resid_size."""
    if not anchor_dates:
        return
    last_anchor = anchor_dates[-1]
    snap = _load_value_snapshot(last_anchor)
    if snap.empty:
        return
    new_ic_rows = []
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            DELETE FROM {VALUE_IC_CACHE_TBL} WHERE anchor_date = :dt
        """), {'dt': last_anchor.date()})
    for horizon in VALUE_HORIZONS:
        fwd = _compute_value_fwd_resid(resid_size, last_anchor, horizon,
                                        all_px_dates)
        if fwd.empty:
            continue
        for m in VALUE_METRICS:
            ic = _compute_value_ic_for_metric(snap, fwd, pd.Series(), m)
            if not np.isnan(ic):
                new_ic_rows.append({'anchor_date': last_anchor.date(),
                                    'metric': m, 'horizon': horizon,
                                    'ic': float(ic)})
    if new_ic_rows:
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {VALUE_IC_CACHE_TBL}
                    (anchor_date, metric, horizon, ic)
                VALUES (:anchor_date, :metric, :horizon, :ic)
                ON CONFLICT DO NOTHING
            """), new_ic_rows)


def _load_value_snapshot(anchor_date):
    """Load value fundamental snapshot."""
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT ticker, {', '.join(f'"{m}"' for m in VALUE_METRICS)}
                FROM valuation_consolidated WHERE date = :dt
            """), conn, params={'dt': anchor_date.date()})
        if df.empty:
            return pd.DataFrame()
        df['ticker'] = df['ticker'].apply(clean_ticker)
        return df.set_index('ticker')
    except Exception:
        return pd.DataFrame()


def _compute_value_fwd_resid(resid_size, anchor, horizon, all_px_dates):
    """Compound resid_size forward returns over horizon days after anchor."""
    fwd_dates = resid_size.index[resid_size.index > anchor]
    if len(fwd_dates) < horizon:
        return pd.Series(dtype=float)
    window = fwd_dates[:horizon]
    compounded = (1 + resid_size.loc[window]).prod(axis=0) - 1
    return compounded.dropna()


def _compute_value_ic_for_metric(snap, fwd_resid, sectors_s, metric):
    """Compute Spearman IC between value metric ranking and forward residuals."""
    if metric not in snap.columns:
        return np.nan
    scores = pd.to_numeric(snap[metric], errors='coerce').dropna()
    common = scores.index.intersection(fwd_resid.index)
    if len(common) < 20:
        return np.nan
    ic, _ = scipy_stats.spearmanr(scores.reindex(common),
                                    fwd_resid.reindex(common))
    return float(ic) if not np.isnan(ic) else np.nan

def load_ou_reversion_v2(universe, calc_dates, resid_pivot,
                          Pxs_df, volumeTrd_df=None):
    """Load O-U scores from v2_ou_reversion_df cache."""
    already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": V2_OU_TBL}).scalar()
        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT date FROM {V2_OU_TBL}"
                )).fetchall()
            already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    dates_to_calc = pd.DatetimeIndex(
        [d for d in calc_dates if d not in already_done]
    )

    if len(dates_to_calc) > 0:
        print(f"  Computing v2 O-U for {len(dates_to_calc)} new dates "
              f"({len(already_done)} already in cache)...")
        new_df = _compute_ou_for_dates(dates_to_calc, universe,
                                        resid_pivot, Pxs_df, volumeTrd_df)
        if new_df is None or new_df.empty:
            print("  WARNING: O-U computation returned no results")
        else:
            long         = new_df.stack(dropna=False).reset_index()
            long.columns = ['date', 'ticker', 'ou_score']
            long         = long.dropna(subset=['ou_score'])
            long['date'] = pd.to_datetime(long['date'])
            with ENGINE.begin() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {V2_OU_TBL} (
                        date DATE, ticker VARCHAR(20), ou_score NUMERIC,
                        PRIMARY KEY (date, ticker)
                    )
                """))
            long.to_sql(V2_OU_TBL, ENGINE, if_exists='append', index=False)
            print(f"  Saved {len(long):,} rows to '{V2_OU_TBL}'")
    else:
        print(f"  v2 O-U: all {len(calc_dates)} dates already cached")

    date_list = [d.date() for d in calc_dates]
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, ticker, ou_score FROM {V2_OU_TBL}
            WHERE date = ANY(:dates)
        """), {"dates": date_list}).fetchall()

    df           = pd.DataFrame(rows, columns=['date', 'ticker', 'ou_score'])
    df['date']   = pd.to_datetime(df['date'])
    df['ou_score'] = df['ou_score'].astype(float)
    pivot        = df.pivot_table(index='date', columns='ticker',
                                   values='ou_score', aggfunc='last')
    pivot        = pivot.reindex(columns=universe).reindex(calc_dates)
    print(f"  v2 O-U loaded: {pivot.shape}")
    return pivot


def _v2_get_anchor_date():
    """Latest date in v2_factor_residuals_mkt."""
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(
                text(f"SELECT MAX(date) FROM {V2_RESID_MKT}")
            ).fetchone()
        if row and row[0]:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


# ===============================================================================
# INCREMENTAL UPDATE — v2 fast single-date path
# ===============================================================================

def _v2_run_incremental(Pxs_df, sectors_s, volumeTrd_df=None,
                         use_vol_scale=False,
                         VOL_LOWER=0.5, VOL_UPPER=3.0):
    """
    Fast single-date incremental update for v2 model.
    Mirrors _run_incremental from v1 but follows the v2 step sequence
    and writes to v2_* tables.
    """
    dt         = Pxs_df.index[-1]
    calc_dates = pd.DatetimeIndex([dt])
    print(f"\n  v2 incremental update for {dt.date()}")

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    st_dt          = FM_START_DATE   # respects global start date setting
    all_dates      = Pxs_df.index
    ext_loc        = max(0, all_dates.searchsorted(st_dt) - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]

    universe    = get_universe(Pxs_df, sectors_s, extended_st_dt)
    sector_dum  = build_sector_dummies(universe, sectors_s)
    sec_cols    = sector_dum.columns.tolist()
    all_rets    = Pxs_df[universe].pct_change().clip(-0.5, 0.5)

    dynamic_size = load_dynamic_size(universe, Pxs_df, calc_dates)

    print("  Computing characteristics for new date...")

    # Beta
    beta_df = calc_rolling_betas(Pxs_df, universe, calc_dates)

    # Size
    s = dynamic_size.loc[dt, universe].dropna() if dt in dynamic_size.index         else pd.Series()
    size_char_df = pd.DataFrame(
        {dt: zscore(np.log(s.clip(lower=1)))}
    ).T.reindex(columns=universe)
    size_char_df.index.name = 'date'

    # Quality and Value
    ext_dates = Pxs_df.index[Pxs_df.index >= extended_st_dt]
    valid_ext = ext_dates[
        all_rets.loc[ext_dates].notna().sum(axis=1) >= MIN_STOCKS
    ]
    quality_df = _load_quality_scores_pit(universe, valid_ext, Pxs_df, sectors_s)
    value_df   = _load_value_scores_pit(universe, valid_ext)

    # SI composite (load recent window for ffill)
    si_dates         = pd.DatetimeIndex(
        [d for d in valid_ext if d <= dt]
    )[-60:]
    si_composite_full = load_si_composite(universe, si_dates)
    si_composite      = si_composite_full.reindex([dt])

    # GK Vol
    open_df, high_df, low_df = load_ohlc_tables(universe)
    vol_df = calc_vol_factor(Pxs_df, universe, calc_dates,
                             open_df=open_df, high_df=high_df, low_df=low_df)

    # Macro betas (computed against resid_vol later — for incremental,
    # compute against current step input which we approximate with raw rets
    # for the single-date fast path; full recalc uses proper residuals)
    macro_betas = calc_macro_betas(Pxs_df, universe, calc_dates)
    macro_cols  = list(macro_betas.keys())

    # Sector dummies
    sec_char = {col: pd.DataFrame({dt: sector_dum[col]}).T
                for col in sec_cols}

    # ── Step 1: Beta ──────────────────────────────────────────────────────────
    r_mkt, lam_mkt, _ = run_factor_step(
        ['beta'], {'beta': beta_df},
        all_rets, dynamic_size, calc_dates, universe
    )

    # ── Step 2: Quality ⊥ {beta} ─────────────────────────────────────────────
    quality_perp = orthogonalize_char_df(
        quality_df, {'beta': beta_df}, calc_dates, dynamic_size=dynamic_size
    )
    r_quality, lam_quality, _ = run_factor_step(
        ['quality'], {'quality': quality_perp},
        r_mkt, dynamic_size, calc_dates, universe
    )

    # ── Step 4: Idio Momentum ⊥ {beta, quality} ──────────────────────────────
    # Load quality residual history for momentum lookback
    qual_hist = _load_resid_from_db(V2_RESID_QUALITY, universe, 400)
    if not qual_hist.empty and not r_quality.empty:
        qual_hist = pd.concat([
            qual_hist[~qual_hist.index.isin(r_quality.index)], r_quality
        ]).sort_index()
    elif not r_quality.empty:
        qual_hist = r_quality

    if use_vol_scale and volumeTrd_df is not None:
        mom_df = calc_idio_momentum_volscaled(
            qual_hist, volumeTrd_df, calc_dates,
            vol_lower=VOL_LOWER, vol_upper=VOL_UPPER
        )
    else:
        mom_df = calc_idio_momentum(qual_hist, calc_dates)

    if mom_df.empty or dt not in mom_df.index:
        print("  WARNING: momentum not available — aborting")
        return None

    mom_perp = orthogonalize_char_df(
        mom_df, {'beta': beta_df, 'quality': quality_perp},
        calc_dates, dynamic_size=dynamic_size
    )
    r_mom, lam_mom, _ = run_factor_step(
        ['idio_mom'], {'idio_mom': mom_perp},
        r_quality, dynamic_size, calc_dates, universe
    )

    # ── Step 4: Size ⊥ {beta, quality, mom} ──────────────────────────────────
    size_perp = orthogonalize_char_df(
        size_char_df,
        {'beta': beta_df, 'quality': quality_perp, 'idio_mom': mom_perp},
        calc_dates, dynamic_size=dynamic_size
    )
    r_size, lam_size, _ = run_factor_step(
        ['size'], {'size': size_perp},
        r_mom, dynamic_size, calc_dates, universe
    )

    # ── Step 5: Value ⊥ {beta, quality, mom, size} ───────────────────────────
    value_perp = orthogonalize_char_df(
        value_df,
        {'beta': beta_df, 'quality': quality_perp,
         'idio_mom': mom_perp, 'size': size_perp},
        calc_dates, dynamic_size=dynamic_size
    )
    r_value, lam_value, _ = run_factor_step(
        ['value'], {'value': value_perp},
        r_size, dynamic_size, calc_dates, universe
    )

    # ── Step 7: SI ⊥ {all prior} ─────────────────────────────────────────────
    prior_for_si = {
        'beta': beta_df, 'quality': quality_perp,
        'idio_mom': mom_perp, 'size': size_perp, 'value': value_perp
    }
    si_perp = orthogonalize_char_df(
        si_composite, prior_for_si, calc_dates, dynamic_size=dynamic_size
    )
    r_si, lam_si, _ = run_factor_step(
        ['si_composite'], {'si_composite': si_perp},
        r_value, dynamic_size, calc_dates, universe
    )

    # ── Step 8: GK Vol ⊥ {all prior} ─────────────────────────────────────────
    prior_for_vol = dict(prior_for_si)
    prior_for_vol['si_composite'] = si_perp
    vol_perp = orthogonalize_char_df(
        vol_df, prior_for_vol, calc_dates, dynamic_size=dynamic_size
    )
    r_vol, lam_vol, _ = run_factor_step(
        ['vol'], {'vol': vol_perp},
        r_si, dynamic_size, calc_dates, universe
    )

    # ── Step 9: Macro — raw betas to vol residuals, joint Ridge ───────────────
    # For incremental, macro betas are computed against Pxs_df (approximation)
    # Full recalc computes them against the actual vol residuals
    if macro_cols:
        r_macro, lam_macro, _ = run_factor_step_optimal_ridge(
            macro_cols, macro_betas,
            r_vol, dynamic_size, calc_dates, universe,
            lambda_grid=RIDGE_GRID_MACRO, default_lambda=0.5
        )
    else:
        r_macro  = r_vol
        lam_macro = pd.DataFrame()

    # ── Step 10: Sectors — sum-to-zero dummies, Ridge ─────────────────────────
    r_sec, lam_sec, _ = run_factor_step_optimal_ridge(
        sec_cols, {c: sec_char[c] for c in sec_cols},
        r_macro, dynamic_size, calc_dates, universe,
        lambda_grid=RIDGE_GRID_SEC, default_lambda=2.0
    )

    # ── Step 11: O-U on sector residuals ─────────────────────────────────────
    sec_hist = _load_resid_from_db(V2_RESID_SEC, universe, 120)
    if not sec_hist.empty and not r_sec.empty:
        sec_hist = pd.concat([
            sec_hist[~sec_hist.index.isin(r_sec.index)], r_sec
        ]).sort_index()
    elif not r_sec.empty:
        sec_hist = r_sec

    if r_sec.empty or dt not in sec_hist.index:
        print("  WARNING: sector residuals empty — O-U skipped")
        ou_pivot  = pd.DataFrame(index=calc_dates, columns=universe, dtype=float)
        r_ou      = pd.DataFrame()
        lam_ou    = pd.DataFrame()
    else:
        new_ou       = _compute_ou_for_dates(
            calc_dates, universe, sec_hist, Pxs_df,
            volumeTrd_df if use_vol_scale else None
        )
        long         = new_ou.stack(dropna=False).reset_index()
        long.columns = ['date', 'ticker', 'ou_score']
        long         = long.dropna(subset=['ou_score'])
        long['date'] = pd.to_datetime(long['date'])
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {V2_OU_TBL} (
                    date DATE, ticker VARCHAR(20), ou_score NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
            conn.execute(text(
                f"DELETE FROM {V2_OU_TBL} WHERE date = :d"
            ), {"d": dt.date()})
        long.to_sql(V2_OU_TBL, ENGINE, if_exists='append', index=False)
        ou_pivot = new_ou.reindex(columns=universe)
        r_ou, lam_ou, _ = run_factor_step(
            ['ou_reversion'], {'ou_reversion': ou_pivot},
            r_sec, dynamic_size, calc_dates, universe
        )

    # ── Save to DB ────────────────────────────────────────────────────────────
    print("  Saving v2 results to DB...")
    lam_pairs = [
        (lam_mkt,     V2_LAM_MKT),
        (lam_quality, V2_LAM_QUALITY),
        (lam_mom,     V2_LAM_MOM),
        (lam_size,    V2_LAM_SIZE),
        (lam_value,   V2_LAM_VALUE),
        (lam_si,      V2_LAM_SI),
        (lam_vol,     V2_LAM_VOL),
        (lam_macro,   V2_LAM_MACRO) if macro_cols else (None, None),
        (lam_sec,     V2_LAM_SEC),
        (lam_ou,      V2_LAM_OU),
    ]
    for ldf, tbl in lam_pairs:
        if ldf is not None and tbl is not None:
            save_lambdas_incremental(ldf, tbl)

    resid_pairs = [
        (r_mkt,     V2_RESID_MKT),
        (r_quality, V2_RESID_QUALITY),
        (r_mom,     V2_RESID_MOM),
        (r_size,    V2_RESID_SIZE),
        (r_value,   V2_RESID_VALUE),
        (r_si,      V2_RESID_SI),
        (r_vol,     V2_RESID_VOL),
        (r_macro,   V2_RESID_MACRO) if macro_cols else (None, None),
        (r_sec,     V2_RESID_SEC),
        (r_ou,      V2_RESID_OU),
    ]
    for rdf, tbl in resid_pairs:
        if rdf is not None and tbl is not None and not rdf.empty:
            save_residuals_incremental(rdf, tbl)

    # ── Snapshot print ────────────────────────────────────────────────────────
    def _val(ldf, col):
        if ldf is not None and not ldf.empty                 and dt in ldf.index and col in ldf.columns:
            return ldf.loc[dt, col] * 100
        return np.nan

    rows_snap = [
        ('Market Beta',   _val(lam_mkt,     'beta')),
        ('Quality',       _val(lam_quality, 'quality')),
        ('Idio Momentum', _val(lam_mom,     'idio_mom')),
        ('Size',          _val(lam_size,    'size')),
        ('Value',         _val(lam_value,   'value')),
        ('SI Composite',  _val(lam_si,      'si_composite')),
        ('GK Vol',        _val(lam_vol,     'vol')),
        ('O-U Reversion', _val(lam_ou,      'ou_reversion')),
    ]
    for c in macro_cols:
        rows_snap.append((f'Macro: {c}', _val(lam_macro, c)))
    if not lam_sec.empty and dt in lam_sec.index:
        for c in sec_cols:
            if c in lam_sec.columns:
                rows_snap.append((f'Sector: {c}', lam_sec.loc[dt, c] * 100))

    ridge_str = ''
    for ldf, label in [(lam_macro, 'Macro'), (lam_sec, 'Sec')]:
        if ldf is not None and not ldf.empty                 and 'ridge_lambda' in ldf.columns and dt in ldf.index:
            rl = ldf.loc[dt, 'ridge_lambda']
            if not np.isnan(rl):
                ridge_str += f'   {label} Ridge λ: {rl:.2f}'

    # ── Consolidated intercept ───────────────────────────────────────────────
    intercept_val = sum(
        ldf.loc[dt, 'intercept']
        if (ldf is not None and not ldf.empty
            and dt in ldf.index and 'intercept' in ldf.columns)
        else 0.0
        for ldf in [lam_mkt, lam_quality, lam_mom, lam_size,
                    lam_value, lam_si, lam_vol, lam_macro,
                    lam_sec, lam_ou]
    )

    # ── Daily R² ─────────────────────────────────────────────────────────────
    r2_consolidated = np.nan
    if not r_ou.empty and dt in r_ou.index and dt in all_rets.index:
        ou_res = r_ou.loc[dt].dropna()
        raw_r  = all_rets.loc[dt, ou_res.index].dropna()
        idx    = ou_res.index.intersection(raw_r.index)
        if len(idx) > 1:
            r2_consolidated = 1.0 - ou_res[idx].var() / raw_r[idx].var()

    r2_str = f"{r2_consolidated*100:.2f}%" if not np.isnan(r2_consolidated) else "n/a"

    W = 38
    print(f"\n  {'='*(W+14)}")
    print(f"  [v2] {dt.date()}   Intercept: {intercept_val*100:+.2f}%   Daily R²: {r2_str}{ridge_str}")
    print(f"  {'='*(W+14)}")
    print(f"  {'Factor':<{W}} {'Return%':>8}")
    print(f"  {'-'*(W+10)}")
    for factor, val in rows_snap:
        val_str = f"{val:>+8.2f}%" if not np.isnan(val) else f"{'n/a':>9}"
        print(f"  {factor:<{W}} {val_str}")
    print(f"  {'='*(W+14)}")

    return {
        'dt': dt, 'universe': universe,
        'sec_cols': sec_cols, 'macro_cols': macro_cols,
        'lambda_mkt': lam_mkt, 'lambda_quality': lam_quality,
        'lambda_mom': lam_mom, 'lambda_size': lam_size,
        'lambda_value': lam_value, 'lambda_si': lam_si,
        'lambda_vol': lam_vol, 'lambda_macro': lam_macro,
        'lambda_sec': lam_sec, 'lambda_ou': lam_ou,
        'resid_ou': r_ou, 'ou_pivot': ou_pivot,
    }


# ===============================================================================
# FULL RECALCULATION
# ===============================================================================

def _v2_run_full(Pxs_df, sectors_s, st_dt, volumeTrd_df=None,
                  use_vol_scale=False, VOL_LOWER=0.5, VOL_UPPER=3.0):
    """Full recalculation for v2 model."""

    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(st_dt)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    print(f"  Extended start: {extended_st_dt.date()}")

    universe   = get_universe(Pxs_df, sectors_s, extended_st_dt)
    sector_dum = build_sector_dummies(universe, sectors_s)
    sec_cols   = sector_dum.columns.tolist()

    all_rets = Pxs_df[universe].pct_change()
    RETURN_CLIP = 0.50
    n_clipped = int((all_rets.abs() > RETURN_CLIP).sum().sum())
    if n_clipped > 0:
        print(f"  Winsorizing {n_clipped} extreme returns (|ret|>{RETURN_CLIP:.0%})")
    all_rets = all_rets.clip(-RETURN_CLIP, RETURN_CLIP)

    ext_dates  = all_dates[all_dates >= extended_st_dt]

    # Count stocks with genuine (non-stale) returns on each date
    # Stale: price unchanged for >3 consecutive days = backfilled/pre-IPO data
    def _non_stale_count(dt):
        window = all_dates[all_dates <= dt][-(_STALE_RUN_LIMIT + 2):]
        if len(window) < 2:
            return 0
        rets = all_rets.loc[window]
        # A stock is stale on dt if last 3+ returns are all zero
        last_rets = rets.iloc[-_STALE_RUN_LIMIT:]
        stale     = (last_rets == 0).all(axis=0) | last_rets.iloc[-1].isna()
        return int((~stale).sum())

    valid_ext_mask = pd.Series(
        [_non_stale_count(dt) >= MIN_STOCKS for dt in ext_dates],
        index=ext_dates
    )
    valid_ext  = ext_dates[valid_ext_mask]
    valid_days = valid_ext[valid_ext >= st_dt]
    print(f"  Extended dates: {len(valid_ext)} | Valid dates: {len(valid_days)}")
    if len(valid_days) > 0:
        print(f"  First valid date: {valid_days[0].date()}")

    dynamic_size = load_dynamic_size(universe, Pxs_df, valid_ext)
    si_composite = load_si_composite(universe, valid_ext)

    # ── Build characteristics ─────────────────────────────────────────────────
    print("\n  Building characteristics...")

    beta_df     = calc_rolling_betas(Pxs_df, universe, valid_ext)
    print(f"  beta_df: {len(beta_df)} dates, first={beta_df.index[0].date()}, last={beta_df.index[-1].date()}")
    print(f"  beta_df non-null count on first date: {beta_df.iloc[0].notna().sum()}")
    print(f"  dates_ext_common will be: valid_ext ∩ beta_df ∩ size_char_df")
    macro_betas = calc_macro_betas(Pxs_df, universe, valid_ext)
    macro_cols  = list(macro_betas.keys())

    # Size
    print("  Building size characteristic...")
    size_char_dict = {}
    for dt in valid_ext:
        if dt not in dynamic_size.index:
            continue
        s = dynamic_size.loc[dt, universe].dropna()
        s = np.log(s.clip(lower=1))
        if len(s) < MIN_STOCKS:
            continue
        size_char_dict[dt] = zscore(s)
    size_char_df            = pd.DataFrame(size_char_dict).T.reindex(columns=universe)
    size_char_df.index.name = 'date'

    quality_df = _load_quality_scores_pit(universe, valid_ext, Pxs_df, sectors_s)
    value_df   = _load_value_scores_pit(universe, valid_ext)

    open_df, high_df, low_df = load_ohlc_tables(universe)
    vol_df = calc_vol_factor(Pxs_df, universe, valid_ext,
                             open_df=open_df, high_df=high_df, low_df=low_df)

    # Sector dummies expanded to date index
    dates_ext_common = valid_ext.intersection(beta_df.index).intersection(
        size_char_df.index
    )
    print(f"  dates_ext_common: {len(dates_ext_common)} dates, first={dates_ext_common[0].date() if len(dates_ext_common) else 'EMPTY'}")
    sec_char = {col: pd.DataFrame(
        {dt: sector_dum[col] for dt in dates_ext_common}
    ).T for col in sec_cols}

    # ── Common sample ─────────────────────────────────────────────────────────
    print(f"\n  valid_days range: {valid_days[0].date()} → {valid_days[-1].date()} ({len(valid_days)} dates)")
    print(f"  beta_df range   : {beta_df.index[0].date()} → {beta_df.index[-1].date()} ({len(beta_df)} dates)")
    print(f"  size_char_df    : {size_char_df.index[0].date()} → {size_char_df.index[-1].date()} ({len(size_char_df)} dates)")
    print(f"  vol_df          : {vol_df.index[0].date()} → {vol_df.index[-1].date()} ({len(vol_df)} dates)")
    print(f"  si_composite    : {si_composite.index[0].date()} → {si_composite.index[-1].date()} ({len(si_composite)} dates)")
    for col in macro_cols:
        idx = macro_betas[col].index
        print(f"  macro[{col:<20}]: {idx[0].date()} → {idx[-1].date()} ({len(idx)} dates)")

    common_dates = valid_days.intersection(beta_df.index).intersection(
        size_char_df.index
    )
    print(f"\n  Common sample construction (from {len(valid_days)} valid_days):")
    for label, idx in [
        ('vol_df',       vol_df.index),
        ('si_composite', si_composite.index),
    ]:
        before = len(common_dates)
        common_dates = common_dates.intersection(idx)
        if len(common_dates) < before:
            print(f"    after ∩ {label}: {len(common_dates)} "
                  f"(-{before-len(common_dates)})")
    # quality_df and value_df are sparse factors — excluded from common_dates
    # intersection so pre-Sep 2018 dates (no quality data) still compute correctly
    for col in macro_cols:
        before = len(common_dates)
        common_dates = common_dates.intersection(macro_betas[col].index)
        if len(common_dates) < before:
            print(f"    after ∩ macro[{col}]: {len(common_dates)} "
                  f"(-{before-len(common_dates)})")
    print(f"\n  Common sample: {len(common_dates)} dates")

    raw_mat = all_rets.loc[common_dates, universe]
    UFV     = variance_stats(raw_mat, "UFV - Raw Returns (common sample)")

    # ── Step 1: Market Beta ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 1: Market Beta")
    print("="*70)
    resid_mkt_full, lambda_mkt, r2_mkt = run_factor_step(
        ['beta'], {'beta': beta_df},
        all_rets, dynamic_size, dates_ext_common, universe
    )
    print(f"  resid_mkt_full: {len(resid_mkt_full)} dates, first={resid_mkt_full.index[0].date() if len(resid_mkt_full) else 'EMPTY'}")

    # ── PIT Quality Weights Update (after Step 1) ─────────────────────────────
    print("\n  Updating quality PIT weights using mkt residuals...")
    _update_quality_pit_weights(
        all_dates[-1], resid_mkt_full, Pxs_df, sectors_s, universe, all_dates
    )
    quality_df = _load_quality_scores_pit(universe, valid_ext, Pxs_df, sectors_s)

    # ── Step 2: Quality ⊥ {beta} ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 2: Quality")
    print("="*70)
    quality_perp = orthogonalize_char_df(
        quality_df, {'beta': beta_df},
        resid_mkt_full.index, dynamic_size=dynamic_size
    )
    resid_quality_step, lambda_quality, r2_quality = run_factor_step(
        ['quality'], {'quality': quality_perp},
        resid_mkt_full, dynamic_size,
        resid_mkt_full.index, universe
    )
    # For dates where quality step was skipped (no quality data),
    # fall back to resid_mkt_full so the chain continues unbroken
    missing_quality = resid_mkt_full.index.difference(resid_quality_step.index)
    if len(missing_quality) > 0:
        print(f"  Quality step skipped for {len(missing_quality)} dates "
              f"(no quality data) — using mkt residuals as fallback")
        resid_quality_full = pd.concat([
            resid_mkt_full.loc[missing_quality],
            resid_quality_step
        ]).sort_index()
    else:
        resid_quality_full = resid_quality_step

    # ── Step 4: Idio Momentum ⊥ {beta, quality} ──────────────────────────────
    print("\n" + "="*70)
    print("  STEP 4: Idio Momentum (on quality residuals)")
    print("="*70)
    if use_vol_scale and volumeTrd_df is not None:
        mom_df = calc_idio_momentum_volscaled(
            resid_quality_full, volumeTrd_df, valid_ext,
            vol_lower=VOL_LOWER, vol_upper=VOL_UPPER
        )
    else:
        mom_df = calc_idio_momentum(resid_quality_full, valid_ext)

    print("  Orthogonalizing momentum vs beta + quality...")
    mom_perp_dates = common_dates.intersection(mom_df.index)
    before_mom     = len(common_dates)
    common_dates   = mom_perp_dates
    print(f"  Common sample after idio_mom: {len(common_dates)} dates "
          f"(dropped {before_mom - len(common_dates)})")

    mom_perp = orthogonalize_char_df(
        mom_df,
        {'beta': beta_df, 'quality': quality_perp},
        common_dates, dynamic_size=dynamic_size
    )
    resid_mom_full, lambda_mom, r2_mom = run_factor_step(
        ['idio_mom'], {'idio_mom': mom_perp},
        resid_quality_full, dynamic_size,
        resid_quality_full.index.intersection(mom_df.index), universe
    )

    # ── Step 4: Size ⊥ {beta, quality, mom} ──────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 4: Size")
    print("="*70)
    print("  Orthogonalizing size vs beta + quality + momentum...")
    size_perp = orthogonalize_char_df(
        size_char_df,
        {'beta': beta_df, 'quality': quality_perp, 'idio_mom': mom_perp},
        resid_mom_full.index, dynamic_size=dynamic_size
    )
    resid_size_full, lambda_size, r2_size = run_factor_step(
        ['size'], {'size': size_perp},
        resid_mom_full, dynamic_size,
        resid_mom_full.index, universe
    )

    # ── PIT Value Weights Update (after Step 4) ───────────────────────────────
    print("\n  Updating value PIT weights using size residuals...")
    _update_value_pit_weights(
        all_dates[-1], resid_size_full, Pxs_df, sectors_s, universe, all_dates
    )
    # Recompute value scores using updated PIT weights
    _recompute_value_scores(sectors_s)
    value_df = _load_value_scores_pit(universe, valid_ext)

    # ── Step 5: Value ⊥ {beta, quality, mom, size} ───────────────────────────
    print("\n" + "="*70)
    print("  STEP 5: Value")
    print("="*70)
    print("  Orthogonalizing value vs beta + quality + momentum + size...")
    value_perp = orthogonalize_char_df(
        value_df,
        {'beta': beta_df, 'quality': quality_perp,
         'idio_mom': mom_perp, 'size': size_perp},
        resid_size_full.index, dynamic_size=dynamic_size
    )
    resid_value_step, lambda_value, r2_value = run_factor_step(
        ['value'], {'value': value_perp},
        resid_size_full, dynamic_size,
        resid_size_full.index, universe
    )
    # For dates without value data, fall back to size residuals
    missing_value = resid_size_full.index.difference(resid_value_step.index)
    if len(missing_value) > 0:
        print(f"  Value step skipped for {len(missing_value)} dates "
              f"(no value data) — using size residuals as fallback")
        resid_value_full = pd.concat([
            resid_size_full.loc[missing_value],
            resid_value_step
        ]).sort_index()
    else:
        resid_value_full = resid_value_step

    # ── Step 7: SI ⊥ {all prior} ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 7: SI Composite")
    print("="*70)
    prior_for_si = {
        'beta': beta_df, 'quality': quality_perp,
        'idio_mom': mom_perp, 'size': size_perp, 'value': value_perp
    }
    print("  Orthogonalizing SI vs all prior...")
    si_perp_full = orthogonalize_char_df(
        si_composite, prior_for_si,
        resid_value_full.index, dynamic_size=dynamic_size
    )
    si_perp = si_perp_full.reindex(common_dates)
    resid_si_full, lambda_si, r2_si = run_factor_step(
        ['si_composite'], {'si_composite': si_perp},
        resid_value_full, dynamic_size,
        resid_value_full.index.intersection(si_perp_full.index[
            si_perp_full.notna().any(axis=1)
        ]), universe
    )

    # ── Step 8: GK Vol ⊥ {all prior} ─────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 8: GK Volatility")
    print("="*70)
    prior_for_vol = dict(prior_for_si)
    prior_for_vol['si_composite'] = si_perp_full
    print("  Orthogonalizing GK vol vs all prior...")
    vol_perp_full = orthogonalize_char_df(
        vol_df, prior_for_vol,
        resid_si_full.index, dynamic_size=dynamic_size
    )
    vol_perp = vol_perp_full.reindex(common_dates)
    resid_vol_full, lambda_vol, r2_vol = run_factor_step(
        ['vol'], {'vol': vol_perp},
        resid_si_full, dynamic_size,
        resid_si_full.index.intersection(vol_perp_full.index[
            vol_perp_full.notna().any(axis=1)
        ]), universe
    )

    # ── Step 9: Macro — raw betas to vol residuals, joint Ridge ───────────────
    print("\n" + "="*70)
    print("  STEP 9: Macro Factors (raw betas to vol residuals, joint Ridge)")
    print("="*70)
    # Recompute macro betas against vol residuals (true v2 design)
    print("  Computing macro betas against vol residuals...")
    macro_betas_v2 = calc_macro_betas(
        Pxs_df, universe,
        resid_vol_full.index
    )
    # Note: calc_macro_betas uses Pxs_df macro columns as factors but
    # stock returns are approximated by Pxs_df returns. For the full
    # Gram-Schmidt spirit, macro betas should be vs the vol residuals.
    # We achieve this by passing resid_vol as the return series implicitly
    # via the run_factor_step call — the regression target is resid_vol_full.
    macro_cols_v2 = list(macro_betas_v2.keys())

    if macro_cols_v2:
        full_dates_macro = resid_vol_full.index
        for col in macro_cols_v2:
            full_dates_macro = full_dates_macro.intersection(
                macro_betas_v2[col].index
            )
        resid_macro_full, lambda_macro, r2_macro = run_factor_step_optimal_ridge(
            macro_cols_v2, macro_betas_v2,
            resid_vol_full, dynamic_size,
            full_dates_macro, universe,
            lambda_grid=RIDGE_GRID_MACRO, default_lambda=0.5
        )
    else:
        resid_macro_full = resid_vol_full
        lambda_macro     = pd.DataFrame()
        r2_macro         = pd.Series(dtype=float)
        macro_cols_v2    = []

    # ── Step 10: Sectors — sum-to-zero dummies, Ridge ─────────────────────────
    print("\n" + "="*70)
    print("  STEP 10: Sector Dummies (sum-to-zero, Ridge)")
    print("="*70)
    resid_sec_full, lambda_sec, r2_sec = run_factor_step_optimal_ridge(
        sec_cols, {c: sec_char[c] for c in sec_cols},
        resid_macro_full, dynamic_size,
        resid_macro_full.index, universe,
        lambda_grid=RIDGE_GRID_SEC, default_lambda=2.0
    )

    # ── Step 11: O-U on sector residuals ─────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 11: O-U Mean Reversion (on sector residuals)")
    print("="*70)

    ou_already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": V2_OU_TBL}).scalar()
        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT date FROM {V2_OU_TBL}"
                )).fetchall()
            ou_already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    ou_dates_to_calc = pd.DatetimeIndex(
        [d for d in common_dates if d not in ou_already_done]
    )
    print(f"  O-U cached: {len(ou_already_done)} | to compute: "
          f"{len(ou_dates_to_calc)}")

    if len(ou_dates_to_calc) > 0:
        BATCH_SIZE = 200
        batches = [ou_dates_to_calc[i:i+BATCH_SIZE]
                   for i in range(0, len(ou_dates_to_calc), BATCH_SIZE)]
        print(f"  Computing in {len(batches)} batches of {BATCH_SIZE}...")

        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {V2_OU_TBL} (
                    date DATE, ticker VARCHAR(20), ou_score NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))

        total_saved = 0
        for b_idx, batch in enumerate(batches):
            new_ou = _compute_ou_for_dates(
                batch, universe, resid_sec_full, Pxs_df,
                volumeTrd_df if use_vol_scale else None
            )
            if new_ou is None or new_ou.empty:
                print(f"\n  WARNING: batch {b_idx+1} returned no results")
                continue
            long         = new_ou.stack(dropna=False).reset_index()
            long.columns = ['date', 'ticker', 'ou_score']
            long         = long.dropna(subset=['ou_score'])
            long['date'] = pd.to_datetime(long['date'])
            long.to_sql(V2_OU_TBL, ENGINE, if_exists='append', index=False)
            total_saved += len(long)
            print(f"  Batch {b_idx+1}/{len(batches)} saved "
                  f"({len(batch)} dates, {len(long):,} rows)", flush=True)

        print(f"  O-U total saved: {total_saved:,} rows")

    date_list = [d.date() for d in common_dates]
    try:
        with ENGINE.connect() as conn:
            ou_rows = conn.execute(text(f"""
                SELECT date, ticker, ou_score FROM {V2_OU_TBL}
                WHERE date = ANY(:dates)
            """), {"dates": date_list}).fetchall()
    except Exception:
        ou_rows = []

    if ou_rows:
        ou_df             = pd.DataFrame(ou_rows,
                                         columns=['date','ticker','ou_score'])
        ou_df['date']     = pd.to_datetime(ou_df['date'])
        ou_df['ou_score'] = ou_df['ou_score'].astype(float)
        ou_pivot = ou_df.pivot_table(
            index='date', columns='ticker',
            values='ou_score', aggfunc='last'
        ).reindex(columns=universe).reindex(common_dates)
    else:
        ou_pivot = pd.DataFrame(
            index=common_dates, columns=universe, dtype=float
        )

    ou_common = common_dates.intersection(
        ou_pivot.index[ou_pivot.notna().any(axis=1)]
    )
    resid_ou, lambda_ou, r2_ou = run_factor_step(
        ['ou_reversion'], {'ou_reversion': ou_pivot},
        resid_sec_full, dynamic_size,
        ou_common, universe
    )

    # ── Restrict early residuals to common sample for variance stats ──────────
    def _cs(df):
        return df[df.index.isin(common_dates)] if not df.empty else df

    resid_mkt     = _cs(resid_mkt_full)
    resid_quality = _cs(resid_quality_full)
    resid_mom     = _cs(resid_mom_full)
    resid_size    = _cs(resid_size_full)
    resid_value   = _cs(resid_value_full)
    resid_si      = _cs(resid_si_full)
    resid_vol     = _cs(resid_vol_full)
    resid_macro   = _cs(resid_macro_full) if macro_cols_v2 else resid_vol
    resid_sec     = _cs(resid_sec_full)

    # ── Variance reduction summary ────────────────────────────────────────────
    print("\n" + "="*70)
    print("  VARIANCE REDUCTION SUMMARY (v2, common sample)")
    print("="*70)
    mkt_UFV     = variance_stats(resid_mkt,     "mkt_UFV     (+beta)",     UFV)
    quality_UFV = variance_stats(resid_quality, "quality_UFV (+quality)",  mkt_UFV)
    mom_UFV     = variance_stats(resid_mom,     "mom_UFV     (+idio_mom)", quality_UFV)
    size_UFV    = variance_stats(resid_size,    "size_UFV    (+size)",     mom_UFV)
    value_UFV   = variance_stats(resid_value,   "value_UFV   (+value)",    size_UFV)
    si_UFV      = variance_stats(resid_si,      "si_UFV      (+SI)",       value_UFV)
    vol_UFV     = variance_stats(resid_vol,     "vol_UFV     (+GK_vol)",   si_UFV)
    macro_UFV   = variance_stats(resid_macro,   "macro_UFV   (+macro)",    vol_UFV)                   if macro_cols_v2 else vol_UFV
    sec_UFV     = variance_stats(resid_sec,     "sec_UFV     (+sectors)",  macro_UFV)
    ou_UFV      = variance_stats(resid_ou,      "ou_UFV      (+O-U)",      sec_UFV)                   if not resid_ou.empty else sec_UFV

    print(f"\n  {'Step':<44} {'%UFV':>8} {'%prev':>8}")
    print(f"  {'-'*62}")
    for lbl, var, base, prev in [
        ("UFV (raw)",            UFV,         UFV,         None),
        ("+ Beta",               mkt_UFV,     UFV,         UFV),
        ("+ Quality",            quality_UFV, UFV,         mkt_UFV),
        ("+ Idio Momentum",      mom_UFV,     UFV,         quality_UFV),
        ("+ Size",               size_UFV,    UFV,         mom_UFV),
        ("+ Value",              value_UFV,   UFV,         size_UFV),
        ("+ SI",                 si_UFV,      UFV,         value_UFV),
        ("+ GK Vol",             vol_UFV,     UFV,         si_UFV),
        ("+ Macro",              macro_UFV,   UFV,         vol_UFV),
        ("+ Sectors",            sec_UFV,     UFV,         macro_UFV),
        ("+ O-U",                ou_UFV,      UFV,         sec_UFV),
    ]:
        pct_ufv  = f"{var/base*100:.2f}%"
        pct_prev = f"{var/prev*100:.2f}%" if prev else "---"
        print(f"  {lbl:<44} {pct_ufv:>8} {pct_prev:>8}")

    # ── Save to DB ────────────────────────────────────────────────────────────
    print("\n  Saving v2 results to DB...")
    for ldf, tbl in [
        (lambda_mkt[lambda_mkt.index.isin(common_dates)],         V2_LAM_MKT),
        (lambda_quality[lambda_quality.index.isin(common_dates)], V2_LAM_QUALITY),
        (lambda_mom[lambda_mom.index.isin(common_dates)],         V2_LAM_MOM),
        (lambda_size[lambda_size.index.isin(common_dates)],       V2_LAM_SIZE),
        (lambda_value[lambda_value.index.isin(common_dates)],     V2_LAM_VALUE),
        (lambda_si[lambda_si.index.isin(common_dates)],           V2_LAM_SI),
        (lambda_vol[lambda_vol.index.isin(common_dates)],         V2_LAM_VOL),
        (lambda_sec[lambda_sec.index.isin(common_dates)],         V2_LAM_SEC),
        (lambda_ou[lambda_ou.index.isin(ou_common)],              V2_LAM_OU),
    ]:
        save_lambdas(ldf, tbl)
    if macro_cols_v2:
        save_lambdas(
            lambda_macro[lambda_macro.index.isin(common_dates)], V2_LAM_MACRO
        )

    for rdf, tbl in [
        (resid_mkt_full,     V2_RESID_MKT),
        (resid_quality_full, V2_RESID_QUALITY),
        (resid_mom_full,     V2_RESID_MOM),
        (resid_size_full,    V2_RESID_SIZE),
        (resid_value_full,   V2_RESID_VALUE),
        (resid_si_full,      V2_RESID_SI),
        (resid_vol_full,     V2_RESID_VOL),
        (resid_sec_full,     V2_RESID_SEC),
        (resid_ou,           V2_RESID_OU),
    ]:
        save_residuals(rdf, tbl)
    if macro_cols_v2:
        save_residuals(resid_macro_full, V2_RESID_MACRO)

    r2_stats(r2_mkt[r2_mkt.index.isin(common_dates)],             "Step 2: Beta")
    r2_stats(r2_quality[r2_quality.index.isin(common_dates)],     "Step 3: Quality")
    r2_stats(r2_mom[r2_mom.index.isin(common_dates)],             "Step 4: Idio Mom")
    r2_stats(r2_size[r2_size.index.isin(common_dates)],           "Step 5: Size")
    r2_stats(r2_value[r2_value.index.isin(common_dates)],         "Step 6: Value")
    r2_stats(r2_si,                                                "Step 7: SI")
    r2_stats(r2_vol,                                               "Step 8: GK Vol")
    if macro_cols_v2:
        r2_stats(r2_macro[r2_macro.index.isin(common_dates)],     "Step 9: Macro")
    r2_stats(r2_sec[r2_sec.index.isin(common_dates)],             "Step 10: Sectors")
    r2_stats(r2_ou,                                                "Step 11: O-U")

    print_lambda_summary(lambda_mkt,     ['beta'],       "Step 2: Beta",        common_dates)
    print_lambda_summary(lambda_quality, ['quality'],    "Step 3: Quality",     common_dates, annual_col='quality')
    print_lambda_summary(lambda_mom,     ['idio_mom'],   "Step 4: Idio Mom",    common_dates, annual_col='idio_mom')
    print_lambda_summary(lambda_size,    ['size'],       "Step 5: Size",        common_dates, annual_col='size')
    print_lambda_summary(lambda_value,   ['value'],      "Step 6: Value",       common_dates, annual_col='value')
    print_lambda_summary(lambda_si,      ['si_composite'],"Step 7: SI",         common_dates, annual_col='si_composite')
    print_lambda_summary(lambda_vol,     ['vol'],        "Step 8: GK Vol",      common_dates, annual_col='vol')
    if macro_cols_v2:
        print_lambda_summary(lambda_macro, macro_cols_v2, "Step 9: Macro",      common_dates)
    print_lambda_summary(lambda_sec,     sec_cols,       "Step 10: Sectors",    common_dates)
    print_sector_lambdas(lambda_sec, sec_cols, common_dates)
    print_lambda_summary(lambda_ou,      ['ou_reversion'],"Step 11: O-U",       ou_common, annual_col='ou_reversion')

    return {
        'UFV': UFV, 'mkt_UFV': mkt_UFV, 'quality_UFV': quality_UFV,
        'mom_UFV': mom_UFV, 'size_UFV': size_UFV, 'value_UFV': value_UFV,
        'si_UFV': si_UFV, 'vol_UFV': vol_UFV, 'macro_UFV': macro_UFV,
        'sec_UFV': sec_UFV, 'ou_UFV': ou_UFV,
        'resid_mkt': resid_mkt, 'resid_quality': resid_quality,
        'resid_mom': resid_mom, 'resid_size': resid_size,
        'resid_value': resid_value, 'resid_si': resid_si,
        'resid_vol': resid_vol, 'resid_macro': resid_macro,
        'resid_sec': resid_sec, 'resid_ou': resid_ou,
        'resid_mkt_full': resid_mkt_full,
        'resid_quality_full': resid_quality_full,
        'resid_mom_full': resid_mom_full,
        'resid_size_full': resid_size_full,
        'resid_value_full': resid_value_full,
        'resid_si_full': resid_si_full,
        'resid_vol_full': resid_vol_full,
        'resid_macro_full': resid_macro_full if macro_cols_v2 else resid_vol_full,
        'resid_sec_full': resid_sec_full,
        'lambda_mkt': lambda_mkt, 'lambda_quality': lambda_quality,
        'lambda_mom': lambda_mom, 'lambda_size': lambda_size,
        'lambda_value': lambda_value, 'lambda_si': lambda_si,
        'lambda_vol': lambda_vol, 'lambda_macro': lambda_macro,
        'lambda_sec': lambda_sec, 'lambda_ou': lambda_ou,
        'beta_df': beta_df, 'size_char_df': size_char_df,
        'quality_perp': quality_perp, 'mom_perp': mom_perp,
        'size_perp': size_perp, 'value_perp': value_perp,
        'si_perp': si_perp, 'vol_perp': vol_perp,
        'macro_betas': macro_betas_v2, 'macro_cols': macro_cols_v2,
        'dynamic_size': dynamic_size, 'si_composite': si_composite,
        'quality_df': quality_df, 'value_df': value_df,
        'vol_df': vol_df, 'ou_pivot': ou_pivot,
        'universe': universe, 'sec_cols': sec_cols,
        'common_dates': common_dates, 'ou_common': ou_common,
        'st_dt': st_dt, 'extended_st_dt': extended_st_dt,
    }


# ===============================================================================
# ENTRY POINT
# ===============================================================================

def run(Pxs_df, sectors_s, volumeTrd_df=None, force_rebuild_pit=False):
    print("=" * 70)
    print("  FACTOR MODEL v2")
    print("  Sequence: Beta → Quality → Idio Mom → Size → Value → SI → "
          "GK Vol → Macro → Sectors → O-U")
    print("=" * 70)

    if force_rebuild_pit:
        print("  force_rebuild_pit=True — wiping quality and value PIT weight caches...")
        _ensure_quality_pit_table()
        _ensure_value_pit_table()
        _ensure_value_ic_table()
        with ENGINE.begin() as conn:
            conn.execute(text(f"DELETE FROM {QUALITY_WEIGHTS_PIT_TBL}"))
            conn.execute(text(f"DELETE FROM {VALUE_WEIGHTS_PIT_TBL}"))
            conn.execute(text(f"DELETE FROM {VALUE_IC_CACHE_TBL}"))
        print("  PIT caches cleared.")

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    update_input = input(
        "\n  Incremental update? (y/n) [default=y]: "
    ).strip().lower()
    incremental = update_input != 'n'
    print(f"  Mode: {'INCREMENTAL UPDATE' if incremental else 'FULL RECALCULATION'}")

    if incremental and _v2_get_anchor_date() is None:
        print("  No existing v2 data — switching to full recalculation")
        incremental = False

    if incremental:
        vol_input     = input(
            "  Volume-scaled momentum? (y/n) [default=n]: "
        ).strip().lower()
        use_vol_scale = vol_input == 'y'
        VOL_LOWER, VOL_UPPER = 0.5, 3.0
        if use_vol_scale:
            lo = input("    Vol scalar lower bound [default=0.5]: ").strip()
            hi = input("    Vol scalar upper bound [default=3.0]: ").strip()
            VOL_LOWER = float(lo) if lo else 0.5
            VOL_UPPER = float(hi) if hi else 3.0
        return _v2_run_incremental(
            Pxs_df, sectors_s, volumeTrd_df,
            use_vol_scale=use_vol_scale,
            VOL_LOWER=VOL_LOWER, VOL_UPPER=VOL_UPPER
        )

    # Full recalculation
    st_input = input(
        f"\n  Start date (YYYY-MM-DD, or Enter for {FM_START_DATE.date()}): "
    ).strip()
    st_dt    = pd.Timestamp(st_input) if st_input else FM_START_DATE

    vol_input     = input(
        "  Volume-scaled momentum? (y/n) [default=n]: "
    ).strip().lower()
    use_vol_scale = vol_input == 'y'
    VOL_LOWER, VOL_UPPER = 0.5, 3.0
    if use_vol_scale:
        lo = input("    Vol scalar lower bound [default=0.5]: ").strip()
        hi = input("    Vol scalar upper bound [default=3.0]: ").strip()
        VOL_LOWER = float(lo) if lo else 0.5
        VOL_UPPER = float(hi) if hi else 3.0

    print(f"  Start date: {st_dt.date()}")
    print(f"  Ridge λ grid macro: {RIDGE_GRID_MACRO}")
    print(f"  Ridge λ grid sec  : {RIDGE_GRID_SEC}")

    return _v2_run_full(
        Pxs_df, sectors_s, st_dt, volumeTrd_df,
        use_vol_scale=use_vol_scale,
        VOL_LOWER=VOL_LOWER, VOL_UPPER=VOL_UPPER
    )

