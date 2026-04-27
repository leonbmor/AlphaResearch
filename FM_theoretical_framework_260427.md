# Factor Model & MVO Pipeline — Comprehensive Reference
*Last updated: April 2026 (v6)*

---

## 1. OVERVIEW

A sequential Fama-MacBeth cross-sectional factor model implemented in Python, running in a Jupyter notebook kernel. The model strips systematic return sources one by one in a true Gram-Schmidt orthogonalization sequence, producing clean residuals at each step. All data is stored in a PostgreSQL database (`factormodel_db`). The universe consists of ~693 US large-cap stocks (varies with sector mapping updates).

**Database connection:**
```
postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db
```

**Key principle:** Both return residuals AND characteristics themselves are orthogonalized at each step using WLS (weighted by log market cap). Before entering any regression, each new characteristic is regressed cross-sectionally against all prior characteristics. Only the residual enters the regression. This is true Gram-Schmidt in the weighted inner-product space defined by cap weights.

---

## 2. DAILY PIPELINE — EXECUTION SEQUENCE

All scripts run in the same Jupyter kernel in the following order. Each step is fast (incremental) on daily runs — only new dates are computed.

```
Step 1:  factor_model_step1.py         run()                          — kernel-only, incremental
Step 2:  factor_model_v2.py            run()                          — kernel-only, incremental
Step 3:  quality_factor.py             run(Pxs_df, sectors_s)         — fast (cached weights)
Step 4:  value_factor.py               run_ic_study(Pxs_df, sectors_s)— fast (cached weights)
Step 5:  factor_ic_study.py            run_ic_study_alpha()            — fast (IC cached)
                                       compute_rolling_regime_weights() — fast
Step 6:  macro_indicators.py           run_macro_hedge_cached(...,      — fast (incremental)
                                           incremental_only=True)
Step 7:  momentum_exclusions.py        run_exclusions(Pxs_df, pct_df,  — fast (new dates only)
                                           ou_scores_df,
                                           incremental=True)
Step 8:  mvo_backtest.py               run_backtest(...,               — single-pass backtest
                                           mode='incremental')
```

### Notes on Daily Execution
- Steps 1-6 confirm "nothing to compute" on most daily runs — cache confirmations only
- Step 7 (`run_exclusions`) must run before `run_backtest` so exclusion lists are available
- Step 8 (`run_backtest`) in incremental mode only computes new portfolio dates and extends NAV; all X snapshots and portfolio caches are reused

### Full Rebuild Sequence (after parameter changes)
```python
# 1 — Rebuild exclusions
run_exclusions(Pxs_df, pct_df, ou_scores_df, force_rebuild=True)

# 2 — Run backtest (wipes portfolio cache, reuses X snapshots)
results = run_backtest(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df    = volumeTrd_df,
    weights_by_date = weights_by_date,
    mode            = 'rebuild',
    rebuild_cov     = False,    # True only when new stocks added to DB
    hedge_multi     = multi,
    hedges_l        = hedges_l,
)
```

---

## 3. FACTOR MODEL — 11-STEP ARCHITECTURE

### Step Sequence

```
Step 1:  Baseline UFV          Raw return variance (benchmark)
Step 2:  Market Beta           EWMA beta vs SPX, OLS cross-section
Step 3:  Size                  Z-scored log market cap, OLS; size ⊥ {beta}
Step 4:  Macro Factors         7 factors, joint Ridge (k-fold CV per date); each ⊥ {beta, size}
Step 5:  Sector Dummies        sum-to-zero coding, Ridge CV; each ⊥ {beta, size, macro}
Step 6:  Quality               OLS; quality ⊥ {beta, size, macro, sectors}
Step 7:  SI Composite          OLS; SI ⊥ {beta, size, macro, sectors, quality}
Step 8:  GK Volatility         OLS; vol ⊥ {beta, size, macro, sectors, quality, SI}
Step 9:  Idio Momentum         OLS (on vol residuals); mom ⊥ all prior
Step 10: Value                 OLS; value ⊥ all prior
Step 11: O-U Mean Reversion    OLS; final alpha step
```

### Key Constants

```python
BETA_WINDOW      = 252       # rolling window for beta/macro betas (trading days)
BETA_HL          = 126       # EWMA half-life for beta/macro betas
VOL_WINDOW       = 84        # shorter window for vol factor
VOL_HL           = 42        # EWMA half-life for vol factor
MOM_LONG         = 252       # momentum lookback
MOM_SKIP         = 21        # momentum skip period (avoid reversal contamination)
OU_MEANREV_W     = 60        # O-U AR(1) fitting window
OU_MIN_OBS       = 30        # minimum observations for valid O-U fit
OU_ST_REV_W      = 21        # ST reversal fallback window
OU_WEIGHT_REF    = 30.0      # reference half-life for O-U/reversal blend weight
OU_WEIGHT_CAP    = 10.0      # maximum O-U blend weight
MIN_STOCKS       = 150       # minimum stocks required for cross-sectional regression

RIDGE_GRID_MACRO = [0.15, 0.3, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
RIDGE_GRID_SEC   = [0.1, 0.2, 0.4, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
```

### Macro Factors (Step 4)

All are pre-computed daily changes passed in via `Pxs_df` columns:

| Column | Description |
|--------|-------------|
| `USGG2YR` | 2Y nominal rate daily change (bps) |
| `US10Y2Y_SPREAD_CHG` | 10Y-2Y spread daily change (bps) |
| `US10YREAL` | 10Y real yield / inflation breakeven daily change |
| `BE5Y5YFWD` | 5y5y forward breakeven inflation daily change |
| `MOVE` | Interest rate volatility index (MOVE) daily change |
| `Crude` | WTI crude oil daily change |
| `XAUUSD` | Gold daily change |

Each macro beta: `β_im = Cov_EWMA(r_i, d_m) / Var_EWMA(d_m)`, window=252, hl=126, z-scored cross-sectionally.

### Sector Dummies (Step 5)
**Sum-to-zero deviation coding:** for K sectors, each dummy = +1 own sector, -1/(K-1) all others. All sectors included — no reference dropped. Intercept = true equal-weighted market return.

### WLS Regression
All cross-sectional regressions: `w_i = log(market_cap_i)`, normalized to sum to 1.

---

## 4. FACTOR DETAILS

### Quality Factor (Step 6)
Rate-conditioned composite. Loaded from `quality_scores_df` cache.
- **GQF** (Growth Quality): non-2021/22 regime
- **CQF** (Conservative Quality): 2021/22 high-rate regime
- Blend: `(1-q)×GQF + q×CQF`, q from USGG10YR vs 252d MAV, threshold=50bps

```python
GQF_WEIGHTS = {
    'GGP': 0.140405, 'GS': 0.130709, 'GS/S_Vol': 0.118369,
    'GS*r2_S': 0.106796, 'GGP/GP_Vol': 0.097482, 'ROId': 0.090544,
    'GGP*r2_GP': 0.087003, 'FCF_PG': 0.085453, 'HSG': 0.072741, 'PSG': 0.070498,
}
CQF_WEIGHTS = {
    'OM': 0.128893, 'GE/E_Vol': 0.123484, 'ISGD': 0.120801,
    'GE*r2_E': 0.115882, 'OMd*r2_S': 0.113548, 'r&d': 0.091381,
    'LastSGD': 0.083599, 'SGD*r2_S': 0.082127, 'OMd': 0.072732, 'GE': 0.067552,
}
```

### Value Factor (Step 10)
IC-weighted composite. Hardcoded weights from last calibration:
```python
VALUE_WEIGHTS = {'P/S': 0.157, 'P/Ee': 0.143, 'P/Eo': 0.140,
                 'sP/S': 0.157, 'sP/E': 0.132, 'sP/GP': 0.157, 'P/GP': 0.115}
```

### O-U Mean Reversion (Step 11)
AR(1) fit to compounded residual price index. `half_life = ln(2)/(-ln(b))`. Requires `0 < b < 1`. Falls back to 21d ST reversal on residual index if fit invalid. Blended: `ou_weight = min(30/half_life, 10)`.

OU scores stored in `v2_ou_reversion_df` (model_version='v2').

---

## 5. SCRIPT: `quality_factor.py`

**Location:** `/mnt/user-data/outputs/quality_factor.py`

### Daily Usage (fast path)
```python
from quality_factor import run
run(Pxs_df, sectors_s)   # incremental — skips if all anchor dates cached
```

### Full Recalibration
```python
run(Pxs_df, sectors_s, force_recompute=True)
```

### Cache Tables
```
quality_scores_cache    Per-date quality scores    (date, ticker, score)
```
Anchor dates computed quarterly. Daily scores forward-filled between anchors.

---

## 6. SCRIPT: `value_factor.py`

**Location:** `/mnt/user-data/outputs/value_factor.py`

### Daily Usage (fast path)
```python
from value_factor import run_ic_study
run_ic_study(Pxs_df, sectors_s)   # uses hardcoded VALUE_WEIGHTS, incremental
```

### Full Recalibration
```python
run_ic_study(Pxs_df, sectors_s, use_cached_weights=False)
```

### Cache Tables
```
value_scores_cache    Per-date value scores    (date, ticker, score)
```

---

## 7. SCRIPT: `factor_ic_study.py`

**Location:** `/mnt/user-data/outputs/factor_ic_study.py`

### Daily Usage
```python
from factor_ic_study import run_ic_study_alpha, compute_rolling_regime_weights

# IC study — incremental (cached in factor_ic_cache)
ic_results, ic_annual = run_ic_study_alpha(Pxs_df, sectors_s, model_version='v2')

# Factor weights — recomputed from IC cache (fast)
weights_by_year, regime_s, weights_by_date = compute_rolling_regime_weights(
    ic_results, ic_annual, Pxs_df, horizons=['21d', '63d'])
```

### OU Factor IC Study
```python
from factor_ic_study import run_ou_factor_ic_study

ou_ic_df, ou_annual, ou_scores = run_ou_factor_ic_study(
    Pxs_df, sectors_s, model_version='v2')
```

**Key finding — OU IC by year:**
```
2019: IC_mean=+0.045  ICIR=+0.55  (OU works — mean reversion effective)
2020: IC_mean=+0.009  ICIR=+0.08
2021: IC_mean=+0.016  ICIR=+0.18
2022: IC_mean=-0.004  ICIR=-0.04
2023: IC_mean=+0.015  ICIR=+0.28
2024: IC_mean=-0.030  ICIR=-0.50  (AI cycle kills standalone MR)
2025: IC_mean=-0.013  ICIR=-0.12
2026: IC_mean=-0.134  ICIR=-1.07
```
OU as standalone signal not viable in AI cycle (2024+). Used as penalty modifier only in joint filter.

### Cache Tables
```
factor_ic_cache    IC values per factor/horizon/date    Per model_version
```

---

## 8. SCRIPT: `macro_indicators.py`

**Location:** `/mnt/user-data/outputs/macro_indicators.py`

### Daily Usage
```python
from macro_indicators import run_macro_hedge_cached

multi = run_macro_hedge_cached(
    Pxs_df, stocks_l, rates_l,
    hedges_l         = ['QQQ','SPX','IGV','SOXX','XHB','XLB','XLC','XLE',
                        'XLF','XLI','XLP','XLU','XLV','XLY','ARKK','IWM'],
    move_grid        = move_grid,
    qbd_grid         = qbd_grid,
    engine           = ENGINE,
    incremental_only = True,
)
```

### Full Rebuild
```python
multi = run_macro_hedge_cached(..., force_rebuild=True)
```

### Output Structure
```python
multi = {
    'results':     {ticker: {'signal_df': pd.DataFrame, ...}},  # per-instrument
    'qbd_s':       pd.Series,    # QBD macro indicator series
    'raw_move_s':  pd.Series,    # raw MOVE series
    'params_hash': str,
}
# Trigger check: multi['results'][ta]['signal_df']['signal']  (0 or 1)
```

### Key Parameters
```python
MOVE_HL   = 10    MOVE_MAV  = 5     CORR_WIN  = 63
VOL_HL    = 42    MAV_WIN   = 50    REF_ASSET = 'QQQ'
HEDGE_MIN_HOLD = 10               SIGNAL_LOGIC = 'AND'
```

### Cache Tables
```
macro_indicators_daily    QBD + raw MOVE per date         (date, params_hash)
macro_signal_daily        Per-instrument signal+eff        (date, instrument, params_hash)
macro_fitted_params       Monthly refit params             (params_hash)
```

---

## 9. SCRIPT: `momentum_exclusions.py`

**Location:** `/mnt/user-data/outputs/momentum_exclusions.py`

### Overview
Computes and caches a daily list of stocks to **exclude** from portfolio construction based on a joint two-gate condition. Fully self-contained.

**Gate 1 (OU):** `ou_score < OPT_OU_THRESHOLD` — stock is above its residual equilibrium

**Gate 2 (Momentum):** rescaled short-term return > p{OPT_PCT_THRESHOLD} on at least one window — extreme recent momentum

For each date, all stocks passing both gates are scored by a **joint overshooting score:**
```python
joint_score = overshot_magnitude × ou_excess
```

Stocks ranked by joint score descending. No cap applied at this stage — capping controlled by `MR_CAP` in `run_backtest` at consumption time.

### Calibration Results
```
k* p50=40, p75=40  → near-hard exclusion → binary implementation
pct* dominant: p95 (26%), p97 (12%), p99 (14%)
ou_thr* dominant: OU < 0.0 (40%), OU < -0.5 (14%), OU < -1.0 (14%)
Adoption rate: 67.7% of dates
IC improvement: median=+0.006, mean=+0.009
```

### Optimal Parameters (hardcoded)
```python
OPT_PCT_THRESHOLD = 95     # percentile threshold for rescaled return
OPT_OU_THRESHOLD  = 0.0    # OU score threshold
WINDOWS_TD        = {'1M': 21, '2M': 42, '3M': 63}
USE_ROBUST        = False
```

### Daily Usage
```python
from momentum_exclusions import run_exclusions, build_ou_scores_df

ou_scores_df = build_ou_scores_df(model_version='v2')
run_exclusions(Pxs_df, pct_df, ou_scores_df, incremental=True)
```

### Cache Tables
```
momentum_exclusions           (date, ticker, score)
momentum_exclusions_processed (date)
```

---

## 10. SCRIPT: `mvo_backtest.py`

**Location:** `/mnt/user-data/outputs/mvo_backtest.py`

### Entry Point

```python
results = run_backtest(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df         = volumeTrd_df,
    weights_by_date      = weights_by_date,
    mode                 = 'incremental',   # 'incremental' | 'rebuild'
    rebuild_cov          = False,           # True only when new stocks added to DB
    ic                   = 0.03,
    max_weight           = 0.075,
    min_weight           = 0.025,
    zscore_cap           = 2.5,
    pca_var_threshold    = 0.65,
    universe_mult        = 5,
    risk_aversion        = 10,
    hedge_multi          = multi,           # from run_macro_hedge_cached
    hedges_l             = hedges_l,
    hedge_trigger_assets = ['QQQ'],         # omit hedge_multi/hedges_l for 6-strategy run
)
```

User is prompted at runtime for: `top_n` (default 25), `rebal_freq` (default 15 days), `advp_cap` (default 4%), `prefilt_pct`, `conc_factor`, `min_cov_matrices`.

### Mode and Cache Flags

| Flag | Controls | When to set |
|------|----------|-------------|
| `mode='rebuild'` | Wipes `mvo_daily_portfolios` for this params_hash | After any parameter change |
| `mode='incremental'` | Computes only missing dates | Daily runs |
| `rebuild_cov=True` | Wipes and rebuilds `mvo_x_snapshots` | Only when new stocks added to DB |
| `rebuild_cov=False` | Loads cached X snapshots, builds missing on-demand | All other cases |

**Rule of thumb:** parameter calibration → `mode='rebuild', rebuild_cov=False`. Adding stocks to DB → `mode='rebuild', rebuild_cov=True`. Daily update → `mode='incremental'`.

---

### Architecture: Single-Pass, Per-Strategy NAV Tracking

`run_backtest` operates in a **single loop** over all trading days from `MB_START_DATE` to `Pxs_df.index[-1]`. Each of the 9 strategies maintains independent state:

```python
state[s_idx] = {
    'nav':              float,    # cumulative NAV (starts at 1.0)
    'hwm':              float,    # high-water mark
    'dd':               float,    # current drawdown from HWM
    'gross':            float,    # gross exposure (1.0 = fully invested)
    'trough':           float,    # NAV trough since last de-gross (for re-entry)
    'theo_nav':         float,    # theoretical full-exposure NAV (for DD re-entry)
    'weights':          Series,   # current portfolio weights
    'last_rebal':       Timestamp,
    'days_held':        int,
    'regime':           str,      # 'alpha' | 'hybrid' | 'mvo'
    'deployed_regime':  str,      # regime of last actual rebalance
    'dd_level':         int,      # current DD level index (-1 = none)
    'dd_active':        bool,
    'dd_regime_forced': bool,     # True when DD event forces immediate rebalance
    'hedge_acct':       float,    # cumulative hedge P&L
    'active_hedges':    dict,     # {instrument: {entry_px, weight, beta, eff}}
    'weights_by_date':  dict,
    'rebal_log':        list,
    'dd_log':           list,
}
```

**Key benefit:** ADVP filter at each rebalance uses `AUM × state[s]['nav'] × state[s]['gross']` — the exact current AUM for that strategy. No proxies, no second pass.

### Daily Loop Order (per trading day)

```
1. Increment days_held for dynamic strategies
2. If calc_date:
   a. Build X snapshot on-demand (if missing)
   b. Build/load alpha, MVO, hybrid, penalised variants
   c. Apply ADVP filter per strategy with its own AUM
   d. Rebalance static strategies (S0–S4)
3. Dynamic trigger check (S5–S8), every day from min_hold_days onwards
4. Update all NAVs (using current gross before any DD change)
5. Update regime for all strategies
6. Update hedge for S6/S7/S8
7. Apply DD de-grossing daily (takes effect next day)
```

---

### 9 Strategies

| # | Name | Description |
|---|------|-------------|
| S0 | Baseline | Quality factor only, equal weight, monthly rebalance |
| S1 | Pure Alpha | Composite alpha signal, concentrated (conc_factor=2.0), monthly |
| S2 | MVO | Ensemble covariance-optimized, monthly |
| S3 | Hybrid | Simple average of Alpha and MVO weight vectors, monthly |
| S4 | Smart Hybrid | DD-regime switching between Alpha / Hybrid / MVO, monthly |
| S5 | Dynamic | Signal-triggered rebalancing, independent schedule |
| S6 | Dyn+Hedge | Dynamic + macro hedge overlay |
| S7 | DD Policy | Dynamic + Hedge + multi-level drawdown de-grossing |
| S8 | Excl | DD Policy + MR momentum exclusion filter |

**Static strategies (S0–S4):** rebalance on a fixed monthly calendar (`rebal_freq` days). Each uses its own running NAV for ADVP filtering.

**Dynamic strategies (S5–S8):** each has its own rebalancing schedule, triggered independently. All share the same portfolio construction logic but diverge through overlays (hedge, DD, exclusions). Each tracks its own NAV, HWM, DD level, and gross exposure.

---

### Portfolio Construction

#### Regime Selection
The macro rates regime is computed from `regime_s` (exogenous, from `factor_ic_study.py`). The portfolio regime then depends on each strategy's own drawdown:

```
dd ≥ -7.5%   → use alpha portfolio   (exogenous regime also alpha/hybrid/mvo)
dd ≥ -17.5%  → use hybrid portfolio
dd  < -17.5% → use MVO portfolio
```

For dynamic strategies, `deployed_regime` (the regime at last rebalance) is tracked separately from the current signal regime. A regime change triggers rebalancing when `deployed_regime ≠ signal_regime` and `days_held ≥ min_hold_days`.

#### Rebalancing Triggers (Dynamic Strategies)
```python
# Turnover trigger
to_trigger = (TO > to_threshold[deployed_regime]
              AND vol_diff < DYN_VOLDIFF_CAP
              AND days_held >= DYN_MIN_HOLD_DAYS)

# Regime switch trigger (no vol_diff condition)
regime_switch = (deployed_regime != signal_regime
                 AND days_held >= DYN_MIN_HOLD_DAYS)

# De-risk trigger (independent of TO)
derisk = (vol_diff < DYN_VOLDIFF_DERISK
          AND days_held >= DYN_MIN_HOLD_DAYS)

# DD-forced regime override (immediate — no min hold)
dd_regime_override = state[s]['dd_regime_forced']
```

Where `vol_diff = vol_new_portfolio - vol_deployed_portfolio` and TO thresholds are:
```python
DYN_TO_THRESHOLD_ALPHA  = 0.25
DYN_TO_THRESHOLD_HYBRID = 0.30
DYN_TO_THRESHOLD_MVO    = 0.35
DYN_VOLDIFF_CAP         = 0.175   # max vol increase allowed alongside TO trigger
DYN_VOLDIFF_DERISK      = -0.750  # vol de-risk (effectively disabled)
DYN_MIN_HOLD_DAYS       = 10
```

---

### Liquidity Constraints (ADVP Filter)

This is the most critical portfolio construction constraint. It is applied at every rebalance for every strategy using that strategy's exact current AUM.

#### Three-Gate Process

**Gate 1 — Exclusion test** (`_is_liquid`):
```python
median_dv = median(price × volume, last ADV_WINDOW=20 days)
advp_capacity = (median_dv × advp_cap) / current_aum

# Stock is EXCLUDED if:
advp_capacity < min_weight
# i.e. even at minimum allocation, we'd exceed 4% of daily volume
```

Excluded stocks are replaced from the candidates pool, searching first the pre-filtered 5×N candidate universe, then the full ~693-stock composite universe if candidates are exhausted.

**Gate 2 — Weight cap** (applied within tier):
```python
# Stock is CAPPED at:
effective_cap = min(advp_capacity, max_weight)
```

**Gate 3 — Tier-structure preservation**:
The portfolio is split into top half (concentrated) and bottom half (standard). Caps and redistribution are applied within each tier independently:
- Excess from capped top-half stocks redistributes to other uncapped top-half stocks
- Excess from capped bottom-half stocks redistributes to other uncapped bottom-half stocks
- Only if an entire tier is fully constrained does excess spill to the other tier
- Concentration ratio (top_a / bot_a) is maintained whenever liquidity allows

This guarantees: ADVP constraints are always respected; concentration structure is preserved whenever possible; breaking concentration is the last resort, not the default.

#### Concentration Weights
```python
conc_factor = 2.0  # default
top_a = conc_factor / (conc_factor + 1) = 2/3   # top half budget
bot_a = 1 / (conc_factor + 1)           = 1/3   # bottom half budget

# With N=25 stocks: n_top=13, n_bot=12
# Base top weight:    (2/3) / 13 = 5.13%
# Base bottom weight: (1/3) / 12 = 2.78%
```

#### Example (N=25, AUM=$210M, advp_cap=4%)
```
Stock SMR: median_dv=$9.2M → capacity = $9.2M / $210M = 4.38%
Base top weight: 5.13%  →  SMR capped at 4.38%
Excess 0.75% redistributes to other 12 uncapped top-half stocks: +0.06% each
Bottom-half stocks: unchanged
```

#### Key Constants
```python
ADV_WINDOW   = 20     # days for median dollar ADV calculation
VOLUME_WINDOW = 10    # rolling window for volume de-trending
advp_cap     = 0.04   # default 4%, prompted at runtime
min_weight   = 0.025  # single-name floor (also exclusion threshold)
max_weight   = 0.075  # single-name ceiling
```

**Important:** `min_weight` serves a single purpose — the exclusion gate. A stock that passes the exclusion test stays in the portfolio regardless of where tier redistribution lands its weight. It is never dropped post-inclusion for being below `min_weight`.

---

### Hedge Overlay (S6, S7, S8)

Active only when `hedge_multi` and `hedges_l` are provided.

**Trigger:** macro signal from `hedge_multi['results'][ta]['signal_df']['signal']` with **1-day lag**. Fires when previous day's signal = 1 for any asset in `hedge_trigger_assets` (default `['QQQ']`).

**Instrument selection:** `_select_hedge_instruments` ranks candidate instruments from `hedges_l` by beta × effectiveness. Total hedge size = `min(n_instruments × hedge_ratio, max_hedge)`. Allocation proportional to beta × effectiveness score.

**P&L:** hedge positions are short (negative weight). Daily P&L = `-(px_today / px_entry - 1) × weight`. Credited to `hedge_acct` when position closed. HWM and DD track the combined portfolio (stock + hedge) P&L.

```python
HEDGE_RATIO = 0.25    # hedge size per instrument (fraction of NAV)
MAX_HEDGE   = 0.50    # maximum total hedge
EFF_FLOOR   = 0.75    # minimum effectiveness score
CORR_FLOOR  = 0.50    # minimum correlation to portfolio
BETA_WINDOW = 63
CORR_WINDOW = 63
```

---

### Drawdown Policy (S7, S8)

Multi-level de-grossing applied daily after NAV update. Each level fires sequentially — level 2 cannot fire before level 1, etc.

```python
DD_LEVELS = [
    (0.175, 2/5),  # -17.5%: cut 40% of remaining → 60% exposed
    (0.300, 3/7),  # -30.0%: cut 43% of remaining → 34% exposed
    (0.350, 1/2),  # -35.0%: cut 50% of remaining → 17% exposed
    (0.400, 2/3),  # -40.0%: cut 67% of remaining →  6% exposed
    (0.450, 1/1),  # -45.0%: cut 100%              →  0% exposed
]
DD_REENTRY_PCT      = 0.075   # theo_nav recovery from trough needed
DD_REENTRY_CONFIRM  = 5       # days recovery must persist
DD_ANNUAL_RESET_PCT = 0.30    # max YTD DD for Jan 1-10 calendar reset
```

**`theo_nav` tracking:** when de-grossed, `theo_nav` compounds daily at the full-exposure portfolio return (ignoring the de-grossing cost). Recovery is measured as `theo_nav / trough - 1`. This ensures re-entry requires genuine portfolio recovery, not just the passage of time.

**De-grossing takes effect the next day:** the gross exposure change is applied at end of day; NAV for that day uses the pre-change gross.

**On re-entry:** HWM is reset to current NAV (prevents immediate re-trigger). `dd_level` resets to -1.

**DD-forced regime override:** when a de-gross event fires, the strategy is immediately forced to `DD_LEVEL_REGIME[level]` (always 'mvo'). This overrides the normal min_hold constraint — the rebalance to MVO happens on the next check regardless of days_held.

---

### MR Exclusion Filter (S8)

```python
MR_CAP = 0.0    # 0.0 = no exclusions → S8 ≡ S7 exactly (convergence test)
MR_K   = 0.5    # soft penalty mode: divide z-scores by exp(MR_K × MR_score)
```

At each rebalance, stocks with positive MR penalty scores have their composite z-scores reduced via `exp(MR_K × score)` before portfolio construction. The `MR_CAP` parameter controls hard exclusion (top `MR_CAP × n_candidates` stocks excluded entirely).

**Convergence guarantee:** `MR_CAP=0.0, MR_K=0.0` → strategy 8 ≡ strategy 7 exactly.

---

### Caching Architecture

```
mvo_x_snapshots      Monthly X factor snapshots     Per model_version
                     Built on-demand during main loop, not upfront
                     Expensive: ~2-3 min per snapshot on first build
                     Controlled by rebuild_cov flag (not mode flag)

mvo_daily_portfolios Portfolio weights (alpha/mvo/hybrid/excl variants)
                     Per params_hash — wiped on mode='rebuild'
                     Includes 'candidates' entry (full pre-filtered pool)
                     for ADVP replacement fallback
```

**X snapshots** are parameter-independent — they depend only on price history and factor model outputs. They are never wiped by `mode='rebuild'`, only by `rebuild_cov=True`. This makes parameter calibration fast: changing `top_n`, `ic`, `conc_factor` etc. only requires rebuilding the portfolio cache (~seconds per date), not the X snapshots (~minutes).

---

### Runtime Output

At each rebalance of the DD Policy strategy (S7), prints:
```
-- 2024-08-02  [105]  regime=alpha  dd=-18.4%  gross=60%  n=25  eff_N=22.1
   TO=34%  AUM=$312.5M  port_vol=38.2%
MR top-10 (k=0.5): ...
*** ADVP cap active (AUM=$312,500,000, cap=4.0% ADV): ONDS, WULF
SNDK     5.81%  Hardware
AAOI     5.81%  ...
WULF     3.94%  Fintech   ***
ONDS     3.52%  ...       ***
```

Post-run display:
1. Strategy comparison table (CAGR, Vol, Sharpe, MDD, CAGR/DD)
2. Yearly returns for all 9 strategies + AUM
3. DD Policy event log (de-gross and re-entry events)
4. Unified current portfolio table (drifted weights % of AUM, all strategies)
5. 10-day daily P&L table (all strategies)
6. Trade summary (Δ weights for dynamic strategies)

---

### Global Constants

```python
MB_START_DATE  = pd.Timestamp('2019-01-01')
MB_TOP_N       = 25          # default (overridden by prompt)
MB_REBAL_FREQ  = 15          # default rebalance frequency in calendar days
MB_MODEL_VER   = 'v2'
AUM            = 20_000_000  # starting AUM ($20M)
TRADING_COST_BPS = 10        # one-way costs in bps

# Smart Hybrid / DD regime thresholds
SH_DD_ALPHA       = 0.075    # dd below: alpha → hybrid
SH_DD_HYBRID      = 0.175    # dd below: hybrid → MVO

# Hedge
BETA_WINDOW    = 63
CORR_WINDOW    = 63
EFF_MAV_WINDOW = 20
EFF_FLOOR      = 0.75
CORR_FLOOR     = 0.50
HEDGE_RATIO    = 0.25
MAX_HEDGE      = 0.50
TRIGGER_ASSETS = ['QQQ', 'SPY']
```

---

## 11. DATABASE TABLES

### Factor Model
```
factor_residuals_quality     Daily factor residuals (v1)       (date, ticker, ...)
v2_factor_residuals_quality  Daily factor residuals (v2)       (date, ticker, ...)
v2_ou_reversion_df           OU scores per date/ticker         (date, ticker, ou_score, ...)
quality_scores_cache         Quality factor scores             (date, ticker, score)
value_scores_cache           Value factor scores               (date, ticker, score)
factor_ic_cache              Factor IC by date/horizon         (date, factor, horizon, ic)
short_interest_data          SI data from Ortex               (date, ticker, metric, value)
valuation_consolidated       Fundamental valuation metrics     (date, ticker, ...)
```

### MVO / Backtest
```
mvo_x_snapshots              Monthly X factor snapshots        Per model_version
mvo_daily_portfolios         alpha/mvo/hybrid/excl weights     Per params_hash
macro_indicators_daily       QBD + raw MOVE                    Per params_hash
macro_signal_daily           Per-instrument signals + eff      Per params_hash
macro_fitted_params          Monthly refit params              Per params_hash
momentum_exclusions          Excl list with joint scores       Global (date, ticker, score)
momentum_exclusions_processed Processed dates tracker          Global (date)
```

---

## 12. NOTES AND CONVENTIONS

- **Ticker format:** bare tickers (no `' US'`). `clean_ticker()` strips suffix.
- **DB writes:** always upsert.
- **Common sample:** intersection of all dates/stocks where every characteristic available.
- **Volume input:** `volumeTrd_df` is raw daily share volume. `volumeRaw_df` is derived internally as a copy and used for ADV cap calculations.
- **Regime computation:** never cached. Always computed inline from each strategy's own NAV drawdown. `regime_s` (exogenous rates regime) used as baseline; DD overrides when drawdown thresholds are crossed.
- **MR_CAP vs MAX_EXCL_PCT:** `MAX_EXCL_PCT` removed. All capping controlled by `MR_CAP` in `run_backtest` at consumption time.
- **SPX vs SPY:** portfolio uses `SPX` column in `Pxs_df`. SPX can be in `hedges_l` as hedge instrument but NOT as trigger — `hedge_trigger_assets` defaults to `['QQQ']`.
- **Hedge instrument naming:** `hedges_l` must match column names in `Pxs_df` exactly.
- **Size in valuation tables:** stored as market cap in $millions.
- **Per-strategy AUM:** each strategy uses `AUM × nav × gross` for ADVP filtering. No proxy NAVs, no second pass.
- **ADVP min_weight:** the single-name floor serves only as the exclusion gate. Once a stock passes the liquidity test, it is never dropped post-inclusion for falling below min_weight during redistribution.

---

## 13. PENDING WORK

### In Progress
1. **Performance validation** — cross-check run_backtest results vs old run_mvo_backtest reference numbers
2. **MR_CAP calibration** — run with `MR_CAP` ∈ {0.02, 0.03, 0.05} to find optimal exclusion intensity
3. **QUALITY_FLOOR calibration** — test floors at 0.40, 0.45, 0.50

### Completed (Sessions through April 2026)
- [DONE] Factor model v2 — full 11-step pipeline, incremental
- [DONE] Quality/Value/IC factors — hardcoded weights, fully incremental
- [DONE] OU factor IC study — IC negative in AI cycle; used as penalty modifier only
- [DONE] `momentum_exclusions.py` — fully self-contained, joint MR+OU filter
- [DONE] `MAX_EXCL_PCT` removed — capping delegated to `MR_CAP` in backtest
- [DONE] `run_backtest` — unified single-pass architecture replacing run_daily_cache_build + run_mvo_backtest
- [DONE] Per-strategy independent NAV tracking and rebalancing schedules
- [DONE] Per-strategy ADVP filtering with correct AUM (no proxies)
- [DONE] X snapshots built on-demand during main loop (no upfront batch build)
- [DONE] `rebuild_cov` flag decoupled from `mode` — parameter calibration reuses expensive cov matrices
- [DONE] ADVP tier-structure preservation — excess redistributes within tier; concentration maintained whenever liquidity allows
- [DONE] Full-universe fallback for ADVP replacements when candidate pool exhausted
- [DONE] DD logic matched exactly to old code — daily check, theo_nav daily compounding, sequential levels, HWM reset on re-entry
- [DONE] Hedge trigger fixed — uses macro signal from `hedge_multi['results']` with 1-day lag
- [DONE] Hybrid portfolios correctly exceed N stocks (union of alpha + MVO, up to 2×N)
- [DONE] `mvo_backtest_vMR.py` saved as stable checkpoint
