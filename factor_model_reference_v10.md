# Factor Model & MVO Pipeline — Comprehensive Reference
*Last updated: May 2026 (v8)*

---

## 1. OVERVIEW

A sequential Fama-MacBeth cross-sectional factor model implemented in Python, running in a Jupyter notebook kernel. The model strips systematic return sources one by one in a true Gram-Schmidt orthogonalization sequence, producing clean residuals at each step. All data is stored in a PostgreSQL database (`factormodel_db`). The universe consists of ~693 US large-cap stocks (varies with sector mapping updates).

**Database connection:**
```
postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db
```

**Key principle:** Both return residuals AND characteristics themselves are orthogonalized at each step using WLS (weighted by log market cap). Before entering any regression, each new characteristic is regressed cross-sectionally against all prior characteristics. Only the residual enters the regression. This is true Gram-Schmidt in the weighted inner-product space defined by cap weights.

**Lookahead bias:** The pipeline is designed to be fully point-in-time by construction. No information from future dates is ever used at any step. Quality and value factor weights are derived from IC evidence available strictly before each cutoff date. All factor model residuals are computed using only data available up to each calculation date.

**Self-contained factor model:** `factor_model_v2.py` is fully self-contained — no kernel dependency on `factor_model_v1.py` or `quality_factor.py`. Quality and value PIT weight derivation and score computation are integrated inline at the correct sequence points within `run()`.

---

## 2. DAILY PIPELINE — EXECUTION SEQUENCE

The pipeline is structured as 5 sequential stages with clear input/output contracts:

```
Stage 1:  factor_model_v2.py     run()                    — factor model + PIT weights inline
Stage 2:  factor_ic_study.py     run_ic_study_alpha()     — alpha IC study (cached)
                                  compute_rolling_regime_weights()
Stage 3:  macro_indicators.py    run_macro_hedge_cached() — hedge signals (incremental)
Stage 4:  momentum_exclusions.py run_exclusions()         — MR exclusions (incremental)
Stage 5:  mvo_backtest.py        run_backtest()           — portfolio construction + backtest
```

### Stage Details

**Stage 1 — Factor Model (self-contained):**
- Runs Gram-Schmidt 10-step sequence for new date
- After Step 1 (mkt residuals): derives quality PIT weights for any new cutoff dates, reloads quality scores
- After Step 4 (size residuals): derives value PIT weights for any new cutoff dates, recomputes value scores
- Writes `factor_residuals_*`, `quality_scores_df`, `value_scores_df`, `quality_weights_pit`, `value_weights_pit`, `value_ic_bank`, `v2_ou_reversion_df`

**Stage 2 — Alpha IC Study:**
- Reads `factor_residuals_*` from Stage 1
- Outputs `weights_by_date`, `regime_s`, `weights_by_year` (in-memory)

**Stage 3 — Macro Hedge:**
- Independent of factor model — reads `Pxs_df` only
- Outputs `hedge_multi` (in-memory) — must run in **same kernel** as Stage 5

**Stage 4 — MR Exclusions:**
- Reads `v2_ou_reversion_df`, `factor_residuals_sec` from Stage 1
- Fully DB-backed — can run in separate kernel

**Stage 5 — Backtest:**
- Consumes outputs of all prior stages
- Writes `mvo_daily_portfolios`, `mvo_x_snapshots`

### Daily Incremental Calls
```python
# Stage 1
from factor_model_v2 import run
run(Pxs_df, sectors_s)                         # or with volumeTrd_df

# Stage 2
ic_results, ic_annual = run_ic_study_alpha(Pxs_df, sectors_s, model_version='v2')
weights_by_year, regime_s, weights_by_date = compute_rolling_regime_weights(
    ic_results, ic_annual, Pxs_df, horizons=['21d', '63d'], half_life_yrs=2.0)

# Stage 3
multi = run_macro_hedge_cached(..., incremental_only=True)

# Stage 4
run_exclusions(Pxs_df, pct_df, ou_scores_df, incremental=True)

# Stage 5
results = run_backtest(Pxs_df, sectors_s, weights_by_year, regime_s,
                       weights_by_date=weights_by_date, mode='incremental', ...)
```

### Full Rebuild (after parameter changes or data updates)
```python
# Wipe all caches
exec(open('wipe_caches.py').read())

# Full rebuild — PIT weights computed inline
run(Pxs_df, sectors_s, force_rebuild_pit=True)

# Then rest of pipeline
```

### Notes
- `run_macro_hedge_cached` must run in **same kernel** as `run_backtest` — `hedge_multi` is in-memory
- `momentum_exclusions` is fully DB-backed — can run in separate kernel
- `quality_factor.py` is **retired** — all logic integrated into `factor_model_v2.py`
- `value_factor.py` still used for standalone value IC/weight operations if needed, but not required in daily pipeline

---

## 3. FACTOR MODEL V2 — ARCHITECTURE

### Script
`factor_model_v2.py` — fully self-contained, 3800+ lines. No imports from v1 or quality_factor.

### Entry Point
```python
from factor_model_v2 import run

run(Pxs_df, sectors_s)                          # daily incremental
run(Pxs_df, sectors_s, volumeTrd_df=vdf)        # with volume-scaled momentum
run(Pxs_df, sectors_s, force_rebuild_pit=True)  # full rebuild, wipes PIT caches
```

User is prompted at runtime: incremental vs full recalculation, start date, volume scaling options.

### 10-Step Gram-Schmidt Sequence

```
Step 1:  Market Beta        EWMA beta vs SPX, OLS
         → [PIT quality weights updated here using resid_mkt]
         → [quality_scores_df reloaded with fresh weights]
Step 2:  Quality            OLS; quality ⊥ {beta}
Step 3:  Idio Momentum      OLS; mom ⊥ {beta, quality}
Step 4:  Size               Z-scored log market cap; size ⊥ {beta, quality, mom}
         → [PIT value weights updated here using resid_size]
         → [value_scores_df recomputed with fresh weights]
Step 5:  Value              OLS; value ⊥ {beta, quality, mom, size}
Step 6:  SI Composite       OLS; SI ⊥ all prior
Step 7:  GK Volatility      OLS; vol ⊥ all prior
Step 8:  Macro Factors      7 factors, joint Ridge (k-fold CV); each ⊥ all prior
Step 9:  Sector Dummies     sum-to-zero coding, Ridge CV; each ⊥ all prior
Step 10: O-U Mean Reversion OLS on sector residuals; final alpha step
```

### Residuals Used for IC Studies
```
Quality IC   → v2_factor_residuals_mkt   (Step 1 output — only mkt beta removed)
Value IC     → v2_factor_residuals_size  (Step 4 output — mkt, quality, mom, size removed)
```

### Key Constants

```python
BETA_WINDOW      = 252       # rolling window for beta/macro betas
BETA_HL          = 126       # EWMA half-life
VOL_WINDOW       = 84        # GK vol window
VOL_HL           = 42        # GK vol EWMA half-life
MOM_LONG         = 252       # momentum lookback
MOM_SKIP         = 21        # momentum skip (reversal avoidance)
OU_MEANREV_W     = 60        # O-U AR(1) fitting window
OU_MIN_OBS       = 30        # min observations for O-U fit
OU_ST_REV_W      = 21        # ST reversal fallback window
OU_WEIGHT_REF    = 30.0      # O-U/reversal blend reference half-life
OU_WEIGHT_CAP    = 10.0      # O-U blend weight cap
MIN_STOCKS       = 150       # min stocks for cross-sectional regression
FM_START_DATE    = pd.Timestamp('2017-01-01')

# Quality regime signal
QF_MAV_WINDOW    = 252       # rolling MAV window for rate signal
QF_THRESHOLD     = 15        # bps — threshold above/below MAV for regime change
QF_RATE_COL      = '10Y RATE'  # rate column in Pxs_df (in bps, distinct from USGG10YR macro factor)

# Ridge grids
RIDGE_GRID_MACRO = [0.15, 0.3, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
RIDGE_GRID_SEC   = [0.1, 0.2, 0.4, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
```

### Rate Regime Signal
The `'10Y RATE'` column (in bps) is used exclusively for quality regime classification. It is **distinct** from `USGG10YR` which is a macro factor in Step 8.

```python
rate_mom = rate - rate.rolling(252).mean()
q = 0.5   # neutral (default)
if rate_mom >  15:  q = 1.0   # tight regime
if rate_mom < -15:  q = 0.0   # easing regime
```

### Macro Factors (Step 8)

| Column | Description |
|--------|-------------|
| `USGG2YR` | 2Y nominal rate daily change |
| `US10Y2Y_SPREAD_CHG` | 10Y-2Y spread daily change |
| `US10YREAL` | 10Y real yield daily change |
| `BE5Y5YFWD` | 5y5y forward breakeven daily change |
| `MOVE` | Rate vol index daily change |
| `Crude` | WTI crude daily change |
| `XAUUSD` | Gold daily change |

### Sector Dummies (Step 9)
Sum-to-zero deviation coding: each dummy = +1 own sector, -1/(K-1) all others. All sectors included, no reference dropped.

### Sparse Factor Handling
When quality or value have no data for a given date, the chain continues using residuals from the prior step:
- Quality missing → use `resid_mkt` as fallback for `resid_quality`
- Value missing → use `resid_size` as fallback for `resid_value`

Neither factor is in `common_dates` intersection — their absence never blocks other factor computation.

### O-U Batch Saving
Full recalculation saves O-U scores in batches of 200 dates to prevent data loss from crashes. Incremental loads the last 120 days of sector residuals from DB.

---

## 4. QUALITY FACTOR — INTEGRATED PIT WEIGHTS

Quality score computation and weight derivation are fully integrated into `factor_model_v2.py`. `quality_factor.py` is retired.

### Measurement Space and Sector Relativity

**All quality scores are computed in rank space, within sector.** This is a deliberate design choice with two consequences:

1. **Rank space [0, 1]:** raw financial metrics (growth rates, margins, volatility-adjusted ratios) are converted to percentile ranks within their sector before any weighting or aggregation. A score of 1.0 = best in sector, 0.0 = worst. This eliminates sensitivity to the scale and distribution of raw metrics.

2. **Within-sector only:** each stock is ranked against its sector peers, not the full cross-section. A stock's quality score reflects how good it looks *relative to its own sector*, not in absolute terms. This means a highly profitable company in a low-quality sector gets the same score range as a moderately profitable company in a high-quality sector — the model does not express views on absolute sector quality.

The winsorization step (`_quality_winsorize`) clips outliers before ranking to prevent extreme observations from distorting the peer group percentiles.

### Weight Derivation (PIT, after Step 1)

At each cutoff date T = anchor A + 63 trading days:

**Eligible anchors:** all A where `A + 63 trading days < T`

**Regime classification per anchor** (step function using `'10Y RATE'` in bps):
- `rate_mom > 15 bps` → q = 1.0 (tight) → anchor contributes with weight q to CQF, weight (1-q) to GQF
- `rate_mom < -15 bps` → q = 0.0 (easing) → fully to GQF
- otherwise → q = 0.5 → 50/50 split

**IC computation per anchor A, metric M, horizon H (21d or 63d):**
1. Rank stocks within sector on metric M at anchor A (after winsorization)
2. Compound forward `factor_residuals_mkt` returns over H days after A
3. IC = `(top_decile_median - bottom_decile_median) / cross_sectional_std`
4. Scale: GQF gets IC × (1-q), CQF gets IC × q

**Weight derivation (separately for GQF and CQF):**
- Aggregate scaled ICs across all eligible anchors and horizons
- Filter: keep metrics where `avg_sz > 0` AND `avg_sz > median(avg_sz)` AND `avg_t > median(avg_t)`
- Select top 10 by t-stat, normalize t-stats as weights
- If CQF has no qualifying metrics, fall back to GQF weights

### Score Computation (per calc_date)
```python
# Load most recent prior quality_weights_pit entry
gqf_w, cqf_w = get_quality_weights_at(calc_date)

# Get rate regime at calc_date
q = rate_signal(calc_date)   # 0.0 / 0.5 / 1.0

# Rank within sector using PIT weights
gqf_score = weighted_sector_rank(snap, gqf_w)
cqf_score = weighted_sector_rank(snap, cqf_w)

# Blend
score = (1 - q) * gqf_score + q * cqf_score
```

### Metrics
```python
QUALITY_METRICS_ALL = [
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    'OM', 'ROI', 'FCF_PG', 'OMd', 'ROId', 'ISGD', 'r&d',
    'GS/S_Vol', 'HSG/S_Vol', 'PSG/S_Vol', 'GE/E_Vol', 'PIG/E_Vol', 'GGP/GP_Vol',
    'GS*r2_S', 'SGD*r2_S', 'OMd*r2_S', 'GE*r2_E', 'PIG*r2_E', 'GGP*r2_GP',
]
QUALITY_EXCLUDE_METRICS = ['ROE', 'ROE-P', 'ROEd']
QUALITY_MAX_COMPONENTS  = 10
```

### Cache Tables
```
quality_scores_df         Per-date composite scores     (date, ticker, score)
quality_weights_pit       PIT weights per cutoff        (cutoff_date, regime, metric, weight)
valuation_metrics_anchors Monthly fundamental snapshots (date, ticker, metric, value)
v2_factor_residuals_mkt   Mkt beta residuals            (date, ticker, resid)
```

---

## 5. VALUE FACTOR — INTEGRATED PIT WEIGHTS

Value PIT weight derivation integrated into `factor_model_v2.py` after Step 4.

### Measurement Space and Sector Relativity

**All value scores are computed in rank space, within sector** — the same philosophy as the Quality factor.

1. **Rank space [0, 1]:** valuation multiples (P/E, P/S, etc.) are not used as absolute numbers. Within each sector, stocks are ranked by cheapness and the rank is normalised to [0, 1]. A score of 1.0 = cheapest in sector, 0.0 = most expensive.

2. **Sign adjustment for negative multiples:** valuation metrics can be negative for loss-making companies (e.g., negative P/E). Before ranking, negatives are repositioned above the positives using `adj[~pos] = max_positive + abs(negative)`. This ensures loss-making companies rank as more expensive than profitable ones, regardless of the sign of their multiple.

3. **Within-sector only:** a stock's value score reflects cheapness *relative to its sector peers*. A cheap company in an expensive sector scores highly; an expensive company in a cheap sector scores poorly. The model deliberately does not express cross-sector value views — sector-level valuation differences are captured separately by the factor model's sector dummies (Step 9) and market beta structure.

### Metrics
```python
VALUE_METRICS = ['P/S', 'P/Ee', 'P/Eo', 'sP/S', 'sP/E', 'sP/GP', 'P/GP']
```

### Weight Derivation (PIT, after Step 4)

Same cutoff logic (63 trading days). IC computed against `v2_factor_residuals_size`.

**Fallback when < 2 metrics pass IC filter:**
- 1 metric passes: 60% to surviving metric + 40% distributed from most recent prior stable period (prior weights, excluding the surviving metric, renormalized to 40%)
- 0 metrics pass: forward-fill most recent prior stable weights entirely
- No prior stable period: equal weights across all 7 metrics

**IC bank caching:** `value_ic_bank` stores IC per (anchor, metric, horizon). Daily incremental only recomputes last anchor's IC — O(1) cost.

### Cache Tables
```
value_scores_df     Per-date value scores         (date, ticker, score)
value_weights_pit   PIT weights per cutoff        (cutoff_date, metric, weight)
value_ic_bank       IC bank                       (anchor_date, metric, horizon, ic)
valuation_consolidated Valuation fundamentals     (date, ticker, ...)
v2_factor_residuals_size Size residuals           (date, ticker, resid)
```

---

## 6. SCRIPT: `factor_ic_study.py`

### Daily Usage
```python
from factor_ic_study import run_ic_study_alpha, compute_rolling_regime_weights

ic_results, ic_annual = run_ic_study_alpha(Pxs_df, sectors_s, model_version='v2')
weights_by_year, regime_s, weights_by_date = compute_rolling_regime_weights(
    ic_results, ic_annual, Pxs_df, horizons=['21d', '63d'], half_life_yrs=2.0)
```

### Cache Tables
```
factor_ic_cache    IC values per factor/horizon/date
```

---

## 7. SCRIPT: `macro_indicators.py`

### Daily Usage
```python
multi = run_macro_hedge_cached(
    Pxs_df, stocks_l, rates_l,
    hedges_l=['QQQ','SPX','IGV','SOXX','XHB','XLB','XLC','XLE',
              'XLF','XLI','XLP','XLU','XLV','XLY','ARKK','IWM'],
    move_grid=move_grid, qbd_grid=qbd_grid,
    engine=ENGINE, incremental_only=True,
)
```

`hedge_multi` is in-memory — must run in same kernel as `run_backtest`.

### Cache Tables
```
macro_indicators_daily    QBD + raw MOVE per date
macro_signal_daily        Per-instrument signal + effectiveness
macro_fitted_params       Monthly refit parameters
```

---

## 8. SCRIPT: `momentum_exclusions.py`

### Overview
Computes daily exclusion list based on joint two-gate condition:
- Gate 1 (OU): `ou_score < 0` — stock above residual equilibrium
- Gate 2 (Momentum): rescaled short-term return > p95 on at least one window (1M/2M/3M)

### Daily Usage
```python
from momentum_exclusions import run_exclusions, build_ou_scores_df
ou_scores_df = build_ou_scores_df(model_version='v2')
run_exclusions(Pxs_df, pct_df, ou_scores_df, incremental=True)
```

Fully DB-backed — can run in separate kernel.

---

## 9. SCRIPT: `mvo_backtest.py`

### Entry Point
```python
results = run_backtest(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df         = volumeTrd_df,
    weights_by_date      = weights_by_date,
    mode                 = 'incremental',   # 'incremental' | 'rebuild'
    rebuild_cov          = False,
    ic                   = 0.03,
    max_weight           = 0.075,
    min_weight           = 0.025,
    zscore_cap           = 2.5,
    pca_var_threshold    = 0.65,
    universe_mult        = 5,
    risk_aversion        = 10,
    hedge_multi          = multi,
    hedges_l             = hedges_l,
    hedge_trigger_assets = ['QQQ'],
)
```

### Start Date Guardrail
`MB_START_DATE` auto-adjusted to `max(MB_START_DATE, first_quality_pit_cutoff + 12 months)`. Queries `quality_weights_pit` directly via ENGINE.

### 9 Strategies

| # | Name | Description |
|---|------|-------------|
| S0 | Baseline | Quality factor only, equal weight, monthly |
| S1 | Pure Alpha | Composite alpha, concentrated (2x), monthly |
| S2 | MVO | Covariance-optimized, monthly |
| S3 | Hybrid | Average of Alpha + MVO, monthly |
| S4 | Smart Hybrid | DD-regime switching Alpha/Hybrid/MVO, monthly |
| S5 | Dynamic | Signal-triggered rebalancing |
| S6 | Dyn+Hedge | Dynamic + macro hedge overlay |
| S7 | DD Policy | Dynamic + Hedge + multi-level DD de-grossing |
| S8 | Excl | DD Policy + MR momentum exclusion |

### Portfolio Regime (DD-driven)
```
dd ≥ -7.5%   → alpha portfolio
dd ≥ -17.5%  → hybrid portfolio
dd  < -17.5% → MVO portfolio
```

### ADVP Filter
Three-gate: exclusion (capacity < min_weight) → cap (min(advp_capacity, max_weight)) → tier redistribution.

```python
ADV_WINDOW = 20    # days for median dollar ADV
advp_cap   = 0.04  # 4% of daily $ volume
min_weight = 0.025
max_weight = 0.075
```

### Drawdown Policy (S7, S8)
```python
DD_LEVELS = [
    (0.175, 2/5),  # -17.5% → 60% gross
    (0.300, 3/7),  # -30.0% → 34% gross
    (0.350, 1/2),  # -35.0% → 17% gross
    (0.400, 2/3),  # -40.0% →  6% gross
    (0.450, 1/1),  # -45.0% →  0% gross
]
DD_REENTRY_PCT     = 0.075
DD_REENTRY_CONFIRM = 5
```

### Hedge Overlay (S6, S7, S8)
```python
HEDGE_RATIO = 0.25
MAX_HEDGE   = 0.50
EFF_FLOOR   = 0.75
CORR_FLOOR  = 0.50
BETA_WINDOW = 63
CORR_WINDOW = 63
```

### Consistency Diagnostics
Printed at every run start:
```
params_hash        : 44537355ff74
quality_fingerprint: 1314936_2017-01-02_2026-05-01
[HASH 2019-01-01] Quality: n=492 hash=e336ac5a
[HASH 2019-01-01] Value: n=464 hash=fd020c60
[HASH 2019-01-01] composite=b8e463a6 n=580
```

---

## 10. UTILITY: `wipe_caches.py`

Wipes all factor model caches for a clean full rebuild. Preserves source data and independent computations.

```python
exec(open('wipe_caches.py').read())
```

**Wiped:** all `v2_factor_residuals_*`, `v2_lambda_*`, `v2_ou_reversion_df`, `quality_scores_df`, `value_scores_df`, `quality_weights_pit`, `value_weights_pit`, `value_ic_bank`, `factor_ic_cache`, `mvo_daily_portfolios`, `mvo_x_snapshots`

**Preserved:** `dynamic_size_df`, `si_composite_df`, `valuation_metrics_anchors`, `valuation_consolidated`, `momentum_exclusions`, `macro_*`

---

## 11. DATABASE TABLES

### Factor Model (v2)
```
v2_factor_residuals_mkt      Mkt beta residuals          (date, ticker, resid)
v2_factor_residuals_quality  Quality residuals           (date, ticker, resid)
v2_factor_residuals_mom      Momentum residuals          (date, ticker, resid)
v2_factor_residuals_size     Size residuals              (date, ticker, resid)
v2_factor_residuals_value    Value residuals             (date, ticker, resid)
v2_factor_residuals_si       SI residuals                (date, ticker, resid)
v2_factor_residuals_vol      Vol residuals               (date, ticker, resid)
v2_factor_residuals_macro    Macro residuals             (date, ticker, resid)
v2_factor_residuals_sec      Sector residuals            (date, ticker, resid)
v2_factor_residuals_ou       OU residuals                (date, ticker, resid)
v2_ou_reversion_df           OU scores                   (date, ticker, ou_score)
v2_lambda_*                  Factor lambdas (10 tables)  (date, factor, lambda)
```

### Quality & Value
```
quality_scores_df            Quality composite scores    (date, ticker, score)
quality_weights_pit          PIT quality weights         (cutoff_date, regime, metric, weight)
value_scores_df              Value composite scores      (date, ticker, score)
value_weights_pit            PIT value weights           (cutoff_date, metric, weight)
value_ic_bank                Value IC cache              (anchor_date, metric, horizon, ic)
valuation_metrics_anchors    Quality fundamental data    (date, ticker, ...)
valuation_consolidated       Value fundamental data      (date, ticker, ...)
```

### Supporting
```
factor_ic_cache              Factor IC cache             (date, factor, horizon, ic)
dynamic_size_df              Market cap                  (date, ticker, size)
si_composite_df              Short interest composite    (date, ticker, score)
short_interest_data          Raw SI data                 (date, ticker, metric, value)
mvo_x_snapshots              X factor snapshots          Per model_version
mvo_daily_portfolios         Portfolio weights           Per params_hash
macro_indicators_daily       QBD + MOVE                  Per params_hash
macro_signal_daily           Hedge signals               Per params_hash
macro_fitted_params          Hedge params                Per params_hash
momentum_exclusions          Exclusion list              (date, ticker, score)
momentum_exclusions_processed Processed dates            (date)
```

---

## 12. GLOBAL CONSTANTS

```python
# Pipeline
MB_START_DATE    = pd.Timestamp('2019-01-01')  # adjusted by guardrail if needed
FM_START_DATE    = pd.Timestamp('2017-01-01')
AUM              = 5_000_000
TRADING_COST_BPS = 10
MB_MODEL_VER     = 'v2'

# Universe
MIN_STOCKS        = 150
_MIN_HISTORY_DAYS = 50
_STALE_RUN_LIMIT  = 3

# Quality regime
QF_MAV_WINDOW    = 252
QF_THRESHOLD     = 15        # bps
QF_RATE_COL      = '10Y RATE'  # bps column in Pxs_df

# Portfolio
top_n            = 25
rebal_freq       = 15
advp_cap         = 0.04
max_weight       = 0.075
min_weight       = 0.025
conc_factor      = 2.0
universe_mult    = 5
ic               = 0.03
risk_aversion    = 10
pca_var_threshold = 0.65

# DD Policy
DD_REENTRY_PCT      = 0.075
DD_REENTRY_CONFIRM  = 5
SH_DD_ALPHA         = 0.075
SH_DD_HYBRID        = 0.175

# Hedge
HEDGE_RATIO  = 0.25
MAX_HEDGE    = 0.50
EFF_FLOOR    = 0.75
CORR_FLOOR   = 0.50

# MR Exclusion
MR_CAP = 0.0
MR_K   = 0.5
```

---

## 13. NOTES AND CONVENTIONS

- **`quality_factor.py` retired:** all logic integrated into `factor_model_v2.py`
- **`'10Y RATE'` vs `USGG10YR`:** `'10Y RATE'` (bps) is used for regime signal only; `USGG10YR` is a macro factor in Step 8 — distinct columns, no ambiguity
- **Rate regime signal:** step function (0.0/0.5/1.0) based on 15bps threshold vs 252-day MAV of `'10Y RATE'`
- **Ticker format:** `TICKER US` in valuation tables; bare tickers in factor model tables
- **Lookahead bias:** eliminated by construction. No `bfill()` in scores or weights. PIT weights use only IC from anchors with complete forward windows before cutoff
- **Universe stability:** built from `sectors_s.index`, invariant to `Pxs_df` end date
- **Per-date active universe:** ≥50 days genuine price history, stale price detection (>3 consecutive identical closes)
- **Sparse factor fallback:** quality/value missing → prior step residuals carry forward; chain never breaks
- **DB writes:** always upsert (`ON CONFLICT DO NOTHING` or `DO UPDATE`)
- **Macro hedge:** in-memory `hedge_multi` — same kernel as `run_backtest` required
- **O-U batch saving:** 200-date batches, crash-safe; incremental uses 120-day DB lookback

---

## 14. COMPLETED WORK (Sessions through May 2026)

- [DONE] Factor model v2 — self-contained, no v1 kernel dependency
- [DONE] Quality PIT weights integrated inline after Step 1 in `factor_model_v2.run()`
- [DONE] Value PIT weights integrated inline after Step 4 in `factor_model_v2.run()`
- [DONE] `quality_factor.py` retired — all logic in `factor_model_v2.py`
- [DONE] Rate regime column renamed `USGG10YR` → `'10Y RATE'` (bps, no ambiguity with macro factor)
- [DONE] QF_THRESHOLD set to 15 bps (optimal from grid search)
- [DONE] Rate signal reverted to step function (0/0.5/1.0) matching original
- [DONE] IC computation matches original `derive_weights`: sector-ranked, top/bottom decile spread
- [DONE] Value warm-start fallback: 60/40 blend when only 1 metric passes IC filter
- [DONE] Value IC bank cached in DB — O(1) incremental updates
- [DONE] `_compute_ou_for_dates` return statement fixed (was missing, caused None returns)
- [DONE] O-U batch saving (200 dates) — crash-safe full rebuild
- [DONE] `force_rebuild_pit=True` flag in `run()` — wipes PIT caches inline
- [DONE] `wipe_caches.py` utility script
- [DONE] `_recompute_value_scores()` — value scores populated inline after PIT weight update
- [DONE] Quality/value fallback residuals when sparse (pre-data dates compute cleanly)
- [DONE] Both quality and value use "previous step" residuals for IC
- [DONE] Start date guardrail in `run_backtest` — first PIT cutoff + 12 months
- [DONE] Universe stability — `sectors_s`-based, invariant to `Pxs_df` end date
- [DONE] Per-date active universe with stale price detection
- [DONE] All `bfill()` removed from score/weight computation
- [DONE] Consistency diagnostic hashes printed for first 3 calc_dates
- [DONE] `params_hash` and `quality_fingerprint` printed at backtest start
- [DONE] 693 vs 697 universe discrepancy fixed
- [DONE] Yearly returns aligned with consistent 10-char column widths
- [DONE] Reference document updated to v8

---

## 15. MVO_BACKTEST — HYPERPARAMETERS AND run_backtest ARGUMENTS

### 15.1 Top-Level Hyperparameter Constants

All constants live at the top of `mvo_backtest.py` and serve as defaults for `run_backtest` arguments. Grouped by function:

#### Execution
| Constant | Default | Description |
|---|---|---|
| `MB_START_DATE` | `2019-01-01` | Backtest start date |
| `MB_TOP_N` | 25 | Default number of stocks in core portfolio |
| `MB_REBAL_FREQ` | 15 | Default rebalance frequency (trading days) |
| `MB_MODEL_VER` | `'v2'` | Factor model version for DB lookups |
| `AUM` | 5,000,000 | Starting AUM ($) |
| `TRADING_COST_BPS` | 10 | One-way trading cost (bps), scaled by gross exposure |

#### MVO Portfolio Construction
| Constant | Default | Description |
|---|---|---|
| `MVO_LOOKBACK` | 252 | Return history window for covariance estimation (trading days) |
| `MVO_EWMA_HL` | 126 | EWMA half-life for return covariance (trading days) |
| `MVO_PCA_VAR_THRESH` | 0.65 | PCA variance threshold — eigenvectors retained until this fraction explained |
| `MVO_DEFAULT_IC` | 0.04 | Default IC for Grinold-Kahn alpha scaling |
| `MVO_OVERLAP_TARGET` | 0.65 | Target portfolio overlap across covariance matrices |
| `MVO_MAX_WEIGHT` | 0.10 | Hard single-name weight cap |
| `MVO_MIN_WEIGHT` | 0.025 | Floor on non-zero single-name weight |
| `MVO_ZSCORE_CAP` | 2.50 | Composite z-score winsorisation cap |
| `MVO_MIN_MATRIX_COUNT` | 2 | Stock must appear in ≥N covariance matrices to be eligible |

#### F-Matrix (Factor Covariance)
| Constant | Default | Description |
|---|---|---|
| `_MB_F_LOOKBACK` | 60 | Months of monthly lambda history used for F matrix (5 years, strictly PIT) |
| `_MB_F_EWMA_HL` | 12 | EWMA half-life for F matrix factor returns (months). 12 months ≈ 87% weight on trailing year |

#### Regime (Smart Hybrid / Dynamic)
| Constant | Default | Description |
|---|---|---|
| `SH_DD_ALPHA` | 0.075 | Drawdown from HWM below which strategy enters hybrid regime |
| `SH_DD_HYBRID` | 0.175 | Drawdown from HWM below which strategy enters MVO regime |
| `SH_DD_EXIT_ALPHA` | 0.050 | Recovery from trough needed to exit hybrid → alpha |
| `SH_DD_EXIT_HYBRID` | 0.150 | Recovery from trough needed to exit MVO → hybrid |
| `SH_PERSIST_DAYS` | 3 | Consecutive days signal must persist before regime switch fires |

Note: All regime thresholds use **lifetime HWM** (`st['hwm']`) — the all-time NAV peak — ensuring uniform regime behaviour across all strategies (S_DYN, S_HEDGE, S_DD, S_EXCL).

#### Dynamic Rebalancing Triggers
| Constant | Default | Description |
|---|---|---|
| `DYN_TO_THRESHOLD_ALPHA` | 0.25 | Turnover threshold to trigger rebalance in alpha regime |
| `DYN_TO_THRESHOLD_HYBRID` | 0.30 | Turnover threshold in hybrid regime |
| `DYN_TO_THRESHOLD_MVO` | 0.35 | Turnover threshold in MVO regime |
| `DYN_VOLDIFF_CAP` | 0.175 | Maximum vol increase alongside TO trigger (anti-panic guard) |
| `DYN_MIN_HOLD_DAYS` | 10 | Minimum trading days between rebalances |

#### Drawdown (DD) Policy — S_DD and S_EXCL only
| Constant | Default | Description |
|---|---|---|
| `DD_LEVELS` | see below | Sequential de-gross levels — list of `(threshold, cut_fraction)` tuples |
| `DD_LEVEL_REGIME` | all `'mvo'` | Portfolio regime forced at each DD level |
| `DD_REENTRY_PCT` | 0.075 | Theoretical full-exposure portfolio must recover 7.5% from `theo_trough` |
| `DD_REENTRY_CONFIRM` | 5 | Consecutive days recovery condition must hold before re-entry |
| `DD_HWM_WINDOW` | 12 | Rolling HWM lookback **for de-gross trigger only** (months; 0 = lifetime) |

**DD_LEVELS default:**
```python
DD_LEVELS = [
    (0.175, 2/5),  # -17.5% rolling DD → cut 40% → 60% gross
    (0.300, 3/7),  # -30.0% rolling DD → cut 43% → ~34% gross
    (0.350, 1/2),  # -35.0% rolling DD → cut 50% → ~17% gross
    (0.400, 2/3),  # -40.0% rolling DD → cut 67% → ~6% gross
    (0.450, 1/1),  # -45.0% rolling DD → cut 100% → 0% gross
]
```
Levels fire sequentially. De-gross trigger uses `dd_rolling = nav / hwm_rolling - 1` (rolling 12-month HWM). Re-entry trigger uses `theo_nav / theo_trough - 1` (theoretical full-exposure portfolio from its own trough).

#### Universe Filters
| Constant | Default | Description |
|---|---|---|
| `MOM_FILTER` | 2 | Alpha/hybrid candidate pool = `MOM_FILTER × top_n` stocks by momentum |
| `QUALITY_FLOOR` | 0.0 | Minimum quality score for universe inclusion (0 = disabled) |
| `MR_K` | 0.5 | Mean reversion soft-penalty strength (0 = disabled) |
| `MR_CAP` | 0.0 | Top fraction of overextended names hard-excluded (0 = disabled) |

#### Hedge Engine
| Constant | Default | Description |
|---|---|---|
| `BETA_WINDOW` | 63 | Rolling window for beta estimation (trading days) |
| `CORR_WINDOW` | 63 | Rolling window for portfolio-instrument correlation ranking |
| `EFF_MAV_WINDOW` | 20 | MAV window for smoothing effectiveness scores |
| `EFF_FLOOR` | 0.75 | Minimum effectiveness score for instrument to qualify |
| `CORR_FLOOR` | 0.50 | Minimum correlation to portfolio to qualify |
| `HEDGE_RATIO` | 0.25 | Hedge size per qualifying instrument (fraction of NAV) |
| `MAX_HEDGE` | 0.50 | Maximum total hedge across all instruments (fraction of NAV) |
| `TRIGGER_ASSETS` | `['QQQ','SPY']` | Assets checked first for hedge trigger (must pass corr + eff filters) |

#### Tier 2 Overlay
| Constant | Default | Description |
|---|---|---|
| `TIER2_T` | 0 | Total stocks including overlay (0 = overlay disabled) |
| `TIERONE_ALLOC` | 0.90 | Fraction of portfolio allocated to tier 1 core (remainder to overlay) |
| `CORR_FILTER` | 0.50 | Fraction of non-core universe retained after intra-tier correlation filter |

---

### 15.2 run_backtest Arguments

`run_backtest(Pxs_df, sectors_s, weights_by_year, regime_s, **kwargs)` — all keyword arguments shown with their defaults.

#### Required positional
| Argument | Description |
|---|---|
| `Pxs_df` | Price/macro panel DataFrame (dates × tickers+macros) |
| `sectors_s` | Series mapping ticker → sector label |
| `weights_by_year` | PIT IC weights from `run_ic_study_alpha` — dict or DataFrame of factor weights by date |
| `regime_s` | Rate regime Series (0.0/0.5/1.0) by date |

#### Optional inputs
| Argument | Default | Description |
|---|---|---|
| `volumeTrd_df` | `None` | Dollar volume panel for ADVP calculations |
| `weights_by_date` | `None` | Full PIT weights by date (overrides weights_by_year if provided) |
| `hedge_multi` | `None` | Hedge signal dict from `run_macro_hedge_cached` — pass to enable hedge strategies |
| `hedges_l` | `None` | List of hedge instrument tickers |

#### Mode
| Argument | Default | Description |
|---|---|---|
| `mode` | `'incremental'` | `'incremental'` appends new days; `'rebuild'` wipes and rebuilds all caches |
| `rebuild_cov` | `False` | Force covariance matrix rebuild — use only when new stocks added to DB |

#### Portfolio construction
| Argument | Default | Description |
|---|---|---|
| `top_n` | 25 | Core portfolio size |
| `universe_mult` | 5 | MVO candidate pool = `universe_mult × top_n` |
| `conc_factor` | 2.0 | Pure Alpha concentration — score²-based vs equal weight |
| `prefilt_pct` | 0.5 | Pre-filter fraction of universe by quality score before composite ranking |
| `min_weight` | 0.025 | Minimum non-zero single-name weight |
| `max_weight` | 0.10 | Maximum single-name weight (hard cap) |
| `ic` | 0.04 | Grinold-Kahn IC for alpha scaling |
| `zscore_cap` | 2.50 | Composite z-score winsorisation |
| `pca_var_threshold` | 0.65 | PCA variance threshold for factor risk model |
| `risk_aversion` | 10 | MVO risk aversion parameter λ |
| `min_cov_matrices` | 2 | Minimum covariance matrices required for stock eligibility |
| `model_version` | `'v2'` | Factor model version |
| `advp_cap` | 0.04 | ADVP cap — max % of median daily dollar volume per stock |
| `rebal_freq` | 15 | Static strategy rebalance frequency (trading days) |
| `mom_filter` | 2 | Candidate pool multiplier for alpha/hybrid strategies |
| `quality_floor` | 0.0 | Quality floor filter (0 = disabled) |

#### Dynamic rebalancing
| Argument | Default | Description |
|---|---|---|
| `min_hold_days` | 10 | Minimum days between rebalances |
| `to_thresh_alpha` | 0.25 | Turnover trigger in alpha regime |
| `to_thresh_hybrid` | 0.30 | Turnover trigger in hybrid regime |
| `to_thresh_mvo` | 0.35 | Turnover trigger in MVO regime |
| `voldiff_cap` | 0.175 | Max portfolio vol increase permitted alongside TO trigger |

#### Regime switching
| Argument | Default | Description |
|---|---|---|
| `sh_dd_alpha` | 0.075 | Enter hybrid below this lifetime-HWM DD |
| `sh_dd_hybrid` | 0.175 | Enter MVO below this lifetime-HWM DD |
| `sh_dd_exit_alpha` | 0.050 | Recovery to exit hybrid → alpha |
| `sh_dd_exit_hybrid` | 0.150 | Recovery to exit MVO → hybrid |
| `sh_persist_days` | 3 | Days signal must persist before switch |

#### Drawdown policy
| Argument | Default | Description |
|---|---|---|
| `dd_levels` | DD_LEVELS | Sequential de-gross levels (threshold, cut_fraction) |
| `dd_level_regime` | all `'mvo'` | Regime forced at each DD level |
| `dd_reentry_pct` | 0.075 | Theoretical recovery threshold for re-entry |
| `dd_reentry_confirm` | 5 | Confirmation days for re-entry |
| `dd_hwm_window` | 12 | Rolling HWM window for de-gross trigger (months; 0 = lifetime) |

#### Hedge
| Argument | Default | Description |
|---|---|---|
| `eff_floor` | 0.75 | Effectiveness floor for instrument selection |
| `corr_floor` | 0.50 | Correlation floor for instrument selection |
| `hedge_ratio` | 0.25 | Per-instrument hedge size (fraction of NAV) |
| `max_hedge` | 0.50 | Total hedge cap (fraction of NAV) |
| `hedge_trigger_assets` | `['QQQ','SPY']` | Trigger assets checked first |

#### MR exclusion
| Argument | Default | Description |
|---|---|---|
| `mr_k` | 0.5 | Soft-penalty strength for overextended names |
| `mr_cap` | 0.0 | Hard exclusion cap — top fraction excluded (0 = disabled) |

#### Overlay
| Argument | Default | Description |
|---|---|---|
| `tier2_t` | 0 | Total portfolio size including overlay (0 = disabled) |
| `tierone_alloc` | 0.90 | Tier 1 allocation fraction |

#### Misc
| Argument | Default | Description |
|---|---|---|
| `aum` | 5,000,000 | Starting AUM ($) |
| `trading_cost_bps` | 10 | One-way trading cost (bps), deducted from NAV at rebalance, scaled by gross |

---

### 15.3 Key Design Principles (v9, May 2026)

- **SQL for storage only** — all data manipulation in pandas; SQL used only for load/save
- **Trading costs are real** — `_record_cost` deducts `turnover × cost_bps × gross` from NAV immediately at rebalance
- **Dual HWM** — `hwm` (lifetime) drives regime for all strategies; `hwm_rolling` (12-month) drives de-gross trigger for S_DD/S_EXCL only — fully decoupled
- **Uniform regime** — all dynamic strategies (S_DYN, S_HEDGE, S_DD, S_EXCL) use identical lifetime HWM for regime switching, ensuring Dyn+Hedge and DD Policy have identical portfolio construction when DD never fires
- **Theoretical re-entry** — `theo_nav / theo_trough` tracks the full-exposure portfolio from its own lowest point, not the real (de-grossed) NAV trough
- **`_select_hedge_instruments` embedded** — `mvo_backtest.py` is fully self-contained; no external hedge_engine.py dependency
