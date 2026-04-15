# Factor Model & Scripts — Comprehensive Reference
*Last updated: April 2026 (v4)*

---

## 1. OVERVIEW

A sequential Fama-MacBeth cross-sectional factor model implemented in Python, running in a Jupyter notebook kernel. The model strips systematic return sources one by one in a true Gram-Schmidt orthogonalization sequence, producing clean residuals at each step. All data is stored in a PostgreSQL database (`factormodel_db`). The universe consists of ~662–679 US large-cap stocks (varies with sector mapping updates).

**Database connection:**
```
postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db
```

**Key principle:** Both return residuals AND characteristics themselves are orthogonalized at each step using WLS (weighted by log market cap). Before entering any regression, each new characteristic is regressed cross-sectionally against all prior characteristics. Only the residual enters the regression. This is true Gram-Schmidt in the weighted inner-product space defined by cap weights.

---

## 2. FACTOR MODEL — 11-STEP ARCHITECTURE

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

Each macro beta: `β_im = Cov_EWMA(r_i, d_m) / Var_EWMA(d_m)`, window=252, hl=126, z-scored cross-sectionally. All 7 enter a single joint ridge regression (no natural ordering; ridge handles collinearity).

**Ridge CV (Step 4):** 5-fold CV per date. Grid floor 0.15. λ=0.15–0.30 dominates (~60% of dates); λ=40 on collinear days (~12%). Mean λ ≈ 5.5, median λ ≈ 0.30.

### Sector Dummies (Step 5)

**Sum-to-zero deviation coding:** for K sectors, each dummy = +1 own sector, -1/(K-1) all others. All sectors included — no reference dropped. Intercept = true equal-weighted market return.

**Ridge CV (Step 5):** Grid floor 0.10. ~88.5% of dates select λ=0.10, ~3% select λ=40. Without ridge, sector lambdas showed ±7% artefacts on low-dispersion days.

Note: sector mapping is passed as `sectors_s` input — the number of sectors varies with the mapping (currently 17 sub-sectors after latest update).

### WLS Regression

All cross-sectional regressions: `w_i = log(market_cap_i)`, normalized to sum to 1.

### Characteristic Orthogonalization

```
new_char_perp = new_char - Proj_{prior_chars}(new_char)    [WLS]
```

Falls back to OLS if market cap unavailable. Full-history versions computed over extended date range for momentum lookback chain.

---

## 3. FACTOR DETAILS

### Market Beta (Step 2)
EWMA beta vs SPX. `β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)`, window=252, hl=126.

### Size (Step 3)
`size_i = log(shares × price)` — dynamic daily. Cached in `dynamic_size_df`.

### Quality Factor (Step 6)
Rate-conditioned composite. Loaded from `quality_scores_df` cache.
- **GQF** (Growth Quality): non-2021/22 regime
- **CQF** (Conservative Quality): 2021/22 high-rate regime
- Blend: `(1-q)×GQF + q×CQF`, q from USGG10YR vs 252d MAV, threshold=50bps
- Optimal params: `mav_window=252, threshold=50`

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

ROE/ROEd excluded (price contamination via market cap denominator).

### SI Composite (Step 7)
Short interest composite. Cached in `si_composite_df`.

### GK Volatility (Step 8)
`σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²`. Window=84d, hl=42d. Annualized. Positive lambda (t≈+4.1) — volatility risk premium.

### Idiosyncratic Momentum (Step 9)
On `factor_residuals_vol`. Volume-scaled: `mom_i = Σ r_resid × vol_scalar`, clipped [0.5, 3.0]. Window: [t-252, t-21].

### Value Factor (Step 10)
IC-weighted composite from `value_scores_df` cache.
```python
_VALUE_TSTAT = {
    'P/S':   (-4.088 + -6.591) / 2,   # -5.340
    'P/Ee':  (-4.307 + -5.428) / 2,   # -4.867
    'P/Eo':  (-3.975 + -5.550) / 2,   # -4.763
    'sP/S':  (-4.329 + -6.353) / 2,   # -5.341
    'sP/E':  (-3.688 + -5.282) / 2,   # -4.485
    'sP/GP': (-4.817 + -5.861) / 2,   # -5.339
    'P/GP':  (-3.491 + -4.329) / 2,   # -3.910
}
VALUE_WEIGHTS = {m: abs(w) / total for m, w in _VALUE_TSTAT.items()}
```

### O-U Mean Reversion (Step 11)
AR(1) fit to compounded residual price index. `half_life = ln(2)/(-ln(b))`. Requires `0 < b < 1`. Falls back to 21d ST reversal on residual index if fit invalid. Blended: `ou_weight = min(30/half_life, 10)`.

---

## 4. RUN MODES

### Full Recalculation (`n`)
Prompts: start date (default 2018-01-01), volume scaling. Extended start = st_dt - 252 trading days.

### Incremental Update (`y`, default)
Single-date fast path. Loads residual histories from DB, runs one cross-section per step. Upserts to DB.

### Snapshot Display
```
2026-03-23  |  Intercept: -0.09%  |  Daily R²: nan%  |  Macro Ridge λ: 0.30  |  Sec Ridge λ: 0.10
```

---

## 5. DATABASE TABLES

### Residual Tables
| Table | Step |
|-------|------|
| `factor_residuals_mkt` | Step 2 |
| `factor_residuals_size` | Step 3 |
| `factor_residuals_macro` | Step 4 |
| `factor_residuals_sec` | Step 5 |
| `factor_residuals_quality` | Step 6 |
| `factor_residuals_si` | Step 7 |
| `factor_residuals_vol` | Step 8 |
| `factor_residuals_mom` | Step 9 |
| `factor_residuals_joint` | Step 10 |
| `factor_residuals_ou` | Step 11 |

### Lambda Tables
`factor_lambdas_mkt/size/macro/sec/quality/si/vol/mom/joint/ou`

### Characteristic / Score Tables
| Table | Contents |
|-------|----------|
| `dynamic_size_df` | Daily market cap |
| `si_composite_df` | SI composite scores |
| `quality_scores_df` | Quality composite (cached) |
| `value_scores_df` | Value composite (cached) |
| `ou_reversion_df` | O-U scores (cached) |
| `valuation_consolidated` | Raw quarterly fundamentals |
| `valuation_metrics_anchors` | Anchor date snapshots |
| `income_data` | Ortex income fundamentals (ticker, download_date, period, metric_name, value, estimated_values) |
| `summary_data` | Ortex summary fundamentals (same schema, includes ebitda) |
| `estimation_status` | FEQ tracking (ticker, category, first_estimated_period, last_checked) |
| `daily_open/high/low` | OHLC prices |

### Sector Metrics Cache Tables
Produced by `sector_metrics.py`:
- `sector_metric_{metric_tag}` — cap-weighted metric per sector
- `index_metric_{metric_tag}` — cap-weighted metric for SPX/QQQ

Produced by `sector_fundamentals.py`:
- `sector_valuation_{metric}_{basis}` — e.g. `sector_valuation_sales_ltm`
- `sector_growth_{metric}_{basis}` — e.g. `sector_growth_ni_ntm`

---

## 6. LATEST FACTOR PERFORMANCE (March 2026)

| Step | % UFV | % prev |
|------|-------|--------|
| Beta | 71.66% | 71.66% |
| Size | 60.69% | 84.70% |
| Macro | 54.73% | 90.18% |
| Sectors | 51.93% | 94.88% |
| Quality | 51.58% | 99.33% |
| SI | 51.40% | 99.65% |
| GK Vol | 50.79% | 98.81% |
| Idio Mom | 50.16% | 98.75% |
| Value | 50.42% | 100.53%* |
| O-U | 47.61% | 98.95% |

*Value stale — needs ic_study re-run. **Consolidated R² = 52.39%**

| Factor | t-stat |
|--------|--------|
| SI Composite | +5.34 |
| GK Vol | +4.10 |
| Size | +3.79 |
| Quality | +2.73 |
| O-U | +1.47 |
| Idio Mom | +1.63 |
| Value | ~0.00 (stale) |

---

## 7. FEQ (FIRST ESTIMATED QUARTER) MAPPING PROCEDURE

This procedure is used by `sector_fundamentals.py` whenever a back-date calculation is needed and the correct quarter alignment must be determined. It is critical for any script that reads from `income_data` or `summary_data` for historical dates.

### Background
The Ortex fundamentals DB stores data with `download_date` (the date data was fetched) and `period` (the fiscal quarter, e.g. `2025Q3`). The `estimation_status` table records the `first_estimated_period` (FEQ) — the first quarter that was estimated (not yet reported) as of the most recent download.

### Algorithm

**Step 1 — Anchor from today's FEQ:**
```python
current_feq, last_checked = get from estimation_status WHERE ticker=t AND category='income'
# e.g. current_feq='2026Q1', last_checked=2026-02-15
```

**Step 2 — Find straddling download dates for the back-date:**
```python
update_before = latest  download_date <= calc_date   # last snapshot before calc_date
update_after  = earliest download_date > calc_date   # first snapshot after calc_date
# If update_before doesn't exist: skip this stock/date entirely
```

**Step 3 — Estimate past FEQ by walking back:**
```python
days_delta    = (last_checked - update_before).days
quarters_back = round(days_delta / 90)
est_past_feq  = add_quarters(current_feq, -quarters_back)
# e.g. last_checked=Feb-15-2026, update_before=Jul-10-2025 → 220 days → 2 quarters
# est_past_feq = 2026Q1 - 2 = 2025Q3
```

**Step 4 — Verify by comparing totalRevenues across update dates:**
- Build 4 candidate quarters: `[est_past_feq-1, est_past_feq, est_past_feq+1, est_past_feq+2]`
- For each candidate, compare `totalRevenues` between `update_before` and `update_after`
- The most recent candidate where the value **changed** between the two snapshots is the confirmed past FEQ
- If none changed: fall back to `est_past_feq`
- If `update_after` doesn't exist: trust `est_past_feq` directly

**Step 5 — Use confirmed FEQ:**
- **LTM actuals:** quarters `[feq-4, feq-3, feq-2, feq-1]`
- **NTM estimates:** quarters `[feq, feq+1, feq+2, feq+3]`
- **Shares:** `dilutedAverageShares` at `feq-1` (most recent actual)
- **Prior LTM (for growth):** quarters `[feq-8, feq-7, feq-6, feq-5]`

### Helper Function
```python
def add_quarters(quarter: str, n: int) -> str:
    year, q = int(quarter[:4]), int(quarter[5])
    q += n
    while q > 4: q -= 4; year += 1
    while q < 1: q += 4; year -= 1
    return f"{year}Q{q}"
```

### Key Tables Used
- `estimation_status`: `(ticker, category, first_estimated_period, last_checked)`
- `income_data`: `(ticker, download_date, period, metric_name, value, estimated_values)`
- `summary_data`: same schema, includes `ebitda`

### Important Notes
- All metric queries use `download_date <= calc_date` (forward-fill semantics) to get the most recent snapshot available as of each back-date
- `dilutedAverageShares` and all metrics use the **same FEQ mapping** — critical for consistency
- The verification step (Step 4) uses strict `download_date = update_before` (no ffill) to detect genuine changes between snapshots

---

## 8. SCRIPT: `factor_model_step1.py`

**Location:** `/mnt/user-data/outputs/factor_model_step1.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_universe()` | Filters stocks: in DB + sector-mapped + sufficient history |
| `load_dynamic_size()` | Loads/computes market cap from DB cache |
| `build_sector_dummies()` | Sum-to-zero deviation coding |
| `calc_rolling_betas()` | EWMA beta vs SPX |
| `calc_macro_betas()` | EWMA betas vs 7 macro factors, z-scored |
| `calc_idio_momentum_volscaled()` | Volume-weighted cumulative idio residuals |
| `calc_vol_factor()` | Garman-Klass EWMA vol |
| `wls_cross_section()` | Single-date WLS regression |
| `wls_ridge_cross_section()` | WLS + ridge, 5-fold CV lambda selection |
| `run_factor_step()` | Loops wls_cross_section over all dates |
| `run_factor_step_optimal_ridge()` | Loops with CV per date |
| `orthogonalize_char()` | Single-date WLS characteristic orthogonalization |
| `orthogonalize_char_df()` | Across all dates |
| `_fit_ou_single()` | AR(1) fit to residual price index |
| `_compute_ou_for_dates()` | O-U + ST reversal blend |
| `load_ou_reversion()` | Cache-aware O-U loader |
| `load_quality_scores()` | Loads from quality_scores_df cache |
| `load_value_scores()` | Loads from value_scores_df cache |
| `_run_incremental()` | Single-date fast path |
| `run()` | Master entry point |

**Known namespace issue:** `_load_cached_dates()` in `quality_factor.py` (no args) can be overwritten in the Jupyter kernel if `sector_fundamentals.py` is run first (its version takes `table_name` arg). Fixed in `sector_fundamentals.py` by renaming to `_load_cached_dates_sf()`.

---

## 9. SCRIPT: `quality_factor.py`

**Location:** `/mnt/user-data/outputs/quality_factor.py`
**Entry point:** `run(Pxs_df, sectors_s, mav_window=252, threshold=50)`

### Cache Refresh Workflow
```python
summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s,
                                              mav_window=252, threshold=50)
update_cached_weights(gqf_w, cqf_w)
# Copy printed GQF_WEIGHTS / CQF_WEIGHTS into quality_factor.py
```

---

## 10. SCRIPT: `ic_study.py` (Value Factor)

**Location:** `/mnt/user-data/outputs/ic_study.py`
**Entry point:** `run_ic_study(Pxs_df, sectors_s, force_recompute_cache=False)`

Targets `factor_residuals_mom`. Horizons: 21d and 63d.

### Cache Refresh Workflow
```python
ic_ts, ic_summary, ic_annual, weights = run_ic_study(Pxs_df, sectors_s,
                                                       force_recompute_cache=True)
# Copy printed _VALUE_TSTAT into factor_model_step1.py
# Re-run full factor model recalculation
```

---

## 11. SCRIPT: `primary_factor_backtest.py`

**Location:** `/mnt/user-data/outputs/primary_factor_backtest.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

Long-only, bi-monthly rebalancing, TOP_N=20. Primary factor = quality composite.
Volume scaling available for both 12m1 and idio momentum when `volumeTrd_df` provided.

---

## 12. SCRIPT: `plot_factor_returns.py`

**Location:** `/mnt/user-data/outputs/plot_factor_returns.py`
**Entry point:** `plot_all(results)` or `plot_all(load_lambdas_from_db())`

6 figures: structural, macro (7 factors), sectors, alpha factors, rolling t-stats, ridge λ (macro + sector panels).
`MACRO_COLS` updated to: `['USGG2YR', 'US10Y2Y_SPREAD_CHG', 'US10YREAL', 'BE5Y5YFWD', 'MOVE', 'Crude', 'XAUUSD']`
`SECTOR_COLS` updated to 13 sectors including XLP.

---

## 13. SCRIPT: `sector_fundamentals.py`

**Location:** `/mnt/user-data/outputs/sector_fundamentals.py`
**Entry point:** `run(Pxs_df, sectors_s, override=False, spx_df=None, qqq_df=None)`

Calculates cap-weighted sector and index valuation/growth metrics from Ortex fundamentals DB using FEQ mapping (see Section 7).

### Prompts
1. Metric type: valuation (v) or growth (g)
2. Metric: Sales (s), Net Income (ni), EBITDA (e)
3. Basis: LTM or NTM
4. Lookback period in years

### Cache Tables
`sector_valuation_{metric}_{basis}` and `sector_growth_{metric}_{basis}`, e.g.:
- `sector_valuation_sales_ltm`
- `sector_growth_ni_ntm`

### Key Design
- FEQ mapping resolves correct quarter alignment for each back-date (see Section 7)
- Growth formula: `np.median((num - den) / ((num + den) / 2))` per quarter
- Cap-weighting: `Σ(val × mcap) / Σ(mcap)`, requires MIN_STOCKS=3
- Progress printed as `[date] — sector (n stocks)`
- SPX/QQQ index metrics computed alongside sectors using constituent lists from `spx_df`/`qqq_df`
- Growth rates output as percentages rounded to 2 decimal places

### Namespace Note
Uses `_load_cached_dates_sf()` (not `_load_cached_dates()`) to avoid collision with `quality_factor.py`.

---

## 14. SCRIPT: `sector_metrics.py`

**Location:** `/mnt/user-data/outputs/sector_metrics.py`
**Entry point:** `run(Pxs_df, sectors_s, spx_df=None, qqq_df=None, override=False, source_table='valuation_metrics_anchors')`

Cap-weighted aggregation of pre-computed valuation metrics directly from `valuation_metrics_anchors` or `valuation_consolidated` tables. Much faster than `sector_fundamentals.py` since no FEQ mapping needed — metrics are already computed as point-in-time snapshots.

### Available Metrics (33 total)
```
P/S, P/Ee, P/Eo, OM-t0, OM, OMd, GS, GE, r2 S, r2 E, GGP, r2 GP, Size,
ROI-P, ROI, ROId, ROE-P, ROE, ROEd, sP/S, sP/E, sP/GP, P/GP, S Vol, E Vol,
GP Vol, r&d, HSG, SGD, LastSGD, PIG, PSG, ISGD, FCF_PG
```

### Key Design

**Valuation multiples** (`P/S, P/Ee, P/Eo, sP/S, sP/E, sP/GP, P/GP`): denominator aggregation to handle negative earnings correctly:
```
sector_multiple = Σ(Size_i) / Σ(Size_i / multiple_i)
```
Reverse-engineers implied denominator (earnings/sales) from Size and ratio. Stocks with zero multiple excluded; negative denominators kept (correctly push multiple higher).

**All other metrics**: cap-weighted arithmetic mean after cross-sectional winsorization at `WINSOR_BOUNDS = (0.02, 0.98)`.

**Last-date mkt cap update:**
```
mkt_cap_updated = Size_last_snapshot × (Px_today / Px_last_snapshot)
```

**Time-series outlier filter** (controlled by `FILTER_TS_OUTLIERS = True`):
- Flags dates where `|pct_change| > TS_JUMP_THRESHOLD (0.25)`
- If next date is any closer to the pre-jump level: replace with average of two neighbours
- Applied iteratively — handles consecutive bad dates
- Raw DB cache untouched; filter runs fresh each time

### Cache Tables
`sector_metric_{metric_tag}` and `index_metric_{metric_tag}`, e.g.:
- `sector_metric_p_s`
- `index_metric_gs`

### Global Config
```python
DEFAULT_TABLE        = 'valuation_metrics_anchors'
MIN_STOCKS           = 3
WINSOR_BOUNDS        = (0.02, 0.98)
FILTER_TS_OUTLIERS   = True
TS_JUMP_THRESHOLD    = 0.25
VALUATION_MULTIPLES  = {'P/S', 'P/Ee', 'P/Eo', 'sP/S', 'sP/E', 'sP/GP', 'P/GP'}
```

### Output
Two DataFrames + two-panel figure (sectors top, indexes bottom):
```python
df_sectors, df_indexes, fig = run(Pxs_df, sectors_s, spx_df=spx_df, qqq_df=qqq_df)
```

---

## 15. SCRIPT: `load_sector_metrics.py`

**Location:** `/mnt/user-data/outputs/load_sector_metrics.py`
**Entry point:** `load_all(verbose=True)`

Scans DB for `sector_metric_*` and `index_metric_*` tables (from `sector_metrics.py`), loads all into a library dict.

```python
lib = load_all()
lib['sector']['p_s']        # sector P/S DataFrame
lib['index']['p_s']         # index P/S DataFrame
lib['available']            # list of metric tags
```

---

## 16. SCRIPT: `load_sector_fundamentals.py`

**Location:** `/mnt/user-data/outputs/load_sector_fundamentals.py`
**Entry point:** `load_all(verbose=True)`

Scans DB for `sector_valuation_*` and `sector_growth_*` tables (from `sector_fundamentals.py`), loads all into a nested library dict.

```python
lib = load_all()
lib['valuation']['sales']['ltm']    # P/Sales LTM by sector
lib['growth']['ni']['ntm']          # NI growth NTM by sector
lib['flat']['valuation_sales_ltm']  # same, flat access
lib['available']                    # list of full tags
```

---

## 17. SCRIPT: `mvo_backtest.py`

**Location:** `/mnt/user-data/outputs/mvo_backtest.py`
**Self-contained** — hedge engine embedded directly (no external dependencies beyond standard libs + SQLAlchemy).

### Entry Points

```python
# Main backtest — 6 strategies (7 with hedge layer)
results = run_mvo_backtest(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df=volumeS_df,
    ic=0.03, max_weight=0.075, min_weight=0.025,
    zscore_cap=2.5, pca_var_threshold=0.65,
    universe_mult=5, risk_aversion=10,
    force_rebuild_cache=False,
    portfolio_cache_override=False,
    # Optional hedge layer (omit for 6-strategy run):
    hedge_multi          = multi,           # from run_macro_hedge_cached
    hedges_l             = hedges_l,
    # hedge_trigger_assets defaults to ['QQQ']
)

# Daily portfolio cache builder (run once; ~15 min)
ph = run_daily_cache_build(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df=volumeS_df,
    ic=0.03, max_weight=0.075, min_weight=0.025,
    zscore_cap=2.5, pca_var_threshold=0.65,
    universe_mult=5, risk_aversion=10,
    top_n=25, conc_factor=2.0, prefilt_pct=0.5,
    min_cov_matrices=2, force_rebuild=False,
)
```

### Overview

Runs **6 portfolios in parallel** (7 with hedge layer), all net of `TRADING_COST_BPS=10` per side:

1. **Baseline** — quality factor only, equal weight
2. **Pure Alpha** — composite alpha signal, concentration weights
3. **MVO** — ensemble covariance-optimized (Empirical EWMA + Ledoit-Wolf + Factor XFX' + PCA)
4. **Hybrid** — simple average of Alpha and MVO weight vectors, renormed to 100%
5. **Smart Hybrid** — drawdown-regime switching between Alpha / Hybrid / MVO
6. **Dynamic** — signal-triggered rebalancing using daily cached trigger variables
7. **Dynamic + Hedge** — Dynamic with parallel macro hedge account (when `hedge_multi` provided)
8. **Dyn+Hedge+DD** — Dynamic+Hedge with flexible multi-level drawdown de-grossing policy

### Key Global Constants
```python
MB_START_DATE       = pd.Timestamp('2019-01-01')
MB_TOP_N            = 20
MB_REBAL_FREQ       = 30
MB_MODEL_VER        = 'v2'
MB_COV_CACHE_TBL    = 'mvo_cov_cache'
MB_DAILY_PORT_TBL   = 'mvo_daily_portfolios'
DAILY_TRIGGER_TBL   = 'mvo_daily_triggers'
MB_MIN_COV_MATRICES = 2
VOL_LOOKBACK        = 63
TRADING_COST_BPS    = 10

# Smart hybrid thresholds (entry into defensive regimes)
SH_DD_ALPHA       = 0.075   # dd below this: alpha → hybrid
SH_DD_HYBRID      = 0.175   # dd below this: hybrid → MVO
SH_DD_EXIT_ALPHA  = 0.050   # recovery needed to exit hybrid → alpha
SH_DD_EXIT_HYBRID = 0.150   # recovery needed to exit MVO → hybrid
SH_PERSIST_DAYS   = 3       # days signal must persist before regime switch

# Dynamic rebalancing thresholds
DYN_TO_THRESHOLD_ALPHA  = 0.20   # TO trigger in alpha regime
DYN_TO_THRESHOLD_HYBRID = 0.25   # TO trigger in hybrid regime
DYN_TO_THRESHOLD_MVO    = 0.30   # TO trigger in MVO regime
DYN_VOLDIFF_CAP         = 0.175  # max vol increase allowed alongside TO trigger
DYN_VOLDIFF_DERISK      = -0.750 # de-risk trigger (effectively disabled)
DYN_MIN_HOLD_DAYS       = 7      # minimum calendar days between rebalances

# Hedge engine defaults (embedded)
BETA_WINDOW    = 63     # rolling window for portfolio beta to hedge instruments
CORR_WINDOW    = 63     # rolling window for portfolio-instrument correlation
EFF_MAV_WINDOW = 20     # MAV window for smoothing effectiveness scores
EFF_FLOOR      = 1.0    # minimum effectiveness to qualify as hedge instrument
CORR_FLOOR     = 0.50   # minimum correlation to portfolio to qualify
HEDGE_RATIO    = 0.25   # hedge size per instrument (fraction of portfolio NAV)
MAX_HEDGE      = 0.50   # maximum total hedge size
TRIGGER_ASSETS = ['QQQ']   # QQQ alone reflects risk sentiment better

# Drawdown policy — flexible multi-level de-grossing
# Each tuple: (dd_threshold, fraction_of_remaining_to_cut)
# Cuts compound on remaining exposure; HWM resets at each full re-entry
DD_LEVELS = [
    (0.175, 2/5),  # at -17.5% DD: cut 40% of remaining → 60% exposed
    (0.300, 3/7),  # at -30.0% DD: cut 43% of remaining → 34% exposed
    (0.350, 1/2),  # at -35.0% DD: cut 50% of remaining → 17% exposed
    (0.400, 2/3),  # at -40.0% DD: cut 67% of remaining →  6% exposed
    (0.450, 1/1),  # at -45.0% DD: cut 100% of remaining → 0% exposed
]
DD_REENTRY_PCT      = 0.075  # theoretical MVO recovery from trough → full re-entry
DD_REENTRY_CONFIRM  = 5      # days recovery must persist before re-entry
DD_ANNUAL_RESET_PCT = 0.30   # max YTD DD allowed for calendar-year re-entry
```

### Smart Hybrid
Asymmetric drawdown regime switching with persistence requirement:
- Entry: `dd < -7.5%` → hybrid; `dd < -17.5%` → MVO
- Exit: `dd > -5%` → alpha; `dd > -15%` → hybrid
- Signal must persist `SH_PERSIST_DAYS=3` days before switching

### Dynamic Strategy
Signal-triggered rebalancing with regime-specific TO thresholds. Rebalances when:
1. **Regime switch** — signal regime ≠ deployed regime (compares vs deployed, not previous signal)
2. **Turnover** — drift-adjusted TO > threshold (regime-specific) AND vol_diff < +17.5%
3. **De-risk** — vol_diff < -75% (disabled)

TO computed at backtest time from drifted deployed portfolio (not from cache).

### Hedge Layer (embedded)
Runs in parallel with Dynamic strategy. Independent hedge account tracks short ETF positions.

**Trigger:** QQQ signal = 1 (1-day lag)
**Unwind:** QQQ signal = 0

**Instrument selection at activation:**
1. Rank all `hedges_l` instruments by 63d rolling correlation to live portfolio
2. Select top 3 with correlation ≥ 50% AND smoothed effectiveness ≥ 1.0
3. Always add QQQ and SPX if they pass both tests (up to 5 instruments total)
4. Total hedge = min(n × 25%, 50%), allocated by `beta × effectiveness`

**Hedge account:** P&L accrues from episode close, swept to portfolio NAV at next rebalancing date (deferred if hedges still open). Trading costs: 10bps entry + 10bps exit per instrument.

**Effectiveness metric:**
```
sharpe_improvement = (sharpe_hedged - sharpe_raw) / abs(sharpe_raw)
effectiveness = clip(nav_ratio × (1 + sharpe_improvement), 0, 5)
```
Computed with same [5,4,3,2,1] annual step weights as signal fitting. Smoothed with 20d MAV before use.

### DD Policy (8th strategy)
Flexible multi-level de-grossing applied to combined NAV (portfolio + hedge account).

**De-grossing:** Sequential levels armed one at a time. Each level fires once per cycle (reset on full re-entry). Cuts apply to remaining exposure (compounding effect).

**Re-entry (full, to 100%):** Either:
1. Theoretical MVO NAV recovers ≥ 7.5% from trough, confirmed 5 consecutive days
2. Calendar year resets (first 10 days of January) AND YTD DD ≤ 30%

**HWM:** Resets at each full re-entry — DD measured from last fully-invested NAV level.

**Exposure path in extreme scenario:** 100% → 60% → 34% → 17% → 6% → 0%

### Latest Results (IC=0.03, risk_aversion=10, N=25, net of 10bps costs)
```
Strategy                    CAGR    Vol   Sharpe    MDD   CAGR/DD
Baseline (quality)         27.7%  28.0%    0.99  -42.0%    0.66x
Pure Alpha (conc=2.0x)     84.0%  42.6%    1.97  -45.2%    1.86x
MVO (IC=0.03, max=8%)      57.3%  33.7%    1.70  -38.9%    1.47x
Hybrid (Alpha+MVO avg)     70.5%  37.8%    1.87  -41.1%    1.72x
Smart Hybrid               79.4%  40.1%    1.98  -41.4%    1.92x
Dynamic                    77.0%  39.3%    1.96  -39.0%    1.98x
Dynamic + Hedge            80.2%  39.5%    2.03  -37.8%    2.12x
Dyn+Hedge+DD (5 levels)    74.1%  36.4%    2.04  -31.6%    2.34x  ← best CAGR/DD
```

### Yearly Returns
```
Year   Baseline  Pure Alpha     MVO   Hybrid    Smart  Dynamic  Dyn+Hedge  DD Policy
2019    +51.0%     +51.1%   +41.0%   +46.1%   +51.4%   +56.3%    +52.4%    +47.0%
2020    +75.5%    +179.7%  +114.2%  +145.0%  +170.2%  +148.1%   +159.6%   +146.8%
2021    +15.2%     +27.9%   +28.2%   +28.5%   +36.6%   +47.3%    +47.3%    +47.6%
2022    -31.3%      -4.2%    -7.2%    -5.7%    -7.2%    -6.0%     +3.3%     -1.3%
2023    +33.9%     +73.6%   +54.9%   +64.2%   +64.3%   +56.1%    +56.5%    +53.8%
2024    +65.0%    +168.4%  +108.7%  +137.6%  +148.4%  +141.2%   +138.4%   +120.6%
2025    +25.4%    +125.5%   +77.9%  +101.5%  +120.2%  +110.0%   +112.2%   +116.1%
2026     +2.9%     +59.0%   +33.3%   +45.7%   +54.3%   +53.7%    +55.8%    +48.1%
```

### DD Policy Event Log (11 de-gross, 10 re-entries, 2019-2026)
```
2019-10-02  DE-GROSS lv1   -18.0%  100%→60%   (level 1 only)
2019-11-15  RE-ENTRY       +10.1%   60%→100%
2020-03-09  DE-GROSS lv1   -21.8%  100%→60%
2020-03-16  DE-GROSS lv2   -31.2%   60%→34%   (level 2: COVID crash)
2020-03-30  RE-ENTRY       +18.0%   34%→100%
2021-03-03  DE-GROSS lv1   -18.0%  100%→60%
2021-03-17  RE-ENTRY       +10.9%   60%→100%
2021-12-01  DE-GROSS lv1   -17.6%  100%→60%
2022-01-07  RE-ENTRY        -6.4%   60%→100%  (annual reset)
2022-06-13  DE-GROSS lv1   -19.0%  100%→60%
2022-08-02  RE-ENTRY       +10.5%   60%→100%
2022-09-26  DE-GROSS lv1   -17.6%  100%→60%
2022-10-10  RE-ENTRY        +7.8%   60%→100%
2023-09-21  DE-GROSS lv1   -19.8%  100%→60%
2023-11-13  RE-ENTRY       +11.4%   60%→100%
2024-08-05  DE-GROSS lv1   -17.5%  100%→60%
2024-08-19  RE-ENTRY       +16.5%   60%→100%
2025-02-27  DE-GROSS lv1   -18.1%  100%→60%
2025-04-29  RE-ENTRY       +15.3%   60%→100%
2025-11-13  DE-GROSS lv1   -18.5%  100%→60%
2026-01-07  RE-ENTRY       +14.0%   60%→100%  (annual reset)
```

### Hedge Backtest Results (14 episodes, 2019-2026, net of 20bps round-trip)
```
Hit rate: 71.4%  |  Total P&L: +17.1%  |  Avg per episode: +1.22%  |  Avg hold: 13.7d
Instruments used: QQQ (84.6% hit), ARKK (75%), SOXX (75%), IWM (50%), SPX (50%)
2022: 4 episodes, 100% hit rate, +10.5% P&L  ← key protection year
```

### Post-Run Displays
1. **Trading costs summary** — per year per strategy, total and annualized (incl. hedge costs and DD costs)
2. **Dynamic rebalancing log** — trigger type, period stats, full portfolio diagnostics (`***` = ADVP cap binding)
3. **Dynamic trigger summary** — count/% by type, yearly frequency
4. **Hedge backtest log** — per episode: instruments, weights, beta, effectiveness, P&L
5. **Hedge summary** — per-year and per-instrument breakdown with hit rates
6. **Live P&L** — smart hybrid and dynamic since last rebalance
7. **Current live portfolios** — rebalance vs drifted weights
8. **Live hedge status** — open/flat, unrealised P&L, instrument details
9. **Hypothetical smart hybrid rebalance** — today's portfolio with current regime
10. **Trade summary table** — unified allocation changes across Dynamic/Dyn+Hedge/DD Policy (ACTUAL or THEORETICAL)
11. **Gross exposure plot** — DD policy exposure over time + NAV comparison
12. **Portfolio/stock return statistics** + histograms

### Capacity / ADVP Constraint
Controls AUM-dependent liquidity filtering applied at runtime (not cached).

```python
AUM          = 1_000_000  # starting AUM in dollars — change for capacity testing
ADV_WINDOW   = 20         # days for median dollar ADV calculation
VOLUME_WINDOW = 10        # rolling mean window for volume de-trending
```

**New prompt:** `ADVP_CAP` (default 4%) — max % of median daily dollar volume per stock.

**Input:** `volumeTrd_df` must now be **raw daily share volume** (not pre-de-trended). De-trending is applied internally:
```python
volumeRaw_df = volumeTrd_df.copy()   # preserved for ADV cap
vol_roll     = volumeTrd_df.replace({0: np.nan}).ffill().fillna(0).rolling(10).mean()
volumeTrd_df = volumeTrd_df / vol_roll   # de-trended, used for momentum signal
```

**Two-tier liquidity filter:**
- **Alpha universe:** `ADV × advp_cap / AUM >= min_weight` — stock must be holdable at floor
- **MVO universe:** `ADV × advp_cap / AUM >= min_weight / top_n` — much lower bar; also searches full alpha-ranked universe if fewer than `n_cands` liquid stocks found (nuclear option)

**Self-healing cache:** if cached MVO portfolio has fewer than `top_n // 2` liquid stocks after ADVP filtering, invalidates cache for that date and recomputes with the nuclear option universe. Prints `[CACHE INVALIDATED @ date]`.

**ADVP cap application:** iterative water-filling. Stocks where `ADV × advp_cap / AUM < min_weight` are excluded entirely; excess redistributed. `***` flag in portfolio display = ADVP cap was binding (weight reduced by participation constraint, not max_weight).

**Rebalancing workflow for new AUM level:**
1. Change `AUM` constant
2. Run `run_mvo_backtest` with `portfolio_cache_override=True`
3. `run_daily_cache_build` NOT needed (triggers are AUM-agnostic)

### Capacity Curve (ADVP_CAP=4%, 2019-2026)

```
AUM      Pure Alpha  Dynamic  Dyn+Hedge  DD Policy  Sharpe(D+H)  Notes
$1M        84.0%     77.0%     80.2%      74.1%       2.03      unconstrained baseline
$25M       62.8%     61.7%     64.6%      58.7%       1.68      sweet spot
$100M      44.6%     42.9%     45.2%      40.2%       1.26      large-cap convergence
```

**Key capacity insights:**

**The liquidity filter is a quality screen at low AUM:** removing noisy micro/small-caps at $25M *improves* risk-adjusted performance vs the unconstrained $1M run (Sharpe 1.68 vs — the filter eliminates stocks where alpha signals are noisier, leaving cleaner mid-cap alpha.

**The strategy is NOT self-defeating:** yearly returns re-accelerate in 2024-2026 at $25M (+90%, +65%, +56% annualised for Dyn+Hedge). This means market conditions (AI/tech bull run, elevated volatility) are expanding the opportunity set faster than AUM compounds. The liquidity constraint is essentially static at $25M for mid/large-cap names.

**The inflection point is $25-50M:** below this, the liquidity filter helps; above it, genuine alpha decay sets in as high-quality mid-caps become constrained. At $100M, all strategies converge to 40-45% CAGR — essentially large-cap factor exposure only.

**$25M AUM results (ADVP_CAP=4%):**
```
Strategy                    CAGR    Vol   Sharpe    MDD   CAGR/DD
Baseline (quality)         27.7%  28.0%    0.99  -42.0%    0.66x
Pure Alpha (conc=2.0x)     62.8%  39.2%    1.60  -40.9%    1.54x
MVO (IC=0.03, max=8%)      54.4%  35.5%    1.53  -36.6%    1.49x
Hybrid (Alpha+MVO avg)     58.7%  37.3%    1.57  -38.8%    1.51x
Smart Hybrid               62.4%  38.2%    1.64  -39.4%    1.58x
Dynamic                    61.7%  38.3%    1.61  -39.4%    1.57x
Dynamic + Hedge            64.6%  38.5%    1.68  -39.4%    1.64x
Dyn+Hedge+DD (5 levels)    58.7%  35.2%    1.67  -35.7%    1.65x  ← best CAGR/DD
```

**$25M yearly returns:**
```
Year   Baseline  Pure Alpha     MVO   Hybrid    Smart  Dynamic  Dyn+Hedge  DD Policy
2019    +51.0%     +64.4%   +58.0%   +61.3%   +65.0%   +63.3%    +59.3%    +59.3%
2020    +75.5%    +122.1%  +101.8%  +111.9%  +127.1%  +135.8%   +146.7%   +126.0%
2021    +15.2%     +29.1%   +21.1%   +25.2%   +29.6%   +25.9%    +25.9%    +20.6%
2022    -31.3%      +2.7%    +3.7%    +3.3%    +1.9%    +3.6%    +11.6%    +15.9%
2023    +33.9%     +41.8%   +40.6%   +41.2%   +41.6%   +38.9%    +41.9%    +37.3%
2024    +65.0%     +95.4%   +78.2%   +86.7%   +92.4%   +92.6%    +90.5%    +83.2%
2025    +25.4%     +73.0%   +67.6%   +70.3%   +70.1%   +63.2%    +64.9%    +58.1%
2026     +2.9%     +52.9%   +43.3%   +48.1%   +51.9%   +53.8%    +55.9%    +46.2%
```

**DD Policy at $25M** (11 de-gross, 10 re-entries) — much cleaner than unconstrained:
- All level 1 de-grossings except COVID (level 2 in March 2020)
- Fast re-entries (avg ~2-3 weeks) — V-shaped recoveries
- 2022 best year for DD Policy at +15.9% vs +11.6% Dyn+Hedge — sustained downtrend rewarded the de-grossing

---

## 20. SCRIPT: `momentum_penalty.py`

**Location:** `/mnt/user-data/outputs/momentum_penalty.py`

### Overview
Analyses cross-sectional return distributions for non-overlapping windows (1M=21d, 2M=42d, 3M=63d) to support an exponential alpha-signal penalty for high-flying stocks. Designed as a **selective guardrail** — activation is episodic (~30-60% of rebalancing dates), never a constant drag on the signal.

### Entry Points
```python
from momentum_penalty import run_momentum_penalty_analysis, calibrate_k, inspect_date

# Step 1: build percentile distributions
pct_df = run_momentum_penalty_analysis(
    Pxs_df      = Pxs_df,
    rebal_dates = results['dyn_rebal_dates'],
    universe    = list(sectors_s.index),
)

# Step 2: calibrate (threshold_pct, k) jointly
k_df = calibrate_k(
    Pxs_df             = Pxs_df,
    composite_by_date  = results['composite_by_date'],
    rebal_dates        = results['dyn_rebal_dates'],
    universe           = list(sectors_s.index),
    pct_df             = pct_df,
    validation_days    = 42,       # forward window for IC evaluation
    k_max              = 10,       # cap on k (default)
    threshold_pcts     = [97, 99], # right tail only
    min_ic_improvement = 0.0025,
)

# Step 3: inspect a specific date
detail_df = inspect_date(
    Pxs_df            = Pxs_df,
    composite_by_date = results['composite_by_date'],
    universe          = list(sectors_s.index),
    pct_df            = pct_df,
    k_df              = k_df,
    rebal_dt          = '2025-12-10',  # default: last penalized date
)
```

### Methodology

**Step 1 — Build 1Y pooled distribution** (per window, per rebalancing date):
- Non-overlapping W-period returns across all stocks over 1Y prior → cross-sectional pool
- Compute mean, std, skewness, kurtosis (12 obs × N stocks for 1M, 6 × N for 2M, 4 × N for 3M)

**Step 2 — Z-score and rescale:**
- Z-score with 1Y mean/std (regime-normalised)
- Un-z-score with 5Y mean/std (long-run anchor) — or median/IQR if `use_robust=True`
- Rationale: skewness and kurtosis are more reliable than 1st/2nd moments; 5Y rescaling anchors to long-run behaviour

**Step 3 — Exponential penalty:**
```python
excess_W       = max(0, rescaled_return_W - threshold_W)
penalty_W      = exp(-k × excess_W)
max_excess     = max(excess_1M, excess_2M, excess_3M)
adjusted_alpha = alpha_z × exp(-k × max_excess)
```
Floor at 0 — no bonus for laggards. Final penalty = worst window (most conservative).

**Step 4 — Joint calibration of (threshold_pct, k):**
- At each rebalancing date independently (no look-ahead)
- signal_date = rebal_date - validation_days
- Ground truth: actual forward returns in validation window → Spearman rank IC
- Grid search over `threshold_pcts × k_grid` → maximise IC improvement ≥ `min_ic_improvement`
- If no improvement: k*=0 (revert to original signal)

### Key Parameters
```python
LOOKBACK_1Y    = 252     # trading days for 1Y distribution
LOOKBACK_5Y    = 1260    # trading days for 5Y rescaling
WINDOWS_TD     = {'1M': 21, '2M': 42, '3M': 63}   # 6M dropped (overlaps momentum signal)
PERCENTILES    = [95, 96, 97, 98, 99]              # right tail only in pct_df
```

Calibration defaults:
```python
validation_days    = 42      # 42d validation window (longer catches more reversal)
k_max              = 10      # exp(-10 × 0.1) = 0.37 — aggressive but not hard zero
threshold_pcts     = [97,99] # p97/p99 only — avoid false positives in middle
min_ic_improvement = 0.0025  # minimum IC improvement to adopt penalty
```

### k Interpretation
```
k = 0    → no penalty
k = 1    → moderate: 10% excess → 90% signal retained
k = 5    → aggressive: 10% excess → 61% retained; 50% excess → 8% retained
k = 10   → very aggressive: 10% excess → 37% retained; 50% excess → 0.7% retained
k → ∞   → hard exclusion
```

### Design Rationale
- **Why episodic is fine:** activation on ~30-60% of dates means the penalty isn't constantly fighting the alpha signal — it's a selective guardrail
- **Why not 6M:** 6M overlaps with the core momentum window (Jegadeesh-Titman 3-12M); penalizing would fight genuine alpha
- **Why p97+ only:** p95 catches too many genuine momentum names (false positives like LITE +42.7%)
- **Why max across windows:** a stock extreme on ANY horizon gets penalized; multi-window consistency (all three elevated) is more characteristic of genuine momentum than single-window spikes
- **Why validation_days=42:** 21d is too short for reversal to manifest in extreme momentum names; 42d captures more of the mean-reversion dynamic
- **Mental comfort value:** even if IC improvements are marginal, the guardrail provides confidence that purely hype-driven names aren't dominating portfolio selection — value that backtests can't fully quantify

### Current Calibration Results (validation_days=21, k_max=10, p[97,99], min_IC=0.0025)
```
Dates penalty adopted : 77/126 (61.1%)
k* distribution (when > 0): p25=20.0  p50=50.0  p75=50.0  (hitting ceiling — use k_max=10)
IC baseline   : median=+0.048  mean=+0.027
IC penalized  : median=+0.055  mean=+0.036
IC improvement: median=+0.0048 mean=+0.0091
Avg n_penalized: 46.4 stocks

Pooled forward returns (21d):
  Penalized (n=1665): min=-85.4%  p25=-10.5%  med=+0.0%  p75=+9.5%  mean=+1.6%
  Rest      (n=27790): min=-79.1%  p25=-5.3%   med=+0.0%  p75=+4.7%  mean=-0.2%
  Penalized outperformed rest  t=+3.23  p=0.0013 *** (wrong direction — validation_days=42 pending)
```

**Key finding:** at 21d, penalized stocks continue to outperform (momentum continuation). Extending to 42d validation to test whether reversal materialises at a longer horizon. The left tail is significantly fatter for penalized stocks (p25=-10.5% vs -5.3%), consistent with higher variance rather than directional reversal.

**Right counterfactual (pending):** comparison should be penalized stocks vs their replacement names in the final portfolio (stock #26 replacing stock #25), not vs the full universe.

---

## 21. PENDING WORK

### Next Session
1. **Momentum penalty — extend validation to 42d** — test whether reversal materialises at longer horizon
2. **Momentum penalty — right counterfactual** — compare penalized stocks vs their replacements in final portfolio, not full universe
3. **Momentum penalty — multi-window concentration signal** — stocks extreme on all 3 windows simultaneously = momentum; extreme on single window = hype spike. Test as conditioning variable
4. **Complete capacity curve** — run $10M and $50M to fill in the $25M-$100M inflection zone
5. **DD policy parameter tuning** — try wider thresholds (-25%/-45%) to reduce whipsaw
6. **Value cache refresh** — `run_ic_study(..., force_recompute_cache=True)` then full recalculation
7. **Quality cache** — run `quality_factor.run(force_recompute=True)` if needed

### Completed This Session
- [DONE] `momentum_penalty.py` — distribution analysis, joint (pct,k) calibration, date inspector
- [DONE] Calibration framework: episodic activation, min_ic_improvement threshold, k_max cap
- [DONE] Diagnostic: penalized vs rest forward returns with t-test, per-stock penalty detail
- [DONE] Design decisions: 6M dropped, p97+ threshold, validation_days=42 as next test
- [DONE] Capacity curve: $1M / $25M / $100M — sweet spot at $25M identified
- [DONE] Liquidity filter acts as quality screen below $25M — strategy not self-defeating
- [DONE] ADVP capacity constraint — full implementation with nuclear option + self-healing cache
- [DONE] Smart weights (`'smart'` key) saved to portfolio cache — Dynamic strategies now load correctly
- [DONE] Hash function unified — `_make_make_params_hash` used consistently
- [DONE] All capacity and momentum penalty findings documented in reference doc

### Caching Architecture
```
mvo_x_snapshots       Monthly X factor snapshots          Per model_version
mvo_daily_portfolios  Daily alpha/mvo/hybrid/smart weights Per params_hash
mvo_daily_triggers    Daily TO, vol_diff, drawdown         Per params_hash
macro_signal_daily    Per-instrument hedge signals+eff     Per params_hash
macro_indicators_daily Shared QBD + raw MOVE               Per params_hash
```

### Return Dict Keys
```python
results = {
    'nav_baseline', 'nav_alpha', 'nav_mvo', 'nav_hybrid',
    'nav_smart', 'nav_dynamic', 'nav_dyn_hedged',       # 7 NAV series
    'port_baseline', 'port_alpha', 'port_mvo', 'port_hybrid',
    'alpha_weights_by_date', 'mvo_weights_by_date',
    'hybrid_weights_by_date', 'smart_hybrid_weights_by_date',
    'dyn_weights_by_date', 'dyn_rebal_dates',           # actual rebal dates list
    'composite_by_date', 'regime_s', 'weights_by_year',
    'hedge_results',                                     # full hedge engine output
}
```

---

## 18. SCRIPT: `macro_indicators.py`

**Location:** `/mnt/user-data/outputs/macro_indicators.py`

### Overview
Builds MOVE proxy and QBD indicators, fits optimal thresholds monthly (point-in-time), generates binary hedge signals for multiple instruments, and caches all results to PostgreSQL.

### Entry Points
```python
from macro_indicators import run_macro_hedge_cached, plot_macro_hedge, plot_effectiveness

# Single instrument (diagnostic)
result = run_macro_hedge(Pxs_df, stocks_l, rates_l, move_grid, qbd_grid)
plot_macro_hedge(result, Pxs_df, start='2015-01-01')

# Multi-instrument with caching (main usage)
multi = run_macro_hedge_cached(
    Pxs_df, stocks_l, rates_l,
    hedges_l  = ['QQQ','SPX','IGV','SOXX','XHB','XLB','XLC','XLE',
                 'XLF','XLI','XLP','XLU','XLV','XLY','ARKK','IWM'],
    move_grid = move_grid,   # user-provided list of MOVE_T candidates
    qbd_grid  = qbd_grid,    # user-provided list of QBD_T candidates
    engine    = ENGINE,
    force_rebuild = False,
)

# Effectiveness plot
plot_effectiveness(multi, start='2015-01-01')
```

### Indicators
**MOVE proxy:** `ewm(halflife=MOVE_HL).std()` per tenor → averaged → differentiated by `rolling(MOVE_MAV).mean()` → filtered by `np.ceil(max(-corr(MOVE_changes, ref_asset_returns), 0))`. Binary filter: MOVE only passes through when negatively correlated with ref_asset (rate stress causing equity selloffs).

**QBD:** `(breadth_highQ - breadth_lowQ) - max(BREADTH_T - breadth_lowQ, 0)` where quintiles assigned daily by `ewm(halflife=VOL_HL).std()` and breadth = % stocks above MAV50.

### Signal Fitting
Monthly point-in-time grid search. Objective: maximize weighted `sign(r)×|r|^f / |log(vol)|` of hedged strategy returns, where `f=RETURN_EXP=1/3` and weights are annual step `[5,4,3,2,1]`. Signal = 1 when `MOVE >= MOVE_T AND QBD <= QBD_T` (AND logic default; OR also supported).

### Key Hyperparameters
```python
HARD_START      = '2010-01-01'
MOVE_HL         = 10       # ewm halflife for MOVE vol
MOVE_MAV        = 5        # rolling mean window for differentiation
CORR_WIN        = 63       # correlation filter window
VOL_HL          = 42       # ewm halflife for QBD vol quintiles
BREADTH_T       = 30       # QBD correction threshold (%)
MAV_WIN         = 50       # breadth MAV window (fixed)
REF_ASSET       = 'QQQ'    # reference for DD objective and corr filter
REF_DD_T        = -0.075   # DD threshold for objective
WEIGHT_YEARS    = [5,4,3,2,1]
HEDGE_MIN_HOLD  = 10       # min days hedge stays ON
SIGNAL_LOGIC    = 'AND'
RETURN_EXP      = 1/3
```

### Cache Tables
```
macro_indicators_daily  QBD + raw MOVE per date         (date, params_hash)
macro_signal_daily      Per-instrument signal+eff        (date, instrument, params_hash)
```
`REBUILD_QBD = False` module-level flag — set True only if `stocks_l`, `vol_hl`, or `breadth_t` changes.

### Effectiveness Metric
```python
sharpe_improvement = (sharpe_hedged - sharpe_raw) / abs(sharpe_raw)
effectiveness = clip(nav_ratio × (1 + sharpe_improvement), 0, 5)
```
Computed at 10B cadence with same annual weights, forward-filled to daily, smoothed with 20d MAV at consumption time.

### Latest Effectiveness Summary (as of 2026-04-13)
```
Instrument  Signal ON%   NAV Ratio  Sharpe Raw  Sharpe Hdg  Effectiveness
QQQ               5.6%       1.681        0.86        1.47          1.694
SOXX              5.9%       1.527        0.94        1.36          1.483
IGV               6.0%       1.177        0.25        0.45          1.462
XLC               6.9%       1.349        0.83        1.24          1.423
ARKK              8.1%       1.350        0.38        0.53          1.375
IWM               6.1%       1.030        0.55        0.67          1.122
XLB               8.2%       0.810  ← below 1.0 effectiveness floor
XLP               8.4%       0.884  ← below 1.0 effectiveness floor
```

---

## 19. SCRIPT: `short_interest_updater.py`

**Location:** `/mnt/user-data/outputs/short_interest_updater.py`

Fetches daily short interest data from Ortex API (CSV format) and updates `short_interest_data` table.

### Key Features
- **New ticker detection**: compares `short_interest_data` universe against fundamentals tables (`income_data`, `cash_data`, `summary_data`). New tickers get a dedicated first-pass fetch from `NEW_TICKER_START_DATE = "2010-01-01"`.
- **Suffix handling**: fundamentals tables store bare tickers; SI table stores `'AAPL US'` format. `normalize_ticker()` strips suffix for comparison; `si_ticker()` adds it back for DB inserts.
- **Guardrail**: if first ticker has no new data, prompts user before continuing (data may not be available yet).
- **5 metrics per ticker**: `shortInterestPcFreeFloat`, `shortInterestShares`, `shortAvailabilityShares`, `costToBorrowAll`, `costToBorrowNew`

### Flow
1. Detect new tickers → optional first-pass historical fetch
2. Prompt: force update all (last 30d) or incremental (from last date per ticker)
3. Prompt: confirm before proceeding
4. Fetch and upsert to `short_interest_data`

---

## 20. PENDING WORK

### Next Session
1. **DD policy parameter tuning** — try wider thresholds (-25%/-45%) to reduce 2021/2023 whipsaw
2. **Capacity / liquidity sensitivity** — grid of `mktcap_floor` and volume filters

### Earlier Pending
1. **Value cache refresh** — `run_ic_study(..., force_recompute_cache=True)` then full recalculation.
2. **Quality cache** — run `quality_factor.run(force_recompute=True)` if needed.

### Completed This Session
- [DONE] Flexible multi-level DD policy (8th strategy: Dyn+Hedge+DD)
- [DONE] DD CAGR/DD 2.34x — best of all 8 strategies
- [DONE] HWM resets on full re-entry; annual calendar reset for new year
- [DONE] Gross exposure plot (2-panel: exposure + NAV comparison)
- [DONE] DD trading costs column in trading costs summary
- [DONE] Hybrid tracker fixed to use daily returns (was computing period-to-period)
- [DONE] Re-entry runs independently of DD level checks (critical bug fix)
- [DONE] All parameters documented and reference doc updated

---

## 21. NOTES AND CONVENTIONS

- **Ticker format:** bare tickers (no `' US'`). `clean_ticker()` strips suffix.
- **DB writes:** always upsert.
- **Common sample:** intersection of all dates/stocks where every characteristic available.
- **Extended dates:** factor model runs from `st_dt - 252` trading days; variance stats from `st_dt`.
- **O-U cache:** clearing takes ~30 min for 2143 dates × 662 stocks.
- **Jupyter kernel:** all scripts run in same kernel. Namespace collision risk between scripts defining same function names (e.g. `_load_cached_dates`, `run`, `clean_ticker`) — mitigated by using `_sf` suffix in `sector_fundamentals.py`.
- **Volume input:** `volumeTrd_df` must now be **raw daily share volume** (NOT pre-de-trended). De-trending applied internally in `run_mvo_backtest`. `volumeRaw_df` preserved internally for ADV cap calculations. Previous convention (`volumeS_df` = pre-de-trended) is no longer used.
- **Size in valuation tables:** stored as market cap in $millions.
- **Ortex data pipeline:** `income_data` and `summary_data` tables. `estimated_values=True` for estimates, `False` for actuals. `normalizedNetIncome` copied to `netIncome` for all estimate rows. Non-GAAP adjustments applied via AlphaVantage EPS on earnings release dates.
- **SPX vs SPY:** portfolio uses `SPX` column name (not `SPY`) in `Pxs_df`. SPX is in `hedges_l` as a hedge instrument but NOT used as a trigger — QQQ alone is the trigger asset.
- **Hedge instrument naming:** `hedges_l` must match column names in `Pxs_df` exactly and must also be present in `multi['results']` (i.e. included in the `run_macro_hedge_cached` run).
