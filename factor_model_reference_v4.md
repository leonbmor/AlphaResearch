# Factor Model & Scripts — Comprehensive Reference
*Last updated: March 2026 (v4)*

---

## 1. OVERVIEW

A sequential Fama-MacBeth cross-sectional factor model implemented in Python, running in a Jupyter notebook kernel. The model strips systematic return sources one by one in a true Gram-Schmidt orthogonalization sequence, producing clean residuals at each step. All data is stored in a PostgreSQL database (`factormodel_db`). The universe consists of ~662–679 US large-cap stocks (varies with sector mapping updates).

**Database connection:**
```
postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db
```

**Key principle:** Both return residuals AND characteristics themselves are orthogonalized at each step using WLS (weighted by log market cap). Before entering any regression, each new characteristic is regressed cross-sectionally against all prior characteristics. Only the residual enters the regression. This is true Gram-Schmidt in the weighted inner-product space defined by cap weights.

---

## 2. FACTOR MODEL — v1 (ORIGINAL) 11-STEP ARCHITECTURE

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

---

## 3. FACTOR MODEL — v2 (REORDERED) 11-STEP ARCHITECTURE

**Key differences from v1:**
1. Sequence reordered — alpha factors (Quality, Momentum, Value) precede structural/risk factors
2. Momentum computed on quality residuals (v1: vol residuals)
3. Value follows momentum — eliminates price-contamination of value signal
4. Macro and Sector characteristics are NOT Gram-Schmidt orthogonalized — macro uses raw EWMA betas to prior-step residuals (z-scored); sectors use sum-to-zero deviation coding with Ridge CV
5. All tables prefixed `v2_` — v1 tables untouched

### Step Sequence

```
Step 1:  Baseline UFV
Step 2:  Market Beta       input=raw_rets        GS char ⊥ {}
Step 3:  Quality           input=resid_mkt       GS char ⊥ {beta}
Step 4:  Idio Momentum     input=resid_quality   GS char ⊥ {beta, quality}
Step 5:  Size              input=resid_mom       GS char ⊥ {beta, quality, mom}
Step 6:  Value             input=resid_size      GS char ⊥ {beta, quality, mom, size}
Step 7:  SI Composite      input=resid_value     GS char ⊥ {all prior}
Step 8:  GK Volatility     input=resid_si        GS char ⊥ {all prior}
Step 9:  Macro Factors     input=resid_vol       raw betas, joint Ridge CV
Step 10: Sector Dummies    input=resid_macro     sum-to-zero, Ridge CV
Step 11: O-U Mean Rev      input=resid_sec       GS char ⊥ {all prior}
```

### v2 Database Tables

| Table | Contents |
|-------|----------|
| `v2_factor_residuals_mkt/quality/mom/size/value/si/vol/macro/sec/ou` | Residuals per step |
| `v2_factor_lambdas_mkt/quality/mom/size/value/si/vol/macro/sec/ou` | Factor returns per step |
| `v2_ou_reversion_df` | v2 O-U scores cache |
| `v2_value_scores_df` | v2 value scores cache |

### Incremental Update — Dynamic Size Cache Fix

The `dynamic_size_df` table is shared between v1, v2, and MVO backtest scripts. When MVO runs with a 150-stock candidate pool, it partially populates the cache for that date with only 150 stocks. The factor model then finds the date already cached and skips recomputation, producing degenerate size factor results.

**Fix:** Both `_run_incremental` (v1) and `_v2_run_incremental` (v2) purge today's entry from `dynamic_size_df` at the start, before calling `load_dynamic_size`:

```python
# At top of _run_incremental / _v2_run_incremental, after dt = Pxs_df.index[-1]:
with ENGINE.begin() as conn:
    deleted = conn.execute(text("""
        DELETE FROM dynamic_size_df WHERE date = :d
    """), {"d": dt.strftime('%Y-%m-%d')})
    if deleted.rowcount > 0:
        print(f"  Cleared {deleted.rowcount} stale dynamic_size_df rows for {dt.date()}")
```

This ensures the full 679-stock universe is always recomputed fresh on incremental update days.

---

## 4. FACTOR DETAILS

### Market Beta
EWMA beta vs SPX. `β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)`, window=252, hl=126.

### Size
`size_i = log(shares × price)` — dynamic daily. Cached in `dynamic_size_df`.

### Macro Factors

| Column | Description |
|--------|-------------|
| `USGG2YR` | 2Y nominal rate daily change (bps) |
| `US10Y2Y_SPREAD_CHG` | 10Y-2Y spread daily change (bps) |
| `US10YREAL` | 10Y real yield daily change |
| `BE5Y5YFWD` | 5y5y forward breakeven inflation daily change |
| `MOVE` | Interest rate volatility index daily change |
| `Crude` | WTI crude oil daily change |
| `XAUUSD` | Gold daily change |

Each macro beta: `β_im = Cov_EWMA(r_i, d_m) / Var_EWMA(d_m)`, window=252, hl=126, z-scored cross-sectionally. All 7 enter a single joint ridge regression. Ridge CV: 5-fold per date. λ=0.15–0.30 dominates (~60% of dates); λ=40 on collinear days (~12%).

### Sector Dummies
Sum-to-zero deviation coding: for K sectors, each dummy = +1 own sector, -1/(K-1) all others. All sectors included. Intercept = true equal-weighted market return. Ridge CV: ~88.5% of dates select λ=0.10.

### Quality Factor
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

### SI Composite
Short interest composite. Cached in `si_composite_df`.

### GK Volatility
`σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²`. Window=84d, hl=42d. Annualized.

### Idiosyncratic Momentum
On `factor_residuals_vol` (v1) or `v2_factor_residuals_quality` (v2). Volume-scaled: `mom_i = Σ r_resid × vol_scalar`, clipped [0.5, 3.0]. Window: [t-252, t-21].

### Value Factor
IC-weighted composite from `value_scores_df` cache.
```python
_VALUE_TSTAT = {
    'P/S':   -5.340, 'P/Ee':  -4.867, 'P/Eo':  -4.763,
    'sP/S':  -5.341, 'sP/E':  -4.485, 'sP/GP': -5.339, 'P/GP':  -3.910,
}
VALUE_WEIGHTS = {m: abs(w) / total for m, w in _VALUE_TSTAT.items()}
```

### O-U Mean Reversion
AR(1) fit to compounded residual price index. `half_life = ln(2)/(-ln(b))`. Requires `0 < b < 1`. Falls back to 21d ST reversal on residual index if fit invalid. Blended: `ou_weight = min(30/half_life, 10)`.

---

## 5. DATABASE TABLES

### v1 Residual & Lambda Tables
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

`factor_lambdas_mkt/size/macro/sec/quality/si/vol/mom/joint/ou`

### Characteristic / Score Tables
| Table | Contents |
|-------|----------|
| `dynamic_size_df` | Daily market cap (shared across all scripts — see cache fix in §3) |
| `si_composite_df` | SI composite scores |
| `quality_scores_df` | Quality composite (cached) |
| `value_scores_df` | Value composite (cached) |
| `ou_reversion_df` | O-U scores (v1 cached) |
| `valuation_consolidated` | Raw quarterly fundamentals |
| `valuation_metrics_anchors` | Anchor date snapshots |
| `income_data` | Ortex income fundamentals |
| `summary_data` | Ortex summary fundamentals |
| `estimation_status` | FEQ tracking |
| `daily_open/high/low` | OHLC prices |

---

## 6. LATEST FACTOR PERFORMANCE (March 2026, v2)

| Step | % UFV | % prev |
|------|-------|--------|
| Beta | 71.66% | 71.66% |
| Quality | — | — |
| Idio Mom | — | — |
| Size | 60.69% | — |
| Value | — | — |
| SI | 51.40% | 99.65% |
| GK Vol | 50.79% | 98.81% |
| Macro | 54.73% | 90.18% |
| Sectors | 51.93% | 94.88% |
| O-U | 47.61% | 98.95% |

**Consolidated R² = 52.39%**

---

## 7. FEQ (FIRST ESTIMATED QUARTER) MAPPING PROCEDURE

*(unchanged from v3 — see Section 7 of prior document)*

---

## 8. SCRIPT: `factor_model_step1.py` (v1)

**Location:** `/mnt/user-data/outputs/factor_model_step1.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_universe()` | Filters stocks: in DB + sector-mapped + sufficient history |
| `load_dynamic_size()` | Loads/computes market cap from DB cache (`dynamic_size_df`) |
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
| `_run_incremental()` | Single-date fast path |
| `run()` | Master entry point |

---

## 9. SCRIPT: `factor_model_v2.py`

**Location:** `/mnt/user-data/outputs/factor_model_v2.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

Reuses all machinery from v1 kernel. Follows v2 step sequence (§3). All output tables prefixed `v2_`. See §3 for dynamic size cache fix that must be applied to `_v2_run_incremental`.

---

## 10. SCRIPT: `quality_factor.py`

**Location:** `/mnt/user-data/outputs/quality_factor.py`
**Entry point:** `run(Pxs_df, sectors_s, mav_window=252, threshold=50)`

### Cache Refresh Workflow
```python
summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s,
                                              mav_window=252, threshold=50)
update_cached_weights(gqf_w, cqf_w)
```

---

## 11. SCRIPT: `ic_study.py` (Value Factor)

**Location:** `/mnt/user-data/outputs/ic_study.py`
**Entry point:** `run_ic_study(Pxs_df, sectors_s, force_recompute_cache=False)`

Targets `factor_residuals_mom`. Horizons: 21d and 63d.

---

## 12. SCRIPT: `primary_factor_backtest.py`

**Location:** `/mnt/user-data/outputs/primary_factor_backtest.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

Long-only, bi-monthly rebalancing, TOP_N=20. Primary factor = quality composite.

---

## 13. SCRIPT: `composite_backtest.py`

**Location:** `/mnt/user-data/outputs/composite_backtest.py`

Composite alpha backtest: Quality + Idio_Mom + Value + Mom_12M1 (OU excluded). Uses point-in-time rolling regime weights from `factor_ic_study.py`. Greedy constrained stock selector with sector cap.

### Backtest Results (v2, rolling weights, conc=2.0x)
```
2019: +54.29%   2020: +154.86%   2021: +35.10%   2022: -0.69%
2023: +38.59%   2024: +125.57%   2025: +101.77%   2026: +0.03%
CAGR: ~62%   Vol: ~38%   Sharpe: ~1.63   MDD: -41.6%
```

---

## 14. SCRIPT: `factor_ic_study.py`

**Location:** `/mnt/user-data/outputs/factor_ic_study.py`

Alpha factor IC study + regime weight derivation. Produces `weights_by_year` dict used by `composite_backtest.py` and `mvo_backtest.py`.

### Key Design Decisions
- **Point-in-time weights:** `compute_rolling_regime_weights` uses only IC data from years < Y
- **Weight bounds:** [0.10, 0.50] for all factors via `_ics_bounded_normalize`
- **Rates regime:** 0.0/0.5/1.0 based on USGG10YR rolling mean (w1=63, w2=42, threshold=20)
- **Active factors:** Quality, Idio_Mom, Value, Mom_12M1 (OU excluded)

### Rolling Weights Sample (year 2022)
```
          Quality  Idio_Mom  Value  Mom_12M1
regime
0.000     0.339    0.409    0.120   0.132
0.500     0.167    0.167    0.500   0.167
1.000     0.250    0.250    0.250   0.250
```

---

## 15. SCRIPT: `mvo_backtest.py`

**Location:** `/mnt/user-data/outputs/mvo_backtest.py`
**Entry point:** `run_mvo_backtest(Pxs_df, sectors_s, weights_by_year, regime_s, ...)`

Three-way backtest: Baseline (quality) vs Pure Alpha (composite, concentration-weighted) vs MVO (ensemble covariance + Grinold-Kahn alpha).

### Backtest Results (IC=0.04, max=10%, N=30)
```
Year     Baseline   Pure Alpha        MVO
2019      +54.21%      +54.29%     +40.50%
2020      +76.59%     +154.86%    +199.15%
2021      +18.04%      +35.10%     +26.86%
2022      -30.38%       -0.69%     -19.61%
2023      +33.15%      +38.59%     +56.72%
2024      +65.67%     +125.57%    +122.33%
2025      +15.96%     +101.77%    +107.09%
2026      -12.21%       -2.47%      -0.46%

Baseline:   CAGR=25.4%  Vol=27.2%  Sharpe=0.93  MDD=-38.9%
Pure Alpha: CAGR=62.0%  Vol=38.1%  Sharpe=1.63  MDD=-41.6%
```

### Parameters
```python
results = run_mvo_backtest(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df    = volumeS_df,
    ic              = 0.04,         # Grinold-Kahn IC scaling
    max_weight      = 0.10,         # single-name cap
    min_weight      = 0.025,        # single-name floor
    zscore_cap      = 2.5,          # alpha z-score winsorization
    min_matrix_count = 2,           # eligibility filter threshold
    pca_var_threshold = 0.65,       # PCA variance explained
    universe_mult   = 5,            # candidate pool = top_n × universe_mult
    risk_aversion   = 1.0,          # MVO λ parameter
)
```

### Portfolio Construction Flow

**[1/4] Composite alpha scores**
- Loads Quality, Idio_Mom, Value, Mom_12M1
- Combines using point-in-time rolling regime weights from `weights_by_year`
- Built on all rebalance dates

**[2/4] X snapshots (monthly)**
- Builds factor exposure matrix X on month-end dates for all 680 universe stocks
- Used for Factor-driven covariance matrix construction
- Rebuilt on every run (no persistent cache — module-level cache resets on cell re-execution)

**[3/4] Portfolio weights**
- **Baseline:** top-N by quality score, sector-capped, equal weight
- **Pure Alpha:** top-N by composite score, concentration-weighted (conc_factor=2.0x)
- **MVO:** see MVO Weight Solver below

**[4/4] NAV series**
- Daily compounding from rebalance weights
- Buy-and-hold between rebalance dates

### MVO Weight Solver (`_mb_solve_mvo`)

**Step 1 — Eligibility filter**

Run MVO on each of the 4 covariance matrices independently. Count how many matrices select each stock. Apply `min_matrix_count=2` threshold — only use stocks selected by ≥2 matrices.

**2×top_n heuristic:** if fewer than `2×top_n` stocks pass the filter (e.g. <60 for N=30), fall back to full valid universe. This handles early dates (2019-2020) where the universe is sparse and the filter is too restrictive.

```python
if len(eligible_filtered) >= 2 * top_n:
    eligible = eligible_filtered   # filtered: higher quality, consensus stocks
else:
    eligible = valid               # fallback: full 150-stock candidate pool
```

**Step 2 — Adaptive cap to enforce target portfolio size**

Solve MVO on eligible universe with cap-only constraint. Reduce cap iteratively until `n_nonzero >= top_n`. Guaranteed to converge at `cap = 1/top_n` (e.g. 3.33% for N=30).

```python
# Cap schedule for top_n=30, max_weight=0.10:
# 10.0% → 9.17% → 8.33% → 7.50% → 6.67% → 5.83% → 5.00% → 4.17% → 3.33%
# Stops as soon as n_nonzero >= top_n

for cap in caps_to_try:
    w_var = cp.Variable(N)
    prob = cp.Problem(
        cp.Maximize(alpha @ w_var - (λ/2) * quad_form(w_var, Σ)),
        [sum(w_var) == 1, w_var >= 0, w_var <= cap]
    )
    prob.solve(solver=OSQP, fallback=SCS)
    if n_nonzero >= top_n:
        break
```

**Note on equal weight convergence:** when `cap = 1/top_n`, the optimizer is forced to distribute weight equally (no room for differentiation). For meaningful weight dispersion, use `max_weight=0.15–0.20` or `IC=0.02` (lower IC → less aggressive signal → natural dispersion before hitting tight cap). The MVO diagnostics use `IC=0.02` and produce the canonical 5×7.5% + 25×2.61% distribution.

**Step 3 — Floor/cap post-processing**

```python
w_nz  = w_out[w_out > 1e-4]           # keep meaningful weights only
w_out = _mb_floor_then_cap(w_nz, min_weight=min_weight, max_weight=used_cap)
```

`_mb_floor_then_cap`:
1. Clip all weights to `min_weight` from below → renorm
2. Iteratively cap weights above `max_weight`, redistribute excess proportionally → renorm
3. Repeat until no weight exceeds cap or max iterations reached

### Ensemble Covariance Matrix (`_mb_build_cov_matrices`)

Four covariance estimators are computed independently and averaged:

**1. Empirical EWMA (`Sigma_emp`)**
- EWMA covariance of daily returns over 252-day window, half-life=42d
- Applied to return history of candidate stocks
- Most responsive to recent market conditions

**2. Ledoit-Wolf Shrinkage (`Sigma_lw`)**
- Analytical shrinkage toward scaled identity: `Σ_LW = (1-α)·Σ_sample + α·μ·I`
- Shrinkage intensity α estimated analytically (Oracle Approximating Shrinkage)
- Reduces estimation error for high-dimensional covariance matrices
- More stable than pure sample covariance

**3. Factor-driven (`Sigma_factor`)**
- Structural decomposition: `Σ_factor = X·F·X' + Ω_rescaled`
- X = factor exposure matrix (N×K) from monthly X snapshots, reindexed to candidate stocks
- F = factor return covariance (K×K) from EWMA of lambda time series, window=252, hl=42d
- Ω = diagonal idiosyncratic variance from `risk_residuals_v2` table
- Rescaled to match empirical covariance trace: `α = trace(Σ_emp) / trace(Σ_factor_raw)`
- Enforces factor structure — prevents optimizer exploiting spurious correlations

**4. PCA (`Sigma_pca`)**
- Eigendecomposition of empirical covariance
- Retain components explaining `pca_var_threshold=65%` of variance
- Reconstruct: `Σ_pca = V_k · Λ_k · V_k' + diag(residual_var)`
- Removes noise eigenvectors — especially useful in high-dimensionality regimes

**Ensemble:**
```python
Sigma_ens = (Sigma_emp + Sigma_lw + Sigma_factor + Sigma_pca) / 4
```
Equal-weighted average. Combines stability properties of all four estimators.

### Alpha Signal (Grinold-Kahn)

```python
vol_daily  = _mvo_ewma_vol(ret_df[valid], hl=126)   # EWMA daily vol per stock
z_s        = composite_scores.reindex(valid)          # composite alpha z-score
z_capped   = z_s.clip(-zscore_cap, zscore_cap)        # winsorize at ±2.5σ
alpha      = ic * vol_daily * z_capped                # Grinold-Kahn: α = IC × σ × z
```

`IC=0.04` recommended from IC overlap analysis. `IC=0.02` produces better weight dispersion in practice.

### Diagnostic Output Per Date
```
2026-03-25 eligible=89 (filtered) valid=150
  cap=0.1000 n_nonzero=12 target=30
  cap=0.0333 n_nonzero=30 target=30
2026-03-25  MVO: n= 30  min=2.5%  max=6.7%  eff_N=24.3
```

### Known Issues & Pending Work
1. **Equal weight convergence:** at `cap=1/top_n` the optimizer returns equal weights — no differentiation. Try `IC=0.02` and `max_weight=0.15` for meaningful dispersion.
2. **X snapshot rebuild:** ~15-20 min per run (84 month-end dates × 680 stocks). No persistent cache — module-level cache resets on cell re-execution. Future: cache to DB.
3. **MVO full stats pending:** CAGR/Vol/Sharpe/MDD for MVO not yet computed from latest run.

---

## 16. SCRIPT: `mvo_diagnostics.py`

**Location:** `/mnt/user-data/outputs/mvo_diagnostics.py`
**Entry point:** `run_mvo_diagnostics(...)`

Single-date MVO diagnostic tool. Tests 4 IC values, shows overlap analysis, portfolio construction details. Confirmed working at `IC=0.02, max_weight=0.075`, producing:

```
Top 5 stocks: ~7.5% each (at cap)
Remaining 25 stocks: ~2.61% each (at floor = 1/38.3)
```

This is the canonical target distribution for the backtest to replicate.

### Recommended Parameters (from diagnostics)
```python
ic          = 0.02    # lower IC → less aggressive → better weight dispersion
max_weight  = 0.075   # allows top stocks to concentrate at 7.5%
min_weight  = 0.025   # floor ensures 30 stocks minimum
top_n       = 30
```

---

## 17. SCRIPT: `portfolio_risk_decomp.py`

**Location:** `/mnt/user-data/outputs/portfolio_risk_decomp.py`

Portfolio variance decomposition: `Var = w'X·F·X'w + w'Ω·w`. Uses consistent-basis residuals from `risk_residuals_v1/v2`.

---

## 18. SCRIPT: `pnl_attribution.py`

**Location:** `/mnt/user-data/outputs/pnl_attribution.py`

Daily PnL attribution by factor. Decomposes realized returns into factor contributions.

---

## 19. SCRIPT: `factor_loading_tracker.py`

**Location:** `/mnt/user-data/outputs/factor_loading_tracker.py`

Tracks portfolio factor loadings `(X'w)[k]` across rebalance dates. Useful for comparing MVO vs pure alpha factor exposures over time.

---

## 20. PENDING WORK

### Immediate
1. **MVO weight dispersion** — rerun with `IC=0.02, max_weight=0.075` to replicate diagnostics distribution
2. **MVO full summary stats** — CAGR/Vol/Sharpe/MDD from latest run pending
3. **MVO vs Pure Alpha comparison** — determine if MVO adds Sharpe value over pure alpha
4. **Factor loading tracker on MVO** — compare factor exposure distribution vs pure alpha portfolio

### Ongoing
5. **Value cache refresh** — `run_ic_study(..., force_recompute_cache=True)` then full recalculation
6. **Quality cache** — 1966 stale dates with old weights

---

## 21. NOTES AND CONVENTIONS

- **Ticker format:** bare tickers (no `' US'`). `clean_ticker()` strips suffix.
- **DB writes:** always upsert.
- **Common sample:** intersection of all dates/stocks where every characteristic available.
- **Extended dates:** factor model runs from `st_dt - 252` trading days; variance stats from `st_dt`.
- **O-U cache:** clearing takes ~30 min for 2143 dates × 662 stocks.
- **Jupyter kernel:** all scripts run in same kernel. Namespace collision risk between scripts — mitigated by using `_sf` suffix in `sector_fundamentals.py`.
- **Volume scalars:** `volumeTrd_df` = pre-computed `vol(t)/mean_vol[t-10, t-1]`, clipped `[0.5, 3.0]`.
- **Size in valuation tables:** stored as market cap in $millions.
- **dynamic_size_df:** shared cache — MVO backtest pollutes with subset universe. Factor model incremental update must delete today's entry before recomputing (see §3).
- **MVO notebook workflow:** scripts pasted directly into cells. Module-level state (caches, globals) resets on every cell execution. Use notebook-level variables for any state that must persist across runs.
- **Ortex data pipeline:** `income_data` and `summary_data` tables. `estimated_values=True` for estimates, `False` for actuals. `normalizedNetIncome` copied to `netIncome` for all estimate rows.
