# Factor Model v2 — Technical Reference (updated June 2026)

## Project Overview
Sequential Fama-MacBeth cross-sectional factor model (v2) for ~701 US large-cap stocks.
PostgreSQL: `postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db`

## Key Files
- `/mnt/user-data/outputs/factor_model_v2.py` — 4103 lines, full pipeline
- `/mnt/user-data/outputs/factor_model_v2_pre_full_orthog.py` — latest checkpoint
- `/mnt/user-data/outputs/mvo_backtest.py` — main backtest, 10 strategies
- `/mnt/user-data/outputs/mvo_backtest_vMR.py`, `mvo_backtest_v10.py` — rolling checkpoints
- `/mnt/user-data/outputs/pnl_attribution.py` — PnL attribution (sub-sectors included)
- `/mnt/user-data/outputs/risk_decomposition.py` — risk decomposition (sub-sectors included)
- `/mnt/user-data/outputs/index_builder.py` — sector/group index builder with extras_l
- `/mnt/user-data/outputs/valuation_visualizer.py` — valuation multiples visualizer
- `/mnt/user-data/outputs/query_lambdas.py` — lambda history viewer
- `/mnt/user-data/outputs/wipe_caches.py` — full cache wiper
- `/mnt/user-data/outputs/check_weights_cache.py` — reads weights_by_date from memory

---

## Factor Model — Current 12-Step Pipeline

**NEW SEQUENCE (implemented June 2026):**
Beta → Sectors → Size → Quality → Value → SI → GK Vol → Idio Momentum → Macro → O-U → Sub-sectors

**Rationale:**
- Sectors at Step 2 (right after Beta): every downstream factor measured net of sector effects by construction — matches Barra/Axioma commercial practice
- Size and Value within-sector by construction: more interpretable, consistent with within-sector score ranking
- Idio Momentum last among style factors: computed on vol residuals, stripped of all structural/fundamental factors — truly idiosyncratic
- Quality stays early: captures genuine cross-sector quality signal

**Full orthogonalization:** Steps 2 (Sectors), 9 (Macro) orthogonalized vs all prior factors using Gram-Schmidt / `orthogonalize_char_df`

**PIT weight updates:**
- Quality PIT weights: updated using **sector residuals** (Step 2 output) — was market residuals
- Value PIT weights: updated using **quality residuals** (Step 4 output) — was size residuals

---

## Variance Reduction (full 2017-2026 run, new sequence)
```
Step                                             %UFV    %prev
UFV (raw)                                     100.00%      ---
+ Beta                                         75.07%   75.07%
+ Sectors                                      70.66%   94.12%
+ Size                                         69.84%   98.84%
+ Quality                                      58.46%   83.71%
+ Value                                        57.59%   98.51%
+ SI                                           56.97%   98.92%
+ GK Vol                                       55.84%   98.02%
+ Idio Momentum                                54.74%   98.04%
+ Macro                                        51.63%   94.32%
+ O-U                                          49.58%   96.02%
+ Sub-sectors                                  48.45%   97.72%
```
Total: 51.55% of cross-sectional variance explained.

---

## Key Constants (calibrated defaults)
```python
AUM=5_000_000; MB_TOP_N=25; MB_REBAL_FREQ=15 (calendar days)
TRADING_COST_BPS=10; top_n=25; rebal_freq=15; advp_cap=0.04
universe_mult=5; risk_aversion=10; prefilt_pct=0.40; conc_factor=2.0
MOM_FILTER=3; MR_K=0.5; MR_CAP=0.0; TIER2_T=0; TIERONE_ALLOC=0.90
CORR_FILTER=0.70; EFF_FLOOR=1.00; HEDGE_RATIO=0.25; MAX_HEDGE=0.50
DD_REENTRY_PCT=0.150; DD_REENTRY_CONFIRM=5; DD_HWM_WINDOW=12
_MB_F_LOOKBACK=504; _MB_F_EWMA_HL=252 (trading days)
MIN_HIST=126; SUBSEC_MIN_STOCKS=5
DYN_TO_THRESHOLD_ALPHA=0.25; DYN_TO_THRESHOLD_HYBRID=0.30; DYN_TO_THRESHOLD_MVO=0.35
DYN_VOLDIFF_CAP=0.175; DYN_VOLDIFF_DERISK=-0.750 (effectively disabled)
DYN_MIN_HOLD_DAYS=10 (calendar days — ~10 calendar days ≈ 7 trading days)
```

**Important:** `DYN_MIN_HOLD_DAYS` and `generate_calc_dates` both use **calendar days**. 15 calendar days ≈ 10-11 trading days. Dynamic and static strategies naturally converge to similar rebalancing cadence in normal markets.

---

## DB Tables
**Factor model:**
- `v2_lambda_*` (11 tables: mkt, quality, mom, size, value, si, vol, macro, sec, ou, subsec)
- `v2_factor_residuals_*` (same set)
- `v2_ou_reversion_df`, `dynamic_size_df`, `si_composite_df`
- `quality_scores_df`, `value_scores_df`, `quality_weights_pit`, `value_weights_pit`

**MVO backtest:**
- `mvo_nav_cache(params_hash, strategy, date, nav)` — daily NAV per strategy
- `mvo_rebal_cache(params_hash, strategy, date, ticker, weight, gross_factor)` — rebalancing events
- `mvo_state_cache(params_hash, strategy, state_json, last_date)` — serialized strategy state
- `mvo_daily_portfolios`, `mvo_x_snapshots`, `mvo_cov_cache`

---

## MVO Backtest — Architecture

**10 strategies:**
```
S_BASE=0   Baseline        (equal weight, static)
S_ALPHA=1  Pure Alpha      (composite score, static)
S_MVO=2    MVO             (covariance-aware, static)
S_HYB=3    Hybrid          (50/50 alpha+MVO, static)
S_SMART=4  Smart Hybrid    (regime-switching, static)
S_DYN=5    Dynamic         (dynamic rebalancing + regime)
S_HEDGE=6  Dyn+Hedge       (S_DYN + hedge overlay)
S_DD=7     DD Policy       (S_HEDGE + drawdown de-grossing)
S_EXCL=8   Excl            (S_DD + momentum exclusion filter)
S_MVO_HEDGE=9 MVO+Hedge   (pure MVO + dynamic + hedge, no regime switching)
```

**Caching modes:**
- `mode='rebuild'`: clears cache (mvo_nav_cache, mvo_rebal_cache, mvo_state_cache, mvo_daily_portfolios, mvo_x_snapshots, mvo_cov_cache), runs full simulation, saves all data
- `mode='incremental'`: loads cached state, always recomputes last Pxs_df date, runs only new dates

**State rollback (incremental):** when wiping last date, the state is rolled back:
- `nav` → from `mvo_nav_cache` at `_cache_last_dt`
- `weights` → from `mvo_rebal_cache` at last rebalancing ≤ `_cache_last_dt`
- `last_rebal` → rolled back to match weights date (critical for `_check_trigger`)

**Dynamic trigger (`_check_trigger`):**
- Uses calendar days for `days_held = (dt - last_rebal).days`
- Fires when: TO > threshold AND voldiff < cap AND days_held >= min_hold_days
- OR: regime switch AND days_held >= min_hold_days
- OR: derisk (voldiff < -0.750, effectively disabled)

**params_hash:** MD5 of all key hyperparameters. Cache is tied to hash — different params = cache miss = rebuild needed.

---

## Full Rebuild Sequence
After any factor model change:
1. Run `wipe_caches.py` (clears ALL factor + MVO caches)
2. Run `factor_model.run(Pxs_df, sectors_df, volumeTrd_df=volumeTrd_df, force_rebuild_pit=True)` → mode=full
3. Run `factor_ic_study` → rebuilds `weights_by_date` in memory
4. Run `mvo_backtest` with `mode='rebuild', rebuild_cov=True`

**wipe_caches.py now covers:** all v2_factor_residuals_*, v2_lambda_*, v2_ou_reversion_df, quality/value scores, PIT weights, factor_ic_cache, mvo_daily_portfolios, mvo_x_snapshots, mvo_cov_cache, mvo_nav_cache, mvo_rebal_cache, mvo_state_cache, dynamic_size_df, si_composite_df

---

## Incremental Daily Output
```
[v2] 2026-06-08   Intercept: X%   Daily R²: X% → Y% (+sub-sec)   ...
  Factor                    Return%
  Market Beta               ±X%
  Size                      ±X%
  Quality                   ±X%
  Value                     ±X%
  SI Composite              ±X%
  GK Vol                    ±X%
  Idio Momentum             ±X%
  O-U Reversion             ±X%
  Macro: USGG2YR...         ±X%
  Sector: Hardware...       ±X%
  SubSec: AI Accelerators.. ±X%
```

```
Current regime (dynamic strategies):
  Dynamic        mode=alpha     gross=100%  dd=-4.1%
  Dyn+Hedge      mode=alpha     gross=100%  dd=-4.1%
  ...
```

---

## Function Signatures

**Factor model:**
```python
x = run(Pxs_df, sectors_df, volumeTrd_df=volumeTrd_df, force_rebuild_pit=True)
# sectors_df: DataFrame with 'sector' and 'sub_sector' columns
```

**MVO backtest:**
```python
results = run_backtest(Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df=volumeTrd_df, weights_by_date=weights_by_date,
    mode='incremental')  # or 'rebuild'
```

**Risk decomposition:**
```python
result_df, decomp_s = run_risk_decomp(port_s, Pxs_df, sectors_df)
```

**PnL attribution:**
```python
attrib_df, stock_idio_df = run_pnl_attribution(port_df, Pxs_df, sectors_df)
```

**Index builder:**
```python
from index_builder import main
main(Pxs_df, Sectors_df, extras_l=['SPY', 'QQQ', 'SOXX'])
```

---

## Pending Items
- **CUSUM + HMM regime detection** for DD Policy: papers read (Page 1954, Inclan & Tiao 1994, Hamilton 1989, Ang & Bekaert 2002, Nystrup et al. 2015/2017). Implementation pending.
- **P&L attribution and risk mapping** with sub-sector residuals/lambdas: scripts updated, ready to run
- **Factor reordering** (Option 4) now implemented and validated
