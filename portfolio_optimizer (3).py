#!/usr/bin/env python
# coding: utf-8

"""
PORTFOLIO OPTIMIZER  (regime-aware, convex, per-factor $-vol budgets)
=====================================================================
Optimizes a portfolio FROM SCRATCH over a candidate universe (top-ranked
names by Composite), with NO priors on which names to hold. Reuses the
factor risk model (X, F, Omega) so that risk is measured on the same basis
as factor_risk_decomposition.py — the covariance matrix is EXOGENOUS and
regime-driven (252d / 42d-HL EWMA), so a stressed window naturally yields a
more conservative book for the same alpha and risk-aversion.

OBJECTIVE (user-selected):
    1. max return        max  muᵀw
    2. min risk          min  wᵀΣw
    3. max Sharpe        max  muᵀw - (γ/2) wᵀΣw     (risk-aversion form, γ=1)

ALPHA (mu):  scaled from scores_df['Composite']. User gives the expected
    ANNUAL return of the top-ranked name; the rest scale linearly by score:
        muᵢ = top_ret * Compositeᵢ / Composite_top
    (μ and Σ are both annualized, so γ=1 is the log-utility anchor.)

PER-FACTOR RISK BUDGETS:  on the 3 factors with the highest COMPONENT $ vol
    in the INPUT portfolio (naturally Beta / Idio_Mom / GK_Vol). Expressed as
    STANDALONE annual $ vol on the NAV — the convex, directly-controllable
    measure:  |Xw[k]| * sqrt(F_ann[k,k]) * NAV <= budget_k.
    Empty input -> no constraint for that factor.

CONSTRAINTS: long-only, fully invested, per-name cap (ALLOCATION_CAP) and
    floor (ALLOCATION_FLOOR). Exact stock count N (= number of held names in
    the input book) via a convex two-stage relaxation:
        (1) solve floor-free over the universe,
        (2) take the top-N names by weight,
        (3) re-solve with the floor on those N.

OUTPUT: (1) suggested portfolio (%, 2dp), (2) trade deltas vs the current
    book, then an automatic risk decomposition of the optimized portfolio.

DEPENDENCY (Option C): run in the SAME kernel as, and AFTER,
    factor_risk_decomposition.py (reuses its builders + table constants) and
    with factor_model_step1 loaded. A guard fails loudly if anything's absent.

USAGE
    opt = run_optimizer(
        port_s, scores_df, Pxs_df, sectors_df,
        volumeTrd_df=None, model_version='v2', nav=1_000_000.0)
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

# ---- editable globals -------------------------------------------------------
ALLOCATION_CAP     = 0.10     # max per-name weight
ALLOCATION_FLOOR   = 0.01     # min per-name weight (for the held N)
OPT_UNIVERSE_SIZE  = 80       # candidate pool: top-N by Composite rank
GAMMA_RISK_AVERSION = 1.0     # risk-aversion in the max-Sharpe objective
BETA_WINDOW_FALLBACK = 252    # min recent history required per candidate name


# =============================================================================
# Dependency guard (Option C)
# =============================================================================
def _opt_check_dependencies():
    required = {
        'constants': ['RD_SCALAR_TABLES', 'MACRO_COLS', 'ENGINE'],
        'functions': ['_rd_build_F', '_rd_build_X', '_rd_build_omega',
                      '_rd_update_risk_residuals', '_rd_decompose',
                      '_rd_build_w', 'run_risk_decomp'],
        'factor_model': ['calc_rolling_betas', 'calc_vol_factor',
                         'calc_macro_betas', 'load_si_composite'],
    }
    g = globals()
    missing = {grp: [n for n in names if n not in g]
               for grp, names in required.items()}
    missing = {grp: ns for grp, ns in missing.items() if ns}
    if missing:
        lines = ["Missing dependencies — run factor_risk_decomposition.py first "
                 "in the SAME kernel (and load factor_model_step1)."]
        for grp, ns in missing.items():
            lines.append(f"   missing {grp}: {', '.join(ns)}")
        raise NameError("\n".join(lines))


# =============================================================================
# Prompts
# =============================================================================
def _prompt_objective():
    print("\n" + "=" * 66)
    print("  OPTIMIZATION OBJECTIVE")
    print("=" * 66)
    print("  1. Max return         maximise muᵀw")
    print("  2. Min risk           minimise wᵀΣw")
    print("  3. Max Sharpe         maximise muᵀw - (γ/2)wᵀΣw   (γ=%.2f)"
          % GAMMA_RISK_AVERSION)
    print("=" * 66)
    while True:
        raw = input("  Select objective (1/2/3): ").strip()
        if raw in ('1', '2', '3'):
            return {'1': 'max_return', '2': 'min_risk', '3': 'max_sharpe'}[raw]
        print("  invalid.")


def _prompt_top_return():
    while True:
        raw = input("\n  Expected ANNUAL return of the TOP-ranked name "
                    "(e.g. 0.30 = 30%): ").strip()
        try:
            v = float(raw)
            if math.isfinite(v):
                return v
        except ValueError:
            pass
        print("  enter a number, e.g. 0.30")


def _prompt_factor_budgets(focus_factors, input_dollar_vols):
    """Prompt a standalone annual $-vol budget for each focus factor.
    Empty -> None (no constraint). Shows the input book's current $ vol as a
    reference anchor."""
    print("\n" + "=" * 66)
    print("  PER-FACTOR RISK BUDGETS  (standalone annual $ vol on the NAV)")
    print("  Empty = no constraint. Current input-book $ vol shown as anchor.")
    print("=" * 66)
    budgets = {}
    for f in focus_factors:
        cur = input_dollar_vols.get(f, float('nan'))
        cur_s = f"{cur:,.0f}" if (cur == cur) else "n/a"
        raw = input(f"  {f:<12} (current ${cur_s}) budget $: ").strip()
        if raw == "":
            budgets[f] = None
        else:
            try:
                budgets[f] = float(raw)
            except ValueError:
                print(f"    '{raw}' invalid -> no constraint for {f}.")
                budgets[f] = None
    return budgets


# =============================================================================
# Core convex solve
# =============================================================================
def _solve_convex(mu_ann, Sigma_ann, objective, cap, floor_vec,
                  factor_budget_rows):
    """Solve one convex portfolio problem over the given universe.

    mu_ann          : (n,) annual expected returns
    Sigma_ann       : (n,n) annual covariance
    objective       : 'max_return' | 'min_risk' | 'max_sharpe'
    cap             : scalar per-name upper bound
    floor_vec       : (n,) per-name lower bounds (0 in stage 1; floor in stage 2)
    factor_budget_rows : list of (a_vec, bound) enforcing |a_vecᵀw| <= bound
                         (standalone factor $-vol budgets, linear in w)

    Returns (w, success, message).
    """
    n = len(mu_ann)
    g = GAMMA_RISK_AVERSION

    if objective == 'max_return':
        def obj(w):  return -(mu_ann @ w)
        def jac(w):  return -mu_ann
    elif objective == 'min_risk':
        def obj(w):  return float(w @ Sigma_ann @ w)
        def jac(w):  return 2.0 * (Sigma_ann @ w)
    else:  # max_sharpe (risk-aversion form)
        def obj(w):  return -(mu_ann @ w) + 0.5 * g * float(w @ Sigma_ann @ w)
        def jac(w):  return -mu_ann + g * (Sigma_ann @ w)

    # equality: fully invested
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
             'jac': lambda w: np.ones(n)}]
    # factor budgets: two-sided |aᵀw| <= bound  ->  bound - aᵀw >=0 ; bound + aᵀw >=0
    for a_vec, bound in factor_budget_rows:
        cons.append({'type': 'ineq',
                     'fun': (lambda w, a=a_vec, b=bound: b - a @ w),
                     'jac': (lambda w, a=a_vec: -a)})
        cons.append({'type': 'ineq',
                     'fun': (lambda w, a=a_vec, b=bound: b + a @ w),
                     'jac': (lambda w, a=a_vec: a)})

    bounds = [(float(floor_vec[i]), float(cap)) for i in range(n)]

    # feasible warm start: clip equal weight into [floor,cap], renormalise
    w0 = np.clip(np.ones(n) / n, floor_vec, cap)
    s = w0.sum()
    w0 = w0 / s if s > 0 else np.ones(n) / n

    res = minimize(obj, w0, jac=jac, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 1000, 'ftol': 1e-11})
    return res.x, res.success, res.message


# =============================================================================
# Full-universe construction (mirrors run_risk_decomp's universe logic)
# =============================================================================
def _build_full_universe(Pxs_df, sectors_s):
    """Build the full stock universe the same way run_risk_decomp does, so the
    cross-sectional factor exposures (and the vol factor's MIN_STOCKS gate) are
    computed over the whole cross-section — NOT just the optimization
    candidates. The candidates are subset from this afterward.
    """
    extended_st_dt = Pxs_df.index[0]
    try:
        with ENGINE.connect() as conn:
            from sqlalchemy import text as _text
            rows = conn.execute(_text(
                "SELECT DISTINCT ticker FROM income_data")).fetchall()
        db_tickers = {r[0].upper() for r in rows}
    except Exception:
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
        if len(pre_dates) >= 252:
            cd = Pxs_df.loc[pre_dates[-252:], col]
            if isinstance(cd, pd.DataFrame):
                cd = cd.iloc[:, 0]
            if int(cd.notna().sum()) < 126:
                continue
        universe.append(col)
    return universe


# =============================================================================
# Entry point
# =============================================================================
def run_optimizer(port_s, scores_df, Pxs_df, sectors_df,
                  volumeTrd_df=None, model_version='v2',
                  nav=1_000_000.0, objective=None, top_return=None,
                  factor_budgets=None, universe_size=None):
    """Regime-aware convex portfolio optimizer. See module docstring.

    port_s        : pd.Series  ticker -> % allocation (the CURRENT book; used
                    only for N = #held and for the trade deltas — NOT as a
                    name prior).
    scores_df     : pd.DataFrame with a 'Composite' column indexed by ticker.
    Pxs_df, sectors_df, volumeTrd_df, model_version : as in run_risk_decomp.
    nav           : NAV for the $-vol budgets and reporting.
    objective / top_return / factor_budgets / universe_size : optional; prompted
                    if not supplied.
    """
    _opt_check_dependencies()

    if universe_size is None:
        universe_size = OPT_UNIVERSE_SIZE

    # --- resolve sector inputs (Series or DataFrame), matching run_risk_decomp
    if isinstance(sectors_df, pd.DataFrame):
        sectors_s = sectors_df['sector'].copy()
        subsec_s  = (sectors_df['sub_sector'].copy()
                     if 'sub_sector' in sectors_df.columns else None)
    else:
        sectors_s = sectors_df.copy()
        subsec_s  = None
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    if subsec_s is not None:
        subsec_s = subsec_s[~subsec_s.index.duplicated(keep='first')]

    # --- current book: N and held names ------------------------------------
    cur = port_s[port_s.abs() > 0].copy()
    N = int(len(cur))
    print(f"\nCurrent book: {N} held names. Optimizing exactly {N} names "
          f"from a top-{universe_size} candidate universe.\n")

    if 'Composite' not in scores_df.columns:
        raise KeyError("scores_df must have a 'Composite' column.")
    ranked = scores_df['Composite'].dropna().sort_values(ascending=False)

    # feasibility precheck for N names under floor/cap
    if N * ALLOCATION_FLOOR > 1.0 + 1e-9:
        raise ValueError(f"Infeasible: {N} names * floor {ALLOCATION_FLOOR} "
                         f"= {N*ALLOCATION_FLOOR:.2f} > 1. Lower the floor or N.")
    if N * ALLOCATION_CAP < 1.0 - 1e-9:
        raise ValueError(f"Infeasible: {N} names * cap {ALLOCATION_CAP} "
                         f"= {N*ALLOCATION_CAP:.2f} < 1. Raise the cap or N.")

    # --- build the risk model over the FULL universe -----------------------
    # IMPORTANT: the factor exposures X are CROSS-SECTIONAL z-scores and the
    # vol factor requires >= MIN_STOCKS (150) names per date. So X/Omega MUST be
    # built over the full universe (same construction as run_risk_decomp), then
    # SUBSET to the optimization candidates. Building over just the top-80 would
    # both fail MIN_STOCKS (0 vol dates -> crash) AND give wrong z-scores.
    print("[1/4] Building F (factor covariance)...")
    F, factor_names, sec_cols, subsec_cols = _rd_build_F(model_version=model_version)

    print("[2/4] Building FULL universe (for correct cross-sectional exposures)...")
    full_universe = _build_full_universe(Pxs_df, sectors_s)
    print(f"  Full universe: {len(full_universe)} stocks")

    print("[3/4] Building X, Omega over the full universe...")
    X_full = _rd_build_X(full_universe, factor_names, sec_cols, subsec_cols,
                         Pxs_df, sectors_s, subsec_s, volumeTrd_df,
                         model_version=model_version).fillna(0.0)
    risk_resid_df = _rd_update_risk_residuals(
        full_universe, factor_names, sec_cols, subsec_cols,
        Pxs_df, sectors_s, subsec_s, volumeTrd_df, model_version=model_version)
    omega_full = _rd_build_omega(full_universe, risk_resid_df=risk_resid_df,
                                 model_version=model_version).reindex(full_universe).fillna(0.0)

    # --- candidate universe: top-`universe_size` by Composite that are in the
    #     full universe (i.e. have factor exposures). Names ranked high but not
    #     in the full universe (no data / excluded upstream) are reported and
    #     skipped; we backfill from the ranked pool to keep the target size.
    print("[4/4] Selecting top-Composite candidates present in the universe...")
    universe, excluded = [], []
    full_set = set(full_universe)
    for t in ranked.index:
        if len(universe) >= universe_size:
            break
        if t in full_set:
            universe.append(t)
        else:
            excluded.append(t)
    if excluded:
        print(f"  {len(excluded)} top-ranked name(s) absent from the factor "
              f"universe (no exposures — investigate), skipped & backfilled:")
        for t in excluded[:20]:
            print(f"      - {t}")
        if len(excluded) > 20:
            print(f"      ... (+{len(excluded)-20} more)")
    if len(universe) < N:
        raise ValueError(f"Only {len(universe)} ranked names are in the factor "
                         f"universe (need N={N}); check data coverage.")
    print(f"  Candidate universe: {len(universe)} names.")

    # subset X and Omega to the candidates
    X_df = X_full.reindex(universe)
    omega_s = omega_full.reindex(universe).fillna(0.0)

    # annualize
    F_ann = F * 252.0
    X = X_df.reindex(universe)[factor_names].values          # n x K
    omega_ann = omega_s.values * 252.0
    Sigma_ann = X @ F_ann @ X.T + np.diag(omega_ann)          # n x n
    # symmetrize for numerical safety
    Sigma_ann = 0.5 * (Sigma_ann + Sigma_ann.T)

    # --- alpha vector from Composite (annual) ------------------------------
    comp = scores_df['Composite'].reindex(universe)
    comp_top = comp.max()
    if not np.isfinite(comp_top) or comp_top == 0:
        raise ValueError("Composite_top is zero/NaN; cannot scale alpha.")
    if objective is None:
        objective = _prompt_objective()
    need_alpha = objective in ('max_return', 'max_sharpe')
    if need_alpha and top_return is None:
        top_return = _prompt_top_return()
    mu_ann = (top_return * comp / comp_top).fillna(0.0).values if need_alpha \
        else np.zeros(len(universe))

    # --- identify the 3 focus factors from the INPUT book ------------------
    #     (top-3 by component $ vol in the current portfolio). Uses the
    #     already-built FULL-universe X/Omega (correct cross-sectional basis).
    focus_factors, input_dollar_vols = _focus_factors_from_input(
        cur, X_full, omega_full, full_universe, F, F_ann, factor_names,
        sec_cols, nav)

    if factor_budgets is None:
        factor_budgets = _prompt_factor_budgets(focus_factors, input_dollar_vols)

    # translate budgets into linear rows: a = X[:,k]*sqrt(F_ann[k,k])*NAV
    fidx = {f: i for i, f in enumerate(factor_names)}
    budget_rows = []
    for f, b in (factor_budgets or {}).items():
        if b is None or f not in fidx:
            continue
        k = fidx[f]
        a_vec = X[:, k] * math.sqrt(max(F_ann[k, k], 0.0)) * nav
        budget_rows.append((a_vec, float(b)))

    # --- STAGE 1: floor-free convex solve over the whole universe ----------
    print("\n  [stage 1] solving over full universe (no floor)...")
    floor0 = np.zeros(len(universe))
    w1, ok1, msg1 = _solve_convex(mu_ann, Sigma_ann, objective,
                                  ALLOCATION_CAP, floor0, budget_rows)
    if not ok1:
        print(f"  [stage 1] solver warning: {msg1}")

    # --- select top-N by weight -------------------------------------------
    w1_s = pd.Series(w1, index=universe)
    sel = list(w1_s.sort_values(ascending=False).index[:N])
    print(f"  [stage 1] selected top {N} names by weight.")

    # --- STAGE 2: re-solve on the N selected names WITH floor --------------
    print("  [stage 2] re-solving on selected names (with floor)...")
    sub_idx = [universe.index(t) for t in sel]
    mu_sub = mu_ann[sub_idx]
    Sig_sub = Sigma_ann[np.ix_(sub_idx, sub_idx)]
    budget_rows_sub = [(a[sub_idx], b) for a, b in budget_rows]
    floor2 = np.full(N, ALLOCATION_FLOOR)
    w2, ok2, msg2 = _solve_convex(mu_sub, Sig_sub, objective,
                                  ALLOCATION_CAP, floor2, budget_rows_sub)

    # infeasibility: solver flag, structural constraints, AND budget rows
    _budget_ok = True
    _budget_viol = []
    for (a_vec, bound) in budget_rows_sub:
        achieved = abs(float(a_vec @ w2))
        if achieved > bound * (1.0 + 1e-3) + 1.0:   # small $ tolerance
            _budget_ok = False
            _budget_viol.append((achieved, bound))
    infeasible = ((not ok2) or (abs(w2.sum() - 1.0) > 1e-3)
                  or (w2.min() < ALLOCATION_FLOOR - 1e-3)
                  or (w2.max() > ALLOCATION_CAP + 1e-3)
                  or (not _budget_ok))
    if infeasible:
        print("\n  " + "!" * 60)
        print("  INFEASIBLE / non-converged on stage 2 with current budgets.")
        print(f"  solver: {msg2}")
        print("  The factor budget(s) may be too tight to satisfy alongside "
              "long-only + floor/cap + fully-invested on these names.")
        for achieved, bound in _budget_viol:
            print(f"    budget $ {bound:,.0f} unreachable — best achievable "
                  f"$ {achieved:,.0f}")
        print("  Try loosening the tightest budget, or widen universe_size.")
        print("  " + "!" * 60)
        # still report the attempted weights for diagnosis
    w_opt = pd.Series(w2, index=sel)

    # --- OUTPUT: suggested portfolio + trade deltas ------------------------
    _print_optimizer_output(w_opt, cur, objective, factor_budgets,
                            focus_factors, X_df, F_ann, omega_s, nav,
                            factor_names)

    # --- auto risk-decomposition of the optimized book ---------------------
    print("\n  Running risk decomposition on the OPTIMIZED portfolio...\n")
    opt_port_s = (w_opt * 100.0)     # run_risk_decomp expects % allocations
    try:
        run_risk_decomp(opt_port_s, Pxs_df, sectors_df,
                        volumeTrd_df=volumeTrd_df, model_version=model_version,
                        nav=nav)
    except Exception as e:
        print(f"  (risk decomposition of optimized book failed: {e})")

    out = {'weights': w_opt, 'trades': (w_opt.reindex(
              cur.index.union(w_opt.index)).fillna(0.0)
              - (cur / 100.0).reindex(cur.index.union(w_opt.index)).fillna(0.0)),
           'objective': objective, 'focus_factors': focus_factors,
           'budgets': factor_budgets, 'infeasible': infeasible}
    return out


# =============================================================================
# Helpers: focus factors from the input book + output printing
# =============================================================================
def _focus_factors_from_input(cur, X_full, omega_full, full_universe, F, F_ann,
                              factor_names, sec_cols, nav):
    """Decompose the INPUT book to find the top-3 factors by component $ vol,
    using the FULL-universe X/Omega (correct cross-sectional exposures). The
    input holdings are just the subset that gets portfolio weight; their
    exposures come from the full-universe X.
    Returns (focus_factor_names, {factor: standalone_annual_$vol}).
    """
    # held names that are actually in the factor universe
    held = [t for t in cur.index if t in set(full_universe)]
    if not held:
        raise ValueError("None of the input holdings are in the factor "
                         "universe; cannot identify focus factors.")
    w_held = cur.reindex(held).fillna(0.0)
    w_held = (w_held / w_held.sum()).values          # renormalized weights

    Xh = X_full.reindex(held)[factor_names].values    # exposures from full X
    om_h = omega_full.reindex(held).fillna(0.0).values

    result_i, _, _, _ = _rd_decompose(w_held, Xh, F, om_h, factor_names, sec_cols)
    has_bc = 'var_beta_cross_pct' in result_i.columns
    contrib = (result_i['var_1st_pct'] + result_i['var_2nd_pct']
               + (result_i['var_beta_cross_pct'] if has_bc else 0.0)) / 100.0
    pool = [f for f in factor_names if f in result_i.index]
    focus = sorted(pool, key=lambda f: abs(contrib.get(f, 0.0)), reverse=True)[:3]

    # standalone annual $ vol per focus factor for the input book
    fidx = {f: i for i, f in enumerate(factor_names)}
    Xw = w_held @ Xh
    dvols = {}
    for f in focus:
        k = fidx[f]
        dvols[f] = abs(Xw[k]) * math.sqrt(max(F_ann[k, k], 0.0)) * nav
    return focus, dvols


def _print_optimizer_output(w_opt, cur, objective, budgets, focus_factors,
                            X_df, F_ann, omega_s, nav, factor_names):
    print("\n" + "=" * 66)
    print(f"  SUGGESTED PORTFOLIO  ({objective})   —   {len(w_opt)} names")
    print("=" * 66)
    ws = w_opt.sort_values(ascending=False)
    print(f"  {'Ticker':<10}{'Weight %':>10}")
    print("  " + "-" * 22)
    for tk, wv in ws.items():
        print(f"  {tk:<10}{wv*100:>9.2f}%")
    print("  " + "-" * 22)
    print(f"  {'TOTAL':<10}{ws.sum()*100:>9.2f}%")

    # standalone $ vol of the optimized book on the focus factors
    fidx = {f: i for i, f in enumerate(factor_names)}
    print("\n  Optimized standalone annual $ vol on focus factors:")
    for f in focus_factors:
        if f in fidx and f in X_df.columns:
            k = fidx[f]
            xk = X_df.reindex(w_opt.index)[f].fillna(0.0).values
            Xw_k = float(w_opt.values @ xk)
            dv = abs(Xw_k) * math.sqrt(max(F_ann[k, k], 0.0)) * nav
            b = (budgets or {}).get(f)
            btag = "" if b is None else f"   (budget ${b:,.0f})"
            print(f"    {f:<12} ${dv:>12,.0f}{btag}")

    # trade deltas vs current book (union of names)
    alln = cur.index.union(w_opt.index)
    cur_w = (cur / 100.0).reindex(alln).fillna(0.0)
    new_w = w_opt.reindex(alln).fillna(0.0)
    delta = (new_w - cur_w).sort_values(ascending=False)
    print("\n" + "=" * 66)
    print("  TRADES TO IMPLEMENT  (allocation deltas, % points)")
    print("=" * 66)
    print(f"  {'Ticker':<10}{'Current %':>11}{'New %':>10}{'Delta pp':>11}")
    print("  " + "-" * 42)
    for tk in delta.index:
        d = delta[tk]
        if abs(d) < 1e-6:
            continue
        print(f"  {tk:<10}{cur_w[tk]*100:>10.2f}%{new_w[tk]*100:>9.2f}%"
              f"{d*100:>+10.2f}")
    buys  = delta[delta > 0].sum() * 100
    sells = delta[delta < 0].sum() * 100
    print("  " + "-" * 42)
    print(f"  gross turnover (one-way): {delta[delta>0].sum()*100:.2f}%  "
          f"(buys +{buys:.2f} / sells {sells:.2f})")


if __name__ == "__main__":
    print(__doc__)
