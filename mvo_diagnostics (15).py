"""
mvo_diagnostics.py
==================
Diagnostic tool for Mean-Variance Optimization inputs.

Builds and compares four covariance matrix estimates for a candidate
universe of stocks on a given date, alongside the composite alpha signal
scaled to return units via the Grinold-Kahn formula.

Covariance matrices:
  1. Ledoit-Wolf (constant correlation shrinkage target)
  2. Factor-driven: XFX' using factor model exposures
  3. PCA-based: retain components explaining >= PCA_VAR_THRESHOLD of variance
  4. Empirical: EWMA sample covariance
  5. Ensemble: equal-weighted average of the four above

Alpha signal:
  alpha_i = IC * vol_i * z_i   (Grinold-Kahn)
  where IC is a free calibration parameter, vol_i is EWMA daily vol,
  z_i is the composite alpha z-score.

Assumes all factor_model_step1, portfolio_risk_decomp, factor_ic_study
functions are already loaded in the Jupyter kernel.

Entry point
-----------
    diag = run_mvo_diagnostics(
        dt, Pxs_df, sectors_s, composite_scores,
        port_n=30, universe_mult=5,
        ic_values=[0.02, 0.04, 0.06, 0.08],
        model_version='v2', volumeTrd_df=None
    )

Parameters
----------
    dt               : pd.Timestamp — date for diagnostics
    Pxs_df           : pd.DataFrame — price/macro panel
    sectors_s        : pd.Series   — ticker -> sector label
    composite_scores : pd.Series   — composite alpha z-scores (full universe)
    port_n           : int — final portfolio size (default 30)
    universe_mult    : int — candidate universe = port_n * universe_mult
    ic_values           : list — IC values to test for alpha scaling
    model_version       : 'v1' or 'v2'
    volumeTrd_df        : optional vol scalars
    pca_var_threshold   : float — variance explained threshold for PCA (default 0.65)
    max_weight          : float — hard cap on single-name weight (default 0.10)
    min_weight          : float — floor on non-zero positions (default 0.025)
    zscore_cap          : float — winsorization cap on z-scores (default 2.5)
    min_matrix_count    : int   — min matrices selecting a stock for MVO eligibility (default 2)

Returns
-------
    dict with cov matrices, alpha vectors, diagnostics, and plots
"""

import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# PARAMETERS
# ===============================================================================

MVO_LOOKBACK        = 252     # days of return history for covariance estimation
MVO_EWMA_HL         = 126     # EWMA half-life for covariance (more stable)
MVO_PCA_VAR_THRESH  = 0.65    # default variance explained threshold for PCA (overridable)
MVO_DEFAULT_IC      = 0.04    # default IC for Grinold-Kahn scaling
MVO_OVERLAP_TARGET  = 0.65    # target portfolio overlap across cov matrices
MVO_MAX_WEIGHT      = 0.10    # hard cap on single-name weight (default 10%)
MVO_ZSCORE_CAP      = 2.50    # winsorization cap on composite z-scores
MVO_MIN_MATRIX_COUNT = 2      # min matrices that must select a stock for eligibility
MVO_MIN_WEIGHT      = 0.025   # floor on non-zero single-name weight (default 2.5%)


# ===============================================================================
# HELPERS
# ===============================================================================

def _mvo_ewma_cov(ret_df, hl):
    """
    EWMA covariance matrix from a returns DataFrame (T x N).
    Returns np.ndarray (N x N).
    """
    T      = len(ret_df)
    decay  = np.log(2) / hl
    w      = np.exp(-decay * np.arange(T - 1, -1, -1))
    w     /= w.sum()
    v      = ret_df.values
    mu     = (w[:, None] * v).sum(0)
    d      = v - mu
    return (d * w[:, None]).T @ d


def _mvo_ewma_vol(ret_df, hl):
    """EWMA volatility (std) per stock. Returns pd.Series."""
    T     = len(ret_df)
    decay = np.log(2) / hl
    w     = np.exp(-decay * np.arange(T - 1, -1, -1))
    w    /= w.sum()
    v     = ret_df.values
    mu    = (w[:, None] * v).sum(0)
    var   = (w[:, None] * (v - mu) ** 2).sum(0)
    return pd.Series(np.sqrt(var), index=ret_df.columns)


def _mvo_ledoit_wolf(ret_matrix):
    """
    Ledoit-Wolf shrinkage using sklearn's well-tested implementation.
    Shrinks toward the constant-correlation target (Oracle Approximating
    Shrinkage estimator — Ledoit & Wolf 2004).

    Returns (Sigma_lw, rho_bar, shrinkage_coef).
    """
    from sklearn.covariance import LedoitWolf

    T, N = ret_matrix.shape

    # Fit LW on raw returns (sklearn handles its own centering)
    lw      = LedoitWolf(assume_centered=False)
    lw.fit(ret_matrix)
    Sigma_lw    = lw.covariance_
    shrink_coef = float(lw.shrinkage_)

    # Compute mean pairwise correlation from the shrunk matrix for reporting
    std     = np.sqrt(np.diag(Sigma_lw))
    std_mat = np.outer(std, std)
    with np.errstate(invalid='ignore', divide='ignore'):
        Corr  = np.where(std_mat > 0, Sigma_lw / std_mat, 0.0)
    n_off   = N * (N - 1)
    rho_bar = (Corr.sum() - np.trace(Corr)) / n_off if n_off > 0 else 0.0

    return Sigma_lw, rho_bar, shrink_coef


def _mvo_factor_cov(tickers, Pxs_df, sectors_s, volumeTrd_df, model_version):
    """
    Factor-driven covariance: Sigma_factor = X @ F_mat @ X.T
    Uses _rd_build_F and _rd_build_X from portfolio_risk_decomp.py.
    Returns (N x N) covariance matrix.

    Note: Pxs_df should be the FULL price panel (not sliced to dt) so that
    _rd_build_X has enough history for all lookback calculations (vol, beta etc).
    The date is implicitly the last row of Pxs_df.
    """
    F_mat, factor_names, sec_cols = _rd_build_F(model_version=model_version)

    # Use full Pxs_df — _rd_build_X uses Pxs_df.index[-1] as the calc date
    # so pass the already-sliced Pxs_df (caller slices to dt) but ensure
    # there is enough history by checking row count
    if len(Pxs_df) < max(BETA_WINDOW, VOL_WINDOW) + 10:
        warnings.warn(
            f"  _mvo_factor_cov: Pxs_df only has {len(Pxs_df)} rows — "
            f"need at least {max(BETA_WINDOW, VOL_WINDOW) + 10} for reliable X"
        )

    try:
        X_df = _rd_build_X(
            tickers, factor_names, sec_cols,
            Pxs_df, sectors_s, volumeTrd_df,
            model_version=model_version
        ).fillna(0.0)

        # Check for all-zero GK_Vol column — indicates OHLC gap on latest date
        # Robust check: unique non-zero values < 2 means effectively flat
        if ('GK_Vol' in X_df.columns and
                X_df['GK_Vol'].abs().sum() < 1e-6):
            warnings.warn("  GK_Vol is all-zero (OHLC gap) — "
                          "falling back to close-to-close vol")
            ret_window = Pxs_df[tickers].pct_change().dropna(how='all')
            ret_window = ret_window.iloc[-VOL_WINDOW:]
            cc_vol     = ret_window.std()                     # daily std
            cc_vol_mu  = cc_vol.mean()
            cc_vol_sd  = cc_vol.std()
            if cc_vol_sd > 0:
                cc_vol_z = (cc_vol - cc_vol_mu) / cc_vol_sd
            else:
                cc_vol_z = cc_vol * 0.0
            X_df['GK_Vol'] = cc_vol_z.reindex(tickers).fillna(0.0).values
            print(f"  GK_Vol fallback applied: "
                  f"mean={cc_vol_z.mean():.3f}, std={cc_vol_z.std():.3f}")

    except Exception as e:
        warnings.warn(f"  _rd_build_X failed: {e} — using zero X matrix")
        X_df = pd.DataFrame(0.0, index=tickers, columns=factor_names)

    X = X_df.reindex(tickers).fillna(0.0).values   # (N x K)
    Sigma_factor_raw = X @ F_mat @ X.T              # (N x N)

    # Rescale to match empirical return variance level.
    # By construction, z-scored X means XX'/N ≈ I_N so XFX' ≈ c*I —
    # it captures relative factor structure but not absolute variance.
    # We calibrate by matching the mean diagonal to the empirical EWMA cov.
    Sigma_emp_ref = _mvo_ewma_cov(
        pd.DataFrame(
            # Use stock returns implied from Pxs_df for the tickers
            Pxs_df[tickers].pct_change().dropna(how='all').iloc[-MVO_LOOKBACK:]
        ), MVO_EWMA_HL
    )
    emp_mean_var    = np.diag(Sigma_emp_ref).mean()
    factor_mean_var = np.diag(Sigma_factor_raw).mean()

    if factor_mean_var > 1e-12:
        scale        = emp_mean_var / factor_mean_var
        Sigma_factor = Sigma_factor_raw * scale
        print(f"  Factor cov rescaled by {scale:.2f}x "
              f"(factor mean var={factor_mean_var:.2e}, "
              f"empirical mean var={emp_mean_var:.2e})")
    else:
        # Fallback: use empirical as factor-driven if scaling fails
        warnings.warn("  Factor cov is near-zero — using empirical as fallback")
        Sigma_factor = Sigma_emp_ref

    return Sigma_factor, X_df, factor_names


def _mvo_pca_cov(ret_matrix, var_threshold=MVO_PCA_VAR_THRESH):
    """
    PCA-based covariance matrix on raw returns.
    Retains components explaining >= var_threshold of total variance.
    Returns (shrunk N x N cov matrix, n_components retained, var_explained).
    """
    # EWMA covariance
    Sigma    = _mvo_ewma_cov(pd.DataFrame(ret_matrix), MVO_EWMA_HL)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    # Sort descending
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Clip negative eigenvalues (numerical noise)
    eigvals = np.maximum(eigvals, 0.0)

    # Select components explaining var_threshold of variance
    total_var   = eigvals.sum()
    cum_var     = np.cumsum(eigvals) / total_var if total_var > 0 else np.zeros_like(eigvals)
    n_comp      = max(1, int(np.searchsorted(cum_var, var_threshold) + 1))
    var_expl    = float(cum_var[n_comp - 1])

    # Reconstruct covariance from retained components
    Vk          = eigvecs[:, :n_comp]           # (N x k)
    Lk          = np.diag(eigvals[:n_comp])     # (k x k)
    Sigma_pca   = Vk @ Lk @ Vk.T               # (N x N)

    # Add residual variance on diagonal (idiosyncratic noise)
    resid_var   = np.diag(Sigma) - np.diag(Sigma_pca)
    Sigma_pca  += np.diag(np.maximum(resid_var, 0.0))

    return Sigma_pca, n_comp, var_expl, eigvals


def _mvo_condition_number(mat):
    """Condition number of a matrix (ratio of max/min eigenvalue)."""
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.maximum(eigvals, 1e-15)
    return float(eigvals.max() / eigvals.min())


def _mvo_scale_alpha(composite_z, vol_s_daily, ic,
                     zscore_cap=MVO_ZSCORE_CAP):
    """
    Grinold-Kahn alpha scaling in DAILY return units:
        alpha_i = IC * vol_i_daily * z_i_winsorized

    vol_s_daily must be daily volatility (not annualised).
    z-scores are winsorized at ±zscore_cap before scaling to prevent
    extreme alpha outliers from dominating the MVO solution.
    """
    z_capped = composite_z.clip(-zscore_cap, zscore_cap)
    alpha    = ic * vol_s_daily * z_capped
    return alpha.fillna(0.0)


def _mvo_cap_weights(w, max_weight=MVO_MAX_WEIGHT, max_iter=20):
    """
    Apply a hard cap on single-name concentration with iterative renormalization.

    Algorithm:
        1. Cap any weight above max_weight
        2. Distribute excess proportionally to uncapped stocks
        3. Repeat until no weight exceeds cap (typically 2-3 iterations)

    Returns a pd.Series with weights summing to 1, all <= max_weight.
    """
    w = w.copy().clip(lower=0)
    if w.sum() == 0:
        return w
    w = w / w.sum()   # ensure sums to 1 before capping

    for _ in range(max_iter):
        over_mask  = w > max_weight
        if not over_mask.any():
            break
        # Amount to redistribute
        excess     = (w[over_mask] - max_weight).sum()
        # Cap the over-limit stocks
        w[over_mask] = max_weight
        # Redistribute to uncapped stocks proportionally
        under_mask = ~over_mask
        if under_mask.any() and w[under_mask].sum() > 1e-10:
            w[under_mask] += excess * (w[under_mask] / w[under_mask].sum())
        else:
            # Edge case: all stocks at cap — distribute equally
            w += excess / len(w)

    # Final renormalize for floating point safety
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _mvo_apply_floor(w, min_weight=MVO_MIN_WEIGHT, max_weight=MVO_MAX_WEIGHT,
                     max_iter=20):
    """
    Apply minimum weight floor to non-zero positions.

    Stocks with solver weight > 0 but below min_weight are raised to min_weight.
    Stocks with weight == 0 (solver excluded them) remain at zero.
    Excess cost absorbed proportionally from above-floor positions.
    Cap re-applied if any position exceeds max_weight during redistribution.
    """
    w = w.copy().clip(lower=0)
    if w.sum() == 0:
        return w
    w = w / w.sum()

    # Threshold: treat weights below this as solver zeros (numerical noise)
    zero_thresh = 1e-4

    for _ in range(max_iter):
        # Active = solver selected these stocks (weight above noise floor)
        active      = w > zero_thresh
        below_floor = active & (w < min_weight - 1e-9)
        if not below_floor.any():
            break

        shortfall = (min_weight - w[below_floor]).sum()
        w[below_floor] = min_weight

        # Absorb from above-floor positions
        above_floor = active & (w > min_weight + 1e-9)
        if above_floor.any() and w[above_floor].sum() > shortfall + 1e-9:
            w[above_floor] -= shortfall * (w[above_floor] / w[above_floor].sum())
        else:
            # Can't absorb — drop the lowest-weight active stocks
            # until shortfall can be covered
            active_sorted = w[active].sort_values()
            for ticker in active_sorted.index:
                if w[active].sum() <= 0:
                    break
                if w[above_floor].sum() >= shortfall + 1e-9:
                    break
                # Drop this stock
                w[ticker] = 0.0
                active[ticker] = False
                above_floor = active & (w > min_weight + 1e-9)
            # Renorm remaining
            if w[active].sum() > 0:
                w[active] = w[active] / w[active].sum()
            break

        # Re-apply cap
        over_cap = w > max_weight + 1e-9
        if over_cap.any():
            excess = (w[over_cap] - max_weight).sum()
            w[over_cap] = max_weight
            under_cap = active & ~over_cap
            if under_cap.any():
                w[under_cap] += excess * (w[under_cap] / w[under_cap].sum())

    # Zero out numerical noise
    w[w < zero_thresh] = 0.0
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _mvo_portfolio_overlap(w1_tickers, w2_tickers):
    """Fraction of stocks appearing in both portfolios."""
    s1, s2 = set(w1_tickers), set(w2_tickers)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _mvo_min_var_portfolio(Sigma, tickers):
    """Minimum variance portfolio weights (long-only, sum to 1)."""
    try:
        import cvxpy as cp
        N = len(tickers)
        w = cp.Variable(N)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, Sigma)),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return pd.Series(w.value, index=tickers)
    except Exception:
        pass
    # Fallback: equal weight
    return pd.Series(1.0 / len(tickers), index=tickers)


def _mvo_floor_then_cap(w, min_weight, max_weight, max_iter=20):
    """
    Post-solve weight adjustment in two clean steps:

    Step 1 — Floor:
        Raise all w_i to min_weight (including zeros), then renormalize.
        This ensures every stock gets at least min_weight.

    Step 2 — Cap:
        Reduce any w_i > max_weight to max_weight.
        Redistribute excess proportionally to all other stocks.
        Iterate until no violations remain.

    By applying floor first, all weights are non-zero before the cap step,
    so redistribution never creates new zeros.
    """
    w = w.copy().clip(lower=0)
    n = len(w)
    if n == 0:
        return w

    # ── Step 1: Floor ─────────────────────────────────────────────────────────
    w = w.clip(lower=min_weight)   # raise all to floor
    if w.sum() > 0:
        w = w / w.sum()             # renorm to sum=1

    # ── Step 2: Cap (iterative) ───────────────────────────────────────────────
    for _ in range(max_iter):
        over  = w > max_weight + 1e-9
        if not over.any():
            break
        excess       = (w[over] - max_weight).sum()
        w[over]      = max_weight
        under        = ~over                          # all non-capped stocks
        if w[under].sum() > 1e-12:
            w[under] += excess * (w[under] / w[under].sum())
        else:
            # Edge case: everything at cap, spread equally
            w += excess / n

    # Final renorm for floating point safety
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _mvo_max_alpha_portfolio(Sigma, alpha, tickers, risk_aversion=1.0,
                              max_weight=MVO_MAX_WEIGHT,
                              min_weight=MVO_MIN_WEIGHT):
    """
    Mean-variance optimal portfolio with post-solve floor and cap.

    Step 1: Solve unconstrained MVO (long-only, sum=1)
    Step 2: Apply floor — raise all w_i < min_weight to min_weight, renorm
    Step 3: Apply cap  — reduce all w_i > max_weight to max_weight,
                         redistribute excess proportionally to remaining stocks
                         (no zeros exist after step 2, so redistribution is clean)
    """
    try:
        import cvxpy as cp
        N = len(tickers)
        w = cp.Variable(N)
        a = alpha.reindex(tickers).fillna(0.0).values
        prob = cp.Problem(
            cp.Maximize(a @ w - (risk_aversion / 2) * cp.quad_form(w, Sigma)),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            w_sol = pd.Series(np.maximum(w.value, 0.0), index=tickers)
            if w_sol.sum() > 0:
                w_sol = w_sol / w_sol.sum()
            return _mvo_floor_then_cap(w_sol, min_weight, max_weight)

    except Exception as e:
        warnings.warn(f"  MVO solver failed: {e} — falling back to equal weight")

    # Fallback: equal weight
    w_eq = pd.Series(1.0 / len(tickers), index=tickers)
    return _mvo_floor_then_cap(w_eq, min_weight, max_weight)


# ===============================================================================
# MAIN DIAGNOSTIC
# ===============================================================================

def run_mvo_diagnostics(dt, Pxs_df, sectors_s, composite_scores,
                         port_n=30, universe_mult=5,
                         ic_values=None, model_version='v2',
                         volumeTrd_df=None,
                         pca_var_threshold=MVO_PCA_VAR_THRESH,
                         max_weight=MVO_MAX_WEIGHT,
                         min_weight=MVO_MIN_WEIGHT,
                         zscore_cap=MVO_ZSCORE_CAP,
                         min_matrix_count=MVO_MIN_MATRIX_COUNT):
    """
    Build and compare four covariance matrices + alpha signal
    on a given date for the top port_n * universe_mult candidate stocks.

    Parameters
    ----------
    dt               : pd.Timestamp — diagnostic date
    Pxs_df           : pd.DataFrame — price/macro panel
    sectors_s        : pd.Series
    composite_scores : pd.Series   — composite alpha z-scores on date dt
    port_n           : int — final portfolio size
    universe_mult    : int — candidate universe multiplier
    ic_values        : list of float — IC values to test (default [0.02,0.04,0.06,0.08])
    model_version    : 'v1' or 'v2'
    volumeTrd_df     : optional

    Returns
    -------
    dict with all matrices, alpha vectors, diagnostics
    """
    ic_values = ic_values or [0.02, 0.04, 0.06, 0.08]
    n_cands   = port_n * universe_mult

    print(f"\nMVO Diagnostics — {dt.date()}")
    print(f"  Portfolio size: {port_n}  |  "
          f"Candidate universe: {n_cands} (×{universe_mult})\n")

    # ── Candidate universe ────────────────────────────────────────────────────
    scores_on_dt = composite_scores.dropna()
    # Filter to tickers with price data on dt
    valid = [t for t in scores_on_dt.index
             if t in Pxs_df.columns and not np.isnan(Pxs_df.loc[dt, t])]
    scores_on_dt = scores_on_dt.reindex(valid).dropna()
    candidates   = scores_on_dt.nlargest(n_cands).index.tolist()
    N            = len(candidates)
    print(f"  Candidates available: {N} (requested {n_cands})")

    # ── Return matrix (T x N) ─────────────────────────────────────────────────
    pxs_to_dt  = Pxs_df.loc[:dt, candidates]
    ret_df     = pxs_to_dt.pct_change().dropna(how='all')
    ret_df     = ret_df.iloc[-MVO_LOOKBACK:].dropna(axis=1, how='any')
    candidates = ret_df.columns.tolist()
    N          = len(candidates)
    print(f"  Candidates after return filter: {N}  |  "
          f"Return window: {len(ret_df)} days")

    # ── 1. Empirical EWMA covariance ──────────────────────────────────────────
    print("\n[1/4] Building empirical EWMA covariance...")
    Sigma_emp = _mvo_ewma_cov(ret_df, MVO_EWMA_HL)
    cond_emp  = _mvo_condition_number(Sigma_emp)
    print(f"  Shape: {Sigma_emp.shape}  |  "
          f"Condition number: {cond_emp:.2e}")

    # ── 2. Ledoit-Wolf ────────────────────────────────────────────────────────
    print("\n[2/4] Building Ledoit-Wolf shrinkage covariance...")
    Sigma_lw, rho_bar, shrink_alpha = _mvo_ledoit_wolf(ret_df.values)
    cond_lw  = _mvo_condition_number(Sigma_lw)
    print(f"  Shrinkage intensity α: {shrink_alpha:.4f}  |  "
          f"Mean pairwise correlation: {rho_bar:.4f}")
    print(f"  Condition number: {cond_lw:.2e}")

    # ── 3. Factor-driven ──────────────────────────────────────────────────────
    print("\n[3/4] Building factor-driven covariance (XFX')...")
    # Pass full Pxs_df so _rd_build_X has full OHLC/vol lookback history
    Sigma_factor, X_df, factor_names = _mvo_factor_cov(
        candidates, Pxs_df, sectors_s, volumeTrd_df, model_version
    )
    cond_factor = _mvo_condition_number(Sigma_factor)
    print(f"  Condition number: {cond_factor:.2e}")

    # ── 4. PCA ────────────────────────────────────────────────────────────────
    print("\n[4/4] Building PCA covariance...")
    Sigma_pca, n_comp, var_expl, eigvals = _mvo_pca_cov(ret_df.values,
                                                              var_threshold=pca_var_threshold)
    cond_pca = _mvo_condition_number(Sigma_pca)
    print(f"  Components retained: {n_comp}  |  "
          f"Variance explained: {var_expl*100:.1f}%  "
          f"(threshold: {pca_var_threshold*100:.0f}%)")
    print(f"  Condition number: {cond_pca:.2e}")

    # ── 5. Ensemble ───────────────────────────────────────────────────────────
    Sigma_ens = (Sigma_emp + Sigma_lw + Sigma_factor + Sigma_pca) / 4.0
    cond_ens  = _mvo_condition_number(Sigma_ens)

    # ── Alpha signals ─────────────────────────────────────────────────────────
    print("\n  Building alpha signals...")
    # Daily vol — keeps alpha in same units as daily covariance matrix
    vol_s_daily = _mvo_ewma_vol(ret_df, MVO_EWMA_HL)              # daily std
    vol_s_ann   = vol_s_daily * np.sqrt(252)                       # for display
    z_s         = pd.Series(
        scores_on_dt.reindex(candidates).fillna(0.0).values,
        index=candidates
    )
    alpha_by_ic = {}
    for ic in ic_values:
        alpha_by_ic[ic] = _mvo_scale_alpha(z_s, vol_s_daily, ic,
                                            zscore_cap=zscore_cap)

    # ── Condition number summary ──────────────────────────────────────────────
    cond_numbers = {
        'Empirical'    : cond_emp,
        'Ledoit-Wolf'  : cond_lw,
        'Factor-driven': cond_factor,
        'PCA'          : cond_pca,
        'Ensemble'     : cond_ens,
    }

    # ── Portfolio overlap analysis ────────────────────────────────────────────
    print("\n  Computing portfolio overlaps across IC values and matrices...")
    overlap_results = _mvo_overlap_analysis(
        candidates, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens,
        alpha_by_ic, port_n, ic_values, max_weight=max_weight,
        min_weight=min_weight, min_matrix_count=min_matrix_count
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    _mvo_print_summary(cond_numbers, overlap_results, ic_values,
                        vol_s_ann, z_s, alpha_by_ic, rho_bar,
                        shrink_alpha, n_comp, var_expl)

    # ── Recommended portfolio weights ────────────────────────────────────────
    best_ic = min(ic_values,
                  key=lambda ic: abs(overlap_results[ic]['mean_overlap']
                                     - MVO_OVERLAP_TARGET))
    alpha_best = alpha_by_ic[best_ic]

    # Filter to eligible candidates (selected by >= min_matrix_count matrices)
    eligible   = overlap_results[best_ic]['eligible']
    n_eligible = len(eligible)
    n_filtered = len(candidates) - n_eligible
    if n_filtered > 0:
        print(f"  Eligibility filter: {n_eligible} eligible stocks "
              f"({n_filtered} removed — selected by < {min_matrix_count} matrices)")

    # Slice covariance matrices and alpha to eligible universe
    elig_idx   = [candidates.index(t) for t in eligible]
    S_ens_elig = Sigma_ens[np.ix_(elig_idx, elig_idx)]
    alpha_elig = alpha_best.reindex(eligible).fillna(0.0)

    w_ens_elig = _mvo_max_alpha_portfolio(
        S_ens_elig, alpha_elig, eligible, risk_aversion=1.0,
        max_weight=max_weight,
        min_weight=min_weight
    )
    # Reindex back to full candidate list (zeros for ineligible)
    w_ens = w_ens_elig.reindex(candidates).fillna(0.0)
    w_equal     = pd.Series(1.0 / port_n, index=candidates[:port_n])

    _mvo_print_portfolio(w_ens, alpha_best, vol_s_ann, z_s,
                          sectors_s, best_ic, port_n,
                          overlap_results[best_ic]['top_n'])

    # ── Plots ─────────────────────────────────────────────────────────────────
    _mvo_plot(ret_df, candidates, Sigma_emp, Sigma_lw, Sigma_factor,
              Sigma_pca, Sigma_ens, eigvals, n_comp, alpha_by_ic,
              ic_values, vol_s_ann, z_s, overlap_results, cond_numbers)

    return {
        'candidates'    : candidates,
        'cov_empirical' : pd.DataFrame(Sigma_emp,    index=candidates, columns=candidates),
        'cov_lw'        : pd.DataFrame(Sigma_lw,     index=candidates, columns=candidates),
        'cov_factor'    : pd.DataFrame(Sigma_factor, index=candidates, columns=candidates),
        'cov_pca'       : pd.DataFrame(Sigma_pca,    index=candidates, columns=candidates),
        'cov_ensemble'  : pd.DataFrame(Sigma_ens,    index=candidates, columns=candidates),
        'alpha_by_ic'   : alpha_by_ic,
        'vol_s'         : vol_s_ann,
        'z_scores'      : z_s,
        'condition_numbers' : cond_numbers,
        'overlap_results'   : overlap_results,
        'n_pca_components'  : n_comp,
        'pca_var_explained' : var_expl,
        'lw_shrinkage'      : shrink_alpha,
        'lw_rho_bar'        : rho_bar,
        'X_df'              : X_df,
        'factor_names'      : factor_names,
        'w_ensemble'        : w_ens,
        'recommended_ic'    : best_ic,
    }


# ===============================================================================
# OVERLAP ANALYSIS
# ===============================================================================

def _mvo_overlap_analysis(candidates, Sigma_emp, Sigma_lw, Sigma_factor,
                            Sigma_pca, Sigma_ens, alpha_by_ic, port_n, ic_values,
                            max_weight=MVO_MAX_WEIGHT,
                            min_weight=MVO_MIN_WEIGHT,
                            min_matrix_count=MVO_MIN_MATRIX_COUNT):
    """
    For each IC value:
      1. Run MVO with each of the 5 covariance matrices
      2. Filter candidates to those selected by >= min_matrix_count matrices
      3. Re-run MVO on filtered universe with ensemble matrix
      4. Compute pairwise portfolio overlaps

    Returns dict {ic: {matrix_pair: overlap_fraction, eligible: list}}.
    """
    matrices = {
        'Empirical'    : Sigma_emp,
        'Ledoit-Wolf'  : Sigma_lw,
        'Factor-driven': Sigma_factor,
        'PCA'          : Sigma_pca,
        'Ensemble'     : Sigma_ens,
    }
    results = {}

    for ic in ic_values:
        alpha    = alpha_by_ic[ic]
        top_n_by = {}

        for mname, Sigma in matrices.items():
            w = _mvo_max_alpha_portfolio(Sigma, alpha, candidates,
                                          risk_aversion=1.0,
                                          max_weight=max_weight,
                                          min_weight=min_weight)
            top_n_by[mname] = w.nlargest(port_n).index.tolist()

        # Count how many matrices selected each stock
        # Exclude Ensemble from count (it's derived from the others)
        count_matrices = [m for m in matrices.keys() if m != 'Ensemble']
        selection_count = {}
        for ticker in candidates:
            selection_count[ticker] = sum(
                1 for m in count_matrices if ticker in top_n_by[m]
            )

        # Eligible stocks: selected by >= min_matrix_count matrices
        eligible = [t for t in candidates
                    if selection_count.get(t, 0) >= min_matrix_count]

        # Pairwise overlaps (on original top_n_by for comparability)
        pairs  = {}
        mnames = list(matrices.keys())
        for i in range(len(mnames)):
            for j in range(i + 1, len(mnames)):
                m1, m2  = mnames[i], mnames[j]
                overlap = _mvo_portfolio_overlap(top_n_by[m1], top_n_by[m2])
                pairs[f"{m1} vs {m2}"] = overlap

        results[ic] = {
            'top_n'        : top_n_by,
            'overlaps'     : pairs,
            'mean_overlap' : np.mean(list(pairs.values())),
            'eligible'     : eligible,
            'selection_count': selection_count,
        }

    return results


# ===============================================================================
# PORTFOLIO WEIGHT DISPLAY
# ===============================================================================

def _mvo_print_portfolio(w_ens, alpha, vol_s_ann, z_s, sectors_s,
                          best_ic, port_n, top_n_by_matrix):
    """
    Print MVO ensemble portfolio vs equal-weight top-N.
    Shows weights, alpha, vol, z-score and sector for each stock.
    Also shows which matrices agree on each stock.
    """
    # Top positions by weight — only show non-zero allocations
    w_top = w_ens[w_ens > 1e-4].nlargest(port_n)

    # Stocks selected by each matrix
    matrix_selections = {m: set(tickers)
                         for m, tickers in top_n_by_matrix.items()}

    print("\n" + "=" * 90)
    print(f"  MVO ENSEMBLE PORTFOLIO  (IC={best_ic}, top {port_n} by weight)")
    print("=" * 90)
    print(f"  {'Ticker':<8}  {'Weight%':>8}  {'AnnAlpha%':>10}  "
          f"{'AnnVol%':>8}  {'Z-score':>8}  {'Sector':<28}  {'In matrices'}")
    print("  " + "-" * 88)

    matrix_names = list(matrix_selections.keys())
    for ticker in w_top.index:
        w    = w_top[ticker] * 100
        a    = alpha.get(ticker, 0.0) * 252 * 100
        v    = vol_s_ann.get(ticker, 0.0) * 100
        z    = z_s.get(ticker, 0.0)
        sec  = sectors_s.get(ticker, 'Unknown')
        # Which matrices selected this stock?
        in_m = [m[0] for m in matrix_names if ticker in matrix_selections[m]]
        in_m_str = '/'.join(in_m) if in_m else '-'
        print(f"  {ticker:<8}  {w:>7.2f}%  {a:>+9.1f}%  "
              f"{v:>7.1f}%  {z:>+8.3f}  {sec:<28}  {in_m_str}")

    # Summary stats
    total_w   = w_top.sum() * 100
    hhi       = (w_top ** 2).sum()   # Herfindahl index
    eff_n     = 1.0 / hhi if hhi > 0 else 0  # effective N
    ticker_sectors = pd.Series(
        {t: sectors_s.get(t, 'Unknown') for t in w_top.index}
    )
    sec_conc = w_top.groupby(ticker_sectors).sum().sort_values(ascending=False)

    print(f"\nTotal weight shown: {total_w:.1f}%  |  "
f"Effective N (1/HHI): {eff_n:.1f}  |  "
f"Top stock weight: {w_top.max()*100:.1f}%")
    print(f"\nSector concentration:")
    for sec, wt in sec_conc.items():
        bar = '█' * int(wt * 100 / 2)
        print(f"    {sec:<28}  {wt*100:>5.1f}%  {bar}")

    # Comparison: how many stocks overlap with equal-weight top-N
    equal_top = set(w_ens.nlargest(port_n).index)
    # Equal weight just uses the top port_n by alpha score
    alpha_top  = set(alpha.nlargest(port_n).index)
    overlap_vs_alpha = len(equal_top & alpha_top) / len(equal_top | alpha_top)
    print(f"\nMVO vs pure-alpha-rank overlap: {overlap_vs_alpha:.1%}")
    print(f"  (stocks added by MVO diversification: "
          f"{sorted(equal_top - alpha_top)[:10]})")
    print("=" * 90 + "\n")


# ===============================================================================
# PRINT SUMMARY
# ===============================================================================

def _mvo_print_summary(cond_numbers, overlap_results, ic_values,
                        vol_s, z_s, alpha_by_ic, rho_bar,
                        shrink_alpha, n_comp, var_expl):
    print("\n" + "=" * 72)
    print("  MVO DIAGNOSTICS SUMMARY")
    print("=" * 72)

    # Condition numbers
    print("\n  -- Covariance Matrix Condition Numbers --")
    print(f"  {'Matrix':<20}  {'Condition #':>14}  {'Log10':>8}")
    print("  " + "-" * 46)
    for name, cn in cond_numbers.items():
        print(f"  {name:<20}  {cn:>14.2e}  {np.log10(cn):>8.2f}")

    print(f"\n  LW shrinkage intensity: {shrink_alpha:.4f}  |  "
          f"Mean correlation: {rho_bar:.4f}")
    print(f"  PCA components: {n_comp}  |  "
          f"Variance explained: {var_expl*100:.1f}%")

    # Alpha signal summary
    print("\n  -- Alpha Signal Distribution --")
    print(f"  {'IC':>6}  {'Ann. alpha mean':>16}  "
          f"{'p5':>8}  {'p95':>8}  {'Max':>8}")
    print("  " + "-" * 54)
    for ic in ic_values:
        a = alpha_by_ic[ic]
        # alpha is in daily units — annualise for display
        print(f"  {ic:>6.3f}  {a.mean()*252*100:>+15.2f}%  "
              f"{np.percentile(a,5)*252*100:>+8.2f}%  "
              f"{np.percentile(a,95)*252*100:>+8.2f}%  "
              f"{a.max()*252*100:>+8.2f}%")

    # Portfolio overlap
    print("\n  -- Portfolio Overlap Across Matrices (top-N) --")
    print(f"  {'IC':>6}  {'Mean overlap':>13}  {'vs target ({:.0%})'.format(MVO_OVERLAP_TARGET):>18}")
    print("  " + "-" * 42)
    for ic in ic_values:
        mo = overlap_results[ic]['mean_overlap']
        flag = "  ✓" if abs(mo - MVO_OVERLAP_TARGET) < 0.10 else ""
        print(f"  {ic:>6.3f}  {mo:>13.1%}{flag}")

    # Detailed overlap for recommended IC
    best_ic = min(ic_values,
                  key=lambda ic: abs(overlap_results[ic]['mean_overlap']
                                     - MVO_OVERLAP_TARGET))
    print(f"\n  Recommended IC: {best_ic} "
          f"(overlap {overlap_results[best_ic]['mean_overlap']:.1%} "
          f"≈ target {MVO_OVERLAP_TARGET:.0%})")
    print(f"\n  Pairwise overlaps at IC={best_ic}:")
    for pair, ov in overlap_results[best_ic]['overlaps'].items():
        print(f"    {pair:<40}  {ov:.1%}")

    print("=" * 72 + "\n")


# ===============================================================================
# PLOTS
# ===============================================================================

def _mvo_plot(ret_df, candidates, Sigma_emp, Sigma_lw, Sigma_factor,
              Sigma_pca, Sigma_ens, eigvals, n_comp, alpha_by_ic,
              ic_values, vol_s, z_s, overlap_results, cond_numbers):
    """
    Six panels:
      1. Correlation heatmaps (2x2 grid: empirical, LW, factor, PCA)
      2. Eigenvalue spectrum (log scale) + cumulative variance
      3. Alpha signal distribution per IC value
      4. Vol vs alpha scatter (alpha-vol tension)
      5. Portfolio overlap vs IC
      6. Diagonal comparison (per-stock variance across matrices)
    """
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('#FAFAF9')
    gs  = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)

    COLORS = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#EF9F27']

    def to_corr(Sigma):
        std   = np.sqrt(np.diag(Sigma))
        outer = np.outer(std, std)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(outer > 0, Sigma / outer, 0.0)

    # ── Panel 1-4: Correlation heatmaps ──────────────────────────────────────
    hmap_specs = [
        (Sigma_emp,    'Empirical',     gs[0, 0]),
        (Sigma_lw,     'Ledoit-Wolf',   gs[0, 1]),
        (Sigma_factor, 'Factor-driven', gs[0, 2]),
        (Sigma_pca,    'PCA',           gs[1, 0]),
        (Sigma_ens,    'Ensemble',      gs[1, 1]),
    ]
    # Sort stocks by first principal component for better visual structure
    try:
        ev, evec = np.linalg.eigh(Sigma_ens)
        pc1_order = np.argsort(evec[:, -1])
    except Exception:
        pc1_order = np.arange(len(candidates))

    for Sigma, title, g in hmap_specs:
        ax  = fig.add_subplot(g)
        ax.set_facecolor('#FAFAF9')
        C   = to_corr(Sigma)[np.ix_(pc1_order, pc1_order)]
        im  = ax.imshow(C, cmap='RdYlGn', vmin=-0.5, vmax=1.0,
                        aspect='auto', interpolation='nearest')
        ax.set_title(f"{title}\n(cond={cond_numbers.get(title, 0):.1e})",
                     fontsize=9, fontweight='500')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    # ── Panel 5: Eigenvalue spectrum ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#FAFAF9')
    n_show = min(50, len(eigvals))
    ax5.semilogy(range(1, n_show + 1), eigvals[:n_show],
                 color='#378ADD', linewidth=1.5, label='Eigenvalues')
    ax5.axvline(n_comp, color='#D85A30', linewidth=1.2, linestyle='--',
                label=f'PCA cutoff (k={n_comp})')
    ax5.set_xlabel("Component", fontsize=9)
    ax5.set_ylabel("Eigenvalue (log)", fontsize=9)
    ax5.set_title(f"Eigenvalue spectrum\n"
                  f"({MVO_PCA_VAR_THRESH*100:.0f}% var → {n_comp} components)",
                  fontsize=9, fontweight='500')
    ax5.legend(fontsize=8)
    ax5.grid(color='#D3D1C7', linewidth=0.5)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # ── Panel 6: Alpha distribution ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor('#FAFAF9')
    for i, ic in enumerate(ic_values):
        a_ann = alpha_by_ic[ic] * 252 * 100
        ax6.hist(a_ann.values, bins=25, alpha=0.5,
                 color=COLORS[i % len(COLORS)], label=f'IC={ic}')
    ax6.axvline(0, color='#888780', linewidth=0.8, linestyle='--')
    ax6.set_xlabel("Annualized alpha (%)", fontsize=9)
    ax6.set_ylabel("Count", fontsize=9)
    ax6.set_title("Alpha signal distribution\n(Grinold-Kahn scaling)",
                  fontsize=9, fontweight='500')
    ax6.legend(fontsize=8)
    ax6.grid(color='#D3D1C7', linewidth=0.5)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # ── Panel 7: Vol vs alpha scatter ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor('#FAFAF9')
    ic_mid = ic_values[len(ic_values) // 2]
    a_mid  = alpha_by_ic[ic_mid] * 252 * 100
    v_ann  = vol_s * 100
    ax7.scatter(v_ann.values, a_mid.values,
                c=z_s.values, cmap='RdYlGn', alpha=0.6, s=20)
    ax7.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax7.set_xlabel("Annualized vol (%)", fontsize=9)
    ax7.set_ylabel(f"Alpha % (IC={ic_mid})", fontsize=9)
    ax7.set_title("Alpha-vol tension\n(color = composite z-score)",
                  fontsize=9, fontweight='500')
    ax7.grid(color='#D3D1C7', linewidth=0.5)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # ── Panel 8: Portfolio overlap vs IC ──────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor('#FAFAF9')
    mean_overlaps = [overlap_results[ic]['mean_overlap'] for ic in ic_values]
    ax8.plot([str(ic) for ic in ic_values], mean_overlaps,
             color='#7F77DD', linewidth=1.8, marker='o', markersize=6)
    ax8.axhline(MVO_OVERLAP_TARGET, color='#1D9E75', linewidth=1.0,
                linestyle='--', label=f'Target ({MVO_OVERLAP_TARGET:.0%})')
    ax8.set_xlabel("IC value", fontsize=9)
    ax8.set_ylabel("Mean portfolio overlap", fontsize=9)
    ax8.set_title("Portfolio overlap across matrices\nvs IC value",
                  fontsize=9, fontweight='500')
    ax8.legend(fontsize=8)
    ax8.set_ylim(0, 1)
    ax8.grid(color='#D3D1C7', linewidth=0.5)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)

    # ── Panel 9: Per-stock variance comparison ────────────────────────────────
    ax9 = fig.add_subplot(gs[3, :])
    ax9.set_facecolor('#FAFAF9')
    # Sort stocks by empirical variance
    emp_var   = np.diag(Sigma_emp) * 252 * 100
    sort_idx  = np.argsort(emp_var)
    x         = np.arange(len(candidates))
    for Sigma, name, color in [
        (Sigma_emp,    'Empirical',     '#888780'),
        (Sigma_lw,     'Ledoit-Wolf',   '#378ADD'),
        (Sigma_factor, 'Factor-driven', '#1D9E75'),
        (Sigma_pca,    'PCA',           '#D85A30'),
    ]:
        diag_ann = np.diag(Sigma)[sort_idx] * 252 * 100
        ax9.plot(x, diag_ann, label=name, linewidth=1.0, alpha=0.8)
    ax9.set_xlabel("Stock (sorted by empirical variance)", fontsize=9)
    ax9.set_ylabel("Annualized variance (%²)", fontsize=9)
    ax9.set_title("Per-stock variance across matrices",
                  fontsize=9, fontweight='500')
    ax9.legend(fontsize=8, ncol=4, loc='upper left')
    ax9.grid(color='#D3D1C7', linewidth=0.5)
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)

    plt.suptitle(f"MVO Diagnostics — {ret_df.index[-1].date()}  "
                 f"(N={len(candidates)} candidates)",
                 fontsize=13, fontweight='500', y=1.005)
    plt.show()
    return fig
