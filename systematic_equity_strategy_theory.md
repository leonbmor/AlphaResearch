# Systematic Equity Strategy — Theoretical Framework
*Updated June 2026*

> **Deployed sequence (as of June 2026):**
> Beta → Quality → Idio Momentum → Size → Value → SI → GK Vol → Macro → Sectors → O-U → Sub-sectors
>
> A sectors-first reordering (Beta → Sectors → Size → Quality → Value → SI → GK Vol → Idio Mom → Macro → O-U → Sub-sectors) was evaluated theoretically and tested empirically but was not adopted in production — results deteriorated under the new sequence, likely because the original ordering preserves more alpha signal in the early steps for this specific universe and sample period. The theoretical case for sectors-first is documented in Section I for reference but does not reflect the deployed model.

---

## Overview

This document describes the theoretical foundations of a systematic US large-cap equity strategy spanning approximately 701 stocks. The framework integrates four distinct but interrelated modules: a cross-sectional factor model that constructs a multi-dimensional alpha signal from fundamental and technical data; a mean-variance portfolio optimisation engine that translates that signal into optimal weights; a dynamic rebalancing and risk management overlay; and a momentum-aware exclusion filter that guards against allocating to stocks exhibiting crowded, potentially reversal-prone price dynamics. Each module is discussed in terms of its theoretical motivation and the statistical machinery employed.

---

## I. Cross-Sectional Factor Model

### 1.1 The Fama-MacBeth Framework

The starting point is the classical cross-sectional regression framework of Fama and MacBeth (1973). At each date *t*, stock returns are regressed cross-sectionally against a set of characteristics:

$$r_{i,t} = \sum_{k} \lambda_{k,t} \cdot f_{k,i,t} + \varepsilon_{i,t}$$

where $r_{i,t}$ is the return of stock *i*, $f_{k,i,t}$ are stock-level factor exposures, $\lambda_{k,t}$ are the cross-sectional factor premia estimated at *t*, and $\varepsilon_{i,t}$ is the idiosyncratic residual. Factor premia are estimated independently at each date, allowing them to vary with market conditions — a feature well-suited to the time-varying nature of risk premia documented by Cochrane (2011) and Fama and French (1997). The time series of estimated premia informs the expected return model used downstream in portfolio construction.

All cross-sectional regressions are estimated using **Weighted Least Squares (WLS)** with weights equal to the log of market capitalisation, normalised to sum to unity. This ensures that large-cap stocks — which dominate portfolio construction — have proportionally greater influence in determining factor premia, while still preserving the information content of smaller names.

### 1.2 Sequential Orthogonalisation — Gram-Schmidt in Factor Space

The central design choice of the factor model is the **sequential stripping** of systematic return sources through a Gram-Schmidt orthogonalisation procedure. This is in contrast to the conventional approach of including all factors simultaneously in a joint regression, as in Fama and French (1993, 2015) or Carhart (1997).

The procedure works as follows. Before each new factor *k* enters the model, it is regressed cross-sectionally against all previously estimated factors. Only the **orthogonal residual** — the component of the new characteristic that is linearly independent of all prior characteristics — enters the regression. Simultaneously, the return series being explained is the residual from the prior step:

$$\text{Step } k: \quad r^{(k-1)}_{i,t} = \lambda_{k,t} \cdot f^{\perp}_{k,i,t} + r^{(k)}_{i,t}$$

where $r^{(k-1)}$ is the residual entering step *k*, $f^{\perp}_{k}$ is the characteristic orthogonalised to all prior factors, and $r^{(k)}$ is the residual exiting step *k*. This orthogonalisation is performed in the WLS inner product space defined by market-cap weights, ensuring that the decomposition is geometrically consistent with the estimation metric.

The rationale is threefold. First, it prevents multicollinearity from contaminating factor premia estimates — each estimated coefficient measures the marginal contribution of a factor after controlling exactly for all prior factors, addressing the concern raised by Harvey, Liu and Zhu (2016) about spurious factor discovery. Second, it produces clean, non-overlapping residuals at each step. Third, the ordering reflects an economic prior about the hierarchy of systematic return sources, described in detail below.

### 1.3 Factor Hierarchy — Current Sequence (June 2026 Revision)

The model strips systematic variation in eleven sequential steps. The factor ordering reflects an economic prior: alpha signals (Quality, Momentum) enter early to ensure their premia are estimated with maximum precision, structural factors (Size, SI, Vol) follow, and macro/sector overlays come last to prevent their broad cross-sectional variation from absorbing alpha variance.

**The current sequence is:**

$$\text{Beta} \rightarrow \text{Quality} \rightarrow \text{Idio Momentum} \rightarrow \text{Size} \rightarrow \text{Value} \rightarrow \text{SI} \rightarrow \text{GK Vol} \rightarrow \text{Macro} \rightarrow \text{Sectors} \rightarrow \text{O-U} \rightarrow \text{Sub-sectors}$$

A new sub-sector layer (Step 11) was added in 2026 as a granular extension of the sector step. The theoretical motivations for the ordering are developed under the individual step descriptions below.

**Step 1 — Market beta.** EWMA-estimated covariance of each stock's returns with the market index, divided by the EWMA variance of the index. The CAPM of Sharpe (1964) and Lintner (1965) motivates this as the single most pervasive systematic source. Exponential weighting (Riskmetrics, 1996) captures time-variation in betas, consistent with the evidence of beta instability documented by Blume (1975) and Ferson and Harvey (1999).

**Step 2 — Sector effects.** A full set of sector dummies encoded using **sum-to-zero deviation coding**: each dummy equals +1 for the stock's own sector and $-1/(K-1)$ for all other sectors, where *K* is the number of sectors. The sector dummies are **fully orthogonalised** against market beta before the Ridge regression step. This placement — immediately after market beta — is the standard adopted by commercial risk models (Barra, Axioma) and reflects the view that sector membership is the most fundamental structural characteristic after market exposure. Placing sectors at Step 2 ensures that every downstream factor — Quality, Size, Value, Momentum — is estimated within-sector by construction, without any additional explicit sector-neutralisation steps. Moskowitz and Grinblatt (1999) document persistent industry return premia; placing the sector step early prevents those premia from contaminating style factor estimates. Prior to this revision, sectors were estimated at Step 10 (penultimate), which caused style factors to absorb sector-level variation — inflating quality and momentum premia for sectors with systematically high quality or strong trends.

**Step 3 — Size.** Cross-sectional log market capitalisation, updated daily, orthogonalised to beta and sectors. Banz (1981) documented the size effect; Fama and French (1993) formalised it as SMB. Placing size after sectors ensures that the estimated size premium is the within-sector size effect — large-cap Technology stocks are compared to small-cap Technology stocks, not to utilities or REITs. This is consistent with Asness, Frazzini, Israel and Moskowitz (2018) who show the size premium is robust only after controlling for quality, and with the standard treatment in commercial factor models where size is estimated net of industry structure.

**Step 4 — Quality.** A composite quality signal blended from two sub-composites calibrated to different rate regimes, orthogonalised to beta, sectors and size. The quality premium is formalised by Asness, Frazzini and Pedersen (2019) in the "Quality Minus Junk" (QMJ) factor. Novy-Marx (2013) documents that gross profitability is the most powerful quality predictor of future returns. With sectors at Step 2, Quality now measures within-sector quality — a pharmaceutical company is compared to other pharmaceutical companies, not to utilities. This is philosophically consistent with how quality scores are constructed (within-sector ranking) and eliminates the earlier inconsistency where scores were ranked within-sector but regressed before sector effects were removed. The rate-regime conditioning reflects the finding of Asness et al. (2019) that quality premia are time-varying and partially explained by monetary conditions.

**Quality PIT weight update.** Quality point-in-time IC weights are re-estimated at each rebalancing using the **sector residual** (the model residual after Steps 1-2) as the forward return target. This ensures IC estimates are computed on sector-neutral returns, preventing sector rotation from inflating or deflating quality IC estimates. Prior to the June 2026 revision, quality IC was computed against market-adjusted returns (Step 1 residual).

**Step 5 — Value.** IC-weighted composite of seven valuation ratios (price-to-sales, price-to-earnings on estimated and operating bases, price-to-gross-profit), orthogonalised to beta, sectors, size and quality. The value premium is among the oldest documented anomalies: Basu (1977), Fama and French (1992), Lakonishok, Shleifer and Vishny (1994). Placing value after sectors ensures within-sector valuation comparisons — a P/E of 15x in software means something entirely different from 15x in utilities, and the sector step ensures these structural differences are already removed before value enters the regression. Placing value after quality ensures the value premium measures mispricing net of fundamental quality differences, addressing the criticism of Graham and Dodd value measures that they conflate cheap but fundamentally deteriorating businesses with genuinely mispriced ones.

**Value PIT weight update.** Value IC weights are estimated using the **quality residual** (Step 4 residual) as the forward return target — sector, size and quality effects are already stripped, so the IC reflects the incremental predictive power of valuation above and beyond those structural factors.

**Step 6 — Short Interest composite.** Cross-sectional short interest metrics, orthogonalised to all prior factors. The information content of short interest is documented by Asquith, Pathak and Ritter (2005), Diether, Malloy and Scherbina (2002), and Boehmer, Jones and Zhang (2008).

**Step 7 — GK Volatility.** The Garman-Klass (1980) estimator of daily variance using open, high, low and close prices:

$$\sigma^2_{GK} = 0.5(\ln H/L)^2 - (2\ln 2 - 1)(\ln C/O)^2$$

Garman and Klass (1980) show this estimator is approximately 7.4 times more efficient than close-to-close variance. Ang, Hodrick, Xing and Zhang (2006) document both a positive variance risk premium and a negative idiosyncratic volatility premium.

**Step 8 — Idiosyncratic Momentum.** Computed on the **GK Volatility residual** — the return series after stripping market beta, sectors, size, quality, value, short interest and volatility. This is a significant departure from prior practice where momentum was computed on the quality residual (Step 3 in the earlier sequence). By deferring momentum to Step 8, the signal measures persistence in stock returns after removing all structural, fundamental and volatility-related explanations. The resulting momentum is genuinely idiosyncratic in the sense intended by Blitz, Huij and Martens (2011) in their residual momentum framework: it captures behavioural persistence in stock-specific information processing, not momentum that can be attributed to sector rotation, quality persistence, value mean-reversion or volatility clustering. This placement aligns with commercial factor model practice where momentum is estimated on a residual series that has already been stripped of the most important systematic sources. The momentum premium is among the most robust findings in empirical asset pricing, documented by Jegadeesh and Titman (1993, 2001). Volume-scaling follows Lee and Swaminathan (2000).

**Step 9 — Macro factors.** Seven macroeconomic sensitivities: yield curve level, slope, real yields, inflation breakevens, rate volatility (MOVE index), oil, and gold. Motivated by the Intertemporal CAPM of Merton (1973) and the Arbitrage Pricing Theory of Ross (1976). Chen, Roll and Ross (1986) and Petkova (2006) provide empirical grounding. Macro factors are estimated via **joint Ridge regression** (Hoerl and Kennard, 1970) against the idiosyncratic momentum residual, with the Ridge penalty selected by k-fold cross-validation at each date. Macro characteristics are **fully orthogonalised** against all prior factors before the Ridge regression — this ensures macro betas capture genuine macro sensitivity rather than characteristics correlated with sector membership, size or style. Prior to the June 2026 revision, macro characteristics were entered as raw EWMA betas without prior orthogonalisation.

**Step 10 — Ornstein-Uhlenbeck mean reversion.** An AR(1) model is fitted to the compounded cumulative residual price index for each stock, treating the idiosyncratic price path as a discrete-time Ornstein-Uhlenbeck process. The continuous-time OU process (Uhlenbeck and Ornstein, 1930) describes a stochastic variable that reverts to a long-run mean at a rate proportional to its current displacement:

$$dX_t = \kappa(\mu - X_t)\,dt + \sigma\,dW_t$$

where $X_t$ is the log-cumulative residual return, $\mu$ is the equilibrium level, $\kappa > 0$ is the speed of mean reversion, $\sigma$ is the diffusion coefficient, and $W_t$ is a standard Brownian motion. The discrete-time equivalent, estimated by OLS, is:

$$\Delta X_t = a + b X_{t-1} + \eta_t$$

where $b \in (-1, 0)$ implies mean reversion, and the characteristic half-life is $\tau = -\ln(2)/\ln(1+b)$. The **OU score** for stock $i$ is defined as the standardised deviation of the current cumulative residual from its estimated equilibrium. This framework is fully developed in Avellaneda and Lee (2010). With the revised factor ordering, OU residuals are fitted on the **macro residual** — the cleanest available signal after all eleven prior systematic sources have been stripped, including the new sector-first ordering that ensures the OU process tracks genuinely idiosyncratic price dynamics.

**Step 11 — Sub-sector dummies.** A granular extension of the sector step, estimated as Ridge regression of 45 sub-sector binary dummies against the OU residual. Sub-sector dummies are **fully orthogonalised** against all ten prior factors before estimation. Each sub-sector dummy is binary (1 for a stock's own sub-sector, 0 otherwise). The sub-sector layer captures within-sector thematic premia — AI Accelerators vs Analog Semis within Hardware, Cloud Platforms vs Cybersecurity within Software — that the broad sector dummies at Step 2 cannot distinguish. The minimum membership threshold (five stocks per sub-sector) guards against over-fitting in small sub-sectors.

The full variance reduction achieved by the eleven-step model over the 2017-2026 sample is approximately 53.5% of cross-sectional variance, distributed as follows (% of raw UFV):

| Step | Factor | %UFV remaining | Incremental |
|------|--------|---------------|-------------|
| — | Raw | 100.00% | — |
| 1 | Beta | 75.07% | 24.93pp |
| 2 | Sectors | 70.66% | 4.41pp |
| 3 | Size | 69.84% | 0.82pp |
| 4 | Quality | 58.46% | 11.38pp |
| 5 | Value | 57.59% | 0.87pp |
| 6 | SI | 56.97% | 0.62pp |
| 7 | GK Vol | 55.84% | 1.13pp |
| 8 | Idio Momentum | 54.74% | 1.10pp |
| 9 | Macro | 51.63% | 3.11pp |
| 10 | O-U | 49.58% | 2.05pp |
| 11 | Sub-sectors | 48.45% | 1.13pp |

---

## II. Composite Alpha Signal and Factor Weight Estimation

### 2.1 Information Coefficient Framework

The outputs of the factor model are used to estimate the Information Coefficient (IC) of each factor at each date. The IC is the Spearman rank correlation between the factor score (measured at date *t*) and the subsequent stock return over forward horizon *h*:

$$IC_{k,t,h} = \text{Spearman}\left(f_{k,i,t},\ r_{i,t \to t+h}\right)$$

The IC framework is the standard evaluation metric in quantitative equity research (Grinold and Kahn, 2000). Spearman rank correlation is used rather than Pearson to reduce sensitivity to outliers and tail events. The Fundamental Law of Active Management (Grinold, 1989) establishes that the information ratio of an active strategy is approximately $IC \times \sqrt{BR}$, where *BR* is the number of independent bets.

### 2.2 Regime-Conditional Factor Weights

Factor weights vary across three macroeconomic regimes defined by the interest rate environment. Within each regime, factor weights are estimated as the IC-weighted average of each factor's contribution, with a regularisation floor that prevents any factor from receiving zero weight (preventing overfit to short in-sample windows). The regime-conditional approach follows the evidence of Asness et al. (2015) and Ilmanen (2011) that factor premia are time-varying and partially predictable from macro state variables. The resulting point-in-time composite alpha signal is the primary input to portfolio construction.

---

## III. Portfolio Construction — Mean-Variance Optimisation

### 3.1 The MVO Problem

The portfolio construction follows the mean-variance framework of Markowitz (1952, 1959). Given composite alpha signal $\hat{\alpha}_i$, portfolio weights $w$ are chosen to solve:

$$\max_w \quad w^\top \hat{\alpha} - \frac{\lambda}{2} w^\top \Sigma w$$

subject to full investment, position bounds, and sector concentration limits. The risk aversion parameter $\lambda$ controls the signal-exploitation/diversification trade-off. The alpha signal is pre-processed by winsorising extreme observations (Huber, 1981) and calibrated by an IC parameter that sets the economic magnitude of the signal, consistent with the signal-to-noise ratio interpretation of Grinold and Kahn (2000).

Michaud (1989) demonstrated that naive MVO is highly sensitive to estimation error in both expected returns and the covariance matrix — a problem he termed "error maximisation." The ensemble covariance approach and robust signal construction described below directly address this concern.

### 3.2 Covariance Matrix Estimation — Ensemble Approach

With approximately 701 stocks, the sample covariance matrix has far more parameters than can be reliably estimated from a few hundred daily returns. The model employs an **ensemble of four covariance estimators**:

**Estimator 1 — EWMA Empirical Covariance.** The sample covariance matrix computed on a 63-day trailing window with exponential weighting. The EWMA approach was popularised by Riskmetrics (1996) and captures short-term dynamics in the correlation structure, which are known to spike during stress episodes (Longin and Solnik, 2001).

**Estimator 2 — Ledoit-Wolf Shrinkage.** The Oracle Approximating Shrinkage (OAS) estimator of Chen, Wiesel, Eldar and Hero (2010), which shrinks the sample covariance towards a scaled identity matrix:

$$\hat{\Sigma}_{LW} = (1 - \rho) \cdot S + \rho \cdot \mu \cdot I$$

where $S$ is the sample covariance, $\mu = \text{tr}(S)/N$ is the mean eigenvalue, and $\rho$ is chosen analytically to minimise expected Frobenius norm estimation error. This builds on the foundational work of Ledoit and Wolf (2004, 2012).

**Estimator 3 — Factor Structure ($XFX^\top$).** A structured estimator built from the factor model's output:

$$\hat{\Sigma}_{Factor} = X F X^\top + D$$

where $X$ is the $N \times K$ matrix of factor exposures, $F$ is the $K \times K$ factor covariance matrix estimated by EWMA on the time series of cross-sectional factor premia $\lambda_{k,t}$, and $D$ is a diagonal matrix of idiosyncratic variances. This is the standard structured covariance of Barra-type factor models, following Connor and Korajczyk (1988) and Fan, Fan and Lv (2008). The factor covariance matrix $F$ is built from the full eleven-factor lambda series (including macro and sector sub-blocks), with exponential weighting at a 252-day half-life over a 504-day trailing window.

**Estimator 4 — PCA Covariance.** Eigenvalue decomposition of the sample covariance, retaining components that explain a threshold fraction of variance. The regularisation effect of truncated PCA is studied by Johnstone and Lu (2009), who show that in high-dimensional settings, discarding small eigenvalues reduces estimation error in the reconstructed covariance. This estimator captures systematic risk components missed by the factor model — latent themes not explicitly modelled.

**Ensemble combination.** Stocks eligible for the two-stage MVO must appear in at least two of the four estimators' valid stock sets. The ensemble covariance is the simple average $(\hat{\Sigma}_E + \hat{\Sigma}_{LW} + \hat{\Sigma}_F + \hat{\Sigma}_{PCA})/4$. Greenberg, McNeil and Krishnan (2016) show that model averaging of covariance estimators reduces out-of-sample portfolio variance beyond any single estimator, consistent with the forecast combination results of Genre et al. (2013).

---

## IV. Dynamic Rebalancing and Regime Overlay

### 4.1 Regime-Dependent Portfolio Selection

Three portfolio constructions are maintained in parallel — pure alpha, covariance-optimised, and hybrid — with the deployed construction selected based on current drawdown regime. This regime-switching approach is motivated by the work of Guidolin and Timmermann (2007) on regime-switching in optimal portfolio choice, and by the empirical evidence in Asness, Frazzini and Pedersen (2019) that the relative performance of quality and momentum factors varies systematically with credit and macro conditions.

### 4.2 Turnover-Based Rebalancing

The portfolio is rebalanced when the implied turnover against the current drifted holdings exceeds a regime-specific threshold. The turnover is computed against the **drifted portfolio** — the actual holdings as evolved by price moves — not the theoretical static weights. This correctly measures required trades and prevents over-counting turnover in trending markets, following the approach of Platen and Heath (2006) and the practical guidance of Grinold and Kahn (2000) on portfolio rebalancing costs.

The interaction between transaction costs and optimal rebalancing frequency is studied by Leland (2000) and Lynch and Balduzzi (2000), who show that no-trade regions are optimal when costs are proportional and that the optimal region widens with higher costs — consistent with the use of higher TO thresholds (0.30 in hybrid, 0.35 in MVO regime) relative to the alpha regime (0.25).

**Minimum holding period.** A minimum holding period (10 calendar days) gates the turnover trigger: even if computed turnover exceeds the threshold, rebalancing is blocked if insufficient time has elapsed since the last rebalancing. This prevents whipsaw in volatile markets where the portfolio drifts above and below the threshold in quick succession. Regime switches (alpha → hybrid → MVO as drawdown deepens) are similarly gated, preventing rapid portfolio reconstitution in fast-moving drawdowns. The 10-calendar-day minimum corresponds approximately to 7-8 trading days.

---

## V. Macro Hedge Overlay

### 5.1 Hedge Signal Construction

The hedge signal is derived from two indicators. The **MOVE-based indicator** uses rate volatility as a proxy for macro stress, filtered by its correlation with equity returns to ensure the hedge fires only during rates-driven selloffs — consistent with the evidence of Ilmanen (2003) on the time-varying equity-bond correlation and its relationship to the inflation regime. The **QBD (Quality Breadth Dispersion) indicator** measures the relative breadth of high- vs. low-quality stocks above their moving averages, capturing flight-to-quality dynamics documented by Asness, Frazzini and Pedersen (2019).

The AND-logic combination of indicators reduces false positive rates at the cost of some sensitivity, following the signal combination methodology of Ferson and Harvey (1993) who show that conditioning on multiple macro state variables improves out-of-sample forecasting of equity premia.

### 5.2 Instrument Sizing

Hedge size is proportional to beta times effectiveness score, consistent with the minimum-variance hedge ratio derivation of Johnson (1960) and the beta-scaled hedging approach of Stoll and Whaley (1993). The beta is estimated on a trailing rolling window using OLS, and the effectiveness score is a smoothed measure of the instrument's historical contribution to risk-adjusted performance during active hedge periods.

---

## VI. Drawdown Policy — Multi-Level De-grossing

The compounding structure of the de-grossing schedule is motivated by the Kelly criterion literature (Kelly, 1956; Thorp, 2006) and its extension to drawdown management. The Kelly framework implies that optimal bet size shrinks as the portfolio suffers losses, because the geometric mean of portfolio returns is maximised by sizing proportional to the current portfolio value rather than a fixed dollar amount. The sequential de-grossing schedule implements a discrete approximation of this principle: each successive level cuts a fraction of the *remaining* exposure, producing a geometrically declining exposure profile as the drawdown deepens.

The re-entry mechanism using a theoretical reference portfolio (the fully-invested MVO portfolio) during the de-grossed period follows the approach of Grossman and Zhou (1993), who derive the optimal consumption-investment policy with a drawdown constraint and show that the optimal policy involves monitoring a reference process to determine the re-entry point. The persistence confirmation requirement (5 consecutive days of recovery) is a practical implementation of the idea that regime switches should require evidence to be statistically credible, following the hypothesis testing framework of Hamilton (1989) for regime-switching models.

---

## VII. Momentum Exclusion Filter

### 7.1 Motivation

The exclusion filter addresses a specific failure mode of momentum-driven strategies: allocation to stocks whose recent return is driven by crowding and hype rather than fundamental alpha. Hong, Lim and Stein (2000) show that momentum is stronger among stocks with lower analyst coverage and slower information diffusion. Da, Engelberg and Gao (2011) find that stocks with high retail investor attention exhibit strong short-term price pressure followed by reversal — precisely the pattern the exclusion filter is designed to avoid. Jegadeesh and Titman (2001) document that momentum profits are partially reversed over the subsequent 3-5 years, with the reversal more pronounced for the winner portfolio.

### 7.2 Joint Condition Design

The joint condition — extreme recent return AND above idiosyncratic equilibrium — combines two theoretically distinct risk signals. The momentum overextension gate captures the crowding and herding dynamic described by Shleifer and Vishny (1997) in the context of limits to arbitrage. The OU above-equilibrium gate ensures that only stocks that have genuinely overshot their fundamental level are penalised, consistent with the mean-reversion framework of Avellaneda and Lee (2010).

### 7.3 Soft Penalty Implementation

Rather than applying a hard exclusion, the filter operates as a **continuous momentum signal penalty**. For each stock $i$ with joint score $s_i > 0$, the momentum factor signals entering the composite alpha are divided by $\exp(k \cdot s_i)$:

$$f^{\text{adj}}_{i,\text{mom}} = \frac{f_{i,\text{mom}}}{\exp(k \cdot s_i)}$$

This exponential attenuation has several desirable properties. At $s_i = 0$ the penalty is exactly unity. As $s_i$ increases, the momentum contribution fades continuously and proportionally. Because the penalty operates at the composite signal level, it propagates fully into the MVO optimiser: a stock with attenuated momentum not only receives a lower alpha estimate but also affects the portfolio construction through reduced covariance with other momentum names.

---

## VIII. Strategy Architecture

Ten portfolio strategies represent a progressive addition of sophistication, allowing direct attribution of each component's contribution to risk-adjusted performance:

| # | Strategy | Description |
|---|----------|-------------|
| 1 | Baseline | Quality factor only, equal-weighted top-N |
| 2 | Pure Alpha | Full composite signal, concentrated weights |
| 3 | MVO | Composite signal + covariance optimisation |
| 4 | Hybrid | Simple average of Alpha and MVO weight vectors |
| 5 | Smart Hybrid | Regime-switching between Alpha / Hybrid / MVO |
| 6 | Dynamic | Signal-triggered rebalancing with regime-specific TO thresholds |
| 7 | Dynamic + Hedge | Strategy 6 + macro hedge overlay |
| 8 | DD Policy | Strategy 7 + multi-level drawdown de-grossing |
| 9 | Excl | Strategy 8 + momentum soft penalty filter |
| 10 | MVO+Hedge | Pure MVO base + dynamic rebalancing + hedge overlay (no regime switching) |

Strategy 10 (MVO+Hedge) was added to isolate the contribution of the hedge overlay and dynamic rebalancing when applied to a pure covariance-optimised base portfolio, without the regime-switching between alpha and MVO modes that characterises Strategies 6-9. The hypothesis is that MVO's diversification benefit, combined with the hedge overlay's asymmetric downside protection, may produce a superior Sharpe-adjusted profile to the regime-switching approaches.

---

## IX. Key Theoretical Properties

**Orthogonality of factor premia.** The Gram-Schmidt sequential stripping ensures that each factor's estimated premium is orthogonal to all prior factors in the WLS inner product space. This means that premium estimates are free of collinearity bias, addressing the multiple-testing and data-mining concerns raised by Harvey, Liu and Zhu (2016) and McLean and Pontiff (2016).

**Within-sector factor measurement.** The placement of sectors at Step 2 ensures that all subsequent factor premia — quality, size, value, momentum — are estimated within-sector by construction. This eliminates the inconsistency between within-sector score construction and across-sector regression estimation that characterised the prior factor ordering, and aligns the model with standard commercial factor model practice.

**Genuinely idiosyncratic momentum.** The placement of idiosyncratic momentum at Step 8, after all structural and fundamental factors have been stripped, ensures that the momentum signal captures persistence in truly stock-specific return dynamics. This is the cleanest possible implementation of the residual momentum framework of Blitz, Huij and Martens (2011).

**Shrinkage and estimation error.** The ensemble covariance approach addresses the "error maximisation" problem of Michaud (1989). Each estimator shrinks the sample covariance in a different direction, and their ensemble average tends to cancel direction-specific biases — a phenomenon studied formally in the model combination literature (Timmermann, 2006).

**Point-in-time construction.** Every component of the framework is constructed using only information available at the time of decision. This is essential for avoiding the backtest bias documented by Harvey and Liu (2015) and the overfitting concerns raised by Bailey, Borwein, López de Prado and Zhu (2014).

**Regime adaptation.** Multiple layers respond to the market regime: factor weights vary across rate regimes, portfolio construction shifts between alpha and MVO as drawdown develops, the hedge signal adapts monthly, and the exclusion filter conditions on the current cross-sectional distribution. This multi-layer adaptation reduces fragility to any single regime assumption, consistent with the robust portfolio construction philosophy of Ben-Tal, El Ghaoui and Nemirovski (2009).

**Asymmetric downside management.** The hedge overlay and drawdown policy together provide the asymmetric return profile that Grossman and Zhou (1993) show is optimal under drawdown constraints: full participation in upside, reduced exposure in tail events.

---

## X. Future Work — Regime Change Detection: CUSUM and Hidden Markov Models

### 10.1 Motivation

The current drawdown policy is triggered by realised NAV drawdown breaching pre-specified thresholds. This is a clean, interpretable rule, but it is inherently reactive: the portfolio absorbs the first 7.5% of drawdown before the first de-grossing level fires. Distribution-based statistical methods — CUSUM and Hidden Markov Models — offer the prospect of detecting regime shifts earlier, before they fully manifest in NAV, by monitoring changes in the statistical properties of portfolio returns (mean, variance) rather than their cumulative level.

### 10.2 CUSUM (Cumulative Sum Control Chart)

The CUSUM procedure (Page, 1954) was originally developed for industrial quality control and has been widely applied to financial time series for structural break detection (Chu, Stinchcombe and White, 1996; Andreou and Ghysels, 2002). The upper CUSUM statistic for detecting a downward mean shift is:

$$S_t = \max(0, S_{t-1} + (r_t - \mu_0 - k))$$

where $r_t$ is the observed portfolio return, $\mu_0$ is the estimated in-control mean, and $k = \sigma/2$ is the allowance parameter tuned to detect a 1$\sigma$ downward shift. A signal fires when $S_t > h$, where $h$ is a control limit calibrated to achieve a target in-control Average Run Length (ARL). The two-sided CUSUM tracks both downward mean shifts (bear signal) and upward shifts (recovery signal) simultaneously.

For the present application, a **variance CUSUM** is equally important:

$$V_t = \max(0, V_{t-1} + (|r_t - \mu_0|^2 - \sigma_0^2 - k_v))$$

which signals a volatility regime shift — often the earliest measurable indicator of a risk-off transition. Inclan and Tiao (1994) developed the ICSS (Iterated Cumulative Sums of Squares) algorithm specifically for variance break detection in financial return series, and this is the recommended starting point.

CUSUM properties particularly suited to this application: computationally trivial (single recursion per day); Average Run Length well-characterised, allowing explicit calibration of false alarm rate; naturally resets after a signal, allowing monitoring to resume immediately after a regime transition; no distributional assumptions beyond a reference mean and variance.

### 10.3 Hidden Markov Model

The Hidden Markov Model framework (Baum and Petrie, 1966; Hamilton, 1989) models the observed return series as generated by a latent discrete state variable $z_t \in \{1, 2\}$ with its own emission distribution:

$$r_t \mid z_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$$

The state evolves according to a Markov transition matrix $P$ with parameters $p_{11} = P(\text{stay bull})$ and $p_{22} = P(\text{stay bear})$. High diagonal values correspond to persistent regimes — empirically typical of equity markets (Hamilton, 1989; Ang and Bekaert, 2002). Parameters are estimated once by the EM algorithm (Baum-Welch), then applied online via the Hamilton filter recursion, which outputs at each date the **filtered probability** $P(z_t = \text{bear} \mid r_1, \ldots, r_t)$ using only past-and-present information.

The HMM is particularly well-suited to this problem because: (i) it jointly models mean and variance regime shifts in a single coherent framework; (ii) it outputs a probability rather than a binary signal, enabling soft conditioning of de-grossing triggers; (iii) the persistence in $p_{22}$ naturally penalises short-lived excursions below the drawdown threshold, reducing whipsaw; (iv) the filtered probability respects the minimum holding period naturally — it cannot jump from 0% to 100% in a single period. Nystrup et al. (2015) demonstrate in a closely related setting that HMM-driven tactical asset allocation improves risk-adjusted returns out of sample.

### 10.4 Proposed Integration Architecture

The two methods are complementary rather than redundant:

- **CUSUM** serves as an **early-warning tripwire**: fast, sensitive, low-parameter, detects changes before the HMM probability accumulates sufficient evidence. The CUSUM signal triggers a watchlist state but does not itself modify portfolio construction.
- **HMM** serves as the **regime classifier**: richer, probabilistic, slower to confirm. When $P(\text{bear} \mid \text{HMM}) > \theta_{\text{enter}}$ (proposed threshold: 0.70), the portfolio enters de-grossed mode.

The proposed conditioning logic:

$$\text{De-gross trigger: } \quad P(\text{bear} \mid \text{HMM}) > \theta_{\text{enter}} \text{ AND } \Delta\text{NAV} < -\delta_{\text{floor}}$$

$$\text{Re-entry trigger: } \quad P(\text{bear} \mid \text{HMM}) < \theta_{\text{exit}} \text{ AND confirmation for } N \text{ consecutive days}$$

This replaces the current NAV-threshold-only trigger with a joint condition that fires earlier (HMM detects distributional shifts before NAV is impaired) and recovers more cleanly (probabilistic re-entry avoids premature re-grossing in bear market rallies).

The regime probability also enables **continuous de-grossing**: rather than discrete steps (100% → 75% → 50% → 25%), the gross exposure can track $1 - P(\text{bear})$ continuously, smoothing the portfolio response curve and reducing transaction costs from discrete threshold crossings.

### 10.5 Implementation Notes

The HMM should be estimated on **portfolio returns** rather than individual stock returns or factor premia, since the objective is drawdown management at the portfolio level. The CUSUM reference parameters ($\mu_0$, $\sigma_0^2$) should be calibrated on a rolling in-sample window (recommended: 252 trading days) to account for secular shifts in portfolio volatility as AUM grows and the strategy matures.

The minimum holding period remains operative even with CUSUM/HMM active — the probabilistic framework reduces false positives, but the 10-calendar-day gate remains a last-resort guard against whipsaw in rapid oscillation scenarios. The expected benefit of the regime-detection layer is most visible in slow-developing bear markets (2022-style rate shock) where the HMM probability builds over weeks, enabling gradual de-grossing well before the -7.5% NAV threshold would fire.

---

## References

Andreou, E., & Ghysels, E. (2002). Detecting multiple breaks in financial market volatility dynamics. *Journal of Applied Econometrics*, 17(5), 579–600.

Ang, A., Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.

Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*, 61(1), 259–299.

Asness, C. S., Frazzini, A., Israel, R., & Moskowitz, T. J. (2018). Size matters, if you control your junk. *Journal of Financial Economics*, 129(3), 479–509.

Asness, C. S., Frazzini, A., Israel, R., Moskowitz, T. J., & Pedersen, L. H. (2015). Fact, fiction, and value investing. *Journal of Portfolio Management*, 42(1), 34–52.

Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. *Review of Accounting Studies*, 24(1), 34–112.

Asquith, P., Pathak, P. A., & Ritter, J. R. (2005). Short interest, institutional ownership, and stock returns. *Journal of Financial Economics*, 78(2), 243–276.

Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–70.

Banz, R. W. (1981). The relationship between return and market value of common stocks. *Journal of Financial Economics*, 9(1), 3–18.

Basu, S. (1977). Investment performance of common stocks in relation to their price-earnings ratios: A test of the efficient market hypothesis. *Journal of Finance*, 32(3), 663–682.

Baum, L. E., & Petrie, T. (1966). Statistical inference for probabilistic functions of finite state Markov chains. *Annals of Mathematical Statistics*, 37(6), 1554–1563.

Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009). *Robust Optimization*. Princeton University Press.

Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*, 18(3), 506–521.

Blume, M. E. (1975). Betas and their regression tendencies. *Journal of Finance*, 30(3), 785–795.

Boehmer, E., Jones, C. M., & Zhang, X. (2008). Which shorts are informed? *Journal of Finance*, 63(2), 491–527.

Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.

Chen, N. F., Roll, R., & Ross, S. A. (1986). Economic forces and the stock market. *Journal of Business*, 59(3), 383–403.

Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing*, 58(10), 5016–5029.

Chu, C. S. J., Stinchcombe, M., & White, H. (1996). Monitoring structural change. *Econometrica*, 64(5), 1045–1065.

Cochrane, J. H. (2011). Presidential address: Discount rates. *Journal of Finance*, 66(4), 1047–1108.

Connor, G., & Korajczyk, R. A. (1988). Risk and return in an equilibrium APT: Application of a new test methodology. *Journal of Financial Economics*, 21(2), 255–289.

Da, Z., Engelberg, J., & Gao, P. (2011). In search of attention. *Journal of Finance*, 66(5), 1461–1499.

Desai, H., Ramesh, K., Thiagarajan, S. R., & Balachandran, B. V. (2002). An investigation of the informational role of short interest in the Nasdaq market. *Journal of Finance*, 57(5), 2263–2287.

Diether, K. B., Malloy, C. J., & Scherbina, A. (2002). Differences of opinion and the cross section of stock returns. *Journal of Finance*, 57(5), 2113–2141.

Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427–465.

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.

Fama, E. F., & French, K. R. (1997). Industry costs of equity. *Journal of Financial Economics*, 43(2), 153–193.

Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.

Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607–636.

Fan, J., Fan, Y., & Lv, J. (2008). High dimensional covariance matrix estimation using a factor model. *Journal of Econometrics*, 147(1), 186–197.

Ferson, W. E., & Harvey, C. R. (1993). The risk and predictability of international equity returns. *Review of Financial Studies*, 6(3), 527–566.

Ferson, W. E., & Harvey, C. R. (1999). Conditioning variables and the cross section of stock returns. *Journal of Finance*, 54(4), 1325–1360.

Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67–78.

Genre, V., Kenny, G., Meyler, A., & Timmermann, A. (2013). Combining expert forecasts: Can anything beat the simple average? *International Journal of Forecasting*, 29(1), 108–121.

Greenberg, D., McNeil, M., & Krishnan, H. (2016). The two percent dilution. *Financial Analysts Journal*, 72(3), 14–22.

Grinold, R. C. (1989). The fundamental law of active management. *Journal of Portfolio Management*, 15(3), 30–37.

Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* (2nd ed.). McGraw-Hill.

Grossman, S. J., & Zhou, Z. (1993). Optimal investment strategies for controlling drawdowns. *Mathematical Finance*, 3(3), 241–276.

Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*, 31(11), 3503–3544.

Gutierrez, R. C., & Kelley, E. K. (2008). The long-lasting momentum in weekly returns. *Journal of Finance*, 63(1), 415–447.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13–28.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.

Hong, H., Lim, T., & Stein, J. C. (2000). Bad news travels slowly: Size, analyst coverage, and the profitability of momentum strategies. *Journal of Finance*, 55(1), 265–295.

Huber, P. J. (1981). *Robust Statistics*. Wiley.

Ilmanen, A. (2003). Stock-bond correlations. *Journal of Fixed Income*, 13(2), 55–66.

Ilmanen, A. (2011). *Expected Returns: An Investor's Guide to Harvesting Market Rewards*. Wiley Finance.

Inclan, C., & Tiao, G. C. (1994). Use of cumulative sums of squares for retrospective detection of changes of variance. *Journal of the American Statistical Association*, 89(427), 913–923.

Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.

Jegadeesh, N., & Titman, S. (2001). Profitability of momentum strategies: An evaluation of alternative explanations. *Journal of Finance*, 56(2), 699–720.

Johnson, L. L. (1960). The theory of hedging and speculation in commodity futures. *Review of Economic Studies*, 27(3), 139–151.

Johnstone, I. M., & Lu, A. Y. (2009). On consistency and sparsity for principal components analysis in high dimensions. *Journal of the American Statistical Association*, 104(486), 682–693.

Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.

Lakonishok, J., Shleifer, A., & Vishny, R. W. (1994). Contrarian investment, extrapolation, and risk. *Journal of Finance*, 49(5), 1541–1578.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.

Ledoit, O., & Wolf, M. (2012). Nonlinear shrinkage estimation of large-dimensional covariance matrices. *Annals of Statistics*, 40(2), 1024–1060.

Lee, C. M. C., & Swaminathan, B. (2000). Price momentum and trading volume. *Journal of Finance*, 55(5), 2017–2069.

Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.

Leland, H. E. (2000). Optimal portfolio management with transactions costs and capital gains taxes. Working Paper, Haas School of Business, UC Berkeley.

Lintner, J. (1965). The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets. *Review of Economics and Statistics*, 47(1), 13–37.

Longin, F., & Solnik, B. (2001). Extreme correlation of international equity markets. *Journal of Finance*, 56(2), 649–676.

Lynch, A. W., & Balduzzi, P. (2000). Predictability and transaction costs: The impact on rebalancing rules and behavior. *Journal of Finance*, 55(5), 2285–2309.

Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.

Markowitz, H. (1959). *Portfolio Selection: Efficient Diversification of Investments*. Wiley.

McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5–32.

Merton, R. C. (1973). An intertemporal capital asset pricing model. *Econometrica*, 41(5), 867–887.

Michaud, R. O. (1989). The Markowitz optimization enigma: Is 'optimized' optimal? *Financial Analysts Journal*, 45(1), 31–42.

Moskowitz, T. J., & Grinblatt, M. (1999). Do industries explain momentum? *Journal of Finance*, 54(4), 1249–1290.

Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1–28.

Nystrup, P., Hansen, B. W., Madsen, H., & Lindström, E. (2015). Regime-based versus static asset allocation: Letting the data speak. *Journal of Portfolio Management*, 42(1), 103–109.

Nystrup, P., Madsen, H., & Lindström, E. (2017). Long memory of financial time series and hidden Markov models with time-varying parameters. *Journal of Forecasting*, 36(8), 989–1002.

Ornstein, L. S., & Uhlenbeck, G. E. (1930). On the theory of Brownian motion. *Physical Review*, 36(5), 823–841.

Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1-2), 100–115.

Petkova, R. (2006). Do the Fama-French factors proxy for innovations in predictive variables? *Journal of Finance*, 61(2), 581–612.

Platen, E., & Heath, D. (2006). *A Benchmark Approach to Quantitative Finance*. Springer.

Riskmetrics Group. (1996). *RiskMetrics Technical Document* (4th ed.). J.P. Morgan / Reuters.

Ross, S. A. (1976). The arbitrage theory of capital asset pricing. *Journal of Economic Theory*, 13(3), 341–360.

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *Journal of Finance*, 19(3), 425–442.

Shleifer, A., & Vishny, R. W. (1997). The limits of arbitrage. *Journal of Finance*, 52(1), 35–55.

Stoll, H. R., & Whaley, R. E. (1993). *Futures and Options: Theory and Applications*. South-Western Publishing.

Thorp, E. O. (2006). The Kelly criterion in blackjack, sports betting, and the stock market. In *Handbook of Asset and Liability Management* (Vol. 1). Elsevier.

Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J. Granger, & A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 1, pp. 135–196). Elsevier.

Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian motion. *Physical Review*, 36(5), 823.
