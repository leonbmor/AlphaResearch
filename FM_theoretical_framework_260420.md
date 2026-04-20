# Systematic Equity Strategy — Theoretical Framework
*April 2026*

---

## Overview

This document describes the theoretical foundations of a systematic US large-cap equity strategy spanning approximately 680 stocks. The framework integrates four distinct but interrelated modules: a cross-sectional factor model that constructs a multi-dimensional alpha signal from fundamental and technical data; a mean-variance portfolio optimisation engine that translates that signal into optimal weights; a dynamic rebalancing and risk management overlay; and a momentum-aware exclusion filter that guards against allocating to stocks exhibiting crowded, potentially reversal-prone price dynamics. Each module is discussed in terms of its theoretical motivation and the statistical machinery employed.

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

The rationale is threefold. First, it prevents multicollinearity from contaminating factor premia estimates — each estimated coefficient measures the marginal contribution of a factor after controlling exactly for all prior factors, addressing the concern raised by Harvey, Liu and Zhu (2016) about spurious factor discovery. Second, it produces clean, non-overlapping residuals at each step. Third, the ordering reflects an economic prior about which systematic sources are most pervasive: market risk and size are stripped first, macro and sector effects second, and idiosyncratic characteristics last.

### 1.3 Factor Hierarchy

The model strips systematic variation in eleven sequential steps. A critical design choice governs the ordering: **alpha factors** (Quality, Momentum, Value) enter the Gram-Schmidt chain before structural and risk factors (Size, SI, GK Volatility), and macro and sector effects are deferred to the end. This ordering reflects the priority of recovering clean alpha signals — if size or macro were stripped first, they would absorb variance that rightly belongs to the quality and momentum premia, attenuating the signals on which the composite alpha is constructed. The academic motivation for this inversion is developed in Daniel and Titman (1997), who show that alpha characteristics carry information beyond what systematic covariances can explain, and Novy-Marx (2013), who demonstrates that gross profitability interacts with but is distinct from other structural factors.

**Step 1 — Baseline variance benchmark.** Establishes the total unexplained variance at the start of the procedure, serving as a normalisation reference for the variance reduction decomposition.

**Step 2 — Market beta.** EWMA-estimated covariance of each stock's returns with the market index, divided by the EWMA variance of the index. The CAPM of Sharpe (1964) and Lintner (1965) motivates this as the single most pervasive systematic source. Exponential weighting (Riskmetrics, 1996) captures time-variation in betas, consistent with the evidence of beta instability documented by Blume (1975) and Ferson and Harvey (1999). Stripped first because market exposure explains the largest fraction of cross-sectional return variation and would otherwise inflate all subsequent factor loadings.

**Step 3 — Quality.** A composite quality signal blended from two sub-composites calibrated to different rate regimes, orthogonalised to beta. The quality premium is well established: Asness, Frazzini and Pedersen (2019) formalise the "Quality Minus Junk" (QMJ) factor, showing that high-quality stocks — defined by profitability, growth, safety and payout — earn a significant premium. Novy-Marx (2013) documents that gross profitability is the most powerful quality predictor of future returns. Quality enters at Step 3 — immediately after market beta — because it is the most structurally persistent alpha signal and because its early placement ensures that subsequent factors are orthogonal to quality rather than contaminated by it. The rate-regime conditioning reflects the finding of Asness et al. (2019) that quality premia are time-varying and partially explained by monetary conditions.

**Step 4 — Idiosyncratic momentum.** Computed on the **quality residual** — the return series after stripping market beta and quality — rather than on the raw or volatility-stripped residual as in earlier implementations. This is a deliberate departure: by computing momentum on the quality residual, the momentum signal measures persistent outperformance above both market and quality expectations, isolating the pure behavioural component of momentum uncontaminated by the quality premium. The momentum premium is among the most robust findings in empirical asset pricing, documented by Jegadeesh and Titman (1993, 2001). Our residual momentum construction follows Gutierrez and Kelley (2008) and Blitz, Huij and Martens (2011), who show that factor-adjusted momentum is more persistent and less prone to reversal than raw price momentum. The 1-month skip period follows Jegadeesh (1990) and Lehmann (1990) to avoid short-term reversal contamination. Volume-scaling follows Lee and Swaminathan (2000).

**Step 5 — Size.** Cross-sectional log market capitalisation, updated daily, orthogonalised to beta, quality and momentum. Banz (1981) documented the size effect; Fama and French (1993) formalised it as SMB. Placing size after alpha factors is important: a naive early placement would cause size to absorb variation attributable to quality and momentum (since large caps tend to be higher quality), inflating the apparent size premium at the expense of the alpha signals. Post-orthogonalisation, the size loading here captures the pure size premium net of quality and momentum, consistent with Asness, Frazzini, Israel and Moskowitz (2018) who show the size premium is robust only after controlling for quality.

**Step 6 — Value.** IC-weighted composite of seven valuation ratios (price-to-sales, price-to-earnings on estimated and operating bases, price-to-gross-profit), orthogonalised to beta, quality, momentum and size. The value premium is among the oldest documented anomalies: Basu (1977), Fama and French (1992), Lakonishok, Shleifer and Vishny (1994). Novy-Marx (2013) demonstrates that gross-profit-to-price subsumes book-to-market. Placing value after momentum is critical: raw valuation metrics embed stale price information, and without first stripping momentum, the value factor would absorb some of the momentum signal in reverse (cheap stocks tend to be recent losers). Orthogonalising value to momentum eliminates this price-contamination, recovering a purer fundamental mispricing signal.

**Step 7 — Short Interest composite.** Cross-sectional short interest metrics, orthogonalised to all prior factors. The information content of short interest is documented by Asquith, Pathak and Ritter (2005), Diether, Malloy and Scherbina (2002), and Desai, Ramesh, Thiagarajan and Balachandran (2002), all showing that high short interest predicts negative future returns. Boehmer, Jones and Zhang (2008) confirm this using exchange-reported data. At this position in the chain, the SI loading captures incremental short-seller information above and beyond market, quality, momentum, size and value.

**Step 8 — GK Volatility.** The Garman-Klass (1980) estimator of daily variance using open, high, low and close prices:

$$\sigma^2_{GK} = 0.5(\ln H/L)^2 - (2\ln 2 - 1)(\ln C/O)^2$$

Garman and Klass (1980) show this estimator is approximately 7.4 times more efficient than close-to-close variance. Ang, Hodrick, Xing and Zhang (2006) document both a positive variance risk premium and a puzzling negative idiosyncratic volatility premium. At Step 8, after all alpha factors have been stripped, the GK loading captures the residual variance risk compensation that is genuinely orthogonal to quality, momentum, size, value and short interest.

**Step 9 — Macro factors.** Seven macroeconomic sensitivities: yield curve level, slope, real yields, inflation breakevens, rate volatility (MOVE index), oil, and gold. Motivated by the Intertemporal CAPM of Merton (1973) and the Arbitrage Pricing Theory of Ross (1976). Chen, Roll and Ross (1986) and Petkova (2006) provide empirical grounding. Macro factors are deferred to Step 9 — after all alpha factors — because macroeconomic variables are correlated with quality, size and value characteristics. An early macro step would absorb alpha variance under the guise of macro sensitivity. At Step 9, these factors are estimated via **joint Ridge regression** (Hoerl and Kennard, 1970) against the GK volatility residual, with the Ridge penalty selected by k-fold cross-validation at each date. Notably, macro characteristics are **not** Gram-Schmidt orthogonalised; instead, raw EWMA betas to the vol residuals are used, consistent with the interpretation of macro loadings as time-varying conditional betas rather than unconditional alpha characteristics.

**Step 10 — Sector effects.** A full set of sector dummies encoded using **sum-to-zero deviation coding**: each dummy equals +1 for the stock's own sector and $-1/(K-1)$ for all other sectors, where *K* is the number of sectors. This ensures all sectors are simultaneously included without a reference category. Moskowitz and Grinblatt (1999) document persistent industry return premia. Like macro, sector dummies are estimated via Ridge regression rather than Gram-Schmidt orthogonalisation — sectors capture time-varying industry-level premia rather than alpha characteristics, and their placement after macro ensures the two structural overlays do not absorb each other's variance.

**Step 11 — Ornstein-Uhlenbeck mean reversion.** An AR(1) model is fitted to the compounded cumulative residual price index for each stock, treating the idiosyncratic price path as a discrete-time Ornstein-Uhlenbeck process. The continuous-time OU process (Uhlenbeck and Ornstein, 1930) describes a stochastic variable that reverts to a long-run mean at a rate proportional to its current displacement:

$$dX_t = \kappa(\mu - X_t)\,dt + \sigma\,dW_t$$

where $X_t$ is the log-cumulative residual return, $\mu$ is the equilibrium level, $\kappa > 0$ is the speed of mean reversion, $\sigma$ is the diffusion coefficient, and $W_t$ is a standard Brownian motion. The discrete-time equivalent, estimated by OLS, is:

$$\Delta X_t = a + b X_{t-1} + \eta_t$$

where $a = \kappa\mu\,\Delta t$, $b = -\kappa\,\Delta t$, and $b \in (-1, 0)$ implies mean reversion. The mapping between the discrete and continuous parameters gives the equilibrium level $\mu = -a/b$ and the characteristic half-life $\tau = \ln(2)/\kappa = -\ln(2)/\ln(1+b)$. The **OU score** for stock $i$ is defined as the standardised deviation of the current cumulative residual from its estimated equilibrium:

$$s_i = \frac{X_{i,t} - \mu_i}{\hat{\sigma}_i / \sqrt{2\kappa_i}}$$

where the denominator is the stationary standard deviation of the OU process. A negative score ($s_i < 0$) indicates the stock is trading below its estimated fundamental equilibrium — a buy signal; a positive score ($s_i > 0$) indicates it has overshot — a sell signal. This framework is fully developed in Avellaneda and Lee (2010), "Statistical Arbitrage in the U.S. Equities Market" (*Quantitative Finance*, 10(7), 761–782), who derive the closed-form estimation procedure, establish its statistical properties, and demonstrate its application to residual-based mean-reversion strategies across the equity universe. The OU residuals here are fitted on the sector residual — the cleanest available signal after all ten prior systematic sources have been stripped — making the equilibrium estimate as model-pure as possible. The OU score serves as the final alpha signal, blended with a short-term reversal component (Jegadeesh, 1990; Lehmann, 1990) with weight decaying with estimated half-life, so stocks whose process reverts slowly receive less OU alpha weight.

---

## II. Composite Alpha Signal and Factor Weight Estimation

### 2.1 Information Coefficient Framework

The outputs of the factor model are used to estimate the Information Coefficient (IC) of each factor at each date. The IC is the Spearman rank correlation between the factor score (measured at date *t*) and the subsequent stock return over forward horizon *h*:

$$IC_{k,t,h} = \text{Spearman}\left(f_{k,i,t},\ r_{i,t \to t+h}\right)$$

The IC framework is the standard evaluation metric in quantitative equity research (Grinold and Kahn, 2000). Spearman rank correlation is used rather than Pearson to reduce sensitivity to outliers and tail events. The Fundamental Law of Active Management (Grinold, 1989) establishes that the information ratio of an active strategy is approximately $IC \times \sqrt{BR}$, where *BR* is the number of independent bets — motivating the use of IC as the primary weight-setting criterion.

### 2.2 Regime-Conditional Factor Weights

Factor weights vary across three macroeconomic regimes defined by the interest rate environment. Within each regime, factor weights are estimated as the IC-weighted average of each factor's contribution, with a regularisation floor. The regime-conditional approach follows the evidence of Asness et al. (2015) and Ilmanen (2011) that factor premia are time-varying and partially predictable from macro state variables. The resulting point-in-time composite alpha signal is the primary input to portfolio construction.

---

## III. Portfolio Construction — Mean-Variance Optimisation

### 3.1 The MVO Problem

The portfolio construction follows the mean-variance framework of Markowitz (1952, 1959). Given composite alpha signal $\hat{\alpha}_i$, portfolio weights $w$ are chosen to solve:

$$\max_w \quad w^\top \hat{\alpha} - \frac{\lambda}{2} w^\top \Sigma w$$

subject to full investment, position bounds, and sector concentration limits. The risk aversion parameter $\lambda$ controls the signal-exploitation/diversification trade-off. The alpha signal is pre-processed by winsorising extreme observations (Huber, 1981) and calibrated by an IC parameter that sets the economic magnitude of the signal, consistent with the signal-to-noise ratio interpretation of Grinold and Kahn (2000).

Michaud (1989) demonstrated that naive MVO is highly sensitive to estimation error in both expected returns and the covariance matrix — a problem he termed "error maximisation." The ensemble covariance approach and robust signal construction described below directly address this concern.

### 3.2 Covariance Matrix Estimation — Ensemble Approach

With approximately 680 stocks, the sample covariance matrix has far more parameters than can be reliably estimated from a few hundred daily returns. The model employs an **ensemble of four covariance estimators**:

**Estimator 1 — EWMA Empirical Covariance.** The sample covariance matrix computed on a 63-day trailing window with exponential weighting. The EWMA approach was popularised by Riskmetrics (1996) and captures short-term dynamics in the correlation structure, which are known to spike during stress episodes (Longin and Solnik, 2001).

**Estimator 2 — Ledoit-Wolf Shrinkage.** The Oracle Approximating Shrinkage (OAS) estimator of Chen, Wiesel, Eldar and Hero (2010), which shrinks the sample covariance towards a scaled identity matrix:

$$\hat{\Sigma}_{LW} = (1 - \rho) \cdot S + \rho \cdot \mu \cdot I$$

where $S$ is the sample covariance, $\mu = \text{tr}(S)/N$ is the mean eigenvalue, and $\rho$ is chosen analytically to minimise expected Frobenius norm estimation error. This builds on the foundational work of Ledoit and Wolf (2004), who showed analytically that shrinking the sample covariance towards a structured target dramatically reduces out-of-sample MVO portfolio variance. The equally-weighted portfolio target was later generalised to other targets by Ledoit and Wolf (2012), but the scaled identity remains the most commonly used and theoretically motivated choice in the absence of strong prior beliefs about the covariance structure.

**Estimator 3 — Factor Structure (XFX').** A structured estimator built from the factor model's output:

$$\hat{\Sigma}_{Factor} = X F X^\top + D$$

where $X$ is the factor loading matrix, $F$ the factor covariance matrix, and $D$ a diagonal matrix of idiosyncratic variances. This approach follows directly from the APT of Ross (1976) and is the basis of the BARRA/Axioma commercial risk models (see Grinold and Kahn, 2000, Chapter 3). The factor structure estimator is parsimonious — it has $K(K+1)/2 + N$ free parameters rather than $N(N+1)/2$ — and performs well when the factor model genuinely explains most cross-sectional covariation. Connor and Korajczyk (1988) provide theoretical justification for factor-based covariance in the context of the APT.

**Estimator 4 — PCA Reconstruction.** PCA decomposition retaining the leading components that jointly explain 65% of total variance:

$$\hat{\Sigma}_{PCA} = V_k \Lambda_k V_k^\top + \bar{\sigma}^2 I$$

where $V_k$ are the top-k eigenvectors, $\Lambda_k$ the corresponding eigenvalues, and $\bar{\sigma}^2$ the mean residual eigenvalue. PCA regularisation of the covariance matrix is studied by Johnstone and Lu (2009) and Fan, Fan and Lv (2008), who show that standard eigenvalue estimators are biased in high-dimensional settings and that hard thresholding of eigenvalues substantially reduces this bias.

**Ensemble combination.** The four estimates are combined using equal weights. Equal-weighting of model combinations is studied by Timmermann (2006) and Genre, Kenny, Meyler and Timmermann (2013), who show that simple averages are competitive with optimised combinations due to estimation error in the combination weights. The ensemble average inherits fast-moving properties from EWMA, bias-reduction from Ledoit-Wolf, structural parsimony from the factor model, and eigenvalue regularisation from PCA.

---

## IV. Dynamic Rebalancing Framework

### 4.1 Regime-Dependent Portfolio Selection

Three portfolio constructions are maintained in parallel — pure alpha, covariance-optimised, and hybrid — with the deployed construction selected based on current drawdown regime. This regime-switching approach is motivated by the work of Guidolin and Timmermann (2007) on regime-switching in optimal portfolio choice, and by the empirical evidence in Asness, Frazzini and Pedersen (2019) that the relative performance of quality and momentum factors varies systematically with credit and macro conditions.

### 4.2 Turnover-Based Rebalancing

The portfolio is rebalanced when the implied turnover against the current drifted holdings exceeds a regime-specific threshold. The turnover is computed against the **drifted portfolio** — the actual holdings as evolved by price moves — not the theoretical static weights. This correctly measures required trades and prevents over-counting turnover in trending markets, following the approach of Platen and Heath (2006) and the practical guidance of Grinold and Kahn (2000) on portfolio rebalancing costs.

The interaction between transaction costs and optimal rebalancing frequency is studied by Leland (2000) and Lynch and Balduzzi (2000), who show that no-trade regions are optimal when costs are proportional and that the optimal region widens with higher costs — consistent with the use of higher TO thresholds in defensive regimes where expected alpha is lower.

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

The exclusion filter addresses a specific failure mode of momentum-driven strategies: allocation to stocks whose recent return is driven by crowding and hype rather than fundamental alpha. The academic literature distinguishes between fundamental momentum (return driven by earnings revisions and cash flow realisations) and non-fundamental momentum (return driven by investor sentiment and herding). Hong, Lim and Stein (2000) show that momentum is stronger among stocks with lower analyst coverage and slower information diffusion, consistent with a gradual diffusion mechanism. Conversely, Da, Engelberg and Gao (2011) find that stocks with high retail investor attention (measured by Google search volume) exhibit strong short-term price pressure followed by reversal — precisely the pattern the exclusion filter is designed to avoid.

Jegadeesh and Titman (2001) document that momentum profits are partially reversed over the subsequent 3-5 years, with the reversal more pronounced for the winner portfolio — consistent with the hypothesis that winner stocks become overvalued. The 1-3 month horizon of the exclusion filter's momentum overextension gate is designed to catch stocks approaching or at the peak of their momentum cycle before the reversal materialises.

### 7.2 Joint Condition Design

The joint condition — extreme recent return AND above idiosyncratic equilibrium — combines two theoretically distinct risk signals. The momentum overextension gate captures the crowding and herding dynamic described by Shleifer and Vishny (1997) in the context of limits to arbitrage: stocks with extreme recent appreciation are more likely to have become consensus long positions, reducing the universe of marginal buyers and increasing vulnerability to coordinated selling. The OU above-equilibrium gate ensures that only stocks that have genuinely overshot their fundamental level are penalised, consistent with the regime-dependent OU evidence discussed in Section I and the mean-reversion framework of Avellaneda and Lee (2010).

The joint score is defined as the product of two magnitudes: the degree of return overextension relative to its cross-sectional percentile threshold, and the stock's OU excess above equilibrium. This construction follows the spirit of composite signal design advocated by Grinold and Kahn (2000): two signals with complementary predictive content — one measuring behavioural crowding, the other structural overvaluation — produce a more robust criterion than either alone.

### 7.3 Soft Penalty Implementation

Rather than applying a hard exclusion that removes overextended stocks entirely from the investable set, the filter operates as a **continuous momentum signal penalty**. For each stock $i$ with joint score $s_i > 0$, the momentum factor signals (idiosyncratic momentum and 12-month price momentum) entering the composite alpha are divided by $\exp(k \cdot s_i)$, where $k$ is a calibrated scaling parameter:

$$f^{\text{adj}}_{i,\text{mom}} = \frac{f_{i,\text{mom}}}{\exp(k \cdot s_i)}$$

This exponential attenuation has several desirable properties. At $s_i = 0$ the penalty is exactly unity — stocks with no overextension signal are unaffected. As $s_i$ increases, the momentum contribution fades continuously and proportionally, with the rate of attenuation governed by $k$. High-conviction overextension cases (large $s_i$) approach zero momentum contribution, recovering behaviour similar to full exclusion; mild overextension cases receive a proportional discount that preserves residual momentum alpha while reducing crowding risk. Quality and value signals are deliberately left unpenalised — the hypothesis is that overextension is a momentum-specific pathology, not a signal about fundamental value.

Because the penalty operates at the composite signal level, it propagates fully into the MVO optimiser: a stock with attenuated momentum not only receives a lower alpha estimate but also affects the portfolio construction through reduced covariance with other momentum names. This is more theoretically coherent than a post-optimisation exclusion, which can create discontinuities in the weight surface.

---

## VIII. Strategy Architecture Summary

The nine portfolio strategies represent a progressive addition of sophistication, allowing direct attribution of each component's contribution to risk-adjusted performance:

| Strategy | Description |
|----------|-------------|
| 1. Baseline | Quality factor only, equal-weighted top-N |
| 2. Pure Alpha | Full composite signal, concentrated weights |
| 3. MVO | Composite signal + covariance optimisation |
| 4. Hybrid | Simple average of Alpha and MVO weight vectors |
| 5. Smart Hybrid | Regime-switching between Alpha / Hybrid / MVO |
| 6. Dynamic | Signal-triggered rebalancing with regime-specific TO thresholds |
| 7. Dynamic + Hedge | Strategy 6 + macro hedge overlay |
| 8. Dyn+Hedge+DD | Strategy 7 + multi-level drawdown de-grossing policy |
| 9. Dyn+Hedge+DD+Excl | Strategy 8 + momentum soft penalty filter |

---

## IX. Key Theoretical Properties

**Orthogonality of factor premia.** The Gram-Schmidt sequential stripping ensures that each factor's estimated premium is orthogonal to all prior factors in the WLS inner product space. This means that premium estimates are free of collinearity bias, addressing the multiple-testing and data-mining concerns raised by Harvey, Liu and Zhu (2016) and McLean and Pontiff (2016).

**Shrinkage and estimation error.** The ensemble covariance approach addresses the "error maximisation" problem of Michaud (1989). Each estimator shrinks the sample covariance in a different direction, and their ensemble average tends to cancel direction-specific biases — a phenomenon studied formally in the model combination literature (Timmermann, 2006).

**Point-in-time construction.** Every component of the framework is constructed using only information available at the time of decision. This is essential for avoiding the backtest bias documented by Harvey and Liu (2015) and the overfitting concerns raised by Bailey, Borwein, López de Prado and Zhu (2014).

**Regime adaptation.** Multiple layers respond to the market regime: factor weights vary across rate regimes, portfolio construction shifts between alpha and MVO as drawdown develops, the hedge signal adapts monthly, and the exclusion filter conditions on the current cross-sectional distribution. This multi-layer adaptation reduces fragility to any single regime assumption, consistent with the robust portfolio construction philosophy of Ben-Tal, El Ghaoui and Nemirovski (2009).

**Asymmetric downside management.** The hedge overlay and drawdown policy together provide the asymmetric return profile that Grossman and Zhou (1993) show is optimal under drawdown constraints: full participation in upside, reduced exposure in tail events.

---

## References

Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*, 61(1), 259–299.

Asness, C. S., Frazzini, A., Israel, R., & Moskowitz, T. J. (2018). Size matters, if you control your junk. *Journal of Financial Economics*, 129(3), 479–509.

Asness, C. S., Frazzini, A., Israel, R., Moskowitz, T. J., & Pedersen, L. H. (2015). Fact, fiction, and value investing. *Journal of Portfolio Management*, 42(1), 34–52.

Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. *Review of Accounting Studies*, 24(1), 34–112.

Asquith, P., Pathak, P. A., & Ritter, J. R. (2005). Short interest, institutional ownership, and stock returns. *Journal of Financial Economics*, 78(2), 243–276.

Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–70.

Banz, R. W. (1981). The relationship between return and market value of common stocks. *Journal of Financial Economics*, 9(1), 3–18.

Basu, S. (1977). Investment performance of common stocks in relation to their price-earnings ratios: A test of the efficient market hypothesis. *Journal of Finance*, 32(3), 663–682.

Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009). *Robust Optimization*. Princeton University Press.

Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*, 18(3), 506–521.

Blume, M. E. (1975). Betas and their regression tendencies. *Journal of Finance*, 30(3), 785–795.

Boehmer, E., Jones, C. M., & Zhang, X. (2008). Which shorts are informed? *Journal of Finance*, 63(2), 491–527.

Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.

Chen, N. F., Roll, R., & Ross, S. A. (1986). Economic forces and the stock market. *Journal of Business*, 59(3), 383–403.

Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing*, 58(10), 5016–5029.

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

Ornstein, L. S., & Uhlenbeck, G. E. (1930). On the theory of Brownian motion. *Physical Review*, 36(5), 823–841.

Petkova, R. (2006). Do the Fama-French factors proxy for innovations in predictive variables? *Journal of Finance*, 61(2), 581–612.

Platen, E., & Heath, D. (2006). *A Benchmark Approach to Quantitative Finance*. Springer.

Riskmetrics Group. (1996). *RiskMetrics Technical Document* (4th ed.). J.P. Morgan / Reuters.

Ross, S. A. (1976). The arbitrage theory of capital asset pricing. *Journal of Economic Theory*, 13(3), 341–360.

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *Journal of Finance*, 19(3), 425–442.

Shleifer, A., & Vishny, R. W. (1997). The limits of arbitrage. *Journal of Finance*, 52(1), 35–55.

Stoll, H. R., & Whaley, R. E. (1993). *Futures and Options: Theory and Applications*. South-Western Publishing.

Thorp, E. O. (2006). The Kelly criterion in blackjack, sports betting, and the stock market. In *Handbook of Asset and Liability Management* (Vol. 1). Elsevier.

Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J. Granger, & A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 1, pp. 135–196). Elsevier.
