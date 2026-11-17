# E1: membership inference on shared summary statistics (fixed before any run, 2026-11-16)

Task from the Nov 4 meeting (Prof. Xiaoxiao; slides "Nov4_FedGM Pipeline Update", slides 7
and 17): show how membership-inference leakage from the statistics a site shares changes with
the sample size n and the number of covariates d, with false-positive rates. Evaluation only:
the releases are exact (no noise), as the meeting asked ("no need to bother with protection").
The slide-17 pilot code is not in the repository, so E1 rebuilds it.

## Releases (what a site shares about a group of n patients)

Every release is the group's mean of a feature map φ(x) (with n public, equivalent to the sums).

| Release | φ(x) | Numbers D | Used by |
|---|---|---|---|
| `mean_var` | x, x² | 2d | Bret's "mean and variance"; variance-only scaling (R GenMatch) |
| `cov` | x, x xᵀ (upper triangle) | d + d(d+1)/2 | pooled covariance S from (n, T, Q); Khellaf's density-ratio weights |
| `quad` | products of the quadratic basis b(x) = (1, x, x xᵀ) | q(q+1)/2 − 1, q = 1 + d + d(d+1)/2 | FedDSM's per-fold B′B (covariate part; B′Y not included) |

For binary covariates x² = x, so some features coincide; the effective count D_eff is the rank of
the features' population covariance.

## Attack and attacker

The attacker knows the population (a reference sample of it, disjoint from the targets), the
release, and a target record; it decides whether the target is in the group. Score
s(x) = (φ(x) − μ)ᵀ P (m̂ − μ), with μ and P the mean and pseudo-inverse covariance of φ in the
reference sample (eigenvalues below 1e-8 × max dropped; their count is D_eff).

*Correction after the first run (2026-11-16, before any result was used):* the cutoff was applied
to the raw feature covariance, which dropped real features whose scale is tiny next to others
(IST: 7 of 34 kept for means and variances, because blood pressure² has variance ~10⁸;
log-normal fourth-order moments: 430 of 1,000). The cutoff is now on the correlation matrix
(features standardized by their reference SD); a test covers the IST case. All results are from
the corrected run. This is the
whitened inner-product (tracing) attack of Homer et al. 2008 / Sankararaman et al. 2009 /
Dwork et al. 2015.

**Prediction** (Gaussian features, known μ and P): member scores have mean D/n and non-member
scores mean 0, both with variance ≈ D/n, so AUROC ≈ Φ(√(D/(2n))) and
TPR(α) ≈ 1 − Φ(z₁₋α − √(D/n)). n and d enter only through D/n. This reproduces slide 17
(e.g. D = 65, n = 35: 0.83 and 16.8% vs 0.85 and 16.9%).

## Populations

- `gaussian`: d-variate normal, correlation 0.5^|i−j|.
- `lognormal`: exp of the same normal (long-tailed marginals; Bret's long-tail point).
- `ist`: real International Stroke Trial patients, the first d of the 31 covariates in
  `src/dgps/ist.COVARIATES` (3 continuous, then binary). The population is split once into a
  reference part (40%) and a target pool (60%); members and non-members are disjoint draws from
  the pool.

## Grid

- n ∈ {10, 20, 35, 50, 100, 200, 650, 2000}.
- d ∈ {2, 5, 10, 20, 50} for the synthetic populations; {2, 5, 10, 20, 31} for IST.
- `quad` only for d ∈ {2, 5, 10} (D = 2,210 at d = 10) and not on IST.
- Per cell: R = clip(⌈4000/n⌉, 20, 400) releases; each scores its n members and ⌈20000/R⌉ fresh
  non-members; scores are pooled over releases. Reference sample: max(20,000, 20 D) draws,
  capped at 60,000 (IST: the 40% reference part).
- Seeds: `SeedSequence([20261116, population, release, d, n])`; the reference sample per
  (population, release, d).

## Metrics (per cell)

AUROC; TPR at FPR 0.1%, 1%, 10%; membership advantage max(TPR − FPR); the empirical ε lower
bound log(TPR/FPR) at FPR 1%; the ROC curve on a log FPR grid; the prediction above.

## Reading rule

- **Prediction adequate** if every `gaussian` cell with D_eff ≥ 10 has |AUROC − Φ(√(D_eff/(2n)))|
  ≤ 0.03. Cells with D_eff < 10 are reported but not scored (the normal approximation needs many
  features).
- For `lognormal` and `ist`: report the deviation from the prediction at the same D_eff/n,
  especially TPR at 0.1% FPR (tails). No threshold; descriptive.
- Bret's question (mean and variance vs full covariance): compare `mean_var` and `cov` at equal
  n and d.

Outputs: `results/privacy/e1_mia/cells.csv`, `roc.csv`, figures in `results/privacy/e1_mia/figs/`.
