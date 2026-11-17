# E2: reconstruction attacks on Fed-GenMatch's messages (fixed before any run, 2026-11-16)

Task: the professor's privacy evaluation (Nov 4 meeting), applied to the protocol of Bret
Nestor's draft (Sec. 2.2; `src/estimators/fedgm.py`). Evaluation only: the honest-but-curious
coordinator sees exactly what the draft sends. Bret's draft states the reconstruction risks
qualitatively; E2 measures them.

## What the coordinator sees (draft eqs. 7, 9)

Public: the treated rows x̃ᵗ = Σ^{-1/2}xᵗ, every candidate W (it runs the search). Per
evaluation g and site k: the cross terms B⁽ᵏ⁾ = (X̃ᵗ W_g) x̃ᶜ (n_t numbers per control), the norms
C⁽ᵏ⁾, the assignment (which control columns are matched), and the matched controls'
(n*_k, Σx, Σx²).

## Attacks

- **A1, one evaluation (draft bullet 1).** Control j's column of B is M x̃ᶜ_j with
  M = X̃ᵗW (known). The attacker solves least squares for the deviation from the population
  mean μ (min-norm, so unseen directions stay at the mean) and maps back:
  x̂ = μ + Σ^{1/2}(ẑ − Σ^{-1/2}μ) (pseudo-inverses when Σ is singular). C is not used, so the
  attack is conservative.
- **A2, several evaluations (draft bullet 2).** The same with the M's of the first G
  evaluations of the actual search stacked; G ∈ {1, 2, 5, 10, 31, 100}; plus the number of
  evaluations until the stacked M reaches the rank of Σ (full exposure).
- **A3, matched-set moments (not in the draft).** The coordinator knows which controls each
  evaluation matched (it solved the assignment) and receives Σx over them. Control j is
  identified exactly iff e_j lies in the row space of the site's evaluations × controls
  match-indicator matrix. Count the identified controls among those ever matched.
- **A4, the draft's DP (eq. 14).** Laplace noise on every entry of B, scale Δ_i/ε with
  Δ_i = ‖W x̃ᵗ_i‖₂‖Σ^{-1/2}‖₂ρ, ρ = the control pool's diameter (2 max‖x − μ‖₂);
  ε per entry ∈ {0.1, 1, 10}. The attacker uses weighted least squares over G evaluations,
  G ∈ {1, 10, 100}. Implied per-patient ε by basic composition: G · n_t · ε (one control
  changes n_t entries per evaluation).

*Changes after a one-task pilot (dataset 0, n_t = 10, shrunk normaliser), before the full
run:*
- Covariates equal to 0 for every treated patient never enter the cross terms, whatever W is
  (the exposed rank stopped at 21 of 31), so whole-record exact recovery is 0 by construction.
  Added: the share of individual covariate values recovered exactly.
- A4 with plain weighted least squares blew up numerically (errors of 10¹⁶ from noise along
  unexposed directions). The A4 attacker now uses the posterior mean under its population
  prior (Gaussian approximation of the noise), and pseudo-inverses use a 10⁻¹⁰ cutoff.
- Added the utility side of the draft's DP: the share of treated patients whose match is
  unchanged by the eq.-14 noise on B (first evaluation, W = I; C left exact).
- At ε ∈ {0.1, 1, 10} per entry the pilot showed no leakage and no utility (0% of matches
  kept), so the grid adds ε ∈ {100, 1000} per entry to locate the trade-off.

## Data and grid

IST single-arm (Bret's setting; 31 covariates, 3 continuous), datasets 0–9; treated cohort
n_t ∈ {10, 20, 31, 50, all ≈ 235} (subsampled as in `ist.load(n_treated=...)`); normaliser
`treated_raw` (the draft's S_t; singular on most datasets) and `treated_shrunk` (full rank).
The search is the draft's GA (`fedgm._search`, GA_DEFAULTS, seed fixed per dataset), recorded
evaluation by evaluation. Controls at their real sites (countries).

## Metrics (per control, summarized per cell)

- **Exact recovery:** max over covariates of |x̂ − x| / SD < 10⁻⁶ (SD over the control pool).
- **Standardized error:** RMS of (x̂ − x)/SD over covariates; baseline: guessing μ.
- **Binary attributes:** share of the 28 binary covariates recovered correctly after rounding;
  baseline: the majority value.
- **Re-identification:** x̂'s nearest neighbour (standardized Euclidean) among its site's
  controls is the true control; baseline 1/n_site.
- A3: share of ever-matched controls identified exactly; their number.

## Predictions (Bret's claims, plus ours)

- A1: exact recovery for every control iff rank(X̃ᵗ W) = p; with n_t < p only partial. On IST
  binary covariates constant among the treated can make X̃ᵗ rank-deficient even for n_t ≥ p.
- A2: with full-rank Σ, the stacked M reaches rank p after a few evaluations even for n_t < p,
  so every control is recovered exactly. With the draft's singular S_t, only the exposed
  subspace (rank of S_t) is recoverable; the rest stays at the mean.
- A3: most ever-matched controls are identified from the moments alone.
- A4: per-entry noise averages out over evaluations: the error falls with G.

No thresholds: E2 reports these quantities.

Outputs: `results/privacy/e2_fedgm/cells.csv`, figures in `results/privacy/e2_fedgm/figs/`.
