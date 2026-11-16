# Fed-GenMatch normaliser ablation (fixed before any run, 2026-11-15)

Question (Bret Nestor's Fed-GenMatch draft, Sec. 2.2): which covariance normaliser Sigma
should the generalized Mahalanobis distance use, given that the treated (trial) cohort is
shareable and the controls stay at their sites? And does keeping the covariates'
correlation help?

## Method (`src/estimators/fedgm.py`)

Bret's design, run centrally. Without noise the federated protocol is lossless (the
coordinator rebuilds D(W) = A - 2B + C and the matched-control moments exactly), so the
central run gives the federated matches; `tests/test_fedgm.py` checks the per-site
computation against the central one.

- Distance (x_i - x_j)' Sigma^{-1/2} W Sigma^{-1/2} (x_i - x_j); symmetric inverse square
  root; W = diag(w), w in [1e-2, 1e2].
- Optimal 1:m assignment without replacement, m = 1.
- Objective: max |SMD| over X1..X31 with Bret's denominator (treated SD and the matched
  controls' SD, eq. 11).
- GA as in the draft (tournament, BLX-alpha, Gaussian mutation, elitism, plateau stop);
  constants the draft leaves open are ours (`GA_DEFAULTS`: population 100, at most 100
  generations, stop after 10 without improvement; genes are log10 w).
- Estimate: treated mean minus matched-control mean (no bias adjustment). ATT only.

## Arms (Sigma)

| Arm | Sigma | Correlation | Control information |
|---|---|---|---|
| `diag` | diag of treated variances (floored) | no | none |
| `treated_raw` | S_t, the draft's choice; pseudo-inverse when singular | yes | none |
| `treated_shrunk` | D^{1/2} R_alpha D^{1/2}: floored treated variances, treated correlation shrunk to I (Ledoit-Wolf) | yes | none |
| `pooled_within` | ((n_t - 1) S_t + (n_c - 1) S_c) / (n - 2) | yes | per-site sums of x, x x' (one round) |

Floor: a binary covariate's treated variance is p~(1 - p~) with p~ = (sum x + 1)/(n_t + 2);
a continuous one is floored at 1e-6 of its pooled variance.
Each arm is also reported without the search (w = 1, "FedMaha"), which isolates what the
GA adds.

## Data and reading rule

IST `single_arm` (n_t ~ 240 of 3,000; reps 0-99) and `single_arm_nt50` (the same data
with 50 randomly kept treated: a rare-disease-sized cohort; reps 0-99). Primary: RMSE vs
the exact ATT, paired by rep, against FedDSM (x) and, in `single_arm` reps 0-29, the
centralized R GenMatch from Narval. An arm "differs" if the paired RMSE difference is
> 2 SE. Accuracy only (no SE).
