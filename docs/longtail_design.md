# Long-tailed covariates: does Fed-GenMatch beat propensity weighting? (fixed before any run, 2026-11-16)

Bret Nestor's draft (abstract): "On synthetic long-tailed data, we find that Fed-GenMatch
outperforms Fed-IPW", measured by SMD. Our results show balance and accuracy disagree
(N16, N17), so this test scores accuracy against a known effect and reports SMD alongside.

## Data (`src/dgps/longtail.py`)

Single-arm, as in the draft: n = 3,000 patients, p = 10 covariates; about 8% are treated and
form the trial site (H = 0); controls sit at 4 sites (H = 1..4, sizes in ratio 4:3:2:1,
assigned at random).

- Latent Z ~ N(0, R), R_ij = 0.5^|i−j|. Observed covariates X = g(Z) per scenario:
  - `gauss`: X = Z;
  - `t3`: X_j = F⁻¹_{t,3}(Φ(Z_j)) (symmetric heavy tails, Student t with 3 df);
  - `lognormal`: X_j = exp(Z_j) (skewed heavy tails).
- Selection into the trial: P(W = 1) = expit(c + 0.8 · s), s = bᵀu / sd(bᵀu),
  b = (1, 1, 1, 0.5, 0.5, 0, …, 0), c solved so the sample share is 8%:
  - selection `latent`: u = Z (a logistic PS in X is misspecified under heavy tails);
  - selection `observed`: u = standardized X (the logistic PS in X is correct, but heavy
    tails give extreme weights).
- Outcome (on the latent scale, so it is nonlinear in X under heavy tails):
  Y(0) = aᵀZ + 0.5 (Z₁² − 1) + 0.5 Z₂Z₃ + N(0, 1), a = (1, 0.8, 0.6, 0.4, 0.2, 0, …);
  τ_i = 1 + 0.5 Z₁; Y = Y(0) + W τ_i.
- Truth: the sample ATT, mean of τ_i over the treated (in df.attrs["tau_true"]).

Five scenarios: {t3, lognormal} × {latent, observed} and `gauss_latent`; 100 datasets each.
(*Pre-run check:* with Gaussian covariates the two selections give identical data, since
standardized X is Z, so `gauss_observed` was dropped before any results run.)

## Methods (ATT)

- Fed-GenMatch, the draft's design (`fedgm`, normaliser S_t, 1:1): FedGM and FedMaha (W = I).
- FedDSM (ps = "x").
- Pooled PS (centralized; equals an exact federated logistic fit): ATT-IPW-Hajek and ATT-AIPW.
- Centralized forest references (`flex_ref`).

## Reading rule

Per scenario, paired by dataset: Fed-GenMatch (FedGM) vs pooled PS IPW (Hajek), and FedGM vs
FedDSM, by the paired difference in squared error / its SE (|z| > 2: differ). Bret's claim is
supported in a scenario if FedGM has lower RMSE than pooled IPW with z < −2. Also reported:
bias, RMSE, and max |SMD| (the draft's metric) per method. Accuracy only.
