# FedGM benchmark — project brief for Claude Code

## Goal
Benchmark **centralized genetic matching (GenMatch)** against the federated
propensity-score-aggregation estimators of Khellaf, Bellet & Josse
(arXiv:2505.17961, ICML 2026) on multi-site **observational** data.
This decides whether developing a federated/multi-site GenMatch is worthwhile.
Later stages add Meta-GM (per-site GenMatch, sample-size-weighted) and a federated GM.

The RCT paper (Khellaf et al., AISTATS 2025, arXiv:2410.16870) is OUT OF SCOPE for now.

## Estimand and conventions
- Estimand: **ATE** (risk difference). GenMatch/Match default to ATT — always pass `estimand="ATE"`.
- Matching SEs: Abadie–Imbens (as returned by `Matching::Match`).
- Every estimator is exposed as `estimate(df, **cfg) -> dict(tau=..., se=..., extras...)`.
- Data frame columns: covariates `X1..Xd`, site `H` (0..K-1), treatment `W`, outcome `Y`.
- All randomness goes through an explicit `seed` argument. No global seeding inside library code.

## Repo layout
- `third_party/FedCausal-ObservationalStudies/` — authors' code (git submodule). **Never edit.** Wrap it.
- `papers/` — PDFs of the relevant papers. Read them when unsure about a definition.
- `src/estimators/` — thin wrappers (authors' Fed-IPW/AIPW, Meta-SW, our GenMatch variants).
- `src/dgps/` — DGP A and DGP B from the paper (App. D, Tables 2–4), later our own scenarios.
- `experiments/` — scripts + configs that write results to `results/` (CSV) and figures.
- `tests/` — fast checks (few reps) that reproduce the paper's qualitative patterns.

## Notes on the authors' code (observational repo, commits of 2026-09-24, no README)
- `FedCaus_Obs/scripts/DGP_Fed.py` (~3.7k lines): data generation, membership weights (MW),
  density-ratio weights (DW), FedAvg, and class `Simulations_Fed` with `estimate_ipw(...)`
  and `estimate_aipw(...)`.
  - `estimate_ipw` has `final_estimators` defaulting to
    `["Oracle", "Meta-SW*", "Meta-SW", "MW", "DW-param"]` → Meta-SW baseline is included.
  - `estimate_aipw` has `outcome_estimators = ["local", "pool", "1SIVW", "GD"]`.
  - `propensity_estimation`: 'logistic' | 'random_forest' | '1or0'; `use_grf=True` calls R's
    `grf::probability_forest` through rpy2 (default False).
- It does `from utils import basic_utils`, so the `scripts/` folder must be on `sys.path`.

### Using the method on our own data
`Simulations_Fed.estimate_ipw/estimate_aipw` **generate data inside their simulation loop**, so they
cannot take an external data frame. Therefore:
- Implement our own `src/estimators/fed_ps.py` (`estimate(df, ...)`), re-using the authors'
  **standalone functions** by import (don't copy-paste, don't edit the submodule):
  - `federated_logistic_regression`, `MW_estimation` (FedAvg multinomial logistic for P(H|X)),
    `MW_estimation_pooled` (centralized equivalent), `softmax`
  - `membership_weighting_vectorized(omega, hat_e_k)` → global ê(x) = Σ_k ω̂_k(x) ê_k(x)
  - `ipw_function(df, hat_e_name, ht_or_hajek="HT"|"Hajek")`, `aipw_function(df, y1_col, y0_col, e_col)`
  - Local propensities ê_k are fitted inline in `estimate_ipw` (sklearn logistic / RF / grf);
    reimplement that step ourselves (each site fits on its own data, predicts on ALL rows).
- Their functions expect a column `client` with values `"client1".."clientK"` (string labels,
  1-indexed), not our integer `H`. Convert inside the wrapper.
- **Validation test:** on the same generated data/seed, our `estimate()` must reproduce the number
  `Simulations_Fed` gets (DGP A, poor overlap, logistic propensities, 1–2 reps).

### Known caveats in the authors' code (document, don't "fix" upstream)
- Default MW-logistic is `MW_estimation_pooled` (centralized sklearn fit); the FedAvg version runs
  only if `params_for_federation_omega` is passed. Same target, so fine for the benchmark, but report it.
- `train_NN` (MW-NN) trains a 1-hidden-layer (64 units) net on **pooled** data with Adam — it is
  not FedAvg in this code.
- `MW_estimation(..., scale=True)` fits Theta on unscaled X but predicts on scaled X → mismatch.
  Keep `scale=False` (default) or standardize X before calling.
- No standard errors are returned. We compute SEs ourselves: influence-function (sandwich) SE for
  IPW/AIPW, and/or bootstrap — needed for CI coverage.
- `FedCaus_Obs/notebooks/simulations.ipynb` reproduces Fig. 2 (no / poor / good overlap × DGP A/B)
  and Fig. 3 (comparison with Xiong et al. 1S-IVW).
- Not in the repo: Table 1 (site effects with Z) and the Traumabase analysis (restricted data).

## Reproduction targets (qualitative, within Monte Carlo error)
- No local overlap: Meta-SW undefined; Fed-IPW/AIPW-MW unbiased.
- Poor local overlap: Meta-SW biased/unstable; Fed-AIPW unbiased.
- Fed-IPW-DW unbiased under DGP A, biased under DGP B.
- Good overlap: all unbiased; federated estimators have the smallest variance.

## Metrics for every method
bias, RMSE, 95% CI coverage, CI width, post-match/weight balance (max |SMD|), runtime.

## Working rules
- Plan before editing; keep changes small; one estimator or one DGP per task.
- Logic lives in `.py` modules; notebooks only for plots.
- Heavy simulations (hundreds of reps) run from `experiments/` scripts, not interactively.
- Run `pytest -q tests/` before declaring a task done.