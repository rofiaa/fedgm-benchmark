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

## Environment setup
```
mamba env create -f environment.yml
conda activate fedgm
Rscript setup/install_r_packages.R   # rgenoud + Matching from cloud.r-project.org
pytest -q tests/                     # includes a GenMatch-through-rpy2 smoke test
```
- `Matching` and `rgenoud` are not on conda-forge for osx-arm64, so they come from CRAN
  (compiled with the env's `compilers` + `make`) into the env's R library.
- torch comes from conda-forge (`pytorch`): the pip wheel bundles its own libomp and
  aborts on import next to conda's.
- `src/r_env.ensure_r_home()` defaults `R_HOME` to the env's R; otherwise rpy2 may load a
  system R (`/Library/Frameworks`) and segfault. Call it before importing `rpy2.robjects`.
- Arguments forwarded through `...` (e.g. GenMatch → genoud `unif.seed`, `int.seed`) must
  be passed with their dotted names (`**{"unif.seed": 1}`); rpy2 only maps `_`→`.` for
  formal arguments.

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
- **Full log with evidence, commits and tests: `docs/findings.md`** (F1–F6). Summary:
  - F1: membership columns come out in alphabetical site order → wrong for K ≥ 10.
    `fed_ps` reorders them.
  - F2: `ipw_function(..., "Hajek")` per site with the global ê is inconsistent.
    In `fed_ps.estimate`: `ipw_type="HT"` is **primary**, `"Hajek_pooled"` normalizes over
    all sites, `"Hajek"` = the authors' version, **parity tests only**. The notebook's
    Fig. 2/3 runs all use HT.
  - F3: notebook DGP parameters differ from the paper (see DGPs below); Fig. 2's true-ATE
    lines (A 8.72, B 4.78) are not reproduced by any derivable parameter set: unresolved.
  - F4: known FedAvg client drift: E > 1 local steps, fixed step, label-skewed sites.
    `fed_ps.FEDAVG_MW_DEFAULTS` (benchmark): E=1, eta=1, globally standardized X, stop at
    relative Θ change < 1e-6, cap T=1000, rounds reached returned as `mw_rounds`.
    `FEDAVG_MW_NOTEBOOK` (T=30, E=3, eta=0.1, unscaled) is for parity only.
  - F5: with logistic MW, Fed-AIPW is biased in DGP A (both nuisances misspecified), as
    Fig. 2b shows, despite the text's "unbiased at all overlap levels".
  - F6: `train_NN`'s early stopping restores the last epoch (state_dict aliasing).
  - F7: `generate()` was not reproducible (LAPACK-dependent SVD factor of Σ; DGP B's
    unseeded `random.choices`). Fixed in our wrapper: Cholesky sampler + both RNGs
    seeded. Results produced before the fix cannot be regenerated.
- MW-NN in `fed_ps.estimate(mw="nn", seed=...)`: authors' `train_NN` unchanged (pooled
  data, `rerun_NN=True`), seeded via `torch.manual_seed` inside `torch.random.fork_rng`
  (int seeds pass through, for parity). ≈ 2–4 s per fit; the bootstrap retrains it in
  every replicate.
- RF local PS in `fed_ps.estimate(local_ps="rf", seed=...)`: the authors' inline forest
  (2000 trees, max_depth=2, min_samples_leaf=√n_k, `use_grf=False`), one
  `np.random.RandomState(seed)` shared by the site forests in site order (= their
  unseeded forests after `np.random.seed(seed)`). ≈ 5.5 s per estimate (3 sites).
- Oracle in `src/estimators/oracle.py` (`estimate_multi(df)`): the authors' "Oracle",
  IPW with the true e* = Σ ω*_k e*_k and AIPW with the true Y1*/Y0*. It reads
  `df.attrs["oracle"]`, which `generate()` attaches (`oracle_frame`); no other estimator
  reads it. Parity with the authors' Oracle (A: IPW + AIPW, B: IPW).
- `experiments/checks/membership_error.py`: |ω̂ − ω*| per scenario for logistic vs NN MW
  (`results/checks/membership_error.csv`): logistic ≈ 0.24 in DGP A vs 0.016 in B.
- Centralized pooled PS in `src/estimators/pooled_ps.py`: one logistic ê(x) on the pooled
  data (authors' `compute_gamma`), same IPW/AIPW and outcome models as fed_ps.
  `pooled_ps_logit(df)` is also GenMatch's PS covariate, so GM(X+ps) and pooled IPW
  differ only in matching vs weighting.
- GenMatch in `src/estimators/genmatch.py` (`estimate_multi(df, estimands, covariates,
  seed, cores, genmatch_params)`): settings of rofiaa/genmatch-lalonde (D&S) with
  `estimand="ATE"` in both GenMatch and Match; balance matrix X, X², X_iX_j; estimands
  GM, GM-BA (Abadie–Imbens bias adjustment on X), Maha, Maha-BA; AI SE from Match.
  Site H is not used. Cost at pop.size=1000: ≈ 3 min per generation on a DGP A dataset.
- Fed-DW in `fed_ps.estimate(mw="dw_param")`: the authors' "DW-param" (ω̂_k ∝ n_k
  N(x; μ̂_k, Σ̂_k), site sample moments; computed in log space). Parity with their IPW
  "DW-param" and AIPW "GD-DWparam".
- AIPW in `fed_ps.estimate(aipw=True)`: linear outcome models on X + intercept, shared
  by all sites; `outcome="ols"` (default; the authors' "GD" without federation params)
  or `"fedavg"` (authors' clipped FedAvg-SGD, `FEDAVG_OUTCOME_NOTEBOOK`). The authors'
  AIPW fits MW **without** an intercept; our default `mw_intercept=True`, parity tests
  pass `mw_intercept=False`.
- Meta-SW in `src/estimators/meta_sw.py` (`estimate(df, local_ps, ipw_type, aipw, seed,
  bootstrap)`): per-site estimates with the site's own ê_k (and per-site OLS outcome
  models for AIPW), averaged with n_k/n; parity with the authors' "Meta-SW". A site with
  one treatment arm → τ̂ = NaN with `undefined_reason` (never dropped); in the bootstrap
  such replicates count as failed (and in `boot_arm_loss`).
- Frozen result keys (`src/estimators/common.py`, `check_result`): `tau`, `se`,
  `se_method`, `max_smd`, `runtime`; everything else is an extra.
- `generate_multivariate_mixtures_of_normals` (DGP B) uses `np.NaN`, removed in NumPy 2;
  `src/estimators/_authors.py` restores the alias before importing.

### DGPs (`src/dgps/authors.py`, authors' generator)
Defaults follow the **paper** (App. D); the notebook's values are named variants.
| | Default (paper) | Notebook variant | True ATE (poor overlap) |
|---|---|---|---|
| DGP A μ₃ | (2,2,2,1,..) `MU3_PAPER` | (3,3,3,1,..) `MU3_NOTEBOOK` | 8.456 / 9.776 |
| DGP A n_k | 650 | 500 (`N_K_NOTEBOOK`) | — |
| DGP B n | 4000 | 1500 (`N_B_NOTEBOOK`) | 4.831 |
`dgp_a_notebook_params()` / `dgp_b_notebook_params()` reproduce the notebook exactly; parity
tests use them. The paper's μ₃ matches Fig. 2's true-ATE line (≈ 8.7) far better than the
notebook's; a ≈ 0.27 gap to Fig. 2's line (8.72) remains unresolved (F3). γ vectors and the outcome polynomial follow the
notebook's code (they differ from App. D Table 2; F3).
- No standard errors are returned. `fed_ps.estimate` gives both:
  - `se` / `ci_if`: influence-function SE from the per-unit terms, nuisances treated as
    fixed (provisional; conservative for IPW with estimated PS).
  - `bootstrap=B` (benchmark: `BOOTSTRAP_B` = 200): site-stratified bootstrap, rows
    resampled within each site (n_k fixed), local PS / MW (pooled fit, also for
    `mw="logistic_fedavg"`) / outcome models refit per replicate → `se_boot`,
    `ci_boot_percentile`; `boot_arm_loss(_by_site)` counts replicates in which a site lost
    a treatment arm. Seeded by `seed` via `np.random.SeedSequence(seed).spawn(B)`.
  - **No comparison table without the bootstrap.**
- `FedCaus_Obs/notebooks/simulations.ipynb` reproduces Fig. 2 (no / poor / good overlap × DGP A/B)
  and Fig. 3 (comparison with Xiong et al. 1S-IVW).
- Not in the repo: Table 1 (site effects with Z) and the Traumabase analysis (restricted data).

## Reproduction targets (qualitative, within Monte Carlo error)
- No local overlap: Meta-SW undefined; Fed-IPW/AIPW-MW unbiased.
- Poor local overlap: Meta-SW biased/unstable; Fed-AIPW unbiased when either nuisance model is
  well specified (MW-NN, or DGP B). With logistic MW in DGP A it is biased upward, as in
  Fig. 2b (docs/findings.md, F5).
- Fed-IPW-DW unbiased under DGP A, biased under DGP B.
- Good overlap: all unbiased; federated estimators have the smallest variance.

## Experiment harness (`experiments/`)
- `harness.py`: a config file defines `SCENARIOS` (`Scenario(name, generate(seed) -> df,
  truth(n, seed))`), `CONFIGS` and `BASE_SEED`. A `Config` runs on one df and returns
  {estimator label: result}; `single(...)` wraps `estimate`, `multi(...)` wraps
  `estimate_multi` (one shared nuisance fit, one row per estimand, same `config_id`).
  Bootstrap per config: `bootstrap=B, bootstrap_reps=range(...), bootstrap_for=[...]`.
  **New estimators (GenMatch) = new configs**; rows rely only on the frozen keys.
- Seeds: `SeedSequence(BASE_SEED, spawn_key=(scenario, rep[, config]))`, so any rep slice
  reproduces the rows of a full run (tested).
- `run.py <config> --reps 0-29 --jobs 7 --out results/<run>` → one file per (scenario, rep),
  `<out>/rows/<scenario>/rep<NNNN>.csv`, written as each finishes; **resumable** (rerun
  the same command: completed pairs are skipped); prints a projected finish time; SLURM: one array task per rep slice with `--jobs 1`. True ATEs are cached
  in `<out>/truth/`. `summarize.py <out>` → `<out>/summary.csv`.
- Long local runs: `caffeinate -i` (a sleeping Mac stalls the pool).
- Narval: `experiments/slurm/array.sh CONFIG OUT REP_OFFSET SCENARIO...` runs one
  (scenario, rep) per array task (task i → scenario i % S, rep OFFSET + i / S).
  GenMatch runs only on Narval (rgenoud's search need not be bit-identical across
  machines).
- IHDP (`src/dgps/ihdp.py`, data via `setup/download_ihdp.sh` into untracked `data/`):
  100 replications (Shalit et al.), same 747 units / design in every replication (rows put
  in a canonical order), sites H = the 8 IHDP sites (x19..x25 dummies), truth = sample ATE
  per replication in `df.attrs["tau_true"]`. Harness: `Scenario(by_rep=True)` calls
  `generate(rep)`; a frame's `attrs["tau_true"]` overrides the scenario truth, and
  `summarize` scores against each row's truth. `experiments/configs/ihdp_mc.py`: fed / Meta-SW
  on X1..X18 + H, pooled PS and GenMatch on X1..X25 (site dummies); GenMatch balance
  matrix = 25 covariates + squares/products of X1..X6 (`balance_quad`). Fed-DW is not
  applicable (singular site covariances). GenMatch ≈ 23 min per replication on one core.
- FedDSM (our proposed method; docs/findings.md N5–N8) in `src/estimators/feddsm.py`
  (`estimate_multi(df, estimands, target="ATE"|"ATT", ps="x"|"site"|"khellaf", M=4)`):
  match on (logit ê, cross-fitted quadratic prognostic score m̂₀ [, m̂₁]) with
  Mahalanobis, M = 4, same-model bias adjustment. ê by distributed Newton (exact pooled
  MLE; ps="site" adds site indicators); m̂ from per-site, per-fold sufficient statistics.
  The PS follows the dataset's confounding assumption: "x" for the authors' DGPs
  (the site is not a confounder), "site" for IHDP. No SE yet (se = NaN). Privacy: the
  per-site, per-fold sums disclose patients exactly in cells of 1–2 patients; a
  protocol and threat model are pending.
- Centralized flexible references in `src/estimators/flex_outcome.py` (RF AIPW, regression,
  and `DSM_ESTIMANDS`: FedDSM's matching with forest prognostic scores). On ACIC the
  forest score beats AIPW RF + RF; the quadratic score is what limits FedDSM (findings N10, N11).
- `experiments/compile_comparison.py` → `results/comparison/` builds the all-methods ×
  all-datasets table; `docs/evaluation_metrics.md` logs the metrics used in the
  literature (read from the papers; verbatim vs summary marked).
- `experiments/configs/ps_mc.py`: the six-scenario PS-aggregation MC.
  `experiments/mini_mc.py` is superseded (kept for its committed results).

## Metrics for every method
bias, RMSE, 95% CI coverage, CI width, post-match/weight balance (max |SMD|), runtime.

## Working rules
- Plan before editing; keep changes small; one estimator or one DGP per task.
- Logic lives in `.py` modules; notebooks only for plots.
- Heavy simulations (hundreds of reps) run from `experiments/` scripts, not interactively.
- Run `pytest -q tests/` before declaring a task done.