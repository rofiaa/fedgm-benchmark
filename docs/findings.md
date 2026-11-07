# Findings on the authors' code and paper

Issues found while wrapping Khellaf, Bellet & Josse (arXiv:2505.17961v4) and its code
(`third_party/FedCausal-ObservationalStudies`, commit `e5f4409`). The submodule is never
edited; each entry says how our wrapper (`src/estimators/fed_ps.py`) handles it.
Line numbers refer to `FedCaus_Obs/scripts/DGP_Fed.py` unless stated.

| # | Finding | Affects paper results? | Our handling |
|---|---|---|---|
| F1 | Site ordering breaks for K ≥ 10 | Possibly F.1 / Table 1 (K = 30); not in repo | Columns reordered |
| F2 | Site-wise Hajek IPW is inconsistent | **No**: all published figures use HT | `Hajek_pooled` added; `Hajek` parity-only |
| F3 | DGP A/B parameters: notebook ≠ paper; Fig. 2 true ATE not reproduced | Fig. 2 true-ATE line; **unresolved** | Paper values default, notebook as variants |
| F4 | Known FedAvg client drift (E > 1, fixed step, label-skewed sites) | Notebook Fig. 2a AIPW uses E = 3, T = 30 | Default E = 1, standardized X, stopping rule |
| F5 | "Fed-AIPW unbiased at all overlap levels" (text) vs Fig. 2b | Text vs figure | Reproduction target narrowed |
| F6 | `train_NN` early stopping keeps the last epoch, not the best | Minor; MW-NN results | Kept for parity; documented |

---

## F1. Membership-weight columns are in alphabetical site order (K ≥ 10)

**What.** Every membership classifier returns one column per site in *sorted-label*
order, `client1, client10, client11, client2, …`:
- `MW_estimation_pooled`: `LogisticRegression.predict_proba` (l. 360)
- `federated_logistic_regression`: `pd.get_dummies(df["client"])` fixes the order of
  the rows of Θ (l. 296)
- `train_NN`: `LabelEncoder` (l. 1291–1292)
- `MW_RF` branches: `RandomForestClassifier.predict_proba` (l. 1707, 2616)

The columns are then named and combined *positionally* as sites 1..K: `omega_mw_{k+1}`
(l. 334), `omega_mw_pool_{k+1}` (l. 352–353), `mw_logistic{k+1}`, then
`membership_weighting_vectorized(ω, ê_local)` (l. 369), which multiplies column k of ω
with site k's local score. For K ≤ 9 the two orders coincide; from K = 10 site 2's local
score is weighted by site 10's membership probability, and so on.

**Evidence.** 11 well-separated sites (`X1 = 3·H + noise`), authors'
`MW_estimation_pooled`; for each site, the column with the largest mean ω̂ over that
site's rows is `[0, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2]` instead of `[0, 1, …, 10]`.
No code path in the repo or notebook reorders the columns. The notebook only runs K = 3;
the K = 30 experiment (Table 1) and App. F.1 (large K) are not in the repo, so whether
they were affected cannot be checked.

**Our handling.** `fed_ps._site_order_numeric` maps the authors' columns back to numeric
order (`6469dc6`); our FedAvg loop builds classes in numeric order (`d321486`).
Test: `tests/test_fed_ps.py::test_membership_columns_in_numeric_site_order_for_k_ge_10`
(pooled and FedAvg).

---

## F2. The site-wise Hajek estimator is inconsistent

**What.** With `ht_or_hajek="Hajek"`, the final estimators call `ipw_function` (l. 377;
Hajek branch l. 393–398) on each site's rows with the **global** ê, then average the
sites with weights n_k/n (e.g. the `MW_logistic` block, l. 2006–2027). Within site k the
normalizer Σ_{i∈k} W_i/ê(X_i) estimates n_k · E[e_k(X)/e(X) | H = k], which is not n_k
because the global e(X) ≠ e_k(X). So each site's normalized mean is not that site's
E[Y(1)], and the n_k-weighted average does not undo it. HT has no normalizer: its
per-site sums average back to the pooled HT estimator, which is consistent. Normalizing
once over all sites ("pooled Hajek") is also consistent.

**Evidence.** Mini MC, poor local overlap, pooled logistic MW, 100 reps:

| DGP | Site-wise Hajek (authors) bias | Pooled Hajek bias | HT bias |
|---|---|---|---|
| B (MW well specified), τ = 4.83 | −1.035 ± 0.014 (`6de3c74`) | +0.004 ± 0.025 (`d321486`) | +0.015 ± 0.031 |
| A (MW misspecified), τ = 8.46 | −1.736 ± 0.022 (`6de3c74`) | +1.060 ± 0.030 (`d321486`) | +0.943 ± 0.040 |

In A the pooled Hajek shares HT's upward bias, which comes from the misspecified
logistic membership model (the paper: logistic MW is well specified only in DGP B).

**F2 does not affect the published figures.** Every `estimate_ipw` call in
`notebooks/simulations.ipynb` uses `ht_or_hajek="HT"`: Fig. 2 (all three overlap levels,
both DGPs: cells 7, 15, 25, 34, 43, 51) and Fig. 3 (cell 59). `estimate_aipw` has no
Hajek option. The issue only matters to someone calling the code with `"Hajek"`.

**Our handling.** `ipw_type="Hajek_pooled"` (`d321486`); `"Hajek"` reproduces the
authors' estimator and is used only in parity tests. HT is primary in the benchmark.
Tests: `tests/test_fed_ps_parity.py::test_ipw_mw_logistic_matches_authors[*-Hajek]`,
`tests/test_fed_ps.py::test_hajek_pooled_normalizes_over_all_sites`,
`tests/test_fed_ps.py::test_hajek_terms_average_to_tau`.

---

## F3. DGP parameters differ between the notebook and the paper — **unresolved**

**Status.** Timeboxed to one hour (17:59–18:01 on 2026-10-24; closed early after the
checks below ran out of hypotheses). The parameter differences are documented; the
figure's true-ATE values are **not** reproduced by any parameter set we can derive from
the paper or the code. Needs input from the authors.

**What.** `notebooks/simulations.ipynb` vs the paper's App. D:

| Parameter | Notebook | Paper | Default in `src/dgps/authors.py` |
|---|---|---|---|
| DGP A n_k | 500 (`50 * dim_x`) | 650 (Table 3) | 650 (`N_K_NOTEBOOK` = 500) |
| DGP A μ₃ | (3, 3, 3, 1, …) | (2, 2, 2, 1, …) (Table 3) | (2, 2, 2) (`MU3_NOTEBOOK` = 3) |
| DGP B n | 1500 | 4000 (Table 4) | 4000 (`N_B_NOTEBOOK` = 1500) |
| γ_k | leading 0, 11 entries | 10 entries | notebook's |
| Outcome μ₁, μ₀ | squares of X1–X6, interactions X10·intercept / X1·intercept | squares of X1–X5, X9·X10 / X1·X10 (Table 2) | notebook's (`generate_Y`) |
| Reps, poor overlap | `n_simulations = 1` (cells 24, 33) | 800 | — |

The γ shift: `treatment_cols = X1..X10 + ["intercept"]` puts the intercept *last*, so
the notebook's leading 0 multiplies X1 and the last entry multiplies the intercept.
The outcome differences come from `generate_Y`'s polynomial branch iterating over
`outcome_cols` (which include the intercept).

**Fig. 2's true-ATE lines, measured.** Page 7 rendered at 600 dpi; the red line's pixel
row interpolated between the two neighbouring gridlines: **DGP A 8.72** (left panel
8.707, middle 8.734), **DGP B 4.78** (4.776).

**What we get** (Monte Carlo, MC SE ≤ 0.008):

| Candidate | DGP A true ATE | Gap to 8.72 |
|---|---|---|
| Code outcome, notebook μ₃ = (3,3,3) | 9.776 | +1.06 |
| Code outcome, paper μ₃ = (2,2,2) | 8.456 | −0.27 |
| Paper Table 2 outcome formula, paper μ₃ | 7.47 | −1.25 |
| Paper Table 2 outcome formula, notebook μ₃ | 8.80 | +0.08 |
| Plotter `tau_star="expectancy_of_tau"` (linear-β formula) | ≈ 3.05 | — |

DGP B: ours 4.831, the figure 4.78, the notebook's hard-coded `tau_star=4.74` (cell 37):
three different values for the same estimand.

**Exact values (2026-11-04).** Both outcome versions have τ(x) = μ₁ − μ₀ with every
polynomial coefficient differing by 1/3, so the true ATE follows from the Gaussian
moments in closed form. Code outcome: **A 8.450, B 4.833** (our MC truths 8.456 and
4.831 agree, an independent check of `true_ate`). App. D Table 2 formula: A 7.472
(8.806 with the notebook's μ₃), **B 4.028**. Fig. 2's DGP B line (4.78) rules out the
Table 2 formula, so the published figure was produced with the code's outcome; our
setup (code outcome and γ, paper n_k, n and μ₃) is the closest available reproduction.
The DGP A gap (8.72 vs 8.45) remains.

**Ruled out.**
- *Notebook μ₃ with the code's outcome*: 9.78, far from 8.72.
- *Paper's Table 2 formula*: 7.47 with the paper's μ₃. With the notebook's μ₃ it gives
  8.80, closest of all, but that mixes the paper's outcome with the notebook's
  covariates, and the code cannot produce that outcome.
- *Unequal site weights*: per-site ATEs (paper μ₃) are 6.655 / 6.415 / 12.287; 8.72
  needs weight 0.38 on site 3 (0.22 with the notebook's μ₃), not 1/3 from equal n_k.
- *Other μ₃*: the code's outcome reaches 8.72 near μ₃ ≈ (2.25, 2.25, 2.25), which
  neither source uses.
- *The Plotter's other option*: the linear-β formula gives ≈ 3.05.
- *An older code version*: the submodule has one upload commit (`c140d28`) plus a
  folder rename (`e5f4409`); no history to compare.
- *Monte Carlo noise*: MC SE ≤ 0.008, versus gaps of 0.05–0.27.

The *estimates* match the figure better than the true-ATE line does: Fed-MW (logistic)
IPW median ≈ 9.25 in Fig. 2b vs 9.34 in our MC (`d321486`). So the bias we report
relative to our true ATE (+0.89 median) is larger than the figure suggests (≈ +0.55).

**Our handling.** Defaults follow the paper; `dgp_a_notebook_params()` /
`dgp_b_notebook_params()` and the constants above give the notebook's values
(`d321486`). Parity tests run on the notebook's settings
(`tests/conftest.py::dgp_a_poor`); the mini MC runs both μ₃ (`A`, `A_mu3nb`).

---

## F4. FedAvg membership weights: known client drift with E > 1

**What.** This is the well-known *client drift* of FedAvg: with more than one local
step per round (E > 1), a fixed step size η, and heterogeneous (here maximally
label-skewed) sites, FedAvg converges to a fixed point that is not the minimizer of the
global loss, with a bias that shrinks with η·E (e.g. Karimireddy et al., 2020,
"SCAFFOLD"; Li et al., 2020, "On the Convergence of FedAvg on Non-IID Data"). It is not
a coding error. `federated_logistic_regression` (l. 291–317) is FedAvg with E
full-batch local steps; for membership weights every site holds a *single* class
(H = k), the most extreme label skew, so local steps pull Θ toward "everything is my
site". The paper's App. C.2 notes convergence to the centralized solution "with a
suitable choice of learning rate and a moderate number of local steps".
The notebook's `federation_parameters_omega` is T = 30, η = 0.1, E = 3; it is passed in
the no-overlap DGP A **AIPW** cell (cell 8) and the DGP B no-overlap IPW cell (cell 15).

**Evidence.** One DGP A dataset (paper μ₃, seed 7), unscaled X, 30 000 rounds without
early stopping, max |ω̂_FedAvg − ω̂_pooled| and τ̂ difference:

| E | η | max abs ω gap | τ̂ gap |
|---|---|---|---|
| 3 | 0.1 | 0.132 | −0.344 |
| 3 | 0.01 | 0.026 | −0.036 |
| 1 | 0.1 | 1.6e−4 | −0.0004 |

At the notebook's T = 30 the fit is also far from converged (max ω gap 0.48 on the
parity dataset). With E = 1 on unscaled X, η ≥ 0.3 diverges and η = 0.1 needs
1400–2000 rounds.

**Our handling.** `FEDAVG_MW_DEFAULTS`: E = 1, η = 1, X standardized by its global
mean/SD, stop at relative Θ change < 1e−6, cap T = 1000; `mw_rounds` / `mw_converged`
are returned (`d321486`). In the mini MC every rep converged (max 105 / 309 / 934
rounds for A / A_mu3nb / B) and agreed with the pooled fit to 3 decimals.
`FEDAVG_MW_NOTEBOOK` (T = 30, E = 3, η = 0.1, unscaled) reproduces the authors exactly.
Tests: `tests/test_fed_ps.py::test_converged_fedavg_matches_pooled_fit[A, B]`,
`tests/test_fed_ps_parity.py::test_ipw_mw_logistic_matches_authors[logistic_fedavg-*]`.

---

## F5. Fed-AIPW with logistic MW is biased in DGP A (poor overlap), as Fig. 2b shows

**What.** §4.1 says Fed-AIPW "remains unbiased across all overlap levels by double
robustness". In DGP A, however, *both* nuisance models are misspecified for Fed-MW
(logistic): the membership model (the paper: logistic MW is well specified only in
DGP B) and the linear outcome model (the outcome is polynomial). Double robustness then
gives no guarantee. Fig. 2b (top) agrees with this, not with the text: the hatched green
box (Fed-MW logistic, AIPW) sits at ≈ 9.5 against a true-ATE line at ≈ 8.7, while
Fed-MW (NN) AIPW is centred on the line.

**Evidence.** 20 reps, pooled logistic MW, pooled-OLS outcome (`fed_ps.estimate(aipw=True)`):
DGP A (paper μ₃, τ = 8.456) bias +0.75 (SD 0.19); DGP B (τ = 4.831) bias +0.02 (SD 0.15).
Same with the authors' FedAvg-SGD outcome fit (+0.75 / +0.02).

**Our handling.** CLAUDE.md's reproduction target now reads "Fed-AIPW unbiased when
either nuisance model is well specified (MW-NN, or DGP B)". AIPW parity:
`tests/test_fed_ps_parity.py::test_aipw_mw_logistic_matches_authors[ols, fedavg]`.

---

## F6. `train_NN`'s early stopping restores the last epoch, not the best

**What.** `train_NN` (l. 1277–1357) keeps `best_model_state = model.state_dict()`
(l. 1338) when the epoch loss improves and calls `model.load_state_dict(best_model_state)`
after training (l. 1346). `state_dict()` returns references to the live parameter
tensors, not a copy, so the "best" snapshot keeps changing with the model and the
restore is a no-op: the returned net is the one from the last epoch, i.e. 5 epochs
(the patience) after the best training loss, or epoch 100. A copy
(`copy.deepcopy(model.state_dict())`) would be needed.

**Evidence.** Minimal reproduction: `m = nn.Linear(2, 1); best = m.state_dict();
m.weight += 1` → `best["weight"]` has changed too.

**Impact.** Small: the training loss is being minimized with no regularization, and
5 extra Adam epochs rarely move far from the best epoch; the stopping rule itself still
works. MW-NN behaves as the paper reports in our check (20 reps, `mw="nn"`, IF only):
AIPW bias −0.03 ± 0.04 in DGP A (logistic MW: +0.77) and −0.03 ± 0.02 in DGP B.

**Our handling.** `fed_ps` calls the authors' `train_NN` unchanged (parity:
`tests/test_fed_ps_parity.py::test_ipw_mw_nn_matches_authors`,
`::test_aipw_mw_nn_matches_authors`). Also: `train_NN` trains on the **pooled** data
(not FedAvg as in the paper's App. C), and with `rerun_NN=False` it returns the cached
model object instead of probabilities; `fed_ps` always passes `rerun_NN=True`.

---

## F7. The generator was not reproducible from its seed (our wrapper) — **fixed**

**What.** Two sources, found when Narval and the Mac produced different data for the
same `data_seed` (ps_mc reps 0–20: identical seeds, Fed-IPW-logistic τ̂ differing by up
to 1.2; true ATEs differing within MC error).
1. *DGP A and B, across machines.* The covariates are drawn with numpy's legacy
   `multivariate_normal` (directly in DGP B, through `scipy.stats.multivariate_normal.rvs`
   in DGP A), which colours standard normals with an SVD factor of Σ. Every Σ here has
   a repeated eigenvalue (e.g. 1,…,1,6 for I + 0.5J), so the factor is unique only up to
   a rotation within that eigenspace, and Apple Accelerate and OpenBLAS return different
   ones. Same normals, different rotation, different X. On one machine, DGP A repeated
   exactly.
2. *DGP B, on any machine.* `generate_multivariate_mixtures_of_normals` picks mixture
   components with Python's `random.choices`; `generate()` seeded only numpy's RNG.
   Two calls with the same seed gave different data.

**Effect.** Every dataset was a valid draw from the intended distribution, so the
preliminary MC metrics (results/ps_mc, not committed) are statistically valid, but the
datasets cannot be regenerated, so later estimators (GenMatch) could not be run on them.

**Fix** (`src/dgps/authors.py`, `_seeded_generator`; the submodule is unchanged).
`generate()` now seeds and restores both numpy's global RNG and `random`, and swaps in
a Cholesky sampler (`_mvn_cholesky`) for the two multivariate-normal calls. The Cholesky
factor is unique; any factor gives the same N(μ, Σ); the sampler consumes the same
standard normals, so all later draws (W, noise) stay aligned. Datasets therefore differ
from those drawn before the fix (same distribution). Tests: `tests/test_dgps.py`.
Cross-machine agreement is checked on reps 0–1 of the rerun.

## Checks of flagged results

### N1. NN IPW-HT bias in DGP A: not confirmed at 100 reps

**Flag.** A 20-rep check (`f733b73`) gave Fed-IPW-HT with MW-NN a bias of −0.34 ± 0.12
in DGP A (paper μ₃, τ = 8.456).

**Recheck.** 100 reps, IF inference
(`experiments/checks/nn_ipw_dgp_a_recheck.py` → `results/checks/nn_ipw_dgp_a_recheck.csv`):

| Method | Bias ± MC SE | Median bias | SD | Mean IF SE | IF coverage | min ê |
|---|---|---|---|---|---|---|
| NN, IPW-HT | −0.095 ± 0.053 | −0.200 | 0.529 | 0.751 | 0.98 | 1.9e−3 |
| NN, AIPW | −0.036 ± 0.021 | −0.068 | 0.213 | 0.252 | 0.98 | 1.9e−3 |
| logistic, IPW-HT | +0.930 ± 0.037 | +0.892 | 0.370 | 0.642 | 0.92 | 1.9e−2 |

**Conclusion.** The −0.34 was Monte Carlo noise (≈ 2 MC SE). At 100 reps the mean
bias is −0.10 ± 0.05, within 2 MC SE of zero. The median (−0.20) sits below the mean:
NN membership weights push the smallest global ê to ≈ 2e−3 (ten times smaller than
with logistic MW), which gives IPW-HT a right tail. Recheck in the six-scenario MC.

### N2. GenMatch pilot: every matching estimate is biased downward — matching discrepancy, not a bug

**Flag.** GenMatch pilot (`experiments/configs/gm_mc.py`, reps 0–1, all six scenarios,
`results/gm_mc`): GM, GM-BA, Maha and Maha-BA all underestimate the ATE in all 12
datasets, by −0.17 to −0.84, including good overlap, where the PS estimators are
unbiased. AI-SE coverage mostly fails.

**Check.** `experiments/checks/matching_bias.py` → `results/checks/matching_bias.csv`:
Mahalanobis matching with the same covariates and Match settings, compared with the
sample ATE mean(Y1* − Y0*):

| | Error vs sample ATE (12 datasets) |
|---|---|
| Maha (no adjustment) | −0.34 to −0.88 |
| Maha-BA, Z = X (linear; our GM-BA / Maha-BA) | −0.26 to −0.68 |
| Maha-BA, Z = X, X², X_iX_j | −0.10 to +0.10 |

Rebuilding the estimate from `index.treated`/`index.control`/`weights` reproduces
Match's number exactly.

**Conclusion.** The matches are correct: once the bias adjustment can represent the
quadratic outcome, the bias disappears. The bias is 1:1 matching's discrepancy in
11 continuous dimensions times the outcome's curvature (Abadie & Imbens 2006), and it
does not shrink much at n = 2000–4000. Linear bias adjustment removes only part of it.
A quadratic Z gives matching a correctly specified outcome model, while our AIPW
outcome models are linear in X. A fair comparison needs the same outcome-model class
on both sides (open decision).

### N3. GenMatch never reaches balance on the ATE; on the ATT it does (IHDP)

**Flag.** In the IHDP ATE run (`results/ihdp_mc`, 100 replications), GenMatch's
lexical fitness, the smallest balance p-value after matching, lies between 1e-7 and
6e-5 in every replication. The learned weights stay close to uniform, and GM-BA is no
better than Mahalanobis-BA. On the authors' DGPs (`results/gm_mc` pilot) it lies
between 1e-13 and 0.02.

**Check.** On IHDP replication 0, a short search (pop 200, 30 generations) and
MatchBalance on the same balance matrix:

| | Min balance p-value | Worst terms (std. diff. ×100) |
|---|---|---|
| GenMatch, ATE | 4e-7 | x3² (−23), x9 (22), site-5 dummy x23 (−15) |
| Mahalanobis, ATE | 4e-16 | x3² (−47), x1·x3 (31), x23 (−36) |
| GenMatch, ATT | 0.22 | about ±10 |
| Mahalanobis, ATT | 0.007 | x5² (24) |

genmatch-lalonde's IHDP run (ATT) reported 0.25.

**Conclusion.** The code is correct; the estimand is the problem. On the ATE, each of
the 608 controls needs a treated match among only 139 treated, and some regions have
almost none: X18 = 0 has 1 treated and 26 controls, site 5 has 2 treated and 53
controls. The weights cannot create treated units, so the imbalance is structural
(one-sided overlap). In the full ATT run (`results/ihdp_att_cc`, Narval), the min
p-value is 0.15–0.32 in every replication, and GM-BA beats Mahalanobis-BA (RMSE 0.61
vs 0.67).

IHDP ATT results (`experiments/configs/ihdp_att.py`; truth 4 in every replication):

- Errors are heavy-tailed for every method: |error| correlates 0.72 with the
  replication's outcome SD.
- The ATT has to impute the control surface for the treated, and that surface is
  exponential in Hill's surface B.

| | RMSE (100) | RMSE (90 calmer reps*) | Median \|err\| |
|---|---|---|---|
| Meta-SW AIPW | 0.40 | 0.18 | 0.11 |
| Fed-AIPW, NN weights | 0.40 | 0.27 | 0.14 |
| Fed-AIPW, logistic / FedAvg weights | 0.49 | 0.29 | 0.16 |
| Pooled PS-AIPW | 0.53 | 0.20 | 0.10 |
| GM-BA | 0.61 | 0.24 | 0.12 |
| Mahalanobis-BA | 0.67 | 0.29 | 0.13 |

\* Excluding the 10 replications with the largest outcome SD.

Paired |error| differences, GM-BA minus competitor, on the 90 calmer replications:

| Competitor | Difference |
|---|---|
| Fed-AIPW-logistic | −0.050 ± 0.021 |
| Fed-AIPW-NN | −0.030 ± 0.020 |
| Meta-AIPW | +0.027 ± 0.020 |
| Pooled AIPW | +0.021 ± 0.019 |

On the 10 extreme replications, GM-BA is the worst of these (mean |err| 1.25 vs
0.81–1.00). Fed-AIPW's −0.2 bias comes from its single linear outcome model;
Meta-SW's per-site models follow the curvature better.

**Cross-machine.** The PS configs ran on the Mac (`results/ihdp_att_mac`) and on
Narval:

- fed_logistic differs by up to 0.009 in τ̂. The authors' `MW_estimation_pooled`
  lbfgs fit does not converge on IHDP's unscaled covariates (sklearn
  ConvergenceWarning), so its stopping point depends on BLAS.
- fed_nn differs by up to 0.5% (torch); the others by < 1e-4.
- Negligible against the errors above (SD ≥ 0.36).

---

## Compatibility notes (no effect on results)

- `MW_estimation_pooled` passes `multi_class="multinomial"`, removed in scikit-learn 1.8
  → `environment.yml` pins `scikit-learn<1.8` (`e53e2a5`).
- `generate_multivariate_mixtures_of_normals` (DGP B) uses `np.NaN` (l. 451, 453),
  removed in NumPy 2 → alias restored in `src/estimators/_authors.py` (`6de3c74`).
- `MW_estimation(..., scale=True)` fits Θ on unscaled X and predicts on scaled X; we
  never pass `scale=True`.
