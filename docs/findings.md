# Findings on the authors' code and paper

Issues found while wrapping Khellaf, Bellet & Josse (arXiv:2505.17961v4) and its code
(`third_party/FedCausal-ObservationalStudies`, commit `e5f4409`). The submodule is never
edited; each entry says how our wrapper (`src/estimators/fed_ps.py`) handles it.
Line numbers refer to `FedCaus_Obs/scripts/DGP_Fed.py` unless stated.

| # | Finding | Affects paper results? | Our handling |
|---|---|---|---|
| F1 | Site ordering breaks for K ≥ 10 | Possibly F.1 / Table 1 (K = 30); not in repo | Columns reordered |
| F2 | Site-wise Hajek IPW is inconsistent | No (paper/notebook use HT) | `Hajek_pooled` added; `Hajek` parity-only |
| F3 | DGP A/B parameters: notebook ≠ paper | Fig. 2 true ATE | Paper values default, notebook as variants |
| F4 | FedAvg with E > 1 converges to a drifted fit | Notebook Fig. 2a AIPW uses E = 3, T = 30 | Default E = 1, standardized X, stopping rule |

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

**Does it affect the paper?** No: every `estimate_ipw` call in
`notebooks/simulations.ipynb` (Fig. 2, all overlap levels and both DGPs: cells 7, 15,
25, 34, 43, 51; Fig. 3: cell 59) uses `ht_or_hajek="HT"`. `estimate_aipw` has no such
option.

**Our handling.** `ipw_type="Hajek_pooled"` (`d321486`); `"Hajek"` reproduces the
authors' estimator and is used only in parity tests. HT is primary in the benchmark.
Tests: `tests/test_fed_ps_parity.py::test_ipw_mw_logistic_matches_authors[*-Hajek]`,
`tests/test_fed_ps.py::test_hajek_pooled_normalizes_over_all_sites`,
`tests/test_fed_ps.py::test_hajek_terms_average_to_tau`.

---

## F3. DGP parameters differ between the notebook and the paper

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

**Evidence (μ₃).** Monte Carlo true ATE (10⁶ draws, MC SE ≈ 0.007) for DGP A:
μ₃ = (3,3,3) → 9.776; μ₃ = (2,2,2) → 8.456. Fig. 2's true-ATE line for DGP A is at
≈ 8.7 (read off the rendered figure), so the figure was made with something close to
the paper's μ₃, not the notebook's. A gap of ≈ 0.25 remains unexplained (other
parameter differences?). The level of the estimates matches better than the line:
Fed-MW (logistic) IPW median ≈ 9.25 in Fig. 2b vs 9.34 in our MC (`d321486`).
DGP B: our true ATE 4.831 vs the figure's ≈ 4.78 and the notebook's hard-coded
`tau_star=4.74` (cell 37).

**Our handling.** Defaults follow the paper; `dgp_a_notebook_params()` /
`dgp_b_notebook_params()` and the constants above give the notebook's values
(`d321486`). Parity tests run on the notebook's settings
(`tests/conftest.py::dgp_a_poor`); the mini MC runs both μ₃ (`A`, `A_mu3nb`).

---

## F4. FedAvg membership weights with E > 1 converge to a drifted fit

**What.** `federated_logistic_regression` (l. 291–317) is FedAvg with E full-batch local
steps per round. Each site holds a single class (H = k), so local steps pull Θ toward
"everything is my site"; with E > 1 the fixed point differs from the pooled
multinomial-logistic optimum (client drift), by an amount that shrinks with η·E.
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

## Compatibility notes (no effect on results)

- `MW_estimation_pooled` passes `multi_class="multinomial"`, removed in scikit-learn 1.8
  → `environment.yml` pins `scikit-learn<1.8` (`e53e2a5`).
- `generate_multivariate_mixtures_of_normals` (DGP B) uses `np.NaN` (l. 451, 453),
  removed in NumPy 2 → alias restored in `src/estimators/_authors.py` (`6de3c74`).
- `MW_estimation(..., scale=True)` fits Θ on unscaled X and predicts on scaled X; we
  never pass `scale=True`.
