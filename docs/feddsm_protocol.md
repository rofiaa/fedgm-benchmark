# FedDSM protocol: threat model, message ledger, disclosure audit (draft v0 → v1)

Status: draft for discussion with the supervisor (2026-11-08). The statistical method is
`src/estimators/feddsm.py`, which simulates the federation centrally: what each party
would see is computed exactly, but no messages are actually exchanged. This document
fixes what the protocol reveals before any privacy attack is run.

Plan: threat model → what each step reveals → attack evaluation → protection →
utility/privacy trade-off.

## 1. Parties, assets, outputs

- **Sites** k = 1..K hold patient records (x, w, y). Our datasets have K = 3 (authors'
  DGPs), 8 (IHDP) or 4 (ACIC).
- **Server:** coordinates and aggregates. It may also run the matching (v0) or take
  part in secure matching (v1).
- **Private (patient level):** covariates, treatment, outcome, the patient's scores
  (logit ê, m̂₀, m̂₁), match identities (who is matched to whom) and distances.
- **Private (site level), if institutional privacy is in scope — decision pending:**
  treatment prevalence, site-specific coefficients, case mix.
- **Public by design:** the final τ̂ (+ SE once inference exists) and a pre-registered
  balance summary.

## 2. Threat model (proposal)

| | Adversary | Default assumption |
|---|---|---|
| T1 | Server | honest-but-curious: follows the protocol, tries to learn from what it sees |
| T2 | Collusion | the server colludes with at most c sites, c < K − 1. With K = 3 this means no collusion at all (c = 0); state this explicitly |
| T3 | Outside observer | sees only the public outputs |
| — | Malicious sites | out of scope for v1 (crafted updates); revisit later |

Two tiers to evaluate:
- **Tier A (cryptographic):** secure aggregation and secure matching; released
  statistics are exact (no utility loss, no formal guarantee for the outputs).
- **Tier B (A + DP):** differential privacy on released statistics and outputs; formal
  (ε, δ), with utility measured.

## 3. Message ledger of the current pipeline (v0)

Numbers sent per site, from `experiments/checks/feddsm_disclosure_audit.py`, replication 0:

| Step | What a site sends (v0) | Authors' DGPs | IHDP | ACIC 2016 |
|---|---|---|---|---|
| S1 PS, distributed Newton | each iteration: gradient (p) and Hessian (p(p+1)/2) at the current β; a log-likelihood value per trial step | 5–6 iterations × 77 = 385–462 | 8 × 377 = 3,016 | 15 × 3,320 = 49,800 |
| S2 prognostic model | for each arm and each of 5 folds: B′B (q(q+1)/2) and B′Y (q) | q = 66: 11,385 per arm | q = 47: 5,875 per arm | control q = 1,355: **4.6 million**; treated q = 130: 43,225 |
| S3 scores for cross-site matching | each patient's (logit ê, m̂₀ [, m̂₁]) and treatment | n_k × 3–4 values | same | same |
| S4 Mahalanobis scaling | the covariance of the scores (computed centrally in v0) | aggregate | aggregate | aggregate |
| S5 matching | done by whoever holds all scores (v0: central) | match identities, distances | | |
| S6 estimate | v0 code: outcomes of matched pairs (pair differences) | individual outcomes | | |
| S7 balance | weighted covariate means by arm | aggregates | | |

**Addition if score="tuned" is adopted (docs/findings.md N12):** one more number per
(site, arm, fold) cell, Y′Y, and the full product basis for both arms (B′B of up to
q = 1,355 columns also for the treated arm on ACIC). Y′Y with n = 1 gives |Y|. Not yet
audited.

## 4. What v0 discloses (audit results)

| ID | Disclosure | Evidence |
|---|---|---|
| D1 | **Exact disclosure in 1- and 2-patient cells.** For n = 1, the intercept row of B′B is the patient's covariate vector and B′Y's intercept entry is the outcome. For n = 2, Σx and Σxx′ give x₁ − x₂ up to sign. | IHDP rep 0, treated arm: 10 cells with n = 1 and 7 with n = 2. **All 17 reconstructed from the sums alone, max error 6e-15.** None on the authors' DGPs or ACIC (smallest cells 8 and 37). |
| D2 | **More shared numbers than patient values.** When a cell has fewer patients than model terms, its B′B (moments up to order 4) has many more entries than the cell has unknowns. Reconstruction may be possible beyond tiny cells: likely, not yet shown. | Cells smaller than the basis: ACIC every cell (e.g. 186 controls vs 1,355 terms: ≈ 920k numbers for ≈ 15k unknowns); IHDP all control cells; authors' DGP 1–15 cells per arm. |
| D3 | **Differencing.** Fold sums and full-arm fits, and per-site vs global sums, can be subtracted from each other to isolate groups. | structural |
| D4 | **Repeated adaptive views in Newton:** the per-site gradient and Hessian at 5–15 different β. | structural |
| D5 | **Patient-level scores and treatment** pass to the matcher. | structural (v0) |
| D6 | ~~**Matched-pair outcomes** cross sites in the pair-difference form.~~ **Resolved:** `feddsm._local_tau` computes the outcome-local form (§5 item 5). It equals the pair form exactly (`tests/test_feddsm.py`), and stored results reproduce to ≤ 1.4e-14. | fixed 2026-11-08 |
| D7 | **Exact released aggregates** enable membership inference. | structural |

## 5. Proposed v1 (tier A)

1. **Secure aggregation** of S1 and S2: the server sees only sums over sites. With K = 3
   this protects only against the server alone (T2 with c = 0).
2. **Folds defined globally,** so that each released (arm, fold) aggregate pools at
   least n_min patients across sites. n_min is an operational floor, not a guarantee.
3. **Smaller prognostic sufficient statistics on wide data:** ridge or a reduced basis,
   or compute the regression inside MPC and release only coefficients. On ACIC this
   reduces 4.6 million numbers per site (communication and D2).
4. **Fewer PS rounds:** compare exact Newton (5–15 rounds) with one-shot ODAL/ODAL2 and
   with DP logistic regression.
5. **Outcome-local estimator.** It is algebraically identical to the current one:
   τ̂_ATT = (1/n₁)[Σ_treated (Yᵢ − m̂₀(Xᵢ)) − Σ_controls K_j (Y_j − m̂₀(X_j))], where K_j is
   control j's total match weight. Each site learns only its own K_j, computes local
   sums, and they are securely aggregated. Outcomes never leave a site.
6. **Secure top-M matching** on secret-shared or encrypted 2–3-dimensional scores (cf.
   FedKNN). Scores, distances and match identities never in plaintext.
7. **Fixed, pre-registered queries:** each statistic released once; a fixed set of
   balance diagnostics.

Tier B adds DP:
- **Sufficient-statistic perturbation** for the prognostic model. Disjoint folds
  compose in parallel, so all fold aggregates cost one ε.
- **DP logistic regression** for the PS.
- **A DP release** of τ̂ and the balance summary.

## 6. Evaluation plan

1. Verify that v1 (tier A) gives the same estimates as v0, except the PS with ODAL,
   whose utility loss is measured.
2. **Attacks on v1 releases:**
   - reconstruction from the remaining aggregates (D2-type moment attacks, differencing);
   - membership inference on the released coefficients and τ̂ (shadow models; AUC,
     advantage);
   - attribute inference from anything score-derived that becomes visible.
3. **Tier B:** RMSE, coverage and balance against ε; privacy budget accounting.
4. **Communication:** numbers and rounds per site for every step, as in §3.

## 7. Decisions needed

- K and c for T2; whether institutional privacy is in scope.
- Whether tier B (formal DP) is required or tier A suffices for the target venue.
- Whether to use a reduced prognostic basis on wide data (affects D2, communication,
  and possibly accuracy; a post-hoc change to be reported as such).
