# E2b: what FedDSM's messages reveal, IST single-arm (fixed before any run, 2026-11-16)

Companion to E2 (Fed-GenMatch, findings N18): the same data and attacker, applied to the
releases of FedDSM as run (v0; `src/estimators/feddsm.py`, `docs/feddsm_protocol.md`), so the
two methods are compared on what they actually send.

## Setting

IST single-arm datasets 0–9 (all ≈ 238 treated at the trial site, 2,765 controls at 9 country
sites), FedDSM with the settings of the IST runs (ps = "x", ATT, 5 folds, the config seed of
`results/ist_single_arm`). Under Bret's assumptions the trial site's data are public, so only
the control sites' releases are attacked. Non-members are IST patients outside the dataset's
3,000 (the loader's sampling is reproduced exactly); half of them are the attacker's reference
sample (population knowledge), half the non-member pool. The attacker knows the target's
record (covariates and, for releases involving outcomes, its 6-month death).

## Releases attacked (control sites)

1. **Propensity fit (distributed Newton), final iterate:** per site, the Hessian sum
   Σ p(1−p) x̃x̃ᵀ and the gradient sum −Σ p x̃ (x̃ = (x, 1)); members = the site's controls.
   Earlier iterates are ignored (the attack is a lower bound on what Newton reveals).
2. **Prognostic sums:** per site and fold, B′B and B′Y on FedDSM's basis (covariates, squares
   and products of the continuous covariates, intercept; 38 columns on IST); members = the
   cell's patients (18–232).
3. **Scores sent to the matcher:** each control's (logit ê, m̂₀).

## Attacks and metrics

- Releases 1–2: the E1 tracing attack (`src/privacy/mia.py`, whitened on the reference
  sample) on the release's own feature map. Per release: AUROC, TPR at 1% FPR (threshold from
  the non-member pool), D_eff, n, and E1's Gaussian prediction Φ(√(D_eff/2n)).
- Release 3, two threats:
  - *linkage*: an attacker holding the site's candidate records computes their scores (it
    knows the fitted models; for cross-fitted m̂₀ it takes the closest of the 5 fold fits)
    and links each released pair to the nearest record (standardized score space); share
    linked correctly (E2's re-identification);
  - *attribute inference* without records: the 25 nearest reference patients in score space
    predict the covariates (majority for binary, mean for continuous); binary accuracy vs the
    majority guess, standardized error vs guessing the mean.

No thresholds; descriptive, side by side with E2's numbers for Fed-GenMatch.

Outputs: `results/privacy/e2b_feddsm/rows.csv`, `scores.csv`.
