# FedDSM inference plan (draft, 2026-11-08)

**Goal:** a standard error and 95% CI for FedDSM, computable from what sites can share,
with coverage near nominal on the authors' DGPs, IHDP and ACIC 2016. Implementation
starts only if ACIC confirms FedDSM is worth pursuing.

Source markers:
- **[read]** read in this project (docs/evaluation_metrics.md);
- **[mem]** from memory, to verify before citing.

## What makes it hard

- **Standard bootstrap:** invalid for nearest-neighbour matching with a fixed number
  of matches (Abadie & Imbens 2008, Econometrica [mem]); Lin, Ding & Han (2023) cite Otsu & Rai (2017) for a valid bootstrap of a linear form [read].
- **Estimated scores:** the matching variables (ê, m̂) are estimated. Variances that
  treat them as fixed can be wrong. Antonelli et al. (2018): their approximate SE "does
  not account for uncertainty in estimation of the two scores … may lead to
  anti-conservative interval estimates" [read].
- **Theory gaps:** Zhang et al. (2022) give asymptotics only for known scores [read].
  Yang & Zhang (2023) derive an adjusted variance for estimated scores (their Thm 2:
  V_adj = V − γ₁ᵀΣ_U⁻¹γ₁ + γ₂ᵀΣ_θγ₂) and use a weighted bootstrap [read; the
  bootstrap's validity proof not seen].
- **Estimated propensity score, from theory:** for PS matching, ignoring PS estimation
  gives conservative CIs for the ATE but "either too large or too small" ones for the
  ATT (Abadie & Imbens 2016 [read]). FedDSM's ATT coverage must therefore be checked,
  not assumed. (Their result covers PS-only matching; DSM adds the prognostic score.)
- **Our setting:** M = 4 is fixed, and the prognostic score is cross-fitted.

## Candidates

All use the match weights K_i (unit i's total weight as a match, from
`feddsm.match_weights`) and the cross-fitted residuals ε̂_i = Y_i − m̂_{W_i}(X_i).

| # | Method | SE | What sites share (beyond the estimate) | Pros | Cons |
|---|---|---|---|---|---|
| A | **Influence-function plug-in.** The bias-corrected matching estimator is AIPW-like: τ̂ = mean[m̂₁ − m̂₀] + mean[(2W−1)(1 + K_i/M) ε̂_i] (Abadie & Imbens 2011; this exact form is Lin, Ding & Han 2023, eq. 3.3 [read]). Use φ_i = m̂₁ − m̂₀ + (2W−1)(1 + K_i/M) ε̂_i − τ̂ and SE = sd(φ)/√n. ATT analogue: φ_i = [W_i(ε̂₀,i − τ̂) − (1−W_i) K_i ε̂₀,i]/p̂₁. Justified as M grows: with M → ∞ and M/n^γ → 0 this estimator is doubly robust and semiparametrically efficient, and their σ̂² is consistent (Lin, Ding & Han 2023, Thm 4.1 [read]); they call choosing M in finite samples "challenging" and compare σ̂² with Abadie–Imbens' fixed-M variance by simulation | sd(φ)/√n | per site: Σφ, Σφ², n_k (3 numbers) | same form as the IF SE of our other estimators; trivially federable; cheap | ignores score estimation; fixed M = 4 may make it anti-conservative |
| B | **Abadie–Imbens (2006) variance** with σ²(x) estimated by matching within arms on the scores [mem] | closed form | per site: Σ (1 + K_i/M)² σ̂²_i and Σ (m̂₁ − m̂₀ − τ̂)² | the standard matching variance | also treats the scores as fixed; σ̂² needs within-arm matching (a second protocol step) |
| C | **Multiplier (weighted) bootstrap** of the linear representation in A: each site draws its own multipliers ξ_i and sends Σ ξ_i φ_i for each replicate b (cf. Otsu & Rai 2017 [mem]; Yang & Zhang's weighted bootstrap [read]) | SD over B replicates | B numbers per site | no matching re-run; federable | B releases of exact sums (a privacy query channel); same blind spot as A unless φ includes score-estimation terms |
| D | **Yang & Zhang adjusted variance:** A or B plus correction terms for estimating the PS and prognostic models (γ₁, γ₂, Σ_θ) | closed form | the GLM pieces (scores and Hessians), already aggregates in our fits | accounts for estimated scores; principled | most work; derived for their estimator (Euclidean distance, de-biased), needs adapting to ours |

## Recommendation

1. **Implement A** (cheap) and C (as a check on A). Measure coverage and SE/SD on:
   - the authors' DGPs (6 scenarios × 100 reps);
   - IHDP (100);
   - ACIC (77 settings × 20).
2. **If A undercovers,** for example because estimated scores make the SE too small,
   **implement D.**
3. **Report inference like the other estimators:** coverage, CI width and SE/SD, with
   bootstrap coverage on replications 0–29 where applicable.
4. **Protocol:** A adds 3 aggregates per site; C adds B aggregates (to be counted in the
   protocol's query ledger, and DP-protected in tier B).
