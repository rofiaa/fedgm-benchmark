# Open questions for the next meeting (2026-11-16)

Living list; answers go in docs/findings.md when given.

## For Bret: his Fed-GenMatch draft (Bret_GM_derivation.tex)

Our implementation (`src/estimators/fedgm.py`) matches every equation the draft specifies,
checked by tests (eqs. 2, 3, 6–13, the GA's start at W = I and best-seen return). These points
are open because the draft is silent or inconsistent:

1. **Gene scale of the GA.** BLX-α and the Gaussian mutation act on w in [10⁻², 10²]: on the
   linear or the log scale? (We use log10 w; on a linear scale nearly all candidates land in
   [1, 100].) Mutation probability and size per gene?
2. **Search.** Algorithm 1 is a single-candidate accept-if-better loop; the text describes a
   population GA (tournament, BLX-α, elitism). Which one? (We run the text's GA.)
3. **Messages.** Algorithm 1 sends only the cross terms B⁽ᵏ⁾; the text sends B⁽ᵏ⁾ and the norms
   C⁽ᵏ⁾. (We follow the text; the coordinator needs C to build D.)
4. **Singular S_t.** On IST single-arm, 28 of 30 datasets have binary covariates constant
   among the treated, so S_t is singular and Σ^{-1/2} does not exist. Pseudo-inverse (those
   covariates leave the distance), shrinkage, or something else? (N15: the choice does not
   change accuracy.)
5. **Estimand.** Matching the treated cohort estimates the ATT; the results tables say ATE.
6. **SMD when both variances are 0** (a covariate constant in the treated and in the matched
   controls): we use 0 if the means are equal, ∞ otherwise.
7. **DP variant (eq. 14).**
   - Laplace noise is calibrated to ℓ1 sensitivity; eq. 14 bounds an ℓ2 quantity.
   - One control changes a whole column of B⁽ᵏ⁾ (n_t entries) plus C_j, so per-entry ε is not
     per-patient ε; C_j is quadratic in x and needs a bounded domain.
   - The matched-set moments have no simple sensitivity: one control can reshuffle the optimal
     assignment without replacement.
   - Composition over the search (median 5,000 releases in our runs) needs a number.
7b. **Measured (E2, findings N18, IST single-arm):** with his S_t, one search evaluation
   exposes everything the protocol exposes; 80% of controls are re-identified at n_t = 20 and
   over 90% from n_t = 31. New: the matched-set sums (eq. 9) alone identify 97–99% of the
   controls ever matched exactly, because the coordinator knows the matched sets. His DP at
   ε up to 1,000 per entry keeps essentially no matches. Does he want these in the paper's
   privacy section?
8. **Paper claim.** The abstract compares methods by SMD on long-tailed data. Our results show
   balance and accuracy disagree; we suggest RMSE against a known effect (we can run it).
9. **Notation.** The recovery formula's left side should be the raw control x, not x̃; two
   equation references in Algorithm 1 are broken.
10. **His open questions** (end of the Discussion section): why Σ_t: the choice does not change
    accuracy (N15); removing the cross term: impossible, since it is the only piece that needs
    both sites, so that is where protection must go.

## For Prof. Xiaoxiao (and Bret)

1. In the single-arm setting, is the ATT the target, and what is the smallest treated cohort
   the method must support?
2. Does Fed-GenMatch need 1:1 matching without replacement? With replacement each site could
   return only its nearest distances instead of its full distance matrix.
3. What would Fed-GenMatch need to show to justify developing it (accuracy, robustness to a
   wrong outcome model, privacy–utility)? Is there a setting where it should win?
4. May we test a larger number of matches for FedDSM in the single-arm setting (post-hoc)?
5. Privacy evaluation: is the attacker model (knows the population distribution and the
   target record) acceptable? Which group sizes are realistic (rare-disease cohorts, eICU
   sites), so the E1/E2 grids use them?
6. Threat model decisions (docs/feddsm_protocol.md §7): honest-but-curious server; with 3
   sites no collusion can be tolerated; is a site's case mix private?
