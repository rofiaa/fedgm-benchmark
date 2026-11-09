# Evaluation metrics: what the literature reports, and what we cover

A living log (started 2026-11-08). Each entry was read from the paper's own text, not
recalled from memory. Markers:

- **[V]** read verbatim from the paper;
- **[S]** only seen in a tool-generated summary of a section that could not be read
  verbatim (re-check before citing);
- **[N/A]** source not accessible from here.

Add papers as they are read.

## 1. Metrics reported by related papers

| Paper | Setting | Accuracy metrics | Inference | Balance / overlap | Compute / communication | Privacy |
|---|---|---|---|---|---|---|
| Khellaf, Bellet & Josse (arXiv:2505.17961) | federated PS aggregation (Fed-IPW/AIPW), ATE | bias and MSE tables, App. F Tables 5–7 [V]; boxplots in Figs 2–3 [S]; 800 runs [S] | asymptotic variance theorems [V]; bootstrap CIs (150 resamples) on Traumabase [S]; **no coverage reported** [V, in the parts read] | PS-distribution overlap plots (App. E) [V]; no covariate balance | FedAvg rounds T as a parameter [V]; no measured cost | none (no metric) [V] |
| Antonelli et al. 2018, *Biometrics* (arXiv:1612.00424) | double-score matching, ATE (and ATT) | absolute bias, SD, MSE over 1000 reps (Table 1) [V] | approximate SE (Eq. 13) that "underestimates the true standard error" [V]; coverage [S] | standardized differences [S] | — | — |
| Zhang, Yang, Ye, Faries, Lipkovich, Kadziola 2022, *Stat Med* (PMC8918069) | DSM practice, ATT only | bias and variance as boxplots; matching rate [V] | **none**: asymptotics only for known scores [V] | qualitative (within-pair distance) [V] | — | notes that DSM "require[s] outcome information" [V] |
| Yang & Zhang 2023, *Scand J Stat* (arXiv:2001.06049) | multiply robust DSM, ATE/QTE/ATT | error violin plots [S] | adjusted asymptotic variance for estimated scores (Thm 2) [V]; weighted bootstrap [V]; coverage table [S] | SMDs in application [S] | — | — |
| Dorie et al. 2019, *Stat Sci* (ACIC 2016, arXiv:1707.02641) | competition, SATT | bias, RMSE [S] (motivated verbatim [V]); PEHE for individual effects [S] | coverage, interval length [S] | dataset descriptors: overlap, imbalance, alignment, heterogeneity [V] | compute time [S] | — |
| Xiong et al. 2023, *Stat Med* (arXiv:2107.11732) | federated MLE / IPW-MLE / AIPW, one-shot summary statistics | simulation metrics **not verified** | sandwich variances; federated variance by sample-size weighting [V] | — | "one-way and one-time sharing" [V] | motivation only [V] |
| Han, Shen & Zubizarreta 2023 (arXiv:2309.12600) | multiply robust federated target ATE | MAE, RMSE over 500 sims (Tables 1–2) [V] | coverage and CI length [V]; closed-form variance | — | one round for the density-ratio step [V] | claim only, no metric [V] |
| Koga et al. 2023 (arXiv:2310.06237) | DP multi-site ATE; exact matching within sites | MAE vs the non-private estimate and the truth [S] | DP variance [S] | — | — | (ε, δ)-DP; Laplace + smooth sensitivity [V]; utility vs privacy-budget ratio [S] |
| Li, …, Toh 2019, *AJE* (PMC6438804) | distributed networks; PS *or* disease-risk-score matching within sites | agreement with the pooled individual-level analysis (estimates and CIs) [V] | meta-analytic and risk-set CIs [V] | — | what each site shares [V] | aggregate-only sharing (no metric) [V] |
| Diamond & Sekhon 2013, *REStat* 95(3):932–945 (GenMatch) [V, PDF supplied by the user] | genetic matching; ATT, 1:1 with replacement, ties averaged (Sec. II.B) | Sim 1 (Setoguchi/Lee: 10 covariates, 7 PS scenarios A–G of increasing nonlinearity/non-additivity, linear outcome, n = 1,000 and 5,000, 1,000 datasets): absolute bias (% of the true effect) and RMSE (Table 1). Sim 2 (LaLonde-based: 185 treated / 260 controls, effect $1,000, outcome nonlinear in re74/re75, misspecified PS): bias %, RMSE and ratios to GenMatch (Table 2; GenMatch 4.32% / RMSE 512 vs logit PS matching 51.2% / 1,832). LaLonde: ATT vs the experimental benchmark (Table 3) | 95% CIs in the empirical example only (Table 3); none in the simulations | boxplots of the smallest p-value of paired t- and KS tests (Figs 3–4); fitness = lowest p-value; mentions SMD, eQQ, variance ratios | — | — |
| Liu et al. 2025, *Ann Oper Res* (FPSM) | federated PS matching | [N/A]: paywalled | | | | |

## 2. Metrics the literature uses, and our coverage

| Metric | Used by | Our benchmark |
|---|---|---|
| Bias (± Monte Carlo SE) | all | ✓ all methods |
| RMSE / MSE | Khellaf, Antonelli, ACIC, Han | ✓ all |
| MAE, median \|error\| | Han, Koga | ✓ all |
| Empirical SD | Antonelli | ✓ all |
| SE / SD ratio | (standard) | ✓ PS methods, GenMatch; **✗ FedDSM** |
| 95% CI coverage | ACIC, Han, Yang & Zhang | ✓ PS methods (influence-function SE; bootstrap on reps 0–29), GenMatch (Abadie–Imbens SE); **✗ FedDSM (no SE yet)** |
| CI width | ACIC, Han | as for coverage |
| Covariate balance (max \|SMD\|) | Antonelli, Yang & Zhang, GenMatch | ✓ all estimators (weighted or matched sample) |
| KS statistic, variance ratio | Diamond & Sekhon (KS p-values; variance ratios mentioned) | ✗ (GenMatch's fitness uses KS internally) |
| Effective sample size | — | partly (GenMatch n_eff) |
| Overlap diagnostics (PS distributions) | Khellaf | ✗ (scenarios are labelled by overlap level) |
| Robustness across settings (overlap, nonlinearity, alignment) | ACIC | partly (3 overlap levels × 2 DGPs; IHDP) → ACIC 2016 next |
| PEHE (individual effects) | ACIC | n/a (ATE/ATT only) |
| Agreement with pooled analysis | Li et al. | ✓ implicitly (federated fits verified equal to pooled fits) |
| Runtime | ACIC | ✓ (config runtime) |
| Communication rounds / bytes | Khellaf (rounds as parameter), Xiong (one-shot) | partly (FedAvg rounds); **✗ bytes; FedDSM's messages not yet counted** |
| Privacy: (ε, δ), utility vs ε | Koga | ✗ planned (tier B) |
| Privacy: reconstruction error, membership-inference AUC | none of the causal papers above | ✗ planned; deterministic n ≤ 2 disclosure already shown |

## 3. Gaps to close before a final FedDSM decision

1. FedDSM inference (SE, coverage, CI width).
2. Robustness across ACIC 2016 settings (misspecified outcome models).
3. Communication accounting (rounds, numbers sent per site).
4. Privacy metrics under a stated threat model.
5. Read FPSM in full when a readable copy is available.

## 4. Notes from Diamond & Sekhon (2013) relevant to our results

- **Comparators were propensity-score matching variants only:** fixed logit, random
  forest and boosted CART PS matching (Table 1–2). There was no weighting, no outcome
  regression and no doubly robust estimator.
  - GenMatch's published advantage is over PS matching.
  - Our benchmark compares it with doubly robust federated weighting estimators that
    have outcome models, a comparison the paper never made.
- **Simulation 1's outcome is linear in X,** where matching discrepancy is mild.
- **The authors say GenMatch "may or may not decrease the bias"** (Sec. II.B), that
  there is "no claim that it is generally the best", and that it is "unclear how to
  best measure the degree of covariate balance" (Sec. V). Their Fig. 5 shows reliable
  LaLonde estimates "only at the highest fitness values".
- **Their own LaLonde results vary a lot** (Table 3):

  | Sample | GenMatch | Benchmark | Balance (fitness) |
  |---|---|---|---|
  | DW, CPS-1 | $1,734 | $1,794 | 0.21 |
  | DW, PSID-1 | $1,045 | $1,794 | 0.029 |
  | Early RA, CPS-1 | $1,631 | $2,748 | 0.46 |
  | LaLonde sample, CPS-1 | $281 | $886 | 0.23 |

  This matches our N6: an accurate result on CPS-1 is not a general advantage.
- **Our genmatch-lalonde replication:**

  | Sample | Ours | D&S |
  |---|---|---|
  | DW, CPS-1 | $1,698, fitness 0.134 | $1,734, fitness 0.21 |
  | DW, PSID-1 | $400, fitness 0.003 | $1,045, fitness 0.029 |

  D&S's search reached better balance on PSID-1.
