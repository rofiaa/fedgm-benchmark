# IST semi-synthetic benchmark: design (fixed before any run)

Data: International Stroke Trial database v2 (Sandercock, Niewada & Czlonkowska 2011,
doi:10.7488/ds/104, ODC-By), `setup/download_ist.sh`. Covariate list follows Bret
Nestor's `preprocess_IST` (2026-11-12), with the changes below.

## Why semi-synthetic

The randomised aspirin effect on 6-month death is -0.9 pp (SE 0.6): too noisy to rank
estimators, and the trial has no confounding. So we keep IST's real patients,
covariates, countries and **real 6-month death as Y(0)**, simulate treatment from a
known propensity score, and simulate Y(1) from Y(0) with a known effect. The truth is
exact, and the outcome-covariate relation is real (no model of ours generates it).

## Population (`src/dgps/ist.population()`)

- Drop the pilot phase (984 patients; `RATRIAL`/`RASP3` not recorded), patients with
  6-month status unknown (`FDEAD` missing or "U"), and keep the 9 countries with
  >= 500 remaining patients (UK, Italy, Switzerland, Poland, Netherlands, Sweden,
  Australia, Norway, Argentina): **14,231 patients**. Sites H = 0..8, largest first.
- 31 covariates X1..X31: AGE, RSBP, RDELAY (continuous, X1..X3); SEX (F = 1), RSLEEP,
  RATRIAL, RCT, RVISINF, RHEP24, RASP3; RCONSC drowsy / unconscious (alert = base);
  STYPE LACS / POCS / TACS (PACS = base; the 47 "OTH" join the base); RDEF1..8 as
  "yes" and "can't assess" indicators ("no" = base).
- Not used: `DASP14`/`DASPLT` (post-randomisation; `DASPLT` exists only for survivors to
  discharge), `DAYLOCAL`, the randomised arms (`RXASP`, `RXHEP`).
- Y(0) = `FDEAD` (1 = dead at 6 months).

## Replication r of scenario s (`load(rep, scenario, target)`)

1. Subsample n = 3,000 patients without replacement (all from the 14,231).
2. Treatment: e(x, h) = expit(c + gamma * s(x) + delta_h), with s(x) a fixed linear
   index (standardized over the population, sd 1) that makes treatment more likely in
   younger, alert, less severe patients and has an age^2 term (so a linear PS model is
   misspecified):
   s_raw = -0.5 z_age + 0.3 (z_age^2 - 1) - 0.4 z_drowsy - 0.4 z_unconscious - 0.4 z_TACS
           + 0.3 z_SBP - 0.3 z_delay + 0.3 z_infarct_visible - 0.3 z_AF + 0.2 z_prior_aspirin.
   c solves for the population treated share.
3. Effect: a death is prevented with prob q(x) = expit(-1.6 + 0.4 z_age + 0.5 TACS);
   a survivor dies with prob r(x) = expit(-4.0 + 0.5 z_SBP + 0.4 z_age) (bleeding).
   Y(1) drawn from Y(0) accordingly; Y = W Y(1) + (1 - W) Y(0).
4. Truth: tau_i = r(x_i)(1 - Y0_i) - q(x_i) Y0_i (exact given the data);
   ATE = mean over the sample, ATT = mean over the treated. Benefit is larger in old /
   TACS patients, who are treated less, so ATT is closer to 0 than ATE (20-rep means:
   ATE -4.1 pp in every scenario; ATT -2.7 / -1.1 / -1.4 / -0.9 pp in obs_good /
   obs_poor / obs_site / single_arm; naive difference -10 to -17 pp).

| Scenario | gamma | delta_h | Treated share | Estimands | Notes |
|---|---|---|---|---|---|
| `obs_good` | 0.6 | 0 | 40% | ATE, ATT | site not a confounder |
| `obs_poor` | 1.8 | 0 | 40% | ATE, ATT | poor overlap |
| `obs_site` | 1.2 | -1.5..+1.5 | 40% | ATE, ATT | low-mortality countries treat more: **site is a confounder** (country mortality 13-30% beyond case mix) |
| `single_arm` | 1.2 | 0 | 8% | ATT | Bret's setting: treated form one extra "trial" site (H = 9); the 9 country sites hold controls only |

## Estimators

- FedDSM, ps="x" (all) and ps="site" (obs scenarios; undefined in single_arm by design).
- Khellaf et al. Fed-IPW / Fed-AIPW (FedAvg and NN membership) and Meta-SW: obs
  scenarios only (no site has both arms in single_arm).
- Centralized: pooled PS (IPW, AIPW), DSM-RF and AIPW RF + RF (flex_outcome), and
  GenMatch (X + PS logit; balance matrix X plus squares/products of X1..X3; Narval only).
  Centralized methods get the country dummies in obs scenarios (`with_site_dummies`),
  as IHDP; in single_arm they would identify the treated, so not there.
- Accuracy only (FedDSM has no SE yet; no bootstrap): label tables "accuracy only".

## Reps and reading rule

100 reps per scenario (GenMatch: 30, Narval). Primary: RMSE and bias vs the exact
truth, paired by rep. The question for the supervisors: does centralized GenMatch beat
FedDSM on IST (any scenario, paired RMSE difference > 2 SE)?
