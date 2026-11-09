"""FedDSM verification on ACIC 2016 (src/dgps/acic2016.py), ATE.

All 77 competition settings (treatment model, % treated, overlap, response model,
alignment, heterogeneity), replications 0-19 = the competition's replications 1-20;
4 sites assigned from X (the site is not a confounder, so FedDSM uses the covariate-
only PS, ps="x"). Truth per replication: the noiseless ATE from mu0, mu1.
FedDSM settings fixed beforehand (M = 4, quadratic prognostic model with the
products-only-if-basis <= n_arm/2 rule, same-model bias adjustment).
Comparators: the paper's Fed-AIPW / Fed-IPW-Hajek with FedAvg and NN membership,
Meta-SW and pooled PS (all with linear outcome models), plus Fed-AIPW with FedDSM's
own models (FedAIPW-quad). Fed-DW is not run (79 covariates incl. dummies).

    python experiments/run.py experiments/configs/feddsm_acic.py --reps 0-19 \
        --jobs 7 --out results/feddsm_acic
"""

from functools import partial

import numpy as np

from experiments.harness import Scenario, multi
from src.dgps import acic2016
from src.estimators import fed_ps, feddsm, meta_sw, pooled_ps

BASE_SEED = 20261109

SCENARIOS = [Scenario(name=f"acic{s:02d}", generate=partial(acic2016.load, setting=s, target="ATE"),
                      truth=lambda n, seed: (np.nan, np.nan), by_rep=True)
             for s in range(1, acic2016.N_SETTINGS + 1)]

FED = ["IPW-Hajek_pooled", "AIPW"]
META = ["IPW-Hajek", "AIPW"]
CONFIGS = [
    multi("feddsm_x", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target="ATE", ps="x"),
    multi("fed_fedavg", fed_ps.estimate_multi, FED, mw="logistic_fedavg"),
    multi("fed_nn", fed_ps.estimate_multi, FED, mw="nn"),
    multi("meta_logistic", meta_sw.estimate_multi, META),
    multi("pooled_logistic", pooled_ps.estimate_multi, FED),
]
