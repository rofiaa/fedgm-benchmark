"""IHDP benchmark: centralized GenMatch vs federated / Meta-SW / pooled PS on the 100
IHDP replications (src/dgps/ihdp.py), sites = the 8 IHDP sites. Estimand: the sample
ATE of each replication (df.attrs["tau_true"]).

Information sets: the federated and Meta-SW estimators get X = x1..x18 and the site as
H; the centralized ones (pooled PS, GenMatch) get x1..x25, i.e. the site as 7 dummies.
GenMatch's balance matrix follows genmatch-lalonde's IHDP run: the 25 covariates plus
the squares and pairwise products of the 6 continuous ones.
Fed-DW is not run: its Gaussian site densities are singular with IHDP's binary
covariates (constant within some sites).

Inference: IF SE on every rep; site-stratified bootstrap (B = 200) on reps 0-29 for the
IPW-type estimators; Abadie-Imbens SE for matching. The design (X, W) is the same in
every replication, so the spread across replications is outcome noise only.

    python experiments/run.py experiments/configs/ihdp_mc.py --reps 0-99 --jobs 7 \
        --out results/ihdp_mc
"""

import numpy as np

from experiments.harness import Scenario, multi
from src.dgps import ihdp
from src.estimators import fed_ps, genmatch, meta_sw, pooled_ps

BASE_SEED = 20261105
BOOT_B = fed_ps.BOOTSTRAP_B
BOOT_REPS = range(0, 30)

SCENARIOS = [Scenario(name="IHDP", generate=ihdp.load,
                      truth=lambda n, seed: (np.nan, np.nan), by_rep=True)]


def _site_as_h(estimate_multi):
    """Run on the frame without the site dummies (the site enters as H)."""
    def run(df, *args, **kwargs):
        return estimate_multi(ihdp.without_site_dummies(df), *args, **kwargs)
    return run


FED = ["IPW-HT", "IPW-Hajek_pooled", "AIPW"]
FED_IPW = ["IPW-HT", "IPW-Hajek_pooled"]
META = ["IPW-HT", "IPW-Hajek", "AIPW"]
META_IPW = ["IPW-HT", "IPW-Hajek"]

boot = dict(bootstrap=BOOT_B, bootstrap_reps=BOOT_REPS)
CONFIGS = [
    multi("fed_logistic", _site_as_h(fed_ps.estimate_multi), FED, mw="logistic",
          bootstrap_for=FED_IPW, **boot),
    multi("fed_fedavg", _site_as_h(fed_ps.estimate_multi), FED, mw="logistic_fedavg"),
    multi("fed_nn", _site_as_h(fed_ps.estimate_multi), FED, mw="nn",
          bootstrap_for=FED_IPW, **boot),
    multi("meta_logistic", _site_as_h(meta_sw.estimate_multi), META,
          bootstrap_for=META_IPW, **boot),
    multi("pooled_logistic", pooled_ps.estimate_multi, FED, bootstrap_for=FED_IPW, **boot),
    multi("gm_xps", genmatch.estimate_multi, list(genmatch.ESTIMANDS), covariates="X+ps",
          balance_quad=[f"X{i}" for i in range(1, 7)]),
]
