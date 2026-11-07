"""IHDP benchmark with the ATT as the estimand: the same datasets, information sets and
nuisance models as ihdp_mc.py (the ATE run), with every method switched to the ATT.
The truth is each replication's ATT, mean(mu1 - mu0) over the 139 treated, which is 4
by Hill's construction.

Why: on the ATE, GenMatch never reaches balance on IHDP (best balance p-value ~1e-6 in
every replication; 608 controls need matches among 139 treated), while on the ATT it
does (p ~ 0.2, as in genmatch-lalonde). This run measures GenMatch where its search
works, against the PS estimators' ATT versions (fed_ps.ESTIMANDS: controls weighted by
the odds ê/(1-ê); Meta-SW averages the site ATTs with treated counts n1_k / n1).

Inference as in ihdp_mc.py: IF SE on every rep; site-stratified bootstrap (B = 200) on
reps 0-29 for the IPW-type estimators; Abadie-Imbens SE for matching.

    python experiments/run.py experiments/configs/ihdp_att.py --reps 0-99 --jobs 7 \
        --out results/ihdp_att --configs fed_logistic fed_fedavg fed_nn meta_logistic \
        pooled_logistic
GenMatch (gm_xps) runs on Narval (experiments/slurm/array.sh ... IHDP_ATT).
"""

from functools import partial

import numpy as np

from experiments.harness import Scenario, multi
from src.dgps import ihdp
from src.estimators import fed_ps, genmatch, meta_sw, pooled_ps

BASE_SEED = 20261107
BOOT_B = fed_ps.BOOTSTRAP_B
BOOT_REPS = range(0, 30)

SCENARIOS = [Scenario(name="IHDP_ATT", generate=partial(ihdp.load, target="ATT"),
                      truth=lambda n, seed: (np.nan, np.nan), by_rep=True)]


def _site_as_h(estimate_multi):
    """Run on the frame without the site dummies (the site enters as H)."""
    def run(df, *args, **kwargs):
        return estimate_multi(ihdp.without_site_dummies(df), *args, **kwargs)
    return run


FED = ["ATT-IPW-HT", "ATT-IPW-Hajek_pooled", "ATT-AIPW"]
FED_IPW = ["ATT-IPW-HT", "ATT-IPW-Hajek_pooled"]
META = ["ATT-IPW-HT", "ATT-IPW-Hajek", "ATT-AIPW"]
META_IPW = ["ATT-IPW-HT", "ATT-IPW-Hajek"]

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
          balance_quad=[f"X{i}" for i in range(1, 7)], target="ATT"),
]
