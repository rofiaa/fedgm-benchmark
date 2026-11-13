"""IST semi-synthetic benchmark (src/dgps/ist.py; design fixed in docs/ist_design.md), ATE.

Scenarios obs_good / obs_poor / obs_site, replications 0-99 (n = 3,000 patients each,
9 country sites). Federated: FedDSM (ps="x" and ps="site"), the paper's Fed-IPW /
Fed-AIPW (FedAvg and NN membership), Meta-SW. Centralized (with the country dummies, as
IHDP): pooled PS, DSM-RF / AIPW RF + RF, GenMatch. Accuracy only (no bootstrap; FedDSM
has no SE yet).

    python experiments/run.py experiments/configs/ist.py --reps 0-99 --jobs 4 \
        --out results/ist --configs feddsm_x feddsm_site fed_fedavg fed_nn meta_logistic \
        pooled_logistic flex_ref
GenMatch (gm_xps) runs on Narval (experiments/slurm/array.sh ... obs_good obs_poor obs_site).
"""

from functools import partial

import numpy as np

from experiments.harness import Scenario, multi
from src.dgps import ist
from src.estimators import fed_ps, feddsm, flex_outcome, genmatch, meta_sw, pooled_ps

BASE_SEED = 20261113
TARGET = "ATE"
OBS = ["obs_good", "obs_poor", "obs_site"]


def scenarios(names, target):
    return [Scenario(name=s, generate=partial(ist.load, scenario=s, target=target),
                     truth=lambda n, seed: (np.nan, np.nan), by_rep=True) for s in names]


def with_sites(estimate_multi):
    """Centralized estimators see the country dummies (observational scenarios)."""
    def run(df, *args, **kwargs):
        return estimate_multi(ist.with_site_dummies(df), *args, **kwargs)
    return run


def configs(target, site_dummies=True, federated=True):
    p = "ATT-" if target == "ATT" else ""
    wrap = with_sites if site_dummies else (lambda f: f)
    fed = [p + "IPW-Hajek_pooled", p + "AIPW"]
    out = [multi("feddsm_x", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target=target, ps="x")]
    if federated:
        out += [
            multi("feddsm_site", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target=target,
                  ps="site"),
            multi("fed_fedavg", fed_ps.estimate_multi, fed, mw="logistic_fedavg"),
            multi("fed_nn", fed_ps.estimate_multi, fed, mw="nn"),
            multi("meta_logistic", meta_sw.estimate_multi, [p + "IPW-Hajek", p + "AIPW"]),
        ]
    out += [
        multi("pooled_logistic", wrap(pooled_ps.estimate_multi), fed),
        multi("flex_ref", wrap(flex_outcome.estimate_multi), ["DSM-RF", "AIPW-RF-rfps"],
              target=target),
        multi("gm_xps", wrap(genmatch.estimate_multi), list(genmatch.ESTIMANDS),
              covariates="X+ps", balance_quad=["X1", "X2", "X3"], target=target),
    ]
    return out


SCENARIOS = scenarios(OBS, TARGET)
CONFIGS = configs(TARGET)
