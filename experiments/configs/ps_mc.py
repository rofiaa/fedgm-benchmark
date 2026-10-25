"""Propensity-score-aggregation Monte Carlo: Fig. 2's six scenarios
(DGP A / DGP B × no / poor / good local overlap; paper's App. D values, see
src/dgps/authors.py), federated and Meta-SW estimators.

Inference: IF SE on every rep; site-stratified bootstrap (B = 200) on reps 0-29 for the
IPW-type estimators only (AIPW uses the IF SE as its primary SE). FedAvg-MW configs are
IF-only: their bootstrap would refit with the pooled MW and duplicate fed_logistic.

    python experiments/run.py experiments/configs/ps_mc.py --reps 0-99 --jobs 7 \
        --out results/ps_mc
    python experiments/summarize.py results/ps_mc
"""

from functools import partial

from experiments.harness import Scenario, multi
from src.dgps.authors import OVERLAPS, dgp_a_params, dgp_b_params, generate, true_ate
from src.estimators import fed_ps, meta_sw

BASE_SEED = 20261025
BOOT_B = fed_ps.BOOTSTRAP_B
BOOT_REPS = range(0, 30)


def _scenario(dgp, overlap):
    make = dgp_a_params if dgp == "A" else dgp_b_params
    return Scenario(
        name=f"{dgp}_{overlap}",
        generate=lambda seed: generate(make(overlap), dgp, seed)[1],
        truth=lambda n, seed: true_ate(make(overlap), dgp, n, seed),
    )


SCENARIOS = [_scenario(dgp, overlap) for dgp in ("A", "B") for overlap in OVERLAPS]

FED = ["IPW-HT", "IPW-Hajek_pooled", "AIPW"]
FED_IPW = ["IPW-HT", "IPW-Hajek_pooled"]
META = ["IPW-HT", "IPW-Hajek", "AIPW"]
META_IPW = ["IPW-HT", "IPW-Hajek"]

boot = dict(bootstrap=BOOT_B, bootstrap_reps=BOOT_REPS)
CONFIGS = [
    multi("fed_logistic", fed_ps.estimate_multi, FED, mw="logistic",
          bootstrap_for=FED_IPW, **boot),
    multi("fed_fedavg", fed_ps.estimate_multi, FED, mw="logistic_fedavg"),
    multi("fed_nn", fed_ps.estimate_multi, FED, mw="nn", bootstrap_for=FED_IPW, **boot),
    multi("fed_logistic_rfps", fed_ps.estimate_multi, FED, mw="logistic", local_ps="rf",
          bootstrap_for=FED_IPW, **boot),
    multi("meta_logistic", meta_sw.estimate_multi, META, bootstrap_for=META_IPW, **boot),
    multi("meta_rfps", meta_sw.estimate_multi, META, local_ps="rf",
          bootstrap_for=META_IPW, **boot),
]
