"""FedDSM verification on the authors' DGPs, ATT.

Same datasets as ps_mc.py / feddsm_authors.py (same scenario names and BASE_SEED);
the truth is the population ATT (src.dgps.authors.true_att, mean ITE* over the treated
of a 10^6 sample). No stored ATT results exist for the paper's estimators, so their
ATT versions run here: Fed-IPW/AIPW (fed_ps ATT estimands) with FedAvg, NN and DW
membership, Meta-SW and pooled PS.

FedDSM settings as in feddsm_authors.py (fixed before the run). With ps="site" the
control-only site of the no-overlap scenarios drops out of the ATT (its controls can
never be matched under the site-confounding assumption).
Accuracy only: FedDSM has no SE yet.

    python experiments/run.py experiments/configs/feddsm_authors_att.py --reps 0-99 \
        --jobs 7 --out results/feddsm_authors_att
"""

from experiments.configs.ps_mc import BASE_SEED  # noqa: F401  (same datasets)
from experiments.harness import Scenario, multi
from src.dgps.authors import OVERLAPS, dgp_a_params, dgp_b_params, generate, true_att
from src.estimators import fed_ps, feddsm, meta_sw, pooled_ps


def _scenario(dgp, overlap):
    make = dgp_a_params if dgp == "A" else dgp_b_params
    return Scenario(
        name=f"{dgp}_{overlap}",
        generate=lambda seed: generate(make(overlap), dgp, seed)[1],
        truth=lambda n, seed: true_att(make(overlap), dgp, n, seed),
    )


SCENARIOS = [_scenario(dgp, overlap) for dgp in ("A", "B") for overlap in OVERLAPS]

FED = ["ATT-IPW-HT", "ATT-IPW-Hajek_pooled", "ATT-AIPW"]
META = ["ATT-IPW-HT", "ATT-IPW-Hajek", "ATT-AIPW"]

CONFIGS = [
    multi(f"feddsm_{ps}", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target="ATT", ps=ps)
    for ps in feddsm.PS_MODELS
] + [
    multi("fed_fedavg", fed_ps.estimate_multi, FED, mw="logistic_fedavg"),
    multi("fed_nn", fed_ps.estimate_multi, FED, mw="nn"),
    multi("fed_dw", fed_ps.estimate_multi, FED, mw="dw_param"),
    multi("meta_logistic", meta_sw.estimate_multi, META),
    multi("pooled_logistic", pooled_ps.estimate_multi, FED),
]
