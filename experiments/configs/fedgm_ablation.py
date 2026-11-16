"""Fed-GenMatch normaliser ablation (docs/fedgm_ablation.md): Bret Nestor's design
(src/estimators/fedgm.py) with four covariance normalisers, on IST single_arm and on the
same data with 50 treated (single_arm_nt50). References: FedDSM (x) and pooled PS.

    python experiments/run.py experiments/configs/fedgm_ablation.py --reps 0-99 --jobs 6 \
        --out results/fedgm_ablation
"""

from functools import partial

import numpy as np

from experiments.configs.ist import BASE_SEED  # noqa: F401  (same seeds as results/ist_single_arm)
from experiments.harness import Scenario, multi
from src.dgps import ist
from src.estimators import feddsm, fedgm, pooled_ps

TARGET = "ATT"


def _scenario(name, n_treated=None):
    return Scenario(name=name, by_rep=True, truth=lambda n, seed: (np.nan, np.nan),
                    generate=partial(ist.load, scenario="single_arm", target=TARGET,
                                     n_treated=n_treated))


SCENARIOS = [_scenario("single_arm"), _scenario("single_arm_nt50", n_treated=50)]
CONFIGS = [multi(f"fedgm_{s}", fedgm.estimate_multi, list(fedgm.ESTIMANDS), sigma=s,
                 target=TARGET) for s in fedgm.SIGMAS] + [
    multi("feddsm_x", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target=TARGET, ps="x"),
    multi("pooled_logistic", pooled_ps.estimate_multi, ["ATT-IPW-Hajek_pooled", "ATT-AIPW"]),
]
