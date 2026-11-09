"""FedDSM with the tuned prognostic score (score="tuned", docs/findings.md N12) on IHDP, ATE.
Same scenarios, config id (hence folds) and PS as feddsm_ihdp, paired with its FedDSM rows.

    python experiments/run.py experiments/configs/feddsm_ihdp_tuned.py --reps 0-99 --jobs 7 \
        --out results/feddsm_ihdp_tuned
"""

from experiments.configs.feddsm_ihdp import BASE_SEED, SCENARIOS, _site_as_h  # noqa: F401
from experiments.harness import multi
from src.estimators import feddsm

CONFIGS = [multi("feddsm_site", _site_as_h(feddsm.estimate_multi), list(feddsm.ESTIMANDS), target="ATE", ps="site", score="tuned")]
