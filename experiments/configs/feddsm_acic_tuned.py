"""FedDSM with the tuned prognostic score (score="tuned", docs/findings.md N12) on ACIC 2016, ATE.
Same scenarios, config id (hence folds) and PS as feddsm_acic, paired with its FedDSM rows.

    python experiments/run.py experiments/configs/feddsm_acic_tuned.py --reps 0-19 --jobs 7 \
        --out results/feddsm_acic_tuned
"""

from experiments.configs.feddsm_acic import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.harness import multi
from src.estimators import feddsm

CONFIGS = [multi("feddsm_x", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target="ATE", ps="x", score="tuned")]
