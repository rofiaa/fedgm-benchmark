"""FedDSM with the tuned prognostic score (score="tuned", docs/findings.md N12) on the authors' DGPs, ATT.
Same scenarios, config id (hence folds) and PS as feddsm_authors_att, paired with its FedDSM rows.

    python experiments/run.py experiments/configs/feddsm_authors_tuned_att.py --reps 0-99 --jobs 7 \
        --out results/feddsm_authors_tuned_att
"""

from experiments.configs.feddsm_authors_att import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.harness import multi
from src.estimators import feddsm

CONFIGS = [multi("feddsm_x", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target="ATT", ps="x", score="tuned")]
