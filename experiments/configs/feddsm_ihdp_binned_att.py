"""FedDSM with the binned sums-only score and the clipped variant (score="binned",
docs/findings.md N13) on IHDP, ATT. Same scenarios, config id (hence folds) and PS as feddsm_ihdp_att,
paired with its FedDSM rows.

    python experiments/run.py experiments/configs/feddsm_ihdp_binned_att.py --reps 0-99 --jobs 7 \
        --out results/feddsm_ihdp_binned_att
"""

from experiments.configs.feddsm_ihdp_att import BASE_SEED, SCENARIOS, _site_as_h  # noqa: F401
from experiments.harness import multi
from src.estimators import feddsm

CONFIGS = [multi("feddsm_site", _site_as_h(feddsm.estimate_multi), [*feddsm.ESTIMANDS, *feddsm.EXTRA_ESTIMANDS], target="ATT", ps="site",
                 score="binned")]
