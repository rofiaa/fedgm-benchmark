"""FedDSM with the binned sums-only score and the clipped variant (score="binned",
docs/findings.md N13) on the authors' DGPs, ATE. Same scenarios, config id (hence folds) and PS as feddsm_authors,
paired with its FedDSM rows.

    python experiments/run.py experiments/configs/feddsm_authors_binned.py --reps 0-99 --jobs 7 \
        --out results/feddsm_authors_binned
"""

from experiments.configs.feddsm_authors import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.harness import multi
from src.estimators import feddsm

CONFIGS = [multi("feddsm_x", feddsm.estimate_multi, [*feddsm.ESTIMANDS, *feddsm.EXTRA_ESTIMANDS], target="ATE", ps="x",
                 score="binned")]
