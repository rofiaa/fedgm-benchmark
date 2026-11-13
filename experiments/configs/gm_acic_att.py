"""Centralized GenMatch on ACIC 2016, ATT (as gm_acic.py; pairs with feddsm_acic_att.py).

Narval only:
    sbatch --job-name=gm_acic_att --time=24:00:00 --array=0-76 experiments/slurm/array.sh \
        experiments/configs/gm_acic_att.py results/gm_acic_att 0 $(python -c \
        "print(' '.join(f'acic{s:02d}' for s in range(1, 78)))")
"""

from experiments.configs.feddsm_acic_att import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.configs.gm_acic import GM_BUDGET
from experiments.harness import multi
from src.estimators import genmatch

CONFIGS = [multi("gm_xps", genmatch.estimate_multi, list(genmatch.ESTIMANDS),
                 covariates="X+ps", balance_quad=[], target="ATT", genmatch_params=GM_BUDGET)]
