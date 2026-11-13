"""Centralized GenMatch on ACIC 2016 (src/dgps/acic2016.py), ATE: Bret Nestor's
hypothesis that GenMatch helps on high-dimensional data (79 covariates).

Same datasets as feddsm_acic.py (the competition's replications), so rows pair with
FedDSM per (setting, rep). Plan: rep 0 of all 77 settings (77 paired datasets per
estimand), after a timing pilot on 3 settings.

Search budget (reduced from genmatch-lalonde's pop.size 1000, fixed before the run):
at full budget one DGP B dataset (n = 4,000, 10 covariates) took 3.4-6.4 h, and ACIC has
n = 4,802 and 79 covariates. Balance matrix: the 79 covariates, no squares/products
(3,239 columns otherwise). Matching on X + the pooled PS logit, as everywhere else.

Narval only:
    sbatch --job-name=gm_acic --time=24:00:00 --array=0-76 experiments/slurm/array.sh \
        experiments/configs/gm_acic.py results/gm_acic 0 $(python -c \
        "print(' '.join(f'acic{s:02d}' for s in range(1, 78)))")
"""

from experiments.configs.feddsm_acic import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.harness import multi
from src.estimators import genmatch

TARGET = "ATE"
GM_BUDGET = {"pop_size": 200, "max_generations": 30, "wait_generations": 5}

CONFIGS = [multi("gm_xps", genmatch.estimate_multi, list(genmatch.ESTIMANDS),
                 covariates="X+ps", balance_quad=[], target=TARGET, genmatch_params=GM_BUDGET)]
