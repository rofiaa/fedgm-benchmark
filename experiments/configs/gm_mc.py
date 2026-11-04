"""Centralized GenMatch on the ps_mc datasets: same scenarios, BASE_SEED and hence the
same data per (scenario, rep), so rows pair with experiments/configs/ps_mc.py.

One GenMatch search per rep (genmatch-lalonde / D&S settings, ATE, covariates X + the
pooled PS logit) gives GM, GM-BA (bias-adjusted), and the search-free Mahalanobis
matches Maha, Maha-BA. Inference: Abadie-Imbens SE from Match (no bootstrap: it is not
valid for matching estimators).

    python experiments/run.py experiments/configs/gm_mc.py --reps 0-99 --jobs 7 \
        --out results/gm_mc
"""

from experiments.configs.ps_mc import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.harness import multi
from src.estimators import genmatch

CONFIGS = [
    multi("gm_xps", genmatch.estimate_multi, list(genmatch.ESTIMANDS), covariates="X+ps"),
]
