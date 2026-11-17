"""Long-tailed covariates, single-arm ATT (docs/longtail_design.md).

    python experiments/run.py experiments/configs/longtail.py --reps 0-99 --jobs 6 \
        --out results/longtail
"""

from functools import partial

import numpy as np

from experiments.harness import Scenario, multi
from src.dgps import longtail
from src.estimators import feddsm, fedgm, flex_outcome, pooled_ps

BASE_SEED = 20261117
TARGET = "ATT"

SCENARIOS = [Scenario(name=f"{tl}_{sel}", truth=lambda n, seed: (np.nan, np.nan),
                      generate=partial(longtail.generate, tails=tl, selection=sel))
             for tl in longtail.TAILS for sel in longtail.SELECTIONS
             if (tl, sel) != ("gauss", "observed")]   # identical to gauss_latent
CONFIGS = [
    multi("fedgm_raw", fedgm.estimate_multi, list(fedgm.ESTIMANDS), sigma="treated_raw",
          target=TARGET),
    multi("feddsm_x", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target=TARGET, ps="x"),
    multi("pooled_logistic", pooled_ps.estimate_multi, ["ATT-IPW-Hajek_pooled", "ATT-AIPW"]),
    multi("flex_ref", flex_outcome.estimate_multi, ["DSM-RF", "AIPW-RF-rfps"], target=TARGET),
]
