"""FedDSM on IHDP, ATE (100 replications), through src/estimators/feddsm.py.

Same scenario as ihdp_mc.py (same BASE_SEED and name, so the same replications and
truth handling). The site is a confounder in IHDP (Hill's surface uses the site
dummies), so the primary PS is site-aware (ps="site"); the paper's federated PS
("khellaf") is reported too. FedDSM sees X1..X18 + H (the site enters as indicators
via ps="site"), like the federated estimators. Settings fixed beforehand (M = 4,
quadratic prognostic model on X1..X6, same-model bias adjustment).

    python experiments/run.py experiments/configs/feddsm_ihdp.py --reps 0-99 --jobs 7 \
        --out results/feddsm_ihdp
"""

from experiments.configs.ihdp_mc import BASE_SEED, SCENARIOS, _site_as_h  # noqa: F401
from experiments.harness import multi
from src.estimators import feddsm

CONFIGS = [multi(f"feddsm_{ps}", _site_as_h(feddsm.estimate_multi), list(feddsm.ESTIMANDS),
                 target="ATE", ps=ps) for ps in ("site", "khellaf")]
