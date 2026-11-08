"""FedDSM on IHDP, ATT (100 replications); as feddsm_ihdp.py with ihdp_att.py's
scenario (truth = the ATT, 4 by construction).

    python experiments/run.py experiments/configs/feddsm_ihdp_att.py --reps 0-99 \
        --jobs 7 --out results/feddsm_ihdp_att
"""

from experiments.configs.ihdp_att import BASE_SEED, SCENARIOS, _site_as_h  # noqa: F401
from experiments.harness import multi
from src.estimators import feddsm

CONFIGS = [multi(f"feddsm_{ps}", _site_as_h(feddsm.estimate_multi), list(feddsm.ESTIMANDS),
                 target="ATT", ps=ps) for ps in ("site", "khellaf")]
