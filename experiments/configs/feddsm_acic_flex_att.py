"""Centralized flexible-outcome references on ACIC 2016, ATT: same scenarios and
replications as feddsm_acic_att.py, for paired comparison with FedDSM
(src/estimators/flex_outcome.py; not federated).

    python experiments/run.py experiments/configs/feddsm_acic_flex_att.py --reps 0-19 \
        --jobs 7 --out results/feddsm_acic_flex_att
"""

from experiments.configs.feddsm_acic_att import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.harness import multi
from src.estimators import flex_outcome

CONFIGS = [multi("flex_ref", flex_outcome.estimate_multi, ["AIPW-RF", "AIPW-RF-rfps", "REG-RF", "REG-quad"], target="ATT")]
