"""Double-score matching with random-forest prognostic scores on ACIC 2016, ATT
(centralized; docs/findings.md N11): same scenarios and replications as feddsm_acic_att.py.
Is FedDSM's gap to AIPW RF + RF the outcome model or the matching? AIPW-RF-rfps is
refitted here on the same forests.

Reading rule, fixed before the run: remaining paired RMSE gap of DSM-RF-rfps to
AIPW-RF-rfps relative to FedDSM's (ATE 0.116, ATT 0.144). At least 75% closed: the
outcome model was the limit; under 25%: matching itself costs accuracy; else partial.

    python experiments/run.py experiments/configs/feddsm_acic_dsmrf_att.py --reps 0-19 --jobs 7 \
        --out results/feddsm_acic_dsmrf_att
"""

from experiments.configs.feddsm_acic_att import BASE_SEED, SCENARIOS  # noqa: F401
from experiments.harness import multi
from src.estimators import flex_outcome

CONFIGS = [multi("flex_ref", flex_outcome.estimate_multi,
                 [*flex_outcome.DSM_ESTIMANDS, "AIPW-RF-rfps"], target="ATT")]
