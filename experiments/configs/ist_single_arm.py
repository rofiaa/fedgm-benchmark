"""IST semi-synthetic benchmark, Bret Nestor's single-arm setting (docs/ist_design.md):
the treated (8%) form one "trial" site, the 9 country sites hold controls only; ATT.

The paper's estimators and Meta-SW need both arms within a site and FedDSM's site-aware
PS is not identified here, so they are not run. Centralized methods see no country
dummies (they would identify the treated).

    python experiments/run.py experiments/configs/ist_single_arm.py --reps 0-99 --jobs 4 \
        --out results/ist_single_arm --configs feddsm_x pooled_logistic flex_ref
GenMatch (gm_xps) runs on Narval.
"""

from experiments.configs.ist import BASE_SEED, configs, scenarios  # noqa: F401

TARGET = "ATT"
SCENARIOS = scenarios(["single_arm"], TARGET)
CONFIGS = configs(TARGET, site_dummies=False, federated=False)
