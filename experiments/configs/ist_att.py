"""IST semi-synthetic benchmark, ATT: the observational scenarios of ist.py with every
method switched to the ATT (same BASE_SEED, so the same data per (scenario, rep)).

    python experiments/run.py experiments/configs/ist_att.py --reps 0-99 --jobs 4 \
        --out results/ist_att --configs feddsm_x feddsm_site fed_fedavg fed_nn \
        meta_logistic pooled_logistic flex_ref
GenMatch (gm_xps) runs on Narval.
"""

from experiments.configs.ist import BASE_SEED, OBS, configs, scenarios  # noqa: F401

TARGET = "ATT"
SCENARIOS = scenarios(OBS, TARGET)
CONFIGS = configs(TARGET)
