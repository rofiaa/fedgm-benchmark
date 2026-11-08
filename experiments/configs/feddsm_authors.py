"""FedDSM verification on the authors' DGPs, ATE (Khellaf et al.'s six scenarios).

Same SCENARIOS and BASE_SEED as ps_mc.py, so every rep is the same dataset as in
results/ps_mc_v2: the paper's estimators (Fed-IPW/AIPW with logistic, FedAvg, NN and
DW membership, Meta-SW, pooled PS, Oracle) are compared from those rows, paired by rep.
fed_fedavg is rerun here as a check that the datasets are identical.

FedDSM settings were fixed before this run (src/estimators/feddsm.py): M = 4, 5 folds,
quadratic prognostic model, same-model bias adjustment. PS:
  feddsm_x        covariates only -- primary here: in these DGPs the outcome depends on
                  X only, so the site is not a confounder (the paper's assumption);
  feddsm_site     covariates + site indicators (undefined for the ATE when a site has a
                  single arm: no local overlap scenarios);
  feddsm_khellaf  the paper's federated PS (FedAvg membership).
The quadratic prognostic model is well specified for these outcomes (quadratic in X),
so this is a favourable case for FedDSM and for FedAIPW-quad alike.
Accuracy only: FedDSM has no SE yet.

    python experiments/run.py experiments/configs/feddsm_authors.py --reps 0-99 \
        --jobs 7 --out results/feddsm_authors
"""

from experiments.configs.ps_mc import BASE_SEED, SCENARIOS  # noqa: F401  (same datasets)
from experiments.harness import multi
from src.estimators import fed_ps, feddsm

CONFIGS = [
    multi(f"feddsm_{ps}", feddsm.estimate_multi, list(feddsm.ESTIMANDS), target="ATE", ps=ps)
    for ps in feddsm.PS_MODELS
] + [
    multi("fed_fedavg", fed_ps.estimate_multi, ["IPW-Hajek_pooled", "AIPW"],
          mw="logistic_fedavg"),
]
