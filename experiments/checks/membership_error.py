"""Membership-weight accuracy: estimated ω̂_k(x) vs the true ω*_k(x) = P(H=k | X=x).

ω* is the authors' "Oracle" definition (src.dgps.authors.oracle_frame):
  DGP A  ω*_k(x) ∝ n_k N(x; μ_k, Σ_k). The Σ_k differ across sites, so log ω*_k/ω*_j is
         quadratic in x and the (linear) multinomial logistic MW is misspecified.
  DGP B  ω* = multinomial logistic in X (membership_Thetas): the logistic MW is well
         specified.
Datasets and NN seeds are those of experiments/configs/ps_mc.py (reps 0..R-1).

Usage: python experiments/checks/membership_error.py results/checks/membership_error.csv [--reps 5]
Per (scenario, rep, model): mae = mean over units and sites of |ω̂ - ω*|;
p95_max = 95th percentile over units of max_k |ω̂_k - ω*_k|.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.configs.ps_mc import BASE_SEED
from experiments.harness import config_seed, data_seed
from src.dgps.authors import dgp_a_params, dgp_b_params, generate
from src.estimators.fed_ps import estimate

MODELS = {"logistic": ("fed_logistic", {}), "nn": ("fed_nn", {"mw": "nn"})}


def one(job):
    scenario, rep = job
    warnings.simplefilter("ignore")
    dgp, overlap = scenario.split("_")
    params = (dgp_a_params if dgp == "A" else dgp_b_params)(overlap)
    _, df = generate(params, dgp, data_seed(BASE_SEED, scenario, rep))
    orc = df.attrs["oracle"]
    omega_star = orc[[c for c in orc if c.startswith("omega_star_")]].to_numpy()
    rows = []
    for model, (config_id, kw) in MODELS.items():
        out = estimate(df, seed=config_seed(BASE_SEED, scenario, rep, config_id), **kw)
        err = np.abs(out["omega_hat"].to_numpy() - omega_star)
        rows.append(dict(scenario=scenario, rep=rep, model=model, mae=err.mean(),
                         p95_max=np.quantile(err.max(axis=1), 0.95)))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    jobs = [(f"{d}_{o}", r) for d in "AB" for o in ("no", "poor", "good")
            for r in range(a.reps)]
    with ProcessPoolExecutor(a.jobs) as pool:
        df = pd.DataFrame([r for rows in pool.map(one, jobs) for r in rows])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    s = df.groupby(["scenario", "model"], sort=False).agg(
        mae=("mae", "mean"), mae_sd=("mae", "std"), p95_max=("p95_max", "mean"))
    print(s.unstack("model").round(4).to_string())
