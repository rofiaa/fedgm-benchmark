"""Is the GenMatch pilot's downward bias a bug or matching-discrepancy bias?

Mahalanobis matching (no GenMatch search, so seconds per dataset) on the ps_mc
datasets, with the same covariates (X + pooled PS logit) and Match settings as
src.estimators.genmatch. Per dataset:
  sate        mean(Y1* - Y0*), the sample ATE from the oracle potential outcomes
  maha        Match's estimate (= genmatch "Maha")
  from_pairs  Σ w (Y_t - Y_c) / Σ w rebuilt from index.treated/control and weights
              (checks our pair bookkeeping against Match)
  ba_x        bias-adjusted on Z = X (= genmatch "Maha-BA")
  ba_quad     bias-adjusted on Z = X, X², X_iX_j (the outcome is quadratic in X)
If ba_quad ≈ sate while maha and ba_x are biased, the matches are correct and the bias
is the matching discrepancy times the outcome's curvature (not removable by linear BA).

Usage: python experiments/checks/matching_bias.py results/checks/matching_bias.csv [--reps 2]
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.configs.ps_mc import BASE_SEED, SCENARIOS
from experiments.harness import data_seed
from src.estimators.genmatch import _balance_matrix, _r_matrix
from src.estimators.pooled_ps import pooled_ps_logit
from src.r_env import ensure_r_home


def one(scenario, rep, ro, matching):
    df = scenario.generate(data_seed(BASE_SEED, scenario.name, rep))
    orc = df.attrs["oracle"]
    X = df[[f"X{i}" for i in range(1, 11)]].to_numpy(float)
    W, Y = df["W"].to_numpy(int), df["Y"].to_numpy(float)
    base = dict(Tr=ro.FloatVector(W), X=_r_matrix(ro, np.column_stack([X, pooled_ps_logit(df)])),
                estimand="ATE", M=1, replace=True, ties=True, Weight=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = matching.Match(Y=ro.FloatVector(Y), **base)
        est = {k: float(matching.Match(Y=ro.FloatVector(Y), **base, BiasAdjust=True,
                                       Z=_r_matrix(ro, Z)).rx2("est")[0])
               for k, Z in (("ba_x", X), ("ba_quad", _balance_matrix(X)))}
    it = np.asarray(m.rx2("index.treated"), int) - 1
    ic = np.asarray(m.rx2("index.control"), int) - 1
    w = np.asarray(m.rx2("weights"), float)
    return dict(scenario=scenario.name, rep=rep,
                sate=float(np.mean(orc["Y1*"] - orc["Y0*"])),
                maha=float(m.rx2("est")[0]),
                from_pairs=float(np.sum(w * (Y[it] - Y[ic])) / w.sum()), **est)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--reps", type=int, default=2)
    a = p.parse_args()
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    matching = importr("Matching")
    res = pd.DataFrame([one(s, r, ro, matching) for s in SCENARIOS for r in range(a.reps)])
    for c in ("maha", "ba_x", "ba_quad"):
        res[f"{c}_err"] = res[c] - res["sate"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
