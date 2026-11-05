"""IHDP: does matching with more neighbours (M) close the variance gap to Fed-AIPW?

Mahalanobis matching with Abadie-Imbens bias adjustment on X (= ihdp_mc's "Maha-BA"
at M=1; no GenMatch search), same covariates (X1..X25 + pooled PS logit), ATE, on the
100 IHDP replications, for M = 1, 2, 4, 8.

Usage: python experiments/checks/ihdp_matching_m.py results/checks/ihdp_matching_m.csv
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dgps import ihdp
from src.estimators.genmatch import _r_matrix
from src.estimators.pooled_ps import pooled_ps_logit
from src.r_env import ensure_r_home
ensure_r_home()
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
M_ = importr("Matching")
rows = []
for rep in range(100):
    df = ihdp.load(rep); xc = [f"X{i}" for i in range(1, 26)]
    X = df[xc].to_numpy(float)
    Xm = _r_matrix(ro, np.column_stack([X, pooled_ps_logit(df)]))
    for M in (1, 2, 4, 8):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = M_.Match(Y=ro.FloatVector(df.Y), Tr=ro.FloatVector(df.W), X=Xm, estimand="ATE", M=M,
                         replace=True, ties=True, Weight=2, BiasAdjust=True, Z=_r_matrix(ro, X))
        rows.append(dict(rep=rep, M=M, e=float(m.rx2("est")[0]) - df.attrs["tau_true"], se=float(m.rx2("se")[0])))
d = pd.DataFrame(rows)
d.to_csv(sys.argv[1], index=False)
print(d.groupby("M").apply(lambda g: pd.Series(dict(rmse=np.sqrt((g.e**2).mean()), mae=g.e.abs().mean(),
      bias=g.e.mean(), sd=g.e.std(), mean_se=g.se.mean(), cover=(g.e.abs() <= 1.96*g.se).mean())), include_groups=False).round(3))
