"""100-rep recheck of NN-MW IPW-HT bias in DGP A (paper mu_3), IF only.

Usage: python experiments/checks/nn_ipw_dgp_a_recheck.py results/checks/nn_ipw_dgp_a_recheck.csv
See docs/findings.md, N1.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from src.dgps.authors import dgp_a_poor_overlap_params, generate
from src.estimators.fed_ps import estimate

def one(seed):
    warnings.simplefilter("ignore")
    _, df = generate(dgp_a_poor_overlap_params(), "A", seed)
    rows = []
    for label, kw in [("logistic IPW-HT", {}), ("nn IPW-HT", {"mw": "nn"}), ("nn AIPW", {"mw": "nn", "aipw": True})]:
        o = estimate(df, seed=seed, **kw)
        rows.append(dict(seed=seed, method=label, tau=o["tau"], se=o["se"], min_e=float(o["e_hat"].min())))
    return rows

if __name__ == "__main__":
    tau = 8.4559  # results/mini_mc_summary.csv, scenario A
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(777).spawn(100)]
    with ProcessPoolExecutor(7) as pool:
        df = pd.DataFrame([r for rows in pool.map(one, seeds) for r in rows])
    df.to_csv(sys.argv[1], index=False)
    for m, g in df.groupby("method"):
        e = g.tau - tau
        print(f"{m:16s} bias {e.mean():+.3f} ± {g.tau.std(ddof=1)/np.sqrt(len(g)):.3f}  median {e.median():+.3f}  sd {g.tau.std(ddof=1):.3f}  mean IF se {g.se.mean():.3f}  cov_IF {((g.tau-1.96*g.se<=tau)&(tau<=g.tau+1.96*g.se)).mean():.2f}  min ê {g.min_e.min():.2e}")
