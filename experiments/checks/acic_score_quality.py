"""ACIC 2016, replication 0 of each setting: accuracy of the control prognostic score
m̂_0 (FedDSM's quadratic model, random forest), against the true mu.0 on the treated and
as a network could see it (held-out R² on observed Y among controls). docs/findings.md N11.

Writes results/checks/acic_score_quality.csv.
"""

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results" / "checks" / "acic_score_quality.csv"


def r2(m, y, r):
    return 1 - np.sum((y[r] - m[r]) ** 2) / np.sum((y[r] - y[r].mean()) ** 2)


def one(s):
    from src.dgps import acic2016 as A
    from src.estimators.feddsm import _basis, fed_quad
    from src.estimators.flex_outcome import _rf_outcomes
    df = A.load(0, s, "ATT")
    _, acic, inp, _ = A._r()
    mu0 = np.asarray(acic.dgp_2016(inp, int(s), 1).rx2("mu.0"), float)
    cols = [k for k in df if k[0] == "X" and k[1:].isdigit()]
    X, W, Y, H = (df[cols].to_numpy(float), df.W.to_numpy(int), df.Y.to_numpy(float), df.H.to_numpy(int))
    quad = [j for j, k in enumerate(cols) if df[k].nunique() > 2]
    use_int = _basis(X[:1], quad).shape[1] <= (W == 0).sum() / 2
    mq = fed_quad(_basis(X, quad, interactions=use_int), Y, W, H, 0, seed=s)[0]
    mr = _rf_outcomes(X, W, Y, s, (0,))[0]
    t, c = W == 1, W == 0
    return dict(setting=s, quad_r2_true_treated=r2(mq, mu0, t), rf_r2_true_treated=r2(mr, mu0, t),
                quad_heldout_r2=r2(mq, Y, c), rf_heldout_r2=r2(mr, Y, c), true_mu_r2=r2(mu0, Y, c))


if __name__ == "__main__":
    with mp.Pool(7) as p:
        d = pd.DataFrame(p.map(one, range(1, 78), chunksize=2))
    d.to_csv(OUT, index=False)
    print(d.drop(columns="setting").describe().round(3).to_string())
