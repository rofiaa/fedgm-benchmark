"""Single-arm data with long-tailed covariates (docs/longtail_design.md)."""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t

TAILS = ("gauss", "t3", "lognormal")
SELECTIONS = ("latent", "observed")
P = 10
SHARE = 0.08
GAMMA = 0.8
B = np.r_[1.0, 1.0, 1.0, 0.5, 0.5, np.zeros(P - 5)]
A = np.r_[1.0, 0.8, 0.6, 0.4, 0.2, np.zeros(P - 5)]
SITE_RATIO = np.array([4, 3, 2, 1]) / 10


def _intercept(s, share):
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expit(mid + s).mean() < share else (lo, mid)
    return (lo + hi) / 2


def generate(seed, tails="gauss", selection="latent", n=3000):
    if tails not in TAILS or selection not in SELECTIONS:
        raise ValueError((tails, selection))
    rng = np.random.default_rng(seed)
    i = np.arange(P)
    L = np.linalg.cholesky(0.5 ** np.abs(i[:, None] - i[None, :]))
    Z = rng.standard_normal((n, P)) @ L.T
    if tails == "gauss":
        X = Z.copy()
    elif tails == "t3":
        X = t.ppf(norm.cdf(Z), 3)
    else:
        X = np.exp(Z)
    u = Z if selection == "latent" else (X - X.mean(0)) / X.std(0)
    s = u @ B
    s = GAMMA * s / s.std()
    W = (rng.random(n) < expit(_intercept(s, SHARE) + s)).astype(int)
    Y0 = Z @ A + 0.5 * (Z[:, 0] ** 2 - 1) + 0.5 * Z[:, 1] * Z[:, 2] + rng.standard_normal(n)
    tau = 1 + 0.5 * Z[:, 0]
    H = np.where(W == 1, 0, 1 + rng.choice(4, n, p=SITE_RATIO))
    df = pd.DataFrame(X, columns=[f"X{j}" for j in range(1, P + 1)])
    df["H"], df["W"], df["Y"] = H, W, Y0 + W * tau
    df.attrs["tau_true"] = float(tau[W == 1].mean())
    return df
