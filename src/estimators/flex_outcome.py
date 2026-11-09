"""Centralized flexible-outcome references for FedDSM (NOT federated).

Doubly robust and regression estimators with random-forest nuisance models on the pooled
data, to check whether FedDSM beats flexible AIPW or only the linear versions (docs/
findings.md N9). The forests would be hard to federate; these are benchmarks, not
candidates.

Nuisances: outcome forests m̂_w (RandomForestRegressor, 300 trees, min_samples_leaf=5,
max_features=1/3) fitted within each arm; units of the arm get their out-of-bag
prediction (cross-fitting), the other arm the full forest's. PS: the covariate-only
logistic fit (FedDSM's, distributed Newton) or a probability forest (out-of-bag,
clipped to [0.01, 0.99]).

Estimands (target "ATE" or "ATT"):
  "AIPW-RF"        forest outcome, logistic PS. ATE: AIPW (HT form); ATT: the
                   normalized ATT-AIPW of fed_ps.
  "AIPW-RF-norm"   ATE only: forest outcome, logistic PS, normalized weights.
  "AIPW-RF-rfps"   forest outcome and forest PS, normalized weights.
  "REG-RF"         regression only with the forests.
  "REG-quad"       regression only with FedDSM's quadratic prognostic model
                   (feddsm.fed_quad, same interaction rule).
se: influence-function SE with nuisances fixed for the AIPW variants; NaN for regression.
"""

import time

import numpy as np
import pandas as pd

from src.estimators.balance import smd
from src.estimators.fed_ps import _att_terms, _if_se
from src.estimators.feddsm import _basis, fed_logistic_newton, fed_quad

ESTIMANDS = ("AIPW-RF", "AIPW-RF-norm", "AIPW-RF-rfps", "REG-RF", "REG-quad")
_CLIP = 1e-8


def _rf_outcomes(X, W, Y, seed, arms):
    from sklearn.ensemble import RandomForestRegressor
    mu = {}
    for arm in arms:
        a = W == arm
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=1 / 3,
                                   oob_score=True, random_state=int(seed) + arm, n_jobs=1)
        rf.fit(X[a], Y[a])
        pred = rf.predict(X)
        pred[a] = rf.oob_prediction_
        mu[arm] = pred
    return mu


def _rf_ps(X, W, seed):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                oob_score=True, random_state=int(seed), n_jobs=1).fit(X, W)
    return np.clip(rf.oob_decision_function_[:, 1], 0.01, 0.99)


def _ate_norm(W, Y, e, mu):
    n = len(W)
    w1, w0 = W / e, (1 - W) / (1 - e)
    psi = (mu[1] - mu[0] + n * w1 / w1.sum() * (Y - mu[1])
           - n * w0 / w0.sum() * (Y - mu[0]))
    return float(psi.mean()), psi


def estimate_multi(df, estimands=ESTIMANDS, target="ATE", seed=None):
    """{estimand: result}; df: X1..Xd, H, W, Y. Frozen keys plus estimand, target."""
    estimands = list(dict.fromkeys(estimands))
    if set(estimands) - set(ESTIMANDS):
        raise NotImplementedError(sorted(set(estimands) - set(ESTIMANDS)))
    if target not in ("ATE", "ATT"):
        raise ValueError("target must be 'ATE' or 'ATT'")
    if target == "ATT" and "AIPW-RF-norm" in estimands:
        raise ValueError("AIPW-RF-norm is ATE-only; the ATT AIPW is already normalized")
    start = time.perf_counter()
    x_cols = sorted((c for c in df.columns if c[0] == "X" and c[1:].isdigit()),
                    key=lambda c: int(c[1:]))
    X, H, W, Y = (df[x_cols].to_numpy(float), df["H"].to_numpy(int), df["W"].to_numpy(int),
                  df["Y"].to_numpy(float))
    seeds = np.random.SeedSequence(seed).generate_state(4)
    arms = (0,) if target == "ATT" else (0, 1)
    n = len(W)
    mu = _rf_outcomes(X, W, Y, seeds[0], arms) if set(estimands) - {"REG-quad"} else None
    results = {}
    e_log = e_rf = None

    def ps(kind):
        nonlocal e_log, e_rf
        if kind == "log":
            if e_log is None:
                lp, _ = fed_logistic_newton(X, W, H)
                e_log = np.clip(1 / (1 + np.exp(-lp)), _CLIP, 1 - _CLIP)
            return e_log
        if e_rf is None:
            e_rf = _rf_ps(X, W, seeds[1])
        return e_rf

    def add(name, tau, psi=None, weights=None):
        se = float("nan") if psi is None else _if_se(psi)
        bal = float("nan") if weights is None else float(smd(pd.DataFrame(X), W, weights).abs().max())
        results[name] = dict(tau=float(tau), se=se, se_method="influence function, nuisances fixed"
                             if psi is not None else "none", max_smd=bal, estimand=name,
                             target=target)

    for name in estimands:
        if name.startswith("AIPW"):
            e = ps("rf" if name.endswith("rfps") else "log")
            if target == "ATT":
                tau, _, psi = _att_terms(W, Y - mu[0], e, [np.ones(n, bool)], normalize=True)
                w = np.where(W == 1, 1.0, e / (1 - e))
            elif name == "AIPW-RF":
                psi = mu[1] - mu[0] + W * (Y - mu[1]) / e - (1 - W) * (Y - mu[0]) / (1 - e)
                tau, w = float(psi.mean()), np.where(W == 1, 1 / e, 1 / (1 - e))
            else:
                tau, psi = _ate_norm(W, Y, e, mu)
                w = np.where(W == 1, 1 / e, 1 / (1 - e))
            add(name, tau, psi, w)
        elif name == "REG-RF":
            add(name, np.mean((Y - mu[0])[W == 1]) if target == "ATT" else np.mean(mu[1] - mu[0]))
        else:  # REG-quad, FedDSM's prognostic model
            quad = [j for j, c in enumerate(x_cols) if df[c].nunique() > 2]
            m = {}
            for arm in arms:
                use_int = _basis(X[:1], quad).shape[1] <= (W == arm).sum() / 2
                B = _basis(X, quad, interactions=use_int)
                m[arm] = fed_quad(B, Y, W, H, arm, seed=int(seeds[2 + arm]))[0]
            add(name, np.mean((Y - m[0])[W == 1]) if target == "ATT" else np.mean(m[1] - m[0]))
    runtime = time.perf_counter() - start
    for r in results.values():
        r["runtime"] = runtime
    return results
