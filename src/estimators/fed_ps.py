"""Federated IPW/AIPW via propensity-score aggregation (Khellaf, Bellet & Josse,
arXiv:2505.17961), applied to an external data frame.

The authors' `Simulations_Fed.estimate_ipw/estimate_aipw` generate data inside their
simulation loop, so we re-assemble the estimator from their standalone functions:

    ê(x) = Σ_k ω̂_k(x) ê_k(x),   ω̂_k(x) = P̂(H = k | X = x),

with ê_k fitted at site k and predicted on all rows, and
τ̂ = Σ_k (n_k / n) τ̂_k, where τ̂_k is the IPW estimate on site k using the global ê.
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.estimators._authors import dgp_fed
from src.estimators.balance import smd

_IMPLEMENTED = {
    "mw": {"logistic"},
    "local_ps": {"logistic"},
    "ipw_type": {"HT"},
    "aipw": {False},
}


def estimate(
    df,
    mw="logistic",
    local_ps="logistic",
    ipw_type="HT",
    aipw=False,
    seed=None,
    mw_intercept=True,
):
    """Federated IPW estimate of the ATE.

    df: columns X1..Xd, H (sites 0..K-1), W (0/1), Y.
    mw: membership-weight model for P(H|X). "logistic" = authors' MW_estimation_pooled
        (a centralized multinomial fit, the default in their code).
    local_ps: local propensity model e_k(x). "logistic" = authors' compute_gamma.
    ipw_type: "HT" (Horvitz-Thompson).
    mw_intercept: include an intercept in the membership model (the authors' IPW path
        does; their AIPW path does not).
    seed: reserved for the stochastic variants; unused by the deterministic ones.

    Returns dict(tau, se, se_method, e_hat, e_local, omega_hat, n_k, smd, max_smd,
    mw_fit, outcome_fit, runtime).
    """
    for name, value in [("mw", mw), ("local_ps", local_ps),
                        ("ipw_type", ipw_type), ("aipw", aipw)]:
        if value not in _IMPLEMENTED[name]:
            raise NotImplementedError(f"{name}={value!r} is not implemented yet")

    start = time.perf_counter()
    A = dgp_fed()
    data, x_cols, K = _to_authors_frame(df)
    ps_cols = x_cols + ["intercept"]
    mw_cols = x_cols + ["intercept"] if mw_intercept else x_cols

    e_local = _local_propensities(A, data, ps_cols, K)
    omega = _site_order_numeric(
        A.MW_estimation_pooled(data, mw_cols, scale=False).to_numpy(), K
    )
    e_hat = A.membership_weighting_vectorized(pd.DataFrame(omega), pd.DataFrame(e_local))
    data["e_hat"] = e_hat

    site_masks = [data["client"] == _label(k) for k in range(K)]
    n_k = np.array([mask.sum() for mask in site_masks])
    tau_k = [
        A.ipw_function(data[mask], hat_e_name="e_hat", ht_or_hajek=ipw_type)
        for mask in site_masks
    ]
    tau = float(np.average(tau_k, weights=n_k))

    psi = _ipw_terms_ht(data["W"].to_numpy(), data["Y"].to_numpy(), e_hat)
    se = _if_se(psi)

    weights = np.where(data["W"] == 1, 1 / e_hat, 1 / (1 - e_hat))
    balance = smd(data[x_cols], data["W"], weights)

    index = df.index
    return dict(
        tau=tau,
        se=se,
        se_method="influence function, nuisances treated as fixed (provisional)",
        e_hat=pd.Series(e_hat, index=index, name="e_hat"),
        e_local=pd.DataFrame(e_local, index=index, columns=range(K)),
        omega_hat=pd.DataFrame(omega, index=index, columns=range(K)),
        n_k=dict(enumerate(n_k.tolist())),
        smd=balance,
        max_smd=float(balance.abs().max()),
        mw_fit="pooled-centralized",
        outcome_fit=None,
        runtime=time.perf_counter() - start,
    )


def _label(k):
    """Authors' site label for our 0-indexed site k."""
    return f"client{k + 1}"


def _to_authors_frame(df):
    """Our layout (X1..Xd, H, W, Y) -> authors' layout (X1..Xd, intercept, client, W, Y)."""
    x_cols = sorted(
        (c for c in df.columns if c[0] == "X" and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if not x_cols:
        raise ValueError("df has no covariate columns X1..Xd")
    missing = {"H", "W", "Y"} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns {sorted(missing)}")

    H = df["H"].to_numpy()
    if not np.issubdtype(H.dtype, np.integer):
        raise ValueError("H must be integer site labels 0..K-1")
    K = int(H.max()) + 1
    if H.min() != 0 or len(np.unique(H)) != K:
        raise ValueError("H must take every value 0..K-1")
    if not np.isin(df["W"].unique(), [0, 1]).all():
        raise ValueError("W must be 0/1")

    data = pd.DataFrame({c: df[c].to_numpy(dtype=float) for c in x_cols})
    data["intercept"] = 1.0
    data["client"] = [_label(h) for h in H]
    data["W"] = df["W"].to_numpy(dtype=int)
    data["Y"] = df["Y"].to_numpy(dtype=float)
    return data, x_cols, K


def _local_propensities(A, data, ps_cols, K):
    """ê_k fitted on site k, predicted on all rows; (n, K) array.

    A site with a single treatment arm gets the constant ê_k = 0 or 1 (it never / always
    treats), as in the authors' external-control-arm setting.
    """
    e_local = np.empty((len(data), K))
    for k in range(K):
        site = data[data["client"] == _label(k)]
        arms = site["W"].unique()
        if len(arms) == 1:
            e_local[:, k] = float(arms[0])
        else:
            gamma = A.compute_gamma(site, ps_cols)
            e_local[:, k] = A.logistic_function_vectorized(data, ps_cols, gamma)
    return e_local


def _site_order_numeric(proba, K):
    """Reorder classifier columns from alphabetical to numeric site order.

    The authors' classifiers return columns in sorted-label order ("client1",
    "client10", "client2", ...), which differs from numeric order once K >= 10.
    """
    alphabetical = sorted(_label(k) for k in range(K))
    return proba[:, [alphabetical.index(_label(k)) for k in range(K)]]


def _ipw_terms_ht(W, Y, e):
    return W * Y / e - (1 - W) * Y / (1 - e)


def _if_se(psi):
    if not np.all(np.isfinite(psi)):
        warnings.warn("non-finite IPW terms (ê hit 0 or 1); se is NaN")
        return float("nan")
    return float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
