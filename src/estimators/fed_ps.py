"""Federated IPW/AIPW via propensity-score aggregation (Khellaf, Bellet & Josse,
arXiv:2505.17961), applied to an external data frame.

The authors' `Simulations_Fed.estimate_ipw/estimate_aipw` generate data inside their
simulation loop, so we re-assemble the estimator from their standalone functions:

    ê(x) = Σ_k ω̂_k(x) ê_k(x),   ω̂_k(x) = P̂(H = k | X = x),

with ê_k fitted at site k and predicted on all rows, and
τ̂ = Σ_k (n_k / n) τ̂_k, where τ̂_k is the IPW (or AIPW) estimate on site k using the
global ê (and, for AIPW, outcome models μ̂_0, μ̂_1 shared by all sites).
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.estimators._authors import dgp_fed
from src.estimators.balance import smd

_IMPLEMENTED = {
    "mw": {"logistic", "logistic_fedavg"},
    "local_ps": {"logistic"},
    "ipw_type": {"HT", "Hajek", "Hajek_pooled"},
    "aipw": {False, True},
    "outcome": {"ols", "fedavg"},
}

# Outcome-model FedAvg (authors' fedavg_sgd_server_side): the notebook's
# federation_parameters for AIPW. Fixed number of rounds, gradients clipped to [-1, 1].
FEDAVG_OUTCOME_NOTEBOOK = {"T_rounds": 1000, "eta": 0.01, "E_local_steps": 1,
                           "B_batch_size": None}

# Membership-weight FedAvg: run until the relative change in Theta drops below `tol`,
# at most T_rounds rounds. tol=None runs exactly T_rounds (the authors' behaviour).
# E_local_steps=1: with E > 1 and one class per site, FedAvg converges to a different
# point than the pooled fit (client drift; docs/findings.md). standardize=True rescales
# X by its global mean/SD (shareable as per-site sums) so eta=1 converges in < 1000
# rounds; the intercept makes the fitted probabilities invariant to this rescaling.
FEDAVG_MW_DEFAULTS = {"T_rounds": 1000, "eta": 1.0, "E_local_steps": 1,
                      "B_batch_size": None, "tol": 1e-6, "standardize": True}
# The notebook's federation_parameters_omega; used for parity with the authors' code.
FEDAVG_MW_NOTEBOOK = {"T_rounds": 30, "eta": 0.1, "E_local_steps": 3,
                      "B_batch_size": None, "tol": None, "standardize": False}

_MW_FIT = {"logistic": "pooled-centralized", "logistic_fedavg": "fedavg"}
_OUTCOME_FIT = {"ols": "pooled-ols", "fedavg": "fedavg-sgd-clipped"}


def estimate(
    df,
    mw="logistic",
    local_ps="logistic",
    ipw_type="HT",
    aipw=False,
    seed=None,
    mw_intercept=True,
    fedavg_params=None,
    outcome="ols",
    outcome_fedavg_params=None,
):
    """Federated IPW / AIPW estimate of the ATE.

    df: columns X1..Xd, H (sites 0..K-1), W (0/1), Y.
    mw: membership-weight model for P(H|X). "logistic" = authors' MW_estimation_pooled
        (a centralized multinomial fit, the default in their code); "logistic_fedavg" =
        FedAvg multinomial logistic with full-batch local steps (authors' algorithm and
        gradient, plus a stopping rule; see FEDAVG_MW_DEFAULTS).
    local_ps: local propensity model e_k(x). "logistic" = authors' compute_gamma.
    ipw_type: "HT" (Horvitz-Thompson; primary), "Hajek_pooled" (weights normalized over
        all sites), or "Hajek" -- the authors' version, which normalizes within each
        site using the global ê and is inconsistent (docs/findings.md); parity only.
        IPW only; must be "HT" when aipw=True.
    aipw: augmented IPW with linear outcome models μ̂_w(x) = [x, 1]·β̂_w, as the authors'
        "GD-MW_*" estimators.
    outcome: outcome-model fit for aipw=True. "ols" = per-arm OLS on all sites' data
        (the authors' "GD" when params_for_federation is None); "fedavg" = authors'
        fedavg_sgd_server_side (clipped gradients, fixed rounds).
    outcome_fedavg_params: overrides of FEDAVG_OUTCOME_NOTEBOOK for outcome="fedavg".
    mw_intercept: include an intercept in the membership model (the authors' IPW path
        does; their AIPW path does not).
    fedavg_params: overrides of FEDAVG_MW_DEFAULTS for mw="logistic_fedavg". Only
        B_batch_size=None is supported: minibatches are drawn from numpy's global RNG.
    seed: reserved for the stochastic variants; unused by the deterministic ones.

    Returns dict(tau, se, se_method, e_hat, e_local, omega_hat, n_k, smd, max_smd,
    mw_fit, mw_rounds, mw_converged, outcome_fit, outcome_coef, runtime).
    mw_rounds/mw_converged are None unless mw="logistic_fedavg"; outcome_fit and
    outcome_coef (dict beta_1, beta_0 over X1..Xd, intercept) are None unless aipw=True.
    """
    for name, value in [("mw", mw), ("local_ps", local_ps), ("ipw_type", ipw_type),
                        ("aipw", aipw), ("outcome", outcome)]:
        if value not in _IMPLEMENTED[name]:
            raise NotImplementedError(f"{name}={value!r} is not implemented yet")
    if aipw and ipw_type != "HT":
        raise ValueError("ipw_type applies to IPW only; leave it at 'HT' with aipw=True")
    fedavg = {**FEDAVG_MW_DEFAULTS, **(fedavg_params or {})}
    if fedavg["B_batch_size"] is not None:
        raise NotImplementedError("FedAvg with minibatches (B_batch_size) is not supported")

    start = time.perf_counter()
    A = dgp_fed()
    data, x_cols, K = _to_authors_frame(df)
    ps_cols = x_cols + ["intercept"]
    mw_cols = x_cols + ["intercept"] if mw_intercept else x_cols

    e_local = _local_propensities(A, data, ps_cols, K)
    mw_rounds = mw_converged = None
    if mw == "logistic":
        omega = _site_order_numeric(
            A.MW_estimation_pooled(data, mw_cols, scale=False).to_numpy(), K
        )
    else:
        omega, mw_rounds, mw_converged = _fedavg_membership(
            A, data, mw_cols, K, fedavg["T_rounds"], fedavg["eta"],
            fedavg["E_local_steps"], fedavg["tol"], fedavg["standardize"],
        )
    e_hat = A.membership_weighting_vectorized(pd.DataFrame(omega), pd.DataFrame(e_local))
    data["e_hat"] = e_hat

    site_masks = [data["client"] == _label(k) for k in range(K)]
    n_k = np.array([mask.sum() for mask in site_masks])
    W, Y = data["W"].to_numpy(), data["Y"].to_numpy()
    outcome_coef = None
    if aipw:
        beta_1, beta_0 = _outcome_models(A, data, ps_cols, K, outcome,
                                         outcome_fedavg_params)
        outcome_coef = {"beta_1": pd.Series(beta_1, index=ps_cols),
                        "beta_0": pd.Series(beta_0, index=ps_cols)}
        X_out = data[ps_cols].to_numpy()
        data["mu1_hat"], data["mu0_hat"] = X_out @ beta_1, X_out @ beta_0
        tau_k = [
            A.aipw_function(data[mask], y1_col="mu1_hat", y0_col="mu0_hat", e_col="e_hat")
            for mask in site_masks
        ]
        tau = float(np.average(tau_k, weights=n_k))
        psi = _aipw_terms(W, Y, e_hat, data["mu1_hat"].to_numpy(),
                          data["mu0_hat"].to_numpy())
    elif ipw_type == "Hajek_pooled":
        tau = float(A.ipw_function(data, hat_e_name="e_hat", ht_or_hajek="Hajek"))
        all_rows = pd.Series(True, index=data.index)
        psi = _ipw_terms_hajek(W, Y, e_hat, [all_rows], [tau])
    else:
        tau_k = [
            A.ipw_function(data[mask], hat_e_name="e_hat", ht_or_hajek=ipw_type)
            for mask in site_masks
        ]
        tau = float(np.average(tau_k, weights=n_k))
        if ipw_type == "HT":
            psi = _ipw_terms_ht(W, Y, e_hat)
        else:
            psi = _ipw_terms_hajek(W, Y, e_hat, site_masks, tau_k)
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
        mw_fit=_MW_FIT[mw],
        mw_rounds=mw_rounds,
        mw_converged=mw_converged,
        outcome_fit=_OUTCOME_FIT[outcome] if aipw else None,
        outcome_coef=outcome_coef,
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


def _fedavg_membership(A, data, mw_cols, K, T_rounds, eta, E_local_steps, tol,
                       standardize):
    """FedAvg multinomial logistic regression for P(H|X), full-batch local steps.

    Same algorithm as the authors' federated_logistic_regression / MW_estimation
    (their gradient_multi_logistic and softmax; Theta starts at 0; each round, every
    site takes E full-batch gradient steps from the global Theta and the server averages
    with weights n_k/n), with three differences: classes are in numeric site order; the
    loop stops once ||Theta_t - Theta_{t-1}||_F / ||Theta_{t-1}||_F < tol; and with
    standardize=True the non-intercept columns are rescaled by their global mean/SD
    (unlike their scale=True, which fits unscaled and predicts scaled).
    Their per-step shuffle (df.sample) does not affect a full-batch gradient.

    Returns (omega (n, K), rounds run, converged); converged is None when tol is None.
    """
    X = data[mw_cols].to_numpy()
    if standardize:
        cols = [i for i, c in enumerate(mw_cols) if c != "intercept"]
        X = X.copy()
        X[:, cols] = (X[:, cols] - X[:, cols].mean(axis=0)) / X[:, cols].std(axis=0)
    H = np.eye(K)[[int(c.removeprefix("client")) - 1 for c in data["client"]]]
    sites = [H[:, k] == 1 for k in range(K)]
    n_k = np.array([s.sum() for s in sites])

    theta = np.zeros((K, X.shape[1]))
    converged = None if tol is None else False
    for t in range(1, T_rounds + 1):
        updates = []
        for s in sites:
            theta_k = theta
            for _ in range(E_local_steps):
                theta_k = theta_k - eta * A.gradient_multi_logistic(X[s], theta_k, H[s])
            updates.append(theta_k)
        theta_new = np.average(updates, weights=n_k, axis=0)
        change = np.linalg.norm(theta_new - theta) / max(np.linalg.norm(theta), 1e-300)
        theta = theta_new
        if tol is not None and change < tol:
            converged = True
            break
    return A.softmax(X @ theta.T), t, converged


def _outcome_models(A, data, out_cols, K, outcome, outcome_fedavg_params):
    """Linear outcome models per arm, shared by all sites: (beta_1, beta_0)."""
    if outcome == "ols":
        return tuple(
            np.asarray(A.estimate_beta_MLE(data[data["W"] == w], out_cols))
            for w in (1, 0)
        )
    params = {**FEDAVG_OUTCOME_NOTEBOOK, **(outcome_fedavg_params or {})}
    if params["B_batch_size"] is not None:
        raise NotImplementedError("FedAvg with minibatches (B_batch_size) is not supported")
    df_dict = {_label(k): data[data["client"] == _label(k)] for k in range(K)}
    return tuple(
        np.asarray(A.fedavg_sgd_server_side(
            df_dict, out_cols, beta1_or_beta0_or_gamma=arm, **params
        ))
        for arm in ("beta1", "beta0")
    )


def _aipw_terms(W, Y, e, mu1, mu0):
    return mu1 - mu0 + W * (Y - mu1) / e - (1 - W) * (Y - mu0) / (1 - e)


def _ipw_terms_ht(W, Y, e):
    return W * Y / e - (1 - W) * Y / (1 - e)


def _ipw_terms_hajek(W, Y, e, site_masks, tau_k):
    """Linearized per-unit terms of a Hajek estimator normalized within each group in
    `site_masks` (sites for the authors' "Hajek", one all-rows group for "Hajek_pooled").

    For unit i in site k: tau_k + (n_k/N1_k) W (Y - m1_k)/e - (n_k/N0_k) (1-W)(Y - m0_k)/(1-e),
    with N1_k = Σ_k W/e and m1_k the site's weighted treated mean (same for controls).
    The correction terms sum to zero within each site, so mean(psi) equals tau.
    """
    psi = np.empty(len(W))
    for mask, t_k in zip(site_masks, tau_k):
        m = mask.to_numpy()
        w1, w0 = W[m] / e[m], (1 - W[m]) / (1 - e[m])
        n_site = m.sum()
        m1 = np.sum(w1 * Y[m]) / np.sum(w1)
        m0 = np.sum(w0 * Y[m]) / np.sum(w0)
        psi[m] = (
            t_k
            + n_site / np.sum(w1) * w1 * (Y[m] - m1)
            - n_site / np.sum(w0) * w0 * (Y[m] - m0)
        )
    return psi


def _if_se(psi):
    if not np.all(np.isfinite(psi)):
        warnings.warn("non-finite per-unit terms (ê hit 0 or 1); se is NaN")
        return float("nan")
    return float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
