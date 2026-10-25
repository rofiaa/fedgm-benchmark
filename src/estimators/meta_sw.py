"""Meta-analysis with sample-size weights (Meta-SW; Khellaf et al., Def. 3).

Each site estimates its own ATE with its own propensity score ê_k (and, for AIPW, its
own linear outcome models), and the server averages: τ̂ = Σ_k (n_k / n) τ̂_k.
This is the authors' "Meta-SW" final estimator in Simulations_Fed.estimate_ipw
(local propensities, ipw_function per site) and estimate_aipw ("local" outcome
estimators, aipw_function per site), re-assembled from their standalone functions.

A site with a single treatment arm has no local estimate, so τ̂ is NaN (the site is
not dropped); `undefined_reason` says why.
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.estimators._authors import dgp_fed
from src.estimators.balance import smd
from src.estimators.fed_ps import (
    _Z95,
    _aipw_terms,
    _if_se,
    _int_seed,
    _ipw_terms_hajek,
    _ipw_terms_ht,
    _label,
    _local_propensities,
    _site_stratified_bootstrap,
    _to_authors_frame,
)

_IMPLEMENTED = {
    "local_ps": {"logistic", "rf"},
    "ipw_type": {"HT", "Hajek"},
    "aipw": {False, True},
}


def estimate(df, local_ps="logistic", ipw_type="HT", aipw=False, seed=None, bootstrap=0):
    """Meta-SW IPW / AIPW estimate of the ATE.

    df: columns X1..Xd, H (sites 0..K-1), W (0/1), Y.
    local_ps: "logistic" | "rf", as in fed_ps.estimate.
    ipw_type: "HT" or "Hajek". Here Hajek normalizes within each site with that site's
        own ê_k, which is consistent (unlike the global-ê case, docs/findings.md F2).
        IPW only; must be "HT" when aipw=True.
    aipw: local AIPW with per-site, per-arm OLS outcome models on X + intercept.
    seed: seeds RF local PS and the bootstrap, as in fed_ps.estimate.
    bootstrap: B site-stratified bootstrap replicates (0 = off); see
        fed_ps._site_stratified_bootstrap. Replicates in which a site loses an arm give
        τ̂_b = NaN and count as failed.

    Returns the frozen keys (src/estimators/common.py) plus ci_if, tau_k (per site),
    n_k, undefined_reason, smd, local_ps_fit, ps_seed, outcome_fit, and the bootstrap
    extras of fed_ps.estimate when bootstrap > 0. se is the IF SE from the per-unit
    local terms (nuisances treated as fixed).
    """
    for name, value in [("local_ps", local_ps), ("ipw_type", ipw_type), ("aipw", aipw)]:
        if value not in _IMPLEMENTED[name]:
            raise NotImplementedError(f"{name}={value!r} is not implemented")
    if aipw and ipw_type != "HT":
        raise ValueError("ipw_type applies to IPW only; leave it at 'HT' with aipw=True")
    cfg = dict(ipw_type=ipw_type, aipw=aipw, local_ps=local_ps,
               ps_seed=_int_seed(seed, 32) if local_ps == "rf" else None)

    start = time.perf_counter()
    A = dgp_fed()
    data, x_cols, K = _to_authors_frame(df)
    fit = _fit(A, data, x_cols, K, **cfg)

    e_own = fit["e_own"]
    if np.isfinite(fit["tau"]):
        weights = np.where(data["W"] == 1, 1 / e_own, 1 / (1 - e_own))
        balance = smd(data[x_cols], data["W"], weights)
        max_smd = float(balance.abs().max())
    else:
        balance, max_smd = None, float("nan")

    out = dict(
        tau=fit["tau"],
        se=fit["se"],
        se_method="influence function, nuisances treated as fixed (provisional)",
        ci_if=(fit["tau"] - _Z95 * fit["se"], fit["tau"] + _Z95 * fit["se"]),
        max_smd=max_smd,
        tau_k=dict(enumerate(fit["tau_k"])),
        n_k=dict(enumerate(fit["n_k"].tolist())),
        undefined_reason=fit["undefined_reason"],
        e_own=pd.Series(e_own, index=df.index, name="e_own"),
        smd=balance,
        local_ps_fit=local_ps,
        ps_seed=cfg["ps_seed"],
        outcome_fit="local-ols" if aipw else None,
    )
    if bootstrap:
        out.update(_site_stratified_bootstrap(A, data, x_cols, K, cfg, bootstrap, seed,
                                              fit=_fit))
    out["runtime"] = time.perf_counter() - start
    return out


def _fit(A, data, x_cols, K, ipw_type, aipw, local_ps, ps_seed):
    """Per-site estimates with local nuisances, aggregated with n_k / n."""
    data = data.copy()
    ps_cols = x_cols + ["intercept"]
    site_masks = [data["client"] == _label(k) for k in range(K)]
    n_k = np.array([mask.sum() for mask in site_masks])
    W, Y = data["W"].to_numpy(), data["Y"].to_numpy()

    single_arm = [k for k, m in enumerate(site_masks)
                  if len(np.unique(W[m.to_numpy()])) < 2]
    e_local = _local_propensities(A, data, ps_cols, K, local_ps, ps_seed)
    # each unit's own site's score
    e_own = e_local[np.arange(len(data)), [int(c[6:]) - 1 for c in data["client"]]]
    if single_arm:
        nan = float("nan")
        return dict(tau=nan, se=nan, tau_k=[nan] * K, n_k=n_k, e_own=e_own,
                    undefined_reason=f"sites {single_arm} have a single treatment arm")

    for k, mask in enumerate(site_masks):
        data.loc[mask, "e_own"] = e_local[mask.to_numpy(), k]
    if aipw:
        X_out = data[ps_cols].to_numpy()
        mu1, mu0 = np.empty(len(data)), np.empty(len(data))
        for mask in site_masks:
            site = data[mask]
            m = mask.to_numpy()
            beta_1 = np.asarray(A.estimate_beta_MLE(site[site["W"] == 1], ps_cols))
            beta_0 = np.asarray(A.estimate_beta_MLE(site[site["W"] == 0], ps_cols))
            mu1[m], mu0[m] = X_out[m] @ beta_1, X_out[m] @ beta_0
        data["mu1_hat"], data["mu0_hat"] = mu1, mu0
        tau_k = [
            A.aipw_function(data[mask], y1_col="mu1_hat", y0_col="mu0_hat", e_col="e_own")
            for mask in site_masks
        ]
        psi = _aipw_terms(W, Y, e_own, mu1, mu0)
    else:
        tau_k = [
            A.ipw_function(data[mask], hat_e_name="e_own", ht_or_hajek=ipw_type)
            for mask in site_masks
        ]
        if ipw_type == "HT":
            psi = _ipw_terms_ht(W, Y, e_own)
        else:
            psi = _ipw_terms_hajek(W, Y, e_own, site_masks, tau_k)

    tau = float(np.average(tau_k, weights=n_k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = _if_se(psi)
    return dict(tau=tau, se=se, tau_k=[float(t) for t in tau_k], n_k=n_k, e_own=e_own,
                undefined_reason=None)
