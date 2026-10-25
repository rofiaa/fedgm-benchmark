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

ESTIMANDS = ("IPW-HT", "IPW-Hajek", "AIPW")
_IMPLEMENTED = {"local_ps": {"logistic", "rf"}}


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

    Returns the frozen keys (src/estimators/common.py) plus ci_if, estimand, tau_k (per
    site), n_k, undefined_reason, e_own, smd, local_ps_fit, ps_seed, outcome_fit, and
    the bootstrap extras of fed_ps.estimate when bootstrap > 0. se is the IF SE from the
    per-unit local terms (nuisances treated as fixed).
    """
    if aipw and ipw_type != "HT":
        raise ValueError("ipw_type applies to IPW only; leave it at 'HT' with aipw=True")
    name = "AIPW" if aipw else f"IPW-{ipw_type}"
    if name not in ESTIMANDS:
        raise NotImplementedError(f"ipw_type={ipw_type!r} is not implemented")
    return estimate_multi(df, [name], local_ps=local_ps, seed=seed,
                          bootstrap=bootstrap)[name]


def estimate_multi(df, estimands, local_ps="logistic", seed=None, bootstrap=0,
                   bootstrap_for=None):
    """Several Meta-SW estimators from one fit of the local nuisances.

    estimands: names from ESTIMANDS. bootstrap_for: estimands that get the bootstrap
    (default: all when bootstrap > 0). Returns {estimand: result} as in estimate();
    `runtime` covers the whole call and `runtime_shared_by` lists the estimands.
    """
    estimands = list(dict.fromkeys(estimands))
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        raise NotImplementedError(f"estimands {sorted(unknown)} are not implemented")
    if local_ps not in _IMPLEMENTED["local_ps"]:
        raise NotImplementedError(f"local_ps={local_ps!r} is not implemented")
    bootstrap_for = list(estimands) if bootstrap_for is None else list(bootstrap_for)
    if not set(bootstrap_for) <= set(estimands):
        raise ValueError("bootstrap_for must be a subset of estimands")
    if not bootstrap:
        bootstrap_for = []
    cfg = dict(local_ps=local_ps, ps_seed=_int_seed(seed, 32) if local_ps == "rf" else None)

    start = time.perf_counter()
    A = dgp_fed()
    data, x_cols, K = _to_authors_frame(df)
    fit = _fit(A, data, x_cols, K, estimands=estimands, **cfg)

    e_own = fit["e_own"]
    if fit["undefined_reason"] is None:
        weights = np.where(data["W"] == 1, 1 / e_own, 1 / (1 - e_own))
        balance = smd(data[x_cols], data["W"], weights)
        max_smd = float(balance.abs().max())
    else:
        balance, max_smd = None, float("nan")

    boot = {}
    if bootstrap_for:
        boot = _site_stratified_bootstrap(A, data, x_cols, K,
                                          dict(cfg, estimands=bootstrap_for),
                                          bootstrap, seed, fit=_fit_taus)
    runtime = time.perf_counter() - start
    results = {}
    for name in estimands:
        tau, se = fit["est"][name]["tau"], fit["est"][name]["se"]
        results[name] = dict(
            tau=tau,
            se=se,
            se_method="influence function, nuisances treated as fixed (provisional)",
            ci_if=(tau - _Z95 * se, tau + _Z95 * se),
            max_smd=max_smd,
            estimand=name,
            tau_k=dict(enumerate(fit["est"][name]["tau_k"])),
            n_k=dict(enumerate(fit["n_k"].tolist())),
            undefined_reason=fit["undefined_reason"],
            e_own=pd.Series(e_own, index=df.index, name="e_own"),
            smd=balance,
            local_ps_fit=local_ps,
            ps_seed=cfg["ps_seed"],
            outcome_fit="local-ols" if name == "AIPW" else None,
            **boot.get(name, {}),
            runtime=runtime,
            runtime_shared_by=list(estimands),
        )
    return results


def _fit(A, data, x_cols, K, estimands, local_ps, ps_seed):
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
        est = {name: dict(tau=nan, se=nan, tau_k=[nan] * K) for name in estimands}
        return dict(est=est, n_k=n_k, e_own=e_own,
                    undefined_reason=f"sites {single_arm} have a single treatment arm")

    data["e_own"] = e_own
    est = {}
    for name in estimands:
        if name == "AIPW":
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
                A.aipw_function(data[m], y1_col="mu1_hat", y0_col="mu0_hat", e_col="e_own")
                for m in site_masks
            ]
            psi = _aipw_terms(W, Y, e_own, mu1, mu0)
        else:
            ipw_type = name.removeprefix("IPW-")
            tau_k = [
                A.ipw_function(data[m], hat_e_name="e_own", ht_or_hajek=ipw_type)
                for m in site_masks
            ]
            if ipw_type == "HT":
                psi = _ipw_terms_ht(W, Y, e_own)
            else:
                psi = _ipw_terms_hajek(W, Y, e_own, site_masks, tau_k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            se = _if_se(psi)
        est[name] = dict(tau=float(np.average(tau_k, weights=n_k)), se=se,
                         tau_k=[float(t) for t in tau_k])
    return dict(est=est, n_k=n_k, e_own=e_own, undefined_reason=None)


def _fit_taus(A, data, x_cols, K, estimands, **cfg):
    """Bootstrap fit: {estimand: τ̂} (NaN when a site lost an arm)."""
    fit = _fit(A, data, x_cols, K, estimands=estimands, **cfg)
    return {name: fit["est"][name]["tau"] for name in estimands}
