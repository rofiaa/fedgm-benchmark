"""Centralized propensity-score estimators on the pooled data (no federation).

One logistic propensity model ê(x) for P(W=1 | X) fitted on all sites' rows together
(the authors' compute_gamma on X + intercept, as for their local models), used in the
same IPW-HT / IPW-Hajek_pooled / AIPW estimators as fed_ps, with the same pooled OLS
outcome models for AIPW. It is the weighting counterpart of genmatch's covariates="X+ps"
(which matches on the logit of this same ê), so the two differ only in weighting vs
matching. In DGP A the pooled logistic model is misspecified (the true e*(x) =
Σ_k ω*_k(x) e*_k(x) is not logistic in x).
"""

import time

import numpy as np
import pandas as pd

from src.estimators._authors import dgp_fed
from src.estimators.fed_ps import (
    _Z95,
    _balance,
    _estimands,
    _needs_outcome,
    _outcome_models,
    _site_stratified_bootstrap,
    _target,
    _to_authors_frame,
)

ATE_ESTIMANDS = ("IPW-HT", "IPW-Hajek_pooled", "AIPW")  # the default set
ESTIMANDS = ATE_ESTIMANDS + ("ATT-IPW-HT", "ATT-IPW-Hajek_pooled", "ATT-AIPW")  # as fed_ps


def pooled_ps_logit(df):
    """Linear predictor [X, 1]·γ̂ of the pooled logistic propensity model (H, the site,
    is not used; a frame without H is accepted)."""
    A = dgp_fed()
    data, x_cols, _ = _to_authors_frame(df if "H" in df else df.assign(H=0))
    cols = x_cols + ["intercept"]
    return data[cols].to_numpy() @ np.asarray(A.compute_gamma(data, cols))


def estimate_multi(df, estimands=ATE_ESTIMANDS, seed=None, bootstrap=0, bootstrap_for=None):
    """{estimand: result}: frozen keys plus ci_if, estimand, e_hat, smd, ps_fit, and the
    bootstrap extras of fed_ps (site-stratified, ê and outcome models refit) when
    bootstrap > 0."""
    estimands = list(dict.fromkeys(estimands))
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        raise NotImplementedError(f"estimands {sorted(unknown)} are not implemented")
    bootstrap_for = list(estimands) if bootstrap_for is None else list(bootstrap_for)
    if not bootstrap:
        bootstrap_for = []

    start = time.perf_counter()
    A = dgp_fed()
    data, x_cols, K = _to_authors_frame(df)
    nuis = _nuisances(A, data, x_cols, K, need_outcome=_needs_outcome(estimands))
    fits = _estimands(A, data, x_cols, K, nuis, estimands)
    e_hat = nuis["e_hat"]
    balance = _balance(data, x_cols, e_hat, estimands)
    boot = {}
    if bootstrap_for:
        boot = _site_stratified_bootstrap(A, data, x_cols, K,
                                          dict(estimands=bootstrap_for), bootstrap,
                                          seed, fit=_fit_taus)
    runtime = time.perf_counter() - start
    out = {}
    for name in estimands:
        tau, se = fits[name]["tau"], fits[name]["se"]
        out[name] = dict(
            tau=tau,
            se=se,
            se_method="influence function, nuisances treated as fixed (provisional)",
            ci_if=(tau - _Z95 * se, tau + _Z95 * se),
            max_smd=float(balance[_target(name)].abs().max()),
            estimand=name,
            e_hat=pd.Series(e_hat, index=df.index, name="e_hat"),
            smd=balance[_target(name)],
            ps_fit="pooled-logistic",
            outcome_fit="pooled-ols" if name.endswith("AIPW") else None,
            **boot.get(name, {}),
            runtime=runtime,
            runtime_shared_by=list(estimands),
        )
    return out


def estimate(df, estimand="IPW-HT", **cfg):
    return estimate_multi(df, [estimand], **cfg)[estimand]


def _nuisances(A, data, x_cols, K, need_outcome):
    cols = x_cols + ["intercept"]
    e_hat = A.logistic_function_vectorized(data, cols, A.compute_gamma(data, cols))
    n_k = np.array([(data["client"] == f"client{k + 1}").sum() for k in range(K)])
    beta_1 = beta_0 = None
    if need_outcome:
        beta_1, beta_0 = _outcome_models(A, data, cols, K, "ols", None)
    return dict(e_hat=np.asarray(e_hat), n_k=n_k, beta_1=beta_1, beta_0=beta_0)


def _fit_taus(A, data, x_cols, K, estimands):
    nuis = _nuisances(A, data, x_cols, K, need_outcome=_needs_outcome(estimands))
    return {n: r["tau"] for n, r in _estimands(A, data, x_cols, K, nuis, estimands).items()}
