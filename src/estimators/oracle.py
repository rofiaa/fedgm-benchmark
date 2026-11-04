"""Centralized Oracle (Khellaf et al., Def. 1): IPW / AIPW with the true nuisances.

The authors' "Oracle" in Simulations_Fed.estimate_ipw / estimate_aipw: IPW with the
true global propensity e*(x) = Σ_k ω*_k(x) e*_k(x), and AIPW with e* and the true
potential outcomes Y1*, Y0* in place of outcome models (so AIPW-Oracle is essentially
the sample ATE). Not an estimator one can run on real data: it reads the true
nuisances that src.dgps.authors.generate attaches as df.attrs["oracle"].
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.estimators._authors import dgp_fed
from src.estimators.balance import smd
from src.estimators.fed_ps import _Z95, _aipw_terms, _if_se, _ipw_terms_hajek, _ipw_terms_ht

ESTIMANDS = ("IPW-HT", "IPW-Hajek_pooled", "AIPW")


def estimate_multi(df, estimands=ESTIMANDS, seed=None):
    """{estimand: result} with the frozen keys plus ci_if, estimand, e_hat.

    seed: unused (the Oracle is deterministic); accepted for the harness.
    se is the IF SE with the (true) nuisances fixed; max_smd uses the e*-IPW weights.
    """
    estimands = list(dict.fromkeys(estimands))
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        raise NotImplementedError(f"estimands {sorted(unknown)} are not implemented")
    if "oracle" not in df.attrs:
        raise ValueError("df has no attrs['oracle']; generate it with src.dgps.authors")
    start = time.perf_counter()
    A = dgp_fed()
    orc = df.attrs["oracle"]
    data = pd.DataFrame({"W": df["W"].to_numpy(), "Y": df["Y"].to_numpy(),
                         "e_star": orc["e_star"].to_numpy(),
                         "Y1*": orc["Y1*"].to_numpy(), "Y0*": orc["Y0*"].to_numpy()})
    W, Y, e = data["W"].to_numpy(), data["Y"].to_numpy(), data["e_star"].to_numpy()
    x_cols = [c for c in df.columns if c[0] == "X" and c[1:].isdigit()]
    balance = smd(df[x_cols], df["W"], np.where(W == 1, 1 / e, 1 / (1 - e)))

    est = {}
    for name in estimands:
        if name == "AIPW":
            tau = A.aipw_function(data, y1_col="Y1*", y0_col="Y0*", e_col="e_star")
            psi = _aipw_terms(W, Y, e, data["Y1*"].to_numpy(), data["Y0*"].to_numpy())
        elif name == "IPW-HT":
            tau = A.ipw_function(data, hat_e_name="e_star", ht_or_hajek="HT")
            psi = _ipw_terms_ht(W, Y, e)
        else:  # authors' "Hajek" on the whole sample = normalized over all sites
            tau = A.ipw_function(data, hat_e_name="e_star", ht_or_hajek="Hajek")
            psi = _ipw_terms_hajek(W, Y, e, [pd.Series(np.ones(len(W), bool))], [tau])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est[name] = (float(tau), _if_se(psi))
    runtime = time.perf_counter() - start

    return {
        name: dict(
            tau=tau,
            se=se,
            se_method="influence function, true nuisances",
            ci_if=(tau - _Z95 * se, tau + _Z95 * se),
            max_smd=float(balance.abs().max()),
            estimand=name,
            e_hat=pd.Series(e, index=df.index, name="e_star"),
            runtime=runtime,
            runtime_shared_by=list(estimands),
        )
        for name, (tau, se) in est.items()
    }


def estimate(df, estimand="IPW-HT"):
    return estimate_multi(df, [estimand])[estimand]
