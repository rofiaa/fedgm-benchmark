"""Centralized pooled-PS estimators and the PS shared with genmatch."""

import numpy as np
import pandas as pd

from src.estimators.common import check_result
from src.estimators.pooled_ps import estimate_multi, pooled_ps_logit


def _logistic_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + X @ [0.8, -0.5, 0.2]))))
    Y = X @ [1.0, 1.0, -1.0] + W * (2 + X[:, 0]) + rng.normal(size=n)
    return pd.DataFrame(X, columns=["X1", "X2", "X3"]).assign(
        H=rng.integers(0, 2, n), W=W, Y=Y)


def test_ps_is_the_one_genmatch_matches_on():
    df = _logistic_data()
    res = estimate_multi(df, ["IPW-HT"])["IPW-HT"]
    np.testing.assert_allclose(res["e_hat"], 1 / (1 + np.exp(-pooled_ps_logit(df))),
                               rtol=1e-12)
    # the site column plays no role
    np.testing.assert_allclose(pooled_ps_logit(df), pooled_ps_logit(df.drop(columns="H")))


def test_ht_closed_form_and_ate_recovery():
    df = _logistic_data()
    res = estimate_multi(df)
    e, W, Y = res["IPW-HT"]["e_hat"], df["W"], df["Y"]
    np.testing.assert_allclose(res["IPW-HT"]["tau"],
                               np.mean(W * Y / e - (1 - W) * Y / (1 - e)), rtol=1e-12)
    for name, out in res.items():
        check_result(out)
        assert abs(out["tau"] - 2.0) < 3 * out["se"], name


def test_bootstrap_runs():
    df = _logistic_data(n=800)
    out = estimate_multi(df, ["IPW-HT", "AIPW"], seed=1, bootstrap=5,
                         bootstrap_for=["IPW-HT"])
    assert out["IPW-HT"]["boot_ok"] == 5 and "se_boot" not in out["AIPW"]
