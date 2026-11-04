"""GenMatch wrapper: ATE (not ATT), seeding, index conversion, result contract."""

import numpy as np
import pandas as pd
import pytest

from src.estimators.common import check_result
from src.estimators.genmatch import _balance_matrix, estimate_multi

FAST = {"pop_size": 20, "max_generations": 3, "wait_generations": 1,
        "hard_generation_limit": True}


@pytest.fixture(scope="module")
def hetero():
    """τ(x) = 1 + 2 X1 and P(W=1|x) rising in X1, so the ATT (≈ 1.9) is far from the
    ATE (= 1)."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 1 / (1 + np.exp(-1.5 * X[:, 0])))
    Y = X.sum(axis=1) + W * (1 + 2 * X[:, 0]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["X1", "X2", "X3"]).assign(W=W, Y=Y)
    att = float(np.mean(1 + 2 * X[W == 1, 0]))
    return df, att


def test_targets_the_ate(hetero):
    df, att = hetero
    res = estimate_multi(df, ["GM-BA", "Maha-BA"], covariates="X", seed=1,
                         genmatch_params=FAST)
    for name, out in res.items():
        check_result(out)
        assert abs(out["tau"] - 1.0) < 3 * out["se"], name
        assert abs(out["tau"] - att) > 0.5, name


def test_same_seed_same_result(hetero):
    df, _ = hetero
    a = estimate_multi(df, ["GM"], seed=5, genmatch_params=FAST)["GM"]
    b = estimate_multi(df, ["GM"], seed=5, genmatch_params=FAST)["GM"]
    assert a["tau"] == b["tau"] and a["gm_weights"] == b["gm_weights"]
    assert a["gm_seeds"] == b["gm_seeds"]


def test_unit_weights_reproduce_the_estimate(hetero):
    """The unadjusted matching estimate = weighted mean difference over the matched
    sample; checks the 1-based R index conversion."""
    df, _ = hetero
    for name, out in estimate_multi(df, ["GM", "Maha"], seed=2,
                                    genmatch_params=FAST).items():
        u, W, Y = out["unit_weights"], df["W"], df["Y"]
        tau = (np.sum(u[W == 1] * Y[W == 1]) - np.sum(u[W == 0] * Y[W == 0])) \
            / np.sum(u[W == 1])
        np.testing.assert_allclose(tau, out["tau"], rtol=1e-10, err_msg=name)
        np.testing.assert_allclose(u[W == 1].sum(), u[W == 0].sum(), rtol=1e-10)


def test_ps_covariate_and_settings(hetero):
    df, _ = hetero
    out = estimate_multi(df, ["GM"], covariates="X+ps", seed=3, genmatch_params=FAST)["GM"]
    assert list(out["gm_weights"]) == ["X1", "X2", "X3", "ps_logit"]
    assert out["gm_settings"]["covariates"] == "X+ps" and out["max_smd"] >= 0


def test_balance_matrix_terms():
    X = np.arange(6.0).reshape(2, 3)
    B = _balance_matrix(X)
    assert B.shape == (2, 3 + 3 + 3)
    np.testing.assert_array_equal(B[:, 6], X[:, 0] * X[:, 1])
