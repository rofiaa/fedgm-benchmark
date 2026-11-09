"""Centralized flexible-outcome references."""

import numpy as np
import pandas as pd
import pytest

from src.estimators import flex_outcome
from src.estimators.common import check_result


def _data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    Y = np.exp(X[:, 0] / 2) + X[:, 1] + W * (1 + X[:, 0]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["X1", "X2", "X3"]).assign(H=rng.integers(0, 2, n), W=W, Y=Y)
    return df, float(np.mean(1 + X[:, 0])), float(np.mean(1 + X[W == 1, 0]))


@pytest.mark.parametrize("target", ["ATE", "ATT"])
def test_targets(target):
    df, ate, att = _data()
    names = [n for n in flex_outcome.ESTIMANDS if target == "ATE" or n != "AIPW-RF-norm"]
    res = flex_outcome.estimate_multi(df, names, target=target, seed=1)
    truth = ate if target == "ATE" else att
    for name, out in res.items():
        check_result(out)
        assert abs(out["tau"] - truth) < 0.35, (name, out["tau"], truth)
    assert np.isfinite(res["AIPW-RF"]["se"]) and np.isnan(res["REG-RF"]["se"])


def test_att_rejects_norm():
    df, _, _ = _data(n=200)
    with pytest.raises(ValueError):
        flex_outcome.estimate_multi(df, ["AIPW-RF-norm"], target="ATT")


@pytest.mark.parametrize("target", ["ATE", "ATT"])
def test_dsm_rf(target):
    df, ate, att = _data()
    truth = ate if target == "ATE" else att
    res = flex_outcome.estimate_multi(df, list(flex_outcome.DSM_ESTIMANDS), target=target, seed=1)
    for name, out in res.items():
        check_result(out)
        assert np.isnan(out["se"]) and np.isfinite(out["max_smd"])
    for name in ("DSM-RF", "DSM-RF-rfps"):  # no-BA versions keep the matching bias
        assert abs(res[name]["tau"] - truth) < 0.35, (name, res[name]["tau"], truth)
    assert res["DSM-RF"]["tau"] != res["DSM-RF-noBA"]["tau"]
    assert res["DSM-RF"]["tau"] != res["DSM-RF-rfps"]["tau"]


def test_dsm_not_in_defaults():
    assert not set(flex_outcome.DSM_ESTIMANDS) & set(flex_outcome.ESTIMANDS)
