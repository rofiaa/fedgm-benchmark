"""src/estimators/fedgm.py: Bret Nestor's Fed-GenMatch design, run centrally."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from src.estimators import fedgm
from src.estimators.common import check_result


def _single_arm(seed=0, nt=40, nc=400, p=5):
    rng = np.random.default_rng(seed)
    Xc = rng.normal(size=(nc, p))
    Xt = rng.normal(0.5, 0.8, size=(nt, p))
    Xt[:, 1] = Xt[:, 0] + 0.1 * rng.normal(size=nt)     # strongly correlated pair
    Xc[:, 1] = Xc[:, 0] + 0.5 * rng.normal(size=nc)
    Xt[:, 4] = Xc[:, 4] = 0.0
    Xc[: nc // 3, 4] = 1.0                               # binary, constant among treated
    X = np.vstack([Xt, Xc])
    W = np.r_[np.ones(nt, int), np.zeros(nc, int)]
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(1, p + 1)])
    df["H"] = np.r_[np.full(nt, 9), rng.integers(0, 3, nc)]
    df["W"] = W
    df["Y"] = X[:, 0] + 2.0 * W + rng.normal(size=len(W))
    return df


def _split(df):
    X = df[[c for c in df if c[0] == "X"]].to_numpy(float)
    return X[df.W == 1], X[df.W == 0]


@pytest.mark.parametrize("sigma", fedgm.SIGMAS)
def test_federated_protocol_equals_central(sigma):
    """Site-by-site pieces (A - 2B + C, pooled matched moments) give the central
    matches and objective, for any split of the controls over sites."""
    df = _single_arm()
    Xt, Xc = _split(df)
    S, _ = fedgm.normaliser(Xt, Xc, sigma)
    rng = np.random.default_rng(1)
    sites = [Xc[:100], Xc[100:130], Xc[130:]]
    for m in (1, 3):
        w = 10 ** rng.uniform(-2, 2, Xt.shape[1])
        idx = fedgm.assign(fedgm.distances(Xt @ S, Xc @ S, w), m)
        idx_fed, obj_fed = fedgm.federated_objective(Xt, sites, S, w, m)
        np.testing.assert_array_equal(idx, idx_fed)
        assert obj_fed == pytest.approx(fedgm.bret_max_smd(Xt, Xc[idx.ravel()]), rel=1e-9)


def test_assignment_is_optimal_one_to_m_without_replacement():
    rng = np.random.default_rng(2)
    D = rng.random((6, 20))
    idx = fedgm.assign(D, 3)
    assert idx.shape == (6, 3) and len(np.unique(idx)) == 18
    r, c = linear_sum_assignment(np.repeat(D, 3, axis=0))
    assert D[np.arange(6)[:, None], idx].sum() == pytest.approx(D[r // 3, c].sum())
    with pytest.raises(ValueError):
        fedgm.assign(D, 4)


def test_normalisers():
    df = _single_arm()
    Xt, Xc = _split(df)
    raw, info_raw = fedgm.normaliser(Xt, Xc, "treated_raw")
    assert info_raw["sigma_rank"] == Xt.shape[1] - 1     # X5 is constant among treated
    assert np.allclose(raw[:, 4], 0) and np.allclose(raw, raw.T)
    shr, info = fedgm.normaliser(Xt, Xc, "treated_shrunk")
    assert info["sigma_rank"] == Xt.shape[1] and 0 < info["shrinkage"] < 1
    diag, _ = fedgm.normaliser(Xt, Xc, "diag")
    assert np.allclose(diag, np.diag(np.diag(diag)))
    p = 1 / (len(Xt) + 2)                                # floored binary variance
    assert diag[4, 4] == pytest.approx(1 / np.sqrt(p * (1 - p)))
    assert diag[0, 0] == pytest.approx(1 / Xt[:, 0].std(ddof=1))


def test_estimate_multi():
    df = _single_arm()
    out = fedgm.estimate_multi(df, seed=3, sigma="treated_shrunk",
                               ga_params={"pop_size": 20, "max_generations": 8})
    for r in out.values():
        check_result(r)
        assert r["unit_weights"].sum() == 2 * (df.W == 1).sum()
    gm, maha = out["FedGM"], out["FedMaha"]
    assert gm["max_smd_bret"] <= maha["max_smd_bret"] + 1e-12   # W = I is in the population
    assert all(1e-2 - 1e-12 <= v <= 1e2 + 1e-12 for v in gm["fedgm_w"].values())
    again = fedgm.estimate_multi(df, seed=3, sigma="treated_shrunk",
                                 ga_params={"pop_size": 20, "max_generations": 8})
    assert again["FedGM"]["tau"] == gm["tau"] and again["FedGM"]["fedgm_w"] == gm["fedgm_w"]
    with pytest.raises(ValueError):
        fedgm.estimate_multi(df, target="ATE")
