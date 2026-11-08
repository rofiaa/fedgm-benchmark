"""FedDSM (src/estimators/feddsm.py): exact federated fits, targets, single-arm sites."""

import numpy as np
import pandas as pd
import pytest

from src.estimators import feddsm
from src.estimators.common import check_result


def _sites_data(n=3000, seed=0, control_only_site=False):
    """Three sites; τ(x) = 1 + X1 and P(W=1|x) rising in X1, so the ATT (≈ 1.5) differs
    from the ATE (≈ 1); quadratic outcome (the prognostic model is well specified)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    H = rng.integers(0, 3, size=n)
    W = rng.binomial(1, 1 / (1 + np.exp(-(1.0 * X[:, 0] - 0.5 * X[:, 1]))))
    if control_only_site:
        W[H == 1] = 0
    Y = X.sum(axis=1) + 0.5 * X[:, 1] ** 2 + W * (1 + X[:, 0]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["X1", "X2", "X3"]).assign(H=H, W=W, Y=Y)
    return df, float(np.mean(1 + X[:, 0])), float(np.mean(1 + X[W == 1, 0]))


def test_fed_quad_is_the_centralized_fit():
    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(size=(n, 3))
    H, W = rng.integers(0, 3, n), rng.binomial(1, 0.4, n)
    Y = X[:, 0] ** 2 + rng.normal(size=n)
    B = feddsm._basis(X, [0, 1, 2])
    pred, fold = feddsm.fed_quad(B, Y, W, H, arm=0, seed=3)
    central = B @ np.linalg.lstsq(B[W == 0], Y[W == 0], rcond=None)[0]
    for f in range(5):
        tr = (W == 0) & (fold != f)
        central[fold == f] = B[fold == f] @ np.linalg.lstsq(B[tr], Y[tr], rcond=None)[0]
    np.testing.assert_allclose(pred, central, rtol=1e-8, atol=1e-8)
    assert set(np.unique(fold[W == 0])) == set(range(5)) and (fold[W == 1] == -1).all()


def test_fed_newton_is_the_pooled_mle():
    import statsmodels.api as sm
    df, _, _ = _sites_data(n=1500)
    X = df[["X1", "X2", "X3"]].to_numpy()
    lp, _ = feddsm.fed_logistic_newton(X, df.W.to_numpy(), df.H.to_numpy())
    fit = sm.Logit(df.W.to_numpy(), np.column_stack([X, np.ones(len(X))])).fit(disp=0, tol=1e-12)
    np.testing.assert_allclose(lp, np.column_stack([X, np.ones(len(X))]) @ fit.params,
                               atol=1e-7)


@pytest.mark.parametrize("ps", ["x", "site", "khellaf"])
def test_targets(ps):
    df, ate, att = _sites_data()
    for target, truth, other in (("ATE", ate, att), ("ATT", att, ate)):
        res = feddsm.estimate_multi(df, target=target, ps=ps, seed=2)
        for name, out in res.items():
            check_result(out)
            assert out["target"] == target and out["undefined_reason"] is None
            # without bias adjustment, M = 4 matching keeps some discrepancy bias
            tol = 0.3 if name == "FedDSM-noBA" else 0.15
            assert abs(out["tau"] - truth) < tol, (ps, target, name, out["tau"], truth)
        assert abs(res["FedDSM"]["tau"] - other) > 0.25


def test_site_confounded_single_arm_site():
    df, _, att = _sites_data(control_only_site=True)
    ate = feddsm.estimate_multi(df, target="ATE", ps="site", seed=1)
    assert all(np.isnan(r["tau"]) and "not identified" in r["undefined_reason"]
               for r in ate.values())
    out = feddsm.estimate_multi(df, ["FedDSM"], target="ATT", ps="site", seed=1)["FedDSM"]
    assert np.isfinite(out["tau"]) and out["n_active"] == int((df.H != 1).sum())
    # covariate-only PS: the site is not a confounder, every unit stays
    out = feddsm.estimate_multi(df, ["FedDSM"], target="ATE", ps="x", seed=1)["FedDSM"]
    assert np.isfinite(out["tau"]) and out["n_active"] == len(df)


def test_seed_reproducible():
    df, _, _ = _sites_data(n=800)
    a = feddsm.estimate(df, target="ATT", seed=5)
    b = feddsm.estimate(df, target="ATT", seed=5)
    assert a["tau"] == b["tau"]
