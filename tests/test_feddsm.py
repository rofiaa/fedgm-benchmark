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


def test_interaction_rule_keeps_products_on_small_bases():
    """'auto' keeps the pairwise products whenever the basis has <= n_arm/2 columns
    (true for the authors' DGPs and IHDP), and drops them otherwise."""
    df, _, _ = _sites_data(n=600)
    a = feddsm.estimate_multi(df, ["FedDSM"], target="ATE", seed=1)["FedDSM"]
    b = feddsm.estimate_multi(df, ["FedDSM"], target="ATE", seed=1,
                              interactions=True)["FedDSM"]
    assert a["tau"] == b["tau"] and a["interactions_0"] and a["interactions_1"]
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(size=(600, 40)), columns=[f"X{i}" for i in range(1, 41)])
    wide = wide.assign(H=rng.integers(0, 2, 600), W=rng.binomial(1, 0.4, 600),
                       Y=rng.normal(size=600))
    c = feddsm.estimate_multi(wide, ["FedDSM"], target="ATT", seed=1)["FedDSM"]
    assert not c["interactions_0"] and c["n_basis_0"] == 40 + 40 + 1


@pytest.mark.parametrize("est", ["ATT", "ATC"])
def test_outcome_local_estimate_equals_pair_differences(est):
    """The outcome-local form (sites send two sums each) is the pair-difference
    estimate exactly, with and without bias adjustment."""
    df, _, _ = _sites_data(n=900, seed=3)
    W, Y, H = df.W.to_numpy(), df.Y.to_numpy(), df.H.to_numpy()
    rng = np.random.default_rng(4)
    V = np.column_stack([df.X1, df.X2 + rng.normal(scale=0.1, size=len(df))])
    m = df.X1.to_numpy() * 2 + rng.normal(size=len(df))
    for M in (1, 4):
        ti, ci, pw = feddsm._match_pairs(V, W, Y, est, M)
        for mm in (None, m):
            assert feddsm._local_tau(ti, ci, pw, W, Y, H, mm, est) == pytest.approx(
                feddsm._pair_tau(ti, ci, pw, Y, mm), rel=1e-12, abs=1e-12)
        u, K = feddsm.match_weights(ti, ci, pw, len(W), est)
        target = W == (1 if est == "ATT" else 0)
        np.testing.assert_allclose(u[target], 1.0)   # every target unit's weights sum to 1
        assert (u[~target] == 0).all() and (K[target] == 0).all()


def _tuned_data(linear, n=2500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    H, W = rng.integers(0, 3, n), rng.binomial(1, 0.4, n)
    Y = X[:, 0] + (0 if linear else 2 * X[:, 0] * X[:, 1]) + rng.normal(size=n)
    return X, H, W, Y


def test_ridge_path_is_centralized_ridge():
    X, H, W, Y = _tuned_data(False, n=400)
    B = feddsm._basis(X, list(range(6)))
    lam = np.array([0.01, 1.0])
    b = feddsm._ridge_path(B.T @ B, B.T @ Y, lam)
    A = B[:, :-1]
    sd, Ac = A.std(0), A - A.mean(0)
    for j, l in enumerate(lam):
        Z = Ac / sd
        beta = np.linalg.solve(Z.T @ Z + len(Y) * l * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean())) / sd
        np.testing.assert_allclose(b[:-1, j], beta, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(b[-1, j], Y.mean() - A.mean(0) @ beta, rtol=1e-8)


def test_tuned_heldout_error_and_folds_match_direct_computation():
    X, H, W, Y = _tuned_data(False)
    B = feddsm._basis(X, list(range(6)))
    blocks = feddsm._score_blocks(6, 6, 0)
    pred, fold, info = feddsm.fed_tuned(B, Y, W, H, 0, seed=2, blocks=blocks)
    r = W == 0
    assert info["heldout_mse"] == pytest.approx(np.mean((Y[r] - pred[r]) ** 2), rel=1e-8)
    idx, lam = blocks[info["basis"]], np.array([info["lambda"]])
    for f in range(5):  # no unit's own fold enters its prediction
        tr = r & (fold != f)
        b = feddsm._ridge_path(B[tr][:, idx].T @ B[tr][:, idx], B[tr][:, idx].T @ Y[tr], lam)[:, 0]
        np.testing.assert_allclose(pred[fold == f], B[fold == f][:, idx] @ b, rtol=1e-8)


@pytest.mark.parametrize("linear,expected", [(True, "main"), (False, "quad")])
def test_tuned_selects_the_right_basis(linear, expected):
    X, H, W, Y = _tuned_data(linear)
    B = feddsm._basis(X, list(range(6)))
    _, _, info = feddsm.fed_tuned(B, Y, W, H, 0, seed=2, blocks=feddsm._score_blocks(6, 6, 0))
    assert info["basis"] == expected


@pytest.mark.parametrize("target,idx", [("ATE", 1), ("ATT", 2)])
def test_tuned_score_estimates(target, idx):
    df, *truth = _sites_data()
    res = feddsm.estimate_multi(df, list(feddsm.ESTIMANDS), target=target, seed=1, score="tuned")
    for out in res.values():
        check_result(out)
        assert out["score_basis_0"] in ("main", "squares", "quad")
    assert abs(res["FedDSM"]["tau"] - truth[idx - 1]) < 0.2
    with pytest.raises(ValueError):
        feddsm.estimate_multi(df, target=target, score="nope")


def test_binned_blocks_and_selection_on_a_step_surface():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 6))
    H, W = rng.integers(0, 3, n), rng.binomial(1, 0.4, n)
    Y = 3.0 * (X[:, 0] > 0.5) - 2.0 * (X[:, 1] > -0.7) + rng.normal(size=n) * 0.5
    quad = list(range(6))
    edges = feddsm._bin_edges(X[:, quad])
    B = feddsm._basis_binned(X, quad, None, edges)
    n_bins = sum(map(len, edges))
    assert n_bins == 54 and B.shape[1] == 6 + 6 + 15 + 54 + 1
    blocks = feddsm._score_blocks(6, 6, 0, n_bins)
    np.testing.assert_array_equal(B[:, 27], (X[:, 0] > edges[0][0]).astype(float))
    np.testing.assert_array_equal(B[:, blocks["squares"]][:, 6:12], X**2)
    assert (B[:, -1] == 1).all() and blocks["bins"][-1] == B.shape[1] - 1
    _, _, info = feddsm.fed_tuned(B, Y, W, H, 0, seed=2, blocks=blocks)
    assert info["basis"] == "bins"


@pytest.mark.parametrize("target", ["ATE", "ATT"])
def test_binned_and_clip_estimands(target):
    df, ate, att = _sites_data()
    truth = ate if target == "ATE" else att
    names = list(feddsm.ESTIMANDS) + list(feddsm.EXTRA_ESTIMANDS)
    res = feddsm.estimate_multi(df, names, target=target, seed=1, score="binned")
    for name, out in res.items():
        check_result(out)
    assert abs(res["FedDSM"]["tau"] - truth) < 0.2 and abs(res["FedDSM-clip"]["tau"] - truth) < 0.2


def test_clip_changes_only_out_of_range_predictions():
    df, *_ = _sites_data(n=1500)
    a = feddsm.estimate_multi(df, ["FedDSM", "FedDSM-clip"], target="ATT", seed=1)
    # the outcome model rarely leaves the observed range here: clipping is nearly neutral
    assert abs(a["FedDSM"]["tau"] - a["FedDSM-clip"]["tau"]) < 0.05


def test_default_estimands_exclude_clip():
    assert "FedDSM-clip" not in feddsm.ESTIMANDS
