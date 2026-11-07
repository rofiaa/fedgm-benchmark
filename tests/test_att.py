"""ATT versions of the PS estimators (fed_ps, meta_sw, pooled_ps) and GenMatch's
target="ATT"."""

import numpy as np
import pandas as pd
import pytest

from src.estimators import fed_ps, meta_sw, pooled_ps
from src.estimators.common import check_result
from src.estimators.fed_ps import _att_terms


@pytest.fixture(scope="module")
def hetero_sites():
    """Two sites; τ(x) = 1 + 2 X1 and P(W=1|x) rising in X1 (logistic, well specified),
    so the ATT (≈ 1.9) is far from the ATE (= 1); outcome linear in X."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 3))
    H = rng.integers(0, 2, size=n)
    W = rng.binomial(1, 1 / (1 + np.exp(-(1.5 * X[:, 0] - 0.3))))
    Y = X.sum(axis=1) + W * (1 + 2 * X[:, 0]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["X1", "X2", "X3"]).assign(H=H, W=W, Y=Y)
    return df, float(np.mean(1 + 2 * X[W == 1, 0]))


def test_att_terms_by_hand():
    W = np.array([1, 1, 0, 0, 0, 1, 0])
    R = np.array([3.0, 5.0, 1.0, 2.0, 4.0, 6.0, 0.5])
    e = np.array([0.6, 0.5, 0.2, 0.5, 0.4, 0.7, 0.3])
    r = e / (1 - e)
    c = (1 - W) * r
    ht, _, psi_ht = _att_terms(W, R, e, [np.ones(7, bool)], normalize=False)
    assert ht == pytest.approx((np.sum(W * R) - np.sum(c * R)) / W.sum())
    hj, _, psi_hj = _att_terms(W, R, e, [np.ones(7, bool)], normalize=True)
    assert hj == pytest.approx(R[W == 1].mean() - np.sum(c * R) / np.sum(c))
    assert psi_ht.mean() == pytest.approx(ht) and psi_hj.mean() == pytest.approx(hj)
    # groups: treated-count weights, and mean(ψ) = τ̂ still
    groups = [np.arange(7) < 4, np.arange(7) >= 4]
    tau, tau_k, psi = _att_terms(W, R, e, groups, normalize=True)
    assert tau == pytest.approx((2 * tau_k[0] + 1 * tau_k[1]) / 3)
    assert psi.mean() == pytest.approx(tau)


@pytest.mark.parametrize("module, names", [
    (fed_ps, ["ATT-IPW-HT", "ATT-IPW-Hajek_pooled", "ATT-AIPW"]),
    (meta_sw, ["ATT-IPW-HT", "ATT-IPW-Hajek", "ATT-AIPW"]),
    (pooled_ps, ["ATT-IPW-HT", "ATT-IPW-Hajek_pooled", "ATT-AIPW"]),
])
def test_targets_the_att(hetero_sites, module, names):
    df, att = hetero_sites
    for name, out in module.estimate_multi(df, names, seed=1).items():
        check_result(out)
        assert abs(out["tau"] - att) < 4 * out["se"], name
        assert abs(out["tau"] - 1.0) > 0.5, name


def test_one_site_meta_equals_pooled(hetero_sites):
    """With a single site, Meta-SW's local ê and outcome model are the pooled ones."""
    df, _ = hetero_sites
    one = df.assign(H=0)
    m = meta_sw.estimate_multi(one, ["ATT-IPW-Hajek", "ATT-AIPW", "ATT-IPW-HT"])
    p = pooled_ps.estimate_multi(one, ["ATT-IPW-Hajek_pooled", "ATT-AIPW", "ATT-IPW-HT"])
    for a, b in [("ATT-IPW-Hajek", "ATT-IPW-Hajek_pooled"), ("ATT-AIPW", "ATT-AIPW"),
                 ("ATT-IPW-HT", "ATT-IPW-HT")]:
        assert m[a]["tau"] == pytest.approx(p[b]["tau"], rel=1e-10)
        assert m[a]["se"] == pytest.approx(p[b]["se"], rel=1e-10)
        assert m[a]["max_smd"] == pytest.approx(p[b]["max_smd"], rel=1e-10)


def test_att_balance_weights(hetero_sites):
    """ATT rows report balance under the ATT weights (1 treated, ê/(1-ê) controls)."""
    from src.estimators.balance import smd

    df, _ = hetero_sites
    out = pooled_ps.estimate_multi(df, ["IPW-HT", "ATT-IPW-HT"])
    e = out["ATT-IPW-HT"]["e_hat"].to_numpy()
    expected = smd(df[["X1", "X2", "X3"]], df.W, np.where(df.W == 1, 1, e / (1 - e)))
    pd.testing.assert_series_equal(out["ATT-IPW-HT"]["smd"], expected)
    assert out["IPW-HT"]["max_smd"] != out["ATT-IPW-HT"]["max_smd"]


def test_att_bootstrap_runs(hetero_sites):
    df, _ = hetero_sites
    out = fed_ps.estimate_multi(df, ["ATT-IPW-Hajek_pooled", "ATT-AIPW"], seed=3,
                                bootstrap=5, bootstrap_for=["ATT-IPW-Hajek_pooled"])
    assert out["ATT-IPW-Hajek_pooled"]["boot_ok"] == 5
    assert "se_boot" not in out["ATT-AIPW"]


def test_genmatch_targets_the_att():
    from src.estimators.genmatch import estimate_multi

    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 1 / (1 + np.exp(-1.5 * X[:, 0])))
    Y = X.sum(axis=1) + W * (1 + 2 * X[:, 0]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["X1", "X2", "X3"]).assign(W=W, Y=Y)
    att = float(np.mean(1 + 2 * X[W == 1, 0]))
    fast = {"pop_size": 20, "max_generations": 3, "wait_generations": 1,
            "hard_generation_limit": True}
    res = estimate_multi(df, ["GM-BA", "Maha-BA"], covariates="X", seed=1,
                         genmatch_params=fast, target="ATT")
    for name, out in res.items():
        check_result(out)
        assert out["target"] == "ATT"
        assert out["n_pairs"] >= W.sum()
        assert abs(out["tau"] - att) < 3 * out["se"], name
        assert abs(out["tau"] - 1.0) > 0.5, name
