import pytest
import numpy as np
import pandas as pd

from src.estimators.balance import max_abs_smd, smd


def test_smd_uses_pooled_unweighted_sd():
    X = pd.DataFrame({"X1": [0.0, 2.0, 1.0, 3.0]})
    W = np.array([0, 0, 1, 1])
    # means 1 vs 2, both arm variances 2 -> pooled SD sqrt(2)
    assert np.isclose(smd(X, W)["X1"], 1 / np.sqrt(2))


def test_weights_change_means_not_denominator():
    X = pd.DataFrame({"X1": [0.0, 2.0, 1.0, 3.0]})
    W = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 3.0, 1.0, 1.0])  # control mean -> 1.5
    assert np.isclose(max_abs_smd(X, W, weights), 0.5 / np.sqrt(2))


def test_balance_table_against_direct_formulas():
    from scipy.stats import ks_2samp

    from src.estimators.balance import balance_summary, balance_table

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.integers(0, 2, 400).astype(float)})
    W = (rng.random(400) < 0.4).astype(int)
    t = balance_table(X, W)
    np.testing.assert_allclose(t.smd_pooled, smd(X, W))
    assert t.smd_treated["a"] == pytest.approx(
        (X.a[W == 1].mean() - X.a[W == 0].mean()) / X.a[W == 1].std())
    assert t.var_ratio["a"] == pytest.approx(X.a[W == 1].var() / X.a[W == 0].var())
    assert np.isnan(t.var_ratio["b"]) and t.binary["b"]
    assert t.ks["a"] == pytest.approx(ks_2samp(X.a[W == 1], X.a[W == 0]).statistic)
    # integer weights = repeating rows
    w = rng.integers(1, 4, 400).astype(float)
    rep = np.repeat(np.arange(400), w.astype(int))
    tw, tr = balance_table(X, W, w), balance_table(X.iloc[rep], W[rep])
    assert tw.ks["a"] == pytest.approx(tr.ks["a"])
    s = balance_summary(t)
    assert s["max_smd_pooled"] == pytest.approx(t.smd_pooled.abs().max())
