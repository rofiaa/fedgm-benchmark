"""src/privacy: feature maps, metrics, and the Gaussian prediction of the tracing attack."""

import numpy as np
import pytest

from src.privacy import mia, populations


@pytest.mark.parametrize("release", mia.RELEASES)
@pytest.mark.parametrize("d", [1, 3, 5])
def test_feature_counts(release, d):
    X = np.random.default_rng(0).normal(size=(7, d))
    assert mia.features(X, release).shape == (7, mia.n_features(release, d))
    assert np.allclose(mia.feature_mean(X, release), mia.features(X, release).mean(0))


def test_cov_release_gives_pooled_covariance():
    """(n, mean of x, mean of x xᵀ) determine the sample covariance exactly."""
    X = np.random.default_rng(1).normal(size=(50, 4))
    m = mia.feature_mean(X, "cov")
    mu, iu = m[:4], np.triu_indices(4)
    M2 = np.zeros((4, 4))
    M2[iu] = m[4:]
    M2 = M2 + M2.T - np.diag(np.diag(M2))
    S = (M2 - np.outer(mu, mu)) * 50 / 49
    assert np.allclose(S, np.cov(X.T))


def test_roc_metrics_extremes():
    rng = np.random.default_rng(2)
    same = mia.roc_metrics(rng.normal(size=4000), rng.normal(size=20000))
    assert same["auroc"] == pytest.approx(0.5, abs=0.02)
    assert same["tpr_at_0.01"] == pytest.approx(0.01, abs=0.005)
    sep = mia.roc_metrics(rng.normal(10, 1, 1000), rng.normal(size=5000))
    assert sep["auroc"] == 1.0 and sep["tpr_at_0.001"] == 1.0 and sep["advantage"] > 0.99


def test_gaussian_prediction():
    """The attack on a Gaussian population follows Φ(√(D/2n))."""
    pop = populations.make("gaussian", 5)
    rng = np.random.default_rng(3)
    att = mia.TracingAttack(pop.reference(20000, rng), "cov")
    assert att.d_eff == mia.n_features("cov", 5) == 20
    s_in, s_out = mia.run_cell(pop.draw_members, pop.draw_out, att, n=20, reps=150,
                               n_out=100, rng=rng)
    got = mia.roc_metrics(s_in, s_out)["auroc"]
    assert got == pytest.approx(mia.predicted(20, 20)["auroc_pred"], abs=0.03)


def test_binary_features_coincide():
    """For binary covariates x² = x, so mean_var has fewer effective features."""
    X = (np.random.default_rng(4).random((5000, 3)) < 0.3).astype(float)
    assert mia.TracingAttack(X, "mean_var").d_eff == 3


def test_rank_cutoff_is_scale_free():
    """Binary features next to a feature with a huge scale must not be dropped (IST: 31
    covariates, 3 continuous, give 31 means + 3 squares = 34 distinct numbers)."""
    from src.privacy import populations as pops
    att = mia.TracingAttack(pops.make("ist", 31).reference(), "mean_var")
    assert att.d_eff == 34
    rng = np.random.default_rng(6)
    X = np.c_[rng.normal(150, 20, 5000), rng.random((5000, 3)) < 0.3]
    assert mia.TracingAttack(X, "mean_var").d_eff == 1 + 1 + 3


def test_reproducible():
    def once():
        pop = populations.make("lognormal", 3)
        rng = np.random.default_rng(5)
        att = mia.TracingAttack(pop.reference(5000, rng), "mean_var")
        return mia.run_cell(pop.draw_members, pop.draw_out, att, 10, 5, 20, rng)
    a, b = once(), once()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
