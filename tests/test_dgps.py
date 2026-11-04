"""Reproducibility of the authors' generator through src.dgps.authors.generate (F7)."""

import random

import numpy as np
import pandas as pd
import pytest

from src.dgps.authors import _mvn_cholesky, dgp_a_params, dgp_b_params, generate


@pytest.mark.parametrize("dgp, make", [("A", dgp_a_params), ("B", dgp_b_params)])
def test_same_seed_same_data(dgp, make):
    params = make("poor", **({"n_k": 100} if dgp == "A" else {"n": 300}))
    a, b = generate(params, dgp, 7)[1], generate(params, dgp, 7)[1]
    pd.testing.assert_frame_equal(a, b)
    assert not generate(params, dgp, 8)[1].equals(a)


def test_global_rng_states_restored():
    np.random.seed(1)
    random.seed(1)
    np_before, py_before = np.random.random(), random.random()
    np.random.seed(1)
    random.seed(1)
    generate(dgp_b_params("poor", n=200), "B", 3)
    assert np.random.random() == np_before and random.random() == py_before


def test_cholesky_draw_keeps_the_rng_stream_aligned():
    """Same number of standard normals as the legacy sampler, so later draws match."""
    cov = 0.6 * np.eye(4) + 0.4  # repeated eigenvalue, as in DGP A
    np.random.seed(0)
    np.random.multivariate_normal(np.zeros(4), cov, size=5)
    after_legacy = np.random.random()
    np.random.seed(0)
    x = _mvn_cholesky(np.zeros(4), cov, 5)
    assert np.random.random() == after_legacy
    assert x.shape == (5, 4) and _mvn_cholesky(np.zeros(4), cov).shape == (4,)


def test_cholesky_draw_has_the_right_distribution():
    cov = 3 * np.eye(3) + 0.3
    mean = np.array([2.0, 2.0, 1.0])
    np.random.seed(0)
    x = _mvn_cholesky(mean, cov, 200_000)
    np.testing.assert_allclose(x.mean(axis=0), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(x.T), cov, atol=0.05)
