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
