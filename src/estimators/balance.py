"""Covariate balance diagnostics shared by all estimators (weighting and matching)."""

import numpy as np
import pandas as pd


def smd(X, W, weights=None):
    """Standardized mean difference per covariate.

    Numerator: (weighted) treated mean minus (weighted) control mean.
    Denominator: pooled *unweighted* SD, sqrt((s1^2 + s0^2) / 2), computed on the
    full sample before weighting/matching, so it is the same for every method.

    X: DataFrame of covariates; W: 0/1 treatment; weights: per-unit weights
    (IPW weights, matching weights, ...) or None for the raw sample.
    """
    X = pd.DataFrame(X)
    W = np.asarray(W)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=float)
    treated, control = W == 1, W == 0

    denom = np.sqrt(
        (X[treated].var(ddof=1) + X[control].var(ddof=1)) / 2
    )
    mean_1 = np.average(X[treated], axis=0, weights=w[treated])
    mean_0 = np.average(X[control], axis=0, weights=w[control])
    return pd.Series(mean_1 - mean_0, index=X.columns) / denom


def max_abs_smd(X, W, weights=None):
    """Largest |SMD| over covariates (see `smd`)."""
    return float(np.max(np.abs(smd(X, W, weights))))
