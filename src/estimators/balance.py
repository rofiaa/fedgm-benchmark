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


def _wvar(x, w):
    m = np.average(x, axis=0, weights=w)
    return np.average((x - m) ** 2, axis=0, weights=w) * (w.sum() ** 2 / (w.sum() ** 2 - (w ** 2).sum()))


def _wks(x1, w1, x0, w0):
    """Weighted two-sample Kolmogorov-Smirnov statistic (max |F1 - F0|)."""
    grid = np.union1d(x1, x0)
    o1, o0 = np.argsort(x1), np.argsort(x0)
    c1 = np.concatenate([[0], np.cumsum(w1[o1])]) / w1.sum()
    c0 = np.concatenate([[0], np.cumsum(w0[o0])]) / w0.sum()
    F1 = c1[np.searchsorted(x1[o1], grid, side="right")]
    F0 = c0[np.searchsorted(x0[o0], grid, side="right")]
    return float(np.max(np.abs(F1 - F0)))


def balance_table(X, W, weights=None):
    """Per-covariate balance under per-unit weights (None = the raw sample).

    smd_pooled:  as `smd` (pooled SD of the unweighted sample, the same for all methods).
    smd_treated: same numerator over the treated SD of the unweighted sample (the ATT
                 convention, e.g. cobalt's default for the ATT).
    var_ratio:   weighted treated variance / weighted control variance (NaN for binary
                 covariates, where it is determined by the proportions).
    ks:          weighted Kolmogorov-Smirnov statistic.
    binary:      covariate takes two values.
    """
    X = pd.DataFrame(X)
    W = np.asarray(W)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=float)
    t, c = W == 1, W == 0
    A = X.to_numpy(float)
    diff = np.average(A[t], axis=0, weights=w[t]) - np.average(A[c], axis=0, weights=w[c])
    sd_t, sd_c = A[t].std(axis=0, ddof=1), A[c].std(axis=0, ddof=1)
    binary = np.array([len(np.unique(A[:, j])) <= 2 for j in range(A.shape[1])])
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = _wvar(A[t], w[t]) / _wvar(A[c], w[c])
        out = pd.DataFrame({
            "smd_pooled": diff / np.sqrt((sd_t ** 2 + sd_c ** 2) / 2),
            "smd_treated": diff / sd_t,
            "var_ratio": np.where(binary, np.nan, vr),
            "ks": [_wks(A[t, j], w[t], A[c, j], w[c]) for j in range(A.shape[1])],
            "binary": binary,
        }, index=X.columns)
    return out


def balance_summary(table):
    """One-row summary of `balance_table`: max |SMD| (both denominators), share of
    covariates with |SMD| > 0.1, worst variance ratio folded as max(VR, 1/VR) and the
    share outside [0.5, 2] (continuous covariates only), max KS."""
    cont = table[~table.binary]
    lvr = np.abs(np.log(cont.var_ratio))
    return dict(max_smd_pooled=float(table.smd_pooled.abs().max()),
                max_smd_treated=float(table.smd_treated.abs().replace(np.inf, np.nan).max()),
                share_smd_gt_0_1=float((table.smd_pooled.abs() > 0.1).mean()),
                worst_var_ratio=float(np.exp(lvr.max()) if len(cont) else np.nan),
                share_vr_outside=float(((cont.var_ratio < 0.5) | (cont.var_ratio > 2)).mean())
                if len(cont) else np.nan,
                max_ks=float(table.ks.max()))
