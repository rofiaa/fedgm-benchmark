"""Smoke test: Matching::GenMatch + Match run through rpy2 (CRAN install, see CLAUDE.md)."""

import numpy as np

from src.r_env import ensure_r_home


def test_genmatch_runs_through_rpy2():
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    matching = importr("Matching")

    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 2))
    W = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    Y = X.sum(axis=1) + 2 * W + rng.normal(size=n)

    X_r = ro.r.matrix(ro.FloatVector(X.ravel(order="F")), nrow=n)
    W_r = ro.FloatVector(W)
    gen = matching.GenMatch(
        Tr=W_r, X=X_r, estimand="ATE",
        pop_size=10, max_generations=2, wait_generations=1, print_level=0,
        # forwarded to rgenoud::genoud via `...`, so rpy2 does not map "_" -> "."
        **{"unif.seed": 1, "int.seed": 2},
    )
    m = matching.Match(Y=ro.FloatVector(Y), Tr=W_r, X=X_r, estimand="ATE",
                       Weight_matrix=gen)

    est = m.rx2("est")[0]
    se = m.rx2("se")[0]
    assert np.isfinite(est) and np.isfinite(se) and se > 0
