"""What FedDSM's messages reveal (docs/privacy_e2b.md): its releases on IST single-arm,
reproduced from src/estimators/feddsm.py, and their feature maps for the E1 tracing attack.

Rows of the attacked matrices are patients' (x, y): covariates then the outcome in the last
column, so one feature map can use both.
"""

import numpy as np

from src.estimators import feddsm


def ps_features(beta):
    """Feature map of a control site's final Newton release: p(1−p)·vech(x̃x̃ᵀ) (Hessian sum)
    and −p·x̃ (gradient sum for W = 0), x̃ = (x, 1), p = expit(x̃ᵀβ)."""
    def phi(XY):
        Xi = np.column_stack([XY[:, :-1], np.ones(len(XY))])
        p = 1 / (1 + np.exp(-Xi @ beta))
        iu = np.triu_indices(Xi.shape[1])
        H = (Xi[:, iu[0]] * Xi[:, iu[1]]) * (p * (1 - p))[:, None]
        return np.hstack([H, -p[:, None] * Xi])
    return phi


def prognostic_features(quad, interactions):
    """Feature map of a (site, fold) prognostic release: vech(b bᵀ) without the count, and
    b·y, on FedDSM's basis b(x)."""
    def phi(XY):
        B = feddsm._basis(XY[:, :-1], quad, None, interactions=interactions)
        iu = np.triu_indices(B.shape[1])
        BB = B[:, iu[0]] * B[:, iu[1]]
        return np.hstack([BB[:, :-1], B * XY[:, -1:]])  # last vech entry is 1·1 = the count
    return phi


def releases(df, seed):
    """FedDSM's internals on a single-arm frame (ps = "x", ATT), as estimate_multi computes
    them: (beta of the propensity fit, quad column indices, interactions flag, basis B,
    control fold labels, logit ê clipped as in FedDSM, cross-fitted m̂₀)."""
    x_cols = sorted((c for c in df if c[0] == "X" and c[1:].isdigit()), key=lambda c: int(c[1:]))
    X = df[x_cols].to_numpy(float)
    H, W, Y = df.H.to_numpy(int), df.W.to_numpy(int), df.Y.to_numpy(float)
    quad = [i for i, c in enumerate(x_cols) if df[c].nunique() > 2]
    lp_raw, _ = feddsm.fed_logistic_newton(X, W, H)
    Xi = np.column_stack([X, np.ones(len(X))])
    beta = np.linalg.lstsq(Xi, lp_raw, rcond=None)[0]
    e = np.clip(1 / (1 + np.exp(-lp_raw)), feddsm._CLIP, 1 - feddsm._CLIP)
    lp = np.log(e / (1 - e))
    n0 = int((W == 0).sum())
    use_int = feddsm._basis(X[:1], quad, None).shape[1] <= n0 / 2
    B = feddsm._basis(X, quad, None, interactions=use_int)
    fold_seed = int(np.random.SeedSequence(seed).generate_state(2)[0])
    mu0, fold = feddsm.fed_quad(B, Y, W, H, 0, seed=fold_seed)
    return dict(beta=beta, quad=quad, interactions=use_int, B=B, fold=fold, lp=lp, mu0=mu0)


def fold_fits(B, Y, W, fold, folds=5):
    """The 5 cross-fitted prognostic coefficient vectors (what the sums determine)."""
    c = W == 0
    G, g = B[c].T @ B[c], B[c].T @ Y[c]
    out = []
    for f in range(folds):
        r = c & (fold == f)
        out.append(np.linalg.lstsq(G - B[r].T @ B[r], g - B[r].T @ Y[r], rcond=None)[0])
    return np.array(out)
