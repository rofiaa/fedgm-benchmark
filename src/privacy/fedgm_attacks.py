"""Reconstruction attacks on the messages of Bret Nestor's Fed-GenMatch protocol
(docs/privacy_e2.md; protocol in src/estimators/fedgm.py).

Row convention: whitened rows z = x S (S = Sigma^{-1/2}, symmetric), so x = z S⁺ on the
range of S. For candidate weights w the cross terms of control j are M z_jᵀ with
M = (x̃ᵗ ⊙ w) (one row per treated patient).
"""

import numpy as np

from src.estimators import fedgm

_TOL = 1e-9


def recorded_search(Xt, Xc, sigma, m=1, seed=None, ga_params=None):
    """Run the draft's search exactly as fedgm.estimate_multi does, recording every
    evaluation: (S, info, [(w, matched control indices), ...])."""
    S, info = fedgm.normaliser(Xt, Xc, sigma)
    Tt, Tc = Xt @ S, Xc @ S
    log = []

    def fitness(g):
        w = 10.0 ** g
        idx = fedgm.assign(fedgm.distances(Tt, Tc, w), m).ravel()
        log.append((w, idx.copy()))
        return fedgm.bret_max_smd(Xt, Xc[idx])

    fedgm._search(fitness, Xt.shape[1], np.random.default_rng(seed),
                  **dict(fedgm.GA_DEFAULTS, **(ga_params or {})))
    return S, info, log


def _range_basis(S):
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    return vecs[:, np.abs(vals) > _TOL * np.abs(vals).max()]


def evals_to_full_exposure(Tt, S, ws):
    """Number of evaluations after which the stacked M spans the range of S (every
    control then recoverable exactly from the cross terms); None if never."""
    U = _range_basis(S)
    r, rows = U.shape[1], np.empty((0, U.shape[1]))
    for g, w in enumerate(ws, 1):
        rows = np.vstack([rows, (Tt * w) @ U])
        if np.linalg.matrix_rank(rows, tol=_TOL * max(1.0, np.abs(rows).max())) == r:
            return g
        if len(rows) > 4 * r:  # keep the stack small: an orthonormal basis of its row space
            _, sv, vt = np.linalg.svd(rows, full_matrices=False)
            rows = vt[sv > _TOL * sv.max()]
    return None


def recover_from_cross_terms(Tt, S, Xc, mu, ws, eps=None, rho=None, rng=None, chunk=300):
    """Least-squares reconstruction of the controls from the cross terms of the
    evaluations with weights `ws`; unseen directions stay at the population mean mu.

    eps: the draft's DP (eq. 14): Laplace noise per entry with scale Δ_i/ε,
    Δ_i = ‖w ⊙ x̃ᵗ_i‖₂ ‖S‖₂ ρ. The attacker then uses the posterior mean under its
    population prior z ~ N(μ_z, Σ_z) with a Gaussian approximation of the Laplace noise
    (variance 2 after scaling), so it never does worse than guessing the mean by much.
    Returns (X_hat, rank of the stacked M on the range of S)."""
    U = _range_basis(S)
    M = np.vstack([Tt * w for w in ws])
    mu_z = mu @ S
    Zd = Xc @ S - mu_z                                   # deviations, (n_c, p)
    if eps is None:
        G = np.linalg.pinv(M, rcond=1e-10)
        dz = (G @ (M @ Zd.T)).T
    else:
        scale = np.linalg.norm(M, axis=1) * np.linalg.norm(S, 2) * rho / eps
        Mw = M / scale[:, None]
        prior_prec = np.linalg.pinv(np.cov(Zd, rowvar=False), rcond=1e-10)
        G = np.linalg.pinv(Mw.T @ Mw / 2 + prior_prec, rcond=1e-10) @ Mw.T / 2
        dz = (G @ (Mw @ Zd.T)).T
        for a in range(0, len(Xc), chunk):
            L = rng.laplace(0.0, 1.0, (len(M), min(chunk, len(Xc) - a)))
            dz[a:a + chunk] += (G @ L).T
    X_hat = mu + dz @ np.linalg.pinv(S, rcond=1e-10)
    rank = int(np.linalg.matrix_rank(M @ U, tol=_TOL * max(1.0, np.abs(M).max())))
    return X_hat, rank


def match_agreement_under_dp(Tt, Tc, S, w, eps, rho, rng, m=1):
    """Utility side of the draft's DP: share of treated units whose matched controls are
    unchanged when the cross terms B carry the eq.-14 Laplace noise (C left exact, which
    favours the DP variant)."""
    clean = fedgm.assign(fedgm.distances(Tt, Tc, w), m)
    scale = np.linalg.norm(Tt * w, axis=1) * np.linalg.norm(S, 2) * rho / eps
    B = (Tt * w) @ Tc.T + rng.laplace(0.0, 1.0, (len(Tt), len(Tc))) * scale[:, None]
    D = (Tt ** 2 * w).sum(1)[:, None] - 2 * B + (Tc ** 2 * w).sum(1)[None]
    noisy = fedgm.assign(D, m)
    return float(np.mean([set(a) == set(b) for a, b in zip(clean, noisy)]))


def identified_by_moments(match_sets, n_controls):
    """Controls whose covariates the coordinator can solve exactly from the matched-set
    sums of one site: e_j in the row space of the (evaluations × controls) indicator.
    match_sets: iterable of index arrays (the site's matched controls per evaluation).
    Returns (identified mask over the site's controls, ever-matched mask)."""
    rows = {tuple(sorted(s)) for s in match_sets}
    ever = np.zeros(n_controls, bool)
    for r in rows:
        ever[list(r)] = True
    cols = np.flatnonzero(ever)
    if len(cols) == 0:
        return np.zeros(n_controls, bool), ever
    pos = {c: i for i, c in enumerate(cols)}
    E = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        E[i, [pos[c] for c in r]] = 1.0
    vals, vecs = np.linalg.eigh(E.T @ E)          # null space of E = that of EᵀE
    null = vecs[:, vals < 1e-9 * vals.max()]
    ident = np.zeros(n_controls, bool)
    ident[cols] = np.all(np.abs(null) < 1e-6, axis=1) if null.shape[1] else True
    return ident, ever


def recovery_metrics(X_hat, X, mu, sd, binary, site):
    """Per-cell summary of a reconstruction (see docs/privacy_e2.md)."""
    err = (X_hat - X) / sd
    base = (mu - X) / sd
    out = {
        "exact_share": float(np.mean(np.abs(err).max(1) < 1e-6)),
        "exact_value_share": float(np.mean(np.abs(err) < 1e-6)),
        "std_err_median": float(np.median(np.sqrt((err ** 2).mean(1)))),
        "std_err_baseline": float(np.median(np.sqrt((base ** 2).mean(1)))),
    }
    if binary.any():
        hat_b = (X_hat[:, binary] > 0.5).astype(float)
        out["binary_acc"] = float(np.mean(hat_b == X[:, binary]))
        out["binary_acc_baseline"] = float(np.mean((mu[binary] > 0.5) == X[:, binary]))
    hits = 0
    for s in np.unique(site):
        k = np.flatnonzero(site == s)
        A, B = X_hat[k] / sd, X[k] / sd
        d = (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None]
        hits += int((np.argmin(d, axis=1) == np.arange(len(k))).sum())
    out["reid_share"] = hits / len(X)
    out["reid_baseline"] = float(len(np.unique(site)) / len(X))
    return out
