"""Bret Nestor's federated genetic matching (Fed-GenMatch draft, 2026-11, Sec. 2.2), run
centrally, with a choice of the covariance normaliser Sigma (docs/fedgm_ablation.md).

Design (draft eqs. 2-13), single-arm ATT:
  - distance d^2(i, j) = (x_i - x_j)' S W S (x_i - x_j), S = Sigma^{-1/2} the symmetric
    inverse square root (eigendecomposition), W = diag(w), w in [1e-2, 1e2];
  - optimal 1:m assignment without replacement (each treated unit gets m distinct
    controls, each control is used at most once): linear_sum_assignment on the
    m-duplicated treated rows;
  - objective: max_l |SMD_l|, denominator sqrt((s_t^2 + s_c^2(W)) / 2) with s_c^2 over
    the matched controls (eq. 11);
  - search: GA with tournament selection, BLX-alpha crossover, Gaussian mutation clipped
    to the box, (N + N) elitism, stop after max_generations or after wait_generations
    without improvement. The draft leaves the GA's constants and the gene scale open;
    ours are GA_DEFAULTS, with genes log10(w);
  - estimate: mean treated Y minus mean matched-control Y (no bias adjustment).

Checked equation by equation in tests/test_fedgm.py (eq. 2 against a literal evaluation,
eq. 8 against brute-force enumeration, eq. 10-12 against sample moments, the GA's
best-seen and W = I start). Where the draft is silent or inconsistent, our choices:
  - genes are log10(w) (the draft does not say on which scale BLX and the mutation act);
    mutation hits each gene with probability 1/p, N(0, 0.25) in log10 units;
  - the search is the population GA of the text, not Algorithm 1's single-candidate
    accept-if-better loop;
  - S_t singular (a covariate constant among the treated): pseudo-inverse square root,
    so that direction does not enter the distance (the draft assumes S_t is PD);
  - SMD with a zero denominator: 0 if the means are equal, inf otherwise;
  - the effect is the ATT (treated mean minus matched-control mean); the draft's
    results tables say "ATE";
  - the DP variant (Laplace noise on B, C and the moments) is not implemented.

Without noise the federated protocol is lossless: the coordinator reassembles
D(W) = A - 2B + C exactly and pools the matched-control moments from per-site sums, so
running it centrally gives the federated matches. `federated_objective` implements the
per-site computation; tests/test_fedgm.py checks it against the central one.

Sigma (SIGMAS):
  "diag"            diag of the treated variances (floored): per-covariate scaling, as
                    Matching::GenMatch does, but with the treated SD
  "treated_raw"     S_t, the draft's choice; pseudo-inverse when singular (directions
                    with no treated variation do not enter the distance; sigma_rank)
  "treated_shrunk"  D^{1/2} R_alpha D^{1/2}: floored treated variances D, the treated
                    correlation R shrunk toward I by the Ledoit-Wolf weight alpha
  "pooled_within"   ((n_t - 1) S_t + (n_c - 1) S_c) / (n - 2): needs the control sites'
                    sums of x and x x' (one round)
Floor: a binary covariate's treated variance is p~(1 - p~), p~ = (sum x + 1)/(n_t + 2);
a continuous one is at least 1e-6 times its pooled variance.

Estimands: "FedGM" (searched w) and "FedMaha" (w = 1: the same normaliser, no search).
"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import ledoit_wolf_shrinkage

from src.estimators.balance import max_abs_smd

ESTIMANDS = ("FedGM", "FedMaha")
SIGMAS = ("diag", "treated_raw", "treated_shrunk", "pooled_within")
GA_DEFAULTS = {"pop_size": 100, "max_generations": 100, "wait_generations": 10,
               "tournament": 2, "blx_alpha": 0.5, "mutation_sd": 0.25,
               "log10_bounds": (-2.0, 2.0)}
_EIG_TOL = 1e-10


def _inv_sqrt(sigma):
    """Symmetric (pseudo-)inverse square root and the rank used."""
    vals, vecs = np.linalg.eigh((sigma + sigma.T) / 2)
    keep = vals > _EIG_TOL * max(vals.max(), _EIG_TOL)
    inv = np.zeros_like(vals)
    inv[keep] = 1 / np.sqrt(vals[keep])
    return (vecs * inv) @ vecs.T, int(keep.sum())


def _floored_treated_var(Xt, X):
    var = Xt.var(0, ddof=1)
    binary = np.all((X == 0) | (X == 1), axis=0)
    p = (Xt.sum(0) + 1) / (len(Xt) + 2)
    var = np.where(binary, p * (1 - p), var)
    return np.maximum(var, 1e-6 * X.var(0, ddof=1)), binary


def normaliser(Xt, Xc, sigma):
    """(Sigma^{-1/2}, info) for the arm `sigma` (see the module docstring)."""
    X = np.vstack([Xt, Xc])
    info = {}
    if sigma in ("diag", "treated_shrunk"):
        var, _ = _floored_treated_var(Xt, X)
        if sigma == "diag":
            cov = np.diag(var)
        else:
            Z = (Xt - Xt.mean(0)) / np.sqrt(var)
            R = np.cov(Z, rowvar=False)
            np.fill_diagonal(R, 1.0)
            alpha = float(ledoit_wolf_shrinkage(Z, assume_centered=True))
            info["shrinkage"] = alpha
            sd = np.sqrt(var)
            cov = ((1 - alpha) * R + alpha * np.eye(len(var))) * np.outer(sd, sd)
    elif sigma == "treated_raw":
        cov = np.cov(Xt, rowvar=False)
    elif sigma == "pooled_within":
        nt, nc = len(Xt), len(Xc)
        cov = ((nt - 1) * np.cov(Xt, rowvar=False) + (nc - 1) * np.cov(Xc, rowvar=False)) \
            / (nt + nc - 2)
    else:
        raise ValueError(f"sigma must be one of {SIGMAS}")
    S, rank = _inv_sqrt(cov)
    info["sigma_rank"] = rank
    return S, info


def distances(Tt, Tc, w):
    """Squared generalized Mahalanobis distances between whitened treated rows Tt and
    whitened control rows Tc (draft eq. 7: A - 2B + C)."""
    return (Tt ** 2 * w).sum(1)[:, None] - 2 * (Tt * w) @ Tc.T + (Tc ** 2 * w).sum(1)[None]


def assign(D, m):
    """Optimal 1:m matching without replacement: (n_t, m) control indices."""
    nt, nc = D.shape
    if m * nt > nc:
        raise ValueError(f"1:{m} matching needs {m * nt} controls, have {nc}")
    rows, cols = linear_sum_assignment(np.repeat(D, m, axis=0))
    out = np.empty(m * nt, dtype=int)
    out[rows] = cols
    return out.reshape(nt, m)


def bret_max_smd(Xt, Xc_matched):
    """max_l |SMD_l| with the draft's denominator (eq. 11)."""
    diff = Xt.mean(0) - Xc_matched.mean(0)
    den = np.sqrt((Xt.var(0, ddof=1) + Xc_matched.var(0, ddof=1)) / 2)
    return _max_ratio(diff, den)


def _max_ratio(diff, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > 1e-12, np.abs(diff) / den,
                     np.where(np.abs(diff) < 1e-12, 0.0, np.inf))
    return float(s.max())


def federated_objective(Xt, Xc_sites, S, w, m):
    """The draft's protocol, site by site: the trial site broadcasts the whitened
    treated rows and A; site k returns B_k, C_k; the coordinator assigns; site k returns
    the matched controls' (n_k, sum x, sum x^2); the coordinator pools (eq. 10-12).
    Returns (matched indices into the stacked controls, objective)."""
    Tt = Xt @ S
    A = (Tt ** 2 * w).sum(1)
    B = [(Tt * w) @ (Xk @ S).T for Xk in Xc_sites]
    C = [((Xk @ S) ** 2 * w).sum(1) for Xk in Xc_sites]
    D = A[:, None] - 2 * np.hstack(B) + np.concatenate(C)[None]
    idx = assign(D, m)
    offsets = np.cumsum([0] + [len(Xk) for Xk in Xc_sites])
    n_s = s_s = q_s = 0
    for k, Xk in enumerate(Xc_sites):
        local = idx.ravel()[(idx.ravel() >= offsets[k]) & (idx.ravel() < offsets[k + 1])]
        Mk = Xk[local - offsets[k]]
        n_s, s_s, q_s = n_s + len(Mk), s_s + Mk.sum(0), q_s + (Mk ** 2).sum(0)
    mean_c = s_s / n_s
    var_c = np.maximum(q_s / (n_s - 1) - n_s / (n_s - 1) * mean_c ** 2, 0.0)
    den = np.sqrt((Xt.var(0, ddof=1) + var_c) / 2)
    return idx, _max_ratio(Xt.mean(0) - mean_c, den)


def _search(fitness, p, rng, pop_size, max_generations, wait_generations, tournament,
            blx_alpha, mutation_sd, log10_bounds):
    """The draft's GA over log10(w). Returns (best genes, best fitness, generations,
    evaluations)."""
    lo, hi = log10_bounds
    pop = rng.uniform(lo, hi, (pop_size, p))
    pop[0] = 0.0  # W = I
    fit = np.array([fitness(g) for g in pop])
    n_evals, best, stall, gen = pop_size, fit.min(), 0, 0
    rate = 1.0 / p

    def pick():
        cand = rng.integers(pop_size, size=tournament)
        return pop[cand[np.argmin(fit[cand])]]

    for gen in range(1, max_generations + 1):
        kids = np.empty_like(pop)
        for k in range(pop_size):
            a, b = pick(), pick()
            low, high = np.minimum(a, b), np.maximum(a, b)
            child = rng.uniform(low - blx_alpha * (high - low), high + blx_alpha * (high - low))
            mut = rng.random(p) < rate
            child[mut] += rng.normal(0.0, mutation_sd, mut.sum())
            kids[k] = np.clip(child, lo, hi)
        kfit = np.array([fitness(g) for g in kids])
        n_evals += pop_size
        allp, allf = np.vstack([pop, kids]), np.concatenate([fit, kfit])
        order = np.argsort(allf, kind="stable")[:pop_size]
        pop, fit = allp[order], allf[order]
        if fit[0] < best - 1e-12:
            best, stall = fit[0], 0
        else:
            stall += 1
            if stall >= wait_generations:
                break
    return pop[0], float(fit[0]), gen, n_evals


def estimate(df, estimand="FedGM", **cfg):
    return estimate_multi(df, [estimand], **cfg)[estimand]


def estimate_multi(df, estimands=ESTIMANDS, sigma="treated_raw", m=1, seed=None,
                   ga_params=None, target="ATT"):
    """{estimand: result} for the ATT; one search serves "FedGM", "FedMaha" needs none.

    The default sigma is the draft's (S_t); docs/fedgm_ablation.md runs all four arms.

    df: X1..Xd, W, Y (H is ignored: the protocol's result does not depend on how the
    controls are split over sites). Extras: max_smd_bret (the objective), fedgm_w
    ({covariate: w}), sigma, m, sigma_rank, shrinkage, generations, n_evals,
    unit_weights."""
    t0 = time.perf_counter()
    if target != "ATT":
        raise ValueError("Fed-GenMatch is a single-arm (ATT) design")
    cols = [c for c in df if c[0] == "X" and c[1:].isdigit()]
    X = df[cols].to_numpy(float)
    W = df["W"].to_numpy()
    Y = df["Y"].to_numpy(float)
    Xt, Xc, Yt, Yc = X[W == 1], X[W == 0], Y[W == 1], Y[W == 0]
    S, info = normaliser(Xt, Xc, sigma)
    Tt, Tc = Xt @ S, Xc @ S
    p = len(cols)

    def run(w):
        idx = assign(distances(Tt, Tc, w), m)
        return idx, bret_max_smd(Xt, Xc[idx.ravel()])

    fits = {}
    if "FedMaha" in estimands:
        fits["FedMaha"] = (np.ones(p), 0, 1)
    if "FedGM" in estimands:
        params = dict(GA_DEFAULTS, **(ga_params or {}))
        genes, _, gens, n_evals = _search(lambda g: run(10.0 ** g)[1], p,
                                          np.random.default_rng(seed), **params)
        fits["FedGM"] = (10.0 ** genes, gens, n_evals)
    shared = time.perf_counter() - t0

    out = {}
    for est in estimands:
        w, gens, n_evals = fits[est]
        idx, obj = run(w)
        uw = np.zeros(len(df))
        ctrl_pos = np.flatnonzero(W == 0)
        uw[W == 1] = 1.0
        uw[ctrl_pos[idx.ravel()]] = 1.0
        out[est] = {
            "tau": float(Yt.mean() - Yc[idx.ravel()].mean()),
            "se": float("nan"), "se_method": "none",
            "max_smd": max_abs_smd(df[cols], W, uw),
            "runtime": shared,
            "max_smd_bret": obj, "fedgm_w": dict(zip(cols, map(float, w))),
            "sigma": sigma, "m": m, "generations": gens, "n_evals": n_evals,
            "unit_weights": pd.Series(uw, index=df.index, name="w_match"), **info,
        }
    return out
