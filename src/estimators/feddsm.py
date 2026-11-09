"""Federated double-score matching (FedDSM): our proposed method (docs/findings.md N5-N7).

Match on low-dimensional scores instead of the covariates (double-score matching;
Leacy & Stuart 2014, Antonelli et al. 2018), with every model fitted exactly from
statistics the sites can share:

  ê(x):   logistic propensity score on the confounder set, fitted by distributed
          Newton-Raphson (each site sends its gradient and Hessian sums per iteration:
          the exact pooled MLE). ps="x": covariates only (the site is not a
          confounder, as in Khellaf et al.'s DGPs); ps="site": covariates + site
          indicators (the site is a confounder, as in IHDP); ps="khellaf": the
          federated PS of fed_ps (local logistic ê_k, FedAvg membership weights).
  m̂_w(x): prognostic score for arm w: least squares on main effects, squares and
          pairwise products of the continuous covariates (+ site indicators when
          ps="site") and an intercept -- the products only while this basis has at
          most half as many columns as arm w has units (otherwise main effects +
          squares; rule fixed before the ACIC 2016 run, and every earlier dataset
          keeps its products), from summed per-site, per-fold B'B and B'Y:
          the exact pooled fit. Cross-fitted: each site splits its arm-w units into
          `folds` folds; a unit's score comes from the fit without its fold, so no
          unit's own outcome enters its score. Units of the other arm get the full fit.
  matching: Mahalanobis nearest neighbours with replacement (Matching::Match,
          Weight=2, ties) on V = (logit ê, m̂_0) for the ATT and (logit ê, m̂_0, m̂_1)
          for the ATE (ATT matches of the treated + ATC matches of the controls,
          combined with weights n1/n and n0/n). M = 4 by default.
  bias adjustment: each pair's outcome difference is corrected by the same prognostic
          model, m̂(X_i) - m̂(X_j).

Prespecified settings (fixed before the cross-dataset verification): M = 4, 5 folds,
quadratic prognostic model, bias adjustment by the same model. ps follows the
dataset's confounding assumption.

Estimands: "FedDSM" (with bias adjustment), "FedDSM-noBA", and "FedAIPW-quad", AIPW
with the same ê and m̂ (fully federated; the like-for-like weighting comparator).
target: "ATE" or "ATT". With ps="site", sites with a single arm make the target
undefined where the site-confounding assumption leaves units without comparable
counterparts (ATE: any single-arm site; ATT: a treated-only site); control-only sites
drop out of the ATT (they can never be matched). Undefined -> tau NaN with
`undefined_reason`.

score="tuned" (post-hoc, docs/findings.md N12; default "ols" is the prespecified model):
the prognostic model is chosen per arm over basis x ridge penalty by held-out squared
error, from the same per-site, per-fold sums plus Y'Y per cell (fed_tuned). Bases: main
effects, + squares, + products; lambda / n on standardized columns in SCORE_LAMBDAS; the
intercept is not penalized. A unit's score still excludes its fold's outcomes, except
through this one choice of (basis, lambda).

No standard error yet (se = NaN): matching on estimated scores needs an adjusted
variance; inference is future work.
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.estimators.balance import smd
from src.r_env import ensure_r_home

ESTIMANDS = ("FedDSM", "FedDSM-noBA", "FedAIPW-quad")
PS_MODELS = ("x", "site", "khellaf")
_SE_METHOD = "none (FedDSM inference not implemented yet)"
_CLIP = 1e-8
SCORE_LAMBDAS = np.logspace(-4, 2, 13)


def estimate(df, estimand="FedDSM", **cfg):
    return estimate_multi(df, [estimand], **cfg)[estimand]


def estimate_multi(df, estimands=ESTIMANDS, target="ATE", ps="x", M=4, folds=5,
                   quad_cols=None, seed=None, interactions="auto", score="ols"):
    """{estimand: result}. df: X1..Xd, H (sites 0..K-1), W, Y.

    quad_cols: covariates whose squares and pairwise products enter the prognostic
        model (default: those with more than two distinct values).
    interactions: "auto" (products while the basis has <= n_arm / 2 columns), True or
        False.
    score: "ols" (least squares, the prespecified model) or "tuned" (fed_tuned); with
        "tuned", `interactions` is ignored and extras gain score_basis_{arm},
        score_lambda_{arm}.
    seed: fold assignment (np.random.SeedSequence(seed)); for ps="khellaf" also
        passed to fed_ps.
    Extras: target, ps_model, M, max_smd_x, max_smd_site (when ps="site"),
    undefined_reason, n_active, ps_iterations, n_basis_0 / n_basis_1,
    interactions_0 / interactions_1.
    """
    estimands = list(dict.fromkeys(estimands))
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        raise NotImplementedError(f"estimands {sorted(unknown)} are not implemented")
    if target not in ("ATE", "ATT"):
        raise ValueError("target must be 'ATE' or 'ATT'")
    if ps not in PS_MODELS:
        raise ValueError(f"ps must be one of {PS_MODELS}")
    if score not in ("ols", "tuned"):
        raise ValueError("score must be 'ols' or 'tuned'")
    start = time.perf_counter()
    x_cols = sorted((c for c in df.columns if c[0] == "X" and c[1:].isdigit()),
                    key=lambda c: int(c[1:]))
    X = df[x_cols].to_numpy(float)
    H = df["H"].to_numpy(int)
    W = df["W"].to_numpy(int)
    Y = df["Y"].to_numpy(float)
    K = int(H.max()) + 1
    if quad_cols is None:
        quad_cols = [c for c in x_cols if df[c].nunique() > 2]
    quad = [x_cols.index(c) for c in quad_cols]

    n_treated = np.bincount(H, weights=W, minlength=K)
    n_site = np.bincount(H, minlength=K)
    ctrl_only = [k for k in range(K) if n_treated[k] == 0]
    trt_only = [k for k in range(K) if n_treated[k] == n_site[k]]
    reason = None
    active = np.ones(len(W), bool)
    if ps == "site":
        if target == "ATE" and (ctrl_only or trt_only):
            reason = (f"site-confounded ATE not identified: single-arm sites "
                      f"{sorted(ctrl_only + trt_only)}")
        elif target == "ATT" and trt_only:
            reason = f"site-confounded ATT not identified: treated-only sites {trt_only}"
        active = ~np.isin(H, ctrl_only)
    extras = dict(target=target, ps_model=ps, M=M, undefined_reason=reason,
                  n_active=int(active.sum()))
    if reason is not None:
        return _undefined(estimands, extras, start)

    Xa, Ha, Wa, Ya = X[active], H[active], W[active], Y[active]
    sites_a = np.unique(Ha)
    D = (Ha[:, None] == sites_a[None, 1:]).astype(float)  # site indicators, first = base
    conf = np.column_stack([Xa, D]) if ps == "site" else Xa

    if ps == "khellaf":
        from src.estimators import fed_ps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = fed_ps.estimate_multi(df[x_cols + ["H", "W", "Y"]], ["IPW-HT"],
                                      mw="logistic_fedavg", seed=seed)["IPW-HT"]["e_hat"]
        e = np.clip(e.to_numpy()[active], _CLIP, 1 - _CLIP)
        lp, iters = np.log(e / (1 - e)), None
    else:
        lp, iters = fed_logistic_newton(conf, Wa, Ha)
        e = np.clip(1 / (1 + np.exp(-lp)), _CLIP, 1 - _CLIP)
        lp = np.log(e / (1 - e))

    fold_seed = np.random.SeedSequence(seed).generate_state(2)
    mu = {}
    extras["ps_iterations"] = iters
    for arm in ((0,) if target == "ATT" else (0, 1)):
        if score == "tuned":
            Dm = D if ps == "site" else None
            B = _basis(Xa, quad, Dm, interactions=True)
            blocks = _score_blocks(Xa.shape[1], len(quad), 0 if Dm is None else Dm.shape[1])
            mu[arm], _, info = fed_tuned(B, Ya, Wa, Ha, arm, seed=int(fold_seed[arm]),
                                         blocks=blocks, folds=folds)
            extras[f"n_basis_{arm}"], extras[f"interactions_{arm}"] = (
                len(blocks[info["basis"]]), info["basis"] == "quad")
            extras[f"score_basis_{arm}"], extras[f"score_lambda_{arm}"] = (
                info["basis"], info["lambda"])
            continue
        n_arm = int((Wa == arm).sum())  # an aggregate the sites can share
        use_int = interactions
        if interactions == "auto":
            use_int = _basis(Xa[:1], quad, None if ps != "site" else D[:1]).shape[1] <= n_arm / 2
        B = _basis(Xa, quad, D if ps == "site" else None, interactions=bool(use_int))
        mu[arm] = fed_quad(B, Ya, Wa, Ha, arm, seed=int(fold_seed[arm]), folds=folds)[0]
        extras[f"n_basis_{arm}"], extras[f"interactions_{arm}"] = B.shape[1], bool(use_int)

    results = {}
    if any(e_.startswith("FedDSM") for e_ in estimands):
        taus, unit_w = _match(lp, mu, Wa, Ya, Ha, target, M)
        bal = smd(pd.DataFrame(conf), Wa, unit_w).abs()
        for name, key in (("FedDSM", "quad"), ("FedDSM-noBA", "none")):
            if name in estimands:
                results[name] = dict(tau=float(taus[key]), se=float("nan"),
                                     se_method=_SE_METHOD, max_smd=float(bal.max()),
                                     **_balance_extras(bal, Xa.shape[1], ps), **extras)
    if "FedAIPW-quad" in estimands:
        if target == "ATT":
            from src.estimators.fed_ps import _att_terms
            tau = _att_terms(Wa, Ya - mu[0], e, [np.ones(len(Wa), bool)], normalize=True)[0]
            w = np.where(Wa == 1, 1.0, e / (1 - e))
        else:
            tau = float(np.mean(mu[1] - mu[0] + Wa * (Ya - mu[1]) / e
                                - (1 - Wa) * (Ya - mu[0]) / (1 - e)))
            w = np.where(Wa == 1, 1 / e, 1 / (1 - e))
        bal = smd(pd.DataFrame(conf), Wa, w).abs()
        results["FedAIPW-quad"] = dict(tau=float(tau), se=float("nan"), se_method=_SE_METHOD,
                                       max_smd=float(bal.max()),
                                       **_balance_extras(bal, Xa.shape[1], ps), **extras)
    runtime = time.perf_counter() - start
    for r in results.values():
        r["runtime"] = runtime
        r["runtime_shared_by"] = list(estimands)
    return results


def fed_logistic_newton(Xd, W, H, tol=1e-10, max_iter=100):
    """Logistic MLE of W on [Xd, 1] from per-site gradient / Hessian sums (distributed
    Newton-Raphson with step halving on the summed log-likelihood). Returns (linear
    predictor for every row, iterations)."""
    Xi = np.column_stack([Xd, np.ones(len(Xd))])
    sites = np.unique(H)

    def loglik(b):
        z = Xi @ b
        return float(np.sum(W * z - np.logaddexp(0, z)))

    beta = np.zeros(Xi.shape[1])
    ll = loglik(beta)
    for it in range(1, max_iter + 1):
        grad, hess = np.zeros_like(beta), np.zeros((len(beta), len(beta)))
        for k in sites:  # what site k sends
            r = H == k
            p = 1 / (1 + np.exp(-Xi[r] @ beta))
            grad += Xi[r].T @ (W[r] - p)
            hess += (Xi[r] * (p * (1 - p))[:, None]).T @ Xi[r]
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        t = 1.0
        while loglik(beta + t * step) < ll - 1e-12 and t > 1e-8:
            t /= 2
        beta = beta + t * step
        ll = loglik(beta)
        if np.max(np.abs(t * step)) < tol:
            break
    return Xi @ beta, it


def fed_quad(B, Y, W, H, arm, seed, folds=5):
    """Least squares of Y on the basis B among arm-`arm` units from summed per-site,
    per-fold sufficient statistics (B'B, B'Y), cross-fitted with site-wise folds.
    Returns (predictions for every row, fold labels; -1 for the other arm)."""
    n, p = B.shape
    fold = fold_labels(W, H, arm, seed, folds)
    G, g = np.zeros((folds, p, p)), np.zeros((folds, p))
    for k in np.unique(H):  # what site k sends: its per-fold B'B and B'Y
        for f in range(folds):
            r = (H == k) & (fold == f)
            G[f] += B[r].T @ B[r]
            g[f] += B[r].T @ Y[r]
    pred = B @ np.linalg.lstsq(G.sum(0), g.sum(0), rcond=None)[0]
    for f in range(folds):
        beta = np.linalg.lstsq(G.sum(0) - G[f], g.sum(0) - g[f], rcond=None)[0]
        pred[fold == f] = B[fold == f] @ beta
    return pred, fold


def _score_blocks(d, q, m):
    """Column indices of the nested bases within _basis(interactions=True), in order of
    size (a tie in held-out error goes to the simpler one)."""
    n_prod = q * (q - 1) // 2
    x, sq, prod = np.arange(d), np.arange(d, d + q), np.arange(d + q, d + q + n_prod)
    rest = np.arange(d + q + n_prod, d + q + n_prod + m + 1)  # site indicators, intercept
    return {"main": np.r_[x, rest], "squares": np.r_[x, sq, rest],
            "quad": np.r_[x, sq, prod, rest]}


def _ridge_path(G, g, lams):
    """Ridge coefficients (one column per lambda) from B'B and B'Y alone. The last column
    of B is the intercept (unpenalized); the others are centred and scaled to unit
    variance from the same sums; the penalty is lambda * n * ||scaled beta||^2."""
    n = G[-1, -1]
    xbar, ybar = G[:-1, -1] / n, g[-1] / n
    Gc = G[:-1, :-1] - n * np.outer(xbar, xbar)
    gc = g[:-1] - n * xbar * ybar
    sd = np.sqrt(np.maximum(np.diag(Gc), 0) / n)
    sd[sd < 1e-12] = np.inf  # constant in this training set: coefficient 0
    s, V = np.linalg.eigh(Gc / np.outer(sd, sd))
    a = V.T @ (gc / sd)
    beta = (V @ (a[:, None] / (np.maximum(s, 0)[:, None] + n * lams[None, :]))) / sd[:, None]
    return np.vstack([beta, ybar - xbar @ beta])


def fed_tuned(B, Y, W, H, arm, seed, blocks, folds=5, lams=SCORE_LAMBDAS):
    """Ridge prognostic model of arm `arm`, choosing (basis, lambda) by held-out squared
    error over the folds. Uses only per-site, per-fold sums: B'B, B'Y and Y'Y. B's last
    column must be the intercept and every block must end with it. Returns (predictions
    for every row, fold labels, {"basis", "lambda", "heldout_mse"}); units of the arm get
    the fit without their fold, the other arm the full fit."""
    fold = fold_labels(W, H, arm, seed, folds)
    p = B.shape[1]
    G, g, yy = np.zeros((folds, p, p)), np.zeros((folds, p)), np.zeros(folds)
    for k in np.unique(H):  # what site k sends
        for f in range(folds):
            r = (H == k) & (fold == f)
            G[f] += B[r].T @ B[r]
            g[f] += B[r].T @ Y[r]
            yy[f] += Y[r] @ Y[r]
    Gt, gt = G.sum(0), g.sum(0)
    best = dict(err=np.inf)
    for name, idx in blocks.items():
        ix = np.ix_(idx, idx)
        fits, err = [], np.zeros(len(lams))
        for f in range(folds):
            b = _ridge_path((Gt - G[f])[ix], (gt - g[f])[idx], lams)
            err += yy[f] - 2 * g[f, idx] @ b + np.sum(b * (G[f][ix] @ b), axis=0)
            fits.append(b)
        j = int(np.argmin(err))
        if err[j] < best["err"]:
            best = dict(err=float(err[j]), name=name, idx=idx, j=j, fits=fits)
    idx, j = best["idx"], best["j"]
    pred = B[:, idx] @ _ridge_path(Gt[np.ix_(idx, idx)], gt[idx], lams[[j]])[:, 0]
    for f in range(folds):
        pred[fold == f] = B[fold == f][:, idx] @ best["fits"][f][:, j]
    n_arm = int(Gt[-1, -1])
    return pred, fold, dict(basis=best["name"], **{"lambda": float(lams[j])},
                            heldout_mse=best["err"] / n_arm)


def fold_labels(W, H, arm, seed, folds=5):
    """Cross-fitting folds: each site splits its arm-`arm` units into `folds` folds
    (site order, one RNG); -1 for the other arm. These (site, arm, fold) cells are
    what fed_quad's per-site sufficient statistics are computed over."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(W), -1)
    for k in np.unique(H):
        rows = np.flatnonzero((W == arm) & (H == k))
        fold[rows] = rng.permutation(len(rows)) % folds
    return fold


def _basis(X, quad, D=None, interactions=True):
    C = X[:, quad]
    d = C.shape[1]
    cols = [X, C**2]
    if interactions:
        cols += [C[:, [i]] * C[:, [j]] for i in range(d) for j in range(i + 1, d)]
    if D is not None:
        cols.append(D)
    cols.append(np.ones((len(X), 1)))
    return np.column_stack(cols)


def _match(lp, mu, W, Y, H, target, M):
    """Mahalanobis matching on the scores; returns ({"none", "quad"}: τ̂, unit weights).

    τ̂ is computed in its outcome-local form (_local_tau): each site needs only the
    match weights of its own patients and sends two sums, so no outcome leaves a site.
    It equals the pair-difference form (_pair_tau) exactly (tests/test_feddsm.py)."""
    V = np.column_stack([lp, mu[0]] + ([mu[1]] if target == "ATE" else []))
    n, n1 = len(W), int(W.sum())
    unit_w = np.zeros(n)
    parts = {}
    for est in (("ATT",) if target == "ATT" else ("ATT", "ATC")):
        ti, ci, pw = _match_pairs(V, W, Y, est, M)
        np.add.at(unit_w, ti, pw)
        np.add.at(unit_w, ci, pw)
        m_ = mu[0] if est == "ATT" else mu[1]
        parts[est] = {"none": _local_tau(ti, ci, pw, W, Y, H, None, est),
                      "quad": _local_tau(ti, ci, pw, W, Y, H, m_, est)}
    if target == "ATT":
        return parts["ATT"], unit_w
    return {k: (n1 * parts["ATT"][k] + (n - n1) * parts["ATC"][k]) / n
            for k in ("none", "quad")}, unit_w


def _match_pairs(V, W, Y, est, M):
    """Matching::Match pairs (0-based treated index, control index, pair weight)."""
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    from src.estimators.genmatch import _r_matrix

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = importr("Matching").Match(Y=ro.FloatVector(Y), Tr=ro.FloatVector(W),
                                      X=_r_matrix(ro, V), estimand=est, M=M,
                                      replace=True, ties=True, Weight=2)
    return (np.asarray(m.rx2("index.treated"), int) - 1,
            np.asarray(m.rx2("index.control"), int) - 1,
            np.asarray(m.rx2("weights"), float))


def match_weights(ti, ci, pw, n, est):
    """(u, K): u_i = total pair weight of unit i as a target unit (the treated for the
    ATT, the controls for the ATC), K_j = its total weight as a match (donor)."""
    own, donor = (ti, ci) if est == "ATT" else (ci, ti)
    u, K = np.zeros(n), np.zeros(n)
    np.add.at(u, own, pw)
    np.add.at(K, donor, pw)
    return u, K


def _local_tau(ti, ci, pw, W, Y, H, m, est):
    """Outcome-local matching estimate. With ω_i = u_i + K_i and σ_i = +1 (treated) or
    −1 (control), τ̂ = Σ_k S_k / Σ_k N_k, where site k computes from its own patients
    S_k = Σ_{i in k} σ_i ω_i (Y_i − m(X_i)) and N_k = Σ_{i in k} u_i (m = 0 without
    bias adjustment). Sites need only their own patients' u and K."""
    u, K = match_weights(ti, ci, pw, len(W), est)
    r = Y if m is None else Y - m
    contrib = np.where(W == 1, 1.0, -1.0) * (u + K) * r
    S = np.bincount(H, weights=contrib)
    N = np.bincount(H, weights=u)
    return float(S.sum() / N.sum())


def _pair_tau(ti, ci, pw, Y, m):
    """Pair-difference form (needs the matched patients' outcomes in one place); kept
    for the equivalence test only."""
    d = Y[ti] - Y[ci]
    if m is not None:
        d = d - (m[ti] - m[ci])
    return float(np.sum(pw * d) / pw.sum())


def _balance_extras(bal, d, ps):
    out = {"max_smd_x": float(bal.iloc[:d].max())}
    if ps == "site":
        out["max_smd_site"] = float(bal.iloc[d:].max()) if len(bal) > d else float("nan")
    return out


def _undefined(estimands, extras, start):
    nan = float("nan")
    return {name: dict(tau=nan, se=nan, se_method=_SE_METHOD, max_smd=nan,
                       runtime=time.perf_counter() - start, **extras)
            for name in estimands}
