"""Centralized genetic matching (Diamond & Sekhon, 2013) on the pooled data, via R's
Matching::GenMatch + Match through rpy2.

Settings follow the genmatch-lalonde pipeline (rofiaa/genmatch-lalonde, R/00_config.R,
R/04_matching.R), which follows D&S, except that the estimand is the ATE:
  - matching covariates: X1..Xd, plus the logit of the pooled logistic propensity
    score when covariates="X+ps" (pooled_ps.pooled_ps_logit: the same ê that
    pooled_ps weights with, so GM and pooled IPW differ only in matching vs weighting);
  - balance matrix (GenMatch's fitness): X, X², and all pairwise products X_i X_j
    (the propensity score is not included, as in genmatch-lalonde); balance_quad
    restricts the squares and products to some covariates (genmatch-lalonde's IHDP
    run: the 6 continuous ones);
  - fit.func="pvals" (lexical: smallest paired-t / KS p-value first), ks=TRUE,
    nboots=0, pop.size=1000, max.generations=100 (soft limit), wait.generations=10;
  - Match: M=1, with replacement, weighted ties, no caliper, Weight.matrix from
    GenMatch on the *same* X, estimand="ATE" in both calls.
The site H is not used: in the DGPs here the outcome depends on X only, so H affects
Y only through X and treatment (matching on X alone is valid; see docs/findings.md).

Estimands (one GenMatch search serves all of them):
  "GM"       GenMatch-weighted matching (D&S standard; no bias adjustment)
  "GM-BA"    the same matches with Abadie-Imbens regression bias adjustment on X
  "Maha"     plain Mahalanobis matching on the same covariates (no search)
  "Maha-BA"  Mahalanobis with bias adjustment
se: Abadie-Imbens SE returned by Match (Var.calc=0, homoskedastic; the DGPs' noise is
homoskedastic). It treats the GenMatch weights as fixed.
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.estimators.balance import smd
from src.estimators.pooled_ps import pooled_ps_logit
from src.r_env import ensure_r_home

ESTIMANDS = ("GM", "GM-BA", "Maha", "Maha-BA")

# genmatch-lalonde's R/00_config.R (the search budget there is an implementation choice,
# not a published D&S setting)
GENMATCH_DEFAULTS = {"pop_size": 1000, "max_generations": 100, "wait_generations": 10,
                     "hard_generation_limit": False}


def estimate(df, estimand="GM", **cfg):
    return estimate_multi(df, [estimand], **cfg)[estimand]


def estimate_multi(df, estimands=ESTIMANDS, covariates="X+ps", seed=None, cores=1,
                   genmatch_params=None, balance_quad=None):
    """{estimand: result}; one GenMatch search shared by "GM" and "GM-BA".

    df: columns X1..Xd, W (0/1), Y (H, if present, is ignored).
    covariates: "X" or "X+ps" (X plus the pooled logistic PS logit; D&S's choice).
    seed: GenMatch's unif.seed / int.seed are drawn from SeedSequence(seed)
        (None -> fresh entropy, returned as gm_seeds).
    cores: >1 runs the GenMatch search on a local R parallel cluster.
    genmatch_params: overrides of GENMATCH_DEFAULTS.
    balance_quad: covariate names whose squares and pairwise products enter the
        balance matrix (None = all of X1..Xd); every covariate enters linearly.

    Returns the frozen keys plus ci_ai (normal 95% CI), estimand, covariates, smd,
    unit_weights (each unit's total pair weight in the matched sample), n_eff (Kish effective sample size of the matched weights), n_pairs, gm_fitness
    (GenMatch's lexical fitness vector), gm_min_pvalue (its first entry: the smallest
    balance p-value after matching),
    gm_weights (learned diagonal weights, per matching covariate), gm_seeds,
    gm_settings, runtime_shared_by. runtime covers the whole call.
    """
    estimands = list(dict.fromkeys(estimands))
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        raise NotImplementedError(f"estimands {sorted(unknown)} are not implemented")
    if covariates not in ("X", "X+ps"):
        raise ValueError("covariates must be 'X' or 'X+ps'")
    gm = {**GENMATCH_DEFAULTS, **(genmatch_params or {})}
    seeds = np.random.SeedSequence(seed).generate_state(2) % (2**31 - 1) + 1
    unif_seed, int_seed = int(seeds[0]), int(seeds[1])

    start = time.perf_counter()
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    matching = importr("Matching")
    x_cols = sorted((c for c in df.columns if c[0] == "X" and c[1:].isdigit()),
                    key=lambda c: int(c[1:]))
    X = df[x_cols].to_numpy(dtype=float)
    W = df["W"].to_numpy(dtype=int)
    Y = df["Y"].to_numpy(dtype=float)
    if set(np.unique(W)) != {0, 1}:
        raise ValueError("W must contain both arms")
    match_cols = list(x_cols)
    X_match = X
    if covariates == "X+ps":
        X_match = np.column_stack([X, pooled_ps_logit(df)])
        match_cols.append("ps_logit")
    quad = None if balance_quad is None else [x_cols.index(c) for c in balance_quad]
    Z = X  # bias-adjustment regressors: the covariates (Match's default Z = X)

    r = dict(Tr=ro.FloatVector(W), X=_r_matrix(ro, X_match), estimand="ATE",
             M=1, replace=True, ties=True)
    y_r, z_r = ro.FloatVector(Y), _r_matrix(ro, Z)

    need_gm = any(e.startswith("GM") for e in estimands)
    gen = None
    if need_gm:
        cluster = False
        if cores > 1:
            cluster = ro.r("parallel::makeCluster")(cores)
        try:
            gen = matching.GenMatch(
                BalanceMatrix=_r_matrix(ro, _balance_matrix(X, quad)),
                pop_size=gm["pop_size"], max_generations=gm["max_generations"],
                wait_generations=gm["wait_generations"],
                hard_generation_limit=gm["hard_generation_limit"],
                fit_func="pvals", ks=True, nboots=0, print_level=0, balance=False,
                cluster=cluster,
                **r, **{"unif.seed": unif_seed, "int.seed": int_seed},
            )
        finally:
            if cores > 1:
                ro.r("parallel::stopCluster")(cluster)

    results = {}
    fits = {}
    for name in estimands:
        weight = {"Weight": 3, "Weight_matrix": gen} if name.startswith("GM") \
            else {"Weight": 2}
        bias_adjust = name.endswith("-BA")
        extra = {"BiasAdjust": True, "Z": z_r} if bias_adjust else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = matching.Match(Y=y_r, **r, **weight, **extra)
        fits[name] = m
    runtime = time.perf_counter() - start

    gm_fitness = gm_weights = None
    if gen is not None:
        gm_fitness = list(np.asarray(gen.rx2("value"), dtype=float))
        gm_weights = dict(zip(match_cols, np.diag(np.asarray(gen.rx2("Weight.matrix")))))
    quad_desc = "all" if balance_quad is None else "+".join(balance_quad)
    settings = dict(covariates=covariates,
                    balance_matrix=f"X; squares and products of {quad_desc}",
                    fit_func="pvals", ks=True, M=1, replace=True, ties=True, **gm)
    for name, m in fits.items():
        tau = float(m.rx2("est")[0])
        se = float(m.rx2("se")[0])
        unit_w = _unit_weights(m, len(W))
        balance = smd(df[x_cols], df["W"], unit_w.to_numpy())
        results[name] = dict(
            tau=tau,
            se=se,
            se_method="Abadie-Imbens (Matching::Match, Var.calc=0)"
                      + ("; GenMatch weights treated as fixed" if name.startswith("GM")
                         else ""),
            ci_ai=(tau - 1.959963984540054 * se, tau + 1.959963984540054 * se),
            max_smd=float(balance.abs().max()),
            estimand=name,
            covariates=covariates,
            smd=balance,
            unit_weights=pd.Series(unit_w.to_numpy(), index=df.index, name="w_match"),
            n_eff=float(unit_w.sum() ** 2 / np.sum(unit_w**2)),
            n_pairs=int(len(np.asarray(m.rx2("index.treated")))),
            gm_fitness=gm_fitness if name.startswith("GM") else None,
            gm_min_pvalue=gm_fitness[0] if name.startswith("GM") else None,
            gm_weights=gm_weights if name.startswith("GM") else None,
            gm_seeds=(unif_seed, int_seed) if name.startswith("GM") else None,
            gm_settings=settings if name.startswith("GM") else None,
            runtime=runtime,
            runtime_shared_by=list(estimands),
        )
    return results


def _r_matrix(ro, A):
    A = np.asarray(A, dtype=float)
    return ro.r.matrix(ro.FloatVector(A.ravel(order="F")), nrow=A.shape[0])


def _balance_matrix(X, quad=None):
    """X, plus the squares and pairwise products of the columns `quad` (default: all)
    (genmatch-lalonde's balance targets)."""
    C = X if quad is None else X[:, quad]
    d = C.shape[1]
    cols = [X, C**2] + [C[:, [i]] * C[:, [j]] for i in range(d) for j in range(i + 1, d)]
    return np.column_stack(cols)


def _unit_weights(m, n):
    """Per-unit weight in the matched sample: the sum of Match's pair weights over the
    pairs in which the unit appears (as itself or as a match), as in genmatch-lalonde's
    match_unit_weights."""
    w = np.zeros(n)
    pair_w = np.asarray(m.rx2("weights"), dtype=float)
    np.add.at(w, np.asarray(m.rx2("index.treated"), dtype=int) - 1, pair_w)
    np.add.at(w, np.asarray(m.rx2("index.control"), dtype=int) - 1, pair_w)
    return pd.Series(w)
