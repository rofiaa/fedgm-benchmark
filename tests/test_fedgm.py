"""src/estimators/fedgm.py: Bret Nestor's Fed-GenMatch design, run centrally."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from src.estimators import fedgm
from src.estimators.common import check_result


def _single_arm(seed=0, nt=40, nc=400, p=5):
    rng = np.random.default_rng(seed)
    Xc = rng.normal(size=(nc, p))
    Xt = rng.normal(0.5, 0.8, size=(nt, p))
    Xt[:, 1] = Xt[:, 0] + 0.1 * rng.normal(size=nt)     # strongly correlated pair
    Xc[:, 1] = Xc[:, 0] + 0.5 * rng.normal(size=nc)
    Xt[:, 4] = Xc[:, 4] = 0.0
    Xc[: nc // 3, 4] = 1.0                               # binary, constant among treated
    X = np.vstack([Xt, Xc])
    W = np.r_[np.ones(nt, int), np.zeros(nc, int)]
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(1, p + 1)])
    df["H"] = np.r_[np.full(nt, 9), rng.integers(0, 3, nc)]
    df["W"] = W
    df["Y"] = X[:, 0] + 2.0 * W + rng.normal(size=len(W))
    return df


def _split(df):
    X = df[[c for c in df if c[0] == "X"]].to_numpy(float)
    return X[df.W == 1], X[df.W == 0]


@pytest.mark.parametrize("sigma", fedgm.SIGMAS)
def test_federated_protocol_equals_central(sigma):
    """Site-by-site pieces (A - 2B + C, pooled matched moments) give the central
    matches and objective, for any split of the controls over sites."""
    df = _single_arm()
    Xt, Xc = _split(df)
    S, _ = fedgm.normaliser(Xt, Xc, sigma)
    rng = np.random.default_rng(1)
    sites = [Xc[:100], Xc[100:130], Xc[130:]]
    for m in (1, 3):
        w = 10 ** rng.uniform(-2, 2, Xt.shape[1])
        idx = fedgm.assign(fedgm.distances(Xt @ S, Xc @ S, w), m)
        idx_fed, obj_fed = fedgm.federated_objective(Xt, sites, S, w, m)
        np.testing.assert_array_equal(idx, idx_fed)
        assert obj_fed == pytest.approx(fedgm.bret_max_smd(Xt, Xc[idx.ravel()]), rel=1e-9)


def test_assignment_is_optimal_one_to_m_without_replacement():
    rng = np.random.default_rng(2)
    D = rng.random((6, 20))
    idx = fedgm.assign(D, 3)
    assert idx.shape == (6, 3) and len(np.unique(idx)) == 18
    r, c = linear_sum_assignment(np.repeat(D, 3, axis=0))
    assert D[np.arange(6)[:, None], idx].sum() == pytest.approx(D[r // 3, c].sum())
    with pytest.raises(ValueError):
        fedgm.assign(D, 4)


def test_normalisers():
    df = _single_arm()
    Xt, Xc = _split(df)
    raw, info_raw = fedgm.normaliser(Xt, Xc, "treated_raw")
    assert info_raw["sigma_rank"] == Xt.shape[1] - 1     # X5 is constant among treated
    assert np.allclose(raw[:, 4], 0) and np.allclose(raw, raw.T)
    shr, info = fedgm.normaliser(Xt, Xc, "treated_shrunk")
    assert info["sigma_rank"] == Xt.shape[1] and 0 < info["shrinkage"] < 1
    diag, _ = fedgm.normaliser(Xt, Xc, "diag")
    assert np.allclose(diag, np.diag(np.diag(diag)))
    p = 1 / (len(Xt) + 2)                                # floored binary variance
    assert diag[4, 4] == pytest.approx(1 / np.sqrt(p * (1 - p)))
    assert diag[0, 0] == pytest.approx(1 / Xt[:, 0].std(ddof=1))


def test_estimate_multi():
    df = _single_arm()
    out = fedgm.estimate_multi(df, seed=3, sigma="treated_shrunk",
                               ga_params={"pop_size": 20, "max_generations": 8})
    for r in out.values():
        check_result(r)
        assert r["unit_weights"].sum() == 2 * (df.W == 1).sum()
    gm, maha = out["FedGM"], out["FedMaha"]
    assert gm["max_smd_bret"] <= maha["max_smd_bret"] + 1e-12   # W = I is in the population
    assert all(1e-2 - 1e-12 <= v <= 1e2 + 1e-12 for v in gm["fedgm_w"].values())
    again = fedgm.estimate_multi(df, seed=3, sigma="treated_shrunk",
                                 ga_params={"pop_size": 20, "max_generations": 8})
    assert again["FedGM"]["tau"] == gm["tau"] and again["FedGM"]["fedgm_w"] == gm["fedgm_w"]
    with pytest.raises(ValueError):
        fedgm.estimate_multi(df, target="ATE")


# ---- Equation-level checks against the draft (Bret_GM_derivation.tex, Sec. 2.1-2.2) ----

def test_eq2_distance_from_the_formula():
    """distances() equals eq. 2 squared, computed literally with Sigma^{-1/2} from an
    independent symmetric square root (scipy.linalg.sqrtm)."""
    from scipy.linalg import sqrtm
    rng = np.random.default_rng(10)
    p = 4
    A = rng.normal(size=(p, p))
    sigma = A @ A.T + p * np.eye(p)
    S, rank = fedgm._inv_sqrt(sigma)
    S_ref = np.linalg.inv(np.real(sqrtm(sigma)))
    assert rank == p and np.allclose(S, S.T) and np.allclose(S, S_ref)
    assert np.allclose(S @ sigma @ S, np.eye(p))
    Xt, Xc, w = rng.normal(size=(3, p)), rng.normal(size=(5, p)), 10 ** rng.uniform(-2, 2, p)
    D = fedgm.distances(Xt @ S, Xc @ S, w)
    for i in range(3):
        for j in range(5):
            v = Xt[i] - Xc[j]
            assert D[i, j] == pytest.approx(v @ S_ref.T @ np.diag(w) @ S_ref @ v, rel=1e-10)


@pytest.mark.parametrize("m", [1, 2])
def test_eq8_assignment_is_the_brute_force_optimum(m):
    """Eq. 8: minimum total squared distance over all 1:m matchings without replacement,
    by enumeration."""
    from itertools import permutations
    rng = np.random.default_rng(11 + m)
    nt, nc = 3, 6
    D = rng.random((nt, nc))
    best = min(sum(D[i, perm[m * i + r]] for i in range(nt) for r in range(m))
               for perm in permutations(range(nc), m * nt))
    idx = fedgm.assign(D, m)
    assert len(np.unique(idx)) == m * nt
    assert D[np.arange(nt)[:, None], idx].sum() == pytest.approx(best)


def test_eq10_to_12_pooled_smd():
    """Eq. 10: pooled moments equal the matched controls' sample mean and variance;
    eq. 11-12: SMD with sqrt((s_t^2 + s_c^2)/2), s_t^2 = [S_t]_ll (ddof 1)."""
    rng = np.random.default_rng(12)
    Xt, Mc = rng.normal(size=(20, 3)), rng.normal(1, 2, size=(20, 3))
    sites = [Mc[:7], Mc[7:12], Mc[12:]]
    n = sum(len(s) for s in sites)
    s_sum, q_sum = sum(s.sum(0) for s in sites), sum((s ** 2).sum(0) for s in sites)
    mean_c = s_sum / n
    var_c = q_sum / (n - 1) - n / (n - 1) * mean_c ** 2
    assert np.allclose(mean_c, Mc.mean(0)) and np.allclose(var_c, Mc.var(0, ddof=1))
    smd = (Xt.mean(0) - mean_c) / np.sqrt((np.cov(Xt, rowvar=False).diagonal() + var_c) / 2)
    assert fedgm.bret_max_smd(Xt, Mc) == pytest.approx(np.abs(smd).max())


def test_search_returns_best_seen_within_box_and_starts_at_identity():
    """The GA keeps W = I in the first population, stays in the box, and returns the best
    individual evaluated (elitism)."""
    rng = np.random.default_rng(13)
    target = rng.uniform(-2, 2, 4)
    seen = []

    def fitness(g):
        seen.append((g.copy(), float(((g - target) ** 2).sum())))
        return seen[-1][1]

    genes, best, gens, n_evals = fedgm._search(fitness, 4, np.random.default_rng(1),
                                               **dict(fedgm.GA_DEFAULTS, pop_size=12,
                                                      max_generations=15))
    assert np.array_equal(seen[0][0], np.zeros(4))
    assert all((g >= -2).all() and (g <= 2).all() for g, _ in seen)
    assert n_evals == len(seen) and best == min(f for _, f in seen)
    assert fitness(genes) == pytest.approx(best)


def test_estimate_is_treated_mean_minus_matched_control_mean():
    df = _single_arm(seed=14)
    out = fedgm.estimate_multi(df, ["FedMaha"])["FedMaha"]
    w = out["unit_weights"].to_numpy()
    t, c = (df.W == 1).to_numpy(), (df.W == 0).to_numpy() & (w > 0)
    assert out["sigma"] == "treated_raw"
    assert out["tau"] == pytest.approx(df.Y[t].mean() - df.Y[c].mean())
