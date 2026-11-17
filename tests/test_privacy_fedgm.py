"""src/privacy/fedgm_attacks.py: reconstruction from Fed-GenMatch's messages."""

import numpy as np
import pytest

from src.estimators import fedgm
from src.privacy import fedgm_attacks as fa
from tests.test_fedgm import _single_arm


def _data(nt, nc=300, p=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.5, 1, (nt, p)), rng.normal(size=(nc, p))


def test_recorded_search_is_the_estimators_search():
    df = _single_arm(seed=20)
    X = df[[c for c in df if c[0] == "X"]].to_numpy(float)
    Xt, Xc = X[df.W == 1], X[df.W == 0]
    ga = {"pop_size": 12, "max_generations": 6}
    _, _, log = fa.recorded_search(Xt, Xc, "treated_shrunk", seed=4, ga_params=ga)
    out = fedgm.estimate_multi(df, ["FedGM"], sigma="treated_shrunk", seed=4,
                               ga_params=ga)["FedGM"]
    objs = [fedgm.bret_max_smd(Xt, Xc[idx]) for _, idx in log]
    assert len(log) == out["n_evals"] and min(objs) == pytest.approx(out["max_smd_bret"])


def test_one_evaluation_recovers_everything_when_nt_ge_p():
    Xt, Xc = _data(nt=10)
    S, _ = fedgm.normaliser(Xt, Xc, "treated_shrunk")
    X_hat, rank = fa.recover_from_cross_terms(Xt @ S, S, Xc, Xc.mean(0), [np.ones(6)])
    assert rank == 6 and np.allclose(X_hat, Xc, atol=1e-8)


def test_several_evaluations_recover_when_nt_lt_p():
    Xt, Xc = _data(nt=2)
    S, _ = fedgm.normaliser(Xt, Xc, "diag")      # full rank for any n_t
    ws = [10 ** np.random.default_rng(g).uniform(-2, 2, 6) for g in range(5)]
    X1, r1 = fa.recover_from_cross_terms(Xt @ S, S, Xc, Xc.mean(0), ws[:1])
    assert r1 == 2 and not np.allclose(X1, Xc, atol=1e-3)
    g = fa.evals_to_full_exposure(Xt @ S, S, ws)
    assert g == 3                                  # 2 + 2 + 2 >= 6 dimensions
    Xg, rg = fa.recover_from_cross_terms(Xt @ S, S, Xc, Xc.mean(0), ws[:g])
    assert rg == 6 and np.allclose(Xg, Xc, atol=1e-6)


def test_moments_identify_controls():
    sets = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [4, 5], [4, 5, 0]]
    ident, ever = fa.identified_by_moments([np.array(s) for s in sets], 8)
    assert ever.tolist() == [True] * 6 + [False] * 2
    # 0..3 solvable from the first three sets with 4 unknowns? 3 equations: not all; but
    # {4,5,0} - {4,5} gives control 0 exactly, and then 1,2,3 follow
    assert ident[:4].all() and not ident[4] and not ident[5] and not ident[6:].any()


def test_laplace_noise_averages_out():
    Xt, Xc = _data(nt=10, nc=200)
    S, _ = fedgm.normaliser(Xt, Xc, "treated_shrunk")
    rho = 2 * np.linalg.norm(Xc - Xc.mean(0), axis=1).max()
    errs = []
    for G in (1, 100):
        X_hat, _ = fa.recover_from_cross_terms(Xt @ S, S, Xc, Xc.mean(0), [np.ones(6)] * G,
                                               eps=10.0, rho=rho,
                                               rng=np.random.default_rng(1))
        errs.append(np.sqrt(((X_hat - Xc) ** 2).mean()))
    assert errs[1] < errs[0] / 3


def test_metrics_perfect_and_baseline():
    rng = np.random.default_rng(2)
    X = np.c_[rng.normal(size=(50, 2)), rng.random((50, 2)) < 0.4]
    mu, sd = X.mean(0), X.std(0)
    site = np.repeat([0, 1], 25)
    m = fa.recovery_metrics(X.copy(), X, mu, sd, np.array([0, 0, 1, 1], bool), site)
    assert m["exact_share"] == 1 and m["reid_share"] == 1 and m["binary_acc"] == 1
    assert m["reid_baseline"] == pytest.approx(2 / 50)
