"""Site-stratified bootstrap in fed_ps.estimate."""

import numpy as np

from src.estimators.fed_ps import estimate


def test_bootstrap_is_seeded_and_leaves_point_estimate_alone(dgp_a_poor):
    _, _, df = dgp_a_poor
    plain = estimate(df)
    a = estimate(df, bootstrap=30, seed=7)
    b = estimate(df, bootstrap=30, seed=7)
    c = estimate(df, bootstrap=30, seed=8)

    np.testing.assert_array_equal(a["boot_taus"], b["boot_taus"])
    assert not np.array_equal(a["boot_taus"], c["boot_taus"])
    assert a["tau"] == plain["tau"] and a["se"] == plain["se"]
    assert a["boot_ok"] == 30 and a["boot_failed"] == 0
    assert np.isclose(a["se_boot"], np.std(a["boot_taus"], ddof=1))
    lo, hi = a["ci_boot_percentile"]
    assert lo < a["tau"] < hi


def test_arm_loss_is_counted_per_site(dgp_a_poor):
    """Site 2 keeps a single treated unit: a within-site resample of size n_k misses it
    with probability (1 - 1/n_k)^n_k ≈ e^-1."""
    _, _, df = dgp_a_poor
    df = df.copy()
    site2 = df.index[df["H"] == 1]
    df.loc[site2, "W"] = 0
    df.loc[site2[0], "W"] = 1

    out = estimate(df, bootstrap=200, seed=1)

    assert out["boot_arm_loss_by_site"][0] == out["boot_arm_loss_by_site"][2] == 0
    assert out["boot_arm_loss"] == out["boot_arm_loss_by_site"][1]
    assert 0.25 < out["boot_arm_loss"] / 200 < 0.5
    assert out["boot_ok"] == 200  # arm-loss replicates are still estimated


def test_fedavg_bootstrap_refits_with_pooled_mw(dgp_a_poor):
    _, _, df = dgp_a_poor
    out = estimate(df, mw="logistic_fedavg", bootstrap=5, seed=0)
    assert out["mw_fit"] == "fedavg" and out["boot_mw_fit"] == "pooled-centralized"


def test_aipw_bootstrap_runs(dgp_a_poor):
    _, _, df = dgp_a_poor
    out = estimate(df, aipw=True, bootstrap=20, seed=3)
    assert out["boot_ok"] == 20 and np.isfinite(out["se_boot"]) and out["se_boot"] > 0


def test_nn_bootstrap_retrains_with_seeded_nets(dgp_a_poor):
    _, _, df = dgp_a_poor
    a = estimate(df, mw="nn", bootstrap=3, seed=9)
    b = estimate(df, mw="nn", bootstrap=3, seed=9)
    assert a["boot_mw_fit"] == "nn-pooled" and a["boot_ok"] == 3
    np.testing.assert_array_equal(a["boot_taus"], b["boot_taus"])
