"""Meta-SW: parity with the authors' "Meta-SW" final estimator, and our extensions."""

from unittest import mock

import numpy as np
import pytest

from src.estimators import fed_ps, meta_sw
from src.estimators._authors import dgp_fed
from src.estimators.common import REQUIRED_KEYS, check_result


def _authors(params, df_auth, aipw, ipw_type="HT", propensity="logistic", np_seed=None):
    Sim = dgp_fed().Simulations_Fed
    state = np.random.get_state()
    if np_seed is not None:
        np.random.seed(np_seed)
    try:
        with mock.patch.object(Sim, "combine_data",
                               lambda self, dict_data=None: df_auth.copy()):
            sim = Sim(params, n_simulations=1)
            if aipw:
                results, _ = sim.estimate_aipw(
                    propensity_federations=["local"], outcome_estimators=["local"],
                    final_estimators=["Meta-SW"], propensity_estimation=propensity,
                    use_grf=False, generate_by_center=True, scale=False,
                    print_overlap=False,
                )
            else:
                results, _, _ = sim.estimate_ipw(
                    ht_or_hajek=ipw_type, propensity_estimation=propensity, use_grf=False,
                    propensity_federations=["local"], final_estimators=["Meta-SW"],
                    generate_by_center=True, print_overlap=False, scale=False,
                )
    finally:
        np.random.set_state(state)
    assert len(results["Meta-SW"]) == 1
    return results["Meta-SW"][0]


@pytest.mark.parametrize("ipw_type", ["HT", "Hajek"])
def test_meta_sw_ipw_matches_authors(dgp_a_poor, ipw_type):
    params, df_auth, df = dgp_a_poor
    theirs = _authors(params, df_auth, aipw=False, ipw_type=ipw_type)
    ours = meta_sw.estimate(df, ipw_type=ipw_type)
    np.testing.assert_allclose(ours["tau"], theirs, rtol=1e-10)


def test_meta_sw_aipw_matches_authors(dgp_a_poor):
    params, df_auth, df = dgp_a_poor
    theirs = _authors(params, df_auth, aipw=True)
    ours = meta_sw.estimate(df, aipw=True)
    np.testing.assert_allclose(ours["tau"], theirs, rtol=1e-10)
    assert ours["outcome_fit"] == "local-ols"


def test_meta_sw_rf_matches_authors(dgp_a_poor):
    params, df_auth, df = dgp_a_poor
    theirs = _authors(params, df_auth, aipw=False, propensity="random_forest", np_seed=31)
    ours = meta_sw.estimate(df, local_ps="rf", seed=31)
    np.testing.assert_allclose(ours["tau"], theirs, rtol=1e-10)


def test_meta_sw_terms_and_site_estimates(dgp_a_poor):
    _, _, df = dgp_a_poor
    out = meta_sw.estimate(df)
    n_k = np.array(list(out["n_k"].values()))
    tau_k = np.array(list(out["tau_k"].values()))
    assert np.isclose(out["tau"], np.average(tau_k, weights=n_k), rtol=1e-12)
    W, Y, e = df["W"].to_numpy(), df["Y"].to_numpy(), out["e_own"].to_numpy()
    psi = W * Y / e - (1 - W) * Y / (1 - e)
    assert np.isclose(np.std(psi, ddof=1) / np.sqrt(len(psi)), out["se"], rtol=1e-12)


@pytest.mark.parametrize("aipw", [False, True])
def test_meta_sw_is_nan_with_a_single_arm_site(dgp_a_poor, aipw):
    _, _, df = dgp_a_poor
    df = df.copy()
    df.loc[df["H"] == 1, "W"] = 0
    out = check_result(meta_sw.estimate(df, aipw=aipw))
    assert np.isnan(out["tau"]) and np.isnan(out["se"]) and np.isnan(out["max_smd"])
    assert "[1]" in out["undefined_reason"]
    # the federated estimator is still defined on the same data
    assert np.isfinite(fed_ps.estimate(df, aipw=aipw)["tau"])


def test_meta_sw_bootstrap_counts_arm_loss_as_failure(dgp_a_poor):
    _, _, df = dgp_a_poor
    df = df.copy()
    site2 = df.index[df["H"] == 1]
    df.loc[site2, "W"] = 0
    df.loc[site2[:2], "W"] = 1  # two treated units: lost in ~ e^-2 of resamples
    out = meta_sw.estimate(df, bootstrap=100, seed=2)
    assert out["boot_arm_loss"] > 0
    assert out["boot_failed"] == out["boot_arm_loss"]
    assert out["boot_ok"] == 100 - out["boot_arm_loss"]


@pytest.mark.parametrize("call", [
    lambda df: fed_ps.estimate(df),
    lambda df: fed_ps.estimate(df, aipw=True, bootstrap=2, seed=0),
    lambda df: meta_sw.estimate(df),
    lambda df: meta_sw.estimate(df, aipw=True, bootstrap=2, seed=0),
])
def test_frozen_keys(dgp_a_poor, call):
    _, _, df = dgp_a_poor
    out = check_result(call(df))
    assert set(REQUIRED_KEYS) <= set(out)
