"""Our fed_ps.estimate must reproduce the authors' Simulations_Fed on the same data.

Their estimate_ipw draws a fresh dataset inside its loop via
Simulations_Fed(...).combine_data(); for the duration of each call we patch that method
to return the fixture's dataset, and run one repetition.
"""

from unittest import mock

import numpy as np
import pytest

from src.estimators._authors import dgp_fed
from src.estimators.fed_ps import FEDAVG_MW_NOTEBOOK, FEDAVG_OUTCOME_NOTEBOOK, estimate


def run_authors_ipw(params, df_auth, ipw_type, omega_params=None):
    Sim = dgp_fed().Simulations_Fed
    with mock.patch.object(Sim, "combine_data", lambda self, dict_data=None: df_auth.copy()):
        results, df_out, _ = Sim(params, n_simulations=1).estimate_ipw(
            ht_or_hajek=ipw_type,
            propensity_estimation="logistic",
            # "local" is required: MW alone never triggers the local-PS fits
            propensity_federations=["MW_logistic", "local"],
            final_estimators=["MW_logistic"],
            params_for_federation_omega=omega_params,
            generate_by_center=True,
            print_overlap=False,
            scale=False,
        )
    assert len(results["MW_logistic"]) == 1
    return results["MW_logistic"][0], df_out


@pytest.mark.parametrize("ipw_type", ["HT", "Hajek"])
@pytest.mark.parametrize("mw", ["logistic", "logistic_fedavg"])
def test_ipw_mw_logistic_matches_authors(dgp_a_poor, ipw_type, mw):
    params, df_auth, df = dgp_a_poor
    # their FedAvg runs only when params_for_federation_omega is passed
    omega_params = None
    if mw == "logistic_fedavg":  # their MW_estimation has no stopping rule / rescaling
        omega_params = {k: v for k, v in FEDAVG_MW_NOTEBOOK.items()
                        if k not in ("tol", "standardize")}
    tau_theirs, df_theirs = run_authors_ipw(params, df_auth, ipw_type, omega_params)

    ours = estimate(df, mw=mw, local_ps="logistic", ipw_type=ipw_type,
                    fedavg_params=FEDAVG_MW_NOTEBOOK)

    their_local = df_theirs[[f"hat_e_local_{k}" for k in (1, 2, 3)]].to_numpy()
    np.testing.assert_allclose(ours["e_local"].to_numpy(), their_local, rtol=1e-12)
    np.testing.assert_allclose(
        ours["e_hat"].to_numpy(), df_theirs["hat_e_MW_logistic"].to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(ours["tau"], tau_theirs, rtol=1e-10)
    assert np.isfinite(ours["se"]) and ours["se"] > 0


def run_authors_aipw(params, df_auth, outcome_params=None):
    Sim = dgp_fed().Simulations_Fed
    with mock.patch.object(Sim, "combine_data", lambda self, dict_data=None: df_auth.copy()):
        results, df_out = Sim(params, n_simulations=1).estimate_aipw(
            propensity_federations=["MW_logistic", "local"],
            outcome_estimators=["GD"],
            final_estimators=["GD-MW_logistic"],
            propensity_estimation="logistic",
            generate_by_center=True,
            params_for_federation=outcome_params,  # None -> pooled OLS
            params_for_federation_omega=None,       # pooled multinomial logistic
            scale=False,
            print_overlap=False,
        )
    assert len(results["GD-MW_logistic"]) == 1
    return results["GD-MW_logistic"][0], df_out


@pytest.mark.parametrize("outcome", ["ols", "fedavg"])
def test_aipw_mw_logistic_matches_authors(dgp_a_poor, outcome):
    params, df_auth, df = dgp_a_poor
    outcome_params = FEDAVG_OUTCOME_NOTEBOOK if outcome == "fedavg" else None
    tau_theirs, df_theirs = run_authors_aipw(params, df_auth, outcome_params)

    # their AIPW path fits the membership model on X only (no intercept)
    ours = estimate(df, aipw=True, mw="logistic", mw_intercept=False, outcome=outcome)

    np.testing.assert_allclose(
        ours["e_hat"].to_numpy(), df_theirs["hat_e_MW_logistic"].to_numpy(), rtol=1e-12
    )
    X_out = df_theirs[list(ours["outcome_coef"]["beta_1"].index)].to_numpy()
    np.testing.assert_allclose(X_out @ ours["outcome_coef"]["beta_1"].to_numpy(),
                               df_theirs["hat_mu1_GD"].to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(X_out @ ours["outcome_coef"]["beta_0"].to_numpy(),
                               df_theirs["hat_mu0_GD"].to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(ours["tau"], tau_theirs, rtol=1e-10)
    assert ours["outcome_fit"] == {"ols": "pooled-ols", "fedavg": "fedavg-sgd-clipped"}[outcome]
    assert np.isfinite(ours["se"]) and ours["se"] > 0


def test_ipw_mw_nn_matches_authors(dgp_a_poor):
    """train_NN draws from torch's global RNG; seeding it with the same int before
    their run as our `seed` must give the same net, hence the same estimate."""
    import torch

    params, df_auth, df = dgp_a_poor
    Sim = dgp_fed().Simulations_Fed
    torch.manual_seed(11)
    with mock.patch.object(Sim, "combine_data", lambda self, dict_data=None: df_auth.copy()):
        results, df_theirs, _ = Sim(params, n_simulations=1).estimate_ipw(
            ht_or_hajek="HT",
            propensity_estimation="logistic",
            propensity_federations=["MW_NN", "local"],
            final_estimators=["MW_NN"],
            generate_by_center=True,
            print_overlap=False,
            rerun_NN=True,
            scale=False,
        )

    ours = estimate(df, mw="nn", seed=11)

    assert ours["mw_fit"] == "nn-pooled" and ours["mw_seed"] == 11
    np.testing.assert_allclose(
        ours["e_hat"].to_numpy(), df_theirs["hat_e_MW_NN"].to_numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(ours["tau"], results["MW_NN"][0], rtol=1e-6)


def test_aipw_mw_nn_matches_authors(dgp_a_poor):
    import torch

    params, df_auth, df = dgp_a_poor
    Sim = dgp_fed().Simulations_Fed
    torch.manual_seed(12)
    with mock.patch.object(Sim, "combine_data", lambda self, dict_data=None: df_auth.copy()):
        results, _ = Sim(params, n_simulations=1).estimate_aipw(
            propensity_federations=["MW_NN", "local"],
            outcome_estimators=["GD"],
            final_estimators=["GD-MW_NN"],
            propensity_estimation="logistic",
            generate_by_center=True,
            params_for_federation=None,
            params_for_federation_omega=None,
            scale=False,
            print_overlap=False,
            rerun_NN=True,
        )

    ours = estimate(df, mw="nn", aipw=True, seed=12)

    np.testing.assert_allclose(ours["tau"], results["GD-MW_NN"][0], rtol=1e-6)
