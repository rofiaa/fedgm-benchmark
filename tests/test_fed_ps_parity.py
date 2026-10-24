"""Our fed_ps.estimate must reproduce the authors' Simulations_Fed on the same data.

Their estimate_ipw draws a fresh dataset inside its loop via
Simulations_Fed(...).combine_data(); for the duration of each call we patch that method
to return the fixture's dataset, and run one repetition.
"""

from unittest import mock

import numpy as np
import pytest

from src.estimators._authors import dgp_fed
from src.estimators.fed_ps import estimate


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
def test_ipw_mw_logistic_matches_authors(dgp_a_poor, ipw_type):
    params, df_auth, df = dgp_a_poor
    tau_theirs, df_theirs = run_authors_ipw(params, df_auth, ipw_type)

    ours = estimate(df, mw="logistic", local_ps="logistic", ipw_type=ipw_type)

    their_local = df_theirs[[f"hat_e_local_{k}" for k in (1, 2, 3)]].to_numpy()
    np.testing.assert_allclose(ours["e_local"].to_numpy(), their_local, rtol=1e-12)
    np.testing.assert_allclose(
        ours["e_hat"].to_numpy(), df_theirs["hat_e_MW_logistic"].to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(ours["tau"], tau_theirs, rtol=1e-10)
    assert np.isfinite(ours["se"]) and ours["se"] > 0
