"""The Oracle must reproduce the authors' "Oracle" (true e*, and Y1*/Y0* for AIPW)."""

from unittest import mock

import numpy as np
import pytest

from src.dgps.authors import dgp_b_params, generate
from src.estimators._authors import dgp_fed
from src.estimators.common import check_result
from src.estimators.oracle import estimate_multi


def _authors(params, df_auth, by_center, aipw):
    Sim = dgp_fed().Simulations_Fed
    data = lambda self, *a, **k: df_auth.copy()  # noqa: E731
    with mock.patch.object(Sim, "combine_data", data), \
            mock.patch.object(Sim, "make_data_by_H_given_X", data):
        sim = Sim(params, n_simulations=1)
        common = dict(propensity_estimation="logistic", final_estimators=["Oracle"],
                      generate_by_center=by_center, print_overlap=False, scale=False)
        if aipw:
            res, df_out = sim.estimate_aipw(propensity_federations=["oracle", "local"],
                                            outcome_estimators=["GD"], params_for_federation=None,
                                            params_for_federation_omega=None, **common)
        else:
            res, df_out, _ = sim.estimate_ipw(ht_or_hajek="HT",
                                              propensity_federations=["oracle", "local"],
                                              **common)
    return res["Oracle"][0], df_out["e_oracle"].to_numpy()


@pytest.mark.parametrize("aipw", [False, True])
def test_oracle_matches_authors_dgp_a(dgp_a_poor, aipw):
    params, df_auth, df = dgp_a_poor
    tau, e_theirs = _authors(params, df_auth, True, aipw)
    name = "AIPW" if aipw else "IPW-HT"
    ours = check_result(estimate_multi(df, [name])[name])
    np.testing.assert_allclose(ours["e_hat"].to_numpy(), e_theirs, rtol=1e-9)
    np.testing.assert_allclose(ours["tau"], tau, rtol=1e-9)


def test_oracle_matches_authors_dgp_b():
    params = dgp_b_params("poor", n=600)
    df_auth, df = generate(params, "B", seed=0)
    tau, e_theirs = _authors(params, df_auth, False, aipw=False)
    ours = estimate_multi(df, ["IPW-HT"])["IPW-HT"]
    np.testing.assert_allclose(ours["e_hat"].to_numpy(), e_theirs, rtol=1e-9)
    np.testing.assert_allclose(ours["tau"], tau, rtol=1e-9)


def test_oracle_needs_the_true_nuisances(dgp_a_poor):
    df = dgp_a_poor[2].copy()
    df.attrs = {}
    with pytest.raises(ValueError):
        estimate_multi(df)
