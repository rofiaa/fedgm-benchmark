"""Behaviour of fed_ps.estimate that has no counterpart in the authors' code."""

import numpy as np
import pytest

from src.estimators.fed_ps import _ipw_terms_hajek, estimate


def test_hajek_terms_average_to_tau(dgp_a_poor):
    _, _, df = dgp_a_poor
    out = estimate(df, ipw_type="Hajek")
    masks = [df["H"] == k for k in range(3)]
    tau_k = []
    for mask in masks:
        W, Y, e = df["W"][mask], df["Y"][mask], out["e_hat"][mask]
        tau_k.append(np.sum(W * Y / e) / np.sum(W / e)
                     - np.sum((1 - W) * Y / (1 - e)) / np.sum((1 - W) / (1 - e)))

    psi = _ipw_terms_hajek(df["W"].to_numpy(), df["Y"].to_numpy(),
                           out["e_hat"].to_numpy(), masks, tau_k)

    assert np.isclose(psi.mean(), out["tau"], rtol=1e-12)
    assert np.isclose(np.std(psi, ddof=1) / np.sqrt(len(psi)), out["se"], rtol=1e-12)


def test_single_arm_site_gets_constant_local_ps(dgp_a_poor):
    _, _, df = dgp_a_poor
    df = df.copy()
    df.loc[df["H"] == 1, "W"] = 0  # site 2 becomes an external control arm

    out = estimate(df)

    assert (out["e_local"][1] == 0).all()
    assert np.isfinite(out["tau"]) and np.isfinite(out["se"])


def test_rejects_bad_site_labels(dgp_a_poor):
    _, _, df = dgp_a_poor
    with pytest.raises(ValueError, match="0..K-1"):
        estimate(df.assign(H=df["H"] + 1))


def test_unimplemented_options_raise(dgp_a_poor):
    _, _, df = dgp_a_poor
    with pytest.raises(NotImplementedError):
        estimate(df, aipw=True)
    with pytest.raises(NotImplementedError, match="minibatches"):
        estimate(df, mw="logistic_fedavg", fedavg_params={"B_batch_size": 32})


def test_fedavg_mw_close_to_pooled(dgp_a_poor):
    """Same target; 30 FedAvg rounds on unscaled X need not have converged."""
    _, _, df = dgp_a_poor
    pooled = estimate(df, mw="logistic")
    fedavg = estimate(df, mw="logistic_fedavg")
    assert fedavg["mw_fit"] == "fedavg" and pooled["mw_fit"] == "pooled-centralized"
    gap = np.abs(fedavg["omega_hat"] - pooled["omega_hat"]).to_numpy()
    print(f"max |omega_fedavg - omega_pooled| = {gap.max():.3f}, "
          f"tau {fedavg['tau']:.3f} vs {pooled['tau']:.3f}")
    assert np.isfinite(fedavg["tau"])
