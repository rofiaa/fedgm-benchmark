"""Behaviour of fed_ps.estimate that has no counterpart in the authors' code."""

import numpy as np
import pandas as pd
import pytest

from src.dgps.authors import dgp_b_poor_overlap_params, generate
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


@pytest.mark.parametrize("mw", ["logistic", "logistic_fedavg"])
def test_membership_columns_in_numeric_site_order_for_k_ge_10(mw):
    """The authors' classifiers order classes alphabetically ("client10" before
    "client2"); ω̂ column k must still belong to site k (docs/findings.md, F1)."""
    rng = np.random.default_rng(0)
    K, n_per = 11, 80
    H = np.repeat(np.arange(K), n_per)
    X1 = 3.0 * H + rng.normal(size=H.size)  # sites well separated along X1
    df = pd.DataFrame({"X1": X1, "X2": rng.normal(size=H.size), "H": H,
                       "W": rng.binomial(1, 0.5, H.size)})
    df["Y"] = df["X1"] + df["W"] + rng.normal(size=H.size)

    omega = estimate(df, mw=mw)["omega_hat"].to_numpy()

    mean_omega_by_site = np.array([omega[H == k].mean(axis=0) for k in range(K)])
    assert (mean_omega_by_site.argmax(axis=1) == np.arange(K)).all()


def test_rejects_bad_site_labels(dgp_a_poor):
    _, _, df = dgp_a_poor
    with pytest.raises(ValueError, match="0..K-1"):
        estimate(df.assign(H=df["H"] + 1))


def test_unimplemented_options_raise(dgp_a_poor):
    _, _, df = dgp_a_poor
    with pytest.raises(NotImplementedError):
        estimate(df, local_ps="grf")
    with pytest.raises(ValueError, match="IPW only"):
        estimate(df, aipw=True, ipw_type="Hajek_pooled")
    with pytest.raises(NotImplementedError, match="minibatches"):
        estimate(df, mw="logistic_fedavg", fedavg_params={"B_batch_size": 32})


@pytest.mark.parametrize("dgp", ["A", "B"])
def test_converged_fedavg_matches_pooled_fit(dgp_a_poor, dgp):
    """With the default settings FedAvg converges within the cap to the pooled
    multinomial-logistic fit (up to the pooled solver's own tolerance)."""
    if dgp == "A":
        _, _, df = dgp_a_poor
    else:
        _, df = generate(dgp_b_poor_overlap_params(), "B", seed=3)
    pooled = estimate(df, mw="logistic")
    fedavg = estimate(df, mw="logistic_fedavg")

    assert fedavg["mw_fit"] == "fedavg" and pooled["mw_fit"] == "pooled-centralized"
    assert fedavg["mw_converged"] and fedavg["mw_rounds"] < 1000
    gap = np.abs(fedavg["omega_hat"] - pooled["omega_hat"]).to_numpy().max()
    assert gap < 5e-3
    assert abs(fedavg["tau"] - pooled["tau"]) < 1e-2


def test_aipw_terms_average_to_tau(dgp_a_poor):
    _, _, df = dgp_a_poor
    out = estimate(df, aipw=True)
    X = df[[f"X{i}" for i in range(1, 11)]].assign(intercept=1.0).to_numpy()
    mu1 = X @ out["outcome_coef"]["beta_1"].to_numpy()
    mu0 = X @ out["outcome_coef"]["beta_0"].to_numpy()
    W, Y, e = df["W"].to_numpy(), df["Y"].to_numpy(), out["e_hat"].to_numpy()
    psi = mu1 - mu0 + W * (Y - mu1) / e - (1 - W) * (Y - mu0) / (1 - e)

    assert np.isclose(psi.mean(), out["tau"], rtol=1e-12)
    assert np.isclose(np.std(psi, ddof=1) / np.sqrt(len(psi)), out["se"], rtol=1e-12)
    assert out["outcome_fit"] == "pooled-ols"


def test_hajek_pooled_normalizes_over_all_sites(dgp_a_poor):
    _, _, df = dgp_a_poor
    out = estimate(df, ipw_type="Hajek_pooled")
    W, Y, e = df["W"], df["Y"], out["e_hat"]
    expected = (np.sum(W * Y / e) / np.sum(W / e)
                - np.sum((1 - W) * Y / (1 - e)) / np.sum((1 - W) / (1 - e)))
    assert np.isclose(out["tau"], expected, rtol=1e-12)

    all_rows = pd.Series(True, index=df.index)
    psi = _ipw_terms_hajek(W.to_numpy(), Y.to_numpy(), e.to_numpy(), [all_rows], [out["tau"]])
    assert np.isclose(psi.mean(), out["tau"], rtol=1e-12)
    assert np.isclose(np.std(psi, ddof=1) / np.sqrt(len(psi)), out["se"], rtol=1e-12)


def test_nn_mw_is_seeded_and_restores_torch_rng(dgp_a_poor):
    import torch

    _, _, df = dgp_a_poor
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)

    a = estimate(df, mw="nn", seed=5)

    assert torch.equal(torch.rand(1), expected_next)  # global torch RNG untouched
    b = estimate(df, mw="nn", seed=5)
    c = estimate(df, mw="nn", seed=6)
    assert a["tau"] == b["tau"] and a["tau"] != c["tau"]
    assert np.allclose(a["omega_hat"].sum(axis=1), 1)


def test_rf_local_ps_is_seeded_and_bounded(dgp_a_poor):
    _, _, df = dgp_a_poor
    a = estimate(df, local_ps="rf", seed=4)
    b = estimate(df, local_ps="rf", seed=4)
    c = estimate(df, local_ps="rf", seed=5)
    assert a["tau"] == b["tau"] and a["tau"] != c["tau"]
    e = a["e_local"].to_numpy()
    assert ((e > 0) & (e < 1)).all()  # depth-2 forests with large leaves never hit 0/1
