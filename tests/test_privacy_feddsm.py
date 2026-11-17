"""src/privacy/feddsm_attacks.py: FedDSM's releases reproduced, and their feature maps."""

import numpy as np
import pytest

from src.estimators import feddsm
from src.privacy import feddsm_attacks as fa, mia
from tests.test_fedgm import _single_arm


def test_releases_reproduce_feddsm():
    """The reproduced scores give FedDSM's own estimate, and the fold fits give m̂₀."""
    df = _single_arm(seed=30, nt=60, nc=600)
    df["Y"] = (df.Y > df.Y.median()).astype(float)
    R = fa.releases(df, seed=7)
    out = feddsm.estimate_multi(df, ["FedDSM-noBA"], target="ATT", ps="x", seed=7)
    W, Y = df.W.to_numpy(), df.Y.to_numpy()
    taus, _ = feddsm._match(R["lp"], {0: R["mu0"]}, W, Y, df.H.to_numpy(), "ATT", 4)
    assert taus["none"] == pytest.approx(out["FedDSM-noBA"]["tau"])
    fits = fa.fold_fits(R["B"], Y, W, R["fold"])
    for f in range(5):
        r = (W == 0) & (R["fold"] == f)
        assert np.allclose(R["B"][r] @ fits[f], R["mu0"][r])


def test_feature_maps_sum_to_the_released_sums():
    df = _single_arm(seed=31, nt=40, nc=300)
    R = fa.releases(df, seed=1)
    X = df[[c for c in df if c[0] == "X"]].to_numpy(float)
    XY = np.column_stack([X, df.Y.to_numpy()])
    site = (df.W == 0).to_numpy() & (df.H == 1).to_numpy()
    Xi = np.column_stack([X[site], np.ones(site.sum())])
    p = 1 / (1 + np.exp(-Xi @ R["beta"]))
    hess = (Xi * (p * (1 - p))[:, None]).T @ Xi
    phi = fa.ps_features(R["beta"])(XY[site]).sum(0)
    iu = np.triu_indices(Xi.shape[1])
    assert np.allclose(phi[:len(iu[0])], hess[iu]) and np.allclose(phi[len(iu[0]):], -Xi.T @ p)
    cell = site & (R["fold"] == 0)
    B = R["B"][cell]
    phi2 = fa.prognostic_features(R["quad"], R["interactions"])(XY[cell]).sum(0)
    iq = np.triu_indices(B.shape[1])
    assert np.allclose(phi2[:len(iq[0]) - 1], (B.T @ B)[iq][:-1])
    assert np.allclose(phi2[len(iq[0]) - 1:], B.T @ df.Y.to_numpy()[cell])
    assert mia.features(XY[:3], fa.ps_features(R["beta"])).shape[0] == 3
