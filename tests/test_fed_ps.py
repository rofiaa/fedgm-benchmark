"""Behaviour of fed_ps.estimate that has no counterpart in the authors' code."""

import numpy as np
import pytest

from src.estimators.fed_ps import estimate


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
