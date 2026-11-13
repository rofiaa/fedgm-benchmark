"""src/dgps/ist.py (skipped without the data; setup/download_ist.sh)."""

import numpy as np
import pandas as pd
import pytest

from src.dgps import ist

pytestmark = pytest.mark.skipif(not (ist.DATA_DIR / "IST_corrected.csv").exists(),
                                reason="IST data not downloaded")


def test_population():
    pop = ist.population()
    assert len(pop) == 14231
    assert pop.H.nunique() == 9 and pop.groupby("H").size().is_monotonic_decreasing
    assert pop.country.iloc[(pop.H == 0).argmax()] == "UK"
    assert set(np.unique(pop[ist.COVARIATES[3:]])) == {0.0, 1.0}
    assert pop.Y0.mean() == pytest.approx(0.2365, abs=0.001)


def test_reproducible_and_reps_differ():
    a, b = ist.load(3, "obs_poor"), ist.load(3, "obs_poor")
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs == b.attrs
    assert not a.equals(ist.load(4, "obs_poor"))
    assert [c for c in a if c.startswith("X")] == [f"X{i}" for i in range(1, 32)]
    assert len(a) == ist.N_SUB


def test_truth_and_outcomes_consistent():
    df = ist.load(0, "obs_good")
    q, r = ist.effect_probs(pd.DataFrame(df[[f"X{i}" for i in range(1, 32)]].to_numpy(),
                                         columns=ist.COVARIATES))
    assert 0 < q.min() and q.max() < 1 and r.max() < 0.3 and r.mean() < 0.05
    assert df.attrs["ate_true"] < 0 and df.attrs["att_true"] < 0
    assert ist.load(0, "obs_good", target="ATT").attrs["tau_true"] == df.attrs["att_true"]


def test_single_arm():
    df = ist.load(0, "single_arm", target="ATT")
    assert df.H[df.W == 1].nunique() == 1 and df.H[df.W == 1].iloc[0] == 9
    assert (df.W[df.H < 9] == 0).all()
    assert 0.05 < df.W.mean() < 0.11
    with pytest.raises(ValueError):
        ist.load(0, "single_arm", target="ATE")
    with pytest.raises(ValueError):
        ist.with_site_dummies(df)


def test_site_dummies():
    df = ist.with_site_dummies(ist.load(0, "obs_site"))
    assert [c for c in df if c.startswith("X")] == [f"X{i}" for i in range(1, 40)]
    assert (df[[f"X{i}" for i in range(32, 40)]].sum(1) == (df.H > 0)).all()
    assert df.attrs["tau_true"] == ist.load(0, "obs_site").attrs["tau_true"]
