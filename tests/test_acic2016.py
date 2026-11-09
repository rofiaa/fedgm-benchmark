"""src/dgps/acic2016.py (skipped without the aciccomp2016 R package)."""

import numpy as np
import pytest


def _have_acic():
    try:
        from src.dgps import acic2016
        acic2016._r()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_acic(), reason="aciccomp2016 not installed")


def test_frame_truth_and_reproducibility():
    from src.dgps import acic2016
    a = acic2016.load(0, 1)
    assert a.shape == (4802, 79 + 3)
    assert sorted(a.H.unique()) == list(range(acic2016.N_SITES))
    assert set(a.W.unique()) == {0, 1}
    assert a.attrs["tau_true"] == a.attrs["ate_true"] != a.attrs["att_true"]
    b = acic2016.load(0, 1, target="ATT")
    assert b.attrs["tau_true"] == a.attrs["att_true"]
    assert a.Y.equals(b.Y) and a.H.equals(b.H)          # same replication, same data
    assert not a.Y.equals(acic2016.load(1, 1).Y)
    assert len(acic2016.settings()) == acic2016.N_SETTINGS
    with pytest.raises(ValueError):
        acic2016.load(100, 1)


def test_sites_depend_on_covariates_only():
    from src.dgps import acic2016
    X, _ = acic2016.covariates()
    h1, h2 = acic2016._sites(X, 3, 5), acic2016._sites(X, 3, 5)
    assert np.array_equal(h1, h2) and not np.array_equal(h1, acic2016._sites(X, 3, 6))
