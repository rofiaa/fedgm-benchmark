"""src/dgps/longtail.py: single-arm data with long-tailed covariates."""

import numpy as np
import pytest

from src.dgps import longtail


@pytest.mark.parametrize("tails", longtail.TAILS)
@pytest.mark.parametrize("selection", longtail.SELECTIONS)
def test_generate(tails, selection):
    df = longtail.generate(3, tails, selection)
    assert df.W.mean() == pytest.approx(longtail.SHARE, abs=0.02)
    assert (df.H[df.W == 1] == 0).all() and set(df.H[df.W == 0]) == {1, 2, 3, 4}
    assert np.isfinite(df.attrs["tau_true"])
    df2 = longtail.generate(3, tails, selection)
    assert df.equals(df2) and df.attrs == df2.attrs


def test_tails_are_heavy():
    k = {tl: float(longtail.generate(0, tl).X1.kurt()) for tl in longtail.TAILS}
    assert k["gauss"] < 1 < 10 < k["t3"] and k["lognormal"] > 10
