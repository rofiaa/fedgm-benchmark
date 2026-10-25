"""estimate_multi: one nuisance fit shared by several estimators."""

import numpy as np

from src.estimators import fed_ps, meta_sw
from src.estimators.common import check_result


def test_fed_multi_equals_separate_calls(dgp_a_poor):
    _, _, df = dgp_a_poor
    multi = fed_ps.estimate_multi(df, ["IPW-HT", "IPW-Hajek_pooled", "AIPW"])
    for name, kw in [("IPW-HT", {}), ("IPW-Hajek_pooled", {"ipw_type": "Hajek_pooled"}),
                     ("AIPW", {"aipw": True})]:
        single = fed_ps.estimate(df, **kw)
        check_result(multi[name])
        assert multi[name]["tau"] == single["tau"] and multi[name]["se"] == single["se"]
        assert multi[name]["estimand"] == name
        assert multi[name]["runtime_shared_by"] == ["IPW-HT", "IPW-Hajek_pooled", "AIPW"]


def test_fed_multi_shared_bootstrap_only_where_asked(dgp_a_poor):
    _, _, df = dgp_a_poor
    multi = fed_ps.estimate_multi(df, ["IPW-HT", "IPW-Hajek_pooled", "AIPW"],
                                  bootstrap=20, bootstrap_for=["IPW-HT", "IPW-Hajek_pooled"],
                                  seed=4)
    assert "se_boot" not in multi["AIPW"]
    single = fed_ps.estimate(df, bootstrap=20, seed=4)
    # same seed -> same resamples -> same replicate estimates as a stand-alone call
    np.testing.assert_array_equal(multi["IPW-HT"]["boot_taus"], single["boot_taus"])
    assert multi["IPW-HT"]["boot_arm_loss"] == multi["IPW-Hajek_pooled"]["boot_arm_loss"]


def test_meta_multi_equals_separate_calls(dgp_a_poor):
    _, _, df = dgp_a_poor
    multi = meta_sw.estimate_multi(df, ["IPW-HT", "AIPW"], bootstrap=10,
                                   bootstrap_for=["IPW-HT"], seed=1)
    assert multi["IPW-HT"]["tau"] == meta_sw.estimate(df)["tau"]
    assert multi["AIPW"]["tau"] == meta_sw.estimate(df, aipw=True)["tau"]
    np.testing.assert_array_equal(
        multi["IPW-HT"]["boot_taus"], meta_sw.estimate(df, bootstrap=10, seed=1)["boot_taus"]
    )
    assert "se_boot" not in multi["AIPW"]
