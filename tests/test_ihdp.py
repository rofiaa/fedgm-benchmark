"""src/dgps/ihdp.py and the harness's fixed-replication scenarios (skipped without the
data; setup/download_ihdp.sh)."""

import numpy as np
import pandas as pd
import pytest

from src.dgps import ihdp

pytestmark = pytest.mark.skipif(not (ihdp.DATA_DIR / "ihdp_npci_1-100.train.npz").exists(),
                                reason="IHDP data not downloaded")


def test_design_identical_across_replications():
    a, b = ihdp.load(0), ihdp.load(57)
    cols = [c for c in a if c != "Y"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert len(a) == 747 and a.W.sum() == 139
    assert sorted(a.H.unique()) == list(range(8))
    assert a.groupby("H").W.sum().tolist() == [32, 21, 9, 24, 5, 2, 8, 38]
    assert not a.Y.equals(b.Y)


def test_truth_attached():
    df = ihdp.load(0)
    assert df.attrs["att_true"] == pytest.approx(4.0)
    assert df.attrs["tau_true"] == pytest.approx(4.016, abs=1e-3)  # CEVAE replication 1


def test_att_target():
    df = ihdp.load(0, target="ATT")
    assert df.attrs["tau_true"] == pytest.approx(4.0)
    assert df.attrs["ate_true"] == ihdp.load(0).attrs["tau_true"]
    with pytest.raises(ValueError):
        ihdp.load(0, target="ATC")


def test_without_site_dummies():
    df = ihdp.without_site_dummies(ihdp.load(0))
    assert [c for c in df if c.startswith("X")] == [f"X{i}" for i in range(1, 19)]
    assert df.attrs["tau_true"] == ihdp.load(0).attrs["tau_true"]


def test_harness_uses_rep_and_per_rep_truth(tmp_path):
    from experiments.harness import Scenario, run_rep, single
    from experiments.summarize import summarize

    scen = Scenario("IHDP", generate=ihdp.load, truth=lambda n, s: (np.nan, np.nan),
                    by_rep=True)
    naive = single("diff", lambda df, seed: dict(
        tau=df.Y[df.W == 1].mean() - df.Y[df.W == 0].mean(), se=0.1, se_method="none",
        max_smd=0.0, runtime=0.0))
    rows = pd.DataFrame(run_rep(scen, [naive], 3, 1, np.nan)
                        + run_rep(scen, [naive], 4, 1, np.nan))
    assert rows.tau_true.tolist() == [ihdp.load(3).attrs["tau_true"],
                                      ihdp.load(4).attrs["tau_true"]]
    s = summarize(rows).iloc[0]
    assert s.bias == pytest.approx((rows.tau - rows.tau_true).mean())
