"""experiments/harness.py: reproducible rep slices, multi-row configs, error rows."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.harness import flatten, parse_reps, run
from experiments.summarize import load_rows, summarize

CONFIG = Path(__file__).parent / "configs" / "tiny.py"


def test_parse_reps():
    assert parse_reps("0-3") == [0, 1, 2, 3]
    assert parse_reps("5") == [5]
    assert parse_reps("0-1,4-5") == [0, 1, 4, 5]


def test_rep_slices_reproduce_a_full_run(tmp_path):
    run(CONFIG, [0, 1, 2], tmp_path / "full")
    run(CONFIG, [0], tmp_path / "a")
    run(CONFIG, [1, 2], tmp_path / "b")
    full = load_rows(tmp_path / "full").sort_values(["rep", "config_id", "estimator"])
    parts = pd.concat([load_rows(tmp_path / "a"), load_rows(tmp_path / "b")])
    parts = parts.sort_values(["rep", "config_id", "estimator"])
    cols = ["rep", "data_seed", "config_seed", "config_id", "estimator", "tau", "se",
            "se_boot"]
    pd.testing.assert_frame_equal(full[cols].reset_index(drop=True),
                                  parts[cols].reset_index(drop=True))


def test_rows_frozen_keys_multi_rows_bootstrap_and_errors(tmp_path):
    run(CONFIG, [0, 1], tmp_path)
    rows = load_rows(tmp_path)
    assert {"tau", "se", "se_method", "max_smd", "runtime", "config_runtime"} <= set(rows)
    fed = rows[rows.config_id == "fed"]
    assert sorted(fed.estimator.unique()) == ["AIPW", "IPW-HT"]  # one fit, two rows
    boot = fed.dropna(subset=["se_boot"])
    assert list(zip(boot.rep, boot.estimator)) == [(0, "IPW-HT")]  # rep 0, IPW only
    broken = rows[rows.config_id == "broken"]
    assert len(broken) == 2 and broken.error.str.contains("deliberate").all()
    assert np.isnan(broken.tau).all()
    assert (rows[rows.config_id != "broken"].error.fillna("") == "").all()

    summary = summarize(rows)
    assert len(summary) == 4  # fed x 2, meta_ht, broken
    assert summary.loc[summary.config_id == "broken", "n_errors"].item() == 2


def test_flatten_keeps_scalars_and_splits_pairs():
    row = flatten(dict(tau=1.0, se=0.5, se_method="m", max_smd=0.1, runtime=0.2,
                       ci_if=(0.0, 2.0), n_k={0: 3, 1: 4}, names=["a", "b"],
                       arr=np.zeros(3), undefined_reason=None))
    assert row["ci_if_lo"] == 0.0 and row["ci_if_hi"] == 2.0
    assert row["n_k_0"] == 3 and row["names"] == "a;b"
    assert "arr" not in row and row["undefined_reason"] is None
