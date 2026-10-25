"""Combine harness row files and score each (scenario, config, estimator).

    python experiments/summarize.py results/ps_mc   # reads results/ps_mc/rows/*.csv

Writes <dir>/summary.csv. Metrics on all reps with a finite τ̂: bias (± MC SE), median
bias, empirical SD, RMSE, mean IF SE and its ratio to the empirical SD, IF 95%
coverage and width, mean max |SMD|, runtimes. On the reps that carry a bootstrap:
mean bootstrap SE (ratio to the empirical SD over all reps), coverage of the
bootstrap-SE normal CI and of the percentile CI, their widths, IF coverage on the same
reps (like-for-like), and bootstrap failures / arm-loss counts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

Z95 = 1.959963984540054
OPTIONAL = ("mw_rounds_max", "n_boot_reps", "mean_se_boot", "se_boot_ratio", "cov_boot_se",
            "cov_boot_pct", "cov_if_on_boot_reps", "width_boot_se", "width_boot_pct",
            "boot_failed_total", "boot_arm_loss_total", "boot_arm_loss_reps")


def _covers(lo, hi, truth):
    return float(((lo <= truth) & (truth <= hi)).mean()) if len(lo) else np.nan


def score(g):
    truth = g["tau_true"].iloc[0]
    ok = g[np.isfinite(g["tau"])]
    err = ok["tau"] - truth
    sd = ok["tau"].std(ddof=1)
    has_se = ok[np.isfinite(ok["se"])]
    out = {
        "tau_true": truth,
        "n_reps": len(g),
        "n_errors": int(g["error"].astype("string").fillna("").ne("").sum()),
        "n_undefined": int((~np.isfinite(g["tau"])).sum()),
        "bias": err.mean(),
        "bias_mcse": sd / np.sqrt(len(ok)) if len(ok) > 1 else np.nan,
        "median_bias": err.median(),
        "emp_sd": sd,
        "rmse": float(np.sqrt(np.mean(err**2))) if len(ok) else np.nan,
        "mean_se_if": has_se["se"].mean(),
        "se_if_ratio": has_se["se"].mean() / sd,
        "cov_if": _covers(has_se["tau"] - Z95 * has_se["se"],
                          has_se["tau"] + Z95 * has_se["se"], truth),
        "width_if": (2 * Z95 * has_se["se"]).mean(),
        "mean_max_smd": ok["max_smd"].mean(),
        "mean_runtime_s": ok["runtime"].mean(),
        "mean_config_runtime_s": g["config_runtime"].mean(),
    }
    # optional metrics: same keys for every group, NaN when not applicable
    out.update(dict.fromkeys(OPTIONAL, np.nan))
    if "mw_rounds" in ok and ok["mw_rounds"].notna().any():
        out["mw_rounds_max"] = ok["mw_rounds"].max()
    if "se_boot" in ok and ok["se_boot"].notna().any():
        b = ok[ok["se_boot"].notna()]
        out.update({
            "n_boot_reps": len(b),
            "mean_se_boot": b["se_boot"].mean(),
            "se_boot_ratio": b["se_boot"].mean() / sd,
            "cov_boot_se": _covers(b["tau"] - Z95 * b["se_boot"],
                                   b["tau"] + Z95 * b["se_boot"], truth),
            "cov_boot_pct": _covers(b["ci_boot_percentile_lo"],
                                    b["ci_boot_percentile_hi"], truth),
            "cov_if_on_boot_reps": _covers(b["tau"] - Z95 * b["se"],
                                           b["tau"] + Z95 * b["se"], truth),
            "width_boot_se": (2 * Z95 * b["se_boot"]).mean(),
            "width_boot_pct": (b["ci_boot_percentile_hi"]
                               - b["ci_boot_percentile_lo"]).mean(),
            "boot_failed_total": b["boot_failed"].sum(),
            "boot_arm_loss_total": b["boot_arm_loss"].sum(),
            "boot_arm_loss_reps": int((b["boot_arm_loss"] > 0).sum()),
        })
    return pd.Series(out)


def summarize(rows):
    return (rows.groupby(["scenario", "config_id", "estimator"], sort=False)
            .apply(score, include_groups=False).reset_index())


def load_rows(out_dir):
    files = sorted(Path(out_dir, "rows").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no row files in {out_dir}/rows")
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = rows.duplicated(["scenario", "rep", "config_id", "estimator"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate (scenario, rep, config, estimator) rows")
    return rows


def main():
    out_dir = Path(sys.argv[1])
    summary = summarize(load_rows(out_dir))
    summary.to_csv(out_dir / "summary.csv", index=False)
    cols = ["scenario", "config_id", "estimator", "n_reps", "n_undefined", "bias",
            "bias_mcse", "median_bias", "emp_sd", "se_if_ratio", "cov_if",
            "n_boot_reps", "se_boot_ratio", "cov_boot_se", "cov_boot_pct",
            "cov_if_on_boot_reps", "boot_arm_loss_total"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 500, "display.float_format", "{:.3f}".format):
        print(summary[[c for c in cols if c in summary]].to_string(index=False))
    print(f"-> {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
