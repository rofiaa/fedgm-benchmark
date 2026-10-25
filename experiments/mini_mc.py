"""Mini Monte Carlo: Fed-IPW / Fed-AIPW with logistic membership weights, poor local
overlap; influence-function vs site-stratified bootstrap inference.

For each scenario, draws `--reps` datasets with the authors' generator and runs
fed_ps.estimate for every method below. Methods with the pooled MW fit also get a
site-stratified bootstrap (`--boot` replicates, default fed_ps.BOOTSTRAP_B); all methods
in a rep share the bootstrap seed, so they see the same resamples. FedAvg rows report IF
inference only: their bootstrap would refit with the pooled MW (fed_ps docs) and so
duplicate the pooled rows. The authors' site-wise "Hajek" is parity-only and not run
(docs/findings.md, F2).

Per scenario x method: mean and median bias vs the true ATE, empirical SD, mean IF SE,
mean bootstrap SE, RMSE, 95% coverage and width of the IF CI, the bootstrap-SE normal
CI and the bootstrap percentile CI, bootstrap replicates in which a site lost an arm,
and FedAvg rounds reached.

Scenarios (paper's App. D values unless noted; see src/dgps/authors.py):
  A          DGP A, n_k = 650, paper mu_3 = (2,2,2,1,..)
  A_mu3nb    DGP A, n_k = 650, notebook mu_3 = (3,3,3,1,..)
  B          DGP B, n = 4000

Usage: python experiments/mini_mc.py [--reps 100] [--boot 200] [--jobs 7] [--seed 2026]
Writes <out>/mini_mc_reps.csv (one row per rep x method) and <out>/mini_mc_summary.csv.
"""

import os

# one BLAS/OpenMP thread per worker process; set before numpy is imported
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dgps.authors import (  # noqa: E402
    MU3_NOTEBOOK,
    dgp_a_poor_overlap_params,
    dgp_b_poor_overlap_params,
    generate,
    true_ate,
)
from src.estimators.fed_ps import BOOTSTRAP_B, estimate  # noqa: E402

SCENARIOS = {
    "A": ("A", dgp_a_poor_overlap_params),
    "A_mu3nb": ("A", lambda: dgp_a_poor_overlap_params(mu3_first=MU3_NOTEBOOK)),
    "B": ("B", dgp_b_poor_overlap_params),
}
# (label, estimate kwargs, bootstrap?)
METHODS = [
    ("IPW-HT", dict(mw="logistic", ipw_type="HT"), True),
    ("IPW-Hajek_pooled", dict(mw="logistic", ipw_type="Hajek_pooled"), True),
    ("AIPW", dict(mw="logistic", aipw=True), True),
    ("IPW-HT", dict(mw="logistic_fedavg", ipw_type="HT"), False),
    ("IPW-Hajek_pooled", dict(mw="logistic_fedavg", ipw_type="Hajek_pooled"), False),
    ("AIPW", dict(mw="logistic_fedavg", aipw=True), False),
]


def run_rep(name, rep, seed, tau_true, boot):
    """All methods on one dataset; returns a list of row dicts."""
    dgp, make_params = SCENARIOS[name]
    _, df = generate(make_params(), dgp, seed)
    rows = []
    for label, kwargs, use_boot in METHODS:
        row = dict(scenario=name, dgp=dgp, rep=rep, seed=seed, method=label,
                   mw=kwargs["mw"], tau_true=tau_true)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = estimate(df, **kwargs, bootstrap=boot if use_boot else 0, seed=seed)
            row.update(tau=out["tau"], se=out["se"], max_smd=out["max_smd"],
                       min_e_hat=float(out["e_hat"].min()),
                       max_e_hat=float(out["e_hat"].max()),
                       mw_rounds=out["mw_rounds"], mw_converged=out["mw_converged"],
                       runtime=out["runtime"], error="")
            if use_boot:
                lo, hi = out["ci_boot_percentile"]
                row.update(se_boot=out["se_boot"], boot_pct_lo=lo, boot_pct_hi=hi,
                           boot_ok=out["boot_ok"], boot_failed=out["boot_failed"],
                           boot_arm_loss=out["boot_arm_loss"],
                           **{f"boot_arm_loss_site{k}": v
                              for k, v in out["boot_arm_loss_by_site"].items()})
        except Exception as exc:  # record and keep going
            row.update(tau=np.nan, se=np.nan, error=repr(exc))
        rows.append(row)
    return rows


def run_scenario(name, reps, seed_seq, truth_n, boot, pool):
    dgp, make_params = SCENARIOS[name]
    truth_seed, *rep_seeds = (
        int(s.generate_state(1)[0]) for s in seed_seq.spawn(reps + 1)
    )
    tau_true, tau_true_mcse = true_ate(make_params(), dgp, truth_n, truth_seed)
    print(f"[{name}] true ATE {tau_true:.4f} (MC SE {tau_true_mcse:.4f})", flush=True)

    futures = [pool.submit(run_rep, name, rep, seed, tau_true, boot)
               for rep, seed in enumerate(rep_seeds)]
    rows = []
    for i, fut in enumerate(futures, 1):
        rows.extend(fut.result())
        if i % 20 == 0:
            print(f"[{name}] {i}/{reps} reps", flush=True)
    return pd.DataFrame(rows)


def _covered(lo, hi, truth):
    return ((lo <= truth) & (truth <= hi)).mean()


def summarize(reps_df):
    z = 1.959963984540054

    def one(g):
        ok = g.dropna(subset=["tau", "se"])
        err = ok["tau"] - ok["tau_true"]
        truth = ok["tau_true"]
        out = {
            "tau_true": g["tau_true"].iloc[0],
            "n_ok": len(ok),
            "n_failed": len(g) - len(ok),
            "mean_tau": ok["tau"].mean(),
            "bias": err.mean(),
            "bias_mcse": ok["tau"].std(ddof=1) / np.sqrt(len(ok)),
            "median_bias": err.median(),
            "emp_sd": ok["tau"].std(ddof=1),
            "rmse": np.sqrt(np.mean(err**2)),
            "mean_se_if": ok["se"].mean(),
            "se_if_ratio": ok["se"].mean() / ok["tau"].std(ddof=1),
            "cov_if": _covered(ok["tau"] - z * ok["se"], ok["tau"] + z * ok["se"], truth),
            "width_if": (2 * z * ok["se"]).mean(),
            "mean_max_smd": ok["max_smd"].mean(),
            "mean_runtime_s": ok["runtime"].mean(),
            "mw_rounds_median": ok["mw_rounds"].median(),
            "mw_rounds_max": ok["mw_rounds"].max(),
            "mw_converged_frac": ok["mw_converged"].dropna().astype(bool).mean()
            if ok["mw_converged"].notna().any() else np.nan,
        }
        boot_keys = ["mean_se_boot", "se_boot_ratio", "cov_boot_se", "width_boot_se",
                     "cov_boot_pct", "width_boot_pct", "boot_failed_total",
                     "boot_arm_loss_total", "boot_arm_loss_reps"]
        out.update(dict.fromkeys(boot_keys, np.nan))  # same columns for every group
        if "se_boot" in ok and ok["se_boot"].notna().any():
            sb = ok["se_boot"]
            out.update({
                "mean_se_boot": sb.mean(),
                "se_boot_ratio": sb.mean() / ok["tau"].std(ddof=1),
                "cov_boot_se": _covered(ok["tau"] - z * sb, ok["tau"] + z * sb, truth),
                "width_boot_se": (2 * z * sb).mean(),
                "cov_boot_pct": _covered(ok["boot_pct_lo"], ok["boot_pct_hi"], truth),
                "width_boot_pct": (ok["boot_pct_hi"] - ok["boot_pct_lo"]).mean(),
                "boot_failed_total": ok["boot_failed"].sum(),
                "boot_arm_loss_total": ok["boot_arm_loss"].sum(),
                "boot_arm_loss_reps": int((ok["boot_arm_loss"] > 0).sum()),
            })
        return pd.Series(out)

    return (
        reps_df.groupby(["scenario", "mw", "method"], sort=False)
        .apply(one, include_groups=False)
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--boot", type=int, default=BOOTSTRAP_B,
                        help="bootstrap replicates per estimate (0 = IF only)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--truth-n", type=int, default=1_000_000,
                        help="sample size for the Monte Carlo true ATE")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    start = time.perf_counter()
    root = np.random.SeedSequence(args.seed)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        reps_df = pd.concat(
            [run_scenario(name, args.reps, seq, args.truth_n, args.boot, pool)
             for name, seq in zip(args.scenarios, root.spawn(len(args.scenarios)))],
            ignore_index=True,
        )
    summary = summarize(reps_df)

    args.out.mkdir(parents=True, exist_ok=True)
    reps_df.to_csv(args.out / "mini_mc_reps.csv", index=False)
    summary.to_csv(args.out / "mini_mc_summary.csv", index=False)

    cols = ["scenario", "mw", "method", "bias", "bias_mcse", "median_bias", "emp_sd",
            "mean_se_if", "cov_if", "mean_se_boot", "cov_boot_se", "cov_boot_pct",
            "boot_arm_loss_total", "mw_rounds_max"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(summary[[c for c in cols if c in summary]])
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}/mini_mc_*.csv")


if __name__ == "__main__":
    main()
