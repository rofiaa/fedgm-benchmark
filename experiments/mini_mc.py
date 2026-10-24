"""Mini Monte Carlo: Fed-IPW with logistic membership weights, poor local overlap.

For each scenario, draws `--reps` datasets with the authors' generator and runs
fed_ps.estimate with mw in {pooled logistic, FedAvg logistic} x ipw_type in {HT, Hajek}.
Reports bias vs the true ATE, empirical SD, mean IF SE, RMSE, 95% CI coverage and width.

Scenarios (sample sizes follow the paper, App. D; everything else follows the notebook):
  A        DGP A, n_k = 650 per site (notebook: 500), notebook mu_3 = (3,3,3,1,..)
  A_mu3p   same, but the paper's mu_3 = (2,2,2,1,..) (Table 3) -- for the Fig. 2 comparison
  B        DGP B, n = 4000 (notebook: 1500)

Usage: python experiments/mini_mc.py [--reps 100] [--seed 2026] [--out results]
Writes <out>/mini_mc_reps.csv (one row per rep x method) and <out>/mini_mc_summary.csv.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dgps.authors import (  # noqa: E402
    dgp_a_poor_overlap_params,
    dgp_b_poor_overlap_params,
    generate,
    true_ate,
)
from src.estimators.fed_ps import estimate  # noqa: E402

SCENARIOS = {
    "A": ("A", lambda: dgp_a_poor_overlap_params(n_k=650)),
    "A_mu3p": ("A", lambda: dgp_a_poor_overlap_params(n_k=650, mu3_first=2)),
    "B": ("B", lambda: dgp_b_poor_overlap_params(n=4000)),
}
METHODS = [
    (mw, ipw_type)
    for mw in ("logistic", "logistic_fedavg")
    for ipw_type in ("HT", "Hajek")
]


def run_scenario(name, reps, seed_seq, truth_n):
    dgp, make_params = SCENARIOS[name]
    params = make_params()
    truth_seed, *rep_seeds = (
        int(s.generate_state(1)[0]) for s in seed_seq.spawn(reps + 1)
    )
    tau_true, tau_true_mcse = true_ate(params, dgp, truth_n, truth_seed)
    print(f"[{name}] true ATE {tau_true:.4f} (MC SE {tau_true_mcse:.4f})", flush=True)

    rows = []
    for rep, seed in enumerate(rep_seeds):
        _, df = generate(params, dgp, seed)
        for mw, ipw_type in METHODS:
            row = dict(scenario=name, dgp=dgp, rep=rep, seed=seed, mw=mw,
                       ipw_type=ipw_type, tau_true=tau_true)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = estimate(df, mw=mw, ipw_type=ipw_type)
                row.update(tau=out["tau"], se=out["se"], max_smd=out["max_smd"],
                           min_e_hat=float(out["e_hat"].min()),
                           max_e_hat=float(out["e_hat"].max()),
                           runtime=out["runtime"], error="")
            except Exception as exc:  # record and keep going
                row.update(tau=np.nan, se=np.nan, error=repr(exc))
            rows.append(row)
        if (rep + 1) % 20 == 0:
            print(f"[{name}] {rep + 1}/{reps} reps", flush=True)
    return pd.DataFrame(rows)


def summarize(reps_df):
    def one(g):
        ok = g.dropna(subset=["tau", "se"])
        err = ok["tau"] - ok["tau_true"]
        covered = (ok["tau"] - 1.96 * ok["se"] <= ok["tau_true"]) & (
            ok["tau_true"] <= ok["tau"] + 1.96 * ok["se"]
        )
        return pd.Series({
            "tau_true": g["tau_true"].iloc[0],
            "n_ok": len(ok),
            "n_failed": len(g) - len(ok),
            "mean_tau": ok["tau"].mean(),
            "bias": err.mean(),
            "bias_mcse": ok["tau"].std(ddof=1) / np.sqrt(len(ok)),
            "emp_sd": ok["tau"].std(ddof=1),
            "mean_se": ok["se"].mean(),
            "se_ratio": ok["se"].mean() / ok["tau"].std(ddof=1),
            "rmse": np.sqrt(np.mean(err**2)),
            "coverage": covered.mean(),
            "ci_width": (2 * 1.96 * ok["se"]).mean(),
            "mean_max_smd": ok["max_smd"].mean(),
            "mean_runtime_s": ok["runtime"].mean(),
        })

    return (
        reps_df.groupby(["scenario", "mw", "ipw_type"], sort=False)
        .apply(one, include_groups=False)
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--truth-n", type=int, default=1_000_000,
                        help="sample size for the Monte Carlo true ATE")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    start = time.perf_counter()
    root = np.random.SeedSequence(args.seed)
    reps_df = pd.concat(
        [run_scenario(name, args.reps, seq, args.truth_n)
         for name, seq in zip(args.scenarios, root.spawn(len(args.scenarios)))],
        ignore_index=True,
    )
    summary = summarize(reps_df)

    args.out.mkdir(parents=True, exist_ok=True)
    reps_df.to_csv(args.out / "mini_mc_reps.csv", index=False)
    summary.to_csv(args.out / "mini_mc_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(summary[["scenario", "mw", "ipw_type", "tau_true", "n_failed", "bias",
                       "bias_mcse", "emp_sd", "mean_se", "rmse", "coverage",
                       "ci_width"]])
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}/mini_mc_*.csv")


if __name__ == "__main__":
    main()
