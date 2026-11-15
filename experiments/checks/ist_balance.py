"""Balance on the IST benchmark beyond max |SMD|: SMD with the pooled and the treated-only
SD, variance ratios and KS statistics, for every IST dataset with a GenMatch result.

Methods (each on the same dataset as in the runs): raw sample; FedDSM (ps="x"; also
ps="site" in the observational scenarios); GenMatch, rebuilt from its saved weights
(genmatch.rematch: no new search; tau_rematch is checked against the saved tau);
Mahalanobis matching; pooled-PS IPW; the paper's Fed-IPW (FedAvg membership).
Balance is measured on the 31 patient covariates for every method (one covariate set);
max_smd_site adds the country indicators in the observational scenarios.

Runs where GenMatch ran (Narval: the pooled PS logit GenMatch matches on is machine-
dependent, so a rebuilt match agrees exactly only on the machine of the run):
    python -m experiments.checks.ist_balance --jobs 8
Writes results/checks/ist_balance_rows.csv and ist_balance_summary.csv.
"""

import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "checks"
RUNS = {"ist_cc": "ATE", "ist_att_cc": "ATT", "ist_single_arm_cc": "ATT"}
N_COV = 31


def _ps_weights(e, W, target):
    e = np.clip(np.asarray(e, float), 1e-8, 1 - 1e-8)
    return np.where(W == 1, 1 / e, 1 / (1 - e)) if target == "ATE" \
        else np.where(W == 1, 1.0, e / (1 - e))


def one_dataset(task):
    run, scenario, rep, target = task
    warnings.simplefilter("ignore")
    from src.dgps import ist
    from src.estimators import fed_ps, feddsm, genmatch, pooled_ps
    from src.estimators.balance import balance_summary, balance_table

    saved = pd.read_csv(ROOT / "results" / run / "rows" / scenario / f"rep{rep:04d}.csv")
    df = ist.load(rep, scenario, target=target)
    obs = df.attrs["design"] != "single_arm"
    dfc = ist.with_site_dummies(df) if obs else df  # what the centralized methods saw
    W = df["W"].to_numpy()
    cov = df[[f"X{i}" for i in range(1, N_COV + 1)]]
    site = dfc[[c for c in dfc if c.startswith("X") and int(c[1:]) > N_COV]]

    weights = {"raw": np.ones(len(df))}
    taus = {}
    for ps in (("x", "site") if obs else ("x",)):
        r = feddsm.estimate_multi(df, ["FedDSM"], target=target, ps=ps,
                                  seed=int(saved.loc[saved.config_id == f"feddsm_{ps}",
                                                     "config_seed"].iloc[0]))["FedDSM"]
        weights[f"FedDSM ({ps})"], taus[f"FedDSM ({ps})"] = r["unit_weights"].to_numpy(), r["tau"]
    g = saved[saved.estimator == "GM"].iloc[0]
    gm_w = {c[len("gm_weights_"):]: g[c] for c in saved.columns if c.startswith("gm_weights_")}
    tau_gm, w_gm = genmatch.rematch(dfc, gm_w, target=target)
    weights["GenMatch"], taus["GenMatch"] = w_gm.to_numpy(), tau_gm
    r = genmatch.estimate_multi(dfc, ["Maha"], target=target)["Maha"]
    weights["Mahalanobis"], taus["Mahalanobis"] = r["unit_weights"].to_numpy(), r["tau"]
    p = "ATT-" if target == "ATT" else ""
    r = pooled_ps.estimate_multi(dfc, [p + "IPW-Hajek_pooled"])[p + "IPW-Hajek_pooled"]
    weights["Pooled PS IPW"], taus["Pooled PS IPW"] = _ps_weights(r["e_hat"], W, target), r["tau"]
    if obs:
        r = fed_ps.estimate_multi(df, [p + "IPW-Hajek_pooled"], mw="logistic_fedavg",
                                  seed=1)[p + "IPW-Hajek_pooled"]
        weights["Fed-IPW (paper, FedAvg)"] = _ps_weights(r["e_hat"], W, target)
        taus["Fed-IPW (paper, FedAvg)"] = r["tau"]

    rows = []
    for method, w in weights.items():
        row = dict(run=run, scenario=scenario, rep=rep, target=target, method=method,
                   tau=taus.get(method, np.nan), tau_true=df.attrs["tau_true"],
                   **balance_summary(balance_table(cov, W, w)))
        if obs:
            row["max_smd_site"] = float(balance_table(site, W, w).smd_pooled.abs().max())
        if method == "GenMatch":
            row["tau_saved"] = float(g.tau)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    args = ap.parse_args()
    tasks = []
    for run in args.runs:
        for f in sorted((ROOT / "results" / run / "rows").glob("*/rep*.csv")):
            d = pd.read_csv(f, usecols=["estimator", "tau"])
            if ((d.estimator == "GM") & d.tau.notna()).any():
                tasks.append((run, f.parent.name, int(f.stem[3:]), RUNS[run]))
    print(f"{len(tasks)} datasets")
    with Pool(args.jobs) as pool:
        rows = [r for rs in pool.imap_unordered(one_dataset, tasks) for r in rs]
    out = pd.DataFrame(rows).sort_values(["run", "scenario", "rep", "method"])
    out["abs_err_pp"] = (out.tau - out.tau_true).abs() * 100
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "ist_balance_rows.csv", index=False)
    gm = out[out.method == "GenMatch"]
    print("GenMatch rebuilt: max |tau_rematch - tau_saved| =",
          float((gm.tau - gm.tau_saved).abs().max()))
    cols = ["max_smd_pooled", "max_smd_treated", "share_smd_gt_0_1", "worst_var_ratio",
            "share_vr_outside", "max_ks"] + (["max_smd_site"] if "max_smd_site" in out else [])
    summ = out.groupby(["target", "scenario", "method"]).agg(
        n=("rep", "size"), **{c: (c, "median") for c in cols},
        rmse_pp=("abs_err_pp", lambda e: float(np.sqrt(np.mean(e ** 2)))))
    summ.round(4).to_csv(OUT / "ist_balance_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
