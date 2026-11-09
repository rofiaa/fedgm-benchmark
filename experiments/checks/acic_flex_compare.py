"""FedDSM vs the centralized flexible-outcome references on ACIC 2016 (docs/findings.md N10).

Paired on (scenario, rep): both runs use the same 77 settings x 20 replications.
Writes results/checks/acic_flex_{overall,paired,knobs}.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "checks"
RUNS = {"ATE": ("feddsm_acic", "feddsm_acic_flex"), "ATT": ("feddsm_acic_att", "feddsm_acic_flex_att")}
KEEP = {"feddsm_x": ("FedDSM", "FedDSM-noBA", "FedAIPW-quad"),
        "fed_fedavg": ("AIPW", "ATT-AIPW")}


def load(run):
    rows = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "results" / run / "rows").glob("*/rep*.csv")))
    return rows.assign(err=rows["tau"] - rows["tau_true"], setting=rows["scenario"].str[4:].astype(int))


def frame(estimand):
    base, flex = (load(r) for r in RUNS[estimand])
    keep = base[[e in KEEP.get(c, ()) for c, e in zip(base.config_id, base.estimator)]].copy()
    keep["method"] = np.where(keep.config_id == "fed_fedavg", "Fed-AIPW (paper)", keep.estimator)
    flex["method"] = flex.estimator
    cols = ["scenario", "setting", "rep", "method", "err", "max_smd"]
    return pd.concat([keep[cols], flex[cols]])


def per_setting(d):
    return d.groupby(["setting", "method"]).err.agg(
        rmse=lambda e: np.sqrt(np.mean(e ** 2)), bias="mean", mae=lambda e: e.abs().mean(),
        sd="std", worst=lambda e: e.abs().max()).reset_index()


def overall(d, estimand):
    p = per_setting(d).groupby("method")
    out = p.agg(rmse=("rmse", "mean"), median_rmse=("rmse", "median"), worst_rmse=("rmse", "max"),
                mae=("mae", "mean"), mean_bias=("bias", "mean"), sd=("sd", "mean")).reset_index()
    out["blowups"] = out.method.map(d.assign(b=d.err.abs() > 2).groupby("method").b.sum())
    out["n_undefined"] = out.method.map(d.groupby("method").err.apply(lambda e: e.isna().sum()))
    out.insert(0, "estimand", estimand)
    return out.sort_values("rmse")


def paired(d, estimand, ref="FedDSM"):
    """FedDSM minus each method; se over the 77 settings (settings are the unit)."""
    p = per_setting(d).pivot(index="setting", columns="method", values="rmse")
    abs_err = d.assign(a=d.err.abs()).pivot_table(index="setting", columns="method", values="a")
    out = []
    for m in p.columns.drop(ref):
        diff, mdiff = p[ref] - p[m], abs_err[ref] - abs_err[m]
        out.append(dict(estimand=estimand, method=m, rmse_diff=diff.mean(),
                        rmse_diff_se=diff.std() / np.sqrt(len(diff)),
                        mae_diff=mdiff.mean(), mae_diff_se=mdiff.std() / np.sqrt(len(mdiff)),
                        feddsm_better=int((diff < 0).sum()), settings=len(diff)))
    return pd.DataFrame(out).sort_values("rmse_diff")


def knobs(d, estimand):
    from src.dgps.acic2016 import settings
    par = settings().set_index("setting")
    p = per_setting(d).join(par, on="setting")
    out = []
    for k in par.columns:
        t = p.groupby([k, "method"]).rmse.mean().reset_index().rename(columns={k: "level"})
        out.append(t.assign(knob=k, estimand=estimand))
    return pd.concat(out)


def main():
    res = {"overall": [], "paired": [], "knobs": []}
    for estimand in RUNS:
        d = frame(estimand)
        res["overall"].append(overall(d, estimand))
        res["paired"].append(paired(d, estimand))
        res["knobs"].append(knobs(d, estimand))
    OUT.mkdir(exist_ok=True)
    for k, v in res.items():
        t = pd.concat(v)
        t.to_csv(OUT / f"acic_flex_{k}.csv", index=False)
        if k != "knobs":
            print(t.round(3).to_string(index=False), "\n")
    t = pd.concat(res["knobs"])
    for est in RUNS:
        w = t[(t.estimand == est) & t.method.isin(["FedDSM", "AIPW-RF", "AIPW-RF-rfps", "REG-RF"])
              ].pivot_table(index=["knob", "level"], columns="method", values="rmse", sort=False)
        print(est, "\n", w.round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
