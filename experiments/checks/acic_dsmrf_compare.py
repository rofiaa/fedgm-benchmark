"""Double-score matching with forest prognostic scores vs FedDSM and the flexible
references on ACIC 2016 (docs/findings.md N11). Paired on (setting, rep).

Reading rule, fixed before the run: the share of FedDSM's paired RMSE gap to AIPW RF + RF
closed by DSM-RF-rfps. At least 75%: the outcome model was the limit; under 25%: matching
itself costs accuracy; else partial.

Writes results/checks/acic_dsmrf_{overall,paired,knobs}.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.dgps.acic2016 import settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "checks"
RUNS = {"ATE": ("feddsm_acic", "feddsm_acic_flex", "feddsm_acic_dsmrf"),
        "ATT": ("feddsm_acic_att", "feddsm_acic_flex_att", "feddsm_acic_dsmrf_att")}
REF, BEST = "AIPW-RF-rfps", "DSM-RF-rfps"


def load(run):
    rows = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "results" / run / "rows").glob("*/rep*.csv")))
    return rows.assign(err=rows["tau"] - rows["tau_true"], setting=rows["scenario"].str[4:].astype(int))


def frame(estimand):
    base, flex, dsm = (load(r) for r in RUNS[estimand])
    base = base[(base.config_id == "feddsm_x") & (base.estimator == "FedDSM")]
    flex = flex[flex.estimator.isin(["REG-RF", "REG-quad"])]
    return pd.concat([base, flex, dsm])[["setting", "rep", "estimator", "err", "max_smd"]].rename(
        columns={"estimator": "method"})


def per_setting(d):
    return d.groupby(["setting", "method"]).err.agg(
        rmse=lambda e: np.sqrt(np.mean(e ** 2)), bias="mean", mae=lambda e: e.abs().mean()).reset_index()


def overall(d, estimand):
    t = per_setting(d).groupby("method").agg(
        rmse=("rmse", "mean"), median_rmse=("rmse", "median"), worst_rmse=("rmse", "max"),
        mae=("mae", "mean"), mean_bias=("bias", "mean")).reset_index()
    g = d.groupby("method")
    t["blowups"] = t.method.map(g.err.apply(lambda e: (e.abs() > 2).sum()))
    t["max_abs_err"] = t.method.map(g.err.apply(lambda e: e.abs().max()))
    t["n_undefined"] = t.method.map(g.err.apply(lambda e: e.isna().sum()))
    t["max_smd"] = t.method.map(g.max_smd.mean())
    t["balance_le_0_12"] = t.method.map(
        d.groupby(["setting", "method"]).max_smd.mean().le(0.12).groupby("method").sum())
    t.insert(0, "estimand", estimand)
    return t.sort_values("rmse")


def paired(d, estimand):
    p = per_setting(d).pivot(index="setting", columns="method", values="rmse")
    base_gap = (p.FedDSM - p[REF]).mean()
    out = []
    for m in p.columns.drop(REF):
        diff = p[m] - p[REF]
        out.append(dict(estimand=estimand, method=m, vs=REF, rmse_diff=diff.mean(),
                        se=diff.std() / np.sqrt(len(diff)), wins=int((diff < 0).sum()),
                        share_of_feddsm_gap_closed=1 - diff.mean() / base_gap))
    for m, ref in (("DSM-RF", "REG-RF"), (BEST, "REG-RF"), ("DSM-RF", "FedDSM"), (BEST, "FedDSM"),
                   ("DSM-RF", BEST)):
        diff = p[m] - p[ref]
        out.append(dict(estimand=estimand, method=m, vs=ref, rmse_diff=diff.mean(),
                        se=diff.std() / np.sqrt(len(diff)), wins=int((diff < 0).sum())))
    return pd.DataFrame(out)


def knobs(d, estimand):
    par = settings().set_index("setting")
    p = per_setting(d).join(par, on="setting")
    return pd.concat(p.groupby([k, "method"], sort=False).rmse.mean().reset_index().rename(
        columns={k: "level"}).assign(knob=k, estimand=estimand) for k in par.columns)


def main():
    res = {"overall": [], "paired": [], "knobs": []}
    for estimand in RUNS:
        d = frame(estimand)
        res["overall"].append(overall(d, estimand))
        res["paired"].append(paired(d, estimand))
        res["knobs"].append(knobs(d, estimand))
    for k, v in res.items():
        t = pd.concat(v)
        t.to_csv(OUT / f"acic_dsmrf_{k}.csv", index=False)
        if k != "knobs":
            print(t.round(3).to_string(index=False), "\n")
    t = pd.concat(res["knobs"])
    for est in RUNS:
        w = t[(t.estimand == est) & t.method.isin(["FedDSM", "DSM-RF", BEST, REF])].pivot_table(
            index=["knob", "level"], columns="method", values="rmse", sort=False)
        print(est, "\n", w.round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
