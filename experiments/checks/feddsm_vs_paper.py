"""FedDSM (prespecified quadratic score, primary PS) vs every version of Khellaf et al.'s
estimators in our runs (Fed-IPW / Fed-AIPW, all membership models), paired on (scenario,
rep). Khellaf et al. define the ATE; their ATT versions are our extension. LaLonde is not
included: it has no sites, so the federated estimators do not apply.

Writes results/checks/feddsm_vs_paper.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "checks" / "feddsm_vs_paper.csv"
CELLS = [("Authors' DGPs", "ATE", "feddsm_authors", "ps_mc_v2", "feddsm_x"),
         ("Authors' DGPs", "ATT", "feddsm_authors_att", "feddsm_authors_att", "feddsm_x"),
         ("IHDP", "ATE", "feddsm_ihdp", "ihdp_mc", "feddsm_site"),
         ("IHDP", "ATT", "feddsm_ihdp_att", "ihdp_att_cc", "feddsm_site"),
         ("ACIC 2016", "ATE", "feddsm_acic", "feddsm_acic", "feddsm_x"),
         ("ACIC 2016", "ATT", "feddsm_acic_att", "feddsm_acic_att", "feddsm_x")]


def load(run):
    d = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "results" / run / "rows").glob("*/rep*.csv")))
    return d.assign(err=d["tau"] - d["tau_true"])


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def compare(data, est, ours, paper, cfg):
    a, b = load(ours), load(paper)
    me = a[(a.config_id == cfg) & (a.estimator == "FedDSM")].set_index(["scenario", "rep"]).err
    # paper's: config fed_*, any Fed-IPW / Fed-AIPW version
    b = b[b.config_id.str.startswith("fed_")]
    out = []
    for (cf, en), g in b.groupby(["config_id", "estimator"]):
        p = pd.DataFrame({"me": me, "pa": g.set_index(["scenario", "rep"]).err}).dropna()
        if data == "ACIC 2016":
            per = p.groupby(level="scenario").agg(me=("me", rmse), pa=("pa", rmse))
            d = per.me - per.pa
            mn = p.assign(a=p.me.abs() - p.pa.abs()).groupby(level="scenario").a.mean()
            out.append(dict(data=data, estimand=est, scenario="All 77", config=cf, estimator=en, unit="setting",
                            n=len(per), rmse_feddsm=per.me.mean(), rmse_paper=per.pa.mean(),
                            rmse_diff=d.mean(), abs_err_diff=mn.mean(), abs_err_diff_se=mn.std() / np.sqrt(len(mn)),
                            feddsm_better_units=int((d < 0).sum())))
            continue
        for sc, s in p.groupby(level="scenario"):
            a_ = s.me.abs() - s.pa.abs()
            out.append(dict(data=data, estimand=est, scenario=sc, config=cf, estimator=en, unit="rep", n=len(s),
                            rmse_feddsm=rmse(s.me), rmse_paper=rmse(s.pa), rmse_diff=rmse(s.me) - rmse(s.pa),
                            abs_err_diff=a_.mean(), abs_err_diff_se=a_.std() / np.sqrt(len(a_)),
                            feddsm_better_units=int((a_ < 0).sum())))
    return out


def main():
    d = pd.DataFrame([r for c in CELLS for r in compare(*c)])
    d["headline"] = (d.config == "fed_fedavg") & d.estimator.str.endswith("AIPW")
    d["signif_better"] = d.abs_err_diff + 1.96 * d.abs_err_diff_se < 0
    d["signif_worse"] = d.abs_err_diff - 1.96 * d.abs_err_diff_se > 0
    d.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    head = d[d.headline].drop(columns=["headline", "unit", "n"])
    print(head.round(3).to_string(index=False))
    best = d.loc[d.groupby(["data", "estimand", "scenario"]).rmse_paper.idxmin()]
    print("\nBEST paper version per cell\n", best[["data", "estimand", "scenario", "config", "estimator", "rmse_feddsm",
          "rmse_paper", "rmse_diff", "abs_err_diff", "abs_err_diff_se", "signif_better", "signif_worse"]].round(3).to_string(index=False))
    for k, t in (("headline", d[d.headline]), ("best", best)):
        print(k, t.groupby("estimand").agg(cells=("rmse_diff", "size"), lower_rmse=("rmse_diff", lambda x: (x < 0).sum()),
              sig_better=("signif_better", "sum"), sig_worse=("signif_worse", "sum")).to_string())


if __name__ == "__main__":
    main()
