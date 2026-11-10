"""Binned-score FedDSM (and FedDSM-clip) vs the quadratic FedDSM, paired on (scenario,
rep), and on ACIC vs centralized forest-score matching. docs/findings.md N13 (rule fixed
before the run).

Writes results/checks/feddsm_binned_{acic,cells,choices}.csv.
"""

import numpy as np
import pandas as pd

from experiments.checks.feddsm_tuned_compare import DSM_RF, OUT, load, rmse

CELLS = {("ACIC", "ATE"): ("feddsm_acic", "feddsm_acic_binned", "feddsm_x"),
         ("ACIC", "ATT"): ("feddsm_acic_att", "feddsm_acic_binned_att", "feddsm_x"),
         ("Authors", "ATE"): ("feddsm_authors", "feddsm_authors_binned", "feddsm_x"),
         ("Authors", "ATT"): ("feddsm_authors_att", "feddsm_authors_binned_att", "feddsm_x"),
         ("IHDP", "ATE"): ("feddsm_ihdp", "feddsm_ihdp_binned", "feddsm_site"),
         ("IHDP", "ATT"): ("feddsm_ihdp_att", "feddsm_ihdp_binned_att", "feddsm_site")}
# (estimator in the binned run, estimator in the quadratic run it is paired with)
PAIRS = [("FedDSM", "FedDSM"), ("FedDSM-noBA", "FedDSM-noBA"), ("FedDSM-clip", "FedDSM"),
         ("FedAIPW-quad", "FedAIPW-quad")]


def paired(old, new, name, ref):
    keys = ["scenario", "rep"]
    return pd.DataFrame({"quad": old[old.estimator == ref].set_index(keys).err,
                         "new": new[new.estimator == name].set_index(keys).err}).dropna()


def acic_row(est, name, p, new):
    per = p.groupby(level="scenario").agg(quad=("quad", rmse), new=("new", rmse))
    row = dict(estimand=est, estimator=name, rmse_quad=per.quad.mean(), rmse_new=per.new.mean(),
               median_quad=per.quad.median(), median_new=per.new.median(), worst_new=per.new.max(),
               diff=(per.new - per.quad).mean(), diff_se=(per.new - per.quad).std() / np.sqrt(len(per)),
               wins=int((per.new < per.quad).sum()), blowups_quad=int((p.quad.abs() > 2).sum()),
               blowups_new=int((p.new.abs() > 2).sum()), max_abs_new=p.new.abs().max(),
               mean_bias_new=p.groupby(level="scenario").new.mean().mean())
    if name != "FedAIPW-quad":
        rf = load(DSM_RF[est])
        prf = rf[rf.estimator == "DSM-RF"].set_index(["scenario", "rep"]).err.groupby(
            level="scenario").apply(rmse).reindex(per.index)
        row.update(rmse_dsm_rf=prf.mean(), gap_closed=1 - (per.new - prf).mean() / (per.quad - prf).mean(),
                   diff_vs_dsm_rf=(per.new - prf).mean(),
                   diff_vs_dsm_rf_se=(per.new - prf).std() / np.sqrt(len(per)),
                   beats_dsm_rf=int((per.new < prf).sum()))
    row["balance_le_0_12"] = int((new[new.estimator == name].groupby("scenario").max_smd.mean() <= 0.12).sum())
    return row


def main():
    acic, cells, choices = [], [], []
    for (data, est), (o, n, cfg) in CELLS.items():
        old, new = load(o), load(n)
        old = old[old.config_id == cfg]
        x = new[new.estimator == "FedDSM"]
        for col in ("score_basis_0", "score_basis_1"):
            if col in new and new[col].notna().any():
                choices.append(x[col].value_counts(normalize=True).rename("share").reset_index().rename(
                    columns={col: "basis"}).assign(data=data, estimand=est, arm=col[-1],
                                                   median_lambda=x[f"score_lambda_{col[-1]}"].median()))
        for name, ref in PAIRS:
            p = paired(old, new, name, ref)
            if data == "ACIC":
                acic.append(acic_row(est, name, p, new))
                continue
            for sc, g in p.groupby(level="scenario"):
                d = g.new.abs() - g.quad.abs()
                cells.append(dict(data=data, estimand=est, scenario=sc, estimator=name, n=len(g),
                                  rmse_quad=rmse(g.quad), rmse_new=rmse(g.new),
                                  rmse_change=rmse(g.new) - rmse(g.quad), abs_err_diff=d.mean(),
                                  abs_err_diff_se=d.std() / np.sqrt(len(d))))
    pd.DataFrame(acic).to_csv(OUT / "feddsm_binned_acic.csv", index=False)
    pd.DataFrame(cells).to_csv(OUT / "feddsm_binned_cells.csv", index=False)
    pd.concat(choices).to_csv(OUT / "feddsm_binned_choices.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(acic).round(3).T.to_string())
    print(pd.DataFrame(cells).query("estimator in ['FedDSM', 'FedDSM-clip']").round(3).to_string(index=False))
    print(pd.concat(choices).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
