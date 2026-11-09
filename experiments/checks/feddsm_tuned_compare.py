"""Tuned-score FedDSM vs the quadratic FedDSM (paired on scenario, rep), and on ACIC vs
centralized forest-score matching. docs/findings.md N12 (rule fixed before the run).

Writes results/checks/feddsm_tuned_{acic,cells,choices}.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "checks"
CELLS = {("ACIC", "ATE"): ("feddsm_acic", "feddsm_acic_tuned", "feddsm_x"),
         ("ACIC", "ATT"): ("feddsm_acic_att", "feddsm_acic_tuned_att", "feddsm_x"),
         ("Authors", "ATE"): ("feddsm_authors", "feddsm_authors_tuned", "feddsm_x"),
         ("Authors", "ATT"): ("feddsm_authors_att", "feddsm_authors_tuned_att", "feddsm_x"),
         ("IHDP", "ATE"): ("feddsm_ihdp", "feddsm_ihdp_tuned", "feddsm_site"),
         ("IHDP", "ATT"): ("feddsm_ihdp_att", "feddsm_ihdp_tuned_att", "feddsm_site")}
DSM_RF = {"ATE": "feddsm_acic_dsmrf", "ATT": "feddsm_acic_dsmrf_att"}


def load(run):
    rows = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "results" / run / "rows").glob("*/rep*.csv")))
    return rows.assign(err=rows["tau"] - rows["tau_true"])


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def paired(old, new, estimator="FedDSM"):
    keys = ["scenario", "rep"]
    a = old[old.estimator == estimator].set_index(keys).err
    b = new[new.estimator == estimator].set_index(keys).err
    return pd.DataFrame({"ols": a, "tuned": b}).dropna()


def main():
    acic, cells, choices = [], [], []
    for (data, est), (o, n, cfg) in CELLS.items():
        old, new = load(o), load(n)
        old = old[old.config_id == cfg]
        for basis_col in ("score_basis_0", "score_basis_1"):
            if basis_col in new and new[basis_col].notna().any():
                x = new[new.estimator == "FedDSM"]
                choices.append(x[basis_col].value_counts(normalize=True).rename("share").reset_index().rename(
                    columns={basis_col: "basis"}).assign(data=data, estimand=est, arm=basis_col[-1],
                                                        median_lambda=x[f"score_lambda_{basis_col[-1]}"].median()))
        for name in ("FedDSM", "FedDSM-noBA", "FedAIPW-quad"):
            p = paired(old, new, name)
            if data != "ACIC":
                for sc, g in p.groupby(level="scenario"):
                    d = g.tuned.abs() - g.ols.abs()
                    cells.append(dict(data=data, estimand=est, scenario=sc, estimator=name, n=len(g),
                                      rmse_ols=rmse(g.ols), rmse_tuned=rmse(g.tuned),
                                      rmse_change=rmse(g.tuned) - rmse(g.ols),
                                      abs_err_diff=d.mean(), abs_err_diff_se=d.std() / np.sqrt(len(d))))
                continue
            per = p.groupby(level="scenario").agg(ols=("ols", rmse), tuned=("tuned", rmse))
            row = dict(estimand=est, estimator=name, rmse_ols=per.ols.mean(), rmse_tuned=per.tuned.mean(),
                       median_ols=per.ols.median(), median_tuned=per.tuned.median(),
                       worst_tuned=per.tuned.max(), diff=(per.tuned - per.ols).mean(),
                       diff_se=(per.tuned - per.ols).std() / np.sqrt(len(per)),
                       wins=int((per.tuned < per.ols).sum()), blowups_ols=int((p.ols.abs() > 2).sum()),
                       blowups_tuned=int((p.tuned.abs() > 2).sum()), max_abs_tuned=p.tuned.abs().max(),
                       mean_bias_tuned=p.groupby(level="scenario").tuned.mean().mean())
            if name != "FedAIPW-quad":
                rf = load(DSM_RF[est])
                r = rf[rf.estimator == "DSM-RF"].set_index(["scenario", "rep"]).err
                prf = r.groupby(level="scenario").apply(rmse).reindex(per.index)
                row.update(rmse_dsm_rf=prf.mean(), gap_closed=1 - (per.tuned - prf).mean() / (per.ols - prf).mean(),
                           diff_vs_dsm_rf=(per.tuned - prf).mean(),
                           diff_vs_dsm_rf_se=(per.tuned - prf).std() / np.sqrt(len(per)),
                           beats_dsm_rf=int((per.tuned < prf).sum()))
            bal = new[new.estimator == name].groupby("scenario").max_smd.mean()
            row["balance_le_0_12"] = int((bal <= 0.12).sum())
            acic.append(row)
    pd.DataFrame(acic).to_csv(OUT / "feddsm_tuned_acic.csv", index=False)
    pd.DataFrame(cells).to_csv(OUT / "feddsm_tuned_cells.csv", index=False)
    pd.concat(choices).to_csv(OUT / "feddsm_tuned_choices.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(acic).round(3).T.to_string())
    print(pd.DataFrame(cells).query("estimator == 'FedDSM'").round(3).to_string(index=False))
    print(pd.concat(choices).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
