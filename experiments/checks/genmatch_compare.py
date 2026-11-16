"""Centralized R GenMatch (Narval) vs the other methods, on the datasets where GenMatch ran.

IST (results/ist_cc ATE, ist_att_cc ATT, ist_single_arm_cc ATT; 30 reps per scenario,
every method from the same Narval run) and ACIC 2016 ATT (results/gm_acic_att: rep 0 of
each of the 77 settings, paired with results/feddsm_acic_att and feddsm_acic_flex_att).

    python experiments/checks/genmatch_compare.py   # -> results/checks/genmatch_compare.csv

Paired rule as in N14: z = mean(e^2 - e_FedDSM^2) / SE over the shared datasets
(> 2: FedDSM better). IST in percentage points, ACIC in outcome units. FedDSM is the
scenario's primary PS (site-aware in obs_site).
Accuracy only. GM success = share of datasets with GenMatch's min balance p-value > 0.05.

Balance (bal_*; medians over datasets, share_smd_gt_0_1 a mean):
  IST: results/checks/ist_balance_rows.csv (experiments/checks/ist_balance.py), the same 31
       patient covariates for every method (the runs' own max_smd key is not comparable on
       IST: GenMatch's includes the country indicators). Only datasets the balance check
       covers (bal_n); GM and GM-BA share one match (as do Maha and Maha-BA); PS
       weighting balance is shown on the IPW rows.
  ACIC: the runs' max_smd key (max |SMD| over X, pooled SD of the unweighted sample; the
       same covariates for every method).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
IST = [("ist_cc", "ATE"), ("ist_att_cc", "ATT"), ("ist_single_arm_cc", "ATT")]
ACIC = ["gm_acic_att", "feddsm_acic_att", "feddsm_acic_flex_att", "feddsm_acic_dsmrf_att"]
BAL_ROWS = RES / "checks" / "ist_balance_rows.csv"
BAL_METHOD = {  # balance-check method -> labels in the runs ({t} = "" for ATE, "ATT-" for ATT)
    "FedDSM (x)": ["feddsm_x/FedDSM", "feddsm_x/FedDSM-noBA"],
    "FedDSM (site)": ["feddsm_site/FedDSM", "feddsm_site/FedDSM-noBA"],
    "GenMatch": ["gm_xps/GM", "gm_xps/GM-BA"],
    "Mahalanobis": ["gm_xps/Maha", "gm_xps/Maha-BA"],
    "Pooled PS IPW": ["pooled_logistic/{t}IPW-Hajek_pooled"],
    "Fed-IPW (paper, FedAvg)": ["fed_fedavg/{t}IPW-Hajek_pooled"],
    "raw": ["raw (unadjusted)"],
}
BAL_COLS = ["max_smd_pooled", "max_smd_treated", "share_smd_gt_0_1", "max_ks", "max_smd_site"]


def _load(name):
    d = pd.concat(pd.read_csv(f) for f in sorted((RES / name / "rows").glob("*/*.csv")))
    if "error" in d:
        assert d["error"].isna().all(), name
    d = d.copy()
    d["label"] = d["config_id"] + "/" + d["estimator"]
    d["err"] = d["tau"] - d["tau_true"]
    return d


def _table(d, ref_label, dataset, estimand, scenario, scale):
    keys = ["scenario", "rep"]
    gm_keys = d[d.config_id == "gm_xps"][keys].drop_duplicates()
    d = d.merge(gm_keys, on=keys)
    ref = d[d.label == ref_label].set_index(keys).err
    rows = []
    for lab, g in d.groupby("label"):
        e = g.set_index(keys).err.reindex(ref.index)
        if e.isna().any():
            continue
        dd = e ** 2 - ref ** 2
        sd = dd.std(ddof=1)
        row = dict(dataset=dataset, estimand=estimand, scenario=scenario, label=lab,
                   n=len(e), units="pp" if scale == 100 else "outcome",
                   bias=scale * e.mean(), rmse=scale * np.sqrt((e ** 2).mean()),
                   z_vs_feddsm=dd.mean() / (sd / np.sqrt(len(dd))) if sd > 0 else np.nan,
                   median_runtime_s=g.runtime.median())
        if "gm_min_pvalue" in g and g.gm_min_pvalue.notna().any():
            row["gm_success"] = (g.gm_min_pvalue > 0.05).mean()
        if dataset == "ACIC 2016":
            row["bal_n"], row["bal_max_smd_pooled"] = len(g), g.max_smd.median()
        rows.append(row)
    return rows


def _add_ist_balance(out):
    """Balance-check medians for the IST rows, on the datasets the check covers."""
    if not BAL_ROWS.exists():
        return out
    b = pd.read_csv(BAL_ROWS)
    recs = []
    for (t, sc, m), g in b.groupby(["target", "scenario", "method"]):
        stats = {f"bal_{c}": (g[c].mean() if c == "share_smd_gt_0_1" else g[c].median())
                 for c in BAL_COLS}
        for lab in BAL_METHOD.get(m, []):
            recs.append(dict(dataset="IST", estimand=t, scenario=sc,
                             label=lab.format(t="" if t == "ATE" else "ATT-"),
                             bal_n=g.rep.nunique(), **stats))
    bal = pd.DataFrame(recs)
    ist = out[out.dataset == "IST"].drop(columns=[c for c in out if c.startswith("bal_")])
    ist = ist.merge(bal, on=["dataset", "estimand", "scenario", "label"], how="outer")
    return pd.concat([ist, out[out.dataset != "IST"]], ignore_index=True)


def main():
    rows = []
    for name, est in IST:
        d = _load(name)
        for sc, g in d.groupby("scenario"):
            ref = "feddsm_site/FedDSM" if sc == "obs_site" else "feddsm_x/FedDSM"
            rows += _table(g, ref, "IST", est, sc, 100)
    acic = pd.concat([_load(n) for n in ACIC])
    acic = acic.drop_duplicates(["scenario", "rep", "label"])
    rows += _table(acic, "feddsm_x/FedDSM", "ACIC 2016", "ATT", "77 settings, rep 0", 1)
    out = _add_ist_balance(pd.DataFrame(rows))
    out.to_csv(RES / "checks" / "genmatch_compare.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
