"""Centralized R GenMatch (Narval) vs the other methods, on the datasets where GenMatch ran.

IST (results/ist_cc ATE, ist_att_cc ATT, ist_single_arm_cc ATT; 30 reps per scenario,
every method from the same Narval run) and ACIC 2016 ATT (results/gm_acic_att: rep 0 of
each of the 77 settings, paired with results/feddsm_acic_att and feddsm_acic_flex_att).

    python experiments/checks/genmatch_compare.py   # -> results/checks/genmatch_compare.csv

Paired rule as in N14: z = mean(e^2 - e_FedDSM^2) / SE over the shared datasets
(> 2: FedDSM better). IST in percentage points, ACIC in outcome units. FedDSM is the
scenario's primary PS (site-aware in obs_site).
Accuracy only. GM success = share of datasets with GenMatch's min balance p-value > 0.05.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
IST = [("ist_cc", "ATE"), ("ist_att_cc", "ATT"), ("ist_single_arm_cc", "ATT")]
ACIC = ["gm_acic_att", "feddsm_acic_att", "feddsm_acic_flex_att", "feddsm_acic_dsmrf_att"]


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
        rows.append(row)
    return rows


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
    out = pd.DataFrame(rows)
    out.to_csv(RES / "checks" / "genmatch_compare.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
