"""Fed-GenMatch normaliser ablation (docs/fedgm_ablation.md): accuracy per arm, paired by
rep against FedDSM and (single_arm reps 0-29) the centralized R GenMatch from Narval.

    python experiments/checks/fedgm_ablation_summary.py
        -> results/fedgm_ablation/paired.csv

Paired rule: an arm differs from the reference if the mean paired difference in squared
error is > 2 SE (z = mean(d) / (sd(d) / sqrt(n)), d = e_arm^2 - e_ref^2). Accuracy only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / "results" / "fedgm_ablation"
GM = ROOT / "results" / "ist_single_arm_cc"
REF = "feddsm_x/FedDSM"


def _rows(out):
    d = pd.concat(pd.read_csv(f) for f in sorted((out / "rows").glob("*/*.csv")))
    assert d["error"].isna().all() if "error" in d else True
    d["label"] = d["config_id"] + "/" + d["estimator"]
    d["err"] = d["tau"] - d["tau_true"]
    return d


def _paired(e, ref):
    d = e ** 2 - ref ** 2
    rmse, rmse_ref = np.sqrt((e ** 2).mean()), np.sqrt((ref ** 2).mean())
    return rmse - rmse_ref, d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def summarize(d, refs):
    out = []
    for (sc, lab), g in d.groupby(["scenario", "label"]):
        g = g.set_index("rep").sort_index()
        row = {"scenario": sc, "label": lab, "reps": f"{g.index.min()}-{g.index.max()}",
               "n": len(g), "bias_pp": 100 * g.err.mean(),
               "rmse_pp": 100 * np.sqrt((g.err ** 2).mean()),
               "mc_se_bias_pp": 100 * g.err.std(ddof=1) / np.sqrt(len(g)),
               "median_runtime_s": g.runtime.median()}
        for c in ("max_smd_bret", "generations", "n_evals", "sigma_rank", "shrinkage"):
            if c in g and g[c].notna().any():
                row[f"median_{c}"] = g[c].median()
        for name, r in refs.items():
            r = r[r.scenario == sc].set_index("rep").err.reindex(g.index)
            if r.notna().all() and lab != name:
                diff, z = _paired(g.err, r)
                row[f"rmse_diff_vs_{name}_pp"], row[f"z_vs_{name}"] = 100 * diff, z
        out.append(row)
    return pd.DataFrame(out)


def main():
    d = _rows(ABL)
    gm = _rows(GM)
    gm = gm[gm.config_id == "gm_xps"]
    gm["label"] = "R_" + gm["estimator"]
    sub = d[(d.scenario == "single_arm") & d.rep.isin(gm.rep.unique())]
    assert np.allclose(sub.groupby("rep").tau_true.first().sort_index(),
                       gm.groupby("rep").tau_true.first().sort_index())
    ref = d[d.label == REF]
    full = summarize(d, {"FedDSM": ref})
    first30 = summarize(pd.concat([sub, gm]), {"FedDSM": ref, "R_GM": gm[gm.label == "R_GM"]})
    first30["scenario"] = "single_arm_reps0-29"
    res = pd.concat([full, first30], ignore_index=True)
    res.to_csv(ABL / "paired.csv", index=False)
    cols = ["scenario", "label", "n", "bias_pp", "rmse_pp", "rmse_diff_vs_FedDSM_pp",
            "z_vs_FedDSM", "rmse_diff_vs_R_GM_pp", "z_vs_R_GM", "median_max_smd_bret",
            "median_sigma_rank", "median_runtime_s"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(res[[c for c in cols if c in res]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
