"""Long-tailed covariates (docs/longtail_design.md): accuracy and balance per method and
scenario, paired by dataset against Fed-GenMatch.

    python experiments/checks/longtail_summary.py   # -> results/longtail/paired.csv

z = mean(e_ref^2 - e_FedGM^2) / SE over datasets (> 2: Fed-GenMatch more accurate).
Bret's claim (FedGM beats PS weighting on long tails) is supported in a scenario if the
z against pooled IPW is > 2. Accuracy only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "longtail"
GM = "fedgm_raw/FedGM"


def main():
    d = pd.concat(pd.read_csv(f) for f in sorted((RES / "rows").glob("*/*.csv")))
    assert d["error"].isna().all() if "error" in d else True
    d = d.copy()
    d["label"] = d.config_id + "/" + d.estimator
    d["err"] = d.tau - d.tau_true
    rows = []
    for sc, g in d.groupby("scenario"):
        ref = g[g.label == GM].set_index("rep").err
        for lab, h in g.groupby("label"):
            e = h.set_index("rep").err.reindex(ref.index)
            dd = e ** 2 - ref ** 2
            z = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if lab != GM else np.nan
            bret = (float(np.median(h.max_smd_bret))
                    if lab.startswith("fedgm") and "max_smd_bret" in h else np.nan)
            rows.append(dict(scenario=sc, label=lab, n=int(e.notna().sum()),
                             bias=e.mean(), rmse=np.sqrt((e ** 2).mean()),
                             z_vs_fedgm=z, median_max_smd=float(np.median(h.max_smd)),
                             median_max_smd_bret=bret,
                             median_runtime_s=h.runtime.median()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "paired.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
