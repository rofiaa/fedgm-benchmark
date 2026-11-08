"""One comparison table: every method x every dataset x the standard metrics.

Sources (all committed results; nothing is rerun):
  Authors' DGPs (6 scenarios), ATE: results/ps_mc_v2, results/feddsm_authors,
      results/gm_mc (GenMatch pilot: 2 reps per scenario, flagged)
  Authors' DGPs, ATT: results/feddsm_authors_att
  IHDP (100 replications), ATE: results/ihdp_mc, results/feddsm_ihdp
  IHDP, ATT: results/ihdp_att_cc, results/feddsm_ihdp_att
  LaLonde (one sample each; error vs the experimental benchmark): results/checks/
      lalonde_ps.csv, lalonde_feddsm.csv, and genmatch-lalonde's GenMatch estimates.
Metrics per (dataset, estimand, scenario, method): experiments/summarize.score (bias,
MC SE, RMSE, empirical SD, IF-SE ratio and coverage, CI width, bootstrap coverage,
max |SMD|, runtime) plus MAE and median |error|.

    python experiments/compile_comparison.py   # -> results/comparison/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.summarize import load_rows, score  # noqa: E402

OUT = ROOT / "results" / "comparison"
SOURCES = [  # (results dir, dataset, estimand, configs to drop, note)
    ("ps_mc_v2", "Authors' DGPs", "ATE", [], ""),
    ("feddsm_authors", "Authors' DGPs", "ATE", ["fed_fedavg"], ""),
    ("gm_mc", "Authors' DGPs", "ATE", [], "pilot: 2 reps per scenario"),
    ("feddsm_authors_att", "Authors' DGPs", "ATT", [], ""),
    ("ihdp_mc", "IHDP", "ATE", [], ""),
    ("feddsm_ihdp", "IHDP", "ATE", [], ""),
    ("ihdp_att_cc", "IHDP", "ATT", [], ""),
    ("feddsm_ihdp_att", "IHDP", "ATT", [], ""),
]
# primary FedDSM PS per dataset (from the dataset's confounding assumption)
PRIMARY_PS = {"Authors' DGPs": "x", "IHDP": "site", "LaLonde": "x"}
LALONDE_GM = "/Users/chensirong/Documents/casual_analysis+federated_learning/genmatch-lalonde"

PS_NAME = {"x": "PS on covariates", "site": "site-aware PS", "khellaf": "paper's federated PS"}
EST = {"IPW-HT": "IPW-HT", "IPW-Hajek_pooled": "IPW-Hajek", "IPW-Hajek": "IPW-Hajek",
       "AIPW": "AIPW"}


def label(config, estimator):
    """(label, family, federated, headline key) for a (config_id, estimator) pair."""
    est = EST.get(estimator.removeprefix("ATT-"), estimator)
    if config.startswith("feddsm_"):
        ps = config.removeprefix("feddsm_")
        name = {"FedDSM": "FedDSM", "FedDSM-noBA": "FedDSM (no bias adj.)",
                "FedAIPW-quad": "Fed-AIPW-quad (FedDSM's models)"}[estimator]
        key = {"FedDSM": "feddsm", "FedDSM-noBA": "feddsm_noba",
               "FedAIPW-quad": "fedaipw_quad"}[estimator]
        return f"{name} · {PS_NAME[ps]}", "FedDSM (ours)", "yes", f"{key}:{ps}"
    table = {
        "fed_fedavg": (f"Fed-{est} · FedAvg logistic MW (paper)", "Federated PS (paper)",
                       "yes", f"fed_{est}:fedavg"),
        "fed_logistic": (f"Fed-{est} · logistic MW, pooled fit", "Federated PS (paper)",
                         "partly", f"fed_{est}:logistic"),
        "fed_nn": (f"Fed-{est} · NN MW†", "Federated PS (paper)", "partly",
                   f"fed_{est}:nn"),
        "fed_dw": (f"Fed-{est} · density-ratio MW", "Federated PS (paper)", "yes",
                   f"fed_{est}:dw"),
        "fed_logistic_rfps": (f"Fed-{est} · logistic MW + RF local PS",
                              "Federated PS (paper)", "partly", f"fed_{est}:rfps"),
        "meta_logistic": (f"Meta-SW {est}", "Meta-analysis", "yes", f"meta_{est}"),
        "meta_rfps": (f"Meta-SW {est} · RF PS", "Meta-analysis", "yes", f"meta_{est}:rf"),
        "pooled_logistic": (f"Pooled PS {est}", "Centralized PS", "no", f"pooled_{est}"),
        "oracle": (f"Oracle {est} (true nuisances)", "Oracle", "—", f"oracle_{est}"),
    }
    if config in table:
        return table[config]
    if config == "gm_xps":
        name = {"GM": "GenMatch", "GM-BA": "GenMatch + BA", "Maha": "Mahalanobis",
                "Maha-BA": "Mahalanobis + BA"}[estimator]
        return name, "Centralized matching", "no", estimator.lower()
    raise KeyError((config, estimator))


def simulated():
    out = []
    for d, dataset, estimand, drop, note in SOURCES:
        rows = load_rows(ROOT / "results" / d)
        rows = rows[~rows.config_id.isin(drop)]
        for (scen, cfg, est), g in rows.groupby(["scenario", "config_id", "estimator"]):
            s = score(g)
            err = (g["tau"] - g["tau_true"])[np.isfinite(g["tau"])]
            lab, fam, fed, key = label(cfg, est)
            out.append(dict(
                dataset=dataset, estimand=estimand,
                scenario="IHDP" if scen.startswith("IHDP") else scen,
                config_id=cfg, estimator=est, label=lab, family=fam, federated=fed,
                key=key, note=note, n_reps=int(s.n_reps), n_undefined=int(s.n_undefined),
                tau_true=s.tau_true, bias=s.bias, bias_mcse=s.bias_mcse, rmse=s.rmse,
                mae=float(err.abs().mean()) if len(err) else np.nan,
                median_abs_err=float(err.abs().median()) if len(err) else np.nan,
                emp_sd=s.emp_sd, se_ratio=s.se_if_ratio, coverage=s.cov_if,
                ci_width=s.width_if, coverage_boot=s.cov_boot_pct,
                n_boot_reps=s.n_boot_reps, max_smd=s.mean_max_smd,
                runtime_s=s.mean_config_runtime_s, source=f"results/{d}"))
    return pd.DataFrame(out)


def lalonde():
    bench = 1794.343
    ps = pd.read_csv(ROOT / "results/checks/lalonde_ps.csv")
    fd = pd.read_csv(ROOT / "results/checks/lalonde_feddsm.csv")
    names = {"pooled_ATT-IPW-HT": ("Pooled PS IPW-HT", "Centralized PS", "pooled_IPW-HT"),
             "pooled_ATT-IPW-Hajek_pooled": ("Pooled PS IPW-Hajek", "Centralized PS",
                                             "pooled_IPW-Hajek"),
             "pooled_ATT-AIPW": ("Pooled PS AIPW", "Centralized PS", "pooled_AIPW"),
             "aipw_rf": ("Pooled PS AIPW · RF outcome", "Centralized PS", "pooled_AIPW_rf"),
             "maha_xps_M4_lin": ("Mahalanobis + BA (M = 4)", "Centralized matching", "maha-ba4"),
             "maha_xps_M1_lin": ("Mahalanobis + BA", "Centralized matching", "maha-ba"),
             "maha_xps_M1_none": ("Mahalanobis", "Centralized matching", "maha")}
    out = []
    for _, r in ps[ps.method.isin(names) & (ps["sample"] != "experimental")].iterrows():
        lab, fam, key = names[r.method]
        cov = float(r.ci_lo <= bench <= r.ci_hi) if np.isfinite(r.ci_lo) else np.nan
        out.append(dict(dataset="LaLonde", estimand="ATT", scenario=r["sample"].upper(),
                        label=lab, family=fam, federated="no", key=key, tau=r.tau,
                        err=r.tau - bench, se=r.se, covers=cov))
    for _, r in fd[fd["sample"] != "experimental"].iterrows():
        lab, fam, fed, key = label("feddsm_x", r.method)
        out.append(dict(dataset="LaLonde", estimand="ATT", scenario=r["sample"].upper(),
                        label=lab, family=fam, federated=fed, key=key, tau=r.tau,
                        err=r.tau - bench, max_smd=r.max_smd))
    for s in ("psid1", "cps1"):
        f = Path(LALONDE_GM) / f"outputs_paper_{s}" / "tables" / "result_vs_paper.csv"
        if f.exists():
            g = pd.read_csv(f).iloc[0]
            lo, hi = g.our_att - 1.96 * g.our_se, g.our_att + 1.96 * g.our_se
            out.append(dict(dataset="LaLonde", estimand="ATT", scenario=s.upper(),
                            label="GenMatch (genmatch-lalonde)", family="Centralized matching",
                            federated="no", key="gm", tau=g.our_att, err=g.our_att - bench,
                            se=g.our_se, covers=float(lo <= bench <= hi),
                            max_smd=g.our_max_abs_smd))
    d = pd.DataFrame(out)
    d["n_reps"], d["bias"], d["rmse"], d["mae"] = 1, d.err, d.err.abs(), d.err.abs()
    d["coverage"] = d.pop("covers") if "covers" in d else np.nan
    d["tau_true"], d["source"] = bench, "results/checks/lalonde_*.csv"
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([simulated(), lalonde()], ignore_index=True)
    df["primary"] = df.apply(
        lambda r: (not str(r.key).startswith(("feddsm", "fedaipw_quad")))
        or str(r.key).endswith(":" + PRIMARY_PS[r.dataset]), axis=1)
    df.to_csv(OUT / "comparison_long.csv", index=False)
    cols = ["dataset", "estimand", "scenario", "label", "family", "federated", "key",
            "primary", "note", "n_reps", "n_undefined", "tau_true", "bias", "bias_mcse",
            "rmse", "mae", "median_abs_err", "emp_sd", "se_ratio", "coverage", "ci_width",
            "coverage_boot", "n_boot_reps", "max_smd", "runtime_s"]
    recs = df[[c for c in cols if c in df]].replace({np.nan: None}).to_dict("records")
    (OUT / "comparison.json").write_text(json.dumps(recs, default=float))
    print(f"{len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
