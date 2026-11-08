"""LaLonde (Dehejia-Wahba): the PS baselines and DSM on GenMatch's home data.

Completes the comparison of genmatch-lalonde's D&S replication (GenMatch ATT: CPS-1
$1,698, PSID-1 $400; D&S report $1,734 / $1,045). Target: the experimental ATT of the
NSW-DW sample, $1,794 (SE $671), the usual benchmark.

Samples: the 185 NSW-DW treated + PSID-1 (2,490) or CPS-1 (15,992) controls; also the
experimental NSW-DW sample (260 randomized controls) as a sanity check. Covariates as in
genmatch-lalonde: age, educ, black, hispanic, married, nodegree, re74, re75 (earnings in
$1,000 for numerical stability; estimates are reported in $). Outcome re78. ATT.

Estimators (centralized; LaLonde has no sites):
  pooled_ps ATT-IPW-HT / ATT-IPW-Hajek_pooled / ATT-AIPW (additive logistic PS on the 8
      covariates = genmatch-lalonde's PS; linear outcome model): IF SE and bootstrap
      (B = 200; percentile CI);
  aipw_rf: normalized ATT-AIPW with a random-forest control outcome model (out-of-bag
      for the controls), bootstrap B = 200 refitting PS and forest;
  reg_ols / reg_rf: mean over the treated of Y - m̂_0(X);
  maha_xps_M{1,4}: Mahalanobis on the covariates + PS logit (genmatch-lalonde's matching
      covariates without the GenMatch weights), none / Match's linear BA (AI SE) / RF BA;
  dsm_M{1,4}: Mahalanobis on (PS logit, m̂_0 RF), none (AI SE) / RF BA.
Matching gets no bootstrap (invalid for matching; Abadie & Imbens 2008).

Usage: python experiments/checks/lalonde_ps.py DW_DIR results/checks/lalonde_ps.csv
  DW_DIR: genmatch-lalonde's data/dw (nsw_dw.rda, psid_controls.rda, cps_controls.rda)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENCHMARK, BENCHMARK_SE = 1794.343, 670.997
COVS = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
SEED = 20261108
B = 200


def _load(dw_dir):
    from src.r_env import ensure_r_home
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    out = {}
    for name in ("nsw_dw", "psid_controls", "cps_controls"):
        env = ro.Environment()
        obj = ro.r["load"](str(Path(dw_dir) / f"{name}.rda"), envir=env)[0]
        with localconverter(ro.default_converter + pandas2ri.converter):
            out[name] = ro.conversion.rpy2py(env[obj])
    return out


def _frame(d):
    X = d[COVS].astype(float).copy()
    X[["re74", "re75"]] /= 1000.0
    df = pd.DataFrame(X.to_numpy(), columns=[f"X{i}" for i in range(1, 9)])
    return df.assign(H=0, W=d["treat"].astype(int).to_numpy(), Y=d["re78"].astype(float).to_numpy())


def _rf_mu0(X, W, Y, seed):
    from sklearn.ensemble import RandomForestRegressor
    c = W == 0
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=1 / 3,
                               oob_score=True, random_state=seed, n_jobs=4).fit(X[c], Y[c])
    m = rf.predict(X)
    m[c] = rf.oob_prediction_
    return m


def _aipw_rf(df, seed):
    from src.estimators.fed_ps import _att_terms
    from src.estimators.pooled_ps import pooled_ps_logit
    X, W, Y = df.filter(like="X").to_numpy(), df.W.to_numpy(), df.Y.to_numpy()
    e = 1 / (1 + np.exp(-pooled_ps_logit(df)))
    return _att_terms(W, Y - _rf_mu0(X, W, Y, seed), e, [np.ones(len(W), bool)], True)[0]


def _run(sample, df):
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    from src.estimators import pooled_ps
    from src.estimators.genmatch import _r_matrix
    from src.estimators.pooled_ps import pooled_ps_logit

    rows = []

    def add(method, tau, se=np.nan, lo=np.nan, hi=np.nan, inference=""):
        rows.append(dict(sample=sample, method=method, tau=tau, se=se, ci_lo=lo, ci_hi=hi,
                         inference=inference, err=tau - BENCHMARK))

    X, W, Y = df.filter(like="X").to_numpy(), df.W.to_numpy(), df.Y.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pooled_ps.estimate_multi(df, ["ATT-IPW-HT", "ATT-IPW-Hajek_pooled", "ATT-AIPW"],
                                       seed=SEED, bootstrap=B,
                                       bootstrap_for=["ATT-IPW-HT", "ATT-IPW-Hajek_pooled", "ATT-AIPW"])
    for name, r in res.items():
        lo, hi = r["ci_boot_percentile"]
        add(f"pooled_{name}", r["tau"], r["se_boot"], lo, hi, f"bootstrap B={B} (IF SE {r['se']:.0f})")

    # AIPW with a random-forest outcome model, bootstrap refitting PS and forest
    tau = _aipw_rf(df, SEED)
    rng = np.random.default_rng(SEED)
    boot = []
    for b in range(B):
        idx = rng.integers(0, len(df), len(df))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot.append(_aipw_rf(df.iloc[idx].reset_index(drop=True), SEED + 1 + b))
        except Exception:
            pass
    boot = np.array(boot)
    add("aipw_rf", tau, boot.std(ddof=1), *np.percentile(boot, [2.5, 97.5]),
        f"bootstrap B={len(boot)}")

    beta = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))])[W == 0], Y[W == 0], rcond=None)[0]
    mu_ols = np.column_stack([X, np.ones(len(X))]) @ beta
    mu_rf = _rf_mu0(X, W, Y, SEED)
    add("reg_ols", float(np.mean((Y - mu_ols)[W == 1])))
    add("reg_rf", float(np.mean((Y - mu_rf)[W == 1])))

    M_ = importr("Matching")
    lp = pooled_ps_logit(df)
    sets = {"maha_xps": np.column_stack([X, lp]), "dsm": np.column_stack([lp, mu_rf])}
    for sname, V in sets.items():
        for M in (1, 4):
            kw = dict(Y=ro.FloatVector(Y), Tr=ro.FloatVector(W), X=_r_matrix(ro, V),
                      estimand="ATT", M=M, replace=True, ties=True, Weight=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = M_.Match(**kw)
                ti = np.asarray(m.rx2("index.treated"), int) - 1
                ci = np.asarray(m.rx2("index.control"), int) - 1
                pw = np.asarray(m.rx2("weights"), float)
                est, se = float(m.rx2("est")[0]), float(m.rx2("se")[0])
                add(f"{sname}_M{M}_none", est, se, est - 1.96 * se, est + 1.96 * se, "Abadie-Imbens")
                if sname == "maha_xps":
                    mb = M_.Match(**kw, BiasAdjust=True, Z=_r_matrix(ro, X))
                    est, se = float(mb.rx2("est")[0]), float(mb.rx2("se")[0])
                    add(f"{sname}_M{M}_lin", est, se, est - 1.96 * se, est + 1.96 * se,
                        "Abadie-Imbens (Match BiasAdjust, Z = covariates)")
            d = Y[ti] - Y[ci] - (mu_rf[ti] - mu_rf[ci])
            add(f"{sname}_M{M}_rf", float(np.sum(pw * d) / pw.sum()))
    return rows


def main():
    dw_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = _load(dw_dir)
    nsw = data["nsw_dw"]
    treated = nsw[nsw["treat"] == 1]
    exp_est = treated["re78"].mean() - nsw[nsw["treat"] == 0]["re78"].mean()
    print(f"experimental benchmark recomputed: {exp_est:.1f}")
    samples = {"experimental": nsw,
               "psid1": pd.concat([treated, data["psid_controls"].assign(treat=0)]),
               "cps1": pd.concat([treated, data["cps_controls"].assign(treat=0)])}
    rows = []
    for name, d in samples.items():
        t = time.time()
        rows += _run(name, _frame(d.reset_index(drop=True)))
        print(f"{name}: n = {len(d)}, {time.time() - t:.0f} s")
    res = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    res["err_in_benchmark_se"] = res.err / BENCHMARK_SE
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for s in samples:
            print(f"\n== {s}")
            print(res[res["sample"] == s][["method", "tau", "se", "ci_lo", "ci_hi", "err",
                                           "err_in_benchmark_se"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
