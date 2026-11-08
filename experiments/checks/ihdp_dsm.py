"""IHDP: double-score matching (DSM) vs matching on X + PS and vs AIPW, same nuisances.

DSM (Leacy & Stuart 2014; Antonelli et al. 2018): match on low-dimensional scores
instead of the covariates. Mahalanobis matching (Match, Weight=2) on
  ATT: V = (logit ê(x), m̂_0(x))           -- treated matched to controls
  ATE: V = (logit ê(x), m̂_0(x), m̂_1(x))   -- ATT and ATC matches combined, as in
                                             ihdp_flexible_ba.py
The prognostic scores m̂_w are the cross-fitted outcome models of ihdp_flexible_ba.py
(same seeds, so identical numbers): "rf" (out-of-bag for the arm's own units) or "quad"
(5-fold cross-fitted). No unit's observed Y enters its own score. ê: pooled logistic
(X1..X25), federated logistic MW or federated NN MW (X1..X18, site as H), as there.

Match sets (each with M in {1, 4, 8} and bias adjustment none / lin / quad / rf):
  dsm_{pooled,fed_log,fed_nn}_{rf,quad}  the double score
  prog_{rf,quad}                         prognostic score(s) only (ablation)
  xps_maha, xps_gm                       X1..X25 + pooled PS logit, Mahalanobis or the
                                         stored GenMatch weights (as in ihdp_flexible_ba)
Also records each matched sample's max |SMD| over X1..X25 (src.estimators.balance.smd;
unit weights = summed pair weights, as genmatch.py). AIPW / regression comparators come
from results/checks/ihdp_flexible_ba.csv (same nuisances).

Usage: python experiments/checks/ihdp_dsm.py results/checks/ihdp_dsm.csv
"""
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.checks.ihdp_flexible_ba import XC, _match_pairs, _outcome_models, _stored  # noqa: E402

MS = (1, 4, 8)
BAS = ("none", "lin", "quad", "rf")


def _one_rep(args):
    target, rep, gm_w, nn_seed = args
    from src.r_env import ensure_r_home
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    from src.dgps import ihdp
    from src.estimators import fed_ps
    from src.estimators.balance import smd
    from src.estimators.genmatch import _r_matrix
    from src.estimators.pooled_ps import pooled_ps_logit

    M_ = importr("Matching")
    df = ihdp.load(rep, target=target)
    truth = df.attrs["tau_true"]
    X, W, Y = df[XC].to_numpy(float), df.W.to_numpy(int), df.Y.to_numpy(float)
    n, n1 = len(W), W.sum()
    mu = _outcome_models(X, W, Y, seed=1000 + rep)

    logit = {"pooled": pooled_ps_logit(df)}
    d18 = ihdp.without_site_dummies(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, mw, kw in (("fed_log", "logistic", {}), ("fed_nn", "nn", {"seed": int(nn_seed)})):
            e = fed_ps.estimate_multi(d18, ["IPW-HT"], mw=mw, **kw)["IPW-HT"]["e_hat"].to_numpy()
            logit[name] = np.log(e / (1 - e))

    def scores(mod):
        return [mu[mod][0]] if target == "ATT" else [mu[mod][0], mu[mod][1]]

    sets = {"xps_maha": (np.column_stack([X, logit["pooled"]]), None),
            "xps_gm": (np.column_stack([X, logit["pooled"]]), np.diag(gm_w))}
    for mod in ("rf", "quad"):
        sets[f"prog_{mod}"] = (np.column_stack(scores(mod)), None)
        for ps in ("pooled", "fed_log", "fed_nn"):
            sets[f"dsm_{ps}_{mod}"] = (np.column_stack([logit[ps]] + scores(mod)), None)

    rows = []
    for sname, (V, wm) in sets.items():
        Vr = _r_matrix(ro, V)
        for M in MS:
            parts, unit_w = {}, np.zeros(n)
            for est in (("ATT",) if target == "ATT" else ("ATT", "ATC")):
                ti, ci, pw, _ = _match_pairs(M_, ro, Y, W, Vr, est, M, wm)
                np.add.at(unit_w, ti, pw)
                np.add.at(unit_w, ci, pw)
                d = Y[ti] - Y[ci]
                arm = 0 if est == "ATT" else 1
                parts[est] = {"none": np.sum(pw * d) / pw.sum()}
                for ba in BAS[1:]:
                    m = mu[ba][arm]
                    parts[est][ba] = np.sum(pw * (d - (m[ti] - m[ci]))) / pw.sum()
            max_smd = float(smd(df[XC], W, unit_w).abs().max())
            for ba in BAS:
                tau = (parts["ATT"][ba] if target == "ATT"
                       else (n1 * parts["ATT"][ba] + (n - n1) * parts["ATC"][ba]) / n)
                rows.append(dict(target=target, rep=rep, method=f"{sname}_M{M}_{ba}",
                                 set=sname, M=M, ba=ba, tau=tau, tau_true=truth,
                                 err=tau - truth, max_smd=max_smd))
    return rows


def main():
    out = Path(sys.argv[1])
    tasks = []
    for target in ("ATT", "ATE"):
        rows = _stored(target)
        gm_rows = rows[(rows.config_id == "gm_xps") & (rows.estimator == "GM")]
        for rep in range(100):
            g = gm_rows[gm_rows.rep == rep].iloc[0]
            gm_w = g[[f"gm_weights_{c}" for c in XC + ["ps_logit"]]].to_numpy(float)
            nn_seed = rows[(rows.rep == rep) & (rows.config_id == "fed_nn")].config_seed.iloc[0]
            tasks.append((target, rep, gm_w, nn_seed))
    with mp.get_context("spawn").Pool(6) as pool:
        res = [r for rows in pool.imap_unordered(_one_rep, tasks) for r in rows]
    d = pd.DataFrame(res).sort_values(["target", "rep", "method"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)

    # sanity: xps sets reproduce ihdp_flexible_ba's matching estimates
    f = pd.read_csv(ROOT / "results/checks/ihdp_flexible_ba.csv")
    for a, b in (("xps_maha_M4_rf", "match_maha_M4_rf"), ("xps_gm_M1_none", "match_gm_M1_none")):
        x = d[d.method == a].set_index(["target", "rep"]).tau
        y = f[f.method == b].set_index(["target", "rep"]).tau.reindex(x.index)
        print(f"sanity {a} vs {b}: max |diff| =", float((x - y).abs().max()))

    rm = lambda e: np.sqrt(np.mean(e**2))  # noqa: E731
    summ = d.groupby(["target", "method", "set", "M", "ba"]).agg(
        bias=("err", "mean"), rmse=("err", rm), mae=("err", lambda e: e.abs().mean()),
        max_smd=("max_smd", "mean")).reset_index()
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        for t in ("ATT", "ATE"):
            s = summ[summ.target == t]
            best = s.loc[s.groupby("set").rmse.idxmin()].sort_values("rmse")
            print(f"\n== {t}: best (M, BA) per match set")
            print(best[["set", "M", "ba", "bias", "rmse", "mae", "max_smd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
