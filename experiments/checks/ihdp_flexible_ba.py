"""IHDP: matching vs AIPW when both get the same outcome model (linear, quadratic or
random forest).

Question: is GenMatch's gap to the PS baselines on IHDP due to matching itself or to
the linear outcome adjustment? Every estimator below uses the same outcome models
μ̂_w(x) for arm w, fitted on X1..X25 within each arm:
  "lin": OLS with intercept (in-sample);
  "rf":  RandomForestRegressor(500 trees, min_samples_leaf=5, max_features=1/3);
         units of the arm the forest was fitted on get their out-of-bag prediction
         (cross-fitting), the other arm the full forest's;
  "quad": OLS on X1..X25 plus the squares and pairwise products of X1..X6 (GenMatch's
         balance terms) and an intercept; smooth, unlike the forest. Units of the arm
         get 5-fold cross-fitted predictions (47 parameters vs 139 treated).
Outcome models are centralized here (a test of principle; a federated version would
need a federation-friendly flexible model).

Estimators, for target ATT (truth: att_true = 4) and ATE (truth: ate_true):
  match_{gm,maha}_M{1,2,4,8}_{none,lin,quad,rf}: matching on X1..X25 + pooled PS logit with
      the GenMatch weights stored in that replication's gm_xps rows (the search ran
      with M=1; reused here for M > 1) or Mahalanobis (Weight=2). ATT: treated matched
      to controls; ATC: controls matched to treated; ATE = (n1 ATT + n0 ATC) / n.
      Bias adjustment per pair: subtract μ̂_0(X_t) - μ̂_0(X_c) (ATT) or μ̂_1(X_t) -
      μ̂_1(X_c) (ATC).
  aipw_{pooled,fed_log,fed_nn}_{lin,quad,rf}: ATT: the normalized ATT-AIPW of fed_ps
      (odds weights on Y - μ̂_0); ATE: AIPW (HT form). ê from pooled logistic (X1..X25),
      federated logistic MW or federated NN MW (X1..X18, site as H; NN seeded with the
      run's config seed, as in the harness).
  reg_{lin,quad,rf}: regression only. ATT: mean over treated of Y - μ̂_0(X); ATE:
      mean of μ̂_1 - μ̂_0.
Point accuracy only (no SEs): bias, RMSE, MAE, and paired |error| differences.
Sanity: match_{gm,maha}_M1_none reproduces the stored GM / Maha estimates.

Usage: python experiments/checks/ihdp_flexible_ba.py results/checks/ihdp_flexible_ba.csv
"""
import glob
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUNS = {"ATT": ("results/ihdp_att_cc/rows/IHDP_ATT", "fed_nn"),
        "ATE": ("results/ihdp_mc/rows/IHDP", "fed_nn")}
MS = (1, 2, 4, 8)
XC = [f"X{i}" for i in range(1, 26)]


def _stored(target):
    rows = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / RUNS[target][0] / "rep*.csv"))])
    return rows


def _outcome_models(X, W, Y, seed):
    """{"lin"/"quad"/"rf": (mu0, mu1)} on all units."""
    from sklearn.ensemble import RandomForestRegressor

    from src.estimators.genmatch import _balance_matrix
    Xi = np.column_stack([X, np.ones(len(X))])
    Xq = np.column_stack([_balance_matrix(X, list(range(6))), np.ones(len(X))])
    out = {"lin": [None, None], "quad": [None, None], "rf": [None, None]}
    for w in (0, 1):
        arm = W == w
        beta = np.linalg.lstsq(Xi[arm], Y[arm], rcond=None)[0]
        out["lin"][w] = Xi @ beta
        idx = np.flatnonzero(arm)
        pred = Xq @ np.linalg.lstsq(Xq[arm], Y[arm], rcond=None)[0]
        folds = np.random.default_rng(seed + 10 + w).permutation(len(idx)) % 5
        for f in range(5):
            tr, te = idx[folds != f], idx[folds == f]
            pred[te] = Xq[te] @ np.linalg.lstsq(Xq[tr], Y[tr], rcond=None)[0]
        out["quad"][w] = pred
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=1 / 3,
                                   oob_score=True, random_state=seed + w, n_jobs=1)
        rf.fit(X[arm], Y[arm])
        pred = rf.predict(X)
        pred[arm] = rf.oob_prediction_
        out["rf"][w] = pred
    return {k: tuple(v) for k, v in out.items()}


def _match_pairs(M_, ro, Y, W, Xm, est, M, weight_matrix):
    from src.estimators.genmatch import _r_matrix
    kw = dict(Weight=2) if weight_matrix is None else dict(
        Weight=3, Weight_matrix=_r_matrix(ro, weight_matrix))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = M_.Match(Y=ro.FloatVector(Y), Tr=ro.FloatVector(W), X=Xm, estimand=est, M=M,
                     replace=True, ties=True, **kw)
    return (np.asarray(m.rx2("index.treated"), int) - 1,
            np.asarray(m.rx2("index.control"), int) - 1,
            np.asarray(m.rx2("weights"), float), float(m.rx2("est")[0]))


def _one_rep(args):
    target, rep, gm_w, cfg_seed_nn = args
    from src.r_env import ensure_r_home
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    from src.dgps import ihdp
    from src.estimators import fed_ps
    from src.estimators.fed_ps import _att_terms
    from src.estimators.genmatch import _r_matrix
    from src.estimators.pooled_ps import pooled_ps_logit

    M_ = importr("Matching")
    df = ihdp.load(rep, target=target)
    truth = df.attrs["tau_true"]
    X, W, Y = df[XC].to_numpy(float), df.W.to_numpy(int), df.Y.to_numpy(float)
    n, n1 = len(W), W.sum()
    Xm = _r_matrix(ro, np.column_stack([X, pooled_ps_logit(df)]))
    mu = _outcome_models(X, W, Y, seed=1000 + rep)
    rows = []

    def add(method, tau, **kw):
        rows.append(dict(target=target, rep=rep, method=method, tau=tau, tau_true=truth,
                         err=tau - truth, **kw))

    weights = {"gm": np.diag(gm_w), "maha": None}
    for wname, wm in weights.items():
        for M in MS:
            parts = {}
            for est in (("ATT",) if target == "ATT" else ("ATT", "ATC")):
                ti, ci, pw, est_r = _match_pairs(M_, ro, Y, W, Xm, est, M, wm)
                d = Y[ti] - Y[ci]
                parts[est] = {"none": np.sum(pw * d) / pw.sum(), "r_est": est_r}
                arm = 0 if est == "ATT" else 1
                for ba in ("lin", "quad", "rf"):
                    m = mu[ba][arm]
                    parts[est][ba] = np.sum(pw * (d - (m[ti] - m[ci]))) / pw.sum()
            for ba in ("none", "lin", "quad", "rf"):
                tau = (parts["ATT"][ba] if target == "ATT"
                       else (n1 * parts["ATT"][ba] + (n - n1) * parts["ATC"][ba]) / n)
                add(f"match_{wname}_M{M}_{ba}", tau, family="match", weights=wname, M=M, ba=ba)

    # propensity scores
    ps = {"pooled": 1 / (1 + np.exp(-pooled_ps_logit(df)))}
    d18 = ihdp.without_site_dummies(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps["fed_log"] = fed_ps.estimate_multi(d18, ["IPW-HT"], mw="logistic")["IPW-HT"]["e_hat"].to_numpy()
        ps["fed_nn"] = fed_ps.estimate_multi(d18, ["IPW-HT"], mw="nn",
                                             seed=int(cfg_seed_nn))["IPW-HT"]["e_hat"].to_numpy()
    for pname, e in ps.items():
        for ba in ("lin", "quad", "rf"):
            mu0, mu1 = mu[ba]
            if target == "ATT":
                tau = _att_terms(W, Y - mu0, e, [np.ones(n, bool)], normalize=True)[0]
            else:
                tau = float(np.mean(mu1 - mu0 + W * (Y - mu1) / e - (1 - W) * (Y - mu0) / (1 - e)))
            add(f"aipw_{pname}_{ba}", tau, family="aipw", weights=pname, ba=ba)
    for ba in ("lin", "quad", "rf"):
        mu0, mu1 = mu[ba]
        tau = float(np.mean((Y - mu0)[W == 1])) if target == "ATT" else float(np.mean(mu1 - mu0))
        add(f"reg_{ba}", tau, family="reg", ba=ba)
    return rows


def main():
    out = Path(sys.argv[1])
    tasks, stored_est = [], []
    for target in ("ATT", "ATE"):
        rows = _stored(target)
        gm_rows = rows[rows.config_id == "gm_xps"]
        for rep in range(100):
            g = gm_rows[(gm_rows.rep == rep) & (gm_rows.estimator == "GM")].iloc[0]
            gm_w = g[[f"gm_weights_{c}" for c in XC + ["ps_logit"]]].to_numpy(float)
            nn_seed = rows[(rows.rep == rep) & (rows.config_id == "fed_nn")].config_seed.iloc[0]
            tasks.append((target, rep, gm_w, nn_seed))
            for est in ("GM", "Maha"):
                stored_est.append(dict(target=target, rep=rep, est=est,
                                       tau=gm_rows[(gm_rows.rep == rep) & (gm_rows.estimator == est)].tau.iloc[0]))
    with mp.get_context("spawn").Pool(6) as pool:
        res = [r for rows in pool.imap_unordered(_one_rep, tasks) for r in rows]
    d = pd.DataFrame(res).sort_values(["target", "rep", "method"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)

    # sanity: M=1 unadjusted matching reproduces the stored estimates
    s = pd.DataFrame(stored_est)
    for est, meth in (("GM", "match_gm_M1_none"), ("Maha", "match_maha_M1_none")):
        a = s[s.est == est].set_index(["target", "rep"]).tau
        b = d[d.method == meth].set_index(["target", "rep"]).tau
        print(f"sanity {est}: max |stored - recomputed| =", float((a - b.reindex(a.index)).abs().max()))

    summ = d.groupby(["target", "method"]).err.agg(
        bias="mean", rmse=lambda e: np.sqrt(np.mean(e**2)), mae=lambda e: e.abs().mean(),
        med=lambda e: e.abs().median()).reset_index()
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for t in ("ATT", "ATE"):
            print(f"\n== {t}")
            print(summ[summ.target == t].sort_values("rmse").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
