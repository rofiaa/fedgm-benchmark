"""IHDP: federated double-score matching (FedDSM), simulated centrally.

Scores, each computed only from what a federated protocol would share:
  ê(x):    federated PS of fed_ps (local logistic ê_k, membership by FedAvg logistic
           regression: mw="logistic_fedavg", genuinely federated; secondary mw="nn",
           the authors' NN, which they train on pooled data). Uses X1..X18 (site = H).
           "sitelogit": a site-aware PS, logistic regression of W on X1..X25 (site
           dummies included) + intercept, fitted by distributed Newton-Raphson (sites
           send their gradient and Hessian sums each iteration; fed_logistic_newton):
           exactly the pooled logistic fit of pooled_ps (checked).
  m̂_w(x):  quadratic outcome model for arm w (X1..X25, squares and products of X1..X6,
           intercept; the "quad" model of ihdp_flexible_ba.py), fitted by least squares
           from summed per-site, per-fold sufficient statistics (B'B, B'Y): exactly
           the centralized fit. Cross-fitting: each site splits its arm-w units into 5
           folds; a unit's prediction comes from the model without its fold. Units of
           the other arm get the full fit. Sanity check against a centralized lstsq with
           the same folds.
Matching (Match, Mahalanobis, with replacement) on V = (logit ê, m̂_0) for the ATT and
(logit ê, m̂_0, m̂_1) for the ATE (ATT and ATC matches combined), M in {1, 4, 8}:
  within: exact on site (the 7 site dummies) -- each site matches its own patients,
          with the globally shared scaling. ATT: no patient-level data leave a site.
  cross:  across sites, scores only (N5's version);
  soft:   across sites with the site dummies added to the Mahalanobis distance.
Bias adjustment: none or the same federated quadratic model ("quad").
Comparators (same 100 replications, paired):
  fedaipw_quad (computed here): Fed-AIPW with the federated ê and the same federated
      quadratic outcome model -- fully federated, same nuisances as FedDSM;
  from earlier runs: Fed-AIPW as in the paper (fed_fedavg, linear outcome), Meta-SW
      AIPW (ATT), Fed-AIPW with a centralized RF outcome model, centralized DSM (N5).
Balance: max |SMD| over X1..X25, over X1..X18 and over the site dummies X19..X25.

Usage: python experiments/checks/ihdp_feddsm.py results/checks/ihdp_feddsm.csv
"""
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.checks.ihdp_flexible_ba import XC, _stored  # noqa: E402

MS = (1, 4, 8)
SITE = [f"X{i}" for i in range(19, 26)]
X18 = [f"X{i}" for i in range(1, 19)]


def fed_quad(Bm, Y, W, H, arm, seed, folds=5):
    """Federated least squares of Y on the basis Bm among arm-`arm` units from per-site,
    per-fold sums. Returns (predictions for all units, fold labels)."""
    n, p = Bm.shape
    rng = np.random.default_rng(seed)
    fold = np.full(n, -1)
    for k in np.unique(H):
        rows = np.flatnonzero((W == arm) & (H == k))
        fold[rows] = rng.permutation(len(rows)) % folds
    G, g = np.zeros((folds, p, p)), np.zeros((folds, p))
    for k in np.unique(H):  # what site k sends: its per-fold B'B and B'Y
        for f in range(folds):
            r = (H == k) & (fold == f)
            G[f] += Bm[r].T @ Bm[r]
            g[f] += Bm[r].T @ Y[r]
    pred = Bm @ np.linalg.lstsq(G.sum(0), g.sum(0), rcond=None)[0]
    for f in range(folds):
        beta = np.linalg.lstsq(G.sum(0) - G[f], g.sum(0) - g[f], rcond=None)[0]
        pred[fold == f] = Bm[fold == f] @ beta
    return pred, fold


def fed_logistic_newton(Xd, W, H, tol=1e-10, max_iter=100):
    """Logistic MLE from per-site gradient / Hessian sums (distributed Newton-Raphson).
    Returns the linear predictor for every unit."""
    Xi = np.column_stack([Xd, np.ones(len(Xd))])
    beta = np.zeros(Xi.shape[1])
    for _ in range(max_iter):
        grad, hess = np.zeros_like(beta), np.zeros((len(beta), len(beta)))
        for k in np.unique(H):  # what site k sends
            r = H == k
            p = 1 / (1 + np.exp(-Xi[r] @ beta))
            grad += Xi[r].T @ (W[r] - p)
            hess += (Xi[r] * (p * (1 - p))[:, None]).T @ Xi[r]
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return Xi @ beta


def _central_quad(Bm, Y, W, arm, fold):
    pred = Bm @ np.linalg.lstsq(Bm[W == arm], Y[W == arm], rcond=None)[0]
    for f in range(fold.max() + 1):
        tr = (W == arm) & (fold != f) & (fold >= 0)
        pred[fold == f] = Bm[fold == f] @ np.linalg.lstsq(Bm[tr], Y[tr], rcond=None)[0]
    return pred


def _one_rep(args):
    target, rep, nn_seed = args
    from src.r_env import ensure_r_home
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    from src.dgps import ihdp
    from src.estimators import fed_ps
    from src.estimators.balance import smd
    from src.estimators.fed_ps import _att_terms
    from src.estimators.genmatch import _balance_matrix, _r_matrix

    M_ = importr("Matching")
    df = ihdp.load(rep, target=target)
    truth = df.attrs["tau_true"]
    X, W, Y, H = (df[XC].to_numpy(float), df.W.to_numpy(int), df.Y.to_numpy(float),
                  df.H.to_numpy(int))
    n, n1 = len(W), W.sum()
    Bm = np.column_stack([_balance_matrix(X, list(range(6))), np.ones(n)])
    mu, check = {}, 0.0
    for arm in (0, 1):
        mu[arm], fold = fed_quad(Bm, Y, W, H, arm, seed=2000 + 10 * rep + arm)
        check = max(check, float(np.max(np.abs(mu[arm] - _central_quad(Bm, Y, W, arm, fold)))
                                 / np.std(Y)))

    d18 = ihdp.without_site_dummies(df)
    from src.estimators.pooled_ps import pooled_ps_logit
    logit = {"sitelogit": fed_logistic_newton(X, W, H)}
    ps_check = float(np.max(np.abs(logit["sitelogit"] - pooled_ps_logit(df))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, mw, kw in (("fedavg", "logistic_fedavg", {}), ("nn", "nn", {"seed": int(nn_seed)})):
            e = fed_ps.estimate_multi(d18, ["IPW-HT"], mw=mw, **kw)["IPW-HT"]["e_hat"].to_numpy()
            logit[name] = np.log(e / (1 - e))

    rows = []

    def add(method, tau, **kw):
        rows.append(dict(target=target, rep=rep, method=method, tau=tau, tau_true=truth,
                         err=tau - truth, fedquad_vs_central=check,
                         sitelogit_vs_pooled=ps_check, **kw))

    # fully federated AIPW with the same nuisances
    for ps in ("fedavg", "nn", "sitelogit"):
        e = 1 / (1 + np.exp(-logit[ps]))
        if target == "ATT":
            tau = _att_terms(W, Y - mu[0], e, [np.ones(n, bool)], normalize=True)[0]
        else:
            tau = float(np.mean(mu[1] - mu[0] + W * (Y - mu[1]) / e - (1 - W) * (Y - mu[0]) / (1 - e)))
        add(f"fedaipw_quad_{ps}", tau, family="aipw", ps=ps)

    dummies = df[SITE].to_numpy(float)
    for ps in ("fedavg", "nn", "sitelogit"):
        V = np.column_stack([logit[ps], mu[0]] + ([] if target == "ATT" else [mu[1]]))
        for site in ("within", "cross", "soft"):
            Xm = V if site == "cross" else np.column_stack([V, dummies])
            exact = ([False] * V.shape[1] + [True] * 7) if site == "within" else None
            for M in MS:
                parts, unit_w, failed = {}, np.zeros(n), None
                for est in (("ATT",) if target == "ATT" else ("ATT", "ATC")):
                    kw = dict(Y=ro.FloatVector(Y), Tr=ro.FloatVector(W), X=_r_matrix(ro, Xm),
                              estimand=est, M=M, replace=True, ties=True, Weight=2)
                    if exact is not None:
                        kw["exact"] = ro.BoolVector(exact)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            m = M_.Match(**kw)
                        ti = np.asarray(m.rx2("index.treated"), int) - 1
                        ci = np.asarray(m.rx2("index.control"), int) - 1
                        pw = np.asarray(m.rx2("weights"), float)
                    except Exception as exc:  # e.g. no admissible match within a site
                        failed = str(exc).strip().splitlines()[0][:120]
                        break
                    # matched-away units: Match drops units without an exact match
                    np.add.at(unit_w, ti, pw)
                    np.add.at(unit_w, ci, pw)
                    dY = Y[ti] - Y[ci]
                    arm = 0 if est == "ATT" else 1
                    m_ = mu[arm]
                    parts[est] = {"none": np.sum(pw * dY) / pw.sum(),
                                  "quad": np.sum(pw * (dY - (m_[ti] - m_[ci]))) / pw.sum(),
                                  "n_matched": len(np.unique(ti if est == "ATT" else ci))}
                for ba in ("none", "quad"):
                    if failed:
                        add(f"feddsm_{ps}_{site}_M{M}_{ba}", np.nan, family="dsm", ps=ps,
                            site=site, M=M, ba=ba, error=failed)
                        continue
                    tau = (parts["ATT"][ba] if target == "ATT"
                           else (n1 * parts["ATT"][ba] + (n - n1) * parts["ATC"][ba]) / n)
                    s = smd(df[XC], W, unit_w).abs()
                    add(f"feddsm_{ps}_{site}_M{M}_{ba}", tau, family="dsm", ps=ps, site=site,
                        M=M, ba=ba, max_smd=float(s.max()), max_smd_x18=float(s[X18].max()),
                        max_smd_site=float(s[SITE].max()),
                        n_target_matched=parts["ATT"]["n_matched"])
    return rows


def _comparators(target):
    """{method: Series(err by rep)} from earlier runs."""
    out = {}
    rows = _stored(target)
    rows = rows.assign(err=rows.tau - rows.tau_true)
    aipw = "ATT-AIPW" if target == "ATT" else "AIPW"
    for cfg, label in (("fed_fedavg", "fedaipw_linear_paper"), ("fed_nn", "fedaipw_linear_nn"),
                       ("meta_logistic", "meta_aipw")):
        r = rows[(rows.config_id == cfg) & (rows.estimator == aipw)]
        out[label] = r.set_index("rep").err
    f = pd.read_csv(ROOT / "results/checks/ihdp_flexible_ba.csv")
    f = f[f.target == target]
    for m in ("aipw_fed_log_rf", "aipw_fed_nn_rf"):
        out[m + "_central_outcome"] = f[f.method == m].set_index("rep").err
    d = pd.read_csv(ROOT / "results/checks/ihdp_dsm.csv")
    d = d[d.target == target]
    for m in ("dsm_pooled_quad_M4_quad", "dsm_fed_log_quad_M4_quad"):
        out["central_" + m] = d[d.method == m].set_index("rep").err
    return out


def main():
    out = Path(sys.argv[1])
    tasks = []
    for target in ("ATT", "ATE"):
        rows = _stored(target)
        for rep in range(100):
            nn_seed = rows[(rows.rep == rep) & (rows.config_id == "fed_nn")].config_seed.iloc[0]
            tasks.append((target, rep, nn_seed))
    with mp.get_context("spawn").Pool(6) as pool:
        res = [r for rows in pool.imap_unordered(_one_rep, tasks) for r in rows]
    d = pd.DataFrame(res).sort_values(["target", "rep", "method"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print("federated vs centralized quadratic fit, max |diff| / sd(Y):",
          float(d.fedquad_vs_central.max()))
    print("distributed Newton vs pooled logistic PS, max |logit diff|:",
          float(d.sitelogit_vs_pooled.max()))
    if "error" in d:
        print("failed cells:", d.error.notna().sum(), d.error.dropna().unique()[:3])

    rm = lambda e: np.sqrt(np.mean(e**2))  # noqa: E731
    for t in ("ATT", "ATE"):
        x = d[d.target == t]
        print(f"\n===== {t}")
        cols = {c: "mean" for c in ("max_smd", "max_smd_x18", "max_smd_site", "n_target_matched")
                if c in x}
        s = x.groupby("method").agg(rmse=("err", rm), mae=("err", lambda e: e.abs().mean()),
                                    bias=("err", "mean"), n_nan=("err", lambda e: e.isna().sum()),
                                    **{k: (k, v) for k, v in cols.items()})
        comp = _comparators(t)
        for k, v in comp.items():
            s.loc[k, ["rmse", "mae", "bias"]] = [rm(v), v.abs().mean(), v.mean()]
        with pd.option_context("display.width", 220, "display.max_rows", 200):
            print(s.sort_values("rmse").round(3).to_string())


if __name__ == "__main__":
    main()
