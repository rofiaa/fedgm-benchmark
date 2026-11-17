"""E2b: what FedDSM's messages reveal, IST single-arm (docs/privacy_e2b.md).

    python experiments/privacy/e2b_feddsm.py --jobs 5
        -> results/privacy/e2b_feddsm/rows.csv (releases 1-2), scores.csv (release 3)
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.configs.ist import BASE_SEED as IST_BASE  # noqa: E402
from experiments.harness import config_seed  # noqa: E402
from src.dgps import ist  # noqa: E402
from src.privacy import feddsm_attacks as fa, mia  # noqa: E402

OUT = ROOT / "results" / "privacy" / "e2b_feddsm"
REPS = range(10)
K_NN = 25


def _outsiders(rep, df):
    """IST patients outside the dataset (the loader's draw, reproduced and checked), as
    (x, y) rows with y = real 6-month death; split into reference and non-member pool."""
    pop = ist.population()
    rng = np.random.default_rng(np.random.SeedSequence([rep, ist._SCEN_ID["single_arm"], 1995]))
    idx = np.sort(rng.choice(len(pop), size=len(df), replace=False))
    X_ds = pop.iloc[idx][ist.COVARIATES].to_numpy(float)
    assert np.array_equal(X_ds, df[[f"X{i}" for i in range(1, 32)]].to_numpy(float))
    rest = np.setdiff1d(np.arange(len(pop)), idx)
    XY = np.column_stack([pop.iloc[rest][ist.COVARIATES].to_numpy(float),
                          pop.iloc[rest]["Y0"].to_numpy(float)])
    perm = np.random.default_rng(20261116 + rep).permutation(len(XY))
    half = len(XY) // 2
    return XY[perm[:half]], XY[perm[half:]]


def _mia_row(att, members, pool, **info):
    m_hat = mia.feature_mean(members, att.release)
    s_in, s_out = att.scores(members, m_hat), att.scores(pool, m_hat)
    return dict(info, n=len(members), D_eff=att.d_eff, **mia.roc_metrics(s_in, s_out),
                **mia.predicted(att.d_eff, len(members)))


def run_rep(rep):
    df = ist.load(rep, "single_arm", target="ATT")
    seed = config_seed(IST_BASE, "single_arm", rep, "feddsm_x")
    R = fa.releases(df, seed)
    X = df[[f"X{i}" for i in range(1, 32)]].to_numpy(float)
    W, H, Y = df.W.to_numpy(), df.H.to_numpy(), df.Y.to_numpy(float)
    XY = np.column_stack([X, Y])
    ref, pool = _outsiders(rep, df)
    sites = sorted(set(H[W == 0]))
    rows = []
    att_ps = mia.TracingAttack(ref, fa.ps_features(R["beta"]))
    att_pr = mia.TracingAttack(ref, fa.prognostic_features(R["quad"], R["interactions"]))
    for k in sites:
        ctrl = (W == 0) & (H == k)
        rows.append(_mia_row(att_ps, XY[ctrl], pool, rep=rep, release="ps_newton_final",
                             site=k, fold=-1))
        for f in range(5):
            cell = ctrl & (R["fold"] == f)
            rows.append(_mia_row(att_pr, XY[cell], pool, rep=rep, release="prognostic_fold",
                                 site=k, fold=f))
    # release 3: the scores sent to the matcher
    fits = fa.fold_fits(R["B"], Y, W, R["fold"])
    c = W == 0
    G, g = R["B"][c].T @ R["B"][c], R["B"][c].T @ Y[c]
    full = np.linalg.lstsq(G, g, rcond=None)[0]

    def lp_of(Xm):
        Xi = np.column_stack([Xm, np.ones(len(Xm))])
        e = np.clip(1 / (1 + np.exp(-Xi @ R["beta"])), 1e-6, 1 - 1e-6)
        return np.log(e / (1 - e))

    rel = np.column_stack([R["lp"], R["mu0"]])
    linked = total = 0
    for k in sites:
        ctrl = np.flatnonzero(c & (H == k))
        sd = rel[ctrl].std(0)
        cand_lp = lp_of(X[ctrl])
        cand_m = R["B"][ctrl] @ fits.T                      # (n_k, 5): every fold's fit
        d = np.full((len(ctrl), len(ctrl)), np.inf)
        for f in range(5):
            dd = ((rel[ctrl, None, 0] - cand_lp[None, :]) / sd[0]) ** 2 + \
                 ((rel[ctrl, None, 1] - cand_m[None, :, f]) / sd[1]) ** 2
            d = np.minimum(d, dd)
        linked += int((np.argmin(d, axis=1) == np.arange(len(ctrl))).sum())
        total += len(ctrl)
    from src.estimators import feddsm
    B_ref = feddsm._basis(ref[:, :-1], R["quad"], None, interactions=R["interactions"])
    ref_s = np.column_stack([lp_of(ref[:, :-1]), B_ref @ full])
    sd = ref_s.std(0)
    tgt = rel[c] / sd
    A = ref_s / sd
    nn = np.argsort(((tgt ** 2).sum(1)[:, None] - 2 * tgt @ A.T + (A ** 2).sum(1)[None]),
                    axis=1)[:, :K_NN]
    Xc, Xr = X[c], ref[:, :-1]
    binary = np.all((Xr == 0) | (Xr == 1), axis=0)
    pred = Xr[nn].mean(1)
    pop_mu, pop_sd = Xr.mean(0), Xr.std(0)
    acc = float(np.mean((pred[:, binary] > 0.5) == Xc[:, binary]))
    acc0 = float(np.mean((pop_mu[binary] > 0.5) == Xc[:, binary]))
    err = np.median(np.sqrt((((pred - Xc) / pop_sd) ** 2).mean(1)))
    err0 = np.median(np.sqrt((((pop_mu - Xc) / pop_sd) ** 2).mean(1)))
    scores = dict(rep=rep, linkage_share=linked / total, linkage_baseline=len(sites) / total,
                  binary_acc=acc, binary_acc_baseline=acc0, std_err_median=float(err),
                  std_err_baseline=float(err0), n_controls=total, n_basis=R["B"].shape[1],
                  interactions=R["interactions"])
    print(f"rep {rep}: linkage {linked / total:.3f}", flush=True)
    return pd.DataFrame(rows), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.jobs) as p:
        res = p.map(run_rep, list(REPS), chunksize=1)
    pd.concat([r[0] for r in res]).to_csv(OUT / "rows.csv", index=False)
    pd.DataFrame([r[1] for r in res]).to_csv(OUT / "scores.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
