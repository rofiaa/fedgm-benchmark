"""E2: reconstruction attacks on Fed-GenMatch's messages, IST single-arm (docs/privacy_e2.md).

    python experiments/privacy/e2_fedgm.py --jobs 6
        -> results/privacy/e2_fedgm/rows/*.csv (one per task, resumable), cells.csv
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.dgps import ist  # noqa: E402
from src.privacy import fedgm_attacks as fa  # noqa: E402

OUT = ROOT / "results" / "privacy" / "e2_fedgm"
BASE = 20261116
REPS = range(10)
NTS = (10, 20, 31, 50, None)          # None = all treated (~235)
SIGMAS = ("treated_raw", "treated_shrunk")
GS = (1, 2, 5, 10, 31, 100)
DP_GS, DP_EPS = (1, 10, 100), (0.1, 1.0, 10.0, 100.0, 1000.0)


def _seed(*key):
    return int(np.random.SeedSequence([BASE, *key]).generate_state(1)[0])


def run_task(task):
    rep, nt, sigma = task
    tag = f"rep{rep:02d}_nt{nt or 'all'}_{sigma}"
    path = OUT / "rows" / f"{tag}.csv"
    if path.exists():
        return path
    t0 = time.time()
    df = ist.load(rep, "single_arm", target="ATT", n_treated=nt)
    cols = [c for c in df if c[0] == "X" and c[1:].isdigit()]
    X, W = df[cols].to_numpy(float), df.W.to_numpy()
    Xt, Xc, site = X[W == 1], X[W == 0], df.H.to_numpy()[W == 0]
    sid = SIGMAS.index(sigma)
    S, info, log = fa.recorded_search(Xt, Xc, sigma, seed=_seed(rep, nt or 0, sid))
    Tt = Xt @ S
    mu, sd = Xc.mean(0), Xc.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    binary = np.all((Xc == 0) | (Xc == 1), axis=0)
    ws = [w for w, _ in log]
    common = dict(rep=rep, n_t=len(Xt), n_t_label=nt or "all", sigma=sigma,
                  sigma_rank=info["sigma_rank"], n_controls=len(Xc), evals_total=len(log),
                  evals_to_full=fa.evals_to_full_exposure(Tt, S, ws))
    rows = []
    for G in GS:
        if G > len(ws):
            continue
        X_hat, rank = fa.recover_from_cross_terms(Tt, S, Xc, mu, ws[:G])
        rows.append(dict(common, attack="cross_terms", G=G, eps=np.nan, rank=rank,
                         **fa.recovery_metrics(X_hat, Xc, mu, sd, binary, site)))
    rho = 2 * np.linalg.norm(Xc - mu, axis=1).max()
    for G in DP_GS:
        if G > len(ws):
            continue
        for eps in DP_EPS:
            rng = np.random.default_rng(_seed(rep, nt or 0, sid, G, int(eps * 10)))
            X_hat, rank = fa.recover_from_cross_terms(Tt, S, Xc, mu, ws[:G], eps=eps, rho=rho,
                                                      rng=rng)
            agree = (fa.match_agreement_under_dp(Tt, Xc @ S, S, ws[0], eps, rho, rng)
                     if G == 1 else np.nan)
            rows.append(dict(common, attack="cross_terms_dp", G=G, eps=eps, rank=rank,
                             eps_per_patient=G * len(Xt) * eps, match_agreement=agree,
                             **fa.recovery_metrics(X_hat, Xc, mu, sd, binary, site)))
    n_ever = n_ident = 0
    for s in np.unique(site):
        k = np.flatnonzero(site == s)
        pos = np.full(len(Xc), -1)
        pos[k] = np.arange(len(k))
        sets = [pos[idx][pos[idx] >= 0] for _, idx in log]
        ident, ever = fa.identified_by_moments(sets, len(k))
        n_ever, n_ident = n_ever + int(ever.sum()), n_ident + int(ident.sum())
    rows.append(dict(common, attack="matched_moments", G=len(log), eps=np.nan,
                     n_ever_matched=n_ever, n_identified=n_ident,
                     identified_share_of_matched=n_ident / max(n_ever, 1),
                     identified_share_of_all=n_ident / len(Xc)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"{tag}: {len(log)} evals, full exposure after {common['evals_to_full']}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    tasks = [(r, nt, s) for nt in NTS[::-1] for r in REPS for s in SIGMAS]
    with Pool(args.jobs) as pool:
        paths = pool.map(run_task, tasks, chunksize=1)
    cells = pd.concat(pd.read_csv(p) for p in paths)
    cells.to_csv(OUT / "cells.csv", index=False)
    print(f"wrote {len(cells)} rows to {OUT / 'cells.csv'}")


if __name__ == "__main__":
    main()
