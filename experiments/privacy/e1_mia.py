"""E1: membership-inference sensitivity curves for shared statistics (docs/privacy_e1.md).

    python experiments/privacy/e1_mia.py --jobs 6
        -> results/privacy/e1_mia/cells.csv, roc.csv
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
from src.privacy import mia, populations  # noqa: E402

OUT = ROOT / "results" / "privacy" / "e1_mia"
BASE = 20261116
NS = (10, 20, 35, 50, 100, 200, 650, 2000)
DS = {"gaussian": (2, 5, 10, 20, 50), "lognormal": (2, 5, 10, 20, 50),
      "ist": (2, 5, 10, 20, 31)}
QUAD_DS = (2, 5, 10)
ROC_GRID = np.geomspace(1e-4, 1, 41)


def groups():
    for p in populations.POPULATIONS:
        for release in mia.RELEASES:
            for d in DS[p]:
                if release == "quad" and (p == "ist" or d not in QUAD_DS):
                    continue
                yield p, release, d


def _ids(p, release):
    return populations.POPULATIONS.index(p), mia.RELEASES.index(release)


def run_group(task):
    p, release, d = task
    t0 = time.time()
    pid, rid = _ids(p, release)
    pop = populations.make(p, d)
    D = mia.n_features(release, d)
    ref_rng = np.random.default_rng(np.random.SeedSequence([BASE, pid, rid, d]))
    X_ref = pop.reference(int(min(max(20000, 20 * D), 60000)), ref_rng)
    attack = mia.TracingAttack(X_ref, release)
    cells, rocs = [], []
    for n in NS:
        reps = int(np.clip(np.ceil(4000 / n), 20, 400))
        n_out = int(np.ceil(20000 / reps))
        rng = np.random.default_rng(np.random.SeedSequence([BASE, pid, rid, d, n]))
        s_in, s_out = mia.run_cell(pop.draw_members, pop.draw_out, attack, n, reps, n_out, rng)
        row = dict(population=p, release=release, d=d, n=n, D=D, D_eff=attack.d_eff,
                   n_ref=len(X_ref), reps=reps, n_in=len(s_in), n_out=len(s_out),
                   **mia.roc_metrics(s_in, s_out), **mia.predicted(attack.d_eff, n))
        cells.append(row)
        rocs.append(pd.DataFrame(dict(population=p, release=release, d=d, n=n, fpr=ROC_GRID,
                                      tpr=mia.roc_on_grid(s_in, s_out, ROC_GRID))))
    print(f"{p:9s} {release:8s} d={d:2d} D_eff={attack.d_eff:5d} {time.time() - t0:6.1f}s",
          flush=True)
    return pd.DataFrame(cells), pd.concat(rocs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = sorted(groups(), key=lambda t: -mia.n_features(t[1], t[2]))  # big ones first
    with Pool(args.jobs) as pool:
        res = pool.map(run_group, tasks, chunksize=1)
    cells = pd.concat([r[0] for r in res]).sort_values(["population", "release", "d", "n"])
    roc = pd.concat([r[1] for r in res]).sort_values(["population", "release", "d", "n", "fpr"])
    cells.to_csv(OUT / "cells.csv", index=False)
    roc.to_csv(OUT / "roc.csv", index=False)
    print(f"wrote {len(cells)} cells to {OUT}")


if __name__ == "__main__":
    main()
