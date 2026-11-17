"""E1, post-hoc (after the pre-registered prediction failed): the prediction with the member's
own term kept. A member's score is ‖φ̃ᵢ‖²/n + φ̃ᵢᵀ Σ_{j≠i} φ̃ⱼ / n, so with V = Var ‖φ̃‖²
(squared Mahalanobis norm of the features in the population)

    AUROC ≈ Φ( (D/n) / √(D/n + (n−1)D/n² + V/n²) ).

Gaussian features have V = 2D, which gives back Φ(√(D/2n)) for large n; heavy-tailed features
(fourth-order moments, log-normal data) have V ≫ 2D. V comes from the attacker's reference
sample, the same one the attack uses.

    python experiments/privacy/e1_refined.py   # -> results/privacy/e1_mia/cells_refined.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.privacy.e1_mia import BASE, OUT, _ids, groups  # noqa: E402
from src.privacy import mia, populations  # noqa: E402


def norm_var(p, release, d):
    pid, rid = _ids(p, release)
    pop = populations.make(p, d)
    D = mia.n_features(release, d)
    rng = np.random.default_rng(np.random.SeedSequence([BASE, pid, rid, d]))
    X_ref = pop.reference(int(min(max(20000, 20 * D), 60000)), rng)
    att = mia.TracingAttack(X_ref, release)
    q = np.concatenate([np.einsum("ij,jk,ik->i", F - att.mu, att.P, F - att.mu)
                        for F in mia._chunks(X_ref, release)])
    return att.d_eff, float(q.mean()), float(q.var())


def main():
    cells = pd.read_csv(OUT / "cells.csv")
    rows = [dict(population=p, release=r, d=d, **dict(zip(("D_chk", "mean_norm2", "V"),
                                                          norm_var(p, r, d))))
            for p, r, d in groups()]
    v = pd.DataFrame(rows)
    c = cells.merge(v, on=["population", "release", "d"])
    D, n = c.D_eff, c.n
    c["auroc_pred_refined"] = norm.cdf((D / n) / np.sqrt(D / n + (n - 1) * D / n ** 2
                                                         + c.V / n ** 2))
    c["V_over_2D"] = c.V / (2 * D)
    c.to_csv(OUT / "cells_refined.csv", index=False)
    c["dev"] = c.auroc - c.auroc_pred
    c["dev_refined"] = c.auroc - c.auroc_pred_refined
    s = c[c.D_eff >= 10]
    print(s.groupby(["population", "release"])[["dev", "dev_refined"]]
          .agg(lambda x: x.abs().max()).round(3))
    print(v.assign(V_over_2D=v.V / (2 * v.D_chk)).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
