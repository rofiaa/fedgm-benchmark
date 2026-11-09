"""Disclosure audit of the current FedDSM pipeline (v0), as run in our experiments.

For one replication of each dataset, with the exact fold assignment of our runs:
  1. Cells: every (site, arm, fold) cell whose sums B'B and B'Y a site sends for the
     prognostic model, by size. Cells of 1 patient disclose that patient's covariates and
     outcome exactly (intercept row of B'B = x, B'Y's intercept entry = y); cells of 2
     disclose both patients up to their order (Σx and Σxx' determine x1 - x2 up to
     sign). Larger cells: higher moments still constrain the patients (not quantified).
  2. Demonstration: reconstructs every 1- and 2-patient cell from its sums alone and
     reports the error against the true covariates and outcomes.
  3. Message sizes: numbers each site sends per step (PS by distributed Newton;
     prognostic sums; per-patient scores for cross-site matching).
Datasets: authors' DGPs (6 scenarios), IHDP, ACIC 2016 setting 1 (rep 0; ATE runs, so
both arms' prognostic models).

Usage: python experiments/checks/feddsm_disclosure_audit.py results/checks/feddsm_disclosure_audit.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.harness import config_seed, data_seed, load_config  # noqa: E402
from src.estimators import feddsm  # noqa: E402


def _cases():
    """(dataset, scenario, frame, ps, seed of the FedDSM config) for rep 0."""
    m = load_config(ROOT / "experiments/configs/feddsm_authors.py")
    for s in m.SCENARIOS:
        yield ("Authors' DGPs", s.name, s.generate(data_seed(m.BASE_SEED, s.name, 0)), "x",
               config_seed(m.BASE_SEED, s.name, 0, "feddsm_x"))
    from src.dgps import ihdp
    m = load_config(ROOT / "experiments/configs/feddsm_ihdp.py")
    s = m.SCENARIOS[0]
    yield ("IHDP", "IHDP", ihdp.without_site_dummies(s.generate(0)), "site",
           config_seed(m.BASE_SEED, s.name, 0, "feddsm_site"))
    try:
        m = load_config(ROOT / "experiments/configs/feddsm_acic.py")
        s = m.SCENARIOS[0]
        yield ("ACIC 2016", s.name, s.generate(0), "x",
               config_seed(m.BASE_SEED, s.name, 0, "feddsm_x"))
    except Exception as exc:  # aciccomp2016 not installed
        print("ACIC skipped:", exc)


def _reconstruct(Bc, Yc, d):
    """Recover the patients of a 1- or 2-patient cell from (B'B, B'Y) only.
    Returns the max abs error over covariates and outcomes (order-free for n = 2)."""
    G, g = Bc.T @ Bc, Bc.T @ Yc
    i0 = Bc.shape[1] - 1                       # intercept column (last)
    s, S = G[i0, :d], G[:d, :d]                # Σx, Σxx'
    x_true = Bc[:, :d]
    if len(Bc) == 1:
        return max(np.abs(s - x_true[0]).max(), abs(g[i0] - Yc[0]))
    D = S - np.outer(s, s) / 2                 # = (x1-x2)(x1-x2)'/2
    w, V = np.linalg.eigh(D)
    diff = V[:, -1] * np.sqrt(max(2 * w[-1], 0))
    c1, c2 = (s + diff) / 2, (s - diff) / 2
    A = np.column_stack([np.r_[1, c1], np.r_[1, c2]])
    y12 = np.linalg.lstsq(A, np.r_[g[i0], g[:d]], rcond=None)[0]
    err_a = max(np.abs(c1 - x_true[0]).max(), np.abs(c2 - x_true[1]).max(),
                abs(y12[0] - Yc[0]), abs(y12[1] - Yc[1]))
    err_b = max(np.abs(c1 - x_true[1]).max(), np.abs(c2 - x_true[0]).max(),
                abs(y12[0] - Yc[1]), abs(y12[1] - Yc[0]))
    return min(err_a, err_b)


def audit(dataset, scen, df, ps, seed):
    x_cols = sorted((c for c in df if c[0] == "X" and c[1:].isdigit()), key=lambda c: int(c[1:]))
    X, H, W, Y = (df[x_cols].to_numpy(float), df.H.to_numpy(int), df.W.to_numpy(int),
                  df.Y.to_numpy(float))
    K, d = H.max() + 1, len(x_cols)
    quad = [i for i, c in enumerate(x_cols) if df[c].nunique() > 2]
    D = (H[:, None] == np.arange(1, K)[None, :]).astype(float) if ps == "site" else None
    conf = X if D is None else np.column_stack([X, D])
    _, iters = feddsm.fed_logistic_newton(conf, W, H)
    p = conf.shape[1] + 1
    fold_seed = np.random.SeedSequence(seed).generate_state(2)
    rows, recon = [], []
    for arm in (0, 1):
        q_full = feddsm._basis(X[:1], quad, None if D is None else D[:1]).shape[1]
        use_int = q_full <= (W == arm).sum() / 2
        B = feddsm._basis(X, quad, D, interactions=use_int)
        q = B.shape[1]
        fold = feddsm.fold_labels(W, H, arm, int(fold_seed[arm]))
        sizes = [int(((H == k) & (fold == f)).sum()) for k in range(K) for f in range(5)]
        for k in range(K):
            for f in range(5):
                idx = np.flatnonzero((H == k) & (fold == f))
                if 1 <= len(idx) <= 2:
                    recon.append(_reconstruct(B[idx], Y[idx], d))
        sizes = np.array(sizes)
        rows.append(dict(
            dataset=dataset, scenario=scen, arm=arm, ps=ps, sites=K, covariates=d,
            cells=len(sizes), cells_n0=int((sizes == 0).sum()), cells_n1=int((sizes == 1).sum()),
            cells_n2=int((sizes == 2).sum()), cells_n3_4=int(((sizes >= 3) & (sizes <= 4)).sum()),
            cells_n5_9=int(((sizes >= 5) & (sizes <= 9)).sum()),
            cells_n10plus=int((sizes >= 10).sum()), min_nonempty_cell=int(sizes[sizes > 0].min()),
            cells_below_basis_size=int(((sizes > 0) & (sizes < q)).sum()),
            prognostic_terms=q, prognostic_numbers_per_site=5 * (q * (q + 1) // 2 + q),
            ps_params=p, newton_iterations=iters,
            ps_numbers_per_site=iters * (p + p * (p + 1) // 2),
            score_values_per_site=int(round(len(W) / K)) * (2 + (1 if arm == 1 else 0) + 1)))
    return rows, recon


def main():
    out = Path(sys.argv[1])
    rows, recon_all = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case in _cases():
            r, recon = audit(*case)
            rows += r
            recon_all += [(case[0], case[1], e) for e in recon]
    res = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res[["dataset", "scenario", "arm", "cells", "cells_n1", "cells_n2", "cells_n3_4",
                   "cells_n5_9", "min_nonempty_cell", "cells_below_basis_size",
                   "prognostic_terms", "prognostic_numbers_per_site", "newton_iterations",
                   "ps_numbers_per_site"]].to_string(index=False))
    if recon_all:
        r = pd.DataFrame(recon_all, columns=["dataset", "scenario", "max_abs_error"])
        print("\nReconstructed 1- and 2-patient cells from their sums alone:")
        print(r.groupby(["dataset", "scenario"]).max_abs_error.agg(["count", "max"]).to_string())


if __name__ == "__main__":
    main()
