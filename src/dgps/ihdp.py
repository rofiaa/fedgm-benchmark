"""IHDP semi-synthetic benchmark (Hill 2011) as a multi-site observational dataset.

Real covariates and treatment from the Infant Health and Development Program, made
observational by Hill by removing a non-random subset of the treated (139 treated, 608
controls, n = 747); outcomes simulated from the covariates (response surface B, the
100 replications of Shalit et al. 2017; setup/download_ihdp.sh). Every replication has
the same units, treatment and covariates; only the outcomes change.

Covariates x1..x25: x1..x6 continuous, x7..x18 binary (x14 coded 1/2), x19..x25 dummies
of 7 of the 8 IHDP sites (none = site 0). Sites are H = 0..7. Four sites have 2-9
treated units, so local overlap is poor in a natural way.

Truth per replication (attached as df.attrs): tau_true = mean(mu1 - mu0) over the 747
units (the sample ATE, the usual IHDP target), att_true = the same over the treated
(4 by construction).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ihdp100"
N_REPS = 100
SITE_DUMMIES = [f"X{i}" for i in range(19, 26)]


@lru_cache(maxsize=2)
def _arrays(data_dir):
    parts = [np.load(Path(data_dir) / f"ihdp_npci_1-100.{s}.npz", allow_pickle=False)
             for s in ("train", "test")]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("x", "t", "yf", "mu0", "mu1")}


def load(rep, data_dir=DATA_DIR):
    """Replication `rep` (0..99) as our frame: X1..X25, H, W, Y.

    Rows are put in one canonical order (lexicographic in x), so the design (X, W, H)
    is identical across replications (the files shuffle rows per replication).
    """
    a = _arrays(str(data_dir))
    x = a["x"][:, :, rep]
    order = np.lexsort(x.T[::-1])
    x, w = x[order], a["t"][order, rep].astype(int)
    mu0, mu1 = a["mu0"][order, rep], a["mu1"][order, rep]
    df = pd.DataFrame(x, columns=[f"X{i}" for i in range(1, 26)])
    dummies = x[:, 18:25]
    df["H"] = np.where(dummies.sum(1) == 0, 0, dummies.argmax(1) + 1)
    df["W"] = w
    df["Y"] = a["yf"][order, rep]
    df.attrs["tau_true"] = float(np.mean(mu1 - mu0))
    df.attrs["att_true"] = float(np.mean((mu1 - mu0)[w == 1]))
    return df


def without_site_dummies(df):
    """The frame without X19..X25 (for estimators that take the site as H: with the
    site dummies in X, P(H | X) is degenerate and they are constant within a site)."""
    return df.drop(columns=SITE_DUMMIES)
