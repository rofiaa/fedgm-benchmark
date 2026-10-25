"""DGP A and DGP B generated with the authors' own code (Simulations_Fed).

Parameters are copied from FedCaus_Obs/notebooks/simulations.ipynb, cells 1, 22-24
(DGP A, "Some overlap") and 31-33 (DGP B). Where the notebook and the paper's App. D
disagree, the defaults follow the paper and the notebook value is a named variant
(see docs/findings.md):
  - DGP A n_k: paper Table 3: 650 (default), notebook 500 (N_K_NOTEBOOK)
  - DGP A mu_3: paper (2,2,2,1,..) (MU3_PAPER, default), notebook (3,3,3,1,..)
    (MU3_NOTEBOOK). The paper's value reproduces Fig. 2's true-ATE line.
  - DGP B n: paper Table 4: 4000 (default), notebook 1500 (N_B_NOTEBOOK)
  - gamma vectors: kept as in the notebook, whose leading 0 multiplies X1
    (treatment_cols puts the intercept last), i.e. shifted by one position relative
    to App. D.
`dgp_a_notebook_params()` / `dgp_b_notebook_params()` give the notebook's settings.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.estimators._authors import dgp_fed

DIM_X = 10
X_COLS = [f"X{i}" for i in range(1, DIM_X + 1)]

N_K_PAPER, N_K_NOTEBOOK = 650, 50 * DIM_X
MU3_PAPER, MU3_NOTEBOOK = 2, 3
N_B_PAPER, N_B_NOTEBOOK = 4000, 1500

GAMMA_1 = [0, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25]
GAMMA_2_POOR = [0, -2.5, -1, -0.15, -0.15, 0, -0.15, -1, -0.15, -0.15, 0]
GAMMA_3 = [0, 0.15, -0.15, 0.15, -0.15, 0.15, -0.15, 0.15, -0.15, 0.15, -0.15]
THETAS_B = np.array([
    [0, -0.5, -0.5, 0.2, -0.5, -0.5, 0.2, -0.5, -0.5, 0.2, 0.2],
    [0, 0.5, 0.5, 0.2, 0.5, 0.5, 0.2, 0.5, 0.5, 0.2, 0.5],
    [0, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
])


def _shared_params():
    treatment_cols = X_COLS + ["intercept"]
    beta_1 = [2.5 * i / DIM_X - 2 for i in range(DIM_X + 1)]
    beta_1[0] += 0.15
    return {
        "dim_x": DIM_X,
        "beta_1": beta_1,  # unused by the polynomial outcome, required by DGP_Fed
        "beta_0": [2 * i / DIM_X - 2 for i in range(DIM_X + 1)],
        "gamma": [0.5 * i / DIM_X - 0.25 for i in range(DIM_X + 1)],
        "Y_is_linear": "polynomial",
        "sample_size": 150,
        "sigma2": 1,
        "rct_binomial_treatment": False,
        "p_k": 0.5,
        "random_intercept": False,
        "treatment_cols": treatment_cols,
        "outcome_cols": treatment_cols,
        "show_hidden_variables": True,
        "W_model": "logistic",
        "sorting_columns": X_COLS,
        "GaussianMixtures_X": False,
    }


def dgp_a_poor_overlap_params(n_k=N_K_PAPER, mu3_first=MU3_PAPER):
    """DGP A, poor local overlap: site k draws n_k points from N(mu_k, Sigma_k)."""
    rest = [1] * (DIM_X - 3)
    specific = {
        "client1": {"gamma": GAMMA_1, "mean_covariates": [1, 1, 1] + rest,
                    "cov_covariates": np.eye(DIM_X) + 0.5},
        "client2": {"gamma": GAMMA_2_POOR, "mean_covariates": [1.5, 1.5, 1.5] + rest,
                    "cov_covariates": 0.6 * np.eye(DIM_X) + 0.4},
        "client3": {"gamma": GAMMA_3, "mean_covariates": [mu3_first] * 3 + rest,
                    "cov_covariates": 3 * np.eye(DIM_X) + 0.3},
        "global_population": {},
    }
    for name in ("client1", "client2", "client3"):
        specific[name].update(sample_size=n_k, W_model="logistic")
    shared = _shared_params()
    return {name: {**shared, **spec} for name, spec in specific.items()}


def dgp_a_notebook_params():
    """DGP A, poor local overlap, exactly as in the notebook (n_k=500, mu_3=(3,3,3,..))."""
    return dgp_a_poor_overlap_params(n_k=N_K_NOTEBOOK, mu3_first=MU3_NOTEBOOK)


def dgp_b_notebook_params():
    """DGP B, poor local overlap, exactly as in the notebook (n=1500)."""
    return dgp_b_poor_overlap_params(n=N_B_NOTEBOOK)


def dgp_b_poor_overlap_params(n=N_B_PAPER):
    """DGP B, poor local overlap: n points from a 2-component Gaussian mixture, sent to
    sites by a multinomial logistic model in X."""
    shared = _shared_params()
    shared.update(
        mean_covariates=[np.zeros(DIM_X), np.full(DIM_X, 1.5)],
        cov_covariates=[np.eye(DIM_X), np.eye(DIM_X) + 0.5],
        GaussianMixtures_X=True,
    )
    specific = {
        "client1": {"gamma": GAMMA_1},
        "client2": {"gamma": GAMMA_2_POOR},
        "client3": {"gamma": GAMMA_3},
        "global_population": {
            "sample_size": n,
            "membership_Thetas": THETAS_B,
            "mixture_weights": [2 / 3, 1 / 3],
        },
    }
    return {name: {**shared, **spec} for name, spec in specific.items()}


@contextmanager
def _numpy_global_seed(seed):
    """The authors' generator draws from numpy's global RNG; seed it, then restore."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def generate(params, dgp, seed):
    """One dataset from the authors' generator: (authors' frame, our frame).

    dgp: "A" (sites sampled separately) or "B" (sites assigned given X).
    """
    sim = dgp_fed().Simulations_Fed(params, n_simulations=1)
    with _numpy_global_seed(seed):
        df_auth = sim.combine_data() if dgp == "A" else sim.make_data_by_H_given_X()
    return df_auth, to_our_frame(df_auth)


def to_our_frame(df_auth):
    """Authors' frame -> our layout (X1..Xd, H 0..K-1, W, Y)."""
    out = df_auth[X_COLS].reset_index(drop=True).copy()
    out["H"] = df_auth["client"].str.removeprefix("client").astype(int).to_numpy() - 1
    out["W"] = df_auth["W"].astype(int).to_numpy()
    out["Y"] = df_auth["Y"].to_numpy()
    return out


def true_ate(params, dgp, n_total, seed):
    """Monte Carlo ATE: mean of the generator's ITE* on one large sample.

    Like the authors' Plotter("expectancy_of_y1_y0"), but with an explicit size and
    seed. Returns (tau, Monte Carlo SE).
    """
    params = {name: dict(p) for name, p in params.items()}
    if dgp == "A":
        sites = [name for name in params if name.startswith("client")]
        for name in sites:
            params[name]["sample_size"] = n_total // len(sites)
    else:
        params["global_population"]["sample_size"] = n_total
    df_auth, _ = generate(params, dgp, seed)
    ite = df_auth["ITE*"].to_numpy()
    return float(ite.mean()), float(ite.std(ddof=1) / np.sqrt(len(ite)))
