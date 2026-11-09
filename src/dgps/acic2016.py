"""ACIC 2016 (Dorie et al. 2019, Stat. Sci.) as a multi-site observational benchmark.

Covariates: the competition's 58 real covariates from the Collaborative Perinatal
Project (aciccomp2016::input_2016, n = 4802); the 3 categorical ones are one-hot coded
(first level dropped), giving 79 numeric covariates X1..X79. Treatment and outcomes:
aciccomp2016::dgp_2016(input, setting, seed) for setting 1..77, the competition's grid
over treatment model, % treated, overlap, response model, alignment and effect
heterogeneity (aciccomp2016::parameters_2016). R package: remotes::install_github(
"vdorie/aciccomp/2016").

Sites: ACIC has none, so K = 4 sites are assigned per replication by a multinomial
logistic model on three standardized covariates (the first three with more than ten
distinct values): P(H = k | x) ∝ exp(x_(k)) for k = 1..3, site 0 the baseline. Sites
differ in covariate mix but overlap, and H depends on X only, so the site is not a
confounder (the covariate-only PS applies).

Truth (df.attrs): ate_true = mean(mu1 - mu0), att_true = the same over the treated
(noiseless conditional means); tau_true is the one named by `target`.
Replications: rep 0..99 is the competition's own replication rep + 1 (dgp_2016's
random.seed must be in 1..100), so the datasets are the competition's; sites from
np.random.SeedSequence([setting, rep, 2016]).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from src.r_env import ensure_r_home

N_SETTINGS = 77
N_SITES = 4


@lru_cache(maxsize=1)
def _r():
    ensure_r_home()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    acic = importr("aciccomp2016")
    return ro, acic, ro.r("aciccomp2016::input_2016"), ro.r("aciccomp2016::parameters_2016")


@lru_cache(maxsize=1)
def covariates():
    """The 79 numeric covariates (DataFrame X1..X79) and the original names."""
    ro, _, inp, _ = _r()
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        x = ro.conversion.rpy2py(inp)
    x = pd.get_dummies(x, columns=[c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)
                                   or x[c].dtype == object], drop_first=True, dtype=float)
    names = list(x.columns)
    X = pd.DataFrame(x.to_numpy(float), columns=[f"X{i}" for i in range(1, len(names) + 1)])
    return X, names


@lru_cache(maxsize=1)
def settings():
    ro, _, _, par = _r()
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        p = ro.conversion.rpy2py(par)
    return p.reset_index(drop=True).assign(setting=range(1, N_SETTINGS + 1))


def _sites(X, setting, rep):
    cols = [c for c in X.columns if X[c].nunique() > 10][:3]
    Z = ((X[cols] - X[cols].mean()) / X[cols].std()).to_numpy()
    scores = np.column_stack([np.zeros(len(Z)), Z])
    p = np.exp(scores - scores.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    rng = np.random.default_rng(np.random.SeedSequence([setting, rep, 2016]))
    u = rng.random(len(p))[:, None]
    return (u > np.cumsum(p, 1)).sum(1)


def load(rep, setting, target="ATE"):
    """Replication `rep` of ACIC 2016 setting `setting` (1..77): X1..X79, H, W, Y."""
    if target not in ("ATE", "ATT"):
        raise ValueError("target must be 'ATE' or 'ATT'")
    _, acic, inp, _ = _r()
    X, _ = covariates()
    if not 0 <= rep < 100:
        raise ValueError("rep must be in 0..99 (the competition's 100 replications)")
    sim = acic.dgp_2016(inp, int(setting), int(rep) + 1)
    get = lambda k: np.asarray(sim.rx2(k), dtype=float)  # noqa: E731
    w, y, mu0, mu1 = get("z").astype(int), get("y"), get("mu.0"), get("mu.1")
    df = X.copy()
    df["H"] = _sites(X, setting, rep)
    df["W"], df["Y"] = w, y
    df.attrs["ate_true"] = float(np.mean(mu1 - mu0))
    df.attrs["att_true"] = float(np.mean((mu1 - mu0)[w == 1]))
    df.attrs["tau_true"] = df.attrs[f"{target.lower()}_true"]
    df.attrs["setting"] = int(setting)
    return df
