"""International Stroke Trial (IST) as a semi-synthetic multi-site benchmark.

Real patients, covariates, countries and 6-month death (as Y(0)); treatment and Y(1)
simulated with a known effect, so the truth is exact. Design and parameters (fixed
before any run): docs/ist_design.md. Data: setup/download_ist.sh -> data/ist/.

Columns: X1..X31 (COVARIATES), H (country site 0..8, largest first; in "single_arm" the
treated form site 9), W, Y. df.attrs: ate_true, att_true, tau_true (the one named by
`target`), scenario, design.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ist"
N_SUB = 3000
MIN_SITE = 500

COVARIATES = (["AGE", "RSBP", "RDELAY", "SEX_F", "RSLEEP", "RATRIAL", "RCT", "RVISINF",
               "RHEP24", "RASP3", "RCONSC_D", "RCONSC_U", "STYPE_LACS", "STYPE_POCS",
               "STYPE_TACS"]
              + [f"RDEF{i}_{s}" for i in range(1, 9) for s in ("Y", "C")])

# (gamma, site shift range, treated share, design)
SCENARIOS = {
    "obs_good": (0.6, 0.0, 0.40, "observational"),
    "obs_poor": (1.8, 0.0, 0.40, "observational"),
    "obs_site": (1.2, 1.5, 0.40, "observational"),
    "single_arm": (1.2, 0.0, 0.08, "single_arm"),
}
_SCEN_ID = {name: i for i, name in enumerate(SCENARIOS)}

# treatment index s_raw(x) on standardized covariates (age^2 handled separately)
_PS_COEF = {"AGE": -0.5, "RCONSC_D": -0.4, "RCONSC_U": -0.4, "STYPE_TACS": -0.4,
            "RSBP": 0.3, "RDELAY": -0.3, "RVISINF": 0.3, "RATRIAL": -0.3, "RASP3": 0.2}
_PS_AGE2 = 0.3


def _yn(s):
    return s.map({"Y": 1.0, "N": 0.0})


@lru_cache(maxsize=2)
def population(data_dir=DATA_DIR):
    """The analysis population: covariates (named COVARIATES), country, H, Y0."""
    raw = pd.read_csv(Path(data_dir) / "IST_corrected.csv", encoding="latin-1",
                      low_memory=False)
    raw = raw[raw.RATRIAL.notna() & raw.FDEAD.isin(["Y", "N"])]  # pilot; unknown status
    sizes = raw.COUNTRY.value_counts()
    keep = sorted(sizes[sizes >= MIN_SITE].index, key=lambda c: (-sizes[c], c))
    raw = raw[raw.COUNTRY.isin(keep)].reset_index(drop=True)

    x = pd.DataFrame({"AGE": raw.AGE.astype(float), "RSBP": raw.RSBP.astype(float),
                      "RDELAY": raw.RDELAY.astype(float),
                      "SEX_F": (raw.SEX == "F").astype(float)})
    for c in ("RSLEEP", "RATRIAL", "RCT", "RVISINF", "RHEP24", "RASP3"):
        x[c] = _yn(raw[c])
    for lev in ("D", "U"):
        x[f"RCONSC_{lev}"] = (raw.RCONSC == lev).astype(float)
    for lev in ("LACS", "POCS", "TACS"):  # PACS and the 47 OTH = base
        x[f"STYPE_{lev}"] = (raw.STYPE == lev).astype(float)
    for i in range(1, 9):
        for lev in ("Y", "C"):
            x[f"RDEF{i}_{lev}"] = (raw[f"RDEF{i}"] == lev).astype(float)
    x = x[COVARIATES]
    if x.isna().any().any():
        raise ValueError("unexpected missing covariates in the IST population")
    x["country"] = raw.COUNTRY.to_numpy()
    x["H"] = raw.COUNTRY.map({c: k for k, c in enumerate(keep)}).to_numpy()
    x["Y0"] = (raw.FDEAD == "Y").astype(int).to_numpy()
    return x


@lru_cache(maxsize=2)
def _model(data_dir=DATA_DIR):
    """Fixed population constants: standardization, PS index, intercepts, site shifts."""
    pop = population(data_dir)
    X = pop[COVARIATES]
    mean, sd = X.mean(), X.std()
    z = (X - mean) / sd
    s_raw = sum(b * z[c] for c, b in _PS_COEF.items()) + _PS_AGE2 * (z["AGE"] ** 2 - 1)
    s_sd = float(s_raw.std())
    s = (s_raw / s_sd).to_numpy()
    # site shifts: from +range (lowest crude mortality) to -range (highest)
    K = int(pop.H.max()) + 1
    rank = pop.groupby("H").Y0.mean().rank(method="first").astype(int).to_numpy() - 1
    shift_unit = np.linspace(1.0, -1.0, K)[rank]
    consts = {}
    for name, (gamma, srange, share, _) in SCENARIOS.items():
        d = srange * shift_unit[pop.H.to_numpy()]
        consts[name] = _solve_intercept(gamma * s + d, share)
    return dict(mean=mean, sd=sd, s_sd=s_sd, shift_unit=shift_unit, intercept=consts)


def _solve_intercept(lin, share):
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + lin)))) < share:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def propensity(X, H, scenario, data_dir=DATA_DIR):
    """True e(x, h) for covariates X (DataFrame with COVARIATES) and country sites H."""
    gamma, srange, _, _ = SCENARIOS[scenario]
    m = _model(data_dir)
    z = (X[COVARIATES] - m["mean"]) / m["sd"]
    s_raw = sum(b * z[c] for c, b in _PS_COEF.items()) + _PS_AGE2 * (z["AGE"] ** 2 - 1)
    lin = m["intercept"][scenario] + gamma * s_raw.to_numpy() / m["s_sd"] \
        + srange * m["shift_unit"][np.asarray(H, int)]
    return 1 / (1 + np.exp(-lin))


def effect_probs(X, data_dir=DATA_DIR):
    """(q, r): P(death prevented | Y0 = 1, x), P(death caused | Y0 = 0, x)."""
    m = _model(data_dir)
    z_age = ((X["AGE"] - m["mean"]["AGE"]) / m["sd"]["AGE"]).to_numpy()
    z_sbp = ((X["RSBP"] - m["mean"]["RSBP"]) / m["sd"]["RSBP"]).to_numpy()
    q = 1 / (1 + np.exp(-(-1.6 + 0.4 * z_age + 0.5 * X["STYPE_TACS"].to_numpy())))
    r = 1 / (1 + np.exp(-(-4.0 + 0.5 * z_sbp + 0.4 * z_age)))
    return q, r


def load(rep, scenario, target="ATE", n=N_SUB, data_dir=DATA_DIR):
    """Replication `rep` of `scenario` (SCENARIOS) as X1..X31, H, W, Y."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {list(SCENARIOS)}")
    if target not in ("ATE", "ATT"):
        raise ValueError("target must be 'ATE' or 'ATT'")
    design = SCENARIOS[scenario][3]
    if design == "single_arm" and target != "ATT":
        raise ValueError("single_arm supports the ATT only")
    pop = population(data_dir)
    rng = np.random.default_rng(np.random.SeedSequence([int(rep), _SCEN_ID[scenario], 1995]))
    sub = pop.iloc[np.sort(rng.choice(len(pop), size=n, replace=False))].reset_index(drop=True)
    X, H, Y0 = sub[COVARIATES], sub["H"].to_numpy(), sub["Y0"].to_numpy()

    e = propensity(X, H, scenario, data_dir)
    W = (rng.random(n) < e).astype(int)
    q, r = effect_probs(X, data_dir)
    u = rng.random(n)
    Y1 = np.where(Y0 == 1, (u >= q).astype(int), (u < r).astype(int))
    tau_i = r * (1 - Y0) - q * Y0

    df = pd.DataFrame(X.to_numpy(float), columns=[f"X{i}" for i in range(1, len(COVARIATES) + 1)])
    df["H"] = np.where(W == 1, int(pop.H.max()) + 1, H) if design == "single_arm" else H
    df["W"] = W
    df["Y"] = np.where(W == 1, Y1, Y0).astype(float)
    df.attrs["ate_true"] = float(tau_i.mean())
    df.attrs["att_true"] = float(tau_i[W == 1].mean())
    df.attrs["tau_true"] = df.attrs[f"{target.lower()}_true"]
    df.attrs["scenario"] = scenario
    df.attrs["design"] = design
    return df


def with_site_dummies(df):
    """Append the country indicators (sites 1..K-1) as further X columns, for the
    centralized estimators in the observational scenarios (as IHDP's x19..x25)."""
    if df.attrs.get("design") == "single_arm":
        raise ValueError("site dummies identify the treated in single_arm")
    d = len([c for c in df if c[0] == "X" and c[1:].isdigit()])
    out = df.copy()
    for k in range(1, int(df.H.max()) + 1):
        out[f"X{d + k}"] = (df.H == k).astype(float)
    out.attrs = dict(df.attrs)
    return out
