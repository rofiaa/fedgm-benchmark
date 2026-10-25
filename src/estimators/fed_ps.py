"""Federated IPW/AIPW via propensity-score aggregation (Khellaf, Bellet & Josse,
arXiv:2505.17961), applied to an external data frame.

The authors' `Simulations_Fed.estimate_ipw/estimate_aipw` generate data inside their
simulation loop, so we re-assemble the estimator from their standalone functions:

    ê(x) = Σ_k ω̂_k(x) ê_k(x),   ω̂_k(x) = P̂(H = k | X = x),

with ê_k fitted at site k and predicted on all rows, and
τ̂ = Σ_k (n_k / n) τ̂_k, where τ̂_k is the IPW (or AIPW) estimate on site k using the
global ê (and, for AIPW, outcome models μ̂_0, μ̂_1 shared by all sites).
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.estimators._authors import dgp_fed
from src.estimators.balance import smd

_IMPLEMENTED = {
    "mw": {"logistic", "logistic_fedavg", "nn"},
    "local_ps": {"logistic", "rf"},
    "ipw_type": {"HT", "Hajek", "Hajek_pooled"},
    "aipw": {False, True},
    "outcome": {"ols", "fedavg"},
}

# Outcome-model FedAvg (authors' fedavg_sgd_server_side): the notebook's
# federation_parameters for AIPW. Fixed number of rounds, gradients clipped to [-1, 1].
FEDAVG_OUTCOME_NOTEBOOK = {"T_rounds": 1000, "eta": 0.01, "E_local_steps": 1,
                           "B_batch_size": None}

# Membership-weight FedAvg: run until the relative change in Theta drops below `tol`,
# at most T_rounds rounds. tol=None runs exactly T_rounds (the authors' behaviour).
# E_local_steps=1: with E > 1 and one class per site, FedAvg converges to a different
# point than the pooled fit (client drift; docs/findings.md). standardize=True rescales
# X by its global mean/SD (shareable as per-site sums) so eta=1 converges in < 1000
# rounds; the intercept makes the fitted probabilities invariant to this rescaling.
FEDAVG_MW_DEFAULTS = {"T_rounds": 1000, "eta": 1.0, "E_local_steps": 1,
                      "B_batch_size": None, "tol": 1e-6, "standardize": True}
# The notebook's federation_parameters_omega; used for parity with the authors' code.
FEDAVG_MW_NOTEBOOK = {"T_rounds": 30, "eta": 0.1, "E_local_steps": 3,
                      "B_batch_size": None, "tol": None, "standardize": False}

BOOTSTRAP_B = 200
_Z95 = 1.959963984540054

_MW_FIT = {"logistic": "pooled-centralized", "logistic_fedavg": "fedavg",
           "nn": "nn-pooled"}
_OUTCOME_FIT = {"ols": "pooled-ols", "fedavg": "fedavg-sgd-clipped"}


ESTIMANDS = ("IPW-HT", "IPW-Hajek_pooled", "IPW-Hajek", "AIPW")


def estimate(
    df,
    mw="logistic",
    local_ps="logistic",
    ipw_type="HT",
    aipw=False,
    seed=None,
    mw_intercept=True,
    fedavg_params=None,
    outcome="ols",
    outcome_fedavg_params=None,
    bootstrap=0,
):
    """Federated IPW / AIPW estimate of the ATE.

    df: columns X1..Xd, H (sites 0..K-1), W (0/1), Y.
    mw: membership-weight model for P(H|X). "logistic" = authors' MW_estimation_pooled
        (a centralized multinomial fit, the default in their code); "logistic_fedavg" =
        FedAvg multinomial logistic with full-batch local steps (authors' algorithm and
        gradient, plus a stopping rule; see FEDAVG_MW_DEFAULTS); "nn" = authors'
        train_NN (one hidden layer of 64 units, Adam, early stopping on the training
        loss, X standardized), trained on the **pooled** data, not by FedAvg.
    local_ps: local propensity model e_k(x), fitted at site k on X + intercept.
        "logistic" = authors' compute_gamma; "rf" = the authors' inline sklearn forest
        (RandomForestClassifier(n_estimators=2000, min_samples_leaf=int(sqrt(n_k)),
        max_depth=2); their use_grf=False path), seeded through `seed`.
    ipw_type: "HT" (Horvitz-Thompson; primary), "Hajek_pooled" (weights normalized over
        all sites), or "Hajek" -- the authors' version, which normalizes within each
        site using the global ê and is inconsistent (docs/findings.md); parity only.
        IPW only; must be "HT" when aipw=True.
    aipw: augmented IPW with linear outcome models μ̂_w(x) = [x, 1]·β̂_w, as the authors'
        "GD-MW_*" estimators.
    outcome: outcome-model fit for aipw=True. "ols" = per-arm OLS on all sites' data
        (the authors' "GD" when params_for_federation is None); "fedavg" = authors'
        fedavg_sgd_server_side (clipped gradients, fixed rounds).
    outcome_fedavg_params: overrides of FEDAVG_OUTCOME_NOTEBOOK for outcome="fedavg".
    mw_intercept: include an intercept in the membership model (the authors' IPW path
        does; their AIPW path does not).
    fedavg_params: overrides of FEDAVG_MW_DEFAULTS for mw="logistic_fedavg". Only
        B_batch_size=None is supported: minibatches are drawn from numpy's global RNG.
    bootstrap: number of site-stratified bootstrap replicates B (0 = off; the benchmark
        uses BOOTSTRAP_B = 200). See _site_stratified_bootstrap.
    seed: seeds the bootstrap through np.random.SeedSequence(seed) and its spawned
        children; None draws fresh entropy (returned as boot_seed_entropy). For mw="nn"
        it also seeds torch (inside torch.random.fork_rng, so the global torch RNG is
        restored): an int seed is passed to torch.manual_seed as is, anything else is
        mapped through SeedSequence; the value used is returned as mw_seed. For
        local_ps="rf" it seeds one np.random.RandomState shared by the K site forests
        (fitted in site order), which reproduces the authors' unseeded forests after a
        global np.random.seed(seed); returned as ps_seed.

    Returns the frozen keys (src/estimators/common.py) plus ci_if, estimand, e_hat,
    e_local, omega_hat, n_k, smd, mw_fit, mw_rounds, mw_converged, mw_seed,
    local_ps_fit, ps_seed, outcome_fit, outcome_coef. se / ci_if are the IF-based SE
    and normal 95% CI. mw_rounds/mw_converged are None unless mw="logistic_fedavg";
    outcome_fit and outcome_coef (dict beta_1, beta_0 over X1..Xd, intercept) are None
    unless aipw=True. With bootstrap > 0, also: se_boot, ci_boot_percentile (2.5%,
    97.5%), boot_B, boot_ok, boot_failed, boot_arm_loss (replicates where a site lost a
    treatment arm), boot_arm_loss_by_site, boot_mw_fit, boot_seed_entropy, boot_taus.
    """
    if aipw and ipw_type != "HT":
        raise ValueError("ipw_type applies to IPW only; leave it at 'HT' with aipw=True")
    if ipw_type not in _IMPLEMENTED["ipw_type"] or aipw not in _IMPLEMENTED["aipw"]:
        raise NotImplementedError(f"ipw_type={ipw_type!r}, aipw={aipw!r}")
    name = "AIPW" if aipw else f"IPW-{ipw_type}"
    return estimate_multi(
        df, [name], mw=mw, local_ps=local_ps, seed=seed, mw_intercept=mw_intercept,
        fedavg_params=fedavg_params, outcome=outcome,
        outcome_fedavg_params=outcome_fedavg_params, bootstrap=bootstrap,
    )[name]


def estimate_multi(
    df,
    estimands,
    mw="logistic",
    local_ps="logistic",
    seed=None,
    mw_intercept=True,
    fedavg_params=None,
    outcome="ols",
    outcome_fedavg_params=None,
    bootstrap=0,
    bootstrap_for=None,
):
    """Several federated estimators from **one** fit of the nuisances.

    estimands: names from ESTIMANDS ("IPW-HT", "IPW-Hajek_pooled", "IPW-Hajek", "AIPW").
    bootstrap_for: the estimands that get the site-stratified bootstrap (default: all
        of them when bootstrap > 0). Each bootstrap replicate refits the nuisances once
        and evaluates every estimand in bootstrap_for on them.
    Other arguments as in estimate().

    Returns {estimand: result}, each result with the same keys as estimate(); `runtime`
    is the wall-clock time of the whole call (shared fits included) and
    `runtime_shared_by` lists the estimands it covers.
    """
    estimands = list(dict.fromkeys(estimands))
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        raise NotImplementedError(f"estimands {sorted(unknown)} are not implemented")
    for name, value in [("mw", mw), ("local_ps", local_ps), ("outcome", outcome)]:
        if value not in _IMPLEMENTED[name]:
            raise NotImplementedError(f"{name}={value!r} is not implemented yet")
    fedavg = {**FEDAVG_MW_DEFAULTS, **(fedavg_params or {})}
    if fedavg["B_batch_size"] is not None:
        raise NotImplementedError("FedAvg with minibatches (B_batch_size) is not supported")
    bootstrap_for = list(estimands) if bootstrap_for is None else list(bootstrap_for)
    if not set(bootstrap_for) <= set(estimands):
        raise ValueError("bootstrap_for must be a subset of estimands")
    if not bootstrap:
        bootstrap_for = []
    aipw = "AIPW" in estimands
    cfg = dict(mw=mw, mw_intercept=mw_intercept, fedavg=fedavg, local_ps=local_ps,
               mw_seed=_int_seed(seed, 63) if mw == "nn" else None,
               ps_seed=_int_seed(seed, 32) if local_ps == "rf" else None,
               outcome=outcome, outcome_fedavg_params=outcome_fedavg_params)

    start = time.perf_counter()
    A = dgp_fed()
    data, x_cols, K = _to_authors_frame(df)
    nuis = _fit_nuisances(A, data, x_cols, K, need_outcome=aipw, **cfg)
    fits = _estimands(A, data, x_cols, K, nuis, estimands)
    e_hat = nuis["e_hat"]

    weights = np.where(data["W"] == 1, 1 / e_hat, 1 / (1 - e_hat))
    balance = smd(data[x_cols], data["W"], weights)
    index = df.index
    shared = dict(
        e_hat=pd.Series(e_hat, index=index, name="e_hat"),
        e_local=pd.DataFrame(nuis["e_local"], index=index, columns=range(K)),
        omega_hat=pd.DataFrame(nuis["omega"], index=index, columns=range(K)),
        n_k=dict(enumerate(nuis["n_k"].tolist())),
        smd=balance,
        max_smd=float(balance.abs().max()),
        mw_fit=_MW_FIT[mw],
        mw_rounds=nuis["mw_rounds"],
        mw_converged=nuis["mw_converged"],
        mw_seed=cfg["mw_seed"],
        local_ps_fit=local_ps,
        ps_seed=cfg["ps_seed"],
    )
    outcome_coef = None
    if aipw:
        outcome_coef = {"beta_1": pd.Series(nuis["beta_1"], index=x_cols + ["intercept"]),
                        "beta_0": pd.Series(nuis["beta_0"], index=x_cols + ["intercept"])}

    boot = {}
    if bootstrap_for:
        boot_cfg = dict(cfg, estimands=bootstrap_for)
        boot = _site_stratified_bootstrap(A, data, x_cols, K, boot_cfg, bootstrap, seed,
                                          fit=_fit_taus)

    runtime = time.perf_counter() - start
    results = {}
    for name in estimands:
        tau, se = fits[name]["tau"], fits[name]["se"]
        results[name] = dict(
            tau=tau,
            se=se,
            se_method="influence function, nuisances treated as fixed (provisional)",
            ci_if=(tau - _Z95 * se, tau + _Z95 * se),
            estimand=name,
            **shared,
            outcome_fit=_OUTCOME_FIT[outcome] if name == "AIPW" else None,
            outcome_coef=outcome_coef if name == "AIPW" else None,
            **boot.get(name, {}),
            runtime=runtime,
            runtime_shared_by=list(estimands),
        )
    return results


def _fit_nuisances(A, data, x_cols, K, mw, mw_intercept, fedavg, local_ps, mw_seed,
                   ps_seed, outcome, outcome_fedavg_params, need_outcome):
    """Local PS, membership weights, global ê and (if needed) outcome models."""
    ps_cols = x_cols + ["intercept"]
    mw_cols = x_cols + ["intercept"] if mw_intercept else x_cols

    e_local = _local_propensities(A, data, ps_cols, K, local_ps, ps_seed)
    mw_rounds = mw_converged = None
    if mw == "logistic":
        omega = _site_order_numeric(
            A.MW_estimation_pooled(data, mw_cols, scale=False).to_numpy(), K
        )
    elif mw == "nn":
        omega = _nn_membership(A, data, x_cols, K, mw_seed)
    else:
        omega, mw_rounds, mw_converged = _fedavg_membership(
            A, data, mw_cols, K, fedavg["T_rounds"], fedavg["eta"],
            fedavg["E_local_steps"], fedavg["tol"], fedavg["standardize"],
        )
    e_hat = A.membership_weighting_vectorized(pd.DataFrame(omega), pd.DataFrame(e_local))
    n_k = np.array([(data["client"] == _label(k)).sum() for k in range(K)])
    beta_1 = beta_0 = None
    if need_outcome:
        beta_1, beta_0 = _outcome_models(A, data, ps_cols, K, outcome,
                                         outcome_fedavg_params)
    return dict(e_local=e_local, omega=omega, e_hat=e_hat, n_k=n_k, mw_rounds=mw_rounds,
                mw_converged=mw_converged, beta_1=beta_1, beta_0=beta_0)


def _estimands(A, data, x_cols, K, nuis, names):
    """τ̂ and IF SE of each estimand in `names` from fitted nuisances."""
    data = data.copy()
    e_hat = nuis["e_hat"]
    data["e_hat"] = e_hat
    site_masks = [data["client"] == _label(k) for k in range(K)]
    n_k = nuis["n_k"]
    W, Y = data["W"].to_numpy(), data["Y"].to_numpy()
    out = {}
    for name in names:
        if name == "AIPW":
            X_out = data[x_cols + ["intercept"]].to_numpy()
            mu1, mu0 = X_out @ nuis["beta_1"], X_out @ nuis["beta_0"]
            data["mu1_hat"], data["mu0_hat"] = mu1, mu0
            tau_k = [
                A.aipw_function(data[m], y1_col="mu1_hat", y0_col="mu0_hat", e_col="e_hat")
                for m in site_masks
            ]
            tau = float(np.average(tau_k, weights=n_k))
            psi = _aipw_terms(W, Y, e_hat, mu1, mu0)
        elif name == "IPW-Hajek_pooled":
            tau = float(A.ipw_function(data, hat_e_name="e_hat", ht_or_hajek="Hajek"))
            all_rows = pd.Series(True, index=data.index)
            psi = _ipw_terms_hajek(W, Y, e_hat, [all_rows], [tau])
        else:
            ipw_type = name.removeprefix("IPW-")
            tau_k = [
                A.ipw_function(data[m], hat_e_name="e_hat", ht_or_hajek=ipw_type)
                for m in site_masks
            ]
            tau = float(np.average(tau_k, weights=n_k))
            if ipw_type == "HT":
                psi = _ipw_terms_ht(W, Y, e_hat)
            else:
                psi = _ipw_terms_hajek(W, Y, e_hat, site_masks, tau_k)
        out[name] = dict(tau=tau, se=_if_se(psi))
    return out


def _fit_taus(A, data, x_cols, K, estimands, **cfg):
    """Bootstrap fit: refit the nuisances once, return {estimand: τ̂}."""
    nuis = _fit_nuisances(A, data, x_cols, K, need_outcome="AIPW" in estimands, **cfg)
    return {name: r["tau"] for name, r in _estimands(A, data, x_cols, K, nuis,
                                                      estimands).items()}


def _site_stratified_bootstrap(A, data, x_cols, K, cfg, B, seed, fit):
    """Nonparametric bootstrap that resamples rows within each site (n_k fixed) and
    refits every nuisance (local PS, MW, outcome models) in each replicate.

    `fit(A, data, x_cols, K, **cfg)` returns {estimand: τ̂} for one dataset; one call per
    replicate, so all estimands share the replicate's fits. Returns {estimand: extras}.

    MW is refit with the pooled multinomial logistic fit even when mw="logistic_fedavg":
    converged FedAvg reaches the same fit (tests/test_fed_ps.py), at a fraction of the
    cost. mw="nn" retrains the net in every replicate (≈ 2-4 s each).
    Replicate b draws from np.random.default_rng(SeedSequence(seed).spawn(B)[b]); its NN
    and RF seeds come from two further children of that sequence.
    A replicate in which some site that had both arms keeps only one is counted in
    boot_arm_loss; it is still estimated (fed_ps: that site's ê_k becomes the constant 0
    or 1; meta_sw: τ̂_b is NaN and counts as failed).
    """
    seq = np.random.SeedSequence(seed)
    site_rows = [np.flatnonzero(data["client"].to_numpy() == _label(k)) for k in range(K)]
    W = data["W"].to_numpy()
    had_both = [len(np.unique(W[rows])) == 2 for rows in site_rows]
    boot_cfg = dict(cfg)
    if boot_cfg.get("mw") == "logistic_fedavg":
        boot_cfg["mw"] = "logistic"
    names = list(cfg["estimands"])

    taus = {name: np.full(B, np.nan) for name in names}
    arm_loss_by_site = np.zeros(K, dtype=int)
    arm_loss = 0
    for b, child in enumerate(seq.spawn(B)):
        rng = np.random.default_rng(child)
        nn_seq, rf_seq = child.spawn(2)
        if boot_cfg.get("mw") == "nn":
            boot_cfg["mw_seed"] = _int_seed(nn_seq, 63)
        if boot_cfg.get("local_ps") == "rf":
            boot_cfg["ps_seed"] = _int_seed(rf_seq, 32)
        draws = [rng.choice(rows, size=len(rows)) for rows in site_rows]
        lost = [both and len(np.unique(W[d])) < 2 for both, d in zip(had_both, draws)]
        idx = np.concatenate(draws)
        arm_loss += any(lost)
        arm_loss_by_site += np.array(lost)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = fit(A, data.iloc[idx].reset_index(drop=True), x_cols, K, **boot_cfg)
            for name in names:
                taus[name][b] = rep[name]
        except Exception:  # e.g. a singular refit; counted as failed, not fatal
            pass

    shared = dict(
        boot_B=B,
        boot_arm_loss=int(arm_loss),
        boot_arm_loss_by_site=dict(enumerate(arm_loss_by_site.tolist())),
        boot_mw_fit=_MW_FIT.get(boot_cfg.get("mw")),
        boot_seed_entropy=seq.entropy,
    )
    out = {}
    for name in names:
        ok = taus[name][np.isfinite(taus[name])]
        out[name] = dict(
            se_boot=float(np.std(ok, ddof=1)) if len(ok) > 1 else float("nan"),
            ci_boot_percentile=tuple(float(q) for q in np.percentile(ok, [2.5, 97.5]))
            if len(ok) else (float("nan"), float("nan")),
            boot_ok=len(ok),
            boot_failed=B - len(ok),
            boot_taus=taus[name],
            **shared,
        )
    return out


def _label(k):
    """Authors' site label for our 0-indexed site k."""
    return f"client{k + 1}"


def _to_authors_frame(df):
    """Our layout (X1..Xd, H, W, Y) -> authors' layout (X1..Xd, intercept, client, W, Y)."""
    x_cols = sorted(
        (c for c in df.columns if c[0] == "X" and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if not x_cols:
        raise ValueError("df has no covariate columns X1..Xd")
    missing = {"H", "W", "Y"} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns {sorted(missing)}")

    H = df["H"].to_numpy()
    if not np.issubdtype(H.dtype, np.integer):
        raise ValueError("H must be integer site labels 0..K-1")
    K = int(H.max()) + 1
    if H.min() != 0 or len(np.unique(H)) != K:
        raise ValueError("H must take every value 0..K-1")
    if not np.isin(df["W"].unique(), [0, 1]).all():
        raise ValueError("W must be 0/1")

    data = pd.DataFrame({c: df[c].to_numpy(dtype=float) for c in x_cols})
    data["intercept"] = 1.0
    data["client"] = [_label(h) for h in H]
    data["W"] = df["W"].to_numpy(dtype=int)
    data["Y"] = df["Y"].to_numpy(dtype=float)
    return data, x_cols, K


def _local_propensities(A, data, ps_cols, K, local_ps="logistic", ps_seed=None):
    """ê_k fitted on site k, predicted on all rows; (n, K) array.

    A site with a single treatment arm gets the constant ê_k = 0 or 1 (it never / always
    treats), as in the authors' external-control-arm setting.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(ps_seed) if local_ps == "rf" else None
    e_local = np.empty((len(data), K))
    for k in range(K):
        site = data[data["client"] == _label(k)]
        arms = site["W"].unique()
        if len(arms) == 1:
            e_local[:, k] = float(arms[0])
        elif local_ps == "rf":
            forest = RandomForestClassifier(
                n_estimators=2000, min_samples_leaf=int(np.sqrt(len(site))),
                max_depth=2, random_state=rng,
            ).fit(site[ps_cols].to_numpy(), site["W"].to_numpy())
            e_local[:, k] = forest.predict_proba(data[ps_cols].to_numpy())[:, 1]
        else:
            gamma = A.compute_gamma(site, ps_cols)
            e_local[:, k] = A.logistic_function_vectorized(data, ps_cols, gamma)
    return e_local


def _site_order_numeric(proba, K):
    """Reorder classifier columns from alphabetical to numeric site order.

    The authors' classifiers return columns in sorted-label order ("client1",
    "client10", "client2", ...), which differs from numeric order once K >= 10.
    """
    alphabetical = sorted(_label(k) for k in range(K))
    return proba[:, [alphabetical.index(_label(k)) for k in range(K)]]


def _fedavg_membership(A, data, mw_cols, K, T_rounds, eta, E_local_steps, tol,
                       standardize):
    """FedAvg multinomial logistic regression for P(H|X), full-batch local steps.

    Same algorithm as the authors' federated_logistic_regression / MW_estimation
    (their gradient_multi_logistic and softmax; Theta starts at 0; each round, every
    site takes E full-batch gradient steps from the global Theta and the server averages
    with weights n_k/n), with three differences: classes are in numeric site order; the
    loop stops once ||Theta_t - Theta_{t-1}||_F / ||Theta_{t-1}||_F < tol; and with
    standardize=True the non-intercept columns are rescaled by their global mean/SD
    (unlike their scale=True, which fits unscaled and predicts scaled).
    Their per-step shuffle (df.sample) does not affect a full-batch gradient.

    Returns (omega (n, K), rounds run, converged); converged is None when tol is None.
    """
    X = data[mw_cols].to_numpy()
    if standardize:
        cols = [i for i, c in enumerate(mw_cols) if c != "intercept"]
        X = X.copy()
        X[:, cols] = (X[:, cols] - X[:, cols].mean(axis=0)) / X[:, cols].std(axis=0)
    H = np.eye(K)[[int(c.removeprefix("client")) - 1 for c in data["client"]]]
    sites = [H[:, k] == 1 for k in range(K)]
    n_k = np.array([s.sum() for s in sites])

    theta = np.zeros((K, X.shape[1]))
    converged = None if tol is None else False
    for t in range(1, T_rounds + 1):
        updates = []
        for s in sites:
            theta_k = theta
            for _ in range(E_local_steps):
                theta_k = theta_k - eta * A.gradient_multi_logistic(X[s], theta_k, H[s])
            updates.append(theta_k)
        theta_new = np.average(updates, weights=n_k, axis=0)
        change = np.linalg.norm(theta_new - theta) / max(np.linalg.norm(theta), 1e-300)
        theta = theta_new
        if tol is not None and change < tol:
            converged = True
            break
    return A.softmax(X @ theta.T), t, converged


def _outcome_models(A, data, out_cols, K, outcome, outcome_fedavg_params):
    """Linear outcome models per arm, shared by all sites: (beta_1, beta_0)."""
    if outcome == "ols":
        return tuple(
            np.asarray(A.estimate_beta_MLE(data[data["W"] == w], out_cols))
            for w in (1, 0)
        )
    params = {**FEDAVG_OUTCOME_NOTEBOOK, **(outcome_fedavg_params or {})}
    if params["B_batch_size"] is not None:
        raise NotImplementedError("FedAvg with minibatches (B_batch_size) is not supported")
    df_dict = {_label(k): data[data["client"] == _label(k)] for k in range(K)}
    return tuple(
        np.asarray(A.fedavg_sgd_server_side(
            df_dict, out_cols, beta1_or_beta0_or_gamma=arm, **params
        ))
        for arm in ("beta1", "beta0")
    )


def _aipw_terms(W, Y, e, mu1, mu0):
    return mu1 - mu0 + W * (Y - mu1) / e - (1 - W) * (Y - mu0) / (1 - e)


def _int_seed(seed, bits):
    """A non-negative int below 2**bits (63 for torch.manual_seed, 32 for
    np.random.RandomState). Ints in range pass through, for parity with the authors'
    torch.manual_seed(s) / np.random.seed(s); None, sequences and SeedSequences are
    mapped through SeedSequence."""
    if isinstance(seed, (int, np.integer)) and 0 <= seed < 2**bits:
        return int(seed)
    if not isinstance(seed, np.random.SeedSequence):
        seed = np.random.SeedSequence(seed)
    return int(seed.generate_state(1, np.uint64)[0] >> (64 - bits))


def _nn_membership(A, data, x_cols, K, torch_seed):
    """Authors' Simulations_Fed.train_NN, called unbound with a minimal stand-in for
    `self` (it only reads sorting_columns and caches nn_model). rerun_NN=True: with the
    default False it would return the cached model object instead of probabilities.
    Columns come out in LabelEncoder (alphabetical) site order and are reordered."""
    import types

    import torch

    stub = types.SimpleNamespace(sorting_columns=x_cols)
    with torch.random.fork_rng():
        torch.manual_seed(torch_seed)
        proba = A.Simulations_Fed.train_NN(stub, data, rerun_NN=True)
    return _site_order_numeric(np.asarray(proba), K)


def _ipw_terms_ht(W, Y, e):
    return W * Y / e - (1 - W) * Y / (1 - e)


def _ipw_terms_hajek(W, Y, e, site_masks, tau_k):
    """Linearized per-unit terms of a Hajek estimator normalized within each group in
    `site_masks` (sites for the authors' "Hajek", one all-rows group for "Hajek_pooled").

    For unit i in site k: tau_k + (n_k/N1_k) W (Y - m1_k)/e - (n_k/N0_k) (1-W)(Y - m0_k)/(1-e),
    with N1_k = Σ_k W/e and m1_k the site's weighted treated mean (same for controls).
    The correction terms sum to zero within each site, so mean(psi) equals tau.
    """
    psi = np.empty(len(W))
    for mask, t_k in zip(site_masks, tau_k):
        m = mask.to_numpy()
        w1, w0 = W[m] / e[m], (1 - W[m]) / (1 - e[m])
        n_site = m.sum()
        m1 = np.sum(w1 * Y[m]) / np.sum(w1)
        m0 = np.sum(w0 * Y[m]) / np.sum(w0)
        psi[m] = (
            t_k
            + n_site / np.sum(w1) * w1 * (Y[m] - m1)
            - n_site / np.sum(w0) * w0 * (Y[m] - m0)
        )
    return psi


def _if_se(psi):
    if not np.all(np.isfinite(psi)):
        warnings.warn("non-finite per-unit terms (ê hit 0 or 1); se is NaN")
        return float("nan")
    return float(np.std(psi, ddof=1) / np.sqrt(len(psi)))
