"""Membership inference on released summary statistics (docs/privacy_e1.md).

A release is the group mean of a feature map φ(x) over n patients (equivalent to the sums,
since n is public). The attacker knows the population through a reference sample and scores a
target record x by the whitened inner-product (tracing) statistic

    s(x) = (φ(x) − μ)ᵀ P (m̂ − μ),

μ and P the reference mean and pseudo-inverse covariance of φ. Prediction for Gaussian
features: AUROC ≈ Φ(√(D/(2n))), TPR(α) ≈ 1 − Φ(z₁₋α − √(D/n)).
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

RELEASES = ("mean_var", "cov", "quad")
FPRS = (1e-3, 1e-2, 1e-1)
_CHUNK = 2000


def n_features(release, d):
    """Number of released numbers D (before removing coincident features)."""
    if release == "mean_var":
        return 2 * d
    if release == "cov":
        return d + d * (d + 1) // 2
    if release == "quad":
        q = 1 + d + d * (d + 1) // 2
        return q * (q + 1) // 2 - 1
    raise ValueError(f"unknown release {release!r}")


def features(X, release):
    """φ(X), one row per patient. `quad` drops the constant b₀b₀ = 1 entry. `release` may
    also be a callable feature map X -> φ(X) (e.g. a method's own released sums)."""
    if callable(release):
        return np.asarray(release(X), float)
    X = np.asarray(X, float)
    n, d = X.shape
    if release == "mean_var":
        return np.hstack([X, X ** 2])
    iu = np.triu_indices(d)
    second = X[:, iu[0]] * X[:, iu[1]]
    if release == "cov":
        return np.hstack([X, second])
    if release == "quad":
        B = np.hstack([np.ones((n, 1)), X, second])
        iq = np.triu_indices(B.shape[1])
        return (B[:, iq[0]] * B[:, iq[1]])[:, 1:]
    raise ValueError(f"unknown release {release!r}")


def _chunks(X, release):
    for i in range(0, len(X), _CHUNK):
        yield features(X[i:i + _CHUNK], release)


def feature_mean(X, release):
    """The release m̂: mean of φ over the group, computed in chunks."""
    total = sum(F.sum(axis=0) for F in _chunks(X, release))
    return total / len(X)


class TracingAttack:
    """Whitened inner-product attack fitted on a reference sample of the population."""

    def __init__(self, X_ref, release, rcond=1e-8):
        self.release = release
        n = len(X_ref)
        s1 = s2 = 0.0
        for F in _chunks(X_ref, release):
            s1 = s1 + F.sum(axis=0)
            s2 = s2 + F.T @ F
        self.mu = s1 / n
        cov = (s2 - n * np.outer(self.mu, self.mu)) / (n - 1)
        # rank cutoff on the correlation scale: features differ in scale by many orders of
        # magnitude (x vs x⁴, binary vs blood pressure²), and a cutoff on the raw covariance
        # would drop real features
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        ok = sd > 0
        R = cov[np.ix_(ok, ok)] / np.outer(sd[ok], sd[ok])
        lam, V = np.linalg.eigh(R)
        keep = lam > rcond * lam.max()
        self.d_eff = int(keep.sum())
        P = np.zeros_like(cov)
        P[np.ix_(ok, ok)] = ((V[:, keep] / lam[keep]) @ V[:, keep].T) / np.outer(sd[ok], sd[ok])
        self.P = P

    def scores(self, X_targets, m_hat):
        v = self.P @ (m_hat - self.mu)
        return np.concatenate([(F - self.mu) @ v for F in _chunks(X_targets, self.release)])


def roc_metrics(s_in, s_out, fprs=FPRS):
    """AUROC, TPR at fixed FPRs (threshold = non-member quantile), advantage, ε lower bound."""
    s_in, s_out = np.asarray(s_in, float), np.asarray(s_out, float)
    y = np.r_[np.ones(len(s_in)), np.zeros(len(s_out))]
    s = np.r_[s_in, s_out]
    out = {"auroc": float(roc_auc_score(y, s))}
    fpr_c, tpr_c, _ = roc_curve(y, s)
    out["advantage"] = float(np.max(tpr_c - fpr_c))
    for a in fprs:
        thr = np.quantile(s_out, 1 - a, method="higher")
        tpr, fpr = float(np.mean(s_in > thr)), float(np.mean(s_out > thr))
        out[f"tpr_at_{a:g}"] = tpr
        out[f"fpr_at_{a:g}"] = fpr
    a = 1e-2
    tpr, fpr = out[f"tpr_at_{a:g}"], max(out[f"fpr_at_{a:g}"], 1 / len(s_out))
    out["eps_lb_at_0.01"] = float(np.log(tpr / fpr)) if tpr > fpr else 0.0
    return out


def roc_on_grid(s_in, s_out, grid):
    """TPR at each FPR in `grid` (threshold = non-member quantile)."""
    s_in, s_out = np.asarray(s_in, float), np.asarray(s_out, float)
    thr = np.quantile(s_out, 1 - np.asarray(grid), method="higher")
    return np.array([np.mean(s_in > t) for t in thr])


def predicted(D, n, fprs=FPRS):
    """Gaussian prediction: AUROC = Φ(√(D/2n)), TPR(α) = 1 − Φ(z₁₋α − √(D/n))."""
    r = np.sqrt(D / n)
    out = {"auroc_pred": float(norm.cdf(r / np.sqrt(2)))}
    for a in fprs:
        out[f"tpr_at_{a:g}_pred"] = float(1 - norm.cdf(norm.ppf(1 - a) - r))
    return out


def run_cell(draw_members, draw_out, attack, n, reps, n_out, rng):
    """Pool member and non-member scores over `reps` releases of groups of size n.

    draw_members(n, rng) and draw_out(m, rng, exclude) return covariate arrays; for a finite
    population draw_out must avoid the current members (it receives their indices).
    """
    s_in, s_out = [], []
    for _ in range(reps):
        Xm, idx = draw_members(n, rng)
        m_hat = feature_mean(Xm, attack.release)
        Xo = draw_out(n_out, rng, idx)
        s_in.append(attack.scores(Xm, m_hat))
        s_out.append(attack.scores(Xo, m_hat))
    return np.concatenate(s_in), np.concatenate(s_out)
