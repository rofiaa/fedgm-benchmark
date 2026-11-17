"""Populations for the membership-inference experiments (docs/privacy_e1.md)."""

import numpy as np

POPULATIONS = ("gaussian", "lognormal", "ist")
RHO = 0.5
IST_SPLIT_SEED = 20261116
IST_REF_SHARE = 0.4


class Synthetic:
    """d-variate normal with correlation RHO^|i-j|, or its exponential (`lognormal`)."""

    def __init__(self, name, d):
        if name not in ("gaussian", "lognormal"):
            raise ValueError(name)
        self.name, self.d = name, d
        i = np.arange(d)
        self._L = np.linalg.cholesky(RHO ** np.abs(i[:, None] - i[None, :]))

    def _draw(self, m, rng):
        Z = rng.standard_normal((m, self.d)) @ self._L.T
        return np.exp(Z) if self.name == "lognormal" else Z

    def reference(self, size, rng):
        return self._draw(size, rng)

    def draw_members(self, n, rng):
        return self._draw(n, rng), None

    def draw_out(self, m, rng, exclude=None):
        return self._draw(m, rng)


class IST:
    """Real IST patients, the first d covariates of src.dgps.ist.COVARIATES.

    Split once (fixed seed) into a reference part for the attacker and a target pool; members
    and non-members are disjoint draws without replacement from the pool.
    """

    def __init__(self, d):
        from src.dgps import ist
        cols = ist.COVARIATES[:d]
        if len(cols) < d:
            raise ValueError(f"IST has {len(ist.COVARIATES)} covariates")
        X = ist.population()[cols].to_numpy(float)
        perm = np.random.default_rng(IST_SPLIT_SEED).permutation(len(X))
        n_ref = int(IST_REF_SHARE * len(X))
        self.name, self.d, self.columns = "ist", d, cols
        self._ref, self._pool = X[perm[:n_ref]], X[perm[n_ref:]]

    def reference(self, size=None, rng=None):
        return self._ref

    def draw_members(self, n, rng):
        idx = rng.choice(len(self._pool), n, replace=False)
        return self._pool[idx], idx

    def draw_out(self, m, rng, exclude):
        rest = np.setdiff1d(np.arange(len(self._pool)), exclude)
        return self._pool[rng.choice(rest, m, replace=False)]


def make(name, d):
    return IST(d) if name == "ist" else Synthetic(name, d)
