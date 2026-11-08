"""LaLonde (Dehejia-Wahba): FedDSM (src/estimators/feddsm.py) with its prespecified
settings, ATT, against the experimental benchmark ($1,794). One site (H = 0), so the PS
is covariate-only (ps="x"); the quadratic prognostic terms are those of age, educ, re74,
re75 (the covariates with more than two values), as in genmatch-lalonde's balance
matrix. Complements lalonde_ps.py (same samples and covariates).

Usage: python experiments/checks/lalonde_feddsm.py DW_DIR results/checks/lalonde_feddsm.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.checks.lalonde_ps import BENCHMARK, SEED, _frame, _load  # noqa: E402
from src.estimators import feddsm  # noqa: E402


def main():
    data = _load(Path(sys.argv[1]))
    nsw = data["nsw_dw"]
    treated = nsw[nsw["treat"] == 1]
    samples = {"experimental": nsw,
               "psid1": pd.concat([treated, data["psid_controls"].assign(treat=0)]),
               "cps1": pd.concat([treated, data["cps_controls"].assign(treat=0)])}
    rows = []
    for name, d in samples.items():
        df = _frame(d.reset_index(drop=True))
        for est, r in feddsm.estimate_multi(df, target="ATT", ps="x", seed=SEED).items():
            rows.append(dict(sample=name, method=est, tau=r["tau"] * 1.0,
                             err=r["tau"] - BENCHMARK, max_smd=r["max_smd"],
                             runtime=r["runtime"]))
    res = pd.DataFrame(rows)
    res.to_csv(sys.argv[2], index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
