"""The contract every estimator's `estimate(df, **cfg)` honours.

The returned dict always has the frozen keys below; anything else is an estimator-
specific extra. The experiment harness relies on the frozen keys only.

    tau        float  point estimate of the ATE (NaN if undefined, e.g. Meta-SW with a
                      single-arm site)
    se         float  primary standard error (NaN if unavailable)
    se_method  str    how `se` was obtained
    max_smd    float  max |SMD| after weighting/matching (src/estimators/balance.py;
                      NaN if undefined)
    runtime    float  wall-clock seconds for the whole call, bootstrap included
"""

import math

REQUIRED_KEYS = ("tau", "se", "se_method", "max_smd", "runtime")


def check_result(out):
    """Raise if `out` misses a frozen key or a frozen key has the wrong type."""
    missing = [k for k in REQUIRED_KEYS if k not in out]
    if missing:
        raise KeyError(f"estimator result is missing frozen keys {missing}")
    for key in ("tau", "se", "max_smd", "runtime"):
        value = out[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{key} must be a float, got {type(value).__name__}")
    if not isinstance(out["se_method"], str):
        raise TypeError("se_method must be a str")
    if math.isnan(out["runtime"]):
        raise ValueError("runtime must be set")
    return out
