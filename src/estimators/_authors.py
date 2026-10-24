"""Lazy access to the authors' ``DGP_Fed`` module (git submodule, never edited).

Importing ``DGP_Fed`` pulls in torch, densratio and rpy2 (which needs R), so it is
deferred until an estimator actually needs it.
"""

import importlib
import sys
from pathlib import Path

import numpy as np

from src.r_env import ensure_r_home

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "FedCausal-ObservationalStudies"
    / "FedCaus_Obs"
    / "scripts"
)

_module = None


def dgp_fed():
    """Return the authors' ``DGP_Fed`` module, importing it on first use."""
    global _module
    if _module is None:
        # DGP_Fed does `from utils import basic_utils`; any other `utils` already
        # imported would shadow the authors' one.
        loaded = sys.modules.get("utils")
        if loaded is not None:
            loaded_dir = Path(getattr(loaded, "__file__", "") or "").resolve().parent
            if loaded_dir != SCRIPTS_DIR:
                raise ImportError(
                    f"A different 'utils' module is already imported ({loaded_dir}); "
                    "the authors' DGP_Fed needs its own scripts/utils.py."
                )
        ensure_r_home()  # DGP_Fed imports rpy2.robjects at module level
        # generate_multivariate_mixtures_of_normals (DGP B) uses np.NaN, removed in
        # NumPy 2.0; it is the only removed name the authors' code uses.
        if "NaN" not in np.__dict__:
            np.NaN = np.nan
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        # Keep the submodule clean: no __pycache__ written into it.
        dont_write = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            _module = importlib.import_module("DGP_Fed")
        finally:
            sys.dont_write_bytecode = dont_write
    return _module
