"""Lazy access to the authors' ``DGP_Fed`` module (git submodule, never edited).

Importing ``DGP_Fed`` pulls in torch, densratio and rpy2 (which needs R), so it is
deferred until an estimator actually needs it.
"""

import importlib
import os
import sys
from pathlib import Path

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
        # DGP_Fed imports rpy2.robjects at module level. Without R_HOME, rpy2 can pick
        # up a system R (e.g. /Library/Frameworks) instead of the env's and segfault.
        env_r_home = Path(sys.prefix) / "lib" / "R"
        if "R_HOME" not in os.environ and env_r_home.is_dir():
            os.environ["R_HOME"] = str(env_r_home)
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        _module = importlib.import_module("DGP_Fed")
    return _module
