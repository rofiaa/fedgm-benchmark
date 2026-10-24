"""Point rpy2 at the active env's R before anything imports rpy2.robjects."""

import os
import sys
from pathlib import Path


def ensure_r_home():
    """Default R_HOME to this env's R.

    Without it, rpy2 can pick up a system R (e.g. /Library/Frameworks) instead of the
    env's and segfault. An R_HOME set by the user is left alone.
    """
    env_r_home = Path(sys.prefix) / "lib" / "R"
    if "R_HOME" not in os.environ and env_r_home.is_dir():
        os.environ["R_HOME"] = str(env_r_home)
