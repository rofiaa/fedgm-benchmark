"""Run an experiment config through the harness.

    python experiments/run.py experiments/configs/ps_mc.py --reps 0-99 --jobs 7 \
        --out results/ps_mc [--scenarios A_poor B_poor] [--configs fed_logistic]

--reps takes a range ("0-29"), a single rep ("5") or a list ("0-9,20-29"); each rep's
seeds depend only on (BASE_SEED, scenario, rep, config), so SLURM array tasks can each
run a slice, e.g. --reps $((SLURM_ARRAY_TASK_ID*10))-$((SLURM_ARRAY_TASK_ID*10+9))
--jobs 1. Each (scenario, rep) goes to <out>/rows/<scenario>/rep<NNNN>.csv when it
finishes; rerunning the same command resumes, skipping completed (scenario, rep) pairs.
A progress line with the projected finish time follows each pair. Combine and score the
rows with experiments/summarize.py.
"""

import os

# one BLAS/OpenMP thread per process; set before numpy is imported
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.harness import parse_reps, run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="experiment config file")
    parser.add_argument("--reps", required=True, help='e.g. "0-99", "0-29", "5"')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    start = time.perf_counter()
    written = run(args.config, parse_reps(args.reps), args.out,
                  scenario_names=args.scenarios, config_ids=args.configs, jobs=args.jobs)
    for path in written:
        print(f"wrote {path}")
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
