#!/bin/bash
# One check script (experiments/checks/NAME.py) as a single SLURM job:
#   sbatch --job-name=NAME --time=HH:MM:SS --cpus-per-task=C experiments/slurm/check.sh \
#       NAME [args ...]
#SBATCH --account=def-jhchen
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11 r/4.5.0
source ~/venvs/fedgm/bin/activate
export R_LIBS_USER=~/R/library/4.5

NAME=$1; shift
cd ~/fedgm-benchmark
python -m "experiments.checks.$NAME" "$@"
