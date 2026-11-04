#!/bin/bash
# One (scenario, rep) per SLURM array task, through experiments/run.py (resumable).
#
#   sbatch --job-name=NAME --time=HH:MM:SS --array=0-N experiments/slurm/array.sh \
#       CONFIG OUT REP_OFFSET SCENARIO [SCENARIO ...]
#
# Task i runs scenario SCENARIOS[i % S] at rep REP_OFFSET + i / S (S = number of
# scenarios), so --array=0-$((S*R-1)) covers R reps of every scenario. Completed pairs
# in OUT are skipped, so a timed-out task can simply be resubmitted.
#SBATCH --account=def-jhchen
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
module load StdEnv/2023 python/3.11 r/4.5.0
source ~/venvs/fedgm/bin/activate
export R_LIBS_USER=~/R/library/4.5

CONFIG=$1; OUT=$2; OFFSET=$3; shift 3
SCENARIOS=("$@")
S=${#SCENARIOS[@]}
SCENARIO=${SCENARIOS[$((SLURM_ARRAY_TASK_ID % S))]}
REP=$((OFFSET + SLURM_ARRAY_TASK_ID / S))

cd ~/fedgm-benchmark
python experiments/run.py "$CONFIG" --reps "$REP" --scenarios "$SCENARIO" --jobs 1 --out "$OUT"
