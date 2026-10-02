#!/bin/bash
#SBATCH --job-name=sorting-parallel
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=out_results/counterfactuals/sorting_curve/slurm-%j.out
#SBATCH --error=out_results/counterfactuals/sorting_curve/slurm-%j.err
set -euo pipefail
if [[ -f ../.venv/bin/activate ]]; then source ../.venv/bin/activate; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sorting-curve-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
python -m counterfactuals.sorting_curve.run_remaining_parallel "$@"
