#!/bin/bash
#SBATCH --job-name=sorting-padding-test
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=out_results/counterfactuals/sorting_curve/slurm-%j.out
#SBATCH --error=out_results/counterfactuals/sorting_curve/slurm-%j.err
set -euo pipefail
source ../.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sorting-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
python -m counterfactuals.sorting_curve.validate_compact
