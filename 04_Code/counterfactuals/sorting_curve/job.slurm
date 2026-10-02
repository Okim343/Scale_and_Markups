#!/bin/bash
#SBATCH --job-name=sorting-curve
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=out_results/counterfactuals/sorting_curve/slurm-%j.out
#SBATCH --error=out_results/counterfactuals/sorting_curve/slurm-%j.err
set -euo pipefail
if [[ -f ../.venv/bin/activate ]]; then source ../.venv/bin/activate; fi
# Solver workers use processes; keep BLAS threads at one to avoid oversubscription.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sorting-curve-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
python -m counterfactuals.sorting_curve.run_sorting_curve --workers "$SLURM_CPUS_PER_TASK" "$@"
