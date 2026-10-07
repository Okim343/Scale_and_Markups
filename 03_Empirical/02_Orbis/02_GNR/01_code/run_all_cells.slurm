#!/bin/bash
# Orbis Spain GNR: all cells, seed checks, combine. Submit from 03_Empirical/02_Orbis/02_GNR:
#   mkdir -p 01_intermediary/logs && sbatch 01_code/run_all_cells.slurm [cell ...]
# Without arguments runs every cell in config cells.list. A failing cell is logged and skipped;
# 02_combine_cells.py writes 03_outdata/ only when every cell finished with the current config.
#SBATCH --job-name=gnr_orbis_es
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --output=01_intermediary/logs/%x-%j.out
#SBATCH --error=01_intermediary/logs/%x-%j.err

set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # parallelism is across starts
source "${GNR_VENV:-$HOME/venvs/scale_markups_gnr}/bin/activate"
cd "$SLURM_SUBMIT_DIR/01_code"
CELLS=("$@")
if [ ${#CELLS[@]} -eq 0 ]; then
  CELLS=($(python -c "import yaml; print(*yaml.safe_load(open('config_gnr_orbis_es.yaml'))['cells']['list'])"))
fi
for C in "${CELLS[@]}"; do
  echo "=== cell $C start $(date +%T)"
  if ! python 01_run_gnr_cell.py --cell "$C" --workers "$SLURM_CPUS_PER_TASK" --force; then
    echo "FAILED cell $C (run)"; continue
  fi
  python 03_seed_check_cell.py --cell "$C" --seeds 1982 7 123 9999 --workers "$SLURM_CPUS_PER_TASK" \
    || echo "FAILED cell $C (seed check)"
  echo "=== cell $C finished $(date +%T)"
done
python 02_combine_cells.py || echo "FAILED combine"
echo "=== job finished $(date +%T)"
