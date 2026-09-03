#!/bin/bash
#SBATCH --job-name=stamp-vg-benchmark
#SBATCH --output=benchmark_same_node_%j.out
#SBATCH --error=benchmark_same_node_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$HOME/VeraGrid/src:$HOME/STAMP_Public"
export VERAGRID_ROOT="$HOME/VeraGrid"
export NUMBA_CACHE_DIR="/tmp/numba-$USER-$SLURM_JOB_ID"
export MPLCONFIGDIR="/tmp/mpl-$USER-$SLURM_JOB_ID"

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket'

module load matlab2026a
cd "$HOME/STAMP_Public/STAMP"
matlab -singleCompThread -batch "benchmark_wscc_stamp"

module purge
module load python3.12
cd "$HOME/STAMP_Public"
"$HOME/venvs/veragrid-benchmark/bin/python" scripts/benchmark_veragrid_wscc_ssa.py
