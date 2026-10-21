#!/bin/bash
# Rerun Part 2 on the server (full data), then rebuild figures, tables and results.csv.
#
# Run it either way:
#   bash run_part2_server.sh          inside an interactive session on the compute node
#                                     (use tmux/screen so it survives a disconnect)
#   sbatch run_part2_server.sh        as a SLURM batch job (#SBATCH lines below)
#
# Old Part 2 results are moved to ../part2_backup_<date>/ first, so a partial
# rerun never mixes with the old numbers. Part 1 results are kept (Part 2
# reads the exact-search time from them).
#
#SBATCH --job-name=a3-part2
#SBATCH --nodelist=gnode006
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=part2_%j.log
set -euo pipefail

# sbatch runs a copy of the script, so locate the code via the submit directory.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
source .venv/bin/activate

LOG="part2_run_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

echo "== $(date) on $(hostname) =="
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)   (allocated CPUs: $(nproc))"
if command -v squeue >/dev/null; then
    echo "jobs on this node:"; squeue -w "$(hostname)" 2>/dev/null || true
fi

if [ ! -f results/raw/part1_sift.csv ] || [ ! -f results/raw/part1_glove.csv ]; then
    echo "results/raw/part1_*.csv missing: run part1_exact.py first" >&2
    exit 1
fi

if ls results/raw/part2_*.csv >/dev/null 2>&1; then
    BACKUP="../part2_backup_$(date +%Y%m%d_%H%M)"
    mkdir -p "$BACKUP"
    mv results/raw/part2_*.csv "$BACKUP"/
    echo "old Part 2 results moved to $BACKUP"
fi

START=$(date +%s)
python3 -u part2_bench.py
python3 part2_report.py
python3 collect_results.py
echo "== done in $(( ($(date +%s) - START) / 60 )) min; log: $LOG =="
