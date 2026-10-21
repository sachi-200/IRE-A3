#!/bin/bash
# Run Part 3 (compression) on the server, then rebuild figures, tables and results.csv.
#
#   bash run_part3_server.sh     inside an interactive session (use tmux)
#   sbatch run_part3_server.sh   as a SLURM batch job (submit from ~/A3/code)
#
#SBATCH --job-name=a3-part3
#SBATCH --nodelist=gnode006
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=part3_%j.log
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
PY=.venv/bin/python3

LOG="part3_run_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

echo "== $(date) on $(hostname) =="
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)   (allocated CPUs: $(nproc))"

LOAD1=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$LOAD1" -v c="$(getconf _NPROCESSORS_CONF)" 'BEGIN { exit !(l > c) }'; then
    echo "WARNING: load average $LOAD1 is above the node's CPU count; timings will be slower than usual"
fi

for f in results/raw/part1_sift.csv results/raw/part2_sift_ivfflat.csv; do
    [ -f "$f" ] || { echo "$f missing: run Parts 1 and 2 first" >&2; exit 1; }
done

if ls results/raw/part3_*.csv >/dev/null 2>&1; then
    BACKUP="../part3_backup_$(date +%Y%m%d_%H%M)"
    mkdir -p "$BACKUP"
    mv results/raw/part3_*.csv "$BACKUP"/
    echo "old Part 3 results moved to $BACKUP"
fi

START=$(date +%s)
$PY -u part3_compression.py
$PY part3_report.py
$PY collect_results.py
echo "== done in $(( ($(date +%s) - START) / 60 )) min; log: $LOG =="
