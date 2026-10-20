"""Merge every part's benchmark rows into results.csv and record machine specs.

  results/raw/part*_<dataset>.csv  ->  results/results.csv
  machine specs and library versions ->  results/machine.txt
"""
import argparse
import csv
import glob
import os

from common import BENCH_COLUMNS, machine_info, results_dir, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    out = results_dir(args.quick)

    rows = []
    for path in sorted(glob.glob(os.path.join(out, "raw", "part*.csv"))):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "recall_at_10" not in (reader.fieldnames or []):
                continue  # not a benchmark file (e.g. part1_checks_*.csv)
            rows.extend({c: r.get(c, "") for c in BENCH_COLUMNS} for r in reader)
    write_csv(os.path.join(out, "results.csv"), rows, BENCH_COLUMNS)
    print(f"results.csv: {len(rows)} benchmark rows")

    with open(os.path.join(out, "machine.txt"), "w") as f:
        for k, v in machine_info().items():
            f.write(f"{k}: {v}\n")
    print(f"machine specs written to {os.path.join(out, 'machine.txt')}")


if __name__ == "__main__":
    main()
