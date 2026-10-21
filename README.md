# CS4.406 Assignment 3: Benchmarking ANN Search

Benchmarks exact search, HNSW, IVF-Flat and IVF-PQ on SIFT-1M and GloVe-100
(ann-benchmarks versions). GloVe vectors are L2-normalised and Euclidean
distance is used throughout.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

The datasets (about 1 GB in total) are downloaded on first use into `data/`.
To keep them elsewhere, for example on a server's scratch disk:

```bash
export ANN_DATA_DIR=/path/to/ann-data
```

If the machine has no internet access, copy the two files there yourself:

```bash
curl -L -o "$ANN_DATA_DIR/sift-128-euclidean.hdf5" http://ann-benchmarks.com/sift-128-euclidean.hdf5
curl -L -o "$ANN_DATA_DIR/glove-100-angular.hdf5"  http://ann-benchmarks.com/glove-100-angular.hdf5
```

## Reproducing the report

```bash
bash run_all.sh            # full data; writes results/
bash run_all.sh --quick    # first 100,000 vectors only; writes results_quick/ (< 15 min)
```

`--quick` indexes only the first 100K base vectors of each dataset and
recomputes the exact neighbours for that subset (the neighbours stored in the
data files refer to the full sets). It checks that the whole pipeline works;
its numbers are not the ones in the report.

Multi-threaded runs use one thread per physical core by default
(hyperthreads made FAISS slower in testing). Set `ANN_THREADS=N` to override,
for example on a shared server. All single-query timings use one thread.

Queries are split in half: the first 5,000 are used for tuning and the last
5,000 for every reported number.

Part 2 measures recall on all 5,000 queries of each half (with a
multi-threaded batch search, which returns the same neighbours) and speed
one query at a time on one thread, as the median of 3 runs. The reported QPS
times all 5,000 test queries (`--time-n-test`); the tune-half QPS, used only
to choose settings, times the first 1,000 tune queries (`--time-n-tune`).
Every timing row records the machine's 1-minute load average (`load_avg`),
since runs were made on a shared node. Settings are chosen using tune-half
numbers only (Pareto frontier, and the fastest setting reaching recall 0.90 /
0.95); the reported values are the test-half numbers of those settings.
A sweep stops early once tune recall reaches 0.99, stops improving, or the
index becomes slower than exact search.

Part 2 takes a few hours on the full data. Each index type can be run
separately, and results are saved after every build setting:

```bash
python3 part2_bench.py --datasets sift --indexes hnsw
python3 part2_report.py      # plots/tables from whatever has finished
```

## Outputs

| Path | Contents |
|---|---|
| `results/results.csv` | one row per benchmark run behind a number in the report |
| `results/raw/` | per-part CSVs that `results.csv` is built from, plus Part 1 agreement checks |
| `results/figures/` | all figures |
| `results/machine.txt` | CPU, RAM, OS and library versions of the machine that produced the results |

## Files

| File | Purpose |
|---|---|
| `common.py` | paths, quick mode, thread count, machine info, CSV writing |
| `data.py` | download and load datasets, normalise GloVe, query split |
| `recall.py` | tie-aware recall@10 |
| `tests/test_recall.py` | unit tests for recall (`python3 -m pytest tests`) |
| `part1_exact.py` | Part 1: exact baseline, agreement check, single vs batch timing |
| `part2_bench.py` | Part 2: build HNSW / IVF-Flat / IVF-PQ settings and sweep efSearch / nprobe |
| `part2_report.py` | Part 2: trade-off plots and the QPS / cost tables, from the saved CSVs |
| `part3_compression.py` | Part 3: IVF-PQ with m = 8/16/32/64, plain and with exact re-ranking of the top 100 (SIFT) |
| `part3_report.py` | Part 3: speed/recall and memory/recall figure and table |
| `run_part3_server.sh` | runs Part 3 on the SLURM server (`bash` or `sbatch`) |
| `collect_results.py` | merges per-part CSVs into `results.csv`, writes `machine.txt` |
| `hypothesis.md` | hypotheses for Parts 3 and 4, committed before running them |
| `run_all.sh` | runs everything |
| `run_part2_server.sh` | reruns Part 2 on the SLURM server (`bash` or `sbatch`), backing up old results |
