"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ (FAISS) on SIFT and GloVe.

For each index type, a few build settings are tried, and for each one the
search-time parameter (efSearch or nprobe) is swept from low recall up to
about 0.99.

For every (build setting, search parameter) point we measure:
  * recall@10 (tie-aware) on the tune half and on the test half, using a
    multi-threaded batch search (results are identical to one-at-a-time
    search; only the timing differs);
  * speed, one query at a time on a single thread, as the median of
    --repeats runs over the first --time-n queries of each half.

The tune-half numbers are used to choose settings (part2_report.py); the
test-half numbers are the ones reported.

Writes results/raw/part2_<dataset>_<index>.csv, updated after every build
setting so that a run killed by a time limit keeps what it finished.

Usage:
  python part2_bench.py                               # everything (~1-2 h)
  python part2_bench.py --datasets sift --indexes hnsw
  python part2_bench.py --quick                       # first 100K vectors
"""
import argparse
import csv
import gc
import os
import time

import common  # noqa: F401  (must come before faiss: sets OPENBLAS_NUM_THREADS)
import faiss
import numpy as np

from common import (BENCH_COLUMNS, DATASETS, default_threads, machine_name,
                    results_dir, timestamp, write_csv)
from data import load_dataset, load_exact, test_split, tune_split
from recall import exact_distances, recall_at_k

K = 10
INDEXES = ["hnsw", "ivfflat", "ivfpq"]

EF_CONSTRUCTION = 200   # fixed, so only M varies between HNSW builds
PQ_NBITS = 8            # 256 centroids per sub-quantizer

# Build settings. PQ sub-vector counts m must divide the dimension: SIFT
# (d=128) gets sub-vectors of 8/4/2 dimensions, GloVe (d=100) of 10/4/2.
PQ_M = {"sift": (16, 32, 64), "glove": (10, 25, 50)}
FULL_BUILDS = {
    "hnsw": [{"M": 8}, {"M": 16}, {"M": 32}],
    "ivfflat": [{"nlist": 1024}, {"nlist": 4096}, {"nlist": 16384}],
    "ivfpq": {ds: [{"nlist": 4096, "m": m} for m in ms] for ds, ms in PQ_M.items()},
}
QUICK_BUILDS = {  # 100K vectors: fewer, smaller lists so k-means has enough points
    "hnsw": [{"M": 8}, {"M": 16}, {"M": 32}],
    "ivfflat": [{"nlist": 256}, {"nlist": 1024}],
    "ivfpq": {ds: [{"nlist": 256, "m": m} for m in ms] for ds, ms in PQ_M.items()},
}

EF_SEARCH = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
NPROBE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Stop a sweep early (decided on the tune half) once recall is high enough,
# has stopped improving, or the index is already slower than exact search.
STOP_RECALL = 0.99
PLATEAU_GAIN = 0.003    # over the last 2 steps
PLATEAU_WINDOW = 2


def builds_for(kind, name, quick):
    b = (QUICK_BUILDS if quick else FULL_BUILDS)[kind]
    return b[name] if isinstance(b, dict) else b


def build_params_str(kind, p):
    if kind == "hnsw":
        return f"M={p['M']},efC={EF_CONSTRUCTION}"
    if kind == "ivfflat":
        return f"nlist={p['nlist']}"
    return f"nlist={p['nlist']},m={p['m']},nbits={PQ_NBITS}"


def build_index(kind, p, base):
    """Build (train + add) an index; returns (index, seconds)."""
    d = base.shape[1]
    t0 = time.perf_counter()
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, p["M"])
        index.hnsw.efConstruction = EF_CONSTRUCTION
    elif kind == "ivfflat":
        index = faiss.index_factory(d, f"IVF{p['nlist']},Flat")
    else:
        index = faiss.index_factory(d, f"IVF{p['nlist']},PQ{p['m']}x{PQ_NBITS}")
        # The factory turns on polysemous training, which reorders PQ codes for
        # Hamming-distance filtering. We never use that search mode, and it made
        # training 7-9x slower, so it is switched off.
        index.do_polysemous_training = False
    index.train(base)
    index.add(base)
    return index, time.perf_counter() - t0


def sweep_values(kind, p):
    if kind == "hnsw":
        return [("efSearch", v) for v in EF_SEARCH]
    return [("nprobe", v) for v in NPROBE if v <= p["nlist"]]


def set_search_param(index, key, value):
    if key == "efSearch":
        index.hnsw.efSearch = value
    else:
        faiss.extract_index_ivf(index).nprobe = value


def time_one_at_a_time(index, q, repeats):
    """Median seconds to search q one query at a time (caller sets 1 thread)."""
    index.search(q[:20], K)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(len(q)):
            index.search(q[i:i + 1], K)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def exact_single_ms(out, name):
    """Part 1's single-query exact time, used to stop hopeless sweeps."""
    path = os.path.join(out, "raw", f"part1_{name}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["mode"] == "single":
                return float(r["ms_per_query"])
    return None


def run_dataset(name, kinds, args, out):
    print(f"\n=== {name.upper()}{' (quick)' if args.quick else ''} ===")
    ds = load_dataset(name, args.quick)
    base, queries = ds["base"], ds["queries"]
    nb = len(base)
    gt_I, _ = load_exact(name, args.quick)
    gt_d = exact_distances(queries, base, gt_I[:, :K])  # float64, all 10k queries
    halves = {"tune": (tune_split(queries), tune_split(gt_d)),
              "test": (test_split(queries), test_split(gt_d))}
    exact_ms = exact_single_ms(out, name)
    print(f"  base {base.shape}; exact single-query: "
          f"{f'{exact_ms:.2f} ms' if exact_ms else 'unknown (run part 1)'}")

    for kind in kinds:
        rows = []
        path = os.path.join(out, "raw", f"part2_{name}_{kind}.csv")
        for p in builds_for(kind, name, args.quick):
            bstr = build_params_str(kind, p)
            faiss.omp_set_num_threads(args.threads)
            index, build_s = build_index(kind, p, base)
            index_bytes = int(faiss.serialize_index(index).nbytes)
            print(f"\n  [{kind} {bstr}] build {build_s:.1f}s ({args.threads} threads), "
                  f"{index_bytes / 1e6:.1f} MB, {index_bytes / nb:.1f} B/vector")

            history = []
            for key, val in sweep_values(kind, p):
                set_search_param(index, key, val)

                # Recall: batch search on all threads.
                faiss.omp_set_num_threads(args.threads)
                _, I = index.search(queries, K)
                rec_d = exact_distances(queries, base, I)
                recall = {"tune": recall_at_k(tune_split(rec_d), halves["tune"][1], K).mean(),
                          "test": recall_at_k(test_split(rec_d), halves["test"][1], K).mean()}

                # Speed: one query at a time, single thread.
                faiss.omp_set_num_threads(1)
                ms = {}
                for half, (q, _) in halves.items():
                    n = min(args.time_n, len(q))
                    secs = time_one_at_a_time(index, q[:n], args.repeats)
                    ms[half] = secs / n * 1e3
                    rows.append({
                        "part": 2, "dataset": name, "n_base": nb, "n_queries": len(q),
                        "n_timed": n, "query_set": half, "index": kind,
                        "build_params": bstr, "search_params": f"{key}={val}",
                        "mode": "single", "threads": 1, "repeats": args.repeats,
                        "recall_at_10": float(recall[half]), "qps": 1e3 / ms[half],
                        "ms_per_query": ms[half], "build_s": build_s,
                        "index_bytes": index_bytes, "bytes_per_vector": index_bytes / nb,
                        "machine": machine_name(), "timestamp": timestamp(),
                    })
                print(f"    {key}={val:<5d} recall tune {recall['tune']:.4f} test "
                      f"{recall['test']:.4f}   {ms['test']:7.3f} ms/q  "
                      f"{1e3 / ms['test']:9.1f} QPS")

                history.append(recall["tune"])
                if recall["tune"] >= STOP_RECALL:
                    break
                if (len(history) > PLATEAU_WINDOW
                        and abs(history[-1] - history[-1 - PLATEAU_WINDOW]) < PLATEAU_GAIN):
                    print("    (recall has plateaued; stopping sweep)")
                    break
                if exact_ms is not None and ms["tune"] > exact_ms:
                    print("    (slower than exact search; stopping sweep)")
                    break

            write_csv(path, rows, BENCH_COLUMNS)  # save after every build setting
            del index
            gc.collect()
        print(f"  saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--indexes", nargs="+", default=INDEXES, choices=INDEXES)
    p.add_argument("--quick", action="store_true",
                   help="index only the first 100K vectors (for run_all.sh --quick)")
    p.add_argument("--time-n", type=int, default=None,
                   help="queries timed per half (default 1000; 100 with --quick)")
    p.add_argument("--repeats", type=int, default=3, help="timing repeats (median)")
    p.add_argument("--threads", type=int, default=default_threads(),
                   help="threads for building and for the recall batch search")
    args = p.parse_args()
    if args.time_n is None:
        args.time_n = 100 if args.quick else 1000

    out = results_dir(args.quick)
    for name in args.datasets:
        run_dataset(name, args.indexes, args, out)


if __name__ == "__main__":
    main()
