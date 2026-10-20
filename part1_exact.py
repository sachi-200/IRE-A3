"""Part 1: exact search baseline with FAISS.

  1. Build IndexFlatL2 (exact brute force) for SIFT and GloVe (normalised).
  2. Run exact search for all 10,000 queries and save the top-100; later
     parts use these as ground truth.
  3. (full mode only) Check our exact results against the neighbours stored
     in the HDF5 file, with ties handled (float64 distances).
  4. Time exact search on the 5,000 test queries: one query at a time on a
     single thread (the same setting as Part 2), and as one batch.

Writes results/raw/part1_<dataset>.csv (benchmark rows) and
results/raw/part1_checks_<dataset>.csv (agreement checks).

Usage:
  python part1_exact.py                       # both datasets, full data
  python part1_exact.py --quick               # first 100K vectors only
  python part1_exact.py --datasets sift --single-n 500   # quick look
"""
import argparse
import os
import time

import common  # noqa: F401  (must come before faiss: sets OPENBLAS_NUM_THREADS)
import faiss
import numpy as np

from common import (BENCH_COLUMNS, DATASETS, default_threads, machine_name,
                    results_dir, timestamp, write_csv)
from data import N_TUNE, exact_paths, load_dataset, test_split
from recall import exact_distances, recall_at_k

K = 10
K_SAVE = 100  # keep top-100 exact neighbours for later parts

CHECK_COLUMNS = [
    "dataset", "n_base", "identical_id_sets_frac", "recall_ours_vs_file",
    "recall_file_vs_ours", "queries_with_real_diff", "max_dist_gap",
    "max_err_vs_stored_dists", "queries_tied_at_10", "machine",
]


def count_ties_at_k(base, queries, I):
    """Queries whose 10th and 11th exact neighbours are equally far."""
    d11 = np.sort(exact_distances(queries, base, I[:, :K + 1]), 1)
    return int(np.isclose(d11[:, K - 1], d11[:, K], rtol=1e-6, atol=1e-9).sum())


def check_against_file(name, base, queries, I, D, gt_ids, gt_dists):
    """Compare our exact top-10 with the file's top-10, handling ties."""
    ours = I[:, :K]
    theirs = gt_ids[:, :K]

    # Strict check: identical id sets.
    same_set = np.array([set(a) == set(b) for a, b in zip(ours, theirs)])

    # Tie-aware check: recompute both lists' distances in float64.
    d_ours = exact_distances(queries, base, ours)
    d_theirs = exact_distances(queries, base, theirs)
    r_ours_vs_file = recall_at_k(d_ours, d_theirs, k=K)   # ours judged by file
    r_file_vs_ours = recall_at_k(d_theirs, d_ours, k=K)   # file judged by ours
    max_gap = np.abs(np.sort(d_ours, 1) - np.sort(d_theirs, 1)).max()

    # The file stores its own distances: Euclidean (not squared) for SIFT,
    # and angular = 1 - cos for GloVe. For unit vectors, 1 - cos = ||a-b||^2 / 2.
    if name == "sift":
        file_equiv = np.sqrt(np.maximum(D[:, :K].astype(np.float64), 0))
    else:
        file_equiv = D[:, :K].astype(np.float64) / 2
    stored_dist_err = np.abs(file_equiv - gt_dists[:, :K]).max()

    n_bad = int((r_ours_vs_file < 1).sum())
    print(f"  identical top-10 id sets:           {same_set.mean() * 100:.2f}% of queries "
          f"({(~same_set).sum()} differ)")
    print(f"  tie-aware recall@10 ours vs file:   {r_ours_vs_file.mean():.6f}")
    print(f"  tie-aware recall@10 file vs ours:   {r_file_vs_ours.mean():.6f}")
    print(f"  queries with a real (non-tie) diff: {n_bad}")
    print(f"  max |dist diff| between the lists:  {max_gap:.3g}")
    print(f"  max diff vs stored file distances:  {stored_dist_err:.3g}")
    return {
        "identical_id_sets_frac": float(same_set.mean()),
        "recall_ours_vs_file": float(r_ours_vs_file.mean()),
        "recall_file_vs_ours": float(r_file_vs_ours.mean()),
        "queries_with_real_diff": n_bad,
        "max_dist_gap": float(max_gap),
        "max_err_vs_stored_dists": float(stored_dist_err),
    }


def time_batch(index, q, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        index.search(q, K)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_single(index, q, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(len(q)):
            index.search(q[i:i + 1], K)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run(name, quick, single_n, repeats, threads):
    print(f"\n=== {name.upper()}{' (quick)' if quick else ''} ===")
    ds = load_dataset(name, quick)
    base, queries = ds["base"], ds["queries"]
    nb, d = base.shape
    print(f"  base {base.shape}, queries {queries.shape}, threads={threads}")

    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(d)
    t0 = time.perf_counter()
    index.add(base)
    build_s = time.perf_counter() - t0
    index_bytes = int(faiss.serialize_index(index).nbytes)
    print(f"  index build (add): {build_s:.2f}s, size {index_bytes / 1e6:.1f} MB")

    # Exact search for all queries; save as ground truth for later parts.
    t0 = time.perf_counter()
    D, I = index.search(queries, K_SAVE)
    print(f"  exact top-{K_SAVE} for all {len(queries)} queries: {time.perf_counter() - t0:.2f}s")
    i_path, d_path = exact_paths(name, quick)
    np.save(i_path, I)
    np.save(d_path, D)

    check = {"dataset": name, "n_base": nb, "machine": machine_name(),
             "queries_tied_at_10": count_ties_at_k(base, queries, I)}
    if not quick:
        print("  -- agreement with stored ground truth (all 10,000 queries) --")
        check.update(check_against_file(name, base, queries, I, D, ds["gt_ids"], ds["gt_dists"]))
    print(f"  queries with a tie at 10th/11th:    {check['queries_tied_at_10']}")

    # Exact search is its own ground truth, so recall is 1 by construction;
    # computed anyway as a sanity check of the pipeline.
    q_test = test_split(queries)
    I_test = test_split(I)[:, :K]
    d_test = exact_distances(q_test, base, I_test)
    recall = float(recall_at_k(d_test, d_test, k=K).mean())
    assert recall == 1.0

    # Timing on the held-out test half only.
    n_single = len(q_test) if single_n <= 0 else min(single_n, len(q_test))
    index.search(q_test[:100], K)  # warm-up
    print(f"  -- timing on test queries {N_TUNE}..{len(queries) - 1} (k={K}, "
          f"median of {repeats}) --")

    runs = []  # (mode, threads, n_queries, seconds)
    faiss.omp_set_num_threads(1)
    runs.append(("single", 1, n_single, time_single(index, q_test[:n_single], repeats)))
    runs.append(("batch", 1, len(q_test), time_batch(index, q_test, repeats)))
    if threads > 1:
        faiss.omp_set_num_threads(threads)
        runs.append(("batch", threads, len(q_test), time_batch(index, q_test, repeats)))

    rows = []
    for mode, t, n, secs in runs:
        per_q = secs / n
        print(f"  {mode:6s} {t:2d} thread(s), {n} q: {secs:8.2f}s  "
              f"{per_q * 1e3:8.3f} ms/q  {1 / per_q:9.1f} QPS")
        rows.append({
            "part": 1, "dataset": name, "n_base": nb, "n_queries": len(q_test), "n_timed": n,
            "query_set": "test", "index": "Flat", "build_params": "",
            "search_params": "", "mode": mode, "threads": t, "repeats": repeats,
            "recall_at_10": recall, "qps": 1 / per_q, "ms_per_query": per_q * 1e3,
            "build_s": build_s, "index_bytes": index_bytes,
            "bytes_per_vector": index_bytes / nb,
            "machine": machine_name(), "timestamp": timestamp(),
        })
    single_q = rows[0]["ms_per_query"]
    for r in rows[1:]:
        print(f"  speed-up of batch ({r['threads']} thread(s)) over single: "
              f"{single_q / r['ms_per_query']:.1f}x")
    return rows, check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--quick", action="store_true",
                   help="index only the first 100K vectors (for run_all.sh --quick)")
    p.add_argument("--single-n", type=int, default=None,
                   help="queries to time one-at-a-time "
                        "(default: all 5,000 test queries; 1,000 with --quick)")
    p.add_argument("--repeats", type=int, default=3, help="timing repeats (median)")
    p.add_argument("--threads", type=int, default=default_threads(),
                   help="threads for the multi-threaded batch run "
                        "(default: physical cores, or $ANN_THREADS)")
    args = p.parse_args()
    if args.single_n is None:
        args.single_n = 1000 if args.quick else 0

    out = results_dir(args.quick)
    for name in args.datasets:
        rows, check = run(name, args.quick, args.single_n, args.repeats, args.threads)
        write_csv(os.path.join(out, "raw", f"part1_{name}.csv"), rows, BENCH_COLUMNS)
        write_csv(os.path.join(out, "raw", f"part1_checks_{name}.csv"), [check], CHECK_COLUMNS)
        print(f"  saved results to {os.path.join(out, 'raw')}")


if __name__ == "__main__":
    main()
