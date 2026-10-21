"""Part 3 (Compression): PQ code size, with and without exact re-ranking, on SIFT.

For each number of PQ sub-vectors m (nlist fixed, 8 bits per sub-vector):
  * plain IVF-PQ: neighbours ranked by approximate (PQ) distances;
  * IVF-PQ + re-ranking: IVF-PQ returns its top 100 candidates, whose exact
    distances are then computed from the original vectors (FAISS
    IndexRefineFlat, k_factor = 100 / 10). This needs the raw vectors as well
    as the codes, which the memory numbers include.
Both variants share the same trained IVF-PQ index and are swept over nprobe
with exactly the same protocol as Part 2 (part2_bench.sweep).

IVF-Flat with the same nlist is re-measured in the same run as a
full-precision control, so that its speed is compared under the same machine
load as the PQ variants (timings on the shared server varied by up to 1.7x
between runs).

Writes results/raw/part3_sift_compression.csv.

Usage:
  python part3_compression.py            # full SIFT-1M
  python part3_compression.py --quick    # first 100K vectors
"""
import argparse
import gc
import os
import time

import common  # noqa: F401  (must come before faiss: sets OPENBLAS_NUM_THREADS)
import faiss

from common import BENCH_COLUMNS, default_threads, results_dir, write_csv
from part2_bench import K, NPROBE, build_index, load_context, sweep

DATASET = "sift"
PQ_M = (8, 16, 32, 64)        # sub-vectors of 16/8/4/2 dimensions (d = 128)
NLIST = 4096                  # same as Part 2's IVF-PQ
NLIST_QUICK = 256
RERANK = 100                  # candidates re-ranked with exact distances


def set_nprobe(ivf_index):
    """Setter that changes nprobe on the IVF-PQ inside either variant."""
    def setter(_index, key, value):
        assert key == "nprobe"
        faiss.extract_index_ivf(ivf_index).nprobe = value
    return setter


def run_m(m, nlist, ctx, args):
    """Build IVF-PQ with m sub-vectors; sweep it plain and with re-ranking."""
    base, nb = ctx["base"], ctx["nb"]
    bstr = f"nlist={nlist},m={m},nbits=8"
    values = [("nprobe", v) for v in NPROBE if v <= nlist]

    faiss.omp_set_num_threads(args.threads)
    pq, pq_build_s = build_index("ivfpq", {"nlist": nlist, "m": m}, base)
    pq_bytes = int(faiss.serialize_index(pq).nbytes)

    # Re-ranking wrapper: stores a copy of the raw vectors next to the codes.
    t0 = time.perf_counter()
    refine = faiss.IndexRefineFlat(pq, faiss.swig_ptr(base))
    refine.k_factor = RERANK / K
    rr_build_s = pq_build_s + time.perf_counter() - t0
    rr_bytes = int(faiss.serialize_index(refine).nbytes)

    rows = []
    for name, index, build_s, nbytes in [
            ("ivfpq", pq, pq_build_s, pq_bytes),
            (f"ivfpq+rerank{RERANK}", refine, rr_build_s, rr_bytes)]:
        print(f"\n  [{name} {bstr}] build {build_s:.1f}s, {nbytes / 1e6:.1f} MB, "
              f"{nbytes / nb:.1f} B/vector")
        rows += sweep(index, values, ctx, args,
                      {"part": 3, "index": name, "build_params": bstr,
                       "build_s": build_s, "index_bytes": nbytes},
                      set_param=set_nprobe(pq))
    del refine, pq
    gc.collect()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="index only the first 100K vectors (for run_all.sh --quick)")
    p.add_argument("--m", type=int, nargs="+", default=list(PQ_M),
                   help="numbers of PQ sub-vectors to try (must divide 128)")
    p.add_argument("--time-n-tune", type=int, default=None,
                   help="tune queries timed (default 1000; 100 with --quick)")
    p.add_argument("--time-n-test", type=int, default=None,
                   help="test queries timed (default 0 = all 5,000; 100 with --quick)")
    p.add_argument("--repeats", type=int, default=3, help="timing repeats (median)")
    p.add_argument("--threads", type=int, default=default_threads(),
                   help="threads for building and for the recall batch search")
    args = p.parse_args()
    if args.time_n_tune is None:
        args.time_n_tune = 100 if args.quick else 1000
    if args.time_n_test is None:
        args.time_n_test = 100 if args.quick else 0

    out = results_dir(args.quick)
    nlist = NLIST_QUICK if args.quick else NLIST
    print(f"\n=== Part 3: compression on {DATASET.upper()}{' (quick)' if args.quick else ''} ===")
    ctx = load_context(DATASET, args, out)
    path = os.path.join(out, "raw", f"part3_{DATASET}_compression.csv")
    rows = []
    for m in args.m:
        rows += run_m(m, nlist, ctx, args)
        write_csv(path, rows, BENCH_COLUMNS)  # save after every m

    # Full-precision control: IVF-Flat, same nlist, same run.
    faiss.omp_set_num_threads(args.threads)
    flat, build_s = build_index("ivfflat", {"nlist": nlist}, ctx["base"])
    nbytes = int(faiss.serialize_index(flat).nbytes)
    print(f"\n  [ivfflat nlist={nlist}] (control) build {build_s:.1f}s, {nbytes / 1e6:.1f} MB, "
          f"{nbytes / ctx['nb']:.1f} B/vector")
    rows += sweep(flat, [("nprobe", v) for v in NPROBE if v <= nlist], ctx, args,
                  {"part": 3, "index": "ivfflat", "build_params": f"nlist={nlist}",
                   "build_s": build_s, "index_bytes": nbytes})
    write_csv(path, rows, BENCH_COLUMNS)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
