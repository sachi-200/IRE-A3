"""Shared settings: paths, quick mode, thread count, machine info, CSV output.

Import this module BEFORE faiss: it limits OpenBLAS to one thread, which must
happen before the library is loaded. FAISS parallelises with OpenMP itself,
and letting OpenBLAS add its own threads on top oversubscribes the cores.
"""
import csv
import os
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# Large files (datasets, exact neighbours) live here; override on a server
# with e.g. `export ANN_DATA_DIR=/scratch/$USER/ann-data`.
DATA_DIR = os.environ.get("ANN_DATA_DIR", os.path.join(CODE_DIR, "data"))

QUICK_N_BASE = 100_000  # --quick: index only the first 100K vectors

DATASETS = ["sift", "glove"]

# One row per benchmark run; every part writes rows with these columns.
BENCH_COLUMNS = [
    "part", "dataset", "n_base", "n_queries", "n_timed", "query_set",
    "index", "build_params", "search_params", "mode", "threads", "repeats",
    "recall_at_10", "qps", "ms_per_query", "build_s", "index_bytes", "bytes_per_vector",
    "machine", "timestamp",
]


def results_dir(quick):
    """results/ for full runs, results_quick/ for --quick, so they never mix."""
    d = os.path.join(CODE_DIR, "results_quick" if quick else "results")
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    return d


def physical_cores():
    """Physical cores available to this process (hyperthreads not counted)."""
    try:
        cores = set()
        for cpu in os.sched_getaffinity(0):
            topo = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
            with open(topo + "physical_package_id") as f:
                pkg = f.read().strip()
            with open(topo + "core_id") as f:
                core = f.read().strip()
            cores.add((pkg, core))
        return len(cores)
    except (OSError, AttributeError):
        return os.cpu_count() or 1


def default_threads():
    """ANN_THREADS if set (useful on a shared server), else physical cores."""
    return int(os.environ.get("ANN_THREADS", physical_cores()))


def machine_name():
    return platform.node()


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_csv(path, rows, columns):
    """Overwrite path with rows (list of dicts). Missing keys become empty."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def machine_info():
    """Hardware and library versions, for the report's 'specs' paragraph."""
    info = {"hostname": platform.node(), "os": platform.platform(),
            "python": platform.python_version()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
            info["ram_gb"] = f"{kb / 1024 ** 2:.1f}"
    except OSError:
        pass
    info["logical_cpus"] = os.cpu_count()  # whole machine
    try:
        info["cpus_allocated"] = len(os.sched_getaffinity(0))  # e.g. SLURM limit
    except AttributeError:
        pass
    info["physical_cores"] = physical_cores()  # of the allocated CPUs
    info["threads_used"] = default_threads()
    for mod in ["numpy", "faiss", "h5py", "hnswlib", "matplotlib"]:
        try:
            info[mod] = getattr(__import__(mod), "__version__", "installed")
        except ImportError:
            info[mod] = "not installed"
    return info
